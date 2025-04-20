"""Tests for the RL Scheduler Integration module."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import numpy as np

from services.reinforcement_learning_service.scheduler_integration import RLSchedulerIntegration


@pytest.fixture
def mock_rl_service():
    """Create a mock RL service."""
    mock = Mock()
    mock.get_sample_count.return_value = 2000
    mock.get_current_state.return_value = np.zeros(10)
    mock.get_action.return_value = np.zeros(5)
    mock.get_best_reward.return_value = 100.0
    mock.get_current_metrics.return_value = {"cost": 1000.0, "conversion_rate": 0.1, "roas": 2.5}
    mock.get_baseline_metrics.return_value = {"cost": 900.0, "conversion_rate": 0.09, "roas": 2.3}
    mock.evaluate_episode.return_value = {
        "reward": 110.0,
        "performance_ratio": 0.85,
        "safety_violations": 0,
    }
    return mock


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler."""
    mock = Mock()
    mock.add_task.return_value = "task_123"
    return mock


@pytest.fixture
def integration(mock_rl_service, mock_scheduler):
    """Create an integration instance with mocks."""
    return RLSchedulerIntegration(
        rl_service=mock_rl_service,
        scheduler=mock_scheduler,
        config={"training": {"min_samples": 1000}},
    )


def test_initialization(integration):
    """Test initialization of scheduler integration."""
    assert integration.rl_service is not None
    assert integration.scheduler is not None
    assert integration.config["training"]["min_samples"] == 1000
    assert len(integration.performance_history) == 0
    assert len(integration.safety_violations) == 0


def test_schedule_all_tasks(integration, mock_scheduler):
    """Test scheduling all tasks."""
    task_ids = integration.schedule_all_tasks()

    assert len(task_ids) == 4
    assert all(id == "task_123" for id in task_ids.values())
    assert mock_scheduler.add_task.call_count == 4


def test_training_task(integration, mock_rl_service):
    """Test training task execution."""
    task_id = integration.schedule_training()

    assert task_id == "task_123"
    assert mock_rl_service.get_sample_count.called

    # Get the training function
    training_func = integration.scheduler.add_task.call_args[1]["function"]

    # Execute training
    metrics = training_func()

    assert mock_rl_service.train_temporal.called
    assert mock_rl_service.evaluate_episode.called
    assert mock_rl_service.save_policy.called


def test_inference_task_safe_hours(integration, mock_rl_service):
    """Test inference task during safe hours."""
    with patch("datetime.datetime") as mock_dt:
        mock_dt.now.return_value.hour = 14  # 2 PM

        task_id = integration.schedule_inference()
        inference_func = integration.scheduler.add_task.call_args[1]["function"]

        metrics = inference_func()

        assert mock_rl_service.get_current_state.called
        assert mock_rl_service.get_action.called
        assert mock_rl_service.apply_action_safely.called
        assert len(integration.performance_history) > 0


def test_inference_task_unsafe_hours(integration, mock_rl_service):
    """Test inference task during unsafe hours."""
    with patch("datetime.datetime") as mock_dt:
        mock_dt.now.return_value.hour = 2  # 2 AM

        task_id = integration.schedule_inference()
        inference_func = integration.scheduler.add_task.call_args[1]["function"]

        metrics = inference_func()

        assert not mock_rl_service.get_current_state.called
        assert not mock_rl_service.get_action.called
        assert len(integration.performance_history) == 0


def test_evaluation_task(integration, mock_rl_service):
    """Test evaluation task execution."""
    task_id = integration.schedule_evaluation()
    eval_func = integration.scheduler.add_task.call_args[1]["function"]

    metrics = eval_func()

    assert mock_rl_service.evaluate_episode.called
    assert metrics["mean_reward"] > 0
    assert metrics["performance_ratio"] > 0


def test_safety_checks_no_violations(integration, mock_rl_service):
    """Test safety checks with no violations."""
    task_id = integration.schedule_safety_checks()
    safety_func = integration.scheduler.add_task.call_args[1]["function"]

    result = safety_func()

    assert mock_rl_service.get_current_metrics.called
    assert mock_rl_service.get_baseline_metrics.called
    assert len(result["violations"]) == 0
    assert len(integration.safety_violations) == 0


def test_safety_checks_with_violations(integration, mock_rl_service):
    """Test safety checks with violations."""
    mock_rl_service.get_current_metrics.return_value = {
        "cost": 2000.0,  # 100% increase
        "conversion_rate": 0.05,  # 50% decrease
        "roas": 1.5,  # 35% decrease
    }

    task_id = integration.schedule_safety_checks()
    safety_func = integration.scheduler.add_task.call_args[1]["function"]

    result = safety_func()

    assert len(result["violations"]) > 0
    assert "excessive_cost_increase" in result["violations"]
    assert "low_conversion_rate" in result["violations"]
    assert "low_roas" in result["violations"]
    assert len(integration.safety_violations) == 1
    assert mock_rl_service.load_policy.called
    assert mock_rl_service.reset_exploration.called
    assert mock_rl_service.set_action_constraints.called


def test_evaluate_policy(integration, mock_rl_service):
    """Test policy evaluation."""
    metrics = integration._evaluate_policy(episodes=3)

    assert mock_rl_service.evaluate_episode.call_count == 3
    assert "mean_reward" in metrics
    assert "performance_ratio" in metrics
    assert "safety_violations" in metrics


def test_check_safety_violations(integration):
    """Test safety violation checks."""
    current = {"cost": 1000.0, "conversion_rate": 0.08, "roas": 2.0}
    baseline = {"cost": 800.0, "conversion_rate": 0.1, "roas": 2.5}

    violations = integration._check_safety_violations(current, baseline)

    assert len(violations) == 2
    assert "low_conversion_rate" in violations
    assert "low_roas" in violations


def test_apply_safety_measures(integration, mock_rl_service):
    """Test applying safety measures."""
    integration._apply_safety_measures()

    assert mock_rl_service.load_policy.called_with("best_model")
    assert mock_rl_service.reset_exploration.called
    assert mock_rl_service.set_action_constraints.called


def test_get_history(integration):
    """Test getting performance and violation history."""
    # Add some test data
    integration.performance_history.append({"reward": 100})
    integration.safety_violations.append(
        {"timestamp": datetime.now(), "violations": ["test_violation"]}
    )

    perf_history = integration.get_performance_history()
    violation_history = integration.get_safety_violations()

    assert len(perf_history) == 1
    assert len(violation_history) == 1
    assert perf_history[0]["reward"] == 100
    assert violation_history[0]["violations"] == ["test_violation"]
