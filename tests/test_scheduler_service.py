import pytest
import os
import json
import time
from datetime import datetime, timedelta
from services.scheduler_service.scheduler_service import SchedulerService
from unittest.mock import Mock, patch


class TestSchedulerService:
    """Test class for SchedulerService"""

    @pytest.fixture
    def scheduler_service(self):
        """Create a SchedulerService instance for testing"""
        service = SchedulerService()

        # Clean up any existing test file before starting
        if os.path.exists("data/test_tasks.json"):
            os.remove("data/test_tasks.json")

        # Set the tasks file to a test file
        service.tasks_file = "data/test_tasks.json"

        # Return the service
        yield service

        # Clean up after tests
        if os.path.exists("data/test_tasks.json"):
            os.remove("data/test_tasks.json")

    def test_init(self, scheduler_service):
        """Test initialization of SchedulerService"""
        assert scheduler_service is not None
        assert scheduler_service.tasks is not None
        assert scheduler_service.tasks.get("tasks", None) is not None
        assert isinstance(scheduler_service.tasks["tasks"], list)

    def test_add_task(self, scheduler_service):
        """Test adding a task"""
        # Add a test task
        task = scheduler_service.add_task(
            task_name="Test Task",
            task_type="test_type",
            schedule_type="daily",
            schedule_time="12:00",
            parameters={"param1": "value1"},
            enabled=True,
        )

        # Verify the task was added
        assert task is not None
        assert task["name"] == "Test Task"
        assert task["type"] == "test_type"
        assert task["schedule_type"] == "daily"
        assert task["schedule_time"] == "12:00"
        assert task["parameters"] == {"param1": "value1"}
        assert task["enabled"] is True

        # Verify the task is in the service's tasks list
        tasks = scheduler_service.get_tasks()
        assert len(tasks) == 1
        assert tasks[0]["id"] == task["id"]

    def test_get_tasks(self, scheduler_service):
        """Test getting tasks"""
        # Add a few tasks
        task1 = scheduler_service.add_task(
            task_name="Task 1",
            task_type="test_type",
            schedule_type="daily",
            schedule_time="12:00",
            enabled=True,
        )

        task2 = scheduler_service.add_task(
            task_name="Task 2",
            task_type="test_type",
            schedule_type="daily",
            schedule_time="13:00",
            enabled=False,
        )

        # Test getting all tasks (including disabled)
        all_tasks = scheduler_service.get_tasks(include_disabled=True)
        assert len(all_tasks) == 2

        # Test getting only enabled tasks
        enabled_tasks = scheduler_service.get_tasks(include_disabled=False)
        assert len(enabled_tasks) == 1
        assert enabled_tasks[0]["name"] == "Task 1"

    def test_update_task(self, scheduler_service):
        """Test updating a task"""
        # Add a test task
        task = scheduler_service.add_task(
            task_name="Original Name",
            task_type="original_type",
            schedule_type="daily",
            schedule_time="12:00",
            enabled=True,
        )

        # Update the task
        updated_task = scheduler_service.update_task(
            task["id"], {"name": "Updated Name", "type": "updated_type", "enabled": False}
        )

        # Verify the task was updated
        assert updated_task["name"] == "Updated Name"
        assert updated_task["type"] == "updated_type"
        assert updated_task["enabled"] is False
        assert updated_task["schedule_type"] == "daily"  # Unchanged

        # Get the task and verify
        task_from_get = scheduler_service.get_task(task["id"])
        assert task_from_get["name"] == "Updated Name"

    def test_delete_task(self, scheduler_service):
        """Test deleting a task"""
        # Add a test task
        task = scheduler_service.add_task(
            task_name="Task to Delete",
            task_type="test_type",
            schedule_type="daily",
            schedule_time="12:00",
            enabled=True,
        )

        # Verify it was added
        tasks_before = scheduler_service.get_tasks(include_disabled=True)
        assert len(tasks_before) == 1

        # Delete the task
        result = scheduler_service.delete_task(task["id"])
        assert result is True

        # Verify it was deleted
        tasks_after = scheduler_service.get_tasks(include_disabled=True)
        assert len(tasks_after) == 0

    def test_save_and_load_tasks(self, scheduler_service):
        """Test saving and loading tasks"""
        # Add a test task
        task = scheduler_service.add_task(
            task_name="Persistent Task",
            task_type="test_type",
            schedule_type="daily",
            schedule_time="12:00",
            enabled=True,
        )

        # Force a save
        scheduler_service._save_tasks()

        # Create a new service with the same tasks file
        new_service = SchedulerService()
        new_service.tasks_file = scheduler_service.tasks_file

        # Load tasks
        new_service.tasks = new_service._load_tasks()

        # Verify the task was loaded
        assert len(new_service.tasks["tasks"]) == 1
        assert new_service.tasks["tasks"][0]["name"] == "Persistent Task"

    def test_start_stop(self, scheduler_service):
        """Test starting and stopping the scheduler"""
        # Add a test task
        task = scheduler_service.add_task(
            task_name="Scheduled Task",
            task_type="test_type",
            schedule_type="one_time",
            # Schedule for 5 seconds in the future
            schedule_time=(datetime.now() + timedelta(seconds=5)).isoformat(),
            enabled=True,
        )

        # Start the scheduler
        scheduler_service.start()
        assert scheduler_service.is_running is True

        # Stop the scheduler
        scheduler_service.stop()
        assert scheduler_service.is_running is False

    def test_execute_task_now(self, scheduler_service):
        """Test executing a task immediately"""
        # Add a test task
        task = scheduler_service.add_task(
            task_name="Immediate Task",
            task_type="test_type",
            schedule_type="daily",
            schedule_time="12:00",
            enabled=True,
        )

        # Execute the task immediately
        result = scheduler_service.execute_task_now(task["id"])
        assert result["status"] == "success"

        # Verify the task execution was recorded in history
        history = scheduler_service.get_execution_history(task["id"])
        assert len(history) == 1
        assert history[0]["task_name"] == "Immediate Task"
        assert history[0]["status"] == "success"

        # Verify the task status was updated
        updated_task = scheduler_service.get_task(task["id"])
        assert updated_task["status"] == "completed"
        assert updated_task["execution_count"] == 1


@pytest.fixture
def service():
    """Create a SchedulerService instance for testing."""
    config = {"ads_client": Mock(), "optimizer": Mock()}
    return SchedulerService(config)


@pytest.fixture
def mock_ads_agent():
    """Create a mock AdsAgent for testing."""
    agent = Mock()
    agent.analyze_bid_impact.return_value = {
        "absolute_effect": 100,
        "relative_effect": 0.2,
        "p_value": 0.01,
    }
    agent.build_campaign_control_group.return_value = {
        "control_campaigns": {"123": 0.7, "456": 0.3},
        "validation_metrics": {"num_controls": 2},
    }
    return agent


def test_schedule_causal_analysis(service, mock_ads_agent):
    """Test scheduling causal impact analysis tasks."""
    service.ads_agent = mock_ads_agent

    # Test daily schedule
    campaign_ids = ["789", "012"]
    metrics = ["clicks", "conversions"]

    task_ids = service.schedule_causal_analysis(
        campaign_ids=campaign_ids, metrics=metrics, schedule_type="daily", hour=10, minute=30
    )

    assert len(task_ids) == len(campaign_ids) * len(metrics)
    assert all(isinstance(tid, str) for tid in task_ids)

    # Test weekly schedule
    task_ids = service.schedule_causal_analysis(
        campaign_ids=campaign_ids,
        metrics=metrics,
        schedule_type="weekly",
        day_of_week="monday",
        hour=9,
        minute=0,
    )

    assert len(task_ids) == len(campaign_ids) * len(metrics)

    # Test default schedule parameters
    task_ids = service.schedule_causal_analysis(campaign_ids=["789"], metrics=["clicks"])

    assert len(task_ids) == 1
    assert service.default_schedules["causal_impact"]["hour"] == 6
    assert service.default_schedules["causal_impact"]["minute"] == 0


def test_schedule_control_group_updates(service, mock_ads_agent):
    """Test scheduling control group update tasks."""
    service.ads_agent = mock_ads_agent

    campaign_ids = ["789", "012"]
    metrics = ["clicks", "conversions"]

    task_ids = service.schedule_control_group_updates(
        campaign_ids=campaign_ids,
        metrics=metrics,
        schedule_type="weekly",
        hour=5,
        minute=30,
        day_of_week="monday",
    )

    assert len(task_ids) == len(campaign_ids) * len(metrics)
    assert all(isinstance(tid, str) for tid in task_ids)

    # Test default schedule parameters
    task_ids = service.schedule_control_group_updates(campaign_ids=["789"], metrics=["clicks"])

    assert len(task_ids) == 1
    assert service.default_schedules["control_group_update"]["hour"] == 5
    assert service.default_schedules["control_group_update"]["minute"] == 0
    assert service.default_schedules["control_group_update"]["day"] == "monday"


def test_run_causal_analysis(service, mock_ads_agent):
    """Test running causal analysis task."""
    service.ads_agent = mock_ads_agent

    result = service._run_causal_analysis("789", "clicks")

    assert result["absolute_effect"] == 100
    assert result["relative_effect"] == 0.2
    assert result["p_value"] == 0.01

    mock_ads_agent.analyze_bid_impact.assert_called_once_with(campaign_id="789", metric="clicks")


def test_update_control_group(service, mock_ads_agent):
    """Test updating control group task."""
    service.ads_agent = mock_ads_agent

    result = service._update_control_group("789", "clicks")

    assert result["control_campaigns"] == {"123": 0.7, "456": 0.3}
    assert result["validation_metrics"]["num_controls"] == 2

    mock_ads_agent.build_campaign_control_group.assert_called_once_with(
        target_campaign_id="789", metric="clicks"
    )


def test_error_handling(service, mock_ads_agent):
    """Test error handling in scheduled tasks."""
    service.ads_agent = mock_ads_agent
    mock_ads_agent.analyze_bid_impact.side_effect = Exception("API error")

    with pytest.raises(Exception):
        service._run_causal_analysis("789", "clicks")

    mock_ads_agent.build_campaign_control_group.side_effect = Exception("Data error")

    with pytest.raises(Exception):
        service._update_control_group("789", "clicks")
