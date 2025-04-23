"""
Scheduler integration for the Reinforcement Learning Service.

This module handles scheduling of training, inference, evaluation, and safety checks
for the reinforcement learning service.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np
import time
import threading

from .ads_environment import AdsEnvironment  # Assuming AdsEnvironment is in the same dir
from .policy_storage import PolicyStorage
from .replay_buffer import ReplayBuffer
from ..base_service import BaseService

from services.reinforcement_learning_service import ReinforcementLearningService


class RLSchedulerIntegration:
    """Handles scheduling and coordination of RL service tasks."""

    def __init__(
        self,
        rl_service: ReinforcementLearningService,
        scheduler: Any,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the RL scheduler integration.

        Args:
            rl_service: ReinforcementLearningService instance
            scheduler: AdsScheduler instance
            config: Configuration dictionary
            logger: Logger instance
        """
        self.rl_service = rl_service
        self.scheduler = scheduler
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Default schedule configuration
        self.default_config = {
            "training": {
                "frequency": "daily",
                "hour": 2,  # 2 AM
                "minute": 0,
                "days_between": 1,
                "min_samples": 1000,
                "evaluation_episodes": 5,
            },
            "inference": {
                "frequency": "hourly",
                "minute": 30,  # Run at :30 of each hour
                "safe_hours": list(range(7, 22)),  # 7 AM to 10 PM
            },
            "evaluation": {
                "frequency": "daily",
                "hour": 1,  # 1 AM
                "minute": 0,
                "min_performance_ratio": 0.8,
            },
            "safety": {
                "check_frequency": "hourly",
                "minute": 15,
                "max_bid_change": 0.5,  # Maximum 50% change
                "max_budget_change": 0.3,  # Maximum 30% change
                "min_performance_ratio": 0.7,
            },
        }

        # Update with provided config
        self.default_config.update(self.config)

        # Track performance metrics
        self.performance_history = []
        self.safety_violations = []

        self.logger.info("RL Scheduler Integration initialized")

    def schedule_all_tasks(self) -> Dict[str, str]:
        """
        Schedule all RL service tasks.

        Returns:
            Dictionary of task IDs by category
        """
        task_ids = {}

        # Schedule training
        task_ids["training"] = self.schedule_training()

        # Schedule inference/optimization
        task_ids["inference"] = self.schedule_inference()

        # Schedule evaluation
        task_ids["evaluation"] = self.schedule_evaluation()

        # Schedule safety checks
        task_ids["safety"] = self.schedule_safety_checks()

        return task_ids

    def schedule_training(self) -> str:
        """Schedule the training loop."""
        config = self.default_config["training"]

        def training_task():
            try:
                # Check if we have enough new samples
                if self.rl_service.get_sample_count() < config["min_samples"]:
                    self.logger.info("Insufficient samples for training")
                    return

                # Run training with temporal considerations
                metrics = self.rl_service.train_temporal(
                    total_days=config["days_between"],
                    evaluation_days=1,
                    steps_per_day=24,
                    min_performance_threshold=0.8,
                )

                self.logger.info(f"Training completed: {metrics}")

                # Evaluate after training
                eval_metrics = self._evaluate_policy(episodes=config["evaluation_episodes"])

                if eval_metrics["mean_reward"] > self.rl_service.get_best_reward():
                    self.logger.info("New best model achieved")
                    self.rl_service.save_policy("best_model")

                return metrics

            except Exception as e:
                self.logger.error(f"Training task failed: {str(e)}")
                raise

        task_id = self.scheduler.add_task(
            name="RL Training",
            function=training_task,
            schedule_type=config["frequency"],
            hour=config["hour"],
            minute=config["minute"],
        )

        return task_id

    def schedule_inference(self) -> str:
        """Schedule the inference/optimization loop."""
        config = self.default_config["inference"]

        def inference_task():
            try:
                current_hour = datetime.now().hour

                # Only run during safe hours
                if current_hour not in config["safe_hours"]:
                    self.logger.info(f"Skipping inference at hour {current_hour}")
                    return

                # Get current state
                state = self.rl_service.get_current_state()

                # Get action with safety constraints
                action = self.rl_service.get_action(state, deterministic=True)

                # Apply action with safety checks
                success, metrics = self.rl_service.apply_action_safely(
                    action,
                    max_bid_change=self.default_config["safety"]["max_bid_change"],
                    max_budget_change=self.default_config["safety"]["max_budget_change"],
                )

                if success:
                    self.performance_history.append(metrics)
                    self.logger.info(f"Inference applied successfully: {metrics}")
                else:
                    self.logger.warning("Inference action rejected by safety checks")

                return metrics

            except Exception as e:
                self.logger.error(f"Inference task failed: {str(e)}")
                raise

        task_id = self.scheduler.add_task(
            name="RL Inference",
            function=inference_task,
            schedule_type=config["frequency"],
            minute=config["minute"],
        )

        return task_id

    def schedule_evaluation(self) -> str:
        """Schedule the evaluation loop."""
        config = self.default_config["evaluation"]

        def evaluation_task():
            try:
                # Run evaluation episodes
                metrics = self._evaluate_policy()

                # Check performance against baseline
                if metrics["performance_ratio"] < config["min_performance_ratio"]:
                    self.logger.warning(
                        f"Performance below threshold: {metrics['performance_ratio']:.2f} "
                        f"vs required {config['min_performance_ratio']}"
                    )
                    self._apply_safety_measures()

                return metrics

            except Exception as e:
                self.logger.error(f"Evaluation task failed: {str(e)}")
                raise

        task_id = self.scheduler.add_task(
            name="RL Evaluation",
            function=evaluation_task,
            schedule_type=config["frequency"],
            hour=config["hour"],
            minute=config["minute"],
        )

        return task_id

    def schedule_safety_checks(self) -> str:
        """Schedule safety check loop."""
        config = self.default_config["safety"]

        def safety_task():
            try:
                # Get current metrics
                current_metrics = self.rl_service.get_current_metrics()

                # Get baseline metrics
                baseline_metrics = self.rl_service.get_baseline_metrics()

                # Check for violations
                violations = self._check_safety_violations(current_metrics, baseline_metrics)

                if violations:
                    self.safety_violations.append(
                        {"timestamp": datetime.now(), "violations": violations}
                    )

                    self.logger.warning(f"Safety violations detected: {violations}")
                    self._apply_safety_measures()

                return {"violations": violations, "metrics": current_metrics}

            except Exception as e:
                self.logger.error(f"Safety check failed: {str(e)}")
                raise

        task_id = self.scheduler.add_task(
            name="RL Safety Check",
            function=safety_task,
            schedule_type=config["check_frequency"],
            minute=config["minute"],
        )

        return task_id

    def _evaluate_policy(self, episodes: int = 5) -> Dict[str, float]:
        """Run evaluation episodes."""
        try:
            eval_metrics = []

            for _ in range(episodes):
                metrics = self.rl_service.evaluate_episode(deterministic=True)
                eval_metrics.append(metrics)

            mean_metrics = {
                k: np.mean([m[k] for m in eval_metrics]) for k in eval_metrics[0].keys()
            }

            return {
                "mean_reward": mean_metrics["reward"],
                "performance_ratio": mean_metrics["performance_ratio"],
                "safety_violations": mean_metrics["safety_violations"],
            }

        except Exception as e:
            self.logger.error(f"Policy evaluation failed: {str(e)}")
            return {
                "mean_reward": float("-inf"),
                "performance_ratio": 0.0,
                "safety_violations": float("inf"),
            }

    def _check_safety_violations(
        self, current_metrics: Dict[str, float], baseline_metrics: Dict[str, float]
    ) -> List[str]:
        """Check for safety violations."""
        violations = []
        config = self.default_config["safety"]

        # Check cost increase
        if current_metrics.get("cost", 0) > baseline_metrics.get("cost", 0) * (
            1 + config["max_budget_change"]
        ):
            violations.append("excessive_cost_increase")

        # Check conversion rate drop
        if (
            current_metrics.get("conversion_rate", 0)
            < baseline_metrics.get("conversion_rate", 0) * config["min_performance_ratio"]
        ):
            violations.append("low_conversion_rate")

        # Check ROAS drop
        if (
            current_metrics.get("roas", 0)
            < baseline_metrics.get("roas", 0) * config["min_performance_ratio"]
        ):
            violations.append("low_roas")

        return violations

    def _apply_safety_measures(self):
        """Apply safety measures when violations occur."""
        try:
            # Load best performing model
            self.rl_service.load_policy("best_model")

            # Reset exploration parameters
            self.rl_service.reset_exploration()

            # Reduce action magnitude
            self.rl_service.set_action_constraints(
                max_bid_change=self.default_config["safety"]["max_bid_change"] * 0.5,
                max_budget_change=self.default_config["safety"]["max_budget_change"] * 0.5,
            )

            self.logger.info("Applied safety measures")

        except Exception as e:
            self.logger.error(f"Failed to apply safety measures: {str(e)}")

    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get historical performance data."""
        return self.performance_history

    def get_safety_violations(self) -> List[Dict[str, Any]]:
        """Get historical safety violations."""
        return self.safety_violations
