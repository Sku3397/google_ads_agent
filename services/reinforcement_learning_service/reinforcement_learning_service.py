"""
Reinforcement Learning Service for Google Ads Optimization

This module implements a reinforcement learning service that optimizes bidding strategies
and campaign management through deep RL algorithms (PPO, DQN, and A3C).
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import A2C as A3C

from services.base_service import BaseService
from .ads_environment import GoogleAdsEnv
from .policy_models import BiddingPolicy, KeywordPolicy


class ReinforcementLearningService(BaseService):
    """Service for applying reinforcement learning to Google Ads optimization"""

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the RL service.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance
            config: Configuration dictionary with RL parameters
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        # Load RL-specific config with defaults
        rl_base_config = {
            "algorithm": "PPO",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "target_update_interval": 500,
            "multi_objective": {
                "enabled": False,
                "objectives": ["conversions", "cost", "impressions"],
                "weights": [0.6, 0.2, 0.2],
            },
            "safety": {
                "max_bid_change": 0.3,  # 30% maximum bid change
                "max_budget_change": 0.2,  # 20% maximum budget change
                "min_performance_ratio": 0.8,  # 80% of baseline minimum
                "recovery_strategy": "rollback",  # rollback or conservative
            },
        }
        rl_user_config = config.get("rl_config", {}) if config else {}
        self.rl_config = {**rl_base_config, **rl_user_config}

        # Initialize environments
        self.train_env: Optional[Union[DummyVecEnv, VecNormalize]] = None
        self.eval_env: Optional[Union[DummyVecEnv, VecNormalize]] = None

        # Initialize policies
        self.bidding_policy = None
        self.keyword_policy = None

        # Track training progress
        self.training_metrics: Dict[str, Any] = {
            "episodes": 0,
            "total_timesteps": 0,
            "mean_reward": 0.0,
            "best_reward": float("-inf"),
            "start_time": None,
            "training_time": 0,
        }

        # For multi-objective RL
        self.objective_weights: np.ndarray = np.array(self.rl_config["multi_objective"]["weights"])

        # Action constraints for safety
        self.action_constraints: Dict[str, float] = {
            "max_bid_change": self.rl_config["safety"]["max_bid_change"],
            "max_budget_change": self.rl_config["safety"]["max_budget_change"],
        }

        # Sample buffer for experience collection
        self.sample_count: int = 0

        self.logger.info("ReinforcementLearningService initialized")

    def setup_environments(self, campaign_ids: List[str]) -> None:
        """
        Set up training and evaluation environments for the given campaigns.

        Args:
            campaign_ids: List of campaign IDs to include in training
        """
        try:
            # Create environments
            env_config = {
                "campaign_ids": campaign_ids,
                "ads_api": self.ads_api,
                "history_length": 30,
                "action_space_type": self.rl_config["algorithm"],
                "multi_objective": self.rl_config["multi_objective"]["enabled"],
                "objectives": self.rl_config["multi_objective"]["objectives"],
            }

            # Create base environments
            def train_env_fn():
                return GoogleAdsEnv(config={**env_config, "mode": "train"})

            def eval_env_fn():
                return GoogleAdsEnv(config={**env_config, "mode": "eval"})

            # Vectorize the environments
            train_vec_env = DummyVecEnv([train_env_fn])
            eval_vec_env = DummyVecEnv([eval_env_fn])

            # Normalize observations and rewards for more stable training
            self.train_env = VecNormalize(
                train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
            )

            self.eval_env = VecNormalize(
                eval_vec_env,
                norm_obs=True,
                norm_reward=False,  # Don't normalize reward for evaluation
                training=False,  # Don't update normalization stats during evaluation
            )

            self.logger.info(
                f"Environments created for {len(campaign_ids)} campaigns with algorithm {self.rl_config['algorithm']}"
            )

        except Exception as e:
            self.logger.error(f"Error setting up environments: {str(e)}")
            raise

    def initialize_policies(self) -> None:
        """Initialize or load policy networks for the selected algorithm"""
        try:
            if not self.train_env:
                raise ValueError("Environment must be set up before initializing policies")

            # Common kwargs for all algorithms
            common_kwargs = {
                "verbose": 1,
                "tensorboard_log": "./logs/tensorboard/",
            }

            if self.rl_config["algorithm"] == "PPO":
                self.bidding_policy = PPO(
                    "MlpPolicy",
                    self.train_env,
                    learning_rate=self.rl_config["learning_rate"],
                    n_steps=self.rl_config["n_steps"],
                    batch_size=self.rl_config["batch_size"],
                    n_epochs=self.rl_config["n_epochs"],
                    gamma=self.rl_config["gamma"],
                    gae_lambda=self.rl_config["gae_lambda"],
                    clip_range=self.rl_config["clip_range"],
                    ent_coef=self.rl_config["ent_coef"],
                    max_grad_norm=self.rl_config["max_grad_norm"],
                    vf_coef=self.rl_config["vf_coef"],
                    **common_kwargs,
                )
            elif self.rl_config["algorithm"] == "DQN":
                self.bidding_policy = DQN(
                    "MlpPolicy",
                    self.train_env,
                    learning_rate=self.rl_config["learning_rate"],
                    buffer_size=self.rl_config["buffer_size"],
                    learning_starts=self.rl_config["learning_starts"],
                    batch_size=self.rl_config["batch_size"],
                    gamma=self.rl_config["gamma"],
                    exploration_fraction=self.rl_config["exploration_fraction"],
                    exploration_initial_eps=self.rl_config["exploration_initial_eps"],
                    exploration_final_eps=self.rl_config["exploration_final_eps"],
                    target_update_interval=self.rl_config["target_update_interval"],
                    **common_kwargs,
                )
            elif self.rl_config["algorithm"] == "A3C":
                self.bidding_policy = A3C(
                    "MlpPolicy",
                    self.train_env,
                    learning_rate=self.rl_config["learning_rate"],
                    n_steps=self.rl_config["n_steps"],
                    gamma=self.rl_config["gamma"],
                    gae_lambda=self.rl_config["gae_lambda"],
                    ent_coef=self.rl_config["ent_coef"],
                    max_grad_norm=self.rl_config["max_grad_norm"],
                    vf_coef=self.rl_config["vf_coef"],
                    **common_kwargs,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.rl_config['algorithm']}")

            self.logger.info(f"Initialized {self.rl_config['algorithm']} policy")

        except Exception as e:
            self.logger.error(f"Error initializing policies: {str(e)}")
            raise

    def train_policy(self, total_timesteps: int = 100000, eval_freq: int = 10000) -> Dict[str, Any]:
        """
        Train the RL policies with advanced callbacks.

        Args:
            total_timesteps: Total number of environment steps to train for
            eval_freq: How often to evaluate the policy

        Returns:
            Training metrics dictionary
        """
        try:
            if not self.train_env or not self.bidding_policy:
                raise ValueError("Environments and policies must be initialized first")

            self.training_metrics["start_time"] = datetime.now()

            # Setup callbacks
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path="./models/best_model",
                log_path="./logs/eval",
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                callback_on_new_best=self._on_new_best_model,
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=eval_freq,
                save_path="./models/checkpoints/",
                name_prefix=f"{self.rl_config['algorithm']}_model",
                save_replay_buffer=True if self.rl_config["algorithm"] == "DQN" else False,
                save_vecnormalize=True,
            )

            callback_list = CallbackList([eval_callback, checkpoint_callback])

            # Train the policy
            self.bidding_policy.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                tb_log_name=f"{self.rl_config['algorithm']}_run_{datetime.now().strftime('%Y%m%d_%H%M')}",
            )

            # Update metrics
            if hasattr(self.train_env, "get_attr"):
                self.training_metrics["episodes"] += self.train_env.get_attr("episode_count")[0]

            self.training_metrics["total_timesteps"] += total_timesteps
            self.training_metrics["training_time"] += (
                datetime.now() - self.training_metrics["start_time"]
            ).total_seconds()

            # Final evaluation
            mean_reward, std_reward = evaluate_policy(
                self.bidding_policy, self.eval_env, n_eval_episodes=10, deterministic=True
            )

            self.training_metrics["mean_reward"] = mean_reward

            # Save final model
            self.save_policy("final_model")

            self.logger.info(
                f"Training completed: {total_timesteps} steps in "
                f"{self.training_metrics['training_time']:.2f} seconds, "
                f"final reward: {mean_reward:.2f}±{std_reward:.2f}"
            )

            return self.training_metrics

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def _on_new_best_model(self) -> None:
        """Callback that gets triggered when a new best model is found"""
        self.logger.info("New best model found during training")
        # Additional custom logic like notifying other services could go here

    def save_policy(self, name: str) -> None:
        """
        Save the current policy and environment normalization to disk.

        Args:
            name: Name to save the policy under
        """
        try:
            os.makedirs("./models", exist_ok=True)

            save_path = f"./models/{name}"
            self.bidding_policy.save(save_path)

            # Save VecNormalize stats
            if isinstance(self.train_env, VecNormalize):
                self.train_env.save(f"{save_path}_vecnormalize.pkl")

            self.logger.info(f"Policy saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving policy: {str(e)}")
            raise

    def load_policy(self, name: str) -> None:
        """
        Load a saved policy and environment normalization from disk.

        Args:
            name: Name of the policy to load
        """
        try:
            load_path = f"./models/{name}"

            # Load the appropriate policy class
            if self.rl_config["algorithm"] == "PPO":
                self.bidding_policy = PPO.load(load_path, env=self.train_env)
            elif self.rl_config["algorithm"] == "DQN":
                self.bidding_policy = DQN.load(load_path, env=self.train_env)
            elif self.rl_config["algorithm"] == "A3C":
                self.bidding_policy = A3C.load(load_path, env=self.train_env)

            # Load VecNormalize stats if available
            vec_normalize_path = f"{load_path}_vecnormalize.pkl"
            if os.path.exists(vec_normalize_path) and isinstance(self.train_env, VecNormalize):
                self.train_env = VecNormalize.load(vec_normalize_path, self.train_env.venv)
                # Don't update normalization stats during evaluation
                self.train_env.training = False

                # Also update eval_env
                if isinstance(self.eval_env, VecNormalize):
                    self.eval_env = VecNormalize.load(vec_normalize_path, self.eval_env.venv)
                    self.eval_env.training = False

            self.logger.info(f"Policy loaded from {load_path}")

        except Exception as e:
            self.logger.error(f"Error loading policy: {str(e)}")
            raise

    def get_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get an action from the policy for the given state with safety constraints.

        Args:
            state: Current environment state
            deterministic: Whether to use deterministic or stochastic actions

        Returns:
            Action array from the policy, constrained by safety limits
        """
        try:
            if self.bidding_policy is None:
                raise ValueError("Policy is not initialized or loaded")

            # Get action from policy
            action, _ = self.bidding_policy.predict(state, deterministic=deterministic)

            # Apply safety constraints
            action = self._apply_safety_constraints(action)

            return action

        except Exception as e:
            self.logger.error(f"Error getting action: {str(e)}")
            # Return a safe default action in case of error
            return np.zeros(self.train_env.action_space.shape[0])

    def _apply_safety_constraints(self, action: np.ndarray) -> np.ndarray:
        """
        Apply safety constraints to actions to prevent excessive changes.

        Args:
            action: Raw action from policy

        Returns:
            Constrained action
        """
        # If bidding action (assume first dimension controls bid multiplier)
        if action.shape[0] >= 1:
            # Get bid multiplier (assuming action[0] represents bid adjustment)
            bid_multiplier = action[0]

            # Apply constraint
            max_change = self.action_constraints["max_bid_change"]
            constrained_multiplier = np.clip(bid_multiplier, 1.0 - max_change, 1.0 + max_change)

            # Update action with constrained value
            action[0] = constrained_multiplier

        # If budget action (assume second dimension controls budget)
        if action.shape[0] >= 2:
            # Get budget multiplier
            budget_multiplier = action[1]

            # Apply constraint
            max_change = self.action_constraints["max_budget_change"]
            constrained_multiplier = np.clip(budget_multiplier, 1.0 - max_change, 1.0 + max_change)

            # Update action with constrained value
            action[1] = constrained_multiplier

        return action

    def update_from_feedback(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update the policy from a single step of environment feedback.

        Args:
            state: Starting state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether the episode ended
        """
        try:
            # Add to replay buffer if using DQN
            if self.rl_config["algorithm"] == "DQN" and hasattr(
                self.bidding_policy, "replay_buffer"
            ):
                self.bidding_policy.replay_buffer.add(state, action, reward, next_state, done)

            # Increment sample count
            self.sample_count += 1

            self.logger.debug(f"Feedback processed: reward={reward:.2f}, " f"done={done}")

        except Exception as e:
            self.logger.error(f"Error updating from feedback: {str(e)}")
            raise

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        return self.training_metrics.copy()

    def get_sample_count(self) -> int:
        """Get the current number of collected samples"""
        return self.sample_count

    def reset_exploration(self) -> None:
        """Reset exploration parameters to initial values for safety"""
        try:
            if self.rl_config["algorithm"] == "DQN" and hasattr(
                self.bidding_policy, "exploration_rate"
            ):
                self.bidding_policy.exploration_rate = self.rl_config["exploration_initial_eps"]

            self.logger.info("Reset exploration parameters to initial values")
        except Exception as e:
            self.logger.error(f"Error resetting exploration: {str(e)}")

    def set_action_constraints(self, max_bid_change: float, max_budget_change: float) -> None:
        """
        Set constraints on action magnitudes for safety.

        Args:
            max_bid_change: Maximum allowed bid change (0.0-1.0)
            max_budget_change: Maximum allowed budget change (0.0-1.0)
        """
        if not 0.0 <= max_bid_change <= 1.0:
            raise ValueError("max_bid_change must be between 0.0 and 1.0")

        if not 0.0 <= max_budget_change <= 1.0:
            raise ValueError("max_budget_change must be between 0.0 and 1.0")

        self.action_constraints["max_bid_change"] = max_bid_change
        self.action_constraints["max_budget_change"] = max_budget_change

        self.logger.info(
            f"Action constraints updated: max_bid_change={max_bid_change}, "
            f"max_budget_change={max_budget_change}"
        )

    def get_best_reward(self) -> float:
        """Get the best reward achieved during training"""
        return self.training_metrics["best_reward"]

    def get_current_state(self) -> np.ndarray:
        """Get the current state observation from the environment"""
        try:
            if not self.train_env:
                raise ValueError("Environment not initialized")

            # Get current state from environment
            if hasattr(self.train_env, "get_attr"):
                current_state = self.train_env.get_attr("_get_current_state")[0]()
            else:
                # Fallback for non-vectorized environments
                current_state = self.train_env.observation_space.sample()
                self.logger.warning("Using random state as fallback")

            return current_state

        except Exception as e:
            self.logger.error(f"Error getting current state: {str(e)}")
            # Return zero state as fallback
            return np.zeros(self.train_env.observation_space.shape)

    def apply_action_safely(
        self,
        action: np.ndarray,
        max_bid_change: Optional[float] = None,
        max_budget_change: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply an action with safety constraints and monitoring.

        Args:
            action: Action to apply
            max_bid_change: Override for maximum bid change
            max_budget_change: Override for maximum budget change

        Returns:
            success: Whether the action was successfully applied
            metrics: Performance metrics after applying the action
        """
        try:
            # Update constraints if provided
            if max_bid_change is not None:
                self.action_constraints["max_bid_change"] = max_bid_change

            if max_budget_change is not None:
                self.action_constraints["max_budget_change"] = max_budget_change

            # Apply safety constraints
            safe_action = self._apply_safety_constraints(action)

            # Get baseline metrics before applying
            baseline_metrics = self._get_current_metrics()

            # Apply the action to the environment
            if hasattr(self.train_env, "step"):
                next_state, reward, done, info = self.train_env.step(safe_action)
            else:
                # If we can't apply to environment, simulate application
                self.logger.warning("Cannot apply action to environment, simulating")
                next_state = self.get_current_state()
                reward = 0.0
                done = False
                info = {}

            # Get current metrics after applying
            current_metrics = self._get_current_metrics()

            # Calculate performance ratio
            performance_ratio = self._calculate_performance_ratio(current_metrics, baseline_metrics)

            # Check for safety violations
            safety_violations = self._check_safety_violations(current_metrics, baseline_metrics)

            metrics = {
                "action": safe_action.tolist(),
                "reward": reward,
                "performance_ratio": performance_ratio,
                "safety_violations": safety_violations,
                "current_metrics": current_metrics,
                "baseline_metrics": baseline_metrics,
            }

            # Determine success based on safety violations
            success = len(safety_violations) == 0

            if not success:
                self.logger.warning(
                    f"Action application failed due to safety violations: {safety_violations}"
                )
                self._handle_safety_violation(safety_violations)
            else:
                self.logger.info(
                    f"Action applied successfully: reward={reward:.2f}, "
                    f"performance_ratio={performance_ratio:.2f}"
                )

            return success, metrics

        except Exception as e:
            self.logger.error(f"Error applying action safely: {str(e)}")
            return False, {"error": str(e)}

    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics from Google Ads API"""
        try:
            if not self.ads_api:
                return {}

            # Get campaign performance data
            campaigns = self.ads_api.get_campaign_performance(days_ago=7)

            if not campaigns:
                return {}

            # Aggregate metrics across campaigns
            metrics = {
                "impressions": 0,
                "clicks": 0,
                "conversions": 0,
                "cost": 0,
                "average_cpc": 0,
            }

            for campaign in campaigns:
                for key in metrics:
                    if key in campaign:
                        metrics[key] += campaign[key]

            # Calculate derived metrics
            if metrics["impressions"] > 0:
                metrics["ctr"] = metrics["clicks"] / metrics["impressions"]
            else:
                metrics["ctr"] = 0

            if metrics["clicks"] > 0:
                metrics["conversion_rate"] = metrics["conversions"] / metrics["clicks"]
                metrics["cost_per_conversion"] = (
                    metrics["cost"] / metrics["conversions"] if metrics["conversions"] > 0 else 0
                )
            else:
                metrics["conversion_rate"] = 0
                metrics["cost_per_conversion"] = 0

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting current metrics: {str(e)}")
            return {}

    def _calculate_performance_ratio(
        self, current_metrics: Dict[str, float], baseline_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate performance ratio compared to baseline with multiple objectives.

        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics

        Returns:
            Weighted performance ratio
        """
        try:
            # Use multi-objective weights if enabled
            if self.rl_config["multi_objective"]["enabled"]:
                objectives = self.rl_config["multi_objective"]["objectives"]
                weights = self.rl_config["multi_objective"]["weights"]
            else:
                # Default single-objective focus on conversions
                objectives = ["conversions", "cost", "impressions"]
                weights = [0.7, 0.2, 0.1]

            weighted_ratio = 0.0
            total_weight = sum(weights)

            # Normalize weights
            normalized_weights = [w / total_weight for w in weights]

            for obj, weight in zip(objectives, normalized_weights):
                if obj in baseline_metrics and baseline_metrics[obj] > 0:
                    current_value = current_metrics.get(obj, 0)
                    baseline_value = baseline_metrics[obj]

                    # For cost, lower is better so invert the ratio
                    if obj == "cost":
                        if current_value > 0:
                            ratio = baseline_value / current_value
                        else:
                            ratio = 2.0  # Arbitrary high value if cost is zero
                    else:
                        ratio = current_value / baseline_value

                    weighted_ratio += ratio * weight

            return weighted_ratio

        except Exception as e:
            self.logger.error(f"Error calculating performance ratio: {str(e)}")
            return 0.0

    def _check_safety_violations(
        self, current_metrics: Dict[str, float], baseline_metrics: Dict[str, float]
    ) -> List[str]:
        """
        Check for safety violations in current performance.

        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics

        Returns:
            List of safety violation types detected
        """
        violations = []
        safety_config = self.rl_config["safety"]

        # Only check if we have both metrics
        if not current_metrics or not baseline_metrics:
            return violations

        # Check cost increase
        if current_metrics.get("cost", 0) > baseline_metrics.get("cost", 0) * (
            1 + safety_config["max_budget_change"]
        ):
            violations.append("excessive_cost_increase")

        # Check conversion rate drop
        baseline_conv_rate = baseline_metrics.get("conversion_rate", 0)
        current_conv_rate = current_metrics.get("conversion_rate", 0)

        if (
            baseline_conv_rate > 0
            and current_conv_rate < baseline_conv_rate * safety_config["min_performance_ratio"]
        ):
            violations.append("conversion_rate_drop")

        # Check CTR drop
        baseline_ctr = baseline_metrics.get("ctr", 0)
        current_ctr = current_metrics.get("ctr", 0)

        if baseline_ctr > 0 and current_ctr < baseline_ctr * safety_config["min_performance_ratio"]:
            violations.append("ctr_drop")

        # Check for abnormal CPC increase
        if current_metrics.get("average_cpc", 0) > baseline_metrics.get("average_cpc", 0) * (
            1 + safety_config["max_bid_change"]
        ):
            violations.append("cpc_increase")

        return violations

    def _handle_safety_violation(self, violations: List[str]) -> None:
        """
        Handle safety violations based on configured recovery strategy.

        Args:
            violations: List of safety violation types
        """
        recovery_strategy = self.rl_config["safety"]["recovery_strategy"]

        if recovery_strategy == "rollback":
            # Load best model and reset exploration
            self.load_policy("best_model")
            self.reset_exploration()
            self.logger.info("Applied rollback safety measure: loaded best model")

        elif recovery_strategy == "conservative":
            # Reduce action constraints by 50%
            self.set_action_constraints(
                max_bid_change=self.action_constraints["max_bid_change"] * 0.5,
                max_budget_change=self.action_constraints["max_budget_change"] * 0.5,
            )
            self.logger.info("Applied conservative safety measure: reduced action magnitude")

        else:
            self.logger.warning(f"Unknown recovery strategy: {recovery_strategy}")

    def get_baseline_metrics(self) -> Dict[str, float]:
        """Get baseline performance metrics for comparison"""
        return self._get_current_metrics()

    def evaluate_episode(self, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate a single episode with the current policy.

        Args:
            deterministic: Whether to use deterministic policy actions

        Returns:
            Episode metrics
        """
        try:
            if not self.eval_env:
                raise ValueError("Evaluation environment not initialized")

            state = self.eval_env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            # Get baseline metrics
            baseline_metrics = self._get_current_metrics()

            # Run the episode
            while not done:
                action = self.get_action(state, deterministic=deterministic)
                next_state, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                state = next_state
                steps += 1

            # Get current metrics
            current_metrics = self._get_current_metrics()

            # Calculate performance ratio
            performance_ratio = self._calculate_performance_ratio(current_metrics, baseline_metrics)

            # Check for safety violations
            safety_violations = self._check_safety_violations(current_metrics, baseline_metrics)

            return {
                "reward": total_reward,
                "steps": steps,
                "performance_ratio": performance_ratio,
                "safety_violations": len(safety_violations),
            }

        except Exception as e:
            self.logger.error(f"Error evaluating episode: {str(e)}")
            return {"reward": 0.0, "steps": 0, "performance_ratio": 0.0, "safety_violations": 0}

    def train_multi_objective(
        self,
        campaign_ids: List[str],
        objectives: List[str],
        weights: List[float],
        total_timesteps: int = 100000,
    ) -> Dict[str, Any]:
        """
        Train a multi-objective reinforcement learning policy.

        Args:
            campaign_ids: List of campaign IDs to train on
            objectives: List of objective metrics to optimize
            weights: Weights for each objective
            total_timesteps: Total environment steps to train for

        Returns:
            Training metrics
        """
        try:
            # Update multi-objective configuration
            self.rl_config["multi_objective"]["enabled"] = True
            self.rl_config["multi_objective"]["objectives"] = objectives
            self.rl_config["multi_objective"]["weights"] = weights

            # Normalize weights
            total_weight = sum(weights)
            self.objective_weights = np.array([w / total_weight for w in weights])

            # Setup environments with multi-objective reward
            self.setup_environments(campaign_ids)

            # Initialize policies
            self.initialize_policies()

            # Train the multi-objective policy
            return self.train_policy(total_timesteps=total_timesteps)

        except Exception as e:
            self.logger.error(f"Error in multi-objective training: {str(e)}")
            raise

    def train_multi_campaign(
        self,
        campaign_ids: List[str],
        campaign_budgets: Dict[str, float],
        total_timesteps: int = 100000,
    ) -> Dict[str, Any]:
        """
        Train across multiple campaigns with budget allocation optimization.

        Args:
            campaign_ids: List of campaign IDs to train on
            campaign_budgets: Dictionary of initial budgets by campaign ID
            total_timesteps: Total environment steps to train for

        Returns:
            Training metrics
        """
        try:
            # Enable multi-objective with focus on overall performance
            self.rl_config["multi_objective"]["enabled"] = True
            self.rl_config["multi_objective"]["objectives"] = [
                "conversions",
                "cost",
                "conversion_value",
                "revenue",
            ]
            self.rl_config["multi_objective"]["weights"] = [0.4, 0.2, 0.3, 0.1]

            # Setup environments with multi-campaign configuration
            env_config = {
                "campaign_ids": campaign_ids,
                "campaign_budgets": campaign_budgets,
                "multi_campaign": True,
            }

            # Setup and initialize
            self.setup_environments(campaign_ids)
            self.initialize_policies()

            # Train with additional callbacks for budget allocation
            return self.train_policy(total_timesteps=total_timesteps)

        except Exception as e:
            self.logger.error(f"Error in multi-campaign training: {str(e)}")
            raise
