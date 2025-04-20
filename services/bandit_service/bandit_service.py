"""
Bandit Service for Google Ads Management System

This module implements multi-armed bandit algorithms for dynamic budget allocation,
testing, and optimization of Google Ads campaigns, ad groups, and keywords.
Includes Thompson Sampling, UCB, and contextual bandits for optimized decision-making.
"""

import logging
import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# Optional imports for advanced functionality
try:
    import matplotlib
    import matplotlib.pyplot as plt

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Matplotlib not available, visualization features disabled")

try:
    import pymc3 as pm

    BAYESIAN_MODELING_AVAILABLE = True
except ImportError:
    try:
        import pymc as pm

        BAYESIAN_MODELING_AVAILABLE = True
    except ImportError:
        BAYESIAN_MODELING_AVAILABLE = False
        logging.warning("PyMC not available, advanced Bayesian bandits disabled")

from services.base_service import BaseService

logger = logging.getLogger(__name__)


class BanditAlgorithm(str, Enum):
    """Enum for supported bandit algorithms."""

    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "ucb"
    EPSILON_GREEDY = "epsilon_greedy"
    CONTEXTUAL = "contextual"
    DYNAMIC_THOMPSON = "dynamic_thompson"
    BAYESIAN = "bayesian"  # Added new Bayesian algorithm
    LIN_UCB = "lin_ucb"  # Added new LinUCB algorithm


class BanditService(BaseService):
    """
    Service for optimizing Google Ads using multi-armed bandit algorithms.

    This service manages various bandit instances for dynamic budget allocation,
    creative testing, and keyword optimization. It supports:
    - Thompson Sampling (Bayesian) bandits
    - Upper Confidence Bound (UCB) bandits
    - Epsilon-Greedy with dynamic exploration
    - Contextual bandits for personalized decisions
    - Non-stationary reward handling for adapting to market changes
    - Full Bayesian inference with MCMC for complex reward distributions
    - Linear UCB for contextual decisions with linear payoff models
    """

    def __init__(
        self, client: GoogleAdsClient, customer_id: str, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the BanditService.

        Args:
            client: The Google Ads API client
            customer_id: The Google Ads customer ID
            config: Optional configuration dictionary
        """
        super().__init__(client, customer_id)

        # Initialize configuration with defaults
        self.config = config or {}
        self.data_path = self.config.get("data_path", "data/bandits")
        os.makedirs(self.data_path, exist_ok=True)

        # Default priors for Beta distribution (Thompson Sampling)
        self.alpha_prior = self.config.get("alpha_prior", 1.0)  # Prior for success
        self.beta_prior = self.config.get("beta_prior", 1.0)  # Prior for failure

        # Parameters for different algorithms
        self.epsilon = self.config.get("epsilon", 0.1)  # For epsilon-greedy
        self.ucb_alpha = self.config.get("ucb_alpha", 1.0)  # Exploration parameter for UCB
        self.discount_factor = self.config.get(
            "discount_factor", 0.95
        )  # For non-stationary rewards

        # New parameters for advanced algorithms
        self.bayesian_samples = self.config.get("bayesian_samples", 1000)  # Number of MCMC samples
        self.lin_ucb_alpha = self.config.get(
            "lin_ucb_alpha", 1.0
        )  # Exploration parameter for LinUCB
        self.context_dim = self.config.get(
            "context_dim", 10
        )  # Default dimension for context features

        # Initialize bandits registry and metadata
        self.bandits = {}
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_bandits": 0,
            "total_updates": 0,
        }

        self.logger.info("BanditService initialized with enhanced algorithms")

    def initialize_bandit(
        self,
        name: str,
        arms: List[str],
        algorithm: Union[str, BanditAlgorithm] = BanditAlgorithm.THOMPSON_SAMPLING,
        context_features: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Initialize a new bandit with the specified arms and algorithm.

        Args:
            name: A unique name for the bandit
            arms: List of arm IDs (e.g., campaign IDs, keyword IDs)
            algorithm: The bandit algorithm to use
            context_features: List of feature names for contextual bandits
            metadata: Additional metadata about this bandit

        Returns:
            Dictionary with initialization status
        """
        try:
            bandit_id = f"{name}_{int(time.time())}"

            # Convert string algorithm to enum if needed
            if isinstance(algorithm, str):
                algorithm = BanditAlgorithm(algorithm)

            # Base bandit structure
            bandit = {
                "id": bandit_id,
                "name": name,
                "algorithm": algorithm,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "arms": {},
                "total_pulls": 0,
                "total_rewards": 0.0,
                "metadata": metadata or {},
            }

            # Initialize algorithm-specific parameters
            if algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
                bandit["arms"] = {
                    arm: {
                        "alpha": self.alpha_prior,
                        "beta": self.beta_prior,
                        "pulls": 0,
                        "rewards": 0.0,
                        "last_updated": datetime.now().isoformat(),
                    }
                    for arm in arms
                }

            elif algorithm == BanditAlgorithm.UCB:
                bandit["arms"] = {
                    arm: {
                        "mean_reward": 0.0,
                        "pulls": 0,
                        "rewards": 0.0,
                        "ucb_score": float("inf"),  # Initial score for first-time selection
                        "last_updated": datetime.now().isoformat(),
                    }
                    for arm in arms
                }
                bandit["ucb_alpha"] = self.ucb_alpha

            elif algorithm == BanditAlgorithm.EPSILON_GREEDY:
                bandit["arms"] = {
                    arm: {
                        "mean_reward": 0.0,
                        "pulls": 0,
                        "rewards": 0.0,
                        "last_updated": datetime.now().isoformat(),
                    }
                    for arm in arms
                }
                bandit["epsilon"] = self.epsilon
                bandit["min_epsilon"] = self.config.get("min_epsilon", 0.01)
                bandit["epsilon_decay"] = self.config.get("epsilon_decay", 0.999)

            elif algorithm == BanditAlgorithm.DYNAMIC_THOMPSON:
                bandit["arms"] = {
                    arm: {
                        "alpha": self.alpha_prior,
                        "beta": self.beta_prior,
                        "pulls": 0,
                        "rewards": 0.0,
                        "time_weights": [],  # For time-weighted updates
                        "last_updated": datetime.now().isoformat(),
                    }
                    for arm in arms
                }
                bandit["discount_factor"] = self.discount_factor

            elif algorithm == BanditAlgorithm.CONTEXTUAL:
                if not context_features:
                    raise ValueError("Context features must be provided for contextual bandits")

                bandit["arms"] = {
                    arm: {
                        "model": self._initialize_linear_model(len(context_features)),
                        "context_history": [],
                        "reward_history": [],
                        "pulls": 0,
                        "rewards": 0.0,
                        "last_updated": datetime.now().isoformat(),
                    }
                    for arm in arms
                }
                bandit["context_features"] = context_features

            # New algorithm: Bayesian bandit with full MCMC
            elif algorithm == BanditAlgorithm.BAYESIAN:
                if not BAYESIAN_MODELING_AVAILABLE:
                    raise ValueError("PyMC is required for Bayesian bandits but not available")

                bandit["arms"] = {
                    arm: {
                        "rewards": [],  # Store actual reward values
                        "pulls": 0,
                        "total_reward": 0.0,
                        "mean_estimate": 0.0,
                        "std_estimate": 1.0,  # Initial uncertainty
                        "last_updated": datetime.now().isoformat(),
                    }
                    for arm in arms
                }
                bandit["mcmc_samples"] = self.bayesian_samples
                bandit["prior_mean"] = self.config.get("prior_mean", 0.0)
                bandit["prior_std"] = self.config.get("prior_std", 1.0)
                bandit["model_trace"] = None  # Will store latest MCMC trace

            # New algorithm: Linear UCB for contextual bandits
            elif algorithm == BanditAlgorithm.LIN_UCB:
                if not context_features:
                    raise ValueError("Context features must be provided for LinUCB")

                context_dim = len(context_features)
                bandit["arms"] = {
                    arm: {
                        "A": np.identity(context_dim),  # A matrix for LinUCB
                        "b": np.zeros(context_dim),  # b vector for LinUCB
                        "theta": np.zeros(context_dim),  # Estimated parameter vector
                        "context_history": [],
                        "reward_history": [],
                        "pulls": 0,
                        "total_reward": 0.0,
                        "last_updated": datetime.now().isoformat(),
                    }
                    for arm in arms
                }
                bandit["context_features"] = context_features
                bandit["context_dim"] = context_dim
                bandit["alpha"] = self.lin_ucb_alpha  # Exploration parameter

            # Add to bandits registry
            self.bandits[bandit_id] = bandit

            # Update metadata
            self.metadata["total_bandits"] += 1
            self.metadata["last_updated"] = datetime.now().isoformat()

            self.logger.info(
                f"Initialized bandit '{bandit_id}' with algorithm '{algorithm}' and {len(arms)} arms"
            )

            return {
                "status": "success",
                "bandit_id": bandit_id,
                "message": f"Bandit '{name}' initialized with algorithm '{algorithm}' and {len(arms)} arms",
            }

        except Exception as e:
            error_message = f"Error initializing bandit: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def _initialize_linear_model(self, n_features: int) -> Dict[str, np.ndarray]:
        """Initialize a simple linear model for contextual bandits."""
        return {
            "weights": np.zeros(n_features),
            "A_inv": np.identity(n_features),  # Inverse design matrix
            "b": np.zeros(n_features),  # Response vector
            "alpha": 1.0,  # Regularization parameter
        }

    def update_bandit(
        self, bandit_id: str, arm_id: str, reward: float, context: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Update bandit statistics after observing a reward.

        Args:
            bandit_id: The ID of the bandit to update
            arm_id: The arm ID to update
            reward: The observed reward
            context: Optional context features for contextual bandits

        Returns:
            Dictionary with update status
        """
        try:
            if bandit_id not in self.bandits:
                raise ValueError(f"Bandit {bandit_id} not found")

            bandit = self.bandits[bandit_id]
            if arm_id not in bandit["arms"]:
                raise ValueError(f"Arm {arm_id} not found in bandit {bandit_id}")

            # Update common metrics
            bandit["total_pulls"] += 1
            bandit["total_rewards"] += reward
            bandit["last_updated"] = datetime.now().isoformat()

            # Call algorithm-specific update method
            algorithm = bandit["algorithm"]

            if algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
                self._update_thompson(bandit, arm_id, reward)
            elif algorithm == BanditAlgorithm.UCB:
                self._update_ucb(bandit, arm_id, reward)
            elif algorithm == BanditAlgorithm.EPSILON_GREEDY:
                self._update_epsilon_greedy(bandit, arm_id, reward)
            elif algorithm == BanditAlgorithm.DYNAMIC_THOMPSON:
                self._update_dynamic_thompson(bandit, arm_id, reward)
            elif algorithm == BanditAlgorithm.CONTEXTUAL:
                if not context:
                    raise ValueError("Context must be provided for contextual bandits")
                self._update_contextual(bandit, arm_id, reward, context)
            elif algorithm == BanditAlgorithm.BAYESIAN:
                self._update_bayesian(bandit, arm_id, reward)
            elif algorithm == BanditAlgorithm.LIN_UCB:
                if not context:
                    raise ValueError("Context must be provided for LinUCB")
                self._update_lin_ucb(bandit, arm_id, reward, context)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # Update metadata
            self.metadata["total_updates"] += 1
            self.metadata["last_updated"] = datetime.now().isoformat()

            self.logger.info(f"Updated bandit {bandit_id}, arm {arm_id} with reward {reward:.4f}")

            return {
                "status": "success",
                "bandit_id": bandit_id,
                "arm_id": arm_id,
                "reward": reward,
                "message": f"Successfully updated bandit {bandit_id}, arm {arm_id}",
            }

        except Exception as e:
            error_message = f"Error updating bandit: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def select_arm(
        self, bandit_id: str, context: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Select the next arm to pull based on the bandit algorithm.

        Args:
            bandit_id: The ID of the bandit to select from
            context: Optional context features for contextual bandits

        Returns:
            Dictionary with selected arm and selection details
        """
        try:
            if bandit_id not in self.bandits:
                raise ValueError(f"Bandit {bandit_id} not found")

            bandit = self.bandits[bandit_id]
            algorithm = bandit["algorithm"]

            # Call the appropriate algorithm-specific method
            if algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
                return self._select_arm_thompson(bandit)
            elif algorithm == BanditAlgorithm.UCB:
                return self._select_arm_ucb(bandit)
            elif algorithm == BanditAlgorithm.EPSILON_GREEDY:
                return self._select_arm_epsilon_greedy(bandit)
            elif algorithm == BanditAlgorithm.DYNAMIC_THOMPSON:
                return self._select_arm_dynamic_thompson(bandit)
            elif algorithm == BanditAlgorithm.CONTEXTUAL:
                if not context:
                    raise ValueError("Context must be provided for contextual bandits")
                return self._select_arm_contextual(bandit, context)
            elif algorithm == BanditAlgorithm.BAYESIAN:
                return self._select_arm_bayesian(bandit)
            elif algorithm == BanditAlgorithm.LIN_UCB:
                if not context:
                    raise ValueError("Context must be provided for LinUCB")
                return self._select_arm_lin_ucb(bandit, context)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        except Exception as e:
            error_message = f"Error selecting arm: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def _select_arm_thompson(self, bandit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an arm using Thompson Sampling.

        This method samples from each arm's Beta distribution and selects
        the arm with the highest sampled value.
        """
        samples = {}
        sample_values = {}

        for arm_id, arm in bandit["arms"].items():
            # Sample from Beta(alpha, beta) distribution
            sample = np.random.beta(arm["alpha"], arm["beta"])
            samples[arm_id] = sample
            sample_values[arm_id] = {"alpha": arm["alpha"], "beta": arm["beta"], "sample": sample}

        # Select arm with highest sample
        selected_arm = max(samples, key=samples.get)

        self.logger.info(
            f"Thompson Sampling selected arm {selected_arm} with sample value {samples[selected_arm]:.4f}"
        )

        return {
            "status": "success",
            "selected_arm": selected_arm,
            "selection_value": samples[selected_arm],
            "all_samples": sample_values,
            "algorithm": "THOMPSON_SAMPLING",
            "selection_type": "exploitation",
            "rationale": f"Thompson Sampling chose arm with highest sample value: {samples[selected_arm]:.4f}",
        }

    def _select_arm_ucb(self, bandit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an arm using Upper Confidence Bound.

        UCB balances exploration and exploitation by selecting the arm
        with the highest upper confidence bound.
        """
        total_pulls = bandit["total_pulls"]
        ucb_values = {}

        # Small constant to avoid division by zero
        min_pulls = 0.0001

        for arm_id, arm in bandit["arms"].items():
            # Calculate UCB score
            mean_reward = arm["mean_reward"]
            pulls = max(arm["pulls"], min_pulls)

            # If total_pulls is 0, use a high value for exploration
            if total_pulls == 0:
                ucb = float("inf")
            else:
                exploration_term = bandit["ucb_alpha"] * np.sqrt(2 * np.log(total_pulls) / pulls)
                ucb = mean_reward + exploration_term

            # Update the UCB score in the bandit
            arm["ucb_score"] = ucb
            ucb_values[arm_id] = {"mean_reward": mean_reward, "pulls": pulls, "ucb_score": ucb}

        # Select arm with highest UCB
        selected_arm = max(ucb_values, key=lambda x: ucb_values[x]["ucb_score"])

        self.logger.info(
            f"UCB selected arm {selected_arm} with UCB value {ucb_values[selected_arm]['ucb_score']:.4f}"
        )

        return {
            "status": "success",
            "selected_arm": selected_arm,
            "selection_value": ucb_values[selected_arm]["ucb_score"],
            "all_ucb_values": ucb_values,
            "algorithm": "UCB",
            "selection_type": "balanced",
            "rationale": f"UCB chose arm with highest upper confidence bound: {ucb_values[selected_arm]['ucb_score']:.4f}",
        }

    def _select_arm_epsilon_greedy(self, bandit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an arm using Epsilon-Greedy.

        With probability (1-epsilon), select the best arm (exploitation).
        With probability epsilon, select a random arm (exploration).
        """
        epsilon = bandit["epsilon"]
        arm_ids = list(bandit["arms"].keys())

        # Decide whether to explore or exploit
        if np.random.random() < epsilon:
            # Exploration: random arm
            selected_arm = np.random.choice(arm_ids)
            selection_type = "exploration"
            rationale = f"Random exploration with epsilon={epsilon:.4f}"
        else:
            # Exploitation: best arm so far
            mean_rewards = {arm_id: arm["mean_reward"] for arm_id, arm in bandit["arms"].items()}

            if all(reward == 0 for reward in mean_rewards.values()):
                # If all rewards are 0, choose randomly
                selected_arm = np.random.choice(arm_ids)
                selection_type = "exploration"
                rationale = "All arms have equal mean reward (0), choosing randomly"
            else:
                # Choose arm with highest mean reward
                selected_arm = max(mean_rewards, key=mean_rewards.get)
                selection_type = "exploitation"
                rationale = f"Chose arm with highest mean reward: {mean_rewards[selected_arm]:.4f}"

        # Decay epsilon if configured
        if "epsilon_decay" in bandit and bandit["epsilon"] > bandit.get("min_epsilon", 0.01):
            bandit["epsilon"] *= bandit["epsilon_decay"]

        self.logger.info(f"Epsilon-Greedy ({selection_type}) selected arm {selected_arm}")

        return {
            "status": "success",
            "selected_arm": selected_arm,
            "current_epsilon": bandit["epsilon"],
            "algorithm": "EPSILON_GREEDY",
            "selection_type": selection_type,
            "rationale": rationale,
        }

    def _select_arm_dynamic_thompson(self, bandit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an arm using Dynamic Thompson Sampling.

        Similar to standard Thompson Sampling, but with time-discounted rewards
        to adapt to non-stationary environments.
        """
        samples = {}
        sample_values = {}

        for arm_id, arm in bandit["arms"].items():
            # Sample from Beta(alpha, beta) distribution
            sample = np.random.beta(arm["alpha"], arm["beta"])
            samples[arm_id] = sample
            sample_values[arm_id] = {
                "alpha": arm["alpha"],
                "beta": arm["beta"],
                "time_weighted_reward": sum(arm.get("time_weights", [])),
                "sample": sample,
            }

        # Select arm with highest sample
        selected_arm = max(samples, key=samples.get)

        self.logger.info(
            f"Dynamic Thompson Sampling selected arm {selected_arm} with sample value {samples[selected_arm]:.4f}"
        )

        return {
            "status": "success",
            "selected_arm": selected_arm,
            "selection_value": samples[selected_arm],
            "all_samples": sample_values,
            "algorithm": "DYNAMIC_THOMPSON",
            "selection_type": "time_adaptive",
            "rationale": f"Dynamic Thompson Sampling chose arm with highest sample value: {samples[selected_arm]:.4f}",
        }

    def _select_arm_contextual(
        self, bandit: Dict[str, Any], context: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Select an arm using contextual bandit.

        Uses a linear model to predict rewards for each arm given the context.
        """
        context_features = bandit["context_features"]

        # Convert context dict to feature vector
        feature_vector = np.array([context.get(feature, 0.0) for feature in context_features])

        predicted_rewards = {}
        confidence_bounds = {}

        for arm_id, arm in bandit["arms"].items():
            model = arm["model"]

            # Predicted reward: w^T x
            mean_reward = np.dot(model["weights"], feature_vector)

            # Confidence bound: alpha * sqrt(x^T A^-1 x)
            confidence = model["alpha"] * np.sqrt(
                np.dot(feature_vector, np.dot(model["A_inv"], feature_vector))
            )

            # Upper confidence bound
            ucb = mean_reward + confidence

            predicted_rewards[arm_id] = mean_reward
            confidence_bounds[arm_id] = {"mean": mean_reward, "confidence": confidence, "ucb": ucb}

        # Select arm with highest UCB
        selected_arm = max(confidence_bounds, key=lambda x: confidence_bounds[x]["ucb"])

        self.logger.info(
            f"Contextual bandit selected arm {selected_arm} with UCB {confidence_bounds[selected_arm]['ucb']:.4f}"
        )

        return {
            "status": "success",
            "selected_arm": selected_arm,
            "context": context,
            "predicted_rewards": predicted_rewards,
            "confidence_bounds": confidence_bounds,
            "algorithm": "CONTEXTUAL",
            "selection_type": "contextual",
            "rationale": f"Contextual bandit chose arm with highest predicted reward given context",
        }

    def _select_arm_bayesian(self, bandit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an arm using full Bayesian modeling with MCMC.

        This method uses PyMC to perform Bayesian inference on the reward
        distributions of each arm and selects based on posterior samples.
        """
        if not BAYESIAN_MODELING_AVAILABLE:
            return {
                "status": "failed",
                "message": "PyMC is required for Bayesian bandits but not available",
            }

        try:
            arms = bandit["arms"]
            arm_ids = list(arms.keys())

            # If we need to run MCMC first (or refresh it)
            if bandit.get("model_trace") is None or bandit.get("refresh_model", False):
                # Only run MCMC if we have enough data
                has_data = any(len(arm["rewards"]) > 3 for arm in arms.values())

                if has_data:
                    # Create PyMC model
                    with pm.Model() as model:
                        # Priors for each arm
                        arm_means = {}
                        for arm_id in arm_ids:
                            # Use informative prior if we have data
                            if len(arms[arm_id]["rewards"]) > 0:
                                mu = (
                                    np.mean(arms[arm_id]["rewards"])
                                    if arms[arm_id]["rewards"]
                                    else bandit["prior_mean"]
                                )
                                sigma = (
                                    np.std(arms[arm_id]["rewards"])
                                    if len(arms[arm_id]["rewards"]) > 1
                                    else bandit["prior_std"]
                                )
                                # Prevent too small sigma
                                sigma = max(sigma, 0.01)
                            else:
                                mu = bandit["prior_mean"]
                                sigma = bandit["prior_std"]

                            arm_means[arm_id] = pm.Normal(f"mean_{arm_id}", mu=mu, sigma=sigma)

                        # Likelihood for each arm
                        for arm_id in arm_ids:
                            if len(arms[arm_id]["rewards"]) > 0:
                                pm.Normal(
                                    f"obs_{arm_id}",
                                    mu=arm_means[arm_id],
                                    sigma=sigma,
                                    observed=arms[arm_id]["rewards"],
                                )

                        # Run MCMC
                        trace = pm.sample(
                            bandit["mcmc_samples"], tune=500, chains=2, progressbar=False
                        )

                        # Store trace in bandit
                        bandit["model_trace"] = trace
                        bandit["refresh_model"] = False

                        # Update arm estimates
                        for arm_id in arm_ids:
                            if f"mean_{arm_id}" in trace:
                                arms[arm_id]["mean_estimate"] = np.mean(trace[f"mean_{arm_id}"])
                                arms[arm_id]["std_estimate"] = np.std(trace[f"mean_{arm_id}"])

                # If we don't have enough data, use simple estimates
                else:
                    for arm_id in arm_ids:
                        arm = arms[arm_id]
                        if arm["pulls"] > 0:
                            arm["mean_estimate"] = arm["total_reward"] / arm["pulls"]
                        else:
                            arm["mean_estimate"] = bandit["prior_mean"]
                        arm["std_estimate"] = bandit["prior_std"]

            # Sample from posterior for each arm
            samples = {}
            for arm_id in arm_ids:
                arm = arms[arm_id]
                # If we have a trace, sample from it
                if (
                    bandit.get("model_trace") is not None
                    and f"mean_{arm_id}" in bandit["model_trace"]
                ):
                    # Get a random sample from the trace
                    trace_samples = bandit["model_trace"][f"mean_{arm_id}"]
                    sample_idx = np.random.randint(len(trace_samples))
                    sample = trace_samples[sample_idx]
                else:
                    # Otherwise sample from normal using estimates
                    sample = np.random.normal(arm["mean_estimate"], arm["std_estimate"])

                samples[arm_id] = sample

            # Select arm with highest sample
            selected_arm = max(samples, key=samples.get)

            # Create detailed report
            arm_stats = {}
            for arm_id in arm_ids:
                arm = arms[arm_id]
                arm_stats[arm_id] = {
                    "mean_estimate": arm["mean_estimate"],
                    "std_estimate": arm["std_estimate"],
                    "sample": samples[arm_id],
                    "pulls": arm["pulls"],
                    "data_points": len(arm.get("rewards", [])),
                }

            self.logger.info(
                f"Bayesian MCMC selected arm {selected_arm} with sample value {samples[selected_arm]:.4f}"
            )

            return {
                "status": "success",
                "selected_arm": selected_arm,
                "selection_value": samples[selected_arm],
                "arm_statistics": arm_stats,
                "algorithm": "BAYESIAN",
                "selection_type": "posterior_sampling",
                "rationale": f"Bayesian model with MCMC chose arm with highest posterior sample: {samples[selected_arm]:.4f}",
            }

        except Exception as e:
            error_message = f"Error in Bayesian arm selection: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def _select_arm_lin_ucb(
        self, bandit: Dict[str, Any], context: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Select an arm using Linear UCB algorithm.

        LinUCB is a contextual bandit algorithm that models the reward as a linear
        function of the context features.

        Args:
            bandit: The bandit data structure
            context: Context features for the current decision

        Returns:
            Dictionary with selected arm and selection details
        """
        try:
            arms = bandit["arms"]
            arm_ids = list(arms.keys())
            context_features = bandit["context_features"]

            # Convert context to feature vector
            x_t = np.array([context.get(feature, 0.0) for feature in context_features])

            # Compute UCB for each arm
            ucb_values = {}
            for arm_id in arm_ids:
                arm = arms[arm_id]

                # Get parameters
                A = arm["A"]  # Precision matrix
                b = arm["b"]  # Response vector

                # Compute theta (best linear predictor)
                A_inv = np.linalg.inv(A)
                theta = A_inv.dot(b)
                arm["theta"] = theta  # Store for later use

                # Expected reward (mean)
                expected_reward = theta.dot(x_t)

                # Confidence width
                alpha = bandit["alpha"]
                confidence_width = alpha * np.sqrt(x_t.dot(A_inv).dot(x_t))

                # UCB score
                ucb = expected_reward + confidence_width

                ucb_values[arm_id] = {
                    "expected_reward": float(expected_reward),
                    "confidence_width": float(confidence_width),
                    "ucb": float(ucb),
                }

            # Select arm with highest UCB
            selected_arm = max(ucb_values, key=lambda a: ucb_values[a]["ucb"])

            self.logger.info(
                f"LinUCB selected arm {selected_arm} with UCB value {ucb_values[selected_arm]['ucb']:.4f}"
            )

            return {
                "status": "success",
                "selected_arm": selected_arm,
                "ucb_values": ucb_values,
                "algorithm": "LIN_UCB",
                "selection_type": "contextual",
                "context": context,
                "rationale": f"LinUCB chose arm with highest UCB score: {ucb_values[selected_arm]['ucb']:.4f}",
            }

        except Exception as e:
            error_message = f"Error in LinUCB arm selection: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def _update_thompson(self, bandit: Dict[str, Any], arm_id: str, reward: float) -> None:
        """Update Thompson Sampling parameters for an arm."""
        arm = bandit["arms"][arm_id]

        # Ensure reward is between 0 and 1 for Beta distribution
        reward = max(0.0, min(1.0, reward))

        # Update Beta distribution parameters
        arm["alpha"] += reward
        arm["beta"] += 1.0 - reward

        # Update reward tracking
        arm["pulls"] += 1
        arm["rewards"] += reward
        arm["last_updated"] = datetime.now().isoformat()

    def _update_ucb(self, bandit: Dict[str, Any], arm_id: str, reward: float) -> None:
        """Update UCB parameters for an arm."""
        arm = bandit["arms"][arm_id]

        # Update pull count and total reward
        arm["pulls"] += 1
        arm["rewards"] += reward

        # Update mean reward
        arm["mean_reward"] = arm["rewards"] / arm["pulls"]
        arm["last_updated"] = datetime.now().isoformat()

    def _update_epsilon_greedy(self, bandit: Dict[str, Any], arm_id: str, reward: float) -> None:
        """Update Epsilon-Greedy parameters for an arm."""
        arm = bandit["arms"][arm_id]

        # Update pull count and total reward
        arm["pulls"] += 1
        arm["rewards"] += reward

        # Update mean reward
        arm["mean_reward"] = arm["rewards"] / arm["pulls"]
        arm["last_updated"] = datetime.now().isoformat()

    def _update_dynamic_thompson(self, bandit: Dict[str, Any], arm_id: str, reward: float) -> None:
        """Update Dynamic Thompson Sampling parameters for an arm."""
        arm = bandit["arms"][arm_id]
        discount = bandit["discount_factor"]

        # Ensure reward is between 0 and 1 for Beta distribution
        reward = max(0.0, min(1.0, reward))

        # Apply time-discounting to previous rewards
        if "time_weights" not in arm:
            arm["time_weights"] = []

        # Apply discount to existing weights and add new reward
        arm["time_weights"] = [w * discount for w in arm["time_weights"]] + [reward]

        # Recalculate effective alpha and beta from time-weighted rewards
        weighted_rewards = arm["time_weights"]
        effective_pulls = len(weighted_rewards)
        effective_rewards = sum(weighted_rewards)

        arm["alpha"] = self.alpha_prior + effective_rewards
        arm["beta"] = self.beta_prior + (effective_pulls - effective_rewards)

        # Update reward tracking
        arm["pulls"] += 1
        arm["rewards"] += reward
        arm["last_updated"] = datetime.now().isoformat()

    def _update_contextual(
        self, bandit: Dict[str, Any], arm_id: str, reward: float, context: Dict[str, float]
    ) -> None:
        """Update contextual bandit model for an arm."""
        arm = bandit["arms"][arm_id]
        context_features = bandit["context_features"]

        # Convert context dict to feature vector
        feature_vector = np.array([context.get(feature, 0.0) for feature in context_features])

        # Get the linear model
        model = arm["model"]

        # Update the model using ridge regression update rule
        A_inv = model["A_inv"]
        b = model["b"]

        # Sherman-Morrison formula for rank-1 update of A_inv
        v = np.dot(A_inv, feature_vector)
        denominator = 1 + np.dot(feature_vector, v)
        A_inv_new = A_inv - np.outer(v, v) / denominator

        # Update A_inv and b
        model["A_inv"] = A_inv_new
        model["b"] = b + feature_vector * reward

        # Update weights
        model["weights"] = np.dot(A_inv_new, model["b"])

        # Store context and reward in history
        arm["context_history"].append(feature_vector.tolist())
        arm["reward_history"].append(reward)

        # Update reward tracking
        arm["pulls"] += 1
        arm["rewards"] += reward
        arm["last_updated"] = datetime.now().isoformat()

    def _update_bayesian(self, bandit: Dict[str, Any], arm_id: str, reward: float) -> None:
        """
        Update Bayesian bandit with new reward observation.

        Args:
            bandit: The bandit data structure
            arm_id: The arm ID to update
            reward: The observed reward
        """
        arm = bandit["arms"][arm_id]

        # Update basic statistics
        arm["pulls"] += 1
        arm["total_reward"] += reward
        arm["rewards"].append(reward)
        arm["last_updated"] = datetime.now().isoformat()

        # Compute simple estimates
        arm["mean_estimate"] = arm["total_reward"] / arm["pulls"]

        # Flag that we need to refresh the model
        bandit["refresh_model"] = True

    def _update_lin_ucb(
        self, bandit: Dict[str, Any], arm_id: str, reward: float, context: Dict[str, float]
    ) -> None:
        """
        Update Linear UCB bandit with new observation.

        Args:
            bandit: The bandit data structure
            arm_id: The arm ID to update
            reward: The observed reward
            context: The context features that were used for the decision
        """
        arm = bandit["arms"][arm_id]
        context_features = bandit["context_features"]

        # Convert context to feature vector
        x_t = np.array([context.get(feature, 0.0) for feature in context_features])

        # Update basic statistics
        arm["pulls"] += 1
        arm["total_reward"] += reward
        arm["context_history"].append(x_t.tolist())
        arm["reward_history"].append(reward)
        arm["last_updated"] = datetime.now().isoformat()

        # Update A and b for LinUCB
        arm["A"] += np.outer(x_t, x_t)
        arm["b"] += reward * x_t

        # Re-compute theta
        A_inv = np.linalg.inv(arm["A"])
        arm["theta"] = A_inv.dot(arm["b"])

    def allocate_budget(self, bandit_id: str, total_budget: float) -> Dict[str, Any]:
        """
        Allocate budget across arms based on bandit model.

        Args:
            bandit_id: The ID of the bandit to use for allocation
            total_budget: Total budget to allocate

        Returns:
            Dictionary with budget allocations
        """
        try:
            if bandit_id not in self.bandits:
                raise ValueError(f"Bandit {bandit_id} not found")

            bandit = self.bandits[bandit_id]
            algorithm = bandit["algorithm"]
            arms = list(bandit["arms"].keys())

            # Ensure positive budget
            if total_budget <= 0:
                raise ValueError("Total budget must be positive")

            # Get bandit stats for allocation decisions
            stats_result = self.get_bandit_stats(bandit_id)
            if stats_result["status"] != "success":
                raise ValueError(stats_result["message"])

            stats = stats_result["stats"]

            # Allocation strategy depends on algorithm and state
            if algorithm in [BanditAlgorithm.THOMPSON_SAMPLING, BanditAlgorithm.DYNAMIC_THOMPSON]:
                # Thompson Sampling: Allocate based on probability of being best
                # Run multiple samples to estimate probability
                best_counts = {arm: 0 for arm in arms}
                num_samples = 10000

                for _ in range(num_samples):
                    samples = {}
                    for arm_id in arms:
                        arm_stats = stats["arm_stats"][arm_id]
                        alpha = arm_stats["alpha"]
                        beta = arm_stats["beta"]
                        samples[arm_id] = np.random.beta(alpha, beta)

                    best_arm = max(samples, key=samples.get)
                    best_counts[best_arm] += 1

                # Convert counts to probabilities
                probs = {arm: count / num_samples for arm, count in best_counts.items()}

                # Allocate based on probability, with minimum allocation for exploration
                min_allocation = total_budget * 0.1 / len(arms)
                remaining_budget = total_budget - (min_allocation * len(arms))

                allocations = {arm: min_allocation for arm in arms}
                for arm, prob in probs.items():
                    allocations[arm] += remaining_budget * prob

            elif algorithm == BanditAlgorithm.UCB:
                # UCB: Allocate proportionally to UCB scores
                ucb_scores = {arm: stats["arm_stats"][arm].get("ucb_score", 0.0) for arm in arms}

                # Normalize scores (handling negatives or zeros)
                min_score = min(ucb_scores.values())
                if min_score < 0:
                    adjusted_scores = {
                        arm: score - min_score + 0.01 for arm, score in ucb_scores.items()
                    }
                else:
                    adjusted_scores = {arm: max(score, 0.01) for arm, score in ucb_scores.items()}

                total_score = sum(adjusted_scores.values())

                # Allocate proportionally to adjusted UCB scores
                allocations = {
                    arm: (score / total_score) * total_budget
                    for arm, score in adjusted_scores.items()
                }

            elif algorithm == BanditAlgorithm.EPSILON_GREEDY:
                # Epsilon-Greedy: Allocate (1-epsilon) to best arm, epsilon equally among all
                epsilon = bandit.get("epsilon", self.epsilon)

                mean_rewards = {
                    arm: stats["arm_stats"][arm].get("mean_reward", 0.0) for arm in arms
                }

                # Find best arm
                if all(reward == 0 for reward in mean_rewards.values()):
                    # If all rewards are 0, allocate equally
                    allocations = {arm: total_budget / len(arms) for arm in arms}
                else:
                    best_arm = max(mean_rewards, key=mean_rewards.get)

                    # Allocate epsilon uniformly for exploration
                    exploration_budget = total_budget * epsilon
                    exploitation_budget = total_budget - exploration_budget

                    # Base allocation for all arms (exploration)
                    allocations = {arm: exploration_budget / len(arms) for arm in arms}

                    # Additional allocation for best arm (exploitation)
                    allocations[best_arm] += exploitation_budget

            elif algorithm == BanditAlgorithm.CONTEXTUAL:
                # For contextual bandits: This would require context, so use a sensible default
                # Allocate based on historical performance
                mean_rewards = {
                    arm: stats["arm_stats"][arm].get("mean_reward", 0.0) for arm in arms
                }

                # Handle the case where all rewards are 0
                if all(reward == 0 for reward in mean_rewards.values()):
                    allocations = {arm: total_budget / len(arms) for arm in arms}
                else:
                    # Normalize rewards
                    min_reward = min(mean_rewards.values())
                    if min_reward < 0:
                        adjusted_rewards = {
                            arm: reward - min_reward + 0.01 for arm, reward in mean_rewards.items()
                        }
                    else:
                        adjusted_rewards = {
                            arm: max(reward, 0.01) for arm, reward in mean_rewards.items()
                        }

                    total_adjusted = sum(adjusted_rewards.values())

                    # Allocate proportionally to adjusted rewards
                    allocations = {
                        arm: (reward / total_adjusted) * total_budget
                        for arm, reward in adjusted_rewards.items()
                    }

            else:
                # Default: Allocate equally
                allocations = {arm: total_budget / len(arms) for arm in arms}

            # Round to 2 decimal places for budget clarity
            allocations = {arm: round(budget, 2) for arm, budget in allocations.items()}

            # Log allocations
            self.logger.info(f"Allocated budget for bandit {bandit_id}: {allocations}")

            return {
                "status": "success",
                "allocations": allocations,
                "total_budget": total_budget,
                "message": f"Budget allocated across {len(allocations)} arms using {algorithm}",
            }

        except Exception as e:
            error_message = f"Error allocating budget: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def optimize_campaigns(
        self, campaign_ids: List[str], total_budget: float, days: int = 30
    ) -> Dict[str, Any]:
        """
        Optimize budget allocation across campaigns using historical performance.

        Args:
            campaign_ids: List of campaign IDs to optimize
            total_budget: Total budget to allocate
            days: Days of historical data to consider

        Returns:
            Dictionary with optimization results
        """
        try:
            # Check if campaigns exist
            if not campaign_ids:
                raise ValueError("No campaign IDs provided")

            # Create a unique name for this optimization
            optimization_name = f"campaign_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get historical performance data
            performance_data = self._get_campaign_performance(campaign_ids, days)

            if not performance_data:
                return {
                    "status": "failed",
                    "message": "No performance data available for campaigns",
                }

            # Initialize bandit with campaign IDs
            init_result = self.initialize_bandit(
                name=optimization_name,
                arms=campaign_ids,
                algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
                metadata={
                    "type": "campaign_optimization",
                    "days": days,
                    "total_budget": total_budget,
                },
            )

            if init_result["status"] != "success":
                return init_result

            bandit_id = init_result["bandit_id"]

            # Update bandit with historical performance
            for campaign in performance_data:
                campaign_id = campaign.get("campaign_id", "")

                if campaign_id not in campaign_ids:
                    continue

                # Calculate CTR and conversion rates
                impressions = campaign.get("impressions", 0)
                clicks = campaign.get("clicks", 0)
                conversions = campaign.get("conversions", 0)

                ctr = clicks / impressions if impressions > 0 else 0
                conv_rate = conversions / clicks if clicks > 0 else 0

                # Use conversion rate as reward (normalized to 0-1)
                # This prioritizes campaigns with better conversion performance
                reward = min(1.0, max(0.0, conv_rate))

                # Update bandit with this campaign's performance
                self.update_bandit(bandit_id, campaign_id, reward)

            # Allocate budget based on bandit model
            allocation_result = self.allocate_budget(bandit_id, total_budget)

            if allocation_result["status"] != "success":
                return allocation_result

            # Prepare final recommendation with additional context
            recommendations = []
            allocations = allocation_result["allocations"]

            for campaign in performance_data:
                campaign_id = campaign.get("campaign_id", "")

                if campaign_id not in allocations:
                    continue

                # Calculate key metrics
                clicks = campaign.get("clicks", 0)
                conversions = campaign.get("conversions", 0)
                cost = campaign.get("cost", 0.0)

                cpa = cost / conversions if conversions > 0 else 0
                current_budget = campaign.get("budget", 0.0)
                recommended_budget = allocations[campaign_id]

                recommendations.append(
                    {
                        "campaign_id": campaign_id,
                        "campaign_name": campaign.get("campaign_name", ""),
                        "current_budget": current_budget,
                        "recommended_budget": recommended_budget,
                        "change_pct": (
                            ((recommended_budget - current_budget) / current_budget * 100)
                            if current_budget > 0
                            else 0
                        ),
                        "conversions": conversions,
                        "cost": cost,
                        "cpa": cpa,
                        "rationale": self._generate_budget_rationale(
                            current_budget, recommended_budget, conversions, cost
                        ),
                    }
                )

            # Sort recommendations by recommended budget (descending)
            recommendations.sort(key=lambda x: x["recommended_budget"], reverse=True)

            return {
                "status": "success",
                "bandit_id": bandit_id,
                "total_budget": total_budget,
                "recommendations": recommendations,
                "message": f"Budget optimization complete for {len(campaign_ids)} campaigns",
            }

        except Exception as e:
            error_message = f"Error optimizing campaigns: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def _get_campaign_performance(self, campaign_ids: List[str], days: int) -> List[Dict[str, Any]]:
        """
        Get campaign performance data from the Google Ads API.

        Args:
            campaign_ids: List of campaign IDs to get data for
            days: Number of days of historical data

        Returns:
            List of campaign performance dictionaries
        """
        try:
            # Using the Google Ads API client to get campaign performance
            ga_service = self.client.get_service("GoogleAdsService")

            # Construct query for campaign performance
            campaign_ids_str = ", ".join([f"'{cid}'" for cid in campaign_ids])

            query = f"""
                SELECT
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign_budget.amount_micros,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.conversions,
                    metrics.cost_micros
                FROM campaign
                WHERE campaign.id IN ({campaign_ids_str})
                AND segments.date DURING LAST_{days}_DAYS
                AND campaign.status = 'ENABLED'
            """

            try:
                response = ga_service.search(customer_id=self.customer_id, query=query)

                # Process the response
                campaigns = []

                for row in response:
                    campaign = {
                        "campaign_id": row.campaign.id,
                        "campaign_name": row.campaign.name,
                        "status": row.campaign.status.name,
                        "budget": (
                            row.campaign_budget.amount_micros / 1000000
                            if hasattr(row, "campaign_budget")
                            else 0.0
                        ),
                        "impressions": row.metrics.impressions,
                        "clicks": row.metrics.clicks,
                        "conversions": row.metrics.conversions,
                        "cost": row.metrics.cost_micros / 1000000,
                    }
                    campaigns.append(campaign)

                return campaigns

            except GoogleAdsException as ex:
                self.logger.error(f"Google Ads API error: {ex}")
                return []

        except Exception as e:
            self.logger.error(f"Error getting campaign performance: {str(e)}")
            return []

    def _generate_budget_rationale(
        self, current_budget: float, recommended_budget: float, conversions: float, cost: float
    ) -> str:
        """Generate human-readable rationale for budget recommendation."""
        if recommended_budget > current_budget * 1.2:
            return "Significant budget increase recommended due to strong conversion performance."
        elif recommended_budget > current_budget:
            return "Moderate budget increase to capitalize on positive performance."
        elif recommended_budget < current_budget * 0.8:
            return "Significant budget reduction recommended due to underperformance."
        elif recommended_budget < current_budget:
            return "Slight budget reduction to optimize overall portfolio performance."
        else:
            return "Maintain current budget based on stable performance."

    def optimize_ad_creatives(
        self, ad_group_id: str, creative_ids: List[str], days: int = 30
    ) -> Dict[str, Any]:
        """
        Optimize ad creative selection using a multi-armed bandit.

        Args:
            ad_group_id: The ad group ID containing the creatives
            creative_ids: List of ad creative IDs to optimize
            days: Days of historical data to consider

        Returns:
            Dictionary with optimization results
        """
        try:
            # Check if creatives exist
            if not creative_ids:
                raise ValueError("No creative IDs provided")

            # Create a unique name for this optimization
            optimization_name = (
                f"creative_opt_{ad_group_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Get historical performance data for the creatives
            performance_data = self._get_creative_performance(ad_group_id, creative_ids, days)

            if not performance_data:
                return {
                    "status": "failed",
                    "message": "No performance data available for ad creatives",
                }

            # Initialize bandit with creative IDs
            init_result = self.initialize_bandit(
                name=optimization_name,
                arms=creative_ids,
                algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
                metadata={
                    "type": "creative_optimization",
                    "ad_group_id": ad_group_id,
                    "days": days,
                },
            )

            if init_result["status"] != "success":
                return init_result

            bandit_id = init_result["bandit_id"]

            # Update bandit with historical performance
            for creative in performance_data:
                creative_id = creative.get("ad_id", "")

                if creative_id not in creative_ids:
                    continue

                # Calculate CTR and conversion rates
                impressions = creative.get("impressions", 0)
                clicks = creative.get("clicks", 0)
                conversions = creative.get("conversions", 0)

                ctr = clicks / impressions if impressions > 0 else 0
                conv_rate = conversions / clicks if clicks > 0 else 0

                # Combine CTR and conversion rate as reward
                # This balances both engagement and conversion performance
                reward = min(1.0, max(0.0, (ctr * 0.3) + (conv_rate * 0.7)))

                # Update bandit with this creative's performance
                self.update_bandit(bandit_id, creative_id, reward)

            # Select the best creative
            selection_result = self.select_arm(bandit_id)

            if selection_result["status"] != "success":
                return selection_result

            selected_creative = selection_result["selected_arm"]

            # Get bandit stats for more detailed results
            stats_result = self.get_bandit_stats(bandit_id)

            if stats_result["status"] != "success":
                return stats_result

            # Prepare final recommendation with additional context
            recommendations = []
            arms_stats = stats_result["stats"]["arm_stats"]

            for creative in performance_data:
                creative_id = creative.get("ad_id", "")

                if creative_id not in arms_stats:
                    continue

                # Get key metrics
                impressions = creative.get("impressions", 0)
                clicks = creative.get("clicks", 0)
                conversions = creative.get("conversions", 0)

                ctr = clicks / impressions if impressions > 0 else 0
                conv_rate = conversions / clicks if clicks > 0 else 0

                # Calculate confidence based on bandit model
                arm_stats = arms_stats[creative_id]
                confidence = 0.0

                if "alpha" in arm_stats and "beta" in arm_stats:
                    # For Beta distribution, confidence is related to sample size
                    alpha = arm_stats["alpha"]
                    beta = arm_stats["beta"]
                    total_samples = alpha + beta - self.alpha_prior - self.beta_prior
                    confidence = min(1.0, total_samples / 100)  # Scale to 0-1

                recommendations.append(
                    {
                        "creative_id": creative_id,
                        "headline": creative.get("headline", ""),
                        "description": creative.get("description", ""),
                        "impressions": impressions,
                        "clicks": clicks,
                        "conversions": conversions,
                        "ctr": ctr,
                        "conversion_rate": conv_rate,
                        "is_recommended": creative_id == selected_creative,
                        "confidence": confidence,
                        "rationale": self._generate_creative_rationale(
                            creative_id, selected_creative, ctr, conv_rate
                        ),
                    }
                )

            # Sort recommendations by is_recommended (True first), then by CTR
            recommendations.sort(key=lambda x: (-int(x["is_recommended"]), -x["ctr"]))

            return {
                "status": "success",
                "bandit_id": bandit_id,
                "ad_group_id": ad_group_id,
                "recommended_creative": selected_creative,
                "recommendations": recommendations,
                "message": f"Creative optimization complete for {len(creative_ids)} ads",
            }

        except Exception as e:
            error_message = f"Error optimizing ad creatives: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def _get_creative_performance(
        self, ad_group_id: str, creative_ids: List[str], days: int
    ) -> List[Dict[str, Any]]:
        """
        Get ad creative performance data from the Google Ads API.

        Args:
            ad_group_id: The ad group ID containing the creatives
            creative_ids: List of ad creative IDs to get data for
            days: Number of days of historical data

        Returns:
            List of ad creative performance dictionaries
        """
        try:
            # Using the Google Ads API client to get ad performance
            ga_service = self.client.get_service("GoogleAdsService")

            # Construct query for ad performance
            creative_ids_str = ", ".join([f"'{cid}'" for cid in creative_ids])

            query = f"""
                SELECT
                    ad_group_ad.ad.id,
                    ad_group_ad.ad.expanded_text_ad.headline_part1,
                    ad_group_ad.ad.expanded_text_ad.headline_part2,
                    ad_group_ad.ad.expanded_text_ad.description,
                    ad_group_ad.status,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.conversions,
                    metrics.cost_micros
                FROM ad_group_ad
                WHERE ad_group.id = '{ad_group_id}'
                AND ad_group_ad.ad.id IN ({creative_ids_str})
                AND segments.date DURING LAST_{days}_DAYS
                AND ad_group_ad.status = 'ENABLED'
            """

            try:
                response = ga_service.search(customer_id=self.customer_id, query=query)

                # Process the response
                creatives = []

                for row in response:
                    ad = row.ad_group_ad.ad
                    creative = {
                        "ad_id": ad.id,
                        "headline": f"{ad.expanded_text_ad.headline_part1} - {ad.expanded_text_ad.headline_part2}",
                        "description": ad.expanded_text_ad.description,
                        "status": row.ad_group_ad.status.name,
                        "impressions": row.metrics.impressions,
                        "clicks": row.metrics.clicks,
                        "conversions": row.metrics.conversions,
                        "cost": row.metrics.cost_micros / 1000000,
                    }
                    creatives.append(creative)

                return creatives

            except GoogleAdsException as ex:
                self.logger.error(f"Google Ads API error: {ex}")
                return []

        except Exception as e:
            self.logger.error(f"Error getting ad creative performance: {str(e)}")
            return []

    def _generate_creative_rationale(
        self, creative_id: str, selected_id: str, ctr: float, conv_rate: float
    ) -> str:
        """Generate human-readable rationale for creative recommendation."""
        if creative_id == selected_id:
            if conv_rate > 0.1:
                return "Recommended due to strong conversion performance."
            elif ctr > 0.05:
                return "Recommended due to high click-through rate."
            else:
                return "Recommended based on optimal balance of engagement and conversions."
        else:
            if ctr < 0.01:
                return "Not recommended due to low click-through rate."
            elif conv_rate < 0.01:
                return "Not recommended due to poor conversion performance."
            else:
                return "Performs adequately but not optimal based on collected data."

    def run(self, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the BanditService with the specified parameters.

        This method is called by the scheduler to perform routine operations.

        Args:
            parameters: Optional parameters for the run

        Returns:
            Dictionary with run results
        """
        start_time = datetime.now()

        try:
            self.logger.info("Starting BanditService run")

            # Default parameters
            params = parameters or {}
            action = params.get("action", "maintain")

            if action == "optimize_campaigns":
                # Optimize campaign budgets
                campaign_ids = params.get("campaign_ids", [])
                total_budget = params.get("total_budget", 0.0)
                days = params.get("days", 30)

                if not campaign_ids or total_budget <= 0:
                    self.logger.error("Invalid parameters for campaign optimization")
                    return {
                        "status": "failed",
                        "message": "Invalid parameters: campaign_ids and total_budget required",
                    }

                result = self.optimize_campaigns(campaign_ids, total_budget, days)

            elif action == "optimize_creatives":
                # Optimize ad creatives
                ad_group_id = params.get("ad_group_id", "")
                creative_ids = params.get("creative_ids", [])
                days = params.get("days", 30)

                if not ad_group_id or not creative_ids:
                    self.logger.error("Invalid parameters for creative optimization")
                    return {
                        "status": "failed",
                        "message": "Invalid parameters: ad_group_id and creative_ids required",
                    }

                result = self.optimize_ad_creatives(ad_group_id, creative_ids, days)

            elif action == "maintain":
                # Maintenance tasks: save current state, clean up old bandits
                # Save current state
                save_result = self.save_bandits()

                # Clean up old bandits (older than 30 days)
                threshold = datetime.now() - timedelta(days=30)
                old_bandits = []

                for bandit_id, bandit in list(self.bandits.items()):
                    created_at = datetime.fromisoformat(
                        bandit.get("created_at", datetime.now().isoformat())
                    )
                    if created_at < threshold and not bandit.get("metadata", {}).get(
                        "persistent", False
                    ):
                        old_bandits.append(bandit_id)
                        del self.bandits[bandit_id]

                self.metadata["total_bandits"] = len(self.bandits)
                self.metadata["last_updated"] = datetime.now().isoformat()

                result = {
                    "status": "success",
                    "message": f"Maintenance completed, saved state and removed {len(old_bandits)} old bandits",
                    "save_result": save_result,
                    "removed_bandits": old_bandits,
                }

            else:
                result = {"status": "failed", "message": f"Unknown action: {action}"}

            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"BanditService run completed in {execution_time:.2f}s")

            # Add execution time to result
            if isinstance(result, dict):
                result["execution_time_seconds"] = execution_time

            return result

        except Exception as e:
            error_message = f"Error running BanditService: {str(e)}"
            self.logger.error(error_message)

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "status": "failed",
                "message": error_message,
                "execution_time_seconds": execution_time,
            }

    def visualize_bandit_performance(
        self, bandit_id: str, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize the performance of a bandit over time.

        Args:
            bandit_id: The ID of the bandit to visualize
            output_file: Optional output file path to save the visualization

        Returns:
            Dictionary with visualization results and statistics
        """
        if not VISUALIZATION_AVAILABLE:
            return {
                "status": "failed",
                "message": "Matplotlib not available, visualization features disabled",
            }

        try:
            if bandit_id not in self.bandits:
                raise ValueError(f"Bandit {bandit_id} not found")

            bandit = self.bandits[bandit_id]
            arms = bandit["arms"]
            algorithm = bandit["algorithm"]

            # Create figure with multiple subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Bandit Performance: {bandit['name']} ({algorithm})", fontsize=16)

            # 1. Mean rewards by arm
            ax1 = axs[0, 0]
            arm_names = []
            mean_rewards = []
            std_errors = []
            pulls = []

            for arm_id, arm in arms.items():
                arm_names.append(arm_id)
                pulls.append(arm["pulls"])

                if arm["pulls"] > 0:
                    if algorithm == BanditAlgorithm.BAYESIAN:
                        mean_rewards.append(arm["mean_estimate"])
                        std_errors.append(arm["std_estimate"] / np.sqrt(arm["pulls"]))
                    elif algorithm in (
                        BanditAlgorithm.THOMPSON_SAMPLING,
                        BanditAlgorithm.DYNAMIC_THOMPSON,
                    ):
                        # Beta distribution mean and std
                        alpha, beta = arm["alpha"], arm["beta"]
                        mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0
                        std = (
                            np.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
                            if (alpha + beta) > 1
                            else 0
                        )
                        mean_rewards.append(mean)
                        std_errors.append(std)
                    else:
                        # Default case
                        mean = arm["rewards"] / arm["pulls"] if "rewards" in arm else 0
                        mean_rewards.append(mean)
                        std_errors.append(0)  # No std info for some algorithms
                else:
                    mean_rewards.append(0)
                    std_errors.append(0)

            # Sort by mean reward
            sorted_indices = np.argsort(mean_rewards)[::-1]
            sorted_arm_names = [arm_names[i] for i in sorted_indices]
            sorted_mean_rewards = [mean_rewards[i] for i in sorted_indices]
            sorted_std_errors = [std_errors[i] for i in sorted_indices]

            # Plot mean rewards
            bars = ax1.bar(
                np.arange(len(sorted_arm_names)),
                sorted_mean_rewards,
                yerr=sorted_std_errors,
                align="center",
                alpha=0.7,
                color="skyblue",
                ecolor="black",
                capsize=10,
            )
            ax1.set_xticks(np.arange(len(sorted_arm_names)))
            ax1.set_xticklabels(sorted_arm_names, rotation=45, ha="right")
            ax1.set_xlabel("Arms")
            ax1.set_ylabel("Mean Reward")
            ax1.set_title("Mean Reward by Arm")

            # Add pull counts as text
            sorted_pulls = [pulls[arm_names.index(arm)] for arm in sorted_arm_names]
            for i, (bar, pull_count) in enumerate(zip(bars, sorted_pulls)):
                ax1.text(
                    i,
                    bar.get_height() + 0.02,
                    f"n={pull_count}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # 2. Reward distributions for top arms (if available)
            ax2 = axs[0, 1]

            # Only for algorithms that store full reward history
            if algorithm in (
                BanditAlgorithm.BAYESIAN,
                BanditAlgorithm.CONTEXTUAL,
                BanditAlgorithm.LIN_UCB,
            ):
                # Get top 3 arms with enough data
                top_arms = [
                    (arm_id, arms[arm_id])
                    for arm_id in sorted_arm_names
                    if len(arms[arm_id].get("rewards", [])) > 5
                ][:3]

                if top_arms:
                    for i, (arm_id, arm) in enumerate(top_arms):
                        rewards = arm.get("rewards", [])
                        if rewards:
                            # Create kernel density estimate
                            try:
                                kde = stats.gaussian_kde(rewards)
                                x = np.linspace(min(rewards), max(rewards), 100)
                                ax2.plot(x, kde(x), label=f"{arm_id} (n={len(rewards)})")
                                ax2.hist(rewards, bins=10, alpha=0.3, density=True)
                            except Exception as e:
                                # Fall back to simple histogram if KDE fails
                                ax2.hist(
                                    rewards,
                                    bins=10,
                                    alpha=0.3,
                                    density=True,
                                    label=f"{arm_id} (n={len(rewards)})",
                                )

                ax2.set_xlabel("Reward Value")
                ax2.set_ylabel("Density")
                ax2.set_title("Reward Distributions (Top Arms)")
                ax2.legend()
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Reward distributions not available for this algorithm",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])

            # 3. Arm pull counts
            ax3 = axs[1, 0]

            # Sort by number of pulls
            pull_sort_indices = np.argsort(pulls)[::-1]
            sorted_by_pulls_arm_names = [arm_names[i] for i in pull_sort_indices]
            sorted_by_pulls_pull_counts = [pulls[i] for i in pull_sort_indices]

            # Plot pull counts
            ax3.bar(
                np.arange(len(sorted_by_pulls_arm_names)),
                sorted_by_pulls_pull_counts,
                align="center",
                alpha=0.7,
                color="lightgreen",
            )
            ax3.set_xticks(np.arange(len(sorted_by_pulls_arm_names)))
            ax3.set_xticklabels(sorted_by_pulls_arm_names, rotation=45, ha="right")
            ax3.set_xlabel("Arms")
            ax3.set_ylabel("Number of Pulls")
            ax3.set_title("Arm Pull Distribution")

            # 4. Estimated probability of being best arm (for Bayesian)
            ax4 = axs[1, 1]

            if algorithm == BanditAlgorithm.BAYESIAN and bandit.get("model_trace") is not None:
                # Compute probability of being best for each arm
                trace = bandit["model_trace"]
                arm_names = []
                best_probs = []

                for arm_id in sorted_arm_names:
                    if f"mean_{arm_id}" in trace:
                        arm_names.append(arm_id)

                        # Count how often this arm has the highest mean in the trace
                        arm_means = {
                            a: trace[f"mean_{a}"] for a in sorted_arm_names if f"mean_{a}" in trace
                        }

                        if arm_means:
                            # Stack all means and find which arm is best at each sample
                            all_means = np.vstack(list(arm_means.values()))
                            best_indices = np.argmax(all_means, axis=0)
                            arm_index = list(arm_means.keys()).index(arm_id)
                            prob_best = np.mean(best_indices == arm_index)
                            best_probs.append(prob_best)

                if arm_names and best_probs:
                    # Sort by probability of being best
                    sort_indices = np.argsort(best_probs)[::-1]
                    sorted_arm_names = [arm_names[i] for i in sort_indices]
                    sorted_best_probs = [best_probs[i] for i in sort_indices]

                    # Plot
                    ax4.bar(
                        np.arange(len(sorted_arm_names)),
                        sorted_best_probs,
                        align="center",
                        alpha=0.7,
                        color="salmon",
                    )
                    ax4.set_xticks(np.arange(len(sorted_arm_names)))
                    ax4.set_xticklabels(sorted_arm_names, rotation=45, ha="right")
                    ax4.set_xlabel("Arms")
                    ax4.set_ylabel("Probability of Being Best")
                    ax4.set_title("Estimated Probability of Being Best Arm")
                    ax4.set_ylim(0, 1)
                else:
                    ax4.text(
                        0.5,
                        0.5,
                        "Not enough MCMC data to estimate probabilities",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    ax4.set_xticks([])
                    ax4.set_yticks([])
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "Probability estimates not available for this algorithm",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax4.set_xticks([])
                ax4.set_yticks([])

            # Adjust layout and save/show plot
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
                return {
                    "status": "success",
                    "message": f"Visualization saved to {output_file}",
                    "statistics": {
                        "algorithm": algorithm,
                        "num_arms": len(arms),
                        "total_pulls": bandit["total_pulls"],
                        "total_rewards": bandit["total_rewards"],
                        "top_arm": sorted_arm_names[0] if sorted_arm_names else None,
                        "top_arm_mean_reward": (
                            sorted_mean_rewards[0] if sorted_mean_rewards else None
                        ),
                    },
                }
            else:
                # Return figure for display in notebooks/UI
                return {
                    "status": "success",
                    "message": "Visualization created",
                    "figure": fig,
                    "statistics": {
                        "algorithm": algorithm,
                        "num_arms": len(arms),
                        "total_pulls": bandit["total_pulls"],
                        "total_rewards": bandit["total_rewards"],
                        "top_arm": sorted_arm_names[0] if sorted_arm_names else None,
                        "top_arm_mean_reward": (
                            sorted_mean_rewards[0] if sorted_mean_rewards else None
                        ),
                    },
                }

        except Exception as e:
            error_message = f"Error visualizing bandit: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def visualize_convergence(
        self, bandit_id: str, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize the convergence of a bandit over time.

        Args:
            bandit_id: The ID of the bandit to visualize
            output_file: Optional output file path to save the visualization

        Returns:
            Dictionary with convergence analysis results
        """
        if not VISUALIZATION_AVAILABLE:
            return {
                "status": "failed",
                "message": "Matplotlib not available, visualization features disabled",
            }

        try:
            if bandit_id not in self.bandits:
                raise ValueError(f"Bandit {bandit_id} not found")

            bandit = self.bandits[bandit_id]
            algorithm = bandit["algorithm"]

            # Check if we have time-series data for convergence
            has_time_series = False

            if algorithm in (
                BanditAlgorithm.BAYESIAN,
                BanditAlgorithm.CONTEXTUAL,
                BanditAlgorithm.LIN_UCB,
            ):
                # These algorithms store full history
                for arm in bandit["arms"].values():
                    if len(arm.get("reward_history", [])) > 10:
                        has_time_series = True
                        break

            if not has_time_series:
                return {
                    "status": "warning",
                    "message": "Not enough historical data for convergence analysis",
                }

            # Create figure
            fig, axs = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f"Bandit Convergence: {bandit['name']} ({algorithm})", fontsize=16)

            # 1. Cumulative rewards over time
            ax1 = axs[0]

            # Get arms with history
            arms_with_history = []
            for arm_id, arm in bandit["arms"].items():
                if len(arm.get("reward_history", [])) > 0:
                    arms_with_history.append((arm_id, arm))

            # Plot cumulative rewards
            for arm_id, arm in arms_with_history:
                rewards = arm.get("reward_history", [])
                if rewards:
                    cumulative_rewards = np.cumsum(rewards)
                    ax1.plot(
                        np.arange(len(cumulative_rewards)),
                        cumulative_rewards,
                        label=f"{arm_id} (n={len(rewards)})",
                    )

            ax1.set_xlabel("Number of Pulls")
            ax1.set_ylabel("Cumulative Reward")
            ax1.set_title("Cumulative Rewards Over Time")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.7)

            # 2. Moving average of rewards (window size 10)
            ax2 = axs[1]
            window_size = min(
                10,
                min(
                    [
                        len(arm.get("reward_history", []))
                        for _, arm in arms_with_history
                        if arm.get("reward_history")
                    ]
                ),
            )

            if window_size > 1:
                for arm_id, arm in arms_with_history:
                    rewards = arm.get("reward_history", [])
                    if len(rewards) > window_size:
                        # Calculate moving average
                        moving_avg = [
                            np.mean(rewards[i : i + window_size])
                            for i in range(len(rewards) - window_size + 1)
                        ]
                        ax2.plot(
                            np.arange(window_size - 1, len(rewards)),
                            moving_avg,
                            label=f"{arm_id} (MA{window_size})",
                        )

                ax2.set_xlabel("Number of Pulls")
                ax2.set_ylabel(f"Moving Average Reward (window={window_size})")
                ax2.set_title("Reward Convergence Over Time")
                ax2.legend()
                ax2.grid(True, linestyle="--", alpha=0.7)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Not enough data for moving average analysis",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])

            # Adjust layout and save/show plot
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close(fig)
                return {
                    "status": "success",
                    "message": f"Convergence visualization saved to {output_file}",
                }
            else:
                # Return figure for display
                return {
                    "status": "success",
                    "message": "Convergence visualization created",
                    "figure": fig,
                }

        except Exception as e:
            error_message = f"Error visualizing convergence: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def compare_campaigns_with_bandits(
        self,
        campaign_ids: List[str],
        days: int = 30,
        create_new_bandit: bool = True,
        bandit_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare campaign performance and create/update a bandit for optimization.

        Args:
            campaign_ids: List of campaign IDs to compare
            days: Number of days of historical data to analyze
            create_new_bandit: Whether to create a new bandit or update existing
            bandit_name: Name for the new bandit (if creating)

        Returns:
            Dictionary with comparison results and bandit details
        """
        try:
            # Get campaign performance data
            performance_data = self._get_campaign_performance(campaign_ids, days)

            if not performance_data:
                return {
                    "status": "warning",
                    "message": "No performance data found for the selected campaigns",
                }

            # Calculate metrics for comparison
            campaign_metrics = {}
            for data in performance_data:
                campaign_id = data["campaign_id"]

                # Calculate key metrics
                impressions = data.get("impressions", 0)
                clicks = data.get("clicks", 0)
                conversions = data.get("conversions", 0)
                cost = data.get("cost", 0)

                ctr = clicks / impressions if impressions > 0 else 0
                conversion_rate = conversions / clicks if clicks > 0 else 0
                cost_per_conversion = cost / conversions if conversions > 0 else float("inf")

                # Combine into metrics dict
                campaign_metrics[campaign_id] = {
                    "name": data.get("campaign_name", "Unknown"),
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "cost": cost,
                    "ctr": ctr,
                    "conversion_rate": conversion_rate,
                    "cost_per_conversion": cost_per_conversion,
                    "reward": (
                        conversions if conversions > 0 else clicks / 100
                    ),  # Default reward function
                }

            # Create or update bandit
            bandit_result = None
            if create_new_bandit:
                if not bandit_name:
                    bandit_name = f"Campaign_Comparison_{datetime.now().strftime('%Y%m%d')}"

                # Initialize a Bayesian bandit for campaign comparison
                bandit_result = self.initialize_bandit(
                    name=bandit_name,
                    arms=campaign_ids,
                    algorithm=BanditAlgorithm.BAYESIAN,
                    metadata={
                        "created_from": "campaign_comparison",
                        "days_analyzed": days,
                        "creation_date": datetime.now().isoformat(),
                    },
                )

                # Initialize with historical data
                if bandit_result.get("status") == "success":
                    bandit_id = bandit_result.get("bandit_id")
                    for campaign_id, metrics in campaign_metrics.items():
                        # Add historical data as multiple reward updates
                        conversions = metrics["conversions"]

                        # For Bayesian, add each conversion as a separate reward
                        # (simplified approach)
                        for _ in range(int(conversions)):
                            self.update_bandit(bandit_id, campaign_id, 1.0)

                        # Add partial conversion if needed
                        fractional_part = conversions - int(conversions)
                        if fractional_part > 0:
                            self.update_bandit(bandit_id, campaign_id, fractional_part)

            # Create visualization
            fig = None
            if VISUALIZATION_AVAILABLE:
                # Create comparison visualization
                fig, ax = plt.subplots(figsize=(12, 6))

                # Extract data for plotting
                campaign_names = [metrics["name"] for _, metrics in campaign_metrics.items()]
                campaign_names = [
                    f"{name[:15]}..." if len(name) > 15 else name for name in campaign_names
                ]

                conversion_rates = [
                    metrics["conversion_rate"] * 100 for _, metrics in campaign_metrics.items()
                ]
                ctrs = [metrics["ctr"] * 100 for _, metrics in campaign_metrics.items()]
                costs_per_conv = [
                    min(metrics["cost_per_conversion"], 100)
                    for _, metrics in campaign_metrics.items()
                ]

                x = np.arange(len(campaign_names))
                width = 0.25

                # Plot metrics
                ax.bar(x - width, conversion_rates, width, label="Conv. Rate (%)", color="green")
                ax.bar(x, ctrs, width, label="CTR (%)", color="blue")
                ax.bar(x + width, costs_per_conv, width, label="CPC ($)", color="red")

                ax.set_title("Campaign Performance Comparison")
                ax.set_xticks(x)
                ax.set_xticklabels(campaign_names, rotation=45, ha="right")
                ax.legend()

                plt.tight_layout()

            return {
                "status": "success",
                "message": "Campaign comparison completed successfully",
                "campaign_metrics": campaign_metrics,
                "bandit_result": bandit_result,
                "figure": fig,
                "recommendation": self._generate_campaign_recommendation(campaign_metrics),
            }

        except Exception as e:
            error_message = f"Error comparing campaigns: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def _generate_campaign_recommendation(
        self, campaign_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate campaign optimization recommendations based on metrics.

        Args:
            campaign_metrics: Dictionary of campaign metrics

        Returns:
            Dictionary with recommendations
        """
        if not campaign_metrics:
            return {"message": "No data available for recommendations"}

        # Find best campaign for different metrics
        best_conversion_rate = (-1, None)
        best_ctr = (-1, None)
        best_cpc = (float("inf"), None)
        best_overall = (-1, None)

        for campaign_id, metrics in campaign_metrics.items():
            # Check conversion rate
            if metrics["conversion_rate"] > best_conversion_rate[0]:
                best_conversion_rate = (metrics["conversion_rate"], campaign_id)

            # Check CTR
            if metrics["ctr"] > best_ctr[0]:
                best_ctr = (metrics["ctr"], campaign_id)

            # Check cost per conversion
            if metrics["conversions"] > 0 and metrics["cost_per_conversion"] < best_cpc[0]:
                best_cpc = (metrics["cost_per_conversion"], campaign_id)

            # Calculate overall score (higher is better)
            overall_score = (
                metrics["conversion_rate"] * 10  # Weight conversion rate highly
                + metrics["ctr"]  # Add CTR
                - min(metrics["cost_per_conversion"] / 100, 1)  # Subtract normalized CPC
            )

            if overall_score > best_overall[0]:
                best_overall = (overall_score, campaign_id)

        # Generate recommendations
        recommendations = {
            "best_overall_campaign": best_overall[1],
            "best_conversion_rate": {
                "campaign_id": best_conversion_rate[1],
                "rate": best_conversion_rate[0],
            },
            "best_ctr": {"campaign_id": best_ctr[1], "rate": best_ctr[0]},
            "best_cpc": {"campaign_id": best_cpc[1], "value": best_cpc[0]},
        }

        # Add campaign names
        for key in ["best_overall_campaign", "best_conversion_rate", "best_ctr", "best_cpc"]:
            if isinstance(recommendations[key], dict):
                campaign_id = recommendations[key]["campaign_id"]
                if campaign_id and campaign_id in campaign_metrics:
                    recommendations[key]["name"] = campaign_metrics[campaign_id]["name"]
            elif recommendations[key] in campaign_metrics:
                campaign_id = recommendations[key]
                if campaign_id:
                    recommendations[key + "_name"] = campaign_metrics[campaign_id]["name"]

        # Generate action items
        action_items = []

        # If we have a clear winner
        if best_overall[1]:
            action_items.append(
                {
                    "action": "increase_budget",
                    "campaign_id": best_overall[1],
                    "reason": f"Best overall performance with conversion rate {campaign_metrics[best_overall[1]]['conversion_rate']:.2%}",
                }
            )

        # If we have campaigns with poor CPC
        poor_cpc_campaigns = []
        for campaign_id, metrics in campaign_metrics.items():
            if metrics["conversions"] > 0 and metrics["cost_per_conversion"] > 2 * best_cpc[0]:
                poor_cpc_campaigns.append(campaign_id)

        if poor_cpc_campaigns:
            for campaign_id in poor_cpc_campaigns:
                action_items.append(
                    {
                        "action": "reduce_budget",
                        "campaign_id": campaign_id,
                        "reason": f"High cost per conversion ({campaign_metrics[campaign_id]['cost_per_conversion']:.2f}) compared to best ({best_cpc[0]:.2f})",
                    }
                )

        recommendations["action_items"] = action_items

        return recommendations
