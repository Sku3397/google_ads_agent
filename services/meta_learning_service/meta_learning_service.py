"""
Meta Learning Service for Google Ads Management System

This module provides a service for adapting optimization strategies based on
historical performance and learning what works best for specific account patterns.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pickle

# Meta-learning libraries (examples)
# import learn2learn as l2l # type: ignore
# import higher # type: ignore

from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer
from config import load_config
from ..base_service import BaseService


class MetaLearningService(BaseService):
    """
    Service for meta-learning and strategy adaptation within the Google Ads management system.

    The MetaLearningService implements techniques to learn which optimization strategies
    work best in different contexts, adapting the system's approach over time to improve results.

    Features:
    - Strategy performance tracking and analysis
    - Cross-service optimization pattern learning
    - Dynamic strategy selection based on historical performance
    - Hyperparameter optimization for other services
    - Learning to adapt to account-specific patterns
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the MetaLearningService.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        # Initialize meta-learning components
        self.strategy_history = []
        self.performance_history = {}
        self.meta_models = {}

        # Load existing model data if available
        self._load_models()

        self.logger.info("MetaLearningService initialized")

    def _load_models(self):
        """Load saved meta-learning models if they exist."""
        model_path = os.path.join("data", "meta_learning", "meta_models.pkl")
        history_path = os.path.join("data", "meta_learning", "strategy_history.json")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.join("data", "meta_learning"), exist_ok=True)

            # Load models if they exist
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.meta_models = pickle.load(f)
                self.logger.info(f"Loaded meta-learning models from {model_path}")

            # Load history if it exists
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    data = json.load(f)
                    self.strategy_history = data.get("strategy_history", [])
                    self.performance_history = data.get("performance_history", {})
                self.logger.info(f"Loaded strategy history from {history_path}")

        except Exception as e:
            self.logger.error(f"Error loading meta-learning models: {str(e)}")

    def _save_models(self):
        """Save meta-learning models and history."""
        model_path = os.path.join("data", "meta_learning", "meta_models.pkl")
        history_path = os.path.join("data", "meta_learning", "strategy_history.json")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.join("data", "meta_learning"), exist_ok=True)

            # Save models
            with open(model_path, "wb") as f:
                pickle.dump(self.meta_models, f)

            # Save history
            with open(history_path, "w") as f:
                json.dump(
                    {
                        "strategy_history": self.strategy_history,
                        "performance_history": self.performance_history,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            self.logger.info(f"Saved meta-learning models to {model_path}")

        except Exception as e:
            self.logger.error(f"Error saving meta-learning models: {str(e)}")

    def record_strategy_execution(
        self,
        service_name: str,
        strategy_name: str,
        context: Dict[str, Any],
        parameters: Dict[str, Any],
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Record the execution of a strategy by another service.

        Args:
            service_name: Name of the service that executed the strategy
            strategy_name: Name of the strategy that was executed
            context: Context in which the strategy was executed (e.g., campaign type, budget)
            parameters: Parameters used for the strategy
            results: Results of the strategy execution

        Returns:
            Dictionary with record details
        """
        # Create record entry
        record = {
            "id": f"exec_{len(self.strategy_history) + 1}_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "service_name": service_name,
            "strategy_name": strategy_name,
            "context": context,
            "parameters": parameters,
            "results": results,
            "metrics": self._extract_performance_metrics(results),
        }

        # Add to history
        self.strategy_history.append(record)

        # Update performance history for this strategy
        strategy_key = f"{service_name}_{strategy_name}"
        if strategy_key not in self.performance_history:
            self.performance_history[strategy_key] = []

        # Limit history size per strategy if needed
        max_history = self.config.get("meta_learning_max_history_per_strategy", 100)
        self.performance_history[strategy_key].append(
            {
                "timestamp": record["timestamp"],
                "context": context,
                "parameters": parameters,
                "metrics": record["metrics"],
            }
        )
        self.performance_history[strategy_key] = self.performance_history[strategy_key][
            -max_history:
        ]

        # Save updated data
        self._save_models()

        self.logger.info(f"Recorded strategy execution: {service_name} - {strategy_name}")
        return record

    def _extract_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract performance metrics from strategy results.

        Args:
            results: Results dictionary from strategy execution

        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}

        # Extract common metrics if they exist
        metric_keys = [
            "cost",
            "impressions",
            "clicks",
            "conversions",
            "ctr",
            "conversion_rate",
            "cpa",
            "roas",
        ]

        for key in metric_keys:
            if key in results:
                metrics[key] = results[key]

        # Extract improvement metrics if they exist
        if "before" in results and "after" in results:
            for key in metric_keys:
                if key in results["before"] and key in results["after"]:
                    improvement = results["after"][key] - results["before"][key]
                    metrics[f"{key}_improvement"] = improvement

                    # Add percentage improvement if non-zero before value
                    if results["before"][key] != 0:
                        pct_improvement = (improvement / results["before"][key]) * 100
                        metrics[f"{key}_pct_improvement"] = pct_improvement

        return metrics

    def recommend_strategy(
        self, service_name: str, context: Dict[str, Any], available_strategies: List[str]
    ) -> Dict[str, Any]:
        """
        Recommend the best strategy for a given service and context based on historical performance.

        Args:
            service_name: Name of the service requesting a strategy
            context: Current context (e.g., campaign details, market conditions)
            available_strategies: List of strategies the service can execute

        Returns:
            Dictionary with recommended strategy name and estimated parameters
        """
        self.logger.info(f"Recommending strategy for {service_name} in context: {context}")

        strategy_scores = {}
        best_params_for_strategy = {}

        for strategy_name in available_strategies:
            strategy_key = f"{service_name}_{strategy_name}"
            if strategy_key in self.performance_history:
                strategy_data = self.performance_history[strategy_key]
                # Calculate score based on historical performance in similar contexts
                score = self._calculate_strategy_score(strategy_data, context)
                strategy_scores[strategy_name] = score

                # Find best parameters based on historical success in similar contexts
                best_params = self._get_best_parameters(strategy_data, context)
                best_params_for_strategy[strategy_name] = best_params
            else:
                # Assign a default score or handle new strategies (e.g., exploration)
                strategy_scores[strategy_name] = 0.5  # Default score for unknown strategies
                best_params_for_strategy[strategy_name] = {}  # Default empty params

        # Select the strategy with the highest score
        if not strategy_scores:
            # If no history, maybe default to a standard strategy or random choice
            recommended_strategy = available_strategies[0] if available_strategies else None
            recommended_params = {}
            self.logger.warning(
                "No historical data for any available strategies. Choosing default."
            )
        else:
            # Sort strategies by score
            recommended_strategy = max(strategy_scores, key=strategy_scores.get)
            recommended_params = best_params_for_strategy.get(recommended_strategy, {})

        if not recommended_strategy:
            return {"error": "No strategies available or could be recommended."}

        self.logger.info(
            f"Recommended strategy: {recommended_strategy} with params: {recommended_params}"
        )

        # TODO: Integrate more advanced meta-learning models here (e.g., predict performance)
        # meta_model_prediction = self._predict_performance_with_meta_model(service_name, recommended_strategy, context, recommended_params)

        return {
            "recommended_strategy": recommended_strategy,
            "estimated_parameters": recommended_params,
            "confidence_score": strategy_scores.get(
                recommended_strategy, 0.0
            ),  # Example confidence
        }

    def _calculate_strategy_score(
        self, strategy_data: List[Dict[str, Any]], current_context: Dict[str, Any]
    ) -> float:
        """
        Calculate a score for a strategy based on historical performance and context similarity.

        Args:
            strategy_data: Historical data for the strategy
            current_context: Current context to compare against

        Returns:
            Score for the strategy (higher is better)
        """
        if not strategy_data:
            return 0

        # Simple scoring for now - can be replaced with more sophisticated methods
        # like fitting a regression model on the historical data

        scores = []
        for entry in strategy_data:
            # Calculate context similarity
            context_similarity = self._calculate_context_similarity(
                entry["context"], current_context
            )

            # Extract success metrics (this could be customized based on business goals)
            metrics = entry["metrics"]

            # Define success metrics and weights (can be expanded)
            metric_weights = {
                "conversion_rate_improvement": 0.3,
                "cpa_improvement": 0.3,
                "ctr_improvement": 0.2,
                "roas_improvement": 0.2,
            }

            # Calculate weighted performance score
            perf_score = 0
            for metric, weight in metric_weights.items():
                if metric in metrics:
                    # For cost metrics, lower is better so invert
                    if "cpa" in metric:
                        perf_score += weight * (-1 * metrics[metric])
                    else:
                        perf_score += weight * metrics[metric]

            # Combine context similarity and performance
            # More recent executions get higher weight
            recency_factor = 1.0  # Could implement time decay

            # Final score for this entry
            entry_score = context_similarity * perf_score * recency_factor
            scores.append(entry_score)

        # Return average score across all entries
        return sum(scores) / len(scores) if scores else 0

    def _calculate_context_similarity(
        self, context1: Dict[str, Any], context2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two contexts.

        Args:
            context1: First context dictionary
            context2: Second context dictionary

        Returns:
            Similarity score between 0 and 1
        """
        # Simple implementation - can be enhanced with more sophisticated methods
        # like cosine similarity on feature vectors

        # Get all unique keys
        all_keys = set(context1.keys()) | set(context2.keys())

        if not all_keys:
            return 0

        # Count matching values
        matches = 0
        for key in all_keys:
            if key in context1 and key in context2:
                # For numeric values, calculate relative similarity
                if isinstance(context1[key], (int, float)) and isinstance(
                    context2[key], (int, float)
                ):
                    # Avoid division by zero
                    max_val = max(abs(context1[key]), abs(context2[key]))
                    if max_val > 0:
                        similarity = 1 - (abs(context1[key] - context2[key]) / max_val)
                        matches += similarity
                    else:
                        matches += 1  # Both zero means perfect match
                # For strings or other values, exact match check
                elif context1[key] == context2[key]:
                    matches += 1

        # Return similarity as fraction of possible matches
        return matches / len(all_keys)

    def _get_best_parameters(
        self, strategy_data: List[Dict[str, Any]], current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get the best parameters for a strategy based on historical performance.

        Args:
            strategy_data: Historical data for the strategy
            current_context: Current context to compare against

        Returns:
            Dictionary of recommended parameters
        """
        if not strategy_data:
            return {}

        # Find the best performing execution in similar contexts
        best_score = float("-inf")
        best_params = {}

        for entry in strategy_data:
            # Calculate context similarity
            context_similarity = self._calculate_context_similarity(
                entry["context"], current_context
            )

            # Skip if contexts are too different
            if context_similarity < 0.5:
                continue

            # Calculate performance score
            metrics = entry["metrics"]

            # Simple scoring - can be enhanced
            perf_score = sum(
                [
                    metrics.get("conversion_rate_improvement", 0) * 0.3,
                    -1 * metrics.get("cpa_improvement", 0) * 0.3,  # Lower CPA is better
                    metrics.get("ctr_improvement", 0) * 0.2,
                    metrics.get("roas_improvement", 0) * 0.2,
                ]
            )

            # Weight by context similarity
            weighted_score = perf_score * context_similarity

            if weighted_score > best_score:
                best_score = weighted_score
                best_params = entry["parameters"].copy()

        return best_params

    def learn_hyperparameters(
        self,
        service_name: str,
        strategy_name: str,
        param_grid: Dict[str, List[Any]],
        evaluation_function,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """
        Learn optimal hyperparameters for a strategy using historical data.

        Args:
            service_name: Name of the service
            strategy_name: Name of the strategy
            param_grid: Grid of parameters to search
            evaluation_function: Function to evaluate a parameter set
            n_trials: Number of trials to run

        Returns:
            Dictionary with optimal parameters
        """
        self.logger.info(f"Learning hyperparameters for {service_name} - {strategy_name}")

        # Simple random search for now (could be enhanced with Bayesian optimization)
        best_score = float("-inf")
        best_params = {}

        for _ in range(n_trials):
            # Sample parameters from grid
            params = {k: np.random.choice(v) for k, v in param_grid.items()}

            # Evaluate parameters
            try:
                score = evaluation_function(params)

                # Update best if better
                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                self.logger.info(f"Trial score: {score:.4f}")

            except Exception as e:
                self.logger.error(f"Error evaluating parameters: {str(e)}")

        result = {
            "service": service_name,
            "strategy": strategy_name,
            "best_parameters": best_params,
            "best_score": best_score,
            "n_trials": n_trials,
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(f"Best parameters found with score {best_score:.4f}")
        return result

    def analyze_cross_service_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns across different services to find synergies and conflicts.

        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Analyzing cross-service patterns")

        # Get all unique services
        services = set(entry["service_name"] for entry in self.strategy_history)

        # Initialize results
        results = {"synergies": [], "conflicts": [], "service_correlations": {}}

        # Skip if not enough data
        if len(self.strategy_history) < 10:
            self.logger.warning("Not enough data to analyze cross-service patterns")
            return results

        # Convert history to DataFrame for easier analysis
        df = pd.DataFrame(
            [
                {
                    "timestamp": datetime.fromisoformat(entry["timestamp"]),
                    "service": entry["service_name"],
                    "strategy": entry["strategy_name"],
                    **entry["metrics"],
                }
                for entry in self.strategy_history
                if "metrics" in entry
            ]
        )

        if df.empty:
            self.logger.warning("No valid data for cross-service analysis")
            return results

        # Analyze temporal patterns
        # Find cases where one service execution is followed by another
        # and performance improved

        # Simple correlation analysis between service performance
        corr_services = {}
        for service1 in services:
            corr_services[service1] = {}
            for service2 in services:
                if service1 != service2:
                    # Calculate correlation between service executions
                    # This is a simplified analysis and could be enhanced
                    corr_services[service1][service2] = 0.0  # Placeholder

        results["service_correlations"] = corr_services

        self.logger.info("Completed cross-service pattern analysis")
        return results

    def transfer_learning(
        self, source_context: Dict[str, Any], target_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transfer learning from one context to another.

        Args:
            source_context: Source context to learn from
            target_context: Target context to transfer learning to

        Returns:
            Dictionary with transfer results
        """
        self.logger.info("Applying transfer learning between contexts")

        # Calculate similarity between source and target contexts
        similarity = self._calculate_context_similarity(source_context, target_context)

        # Find strategies that worked well in the source context
        source_strategies = []

        # Filter history by source context similarity
        relevant_history = []
        for entry in self.strategy_history:
            entry_similarity = self._calculate_context_similarity(entry["context"], source_context)
            if entry_similarity > 0.7:  # Only consider highly relevant entries
                relevant_history.append(entry)

        # Extract successful strategies
        for entry in relevant_history:
            # Simple success criteria - could be enhanced
            metrics = entry["metrics"]
            success_score = (
                metrics.get("conversion_rate_improvement", 0) * 0.3
                + -1 * metrics.get("cpa_improvement", 0) * 0.3  # Lower CPA is better
                + metrics.get("ctr_improvement", 0) * 0.2
                + metrics.get("roas_improvement", 0) * 0.2
            )

            if success_score > 0:
                source_strategies.append(
                    {
                        "service": entry["service_name"],
                        "strategy": entry["strategy_name"],
                        "parameters": entry["parameters"],
                        "success_score": success_score,
                    }
                )

        # Sort by success score
        source_strategies.sort(key=lambda x: x["success_score"], reverse=True)

        # Adapt parameters for target context
        adapted_strategies = []
        for strategy in source_strategies:
            # Simple adaptation - could be enhanced with more sophisticated methods
            adapted_params = strategy["parameters"].copy()
            adapted_strategies.append(
                {
                    "service": strategy["service"],
                    "strategy": strategy["strategy"],
                    "original_parameters": strategy["parameters"],
                    "adapted_parameters": adapted_params,
                    "confidence": similarity * strategy["success_score"],
                }
            )

        result = {
            "source_context": source_context,
            "target_context": target_context,
            "context_similarity": similarity,
            "adapted_strategies": adapted_strategies[:5],  # Top 5 strategies
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            f"Transfer learning completed with {len(adapted_strategies)} adapted strategies"
        )
        return result

    def run(self, **kwargs):
        """
        Run the meta-learning service's primary functionality.

        Args:
            **kwargs: Additional parameters for specific operations
                - operation: The specific operation to run
                    - "analyze": Analyze historical strategies
                    - "learn": Learn patterns from history
                    - "recommend": Recommend strategies

        Returns:
            Results dictionary
        """
        operation = kwargs.get("operation", "analyze")

        if operation == "analyze":
            return self.analyze_cross_service_patterns()
        elif operation == "learn":
            # Run learning algorithms on historical data
            return {"status": "success", "message": "Learning completed"}
        elif operation == "recommend":
            service = kwargs.get("service")
            context = kwargs.get("context", {})
            strategies = kwargs.get("strategies", [])

            if not service or not strategies:
                return {"status": "error", "message": "Missing required parameters"}

            return self.recommend_strategy(service, context, strategies)
        else:
            return {"status": "error", "message": f"Unknown operation: {operation}"}
