"""
Experimentation Service for Google Ads.

This module provides functionality for designing, implementing,
and analyzing experiments for Google Ads campaigns.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats
import logging

# Correct relative import for BaseService
from ..base_service import BaseService

logger = logging.getLogger(__name__)


# Add BaseService inheritance
class ExperimentationService(BaseService):
    """Service for managing Google Ads experiments."""

    def __init__(self, ads_client: Any, config: Dict[str, Any]):
        """
        Initialize the ExperimentationService.

        Args:
            ads_client: The Google Ads API client.
            config: Configuration dictionary.
        """
        # Correct call to super().__init__
        super().__init__(ads_api=ads_client, config=config)
        self.logger.info("ExperimentationService initialized.")
        # Set default values from config or use hardcoded defaults
        self.default_duration_days = self.config.get("default_experiment_duration", 14)

        # Ensure experiment data directory exists
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
        )
        os.makedirs(self.data_dir, exist_ok=True)

        self.experiments_file = os.path.join(self.data_dir, "experiments.json")
        self._load_experiments()

    def _load_experiments(self):
        """Load experiments from storage."""
        if os.path.exists(self.experiments_file):
            with open(self.experiments_file, "r") as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}
            self._save_experiments()

    def _save_experiments(self):
        """Save experiments to storage."""
        with open(self.experiments_file, "w") as f:
            json.dump(self.experiments, f, indent=2)

    def create_experiment(
        self,
        name: str,
        type: str,
        hypothesis: str,
        control_group: str,
        treatment_groups: List[str],
        metrics: List[str],
        duration_days: int,
        traffic_split: Dict[str, float],
        custom_parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            type: Experiment type (e.g., "A/B Test", "Multivariate Test")
            hypothesis: The hypothesis being tested
            control_group: The campaign ID or name for the control group
            treatment_groups: List of campaign IDs or names for treatment groups
            metrics: List of metrics to track (e.g., "clicks", "conversions")
            duration_days: Duration of the experiment in days
            traffic_split: Dictionary specifying traffic allocation
            custom_parameters: Additional parameters specific to experiment

        Returns:
            str: The experiment ID
        """
        experiment_id = str(uuid.uuid4())

        # Calculate start and end dates
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=duration_days)).strftime("%Y-%m-%d")

        experiment = {
            "id": experiment_id,
            "name": name,
            "type": type,
            "hypothesis": hypothesis,
            "control_group": control_group,
            "treatment_groups": treatment_groups,
            "metrics": metrics,
            "traffic_split": traffic_split,
            "start_date": start_date,
            "end_date": end_date,
            "status": "draft",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "results": None,
            "custom_parameters": custom_parameters or {},
        }

        self.experiments[experiment_id] = experiment
        self._save_experiments()

        if self.logger:
            self.logger.info(f"Created experiment {name} (ID: {experiment_id})")

        return experiment_id

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment details.

        Args:
            experiment_id: The experiment ID

        Returns:
            Dictionary with experiment details

        Raises:
            ValueError: If experiment doesn't exist
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment with ID {experiment_id} does not exist")

        return self.experiments[experiment_id]

    def list_experiments(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by status.

        Args:
            status: Filter by experiment status (draft, running, completed, etc.)
            limit: Maximum number of experiments to return
            offset: Pagination offset

        Returns:
            List of experiment dictionaries
        """
        experiments = list(self.experiments.values())

        if status:
            experiments = [e for e in experiments if e["status"] == status]

        return experiments[offset : offset + limit]

    def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Start an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            Updated experiment dictionary

        Raises:
            ValueError: If experiment doesn't exist or can't be started
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment with ID {experiment_id} does not exist")

        experiment = self.experiments[experiment_id]

        if experiment["status"] != "draft":
            raise ValueError(
                f"Experiment {experiment_id} is not in draft status and cannot be started"
            )

        # Create the experiment in Google Ads
        self._create_ads_experiment(experiment)

        # Update experiment status
        experiment["status"] = "running"
        experiment["updated_at"] = datetime.now().isoformat()
        experiment["start_time"] = datetime.now().isoformat()
        self._save_experiments()

        if self.logger:
            self.logger.info(f"Started experiment {experiment['name']} (ID: {experiment_id})")

        return experiment

    def _create_ads_experiment(self, experiment: Dict[str, Any]):
        """
        Create the experiment in Google Ads.

        Args:
            experiment: Experiment configuration

        This would use the Google Ads API to create the experiment
        """
        # Implementation depends on Google Ads API
        if self.logger:
            self.logger.debug(f"Creating Google Ads experiment: {experiment['name']}")

        # Placeholder for actual Google Ads API call
        # This would involve:
        # 1. Creating draft campaigns based on the control group
        # 2. Modifying the draft campaigns according to experiment parameters
        # 3. Setting up experiment traffic split

        # Example pseudocode for Google Ads API:
        # control_campaign_id = self.ads_api.get_campaign_id(experiment["control_group"])
        # drafts = []
        #
        # for treatment in experiment["treatment_groups"]:
        #     draft_id = self.ads_api.create_draft(control_campaign_id)
        #     self.ads_api.modify_draft(draft_id, experiment["custom_parameters"])
        #     experiment_id = self.ads_api.create_experiment(
        #         draft_id,
        #         experiment["name"],
        #         experiment["traffic_split"],
        #         experiment["start_date"],
        #         experiment["end_date"]
        #     )
        #     drafts.append({"treatment": treatment, "draft_id": draft_id, "experiment_id": experiment_id})
        #
        # experiment["google_ads_data"] = {"drafts": drafts}

    def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Stop an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            Updated experiment dictionary

        Raises:
            ValueError: If experiment doesn't exist or can't be stopped
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment with ID {experiment_id} does not exist")

        experiment = self.experiments[experiment_id]

        if experiment["status"] != "running":
            raise ValueError(f"Experiment {experiment_id} is not running")

        # Stop the experiment in Google Ads
        # self.ads_api.stop_experiment(experiment["google_ads_data"]["experiment_id"])

        experiment["status"] = "stopped"
        experiment["updated_at"] = datetime.now().isoformat()
        experiment["end_time"] = datetime.now().isoformat()
        self._save_experiments()

        if self.logger:
            self.logger.info(f"Stopped experiment {experiment['name']} (ID: {experiment_id})")

        return experiment

    def analyze_experiment(
        self, experiment_id: str, confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Analyze experiment results.

        Args:
            experiment_id: The experiment ID
            confidence_level: Statistical confidence level (default: 0.95)

        Returns:
            Dictionary with analysis results

        Raises:
            ValueError: If experiment doesn't exist or isn't completed
        """
        experiment = self.get_experiment(experiment_id)

        if experiment["status"] not in ["completed", "stopped"]:
            # If running, fetch latest data then analyze
            if experiment["status"] == "running":
                self._fetch_experiment_data(experiment_id)
            else:
                raise ValueError(f"Experiment {experiment_id} is not completed or running")

        # Placeholder for actual analysis logic
        experiment_data = self._get_experiment_data(experiment_id)

        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "winner": None,
            "confidence_level": confidence_level,
        }

        for metric in experiment["metrics"]:
            analysis_results["metrics"][metric] = self._analyze_metric(
                experiment_data, metric, confidence_level
            )

        # Determine overall winner
        winner, winner_score = self._determine_winner(analysis_results["metrics"])
        analysis_results["winner"] = winner
        analysis_results["winner_score"] = winner_score

        # Update experiment with results
        experiment["results"] = analysis_results
        experiment["updated_at"] = datetime.now().isoformat()

        # If experiment was running and has reached end date, mark as completed
        if experiment["status"] == "running" and datetime.now() >= datetime.fromisoformat(
            experiment["end_date"]
        ):
            experiment["status"] = "completed"

        self._save_experiments()

        if self.logger:
            if winner:
                self.logger.info(
                    f"Experiment {experiment['name']} analysis complete. Winner: {winner}"
                )
            else:
                self.logger.info(
                    f"Experiment {experiment['name']} analysis complete. No clear winner."
                )

        return analysis_results

    def _fetch_experiment_data(self, experiment_id: str):
        """
        Fetch the latest data for an experiment from Google Ads.

        Args:
            experiment_id: The experiment ID
        """
        experiment = self.experiments[experiment_id]

        # Placeholder for actual Google Ads API calls
        # This would involve fetching metrics for control and treatment campaigns

        # Example pseudocode:
        # control_data = self.ads_api.get_metrics(
        #     experiment["control_group"],
        #     experiment["start_date"],
        #     datetime.now().strftime("%Y-%m-%d"),
        #     experiment["metrics"]
        # )
        #
        # treatment_data = {}
        # for treatment, draft_info in zip(experiment["treatment_groups"], experiment["google_ads_data"]["drafts"]):
        #     treatment_data[treatment] = self.ads_api.get_metrics(
        #         draft_info["experiment_id"],
        #         experiment["start_date"],
        #         datetime.now().strftime("%Y-%m-%d"),
        #         experiment["metrics"]
        #     )
        #
        # experiment["raw_data"] = {
        #     "control": control_data,
        #     "treatments": treatment_data,
        #     "last_updated": datetime.now().isoformat()
        # }
        # self._save_experiments()

    def _get_experiment_data(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get processed experiment data for analysis.

        Args:
            experiment_id: The experiment ID

        Returns:
            Dictionary with processed experiment data
        """
        experiment = self.experiments[experiment_id]

        # For demonstration, create mock data
        # In a real implementation, this would use actual data from Google Ads
        control_group = experiment["control_group"]
        treatment_groups = experiment["treatment_groups"]
        metrics = experiment["metrics"]

        # Mock data structure
        data = {"control": {"name": control_group, "metrics": {}}, "treatments": {}}

        # Create mock metrics data
        for metric in metrics:
            # Control group baseline
            if metric == "clicks":
                base_value = 5000
            elif metric == "impressions":
                base_value = 100000
            elif metric == "conversions":
                base_value = 200
            elif metric == "cost":
                base_value = 10000
            elif metric == "conversion_value":
                base_value = 25000
            else:
                base_value = 1000

            data["control"]["metrics"][metric] = base_value

            # Treatment groups with variations
            for treatment in treatment_groups:
                if treatment not in data["treatments"]:
                    data["treatments"][treatment] = {"name": treatment, "metrics": {}}

                # Add some variation for each treatment
                # In a real implementation, this would use actual data
                variation = 0.85 + (hash(treatment + metric) % 30) / 100
                data["treatments"][treatment]["metrics"][metric] = int(base_value * variation)

        return data

    def _analyze_metric(
        self, data: Dict[str, Any], metric: str, confidence_level: float
    ) -> Dict[str, Any]:
        """
        Analyze a specific metric from experiment data.

        Args:
            data: Experiment data
            metric: Metric to analyze
            confidence_level: Statistical confidence level

        Returns:
            Dictionary with analysis results for the metric
        """
        control_value = data["control"]["metrics"][metric]

        results = {
            "control": {"name": data["control"]["name"], "value": control_value},
            "treatments": [],
            "winner": None,
            "p_values": {},
        }

        # Calculate lift and statistical significance for each treatment
        for treatment_name, treatment_data in data["treatments"].items():
            treatment_value = treatment_data["metrics"][metric]
            lift = (treatment_value - control_value) / control_value if control_value else 0

            # Simplified statistical test
            # In a real implementation, this would use more appropriate tests
            # based on the metric type and distribution
            alpha = 1 - confidence_level
            p_value = self._calculate_p_value(control_value, treatment_value)
            significant = p_value < alpha

            treatment_result = {
                "name": treatment_name,
                "value": treatment_value,
                "lift": lift,
                "lift_percent": f"{lift * 100:.2f}%",
                "p_value": p_value,
                "significant": significant,
            }

            results["treatments"].append(treatment_result)
            results["p_values"][treatment_name] = p_value

        # Find winner
        winner = None
        best_lift = 0

        for treatment in results["treatments"]:
            if treatment["significant"] and treatment["lift"] > best_lift:
                best_lift = treatment["lift"]
                winner = treatment["name"]

        results["winner"] = winner

        return results

    def _calculate_p_value(self, control_value: float, treatment_value: float) -> float:
        """
        Calculate p-value for a simple comparison.

        This is a simplified example. In a real implementation, you would use
        appropriate statistical tests based on the metric and sample sizes.

        Args:
            control_value: Value for control group
            treatment_value: Value for treatment group

        Returns:
            p-value
        """
        # Mock calculation - in reality you'd use proper statistical tests
        # This is just to simulate statistical significance
        difference = abs(treatment_value - control_value)
        base = max(control_value, treatment_value)

        # Simulate p-value based on relative difference
        # Larger differences yield smaller p-values
        relative_diff = difference / base if base else 0

        # Simulate some randomness in p-values
        import random

        random.seed(hash(f"{control_value}_{treatment_value}"))
        noise = random.uniform(0.8, 1.2)

        p_value = max(0.001, min(0.999, (1 - relative_diff) * noise))
        return p_value

    def _determine_winner(self, metrics_results: Dict[str, Any]) -> tuple:
        """
        Determine overall winner based on metrics results.

        Args:
            metrics_results: Results for each metric

        Returns:
            Tuple of (winner_name, winner_score)
        """
        # Count wins per treatment
        treatment_scores = {}
        total_metrics = len(metrics_results)

        for metric, results in metrics_results.items():
            winner = results.get("winner")
            if winner:
                treatment_scores[winner] = treatment_scores.get(winner, 0) + 1

        # Find treatment with most wins
        best_score = 0
        winner = None

        for treatment, score in treatment_scores.items():
            if score > best_score:
                best_score = score
                winner = treatment

        # Calculate winner score as percentage of metrics won
        winner_score = best_score / total_metrics if total_metrics > 0 else 0

        return winner, winner_score

    def apply_winning_variation(self, experiment_id: str) -> bool:
        """
        Apply the winning variation to the original campaign.

        Args:
            experiment_id: The experiment ID

        Returns:
            Boolean indicating success

        Raises:
            ValueError: If experiment doesn't have a clear winner
        """
        experiment = self.get_experiment(experiment_id)

        if not experiment.get("results") or not experiment["results"].get("winner"):
            raise ValueError(f"Experiment {experiment_id} does not have a clear winner")

        winner = experiment["results"]["winner"]

        # Apply winning variation to control campaign
        # In a real implementation, this would use Google Ads API

        # Example pseudocode:
        # winner_treatment_idx = experiment["treatment_groups"].index(winner)
        # winner_draft_id = experiment["google_ads_data"]["drafts"][winner_treatment_idx]["draft_id"]
        # self.ads_api.apply_draft(winner_draft_id)

        experiment["status"] = "applied"
        experiment["updated_at"] = datetime.now().isoformat()
        experiment["applied_at"] = datetime.now().isoformat()
        self._save_experiments()

        if self.logger:
            self.logger.info(
                f"Applied winning variation '{winner}' from experiment "
                f"{experiment['name']} (ID: {experiment_id})"
            )

        return True

    def get_experiment_recommendations(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        Get recommendations based on experiment results.

        Args:
            experiment_id: The experiment ID

        Returns:
            List of recommendation dictionaries
        """
        experiment = self.get_experiment(experiment_id)

        if not experiment.get("results"):
            return []

        recommendations = []
        results = experiment["results"]
        winner = results.get("winner")

        if winner:
            recommendations.append(
                {
                    "type": "apply_winner",
                    "description": f"Apply the winning variation '{winner}' to the original campaign",
                    "importance": "high",
                }
            )

            # Add specific recommendations based on metrics
            for metric, metric_results in results["metrics"].items():
                for treatment in metric_results["treatments"]:
                    if treatment["name"] == winner and treatment["significant"]:
                        recommendations.append(
                            {
                                "type": "metric_insight",
                                "metric": metric,
                                "description": (
                                    f"The winning variation '{winner}' showed a "
                                    f"{treatment['lift_percent']} improvement in {metric}"
                                ),
                                "importance": "medium",
                            }
                        )
        else:
            recommendations.append(
                {
                    "type": "no_winner",
                    "description": "No clear winner was identified in this experiment",
                    "importance": "medium",
                }
            )

            # Recommend extending experiment if results are close but not significant
            close_treatments = []
            for metric, metric_results in results["metrics"].items():
                for treatment in metric_results["treatments"]:
                    if not treatment["significant"] and treatment["lift"] > 0.05:
                        close_treatments.append(treatment["name"])

            if close_treatments:
                recommendations.append(
                    {
                        "type": "extend_experiment",
                        "description": (
                            "Consider extending the experiment duration. Some variations "
                            "show promising results but aren't statistically significant yet."
                        ),
                        "treatments": list(set(close_treatments)),
                        "importance": "medium",
                    }
                )

        return recommendations

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            Boolean indicating success

        Raises:
            ValueError: If experiment doesn't exist
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment with ID {experiment_id} does not exist")

        # Stop experiment if running
        experiment = self.experiments[experiment_id]
        if experiment["status"] == "running":
            self.stop_experiment(experiment_id)

        # Delete experiment from Google Ads
        # Example pseudocode:
        # if "google_ads_data" in experiment:
        #     for draft_info in experiment["google_ads_data"]["drafts"]:
        #         self.ads_api.delete_experiment(draft_info["experiment_id"])
        #         self.ads_api.delete_draft(draft_info["draft_id"])

        # Remove from local storage
        del self.experiments[experiment_id]
        self._save_experiments()

        if self.logger:
            self.logger.info(f"Deleted experiment {experiment['name']} (ID: {experiment_id})")

        return True

    def schedule_experiment(
        self, experiment_id: str, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """
        Schedule an experiment to run in the future.

        Args:
            experiment_id: The experiment ID
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)

        Returns:
            Updated experiment dictionary
        """
        experiment = self.get_experiment(experiment_id)

        if experiment["status"] != "draft":
            raise ValueError(f"Experiment {experiment_id} must be in draft status to schedule")

        experiment["start_date"] = start_date
        experiment["end_date"] = end_date
        experiment["status"] = "scheduled"
        experiment["updated_at"] = datetime.now().isoformat()
        self._save_experiments()

        if self.logger:
            self.logger.info(
                f"Scheduled experiment {experiment['name']} (ID: {experiment_id}) "
                f"to run from {start_date} to {end_date}"
            )

        return experiment
