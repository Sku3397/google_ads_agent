"""
Portfolio Optimization Service for Google Ads Management System

This module provides portfolio optimization capabilities to allocate budget
across multiple campaigns to maximize overall performance metrics like ROAS,
conversions, or other KPIs.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import json
import cvxpy as cp

from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer
from config import load_config
from ..base_service import BaseService


class PortfolioOptimizationService(BaseService):
    """
    Portfolio Optimization Service for optimizing budget allocation across campaigns.

    This service implements various portfolio optimization algorithms to allocate
    budget across multiple campaigns to maximize overall performance while
    respecting constraints.
    """

    def __init__(self, ads_api=None, optimizer=None, config=None, logger=None):
        """
        Initialize the portfolio optimization service.

        Args:
            ads_api: Google Ads API client
            optimizer: AI optimizer
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

    def optimize_campaign_portfolio(
        self,
        days: int = 30,
        objective: str = "conversions",
        constraint: str = "budget",
        budget_limit: Optional[float] = None,
        campaign_ids: Optional[List[str]] = None,
        algorithm: str = "convex",
        risk_tolerance: float = 0.5,
        min_budget_per_campaign: Optional[float] = None,
        multi_objective_weights: Optional[Dict[str, float]] = None,
        visualize: bool = False,
        output_dir: str = "reports/portfolio_optimization",
    ) -> Dict[str, Any]:
        """
        Optimize budget allocation across campaigns to maximize the objective.

        Args:
            days: Number of days of historical data to use
            objective: Objective function to maximize (conversions, clicks, roas, multi)
            constraint: Type of constraint (budget, target_cpa, target_roas)
            budget_limit: Total budget limit across all campaigns
            campaign_ids: List of campaign IDs to include in the optimization
            algorithm: Optimization algorithm to use (convex, efficient_frontier, multi_objective, bayesian)
            risk_tolerance: Risk tolerance parameter (0-1), higher means more risk-seeking
            min_budget_per_campaign: Minimum budget allocation per campaign
            multi_objective_weights: Weights for different objectives when using multi-objective
            visualize: Whether to generate visualization plots
            output_dir: Directory to save visualizations

        Returns:
            Dictionary with optimization results
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                f"Starting portfolio optimization for {objective} with {constraint} constraint using {algorithm} algorithm"
            )

            # Validate input parameters
            if risk_tolerance < 0 or risk_tolerance > 1:
                return {
                    "status": "error",
                    "message": f"Risk tolerance must be between 0 and 1, got {risk_tolerance}",
                }

            valid_algorithms = ["convex", "efficient_frontier", "multi_objective", "bayesian"]
            if algorithm not in valid_algorithms:
                return {
                    "status": "error",
                    "message": f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}",
                }

            valid_objectives = ["conversions", "clicks", "impressions", "roas", "multi"]
            if objective not in valid_objectives:
                return {
                    "status": "error",
                    "message": f"Invalid objective: {objective}. Must be one of {valid_objectives}",
                }

            # If objective is multi, ensure we have weights
            if objective == "multi" and not multi_objective_weights:
                multi_objective_weights = {
                    "conversions": 0.6,
                    "clicks": 0.2,
                    "impressions": 0.1,
                    "roas": 0.1,
                }

            # Get campaign performance data
            campaigns = self._get_campaign_performance(days, campaign_ids)

            if not campaigns or len(campaigns) < 2:
                self.logger.warning("Insufficient campaign data for portfolio optimization")
                self._track_execution(start_time, False)
                return {
                    "status": "error",
                    "message": "Insufficient campaign data for portfolio optimization. Need at least 2 campaigns.",
                }

            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(campaigns)

            # Set budget limit if not provided
            if budget_limit is None:
                budget_limit = df["budget"].sum()

            # Run portfolio optimization algorithm
            optimization_results = self._run_optimization(
                campaign_df=df,
                objective=objective,
                constraint=constraint,
                budget_limit=budget_limit,
                algorithm=algorithm,
                risk_tolerance=risk_tolerance,
                min_budget_per_campaign=min_budget_per_campaign,
                multi_objective_weights=multi_objective_weights,
            )

            if optimization_results.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"Optimization failed: {optimization_results.get('message', 'Unknown error')}",
                }

            # Prepare recommendations
            recommendations = self._prepare_recommendations(df, optimization_results)

            # Generate visualizations if requested
            visualization_paths = []
            if visualize:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    # Create output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)

                    # 1. Budget Allocation Comparison
                    plt.figure(figsize=(12, 6))

                    # Prepare data
                    campaign_names = [rec["campaign_name"] for rec in recommendations]
                    current_budgets = [rec["current_budget"] for rec in recommendations]
                    recommended_budgets = [rec["recommended_budget"] for rec in recommendations]

                    # Shorten long campaign names
                    campaign_names = [
                        name[:20] + "..." if len(name) > 20 else name for name in campaign_names
                    ]

                    # Create comparison plot
                    x = range(len(campaign_names))
                    width = 0.35

                    plt.bar(
                        [i - width / 2 for i in x], current_budgets, width, label="Current Budget"
                    )
                    plt.bar(
                        [i + width / 2 for i in x],
                        recommended_budgets,
                        width,
                        label="Recommended Budget",
                    )

                    plt.xlabel("Campaign")
                    plt.ylabel("Budget ($)")
                    plt.title("Budget Allocation Comparison")
                    plt.xticks(x, campaign_names, rotation=45, ha="right")
                    plt.legend()
                    plt.tight_layout()

                    # Save the figure
                    budget_plot_path = os.path.join(
                        output_dir,
                        f"budget_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    )
                    plt.savefig(budget_plot_path)
                    plt.close()
                    visualization_paths.append(budget_plot_path)

                    # 2. Performance Metrics Plot
                    if (
                        "expected_conversions" in optimization_results
                        or "expected_clicks" in optimization_results
                        or "expected_roas" in optimization_results
                    ):
                        plt.figure(figsize=(10, 6))

                        metrics_labels = []
                        current_values = []
                        expected_values = []

                        if (
                            "expected_conversions" in optimization_results
                            and optimization_results["expected_conversions"] is not None
                        ):
                            metrics_labels.append("Conversions")
                            current_values.append(df["conversions"].sum())
                            expected_values.append(optimization_results["expected_conversions"])

                        if (
                            "expected_clicks" in optimization_results
                            and optimization_results["expected_clicks"] is not None
                        ):
                            metrics_labels.append("Clicks")
                            current_values.append(df["clicks"].sum())
                            expected_values.append(optimization_results["expected_clicks"])

                        if (
                            "expected_roas" in optimization_results
                            and optimization_results["expected_roas"] is not None
                        ):
                            metrics_labels.append("ROAS")
                            current_roas = (
                                df["conversion_value"].sum() / df["cost"].sum()
                                if df["cost"].sum() > 0
                                else 0
                            )
                            current_values.append(current_roas)
                            expected_values.append(optimization_results["expected_roas"])

                        if metrics_labels:
                            x = range(len(metrics_labels))
                            width = 0.35

                            plt.bar(
                                [i - width / 2 for i in x], current_values, width, label="Current"
                            )
                            plt.bar(
                                [i + width / 2 for i in x], expected_values, width, label="Expected"
                            )

                            plt.xlabel("Metric")
                            plt.ylabel("Value")
                            plt.title("Expected Performance Improvement")
                            plt.xticks(x, metrics_labels)
                            plt.legend()

                            # Add improvement percentage
                            for i, (current, expected) in enumerate(
                                zip(current_values, expected_values)
                            ):
                                if current > 0:
                                    improvement = (expected - current) / current * 100
                                    plt.text(
                                        i,
                                        max(current, expected) * 1.05,
                                        f"+{improvement:.1f}%",
                                        ha="center",
                                        va="bottom",
                                    )

                            plt.tight_layout()

                            # Save the figure
                            metrics_plot_path = os.path.join(
                                output_dir,
                                f"metrics_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            )
                            plt.savefig(metrics_plot_path)
                            plt.close()
                            visualization_paths.append(metrics_plot_path)

                    # 3. Efficient Frontier Plot (if applicable)
                    if algorithm == "efficient_frontier":
                        plt.figure(figsize=(10, 6))

                        # Calculate points along the efficient frontier by varying risk tolerance
                        risk_points = np.linspace(0.1, 0.9, 9)
                        returns = []
                        risks = []

                        for risk_pt in risk_points:
                            ef_result = self._run_efficient_frontier_optimization(
                                campaign_df=df,
                                metrics=self._get_performance_metrics(
                                    df, objective, multi_objective_weights
                                ),
                                budget_limit=budget_limit,
                                risk_tolerance=risk_pt,
                                min_budget=min_budget_per_campaign or 1.0,
                            )

                            if ef_result.get("status") == "success" and "sharpe_ratio" in ef_result:
                                returns.append(
                                    ef_result.get("expected_improvement", 0) + 1
                                )  # +1 to show absolute return
                                risks.append(
                                    1 - risk_pt
                                )  # Use inverse of risk tolerance as risk measure

                        if returns and risks:
                            # Plot efficient frontier
                            plt.plot(risks, returns, "b-", marker="o")

                            # Highlight selected point
                            current_risk = 1 - risk_tolerance
                            current_return = optimization_results.get("expected_improvement", 0) + 1
                            plt.plot(current_risk, current_return, "ro", markersize=10)
                            plt.annotate(
                                "Selected",
                                (current_risk, current_return),
                                xytext=(10, 10),
                                textcoords="offset points",
                            )

                            plt.xlabel("Risk (Lower is Better)")
                            plt.ylabel("Expected Return Multiple")
                            plt.title("Efficient Frontier")
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()

                            # Save the figure
                            ef_plot_path = os.path.join(
                                output_dir,
                                f"efficient_frontier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            )
                            plt.savefig(ef_plot_path)
                            plt.close()
                            visualization_paths.append(ef_plot_path)

                except Exception as e:
                    self.logger.warning(f"Error generating visualizations: {str(e)}")

            result = {
                "status": "success",
                "recommendations": recommendations,
                "optimization_details": {
                    "objective": objective,
                    "constraint": constraint,
                    "budget_limit": budget_limit,
                    "expected_improvement": optimization_results.get("expected_improvement", 0),
                    "algorithm": algorithm,
                    "risk_tolerance": risk_tolerance if algorithm == "efficient_frontier" else None,
                    "solver_status": optimization_results.get("solver_status", "unknown"),
                },
                "timestamp": datetime.now().isoformat(),
            }

            # Add additional metrics based on what's available
            for metric in [
                "expected_conversions",
                "expected_clicks",
                "expected_roas",
                "sharpe_ratio",
            ]:
                if metric in optimization_results:
                    result["optimization_details"][metric] = optimization_results[metric]

            # Add visualization paths if any
            if visualization_paths:
                result["visualization_paths"] = visualization_paths

            self.logger.info(
                f"Portfolio optimization completed successfully with {len(recommendations)} recommendations. Expected improvement: {optimization_results.get('expected_improvement', 0):.2%}"
            )
            self._track_execution(start_time, True)

            return result

        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "error", "message": str(e)}

    def apply_portfolio_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply portfolio optimization recommendations to campaigns.

        Args:
            recommendations: List of budget recommendations by campaign

        Returns:
            Dictionary with application results
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                f"Applying {len(recommendations)} portfolio optimization recommendations"
            )

            results = []
            for rec in recommendations:
                campaign_id = rec.get("campaign_id")
                new_budget = rec.get("recommended_budget")

                if not campaign_id or not new_budget:
                    continue

                # Apply new budget to campaign using the ads API
                if self.ads_api:
                    update_result = self.ads_api.update_campaign_budget(
                        campaign_id=campaign_id, new_budget=new_budget
                    )

                    results.append(
                        {
                            "campaign_id": campaign_id,
                            "previous_budget": rec.get("current_budget"),
                            "new_budget": new_budget,
                            "status": update_result.get("status", "unknown"),
                        }
                    )

            result = {
                "status": "success",
                "applied_count": len([r for r in results if r.get("status") == "success"]),
                "failed_count": len([r for r in results if r.get("status") != "success"]),
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(
                f"Applied {result['applied_count']} portfolio recommendations successfully"
            )
            self._track_execution(start_time, True)

            return result

        except Exception as e:
            self.logger.error(f"Error applying portfolio recommendations: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "failed", "message": str(e)}

    def cross_campaign_keyword_analysis(
        self, days: int = 30, campaign_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze keywords across campaigns to identify overlaps, cannibalization,
        and opportunities for portfolio-level optimization.

        Args:
            days: Number of days of historical data to use
            campaign_ids: List of campaign IDs to include in the analysis

        Returns:
            Dictionary with cross-campaign keyword analysis
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting cross-campaign keyword analysis")

            # Get keyword performance data across campaigns
            keywords = self._get_keyword_performance(days, campaign_ids)

            if not keywords:
                self.logger.warning("Insufficient keyword data for cross-campaign analysis")
                self._track_execution(start_time, False)
                return {
                    "status": "failed",
                    "message": "Insufficient keyword data for cross-campaign analysis",
                }

            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(keywords)

            # Identify overlapping keywords across campaigns
            overlaps = self._identify_keyword_overlaps(df)

            # Identify potential cannibalization
            cannibalization = self._identify_cannibalization(df)

            # Identify performance disparities
            disparities = self._identify_performance_disparities(df)

            result = {
                "status": "success",
                "overlapping_keywords": overlaps,
                "potential_cannibalization": cannibalization,
                "performance_disparities": disparities,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(f"Cross-campaign keyword analysis completed successfully")
            self._track_execution(start_time, True)

            return result

        except Exception as e:
            self.logger.error(f"Error in cross-campaign keyword analysis: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "failed", "message": str(e)}

    def optimize_budget_allocation_over_time(
        self, days: int = 30, forecast_days: int = 30, objective: str = "conversions"
    ) -> Dict[str, Any]:
        """
        Optimize budget allocation over time, accounting for seasonality and trends.

        Args:
            days: Number of days of historical data to use
            forecast_days: Number of days to forecast and optimize for
            objective: Objective function to maximize

        Returns:
            Dictionary with time-based budget allocation plan
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting time-based budget allocation optimization")

            # Get campaign performance data with daily breakdown
            performance_data = self._get_campaign_daily_performance(days)

            if not performance_data:
                self.logger.warning("Insufficient performance data for time-based optimization")
                self._track_execution(start_time, False)
                return {
                    "status": "failed",
                    "message": "Insufficient performance data for time-based optimization",
                }

            # Generate forecasts using time series methods
            forecasts = self._generate_campaign_forecasts(performance_data, forecast_days)

            # Optimize budget allocation for each day in the forecast period
            daily_allocations = self._optimize_daily_allocations(forecasts, objective)

            result = {
                "status": "success",
                "daily_budget_allocations": daily_allocations,
                "optimization_details": {
                    "objective": objective,
                    "forecast_days": forecast_days,
                    "algorithm": "time_series_portfolio_optimization",
                },
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(f"Time-based budget allocation optimization completed successfully")
            self._track_execution(start_time, True)

            return result

        except Exception as e:
            self.logger.error(f"Error in time-based budget allocation: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "failed", "message": str(e)}

    # Helper methods

    def _get_campaign_performance(
        self, days: int, campaign_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get campaign performance data from the Google Ads API.

        Args:
            days: Number of days of historical data to use
            campaign_ids: Optional list of campaign IDs to filter

        Returns:
            List of campaign performance dictionaries
        """
        try:
            if self.ads_api:
                campaigns = self.ads_api.get_campaign_performance(days_ago=days)

                # Filter by campaign IDs if provided
                if campaign_ids:
                    campaigns = [c for c in campaigns if c.get("campaign_id") in campaign_ids]

                return campaigns
            else:
                self.logger.warning("No ads API client available")
                return []
        except Exception as e:
            self.logger.error(f"Error getting campaign performance: {str(e)}")
            return []

    def _get_keyword_performance(
        self, days: int, campaign_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get keyword performance data from the Google Ads API.

        Args:
            days: Number of days of historical data to use
            campaign_ids: Optional list of campaign IDs to filter

        Returns:
            List of keyword performance dictionaries
        """
        try:
            if self.ads_api:
                keywords = self.ads_api.get_keyword_performance(
                    days_ago=days, campaign_ids=campaign_ids
                )
                return keywords
            else:
                self.logger.warning("No ads API client available")
                return []
        except Exception as e:
            self.logger.error(f"Error getting keyword performance: {str(e)}")
            return []

    def _get_campaign_daily_performance(self, days: int) -> Dict[str, pd.DataFrame]:
        """
        Get daily campaign performance data.

        Args:
            days: Number of days of historical data to use

        Returns:
            Dictionary mapping campaign IDs to DataFrames with daily performance
        """
        try:
            if self.ads_api:
                daily_stats = self.ads_api.get_campaign_daily_stats(days_ago=days)
                result = {}

                # Convert to dictionary of DataFrames
                for campaign_id, stats in daily_stats.items():
                    result[campaign_id] = pd.DataFrame(stats)

                return result
            else:
                self.logger.warning("No ads API client available")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting daily campaign performance: {str(e)}")
            return {}

    def _run_optimization(
        self,
        campaign_df: pd.DataFrame,
        objective: str,
        constraint: str,
        budget_limit: float,
        algorithm: str = "convex",
        risk_tolerance: float = 0.5,
        min_budget_per_campaign: Optional[float] = None,
        multi_objective_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Run portfolio optimization using various algorithms.

        Args:
            campaign_df: DataFrame with campaign performance data
            objective: Objective function to maximize (conversions, clicks, roas, multi)
            constraint: Type of constraint (budget, target_cpa, target_roas)
            budget_limit: Total budget limit
            algorithm: Optimization algorithm to use (convex, efficient_frontier, multi_objective, bayesian)
            risk_tolerance: Risk tolerance parameter (0-1), higher means more risk-seeking
            min_budget_per_campaign: Minimum budget allocation per campaign
            multi_objective_weights: Weights for different objectives when using multi-objective optimization

        Returns:
            Dictionary with optimization results
        """
        try:
            # Extract data
            n_campaigns = len(campaign_df)

            # Get performance metrics based on objective
            metrics = self._get_performance_metrics(campaign_df, objective, multi_objective_weights)

            # Set minimum budget per campaign if not provided
            if min_budget_per_campaign is None:
                current_budgets = campaign_df["budget"].values
                min_budget_per_campaign = np.max(
                    [np.min(current_budgets) * 0.1, 1.0]
                )  # At least $1 or 10% of min

            # Select the optimization algorithm
            if algorithm == "convex":
                return self._run_convex_optimization(
                    campaign_df=campaign_df,
                    metrics=metrics,
                    budget_limit=budget_limit,
                    min_budget=min_budget_per_campaign,
                )
            elif algorithm == "efficient_frontier":
                return self._run_efficient_frontier_optimization(
                    campaign_df=campaign_df,
                    metrics=metrics,
                    budget_limit=budget_limit,
                    risk_tolerance=risk_tolerance,
                    min_budget=min_budget_per_campaign,
                )
            elif algorithm == "multi_objective":
                return self._run_multi_objective_optimization(
                    campaign_df=campaign_df,
                    metrics=metrics,
                    budget_limit=budget_limit,
                    min_budget=min_budget_per_campaign,
                )
            elif algorithm == "bayesian":
                return self._run_bayesian_optimization(
                    campaign_df=campaign_df,
                    metrics=metrics,
                    budget_limit=budget_limit,
                    min_budget=min_budget_per_campaign,
                )
            else:
                self.logger.warning(f"Unsupported algorithm: {algorithm}")
                return {"status": "failed", "message": f"Unsupported algorithm: {algorithm}"}

        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def _get_performance_metrics(
        self,
        campaign_df: pd.DataFrame,
        objective: str,
        multi_objective_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics for optimization.

        Args:
            campaign_df: DataFrame with campaign performance data
            objective: Objective function to maximize
            multi_objective_weights: Weights for different objectives when using multi-objective optimization

        Returns:
            Dictionary with performance metrics for optimization
        """
        # Calculate performance metrics
        conversions = campaign_df["conversions"].values
        clicks = campaign_df["clicks"].values
        impressions = campaign_df["impressions"].values
        cost = campaign_df["cost"].values
        conversion_value = campaign_df.get(
            "conversion_value", campaign_df.get("conv_value", conversions * 50)
        ).values

        # Calculate derived metrics
        ctr = np.divide(
            clicks, impressions, out=np.zeros_like(clicks, dtype=float), where=impressions != 0
        )
        cvr = np.divide(
            conversions, clicks, out=np.zeros_like(conversions, dtype=float), where=clicks != 0
        )
        cpa = np.divide(
            cost, conversions, out=np.zeros_like(cost, dtype=float), where=conversions != 0
        )
        roas = np.divide(
            conversion_value,
            cost,
            out=np.zeros_like(conversion_value, dtype=float),
            where=cost != 0,
        )

        # Calculate performance per cost metrics
        conv_per_cost = np.divide(
            conversions, cost, out=np.zeros_like(conversions, dtype=float), where=cost != 0
        )
        clicks_per_cost = np.divide(
            clicks, cost, out=np.zeros_like(clicks, dtype=float), where=cost != 0
        )
        imp_per_cost = np.divide(
            impressions, cost, out=np.zeros_like(impressions, dtype=float), where=cost != 0
        )
        value_per_cost = roas  # Same as ROAS

        # Calculate variance metrics for risk assessment
        conv_variance = self._calculate_variance(campaign_df, "conversions")
        cpa_variance = self._calculate_variance(campaign_df, "cpa")
        roas_variance = self._calculate_variance(campaign_df, "roas")

        # Return all metrics
        return {
            "conversions": conversions,
            "clicks": clicks,
            "impressions": impressions,
            "cost": cost,
            "conversion_value": conversion_value,
            "ctr": ctr,
            "cvr": cvr,
            "cpa": cpa,
            "roas": roas,
            "conv_per_cost": conv_per_cost,
            "clicks_per_cost": clicks_per_cost,
            "imp_per_cost": imp_per_cost,
            "value_per_cost": value_per_cost,
            "conv_variance": conv_variance,
            "cpa_variance": cpa_variance,
            "roas_variance": roas_variance,
            "primary_objective": objective,
            "multi_objective_weights": multi_objective_weights,
        }

    def _calculate_variance(self, campaign_df: pd.DataFrame, metric: str) -> np.ndarray:
        """
        Calculate variance or use estimated variance for risk assessment.

        Args:
            campaign_df: DataFrame with campaign performance data
            metric: Metric to calculate variance for

        Returns:
            Array of variance values
        """
        # Try to use historical variance if available
        if f"{metric}_variance" in campaign_df.columns:
            return campaign_df[f"{metric}_variance"].values

        # Estimate variance based on the metric
        if metric == "conversions":
            # Use Poisson variance approximation: variance ≈ mean
            return campaign_df["conversions"].values
        elif metric == "cpa":
            # Estimate CPA variance using conversions as a proxy (more conversions = less variance)
            conv = campaign_df["conversions"].values
            return np.divide(1, conv, out=np.ones_like(conv), where=conv != 0)
        elif metric == "roas":
            # Estimate ROAS variance
            conv = campaign_df["conversions"].values
            return np.divide(1, conv, out=np.ones_like(conv), where=conv != 0)
        else:
            # Default: use 20% of the metric value as variance
            if metric in campaign_df.columns:
                return campaign_df[metric].values * 0.2
            else:
                return np.ones(len(campaign_df))

    def _run_convex_optimization(
        self,
        campaign_df: pd.DataFrame,
        metrics: Dict[str, Any],
        budget_limit: float,
        min_budget: float,
    ) -> Dict[str, Any]:
        """
        Run portfolio optimization using convex optimization.

        Args:
            campaign_df: DataFrame with campaign performance data
            metrics: Performance metrics from _get_performance_metrics
            budget_limit: Total budget limit
            min_budget: Minimum budget per campaign

        Returns:
            Dictionary with optimization results
        """
        try:
            # Extract data
            n_campaigns = len(campaign_df)
            current_budgets = campaign_df["budget"].values

            # Determine which performance metric to use based on primary objective
            objective = metrics["primary_objective"]
            if objective == "conversions":
                performance_per_cost = metrics["conv_per_cost"]
                current_performance = metrics["conversions"]
            elif objective == "clicks":
                performance_per_cost = metrics["clicks_per_cost"]
                current_performance = metrics["clicks"]
            elif objective == "impressions":
                performance_per_cost = metrics["imp_per_cost"]
                current_performance = metrics["impressions"]
            elif objective == "roas":
                performance_per_cost = metrics["roas"]
                current_performance = metrics["conversion_value"]
            elif objective == "multi" and metrics["multi_objective_weights"]:
                # Combine multiple objectives with weights
                weights = metrics["multi_objective_weights"]
                performance_per_cost = np.zeros(n_campaigns)
                current_performance = 0

                if "conversions" in weights:
                    w = weights["conversions"]
                    performance_per_cost += w * metrics["conv_per_cost"]
                    current_performance += w * np.sum(metrics["conversions"])
                if "clicks" in weights:
                    w = weights["clicks"]
                    performance_per_cost += w * metrics["clicks_per_cost"]
                    current_performance += w * np.sum(metrics["clicks"])
                if "impressions" in weights:
                    w = weights["impressions"]
                    performance_per_cost += w * metrics["imp_per_cost"]
                    current_performance += w * np.sum(metrics["impressions"])
                if "roas" in weights:
                    w = weights["roas"]
                    performance_per_cost += w * metrics["value_per_cost"]
                    current_performance += w * np.sum(metrics["conversion_value"])
            else:
                self.logger.warning(f"Unsupported objective: {objective}")
                return {"status": "failed", "message": f"Unsupported objective: {objective}"}

            # Setup variables and constraints for convex optimization
            budgets = cp.Variable(n_campaigns, nonnegative=True)

            # Objective function: maximize performance given budget allocation
            expected_performance = cp.sum(performance_per_cost * budgets)
            objective_function = cp.Maximize(expected_performance)

            # Constraints
            constraints = [
                cp.sum(budgets) <= budget_limit,  # Total budget constraint
            ]

            # Add minimum budget constraint for each campaign
            for i in range(n_campaigns):
                constraints.append(budgets[i] >= min_budget)

            # Add constraints to ensure allocations are not too far from current
            for i in range(n_campaigns):
                if current_budgets[i] > 0:
                    # Budgets should not change too drastically
                    constraints.append(budgets[i] <= current_budgets[i] * 2)  # Max 2x increase
                    constraints.append(budgets[i] >= current_budgets[i] * 0.5)  # Max 50% decrease

            # Solve the problem
            prob = cp.Problem(objective_function, constraints)
            try:
                prob.solve()
            except cp.SolverError:
                # Try alternative solver if first one fails
                prob.solve(solver=cp.SCS)

            # Check if solved
            if prob.status != "optimal":
                self.logger.warning(f"Optimization did not reach optimal solution: {prob.status}")

            # Extract results
            optimal_budgets = budgets.value

            # Handle NaN or None values
            if optimal_budgets is None or np.any(np.isnan(optimal_budgets)):
                self.logger.warning("Optimization returned invalid budgets, using current budgets")
                optimal_budgets = current_budgets

            # Calculate expected improvement
            expected_new_performance = np.sum(performance_per_cost * optimal_budgets)
            expected_improvement = (
                (expected_new_performance - np.sum(performance_per_cost * current_budgets))
                / np.sum(performance_per_cost * current_budgets)
                if np.sum(performance_per_cost * current_budgets) > 0
                else 0
            )

            # Calculate expected performance by metric
            if objective == "conversions":
                expected_conversions = expected_new_performance
                expected_clicks = None
                expected_roas = None
            elif objective == "clicks":
                expected_conversions = None
                expected_clicks = expected_new_performance
                expected_roas = None
            elif objective == "roas":
                expected_conversions = None
                expected_clicks = None
                expected_roas = expected_new_performance
            elif objective == "multi":
                # Calculate individual metrics from the multi-objective solution
                expected_conversions = np.sum(metrics["conv_per_cost"] * optimal_budgets)
                expected_clicks = np.sum(metrics["clicks_per_cost"] * optimal_budgets)
                expected_roas = np.sum(metrics["value_per_cost"] * optimal_budgets)
            else:
                expected_conversions = None
                expected_clicks = None
                expected_roas = None

            return {
                "status": "success",
                "optimal_budgets": optimal_budgets.tolist(),
                "expected_improvement": float(expected_improvement),
                "expected_conversions": (
                    float(expected_conversions) if expected_conversions is not None else None
                ),
                "expected_clicks": float(expected_clicks) if expected_clicks is not None else None,
                "expected_roas": float(expected_roas) if expected_roas is not None else None,
                "solver_status": prob.status,
            }

        except Exception as e:
            self.logger.error(f"Error in convex optimization: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def _run_efficient_frontier_optimization(
        self,
        campaign_df: pd.DataFrame,
        metrics: Dict[str, Any],
        budget_limit: float,
        risk_tolerance: float,
        min_budget: float,
    ) -> Dict[str, Any]:
        """
        Run efficient frontier optimization balancing returns and risk.

        Args:
            campaign_df: DataFrame with campaign performance data
            metrics: Performance metrics from _get_performance_metrics
            budget_limit: Total budget limit
            risk_tolerance: Risk tolerance parameter (0-1)
            min_budget: Minimum budget per campaign

        Returns:
            Dictionary with optimization results
        """
        try:
            # Extract data
            n_campaigns = len(campaign_df)
            current_budgets = campaign_df["budget"].values

            # Get returns and risk metrics based on objective
            objective = metrics["primary_objective"]

            if objective == "conversions":
                returns = metrics["conv_per_cost"]
                risks = np.sqrt(
                    metrics["conv_variance"] / (metrics["conversions"] + 1)
                )  # Normalized std deviation
                current_performance = metrics["conversions"]
            elif objective == "roas":
                returns = metrics["value_per_cost"]
                risks = np.sqrt(metrics["roas_variance"])
                current_performance = metrics["conversion_value"]
            else:
                # Default to conversion metrics
                returns = metrics["conv_per_cost"]
                risks = np.sqrt(metrics["conv_variance"] / (metrics["conversions"] + 1))
                current_performance = metrics["conversions"]

            # Handle zero/nan values
            returns = np.nan_to_num(returns, nan=0.0)
            risks = np.nan_to_num(risks, nan=1.0)

            # Scale risks to 0-1 range
            if np.max(risks) > 0:
                risks = risks / np.max(risks)

            # Create covariance matrix (simplified: using only diagonal elements)
            # In a real implementation, we'd estimate the full covariance matrix
            cov_matrix = np.diag(risks**2)

            # Set up variables and constraints
            weights = cp.Variable(n_campaigns, nonnegative=True)

            # Calculate expected return and risk
            exp_return = returns @ weights
            risk = cp.quad_form(weights, cov_matrix)

            # Objective: maximize return - (risk_aversion * risk)
            # risk_aversion is (1 - risk_tolerance) to convert tolerance to aversion
            risk_aversion = 1 - risk_tolerance
            objective_function = cp.Maximize(exp_return - risk_aversion * risk)

            # Budget constraint scaled to 1.0 total
            budget_weights = weights * budget_limit / cp.sum(weights)

            # Constraints
            constraints = [
                cp.sum(weights) <= 1.0,  # Weights sum to at most 1
                weights >= min_budget / budget_limit,  # Minimum allocation
            ]

            # Add constraints to ensure allocations are not too far from current
            for i in range(n_campaigns):
                if current_budgets[i] > 0:
                    # Weights should not change too drastically
                    weight_i = current_budgets[i] / np.sum(current_budgets)
                    constraints.append(weights[i] <= weight_i * 2)  # Max 2x increase
                    constraints.append(weights[i] >= weight_i * 0.5)  # Max 50% decrease

            # Solve the problem
            prob = cp.Problem(objective_function, constraints)
            try:
                prob.solve()
            except cp.SolverError:
                # Try alternative solver
                prob.solve(solver=cp.SCS)

            # Check if solved
            if prob.status != "optimal":
                self.logger.warning(
                    f"Efficient frontier optimization did not reach optimal solution: {prob.status}"
                )

            # Convert weights to budget allocation
            if weights.value is not None:
                optimal_weights = weights.value
                optimal_budgets = (
                    optimal_weights * budget_limit / np.sum(optimal_weights)
                    if np.sum(optimal_weights) > 0
                    else current_budgets
                )
            else:
                self.logger.warning("Optimization failed to find a solution, using current budgets")
                optimal_budgets = current_budgets

            # Calculate expected improvement
            expected_new_performance = np.sum(returns * optimal_budgets)
            current_expected_performance = np.sum(returns * current_budgets)
            expected_improvement = (
                (expected_new_performance - current_expected_performance)
                / current_expected_performance
                if current_expected_performance > 0
                else 0
            )

            # Calculate Sharpe ratio (return-to-risk ratio)
            portfolio_return = returns @ optimal_weights
            portfolio_risk = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

            return {
                "status": "success",
                "optimal_budgets": optimal_budgets.tolist(),
                "expected_improvement": float(expected_improvement),
                "sharpe_ratio": float(sharpe_ratio),
                "risk_tolerance_used": risk_tolerance,
                "solver_status": prob.status,
            }

        except Exception as e:
            self.logger.error(f"Error in efficient frontier optimization: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def _run_multi_objective_optimization(
        self,
        campaign_df: pd.DataFrame,
        metrics: Dict[str, Any],
        budget_limit: float,
        min_budget: float,
    ) -> Dict[str, Any]:
        """
        Run multi-objective optimization considering several objectives.

        Args:
            campaign_df: DataFrame with campaign performance data
            metrics: Performance metrics from _get_performance_metrics
            budget_limit: Total budget limit
            min_budget: Minimum budget per campaign

        Returns:
            Dictionary with optimization results
        """
        # Use the convex optimization method with multi-objective weights
        # This is a simplified approach - a more advanced implementation would use
        # proper multi-objective optimization techniques

        # Ensure we have weights
        if not metrics.get("multi_objective_weights"):
            metrics["multi_objective_weights"] = {
                "conversions": 0.6,
                "clicks": 0.2,
                "impressions": 0.1,
                "roas": 0.1,
            }

        # Set primary objective to multi
        metrics["primary_objective"] = "multi"

        # Run standard convex optimization with multi-objective setting
        return self._run_convex_optimization(
            campaign_df=campaign_df,
            metrics=metrics,
            budget_limit=budget_limit,
            min_budget=min_budget,
        )

    def _run_bayesian_optimization(
        self,
        campaign_df: pd.DataFrame,
        metrics: Dict[str, Any],
        budget_limit: float,
        min_budget: float,
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization for budget allocation.
        This is a placeholder for a more sophisticated implementation.

        Args:
            campaign_df: DataFrame with campaign performance data
            metrics: Performance metrics from _get_performance_metrics
            budget_limit: Total budget limit
            min_budget: Minimum budget per campaign

        Returns:
            Dictionary with optimization results
        """
        # For now, fall back to convex optimization
        # A full implementation would use a library like GPyOpt or scikit-optimize
        self.logger.info(
            "Bayesian optimization not fully implemented, using convex optimization instead"
        )
        return self._run_convex_optimization(
            campaign_df=campaign_df,
            metrics=metrics,
            budget_limit=budget_limit,
            min_budget=min_budget,
        )

    def _prepare_recommendations(
        self, campaign_df: pd.DataFrame, optimization_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prepare budget recommendations from optimization results.

        Args:
            campaign_df: DataFrame with campaign performance data
            optimization_results: Results from _run_optimization

        Returns:
            List of budget recommendations by campaign
        """
        recommendations = []

        if optimization_results.get("status") != "success":
            return recommendations

        optimal_budgets = optimization_results.get("optimal_budgets", [])

        for i, row in campaign_df.iterrows():
            if i < len(optimal_budgets):
                campaign_id = row.get("campaign_id")
                current_budget = row.get("budget")
                recommended_budget = round(optimal_budgets[i], 2)

                # Calculate metrics
                change_pct = (
                    (recommended_budget - current_budget) / current_budget * 100
                    if current_budget > 0
                    else 0
                )

                recommendations.append(
                    {
                        "campaign_id": campaign_id,
                        "campaign_name": row.get("campaign_name", ""),
                        "current_budget": current_budget,
                        "recommended_budget": recommended_budget,
                        "change_percentage": change_pct,
                        "current_performance": {
                            "clicks": row.get("clicks", 0),
                            "impressions": row.get("impressions", 0),
                            "conversions": row.get("conversions", 0),
                            "cost": row.get("cost", 0),
                            "conversion_value": row.get("conversion_value", 0),
                        },
                        "confidence": 0.8,  # Placeholder confidence level
                    }
                )

        return recommendations

    def _identify_keyword_overlaps(self, keyword_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify keywords that appear in multiple campaigns.

        Args:
            keyword_df: DataFrame with keyword performance data

        Returns:
            List of overlapping keyword dictionaries
        """
        overlaps = []

        # Group by keyword text
        grouped = keyword_df.groupby("keyword_text")

        for keyword, group in grouped:
            if len(group) > 1:  # Keyword appears in multiple places
                campaigns = group[
                    ["campaign_id", "campaign_name", "ad_group_id", "ad_group_name"]
                ].to_dict("records")

                overlaps.append(
                    {
                        "keyword_text": keyword,
                        "match_type": group["match_type"].iloc[
                            0
                        ],  # Use first match type as example
                        "occurrences": len(group),
                        "campaigns": campaigns,
                        "total_clicks": group["clicks"].sum(),
                        "total_conversions": group["conversions"].sum(),
                        "total_cost": group["cost"].sum(),
                    }
                )

        return overlaps

    def _identify_cannibalization(self, keyword_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify potential keyword cannibalization where keywords compete against each other.

        Args:
            keyword_df: DataFrame with keyword performance data

        Returns:
            List of potential cannibalization dictionaries
        """
        # Placeholder implementation
        return []

    def _identify_performance_disparities(self, keyword_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify keywords with performance disparities across campaigns.

        Args:
            keyword_df: DataFrame with keyword performance data

        Returns:
            List of performance disparity dictionaries
        """
        # Placeholder implementation
        return []

    def _generate_campaign_forecasts(
        self, performance_data: Dict[str, pd.DataFrame], forecast_days: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for each campaign using time series methods.

        Args:
            performance_data: Dictionary mapping campaign IDs to DataFrames
            forecast_days: Number of days to forecast

        Returns:
            Dictionary mapping campaign IDs to forecast DataFrames
        """
        # Placeholder implementation
        forecasts = {}

        for campaign_id, df in performance_data.items():
            # Create a simple forecast dataframe
            future_dates = [datetime.now() + timedelta(days=i) for i in range(forecast_days)]
            forecast_df = pd.DataFrame(
                {
                    "date": future_dates,
                    "forecasted_clicks": [0] * forecast_days,
                    "forecasted_conversions": [0] * forecast_days,
                    "forecasted_cost": [0] * forecast_days,
                    "forecasted_conversion_value": [0] * forecast_days,
                }
            )
            forecasts[campaign_id] = forecast_df

        return forecasts

    def _optimize_daily_allocations(
        self, forecasts: Dict[str, pd.DataFrame], objective: str
    ) -> List[Dict[str, Any]]:
        """
        Optimize budget allocation for each day in the forecast period.

        Args:
            forecasts: Dictionary mapping campaign IDs to forecast DataFrames
            objective: Objective function to maximize

        Returns:
            List of daily budget allocation dictionaries
        """
        # Placeholder implementation
        daily_allocations = []

        return daily_allocations
