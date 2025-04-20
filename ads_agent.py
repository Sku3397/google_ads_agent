#!/usr/bin/env python
"""
Google Ads Autonomous Management System

This is the main entry point for the Google Ads autonomous management system.
It orchestrates all services and provides a unified interface for managing Google Ads campaigns.
"""

import logging
import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Core components
from config import load_config
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer

# Services
from services.audit_service import AuditService
from services.keyword_service import KeywordService
from services.negative_keyword_service import NegativeKeywordService
from services.bid_service import BidService
from services.creative_service import CreativeService
from services.quality_score_service import QualityScoreService
from services.audience_service import AudienceService
from services.reporting_service import ReportingService
from services.anomaly_detection_service import AnomalyDetectionService
from services.scheduler_service import SchedulerService
from services.data_persistence_service import DataPersistenceService
from services.reinforcement_learning_service import ReinforcementLearningService
from services.bandit_service import BanditService
from services.causal_inference_service import CausalInferenceService
from services.meta_learning_service import MetaLearningService
from services.forecasting_service import ForecastingService
from services.experimentation_service import ExperimentationService
from services.personalization_service import PersonalizationService
from services.expert_feedback_service import ExpertFeedbackService
from services.serp_scraper_service import SERPScraperService
from services.ltv_bidding_service import LTVBiddingService
from services.portfolio_optimization_service import PortfolioOptimizationService
from services.self_play_service import SelfPlayService
from services.landing_page_optimization_service import LandingPageOptimizationService
from services.graph_optimization_service import GraphOptimizationService
from services.voice_query_service import VoiceQueryService


class AdsAgent:
    """
    Main orchestrator for the Google Ads autonomous management system.
    This class initializes and coordinates all services, providing a
    unified interface for managing Google Ads campaigns.
    """

    def __init__(self, config_path: str = ".env"):
        """
        Initialize the Ads Agent with all required services.

        Args:
            config_path: Path to the configuration file
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    f"logs/ads_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("AdsAgent")

        # Load configuration
        self.logger.info(f"Loading configuration from {config_path}")
        self.config = load_config(config_path)

        # Initialize core components
        self.logger.info("Initializing Google Ads API client")
        self.ads_api = GoogleAdsAPI(self.config["google_ads"])

        self.logger.info("Initializing optimizer")
        self.optimizer = AdsOptimizer(self.config["google_ai"])

        # Initialize services
        self.logger.info("Initializing services")
        self.services = self._initialize_services()

        self.logger.info("Google Ads agent initialized successfully")

    def _initialize_services(self) -> Dict[str, Any]:
        """
        Initialize all services with required dependencies.

        Returns:
            Dictionary mapping service names to service instances
        """
        services = {}

        # Initialize services with dependencies
        services["audit"] = AuditService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("AuditService"),
        )

        services["keyword"] = KeywordService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("KeywordService"),
        )

        services["negative_keyword"] = NegativeKeywordService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("NegativeKeywordService"),
        )

        services["bid"] = BidService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("BidService"),
        )

        # Initialize the new SchedulerService
        services["scheduler"] = SchedulerService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("SchedulerService"),
        )

        # Initialize the MetaLearningService
        services["meta_learning"] = MetaLearningService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("MetaLearningService"),
        )

        # Initialize the ForecastingService
        services["forecasting"] = ForecastingService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("ForecastingService"),
        )

        # Initialize the ExperimentationService
        services["experimentation"] = ExperimentationService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("ExperimentationService"),
        )

        # Initialize new ReinforcementLearningService
        services["reinforcement_learning"] = ReinforcementLearningService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("ReinforcementLearningService"),
        )

        # Initialize the new BanditService
        services["bandit"] = BanditService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("BanditService"),
        )

        # Initialize the new CausalInferenceService
        services["causal_inference"] = CausalInferenceService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("CausalInferenceService"),
        )

        # Initialize the new CreativeService
        services["creative"] = CreativeService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("CreativeService"),
        )

        # Initialize the new PersonalizationService
        services["personalization"] = PersonalizationService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("PersonalizationService"),
        )

        # Initialize the new ExpertFeedbackService
        services["expert_feedback"] = ExpertFeedbackService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("ExpertFeedbackService"),
        )

        # Initialize the new SERPScraperService
        services["serp_scraper"] = SERPScraperService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("SERPScraperService"),
        )

        # Initialize the new LTVBiddingService
        services["ltv_bidding"] = LTVBiddingService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("LTVBiddingService"),
        )

        # Initialize the new PortfolioOptimizationService
        services["portfolio_optimization"] = PortfolioOptimizationService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("PortfolioOptimizationService"),
        )

        # Initialize the new SelfPlayService
        services["self_play"] = SelfPlayService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("SelfPlayService"),
        )

        # Initialize the new LandingPageOptimizationService
        services["landing_page_optimization"] = LandingPageOptimizationService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("LandingPageOptimizationService"),
        )

        # Initialize the new GraphOptimizationService
        services["graph_optimization"] = GraphOptimizationService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("GraphOptimizationService"),
        )

        # Initialize the new VoiceQueryService
        services["voice_query"] = VoiceQueryService(
            ads_api=self.ads_api,
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger.getChild("VoiceQueryService"),
        )

        # Link SelfPlayService with ReinforcementLearningService
        if "reinforcement_learning" in services and "self_play" in services:
            services["self_play"].initialize_rl_service(services["reinforcement_learning"])

        # Additional services will be initialized similarly
        # We'll add them as they are implemented

        return services

    def run_comprehensive_audit(self, days: int = 30) -> Dict[str, Any]:
        """
        Run a comprehensive audit of the Google Ads account.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with audit results
        """
        self.logger.info(f"Running comprehensive audit for the last {days} days")

        try:
            # Run account structure audit
            audit_results = self.services["audit"].audit_account_structure(days)

            # Additional audit steps will be added as more services are implemented

            return audit_results

        except Exception as e:
            self.logger.error(f"Error running comprehensive audit: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def discover_keywords(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover new keywords for a campaign or the entire account.

        Args:
            campaign_id: Optional campaign ID to target specific campaign

        Returns:
            Dictionary with keyword suggestions
        """
        self.logger.info(f"Discovering keywords for campaign_id={campaign_id or 'all campaigns'}")

        try:
            # Use keyword service to discover new keywords
            keyword_results = self.services["keyword"].discover_new_keywords(campaign_id)

            return keyword_results

        except Exception as e:
            self.logger.error(f"Error discovering keywords: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def analyze_keyword_performance(
        self, days: int = 30, campaign_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze keyword performance and generate optimization recommendations.

        Args:
            days: Number of days to analyze
            campaign_id: Optional campaign ID to target specific campaign

        Returns:
            Dictionary with keyword performance analysis
        """
        self.logger.info(f"Analyzing keyword performance for the last {days} days")

        try:
            # Use keyword service to analyze performance
            performance_results = self.services["keyword"].analyze_keyword_performance(
                days, campaign_id
            )

            return performance_results

        except Exception as e:
            self.logger.error(f"Error analyzing keyword performance: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def run_scheduled_optimization(self):
        """
        Run a scheduled optimization of the Google Ads account.
        This method is meant to be called by a scheduler on a regular basis.

        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Running scheduled optimization")

        try:
            # Run account structure audit
            audit_results = self.services["audit"].audit_account_structure()

            # Analyze keyword performance
            keyword_results = self.services["keyword"].analyze_keyword_performance()

            # Additional optimization steps will be added as more services are implemented

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "audit_results": audit_results,
                "keyword_results": keyword_results,
            }

        except Exception as e:
            self.logger.error(f"Error running scheduled optimization: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def identify_negative_keywords(
        self, days: int = 30, campaign_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Identify potential negative keywords from search query data.

        Args:
            days: Number of days of data to analyze
            campaign_id: Optional campaign ID to filter data

        Returns:
            Dictionary with negative keyword suggestions
        """
        self.logger.info(f"Identifying negative keywords for the last {days} days")

        try:
            # Use negative keyword service to identify negative keywords
            results = self.services["negative_keyword"].identify_negative_keywords(
                days, campaign_id
            )

            return results

        except Exception as e:
            self.logger.error(f"Error identifying negative keywords: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def add_negative_keywords(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add negative keywords based on recommendations.

        Args:
            recommendations: List of negative keyword recommendations

        Returns:
            Dictionary with addition results
        """
        self.logger.info(f"Adding {len(recommendations)} negative keywords")

        try:
            # Use negative keyword service to add negative keywords
            results = self.services["negative_keyword"].add_negative_keywords(recommendations)

            return results

        except Exception as e:
            self.logger.error(f"Error adding negative keywords: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def optimize_keyword_bids(
        self, days: int = 30, campaign_id: Optional[str] = None, strategy: str = "performance_based"
    ) -> Dict[str, Any]:
        """
        Optimize keyword bids based on the specified strategy.

        Args:
            days: Number of days of data to analyze
            campaign_id: Optional campaign ID to filter keywords
            strategy: Bidding strategy to use
                - 'performance_based': General performance metrics
                - 'target_cpa': Target cost per acquisition
                - 'target_roas': Target return on ad spend
                - 'position_based': Position or impression share based

        Returns:
            Dictionary with bid optimization results
        """
        self.logger.info(f"Optimizing keyword bids using '{strategy}' strategy")

        try:
            # Use bid service to optimize keyword bids
            results = self.services["bid"].optimize_keyword_bids(days, campaign_id, strategy)

            return results

        except Exception as e:
            self.logger.error(f"Error optimizing keyword bids: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def apply_bid_recommendations(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply bid recommendations to keywords.

        Args:
            recommendations: List of bid recommendation dictionaries

        Returns:
            Dictionary with application results
        """
        self.logger.info(f"Applying {len(recommendations)} bid recommendations")

        try:
            # Use bid service to apply bid recommendations
            results = self.services["bid"].apply_bid_recommendations(recommendations)

            return results

        except Exception as e:
            self.logger.error(f"Error applying bid recommendations: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def optimize_campaign_budgets(self, days: int = 30) -> Dict[str, Any]:
        """
        Optimize campaign budgets based on performance.

        Args:
            days: Number of days of data to analyze

        Returns:
            Dictionary with budget optimization results
        """
        self.logger.info(f"Optimizing campaign budgets for the last {days} days")

        try:
            # Use bid service to optimize campaign budgets
            results = self.services["bid"].optimize_campaign_budgets(days)

            return results

        except Exception as e:
            self.logger.error(f"Error optimizing campaign budgets: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def train_rl_bidding_policy(
        self, campaign_id: Optional[str] = None, training_episodes: int = 1000
    ) -> Dict[str, Any]:
        """
        Train a reinforcement learning policy for bid optimization.

        Args:
            campaign_id: Optional campaign ID to train a policy for a specific campaign
            training_episodes: Number of episodes to train for

        Returns:
            Dictionary with training results
        """
        self.logger.info(
            f"Training RL bidding policy for campaign_id={campaign_id or 'all campaigns'}"
        )

        try:
            # Use reinforcement learning service to train policy
            results = self.services["reinforcement_learning"].train_policy(
                campaign_id=campaign_id, training_episodes=training_episodes
            )

            return results

        except Exception as e:
            self.logger.error(f"Error training RL bidding policy: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def generate_rl_bid_recommendations(
        self, campaign_id: Optional[str] = None, exploration_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate bid recommendations using reinforcement learning.

        Args:
            campaign_id: Optional campaign ID to generate recommendations for
            exploration_rate: Exploration rate for epsilon-greedy (0.0 to 1.0)

        Returns:
            Dictionary with bid recommendations
        """
        self.logger.info(
            f"Generating RL-based bid recommendations for campaign_id={campaign_id or 'all campaigns'}"
        )

        try:
            # Use reinforcement learning service to generate recommendations
            results = self.services["reinforcement_learning"].generate_bid_recommendations(
                campaign_id=campaign_id, exploration_rate=exploration_rate
            )

            return results

        except Exception as e:
            self.logger.error(f"Error generating RL-based bid recommendations: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def generate_ad_content(
        self,
        campaign_id: str,
        ad_group_id: str,
        keywords: Optional[List[str]] = None,
        product_info: Optional[Dict[str, Any]] = None,
        tone: str = "professional",
    ) -> Dict[str, Any]:
        """
        Generate ad content for a specific ad group using AI.

        Args:
            campaign_id: Campaign ID
            ad_group_id: Ad group ID to generate content for
            keywords: List of target keywords (optional)
            product_info: Product information dictionary (optional)
            tone: Tone for the ad content (professional, conversational, etc.)

        Returns:
            Dictionary with generated ad content
        """
        self.logger.info(
            f"Generating ad content for ad group {ad_group_id} in campaign {campaign_id}"
        )

        try:
            # Use creative service to generate ad content
            content_results = self.services["creative"].generate_ad_content(
                campaign_id=campaign_id,
                ad_group_id=ad_group_id,
                keywords=keywords,
                product_info=product_info,
                tone=tone,
            )

            return content_results

        except Exception as e:
            self.logger.error(f"Error generating ad content: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def analyze_strategy_performance(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the performance of strategies and provide recommendations
        using meta-learning capabilities.

        Args:
            service_name: Optional specific service to analyze

        Returns:
            Dictionary with analysis results and recommendations
        """
        self.logger.info(
            f"Analyzing strategy performance for service_name={service_name or 'all services'}"
        )

        try:
            meta_learning = self.services["meta_learning"]

            # If a specific service is requested, analyze just that service
            if service_name and service_name in self.services:
                # Get the context for the current account state
                context = {
                    "account_id": self.config["google_ads"]["customer_id"],
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    # Add additional context elements as needed
                }

                # Get available strategies for the service
                # This is a simplification - in reality, we would need to
                # query each service for its available strategies
                available_strategies = self._get_available_strategies(service_name)

                # Get recommendation
                recommendation = meta_learning.recommend_strategy(
                    service_name=service_name,
                    context=context,
                    available_strategies=available_strategies,
                )

                return {
                    "service": service_name,
                    "recommendations": recommendation,
                    "timestamp": datetime.now().isoformat(),
                }

            # Analyze patterns across all services
            analysis = meta_learning.analyze_cross_service_patterns()

            # Get per-service recommendations
            recommendations = {}
            for svc_name, service in self.services.items():
                # Skip if it's the meta_learning service itself
                if svc_name == "meta_learning":
                    continue

                # Get available strategies
                available_strategies = self._get_available_strategies(svc_name)

                if available_strategies:
                    # Get the context
                    context = {
                        "account_id": self.config["google_ads"]["customer_id"],
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        # Add additional context elements as needed
                    }

                    # Get recommendation
                    recommendation = meta_learning.recommend_strategy(
                        service_name=svc_name,
                        context=context,
                        available_strategies=available_strategies,
                    )

                    recommendations[svc_name] = recommendation

            return {
                "cross_service_analysis": analysis,
                "service_recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            error_message = f"Error analyzing strategy performance: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def _get_available_strategies(self, service_name: str) -> List[str]:
        """
        Get available strategies for a service.

        Args:
            service_name: Name of the service

        Returns:
            List of available strategy names
        """
        # This is a simplified implementation
        # In reality, we would either have a configuration or ask each service

        strategy_map = {
            "bid": ["performance_bidding", "target_cpa_bidding", "position_based_bidding"],
            "keyword": ["expansion", "refinement", "seasonal_adjustment"],
            "negative_keyword": ["query_analysis", "performance_based", "competitor_exclusion"],
            "creative": ["headline_generation", "description_generation", "ab_testing"],
            "reinforcement_learning": ["ppo_bidding", "dqn_bidding", "a2c_bidding"],
            "bandit": ["thompson_sampling", "ucb", "epsilon_greedy"],
            "causal_inference": ["uplift_analysis", "counterfactual_prediction"],
            "serp_scraper": ["competitor_analysis", "ranking_tracking", "serp_feature_analysis"],
        }

        return strategy_map.get(service_name, [])

    def forecast_performance(
        self,
        forecast_type: str = "metrics",
        metrics: Optional[List[str]] = None,
        days: int = 30,
        campaign_id: Optional[str] = None,
        target_metric: Optional[str] = None,
        target_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate forecasts for performance metrics, budgets, or search trends.

        Args:
            forecast_type: Type of forecast to generate:
                - 'metrics': Forecast performance metrics
                - 'budget': Forecast budget requirements
                - 'trends': Detect and forecast search trends
                - 'demand': Get demand forecasts from Google Ads
            metrics: List of metrics to forecast (for metrics forecast)
            days: Number of days to forecast
            campaign_id: Optional campaign ID to forecast for
            target_metric: Target metric for budget forecasting
            target_value: Target value to achieve for budget forecasting

        Returns:
            Dictionary with forecast results
        """
        self.logger.info(f"Generating {forecast_type} forecast for the next {days} days")

        try:
            forecasting_service = self.services["forecasting"]

            if forecast_type == "metrics":
                # Default metrics if not specified
                if metrics is None:
                    metrics = ["clicks", "impressions", "conversions", "cost"]

                # Generate metric forecasts
                results = forecasting_service.forecast_metrics(
                    metrics=metrics,
                    days_to_forecast=days,
                    campaign_id=campaign_id,
                    model_type="auto",
                )

                return results

            elif forecast_type == "budget":
                # Ensure we have a campaign ID for budget forecasting
                if campaign_id is None:
                    raise ValueError("Campaign ID is required for budget forecasting")

                # Default target metric if not specified
                if target_metric is None:
                    target_metric = "conversions"

                # Generate budget forecast
                results = forecasting_service.forecast_budget(
                    campaign_id=campaign_id,
                    days_to_forecast=days,
                    target_metric=target_metric,
                    target_value=target_value,
                )

                return results

            elif forecast_type == "trends":
                # Detect search trends
                results = forecasting_service.detect_search_trends(
                    days_lookback=max(90, days * 3),  # Use at least 90 days of history
                    min_growth_rate=0.1,
                )

                return results

            elif forecast_type == "demand":
                # Get demand forecasts from Google Ads
                results = forecasting_service.get_demand_forecasts()

                return results

            else:
                return {
                    "status": "error",
                    "message": f"Unknown forecast type: {forecast_type}. Must be 'metrics', 'budget', 'trends', or 'demand'.",
                }

        except Exception as e:
            error_message = f"Error generating forecast: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def create_experiment(
        self,
        name: str,
        type: str,
        hypothesis: str,
        control_group: str,
        treatment_groups: List[str],
        metrics: List[str],
        duration_days: int,
        traffic_split: Optional[Dict[str, float]] = None,
        custom_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new experiment for A/B testing or multivariate testing.

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
            Dictionary with experiment details including ID
        """
        self.logger.info(f"Creating experiment '{name}'")

        try:
            # Generate default traffic split if not provided
            if traffic_split is None:
                # Equal distribution between control and all treatments
                equal_share = 1.0 / (len(treatment_groups) + 1)
                traffic_split = {"control": equal_share}
                for treatment in treatment_groups:
                    traffic_split[treatment] = equal_share

            # Create experiment using experimentation service
            experiment_id = self.services["experimentation"].create_experiment(
                name=name,
                type=type,
                hypothesis=hypothesis,
                control_group=control_group,
                treatment_groups=treatment_groups,
                metrics=metrics,
                duration_days=duration_days,
                traffic_split=traffic_split,
                custom_parameters=custom_parameters,
            )

            # Get the full experiment details
            experiment = self.services["experimentation"].get_experiment(experiment_id)

            return {
                "status": "success",
                "message": f"Created experiment '{name}'",
                "experiment": experiment,
            }

        except Exception as e:
            error_message = f"Error creating experiment: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def list_experiments(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all experiments, optionally filtered by status.

        Args:
            status: Filter by experiment status (draft, running, completed, etc.)
            limit: Maximum number of experiments to return
            offset: Pagination offset

        Returns:
            Dictionary with list of experiments
        """
        self.logger.info(f"Listing experiments with status={status or 'all'}")

        try:
            experiments = self.services["experimentation"].list_experiments(
                status=status, limit=limit, offset=offset
            )

            return {"status": "success", "count": len(experiments), "experiments": experiments}

        except Exception as e:
            error_message = f"Error listing experiments: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Start an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            Dictionary with updated experiment
        """
        self.logger.info(f"Starting experiment {experiment_id}")

        try:
            experiment = self.services["experimentation"].start_experiment(experiment_id)

            return {
                "status": "success",
                "message": f"Started experiment '{experiment['name']}'",
                "experiment": experiment,
            }

        except Exception as e:
            error_message = f"Error starting experiment: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def stop_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Stop an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            Dictionary with updated experiment
        """
        self.logger.info(f"Stopping experiment {experiment_id}")

        try:
            experiment = self.services["experimentation"].stop_experiment(experiment_id)

            return {
                "status": "success",
                "message": f"Stopped experiment '{experiment['name']}'",
                "experiment": experiment,
            }

        except Exception as e:
            error_message = f"Error stopping experiment: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

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
        """
        self.logger.info(f"Analyzing experiment {experiment_id}")

        try:
            results = self.services["experimentation"].analyze_experiment(
                experiment_id=experiment_id, confidence_level=confidence_level
            )

            experiment = self.services["experimentation"].get_experiment(experiment_id)

            return {
                "status": "success",
                "message": f"Analyzed experiment '{experiment['name']}'",
                "experiment_name": experiment["name"],
                "results": results,
            }

        except Exception as e:
            error_message = f"Error analyzing experiment: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def apply_winning_variation(self, experiment_id: str) -> Dict[str, Any]:
        """
        Apply the winning variation from an experiment.

        Args:
            experiment_id: The ID of the experiment

        Returns:
            A dictionary with the result of applying the winning variation
        """
        self.logger.info(f"Applying winning variation for experiment {experiment_id}")

        try:
            # Use experimentation service to apply winning variation
            result = self.services["experimentation"].apply_winning_variation(experiment_id)

            return {
                "status": "success" if result else "failed",
                "message": (
                    f"Applied winning variation for experiment {experiment_id}"
                    if result
                    else f"Failed to apply winning variation for experiment {experiment_id}"
                ),
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            error_message = f"Error applying winning variation: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message, "experiment_id": experiment_id}

    def optimize_campaign_portfolio(
        self,
        days: int = 30,
        objective: str = "conversions",
        constraint: str = "budget",
        budget_limit: Optional[float] = None,
        campaign_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize budget allocation across campaigns to maximize overall performance.

        Args:
            days: Number of days of historical data to use
            objective: Objective function to maximize (conversions, clicks, roas)
            constraint: Type of constraint (budget, target_cpa, target_roas)
            budget_limit: Total budget limit across all campaigns
            campaign_ids: List of campaign IDs to include in the optimization

        Returns:
            Dictionary with optimization results
        """
        self.logger.info(
            f"Optimizing campaign portfolio for {objective} with {constraint} constraint"
        )

        try:
            # Use portfolio optimization service to optimize budget allocation
            results = self.services["portfolio_optimization"].optimize_campaign_portfolio(
                days=days,
                objective=objective,
                constraint=constraint,
                budget_limit=budget_limit,
                campaign_ids=campaign_ids,
            )

            return results

        except Exception as e:
            self.logger.error(f"Error optimizing campaign portfolio: {str(e)}")
            return {"status": "failed", "message": str(e)}

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
        self.logger.info(f"Applying {len(recommendations)} portfolio optimization recommendations")

        try:
            # Use portfolio optimization service to apply recommendations
            results = self.services["portfolio_optimization"].apply_portfolio_recommendations(
                recommendations
            )

            return results

        except Exception as e:
            self.logger.error(f"Error applying portfolio recommendations: {str(e)}")
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
        self.logger.info(f"Analyzing keywords across campaigns")

        try:
            # Use portfolio optimization service to analyze keywords across campaigns
            results = self.services["portfolio_optimization"].cross_campaign_keyword_analysis(
                days=days, campaign_ids=campaign_ids
            )

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing keywords across campaigns: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def get_experiment_recommendations(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get recommendations based on experiment results.

        Args:
            experiment_id: The experiment ID

        Returns:
            Dictionary with recommendations
        """
        self.logger.info(f"Getting recommendations for experiment {experiment_id}")

        try:
            recommendations = self.services["experimentation"].get_experiment_recommendations(
                experiment_id
            )
            experiment = self.services["experimentation"].get_experiment(experiment_id)

            return {
                "status": "success",
                "experiment_name": experiment["name"],
                "recommendations": recommendations,
            }

        except Exception as e:
            error_message = f"Error getting experiment recommendations: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def create_user_segments(self, days: int = 90) -> Dict[str, Any]:
        """
        Create user segments based on user behavior data.

        Args:
            days: Number of days of data to analyze

        Returns:
            Dictionary with user segments
        """
        self.logger.info(f"Creating user segments using {days} days of data")

        try:
            # Get user data from the API
            # In a real implementation, this would fetch actual user data
            # For now, we'll use a placeholder implementation in the service

            # Run the personalization update process
            results = self.services["personalization"].run_personalization_update()

            if results:
                segments = self.services["personalization"].user_segments
                return {
                    "status": "success",
                    "segments": segments,
                    "segment_count": len(segments),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to create user segments",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Error creating user segments: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def get_personalized_bid_adjustments(
        self,
        campaign_id: str,
        ad_group_id: str,
        user_segment: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get personalized bid adjustments for a specific user segment or user data.

        Args:
            campaign_id: Campaign ID
            ad_group_id: Ad group ID
            user_segment: Optional segment ID for the user
            user_data: Optional user data to determine segment

        Returns:
            Dictionary with bid adjustments
        """
        self.logger.info(
            f"Getting personalized bid adjustments for campaign {campaign_id}, ad group {ad_group_id}"
        )

        try:
            personalization_service = self.services["personalization"]

            # If user_segment is not provided but user_data is, determine the segment
            if not user_segment and user_data:
                user_segment = personalization_service.get_segment_for_user(user_data)
                self.logger.info(f"Determined user segment: {user_segment}")

            # If neither is provided, use a default segment
            if not user_segment:
                user_segment = "0"  # Default segment

            # Get bid adjustments
            adjustments = personalization_service.get_personalized_bid_adjustments(
                campaign_id=campaign_id, ad_group_id=ad_group_id, user_segment=user_segment
            )

            return {
                "status": "success",
                "segment_id": user_segment,
                "adjustments": adjustments,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting personalized bid adjustments: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def get_personalized_ads(
        self,
        ad_group_id: str,
        user_segment: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get personalized ad recommendations for a user segment or user data.

        Args:
            ad_group_id: Ad group ID
            user_segment: Optional segment ID for the user
            user_data: Optional user data to determine segment

        Returns:
            Dictionary with personalized ads
        """
        self.logger.info(f"Getting personalized ads for ad group {ad_group_id}")

        try:
            personalization_service = self.services["personalization"]

            # If user_segment is not provided but user_data is, determine the segment
            if not user_segment and user_data:
                user_segment = personalization_service.get_segment_for_user(user_data)
                self.logger.info(f"Determined user segment: {user_segment}")

            # If neither is provided, use a default segment
            if not user_segment:
                user_segment = "0"  # Default segment

            # Get available ads from the API
            if not self.ads_api:
                self.logger.error("Ads API not available")
                return {"status": "failed", "message": "Ads API not available"}

            # Get ads for the ad group
            available_ads = self.ads_api.get_ads(ad_group_id)

            if not available_ads:
                self.logger.warning(f"No ads found for ad group {ad_group_id}")
                return {
                    "status": "warning",
                    "message": f"No ads found for ad group {ad_group_id}",
                    "timestamp": datetime.now().isoformat(),
                }

            # Get personalized ads
            personalized_ads = personalization_service.get_personalized_ads(
                ad_group_id=ad_group_id, user_segment=user_segment, available_ads=available_ads
            )

            # Get ad customizers
            customizers = personalization_service.recommend_ad_customizers(
                ad_group_id=ad_group_id, user_segment=user_segment
            )

            return {
                "status": "success",
                "segment_id": user_segment,
                "ads": personalized_ads,
                "customizers": customizers,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting personalized ads: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def update_segment_performance(
        self, performance_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Update segment performance metrics based on new data.

        Args:
            performance_data: Optional DataFrame with performance metrics by segment

        Returns:
            Dictionary with updated segment performance
        """
        self.logger.info("Updating segment performance metrics")

        try:
            personalization_service = self.services["personalization"]

            if performance_data is None:
                # If no data provided, attempt to get it from the API
                # For now, this will use the mock implementation in the service
                results = personalization_service.run_personalization_update()

                return {
                    "status": "success" if results else "error",
                    "message": (
                        "Segment performance updated"
                        if results
                        else "Failed to update segment performance"
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # Update with provided data
                updated_segments = personalization_service.update_segment_performance(
                    performance_data
                )

                return {
                    "status": "success",
                    "segment_count": len(updated_segments),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Error updating segment performance: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def submit_recommendations_for_review(
        self,
        recommendation_type: str,
        recommendations: List[Dict[str, Any]],
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit recommendations for expert review.

        Args:
            recommendation_type: Type of recommendations (e.g., 'bid_adjustments', 'keywords')
            recommendations: List of recommendation dictionaries
            priority: Priority level ('high', 'normal', 'low')
            metadata: Additional metadata about the recommendations

        Returns:
            Dictionary with submission details
        """
        self.logger.info(
            f"Submitting {len(recommendations)} {recommendation_type} recommendations for expert review"
        )

        try:
            # Use expert feedback service for submission
            result = self.services["expert_feedback"].submit_for_review(
                recommendation_type=recommendation_type,
                recommendations=recommendations,
                priority=priority,
                metadata=metadata,
            )

            return result

        except Exception as e:
            self.logger.error(f"Error submitting recommendations for review: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def get_pending_expert_reviews(
        self,
        expert_id: Optional[str] = None,
        recommendation_type: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a list of pending expert reviews.

        Args:
            expert_id: Optional ID of the expert to filter by expertise
            recommendation_type: Optional type of recommendations to filter
            priority: Optional priority level to filter

        Returns:
            Dictionary with list of pending reviews
        """
        self.logger.info("Retrieving pending expert reviews")

        try:
            # Use expert feedback service to get pending reviews
            result = self.services["expert_feedback"].get_pending_reviews(
                expert_id=expert_id, recommendation_type=recommendation_type, priority=priority
            )

            return result

        except Exception as e:
            self.logger.error(f"Error retrieving pending expert reviews: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def apply_expert_feedback(
        self,
        submission_id: str,
        expert_id: str,
        action: str,
        feedback: Dict[str, Any],
        modified_recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Apply expert feedback to recommendations.

        Args:
            submission_id: ID of the submission to apply feedback to
            expert_id: ID of the expert providing feedback
            action: Action to take ('approve', 'reject', 'modify')
            feedback: Dictionary containing feedback information
            modified_recommendations: List of modified recommendation dictionaries (for 'modify' action)

        Returns:
            Dictionary with feedback application result
        """
        self.logger.info(f"Applying expert feedback for submission {submission_id}")

        try:
            if action == "approve":
                result = self.services["expert_feedback"].approve_recommendations(
                    submission_id=submission_id, expert_id=expert_id, feedback=feedback
                )
            elif action == "reject":
                result = self.services["expert_feedback"].reject_recommendations(
                    submission_id=submission_id, expert_id=expert_id, feedback=feedback
                )
            elif action == "modify":
                if not modified_recommendations:
                    return {
                        "status": "failed",
                        "message": "Modified recommendations required for 'modify' action",
                    }

                result = self.services["expert_feedback"].modify_recommendations(
                    submission_id=submission_id,
                    expert_id=expert_id,
                    modified_recommendations=modified_recommendations,
                    feedback=feedback,
                )
            else:
                return {"status": "failed", "message": f"Unknown action: {action}"}

            return result

        except Exception as e:
            self.logger.error(f"Error applying expert feedback: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def register_expert(
        self, expert_id: str, name: str, email: str, expertise: List[str], role: str = "reviewer"
    ) -> Dict[str, Any]:
        """
        Register a new expert in the system.

        Args:
            expert_id: Unique ID for the expert
            name: Expert's name
            email: Expert's email address
            expertise: List of areas of expertise
            role: Expert's role (reviewer, admin, etc.)

        Returns:
            Dictionary with registration result
        """
        self.logger.info(f"Registering expert {name} ({expert_id})")

        try:
            # Use expert feedback service to register expert
            result = self.services["expert_feedback"].register_expert(
                expert_id=expert_id, name=name, email=email, expertise=expertise, role=role
            )

            return result

        except Exception as e:
            self.logger.error(f"Error registering expert: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def get_experts(self, expertise: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a list of registered experts.

        Args:
            expertise: Optional expertise area to filter by

        Returns:
            Dictionary with list of experts
        """
        self.logger.info("Retrieving experts list")

        try:
            # Use expert feedback service to get experts
            result = self.services["expert_feedback"].get_experts(expertise=expertise)

            return result

        except Exception as e:
            self.logger.error(f"Error retrieving experts: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def learn_from_expert_feedback(self) -> Dict[str, Any]:
        """
        Analyze expert feedback to improve future recommendations.

        Returns:
            Dictionary with learning results
        """
        self.logger.info("Learning from expert feedback")

        try:
            # Use expert feedback service to learn from feedback
            result = self.services["expert_feedback"].learn_from_feedback()

            return result

        except Exception as e:
            self.logger.error(f"Error learning from expert feedback: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def analyze_serp_competitors(self, queries: List[str]) -> Dict[str, Any]:
        """
        Analyze competitor ads across multiple search queries using the SERP scraper.

        Args:
            queries: List of search queries to analyze

        Returns:
            Dictionary with competitor ad analysis
        """
        self.logger.info(f"Analyzing SERP competitors for {len(queries)} queries")

        try:
            # Use the SERP scraper service to analyze competitor ads
            results = self.services["serp_scraper"].analyze_competitor_ads(queries)

            return results

        except Exception as e:
            error_message = f"Error analyzing SERP competitors: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def track_keyword_rankings(self, keywords: List[str], domain: str) -> Dict[str, Any]:
        """
        Track organic rankings for specific keywords and domain.

        Args:
            keywords: List of keywords to track
            domain: Domain to track rankings for

        Returns:
            Dictionary with ranking data
        """
        self.logger.info(f"Tracking rankings for {len(keywords)} keywords on domain {domain}")

        try:
            # Use the SERP scraper service to track keyword rankings
            results = self.services["serp_scraper"].track_keyword_rankings(
                keywords=keywords, domain=domain
            )

            return results

        except Exception as e:
            error_message = f"Error tracking keyword rankings: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def analyze_serp_features(self, queries: List[str]) -> Dict[str, Any]:
        """
        Analyze SERP features across multiple search queries.

        Args:
            queries: List of search queries to analyze

        Returns:
            Dictionary with SERP feature analysis
        """
        self.logger.info(f"Analyzing SERP features for {len(queries)} queries")

        try:
            # Use the SERP scraper service to analyze SERP features
            results = self.services["serp_scraper"].analyze_serp_features(queries)

            return results

        except Exception as e:
            error_message = f"Error analyzing SERP features: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def scrape_single_serp(self, query: str, location: Optional[str] = None) -> Dict[str, Any]:
        """
        Scrape a single SERP for a given query.

        Args:
            query: Search query to scrape
            location: Optional location to use for the search

        Returns:
            Dictionary with SERP data
        """
        self.logger.info(f"Scraping SERP for query: {query}")

        try:
            # Use the SERP scraper service to scrape a single SERP
            result = self.services["serp_scraper"].scrape_serp(query=query, location=location)

            # Convert dataclass to dictionary for consistent return format
            result_dict = {
                "query": result.query,
                "timestamp": result.timestamp,
                "ads_top": result.ads_top,
                "ads_bottom": result.ads_bottom,
                "organic_results": result.organic_results,
                "related_searches": result.related_searches,
                "knowledge_panel": result.knowledge_panel,
                "local_pack": result.local_pack,
                "shopping_results": result.shopping_results,
            }

            return {"status": "success", "data": result_dict}

        except Exception as e:
            error_message = f"Error scraping SERP: {str(e)}"
            self.logger.error(error_message)
            return {"status": "failed", "message": error_message}

    def initialize_self_play_population(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize a population of competing agents for self-play optimization.

        Args:
            campaign_id: Optional campaign ID to focus on

        Returns:
            Dictionary with initialization results
        """
        self.logger.info(
            f"Initializing self-play agent population for campaign_id={campaign_id or 'all campaigns'}"
        )

        try:
            # Use self-play service to initialize population
            results = self.services["self_play"].initialize_population(campaign_id=campaign_id)

            return results

        except Exception as e:
            self.logger.error(f"Error initializing self-play population: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def run_self_play_tournament(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a tournament between competing agents to discover optimal bidding strategies.

        Args:
            campaign_id: Optional campaign ID to focus on

        Returns:
            Dictionary with tournament results
        """
        self.logger.info(
            f"Running self-play tournament for campaign_id={campaign_id or 'all campaigns'}"
        )

        try:
            # Use self-play service to run tournament
            results = self.services["self_play"].run_tournament(campaign_id=campaign_id)

            return results

        except Exception as e:
            self.logger.error(f"Error running self-play tournament: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def evolve_self_play_population(self) -> Dict[str, Any]:
        """
        Evolve the agent population using evolutionary algorithms.

        Returns:
            Dictionary with evolution results
        """
        self.logger.info("Evolving self-play agent population")

        try:
            # Use self-play service to evolve population
            results = self.services["self_play"].evolve_population()

            return results

        except Exception as e:
            self.logger.error(f"Error evolving self-play population: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def get_elite_strategy(self) -> Dict[str, Any]:
        """
        Get the best strategy from the current population.

        Returns:
            Dictionary with the best agent and its strategy
        """
        self.logger.info("Retrieving elite strategy from self-play population")

        try:
            # Use self-play service to get elite strategy
            results = self.services["self_play"].get_elite_strategy()

            return results

        except Exception as e:
            self.logger.error(f"Error getting elite strategy: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def generate_self_play_strategy_report(
        self, tournament_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a report on the evolution of strategies through self-play.

        Args:
            tournament_id: Optional tournament ID to focus on

        Returns:
            Dictionary with strategy evolution report
        """
        self.logger.info(
            f"Generating self-play strategy report for tournament_id={tournament_id or 'latest tournament'}"
        )

        try:
            # Use self-play service to generate strategy report
            results = self.services["self_play"].generate_strategy_report(
                tournament_id=tournament_id
            )

            return results

        except Exception as e:
            self.logger.error(f"Error generating self-play strategy report: {str(e)}")
            return {"status": "failed", "message": str(e)}

    def analyze_bid_impact(
        self,
        campaign_id: str,
        metric: str = "conversions",
        intervention_date: Optional[datetime] = None,
        post_period_days: int = 14,
    ) -> Dict[str, Any]:
        """
        Analyze the causal impact of bid changes on campaign performance.

        Args:
            campaign_id: ID of the campaign to analyze
            metric: Metric to analyze ('clicks', 'conversions', 'cost', etc.)
            intervention_date: Date when bid changes were applied (defaults to 14 days ago)
            post_period_days: Days after intervention to analyze

        Returns:
            Dict containing:
                - absolute_effect: Absolute lift in metric
                - relative_effect: Relative lift as percentage
                - confidence_intervals: Upper/lower bounds
                - p_value: Statistical significance
                - recommendations: List of follow-up actions
        """
        try:
            self.logger.info(f"Analyzing bid impact for campaign {campaign_id}")

            # Default to 14 days ago if no intervention date provided
            if intervention_date is None:
                intervention_date = datetime.now() - timedelta(days=14)

            # Get causal impact analysis
            impact = self.services["causal_inference"].measure_bid_impact(
                campaign_id=campaign_id,
                metric=metric,
                intervention_date=intervention_date,
                post_period_days=post_period_days,
            )

            # Generate recommendations based on impact
            recommendations = []

            if impact["p_value"] < 0.05:  # Statistically significant
                if impact["relative_effect"] > 0.1:  # >10% lift
                    recommendations.append(
                        {
                            "action": "maintain",
                            "reason": f"Bid changes produced significant positive lift of {impact['relative_effect']:.1%}",
                        }
                    )
                elif impact["relative_effect"] < -0.1:  # >10% drop
                    recommendations.append(
                        {
                            "action": "revert",
                            "reason": f"Bid changes produced significant negative impact of {impact['relative_effect']:.1%}",
                        }
                    )
            else:
                recommendations.append(
                    {
                        "action": "monitor",
                        "reason": "Impact not statistically significant yet, continue monitoring",
                    }
                )

            result = {
                **impact,
                "recommendations": recommendations,
                "campaign_id": campaign_id,
                "metric": metric,
                "intervention_date": intervention_date.isoformat(),
                "analysis_date": datetime.now().isoformat(),
            }

            self.logger.info(
                f"Bid impact analysis complete for campaign {campaign_id}: "
                f"{impact['relative_effect']:.1%} lift in {metric}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing bid impact: {str(e)}")
            raise

    def build_campaign_control_group(
        self,
        target_campaign_id: str,
        metric: str = "conversions",
        min_similarity: float = 0.7,
        max_candidates: int = 5,
    ) -> Dict[str, Any]:
        """
        Build a synthetic control group for a campaign using similar campaigns.

        Args:
            target_campaign_id: Campaign to build control for
            metric: Metric to optimize similarity for
            min_similarity: Minimum correlation coefficient (0-1)
            max_candidates: Maximum number of control campaigns

        Returns:
            Dict containing:
                - control_campaigns: List of campaign IDs and weights
                - similarity_scores: Correlation with each control
                - validation_metrics: Control group quality metrics
        """
        try:
            self.logger.info(f"Building control group for campaign {target_campaign_id}")

            # Get all active campaigns
            campaigns = self.ads_api.get_campaign_performance(days=30)
            candidate_ids = [
                c["id"]
                for c in campaigns
                if c["id"] != target_campaign_id and c["status"] == "ENABLED"
            ]

            if not candidate_ids:
                raise ValueError("No candidate campaigns found for control group")

            # Build synthetic control
            control_weights = self.services["causal_inference"].build_synthetic_control(
                target_campaign_id=target_campaign_id,
                candidate_campaign_ids=candidate_ids,
                metric=metric,
            )

            # Filter to best candidates
            top_controls = sorted(control_weights.items(), key=lambda x: abs(x[1]), reverse=True)[
                :max_candidates
            ]

            # Normalize weights to sum to 1
            total_weight = sum(abs(w) for _, w in top_controls)
            normalized_controls = {cid: weight / total_weight for cid, weight in top_controls}

            result = {
                "control_campaigns": normalized_controls,
                "target_campaign_id": target_campaign_id,
                "metric": metric,
                "creation_date": datetime.now().isoformat(),
                "validation_metrics": {
                    "num_controls": len(normalized_controls),
                    "total_weight": 1.0,
                },
            }

            self.logger.info(
                f"Built control group for campaign {target_campaign_id} "
                f"using {len(normalized_controls)} campaigns"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error building control group: {str(e)}")
            raise

    def schedule_impact_analysis(
        self,
        campaign_ids: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        analysis_hour: int = 6,
        analysis_minute: int = 0,
    ) -> Dict[str, Any]:
        """
        Schedule regular causal impact analysis for campaigns.

        Args:
            campaign_ids: List of campaign IDs to analyze (None for all)
            metrics: List of metrics to analyze (defaults to ['conversions', 'cost'])
            analysis_hour: Hour to run analysis (0-23)
            analysis_minute: Minute to run analysis (0-59)

        Returns:
            Dict containing scheduled task information
        """
        try:
            self.logger.info("Scheduling causal impact analysis tasks")

            if metrics is None:
                metrics = ["conversions", "cost"]

            # Get campaigns if not specified
            if campaign_ids is None:
                campaigns = self.ads_api.get_campaign_performance(days=1)
                campaign_ids = [c["id"] for c in campaigns if c["status"] == "ENABLED"]

            scheduled_tasks = []

            # Schedule analysis for each campaign and metric
            for campaign_id in campaign_ids:
                for metric in metrics:
                    task_name = f"causal_impact_analysis_{campaign_id}_{metric}"

                    # Create analysis function with fixed parameters
                    def analyze_task():
                        return self.analyze_bid_impact(campaign_id=campaign_id, metric=metric)

                    # Schedule daily analysis
                    task_id = self.services["scheduler"].schedule_daily(
                        function=analyze_task,
                        hour=analysis_hour,
                        minute=analysis_minute,
                        name=task_name,
                    )

                    scheduled_tasks.append(
                        {
                            "task_id": task_id,
                            "campaign_id": campaign_id,
                            "metric": metric,
                            "schedule": f"{analysis_hour:02d}:{analysis_minute:02d} daily",
                        }
                    )

            result = {
                "scheduled_tasks": scheduled_tasks,
                "num_campaigns": len(campaign_ids),
                "num_metrics": len(metrics),
                "total_tasks": len(scheduled_tasks),
            }

            self.logger.info(f"Scheduled {len(scheduled_tasks)} causal impact analysis tasks")

            return result

        except Exception as e:
            self.logger.error(f"Error scheduling impact analysis: {str(e)}")
            raise


def main():
    """
    Main entry point for the Google Ads autonomous management system.
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Google Ads Autonomous Management System")
    parser.add_argument("--config", default=".env", help="Path to configuration file")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    parser.add_argument("--campaign", help="Campaign ID to target (optional)")
    parser.add_argument("--ad-group", help="Ad group ID to target (for ad content generation)")
    parser.add_argument(
        "--action",
        choices=[
            "audit",
            "keywords",
            "performance",
            "optimize",
            "negative_keywords",
            "optimize_bids",
            "optimize_budgets",
            "train_rl_policy",
            "rl_bid_recommendations",
            "generate_ad_content",
            "analyze_strategies",
            "forecast_metrics",
            "forecast_budget",
            "forecast_trends",
            "forecast_demand",
            "create_experiment",
            "list_experiments",
            "start_experiment",
            "stop_experiment",
            "analyze_experiment",
            "apply_experiment",
            "experiment_recommendations",
            "create_user_segments",
            "get_personalized_bid_adjustments",
            "get_personalized_ads",
            "update_segment_performance",
            "submit_recommendations_for_review",
            "get_pending_expert_reviews",
            "apply_expert_feedback",
            "register_expert",
            "get_experts",
            "learn_from_expert_feedback",
            "analyze_serp_competitors",
            "track_keyword_rankings",
            "analyze_serp_features",
            "scrape_single_serp",
            "initialize_self_play_population",
            "run_self_play_tournament",
            "evolve_self_play_population",
            "get_elite_strategy",
            "generate_self_play_strategy_report",
        ],
        default="audit",
        help="Action to perform",
    )
    parser.add_argument(
        "--strategy",
        choices=["performance_based", "target_cpa", "target_roas", "position_based"],
        default="performance_based",
        help="Bid optimization strategy (for optimize_bids action)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Training episodes for RL policy training"
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=0.1,
        help="Exploration rate for RL-based recommendations (0.0 to 1.0)",
    )
    parser.add_argument(
        "--tone",
        choices=["professional", "conversational", "enthusiastic", "informative"],
        default="professional",
        help="Tone for ad content generation",
    )
    parser.add_argument("--service", help="Service to target for strategy analysis")
    parser.add_argument("--metrics", help="Comma-separated list of metrics to forecast")
    parser.add_argument("--target-metric", help="Target metric for budget forecasting")
    parser.add_argument("--target-value", type=float, help="Target value for budget forecasting")
    parser.add_argument("--experiment-name", help="Name for the experiment")
    parser.add_argument(
        "--experiment-type", choices=["A/B Test", "Multivariate Test"], help="Type of experiment"
    )
    parser.add_argument("--hypothesis", help="Hypothesis for the experiment")
    parser.add_argument("--treatment-groups", help="Comma-separated list of treatment groups")
    parser.add_argument("--experiment-metrics", help="Comma-separated list of metrics to track")
    parser.add_argument("--duration", type=int, default=30, help="Duration of experiment in days")
    parser.add_argument("--experiment-id", help="ID of the experiment to manage")
    parser.add_argument(
        "--confidence", type=float, default=0.95, help="Confidence level for experiment analysis"
    )

    args = parser.parse_args()

    # Initialize the agent
    agent = AdsAgent(config_path=args.config)

    # Perform the requested action
    if args.action == "audit":
        result = agent.run_comprehensive_audit(days=args.days)
    elif args.action == "keywords":
        result = agent.discover_keywords(campaign_id=args.campaign)
    elif args.action == "performance":
        result = agent.analyze_keyword_performance(days=args.days, campaign_id=args.campaign)
    elif args.action == "optimize":
        result = agent.run_scheduled_optimization()
    elif args.action == "negative_keywords":
        result = agent.identify_negative_keywords(days=args.days, campaign_id=args.campaign)
    elif args.action == "optimize_bids":
        result = agent.optimize_keyword_bids(
            days=args.days, campaign_id=args.campaign, strategy=args.strategy
        )
    elif args.action == "optimize_budgets":
        result = agent.optimize_campaign_budgets(days=args.days)
    elif args.action == "train_rl_policy":
        result = agent.train_rl_bidding_policy(
            campaign_id=args.campaign, training_episodes=args.episodes
        )
    elif args.action == "rl_bid_recommendations":
        result = agent.generate_rl_bid_recommendations(
            campaign_id=args.campaign, exploration_rate=args.exploration
        )
    elif args.action == "generate_ad_content":
        if not args.campaign or not args.ad_group:
            print("Error: Campaign ID and Ad Group ID are required for generate_ad_content")
            return

        result = agent.generate_ad_content(
            campaign_id=args.campaign, ad_group_id=args.ad_group, tone=args.tone
        )
    elif args.action == "analyze_strategies":
        result = agent.analyze_strategy_performance(service_name=args.service)
    elif args.action == "forecast_metrics":
        metrics = args.metrics.split(",") if args.metrics else None
        result = agent.forecast_performance(
            forecast_type="metrics", metrics=metrics, days=args.days, campaign_id=args.campaign
        )
    elif args.action == "forecast_budget":
        if not args.campaign:
            print("Error: Campaign ID is required for budget forecasting")
            return

        result = agent.forecast_performance(
            forecast_type="budget",
            days=args.days,
            campaign_id=args.campaign,
            target_metric=args.target_metric,
            target_value=args.target_value,
        )
    elif args.action == "forecast_trends":
        result = agent.forecast_performance(forecast_type="trends", days=args.days)
    elif args.action == "forecast_demand":
        result = agent.forecast_performance(forecast_type="demand")
    elif args.action == "create_experiment":
        if (
            not args.experiment_name
            or not args.experiment_type
            or not args.hypothesis
            or not args.campaign
            or not args.treatment_groups
            or not args.experiment_metrics
        ):
            print("Error: Missing required arguments for create_experiment")
            print(
                "Required: --experiment-name, --experiment-type, --hypothesis, --campaign, --treatment-groups, --experiment-metrics"
            )
            return

        result = agent.create_experiment(
            name=args.experiment_name,
            type=args.experiment_type,
            hypothesis=args.hypothesis,
            control_group=args.campaign,
            treatment_groups=args.treatment_groups.split(","),
            metrics=args.experiment_metrics.split(","),
            duration_days=args.duration,
        )
    elif args.action == "list_experiments":
        result = agent.list_experiments()
    elif args.action == "start_experiment":
        if not args.experiment_id:
            print("Error: Experiment ID is required for start_experiment")
            return

        result = agent.start_experiment(experiment_id=args.experiment_id)
    elif args.action == "stop_experiment":
        if not args.experiment_id:
            print("Error: Experiment ID is required for stop_experiment")
            return

        result = agent.stop_experiment(experiment_id=args.experiment_id)
    elif args.action == "analyze_experiment":
        if not args.experiment_id:
            print("Error: Experiment ID is required for analyze_experiment")
            return

        result = agent.analyze_experiment(
            experiment_id=args.experiment_id, confidence_level=args.confidence
        )
    elif args.action == "apply_experiment":
        if not args.experiment_id:
            print("Error: Experiment ID is required for apply_experiment")
            return

        result = agent.apply_winning_variation(experiment_id=args.experiment_id)
    elif args.action == "experiment_recommendations":
        if not args.experiment_id:
            print("Error: Experiment ID is required for experiment_recommendations")
            return

        result = agent.get_experiment_recommendations(experiment_id=args.experiment_id)
    elif args.action == "create_user_segments":
        result = agent.create_user_segments(days=args.days)
    elif args.action == "get_personalized_bid_adjustments":
        if not args.campaign or not args.ad_group:
            print(
                "Error: Campaign ID and Ad Group ID are required for get_personalized_bid_adjustments"
            )
            return

        result = agent.get_personalized_bid_adjustments(
            campaign_id=args.campaign, ad_group_id=args.ad_group
        )
    elif args.action == "get_personalized_ads":
        if not args.ad_group:
            print("Error: Ad Group ID is required for get_personalized_ads")
            return

        result = agent.get_personalized_ads(ad_group_id=args.ad_group)
    elif args.action == "update_segment_performance":
        result = agent.update_segment_performance()
    elif args.action == "submit_recommendations_for_review":
        if not args.recommendation_type or not args.recommendations:
            print(
                "Error: Recommendation type and recommendations are required for submit_recommendations_for_review"
            )
            return

        result = agent.submit_recommendations_for_review(
            recommendation_type=args.recommendation_type,
            recommendations=json.loads(args.recommendations),
            priority=args.priority,
            metadata=json.loads(args.metadata) if args.metadata else None,
        )
    elif args.action == "get_pending_expert_reviews":
        result = agent.get_pending_expert_reviews(
            expert_id=args.expert_id,
            recommendation_type=args.recommendation_type,
            priority=args.priority,
        )
    elif args.action == "apply_expert_feedback":
        if not args.submission_id or not args.expert_id or not args.action or not args.feedback:
            print(
                "Error: Submission ID, expert ID, action, and feedback are required for apply_expert_feedback"
            )
            return

        result = agent.apply_expert_feedback(
            submission_id=args.submission_id,
            expert_id=args.expert_id,
            action=args.action,
            feedback=json.loads(args.feedback),
            modified_recommendations=(
                json.loads(args.modified_recommendations) if args.modified_recommendations else None
            ),
        )
    elif args.action == "register_expert":
        if not args.expert_id or not args.name or not args.email or not args.expertise:
            print("Error: Expert ID, name, email, and expertise are required for register_expert")
            return

        result = agent.register_expert(
            expert_id=args.expert_id,
            name=args.name,
            email=args.email,
            expertise=json.loads(args.expertise),
            role=args.role,
        )
    elif args.action == "get_experts":
        result = agent.get_experts(expertise=args.expertise)
    elif args.action == "learn_from_expert_feedback":
        result = agent.learn_from_expert_feedback()
    elif args.action == "analyze_serp_competitors":
        result = agent.analyze_serp_competitors(json.loads(args.queries))
    elif args.action == "track_keyword_rankings":
        result = agent.track_keyword_rankings(json.loads(args.keywords), args.domain)
    elif args.action == "analyze_serp_features":
        result = agent.analyze_serp_features(json.loads(args.queries))
    elif args.action == "scrape_single_serp":
        result = agent.scrape_single_serp(args.query, args.location)
    elif args.action == "initialize_self_play_population":
        result = agent.initialize_self_play_population(campaign_id=args.campaign)
    elif args.action == "run_self_play_tournament":
        result = agent.run_self_play_tournament(campaign_id=args.campaign)
    elif args.action == "evolve_self_play_population":
        result = agent.evolve_self_play_population()
    elif args.action == "get_elite_strategy":
        result = agent.get_elite_strategy()
    elif args.action == "generate_self_play_strategy_report":
        result = agent.generate_self_play_strategy_report(tournament_id=args.tournament_id)
    else:
        print(f"Error: Unknown action '{args.action}'")
        return

    # Print the result
    import json

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
