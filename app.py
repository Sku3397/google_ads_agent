"""Main application module for Google Ads Agent.

This module provides autonomous management of Google Ads campaigns through various
optimization services and strategies. It includes functionality for campaign monitoring,
optimization suggestion generation, and automated application of improvements.
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, NoReturn, TypedDict
from datetime import datetime, timedelta
from functools import wraps

from tenacity import retry, stop_after_attempt, wait_exponential
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer
from services.base_service import BaseService
from services.quality_score_service import QualityScoreService
from services.creative_service import CreativeService
from services.reinforcement_learning_service import ReinforcementLearningService
from services.voice_query_service import VoiceQueryService
from services.graph_optimization_service import GraphOptimizationService
from services.expert_feedback_service import ExpertFeedbackService
from services.causal_inference_service import CausalInferenceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(context)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ads_agent.log"),
    ],
)

logger = logging.getLogger(__name__)


class OptimizationSuggestion(TypedDict):
    """Type definition for optimization suggestions."""

    type: str
    entity_type: str
    entity_id: str
    changes: Dict[str, Any]


class AdsAgent:
    """Main class for managing Google Ads campaigns autonomously.

    This class coordinates various optimization services to monitor campaign performance,
    generate optimization suggestions, and apply improvements automatically. It runs
    a background scheduler thread to periodically check and optimize campaigns.

    Attributes:
        config: Configuration dictionary containing Google Ads credentials and settings
        ads_api: Instance of GoogleAdsAPI for interacting with the Ads API
        optimizer: Instance of AdsOptimizer for applying optimizations
        services: Dictionary of optimization services keyed by service name
        scheduler_thread: Background thread for periodic optimization checks
        stop_scheduler: Event to signal the scheduler thread to stop
    """

    def __init__(self, config: Dict[str, str]) -> None:
        """Initialize the AdsAgent.

        Args:
            config: Configuration dictionary containing:
                - client_id: Google Ads API client ID
                - client_secret: Google Ads API client secret
                - developer_token: Google Ads API developer token
                - refresh_token: OAuth2 refresh token
                - customer_id: Google Ads customer ID
                - login_customer_id: Google Ads login customer ID
                - check_interval_seconds: Interval between optimization checks

        Raises:
            KeyError: If required configuration keys are missing
            ValueError: If configuration values are invalid
        """
        self.config = config
        self.ads_api = GoogleAdsAPI(config)
        self.optimizer = AdsOptimizer(self.ads_api)

        # Initialize services
        self.services: Dict[str, BaseService] = {
            "quality_score": QualityScoreService(self.ads_api, config),
            "creative": CreativeService(self.ads_api, config),
            "reinforcement_learning": ReinforcementLearningService(self.ads_api, config),
            "voice_query": VoiceQueryService(self.ads_api, config),
            "graph_optimization": GraphOptimizationService(self.ads_api, config),
            "expert_feedback": ExpertFeedbackService(self.ads_api, config),
            "causal_inference": CausalInferenceService(self.ads_api, config),
        }

        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_scheduler = threading.Event()

    def start(self) -> None:
        """Start the AdsAgent and its scheduler thread.

        This method initializes and starts the background scheduler thread that
        periodically checks and optimizes campaigns.
        """
        logger.info(
            "Starting AdsAgent...", extra={"context": {"services": list(self.services.keys())}}
        )
        self.scheduler_thread = threading.Thread(target=self._run_scheduler_thread)
        self.scheduler_thread.start()

    def stop(self) -> None:
        """Stop the AdsAgent and its scheduler thread.

        This method signals the scheduler thread to stop and waits for it to complete.
        """
        logger.info(
            "Stopping AdsAgent...",
            extra={"context": {"thread_active": bool(self.scheduler_thread)}},
        )
        if self.scheduler_thread:
            self.stop_scheduler.set()
            self.scheduler_thread.join()
        logger.info("AdsAgent stopped.")

    def _run_scheduler_thread(self) -> NoReturn:
        """Run the scheduler thread that periodically checks and optimizes campaigns.

        This method runs in a separate thread and continues until the stop_scheduler
        event is set. It catches and logs any exceptions that occur during execution.
        """
        while not self.stop_scheduler.is_set():
            try:
                self._check_and_optimize_campaigns()
            except Exception as e:
                logger.error(
                    "Error in scheduler thread",
                    extra={"context": {"error": str(e), "error_type": type(e).__name__}},
                )
                logger.exception("Full traceback:")

            # Sleep for the configured interval
            time.sleep(int(self.config.get("check_interval_seconds", 3600)))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _check_and_optimize_campaigns(self) -> None:
        """Check campaign performance and apply optimizations.

        This method retrieves campaign performance data, generates optimization
        suggestions from all services, and applies the suggested optimizations.
        It includes exponential backoff retry logic for transient failures.

        Raises:
            Exception: If there is an error checking or optimizing campaigns
        """
        try:
            # Get campaign performance data
            campaigns = self.ads_api.get_campaign_performance(days_ago=7)

            logger.info(
                "Retrieved campaign performance data",
                extra={"context": {"campaign_count": len(campaigns)}},
            )

            # Get optimization suggestions from each service
            all_suggestions = self._get_optimization_suggestions(campaigns)

            # Apply optimizations
            self._apply_optimizations(all_suggestions)

        except Exception as e:
            logger.error(
                "Error checking and optimizing campaigns",
                extra={"context": {"error": str(e), "error_type": type(e).__name__}},
            )
            raise

    def _get_optimization_suggestions(
        self, campaigns: List[Dict[str, Any]]
    ) -> List[OptimizationSuggestion]:
        """Get optimization suggestions from all services.

        Args:
            campaigns: List of campaign data dictionaries containing performance metrics

        Returns:
            List of optimization suggestions from all services, each containing:
                - type: Type of optimization
                - entity_type: Type of entity to optimize
                - entity_id: ID of entity to optimize
                - changes: Dictionary of changes to apply

        Raises:
            Exception: If there is an error getting suggestions from services
        """
        all_suggestions = []

        for service_name, service in self.services.items():
            try:
                suggestions = service.get_optimization_suggestions(campaigns)
                if suggestions:
                    all_suggestions.extend(suggestions)
                    logger.info(
                        "Retrieved optimization suggestions",
                        extra={
                            "context": {
                                "service": service_name,
                                "suggestion_count": len(suggestions),
                            }
                        },
                    )
            except Exception as e:
                logger.error(
                    "Error getting suggestions from service",
                    extra={
                        "context": {
                            "service": service_name,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                    },
                )
                logger.exception("Full traceback:")

        return all_suggestions

    def _apply_optimizations(self, suggestions: List[OptimizationSuggestion]) -> None:
        """Apply optimization suggestions to campaigns.

        Args:
            suggestions: List of optimization suggestions to apply, each containing:
                - type: Type of optimization
                - entity_type: Type of entity to optimize
                - entity_id: ID of entity to optimize
                - changes: Dictionary of changes to apply

        Raises:
            Exception: If there is an error applying optimizations
        """
        for suggestion in suggestions:
            try:
                success, message = self.optimizer.apply_optimization(
                    suggestion["type"],
                    suggestion["entity_type"],
                    suggestion["entity_id"],
                    suggestion["changes"],
                )

                log_level = logging.INFO if success else logging.WARNING
                logger.log(
                    log_level,
                    "Applied optimization",
                    extra={
                        "context": {
                            "success": success,
                            "message": message,
                            "optimization_type": suggestion["type"],
                            "entity_type": suggestion["entity_type"],
                            "entity_id": suggestion["entity_id"],
                        }
                    },
                )

            except Exception as e:
                logger.error(
                    "Error applying optimization",
                    extra={
                        "context": {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "suggestion": suggestion,
                        }
                    },
                )
                logger.exception("Full traceback:")


def main() -> None:
    """Main entry point for the application.

    This function loads configuration from environment variables, creates and starts
    the AdsAgent, and handles graceful shutdown on keyboard interrupt.

    Environment Variables:
        GOOGLE_ADS_CLIENT_ID: Google Ads API client ID
        GOOGLE_ADS_CLIENT_SECRET: Google Ads API client secret
        GOOGLE_ADS_DEVELOPER_TOKEN: Google Ads API developer token
        GOOGLE_ADS_REFRESH_TOKEN: OAuth2 refresh token
        GOOGLE_ADS_CUSTOMER_ID: Google Ads customer ID
        GOOGLE_ADS_LOGIN_CUSTOMER_ID: Google Ads login customer ID
        CHECK_INTERVAL_SECONDS: Optional interval between optimization checks

    Raises:
        KeyError: If required environment variables are missing
        Exception: If there is an error starting or running the agent
    """
    try:
        # Load configuration from environment variables
        config = {
            "client_id": os.environ["GOOGLE_ADS_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_ADS_CLIENT_SECRET"],
            "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
            "refresh_token": os.environ["GOOGLE_ADS_REFRESH_TOKEN"],
            "customer_id": os.environ["GOOGLE_ADS_CUSTOMER_ID"],
            "login_customer_id": os.environ["GOOGLE_ADS_LOGIN_CUSTOMER_ID"],
            "check_interval_seconds": os.environ.get("CHECK_INTERVAL_SECONDS", "3600"),
        }

        # Create and start the AdsAgent
        agent = AdsAgent(config)
        agent.start()

        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping AdsAgent...")
            agent.stop()

    except KeyError as e:
        logger.error(
            "Missing required environment variable", extra={"context": {"variable": str(e)}}
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            "Error in main", extra={"context": {"error": str(e), "error_type": type(e).__name__}}
        )
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
