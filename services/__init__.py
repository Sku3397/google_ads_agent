"""
Google Ads Management System - Services Package

This package contains modular services for autonomous Google Ads management.
Each service handles a specific aspect of Google Ads campaign management,
optimization, and monitoring.
"""

# Allow services to be imported directly from the package
from services.base_service import BaseService
from services.audit_service import AuditService
from services.keyword_service import KeywordService
from services.negative_keyword_service import NegativeKeywordService
from services.bid_service import BidService
from services.creative_service import CreativeService
from services.quality_score_service import QualityScoreService
from services.reporting_service.reporting_service import ReportingService

# from services.anomaly_detection_service import AnomalyDetectionService # Needs sklearn
from services.scheduler_service.scheduler_service import SchedulerService

# from services.data_persistence_service import DataPersistenceService # Module does not exist
from services.bandit_service import BanditService
from services.causal_inference_service import CausalInferenceService
from services.generative_content_service import GenerativeContentService
from services.data_visualization_service import DataVisualizationService
from services.experimentation_service import ExperimentationService
from services.meta_learning_service import MetaLearningService
from services.forecasting_service import ForecastingService
from services.personalization_service import PersonalizationService
from services.serp_scraper_service import SERPScraperService
from services.portfolio_optimization_service import PortfolioOptimizationService
from services.self_play_service import SelfPlayService
from services.landing_page_optimization_service import LandingPageOptimizationService
from services.graph_optimization_service import GraphOptimizationService
from services.voice_query_service import VoiceQueryService
from services.expert_feedback_service import ExpertFeedbackService
from services.contextual_signal_service import ContextualSignalService
from services.trend_forecasting_service import TrendForecastingService
from services.ltv_bidding_service import LTVBiddingService

__all__ = [
    "BaseService",
    "AuditService",
    "KeywordService",
    "NegativeKeywordService",
    "BidService",
    "CreativeService",
    "QualityScoreService",
    "ReportingService",
    # "AnomalyDetectionService", # Needs sklearn
    "SchedulerService",
    # "DataPersistenceService", # Module does not exist
    # "ReinforcementLearningService", # Needs torch
    "BanditService",
    "CausalInferenceService",
    "GenerativeContentService",
    "DataVisualizationService",
    "ExperimentationService",
    "MetaLearningService",
    "ForecastingService",
    "PersonalizationService",
    "SERPScraperService",
    "PortfolioOptimizationService",
    "SelfPlayService",
    "LandingPageOptimizationService",
    "GraphOptimizationService",
    "VoiceQueryService",
    "ExpertFeedbackService",
    "ContextualSignalService",
    "TrendForecastingService",
    "LTVBiddingService",
]
