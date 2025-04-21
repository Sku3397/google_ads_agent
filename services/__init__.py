"""
Google Ads Management System - Services Package

This package contains modular services for autonomous Google Ads management.
Each service handles a specific aspect of Google Ads campaign management,
optimization, and monitoring.
"""

# Allow services to be imported directly from the package
from typing import Any, Dict, List, Optional
from google.ads.googleads.client import GoogleAdsClient
import logging

# Common service utilities and definitions
from .base_service import BaseService
from .audit_service import AuditService
from .keyword_service import KeywordService
from .negative_keyword_service import NegativeKeywordService
from .bid_service import BidService
from .creative_service import CreativeService
from .quality_score_service import QualityScoreService
from .reporting_service import ReportingService

# from services.anomaly_detection_service import AnomalyDetectionService # Needs sklearn
from .scheduler_service import SchedulerService

# from services.data_persistence_service import DataPersistenceService # Module does not exist
from .bandit_service import BanditService
from .causal_inference_service import CausalInferenceService
from .generative_content_service import GenerativeContentService
from .data_visualization_service import DataVisualizationService
from .experimentation_service import ExperimentationService
from .meta_learning_service import MetaLearningService
from .forecasting_service import ForecastingService
from .personalization_service import PersonalizationService
from .serp_scraper_service import SERPScraperService
from .portfolio_optimization_service import PortfolioOptimizationService
from .self_play_service import SelfPlayService
from .landing_page_optimization_service import LandingPageOptimizationService
from .graph_optimization_service import GraphOptimizationService
from .voice_query_service import VoiceQueryService
from .expert_feedback_service import ExpertFeedbackService
from .contextual_signal_service import ContextualSignalService
from .trend_forecasting_service import TrendForecastingService
from .ltv_bidding_service import LTVBiddingService

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
