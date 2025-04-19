"""
Google Ads Management System - Services Package

This package contains modular services for autonomous Google Ads management.
Each service handles a specific aspect of Google Ads campaign management,
optimization, and monitoring.
"""

# Allow services to be imported directly from the package
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

__all__ = [
    'AuditService',
    'KeywordService',
    'NegativeKeywordService',
    'BidService',
    'CreativeService',
    'QualityScoreService',
    'AudienceService',
    'ReportingService',
    'AnomalyDetectionService',
    'SchedulerService',
    'DataPersistenceService',
    'ReinforcementLearningService'
] 