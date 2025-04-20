"""
Reinforcement Learning Service package for Google Ads optimization.

This service provides advanced reinforcement learning capabilities for
bidding optimization, budget allocation, and strategic decision-making
in Google Ads campaigns.
"""

from services.reinforcement_learning_service.reinforcement_learning_service import (
    ReinforcementLearningService,
    AdsEnvironment,
)

__all__ = ["ReinforcementLearningService", "AdsEnvironment"]

# Service metadata
SERVICE_NAME = "ReinforcementLearningService"
SERVICE_DESCRIPTION = "Reinforcement learning for optimizing Google Ads bidding strategies"
SERVICE_VERSION = "2.0.0"
SUPPORTED_ALGORITHMS = ["DQN", "PPO", "A2C"]
