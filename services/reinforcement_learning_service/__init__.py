"""
Reinforcement Learning Service package for Google Ads optimization.

This service provides advanced reinforcement learning capabilities for
bidding optimization, budget allocation, and strategic decision-making
in Google Ads campaigns.
"""

# This file makes the reinforcement_learning_service directory a Python package.

from .reinforcement_learning_service import ReinforcementLearningService

__all__ = ["ReinforcementLearningService"]

# Service metadata
SERVICE_NAME = "ReinforcementLearningService"
SERVICE_DESCRIPTION = "Reinforcement learning for optimizing Google Ads bidding strategies"
SERVICE_VERSION = "2.0.0"
SUPPORTED_ALGORITHMS = ["DQN", "PPO", "A2C"]
