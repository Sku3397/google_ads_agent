"""
Bandit Service for multi-armed bandit algorithms in Google Ads optimization.

This service implements various bandit algorithms for dynamic budget allocation,
creative testing, and keyword optimization.
"""

# This file makes the bandit_service directory a Python package.

# Define the public API for this package
from .bandit_service import BanditService, BanditAlgorithm

__all__ = ["BanditService", "BanditAlgorithm"]

# Service metadata
SERVICE_NAME = "BanditService"
SERVICE_DESCRIPTION = "Multi-armed bandit algorithms for optimizing Google Ads"
SERVICE_VERSION = "2.0.0"
SUPPORTED_ALGORITHMS = [
    "thompson_sampling",
    "ucb",
    "epsilon_greedy",
    "contextual",
    "dynamic_thompson",
]
