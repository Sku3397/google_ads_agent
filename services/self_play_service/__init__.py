"""
Self Play Service package for Google Ads optimization.

This service provides advanced agent vs agent competition capabilities for
discovering optimal strategies in Google Ads campaigns through competitive
self-play techniques.
"""

# This file makes the self_play_service directory a Python package.

from .self_play_service import SelfPlayService

__all__ = ["SelfPlayService"]

# Service metadata
SERVICE_NAME = "SelfPlayService"
SERVICE_DESCRIPTION = "Agent vs agent optimization for Google Ads bidding strategies"
SERVICE_VERSION = "1.0.0"
SUPPORTED_ALGORITHMS = ["MCTS", "Minimax", "AlphaZero-inspired", "PBT"]
