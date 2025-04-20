"""
Graph Optimization Service for Google Ads campaigns.

This service provides algorithms based on graph theory for optimizing
various aspects of Google Ads campaigns and account structure.
"""

from services.graph_optimization_service.graph_optimization_service import GraphOptimizationService

__all__ = ["GraphOptimizationService"]

# Service metadata
SERVICE_NAME = "GraphOptimizationService"
SERVICE_DESCRIPTION = "Graph theory based optimization for Google Ads"
SERVICE_VERSION = "1.0.0"
SUPPORTED_ALGORITHMS = [
    "network_flow",
    "page_rank",
    "community_detection",
    "path_optimization",
    "node_importance",
]
