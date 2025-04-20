"""
Landing Page Optimization Service for Google Ads campaigns.

This service provides tools for analyzing and optimizing landing pages
to improve conversion rates and user experience.
"""

from services.landing_page_optimization_service.landing_page_optimization_service import (
    LandingPageOptimizationService,
)

__all__ = ["LandingPageOptimizationService"]

# Service metadata
SERVICE_NAME = "LandingPageOptimizationService"
SERVICE_DESCRIPTION = "Landing page optimization for improving conversion rates"
SERVICE_VERSION = "1.0.0"
SUPPORTED_METHODS = [
    "a_b_testing",
    "element_analysis",
    "speed_optimization",
    "content_personalization",
    "form_optimization",
]
