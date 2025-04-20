"""
Simple test script for the Portfolio Optimization Service.
"""

import os
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

from services.portfolio_optimization_service import PortfolioOptimizationService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestPortfolioOptimization")


def test_portfolio_optimization():
    """Test the Portfolio Optimization Service."""

    logger.info("Initializing Portfolio Optimization Service")
    service = PortfolioOptimizationService()

    # Create mock campaign data
    mock_campaigns = [
        {
            "campaign_id": "123456789",
            "campaign_name": "Campaign 1",
            "budget": 100.0,
            "cost": 95.0,
            "clicks": 200,
            "impressions": 10000,
            "conversions": 10,
            "conversion_value": 500,
        },
        {
            "campaign_id": "987654321",
            "campaign_name": "Campaign 2",
            "budget": 200.0,
            "cost": 180.0,
            "clicks": 350,
            "impressions": 15000,
            "conversions": 15,
            "conversion_value": 750,
        },
        {
            "campaign_id": "456789123",
            "campaign_name": "Campaign 3",
            "budget": 150.0,
            "cost": 140.0,
            "clicks": 280,
            "impressions": 12000,
            "conversions": 12,
            "conversion_value": 600,
        },
    ]

    # Mock the _get_campaign_performance method
    service._get_campaign_performance = lambda days, campaign_ids=None: mock_campaigns

    # Test optimize_campaign_portfolio
    logger.info("Testing optimize_campaign_portfolio")
    result = service.optimize_campaign_portfolio(
        days=30, objective="conversions", constraint="budget", budget_limit=500.0
    )

    logger.info(f"Optimization result: {result['status']}")
    if result["status"] == "success":
        logger.info(f"Generated {len(result['recommendations'])} recommendations")
        for rec in result["recommendations"]:
            logger.info(
                f"Campaign: {rec['campaign_name']}, "
                f"Current Budget: ${rec['current_budget']}, "
                f"Recommended Budget: ${rec['recommended_budget']}, "
                f"Change: {rec['change_percentage']:.1f}%"
            )

    return result


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Run test
    result = test_portfolio_optimization()

    print("\nTest completed successfully.")
