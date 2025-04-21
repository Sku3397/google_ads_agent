"""
Unit tests for the BidService.
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime

# Assuming BidService is importable from services.bid_service
# Adjust the import path if necessary
from services.bid_service import BidService, META_LEARNING_AVAILABLE  # Revert to absolute import

# from ..services.bid_service import BidService, META_LEARNING_AVAILABLE # Use relative import


@pytest.fixture
def mock_ads_api():
    """Fixture for mocking the GoogleAdsAPI client."""
    mock = MagicMock()
    mock.client = MagicMock()  # Mock the underlying client if needed
    # Mock specific methods used by BidService
    mock.get_keyword_performance = MagicMock(return_value=[])
    mock.apply_optimization = MagicMock(return_value=(True, "Success"))
    mock.get_campaign_performance = MagicMock(return_value=[])
    # Add mocks for other API calls if needed
    return mock


@pytest.fixture
def mock_meta_learning_service():
    """Fixture for mocking the MetaLearningService."""
    if not META_LEARNING_AVAILABLE:
        return None
    mock = MagicMock()
    # Mock methods used by BidService
    mock.recommend_strategy = MagicMock(
        return_value={
            "recommended_strategy": "performance_based",
            "estimated_parameters": {},
            "confidence_score": 0.8,
        }
    )
    mock.record_strategy_execution = MagicMock()
    return mock


@pytest.fixture
def mock_config():
    """Fixture for providing a sample configuration."""
    return {
        "google_ads": {"customer_id": "1234567890"},
        "bid": {  # Example bid configuration
            "min_data_points": 10,
            "max_bid_increase_pct": 30,
            "max_bid_decrease_pct": 20,
            "min_conversions": 1,
            "target_cpa": 50.0,
            "target_roas": 3.5,
        },
        "account_main_goal": "maximize_conversions",
        "output_dir": "test_output/bids",
        "data_dir": "test_data",
    }


@pytest.fixture
def mock_logger():
    """Fixture for mocking the logger."""
    return MagicMock()


@pytest.fixture
def bid_service(mock_ads_api, mock_config, mock_logger, mock_meta_learning_service):
    """Fixture to create a BidService instance with mocks."""
    # Adjust constructor call based on actual BidService.__init__ signature
    service = BidService(
        client=mock_ads_api.client,  # Pass the mock client if needed directly
        customer_id=mock_config["google_ads"]["customer_id"],
        # Pass the ads_api mock if the service expects the wrapper
        ads_api=mock_ads_api,
        optimizer=None,  # Assuming optimizer is not directly used or mock if needed
        config=mock_config,
        logger=mock_logger,
        meta_learning_service=mock_meta_learning_service,
    )
    # Mock the internal ads_api attribute if it's set differently
    service.ads_api = mock_ads_api
    return service


# --- Test Cases ---


def test_bid_service_initialization(bid_service, mock_config):
    """Test if BidService initializes correctly."""
    assert bid_service is not None
    assert bid_service.customer_id == mock_config["google_ads"]["customer_id"]
    assert bid_service.target_cpa == mock_config["bid"]["target_cpa"]
    assert (
        bid_service.meta_learning_service is not None
        if META_LEARNING_AVAILABLE
        else bid_service.meta_learning_service is None
    )


def test_optimize_keyword_bids_no_data(bid_service, mock_ads_api):
    """Test bid optimization when no keyword data is returned."""
    mock_ads_api.get_keyword_performance.return_value = []
    result = bid_service.optimize_keyword_bids(days=7)
    assert result["status"] == "failed"
    assert "No keyword data available" in result["message"]
    mock_ads_api.get_keyword_performance.assert_called_once()
    # Meta learning should still be called to recommend strategy
    if bid_service.meta_learning_service:
        bid_service.meta_learning_service.recommend_strategy.assert_called_once()
    # Meta learning record should NOT be called if optimization fails early
    if bid_service.meta_learning_service:
        bid_service.meta_learning_service.record_strategy_execution.assert_not_called()


# Add more tests for different strategies:
# - test_optimize_bids_performance_based
# - test_optimize_bids_target_cpa
# - test_optimize_bids_target_roas (requires mocking conversion_value)
# - test_optimize_bids_position_based (requires mocking position metrics)

# Add tests for safety checks:
# - test_apply_bid_safety_checks_increase_cap
# - test_apply_bid_safety_checks_decrease_cap
# - test_apply_bid_safety_checks_min_bid

# Add tests for applying recommendations:
# - test_apply_bid_recommendations_success
# - test_apply_bid_recommendations_partial_failure
# - test_apply_bid_recommendations_api_error

# Add tests for budget optimization (if implemented):
# - test_optimize_campaign_budgets_success
# - test_optimize_campaign_budgets_no_data


# Example test for performance-based strategy (needs expansion)
def test_optimize_bids_performance_based_basic(bid_service, mock_ads_api):
    """Basic test for performance-based bid optimization."""
    mock_keywords = [
        {
            "ad_group_criterion_id": "1",
            "keyword_text": "kw1",
            "match_type": "EXACT",
            "current_bid": 1.0,
            "impressions": 100,
            "clicks": 10,
            "conversions": 2,
            "cost": 10.0,
            "quality_score": 7,
            "ad_group_id": "ag1",
            "ad_group_name": "Ad Group 1",
            "campaign_id": "camp1",
            "campaign_name": "Campaign 1",
        },
        {  # Low performance, should decrease bid
            "ad_group_criterion_id": "2",
            "keyword_text": "kw2",
            "match_type": "BROAD",
            "current_bid": 1.5,
            "impressions": 200,
            "clicks": 1,
            "conversions": 0,
            "cost": 1.5,
            "quality_score": 3,
            "ad_group_id": "ag1",
            "ad_group_name": "Ad Group 1",
            "campaign_id": "camp1",
            "campaign_name": "Campaign 1",
        },
        {  # High performance, should increase bid
            "ad_group_criterion_id": "3",
            "keyword_text": "kw3",
            "match_type": "PHRASE",
            "current_bid": 0.8,
            "impressions": 50,
            "clicks": 15,
            "conversions": 3,
            "cost": 12.0,
            "quality_score": 9,
            "ad_group_id": "ag1",
            "ad_group_name": "Ad Group 1",
            "campaign_id": "camp1",
            "campaign_name": "Campaign 1",
        },
    ]
    mock_ads_api.get_keyword_performance.return_value = mock_keywords
    result = bid_service.optimize_keyword_bids(days=7, strategy="performance_based")

    assert result["status"] == "success"
    assert result["total_recommendations"] == 2  # kw1 is average, kw2 decrease, kw3 increase
    assert len(result["bid_recommendations"]) == 2

    rec1 = next(r for r in result["bid_recommendations"] if r["keyword_id"] == "2")  # kw2 decrease
    assert rec1["recommended_bid"] < 1.5
    assert "Underperforming" in rec1["rationale"]

    rec2 = next(r for r in result["bid_recommendations"] if r["keyword_id"] == "3")  # kw3 increase
    assert rec2["recommended_bid"] > 0.8
    assert "High-performing" in rec2["rationale"]

    if bid_service.meta_learning_service:
        bid_service.meta_learning_service.recommend_strategy.assert_called_once()
        bid_service.meta_learning_service.record_strategy_execution.assert_called_once()


# Placeholder for future tests
@pytest.mark.skip(reason="Target CPA strategy test not fully implemented")
def test_optimize_bids_target_cpa():
    pass


@pytest.mark.skip(reason="Target ROAS strategy test not fully implemented")
def test_optimize_bids_target_roas():
    pass


@pytest.mark.skip(reason="Position based strategy test not fully implemented")
def test_optimize_bids_position_based():
    pass


@pytest.mark.skip(reason="Safety check tests not implemented")
def test_apply_bid_safety_checks():
    pass


@pytest.mark.skip(reason="Apply recommendations tests not implemented")
def test_apply_bid_recommendations():
    pass


# TODO: Add tests for data saving/loading if implemented within the service
# TODO: Add tests for interactions with the optimizer if used directly
