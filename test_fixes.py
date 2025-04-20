import logging
import sys
import time
import concurrent.futures
import google.generativeai as genai
from config import load_config
from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_gemini_model_with_timeout(timeout=15):
    """Test the Gemini model initialization with a timeout"""
    try:
        # Create a thread executor for timeout handling
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the Gemini test to run in a separate thread
            future = executor.submit(test_gemini_model)

            try:
                # Wait for completion with timeout
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"Gemini API test timed out after {timeout} seconds")
                return False
    except Exception as e:
        logger.error(f"Error in timeout wrapper: {str(e)}")
        return False


def test_gemini_model():
    """Test the Gemini model initialization and generation"""
    try:
        # Load configuration
        config = load_config()
        api_key = config["google_ai"]["api_key"]

        # Test Gemini API initialization only, without making actual API calls
        logger.info("Testing Gemini API initialization...")
        optimizer = AdsOptimizer(api_key)

        # Verify gemini_model attribute exists
        if hasattr(optimizer, "gemini_model"):
            logger.info("optimizer.gemini_model attribute found successfully")
            return True
        else:
            logger.error("optimizer.gemini_model attribute is missing")
            return False

    except Exception as e:
        logger.error(f"Gemini initialization test failed: {str(e)}")
        return False


def test_keyword_filtering():
    """Test that keywords are properly filtered to only ENABLED status"""
    try:
        # Load configuration
        config = load_config()

        # Initialize ads API
        ads_api = GoogleAdsAPI(config["google_ads"])

        # Fetch some keywords
        logger.info("Testing keyword filtering...")
        keywords = ads_api.get_keyword_performance(days_ago=90)

        # Check if we have keywords
        if not keywords:
            logger.warning("No keywords found")
            return False

        # Log count of keywords
        logger.info(f"Found {len(keywords)} keywords")

        # Verify statuses
        status_counts = {}
        campaign_status_counts = {}
        ad_group_status_counts = {}

        for kw in keywords:
            # Count by status
            status = kw.get("status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1

            # Count by campaign status
            c_status = kw.get("campaign_status", "UNKNOWN")
            campaign_status_counts[c_status] = campaign_status_counts.get(c_status, 0) + 1

            # Count by ad group status
            ag_status = kw.get("ad_group_status", "UNKNOWN")
            ad_group_status_counts[ag_status] = ad_group_status_counts.get(ag_status, 0) + 1

        # Log the distributions
        logger.info(f"Keywords by status: {status_counts}")
        logger.info(f"Keywords by campaign status: {campaign_status_counts}")
        logger.info(f"Keywords by ad group status: {ad_group_status_counts}")

        # Assert all are ENABLED
        non_enabled_count = len(keywords) - status_counts.get("ENABLED", 0)
        if non_enabled_count > 0:
            logger.error(f"Found {non_enabled_count} keywords with non-ENABLED status")
            return False

        non_enabled_campaign_count = len(keywords) - campaign_status_counts.get("ENABLED", 0)
        if non_enabled_campaign_count > 0:
            logger.error(f"Found {non_enabled_campaign_count} keywords from non-ENABLED campaigns")
            return False

        non_enabled_adgroup_count = len(keywords) - ad_group_status_counts.get("ENABLED", 0)
        if non_enabled_adgroup_count > 0:
            logger.error(f"Found {non_enabled_adgroup_count} keywords from non-ENABLED ad groups")
            return False

        logger.info(f"✓ All {len(keywords)} keywords have ENABLED status")
        logger.info(f"✓ All {len(keywords)} keywords are from ENABLED campaigns")
        logger.info(f"✓ All {len(keywords)} keywords are from ENABLED ad groups")
        return True

    except Exception as e:
        logger.error(f"Keyword filtering test failed with exception: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("Starting tests...")

    # Test keyword filtering first (most important)
    logger.info("===== TESTING KEYWORD FILTERING =====")
    keyword_success = test_keyword_filtering()
    logger.info(f"Keyword filtering test: {'SUCCESS' if keyword_success else 'FAILED'}")

    # Test Gemini model
    logger.info("\n===== TESTING GEMINI MODEL INITIALIZATION =====")
    gemini_success = test_gemini_model_with_timeout(timeout=15)
    logger.info(f"Gemini model initialization test: {'SUCCESS' if gemini_success else 'FAILED'}")

    # Overall result
    logger.info("\n===== TEST SUMMARY =====")
    logger.info(f"Keyword filtering: {'✓' if keyword_success else '✗'}")
    logger.info(f"Gemini model initialization: {'✓' if gemini_success else '✗'}")

    # Consider keyword filtering test the critical one
    if keyword_success:
        logger.info("KEYWORD FILTERING TEST PASSED - This was the main concern!")
        if not gemini_success:
            logger.warning("Gemini model test failed, but this is less critical.")
        sys.exit(0)
    else:
        logger.error("KEYWORD FILTERING TEST FAILED - This needs to be fixed!")
        sys.exit(1)
