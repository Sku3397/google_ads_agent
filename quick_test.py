import logging
import requests
import google.generativeai as genai
from config import load_config
from ads_api import GoogleAdsAPI

# Configure logging to see detailed output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_gemini():
    """Test Gemini API with direct REST API approach"""
    try:
        # Load configuration
        config = load_config()
        api_key = config["google_ai"]["api_key"]

        # First try direct API approach
        logger.info("Testing direct API approach with gemini-pro model...")

        # API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

        # Request payload
        payload = {
            "contents": [{"parts": [{"text": "Write a single sentence about Google Ads."}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            },
        }

        # Make the request
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for non-2xx responses

        # Parse the response
        response_json = response.json()

        # Extract the generated text
        generated_text = ""
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            for part in response_json["candidates"][0]["content"]["parts"]:
                if "text" in part:
                    generated_text += part["text"]

        logger.info(f"Direct API Response: {generated_text}")

        # Try also with SDK (as a fallback test)
        logger.info("Also testing SDK approach...")
        try:
            genai.configure(api_key=api_key)

            # Try with different model names that might work with the SDK
            for model_name in ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro"]:
                try:
                    logger.info(f"Testing SDK with model: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    sdk_response = model.generate_content(
                        "Write a single sentence about Google Ads."
                    )
                    logger.info(f"SDK Response with {model_name}: {sdk_response.text}")
                    logger.info(f"SDK test with {model_name} succeeded")
                    break
                except Exception as model_error:
                    logger.warning(f"SDK failed with model {model_name}: {str(model_error)}")
        except Exception as sdk_error:
            logger.warning(f"SDK approach failed: {str(sdk_error)}")
            logger.info(
                "This is expected if you're having SDK issues. Direct API approach will be used instead."
            )

        return True
    except Exception as e:
        logger.error(f"Gemini test failed: {str(e)}")
        logger.exception("Exception details:")
        return False


def test_ads_api():
    """Test fetching keywords without strict filtering"""
    try:
        # Load configuration
        config = load_config()

        # Initialize ads API
        ads_api = GoogleAdsAPI(config["google_ads"])

        # Fetch some keywords
        keywords = ads_api.get_keyword_performance(days_ago=90)

        # Log results
        keyword_count = len(keywords)
        logger.info(f"Successfully fetched {keyword_count} keywords")

        # Show some details if there are keywords
        if keyword_count > 0:
            # Count by status
            status_counts = {}
            campaign_status_counts = {}
            ad_group_status_counts = {}

            for kw in keywords:
                # Count keywords by status
                status = kw.get("status", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1

                # Count keywords by campaign status
                c_status = kw.get("campaign_status", "UNKNOWN")
                campaign_status_counts[c_status] = campaign_status_counts.get(c_status, 0) + 1

                # Count keywords by ad group status
                ag_status = kw.get("ad_group_status", "UNKNOWN")
                ad_group_status_counts[ag_status] = ad_group_status_counts.get(ag_status, 0) + 1

            # Log the counts
            logger.info(f"Keywords by status: {status_counts}")
            logger.info(f"Keywords by campaign status: {campaign_status_counts}")
            logger.info(f"Keywords by ad group status: {ad_group_status_counts}")

            # Show details of first keyword
            first_kw = keywords[0]
            logger.info(f"First keyword: '{first_kw['keyword_text']}' [{first_kw['match_type']}]")
            logger.info(
                f"  Campaign: {first_kw['campaign_name']} (Status: {first_kw['campaign_status']})"
            )
            logger.info(
                f"  Ad Group: {first_kw['ad_group_name']} (Status: {first_kw['ad_group_status']})"
            )
            logger.info(f"  Keyword Status: {first_kw['status']}")
            logger.info(
                f"  Metrics: {first_kw['clicks']} clicks, {first_kw['impressions']} impressions"
            )

        return keyword_count > 0

    except Exception as e:
        logger.error(f"Ads API test failed: {str(e)}")
        logger.exception("Exception details:")
        return False


if __name__ == "__main__":
    logger.info("===== TESTING GEMINI API =====")
    gemini_success = test_gemini()
    logger.info(f"Gemini API test: {'SUCCESS' if gemini_success else 'FAILED'}")

    logger.info("\n===== TESTING ADS API KEYWORD QUERY =====")
    keywords_success = test_ads_api()
    logger.info(f"Ads API keyword query: {'SUCCESS' if keywords_success else 'FAILED'}")

    # Overall result
    logger.info("\n===== TEST SUMMARY =====")
    logger.info(f"Gemini API: {'✓' if gemini_success else '✗'}")
    logger.info(f"Ads API keyword query: {'✓' if keywords_success else '✗'}")

    if gemini_success and keywords_success:
        logger.info("ALL TESTS PASSED! The system should now work properly.")
    else:
        logger.info("SOME TESTS FAILED. Please check the logs for details.")
