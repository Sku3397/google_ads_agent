import os
import sys
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import modules
try:
    from logger import AdsAgentLogger
    from ads_api import GoogleAdsAPI
    from optimizer import AdsOptimizer
    from chat_interface import ChatInterface
    from config import load_config
    
    logger.info("Successfully imported all modules")
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def test_logger():
    """Test the logger module with UTF-8 encoding"""
    try:
        test_logger = AdsAgentLogger()
        test_logger.info("Test regular message")
        test_logger.info("Test message with emoji: 😀 🚀 🔍")
        test_logger.info("Test message with special characters: ñ é ü ç")
        
        log_file = test_logger.get_latest_log_file()
        if log_file:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "😀" in content:
                    logger.info("Logger UTF-8 encoding test PASSED ✅")
                else:
                    logger.error("Logger UTF-8 encoding test FAILED ❌")
        else:
            logger.warning("Could not find log file to verify UTF-8 encoding")
    except Exception as e:
        logger.error(f"Logger test failed: {str(e)}")

def test_ads_api_date_ranges():
    """Test the ads_api date range handling"""
    try:
        # Create a mock config
        mock_config = {
            'client_id': 'test',
            'client_secret': 'test',
            'developer_token': 'test',
            'refresh_token': 'test',
            'login_customer_id': 'test',
            'customer_id': 'test'
        }
        
        # Create API instance
        api = GoogleAdsAPI(mock_config)
        
        # Test different date ranges
        test_days = [1, 7, 30, 60, 90, 180, 365]
        results = []
        
        for days in test_days:
            try:
                # Test the date range clause method
                clause = api._get_date_range_clause(days)
                
                # Check if it uses the correct format
                is_correct = "BETWEEN" in clause and "segments.date" in clause
                
                results.append({
                    'days': days,
                    'clause': clause,
                    'correct_format': is_correct
                })
                
                logger.info(f"Date range test for {days} days: {'PASSED ✅' if is_correct else 'FAILED ❌'}")
            except Exception as e:
                logger.error(f"Date range test for {days} days failed: {str(e)}")
        
        # Print summary
        success_count = sum(1 for r in results if r['correct_format'])
        logger.info(f"Date range tests: {success_count}/{len(test_days)} passed")
        
    except Exception as e:
        logger.error(f"Ads API date range test failed: {str(e)}")

def test_optimizer():
    """Test the optimizer module with basic campaigns"""
    try:
        # Create mock config suitable for google_ai
        mock_config = {
            # Only api_key is needed by the current __init__
            'api_key': 'mock_google_api_key'
        }

        # Create optimizer instance
        optimizer = AdsOptimizer(mock_config)

        # Create mock campaign data (minimal fields needed)
        mock_campaigns = [
            {
                'id': '123456789',
                'name': 'Test Campaign 1',
                'status': 'ENABLED',
                'days': 30
            }
        ]

        # Create mock keyword data (reduced fields)
        mock_keywords = [
            {
                'campaign_id': '123456789',
                'campaign_name': 'Test Campaign 1',
                'ad_group_id': '98765432',
                'ad_group_name': 'Test Ad Group',
                'keyword_text': 'test keyword',
                'match_type': 'EXACT',
                'status': 'ENABLED',
                'system_serving_status': 'ELIGIBLE',
                'quality_score': 7,
                'current_bid': 0.5,
                'clicks': 100,
                'impressions': 1000,
                'average_cpc': 0.5,
                'conversions': 5,
                'cost': 50.0,
                'top_impression_pct': 0.6,
                'search_impression_share': 0.7,
                'search_top_impression_share': 0.5,
                'days': 30
            }
        ]

        # Test data formatting
        campaign_data = optimizer.format_campaign_data(mock_campaigns)
        keyword_data = optimizer.format_keyword_data(mock_keywords)

        logger.info(f"Campaign data formatting: {'PASSED ✅' if campaign_data else 'FAILED ❌'}")
        logger.info(f"Keyword data formatting: {'PASSED ✅' if keyword_data else 'FAILED ❌'}")

        # Test suggestion parsing (using a sample Gemini-like output format)
        # NOTE: This is a placeholder. A real test would mock the Gemini response
        # and verify the parser handles it correctly.
        sample_suggestions_text = """
        1. [BID_ADJUSTMENT] Increase bid for test keyword
        - Entity: [KEYWORD] ID: "test keyword" [EXACT]
        - Ad Group: Test Ad Group
        - Current Value: Bid $0.50, CPA $10.00, Conv Rate 5.00%, QS 7/10
        - Change: Increase bid from $0.50 to $0.60
        - Rationale: Good CPA and Conv Rate indicate potential for more conversions with higher bid.
        """

        parsed_suggestions = optimizer.parse_suggestions(sample_suggestions_text, mock_campaigns, mock_keywords)
        logger.info(f"Suggestion parsing: {'PASSED ✅' if parsed_suggestions else 'FAILED ❌'}")
        if parsed_suggestions:
            logger.info(f"Parsed suggestion: {parsed_suggestions[0]}")

        # Note: Calling get_optimization_suggestions would require mocking the actual Gemini API call
        # logger.info("Testing suggestion generation (requires mocking Gemini API)...")
        # suggestions = optimizer.get_optimization_suggestions(mock_campaigns, mock_keywords)
        # logger.info(f"Suggestion generation: {'PASSED ✅' if suggestions else 'FAILED ❌'}")

    except Exception as e:
        logger.exception(f"Optimizer test failed: {e}")
        raise

def test_chat_interface():
    """Test the chat interface functionality"""
    try:
        # Create mock objects
        mock_ads_api = type('MockAdsAPI', (), {
            'get_campaign_performance': lambda self, days_ago: [],
            'get_keyword_performance': lambda self, days_ago, campaign_id=None: []
        })()
        
        mock_optimizer = type('MockOptimizer', (), {
            'get_optimization_suggestions': lambda self, campaigns, keywords=None: []
        })()
        
        mock_config = {
            'openai': {'api_key': 'test'}
        }
        
        test_logger = AdsAgentLogger()
        
        # Create chat interface instance
        chat_interface = ChatInterface(mock_ads_api, mock_optimizer, mock_config, test_logger)
        
        # Test command detection
        commands = [
            ('analyze my campaigns', 'analyze_campaigns'),
            ('fetch my keyword data', 'fetch_keywords'),
            ('optimize my account', 'comprehensive_analysis'),
            ('help me understand how to use this', 'help'),
            ('find campaigns with high ctr', 'custom_query'),
            ('schedule weekly reports', 'schedule')
        ]
        
        for message, expected_command in commands:
            detected = chat_interface.detect_command(message)
            result = detected == expected_command
            logger.info(f"Command detection '{message}' -> '{detected}': {'PASSED ✅' if result else 'FAILED ❌'}")
        
        # Test parameter parsing
        param_tests = [
            ('fetch data for the last 60 days', 'fetch_data', {'days': 60}),
            ('schedule daily analysis at 3:30pm', 'schedule', {'hour': 15, 'minute': 30, 'frequency': 'daily'})
        ]
        
        for message, command, expected_params in param_tests:
            params = chat_interface.parse_parameters(message, command)
            # Check if all expected params are present with correct values
            all_correct = all(params.get(k) == v for k, v in expected_params.items())
            logger.info(f"Parameter parsing '{message}': {'PASSED ✅' if all_correct else 'FAILED ❌'}")
            
    except Exception as e:
        logger.error(f"Chat interface test failed: {str(e)}")

def test_unicode_handling():
    """Comprehensive test for Unicode handling throughout the application"""
    try:
        test_logger = AdsAgentLogger()
        
        # Test various Unicode characters
        test_strings = [
            "Regular ASCII text",
            "Emoji test: 😀 🚀 💯 🎉 👍",
            "European characters: áéíóúçñüäöß",
            "Asian characters: 你好 こんにちは 안녕하세요",
            "Symbols: €£¥©®™§±",
            "Mixed content: Product increased by 15% 📈 in region 🌎"
        ]
        
        all_passed = True
        
        for test_str in test_strings:
            # Log the string
            test_logger.info(test_str)
            
            # Get latest log and check content
            log_file = test_logger.get_latest_log_file()
            if log_file:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if test_str not in content:
                        all_passed = False
                        logger.error(f"Unicode test failed for: {test_str}")
            
        logger.info(f"Unicode handling tests: {'PASSED ✅' if all_passed else 'FAILED ❌'}")
        
    except Exception as e:
        logger.error(f"Unicode handling test failed: {str(e)}")

def run_all_tests():
    """Run all tests"""
    logger.info("Starting tests...")
    
    test_logger()
    test_ads_api_date_ranges()
    test_optimizer()
    test_chat_interface()
    test_unicode_handling()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    run_all_tests() 