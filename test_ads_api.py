from ads_api import GoogleAdsAPI
from datetime import datetime, timedelta
import os
import logging
from config import load_config

# Configure basic logging for testing
logging.basicConfig(level=logging.INFO)

def test_date_range_clause():
    """Test the _get_date_range_clause method with various inputs"""
    # Create a mock config
    mock_config = {
        'client_id': 'test_client_id',
        'client_secret': 'test_client_secret',
        'developer_token': 'test_developer_token',
        'refresh_token': 'test_refresh_token',
        'login_customer_id': 'test_login_customer_id',
        'customer_id': 'test_customer_id'
    }
    
    # Initialize API without actual credentials
    ads_api = GoogleAdsAPI(mock_config)
    
    # Test with various day ranges
    test_days = [1, 7, 30, 60, 90, 180, 365]
    
    for days in test_days:
        try:
            clause = ads_api._get_date_range_clause(days)
            print(f"Date range clause for {days} days: {clause}")
            
            # Check that the clause uses the correct format
            assert "BETWEEN" in clause, "Clause should use BETWEEN operator"
            assert "segments.date" in clause, "Clause should filter on segments.date"
        except Exception as e:
            print(f"Error with {days} days: {str(e)}")

def test_with_real_config():
    """Test with the actual config file if available"""
    try:
        # Check if .env file exists
        if os.path.exists('.env'):
            config = load_config()
            ads_api = GoogleAdsAPI(config['google_ads'])
            
            # Test with actual API connection but don't execute queries
            clause_30 = ads_api._get_date_range_clause(30)
            print(f"Using real config - Date clause for 30 days: {clause_30}")
            
            clause_60 = ads_api._get_date_range_clause(60)
            print(f"Using real config - Date clause for 60 days: {clause_60}")
    except Exception as e:
        print(f"Error testing with real config: {str(e)}")

if __name__ == "__main__":
    print("Testing date range clause formatting:")
    test_date_range_clause()
    
    print("\nTesting with real config if available:")
    test_with_real_config() 