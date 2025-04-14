import os
from dotenv import load_dotenv

def load_config():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check if required environment variables are set
    required_vars = [
        'GOOGLE_ADS_CLIENT_ID',
        'GOOGLE_ADS_CLIENT_SECRET',
        'GOOGLE_ADS_DEVELOPER_TOKEN',
        'GOOGLE_ADS_REFRESH_TOKEN',
        'GOOGLE_ADS_LOGIN_CUSTOMER_ID',
        'GOOGLE_ADS_CUSTOMER_ID',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {
        'google_ads': {
            'client_id': os.getenv('GOOGLE_ADS_CLIENT_ID'),
            'client_secret': os.getenv('GOOGLE_ADS_CLIENT_SECRET'),
            'developer_token': os.getenv('GOOGLE_ADS_DEVELOPER_TOKEN'),
            'refresh_token': os.getenv('GOOGLE_ADS_REFRESH_TOKEN'),
            'login_customer_id': os.getenv('GOOGLE_ADS_LOGIN_CUSTOMER_ID'),
            'customer_id': os.getenv('GOOGLE_ADS_CUSTOMER_ID')
        },
        'openai': {
            'api_key': os.getenv('OPENAI_API_KEY')
        }
    } 