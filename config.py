import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config():
    """
    Load configuration from environment variables.
    Prioritizes .env file if it exists.
    """
    env_path = ".env"
    if os.path.exists(env_path):
        logging.info("Loading configuration from .env file...")
        load_dotenv(dotenv_path=env_path)
    else:
        logging.info("No .env file found, relying on system environment variables.")

    config = {
        "google_ads": {
            "client_id": os.getenv("GOOGLE_ADS_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_ADS_CLIENT_SECRET"),
            "developer_token": os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
            "refresh_token": os.getenv("GOOGLE_ADS_REFRESH_TOKEN"),
            "login_customer_id": os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"),
            "customer_id": os.getenv("GOOGLE_ADS_CUSTOMER_ID"),
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "google_ai": {
            "api_key": os.getenv("GOOGLE_API_KEY")
        },
        "scheduler": {
            "log_file": os.getenv("SCHEDULER_LOG_FILE", "logs/scheduler.log")
        }
    }

    # Basic validation (can be expanded)
    if not config["google_ads"]["developer_token"]:
        logging.warning("Google Ads developer token is missing.")
    if not config["google_ai"]["api_key"]:
        logging.warning("Google AI API key is missing.")

    return config 