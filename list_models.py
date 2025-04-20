import google.generativeai as genai
from config import load_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    try:
        # Load API key from config
        config = load_config()
        api_key = config["google_ai"]["api_key"]

        # Configure Gemini
        genai.configure(api_key=api_key)

        # List available models
        logger.info("Listing available models...")
        models = list(genai.list_models())

        if models:
            logger.info(f"Found {len(models)} models:")
            for i, model in enumerate(models, 1):
                logger.info(f"{i}. Name: {model.name}")
                logger.info(f"   Display name: {model.display_name}")
                logger.info(f"   Description: {model.description}")
                logger.info(
                    f"   Generation methods: {', '.join(model.supported_generation_methods)}"
                )
                logger.info(f"   Input token limit: {model.input_token_limit}")
                logger.info(f"   Output token limit: {model.output_token_limit}")
                logger.info("")
        else:
            logger.warning("No models found for your API key.")

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        logger.exception("Exception details:")


if __name__ == "__main__":
    main()
