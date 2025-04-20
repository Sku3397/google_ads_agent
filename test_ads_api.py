import sys
import importlib
import traceback
import logging
import inspect

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("TestAdsAPI")

# Make sure we're using a fresh import
if "ads_api" in sys.modules:
    logger.info("Removing ads_api from sys.modules to force fresh import")
    del sys.modules["ads_api"]

try:
    logger.info("Importing ads_api")
    import ads_api

    logger.info(f"ads_api loaded from: {ads_api.__file__}")

    # Enable more detailed tracing
    def trace_calls(frame, event, arg):
        if event != "call":
            return
        co = frame.f_code
        func_name = co.co_name
        func_line_no = frame.f_lineno
        func_filename = co.co_filename
        caller = frame.f_back
        if caller:
            caller_line_no = caller.f_lineno
            caller_filename = caller.f_code.co_filename
        else:
            caller_filename = "Module initialization"
            caller_line_no = 0

        if "ads_api.py" in func_filename:
            logger.info(
                f"Call to {func_name} on line {func_line_no} of {func_filename} from line {caller_line_no} of {caller_filename}"
            )
        return

    sys.settrace(trace_calls)

    # Create a mock GoogleAdsAPI instance
    # Note: This won't connect to the real API, but will help us trace the code execution
    class MockConfig:
        def __init__(self):
            self.client_id = "mock_client_id"
            self.client_secret = "mock_client_secret"
            self.developer_token = "mock_developer_token"
            self.refresh_token = "mock_refresh_token"
            self.login_customer_id = "mock_customer_id"
            self.use_proto_plus = True

    try:
        mock_config = MockConfig()
        logger.info("Creating mock GoogleAdsAPI instance")

        # Pre-load the get_keyword_performance method to see its source
        if hasattr(ads_api.GoogleAdsAPI, "get_keyword_performance"):
            method = ads_api.GoogleAdsAPI.get_keyword_performance
            logger.info(f"Loaded get_keyword_performance method: {method}")

            # Get the source code
            try:
                source = inspect.getsource(method)
                logger.info("Method source code:")
                for i, line in enumerate(source.split("\n")):
                    logger.info(f"Line {i+1}: {line}")
                # Check for the deprecated field
                if "metrics.average_position" in source:
                    logger.warning("Found deprecated metrics.average_position in the source code!")
                    for i, line in enumerate(source.split("\n")):
                        if "metrics.average_position" in line:
                            logger.error(f"Line {i+1}: {line.strip()}")
            except Exception as e:
                logger.error(f"Error getting source code: {e}")
                traceback.print_exc()
        else:
            logger.error("get_keyword_performance method not found in GoogleAdsAPI class")

        # We won't actually run the API because it requires configuration
        logger.info("Test completed successfully")

    except Exception as e:
        logger.error(f"Error during test: {e}")
        traceback.print_exc()

except ImportError as e:
    logger.error(f"Error importing ads_api: {e}")
    traceback.print_exc()
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    traceback.print_exc()

# Disable tracing
sys.settrace(None)
logger.info("Test script completed.")
