from chat_interface import ChatInterface


class MockObject:
    """Mock object for testing."""

    def __init__(self):
        pass


def test_command_detection():
    """Test the command detection directly using the ChatInterface class."""
    print("Testing command detection...")

    # Create mock objects
    mock_ads_api = MockObject()
    mock_optimizer = MockObject()
    mock_logger = MockObject()

    # Set needed methods/properties
    mock_logger.info = lambda x: None
    mock_logger.error = lambda x: None

    # Mock config
    mock_config = {"openai": {"api_key": "test"}}

    # Create ChatInterface instance
    chat = ChatInterface(mock_ads_api, mock_optimizer, mock_config, mock_logger)

    # Test messages
    test_messages = [
        "fetch my campaign data",
        "get campaign performance",
        "fetch my keyword data",
        "get keyword performance",
        "analyze my campaigns",
        "optimize my campaigns",
        "analyze my keywords",
        "give me keyword suggestions",
        "analyze my entire account",
        "help me use this agent",
        "find campaigns with high CTR",
        "schedule weekly reports",
    ]

    # Test each message
    for msg in test_messages:
        command = chat.detect_command(msg)
        print(f"Message: '{msg}' -> Command: '{command}'")


if __name__ == "__main__":
    test_command_detection()
