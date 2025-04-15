from logger import AdsAgentLogger
import os
import time

def test_logger_with_unicode():
    """Test the logger with various Unicode characters including emojis"""
    logger = AdsAgentLogger()
    
    # Test with regular string
    logger.info("Regular log message")
    
    # Test with emoji
    logger.info("Message with emoji: 😀 🚀 💯 🔍")
    
    # Test with other special characters
    logger.info("Special characters: ñ é ü ç ß ÿ € £ ¥ © ®")
    
    # Test with a mix of characters
    logger.info("Mixed content: Data analysis 📊 shows 45% growth 📈 in Q2")
    
    # Test with non-string object containing emoji
    logger.info({"status": "success", "message": "Operation completed 👍", "code": 200})
    
    # Give a moment for file operations to complete
    time.sleep(1)
    
    # Get the latest log file
    log_file = logger.get_latest_log_file()
    
    if log_file:
        print(f"Checking log file: {log_file}")
        
        # Read the log file directly with UTF-8 encoding
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("\nLog file content (should contain correct Unicode characters):")
        print(content)
        
        # Verify the emoji is correctly encoded
        if "😀" in content:
            print("\nSUCCESS: Emoji correctly stored in the log file")
        else:
            print("\nFAILURE: Emoji not correctly stored in the log file")
    else:
        print("Log file not found")

if __name__ == "__main__":
    test_logger_with_unicode() 