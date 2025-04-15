import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_command_patterns():
    """Test the command pattern matching"""
    print("Starting command pattern test...")
    logger.info("Starting command pattern test...")
    
    # Define command patterns (copy from chat_interface.py)
    command_patterns = {
        'fetch_keywords': r'(fetch|get|retrieve|pull|download).*(keyword|keywords)',
        'fetch_data': r'(fetch|get|retrieve|pull|download).*(data|campaigns|performance)',
        'analyze_keywords': r'(analyze|evaluate|assess|optimization|optimize|suggestions|give me|show|recommend).*(keyword|keywords)',
        'analyze_campaigns': r'(analyze|evaluate|assess|optimization|optimize|suggestions).*(campaign|campaigns)',
        'comprehensive_analysis': r'(analyze|evaluate|assess|optimization|optimize|suggestions).*(account|full|complete|comprehensive)',
        'help': r'help|assist|guide|instructions|commands',
        'custom_query': r'query|search|find|filter',
        'schedule': r'schedule|automate|recurring|daily|weekly'
    }
    
    # Test messages for each command
    test_cases = [
        ('fetch my campaign data', 'fetch_data'),
        ('get performance data', 'fetch_data'),
        ('fetch my keyword data', 'fetch_keywords'),
        ('get keyword performance', 'fetch_keywords'),
        ('analyze my campaigns', 'analyze_campaigns'),
        ('optimize my campaigns', 'analyze_campaigns'),
        ('analyze my keywords', 'analyze_keywords'),
        ('give me keyword suggestions', 'analyze_keywords'),
        ('analyze my entire account', 'comprehensive_analysis'),
        ('help me use this agent', 'help'),
        ('search for high ctr campaigns', 'custom_query'),
        ('schedule weekly reports', 'schedule')
    ]
    
    # Test each case
    passes = 0
    failures = 0
    
    for message, expected_command in test_cases:
        detected_command = None
        
        for command, pattern in command_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                detected_command = command
                break
        
        if detected_command == expected_command:
            logger.info(f"PASS: '{message}' -> '{detected_command}'")
            passes += 1
        else:
            logger.error(f"FAIL: '{message}' -> '{detected_command}' (expected '{expected_command}')")
            failures += 1
    
    logger.info(f"Results: {passes} passed, {failures} failed")
    
if __name__ == "__main__":
    test_command_patterns() 