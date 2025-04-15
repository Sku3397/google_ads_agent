import logging
import os
from datetime import datetime
import sys
import traceback
import glob
import codecs

class AdsAgentLogger:
    """
    Enhanced logger for Google Ads Optimization Agent that provides comprehensive
    error tracking and reporting to both console and file.
    """
    def __init__(self, log_dir="logs"):
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f"ads_agent_{timestamp}.log")
        
        # Configure logger
        self.logger = logging.getLogger("AdsAgent")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers (in case of reinitialization)
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        
        # File handler for all logs - with explicit UTF-8 encoding
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Stream handler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_formatter)
        self.logger.addHandler(stream_handler)
        
        # Store recent logs for GUI display
        self.recent_logs = []
        self.max_recent_logs = 100
        
        # Initialize log history by type
        self.info_logs = []
        self.warning_logs = []
        self.error_logs = []
        self.debug_logs = []
        
    def info(self, message):
        """Log info level message"""
        # Ensure message is properly encoded as string
        message = self._ensure_string(message)
        self.logger.info(message)
        self._add_recent_log("INFO", message)
        self.info_logs.append((datetime.now(), message))
        
    def warning(self, message):
        """Log warning level message"""
        message = self._ensure_string(message)
        self.logger.warning(message)
        self._add_recent_log("WARNING", message)
        self.warning_logs.append((datetime.now(), message))
        
    def error(self, message, include_traceback=True):
        """Log error level message with optional traceback"""
        message = self._ensure_string(message)
        if include_traceback:
            tb = traceback.format_exc()
            if tb and tb != "NoneType: None\n":
                message = f"{message}\nTraceback: {tb}"
        
        self.logger.error(message)
        self._add_recent_log("ERROR", message)
        self.error_logs.append((datetime.now(), message))
        
    def debug(self, message):
        """Log debug level message"""
        message = self._ensure_string(message)
        self.logger.debug(message)
        self._add_recent_log("DEBUG", message)
        self.debug_logs.append((datetime.now(), message))
        
    def exception(self, message):
        """Log exception with traceback"""
        message = self._ensure_string(message)
        self.logger.exception(message)
        tb = traceback.format_exc()
        full_message = f"{message}\nTraceback: {tb}"
        self._add_recent_log("EXCEPTION", full_message)
        self.error_logs.append((datetime.now(), full_message))
        
    def _ensure_string(self, message):
        """Ensure message is a string and handle any problematic Unicode characters"""
        if not isinstance(message, str):
            try:
                message = str(message)
            except UnicodeEncodeError:
                # Handle Unicode encoding error
                message = str(message).encode('utf-8', 'replace').decode('utf-8')
        
        # Ensure the message is properly encoded for log files
        try:
            # Try to encode and decode to catch any character issues
            message.encode('utf-8').decode('utf-8')
        except UnicodeError:
            # Replace problematic characters if encoding fails
            message = message.encode('utf-8', 'replace').decode('utf-8')
            
        return message
        
    def _add_recent_log(self, level, message):
        """Add log to recent logs queue with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.recent_logs.append((timestamp, level, message))
        
        # Maintain max size of recent logs
        if len(self.recent_logs) > self.max_recent_logs:
            self.recent_logs.pop(0)
            
    def get_recent_logs(self, level=None, limit=None):
        """
        Get recent logs, optionally filtered by level
        
        Args:
            level (str, optional): Log level to filter by
            limit (int, optional): Maximum number of logs to return
            
        Returns:
            list: List of recent log tuples (timestamp, level, message)
        """
        if level:
            logs = [log for log in self.recent_logs if log[1] == level.upper()]
        else:
            logs = self.recent_logs.copy()
            
        if limit and len(logs) > limit:
            logs = logs[-limit:]
            
        return logs
        
    def clear_recent_logs(self):
        """Clear recent logs buffer"""
        self.recent_logs = []
        
    def get_error_logs(self):
        """Get all error and exception logs"""
        return self.error_logs.copy()
        
    def get_latest_log_file(self):
        """Get the path to the most recent log file"""
        if not os.path.exists("logs"):
            return None
            
        log_files = glob.glob("logs/ads_agent_*.log")
        if not log_files:
            return None
            
        # Sort by modification time (most recent last)
        log_files.sort(key=lambda x: os.path.getmtime(x))
        return log_files[-1] 