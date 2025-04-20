"""
Base Service for Google Ads Management System

This module provides the BaseService class that all other services inherit from.
It handles common functionality like logging, configuration, and API access.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path


class BaseService:
    """Base class for all services in the Google Ads management system"""

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the base service.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        self.ads_api = ads_api
        self.optimizer = optimizer
        self.config = config or {}

        # Setup logging
        if logger:
            self.logger = logger
        else:
            self.logger = self._setup_logger()

        # Create output directories if they don't exist
        self._ensure_directories()

        # Track metrics for this service
        self.metrics = {
            "invocations": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_run": None,
            "avg_execution_time_ms": 0,
        }

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for this service"""
        logger_name = self.__class__.__name__
        logger = logging.getLogger(logger_name)

        # Only set handlers if they don't exist
        if not logger.handlers:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)

            # Create a file handler
            file_handler = logging.FileHandler(
                f"logs/{logger_name.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
            )

            # Define format
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            # Add handlers
            logger.addHandler(file_handler)

            # Set level
            logger.setLevel(logging.INFO)

        return logger

    def _ensure_directories(self):
        """Ensure that necessary directories exist"""
        directories = ["data", "logs", "reports", "history"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _track_execution(self, start_time: datetime, success: bool):
        """
        Track execution metrics for this service

        Args:
            start_time: When the execution started
            success: Whether the execution was successful
        """
        self.metrics["invocations"] += 1

        if success:
            self.metrics["success_count"] += 1
        else:
            self.metrics["failure_count"] += 1

        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Update average execution time
        prev_avg = self.metrics["avg_execution_time_ms"]
        prev_count = self.metrics["invocations"] - 1

        if prev_count > 0:
            self.metrics["avg_execution_time_ms"] = (
                prev_avg * prev_count + execution_time_ms
            ) / self.metrics["invocations"]
        else:
            self.metrics["avg_execution_time_ms"] = execution_time_ms

        self.metrics["last_run"] = datetime.now().isoformat()

        # Log metrics
        self.logger.info(
            f"Execution tracked: success={success}, "
            f"time={execution_time_ms:.2f}ms, "
            f"avg={self.metrics['avg_execution_time_ms']:.2f}ms"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics for this service"""
        return self.metrics.copy()

    def save_data(self, data: Any, filename: str, directory: str = "data"):
        """
        Save data to a JSON file

        Args:
            data: Data to save
            filename: Name of the file
            directory: Directory to save in
        """
        try:
            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)

            # Full path
            path = os.path.join(directory, filename)

            # Save data
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.info(f"Data saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return False

    def load_data(self, filename: str, directory: str = "data") -> Any:
        """
        Load data from a JSON file

        Args:
            filename: Name of the file
            directory: Directory to load from

        Returns:
            The loaded data or None if file doesn't exist or error
        """
        try:
            # Full path
            path = os.path.join(directory, filename)

            # Check if file exists
            if not os.path.exists(path):
                self.logger.warning(f"File not found: {path}")
                return None

            # Load data
            with open(path, "r") as f:
                data = json.load(f)

            self.logger.info(f"Data loaded from {path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return None
