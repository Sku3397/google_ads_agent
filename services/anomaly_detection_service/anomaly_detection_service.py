"""
Anomaly Detection Service for Google Ads Management System

This module provides anomaly detection capabilities for identifying unusual patterns
in Google Ads campaign performance data that may require attention or intervention.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

from services.base_service import BaseService

class AnomalyDetectionService(BaseService):
    """
    Service for detecting anomalies in Google Ads campaign performance data.
    
    This service uses statistical and machine learning techniques to identify
    unusual patterns in campaign metrics that may indicate issues or opportunities.
    It supports:
    
    - Statistical anomaly detection (Z-score, IQR)
    - Machine learning anomaly detection (Isolation Forest)
    - Time series anomaly detection
    - Threshold-based anomaly detection
    - Alert generation for detected anomalies
    """
    
    def __init__(self, 
                 ads_api=None, 
                 optimizer=None, 
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the AnomalyDetectionService.
        
        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)
        
        # Set default configuration parameters
        self.default_config = {
            "detection_methods": ["statistical", "machine_learning"],
            "sensitivity": 2.0,  # Z-score threshold (lower = more sensitive)
            "min_data_points": 7,  # Minimum number of data points required
            "metrics_to_monitor": [
                "impressions", "clicks", "cost", "conversions", 
                "ctr", "conversion_rate", "cost_per_conversion"
            ],
            "alert_threshold": "high",  # low, medium, high
            "detection_window": 30,  # days
            "training_window": 90,  # days
            "seasonality_adjust": True
        }
        
        # Override defaults with provided configuration
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize ML models
        self.models = {}
        
        # Store detected anomalies
        self.anomalies = []
        
        self.logger.info("AnomalyDetectionService initialized")
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        entity_type: str = 'campaign',
                        methods: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the provided performance data.
        
        Args:
            data: DataFrame with performance data containing metrics over time
            entity_type: Type of entity to analyze ('campaign', 'ad_group', 'keyword')
            methods: List of detection methods to use
                     (default uses configured methods from initialization)
            
        Returns:
            List of detected anomalies with details
        """
        start_time = datetime.now()
        self.logger.info(f"Starting anomaly detection for {entity_type} data")
        
        # Use configured methods if not specified
        methods = methods or self.config["detection_methods"]
        
        # Check if we have enough data
        if len(data) < self.config["min_data_points"]:
            self.logger.warning(f"Insufficient data for anomaly detection. Got {len(data)} points, need {self.config['min_data_points']}")
            return []
        
        # Ensure data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
            else:
                self.logger.error("Data must have a date column or datetime index")
                return []
        
        # Sort data by date
        data = data.sort_index()
        
        # Apply seasonality adjustment if configured and possible
        if self.config["seasonality_adjust"] and len(data) >= 14:  # Need at least 2 weeks for weekday patterns
            data = self._adjust_for_seasonality(data)
        
        # Run detection using each method
        all_anomalies = []
        
        if "statistical" in methods:
            self.logger.info("Running statistical anomaly detection")
            stat_anomalies = self.detect_statistical_anomalies(data, entity_type)
            all_anomalies.extend(stat_anomalies)
            
        if "machine_learning" in methods:
            self.logger.info("Running machine learning anomaly detection")
            ml_anomalies = self.detect_ml_anomalies(data, entity_type)
            all_anomalies.extend(ml_anomalies)
            
        if "threshold" in methods:
            self.logger.info("Running threshold-based anomaly detection")
            threshold_anomalies = self.detect_threshold_anomalies(data, entity_type)
            all_anomalies.extend(threshold_anomalies)
        
        # Deduplicate anomalies (same entity, date, and metric)
        unique_anomalies = self._deduplicate_anomalies(all_anomalies)
        
        # Sort by severity (descending) and date (ascending)
        sorted_anomalies = sorted(
            unique_anomalies,
            key=lambda x: (-x.get('severity_score', 0), x.get('date'))
        )
        
        # Store anomalies for later reference
        self.anomalies = sorted_anomalies
        
        # Save anomalies to file
        self._save_anomalies(sorted_anomalies)
        
        self.logger.info(f"Detected {len(sorted_anomalies)} anomalies in {entity_type} data")
        self._track_execution(start_time, success=True)
        
        return sorted_anomalies
    
    def detect_statistical_anomalies(self, data: pd.DataFrame, 
                                    entity_type: str = 'campaign') -> List[Dict[str, Any]]:
        """
        Detect anomalies using statistical methods (Z-score and IQR).
        
        Args:
            data: DataFrame with performance data
            entity_type: Type of entity being analyzed
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        metrics = self.config["metrics_to_monitor"]
        sensitivity = self.config["sensitivity"]
        
        # Loop through each metric
        for metric in metrics:
            if metric not in data.columns:
                continue
                
            # Get metric values
            values = data[metric].dropna()
            
            if len(values) < self.config["min_data_points"]:
                continue
                
            # Calculate statistical properties
            mean = values.mean()
            std = values.std()
            median = values.median()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            # Z-score method
            z_scores = (values - mean) / std if std > 0 else pd.Series(0, index=values.index)
            
            # IQR method
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # For each data point
            for date, value in values.items():
                z_score = z_scores[date]
                
                # Check if this point is an anomaly by Z-score
                is_zscore_anomaly = abs(z_score) > sensitivity
                
                # Check if this point is an anomaly by IQR
                is_iqr_anomaly = value < lower_bound or value > upper_bound
                
                # If either method detects an anomaly
                if is_zscore_anomaly or is_iqr_anomaly:
                    # Determine direction and severity
                    direction = "increase" if value > mean else "decrease"
                    
                    # Calculate severity score (0-10)
                    if is_zscore_anomaly:
                        severity_score = min(10, abs(z_score) / sensitivity * 5)
                    else:
                        # Calculate how far outside the IQR bounds
                        if value < lower_bound:
                            severity_score = min(10, abs((value - lower_bound) / iqr) * 5)
                        else:
                            severity_score = min(10, abs((value - upper_bound) / iqr) * 5)
                    
                    # Create anomaly record
                    anomaly = {
                        "entity_type": entity_type,
                        "entity_id": data.get("id", "unknown") if "id" in data.columns else data.get("entity_id", "unknown"),
                        "entity_name": data.get("name", "unknown") if "name" in data.columns else data.get("entity_name", "unknown"),
                        "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                        "metric": metric,
                        "value": float(value),
                        "expected_value": float(mean),
                        "direction": direction,
                        "percent_change": float((value - mean) / mean * 100) if mean != 0 else float('inf'),
                        "z_score": float(z_score),
                        "is_zscore_anomaly": bool(is_zscore_anomaly),
                        "is_iqr_anomaly": bool(is_iqr_anomaly),
                        "detection_method": "statistical",
                        "severity_score": float(severity_score),
                        "detection_time": datetime.now().isoformat()
                    }
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_ml_anomalies(self, data: pd.DataFrame, 
                           entity_type: str = 'campaign') -> List[Dict[str, Any]]:
        """
        Detect anomalies using machine learning methods (Isolation Forest).
        
        Args:
            data: DataFrame with performance data
            entity_type: Type of entity being analyzed
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        metrics = self.config["metrics_to_monitor"]
        
        # Filter to only include metrics in the data
        available_metrics = [m for m in metrics if m in data.columns]
        
        if not available_metrics:
            self.logger.warning("No configured metrics found in data for ML anomaly detection")
            return []
        
        try:
            # Prepare data for ML
            features = data[available_metrics].copy()
            
            # Handle NaNs
            features = features.fillna(0)
            
            # Need at least min_data_points rows
            if len(features) < self.config["min_data_points"]:
                self.logger.warning(f"Insufficient data for ML anomaly detection")
                return []
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=0.05,  # Assume 5% of data could be anomalous
                random_state=42,
                n_estimators=100
            )
            
            # Fit and predict
            model.fit(scaled_features)
            predictions = model.predict(scaled_features)
            scores = model.decision_function(scaled_features)
            
            # Interpret results
            for idx, (date, row) in enumerate(data.iterrows()):
                if predictions[idx] == -1:  # -1 indicates anomaly
                    # Determine which metrics contributed most to the anomaly
                    original_metrics = features.iloc[idx].to_dict()
                    scaled_metrics = scaled_features[idx]
                    
                    # Find the metric with the largest deviation
                    largest_deviation_idx = np.argmax(np.abs(scaled_metrics))
                    anomalous_metric = available_metrics[largest_deviation_idx]
                    anomalous_value = original_metrics[anomalous_metric]
                    
                    # Calculate expected value (mean of that metric)
                    expected_value = features[anomalous_metric].mean()
                    
                    # Direction of anomaly
                    direction = "increase" if anomalous_value > expected_value else "decrease"
                    
                    # Severity score (0-10)
                    # Convert anomaly score (-1 to 0) to severity (0-10)
                    # The more negative the score, the more anomalous
                    anomaly_score = scores[idx]
                    severity_score = min(10, ((-anomaly_score + 0.1) * 10))
                    
                    # Create anomaly record
                    anomaly = {
                        "entity_type": entity_type,
                        "entity_id": row.get("id", "unknown") if "id" in row.index else row.get("entity_id", "unknown"),
                        "entity_name": row.get("name", "unknown") if "name" in row.index else row.get("entity_name", "unknown"),
                        "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                        "metric": anomalous_metric,
                        "value": float(anomalous_value),
                        "expected_value": float(expected_value),
                        "direction": direction,
                        "percent_change": float((anomalous_value - expected_value) / expected_value * 100) if expected_value != 0 else float('inf'),
                        "detection_method": "machine_learning",
                        "algorithm": "isolation_forest",
                        "anomaly_score": float(anomaly_score),
                        "severity_score": float(severity_score),
                        "detection_time": datetime.now().isoformat()
                    }
                    
                    anomalies.append(anomaly)
            
            # Save the model for future use
            self.models[f"{entity_type}_isolation_forest"] = {
                "model": model,
                "scaler": scaler,
                "features": available_metrics,
                "trained_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML anomaly detection: {str(e)}")
        
        return anomalies 
    
    def detect_threshold_anomalies(self, data: pd.DataFrame, 
                                  entity_type: str = 'campaign') -> List[Dict[str, Any]]:
        """
        Detect anomalies based on predefined thresholds.
        
        Args:
            data: DataFrame with performance data
            entity_type: Type of entity being analyzed
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Default thresholds by metric
        default_thresholds = {
            "ctr": {"min": 0.01, "max": None, "percent_change": 30},
            "conversion_rate": {"min": 0.01, "max": None, "percent_change": 30},
            "cost_per_conversion": {"min": None, "max": None, "percent_change": 50},
            "impressions": {"min": None, "max": None, "percent_change": 50},
            "clicks": {"min": None, "max": None, "percent_change": 40},
            "conversions": {"min": None, "max": None, "percent_change": 40},
            "cost": {"min": None, "max": None, "percent_change": 40}
        }
        
        # Get thresholds from config if available
        thresholds = self.config.get("thresholds", default_thresholds)
        
        # Calculate rolling averages to compare against
        rolling_window = min(7, len(data) // 2)  # Use 7 days or half the data
        if rolling_window < 3:
            return []  # Not enough data
            
        # For each metric with thresholds
        for metric, threshold_values in thresholds.items():
            if metric not in data.columns:
                continue
                
            # Get metric values
            values = data[metric].dropna()
            
            if len(values) < self.config["min_data_points"]:
                continue
            
            # Calculate rolling average (excluding the current day)
            rolling_avg = values.shift(1).rolling(window=rolling_window, min_periods=3).mean()
            
            # For each data point
            for date, value in values.items():
                # Skip if we don't have a rolling average yet
                if pd.isna(rolling_avg[date]):
                    continue
                    
                avg_value = rolling_avg[date]
                is_anomaly = False
                reason = []
                
                # Check absolute min threshold
                if threshold_values.get("min") is not None and value < threshold_values["min"]:
                    is_anomaly = True
                    reason.append(f"below minimum threshold of {threshold_values['min']}")
                
                # Check absolute max threshold
                if threshold_values.get("max") is not None and value > threshold_values["max"]:
                    is_anomaly = True
                    reason.append(f"above maximum threshold of {threshold_values['max']}")
                
                # Check percent change threshold
                if threshold_values.get("percent_change") is not None and avg_value > 0:
                    percent_change = abs((value - avg_value) / avg_value * 100)
                    if percent_change > threshold_values["percent_change"]:
                        is_anomaly = True
                        direction = "increase" if value > avg_value else "decrease"
                        reason.append(f"{direction} of {percent_change:.1f}% from average")
                
                # If any threshold was crossed
                if is_anomaly:
                    # Direction of change
                    direction = "increase" if value > avg_value else "decrease"
                    
                    # Severity score: how far from threshold as percentage (max 10)
                    if threshold_values.get("percent_change") is not None and avg_value > 0:
                        percent_change = abs((value - avg_value) / avg_value * 100)
                        severity_pct = percent_change / threshold_values["percent_change"]
                        severity_score = min(10, severity_pct * 5)
                    else:
                        severity_score = 5  # Default mid-level severity
                    
                    # Create anomaly record
                    anomaly = {
                        "entity_type": entity_type,
                        "entity_id": data.get("id", "unknown") if "id" in data.columns else data.get("entity_id", "unknown"),
                        "entity_name": data.get("name", "unknown") if "name" in data.columns else data.get("entity_name", "unknown"),
                        "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                        "metric": metric,
                        "value": float(value),
                        "expected_value": float(avg_value),
                        "direction": direction,
                        "percent_change": float((value - avg_value) / avg_value * 100) if avg_value != 0 else float('inf'),
                        "detection_method": "threshold",
                        "reason": "; ".join(reason),
                        "severity_score": float(severity_score),
                        "detection_time": datetime.now().isoformat()
                    }
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def get_anomalies(self, 
                     entity_type: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     min_severity: float = 0.0,
                     metrics: Optional[List[str]] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve detected anomalies with optional filtering.
        
        Args:
            entity_type: Filter by entity type
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            min_severity: Minimum severity score
            metrics: List of metrics to include
            limit: Maximum number of anomalies to return
            
        Returns:
            List of anomalies matching the filter criteria
        """
        # Start with all anomalies
        filtered = self.anomalies
        
        # Apply filters
        if entity_type:
            filtered = [a for a in filtered if a.get("entity_type") == entity_type]
            
        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            filtered = [a for a in filtered if datetime.strptime(a.get("date", "1970-01-01"), "%Y-%m-%d").date() >= start]
            
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
            filtered = [a for a in filtered if datetime.strptime(a.get("date", "2099-12-31"), "%Y-%m-%d").date() <= end]
            
        if min_severity > 0:
            filtered = [a for a in filtered if a.get("severity_score", 0) >= min_severity]
            
        if metrics:
            filtered = [a for a in filtered if a.get("metric") in metrics]
            
        # Sort by severity (desc) and date (asc)
        sorted_anomalies = sorted(
            filtered,
            key=lambda x: (-x.get('severity_score', 0), x.get('date'))
        )
        
        # Apply limit
        return sorted_anomalies[:limit]
    
    def generate_alerts(self, anomalies: Optional[List[Dict[str, Any]]] = None, 
                        min_severity: float = 7.0) -> List[Dict[str, Any]]:
        """
        Generate alerts for high-severity anomalies.
        
        Args:
            anomalies: List of anomalies (uses stored anomalies if None)
            min_severity: Minimum severity threshold for alerts
            
        Returns:
            List of alert objects
        """
        # Use provided anomalies or stored ones
        anomalies_to_process = anomalies if anomalies is not None else self.anomalies
        
        # Filter by severity
        high_severity = [a for a in anomalies_to_process if a.get("severity_score", 0) >= min_severity]
        
        alerts = []
        for anomaly in high_severity:
            # Format a message
            entity_name = anomaly.get("entity_name", "Unknown")
            metric = anomaly.get("metric", "Unknown")
            direction = anomaly.get("direction", "change")
            pct_change = anomaly.get("percent_change", 0)
            
            message = (
                f"Anomaly detected in {entity_name}: "
                f"{metric} shows {direction} of {abs(pct_change):.1f}% "
                f"on {anomaly.get('date', 'unknown date')}"
            )
            
            # Suggest possible actions
            suggestions = self._generate_action_suggestions(anomaly)
            
            alert = {
                "anomaly_id": anomaly.get("id", str(hash(str(anomaly)))),
                "severity": anomaly.get("severity_score", 0),
                "message": message,
                "entity_type": anomaly.get("entity_type"),
                "entity_id": anomaly.get("entity_id"),
                "entity_name": entity_name,
                "metric": metric,
                "date": anomaly.get("date"),
                "detection_time": anomaly.get("detection_time", datetime.now().isoformat()),
                "alert_time": datetime.now().isoformat(),
                "suggestions": suggestions,
                "status": "new"
            }
            
            alerts.append(alert)
        
        # Save alerts
        self._save_alerts(alerts)
        
        return alerts
    
    def _adjust_for_seasonality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust data for known seasonality patterns.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with seasonality adjusted values
        """
        # This is a simplified implementation
        # In a real system, you might use more sophisticated time series decomposition
        
        adjusted_data = data.copy()
        
        # Add day of week
        if isinstance(adjusted_data.index, pd.DatetimeIndex):
            adjusted_data['day_of_week'] = adjusted_data.index.dayofweek
        
        metrics = self.config["metrics_to_monitor"]
        available_metrics = [m for m in metrics if m in data.columns]
        
        # For each metric, adjust for day of week effect
        for metric in available_metrics:
            # Calculate average by day of week
            if 'day_of_week' in adjusted_data.columns:
                day_avgs = adjusted_data.groupby('day_of_week')[metric].mean()
                overall_avg = adjusted_data[metric].mean()
                
                # Calculate seasonality factors
                factors = day_avgs / overall_avg
                
                # Apply adjustment
                for day, factor in factors.items():
                    if factor > 0:
                        mask = adjusted_data['day_of_week'] == day
                        adjusted_data.loc[mask, f"{metric}_adjusted"] = adjusted_data.loc[mask, metric] / factor
            
        # Drop the helper column
        if 'day_of_week' in adjusted_data.columns:
            adjusted_data = adjusted_data.drop('day_of_week', axis=1)
        
        return adjusted_data
    
    def _deduplicate_anomalies(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate anomalies (same entity, date, and metric).
        Keep the one with the highest severity.
        
        Args:
            anomalies: List of anomaly dictionaries
            
        Returns:
            Deduplicated list of anomalies
        """
        if not anomalies:
            return []
            
        # Group by entity, date, and metric
        grouped = {}
        for anomaly in anomalies:
            key = (
                anomaly.get("entity_id", "unknown"),
                anomaly.get("date", "unknown"),
                anomaly.get("metric", "unknown")
            )
            
            if key not in grouped or anomaly.get("severity_score", 0) > grouped[key].get("severity_score", 0):
                grouped[key] = anomaly
                
        return list(grouped.values())
    
    def _save_anomalies(self, anomalies: List[Dict[str, Any]]) -> None:
        """
        Save anomalies to a file for persistence.
        
        Args:
            anomalies: List of anomaly dictionaries
        """
        try:
            # Create directory if it doesn't exist
            directory = "data/anomalies"
            os.makedirs(directory, exist_ok=True)
            
            # Use current date in filename
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"{directory}/anomalies_{date_str}.json"
            
            with open(filename, 'w') as f:
                json.dump(anomalies, f, indent=2)
                
            self.logger.info(f"Saved {len(anomalies)} anomalies to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving anomalies: {str(e)}")
    
    def _save_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """
        Save alerts to a file for persistence.
        
        Args:
            alerts: List of alert dictionaries
        """
        try:
            # Create directory if it doesn't exist
            directory = "data/alerts"
            os.makedirs(directory, exist_ok=True)
            
            # Use current date in filename
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"{directory}/alerts_{date_str}.json"
            
            with open(filename, 'w') as f:
                json.dump(alerts, f, indent=2)
                
            self.logger.info(f"Saved {len(alerts)} alerts to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving alerts: {str(e)}")
    
    def _generate_action_suggestions(self, anomaly: Dict[str, Any]) -> List[str]:
        """
        Generate suggested actions based on the type of anomaly.
        
        Args:
            anomaly: Anomaly dictionary
            
        Returns:
            List of suggested actions
        """
        entity_type = anomaly.get("entity_type", "")
        metric = anomaly.get("metric", "")
        direction = anomaly.get("direction", "")
        
        suggestions = []
        
        # Campaign-level suggestions
        if entity_type == "campaign":
            if metric == "cost" and direction == "increase":
                suggestions.append("Review campaign budget settings")
                suggestions.append("Check for bidding strategy changes")
            
            if metric == "ctr" and direction == "decrease":
                suggestions.append("Review ad copy for relevance")
                suggestions.append("Check if targeting is too broad")
            
            if metric == "conversion_rate" and direction == "decrease":
                suggestions.append("Review landing page experience")
                suggestions.append("Check if conversion tracking is working properly")
            
            if metric == "cost_per_conversion" and direction == "increase":
                suggestions.append("Review bidding strategy")
                suggestions.append("Identify keywords with high cost but low conversions")
        
        # Ad group level suggestions
        elif entity_type == "ad_group":
            if metric == "impressions" and direction == "decrease":
                suggestions.append("Check if budget is limiting ad serving")
                suggestions.append("Review ad group bids")
            
            if metric == "clicks" and direction == "decrease":
                suggestions.append("Review ad copy for relevance")
                suggestions.append("Check if competitors have changed their strategy")
        
        # Default suggestions
        if not suggestions:
            suggestions = [
                f"Review {entity_type} settings",
                f"Analyze {metric} trend over the past 30 days"
            ]
        
        return suggestions
    
    def _track_execution(self, start_time: datetime, success: bool = True) -> None:
        """
        Track execution time and status for performance monitoring.
        
        Args:
            start_time: Start time of execution
            success: Whether execution was successful
        """
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Log execution metrics
        self.logger.info(f"Execution completed in {execution_time:.2f} seconds (success={success})")
        
        # Could store metrics for later analysis
        # self.execution_history.append({
        #     "timestamp": datetime.now().isoformat(),
        #     "execution_time": execution_time,
        #     "success": success
        # })
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about trained models.
        
        Returns:
            Dictionary with model information
        """
        return {
            "models": [
                {
                    "name": name,
                    "type": "IsolationForest" if "isolation_forest" in name else "Unknown",
                    "entity_type": name.split("_")[0] if "_" in name else "unknown",
                    "features": info.get("features", []),
                    "trained_at": info.get("trained_at", "unknown")
                }
                for name, info in self.models.items()
            ],
            "count": len(self.models)
        } 