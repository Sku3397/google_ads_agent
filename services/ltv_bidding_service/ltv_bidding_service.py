"""
LTV Bidding Service for Google Ads Management System

This module provides the LTVBiddingService class that implements lifetime value
based bidding strategies. It optimizes bids based on predicted customer lifetime
value rather than immediate conversion value.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from services.base_service import BaseService


class LTVBiddingService(BaseService):
    """
    Service for optimizing bids based on predicted lifetime value (LTV) of customers.

    This service:
    1. Builds predictive models for customer LTV
    2. Analyzes historical conversion data to predict future value
    3. Adjusts bids based on predicted LTV
    4. Optimizes budget allocation across campaigns based on LTV
    """

    def __init__(self, ads_api=None, optimizer=None, config=None, logger=None):
        """
        Initialize the LTV Bidding Service.

        Args:
            ads_api: Google Ads API client
            optimizer: Optimization service
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        # Initialize model configuration with defaults
        self.model_config = {
            "model_type": "gradient_boosting",  # Options: 'gradient_boosting', 'xgboost', 'lightgbm'
            "hyperparams": {
                "gradient_boosting": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "subsample": 1.0,
                    "random_state": 42,
                },
                "xgboost": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "min_child_weight": 1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "random_state": 42,
                },
                "lightgbm": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": -1,
                    "num_leaves": 31,
                    "min_child_samples": 20,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "regression",
                    "metric": "rmse",
                    "random_state": 42,
                },
            },
            "cv_folds": 5,  # Number of cross-validation folds for hyperparameter tuning
            "feature_selection": True,  # Whether to perform automatic feature selection
            "feature_importance_threshold": 0.01,  # Minimum feature importance to keep
            "scaler": "standard",  # Options: 'standard', 'minmax', 'robust', 'none'
        }

        # Update with user config if provided
        if config and "model_config" in config:
            # Update only the provided parameters, keeping defaults for the rest
            for key, value in config["model_config"].items():
                if key == "hyperparams" and isinstance(value, dict):
                    # For hyperparams, update each model type separately
                    for model_type, model_params in value.items():
                        if model_type in self.model_config["hyperparams"]:
                            self.model_config["hyperparams"][model_type].update(model_params)
                else:
                    self.model_config[key] = value

        # Initialize LTV model attributes
        self.ltv_model = None
        self.feature_columns = None
        self.model_path = os.path.join("data", "ltv_model.joblib")
        self.feature_importance = None
        self.model_metadata = None  # Store model training metadata

        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Load existing model if available
        self._load_model()

        # Try to import advanced ML libraries
        self.xgboost_available = False
        self.lightgbm_available = False

        try:
            import xgboost

            self.xgboost_available = True
        except ImportError:
            self.logger.warning("XGBoost not available. Install with: pip install xgboost")

        try:
            import lightgbm

            self.lightgbm_available = False
        except ImportError:
            self.logger.warning("LightGBM not available. Install with: pip install lightgbm")

        # Add method to check hyperparameter tuning libraries
        try:
            from sklearn.model_selection import RandomizedSearchCV

            self.hyperparam_tuning_available = True
        except ImportError:
            self.hyperparam_tuning_available = False
            self.logger.warning("Hyperparameter tuning requires scikit-learn")

        self.logger.info("LTV Bidding Service initialized")

    def _load_model(self):
        """Load pre-trained LTV model if available"""
        try:
            # Check if model file exists
            if os.path.exists(self.model_path):
                self.ltv_model = joblib.load(self.model_path)
                self.logger.info("Loaded existing LTV model")

                # Load feature importance if available
                importance_path = os.path.join("data", "ltv_feature_importance.joblib")
                if os.path.exists(importance_path):
                    self.feature_importance = joblib.load(importance_path)

                # Load model metadata if available
                metadata_path = os.path.join("data", "ltv_model_metadata.joblib")
                if os.path.exists(metadata_path):
                    self.model_metadata = joblib.load(metadata_path)

                    # Check if we need to load feature columns
                    if self.feature_columns is None and "features" in self.feature_importance:
                        self.feature_columns = self.feature_importance["features"]

                return True
            else:
                self.logger.info("No existing LTV model found")
                return False
        except Exception as e:
            self.logger.error(f"Error loading LTV model: {str(e)}")
            return False

    def _save_model(self):
        """Save the trained LTV model and its metadata"""
        try:
            if self.ltv_model is not None:
                # Create data directory if it doesn't exist
                os.makedirs("data", exist_ok=True)

                # Save the model
                joblib.dump(self.ltv_model, self.model_path)

                # Save feature importance if available
                if self.feature_importance is not None:
                    importance_path = os.path.join("data", "ltv_feature_importance.joblib")
                    joblib.dump(self.feature_importance, importance_path)

                # Save model metadata if available
                if self.model_metadata is not None:
                    metadata_path = os.path.join("data", "ltv_model_metadata.joblib")
                    joblib.dump(self.model_metadata, metadata_path)

                self.logger.info("Saved LTV model and metadata")
                return True
            else:
                self.logger.warning("No LTV model to save")
                return False
        except Exception as e:
            self.logger.error(f"Error saving LTV model: {str(e)}")
            return False

    def train_ltv_model(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        days: int = 365,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "customer_ltv",
        hyperparam_tuning: bool = False,
        model_type: Optional[str] = None,
        custom_hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train an LTV prediction model using historical conversion data.

        Args:
            historical_data: DataFrame containing historical customer data
                If None, data will be fetched from the Ads API
            days: Number of days of historical data to use
            feature_cols: List of feature column names to use for training
            target_col: Name of the target column (LTV value)
            hyperparam_tuning: Whether to perform hyperparameter tuning
            model_type: Type of model to use: 'gradient_boosting', 'xgboost', 'lightgbm'
                If None, uses the value from model_config
            custom_hyperparams: Custom hyperparameters to override defaults

        Returns:
            Dictionary with training results
        """
        start_time = datetime.now()

        # Use provided model_type or fall back to config
        if model_type is not None:
            chosen_model_type = model_type
        else:
            chosen_model_type = self.model_config["model_type"]

        self.logger.info(f"Training LTV model with {days} days of data using {chosen_model_type}")

        # Validate model type
        valid_models = ["gradient_boosting", "xgboost", "lightgbm"]
        if chosen_model_type not in valid_models:
            return {
                "status": "error",
                "message": f"Invalid model type: {chosen_model_type}. Must be one of {valid_models}",
            }

        # Check if advanced models are available when requested
        if chosen_model_type == "xgboost" and not self.xgboost_available:
            return {
                "status": "error",
                "message": "XGBoost not available. Install with: pip install xgboost",
            }

        if chosen_model_type == "lightgbm" and not self.lightgbm_available:
            return {
                "status": "error",
                "message": "LightGBM not available. Install with: pip install lightgbm",
            }

        try:
            # Get data if not provided
            if historical_data is None:
                historical_data = self._fetch_historical_data(days)

                if historical_data is None or len(historical_data) == 0:
                    return {
                        "status": "error",
                        "message": "No historical data available for training",
                    }

            # Default feature columns if not specified
            if feature_cols is None:
                feature_cols = [
                    "geo_location",
                    "device",
                    "keyword_id",
                    "campaign_id",
                    "ad_group_id",
                    "match_type",
                    "conversion_lag_days",
                    "clicks_before_conversion",
                    "impressions_before_conversion",
                    "first_conversion_value",
                    "user_recency_days",
                    "user_frequency",
                    "average_time_on_site",
                    "pages_per_session",
                ]

            # Check if all required columns exist
            for col in feature_cols + [target_col]:
                if col not in historical_data.columns:
                    self.logger.warning(f"Column {col} not found in data")
                    # Skip missing columns
                    if col in feature_cols:
                        feature_cols.remove(col)
                    else:
                        return {
                            "status": "error",
                            "message": f"Target column {target_col} not found in data",
                        }

            self.feature_columns = feature_cols

            # Prepare data
            X = historical_data[feature_cols]
            y = historical_data[target_col]

            # Handle categorical features
            X = pd.get_dummies(X, drop_first=True)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Select scaler based on config
            scaler_type = self.model_config["scaler"]
            if scaler_type == "standard":
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
            elif scaler_type == "minmax":
                from sklearn.preprocessing import MinMaxScaler

                scaler = MinMaxScaler()
            elif scaler_type == "robust":
                from sklearn.preprocessing import RobustScaler

                scaler = RobustScaler()
            else:  # 'none'
                scaler = None

            # Create base model based on chosen type
            if chosen_model_type == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingRegressor

                # Get hyperparameters, using custom if provided
                base_params = self.model_config["hyperparams"]["gradient_boosting"].copy()
                if custom_hyperparams:
                    base_params.update(custom_hyperparams)

                base_model = GradientBoostingRegressor(**base_params)

            elif chosen_model_type == "xgboost":
                import xgboost as xgb

                # Get hyperparameters, using custom if provided
                base_params = self.model_config["hyperparams"]["xgboost"].copy()
                if custom_hyperparams:
                    base_params.update(custom_hyperparams)

                base_model = xgb.XGBRegressor(**base_params)

            elif chosen_model_type == "lightgbm":
                import lightgbm as lgb

                # Get hyperparameters, using custom if provided
                base_params = self.model_config["hyperparams"]["lightgbm"].copy()
                if custom_hyperparams:
                    base_params.update(custom_hyperparams)

                base_model = lgb.LGBMRegressor(**base_params)

            # Hyperparameter tuning if requested
            if hyperparam_tuning and self.hyperparam_tuning_available:
                self.logger.info("Performing hyperparameter tuning")

                from sklearn.model_selection import RandomizedSearchCV

                # Define parameter search space based on model type
                if chosen_model_type == "gradient_boosting":
                    param_distributions = {
                        "n_estimators": [50, 100, 150, 200],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [3, 4, 5, 6, 7],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "subsample": [0.6, 0.8, 1.0],
                    }
                elif chosen_model_type == "xgboost":
                    param_distributions = {
                        "n_estimators": [50, 100, 150, 200],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [3, 4, 5, 6, 7],
                        "min_child_weight": [1, 3, 5],
                        "subsample": [0.6, 0.8, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0],
                        "gamma": [0, 0.1, 0.2],
                    }
                elif chosen_model_type == "lightgbm":
                    param_distributions = {
                        "n_estimators": [50, 100, 150, 200],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "num_leaves": [20, 31, 40, 60],
                        "max_depth": [-1, 5, 10, 15],
                        "min_child_samples": [10, 20, 30],
                        "subsample": [0.6, 0.8, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0],
                    }

                # Set up RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_distributions,
                    n_iter=20,  # Number of parameter settings sampled
                    cv=self.model_config["cv_folds"],
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,  # Use all available cores
                    random_state=42,
                )

                # Fit on training data
                if scaler:
                    X_train_scaled = scaler.fit_transform(X_train)
                    random_search.fit(X_train_scaled, y_train)
                else:
                    random_search.fit(X_train, y_train)

                # Get best estimator
                regressor = random_search.best_estimator_
                best_params = random_search.best_params_

                self.logger.info(f"Best parameters: {best_params}")
            else:
                # Use base model without tuning
                regressor = base_model
                best_params = None

            # Create final pipeline
            if scaler:
                from sklearn.pipeline import Pipeline

                model = Pipeline([("scaler", scaler), ("regressor", regressor)])
            else:
                model = regressor

            # Train final model on all training data
            model.fit(X_train, y_train)
            self.ltv_model = model

            # Save feature importance
            if chosen_model_type == "gradient_boosting":
                feature_importances = (
                    model.named_steps["regressor"].feature_importances_
                    if scaler
                    else model.feature_importances_
                )
            elif chosen_model_type == "xgboost":
                feature_importances = (
                    model.named_steps["regressor"].feature_importances_
                    if scaler
                    else model.feature_importances_
                )
            elif chosen_model_type == "lightgbm":
                feature_importances = (
                    model.named_steps["regressor"].feature_importances_
                    if scaler
                    else model.feature_importances_
                )

            self.feature_importance = {
                "features": X.columns.tolist(),
                "importance": feature_importances.tolist(),
            }

            # Evaluate model
            if scaler:
                X_test_scaled = model.named_steps["scaler"].transform(X_test)
                y_pred = model.named_steps["regressor"].predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Create model metadata
            self.model_metadata = {
                "model_type": chosen_model_type,
                "training_date": datetime.now().isoformat(),
                "data_size": len(historical_data),
                "feature_count": len(X.columns),
                "hyperparameters": (
                    best_params
                    if best_params
                    else (
                        custom_hyperparams
                        if custom_hyperparams
                        else self.model_config["hyperparams"][chosen_model_type]
                    )
                ),
                "performance": {"mse": mse, "rmse": np.sqrt(mse), "r2": r2},
            }

            # Feature selection if enabled
            if self.model_config["feature_selection"]:
                # Sort features by importance
                importances = list(
                    zip(self.feature_importance["features"], self.feature_importance["importance"])
                )
                importances.sort(key=lambda x: x[1], reverse=True)

                # Keep only features above threshold
                threshold = self.model_config["feature_importance_threshold"]
                important_features = [f for f, imp in importances if imp >= threshold]

                self.logger.info(
                    f"Selected {len(important_features)} important features out of {len(X.columns)}"
                )

                # Update feature importance
                self.feature_importance = {
                    "features": [f for f, _ in importances if f in important_features],
                    "importance": [imp for _, imp in importances if _ in important_features],
                }

            # Save the model
            self._save_model()

            self._track_execution(start_time, True)

            # Prepare feature importance for return
            feature_imp_dict = dict(
                zip(self.feature_importance["features"], self.feature_importance["importance"])
            )

            # Sort by importance
            feature_imp_dict = {
                k: v
                for k, v in sorted(feature_imp_dict.items(), key=lambda item: item[1], reverse=True)
            }

            return {
                "status": "success",
                "message": "LTV model training completed",
                "model_type": chosen_model_type,
                "model_metrics": {"mse": mse, "rmse": np.sqrt(mse), "r2": r2},
                "hyperparameters": (
                    best_params
                    if best_params
                    else (
                        custom_hyperparams
                        if custom_hyperparams
                        else self.model_config["hyperparams"][chosen_model_type]
                    )
                ),
                "feature_importance": feature_imp_dict,
                "trained_on_samples": len(X_train),
            }

        except Exception as e:
            self.logger.error(f"Error training LTV model: {str(e)}")
            self._track_execution(start_time, False)

            return {"status": "error", "message": f"Error training LTV model: {str(e)}"}

    def predict_customer_ltv(
        self,
        customer_features: Dict[str, Any],
        return_confidence_interval: bool = False,
        return_feature_contribution: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict the lifetime value for a customer with given features.

        Args:
            customer_features: Dictionary of customer features
            return_confidence_interval: Whether to return confidence intervals
            return_feature_contribution: Whether to return SHAP feature contributions

        Returns:
            Dictionary with prediction results
        """
        start_time = datetime.now()

        try:
            if self.ltv_model is None:
                return {"status": "error", "message": "LTV model not trained yet"}

            # Convert features to DataFrame
            df = pd.DataFrame([customer_features])

            # Ensure all required columns exist
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0  # Default value for missing features

            # Select only the features used by the model
            df = df[self.feature_columns]

            # Handle categorical features
            df = pd.get_dummies(df, drop_first=True)

            # Align columns with the model's expected columns
            model_columns = self.feature_importance["features"]
            for col in model_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[model_columns]

            # Determine model type if metadata is available
            model_type = (
                self.model_metadata.get("model_type", "gradient_boosting")
                if self.model_metadata
                else "gradient_boosting"
            )

            # Check if model has scaler (Pipeline)
            has_scaler = (
                hasattr(self.ltv_model, "named_steps") and "scaler" in self.ltv_model.named_steps
            )

            # Make prediction
            if has_scaler:
                X_scaled = self.ltv_model.named_steps["scaler"].transform(df)
                predicted_ltv = self.ltv_model.named_steps["regressor"].predict(X_scaled)[0]
            else:
                predicted_ltv = self.ltv_model.predict(df)[0]

            # Calculate confidence estimate based on model metadata and data proximity
            confidence_estimate = 0.8  # Default
            confidence_interval = None

            # Calculate confidence interval if requested
            if return_confidence_interval:
                try:
                    # Method depends on model type
                    if model_type == "gradient_boosting":
                        # Use quantile regression with GradientBoostingRegressor
                        if (
                            hasattr(self.ltv_model, "named_steps")
                            and "regressor" in self.ltv_model.named_steps
                        ):
                            regressor = self.ltv_model.named_steps["regressor"]
                        else:
                            regressor = self.ltv_model

                        # Check if regressor supports quantile prediction
                        if hasattr(regressor, "predict") and callable(
                            getattr(regressor, "predict")
                        ):
                            # Set alpha for 95% confidence interval
                            lower_quantile = max(
                                predicted_ltv
                                - 2
                                * np.sqrt(
                                    regressor.loss_.get_init_raw_predictions(
                                        X_scaled if has_scaler else df, regressor.init_
                                    ).squeeze()
                                ),
                                0,
                            )
                            upper_quantile = predicted_ltv + 2 * np.sqrt(
                                regressor.loss_.get_init_raw_predictions(
                                    X_scaled if has_scaler else df, regressor.init_
                                ).squeeze()
                            )
                            confidence_interval = [float(lower_quantile), float(upper_quantile)]

                    elif model_type in ["xgboost", "lightgbm"]:
                        # For tree-based models, approximate CI using leaf variance or bootstrap
                        # This is a simplified approach - more sophisticated methods exist
                        std_estimate = (
                            predicted_ltv * 0.15
                        )  # Estimate standard deviation as 15% of prediction
                        lower_bound = max(
                            predicted_ltv - 1.96 * std_estimate, 0
                        )  # 95% CI lower bound
                        upper_bound = predicted_ltv + 1.96 * std_estimate  # 95% CI upper bound
                        confidence_interval = [float(lower_bound), float(upper_bound)]

                    # Calculate confidence estimate from interval width
                    if confidence_interval:
                        interval_width = confidence_interval[1] - confidence_interval[0]
                        # Narrower intervals = higher confidence
                        confidence_estimate = max(
                            0.5,
                            min(
                                0.95,
                                1.0
                                - (
                                    interval_width / (predicted_ltv * 2)
                                    if predicted_ltv > 0
                                    else 0.5
                                ),
                            ),
                        )

                except Exception as e:
                    self.logger.warning(f"Error calculating confidence interval: {str(e)}")
                    # Fall back to simpler confidence estimate
                    confidence_interval = [
                        max(0, predicted_ltv * 0.7),  # Lower bound: 70% of prediction
                        predicted_ltv * 1.3,  # Upper bound: 130% of prediction
                    ]

            # Calculate feature contributions using SHAP if requested
            feature_contributions = None
            if return_feature_contribution:
                try:
                    import shap

                    if model_type == "gradient_boosting":
                        if has_scaler:
                            explainer = shap.TreeExplainer(self.ltv_model.named_steps["regressor"])
                            shap_values = explainer.shap_values(X_scaled)
                        else:
                            explainer = shap.TreeExplainer(self.ltv_model)
                            shap_values = explainer.shap_values(df)

                    elif model_type == "xgboost":
                        if has_scaler:
                            explainer = shap.TreeExplainer(self.ltv_model.named_steps["regressor"])
                            shap_values = explainer.shap_values(X_scaled)
                        else:
                            explainer = shap.TreeExplainer(self.ltv_model)
                            shap_values = explainer.shap_values(df)

                    elif model_type == "lightgbm":
                        if has_scaler:
                            explainer = shap.TreeExplainer(self.ltv_model.named_steps["regressor"])
                            shap_values = explainer.shap_values(X_scaled)
                        else:
                            explainer = shap.TreeExplainer(self.ltv_model)
                            shap_values = explainer.shap_values(df)

                    # Convert to dictionary of feature contributions
                    if isinstance(shap_values, list):
                        # For multi-output models, take the first output
                        shap_values = shap_values[0] if len(shap_values) > 0 else shap_values

                    feature_contributions = dict(zip(model_columns, shap_values[0]))

                    # Sort by absolute contribution
                    feature_contributions = {
                        k: float(v)
                        for k, v in sorted(
                            feature_contributions.items(),
                            key=lambda item: abs(item[1]),
                            reverse=True,
                        )
                    }

                except ImportError:
                    self.logger.warning(
                        "SHAP library not available. Install with: pip install shap"
                    )
                except Exception as e:
                    self.logger.warning(f"Error calculating feature contributions: {str(e)}")

            self._track_execution(start_time, True)

            result = {
                "status": "success",
                "predicted_ltv": float(predicted_ltv),
                "confidence": float(confidence_estimate),
            }

            if confidence_interval:
                result["confidence_interval"] = confidence_interval

            if feature_contributions:
                result["feature_contributions"] = feature_contributions

                # Add a simple explanation of top contributors
                top_contributors = list(feature_contributions.items())[:5]  # Top 5 contributors
                explanation = []

                for feature, contribution in top_contributors:
                    if contribution > 0:
                        explanation.append(f"{feature} increases LTV by ${abs(contribution):.2f}")
                    else:
                        explanation.append(f"{feature} decreases LTV by ${abs(contribution):.2f}")

                result["explanation"] = explanation

            return result

        except Exception as e:
            self.logger.error(f"Error predicting customer LTV: {str(e)}")
            self._track_execution(start_time, False)

            return {"status": "error", "message": f"Error predicting customer LTV: {str(e)}"}

    def generate_ltv_bid_adjustments(
        self,
        campaign_id: Optional[str] = None,
        ad_group_id: Optional[str] = None,
        min_data_points: int = 100,
        min_confidence: float = 0.7,
        max_bid_adjustment: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Generate bid adjustments based on predicted customer LTV.

        Args:
            campaign_id: Optional campaign ID to focus on
            ad_group_id: Optional ad group ID to focus on
            min_data_points: Minimum number of data points required for adjustment
            min_confidence: Minimum confidence level required for adjustment
            max_bid_adjustment: Maximum bid adjustment factor (e.g., 0.5 = ±50%)

        Returns:
            Dictionary with bid adjustment recommendations
        """
        start_time = datetime.now()
        self.logger.info(f"Generating LTV-based bid adjustments for campaign_id={campaign_id}")

        try:
            if self.ltv_model is None:
                return {"status": "error", "message": "LTV model not trained yet"}

            # Get conversion data
            conversion_data = self._fetch_conversion_data(campaign_id, ad_group_id)

            if conversion_data is None or len(conversion_data) < min_data_points:
                return {
                    "status": "warning",
                    "message": f"Insufficient data points ({len(conversion_data) if conversion_data is not None else 0} < {min_data_points})",
                }

            # Process each keyword/ad group/segment
            results = []
            segments = self._segment_conversion_data(conversion_data)

            for segment_name, segment_data in segments.items():
                # Skip segments with insufficient data
                if (
                    len(segment_data) < min_data_points / 10
                ):  # Require at least some minimum per segment
                    continue

                # Calculate average predicted LTV
                segment_features = self._extract_segment_features(segment_data)
                ltv_predictions = []

                for features in segment_features:
                    prediction = self.predict_customer_ltv(features)
                    if prediction["status"] == "success":
                        ltv_predictions.append(prediction["predicted_ltv"])

                if not ltv_predictions:
                    continue

                avg_ltv = np.mean(ltv_predictions)

                # Calculate relative value compared to overall average
                overall_avg_ltv = np.mean([data["customer_ltv"] for data in conversion_data])
                relative_value = avg_ltv / overall_avg_ltv if overall_avg_ltv > 0 else 1.0

                # Calculate bid adjustment, capped at max_bid_adjustment
                bid_adjustment = min(
                    max(relative_value - 1, -max_bid_adjustment), max_bid_adjustment
                )

                # Calculate confidence based on sample size and variance
                confidence = min(len(segment_data) / min_data_points, 1.0) * (
                    1 - np.std(ltv_predictions) / avg_ltv if avg_ltv > 0 else 0
                )

                if confidence >= min_confidence:
                    segment_parts = segment_name.split("|")

                    results.append(
                        {
                            "segment_type": segment_parts[0],
                            "segment_value": segment_parts[1] if len(segment_parts) > 1 else "",
                            "predicted_ltv": avg_ltv,
                            "relative_value": relative_value,
                            "bid_adjustment": bid_adjustment,
                            "confidence": confidence,
                            "sample_size": len(segment_data),
                        }
                    )

            # Sort results by absolute bid adjustment value
            results.sort(key=lambda x: abs(x["bid_adjustment"]), reverse=True)

            self._track_execution(start_time, True)

            return {
                "status": "success",
                "message": f"Generated {len(results)} LTV-based bid adjustments",
                "adjustments": results,
            }

        except Exception as e:
            self.logger.error(f"Error generating LTV bid adjustments: {str(e)}")
            self._track_execution(start_time, False)

            return {"status": "error", "message": f"Error generating LTV bid adjustments: {str(e)}"}

    def apply_ltv_bid_adjustments(
        self, adjustments: List[Dict[str, Any]], dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply the recommended LTV-based bid adjustments.

        Args:
            adjustments: List of bid adjustment recommendations
            dry_run: If True, simulates the application without making changes

        Returns:
            Dictionary with application results
        """
        start_time = datetime.now()
        self.logger.info(
            f"Applying {len(adjustments)} LTV-based bid adjustments (dry_run={dry_run})"
        )

        try:
            if not adjustments:
                return {"status": "warning", "message": "No adjustments to apply"}

            # Track application results
            results = {"applied": 0, "skipped": 0, "failed": 0, "details": []}

            for adjustment in adjustments:
                segment_type = adjustment["segment_type"]
                segment_value = adjustment["segment_value"]
                bid_adjustment = adjustment["bid_adjustment"]

                try:
                    if dry_run:
                        # Simulate application
                        success = True
                        message = f"Would apply {bid_adjustment:.2%} adjustment to {segment_type}={segment_value}"
                    else:
                        # Actually apply the adjustment
                        success = self._apply_bid_adjustment(
                            segment_type, segment_value, bid_adjustment
                        )
                        message = f"Applied {bid_adjustment:.2%} adjustment to {segment_type}={segment_value}"

                    if success:
                        results["applied"] += 1
                    else:
                        results["failed"] += 1
                        message = f"Failed to apply adjustment to {segment_type}={segment_value}"

                    results["details"].append(
                        {
                            "segment_type": segment_type,
                            "segment_value": segment_value,
                            "bid_adjustment": bid_adjustment,
                            "success": success,
                            "message": message,
                        }
                    )

                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "segment_type": segment_type,
                            "segment_value": segment_value,
                            "bid_adjustment": bid_adjustment,
                            "success": False,
                            "message": f"Error: {str(e)}",
                        }
                    )

            self._track_execution(start_time, True)

            return {
                "status": "success",
                "message": f"Applied {results['applied']} LTV-based bid adjustments ({results['failed']} failed, {results['skipped']} skipped)",
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"Error applying LTV bid adjustments: {str(e)}")
            self._track_execution(start_time, False)

            return {"status": "error", "message": f"Error applying LTV bid adjustments: {str(e)}"}

    def optimize_campaign_budget_by_ltv(
        self, days: int = 30, reallocation_percent: float = 0.2
    ) -> Dict[str, Any]:
        """
        Optimize budget allocation across campaigns based on predicted LTV.

        Args:
            days: Number of days of historical data to use
            reallocation_percent: Maximum percentage of budget to reallocate (0.0 to 1.0)

        Returns:
            Dictionary with budget optimization results
        """
        start_time = datetime.now()
        self.logger.info(
            f"Optimizing campaign budgets based on LTV (reallocation_percent={reallocation_percent})"
        )

        try:
            if self.ltv_model is None:
                return {"status": "error", "message": "LTV model not trained yet"}

            # Get campaign performance data
            campaigns = self._fetch_campaign_performance(days)

            if not campaigns:
                return {"status": "error", "message": "No campaign data available"}

            # Calculate LTV score for each campaign
            for campaign in campaigns:
                # Get recent conversions for this campaign
                conversions = self._fetch_conversion_data(campaign["campaign_id"])

                if not conversions:
                    campaign["ltv_score"] = 0
                    campaign["avg_predicted_ltv"] = 0
                    continue

                # Calculate average predicted LTV
                segment_features = self._extract_segment_features(conversions)
                ltv_predictions = []

                for features in segment_features:
                    prediction = self.predict_customer_ltv(features)
                    if prediction["status"] == "success":
                        ltv_predictions.append(prediction["predicted_ltv"])

                avg_ltv = np.mean(ltv_predictions) if ltv_predictions else 0

                # Calculate LTV score (LTV to Cost ratio)
                cost = campaign.get("cost", 0)
                conversions_count = campaign.get("conversions", 0)

                if cost > 0 and conversions_count > 0:
                    # LTV score = average LTV * number of conversions / cost
                    ltv_score = (avg_ltv * conversions_count) / cost
                else:
                    ltv_score = 0

                campaign["ltv_score"] = ltv_score
                campaign["avg_predicted_ltv"] = avg_ltv

            # Sort campaigns by LTV score
            campaigns.sort(key=lambda x: x["ltv_score"], reverse=True)

            # Calculate budget adjustments
            total_budget = sum(campaign.get("budget", 0) for campaign in campaigns)
            budget_to_reallocate = total_budget * reallocation_percent

            # Allocate budget from low-performing to high-performing campaigns
            high_performers = [c for c in campaigns if c["ltv_score"] > 1.0]
            low_performers = [c for c in campaigns if c["ltv_score"] <= 1.0]

            # Skip if no clear separation of performers
            if not high_performers or not low_performers:
                return {
                    "status": "warning",
                    "message": "No clear high/low performers to adjust budget between",
                }

            # Calculate budget reduction for low performers
            low_performer_budget = sum(campaign.get("budget", 0) for campaign in low_performers)
            reduction_factor = (
                budget_to_reallocate / low_performer_budget if low_performer_budget > 0 else 0
            )

            # Calculate budget increase for high performers (weighted by LTV score)
            total_high_score = sum(campaign["ltv_score"] for campaign in high_performers)

            # Prepare recommendations
            recommendations = []

            for campaign in campaigns:
                current_budget = campaign.get("budget", 0)
                campaign_id = campaign["campaign_id"]

                if campaign in low_performers:
                    # Reduce budget for low performers
                    new_budget = current_budget * (1 - reduction_factor)
                    change = new_budget - current_budget
                    change_percent = change / current_budget if current_budget > 0 else 0
                else:
                    # Increase budget for high performers
                    weight = campaign["ltv_score"] / total_high_score if total_high_score > 0 else 0
                    budget_increase = budget_to_reallocate * weight
                    new_budget = current_budget + budget_increase
                    change = budget_increase
                    change_percent = change / current_budget if current_budget > 0 else 0

                recommendations.append(
                    {
                        "campaign_id": campaign_id,
                        "campaign_name": campaign.get("campaign_name", ""),
                        "current_budget": current_budget,
                        "recommended_budget": new_budget,
                        "change": change,
                        "change_percent": change_percent,
                        "ltv_score": campaign["ltv_score"],
                        "avg_predicted_ltv": campaign["avg_predicted_ltv"],
                    }
                )

            self._track_execution(start_time, True)

            return {
                "status": "success",
                "message": f"Generated budget optimization recommendations for {len(campaigns)} campaigns",
                "recommendations": recommendations,
                "total_budget": total_budget,
                "budget_reallocated": budget_to_reallocate,
            }

        except Exception as e:
            self.logger.error(f"Error optimizing campaign budget by LTV: {str(e)}")
            self._track_execution(start_time, False)

            return {
                "status": "error",
                "message": f"Error optimizing campaign budget by LTV: {str(e)}",
            }

    def evaluate_ltv_performance(self, days: int = 90, lookback_days: int = 180) -> Dict[str, Any]:
        """
        Evaluate performance of LTV-based optimizations.

        Args:
            days: Number of days of optimization data to evaluate
            lookback_days: Number of days prior to the optimization to use as baseline

        Returns:
            Dictionary with evaluation results
        """
        start_time = datetime.now()
        self.logger.info(f"Evaluating LTV bidding performance over {days} days")

        try:
            # Get optimization history
            history = self._get_optimization_history(days)

            if not history:
                return {
                    "status": "warning",
                    "message": "No optimization history found for evaluation",
                }

            # Group optimizations by target (campaign, ad group, etc.)
            grouped_history = {}
            for item in history:
                target = item.get("target_id", "")
                if target not in grouped_history:
                    grouped_history[target] = []
                grouped_history[target].append(item)

            # Evaluate each target
            evaluations = []

            for target_id, target_history in grouped_history.items():
                if not target_history:
                    continue

                # Get performance before and after optimization
                optimization_date = datetime.fromisoformat(target_history[0]["timestamp"])

                before_start = optimization_date - timedelta(days=lookback_days)
                before_end = optimization_date - timedelta(days=1)
                after_start = optimization_date + timedelta(days=1)
                after_end = optimization_date + timedelta(days=lookback_days)

                before_performance = self._get_target_performance(
                    target_id, before_start, before_end
                )

                after_performance = self._get_target_performance(target_id, after_start, after_end)

                if not before_performance or not after_performance:
                    continue

                # Calculate metrics
                metrics = {}

                for metric in ["cost", "clicks", "conversions", "conversion_value"]:
                    before_value = before_performance.get(metric, 0)
                    after_value = after_performance.get(metric, 0)

                    if before_value > 0:
                        change = (after_value - before_value) / before_value
                    else:
                        change = 0

                    metrics[f"{metric}_before"] = before_value
                    metrics[f"{metric}_after"] = after_value
                    metrics[f"{metric}_change"] = change

                # Calculate derived metrics
                before_cpa = before_performance.get("cost", 0) / before_performance.get(
                    "conversions", 1
                )
                after_cpa = after_performance.get("cost", 0) / after_performance.get(
                    "conversions", 1
                )

                before_roas = before_performance.get(
                    "conversion_value", 0
                ) / before_performance.get("cost", 1)
                after_roas = after_performance.get("conversion_value", 0) / after_performance.get(
                    "cost", 1
                )

                metrics["cpa_before"] = before_cpa
                metrics["cpa_after"] = after_cpa
                metrics["cpa_change"] = (
                    (after_cpa - before_cpa) / before_cpa if before_cpa > 0 else 0
                )

                metrics["roas_before"] = before_roas
                metrics["roas_after"] = after_roas
                metrics["roas_change"] = (
                    (after_roas - before_roas) / before_roas if before_roas > 0 else 0
                )

                # Add to evaluations
                evaluations.append(
                    {
                        "target_id": target_id,
                        "target_type": target_history[0].get("target_type", ""),
                        "optimization_date": optimization_date.isoformat(),
                        "num_optimizations": len(target_history),
                        "metrics": metrics,
                    }
                )

            # Calculate aggregate results
            aggregate = {
                "targets_evaluated": len(evaluations),
                "targets_improved": sum(
                    1 for e in evaluations if e["metrics"].get("roas_change", 0) > 0
                ),
                "avg_roas_change": np.mean(
                    [e["metrics"].get("roas_change", 0) for e in evaluations]
                ),
                "avg_cost_change": np.mean(
                    [e["metrics"].get("cost_change", 0) for e in evaluations]
                ),
                "avg_conversions_change": np.mean(
                    [e["metrics"].get("conversions_change", 0) for e in evaluations]
                ),
            }

            self._track_execution(start_time, True)

            return {
                "status": "success",
                "message": f"Evaluated LTV bidding performance on {len(evaluations)} targets",
                "evaluations": evaluations,
                "aggregate": aggregate,
            }

        except Exception as e:
            self.logger.error(f"Error evaluating LTV performance: {str(e)}")
            self._track_execution(start_time, False)

            return {"status": "error", "message": f"Error evaluating LTV performance: {str(e)}"}

    def perform_ltv_cohort_analysis(
        self,
        days: int = 365,
        cohort_period: str = "month",
        min_cohort_size: int = 30,
        retention_type: str = "user",
    ) -> Dict[str, Any]:
        """
        Perform cohort analysis on customer LTV data to understand value development over time.

        Args:
            days: Number of days of historical data to use
            cohort_period: Period to group cohorts ('day', 'week', 'month', 'quarter')
            min_cohort_size: Minimum number of customers in a cohort to include in analysis
            retention_type: Type of retention to analyze ('user', 'revenue', 'conversion')

        Returns:
            Dictionary with cohort analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Performing LTV cohort analysis with {days} days of data")

        try:
            # Get historical conversion data
            historical_data = self._fetch_historical_data(days)

            if historical_data is None or len(historical_data) == 0:
                return {
                    "status": "error",
                    "message": "No historical data available for cohort analysis",
                }

            # Ensure required columns exist
            required_columns = [
                "customer_id",
                "first_conversion_date",
                "conversion_date",
                "conversion_value",
            ]
            for col in required_columns:
                if col not in historical_data.columns:
                    # Try to generate missing columns if possible
                    if (
                        col == "first_conversion_date"
                        and "conversion_date" in historical_data.columns
                        and "customer_id" in historical_data.columns
                    ):
                        # Get first conversion date for each customer
                        first_dates = (
                            historical_data.groupby("customer_id")["conversion_date"]
                            .min()
                            .reset_index()
                        )
                        first_dates.columns = ["customer_id", "first_conversion_date"]
                        historical_data = pd.merge(historical_data, first_dates, on="customer_id")
                    else:
                        return {
                            "status": "error",
                            "message": f"Required column '{col}' not found in historical data",
                        }

            # Convert date columns to datetime if they're not already
            date_columns = ["first_conversion_date", "conversion_date"]
            for col in date_columns:
                if not pd.api.types.is_datetime64_dtype(historical_data[col]):
                    historical_data[col] = pd.to_datetime(historical_data[col])

            # Create cohort period column
            if cohort_period == "day":
                historical_data["cohort_period"] = historical_data[
                    "first_conversion_date"
                ].dt.strftime("%Y-%m-%d")
            elif cohort_period == "week":
                historical_data["cohort_period"] = (
                    historical_data["first_conversion_date"].dt.to_period("W").astype(str)
                )
            elif cohort_period == "month":
                historical_data["cohort_period"] = historical_data[
                    "first_conversion_date"
                ].dt.strftime("%Y-%m")
            elif cohort_period == "quarter":
                historical_data["cohort_period"] = (
                    historical_data["first_conversion_date"].dt.to_period("Q").astype(str)
                )
            else:
                return {
                    "status": "error",
                    "message": f"Invalid cohort_period: {cohort_period}. Must be 'day', 'week', 'month', or 'quarter'",
                }

            # Calculate time since first conversion in the same units as cohort_period
            if cohort_period == "day":
                historical_data["periods_since_first"] = (
                    historical_data["conversion_date"] - historical_data["first_conversion_date"]
                ).dt.days
            elif cohort_period == "week":
                historical_data["periods_since_first"] = (
                    historical_data["conversion_date"] - historical_data["first_conversion_date"]
                ).dt.days // 7
            elif cohort_period == "month":
                historical_data["periods_since_first"] = (
                    historical_data["conversion_date"].dt.year
                    - historical_data["first_conversion_date"].dt.year
                ) * 12 + (
                    historical_data["conversion_date"].dt.month
                    - historical_data["first_conversion_date"].dt.month
                )
            elif cohort_period == "quarter":
                historical_data["periods_since_first"] = (
                    (
                        historical_data["conversion_date"].dt.year
                        - historical_data["first_conversion_date"].dt.year
                    )
                    * 4
                    + ((historical_data["conversion_date"].dt.month - 1) // 3)
                    - ((historical_data["first_conversion_date"].dt.month - 1) // 3)
                )

            # Create cohort analysis based on retention_type
            if retention_type == "user":
                # Count unique customers in each cohort and period
                cohort_data = (
                    historical_data.groupby(["cohort_period", "periods_since_first"])["customer_id"]
                    .nunique()
                    .reset_index()
                )
                cohort_data.columns = ["cohort_period", "periods_since_first", "customer_count"]

                # Get initial size of each cohort
                cohort_sizes = cohort_data[cohort_data["periods_since_first"] == 0].copy()
                cohort_sizes.columns = ["cohort_period", "periods_since_first", "initial_size"]

                # Merge to get retention rates
                cohort_data = pd.merge(
                    cohort_data, cohort_sizes[["cohort_period", "initial_size"]], on="cohort_period"
                )
                cohort_data["retention_rate"] = (
                    cohort_data["customer_count"] / cohort_data["initial_size"]
                )

                # Filter cohorts with minimum size
                valid_cohorts = cohort_sizes[cohort_sizes["initial_size"] >= min_cohort_size][
                    "cohort_period"
                ].unique()
                cohort_data = cohort_data[cohort_data["cohort_period"].isin(valid_cohorts)]

                # Calculate average retention rate by period
                avg_retention = (
                    cohort_data.groupby("periods_since_first")["retention_rate"]
                    .mean()
                    .reset_index()
                )

                # Calculate cumulative retention (percent of customers still active after X periods)
                cumulative_retention = avg_retention.copy()
                cumulative_retention.columns = ["periods_since_first", "avg_retention_rate"]

            elif retention_type == "revenue":
                # Sum conversion value in each cohort and period
                cohort_data = (
                    historical_data.groupby(["cohort_period", "periods_since_first"])[
                        "conversion_value"
                    ]
                    .sum()
                    .reset_index()
                )

                # Get period 0 value (first conversion) for each cohort
                cohort_initial = cohort_data[cohort_data["periods_since_first"] == 0].copy()
                cohort_initial.columns = ["cohort_period", "periods_since_first", "initial_value"]

                # Count customers in each cohort for calculating average values
                cohort_sizes = (
                    historical_data[historical_data["periods_since_first"] == 0]
                    .groupby("cohort_period")["customer_id"]
                    .nunique()
                    .reset_index()
                )
                cohort_sizes.columns = ["cohort_period", "cohort_size"]

                # Merge to get revenue development
                cohort_data = pd.merge(
                    cohort_data,
                    cohort_initial[["cohort_period", "initial_value"]],
                    on="cohort_period",
                )
                cohort_data = pd.merge(cohort_data, cohort_sizes, on="cohort_period")

                # Calculate metrics
                cohort_data["revenue_growth"] = (
                    cohort_data["conversion_value"] / cohort_data["initial_value"] - 1
                )
                cohort_data["avg_value_per_customer"] = (
                    cohort_data["conversion_value"] / cohort_data["cohort_size"]
                )

                # Filter cohorts with minimum size
                valid_cohorts = cohort_sizes[cohort_sizes["cohort_size"] >= min_cohort_size][
                    "cohort_period"
                ].unique()
                cohort_data = cohort_data[cohort_data["cohort_period"].isin(valid_cohorts)]

                # Calculate cumulative revenue by cohort
                cumulative_data = []
                for cohort in valid_cohorts:
                    cohort_cumulative = cohort_data[
                        cohort_data["cohort_period"] == cohort
                    ].sort_values("periods_since_first")
                    cohort_cumulative["cumulative_value"] = cohort_cumulative[
                        "conversion_value"
                    ].cumsum()
                    cohort_cumulative["cumulative_avg_value"] = (
                        cohort_cumulative["cumulative_value"] / cohort_cumulative["cohort_size"]
                    )
                    cumulative_data.append(cohort_cumulative)

                if cumulative_data:
                    cumulative_revenue = pd.concat(cumulative_data)

                    # Calculate average LTV development across cohorts
                    avg_ltv_development = (
                        cumulative_revenue.groupby("periods_since_first")["cumulative_avg_value"]
                        .mean()
                        .reset_index()
                    )
                else:
                    avg_ltv_development = pd.DataFrame(
                        columns=["periods_since_first", "cumulative_avg_value"]
                    )

            elif retention_type == "conversion":
                # Count conversions in each cohort and period
                cohort_data = (
                    historical_data.groupby(["cohort_period", "periods_since_first"])
                    .size()
                    .reset_index(name="conversion_count")
                )

                # Get customer count for each cohort
                customer_counts = (
                    historical_data.groupby("cohort_period")["customer_id"].nunique().reset_index()
                )
                customer_counts.columns = ["cohort_period", "customer_count"]

                # Merge to get conversion rates
                cohort_data = pd.merge(cohort_data, customer_counts, on="cohort_period")
                cohort_data["conversion_rate"] = (
                    cohort_data["conversion_count"] / cohort_data["customer_count"]
                )

                # Filter cohorts with minimum size
                valid_cohorts = customer_counts[
                    customer_counts["customer_count"] >= min_cohort_size
                ]["cohort_period"].unique()
                cohort_data = cohort_data[cohort_data["cohort_period"].isin(valid_cohorts)]

                # Calculate average conversion rate by period
                avg_conversion_rate = (
                    cohort_data.groupby("periods_since_first")["conversion_rate"]
                    .mean()
                    .reset_index()
                )

            # Create pivot table for visualization
            if len(cohort_data) > 0:
                if retention_type == "user":
                    pivot_data = cohort_data.pivot(
                        index="cohort_period",
                        columns="periods_since_first",
                        values="retention_rate",
                    )
                elif retention_type == "revenue":
                    pivot_data = cumulative_revenue.pivot(
                        index="cohort_period",
                        columns="periods_since_first",
                        values="cumulative_avg_value",
                    )
                elif retention_type == "conversion":
                    pivot_data = cohort_data.pivot(
                        index="cohort_period",
                        columns="periods_since_first",
                        values="conversion_rate",
                    )
            else:
                pivot_data = pd.DataFrame()

            # Calculate key LTV metrics based on cohort analysis
            ltv_metrics = {}

            # Prepare results
            results = {
                "status": "success",
                "retention_type": retention_type,
                "cohort_period": cohort_period,
                "analysis_date": datetime.now().isoformat(),
                "cohort_count": len(valid_cohorts),
                "max_periods": (
                    int(cohort_data["periods_since_first"].max()) if len(cohort_data) > 0 else 0
                ),
            }

            # Add type-specific metrics
            if retention_type == "user":
                results.update(
                    {
                        "avg_retention_by_period": avg_retention.to_dict("records"),
                        "retention_matrix": (
                            pivot_data.fillna(0).to_dict() if not pivot_data.empty else {}
                        ),
                    }
                )

                # Calculate average customer lifetime in periods
                if not avg_retention.empty:
                    avg_lifetime_periods = avg_retention["retention_rate"].sum()
                    results["avg_customer_lifetime_periods"] = float(avg_lifetime_periods)

            elif retention_type == "revenue":
                results.update(
                    {
                        "avg_ltv_development": avg_ltv_development.to_dict("records"),
                        "cumulative_revenue_matrix": (
                            pivot_data.fillna(0).to_dict() if not pivot_data.empty else {}
                        ),
                    }
                )

                # Calculate LTV metrics
                if not avg_ltv_development.empty:
                    max_ltv = avg_ltv_development["cumulative_avg_value"].max()
                    ltv_curve = avg_ltv_development["cumulative_avg_value"].tolist()

                    # Calculate LTV:CAC ratio if acquisition cost is available
                    acq_cost_col = "acquisition_cost"
                    if acq_cost_col in historical_data.columns:
                        avg_cac = historical_data[acq_cost_col].mean()
                        ltv_cac_ratio = max_ltv / avg_cac if avg_cac > 0 else None
                        results["ltv_cac_ratio"] = (
                            float(ltv_cac_ratio) if ltv_cac_ratio is not None else None
                        )

                    # Calculate time to break even if acquisition cost is available
                    if acq_cost_col in historical_data.columns and not avg_ltv_development.empty:
                        avg_cac = historical_data[acq_cost_col].mean()
                        if avg_cac > 0:
                            # Find first period where cumulative LTV exceeds CAC
                            for i, row in avg_ltv_development.iterrows():
                                if row["cumulative_avg_value"] >= avg_cac:
                                    results["break_even_period"] = int(row["periods_since_first"])
                                    break

                    results.update({"max_ltv": float(max_ltv), "ltv_curve": ltv_curve})

            elif retention_type == "conversion":
                results.update(
                    {
                        "avg_conversion_rate_by_period": avg_conversion_rate.to_dict("records"),
                        "conversion_rate_matrix": (
                            pivot_data.fillna(0).to_dict() if not pivot_data.empty else {}
                        ),
                    }
                )

                # Calculate repeat purchase probability
                if not avg_conversion_rate.empty and len(avg_conversion_rate) > 1:
                    repeat_rates = avg_conversion_rate[
                        avg_conversion_rate["periods_since_first"] > 0
                    ]["conversion_rate"].tolist()
                    if repeat_rates:
                        avg_repeat_rate = sum(repeat_rates) / len(repeat_rates)
                        results["avg_repeat_purchase_rate"] = float(avg_repeat_rate)

            # Add insights based on analysis
            insights = []

            if retention_type == "user":
                if "avg_customer_lifetime_periods" in results:
                    lifetime_periods = results["avg_customer_lifetime_periods"]
                    insights.append(
                        f"Average customer stays active for {lifetime_periods:.1f} {cohort_period}s"
                    )

                if not avg_retention.empty and len(avg_retention) > 1:
                    retention_drop = (
                        avg_retention.iloc[0]["retention_rate"]
                        - avg_retention.iloc[1]["retention_rate"]
                    ) / avg_retention.iloc[0]["retention_rate"]
                    insights.append(
                        f"Customer retention drops by {retention_drop:.1%} after the first {cohort_period}"
                    )

            elif retention_type == "revenue":
                if "max_ltv" in results:
                    insights.append(f"Average customer lifetime value is ${results['max_ltv']:.2f}")

                if "ltv_cac_ratio" in results and results["ltv_cac_ratio"] is not None:
                    insights.append(f"LTV:CAC ratio is {results['ltv_cac_ratio']:.2f}")

                if "break_even_period" in results:
                    insights.append(
                        f"Average customer breaks even after {results['break_even_period']} {cohort_period}s"
                    )

            elif retention_type == "conversion":
                if "avg_repeat_purchase_rate" in results:
                    insights.append(
                        f"Average repeat purchase rate is {results['avg_repeat_purchase_rate']:.1%} per {cohort_period}"
                    )

            results["insights"] = insights

            # Generate recommendations for bid adjustments based on cohort analysis
            recommendations = []

            if retention_type == "revenue":
                if "ltv_cac_ratio" in results and results["ltv_cac_ratio"] is not None:
                    if results["ltv_cac_ratio"] < 1.0:
                        recommendations.append(
                            "Decrease bids across campaigns as LTV does not exceed acquisition cost"
                        )
                    elif results["ltv_cac_ratio"] > 3.0:
                        recommendations.append(
                            "Increase bids to capture more traffic, as LTV significantly exceeds acquisition cost"
                        )

                # Analyze LTV development for different segments
                segments = ["geo_location", "device", "campaign_id", "ad_group_id"]
                for segment in segments:
                    if segment in historical_data.columns:
                        # Calculate segment-specific LTV
                        segment_ltv = (
                            historical_data.groupby(segment)["conversion_value"].sum()
                            / historical_data.groupby(segment)["customer_id"].nunique()
                        )

                        # Get top and bottom performers
                        if len(segment_ltv) > 1:
                            top_segment = segment_ltv.idxmax()
                            bottom_segment = segment_ltv.idxmin()
                            top_ltv = segment_ltv.max()
                            bottom_ltv = segment_ltv.min()

                            # Only recommend if there's a significant difference
                            if top_ltv > bottom_ltv * 1.5:  # At least 50% difference
                                recommendations.append(
                                    f"Increase bids for {segment}={top_segment} (LTV: ${top_ltv:.2f})"
                                )
                                recommendations.append(
                                    f"Decrease bids for {segment}={bottom_segment} (LTV: ${bottom_ltv:.2f})"
                                )

            results["recommendations"] = recommendations

            self._track_execution(start_time, True)

            return results

        except Exception as e:
            self.logger.error(f"Error performing LTV cohort analysis: {str(e)}")
            self._track_execution(start_time, False)

            return {"status": "error", "message": f"Error performing LTV cohort analysis: {str(e)}"}

    # Helper methods

    def _fetch_historical_data(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical customer data for LTV modeling"""
        # Placeholder - In a real implementation, this would fetch data from the Ads API
        # or other data sources

        self.logger.info(f"Fetching {days} days of historical data")

        # For now, return mock data
        if self.ads_api is None:
            self.logger.warning("No ads_api available, using mock data")
            return self._generate_mock_data(days)

        # TODO: Implement actual data fetching from Google Ads API
        try:
            # In a real implementation, this would use the Ads API to fetch data
            return self._generate_mock_data(days)
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return None

    def _fetch_conversion_data(
        self, campaign_id: Optional[str] = None, ad_group_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch conversion data for the specified campaign/ad group"""
        # Placeholder - would use the Ads API in a real implementation
        return self._generate_mock_conversion_data(campaign_id, ad_group_id)

    def _segment_conversion_data(
        self, conversion_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Segment conversion data by relevant dimensions"""
        segments = {}

        for conversion in conversion_data:
            # Create segments for different dimensions
            for dimension in ["geo_location", "device", "match_type", "ad_group_id"]:
                if dimension in conversion:
                    key = f"{dimension}|{conversion[dimension]}"

                    if key not in segments:
                        segments[key] = []

                    segments[key].append(conversion)

        return segments

    def _extract_segment_features(
        self, conversion_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract feature dictionaries from conversion data segments"""
        features = []

        for conversion in conversion_data:
            # Extract relevant features for LTV prediction
            feature_dict = {}

            for key, value in conversion.items():
                if key != "customer_ltv":  # Don't include the target variable
                    feature_dict[key] = value

            features.append(feature_dict)

        return features

    def _apply_bid_adjustment(
        self, segment_type: str, segment_value: str, adjustment: float
    ) -> bool:
        """Apply bid adjustment to a specific segment"""
        # Placeholder - in a real implementation, this would use the Ads API
        # to apply the bid adjustment

        self.logger.info(f"Applying {adjustment:.2%} adjustment to {segment_type}={segment_value}")

        if self.ads_api is None:
            self.logger.warning("No ads_api available, simulating bid adjustment")
            return True

        # TODO: Implement actual bid adjustment using Google Ads API
        # For now, just log the action and return success
        return True

    def _fetch_campaign_performance(self, days: int) -> List[Dict[str, Any]]:
        """Fetch campaign performance data"""
        # Placeholder - would use the Ads API in a real implementation
        return self._generate_mock_campaign_data(days)

    def _get_optimization_history(self, days: int) -> List[Dict[str, Any]]:
        """Get history of LTV optimization actions"""
        # Placeholder - in a real implementation, this would retrieve from a database
        history_file = os.path.join("data", "ltv_optimization_history.json")

        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)

                # Filter to required timeframe
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                return [item for item in history if item.get("timestamp", "") >= cutoff_date]
            except Exception as e:
                self.logger.error(f"Error loading optimization history: {str(e)}")

        return []

    def _get_target_performance(
        self, target_id: str, start_date: datetime, end_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific target in a date range"""
        # Placeholder - would use the Ads API in a real implementation
        return {
            "cost": np.random.uniform(100, 10000),
            "clicks": np.random.randint(100, 5000),
            "impressions": np.random.randint(1000, 100000),
            "conversions": np.random.randint(1, 500),
            "conversion_value": np.random.uniform(500, 50000),
        }

    # Mock data generation methods for testing/development

    def _generate_mock_data(self, days: int) -> pd.DataFrame:
        """Generate mock customer data for development/testing"""
        n_samples = min(days * 10, 10000)  # Scale with days but cap at reasonable size

        np.random.seed(42)  # For reproducibility

        data = {
            "geo_location": np.random.choice(["US", "UK", "CA", "AU", "DE"], n_samples),
            "device": np.random.choice(["mobile", "desktop", "tablet"], n_samples),
            "keyword_id": np.random.randint(1000, 9999, n_samples),
            "campaign_id": np.random.choice([f"campaign_{i}" for i in range(1, 6)], n_samples),
            "ad_group_id": np.random.choice([f"adgroup_{i}" for i in range(1, 11)], n_samples),
            "match_type": np.random.choice(["exact", "phrase", "broad"], n_samples),
            "conversion_lag_days": np.random.randint(0, 30, n_samples),
            "clicks_before_conversion": np.random.randint(1, 10, n_samples),
            "impressions_before_conversion": np.random.randint(1, 50, n_samples),
            "first_conversion_value": np.random.uniform(10, 500, n_samples),
            "user_recency_days": np.random.randint(0, 365, n_samples),
            "user_frequency": np.random.randint(1, 20, n_samples),
            "average_time_on_site": np.random.uniform(30, 600, n_samples),
            "pages_per_session": np.random.uniform(1, 10, n_samples),
        }

        # Generate LTV with some relationship to features
        ltv_base = data["first_conversion_value"] * (1 + 0.1 * data["user_frequency"])
        device_factor = np.where(
            data["device"] == "desktop", 1.2, np.where(data["device"] == "mobile", 0.8, 1.0)
        )
        geo_factor = np.where(
            data["geo_location"] == "US", 1.3, np.where(data["geo_location"] == "UK", 1.1, 0.9)
        )

        # Add some noise
        noise = np.random.normal(1, 0.3, n_samples)

        data["customer_ltv"] = ltv_base * device_factor * geo_factor * noise

        return pd.DataFrame(data)

    def _generate_mock_conversion_data(
        self, campaign_id: Optional[str] = None, ad_group_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate mock conversion data for development/testing"""
        n_samples = 200

        np.random.seed(42)  # For reproducibility

        # Generate base data
        conversions = []

        for _ in range(n_samples):
            # If campaign_id specified, use it; otherwise random
            cam_id = campaign_id or f"campaign_{np.random.randint(1, 6)}"
            # If ad_group_id specified, use it; otherwise random
            ag_id = ad_group_id or f"adgroup_{np.random.randint(1, 11)}"

            conv = {
                "campaign_id": cam_id,
                "ad_group_id": ag_id,
                "geo_location": np.random.choice(["US", "UK", "CA", "AU", "DE"]),
                "device": np.random.choice(["mobile", "desktop", "tablet"]),
                "keyword_id": f"kw_{np.random.randint(1000, 9999)}",
                "match_type": np.random.choice(["exact", "phrase", "broad"]),
                "conversion_lag_days": np.random.randint(0, 30),
                "clicks_before_conversion": np.random.randint(1, 10),
                "impressions_before_conversion": np.random.randint(1, 50),
                "first_conversion_value": np.random.uniform(10, 500),
                "user_recency_days": np.random.randint(0, 365),
                "user_frequency": np.random.randint(1, 20),
                "average_time_on_site": np.random.uniform(30, 600),
                "pages_per_session": np.random.uniform(1, 10),
                "customer_ltv": np.random.uniform(50, 2000),
            }

            conversions.append(conv)

        return conversions

    def _generate_mock_campaign_data(self, days: int) -> List[Dict[str, Any]]:
        """Generate mock campaign performance data for development/testing"""
        np.random.seed(42)  # For reproducibility

        campaigns = []

        for i in range(1, 6):
            campaign = {
                "campaign_id": f"campaign_{i}",
                "campaign_name": f"Test Campaign {i}",
                "budget": np.random.uniform(1000, 10000),
                "cost": np.random.uniform(500, 9000),
                "clicks": np.random.randint(100, 5000),
                "impressions": np.random.randint(1000, 100000),
                "conversions": np.random.randint(1, 500),
                "conversion_value": np.random.uniform(500, 50000),
            }

            campaigns.append(campaign)

        return campaigns
