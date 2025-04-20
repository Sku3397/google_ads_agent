# Forecasting Service

## Overview

The Forecasting Service is a crucial component of the Google Ads autonomous management system. It enables the system to predict future performance metrics, search trends, and budget requirements, allowing for proactive campaign optimization rather than reactive adjustments.

This service leverages time series forecasting techniques to analyze historical performance data and generate accurate predictions for key metrics like clicks, impressions, conversions, and costs.

## Key Features

- **Performance Metric Forecasting**: Predicts future values for key metrics (clicks, conversions, etc.)
- **Budget Forecasting**: Projects future budget requirements based on goals or historical patterns
- **Search Trend Detection**: Identifies emerging and declining search trends
- **Demand Forecast Integration**: Integrates with Google Ads Insights API for demand forecasts
- **Multiple Forecasting Models**: Uses ARIMA, Exponential Smoothing, and ensemble methods
- **Model Evaluation**: Automatically selects the best forecasting model for each context

## Usage

### Forecasting Performance Metrics

To forecast key performance metrics:

```python
forecast_results = forecasting_service.forecast_metrics(
    metrics=["clicks", "impressions", "conversions", "cost"],
    days_to_forecast=30,
    campaign_id="123456789",  # Optional - forecasts for all campaigns if omitted
    model_type="auto"  # Can be "auto", "arima", "ets", or "ensemble"
)

# Access forecast data
clicks_forecast = forecast_results["metrics"]["clicks"]["forecast"]
clicks_dates = forecast_results["metrics"]["clicks"]["dates"]
clicks_lower_bound = forecast_results["metrics"]["clicks"]["lower_bound"]
clicks_upper_bound = forecast_results["metrics"]["clicks"]["upper_bound"]
```

### Forecasting Budget Requirements

To forecast budget requirements for a campaign:

```python
budget_forecast = forecasting_service.forecast_budget(
    campaign_id="123456789",
    days_to_forecast=30,
    target_metric="conversions",
    target_value=100  # Optional - if omitted, forecasts based on historical patterns
)

# Access budget forecast data
suggested_daily_budget = budget_forecast["suggested_daily_budget"]
total_forecasted_cost = budget_forecast["total_forecasted_cost"]
cpa = budget_forecast["historical_performance"]["cpa"]
```

### Detecting Search Trends

To identify emerging and declining search trends:

```python
trends = forecasting_service.detect_search_trends(
    days_lookback=90,
    min_growth_rate=0.1
)

# Access trend data
trending_keywords = trends["trending_keywords"]
declining_keywords = trends["declining_keywords"]
stable_keywords = trends["stable_volume_keywords"]
```

### Getting Demand Forecasts

To retrieve demand forecasts from Google Ads Insights API:

```python
demand_forecasts = forecasting_service.get_demand_forecasts()

# Check if demand forecasts are available
if demand_forecasts["status"] == "success":
    forecasts = demand_forecasts["demand_forecasts"]
else:
    message = demand_forecasts["message"]
```

## Integration with Google Ads Insights

The ForecastingService can integrate with Google Ads Insights to retrieve demand forecasts, which help you understand predicted upcoming trends relevant to your business. These forecasts use historical data to predict products and services that may experience increased search interest within the next 180 days.

Demand forecasts can help you:
- Review when demand is likely to start increasing
- Identify new events relevant to your business
- Identify upcoming expansion opportunities
- Review demand trends from the current year
- Compare your performance with competitors

For more information on Google Ads demand forecasts, see the [official documentation](https://support.google.com/google-ads/answer/10787044).

## Technical Implementation

### Time Series Models

The service uses several forecasting approaches:

- **ARIMA (AutoRegressive Integrated Moving Average)**: Good for data with trends but no seasonality
- **Exponential Smoothing**: Effective for data with both trend and seasonality
- **Ensemble Methods**: Combines multiple models for more robust forecasts

### Model Selection

For automatic model selection, the service:
1. Splits historical data into training and validation sets
2. Trains multiple models on the training data
3. Evaluates each model on the validation data
4. Selects the model with the lowest Mean Squared Error (MSE)
5. Retrains the selected model on the full dataset

### Data Requirements

For reliable forecasting, the service requires:
- At least 60 days of historical data (ideally 90+ days)
- Daily granularity for most accurate forecasts
- Valid metrics data (clicks, impressions, conversions, cost)

## Future Enhancements

- Integration with additional forecasting models (Prophet, LSTM, etc.)
- Contextual forecasting incorporating external factors (seasonality, holidays, etc.)
- Competitive forecasting based on auction insights data
- Automatic anomaly detection and correction in historical data
- Cross-campaign influence analysis for more accurate forecasts 