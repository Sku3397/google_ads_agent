# Trend Forecasting Service

The Trend Forecasting Service provides advanced trend forecasting and analysis capabilities for Google Ads campaigns. It goes beyond basic forecasting to identify emerging trends, seasonal patterns, and generate comprehensive trend reports.

## Overview

This service enables sophisticated trend analysis and forecasting with features like:

- **Advanced Time Series Forecasting**: Using Prophet, SARIMA, Auto ARIMA, and ensemble methods
- **Emerging Trend Detection**: Identifying keywords with accelerating growth
- **Seasonal Pattern Analysis**: Discovering daily, weekly, monthly, and seasonal patterns
- **Comprehensive Trend Reports**: Generating visualizations and actionable insights
- **Trending Keyword Discovery**: Finding trending keywords in specific industries and locations

## Key Features

- **Multiple Forecasting Models**:
  - Prophet - for handling multiple seasonalities and holidays
  - SARIMA - for stationary time series with regular patterns
  - Auto ARIMA - automatically selecting optimal parameters
  - Ensemble - combining multiple models for better accuracy

- **Forecast Horizons**:
  - Short-term (7 days)
  - Medium-term (30 days)
  - Long-term (90 days)

- **Contextual Analysis**:
  - Integration with external signals (weather, holidays, etc.)
  - Day-of-week patterns
  - Seasonality decomposition

- **Visualization**:
  - Historical trend charts
  - Seasonal pattern visualizations
  - Forecast projections with confidence intervals

## Dependencies

The service relies on the following Python libraries:
- `pandas` and `numpy` for data manipulation
- `fbprophet` for Prophet forecasting
- `statsmodels` for SARIMA models and seasonal decomposition
- `pmdarima` for Auto ARIMA modeling
- `matplotlib` and `seaborn` for visualization

## Usage Examples

### Forecast Keyword Performance

```python
# Initialize the service
trend_service = TrendForecastingService(ads_api=ads_api, config=config)

# Forecast clicks for a keyword over the medium term
forecast = trend_service.forecast_keyword_performance(
    keyword="winter jackets",
    campaign_id="123456789",
    horizon="medium_term",
    metric="clicks",
    model_type="prophet",
    include_external_signals=True
)

print(f"Forecasted clicks: {forecast.forecasted_value:.2f}")
print(f"Forecast range: [{forecast.lower_bound:.2f} - {forecast.upper_bound:.2f}]")
print(f"Confidence: {forecast.confidence:.2f}")
```

### Detect Emerging Trends

```python
# Detect emerging trends in a campaign
emerging_trends = trend_service.detect_emerging_trends(
    campaign_id="123456789",
    lookback_days=90,
    min_growth_rate=0.2
)

for trend in emerging_trends:
    print(f"Keyword: {trend['keyword']}")
    print(f"Growth rate: {trend['growth_rate']:.2%}")
    print(f"Confidence: {trend['confidence']:.2f}")
    print()
```

### Identify Seasonal Patterns

```python
# Identify seasonal patterns in campaign performance
seasonal_patterns = trend_service.identify_seasonal_patterns(
    campaign_id="123456789",
    lookback_days=365,
    metric="clicks"
)

for pattern in seasonal_patterns:
    print(f"Period: {pattern['period_name']}")
    print(f"Strength: {pattern['strength']:.2f}")
    print(f"Peak day: {pattern['peak_day']}")
    print()
```

### Generate Comprehensive Trend Report

```python
# Generate a comprehensive trend report
trend_report = trend_service.generate_trend_report(
    campaign_id="123456789",
    lookback_days=90,
    forecast_horizon="medium_term"
)

# Report includes:
# - Emerging trends
# - Seasonal patterns
# - Top keyword forecasts
# - Overall trend analysis
# - Visualization paths
```

### Discover Trending Keywords

```python
# Discover trending keywords in a specific industry
trending_keywords = trend_service.discover_trending_keywords(
    industry="retail",
    location="New York",
    limit=20
)

for keyword in trending_keywords:
    print(f"Keyword: {keyword['keyword']}")
    print(f"Trend score: {keyword['trend_score']:.2f}")
    print(f"Growth rate: {keyword['growth_rate']:.2%}")
    print()
```

## Integration with Other Services

The Trend Forecasting Service integrates well with:

- **ContextualSignalService**: For incorporating external data signals
- **BidService**: For trend-based bidding strategies
- **KeywordService**: For suggesting trending keywords
- **ExperimentationService**: For testing forecast-driven strategies

## Future Enhancements

- Deep learning models (LSTM, Transformer) for more complex patterns
- Cross-campaign trend analysis
- Competitor trend analysis
- Custom alerting for emerging trends
- Anomaly detection in trend data 