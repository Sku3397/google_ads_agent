# Contextual Signal Service

The Contextual Signal Service enriches ad targeting and optimization with external contextual signals that impact consumer behavior and campaign performance.

## Overview

This service integrates with various external data sources to gather real-time contextual information such as:

- **Weather conditions**: Temperature, precipitation, and extreme weather events that affect consumer behavior
- **News and events**: Current news topics and trending stories relevant to your industry or keywords
- **Industry trends**: Search volume trends and rising topics in your market segment
- **Economic indicators**: Economic data like GDP growth, inflation, unemployment rate by location
- **Social media trends**: Social sentiment and volume for keywords across platforms
- **Seasonality factors**: Current season, upcoming holidays, and industry-specific seasonal patterns

## Key Features

- **Multi-source signal gathering**: Collects data from multiple APIs and sources
- **Intelligent caching**: Caches signals with appropriate expiry times to reduce API calls
- **Signal analysis**: Analyzes signals for keyword-specific relevance and insights
- **Optimization recommendations**: Generates actionable recommendations based on signals
- **Direct optimization application**: Can apply signal-based optimizations directly to campaigns

## Configuration

The service requires API keys for various external data sources. These can be configured in your `.env` file:

```
# Weather API (OpenWeatherMap)
WEATHER_API_KEY=your_openweathermap_api_key

# News API
NEWS_API_KEY=your_newsapi_key

# Trends API (optional)
TRENDS_API_KEY=your_trends_api_key

# Economic Data API (optional)
ECONOMIC_API_KEY=your_economic_api_key

# Social Media API (optional)
SOCIAL_API_KEY=your_social_api_key
```

## Usage Examples

### Get All Contextual Signals

```python
# Initialize the service
contextual_service = ContextualSignalService(ads_api=ads_api, config=config)

# Get all signals for a location, industry, and set of keywords
signals = contextual_service.get_all_signals(
    location="New York",
    industry="Retail",
    keywords=["winter jackets", "snow boots", "holiday gifts"]
)

# Print signal types and counts
for signal_type, signal_list in signals.items():
    print(f"{signal_type}: {len(signal_list)} signals")
```

### Generate Recommendations

```python
# Get recommendations based on signals
recommendations = contextual_service.get_recommendations_from_signals(
    signals=signals,
    keywords=["winter jackets", "snow boots", "holiday gifts"]
)

# Print recommendations
for rec in recommendations:
    print(f"Recommendation: {rec['type']} - {rec['action']}")
    print(f"Confidence: {rec['confidence']}")
    if 'keywords_affected' in rec:
        print(f"Keywords: {', '.join(rec['keywords_affected'])}")
    print()
```

### Apply Optimizations

```python
# Apply signal-based optimizations to a campaign
success, message = contextual_service.apply_signal_based_optimizations(
    campaign_id="1234567890"
)

print(f"Result: {message}")
```

## Integration with Other Services

The Contextual Signal Service integrates well with:

- **BidService**: For applying contextual bid adjustments
- **KeywordService**: For suggesting new keywords based on trends
- **ForecastingService**: To incorporate contextual factors in forecasts
- **ExperimentationService**: To test the impact of contextual optimizations

## Future Enhancements

- Integration with more data sources (Google Trends API, social listening tools)
- More sophisticated NLP for news and content analysis
- Enhanced machine learning models to predict the impact of contextual factors
- Geospatial data analysis for location-specific contextual signals 