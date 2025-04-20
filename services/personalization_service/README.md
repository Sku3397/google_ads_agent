# Personalization Service

## Overview

The Personalization Service optimizes ad delivery by tailoring content, bidding strategies, and targeting parameters based on user characteristics and behavior. This service enables Google Ads campaigns to deliver more relevant ads to different user segments, improving engagement and conversion rates.

## Key Features

- **User Segmentation**: Creates data-driven user segments based on behavior patterns and characteristics
- **Personalized Bidding**: Provides segment-specific bid adjustments for campaigns
- **Ad Content Optimization**: Ranks and recommends ads based on their performance with specific segments
- **Ad Customizer Generation**: Creates dynamic ad content tailored to user segments
- **Performance Tracking**: Monitors and evaluates segment performance over time

## How It Works

1. **Data Collection**: The service collects user interaction data from Google Ads campaigns
2. **Segmentation**: Users are clustered into segments based on behavior and characteristics
3. **Model Training**: Performance models are built for each segment
4. **Optimization**: Ad delivery and bidding strategies are personalized for each segment
5. **Continuous Learning**: Models are updated regularly based on new performance data

## Integration

The Personalization Service integrates with:

- **Reinforcement Learning Service**: For advanced bidding strategies
- **Bandit Service**: For exploration and exploitation of ad variations
- **Creative Service**: For content optimization
- **Keyword Service**: For personalized keyword targeting

## Usage

### Creating User Segments

```python
# Example of creating user segments
from services.personalization_service import PersonalizationService

# Initialize the service
personalization_service = PersonalizationService(ads_api=ads_api)

# Create user segments
user_data = ads_api.get_user_interaction_data(days=30)
segments = personalization_service.create_user_segments(user_data)
```

### Getting Personalized Bid Adjustments

```python
# Example of getting personalized bid adjustments
campaign_id = "1234567890"
ad_group_id = "0987654321"
user_segment = "2"  # Segment ID

adjustments = personalization_service.get_personalized_bid_adjustments(
    campaign_id=campaign_id,
    ad_group_id=ad_group_id,
    user_segment=user_segment
)

# Apply adjustments
print(f"Device adjustment: {adjustments['device']}")
print(f"Location adjustment: {adjustments['location']}")
```

### Ranking Ads for a Segment

```python
# Example of ranking ads for a specific segment
ad_group_id = "0987654321"
user_segment = "3"  # Segment ID
available_ads = ads_api.get_ads(ad_group_id)

# Get personalized ad rankings
ranked_ads = personalization_service.get_personalized_ads(
    ad_group_id=ad_group_id,
    user_segment=user_segment,
    available_ads=available_ads
)

# Top performing ad for this segment
top_ad = ranked_ads[0] if ranked_ads else None
```

## Configuration

The service can be configured through the following parameters:

- `segment_count`: Number of user segments to create (default: 5)
- `min_observations`: Minimum number of observations required for segmentation (default: 100)
- `update_frequency_days`: How often to update the segmentation models (default: 7)
- `data_lookback_days`: Number of days of historical data to use (default: 90)

These can be set in the configuration when initializing the service:

```python
config = {
    'segment_count': 8,
    'min_observations': 200,
    'update_frequency_days': 3,
    'data_lookback_days': 60
}

personalization_service = PersonalizationService(
    ads_api=ads_api,
    config=config
)
```

## Requirements

- NumPy
- Pandas
- scikit-learn (for clustering algorithms)

## Data

The service stores the following data:

- `user_segments.json`: Definitions of user segments
- `personalization_models.json`: Trained models for personalization
- `segment_performance.json`: Performance metrics for each segment 