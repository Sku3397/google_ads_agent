# Experimentation Service

The Experimentation Service provides functionality for designing, implementing, and analyzing experiments for Google Ads campaigns, such as A/B tests and multivariate tests.

## Features

- Create experiments with control and treatment groups
- Start, stop, and analyze experiment results
- Retrieve experiment details and list all experiments
- Apply winning variations to original campaigns
- Schedule and manage experiment lifecycles

## Usage

```python
from services.experimentation_service import ExperimentationService

# Initialize the service
experimentation_service = ExperimentationService(
    ads_api=ads_api,
    optimizer=optimizer,
    config=config,
    logger=logger
)

# Create an experiment
experiment_id = experimentation_service.create_experiment(
    name="Bid Strategy Test",
    type="A/B Test",
    hypothesis="Increasing bids by 20% will improve ROAS",
    control_group="Campaign A",
    treatment_groups=["Campaign A (Test)"],
    metrics=["clicks", "conversions", "cost", "conversion_value"],
    duration_days=14,
    traffic_split={"control": 0.5, "treatment": 0.5}
)

# Start the experiment
experimentation_service.start_experiment(experiment_id)

# After the experiment completes
results = experimentation_service.analyze_experiment(experiment_id)

# Apply winning variation
experimentation_service.apply_winning_variation(experiment_id)
```

## Integration

The Experimentation Service integrates with the following components:

- Google Ads API for creating and managing campaign experiments
- Optimizer for optimizing experiment parameters
- Analytics Service for analyzing experiment results
- Campaign Management Service for applying winning variations

## Data Storage

Experiment data is stored in `data/experiments.json` and includes:
- Experiment configuration
- Status and timeline
- Results and analysis
- Recommendations 