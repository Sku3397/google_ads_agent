# Bandit Service

## Overview

The BanditService implements multi-armed bandit algorithms for optimizing Google Ads campaigns, ad creatives, and budget allocation. It leverages statistical learning techniques to balance exploration and exploitation, allowing for dynamic optimization of advertising decisions over time.

## Features

- **Multiple Bandit Algorithms**:
  - **Thompson Sampling**: Bayesian approach for balancing exploration/exploitation
  - **Upper Confidence Bound (UCB)**: Uses confidence intervals to guide exploration
  - **Epsilon-Greedy**: Simple exploration strategy with tunable exploration rate
  - **Dynamic Thompson Sampling**: Adapts to non-stationary environments with time-weighted rewards
  - **Contextual Bandits**: Makes decisions based on contextual features

- **Applications**:
  - Optimizing budget allocation across campaigns
  - Testing and selecting the best ad creatives
  - Keyword bid optimization
  - Landing page testing

- **Analytics**:
  - Performance visualization
  - Simulation capabilities
  - Historical performance tracking
  - Confidence-based recommendations

## Usage

### Initializing Bandits

Create a new bandit for campaign optimization:

```python
from services.bandit_service import BanditService, BanditAlgorithm

# Initialize the service
bandit_service = BanditService(
    client=google_ads_client,
    customer_id="1234567890",
    config={
        "alpha_prior": 1.0,
        "beta_prior": 1.0,
        "epsilon": 0.1,
        "ucb_alpha": 1.0
    }
)

# Initialize a Thompson Sampling bandit
result = bandit_service.initialize_bandit(
    name="campaign_optimizer",
    arms=["campaign1", "campaign2", "campaign3"],
    algorithm=BanditAlgorithm.THOMPSON_SAMPLING,
    metadata={"type": "campaign_budget"}
)

bandit_id = result["bandit_id"]
```

### Updating Bandits with Performance Data

After observing performance, update the bandit:

```python
# Update with conversion data (reward between 0-1)
bandit_service.update_bandit(
    bandit_id=bandit_id,
    arm_id="campaign1",
    reward=0.75  # e.g., conversion rate or normalized return on ad spend
)
```

### Making Decisions

Use the bandit to select the best arm:

```python
# Get recommendation for next allocation
selection = bandit_service.select_arm(bandit_id)

selected_campaign = selection["selected_arm"]
rationale = selection["rationale"]

print(f"Selected campaign: {selected_campaign}")
print(f"Rationale: {rationale}")
```

### Budget Allocation

Allocate a budget based on bandit recommendations:

```python
# Allocate a $1000 budget
allocation = bandit_service.allocate_budget(
    bandit_id=bandit_id,
    total_budget=1000.0
)

for campaign_id, budget in allocation["allocations"].items():
    print(f"Campaign {campaign_id}: ${budget:.2f}")
```

### End-to-End Campaign Optimization

Optimize campaigns in a single call:

```python
# Optimize multiple campaigns with a $5000 budget
result = bandit_service.optimize_campaigns(
    campaign_ids=["campaign1", "campaign2", "campaign3", "campaign4"],
    total_budget=5000.0,
    days=30  # Use last 30 days of data
)

# Apply recommendations
for rec in result["recommendations"]:
    print(f"Campaign: {rec['campaign_name']}")
    print(f"Current Budget: ${rec['current_budget']:.2f}")
    print(f"Recommended Budget: ${rec['recommended_budget']:.2f}")
    print(f"Rationale: {rec['rationale']}")
    print("---")
```

### Ad Creative Testing

Optimize ad creatives:

```python
result = bandit_service.optimize_ad_creatives(
    ad_group_id="123456789",
    creative_ids=["ad1", "ad2", "ad3"],
    days=14
)

best_creative = result["recommended_creative"]
print(f"Best performing creative: {best_creative}")
```

### Analysis and Visualization

Analyze bandits and visualize performance:

```python
# Get detailed statistics
stats = bandit_service.get_bandit_stats(bandit_id)

# Visualize performance
viz_result = bandit_service.visualize_bandit(
    bandit_id=bandit_id,
    save_path="reports/bandit_performance.png"
)
```

### Simulation

Simulate bandit performance with known reward distributions:

```python
# Simulate with known true reward probabilities
simulation = bandit_service.simulate_bandit(
    bandit_id=bandit_id,
    true_rewards={
        "campaign1": 0.08,  # 8% conversion rate
        "campaign2": 0.05,  # 5% conversion rate
        "campaign3": 0.03   # 3% conversion rate
    },
    num_trials=10000
)

print(f"Total reward: {simulation['simulation_results']['total_reward']}")
print(f"Regret: {simulation['simulation_results']['total_regret']}")
```

### Persistence

Save and load bandits:

```python
# Save current state
bandit_service.save_bandits("data/bandits/campaign_bandits.json")

# Load previous state
bandit_service.load_bandits("data/bandits/campaign_bandits.json")
```

## Configuration

The BanditService accepts these configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_path` | Path to store bandit data | `"data/bandits"` |
| `alpha_prior` | Prior alpha for Beta distribution | `1.0` |
| `beta_prior` | Prior beta for Beta distribution | `1.0` |
| `epsilon` | Exploration rate for Epsilon-Greedy | `0.1` |
| `ucb_alpha` | Exploration parameter for UCB | `1.0` |
| `discount_factor` | Discount factor for non-stationary rewards | `0.95` |

## Algorithm Selection

Choose algorithms based on your specific needs:

- **Thompson Sampling**: Best general-purpose algorithm with excellent performance
- **UCB**: Good for scenarios when tight confidence bounds are important
- **Epsilon-Greedy**: Simple and effective for stable environments
- **Dynamic Thompson**: Best for changing environments (e.g., seasonal trends)
- **Contextual**: When you have relevant context for each decision

## Requirements

- NumPy
- SciPy
- Matplotlib (optional, for visualization)
- PyMC3 (optional, for advanced Bayesian modeling)
- Google Ads API client

## Performance Considerations

- For campaign optimization, weekly updates are usually sufficient
- For ad creative testing, daily updates provide faster learning
- More data points (pulls) lead to higher confidence recommendations
- Consider using higher exploration rates (epsilon) for new campaigns
- For mature campaigns, lower exploration rates focus on exploitation 