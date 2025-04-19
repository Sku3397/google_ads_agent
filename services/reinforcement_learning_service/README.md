# Reinforcement Learning Service

This service provides reinforcement learning (RL) capabilities for optimizing bidding strategies, budget allocation, and other decision-making tasks in Google Ads campaigns.

## Features

- **Auction Simulator**: Builds a simulator of the ad auction environment based on historical auction insights data
- **Policy Training**: Trains RL policies (DQN/PPO) for optimizing bids across campaigns and ad groups
- **Safe Exploration**: Uses epsilon-greedy strategy for balancing exploration with exploitation
- **Bid Recommendations**: Generates optimized bid recommendations using trained policies
- **Policy Evaluation**: Evaluates policy performance before applying changes to live campaigns

## Usage

### Training a Policy

```python
from services.reinforcement_learning_service import ReinforcementLearningService

# Initialize the service
rl_service = ReinforcementLearningService(
    ads_api=ads_api,
    optimizer=optimizer,
    config=config
)

# Train a policy for a specific campaign
result = rl_service.train_policy(
    campaign_id="123456789",
    training_episodes=1000
)

# Generate recommendations using the trained policy
recommendations = rl_service.generate_bid_recommendations(
    campaign_id="123456789",
    exploration_rate=0.1
)
```

### Command-line Interface

```bash
# Train an RL policy for all campaigns
python ads_agent.py --action train_rl_policy --episodes 2000

# Train an RL policy for a specific campaign
python ads_agent.py --action train_rl_policy --campaign 123456789 --episodes 2000

# Generate bid recommendations using RL
python ads_agent.py --action rl_bid_recommendations --exploration 0.05
```

## Configuration

Example configuration settings in `.env` file:

```
# Reinforcement Learning Settings
RL_AUCTION_SIMULATOR_ENABLED=true
RL_EPSILON=0.1
RL_LEARNING_RATE=0.001
RL_DISCOUNT_FACTOR=0.95
RL_BATCH_SIZE=64
RL_MEMORY_SIZE=10000
RL_TARGET_UPDATE_FREQUENCY=100
RL_MODEL_SAVE_PATH=models/rl
```

## Dependencies

This service requires the following Python packages:
- gym
- torch
- stable-baselines3
- scikit-learn

Make sure these are installed by running:
```bash
pip install -r requirements.txt
``` 