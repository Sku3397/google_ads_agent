# Reinforcement Learning Service for Google Ads Optimization

A machine learning service that uses reinforcement learning algorithms (PPO and DQN) to optimize Google Ads campaigns through automated bidding and keyword management.

## Components

### Main Service Class
- `ReinforcementLearningService`: Core service implementing RL algorithms
- Custom OpenAI Gym environment for Google Ads
- Policy models for bidding and keyword management

### Scheduler Integration
- `RLSchedulerIntegration`: Manages scheduling of RL tasks
- Automated training, inference, evaluation, and safety checks
- Configurable schedules and safety thresholds
- Performance tracking and violation monitoring

## Architecture

### State Space
- Campaign performance metrics
- Keyword-level statistics
- Historical performance data
- Market competition indicators

### Action Space
- Bidding actions: Continuous bid adjustments (0.5x to 2.0x)
- Keyword actions: Add, Remove, Pause, Enable

### Policy Networks
1. BiddingPolicy
   - Continuous action space
   - Shared feature extractor
   - Policy and value heads
   - PPO-style action evaluation

2. KeywordPolicy
   - Discrete action space
   - Categorical distributions
   - Value estimation for PPO

### Scheduler Components
1. Training Schedule
   - Daily training at 2 AM
   - Minimum sample requirements
   - Automatic model evaluation
   - Best model preservation

2. Inference Schedule
   - Hourly optimization during safe hours (7 AM - 10 PM)
   - Safety-constrained action application
   - Performance tracking

3. Evaluation Pipeline
   - Daily evaluation at 1 AM
   - Multiple evaluation episodes
   - Performance ratio monitoring
   - Automatic safety measures

4. Safety Checks
   - Hourly monitoring at :15
   - Conversion rate and ROAS tracking
   - Cost increase monitoring
   - Violation history maintenance

## Configuration

### Training Configuration
```python
{
    'frequency': 'daily',
    'hour': 2,
    'minute': 0,
    'days_between': 1,
    'min_samples': 1000,
    'evaluation_episodes': 5
}
```

### Inference Configuration
```python
{
    'frequency': 'hourly',
    'minute': 30,
    'safe_hours': range(7, 22)  # 7 AM to 10 PM
}
```

### Safety Configuration
```python
{
    'check_frequency': 'hourly',
    'minute': 15,
    'max_bid_change': 0.5,  # Maximum 50% change
    'max_budget_change': 0.3,  # Maximum 30% change
    'min_performance_ratio': 0.7
}
```

## Usage

### Basic Setup
```python
from services.reinforcement_learning_service import ReinforcementLearningService
from services.reinforcement_learning_service.scheduler_integration import RLSchedulerIntegration

# Initialize the RL service
rl_service = ReinforcementLearningService(config)

# Initialize the scheduler integration
scheduler_integration = RLSchedulerIntegration(
    rl_service=rl_service,
    scheduler=ads_scheduler,
    config=scheduler_config
)

# Schedule all tasks
task_ids = scheduler_integration.schedule_all_tasks()
```

### Custom Configuration
```python
custom_config = {
    'training': {
        'frequency': 'daily',
        'hour': 3,
        'min_samples': 2000
    },
    'safety': {
        'max_bid_change': 0.3,
        'min_performance_ratio': 0.8
    }
}

scheduler_integration = RLSchedulerIntegration(
    rl_service=rl_service,
    scheduler=ads_scheduler,
    config=custom_config
)
```

## Safety Mechanisms

1. Bid Constraints
   - Maximum bid changes (default: 50%)
   - Budget change limits (default: 30%)
   - Historical performance tracking

2. Performance Monitoring
   - Conversion rate tracking
   - ROAS monitoring
   - Cost increase limits
   - Automatic safety measures

3. Safe Hours
   - Restricted operation hours
   - Configurable safe periods
   - Automatic action blocking

4. Recovery Measures
   - Best model loading
   - Exploration reset
   - Constraint tightening
   - Performance history tracking

## Monitoring

### Performance Metrics
- Training rewards
- Evaluation performance
- Safety violations
- Action statistics

### History Access
```python
# Get performance history
performance_history = scheduler_integration.get_performance_history()

# Get safety violations
violations = scheduler_integration.get_safety_violations()
```

## Dependencies
- Python 3.8+
- PyTorch
- Stable Baselines3
- Google Ads API
- NumPy
- Pandas

## Contributing
1. Follow PEP 8 style guide
2. Add unit tests for new features
3. Update documentation
4. Maintain type hints
5. Test safety measures 