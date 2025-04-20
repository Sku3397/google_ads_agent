# Self Play Service

The Self Play Service is an advanced component of the Google Ads Autonomous Management System that implements agent vs agent competitive simulations to discover robust and optimal bidding strategies through self-play techniques.

## Overview

The Self Play Service implements a competitive framework where multiple bidding strategies compete against each other in simulated environments. By using techniques inspired by competitive AI research, this service can discover strategies that are more robust to market changes and competitor behavior than those found through traditional optimization approaches.

## Key Features

- **Population-based Training (PBT)**: Maintains a diverse population of competing bidding strategies that evolve over time
- **Tournament-style Competition**: Pits strategies against each other in round-robin tournaments to determine the most effective approaches
- **Evolutionary Optimization**: Applies selection, crossover, and mutation to evolve the population, keeping the best strategies and combining their attributes
- **Strategy Distillation**: Extracts insights and patterns from successful strategies for broader application
- **Robustness Evaluation**: Tests strategies against various market conditions to ensure they're robust to changes
- **Self-Play Variations**: Implements various self-play techniques for effective learning
- **Transfer Learning**: Applies successful strategies from one campaign context to related campaigns

## How It Works

1. **Population Initialization**: The service creates a population of agents with different hyperparameters for their bidding strategies
2. **Environment Simulation**: The Google Ads auction environment is simulated based on historical data
3. **Tournament Competition**: Agents compete in tournaments within the simulated environment
4. **Fitness Evaluation**: Each agent's performance is evaluated and assigned a fitness score
5. **Evolutionary Selection**: The best-performing agents are selected to reproduce
6. **Crossover & Mutation**: New agents are created by combining and mutating the hyperparameters of successful agents
7. **Strategy Extraction**: The best strategies are extracted and can be deployed in real campaigns
8. **Continuous Improvement**: The process continues iteratively, with strategies becoming increasingly sophisticated

## Integration with Other Services

The Self Play Service integrates with several other services in the system:

- **Reinforcement Learning Service**: Uses the RL environments and models for agent policies
- **Simulation Service**: Leverages simulation capabilities for realistic environment modeling
- **Meta Learning Service**: Shares insights on successful strategies
- **Scheduler Service**: Runs tournaments and population evolution on schedules

## Usage

### Initialize Agent Population

Initialize a population of competing agents with different hyperparameters:

```python
result = agent.initialize_self_play_population(campaign_id="123456789")
```

### Run Tournament

Run a tournament between competing agents to evaluate their performance:

```python
result = agent.run_self_play_tournament(campaign_id="123456789")
```

### Evolve Population

Evolve the agent population based on tournament results:

```python
result = agent.evolve_self_play_population()
```

### Get Elite Strategy

Retrieve the best strategy from the current population:

```python
result = agent.get_elite_strategy()
```

### Generate Strategy Report

Generate a report on strategy evolution and insights:

```python
result = agent.generate_self_play_strategy_report()
```

## Configuration Options

The Self Play Service offers several configuration options that can be set in the `.env` file or passed directly to the service constructor:

```
# Self Play Service Configuration
SELF_PLAY_POPULATION_SIZE=10         # Number of agents in the population
SELF_PLAY_TOURNAMENT_SIZE=3          # Number of agents in tournament selection
SELF_PLAY_TOURNAMENT_ROUNDS=5        # Number of rounds in each tournament
SELF_PLAY_ELITISM_COUNT=2            # Number of top agents to preserve unchanged
SELF_PLAY_MUTATION_RATE=0.1          # Probability of mutation for each hyperparameter
SELF_PLAY_CROSSOVER_PROBABILITY=0.3  # Probability of crossover between parents
```

## Algorithms

The Self Play Service implements several algorithms:

### Population-Based Training (PBT)

PBT combines hyperparameter optimization with model training, allowing the service to efficiently explore hyperparameter space while training. This is particularly effective for discovering bidding strategies that adapt to changing market conditions.

### Tournament Selection

Tournament selection is used to select parents for reproduction. This method randomly selects k agents from the population and chooses the best one. This balances exploration and exploitation by giving less fit agents a chance to reproduce while still favoring better-performing agents.

### Competitive Self-Play

Agents compete against each other in simulated Google Ads environments, where their bidding strategies are evaluated. This competitive setup drives the evolution of increasingly sophisticated strategies that are robust against various opponent behaviors.

### Evolutionary Operators

The service uses standard evolutionary operators:
- **Selection**: Identifying the fittest individuals
- **Crossover**: Combining aspects of successful strategies
- **Mutation**: Introducing random variations to explore new possibilities

## Data Structures

### Agent Population

The agent population is stored as a dictionary where each entry contains:
- Agent ID
- Hyperparameters for bidding strategy
- Fitness score
- Win/match statistics
- Generation number
- Model path (if applicable)

### Tournament Results

Tournament results include:
- Match outcomes (winner, rewards, etc.)
- Leaderboard rankings
- Fitness updates
- Performance metrics

## Extending the Service

The Self Play Service can be extended in several ways:

1. **New Algorithms**: Implement additional self-play algorithms like MCTS or AlphaZero-inspired approaches
2. **Enhanced Environments**: Create more sophisticated simulation environments that model specific market dynamics
3. **Multi-objective Optimization**: Extend to optimize for multiple objectives simultaneously
4. **Meta-game Analysis**: Analyze the strategic landscape that emerges from agent competition
5. **Hierarchical Strategies**: Implement strategies that operate at different levels (campaign, ad group, keyword)

## Limitations and Considerations

- **Simulation Accuracy**: The quality of strategies depends on the fidelity of the simulation environment
- **Computational Requirements**: Running tournaments and evolution can be computationally intensive
- **Exploration-Exploitation Balance**: Must balance exploration of new strategies with exploitation of known good ones
- **Overfitting**: Strategies might overfit to the simulation environment rather than generalizing to real-world conditions

## Future Enhancements

- **Neural Network Policies**: Implement neural network policies for more sophisticated bidding strategies
- **Multi-agent Reinforcement Learning**: Extend to true multi-agent RL approaches
- **Adversarial Training**: Implement adversarial training to discover robust strategies
- **Strategy Visualization**: Create visualizations of strategy evolution and competition
- **Real-time Adaptation**: Enable real-time adaptation of strategies based on market conditions 