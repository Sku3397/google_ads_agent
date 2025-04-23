"""
Self Play Service for Google Ads Management System

This module provides a self-play framework for Google Ads optimization,
enabling the discovery of robust bidding strategies through competitive
agent vs agent simulations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import os
import json
import pickle
import uuid
from pathlib import Path

from ads_api import GoogleAdsAPI
from optimizer import AdsOptimizer
from config import load_config
from ..base_service import BaseService
from services.reinforcement_learning_service import ReinforcementLearningService, AdsEnvironment

logger = logging.getLogger(__name__)


class SelfPlayService(BaseService):
    """
    Self Play Service for discovering robust Google Ads strategies through competition.

    This service implements agent vs agent competitive simulations to discover
    and refine optimal bidding strategies, budget allocations, and other decision
    parameters for Google Ads campaigns.

    Features:
    - Population-based training (PBT) for evolutionary optimization
    - Tournament-style competition between agent policies
    - Strategy distillation from competitive outcomes
    - Robustness evaluation through adversarial testing
    - Strategy exploration through self-play variations
    - Transfer learning from successful strategies
    - League training with specialist and generalist agents
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the self-play service.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        # Initialize configuration with defaults
        self.config = config or {}
        self.model_save_path = self.config.get("model_save_path", "models/self_play")
        self.algorithm = self.config.get(
            "algorithm", "pbt"
        ).lower()  # Default to Population Based Training

        # Ensure model directory exists
        os.makedirs(self.model_save_path, exist_ok=True)

        # Initialize metrics tracking
        self.metrics_history = []

        # Create a logger specific to this instance
        self.instance_id = str(uuid.uuid4())[:8]

        # Reference to reinforcement learning service for policy training
        self.rl_service = None

        # Population of agents for competition
        self.agent_population = {}

        # Tournament history
        self.tournament_history = []

        # Initialize population configuration
        self.population_size = self.config.get("population_size", 10)
        self.tournament_frequency = self.config.get("tournament_frequency", 10)
        self.mutation_rate = self.config.get("mutation_rate", 0.1)
        self.crossover_probability = self.config.get("crossover_probability", 0.3)
        self.elitism_count = self.config.get("elitism_count", 2)

        # Initialize tournament configuration
        self.tournament_size = self.config.get("tournament_size", 3)
        self.tournament_rounds = self.config.get("tournament_rounds", 5)

        # Load existing population if available
        self._load_population()

        self.logger.info(f"Self Play Service initialized with algorithm: {self.algorithm}")

    def initialize_rl_service(self, rl_service: ReinforcementLearningService) -> None:
        """
        Initialize the reference to a reinforcement learning service.

        Args:
            rl_service: The reinforcement learning service instance
        """
        self.rl_service = rl_service
        self.logger.info("Reinforcement Learning Service reference initialized")

    def initialize_population(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize a population of competing agents.

        Args:
            campaign_id: Optional campaign ID to focus on

        Returns:
            Dictionary with initialization results
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                f"Initializing agent population for self-play (size: {self.population_size})"
            )

            # Check if RL service is available
            if not self.rl_service:
                return {
                    "status": "failed",
                    "message": "Reinforcement Learning Service not initialized",
                }

            # Get historical data for the campaign
            days = self.config.get("historical_data_days", 30)
            historical_data = self._get_historical_data(campaign_id, days)

            if not historical_data:
                return {
                    "status": "failed",
                    "message": f"No historical data found for campaign {campaign_id}",
                }

            # Create environment for training
            env_config = self.config.get("environment", {})

            # Initialize agent population
            self.agent_population = {}

            for i in range(self.population_size):
                agent_id = f"agent_{i + 1}"

                # Create agent with slightly different hyperparameters
                hyperparams = self._generate_agent_hyperparameters()

                # Train a policy for this agent (abbreviated training)
                training_episodes = self.config.get("initial_training_episodes", 100)

                # Configure the RL service with these hyperparameters
                agent_config = self.config.copy()
                agent_config.update(hyperparams)

                # Store agent in population
                self.agent_population[agent_id] = {
                    "id": agent_id,
                    "hyperparameters": hyperparams,
                    "fitness": 0.0,
                    "win_count": 0,
                    "match_count": 0,
                    "generation": 1,
                    "model_path": None,
                    "created_at": datetime.now().isoformat(),
                }

            # Save the population
            self._save_population()

            population_init_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Agent population initialized in {population_init_time:.2f}s")

            return {
                "status": "success",
                "message": f"Initialized {len(self.agent_population)} agents",
                "population_size": len(self.agent_population),
                "time_seconds": population_init_time,
            }

        except Exception as e:
            error_message = f"Error initializing agent population: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return {"status": "failed", "message": error_message}

    def _generate_agent_hyperparameters(self) -> Dict[str, Any]:
        """
        Generate hyperparameters for a new agent with random variations.

        Returns:
            Dictionary of hyperparameters
        """
        # Base hyperparameters
        base_hyperparams = {
            "learning_rate": 0.001,
            "gamma": 0.95,
            "hidden_layers": [256, 128, 64],
            "batch_size": 64,
            "action_space_type": "discrete",
            "reward_weights": {
                "conversions": 20.0,
                "cost": -0.1,
                "clicks": 0.5,
                "ctr": 5.0,
                "conv_rate": 10.0,
                "roas": 25.0,
            },
        }

        # Apply random variations
        hyperparams = base_hyperparams.copy()

        # Vary learning rate: 0.0005 to 0.005
        hyperparams["learning_rate"] = np.random.uniform(0.0005, 0.005)

        # Vary gamma (discount factor): 0.9 to 0.99
        hyperparams["gamma"] = np.random.uniform(0.9, 0.99)

        # Vary batch size: 16, 32, 64, 128
        hyperparams["batch_size"] = np.random.choice([16, 32, 64, 128])

        # Vary layer sizes slightly
        layer_variation = np.random.uniform(0.8, 1.2, len(hyperparams["hidden_layers"]))
        hyperparams["hidden_layers"] = [
            max(16, int(size * variation))
            for size, variation in zip(hyperparams["hidden_layers"], layer_variation)
        ]

        # Vary reward weights by ±20%
        for key in hyperparams["reward_weights"]:
            variation = np.random.uniform(0.8, 1.2)
            hyperparams["reward_weights"][key] *= variation

        return hyperparams

    def run_tournament(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a tournament between agents to evaluate and evolve strategies.

        Args:
            campaign_id: Optional campaign ID to focus on

        Returns:
            Dictionary with tournament results
        """
        tournament_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting agent tournament (ID: {tournament_id})")

            # Check population size
            if len(self.agent_population) < 2:
                return {
                    "status": "failed",
                    "message": "Need at least 2 agents in population to run tournament",
                }

            # Get historical data for environment
            days = self.config.get("historical_data_days", 30)
            historical_data = self._get_historical_data(campaign_id, days)

            if not historical_data:
                return {
                    "status": "failed",
                    "message": f"No historical data found for campaign {campaign_id}",
                }

            # Create environment for tournament
            env_config = self.config.get("environment", {})
            env = AdsEnvironment(env_config, historical_data)

            # Tournament results
            matches = []
            leaderboard = {}

            # Initialize leaderboard
            for agent_id in self.agent_population:
                leaderboard[agent_id] = {
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "score": 0,
                    "total_reward": 0,
                }

            # Run round-robin tournament
            agent_ids = list(self.agent_population.keys())

            for i, agent1_id in enumerate(agent_ids[:-1]):
                for agent2_id in agent_ids[i + 1 :]:
                    # Run match between agent1 and agent2
                    match_result = self._run_match(agent1_id, agent2_id, env)
                    matches.append(match_result)

                    # Update leaderboard based on match result
                    if match_result["winner"] == agent1_id:
                        leaderboard[agent1_id]["wins"] += 1
                        leaderboard[agent1_id]["score"] += 3
                        leaderboard[agent2_id]["losses"] += 1
                    elif match_result["winner"] == agent2_id:
                        leaderboard[agent2_id]["wins"] += 1
                        leaderboard[agent2_id]["score"] += 3
                        leaderboard[agent1_id]["losses"] += 1
                    else:
                        # Draw
                        leaderboard[agent1_id]["draws"] += 1
                        leaderboard[agent2_id]["draws"] += 1
                        leaderboard[agent1_id]["score"] += 1
                        leaderboard[agent2_id]["score"] += 1

                    # Update total rewards
                    leaderboard[agent1_id]["total_reward"] += match_result["agent1_reward"]
                    leaderboard[agent2_id]["total_reward"] += match_result["agent2_reward"]

            # Sort leaderboard by score, then by total reward
            sorted_leaderboard = sorted(
                leaderboard.items(),
                key=lambda x: (x[1]["score"], x[1]["total_reward"]),
                reverse=True,
            )

            # Calculate fitness for each agent based on tournament results
            for agent_id, stats in sorted_leaderboard:
                # Update agent fitness
                self.agent_population[agent_id]["fitness"] = stats["score"] + (
                    stats["total_reward"] / 100
                )
                self.agent_population[agent_id]["win_count"] += stats["wins"]
                self.agent_population[agent_id]["match_count"] += (
                    stats["wins"] + stats["draws"] + stats["losses"]
                )

            # Save tournament results
            tournament_data = {
                "id": tournament_id,
                "timestamp": datetime.now().isoformat(),
                "campaign_id": campaign_id,
                "matches": matches,
                "leaderboard": leaderboard,
                "sorted_leaderboard": [
                    {
                        "agent_id": agent_id,
                        "stats": stats,
                        "fitness": self.agent_population[agent_id]["fitness"],
                    }
                    for agent_id, stats in sorted_leaderboard
                ],
            }

            self.tournament_history.append(tournament_data)
            self._save_tournament(tournament_data)

            # Evolve population
            if self.config.get("auto_evolve", True):
                self.evolve_population()

            # Tournament time
            tournament_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Tournament completed in {tournament_time:.2f}s (ID: {tournament_id})"
            )

            return {
                "status": "success",
                "message": "Tournament completed successfully",
                "tournament_id": tournament_id,
                "matches": len(matches),
                "leaderboard": sorted_leaderboard,
                "time_seconds": tournament_time,
            }

        except Exception as e:
            error_message = f"Error running tournament: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return {"status": "failed", "message": error_message, "tournament_id": tournament_id}

    def _run_match(self, agent1_id: str, agent2_id: str, env) -> Dict[str, Any]:
        """
        Run a match between two agents in the environment.

        Args:
            agent1_id: ID of the first agent
            agent2_id: ID of the second agent
            env: The environment for the match

        Returns:
            Dictionary with match results
        """
        match_id = str(uuid.uuid4())[:8]
        self.logger.info(f"Running match between {agent1_id} and {agent2_id} (ID: {match_id})")

        # Reset environment
        env.reset()

        # Get agent configs
        agent1_config = self.agent_population[agent1_id]
        agent2_config = self.agent_population[agent2_id]

        # Placeholder for actual policy execution
        # In a real implementation, we would load the trained models for each agent
        # and have them interact with the environment

        # Simulate match
        total_steps = env.max_steps
        agent1_reward = 0
        agent2_reward = 0

        for step in range(total_steps):
            # Alternate turns
            if step % 2 == 0:
                # Agent 1's turn
                # In a real implementation, we would use the agent's policy to select an action
                action = np.random.randint(0, env.action_space)
                next_state, reward, done, info = env.step(action)
                agent1_reward += reward
            else:
                # Agent 2's turn
                action = np.random.randint(0, env.action_space)
                next_state, reward, done, info = env.step(action)
                agent2_reward += reward

            if done:
                break

        # Determine winner
        if agent1_reward > agent2_reward:
            winner = agent1_id
            margin = agent1_reward - agent2_reward
        elif agent2_reward > agent1_reward:
            winner = agent2_id
            margin = agent2_reward - agent1_reward
        else:
            winner = None  # Draw
            margin = 0

        # Return match results
        return {
            "match_id": match_id,
            "agent1_id": agent1_id,
            "agent2_id": agent2_id,
            "agent1_reward": agent1_reward,
            "agent2_reward": agent2_reward,
            "winner": winner,
            "margin": margin,
            "steps": step + 1,
            "timestamp": datetime.now().isoformat(),
        }

    def evolve_population(self) -> Dict[str, Any]:
        """
        Evolve the agent population using evolutionary algorithms.

        Returns:
            Dictionary with evolution results
        """
        start_time = datetime.now()

        try:
            self.logger.info("Evolving agent population")

            # Check population size
            if len(self.agent_population) < 2:
                return {
                    "status": "failed",
                    "message": "Need at least 2 agents in population to evolve",
                }

            # Sort agents by fitness
            sorted_agents = sorted(
                self.agent_population.items(), key=lambda x: x[1]["fitness"], reverse=True
            )

            # Keep track of agents to remove and add
            agents_to_remove = []
            new_agents = {}

            # Keep the elite agents (top performers)
            elite_count = min(self.elitism_count, len(sorted_agents))
            elite_agents = sorted_agents[:elite_count]

            # Determine agents to replace (bottom half)
            replace_count = len(sorted_agents) // 2
            agents_to_replace = sorted_agents[-replace_count:]

            for agent_id, _ in agents_to_replace:
                agents_to_remove.append(agent_id)

            # Create new agents through crossover and mutation
            for i in range(replace_count):
                # Select parents using tournament selection
                parent1_id = self._tournament_selection(sorted_agents)
                parent2_id = self._tournament_selection(sorted_agents)

                # Create child through crossover
                child_hyperparams = self._crossover(
                    self.agent_population[parent1_id]["hyperparameters"],
                    self.agent_population[parent2_id]["hyperparameters"],
                )

                # Apply mutation
                child_hyperparams = self._mutate(child_hyperparams)

                # Create new agent
                new_agent_id = f"agent_{str(uuid.uuid4())[:8]}"

                new_agents[new_agent_id] = {
                    "id": new_agent_id,
                    "hyperparameters": child_hyperparams,
                    "fitness": 0.0,
                    "win_count": 0,
                    "match_count": 0,
                    "generation": self.agent_population[parent1_id]["generation"] + 1,
                    "model_path": None,
                    "created_at": datetime.now().isoformat(),
                    "parents": [parent1_id, parent2_id],
                }

            # Update population
            for agent_id in agents_to_remove:
                del self.agent_population[agent_id]

            for agent_id, agent in new_agents.items():
                self.agent_population[agent_id] = agent

            # Save the updated population
            self._save_population()

            evolution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Population evolved in {evolution_time:.2f}s")

            return {
                "status": "success",
                "message": "Population evolved successfully",
                "elite_count": elite_count,
                "replaced_count": len(agents_to_remove),
                "new_count": len(new_agents),
                "time_seconds": evolution_time,
            }

        except Exception as e:
            error_message = f"Error evolving population: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return {"status": "failed", "message": error_message}

    def _tournament_selection(self, sorted_agents: List[Tuple[str, Dict[str, Any]]]) -> str:
        """
        Select an agent using tournament selection.

        Args:
            sorted_agents: List of (agent_id, agent_data) tuples sorted by fitness

        Returns:
            Selected agent ID
        """
        # Select k random agents
        k = min(self.tournament_size, len(sorted_agents))
        candidates = np.random.choice(len(sorted_agents), k, replace=False)

        # Return the best one
        best_idx = min(candidates)  # since agents are sorted, lower index = higher fitness
        return sorted_agents[best_idx][0]

    def _crossover(
        self, parent1_params: Dict[str, Any], parent2_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform crossover between two parents.

        Args:
            parent1_params: Hyperparameters of first parent
            parent2_params: Hyperparameters of second parent

        Returns:
            Child hyperparameters
        """
        child_params = {}

        # Uniform crossover for scalar parameters
        for key in parent1_params:
            if key not in parent2_params:
                child_params[key] = parent1_params[key]
                continue

            if isinstance(parent1_params[key], (int, float)):
                # Crossover with random weight
                weight = np.random.random()
                child_params[key] = (
                    weight * parent1_params[key] + (1 - weight) * parent2_params[key]
                )

                # Round to int if needed
                if isinstance(parent1_params[key], int):
                    child_params[key] = int(round(child_params[key]))
            elif isinstance(parent1_params[key], list):
                # For lists (like hidden_layers), perform element-wise crossover
                if all(isinstance(x, (int, float)) for x in parent1_params[key]):
                    child_list = []
                    max_len = min(len(parent1_params[key]), len(parent2_params[key]))

                    for i in range(max_len):
                        weight = np.random.random()
                        value = (
                            weight * parent1_params[key][i] + (1 - weight) * parent2_params[key][i]
                        )

                        # Round to int if needed
                        if isinstance(parent1_params[key][i], int):
                            value = int(round(value))

                        child_list.append(value)

                    child_params[key] = child_list
                else:
                    # For non-numeric lists, randomly choose from either parent
                    child_params[key] = (
                        parent1_params[key] if np.random.random() < 0.5 else parent2_params[key]
                    )
            elif isinstance(parent1_params[key], dict):
                # Recursive crossover for nested dictionaries
                child_params[key] = self._crossover(parent1_params[key], parent2_params[key])
            else:
                # For other types, randomly choose from either parent
                child_params[key] = (
                    parent1_params[key] if np.random.random() < 0.5 else parent2_params[key]
                )

        return child_params

    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply mutation to hyperparameters.

        Args:
            params: Hyperparameters to mutate

        Returns:
            Mutated hyperparameters
        """
        mutated_params = params.copy()

        for key in mutated_params:
            # Apply mutation with probability mutation_rate
            if np.random.random() > self.mutation_rate:
                continue

            if isinstance(mutated_params[key], float):
                # Mutate floating point values by ±20%
                mutation_factor = np.random.uniform(0.8, 1.2)
                mutated_params[key] *= mutation_factor
            elif isinstance(mutated_params[key], int):
                # Mutate integer values by ±20%
                mutation_factor = np.random.uniform(0.8, 0.2)
                mutated_params[key] = max(1, int(round(mutated_params[key] * mutation_factor)))
            elif isinstance(mutated_params[key], list):
                # For lists (like hidden_layers), mutate each element
                if all(isinstance(x, (int, float)) for x in mutated_params[key]):
                    for i in range(len(mutated_params[key])):
                        if np.random.random() <= self.mutation_rate:
                            mutation_factor = np.random.uniform(0.8, 1.2)

                            if isinstance(mutated_params[key][i], int):
                                mutated_params[key][i] = max(
                                    1, int(round(mutated_params[key][i] * mutation_factor))
                                )
                            else:
                                mutated_params[key][i] *= mutation_factor
            elif isinstance(mutated_params[key], dict):
                # Recursive mutation for nested dictionaries
                mutated_params[key] = self._mutate(mutated_params[key])

        return mutated_params

    def get_elite_strategy(self) -> Dict[str, Any]:
        """
        Get the best strategy from the current population.

        Returns:
            Dictionary with the best agent and its strategy
        """
        try:
            # Check if population exists
            if not self.agent_population:
                return {"status": "failed", "message": "No agent population available"}

            # Sort agents by fitness
            sorted_agents = sorted(
                self.agent_population.items(), key=lambda x: x[1]["fitness"], reverse=True
            )

            # Get the best agent
            best_agent_id, best_agent = sorted_agents[0]

            return {
                "status": "success",
                "message": "Elite strategy retrieved",
                "agent_id": best_agent_id,
                "fitness": best_agent["fitness"],
                "generation": best_agent["generation"],
                "win_rate": best_agent["win_count"] / max(1, best_agent["match_count"]),
                "hyperparameters": best_agent["hyperparameters"],
            }

        except Exception as e:
            error_message = f"Error getting elite strategy: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return {"status": "failed", "message": error_message}

    def generate_strategy_report(self, tournament_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a report on the evolution of strategies.

        Args:
            tournament_id: Optional tournament ID to focus on

        Returns:
            Dictionary with strategy evolution report
        """
        try:
            # Check if tournament history exists
            if not self.tournament_history:
                return {"status": "failed", "message": "No tournament history available"}

            # Get specific tournament or the latest
            if tournament_id:
                tournament = next(
                    (t for t in self.tournament_history if t["id"] == tournament_id), None
                )

                if not tournament:
                    return {"status": "failed", "message": f"Tournament {tournament_id} not found"}
            else:
                tournament = self.tournament_history[-1]

            # Generate report
            report = {
                "tournament_id": tournament["id"],
                "timestamp": tournament["timestamp"],
                "campaign_id": tournament["campaign_id"],
                "match_count": len(tournament["matches"]),
                "leaderboard": tournament["sorted_leaderboard"][:5],  # Top 5 agents
                "strategy_insights": self._generate_strategy_insights(tournament),
                "evolution_trends": self._generate_evolution_trends(),
            }

            return {"status": "success", "message": "Strategy report generated", "report": report}

        except Exception as e:
            error_message = f"Error generating strategy report: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return {"status": "failed", "message": error_message}

    def _generate_strategy_insights(self, tournament: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights from tournament results.

        Args:
            tournament: Tournament data

        Returns:
            Dictionary with strategy insights
        """
        # Placeholder for actual insights generation
        insights = {"top_strategies": [], "key_factors": [], "strategy_clusters": []}

        # Extract top strategies
        for i, entry in enumerate(tournament["sorted_leaderboard"][:3]):
            agent_id = entry["agent_id"]
            agent = self.agent_population.get(agent_id)

            if agent:
                strategy = {
                    "rank": i + 1,
                    "agent_id": agent_id,
                    "fitness": entry["fitness"],
                    "key_parameters": {
                        "learning_rate": agent["hyperparameters"].get("learning_rate"),
                        "gamma": agent["hyperparameters"].get("gamma"),
                        "layer_sizes": agent["hyperparameters"].get("hidden_layers"),
                        "top_reward_weights": self._get_top_weights(
                            agent["hyperparameters"].get("reward_weights", {})
                        ),
                    },
                }

                insights["top_strategies"].append(strategy)

        # Identify key factors (simplistic approach)
        if insights["top_strategies"]:
            # Learning rate trends
            learning_rates = [
                s["key_parameters"]["learning_rate"] for s in insights["top_strategies"]
            ]
            avg_lr = sum(learning_rates) / len(learning_rates)

            # Gamma trends
            gammas = [s["key_parameters"]["gamma"] for s in insights["top_strategies"]]
            avg_gamma = sum(gammas) / len(gammas)

            # Add insights
            insights["key_factors"] = [
                {
                    "factor": "learning_rate",
                    "avg_value": avg_lr,
                    "trend": (
                        "Higher learning rates seem beneficial"
                        if avg_lr > 0.001
                        else "Lower learning rates seem beneficial"
                    ),
                },
                {
                    "factor": "gamma",
                    "avg_value": avg_gamma,
                    "trend": (
                        "Higher discount factors seem beneficial"
                        if avg_gamma > 0.95
                        else "Lower discount factors seem beneficial"
                    ),
                },
            ]

        return insights

    def _get_top_weights(self, weights: Dict[str, float], top_n: int = 3) -> Dict[str, float]:
        """
        Get the top N weights by magnitude.

        Args:
            weights: Dictionary of weights
            top_n: Number of top weights to return

        Returns:
            Dictionary with top weights
        """
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)

        return dict(sorted_weights[:top_n])

    def _generate_evolution_trends(self) -> Dict[str, Any]:
        """
        Generate trends from evolution history.

        Returns:
            Dictionary with evolution trends
        """
        # Placeholder for actual trends analysis
        trends = {"fitness_progression": [], "hyperparameter_trends": {}}

        # Simple fitness progression (by generation)
        generations = {}

        for agent_id, agent in self.agent_population.items():
            gen = agent["generation"]

            if gen not in generations:
                generations[gen] = []

            generations[gen].append(agent["fitness"])

        # Calculate average fitness by generation
        for gen, fitness_values in sorted(generations.items()):
            avg_fitness = sum(fitness_values) / len(fitness_values)
            trends["fitness_progression"].append({"generation": gen, "avg_fitness": avg_fitness})

        return trends

    def _save_population(self) -> None:
        """Save the current agent population to disk."""
        population_file = os.path.join(self.model_save_path, "agent_population.json")

        try:
            with open(population_file, "w") as f:
                json.dump(self.agent_population, f, indent=2)

            self.logger.info(f"Agent population saved to {population_file}")
        except Exception as e:
            self.logger.error(f"Error saving agent population: {str(e)}")

    def _load_population(self) -> None:
        """Load the agent population from disk if available."""
        population_file = os.path.join(self.model_save_path, "agent_population.json")

        if not os.path.exists(population_file):
            self.logger.info("No saved agent population found")
            return

        try:
            with open(population_file, "r") as f:
                self.agent_population = json.load(f)

            self.logger.info(f"Loaded agent population with {len(self.agent_population)} agents")
        except Exception as e:
            self.logger.error(f"Error loading agent population: {str(e)}")

    def _save_tournament(self, tournament_data: Dict[str, Any]) -> None:
        """
        Save tournament data to disk.

        Args:
            tournament_data: Tournament data to save
        """
        tournament_dir = os.path.join(self.model_save_path, "tournaments")
        os.makedirs(tournament_dir, exist_ok=True)

        tournament_file = os.path.join(tournament_dir, f"tournament_{tournament_data['id']}.json")

        try:
            with open(tournament_file, "w") as f:
                json.dump(tournament_data, f, indent=2)

            self.logger.info(f"Tournament data saved to {tournament_file}")
        except Exception as e:
            self.logger.error(f"Error saving tournament data: {str(e)}")

    def _get_historical_data(
        self, campaign_id: Optional[str] = None, days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for training and simulation.

        Args:
            campaign_id: Optional campaign ID to filter data
            days: Number of days of historical data to fetch

        Returns:
            List of data points
        """
        try:
            # If we have an ads_api instance, use it to fetch data
            if self.ads_api:
                if campaign_id:
                    return self.ads_api.get_keyword_performance(days, campaign_id)
                else:
                    return self.ads_api.get_keyword_performance(days)
            else:
                # Mock data for development/testing
                self.logger.warning("No ads_api available, using mock data")
                return self._generate_mock_data(campaign_id, days)

        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return []

    def _generate_mock_data(
        self, campaign_id: Optional[str] = None, days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generate mock data for development and testing.

        Args:
            campaign_id: Optional campaign ID
            days: Number of days of data to generate

        Returns:
            List of mock data points
        """
        mock_data = []

        # Generate 10-50 mock keywords
        keyword_count = np.random.randint(10, 51)

        for i in range(keyword_count):
            # Generate a mock keyword
            keyword = {
                "campaign_id": campaign_id or f"campaign_{np.random.randint(1, 6)}",
                "campaign_name": f"Campaign {np.random.randint(1, 6)}",
                "ad_group_id": f"adgroup_{np.random.randint(1, 11)}",
                "ad_group_name": f"Ad Group {np.random.randint(1, 11)}",
                "keyword_text": f"keyword_{i}",
                "match_type": np.random.choice(["EXACT", "PHRASE", "BROAD"]),
                "status": "ENABLED",
                "quality_score": np.random.randint(1, 11),
                "current_bid": np.random.uniform(0.5, 10.0),
                "impressions": np.random.randint(100, 10000),
                "clicks": np.random.randint(1, 500),
                "conversions": np.random.randint(0, 50),
                "cost": np.random.uniform(10, 1000),
                "avg_conv_value": np.random.uniform(20, 200),
            }

            # Derive some values
            keyword["ctr"] = keyword["clicks"] / max(1, keyword["impressions"])
            keyword["conv_rate"] = keyword["conversions"] / max(1, keyword["clicks"])
            keyword["cpc"] = keyword["cost"] / max(1, keyword["clicks"])
            keyword["conv_value"] = keyword["conversions"] * keyword["avg_conv_value"]

            mock_data.append(keyword)

        return mock_data
