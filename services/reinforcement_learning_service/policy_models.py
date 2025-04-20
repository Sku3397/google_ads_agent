"""
Advanced Policy Models for Reinforcement Learning

This module implements various policy gradient methods including:
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- SAC (Soft Actor-Critic)
"""

import tensorflow as tf
import tensorflow_probability as tfp
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from dataclasses import dataclass
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for policy models"""

    state_dim: int
    action_dim: int
    hidden_layers: List[int] = None
    learning_rate: float = 3e-4
    value_learning_rate: float = 1e-3
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_gae: bool = True
    gae_lambda: float = 0.95
    gamma: float = 0.99
    normalize_advantages: bool = True
    use_critic_ensemble: bool = True
    num_critics: int = 2
    use_layer_norm: bool = True
    activation: str = "tanh"
    initializer: str = "orthogonal"


class PPOPolicy(tf.keras.Model):
    """
    PPO Policy Network with state-of-the-art improvements:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Gradient clipping
    - Orthogonal initialization
    - Layer normalization
    """

    def __init__(self, config: PolicyConfig):
        """Initialize the policy."""
        super().__init__()
        self.config = config

        # Set up layers
        self.hidden_layers = config.hidden_layers or [64, 64]

        # Get activation function
        if config.activation == "tanh":
            self.activation = tf.nn.tanh
        elif config.activation == "relu":
            self.activation = tf.nn.relu
        else:
            self.activation = tf.nn.relu

        # Initialize layers with orthogonal initialization
        initializer = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))

        # Shared feature extractor
        self.shared_layers = []
        for units in self.hidden_layers:
            self.shared_layers.extend(
                [
                    tf.keras.layers.Dense(units, kernel_initializer=initializer, activation=None),
                    (
                        tf.keras.layers.LayerNormalization()
                        if config.use_layer_norm
                        else tf.keras.layers.Lambda(lambda x: x)
                    ),
                    tf.keras.layers.Activation(self.activation),
                ]
            )

        # Policy head (actor)
        self.policy_layers = [
            tf.keras.layers.Dense(64, kernel_initializer=initializer, activation=self.activation),
            tf.keras.layers.Dense(
                config.action_dim,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01),
                name="policy_logits",
            ),
        ]

        # Value head (critic)
        self.value_layers = []
        if config.use_critic_ensemble:
            self.value_ensembles = []
            for _ in range(config.num_critics):
                critic = [
                    tf.keras.layers.Dense(
                        64, kernel_initializer=initializer, activation=self.activation
                    ),
                    tf.keras.layers.Dense(
                        1,
                        kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
                        name=f"value_{_}",
                    ),
                ]
                self.value_ensembles.append(critic)
        else:
            self.value_layers = [
                tf.keras.layers.Dense(
                    64, kernel_initializer=initializer, activation=self.activation
                ),
                tf.keras.layers.Dense(
                    1, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), name="value"
                ),
            ]

        # Build the model
        self.build((None, config.state_dim))

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass through the network."""
        # Shared features
        x = inputs
        for layer in self.shared_layers:
            x = layer(x)

        # Policy head
        policy_out = x
        for layer in self.policy_layers:
            policy_out = layer(policy_out)

        # Value head(s)
        if self.config.use_critic_ensemble:
            values = []
            for critic in self.value_ensembles:
                value_out = x
                for layer in critic:
                    value_out = layer(value_out)
                values.append(value_out)
            value_out = tf.reduce_mean(tf.stack(values, axis=0), axis=0)
        else:
            value_out = x
            for layer in self.value_layers:
                value_out = layer(value_out)

        return policy_out, value_out

    def get_action_distribution(self, logits: tf.Tensor) -> tfp.distributions.Distribution:
        """Get the action distribution from logits."""
        return tfp.distributions.Categorical(logits=logits)

    def evaluate_actions(
        self, states: tf.Tensor, actions: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Evaluate actions and compute log probs and entropy."""
        logits, values = self(states)
        distribution = self.get_action_distribution(logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy, values

    @tf.function
    def train_step(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Training step for PPO."""
        states = data["states"]
        actions = data["actions"]
        advantages = data["advantages"]
        returns = data["returns"]
        old_log_probs = data["old_log_probs"]

        with tf.GradientTape() as tape:
            # Forward pass
            logits, values = self(states)
            distribution = self.get_action_distribution(logits)

            # Calculate log probs and entropy
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy()

            # Compute policy loss with clipping
            ratio = tf.exp(log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
            )

            # PPO policy loss
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )

            # Value loss with clipping
            values_clipped = data["old_values"] + tf.clip_by_value(
                values - data["old_values"], -self.config.clip_ratio, self.config.clip_ratio
            )
            value_loss_1 = tf.square(values - returns)
            value_loss_2 = tf.square(values_clipped - returns)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

            # Entropy bonus
            entropy_loss = -tf.reduce_mean(entropy)

            # Total loss
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                + self.config.entropy_coef * entropy_loss
            )

        # Compute gradients and apply
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)

        # Gradient clipping
        if self.config.max_grad_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Compute approximate KL for early stopping
        approx_kl = 0.5 * tf.reduce_mean(tf.square(old_log_probs - log_probs))

        # Return metrics
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": loss,
            "approx_kl": approx_kl,
            "clip_fraction": tf.reduce_mean(
                tf.cast(tf.greater(tf.abs(ratio - 1.0), self.config.clip_ratio), tf.float32)
            ),
        }


class A2CPolicy(PPOPolicy):
    """
    Advantage Actor-Critic (A2C) Policy Network.
    Inherits from PPO but modifies the loss calculation.
    """

    @tf.function
    def train_step(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Training step for A2C."""
        states = data["states"]
        actions = data["actions"]
        advantages = data["advantages"]
        returns = data["returns"]

        with tf.GradientTape() as tape:
            # Forward pass
            logits, values = self(states)
            distribution = self.get_action_distribution(logits)

            # Calculate log probs and entropy
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy()

            # A2C policy loss (simpler than PPO, no clipping)
            policy_loss = -tf.reduce_mean(log_probs * advantages)

            # Value loss
            value_loss = 0.5 * tf.reduce_mean(tf.square(returns - values))

            # Entropy bonus
            entropy_loss = -tf.reduce_mean(entropy)

            # Total loss
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                + self.config.entropy_coef * entropy_loss
            )

        # Compute and apply gradients with clipping
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)

        if self.config.max_grad_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": loss,
        }


class SACPolicy(tf.keras.Model):
    """
    Soft Actor-Critic (SAC) Policy Network.
    Implements continuous action space with automatic entropy tuning.
    """

    def __init__(self, config: PolicyConfig):
        """Initialize the SAC policy."""
        super().__init__()
        self.config = config

        # Initialize the target entropy (used for automatic entropy tuning)
        self.target_entropy = -np.prod(config.action_dim)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)

        # Build networks
        self.actor = self._build_actor()
        self.critic_1 = self._build_critic()
        self.critic_2 = self._build_critic()
        self.critic_target_1 = self._build_critic()
        self.critic_target_2 = self._build_critic()

        # Copy weights to targets
        self.critic_target_1.set_weights(self.critic_1.get_weights())
        self.critic_target_2.set_weights(self.critic_2.get_weights())

    def _build_actor(self) -> tf.keras.Model:
        """Build the actor network."""
        inputs = tf.keras.Input(shape=(self.config.state_dim,))
        x = inputs

        # Hidden layers
        for units in self.config.hidden_layers:
            x = tf.keras.layers.Dense(
                units, activation="relu", kernel_initializer=tf.keras.initializers.Orthogonal()
            )(x)
            if self.config.use_layer_norm:
                x = tf.keras.layers.LayerNormalization()(x)

        # Output mean and log_std
        mean = tf.keras.layers.Dense(
            self.config.action_dim,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01),
        )(x)

        log_std = tf.keras.layers.Dense(
            self.config.action_dim, kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01)
        )(x)

        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])

    def _build_critic(self) -> tf.keras.Model:
        """Build a critic network."""
        state_inputs = tf.keras.Input(shape=(self.config.state_dim,))
        action_inputs = tf.keras.Input(shape=(self.config.action_dim,))
        x = tf.keras.layers.Concatenate()([state_inputs, action_inputs])

        # Hidden layers
        for units in self.config.hidden_layers:
            x = tf.keras.layers.Dense(
                units, activation="relu", kernel_initializer=tf.keras.initializers.Orthogonal()
            )(x)
            if self.config.use_layer_norm:
                x = tf.keras.layers.LayerNormalization()(x)

        # Q-value output
        q_value = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal())(x)

        return tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=q_value)

    def get_action_distribution(self, state: tf.Tensor) -> tfp.distributions.Distribution:
        """Get the action distribution from the actor."""
        mean, log_std = self.actor(state)
        std = tf.exp(tf.clip_by_value(log_std, -20, 2))
        return tfp.distributions.Normal(mean, std)

    @tf.function
    def train_step(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Training step for SAC."""
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        next_states = data["next_states"]
        dones = data["dones"]

        metrics = {}

        # Update critics
        with tf.GradientTape(persistent=True) as tape:
            # Get next actions and their log probs
            next_dist = self.get_action_distribution(next_states)
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions)

            # Get target Q-values
            target_q1 = self.critic_target_1([next_states, next_actions])
            target_q2 = self.critic_target_2([next_states, next_actions])
            target_q = tf.minimum(target_q1, target_q2)

            # Compute target value with entropy
            target_value = target_q - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.config.gamma * target_value

            # Current Q-values
            current_q1 = self.critic_1([states, actions])
            current_q2 = self.critic_2([states, actions])

            # Compute critic losses
            critic_loss_1 = 0.5 * tf.reduce_mean(tf.square(current_q1 - q_target))
            critic_loss_2 = 0.5 * tf.reduce_mean(tf.square(current_q2 - q_target))

        # Update critics
        critic_1_gradients = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)
        critic_2_gradients = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)

        if self.config.max_grad_norm is not None:
            critic_1_gradients, _ = tf.clip_by_global_norm(
                critic_1_gradients, self.config.max_grad_norm
            )
            critic_2_gradients, _ = tf.clip_by_global_norm(
                critic_2_gradients, self.config.max_grad_norm
            )

        self.critic_1.optimizer.apply_gradients(
            zip(critic_1_gradients, self.critic_1.trainable_variables)
        )
        self.critic_2.optimizer.apply_gradients(
            zip(critic_2_gradients, self.critic_2.trainable_variables)
        )

        metrics.update({"critic_1_loss": critic_loss_1, "critic_2_loss": critic_loss_2})

        # Update actor
        with tf.GradientTape() as tape:
            # Sample actions from current policy
            dist = self.get_action_distribution(states)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            # Compute Q-values for sampled actions
            q1 = self.critic_1([states, actions])
            q2 = self.critic_2([states, actions])
            q = tf.minimum(q1, q2)

            # Actor loss with entropy
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q)

        # Update actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        if self.config.max_grad_norm is not None:
            actor_gradients, _ = tf.clip_by_global_norm(actor_gradients, self.config.max_grad_norm)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        metrics["actor_loss"] = actor_loss

        # Update alpha (automatic entropy tuning)
        with tf.GradientTape() as tape:
            dist = self.get_action_distribution(states)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            alpha_loss = -tf.reduce_mean(self.alpha * (log_probs + self.target_entropy))

        alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
        tf.optimizers.Adam(learning_rate=self.config.learning_rate).apply_gradients(
            zip(alpha_gradients, [self.log_alpha])
        )

        metrics["alpha_loss"] = alpha_loss
        metrics["alpha"] = self.alpha

        # Update target networks with polyak averaging
        self._update_targets()

        return metrics

    def _update_targets(self):
        """Update target networks with polyak averaging."""
        tau = 0.005  # Soft update parameter
        for source, target in [
            (self.critic_1, self.critic_target_1),
            (self.critic_2, self.critic_target_2),
        ]:
            for source_weight, target_weight in zip(
                source.trainable_variables, target.trainable_variables
            ):
                target_weight.assign(tau * source_weight + (1 - tau) * target_weight)


class BiddingPolicy(nn.Module):
    """Policy network for bid optimization"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """
        Initialize the bidding policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head (mean)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Policy head (log std)
        self.log_std_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1)
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            mean, log_std, value estimate
        """
        features = self.feature_net(state)

        # Get mean and log std for action distribution
        mean = self.policy_net(features)
        log_std = self.log_std_net(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # Get value estimate
        value = self.value_net(features)

        return mean, log_std, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: Input state tensor
            deterministic: Whether to return deterministic action

        Returns:
            action, log probability
        """
        mean, log_std, _ = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
        else:
            # Sample from normal distribution
            normal = Normal(mean, std)
            action = normal.rsample()

        # Get log probability
        log_prob = Normal(mean, std).log_prob(action).sum(dim=-1)

        # Clip action to valid range
        action = torch.clamp(action, 0.5, 2.0)  # Valid bid multipliers

        return action, log_prob

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO.

        Args:
            state: Input state tensor
            action: Actions to evaluate

        Returns:
            log probabilities, value estimates, entropy
        """
        mean, log_std, value = self.forward(state)
        std = log_std.exp()

        # Calculate log probabilities
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)

        # Calculate entropy
        entropy = normal.entropy().sum(dim=-1)

        return log_prob, value, entropy


class KeywordPolicy(nn.Module):
    """Policy network for keyword optimization"""

    def __init__(
        self,
        state_dim: int,
        num_actions: int = 4,  # Add, Remove, Pause, Enable
        hidden_dim: int = 256,
    ):
        """
        Initialize the keyword policy network.

        Args:
            state_dim: Dimension of state space
            num_actions: Number of possible actions
            hidden_dim: Size of hidden layers
        """
        super().__init__()

        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head (discrete actions)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            action logits, value estimate
        """
        features = self.feature_net(state)

        # Get action logits and value estimate
        logits = self.policy_net(features)
        value = self.value_net(features)

        return logits, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: Input state tensor
            deterministic: Whether to return deterministic action

        Returns:
            action, log probability
        """
        logits, _ = self.forward(state)

        if deterministic:
            # Take most likely action
            action = torch.argmax(logits, dim=-1)
        else:
            # Sample from categorical distribution
            dist = Categorical(logits=logits)
            action = dist.sample()

        # Get log probability
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO.

        Args:
            state: Input state tensor
            action: Actions to evaluate

        Returns:
            log probabilities, value estimates, entropy
        """
        logits, value = self.forward(state)

        # Calculate log probabilities
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Calculate entropy
        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        return log_prob, value, entropy
