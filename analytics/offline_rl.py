"""
Offline Reinforcement Learning for Hockey Tactics

Implements methodology from:
- "ReLiable: Offline Reinforcement Learning for Tactical Strategies in
  Professional Basketball Games." ACM CIKM 2022.

Key challenges addressed:
- Heterogeneous signals (events, tracking, context)
- Massive action/outcome space
- Incomplete observations
- No environment interaction during training

Hockey translation:
- Train on historical NHL play-by-play data
- Suggest optimal actions given game state
- Account for incomplete tracking data
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from datetime import datetime


class ActionSpace(Enum):
    """Discrete action space for hockey decisions."""
    SHOOT = "shoot"
    PASS_CROSS_ICE = "pass_cross_ice"
    PASS_TAPE_TO_TAPE = "pass_tape_to_tape"
    PASS_SAUCER = "pass_saucer"
    CARRY = "carry"
    DUMP_IN = "dump_in"
    CHIP_OUT = "chip_out"
    CYCLE_LOW = "cycle_low"
    CYCLE_HIGH = "cycle_high"
    HOLD = "hold"  # Protect puck
    REVERSE = "reverse"  # Reverse play direction


class OutcomeSpace(Enum):
    """Possible outcomes from actions."""
    GOAL = "goal"
    SHOT_ON_GOAL = "shot_on_goal"
    SHOT_MISSED = "shot_missed"
    PASS_COMPLETE = "pass_complete"
    PASS_INCOMPLETE = "pass_incomplete"
    TURNOVER = "turnover"
    ZONE_EXIT = "zone_exit"
    ZONE_ENTRY = "zone_entry"
    FACEOFF = "faceoff"
    PENALTY_DRAWN = "penalty_drawn"
    PENALTY_TAKEN = "penalty_taken"


@dataclass
class HistoricalPlay:
    """Single play from historical data."""
    game_id: str
    timestamp: float
    period: int
    time_remaining: float
    state_features: np.ndarray  # Pre-computed features
    action: ActionSpace
    outcome: OutcomeSpace
    reward: float  # Computed reward
    next_state_features: Optional[np.ndarray] = None
    player_id: Optional[str] = None
    zone: str = "neutral"
    score_differential: int = 0
    strength_state: str = "5v5"


@dataclass
class StateFeatures:
    """Feature representation of game state."""
    # Spatial features
    puck_location: Tuple[float, float]
    player_locations: List[Tuple[float, float]]  # 10 players
    puck_velocity: Tuple[float, float]

    # Context features
    score_differential: int
    time_remaining: float
    period: int
    strength_state: str  # "5v5", "5v4", "4v5", etc.
    zone: str

    # Derived features
    passing_lane_quality: float
    shot_quality: float
    pressure_level: float


@dataclass
class BatchExperience:
    """Batch of experiences for training."""
    states: np.ndarray  # (batch, state_dim)
    actions: np.ndarray  # (batch,) action indices
    rewards: np.ndarray  # (batch,)
    next_states: np.ndarray  # (batch, state_dim)
    dones: np.ndarray  # (batch,) terminal flags


@dataclass
class CQLConfig:
    """Configuration for Conservative Q-Learning."""
    state_dim: int = 64
    action_dim: int = 11  # len(ActionSpace)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    gamma: float = 0.99
    alpha: float = 0.5  # CQL regularization weight
    learning_rate: float = 3e-4
    batch_size: int = 256
    tau: float = 0.005  # Target network update


class QNetwork:
    """
    Q-Network for value estimation.

    Estimates Q(s, a) for all actions given state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize weights
        dims = [state_dim] + hidden_dims + [action_dim]
        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.biases.append(np.zeros(dims[i+1]))

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass through network.

        Args:
            state: (batch, state_dim) or (state_dim,)

        Returns:
            Q-values: (batch, action_dim) or (action_dim,)
        """
        x = state
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU
        return x

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        return self.forward(state)

    def get_value(self, state: np.ndarray, action_idx: int) -> float:
        """Get Q-value for specific action."""
        q_values = self.forward(state)
        return q_values[action_idx]


class ConservativeQLearning:
    """
    Conservative Q-Learning (CQL) for offline RL.

    Key insight: Standard Q-learning overestimates values for
    out-of-distribution actions. CQL adds a penalty for high
    Q-values on actions not in the dataset.
    """

    def __init__(self, config: CQLConfig):
        self.config = config

        # Main and target networks
        self.q_network = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        )
        self.target_network = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        )

        # Copy weights to target
        self._update_target(tau=1.0)

    def _update_target(self, tau: float):
        """Soft update of target network."""
        for i in range(len(self.q_network.weights)):
            self.target_network.weights[i] = (
                tau * self.q_network.weights[i] +
                (1 - tau) * self.target_network.weights[i]
            )
            self.target_network.biases[i] = (
                tau * self.q_network.biases[i] +
                (1 - tau) * self.target_network.biases[i]
            )

    def compute_cql_loss(
        self,
        batch: BatchExperience
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute CQL loss.

        CQL loss = TD loss + alpha * (logsumexp(Q) - Q(a_data))
        """
        # Standard TD target
        next_q = self.target_network.forward(batch.next_states)
        max_next_q = np.max(next_q, axis=1)
        td_target = batch.rewards + self.config.gamma * max_next_q * (1 - batch.dones)

        # Current Q-values
        current_q = self.q_network.forward(batch.states)
        current_q_a = current_q[np.arange(len(batch.actions)), batch.actions.astype(int)]

        # TD loss
        td_loss = np.mean((current_q_a - td_target) ** 2)

        # CQL penalty: logsumexp(Q) - Q(a_data)
        # Encourages lower Q-values for actions not in data
        logsumexp_q = np.log(np.sum(np.exp(current_q), axis=1) + 1e-10)
        cql_penalty = np.mean(logsumexp_q - current_q_a)

        # Total loss
        total_loss = td_loss + self.config.alpha * cql_penalty

        return total_loss, {
            'td_loss': td_loss,
            'cql_penalty': cql_penalty,
            'mean_q': np.mean(current_q_a)
        }

    def train_step(
        self,
        batch: BatchExperience,
        lr: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Single training step.

        Uses numerical gradient for simplicity.
        In practice, use automatic differentiation.
        """
        lr = lr or self.config.learning_rate

        # Compute loss and approximate gradients
        loss, metrics = self.compute_cql_loss(batch)

        # Simplified gradient descent (in practice use autodiff)
        eps = 1e-5
        for layer_idx in range(len(self.q_network.weights)):
            # Weight gradient approximation
            for i in range(min(10, self.q_network.weights[layer_idx].shape[0])):
                for j in range(min(10, self.q_network.weights[layer_idx].shape[1])):
                    self.q_network.weights[layer_idx][i, j] += eps
                    loss_plus, _ = self.compute_cql_loss(batch)
                    self.q_network.weights[layer_idx][i, j] -= 2 * eps
                    loss_minus, _ = self.compute_cql_loss(batch)
                    self.q_network.weights[layer_idx][i, j] += eps

                    grad = (loss_plus - loss_minus) / (2 * eps)
                    self.q_network.weights[layer_idx][i, j] -= lr * grad

        # Update target network
        self._update_target(self.config.tau)

        return metrics

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0
    ) -> Tuple[ActionSpace, float]:
        """Select action using learned Q-values."""
        if np.random.random() < epsilon:
            action_idx = np.random.randint(self.config.action_dim)
        else:
            q_values = self.q_network.get_q_values(state)
            action_idx = np.argmax(q_values)

        actions = list(ActionSpace)
        q_value = self.q_network.get_q_values(state)[action_idx]

        return actions[action_idx], q_value


class BehaviorCloning:
    """
    Behavior cloning baseline.

    Learns to imitate actions in the dataset without
    considering outcomes.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        hidden_dims = hidden_dims or [256, 256]

        # Policy network
        dims = [state_dim] + hidden_dims + [action_dim]
        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.biases.append(np.zeros(dims[i+1]))

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass returning action probabilities."""
        x = state
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)

        # Softmax
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def compute_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> float:
        """Cross-entropy loss."""
        probs = self.forward(states)
        log_probs = np.log(probs + 1e-10)

        # Select log prob of actual action
        action_log_probs = log_probs[np.arange(len(actions)), actions.astype(int)]

        return -np.mean(action_log_probs)

    def select_action(
        self,
        state: np.ndarray,
        temperature: float = 1.0
    ) -> Tuple[ActionSpace, float]:
        """Sample action from policy."""
        probs = self.forward(state)
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)

        action_idx = np.random.choice(self.action_dim, p=probs)
        actions = list(ActionSpace)

        return actions[action_idx], probs[action_idx]


class ImplicitQLearning:
    """
    Implicit Q-Learning (IQL) variant.

    Alternative offline RL algorithm that avoids querying
    out-of-distribution actions entirely.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        tau: float = 0.7  # Expectile for value function
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau

        # Q-network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims)

        # Value network V(s) - single output
        dims = [state_dim] + hidden_dims + [1]
        self.v_weights = []
        self.v_biases = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.v_weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.v_biases.append(np.zeros(dims[i+1]))

    def value_forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass for value network."""
        x = state
        for i, (w, b) in enumerate(zip(self.v_weights, self.v_biases)):
            x = x @ w + b
            if i < len(self.v_weights) - 1:
                x = np.maximum(0, x)
        return x.squeeze()

    def expectile_loss(
        self,
        errors: np.ndarray,
        tau: float
    ) -> np.ndarray:
        """Asymmetric expectile loss."""
        weight = np.where(errors >= 0, tau, 1 - tau)
        return weight * errors ** 2

    def compute_iql_loss(
        self,
        batch: BatchExperience,
        gamma: float = 0.99
    ) -> Dict[str, float]:
        """Compute IQL losses."""
        # Q-values for actions in data
        current_q = self.q_network.forward(batch.states)
        current_q_a = current_q[np.arange(len(batch.actions)), batch.actions.astype(int)]

        # Value function
        current_v = self.value_forward(batch.states)
        next_v = self.value_forward(batch.next_states)

        # Value loss: expectile regression toward Q
        v_errors = current_q_a - current_v
        v_loss = np.mean(self.expectile_loss(v_errors, self.tau))

        # Q loss: TD error using V as target
        q_target = batch.rewards + gamma * next_v * (1 - batch.dones)
        q_loss = np.mean((current_q_a - q_target) ** 2)

        return {
            'v_loss': v_loss,
            'q_loss': q_loss,
            'mean_q': np.mean(current_q_a),
            'mean_v': np.mean(current_v)
        }


class OfflineRLTrainer:
    """
    Trainer for offline RL on historical hockey data.
    """

    def __init__(
        self,
        algorithm: str = "cql",
        config: Optional[CQLConfig] = None
    ):
        self.config = config or CQLConfig()

        if algorithm == "cql":
            self.agent = ConservativeQLearning(self.config)
        elif algorithm == "bc":
            self.agent = BehaviorCloning(
                self.config.state_dim,
                self.config.action_dim
            )
        elif algorithm == "iql":
            self.agent = ImplicitQLearning(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dims
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.training_history: List[Dict[str, float]] = []

    def preprocess_data(
        self,
        plays: List[HistoricalPlay]
    ) -> List[BatchExperience]:
        """Convert historical plays to training batches."""
        # Filter plays with next states
        valid_plays = [p for p in plays if p.next_state_features is not None]

        # Create batch
        n = len(valid_plays)
        states = np.array([p.state_features for p in valid_plays])
        actions = np.array([list(ActionSpace).index(p.action) for p in valid_plays])
        rewards = np.array([p.reward for p in valid_plays])
        next_states = np.array([p.next_state_features for p in valid_plays])
        dones = np.zeros(n)  # Could be inferred from plays

        # Split into batches
        batches = []
        for i in range(0, n, self.config.batch_size):
            end = min(i + self.config.batch_size, n)
            batches.append(BatchExperience(
                states=states[i:end],
                actions=actions[i:end],
                rewards=rewards[i:end],
                next_states=next_states[i:end],
                dones=dones[i:end]
            ))

        return batches

    def train(
        self,
        plays: List[HistoricalPlay],
        n_epochs: int = 10
    ) -> List[Dict[str, float]]:
        """Train on historical data."""
        batches = self.preprocess_data(plays)

        for epoch in range(n_epochs):
            epoch_metrics = []

            for batch in batches:
                if hasattr(self.agent, 'train_step'):
                    metrics = self.agent.train_step(batch)
                else:
                    metrics = {'loss': 0.0}

                epoch_metrics.append(metrics)

            # Average epoch metrics
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
            avg_metrics['epoch'] = epoch

            self.training_history.append(avg_metrics)

        return self.training_history

    def evaluate(
        self,
        test_plays: List[HistoricalPlay]
    ) -> Dict[str, float]:
        """Evaluate policy on held-out data."""
        correct_actions = 0
        total_reward = 0.0
        q_values = []

        for play in test_plays:
            # Get agent's action
            action, q_value = self.agent.select_action(play.state_features)
            q_values.append(q_value)

            # Check if matches actual action
            if action == play.action:
                correct_actions += 1

            total_reward += play.reward

        return {
            'action_accuracy': correct_actions / len(test_plays),
            'mean_q_value': np.mean(q_values),
            'mean_reward': total_reward / len(test_plays)
        }


class DatasetBuilder:
    """
    Build offline RL dataset from raw NHL data.
    """

    def __init__(self):
        self.reward_weights = {
            OutcomeSpace.GOAL: 1.0,
            OutcomeSpace.SHOT_ON_GOAL: 0.1,
            OutcomeSpace.SHOT_MISSED: 0.02,
            OutcomeSpace.PASS_COMPLETE: 0.05,
            OutcomeSpace.PASS_INCOMPLETE: -0.02,
            OutcomeSpace.TURNOVER: -0.1,
            OutcomeSpace.ZONE_EXIT: -0.05,
            OutcomeSpace.ZONE_ENTRY: 0.08,
            OutcomeSpace.PENALTY_DRAWN: 0.15,
            OutcomeSpace.PENALTY_TAKEN: -0.15,
        }

    def compute_reward(
        self,
        outcome: OutcomeSpace,
        context: Dict[str, Any]
    ) -> float:
        """Compute reward for outcome given context."""
        base_reward = self.reward_weights.get(outcome, 0.0)

        # Adjust for context
        multiplier = 1.0

        # Higher value in offensive zone
        if context.get('zone') == 'offensive':
            multiplier *= 1.2

        # Higher value when trailing
        if context.get('score_differential', 0) < 0:
            multiplier *= 1.1

        # Lower value late when leading
        if (context.get('score_differential', 0) > 0 and
            context.get('time_remaining', 1200) < 300):
            multiplier *= 0.8

        return base_reward * multiplier

    def build_play(
        self,
        raw_event: Dict[str, Any]
    ) -> HistoricalPlay:
        """Convert raw event to training play."""
        # Extract features (simplified)
        state_features = np.random.randn(64)  # Would be computed from tracking

        # Map event type to action
        action = ActionSpace.PASS_TAPE_TO_TAPE  # Default
        if 'shot' in raw_event.get('type', '').lower():
            action = ActionSpace.SHOOT
        elif 'dump' in raw_event.get('type', '').lower():
            action = ActionSpace.DUMP_IN

        # Determine outcome
        outcome = OutcomeSpace.PASS_COMPLETE  # Default
        if raw_event.get('goal'):
            outcome = OutcomeSpace.GOAL
        elif raw_event.get('shot'):
            outcome = OutcomeSpace.SHOT_ON_GOAL

        # Compute reward
        context = {
            'zone': raw_event.get('zone', 'neutral'),
            'score_differential': raw_event.get('score_diff', 0),
            'time_remaining': raw_event.get('time_remaining', 1200)
        }
        reward = self.compute_reward(outcome, context)

        return HistoricalPlay(
            game_id=raw_event.get('game_id', ''),
            timestamp=raw_event.get('timestamp', 0.0),
            period=raw_event.get('period', 1),
            time_remaining=context['time_remaining'],
            state_features=state_features,
            action=action,
            outcome=outcome,
            reward=reward,
            zone=context['zone'],
            score_differential=context['score_differential']
        )
