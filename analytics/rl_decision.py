"""
Reinforcement Learning Decision Optimizer for Hockey Analytics

This module implements RL-based decision optimization adapted from basketball
and multi-sport research for tactical decision support in hockey.

Applications:
- Evaluate decision quality (shoot vs. pass vs. carry)
- Optimize line deployment based on game state
- Generate tactical recommendations for coaches
- Learn optimal strategies from historical data

References:
    - Yanai, C., et al. (2022). "Q-Ball: Modeling Basketball Games Using
      Deep Reinforcement Learning." AAAI.
    - Wang, J.R., et al. (2018). "The Advantage of Doubling: A Deep RL
      Approach to Studying Double Teams in the NBA." arXiv:1803.02940.
    - "ReLiable: Offline Reinforcement Learning for Tactical Strategies
      in Professional Basketball Games." ACM CIKM 2022.
    - Meng, X. (2025). "AI-powered Tactical Optimization in Dynamic Team
      Sports: A Hierarchical RL Approach." ScienceDirect.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import random


class Action(Enum):
    """Possible actions a player can take."""
    SHOOT = "shoot"
    PASS = "pass"
    CARRY = "carry"
    DUMP = "dump"
    HOLD = "hold"
    CYCLE = "cycle"
    CLEAR = "clear"
    CHANGE = "line_change"


class StrengthState(Enum):
    """Game strength states."""
    FIVE_ON_FIVE = "5v5"
    FIVE_ON_FOUR = "5v4"
    FIVE_ON_THREE = "5v3"
    FOUR_ON_FIVE = "4v5"
    THREE_ON_FIVE = "3v5"
    FOUR_ON_FOUR = "4v4"
    THREE_ON_THREE = "3v3"
    SIX_ON_FIVE = "6v5"


class GameZone(Enum):
    """Ice zones."""
    DEFENSIVE = "defensive"
    NEUTRAL = "neutral"
    OFFENSIVE = "offensive"


@dataclass
class GameState:
    """
    Complete game state representation for RL.

    Encodes all relevant information for decision-making.
    """
    # Score and time
    score_differential: int  # Home - Away
    period: int
    time_remaining: float  # Seconds in period

    # Strength
    strength_state: StrengthState
    penalty_time_remaining: float = 0.0

    # Puck location
    zone: GameZone
    puck_x: float  # Normalized -1 to 1
    puck_y: float  # Normalized -1 to 1

    # Player positions (simplified)
    n_teammates_in_zone: int = 0
    n_opponents_in_zone: int = 0
    closest_defender_distance: float = 0.0

    # Shot opportunity
    shot_quality: float = 0.0  # xG if shooting now
    passing_lanes_open: int = 0

    # Momentum
    recent_shots_for: int = 0
    recent_shots_against: int = 0
    zone_time: float = 0.0  # Seconds in current zone

    # Fatigue
    current_shift_length: float = 0.0
    line_fatigue: float = 0.0  # 0-1

    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for neural network."""
        return np.array([
            self.score_differential / 5,  # Normalize
            self.period / 3,
            self.time_remaining / 1200,
            1.0 if self.strength_state == StrengthState.FIVE_ON_FOUR else 0.0,
            1.0 if self.strength_state == StrengthState.FOUR_ON_FIVE else 0.0,
            1.0 if self.zone == GameZone.OFFENSIVE else 0.0,
            1.0 if self.zone == GameZone.DEFENSIVE else 0.0,
            self.puck_x,
            self.puck_y,
            self.n_teammates_in_zone / 5,
            self.n_opponents_in_zone / 5,
            min(self.closest_defender_distance / 30, 1.0),
            self.shot_quality,
            self.passing_lanes_open / 4,
            self.recent_shots_for / 10,
            self.recent_shots_against / 10,
            min(self.zone_time / 30, 1.0),
            min(self.current_shift_length / 60, 1.0),
            self.line_fatigue,
        ])


@dataclass
class ActionResult:
    """Result of taking an action."""
    success: bool
    new_state: GameState
    reward: float
    terminal: bool = False  # Goal or turnover
    goal_scored: bool = False
    goal_conceded: bool = False


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: GameState
    action: Action
    reward: float
    next_state: GameState
    done: bool


class QNetwork:
    """
    Q-Network for action value estimation.

    Simple neural network that estimates Q(s, a) for all actions.
    In production, this would be a deep neural network.
    """

    def __init__(
        self,
        state_dim: int = 19,
        n_actions: int = 8,
        hidden_dim: int = 64
    ):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimension of state vector
            n_actions: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, n_actions) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass through network.

        Args:
            state: State vector

        Returns:
            Q-values for all actions
        """
        # Hidden layer 1
        h1 = np.maximum(0, state @ self.W1 + self.b1)  # ReLU
        # Hidden layer 2
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        # Output layer (no activation)
        q_values = h2 @ self.W3 + self.b3

        return q_values

    def update(
        self,
        state: np.ndarray,
        action_idx: int,
        target: float,
        learning_rate: float = 0.001
    ):
        """
        Update network weights using gradient descent.

        Args:
            state: State vector
            action_idx: Index of action taken
            target: Target Q-value
            learning_rate: Learning rate
        """
        # Forward pass with cached activations
        h1 = np.maximum(0, state @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        q_values = h2 @ self.W3 + self.b3

        # Compute loss gradient
        loss_grad = np.zeros(self.n_actions)
        loss_grad[action_idx] = q_values[action_idx] - target

        # Backpropagation
        dW3 = np.outer(h2, loss_grad)
        db3 = loss_grad

        dh2 = loss_grad @ self.W3.T
        dh2 = dh2 * (h2 > 0)  # ReLU gradient

        dW2 = np.outer(h1, dh2)
        db2 = dh2

        dh1 = dh2 @ self.W2.T
        dh1 = dh1 * (h1 > 0)  # ReLU gradient

        dW1 = np.outer(state, dh1)
        db1 = dh1

        # Update weights
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


class HockeyDQN:
    """
    Deep Q-Network for Hockey Decision Making.

    Based on Q-Ball (Yanai et al., 2022) and ReLiable (CIKM 2022).

    Learns optimal action policy from historical play-by-play data
    using offline reinforcement learning.
    """

    # Reward structure
    REWARDS = {
        'goal_scored': 1.0,
        'goal_conceded': -1.0,
        'shot_on_goal': 0.05,
        'shot_blocked': -0.02,
        'successful_pass': 0.02,
        'failed_pass': -0.03,
        'zone_entry': 0.03,
        'zone_exit': 0.02,
        'turnover': -0.05,
        'penalty_drawn': 0.1,
        'penalty_taken': -0.1,
    }

    def __init__(
        self,
        state_dim: int = 19,
        n_actions: int = 8,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        learning_rate: float = 0.001,
        buffer_size: int = 10000
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            n_actions: Number of actions
            gamma: Discount factor
            epsilon: Exploration rate
            learning_rate: Learning rate
            buffer_size: Replay buffer size
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.n_actions = n_actions

        # Q-networks (main and target)
        self.q_network = QNetwork(state_dim, n_actions)
        self.target_network = QNetwork(state_dim, n_actions)
        self._copy_weights()

        # Replay buffer
        self.buffer: List[Experience] = []
        self.buffer_size = buffer_size

        # Action mapping
        self.actions = list(Action)[:n_actions]

        # Training stats
        self.training_steps = 0
        self.update_target_every = 100

    def _copy_weights(self):
        """Copy weights from main network to target network."""
        self.target_network.W1 = self.q_network.W1.copy()
        self.target_network.b1 = self.q_network.b1.copy()
        self.target_network.W2 = self.q_network.W2.copy()
        self.target_network.b2 = self.q_network.b2.copy()
        self.target_network.W3 = self.q_network.W3.copy()
        self.target_network.b3 = self.q_network.b3.copy()

    def select_action(
        self,
        state: GameState,
        explore: bool = True
    ) -> Tuple[Action, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current game state
            explore: Whether to use exploration

        Returns:
            Tuple of (selected action, Q-value)
        """
        state_vec = state.to_vector()
        q_values = self.q_network.forward(state_vec)

        # Filter invalid actions based on state
        valid_actions = self._get_valid_actions(state)
        valid_indices = [self.actions.index(a) for a in valid_actions]

        if explore and random.random() < self.epsilon:
            # Explore: random valid action
            action_idx = random.choice(valid_indices)
        else:
            # Exploit: best valid action
            valid_q = [(i, q_values[i]) for i in valid_indices]
            action_idx = max(valid_q, key=lambda x: x[1])[0]

        return self.actions[action_idx], q_values[action_idx]

    def _get_valid_actions(self, state: GameState) -> List[Action]:
        """Get valid actions for current state."""
        valid = []

        if state.zone == GameZone.OFFENSIVE:
            valid.extend([Action.SHOOT, Action.PASS, Action.CARRY, Action.CYCLE])
            if state.shot_quality < 0.05:
                valid.append(Action.HOLD)
        elif state.zone == GameZone.DEFENSIVE:
            valid.extend([Action.PASS, Action.CARRY, Action.CLEAR, Action.DUMP])
        else:  # Neutral zone
            valid.extend([Action.PASS, Action.CARRY, Action.DUMP])

        # Line change always available if not under pressure
        if state.closest_defender_distance > 10:
            valid.append(Action.CHANGE)

        return valid

    def store_experience(self, experience: Experience):
        """Store experience in replay buffer."""
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def train_step(self, batch_size: int = 32) -> float:
        """
        Perform one training step.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Average loss
        """
        if len(self.buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.buffer, batch_size)

        total_loss = 0.0
        for exp in batch:
            state_vec = exp.state.to_vector()
            next_state_vec = exp.next_state.to_vector()
            action_idx = self.actions.index(exp.action)

            # Compute target using Double DQN
            if exp.done:
                target = exp.reward
            else:
                # Use main network to select action
                next_q = self.q_network.forward(next_state_vec)
                best_action = np.argmax(next_q)
                # Use target network to evaluate
                target_q = self.target_network.forward(next_state_vec)
                target = exp.reward + self.gamma * target_q[best_action]

            # Compute current Q-value
            current_q = self.q_network.forward(state_vec)[action_idx]
            loss = (current_q - target) ** 2
            total_loss += loss

            # Update network
            self.q_network.update(state_vec, action_idx, target, self.learning_rate)

        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % self.update_target_every == 0:
            self._copy_weights()

        return total_loss / batch_size

    def get_action_values(self, state: GameState) -> Dict[Action, float]:
        """
        Get Q-values for all actions in a state.

        Args:
            state: Current game state

        Returns:
            Dictionary mapping actions to Q-values
        """
        state_vec = state.to_vector()
        q_values = self.q_network.forward(state_vec)

        return {
            action: q_values[i]
            for i, action in enumerate(self.actions)
        }

    def evaluate_decision(
        self,
        state: GameState,
        action_taken: Action
    ) -> Dict[str, Any]:
        """
        Evaluate a decision that was made.

        Args:
            state: Game state when decision was made
            action_taken: Action that was taken

        Returns:
            Evaluation dictionary with optimal action and value difference
        """
        q_values = self.get_action_values(state)
        valid_actions = self._get_valid_actions(state)

        # Find optimal action among valid ones
        valid_q = {a: q for a, q in q_values.items() if a in valid_actions}
        optimal_action = max(valid_q, key=valid_q.get)
        optimal_value = valid_q[optimal_action]

        # Get value of action taken
        taken_value = q_values.get(action_taken, 0)

        return {
            'action_taken': action_taken.value,
            'optimal_action': optimal_action.value,
            'action_value': taken_value,
            'optimal_value': optimal_value,
            'value_difference': taken_value - optimal_value,
            'was_optimal': action_taken == optimal_action,
            'all_values': {a.value: v for a, v in q_values.items()}
        }


class LineDeploymentOptimizer:
    """
    Optimize line deployment using RL.

    Learns when to use each line based on game state.
    """

    def __init__(self, n_forward_lines: int = 4, n_defense_pairs: int = 3):
        """
        Initialize line deployment optimizer.

        Args:
            n_forward_lines: Number of forward lines
            n_defense_pairs: Number of defense pairs
        """
        self.n_forwards = n_forward_lines
        self.n_defense = n_defense_pairs

        # Q-tables for line selection
        self.forward_q: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(n_forward_lines)
        )
        self.defense_q: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(n_defense_pairs)
        )

        # Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount
        self.epsilon = 0.1  # Exploration

    def _state_key(self, state: GameState) -> str:
        """Convert state to discrete key for Q-table."""
        return (
            f"{state.score_differential:+d}_"
            f"P{state.period}_"
            f"{'PP' if state.strength_state == StrengthState.FIVE_ON_FOUR else ''}"
            f"{'PK' if state.strength_state == StrengthState.FOUR_ON_FIVE else ''}"
            f"{'5v5' if state.strength_state == StrengthState.FIVE_ON_FIVE else ''}_"
            f"T{int(state.time_remaining // 300)}"  # 5-min buckets
        )

    def select_lines(
        self,
        state: GameState,
        line_fatigue: List[float],
        explore: bool = True
    ) -> Tuple[int, int]:
        """
        Select forward line and defense pair.

        Args:
            state: Current game state
            line_fatigue: Fatigue levels for each line
            explore: Whether to explore

        Returns:
            Tuple of (forward_line_index, defense_pair_index)
        """
        key = self._state_key(state)

        # Adjust Q-values by fatigue
        forward_q = self.forward_q[key].copy()
        for i, fatigue in enumerate(line_fatigue[:self.n_forwards]):
            forward_q[i] -= fatigue * 0.5  # Penalty for tired lines

        defense_q = self.defense_q[key].copy()
        for i, fatigue in enumerate(line_fatigue[self.n_forwards:]):
            if i < len(defense_q):
                defense_q[i] -= fatigue * 0.5

        # Epsilon-greedy selection
        if explore and random.random() < self.epsilon:
            forward_line = random.randint(0, self.n_forwards - 1)
            defense_pair = random.randint(0, self.n_defense - 1)
        else:
            forward_line = int(np.argmax(forward_q))
            defense_pair = int(np.argmax(defense_q))

        return forward_line, defense_pair

    def update(
        self,
        state: GameState,
        forward_line: int,
        defense_pair: int,
        reward: float,
        next_state: GameState
    ):
        """
        Update Q-values based on shift outcome.

        Args:
            state: State when lines were deployed
            forward_line: Forward line used
            defense_pair: Defense pair used
            reward: Reward from shift (xG differential, goals, etc.)
            next_state: State after shift
        """
        key = self._state_key(state)
        next_key = self._state_key(next_state)

        # Update forward line Q
        current_q = self.forward_q[key][forward_line]
        next_max_q = np.max(self.forward_q[next_key])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.forward_q[key][forward_line] = new_q

        # Update defense pair Q
        current_q = self.defense_q[key][defense_pair]
        next_max_q = np.max(self.defense_q[next_key])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.defense_q[key][defense_pair] = new_q

    def get_deployment_recommendation(
        self,
        state: GameState,
        line_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get deployment recommendation with explanation.

        Args:
            state: Current game state
            line_names: Optional names for lines

        Returns:
            Recommendation dictionary
        """
        key = self._state_key(state)

        forward_q = self.forward_q[key]
        defense_q = self.defense_q[key]

        rec_forward = int(np.argmax(forward_q))
        rec_defense = int(np.argmax(defense_q))

        return {
            'recommended_forward_line': rec_forward + 1,
            'recommended_defense_pair': rec_defense + 1,
            'forward_line_name': line_names[rec_forward] if line_names else f"Line {rec_forward + 1}",
            'forward_values': forward_q.tolist(),
            'defense_values': defense_q.tolist(),
            'context': {
                'score': state.score_differential,
                'period': state.period,
                'strength': state.strength_state.value,
                'time_bucket': int(state.time_remaining // 300)
            }
        }


class HierarchicalTacticalRL:
    """
    Hierarchical RL for multi-level tactical decisions.

    Based on Meng (2025) hierarchical approach:
    - Game level: Overall strategy
    - Period level: Tactical adjustments
    - Shift level: Line deployment
    - Play level: Individual decisions
    """

    def __init__(self):
        """Initialize hierarchical RL system."""
        # Game-level strategy
        self.game_strategies = ['aggressive', 'balanced', 'defensive', 'comeback']
        self.strategy_q: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.game_strategies))
        )

        # Period-level tactics
        self.period_tactics = ['forecheck', 'neutral_trap', 'protect_lead', 'push']
        self.tactics_q: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.period_tactics))
        )

        # Shift-level deployment
        self.deployment = LineDeploymentOptimizer()

        # Play-level decisions
        self.play_dqn = HockeyDQN()

    def get_game_strategy(self, state: GameState) -> str:
        """
        Determine overall game strategy.

        Args:
            state: Current game state

        Returns:
            Strategy name
        """
        key = f"P{state.period}_{state.score_differential:+d}"
        q_values = self.strategy_q[key]

        # Default strategy based on score
        if state.score_differential <= -2:
            default_idx = self.game_strategies.index('comeback')
        elif state.score_differential >= 2:
            default_idx = self.game_strategies.index('defensive')
        else:
            default_idx = self.game_strategies.index('balanced')

        # Blend learned and default
        if np.max(q_values) > 0:
            idx = int(np.argmax(q_values))
        else:
            idx = default_idx

        return self.game_strategies[idx]

    def get_period_tactics(self, state: GameState, strategy: str) -> str:
        """
        Determine period-level tactics given strategy.

        Args:
            state: Current game state
            strategy: Overall game strategy

        Returns:
            Tactical approach
        """
        key = f"{strategy}_P{state.period}_T{int(state.time_remaining // 300)}"
        q_values = self.tactics_q[key]

        # Default based on strategy and time
        if strategy == 'aggressive':
            default = 'forecheck'
        elif strategy == 'defensive':
            if state.time_remaining < 300:  # Last 5 min
                default = 'protect_lead'
            else:
                default = 'neutral_trap'
        elif strategy == 'comeback':
            if state.time_remaining < 300:
                default = 'push'
            else:
                default = 'forecheck'
        else:
            default = 'forecheck' if state.period < 3 else 'neutral_trap'

        if np.max(q_values) > 0:
            idx = int(np.argmax(q_values))
        else:
            idx = self.period_tactics.index(default)

        return self.period_tactics[idx]

    def get_full_recommendation(
        self,
        state: GameState,
        line_fatigue: List[float]
    ) -> Dict[str, Any]:
        """
        Get complete hierarchical recommendation.

        Args:
            state: Current game state
            line_fatigue: Current fatigue levels

        Returns:
            Multi-level recommendation
        """
        strategy = self.get_game_strategy(state)
        tactics = self.get_period_tactics(state, strategy)
        forward_line, defense_pair = self.deployment.select_lines(
            state, line_fatigue, explore=False
        )

        # Get play-level action values
        action_values = self.play_dqn.get_action_values(state)
        best_action = max(action_values, key=action_values.get)

        return {
            'game_strategy': strategy,
            'period_tactics': tactics,
            'line_deployment': {
                'forward_line': forward_line + 1,
                'defense_pair': defense_pair + 1
            },
            'play_recommendation': {
                'best_action': best_action.value,
                'action_values': {a.value: v for a, v in action_values.items()}
            },
            'context': {
                'score': state.score_differential,
                'period': state.period,
                'time_remaining': state.time_remaining,
                'strength': state.strength_state.value
            }
        }


class DecisionEvaluator:
    """
    Evaluate decision quality over time.

    Tracks how well actual decisions match optimal policy.
    """

    def __init__(self, dqn: HockeyDQN):
        """
        Initialize evaluator.

        Args:
            dqn: Trained DQN model
        """
        self.dqn = dqn
        self.evaluations: List[Dict] = []

    def evaluate_game_decisions(
        self,
        decisions: List[Tuple[GameState, Action]]
    ) -> Dict[str, Any]:
        """
        Evaluate all decisions from a game.

        Args:
            decisions: List of (state, action) tuples

        Returns:
            Game-level evaluation metrics
        """
        total_value_lost = 0.0
        optimal_count = 0
        evaluations = []

        for state, action in decisions:
            eval_result = self.dqn.evaluate_decision(state, action)
            evaluations.append(eval_result)

            if eval_result['was_optimal']:
                optimal_count += 1
            else:
                total_value_lost += abs(eval_result['value_difference'])

        n_decisions = len(decisions)

        return {
            'total_decisions': n_decisions,
            'optimal_decisions': optimal_count,
            'optimal_rate': optimal_count / n_decisions if n_decisions > 0 else 0,
            'total_value_lost': total_value_lost,
            'avg_value_lost': total_value_lost / n_decisions if n_decisions > 0 else 0,
            'evaluations': evaluations
        }

    def get_player_decision_quality(
        self,
        player_decisions: Dict[str, List[Tuple[GameState, Action]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate decision quality by player.

        Args:
            player_decisions: Dictionary mapping player_id to their decisions

        Returns:
            Per-player decision quality metrics
        """
        results = {}

        for player_id, decisions in player_decisions.items():
            eval_result = self.evaluate_game_decisions(decisions)
            results[player_id] = {
                'n_decisions': eval_result['total_decisions'],
                'optimal_rate': eval_result['optimal_rate'],
                'avg_value_lost': eval_result['avg_value_lost']
            }

        return results
