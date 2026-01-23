"""
Pressure Strategy Optimization with Deep RL

Implements pressure and double-team strategy optimization translated
from basketball research.

Key Paper:
- Wang, J.R., et al. (2018). "The Advantage of Doubling: A Deep
  Reinforcement Learning Approach to Studying Double Teams in the NBA."
  arXiv:1803.02940

Key Concepts:
- Deep RL learns optimal double-team/pressure strategies
- NothingButNet (NBNet) CNN architecture
- Maps state-action pairs to expected cumulative reward
- Compares learned strategy vs. actual strategies

Hockey Translation:
- Optimize forechecking pressure strategies
- Model when to pinch vs. stay back as defenseman
- Evaluate aggressive vs. conservative positioning
- Learn optimal 2-on-1 pressure situations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class PressureType(Enum):
    """Types of pressure strategies."""
    PASSIVE = "passive"              # Maintain gap, contain
    MODERATE = "moderate"            # Close gap gradually
    AGGRESSIVE = "aggressive"        # Attack puck carrier
    DOUBLE_TEAM = "double_team"      # Two players pressure
    PINCH = "pinch"                  # Defenseman moves up


class ZoneLocation(Enum):
    """Zones for pressure decisions."""
    OFFENSIVE_ZONE = "oz"
    NEUTRAL_ZONE = "nz"
    DEFENSIVE_ZONE = "dz"
    BEHIND_NET = "behind_net"
    SLOT = "slot"
    POINT = "point"


class GameSituation(Enum):
    """Game situations affecting pressure."""
    EVEN_STRENGTH = "5v5"
    POWER_PLAY = "pp"
    PENALTY_KILL = "pk"
    FOUR_ON_FOUR = "4v4"
    EMPTY_NET = "en"
    PULLING_GOALIE = "pull"


@dataclass
class PressureState:
    """State for pressure decision-making."""
    puck_x: float                    # Puck location
    puck_y: float
    puck_carrier_id: str
    carrier_speed: float             # Carrier's current speed
    carrier_direction: float         # Direction of movement (radians)

    pressurer_positions: List[Tuple[str, float, float]]  # id, x, y
    support_positions: List[Tuple[str, float, float]]    # Nearby teammates

    time_in_zone: float              # Seconds puck in current zone
    score_differential: int          # + means leading
    time_remaining: float            # Seconds left in period

    zone: ZoneLocation
    situation: GameSituation


@dataclass
class PressureAction:
    """Action taken in pressure situation."""
    action_type: PressureType
    primary_pressurer: str           # Player ID
    secondary_pressurer: Optional[str] = None  # For double-team
    target_gap: float = 10.0         # Target distance to puck
    angle_of_approach: float = 0.0   # Radians


@dataclass
class PressureOutcome:
    """Outcome of a pressure action."""
    turnover_forced: bool
    shot_allowed: bool
    zone_exit_allowed: bool
    time_until_outcome: float        # Seconds
    xg_conceded: float               # Expected goals against
    possession_retained: bool


@dataclass
class PressurePolicy:
    """Learned pressure policy."""
    zone_policies: Dict[ZoneLocation, Dict[str, float]]
    situation_adjustments: Dict[GameSituation, float]
    score_adjustments: Dict[int, float]  # score_diff -> adjustment
    optimal_gap_by_zone: Dict[ZoneLocation, float]


class NBNetEncoder:
    """
    NothingButNet-inspired CNN encoder for hockey states.

    Encodes game state into feature representation for
    pressure decision-making.
    """

    def __init__(
        self,
        grid_size: int = 20,
        hidden_dim: int = 64,
    ):
        """Initialize encoder."""
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim

        # Initialize weights
        np.random.seed(42)
        self.conv_weights = {
            'layer1': np.random.randn(3, 3, 4, 16) * 0.1,
            'layer2': np.random.randn(3, 3, 16, 32) * 0.1,
        }
        self.fc_weights = {
            'W1': np.random.randn(32 * 5 * 5, hidden_dim) * 0.1,
            'b1': np.zeros(hidden_dim),
        }

    def state_to_grid(self, state: PressureState) -> np.ndarray:
        """
        Convert state to grid representation.

        Creates multi-channel grid:
        - Channel 0: Puck location
        - Channel 1: Pressurer positions
        - Channel 2: Support positions
        - Channel 3: Velocity/direction field
        """
        grid = np.zeros((self.grid_size, self.grid_size, 4))

        # Normalize coordinates (assuming 200x85 rink)
        def to_grid(x, y):
            gx = int(np.clip(x / 200 * self.grid_size, 0, self.grid_size - 1))
            gy = int(np.clip(y / 85 * self.grid_size, 0, self.grid_size - 1))
            return gx, gy

        # Channel 0: Puck
        px, py = to_grid(state.puck_x, state.puck_y)
        grid[py, px, 0] = 1.0

        # Channel 1: Pressurers
        for _, x, y in state.pressurer_positions:
            gx, gy = to_grid(x, y)
            grid[gy, gx, 1] = 1.0

        # Channel 2: Support
        for _, x, y in state.support_positions:
            gx, gy = to_grid(x, y)
            grid[gy, gx, 2] = 1.0

        # Channel 3: Velocity direction at puck location
        grid[py, px, 3] = state.carrier_speed / 30.0  # Normalize

        return grid

    def encode(self, state: PressureState) -> np.ndarray:
        """Encode state to feature vector."""
        grid = self.state_to_grid(state)

        # Simplified conv operation (in practice use proper CNN)
        h = grid.mean(axis=(0, 1))  # Global average

        # Add contextual features
        context = np.array([
            state.time_in_zone / 30.0,
            state.score_differential / 5.0,
            state.time_remaining / 1200.0,
            state.carrier_speed / 30.0,
            float(state.zone == ZoneLocation.DEFENSIVE_ZONE),
            float(state.zone == ZoneLocation.SLOT),
        ])

        features = np.concatenate([h, context])

        # FC layer
        output = np.tanh(features[:self.hidden_dim] if len(features) >= self.hidden_dim
                        else np.pad(features, (0, self.hidden_dim - len(features))))

        return output


class PressureStrategyRL:
    """
    Deep RL for pressure strategy optimization.

    Learns when and how to apply pressure based on
    game state and historical outcomes.
    """

    def __init__(
        self,
        encoder: Optional[NBNetEncoder] = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
    ):
        """Initialize RL agent."""
        self.encoder = encoder or NBNetEncoder()
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Action space
        self.actions = list(PressureType)
        self.n_actions = len(self.actions)

        # Q-network weights
        np.random.seed(43)
        self.q_weights = {
            'W1': np.random.randn(self.encoder.hidden_dim, 32) * 0.1,
            'b1': np.zeros(32),
            'W2': np.random.randn(32, self.n_actions) * 0.1,
            'b2': np.zeros(self.n_actions),
        }

        # Experience replay
        self.replay_buffer: List[Tuple] = []
        self.max_buffer_size = 10000

    def get_q_values(self, state_features: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions."""
        h1 = np.maximum(0, state_features @ self.q_weights['W1'] + self.q_weights['b1'])
        q_values = h1 @ self.q_weights['W2'] + self.q_weights['b2']
        return q_values

    def select_action(
        self,
        state: PressureState,
        epsilon: float = 0.1,
    ) -> PressureAction:
        """
        Select pressure action using epsilon-greedy policy.
        """
        features = self.encoder.encode(state)
        q_values = self.get_q_values(features)

        # Epsilon-greedy
        if np.random.random() < epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            action_idx = np.argmax(q_values)

        action_type = self.actions[action_idx]

        # Determine primary pressurer (closest to puck)
        if state.pressurer_positions:
            distances = [
                np.sqrt((x - state.puck_x)**2 + (y - state.puck_y)**2)
                for _, x, y in state.pressurer_positions
            ]
            primary_idx = np.argmin(distances)
            primary = state.pressurer_positions[primary_idx][0]

            # Secondary for double-team
            secondary = None
            if action_type == PressureType.DOUBLE_TEAM and len(state.pressurer_positions) > 1:
                sorted_indices = np.argsort(distances)
                secondary = state.pressurer_positions[sorted_indices[1]][0]
        else:
            primary = "unknown"
            secondary = None

        # Target gap based on action type
        gap_map = {
            PressureType.PASSIVE: 15.0,
            PressureType.MODERATE: 10.0,
            PressureType.AGGRESSIVE: 5.0,
            PressureType.DOUBLE_TEAM: 5.0,
            PressureType.PINCH: 8.0,
        }

        return PressureAction(
            action_type=action_type,
            primary_pressurer=primary,
            secondary_pressurer=secondary,
            target_gap=gap_map[action_type],
            angle_of_approach=np.arctan2(
                state.puck_y - 42.5,  # Center of rink
                state.puck_x - 100,   # Center of rink length
            ),
        )

    def compute_reward(self, outcome: PressureOutcome) -> float:
        """
        Compute reward from pressure outcome.

        Rewards:
        - Turnover: +1.0
        - No zone exit: +0.3
        - Quick outcome: +0.2 * (1 - time/10)

        Penalties:
        - Shot allowed: -0.5
        - High xG: -xG_conceded
        - Zone exit: -0.2
        """
        reward = 0.0

        if outcome.turnover_forced:
            reward += 1.0

        if not outcome.zone_exit_allowed:
            reward += 0.3
        else:
            reward -= 0.2

        if outcome.shot_allowed:
            reward -= 0.5

        reward -= outcome.xg_conceded

        # Time bonus (quicker outcomes better)
        time_factor = max(0, 1 - outcome.time_until_outcome / 10)
        reward += 0.2 * time_factor

        return reward

    def store_experience(
        self,
        state: PressureState,
        action: PressureAction,
        reward: float,
        next_state: Optional[PressureState],
        done: bool,
    ):
        """Store experience in replay buffer."""
        features = self.encoder.encode(state)
        next_features = self.encoder.encode(next_state) if next_state else np.zeros_like(features)
        action_idx = self.actions.index(action.action_type)

        self.replay_buffer.append((features, action_idx, reward, next_features, done))

        # Keep buffer bounded
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]

    def train_step(self, batch_size: int = 32) -> float:
        """
        Perform one training step.

        Returns loss.
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        total_loss = 0.0

        for features, action_idx, reward, next_features, done in batch:
            # Current Q-value
            q_values = self.get_q_values(features)
            current_q = q_values[action_idx]

            # Target Q-value
            if done:
                target_q = reward
            else:
                next_q_values = self.get_q_values(next_features)
                target_q = reward + self.gamma * np.max(next_q_values)

            # TD error
            td_error = target_q - current_q
            total_loss += td_error ** 2

            # Update weights (simplified gradient descent)
            # In practice, use proper backpropagation
            self.q_weights['W2'][:, action_idx] += self.learning_rate * td_error * 0.01

        return total_loss / batch_size


class PinchDecisionModel:
    """
    Model for defenseman pinch decisions.

    Pinching is high-risk/high-reward - keeps puck in zone
    but risks odd-man rush if failed.
    """

    def __init__(self):
        """Initialize pinch model."""
        # Risk tolerance by game state
        self.risk_tolerance = {
            (-3, -999): 0.9,   # Down by 3+, high risk tolerance
            (-2, -3): 0.7,
            (-1, -2): 0.5,
            (0, -1): 0.4,
            (0, 1): 0.3,      # Tied or up by 1
            (1, 2): 0.2,
            (2, 3): 0.15,
            (3, 999): 0.1,    # Up by 3+, low risk
        }

    def should_pinch(
        self,
        defenseman_x: float,
        defenseman_y: float,
        puck_x: float,
        puck_y: float,
        partner_x: float,
        partner_y: float,
        forwards_back: int,
        score_diff: int,
        time_remaining: float,
    ) -> Dict[str, Any]:
        """
        Determine if defenseman should pinch.

        Returns recommendation with confidence.
        """
        # Distance to puck
        dist_to_puck = np.sqrt((defenseman_x - puck_x)**2 + (defenseman_y - puck_y)**2)

        # Partner coverage
        partner_depth = 200 - partner_x  # Distance from offensive goal

        # Forward support
        forward_support = forwards_back >= 2

        # Risk tolerance from score
        risk_tol = 0.3
        for (low, high), tol in self.risk_tolerance.items():
            if low <= score_diff <= high:
                risk_tol = tol
                break

        # Late game adjustment
        if time_remaining < 300:  # Last 5 minutes
            if score_diff > 0:
                risk_tol *= 0.5  # More conservative when leading
            elif score_diff < 0:
                risk_tol *= 1.5  # More aggressive when trailing

        # Pinch score calculation
        pinch_score = 0.0

        # Close to puck = higher pinch value
        if dist_to_puck < 15:
            pinch_score += 0.4
        elif dist_to_puck < 25:
            pinch_score += 0.2

        # Partner deep = safer to pinch
        if partner_depth > 50:
            pinch_score += 0.3

        # Forward support
        if forward_support:
            pinch_score += 0.2

        # Puck in corner (good pinch opportunity)
        in_corner = (puck_x > 175 and (puck_y < 20 or puck_y > 65))
        if in_corner:
            pinch_score += 0.2

        # Decision
        should_pinch = pinch_score > (1 - risk_tol)

        return {
            'should_pinch': should_pinch,
            'pinch_score': pinch_score,
            'risk_tolerance': risk_tol,
            'confidence': abs(pinch_score - (1 - risk_tol)) / 0.5,
            'reasons': self._get_reasons(
                dist_to_puck, partner_depth, forward_support, in_corner
            ),
        }

    def _get_reasons(
        self,
        dist: float,
        partner_depth: float,
        forward_support: bool,
        in_corner: bool,
    ) -> List[str]:
        """Get reasons for recommendation."""
        reasons = []

        if dist < 15:
            reasons.append("Close to puck")
        if partner_depth > 50:
            reasons.append("Partner providing depth")
        if forward_support:
            reasons.append("Forward support available")
        if in_corner:
            reasons.append("Puck in corner (good pinch spot)")

        return reasons


class DoubleteamOptimizer:
    """
    Optimizes double-team situations in hockey.

    Determines when 2-on-1 pressure is optimal.
    """

    def __init__(self):
        """Initialize optimizer."""
        # Zone-specific double-team value
        self.zone_dt_value = {
            ZoneLocation.DEFENSIVE_ZONE: 0.3,   # Risky
            ZoneLocation.NEUTRAL_ZONE: 0.5,
            ZoneLocation.OFFENSIVE_ZONE: 0.7,   # Good opportunity
            ZoneLocation.BEHIND_NET: 0.8,       # Great spot
            ZoneLocation.SLOT: 0.2,             # Don't leave slot
            ZoneLocation.POINT: 0.6,
        }

    def evaluate_doubleteam(
        self,
        state: PressureState,
    ) -> Dict[str, Any]:
        """
        Evaluate whether to double-team.

        Returns recommendation with analysis.
        """
        # Base value from zone
        zone_value = self.zone_dt_value.get(state.zone, 0.5)

        # Time in zone factor (longer = more valuable to force turnover)
        time_factor = min(1.0, state.time_in_zone / 10)

        # Carrier skill factor (would need player ratings)
        # Assume average for now
        carrier_factor = 0.5

        # Support availability
        if len(state.pressurer_positions) < 2:
            return {
                'recommend_doubleteam': False,
                'reason': 'Insufficient players for double-team',
                'value': 0.0,
            }

        # Check if leaving someone open
        coverage_risk = self._assess_coverage_risk(state)

        # Calculate value
        dt_value = (
            zone_value * 0.4 +
            time_factor * 0.2 +
            carrier_factor * 0.2 +
            (1 - coverage_risk) * 0.2
        )

        # Score adjustment
        if state.score_differential < 0:
            dt_value *= 1.2  # More aggressive when trailing
        elif state.score_differential > 1:
            dt_value *= 0.8  # More conservative with lead

        recommend = dt_value > 0.5

        # Find best second pressurer
        second_pressurer = self._select_second_pressurer(state)

        return {
            'recommend_doubleteam': recommend,
            'value': float(dt_value),
            'coverage_risk': float(coverage_risk),
            'zone_factor': float(zone_value),
            'second_pressurer': second_pressurer,
            'expected_turnover_prob': float(dt_value * 0.4),  # Rough estimate
        }

    def _assess_coverage_risk(self, state: PressureState) -> float:
        """Assess risk of leaving players open."""
        # Count support vs pressurers
        n_pressurers = len(state.pressurer_positions)
        n_support = len(state.support_positions)

        # Risk increases with fewer support
        if n_support >= 3:
            return 0.2
        elif n_support == 2:
            return 0.4
        elif n_support == 1:
            return 0.6
        else:
            return 0.9

    def _select_second_pressurer(
        self,
        state: PressureState,
    ) -> Optional[str]:
        """Select best player to be second pressurer."""
        if len(state.pressurer_positions) < 2:
            return None

        # Find closest to puck (excluding closest who is primary)
        distances = []
        for pid, x, y in state.pressurer_positions:
            dist = np.sqrt((x - state.puck_x)**2 + (y - state.puck_y)**2)
            distances.append((pid, dist))

        distances.sort(key=lambda x: x[1])

        # Second closest
        if len(distances) >= 2:
            return distances[1][0]
        return None


class PressureAnalytics:
    """
    Analytics for pressure effectiveness.
    """

    def __init__(self):
        """Initialize analytics."""
        self.pressure_events: List[Dict[str, Any]] = []

    def record_event(
        self,
        state: PressureState,
        action: PressureAction,
        outcome: PressureOutcome,
    ):
        """Record a pressure event."""
        self.pressure_events.append({
            'zone': state.zone.value,
            'situation': state.situation.value,
            'action_type': action.action_type.value,
            'turnover': outcome.turnover_forced,
            'shot_allowed': outcome.shot_allowed,
            'xg_conceded': outcome.xg_conceded,
            'time_to_outcome': outcome.time_until_outcome,
        })

    def get_effectiveness_by_type(self) -> Dict[str, Dict[str, float]]:
        """Get effectiveness metrics by pressure type."""
        type_stats: Dict[str, List[Dict]] = defaultdict(list)

        for event in self.pressure_events:
            type_stats[event['action_type']].append(event)

        results = {}
        for ptype, events in type_stats.items():
            if not events:
                continue

            n = len(events)
            results[ptype] = {
                'turnover_rate': sum(e['turnover'] for e in events) / n,
                'shot_allowed_rate': sum(e['shot_allowed'] for e in events) / n,
                'avg_xg_conceded': np.mean([e['xg_conceded'] for e in events]),
                'avg_time_to_outcome': np.mean([e['time_to_outcome'] for e in events]),
                'count': n,
            }

        return results

    def get_zone_analysis(self) -> Dict[str, Dict[str, float]]:
        """Get pressure effectiveness by zone."""
        zone_stats: Dict[str, List[Dict]] = defaultdict(list)

        for event in self.pressure_events:
            zone_stats[event['zone']].append(event)

        results = {}
        for zone, events in zone_stats.items():
            if not events:
                continue

            n = len(events)
            results[zone] = {
                'turnover_rate': sum(e['turnover'] for e in events) / n,
                'best_action': self._find_best_action(events),
                'count': n,
            }

        return results

    def _find_best_action(self, events: List[Dict]) -> str:
        """Find most effective action type for these events."""
        type_turnover: Dict[str, List[bool]] = defaultdict(list)

        for event in events:
            type_turnover[event['action_type']].append(event['turnover'])

        best_type = "passive"
        best_rate = 0.0

        for atype, outcomes in type_turnover.items():
            rate = sum(outcomes) / len(outcomes) if outcomes else 0
            if rate > best_rate:
                best_rate = rate
                best_type = atype

        return best_type
