"""
VAEP (Valuing Actions by Estimating Probabilities) for Hockey Analytics

This module implements the VAEP framework adapted from soccer analytics
(Decroos et al., 2019) for valuing individual player actions in hockey.

Unlike xT which only considers location, VAEP considers:
- The full action sequence (last 3 actions)
- Both offensive and defensive value
- Action type, result, and context

Formula:
    VAEP(a) = ΔP(scores) - ΔP(concedes)

    Where:
    - ΔP(scores) = P(score | game state after a) - P(score | game state before a)
    - ΔP(concedes) = P(concede | game state after a) - P(concede | game state before a)

References:
    - Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019).
      "Actions Speak Louder Than Goals: Valuing Player Actions in Soccer."
      ACM SIGKDD.
    - Van Roy, M., et al. (2020). "Valuing On-the-Ball Actions in Soccer:
      A Critical Comparison of xT and VAEP." AAAI Workshop.
    - ML-KULeuven/socceraction library (SPADL format)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json


class ActionType(Enum):
    """Hockey action types for VAEP calculation (HADL - Hockey Action Description Language)."""
    # Puck movement
    PASS = "pass"
    CARRY = "carry"
    DUMP = "dump"
    CHIP = "chip"
    CLEAR = "clear"

    # Shots
    SHOT = "shot"
    SHOT_BLOCKED = "shot_blocked"
    SHOT_SAVED = "shot_saved"
    GOAL = "goal"

    # Defensive
    TAKEAWAY = "takeaway"
    BLOCK = "block"
    HIT = "hit"
    INTERCEPTION = "interception"

    # Turnovers
    GIVEAWAY = "giveaway"
    TURNOVER = "turnover"
    OFFSIDE = "offside"
    ICING = "icing"

    # Set pieces
    FACEOFF_WIN = "faceoff_win"
    FACEOFF_LOSS = "faceoff_loss"

    # Other
    RECEPTION = "reception"
    LINE_CHANGE = "line_change"
    PENALTY_DRAWN = "penalty_drawn"
    PENALTY_TAKEN = "penalty_taken"


class ActionResult(Enum):
    """Result of an action."""
    SUCCESS = "success"
    FAIL = "fail"
    PARTIAL = "partial"  # e.g., pass completed but under pressure


class BodyPart(Enum):
    """Body part used for action (relevant for shots/passes)."""
    STICK = "stick"
    SKATE = "skate"
    BODY = "body"
    GLOVE = "glove"  # For goalies


class Zone(Enum):
    """Ice zones."""
    DEFENSIVE = "defensive"
    NEUTRAL = "neutral"
    OFFENSIVE = "offensive"
    BEHIND_NET = "behind_net"
    SLOT = "slot"
    CREASE = "crease"


@dataclass
class HADLAction:
    """
    Hockey Action Description Language (HADL) format.

    Standardized representation of hockey actions for VAEP,
    inspired by SPADL (Soccer Player Action Description Language).
    """
    action_id: str
    game_id: str
    period: int
    time_seconds: float  # Seconds into period

    # Core action info
    action_type: ActionType
    result: ActionResult
    player_id: str
    team_id: str

    # Location (normalized 0-1, origin at center ice)
    start_x: float  # -1 (own goal) to 1 (opponent goal)
    start_y: float  # -1 (left boards) to 1 (right boards)
    end_x: Optional[float] = None
    end_y: Optional[float] = None

    # Context
    strength_state: str = "5v5"  # 5v5, 5v4, 4v5, etc.
    score_differential: int = 0  # Positive = team leading
    home_team: bool = True

    # Action details
    body_part: BodyPart = BodyPart.STICK
    under_pressure: bool = False
    fast_break: bool = False

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    """
    Game state representation for VAEP probability models.

    Contains features computed from the last 3 actions.
    """
    # Action sequence features (last 3 actions)
    action_types: List[ActionType]
    action_results: List[ActionResult]

    # Location features
    start_x: List[float]
    start_y: List[float]
    end_x: List[float]
    end_y: List[float]

    # Distance/angle features
    distance_to_goal: List[float]
    angle_to_goal: List[float]
    distance_traveled: List[float]

    # Context features
    strength_state: str
    score_differential: int
    period: int
    time_remaining: float

    # Derived features
    zone: Zone
    is_counter_attack: bool
    puck_speed: float


@dataclass
class VAEPValue:
    """VAEP value decomposition for an action."""
    total_value: float
    offensive_value: float  # ΔP(scores)
    defensive_value: float  # -ΔP(concedes)

    # Probability changes
    p_score_before: float
    p_score_after: float
    p_concede_before: float
    p_concede_after: float

    # Context
    action_type: ActionType
    player_id: str


class HockeyVAEP:
    """
    VAEP Model for Hockey.

    Assigns a value to every on-puck action based on how it changes
    the probability of scoring and conceding.

    The model uses features from the last 3 actions to compute:
    1. P(score in next N actions | current game state)
    2. P(concede in next N actions | current game state)

    Value = ΔP(score) - ΔP(concede)

    Based on Decroos et al. (2019) with hockey-specific adaptations:
    - HADL (Hockey Action Description Language) for action representation
    - Hockey-specific action types (dumps, icings, line changes)
    - Behind-net zones and cycle play patterns
    - Strength state weighting
    """

    # Lookahead window (actions to consider for scoring probability)
    LOOKAHEAD_ACTIONS = 10

    # Feature weights for probability models (learned from data)
    # These are placeholder weights - in production, train on actual data
    FEATURE_WEIGHTS = {
        'distance_to_goal': -0.15,
        'angle_to_goal': 0.08,
        'in_slot': 0.25,
        'behind_net': 0.05,
        'shot_attempt': 0.35,
        'successful_pass': 0.10,
        'turnover': -0.20,
        'pressure': -0.05,
        'counter_attack': 0.12,
        'power_play': 0.15,
        'penalty_kill': -0.10,
    }

    def __init__(
        self,
        n_actions_context: int = 3,
        n_actions_lookahead: int = 10,
        scoring_window_seconds: float = 15.0
    ):
        """
        Initialize VAEP model.

        Args:
            n_actions_context: Number of previous actions to consider
            n_actions_lookahead: Actions to look ahead for scoring probability
            scoring_window_seconds: Time window for scoring probability
        """
        self.n_context = n_actions_context
        self.n_lookahead = n_actions_lookahead
        self.scoring_window = scoring_window_seconds

        # Action history buffer
        self.action_buffer: deque = deque(maxlen=n_actions_context)

        # Trained model coefficients (placeholder - would be learned)
        self.score_model_weights = self._init_score_model()
        self.concede_model_weights = self._init_concede_model()

    def _init_score_model(self) -> Dict[str, float]:
        """Initialize scoring probability model weights."""
        return {
            # Location features
            'distance_to_goal': -0.08,
            'angle_to_goal': 0.04,
            'in_slot': 0.30,
            'in_crease': 0.45,
            'behind_net': 0.08,

            # Action type features
            'shot': 0.40,
            'pass_to_slot': 0.25,
            'carry_forward': 0.12,
            'dump': -0.05,
            'turnover': -0.15,
            'faceoff_win_oz': 0.10,

            # Sequence features
            'consecutive_passes': 0.05,
            'zone_time': 0.03,
            'odd_man_rush': 0.35,

            # Context features
            'power_play': 0.20,
            'score_close': 0.02,
            'third_period': 0.03,

            'intercept': -0.10
        }

    def _init_concede_model(self) -> Dict[str, float]:
        """Initialize conceding probability model weights."""
        return {
            # Location features (in own zone = bad)
            'distance_from_own_goal': -0.06,
            'in_own_slot': 0.25,
            'in_own_crease': 0.40,

            # Action type features
            'giveaway': 0.30,
            'failed_clear': 0.25,
            'icing': 0.15,
            'turnover_in_dz': 0.35,
            'blocked_shot': -0.10,
            'takeaway': -0.15,

            # Sequence features
            'under_pressure': 0.10,
            'extended_dz_time': 0.12,

            # Context features
            'penalty_kill': 0.20,
            'trailing': 0.05,
        }

    def compute_game_state(self, actions: List[HADLAction]) -> GameState:
        """
        Compute game state features from action sequence.

        Args:
            actions: Last N actions (most recent last)

        Returns:
            GameState with computed features
        """
        if not actions:
            return self._empty_game_state()

        # Pad if fewer than n_context actions
        while len(actions) < self.n_context:
            actions = [self._create_dummy_action()] + actions

        recent = actions[-self.n_context:]

        # Extract action features
        action_types = [a.action_type for a in recent]
        action_results = [a.result for a in recent]

        # Location features
        start_x = [a.start_x for a in recent]
        start_y = [a.start_y for a in recent]
        end_x = [a.end_x if a.end_x else a.start_x for a in recent]
        end_y = [a.end_y if a.end_y else a.start_y for a in recent]

        # Distance and angle to goal
        distance_to_goal = [self._distance_to_goal(a.end_x or a.start_x, a.end_y or a.start_y) for a in recent]
        angle_to_goal = [self._angle_to_goal(a.end_x or a.start_x, a.end_y or a.start_y) for a in recent]

        # Distance traveled
        distance_traveled = [
            np.sqrt((a.end_x - a.start_x)**2 + (a.end_y - a.start_y)**2)
            if a.end_x and a.end_y else 0.0
            for a in recent
        ]

        # Current context
        current = recent[-1]
        zone = self._determine_zone(current.end_x or current.start_x)

        # Counter attack detection
        is_counter = self._detect_counter_attack(recent)

        # Puck speed estimation
        puck_speed = self._estimate_puck_speed(recent)

        return GameState(
            action_types=action_types,
            action_results=action_results,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            distance_to_goal=distance_to_goal,
            angle_to_goal=angle_to_goal,
            distance_traveled=distance_traveled,
            strength_state=current.strength_state,
            score_differential=current.score_differential,
            period=current.period,
            time_remaining=(20 * 60) - current.time_seconds,
            zone=zone,
            is_counter_attack=is_counter,
            puck_speed=puck_speed
        )

    def compute_vaep(
        self,
        action: HADLAction,
        state_before: GameState,
        state_after: GameState
    ) -> VAEPValue:
        """
        Compute VAEP value for a single action.

        Args:
            action: The action to value
            state_before: Game state before the action
            state_after: Game state after the action

        Returns:
            VAEPValue with offensive and defensive components
        """
        # Compute scoring probabilities
        p_score_before = self._score_probability(state_before)
        p_score_after = self._score_probability(state_after)

        # Compute conceding probabilities
        p_concede_before = self._concede_probability(state_before)
        p_concede_after = self._concede_probability(state_after)

        # VAEP formula
        offensive_value = p_score_after - p_score_before
        defensive_value = -(p_concede_after - p_concede_before)
        total_value = offensive_value + defensive_value

        return VAEPValue(
            total_value=total_value,
            offensive_value=offensive_value,
            defensive_value=defensive_value,
            p_score_before=p_score_before,
            p_score_after=p_score_after,
            p_concede_before=p_concede_before,
            p_concede_after=p_concede_after,
            action_type=action.action_type,
            player_id=action.player_id
        )

    def _score_probability(self, state: GameState) -> float:
        """
        Compute probability of scoring in next N actions.

        Uses logistic regression with pre-trained weights.
        """
        if not state.action_types:
            return 0.05  # Baseline

        features = self._extract_score_features(state)
        logit = sum(
            self.score_model_weights.get(k, 0) * v
            for k, v in features.items()
        )

        # Add bias
        logit += -2.5  # Base scoring probability ~8%

        return self._sigmoid(logit)

    def _concede_probability(self, state: GameState) -> float:
        """
        Compute probability of conceding in next N actions.

        Uses logistic regression with pre-trained weights.
        """
        if not state.action_types:
            return 0.05  # Baseline

        features = self._extract_concede_features(state)
        logit = sum(
            self.concede_model_weights.get(k, 0) * v
            for k, v in features.items()
        )

        # Add bias
        logit += -2.5  # Base conceding probability ~8%

        return self._sigmoid(logit)

    def _extract_score_features(self, state: GameState) -> Dict[str, float]:
        """Extract features for scoring probability model."""
        features = {}

        # Most recent action
        latest_action = state.action_types[-1] if state.action_types else None
        latest_x = state.end_x[-1] if state.end_x else 0.0
        latest_y = state.end_y[-1] if state.end_y else 0.0

        # Location features
        features['distance_to_goal'] = state.distance_to_goal[-1] if state.distance_to_goal else 1.0
        features['angle_to_goal'] = state.angle_to_goal[-1] if state.angle_to_goal else 0.0
        features['in_slot'] = 1.0 if self._in_slot(latest_x, latest_y) else 0.0
        features['in_crease'] = 1.0 if self._in_crease(latest_x, latest_y) else 0.0
        features['behind_net'] = 1.0 if state.zone == Zone.BEHIND_NET else 0.0

        # Action type features
        features['shot'] = 1.0 if latest_action in [ActionType.SHOT, ActionType.SHOT_SAVED, ActionType.GOAL] else 0.0
        features['pass_to_slot'] = 1.0 if (
            latest_action == ActionType.PASS and
            self._in_slot(latest_x, latest_y)
        ) else 0.0
        features['carry_forward'] = 1.0 if (
            latest_action == ActionType.CARRY and
            state.distance_traveled[-1] > 0.1
        ) else 0.0
        features['dump'] = 1.0 if latest_action == ActionType.DUMP else 0.0
        features['turnover'] = 1.0 if latest_action in [ActionType.GIVEAWAY, ActionType.TURNOVER] else 0.0
        features['faceoff_win_oz'] = 1.0 if (
            latest_action == ActionType.FACEOFF_WIN and
            state.zone == Zone.OFFENSIVE
        ) else 0.0

        # Sequence features
        pass_count = sum(1 for a in state.action_types if a == ActionType.PASS)
        features['consecutive_passes'] = min(pass_count, 5) / 5.0
        features['zone_time'] = 1.0 if state.zone == Zone.OFFENSIVE else 0.0
        features['odd_man_rush'] = 1.0 if state.is_counter_attack else 0.0

        # Context features
        features['power_play'] = 1.0 if '5v4' in state.strength_state or '5v3' in state.strength_state else 0.0
        features['score_close'] = 1.0 if abs(state.score_differential) <= 1 else 0.0
        features['third_period'] = 1.0 if state.period == 3 else 0.0

        return features

    def _extract_concede_features(self, state: GameState) -> Dict[str, float]:
        """Extract features for conceding probability model."""
        features = {}

        latest_action = state.action_types[-1] if state.action_types else None
        latest_x = state.start_x[-1] if state.start_x else 0.0

        # Location features (flip coordinates - own zone is dangerous for conceding)
        features['distance_from_own_goal'] = abs(latest_x - (-1.0))
        features['in_own_slot'] = 1.0 if self._in_slot(-latest_x, state.start_y[-1] if state.start_y else 0) else 0.0
        features['in_own_crease'] = 1.0 if self._in_crease(-latest_x, state.start_y[-1] if state.start_y else 0) else 0.0

        # Action type features
        features['giveaway'] = 1.0 if latest_action == ActionType.GIVEAWAY else 0.0
        features['failed_clear'] = 1.0 if (
            latest_action == ActionType.CLEAR and
            state.action_results[-1] == ActionResult.FAIL
        ) else 0.0
        features['icing'] = 1.0 if latest_action == ActionType.ICING else 0.0
        features['turnover_in_dz'] = 1.0 if (
            latest_action in [ActionType.GIVEAWAY, ActionType.TURNOVER] and
            state.zone == Zone.DEFENSIVE
        ) else 0.0
        features['blocked_shot'] = 1.0 if latest_action == ActionType.BLOCK else 0.0
        features['takeaway'] = 1.0 if latest_action == ActionType.TAKEAWAY else 0.0

        # Sequence features
        features['under_pressure'] = sum(1 for a in state.action_results if a == ActionResult.FAIL) / len(state.action_results) if state.action_results else 0.0
        features['extended_dz_time'] = 1.0 if state.zone == Zone.DEFENSIVE else 0.0

        # Context features
        features['penalty_kill'] = 1.0 if '4v5' in state.strength_state or '3v5' in state.strength_state else 0.0
        features['trailing'] = 1.0 if state.score_differential < 0 else 0.0

        return features

    def _distance_to_goal(self, x: float, y: float) -> float:
        """Calculate distance to opponent goal (normalized)."""
        # Goal at x=1.0, y=0.0 (center)
        return np.sqrt((x - 1.0)**2 + y**2)

    def _angle_to_goal(self, x: float, y: float) -> float:
        """Calculate angle to goal (0 = straight on, 1 = extreme angle)."""
        if x >= 1.0:
            return 0.0
        angle = np.abs(np.arctan2(y, 1.0 - x))
        return angle / (np.pi / 2)  # Normalize to 0-1

    def _determine_zone(self, x: float) -> Zone:
        """Determine ice zone from x coordinate."""
        if x > 0.7:
            return Zone.OFFENSIVE
        elif x > 0.85:
            return Zone.BEHIND_NET
        elif x > 0.3:
            return Zone.NEUTRAL
        else:
            return Zone.DEFENSIVE

    def _in_slot(self, x: float, y: float) -> bool:
        """Check if position is in the slot."""
        return x > 0.75 and abs(y) < 0.25

    def _in_crease(self, x: float, y: float) -> bool:
        """Check if position is in the crease."""
        return x > 0.90 and abs(y) < 0.10

    def _detect_counter_attack(self, actions: List[HADLAction]) -> bool:
        """Detect if current play is a counter-attack/rush."""
        if len(actions) < 2:
            return False

        # Check for rapid zone transition
        first = actions[0]
        last = actions[-1]

        x_change = (last.start_x or 0) - (first.start_x or 0)
        time_change = last.time_seconds - first.time_seconds

        if time_change <= 0:
            return False

        # Fast forward movement
        speed = x_change / time_change
        return speed > 0.15 and x_change > 0.3

    def _estimate_puck_speed(self, actions: List[HADLAction]) -> float:
        """Estimate puck movement speed."""
        if len(actions) < 2:
            return 0.0

        total_dist = 0.0
        total_time = 0.0

        for i in range(1, len(actions)):
            dx = (actions[i].start_x or 0) - (actions[i-1].start_x or 0)
            dy = (actions[i].start_y or 0) - (actions[i-1].start_y or 0)
            total_dist += np.sqrt(dx**2 + dy**2)
            total_time += actions[i].time_seconds - actions[i-1].time_seconds

        return total_dist / total_time if total_time > 0 else 0.0

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _empty_game_state(self) -> GameState:
        """Create empty game state."""
        return GameState(
            action_types=[],
            action_results=[],
            start_x=[],
            start_y=[],
            end_x=[],
            end_y=[],
            distance_to_goal=[],
            angle_to_goal=[],
            distance_traveled=[],
            strength_state="5v5",
            score_differential=0,
            period=1,
            time_remaining=1200.0,
            zone=Zone.NEUTRAL,
            is_counter_attack=False,
            puck_speed=0.0
        )

    def _create_dummy_action(self) -> HADLAction:
        """Create a dummy action for padding."""
        return HADLAction(
            action_id="dummy",
            game_id="",
            period=1,
            time_seconds=0.0,
            action_type=ActionType.LINE_CHANGE,
            result=ActionResult.SUCCESS,
            player_id="",
            team_id="",
            start_x=0.0,
            start_y=0.0
        )


class VAEPAccumulator:
    """
    Accumulate VAEP values for players over time.

    Tracks total offensive and defensive contributions.
    """

    def __init__(self):
        self.model = HockeyVAEP()
        self.player_values: Dict[str, Dict[str, float]] = {}
        self.action_history: List[HADLAction] = []

    def process_action(self, action: HADLAction) -> VAEPValue:
        """
        Process an action and accumulate VAEP for the player.

        Args:
            action: The action to process

        Returns:
            VAEPValue for this action
        """
        # Compute states
        state_before = self.model.compute_game_state(self.action_history[-3:])
        self.action_history.append(action)
        state_after = self.model.compute_game_state(self.action_history[-3:])

        # Compute VAEP
        vaep = self.model.compute_vaep(action, state_before, state_after)

        # Accumulate for player
        player_id = action.player_id
        if player_id not in self.player_values:
            self.player_values[player_id] = {
                'total': 0.0,
                'offensive': 0.0,
                'defensive': 0.0,
                'actions': 0
            }

        self.player_values[player_id]['total'] += vaep.total_value
        self.player_values[player_id]['offensive'] += vaep.offensive_value
        self.player_values[player_id]['defensive'] += vaep.defensive_value
        self.player_values[player_id]['actions'] += 1

        return vaep

    def get_player_rankings(self, min_actions: int = 10) -> List[Dict]:
        """
        Get player rankings by VAEP.

        Args:
            min_actions: Minimum actions to qualify

        Returns:
            List of player dictionaries sorted by total VAEP
        """
        rankings = []
        for player_id, values in self.player_values.items():
            if values['actions'] >= min_actions:
                rankings.append({
                    'player_id': player_id,
                    'total_vaep': values['total'],
                    'offensive_vaep': values['offensive'],
                    'defensive_vaep': values['defensive'],
                    'actions': values['actions'],
                    'vaep_per_action': values['total'] / values['actions']
                })

        return sorted(rankings, key=lambda x: -x['total_vaep'])

    def get_action_type_breakdown(self, player_id: str) -> Dict[str, Dict[str, float]]:
        """
        Get VAEP breakdown by action type for a player.

        Args:
            player_id: Player to analyze

        Returns:
            Dictionary mapping action types to VAEP values
        """
        breakdown: Dict[str, Dict[str, float]] = {}

        for action in self.action_history:
            if action.player_id != player_id:
                continue

            action_type = action.action_type.value
            if action_type not in breakdown:
                breakdown[action_type] = {
                    'total': 0.0,
                    'count': 0
                }

            # Would need to recompute VAEP for accurate breakdown
            # This is a simplified version
            breakdown[action_type]['count'] += 1

        return breakdown

    def reset(self):
        """Reset accumulator for new game/period."""
        self.player_values.clear()
        self.action_history.clear()


class VAEPxTComparison:
    """
    Compare VAEP and xT valuations.

    Based on Van Roy et al. (2020) analysis of differences
    between location-based (xT) and action-based (VAEP) models.
    """

    def __init__(self, vaep_model: HockeyVAEP, xt_values: np.ndarray):
        """
        Initialize comparison.

        Args:
            vaep_model: VAEP model instance
            xt_values: Pre-computed xT grid values
        """
        self.vaep = vaep_model
        self.xt = xt_values

    def compare_action_values(
        self,
        action: HADLAction,
        state_before: GameState,
        state_after: GameState
    ) -> Dict[str, float]:
        """
        Compare VAEP and xT values for an action.

        Returns:
            Dictionary with both valuations and difference
        """
        # VAEP value
        vaep_value = self.vaep.compute_vaep(action, state_before, state_after)

        # xT value (delta between end and start locations)
        start_xt = self._lookup_xt(action.start_x, action.start_y)
        end_xt = self._lookup_xt(action.end_x or action.start_x, action.end_y or action.start_y)
        xt_value = end_xt - start_xt

        return {
            'vaep': vaep_value.total_value,
            'vaep_offensive': vaep_value.offensive_value,
            'vaep_defensive': vaep_value.defensive_value,
            'xt': xt_value,
            'difference': vaep_value.total_value - xt_value,
            'action_type': action.action_type.value
        }

    def _lookup_xt(self, x: float, y: float) -> float:
        """Look up xT value for a position."""
        # Convert normalized coords to grid indices
        grid_x = int((x + 1) / 2 * (self.xt.shape[1] - 1))
        grid_y = int((y + 1) / 2 * (self.xt.shape[0] - 1))

        grid_x = np.clip(grid_x, 0, self.xt.shape[1] - 1)
        grid_y = np.clip(grid_y, 0, self.xt.shape[0] - 1)

        return self.xt[grid_y, grid_x]
