"""
Micro-Action Evaluation (DeepHoops-inspired)

Implements methodology from:
- Sicilia, A., et al. (2019). "DeepHoops: Evaluating Micro-Actions in
  Basketball Using Deep Feature Representations." MIT Sloan.

Key concept: Evaluate individual micro-actions (movements, decisions)
at high temporal resolution rather than just discrete events.

Hockey translation:
- Shift-level player analysis with sub-second granularity
- Evaluate skating decisions, positioning choices
- Credit for "invisible" good plays (forcing turnovers, creating space)
- Deep learning on tracking data for action evaluation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime


class MicroActionType(Enum):
    """Types of micro-actions in hockey."""
    # Movement
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    CHANGE_DIRECTION = "change_direction"
    STOP = "stop"
    CROSSOVER = "crossover"

    # Offensive
    RECEIVE_PASS = "receive_pass"
    CARRY_PUCK = "carry_puck"
    PROTECT_PUCK = "protect_puck"
    CREATE_SPACE = "create_space"
    DRIVE_NET = "drive_net"
    CYCLE_LOW = "cycle_low"

    # Defensive
    GAP_CLOSE = "gap_close"
    STICK_CHECK = "stick_check"
    BODY_POSITION = "body_position"
    BLOCK_LANE = "block_lane"
    BACKCHECK = "backcheck"

    # Transitional
    SUPPORT_PUCK = "support_puck"
    OUTLET_OPTION = "outlet_option"
    LANE_FILL = "lane_fill"


class ActionContext(Enum):
    """Context in which action occurs."""
    OFFENSIVE_POSSESSION = "offensive_possession"
    DEFENSIVE_COVERAGE = "defensive_coverage"
    TRANSITION_OFFENSE = "transition_offense"
    TRANSITION_DEFENSE = "transition_defense"
    NEUTRAL_ZONE = "neutral_zone"
    POWER_PLAY = "power_play"
    PENALTY_KILL = "penalty_kill"


@dataclass
class TrackingFrame:
    """Single frame of tracking data."""
    timestamp: float
    player_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    has_puck: bool = False


@dataclass
class MicroAction:
    """Single micro-action detection."""
    action_type: MicroActionType
    start_time: float
    end_time: float
    player_id: str
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    context: ActionContext
    confidence: float
    raw_value: float = 0.0  # Computed value


@dataclass
class ActionSequence:
    """Sequence of micro-actions for a player."""
    player_id: str
    shift_start: float
    shift_end: float
    actions: List[MicroAction]
    total_value: float = 0.0
    context_breakdown: Dict[ActionContext, float] = field(default_factory=dict)


@dataclass
class ShiftEvaluation:
    """Complete shift evaluation for a player."""
    player_id: str
    shift_number: int
    duration: float
    total_value: float
    offensive_value: float
    defensive_value: float
    transition_value: float
    action_counts: Dict[MicroActionType, int]
    peak_value_moment: Tuple[float, str]  # (time, description)
    comparison_to_average: float  # vs. position average


@dataclass
class ModelConfig:
    """Configuration for micro-action model."""
    sequence_length: int = 25  # frames (1 second at 25 fps)
    hidden_dim: int = 128
    num_layers: int = 2
    action_dim: int = len(MicroActionType)
    context_dim: int = len(ActionContext)


class ActionDetector:
    """
    Detect micro-actions from tracking data.

    Uses acceleration/velocity patterns to identify actions.
    """

    def __init__(
        self,
        acceleration_threshold: float = 2.0,  # m/s^2
        direction_change_threshold: float = 45,  # degrees
        min_action_duration: float = 0.2  # seconds
    ):
        self.accel_thresh = acceleration_threshold
        self.direction_thresh = np.radians(direction_change_threshold)
        self.min_duration = min_action_duration

    def detect_actions(
        self,
        frames: List[TrackingFrame],
        context: ActionContext
    ) -> List[MicroAction]:
        """Detect all micro-actions in frame sequence."""
        actions = []

        # Movement actions
        actions.extend(self._detect_movement_actions(frames, context))

        # Context-specific actions
        if context in [ActionContext.OFFENSIVE_POSSESSION, ActionContext.POWER_PLAY]:
            actions.extend(self._detect_offensive_actions(frames, context))
        elif context in [ActionContext.DEFENSIVE_COVERAGE, ActionContext.PENALTY_KILL]:
            actions.extend(self._detect_defensive_actions(frames, context))
        else:
            actions.extend(self._detect_transition_actions(frames, context))

        # Sort by time
        actions.sort(key=lambda a: a.start_time)
        return actions

    def _detect_movement_actions(
        self,
        frames: List[TrackingFrame],
        context: ActionContext
    ) -> List[MicroAction]:
        """Detect movement-based micro-actions."""
        actions = []

        for i in range(len(frames) - 1):
            curr = frames[i]
            next_frame = frames[i + 1]

            # Calculate acceleration
            ax = next_frame.vx - curr.vx
            ay = next_frame.vy - curr.vy
            accel_mag = np.sqrt(ax**2 + ay**2)

            # Detect acceleration/deceleration
            if accel_mag > self.accel_thresh:
                # Direction of velocity
                if curr.vx != 0 or curr.vy != 0:
                    vel_dir = np.arctan2(curr.vy, curr.vx)
                    accel_dir = np.arctan2(ay, ax)
                    angle_diff = abs(vel_dir - accel_dir)

                    if angle_diff < np.pi / 4:  # Accelerating forward
                        action_type = MicroActionType.ACCELERATE
                    elif angle_diff > 3 * np.pi / 4:  # Decelerating
                        action_type = MicroActionType.DECELERATE
                    else:  # Direction change
                        action_type = MicroActionType.CHANGE_DIRECTION
                else:
                    action_type = MicroActionType.ACCELERATE

                actions.append(MicroAction(
                    action_type=action_type,
                    start_time=curr.timestamp,
                    end_time=next_frame.timestamp,
                    player_id=curr.player_id,
                    start_position=(curr.x, curr.y),
                    end_position=(next_frame.x, next_frame.y),
                    context=context,
                    confidence=min(accel_mag / self.accel_thresh, 1.0)
                ))

            # Detect stops
            curr_speed = np.sqrt(curr.vx**2 + curr.vy**2)
            next_speed = np.sqrt(next_frame.vx**2 + next_frame.vy**2)

            if curr_speed > 3 and next_speed < 0.5:
                actions.append(MicroAction(
                    action_type=MicroActionType.STOP,
                    start_time=curr.timestamp,
                    end_time=next_frame.timestamp,
                    player_id=curr.player_id,
                    start_position=(curr.x, curr.y),
                    end_position=(next_frame.x, next_frame.y),
                    context=context,
                    confidence=0.9
                ))

        return actions

    def _detect_offensive_actions(
        self,
        frames: List[TrackingFrame],
        context: ActionContext
    ) -> List[MicroAction]:
        """Detect offensive micro-actions."""
        actions = []

        puck_frames = [f for f in frames if f.has_puck]

        if puck_frames:
            # Puck carrier actions
            for i in range(len(puck_frames) - 1):
                curr = puck_frames[i]
                next_f = puck_frames[i + 1]

                # Carry detection
                actions.append(MicroAction(
                    action_type=MicroActionType.CARRY_PUCK,
                    start_time=curr.timestamp,
                    end_time=next_f.timestamp,
                    player_id=curr.player_id,
                    start_position=(curr.x, curr.y),
                    end_position=(next_f.x, next_f.y),
                    context=context,
                    confidence=0.95
                ))

                # Protect puck (low speed with puck)
                speed = np.sqrt(curr.vx**2 + curr.vy**2)
                if speed < 2:
                    actions.append(MicroAction(
                        action_type=MicroActionType.PROTECT_PUCK,
                        start_time=curr.timestamp,
                        end_time=next_f.timestamp,
                        player_id=curr.player_id,
                        start_position=(curr.x, curr.y),
                        end_position=(next_f.x, next_f.y),
                        context=context,
                        confidence=0.7
                    ))
        else:
            # Off-puck actions
            for i, frame in enumerate(frames[:-1]):
                # Create space (moving away from defenders)
                # Would need defender positions - simplified here
                speed = np.sqrt(frame.vx**2 + frame.vy**2)
                if speed > 5:  # Fast movement
                    actions.append(MicroAction(
                        action_type=MicroActionType.CREATE_SPACE,
                        start_time=frame.timestamp,
                        end_time=frames[i+1].timestamp,
                        player_id=frame.player_id,
                        start_position=(frame.x, frame.y),
                        end_position=(frames[i+1].x, frames[i+1].y),
                        context=context,
                        confidence=0.6
                    ))

        return actions

    def _detect_defensive_actions(
        self,
        frames: List[TrackingFrame],
        context: ActionContext
    ) -> List[MicroAction]:
        """Detect defensive micro-actions."""
        actions = []

        for i in range(len(frames) - 1):
            curr = frames[i]
            next_f = frames[i + 1]

            # Gap close (moving toward offensive zone)
            if curr.vx > 2:  # Moving toward offensive end
                actions.append(MicroAction(
                    action_type=MicroActionType.GAP_CLOSE,
                    start_time=curr.timestamp,
                    end_time=next_f.timestamp,
                    player_id=curr.player_id,
                    start_position=(curr.x, curr.y),
                    end_position=(next_f.x, next_f.y),
                    context=context,
                    confidence=0.7
                ))

            # Backcheck (moving toward defensive zone quickly)
            if curr.vx < -5:  # Fast retreat
                actions.append(MicroAction(
                    action_type=MicroActionType.BACKCHECK,
                    start_time=curr.timestamp,
                    end_time=next_f.timestamp,
                    player_id=curr.player_id,
                    start_position=(curr.x, curr.y),
                    end_position=(next_f.x, next_f.y),
                    context=context,
                    confidence=0.8
                ))

        return actions

    def _detect_transition_actions(
        self,
        frames: List[TrackingFrame],
        context: ActionContext
    ) -> List[MicroAction]:
        """Detect transitional micro-actions."""
        actions = []

        for i in range(len(frames) - 1):
            curr = frames[i]
            next_f = frames[i + 1]

            # Support puck (moving toward puck area)
            actions.append(MicroAction(
                action_type=MicroActionType.SUPPORT_PUCK,
                start_time=curr.timestamp,
                end_time=next_f.timestamp,
                player_id=curr.player_id,
                start_position=(curr.x, curr.y),
                end_position=(next_f.x, next_f.y),
                context=context,
                confidence=0.5
            ))

        return actions


class ActionValuator:
    """
    Assign value to micro-actions.

    Uses context and outcome to determine value.
    """

    def __init__(self):
        # Base values by action type
        self.base_values = {
            MicroActionType.ACCELERATE: 0.01,
            MicroActionType.DECELERATE: 0.005,
            MicroActionType.CHANGE_DIRECTION: 0.01,
            MicroActionType.STOP: 0.005,
            MicroActionType.CROSSOVER: 0.01,
            MicroActionType.RECEIVE_PASS: 0.02,
            MicroActionType.CARRY_PUCK: 0.015,
            MicroActionType.PROTECT_PUCK: 0.01,
            MicroActionType.CREATE_SPACE: 0.02,
            MicroActionType.DRIVE_NET: 0.03,
            MicroActionType.CYCLE_LOW: 0.015,
            MicroActionType.GAP_CLOSE: 0.02,
            MicroActionType.STICK_CHECK: 0.02,
            MicroActionType.BODY_POSITION: 0.015,
            MicroActionType.BLOCK_LANE: 0.025,
            MicroActionType.BACKCHECK: 0.025,
            MicroActionType.SUPPORT_PUCK: 0.015,
            MicroActionType.OUTLET_OPTION: 0.02,
            MicroActionType.LANE_FILL: 0.015,
        }

        # Context multipliers
        self.context_multipliers = {
            ActionContext.OFFENSIVE_POSSESSION: 1.2,
            ActionContext.DEFENSIVE_COVERAGE: 1.1,
            ActionContext.TRANSITION_OFFENSE: 1.3,
            ActionContext.TRANSITION_DEFENSE: 1.25,
            ActionContext.NEUTRAL_ZONE: 0.9,
            ActionContext.POWER_PLAY: 1.4,
            ActionContext.PENALTY_KILL: 1.5,
        }

    def value_action(
        self,
        action: MicroAction,
        game_state: Optional[Dict] = None
    ) -> float:
        """Calculate value of single micro-action."""
        base = self.base_values.get(action.action_type, 0.01)
        context_mult = self.context_multipliers.get(action.context, 1.0)

        # Position multiplier (dangerous areas worth more)
        pos_mult = self._position_multiplier(action.end_position)

        # Confidence weighting
        value = base * context_mult * pos_mult * action.confidence

        # Game state adjustments
        if game_state:
            # Trailing team benefits from offensive actions
            if game_state.get('trailing', False):
                if action.action_type in [
                    MicroActionType.CREATE_SPACE,
                    MicroActionType.DRIVE_NET,
                    MicroActionType.CARRY_PUCK
                ]:
                    value *= 1.2

        return value

    def _position_multiplier(self, position: Tuple[float, float]) -> float:
        """Higher value for actions in dangerous positions."""
        x, y = position

        # Slot area (high danger)
        if x > 20 and abs(y) < 10:
            return 1.5

        # Offensive zone
        if x > 25:
            return 1.2

        # Defensive zone
        if x < -25:
            return 1.1

        return 1.0

    def value_sequence(
        self,
        actions: List[MicroAction],
        game_state: Optional[Dict] = None
    ) -> float:
        """Calculate total value of action sequence."""
        total = 0.0
        for action in actions:
            action.raw_value = self.value_action(action, game_state)
            total += action.raw_value
        return total


class DeepActionEncoder:
    """
    Deep learning encoder for micro-action sequences.

    LSTM-based encoding of tracking sequences.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # LSTM weights (simplified)
        input_dim = 6  # x, y, vx, vy, ax, ay
        self.W_lstm = np.random.randn(input_dim, config.hidden_dim * 4) * 0.1
        self.U_lstm = np.random.randn(config.hidden_dim, config.hidden_dim * 4) * 0.1
        self.b_lstm = np.zeros(config.hidden_dim * 4)

        # Action classifier
        self.W_action = np.random.randn(config.hidden_dim, config.action_dim) * 0.1
        self.b_action = np.zeros(config.action_dim)

    def encode_sequence(
        self,
        frames: List[TrackingFrame]
    ) -> np.ndarray:
        """Encode frame sequence to hidden representation."""
        h = np.zeros(self.config.hidden_dim)
        c = np.zeros(self.config.hidden_dim)

        for frame in frames:
            x = np.array([
                frame.x / 100, frame.y / 50,  # Normalized position
                frame.vx / 10, frame.vy / 10,  # Normalized velocity
                frame.ax / 5, frame.ay / 5    # Normalized acceleration
            ])

            # LSTM step
            gates = x @ self.W_lstm + h @ self.U_lstm + self.b_lstm
            i, f, o, g = np.split(gates, 4)

            i = 1 / (1 + np.exp(-i))
            f = 1 / (1 + np.exp(-f))
            o = 1 / (1 + np.exp(-o))
            g = np.tanh(g)

            c = f * c + i * g
            h = o * np.tanh(c)

        return h

    def predict_action_type(
        self,
        encoding: np.ndarray
    ) -> Tuple[MicroActionType, float]:
        """Predict action type from encoding."""
        logits = encoding @ self.W_action + self.b_action
        probs = np.exp(logits) / np.sum(np.exp(logits))

        idx = np.argmax(probs)
        action_types = list(MicroActionType)

        return action_types[idx], probs[idx]


class ShiftAnalyzer:
    """
    Analyze complete player shifts using micro-actions.
    """

    def __init__(self):
        self.detector = ActionDetector()
        self.valuator = ActionValuator()
        self.position_averages: Dict[str, float] = {
            'forward': 0.15,
            'defense': 0.12,
            'center': 0.14
        }

    def analyze_shift(
        self,
        frames: List[TrackingFrame],
        player_position: str,
        context_sequence: List[ActionContext],
        game_state: Optional[Dict] = None
    ) -> ShiftEvaluation:
        """Analyze complete shift for a player."""
        if not frames:
            return ShiftEvaluation(
                player_id="unknown",
                shift_number=0,
                duration=0,
                total_value=0,
                offensive_value=0,
                defensive_value=0,
                transition_value=0,
                action_counts={},
                peak_value_moment=(0, "No data"),
                comparison_to_average=0
            )

        player_id = frames[0].player_id
        duration = frames[-1].timestamp - frames[0].timestamp

        # Detect all actions
        all_actions = []
        for context in set(context_sequence):
            context_frames = frames  # Simplified - would filter by time
            actions = self.detector.detect_actions(context_frames, context)
            all_actions.extend(actions)

        # Value actions
        total_value = self.valuator.value_sequence(all_actions, game_state)

        # Categorize values
        offensive_value = sum(
            a.raw_value for a in all_actions
            if a.context in [ActionContext.OFFENSIVE_POSSESSION, ActionContext.POWER_PLAY]
        )
        defensive_value = sum(
            a.raw_value for a in all_actions
            if a.context in [ActionContext.DEFENSIVE_COVERAGE, ActionContext.PENALTY_KILL]
        )
        transition_value = sum(
            a.raw_value for a in all_actions
            if a.context in [ActionContext.TRANSITION_OFFENSE, ActionContext.TRANSITION_DEFENSE]
        )

        # Action counts
        action_counts = {}
        for action in all_actions:
            action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1

        # Peak moment
        if all_actions:
            peak_action = max(all_actions, key=lambda a: a.raw_value)
            peak_moment = (peak_action.start_time, peak_action.action_type.value)
        else:
            peak_moment = (0, "No actions")

        # Comparison to average
        avg = self.position_averages.get(player_position, 0.13)
        comparison = (total_value - avg) / avg if avg > 0 else 0

        return ShiftEvaluation(
            player_id=player_id,
            shift_number=1,  # Would be passed in
            duration=duration,
            total_value=total_value,
            offensive_value=offensive_value,
            defensive_value=defensive_value,
            transition_value=transition_value,
            action_counts=action_counts,
            peak_value_moment=peak_moment,
            comparison_to_average=comparison
        )


class GameAnalyzer:
    """
    Analyze full game using micro-actions.
    """

    def __init__(self):
        self.shift_analyzer = ShiftAnalyzer()

    def analyze_game(
        self,
        player_shifts: Dict[str, List[List[TrackingFrame]]],
        player_positions: Dict[str, str],
        game_state: Optional[Dict] = None
    ) -> Dict[str, List[ShiftEvaluation]]:
        """Analyze all shifts for all players."""
        results = {}

        for player_id, shifts in player_shifts.items():
            player_results = []
            position = player_positions.get(player_id, 'forward')

            for i, shift_frames in enumerate(shifts):
                # Simplified context
                contexts = [ActionContext.OFFENSIVE_POSSESSION] * len(shift_frames)

                eval_result = self.shift_analyzer.analyze_shift(
                    shift_frames, position, contexts, game_state
                )
                eval_result.shift_number = i + 1
                player_results.append(eval_result)

            results[player_id] = player_results

        return results

    def get_player_summary(
        self,
        evaluations: List[ShiftEvaluation]
    ) -> Dict[str, float]:
        """Summarize player performance across shifts."""
        if not evaluations:
            return {}

        return {
            'total_value': sum(e.total_value for e in evaluations),
            'avg_shift_value': np.mean([e.total_value for e in evaluations]),
            'total_offensive': sum(e.offensive_value for e in evaluations),
            'total_defensive': sum(e.defensive_value for e in evaluations),
            'total_transition': sum(e.transition_value for e in evaluations),
            'total_shifts': len(evaluations),
            'total_ice_time': sum(e.duration for e in evaluations),
            'value_per_minute': sum(e.total_value for e in evaluations) / max(sum(e.duration for e in evaluations) / 60, 0.01)
        }
