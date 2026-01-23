"""
Transition and Counterattack Analysis for Hockey Analytics

This module implements models for analyzing transition play and odd-man rushes,
inspired by soccer counterattack detection research using Graph Neural Networks.

Key concepts:
- Odd-man rush quality prediction (2-on-1, 3-on-2)
- Transition speed (time from defensive zone exit to shot)
- Defensive recovery patterns
- Rush success probability modeling

Hockey-specific factors:
- Blue line offside rules affect transition timing
- Dump and chase vs controlled entry
- Speed through neutral zone
- Backchecking effectiveness

References:
- Soccer GNN counterattack research
- "Byline-to-byline speed" transition metrics
- Defending player feature importance studies
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class RushType(Enum):
    """Types of odd-man rushes."""
    BREAKAWAY = "breakaway"   # 1-on-0
    TWO_ON_ONE = "2_on_1"
    THREE_ON_TWO = "3_on_2"
    TWO_ON_TWO = "2_on_2"
    THREE_ON_THREE = "3_on_3"
    OUTNUMBERED = "outnumbered"  # More defenders
    EVEN = "even"


class RushOutcome(Enum):
    """Outcomes of a rush."""
    GOAL = "goal"
    SHOT_ON_GOAL = "shot_on_goal"
    SHOT_MISSED = "shot_missed"
    SHOT_BLOCKED = "shot_blocked"
    TURNOVER = "turnover"
    OFFSIDE = "offside"
    ICING = "icing"
    CLEARED = "cleared"
    CONTINUED_POSSESSION = "continued"


@dataclass
class RushEvent:
    """Represents a single transition/rush event."""
    rush_id: str
    game_id: str
    period: int
    timestamp: float

    # Rush classification
    rush_type: RushType
    attacking_team: str

    # Players involved
    attackers: List[str]
    defenders: List[str]
    puck_carrier: str

    # Spatial/temporal
    zone_exit_time: float
    zone_entry_time: float
    shot_time: Optional[float] = None
    transition_time: float = 0.0  # Time from exit to shot/end

    # Speed metrics
    avg_rush_speed: float = 0.0  # mph
    max_rush_speed: float = 0.0  # mph
    neutral_zone_time: float = 0.0  # seconds

    # Positions at entry
    entry_x: float = 175.0
    entry_y: float = 42.5
    entry_lane: str = "center"
    entry_type: str = "controlled"

    # Defender positions at entry
    defender_avg_depth: float = 0.0  # Average x-position of defenders
    defender_gap: float = 0.0  # Gap between attackers and defenders

    # Goalie state
    goalie_position_x: float = 5.0
    goalie_position_y: float = 42.5
    goalie_depth: float = 3.0

    # Outcome
    outcome: Optional[RushOutcome] = None
    shot_xG: Optional[float] = None
    is_goal: bool = False

    # Computed
    success_probability: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionSequence:
    """Sequence of events in a transition."""
    sequence_id: str
    rush_event: RushEvent
    tracking_frames: List[Dict[str, Any]] = field(default_factory=list)

    # Derived metrics
    speed_profile: List[float] = field(default_factory=list)
    defender_recovery_speeds: List[float] = field(default_factory=list)


class RushSuccessModel:
    """
    Model for predicting odd-man rush success probability.

    Based on research showing defender features have more impact
    than attacker features in counterattack success prediction.

    Key predictive features:
    1. Rush type (numerical advantage)
    2. Entry speed
    3. Defender gap/depth
    4. Goalie position
    5. Angle to goal
    """

    def __init__(self):
        """Initialize the rush success model."""
        # Base success rates by rush type
        self.base_rates = {
            RushType.BREAKAWAY: 0.33,
            RushType.TWO_ON_ONE: 0.25,
            RushType.THREE_ON_TWO: 0.20,
            RushType.TWO_ON_TWO: 0.12,
            RushType.THREE_ON_THREE: 0.15,
            RushType.EVEN: 0.10,
            RushType.OUTNUMBERED: 0.06,
        }

        # Feature weights (learned from data in practice)
        self.weights = {
            'speed_factor': 0.015,      # Per mph above average
            'gap_factor': 0.008,        # Per foot of defender gap
            'goalie_depth_factor': -0.02,  # Aggressive goalie reduces success
            'entry_angle_factor': -0.003,  # Per degree from center
            'transition_time_factor': -0.01,  # Slower transition = worse
        }

        # Track events for model refinement
        self.rush_history: List[RushEvent] = []

    def predict_success_probability(
        self,
        rush: RushEvent
    ) -> Dict[str, Any]:
        """
        Predict success probability for a rush.

        Args:
            rush: RushEvent with rush details

        Returns:
            Dictionary with probability and contributing factors
        """
        # Start with base rate
        base_prob = self.base_rates.get(rush.rush_type, 0.10)

        # Calculate adjustments
        adjustments = {}

        # Speed adjustment (faster = better)
        if rush.avg_rush_speed > 15:  # Average rush speed ~15 mph
            adjustments['speed'] = (rush.avg_rush_speed - 15) * self.weights['speed_factor']
        else:
            adjustments['speed'] = (rush.avg_rush_speed - 15) * self.weights['speed_factor'] * 1.5  # Bigger penalty

        # Defender gap adjustment (bigger gap = better)
        adjustments['gap'] = rush.defender_gap * self.weights['gap_factor']

        # Goalie depth adjustment (aggressive goalie = harder)
        if rush.goalie_depth > 3:  # Aggressive
            adjustments['goalie'] = (rush.goalie_depth - 3) * self.weights['goalie_depth_factor']
        else:
            adjustments['goalie'] = 0

        # Entry angle (center = best)
        entry_angle = abs(rush.entry_y - 42.5)
        adjustments['angle'] = entry_angle * self.weights['entry_angle_factor']

        # Transition time (faster = better)
        if rush.transition_time > 3:  # Slow transition
            adjustments['time'] = (rush.transition_time - 3) * self.weights['transition_time_factor']
        else:
            adjustments['time'] = (3 - rush.transition_time) * abs(self.weights['transition_time_factor'])

        # Calculate final probability
        total_adjustment = sum(adjustments.values())
        probability = base_prob + total_adjustment
        probability = np.clip(probability, 0.02, 0.80)

        return {
            'probability': float(probability),
            'base_rate': base_prob,
            'adjustments': adjustments,
            'total_adjustment': total_adjustment,
            'rush_type': rush.rush_type.value,
        }

    def add_rush(self, rush: RushEvent):
        """Add a rush event for tracking/learning."""
        # Calculate success probability
        prediction = self.predict_success_probability(rush)
        rush.success_probability = prediction['probability']

        self.rush_history.append(rush)

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get model calibration statistics."""
        if not self.rush_history:
            return {'error': 'No data'}

        # Group by predicted probability bins
        bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.0)]
        calibration = []

        for low, high in bins:
            bin_rushes = [
                r for r in self.rush_history
                if low <= (r.success_probability or 0) < high
            ]

            if bin_rushes:
                predicted = np.mean([r.success_probability for r in bin_rushes])
                actual = np.mean([1 if r.is_goal else 0 for r in bin_rushes])

                calibration.append({
                    'bin': f"{low:.1f}-{high:.1f}",
                    'count': len(bin_rushes),
                    'predicted': predicted,
                    'actual': actual,
                    'calibration_error': abs(predicted - actual),
                })

        return {
            'calibration': calibration,
            'total_rushes': len(self.rush_history),
            'total_goals': sum(1 for r in self.rush_history if r.is_goal),
        }


class TransitionSpeedAnalyzer:
    """
    Analyzes transition speed and identifies fast break opportunities.
    """

    def __init__(
        self,
        fast_transition_threshold: float = 3.0,  # seconds
        elite_speed_threshold: float = 20.0,  # mph
    ):
        """
        Initialize the analyzer.

        Args:
            fast_transition_threshold: Seconds for fast transition
            elite_speed_threshold: Speed threshold for elite transition
        """
        self.fast_threshold = fast_transition_threshold
        self.elite_speed = elite_speed_threshold

        self.transitions: List[RushEvent] = []

    def add_transition(self, rush: RushEvent):
        """Add a transition event."""
        self.transitions.append(rush)

    def analyze_transition(
        self,
        zone_exit_time: float,
        zone_entry_time: float,
        shot_time: Optional[float],
        speeds: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze a single transition sequence.

        Args:
            zone_exit_time: Time of defensive zone exit
            zone_entry_time: Time of offensive zone entry
            shot_time: Time of shot (if applicable)
            speeds: Speed readings during transition

        Returns:
            Analysis dictionary
        """
        neutral_zone_time = zone_entry_time - zone_exit_time

        if shot_time:
            total_transition = shot_time - zone_exit_time
        else:
            total_transition = zone_entry_time - zone_exit_time

        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0

        # Classify transition
        if total_transition < self.fast_threshold:
            classification = "fast_break"
        elif total_transition < 5:
            classification = "quick_transition"
        else:
            classification = "slow_transition"

        # Speed classification
        if avg_speed > self.elite_speed:
            speed_class = "elite"
        elif avg_speed > 15:
            speed_class = "above_average"
        elif avg_speed > 10:
            speed_class = "average"
        else:
            speed_class = "slow"

        return {
            'neutral_zone_time': neutral_zone_time,
            'total_transition_time': total_transition,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'classification': classification,
            'speed_class': speed_class,
            'is_fast_break': classification == "fast_break",
        }

    def get_player_transition_stats(
        self,
        player_id: str
    ) -> Dict[str, Any]:
        """Get transition statistics for a player."""
        player_transitions = [
            t for t in self.transitions if t.puck_carrier == player_id
        ]

        if not player_transitions:
            return {'player_id': player_id, 'transitions': 0}

        return {
            'player_id': player_id,
            'transitions': len(player_transitions),
            'avg_transition_time': np.mean([t.transition_time for t in player_transitions]),
            'avg_speed': np.mean([t.avg_rush_speed for t in player_transitions]),
            'fast_breaks': sum(1 for t in player_transitions if t.transition_time < self.fast_threshold),
            'goals': sum(1 for t in player_transitions if t.is_goal),
            'shots': sum(1 for t in player_transitions if t.outcome in [RushOutcome.SHOT_ON_GOAL, RushOutcome.GOAL]),
        }


class DefensiveRecoveryModel:
    """
    Models defensive recovery during transitions.

    Key metrics:
    - Backcheck speed
    - Gap closing rate
    - Recovery angles
    """

    def __init__(self):
        """Initialize the model."""
        self.recovery_events: List[Dict[str, Any]] = []

    def analyze_recovery(
        self,
        defender_positions: List[Dict[str, float]],  # [{x, y, vx, vy}]
        attacker_positions: List[Dict[str, float]],
        puck_position: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze defensive recovery effectiveness.

        Args:
            defender_positions: List of defender position dicts
            attacker_positions: List of attacker position dicts
            puck_position: Puck position dict

        Returns:
            Recovery analysis
        """
        # Calculate gaps
        defender_xs = [d['x'] for d in defender_positions]
        attacker_xs = [a['x'] for a in attacker_positions]

        avg_defender_x = np.mean(defender_xs) if defender_xs else 100
        avg_attacker_x = np.mean(attacker_xs) if attacker_xs else 150

        gap = avg_attacker_x - avg_defender_x

        # Recovery speeds (skating back toward own net)
        recovery_speeds = []
        for defender in defender_positions:
            # Negative vx means skating toward defensive zone
            if defender.get('vx', 0) < 0:
                recovery_speeds.append(abs(defender['vx']))

        avg_recovery_speed = np.mean(recovery_speeds) if recovery_speeds else 0

        # Angle coverage (are defenders between attackers and goal)
        coverage_score = 0
        for defender in defender_positions:
            for attacker in attacker_positions:
                # Is defender between attacker and goal?
                if defender['x'] < attacker['x']:
                    coverage_score += 0.2

        coverage_score = min(coverage_score, 1.0)

        return {
            'gap': gap,
            'avg_recovery_speed': avg_recovery_speed,
            'coverage_score': coverage_score,
            'num_defenders_recovering': len(recovery_speeds),
            'recovery_grade': 'good' if gap < 10 else 'fair' if gap < 20 else 'poor',
        }


def create_sample_rush() -> RushEvent:
    """Create sample rush data for testing."""
    np.random.seed(42)

    rush_type = np.random.choice(list(RushType), p=[0.1, 0.3, 0.25, 0.15, 0.1, 0.05, 0.05])

    if rush_type == RushType.BREAKAWAY:
        n_attackers, n_defenders = 1, 0
    elif rush_type == RushType.TWO_ON_ONE:
        n_attackers, n_defenders = 2, 1
    elif rush_type == RushType.THREE_ON_TWO:
        n_attackers, n_defenders = 3, 2
    else:
        n_attackers, n_defenders = np.random.randint(2, 4), np.random.randint(2, 4)

    transition_time = np.random.uniform(2, 8)
    avg_speed = np.random.uniform(12, 22)

    rush = RushEvent(
        rush_id=f"rush_{np.random.randint(1000)}",
        game_id="game_001",
        period=np.random.randint(1, 4),
        timestamp=np.random.uniform(0, 3600),
        rush_type=rush_type,
        attacking_team="home",
        attackers=[f"player_{i}" for i in range(n_attackers)],
        defenders=[f"defender_{i}" for i in range(n_defenders)],
        puck_carrier="player_0",
        zone_exit_time=0,
        zone_entry_time=transition_time * 0.4,
        shot_time=transition_time,
        transition_time=transition_time,
        avg_rush_speed=avg_speed,
        max_rush_speed=avg_speed + np.random.uniform(2, 5),
        neutral_zone_time=transition_time * 0.4,
        entry_y=np.random.uniform(20, 65),
        defender_avg_depth=np.random.uniform(160, 180),
        defender_gap=np.random.uniform(5, 25),
        goalie_position_x=np.random.uniform(3, 8),
        goalie_depth=np.random.uniform(2, 6),
    )

    # Determine outcome based on rush type
    goal_prob = {
        RushType.BREAKAWAY: 0.35,
        RushType.TWO_ON_ONE: 0.22,
        RushType.THREE_ON_TWO: 0.18,
        RushType.TWO_ON_TWO: 0.10,
    }.get(rush_type, 0.08)

    if np.random.random() < goal_prob:
        rush.outcome = RushOutcome.GOAL
        rush.is_goal = True
    elif np.random.random() < 0.6:
        rush.outcome = RushOutcome.SHOT_ON_GOAL
    else:
        rush.outcome = np.random.choice([
            RushOutcome.SHOT_MISSED,
            RushOutcome.TURNOVER,
            RushOutcome.SHOT_BLOCKED,
        ])

    return rush


if __name__ == "__main__":
    # Demo the transition analysis
    print("Creating Rush Success Model...")

    model = RushSuccessModel()

    # Generate sample rushes
    print("Generating sample rushes...")
    rushes = [create_sample_rush() for _ in range(200)]

    for rush in rushes:
        model.add_rush(rush)

    # Analyze a specific rush
    print("\nSample Rush Analysis:")
    sample_rush = create_sample_rush()
    prediction = model.predict_success_probability(sample_rush)

    print(f"  Rush type: {sample_rush.rush_type.value}")
    print(f"  Base rate: {prediction['base_rate']:.2%}")
    print(f"  Adjustments: {prediction['adjustments']}")
    print(f"  Final probability: {prediction['probability']:.2%}")

    # Calibration stats
    print("\nModel Calibration:")
    calibration = model.get_calibration_stats()
    for bin_stats in calibration.get('calibration', []):
        print(f"  {bin_stats['bin']}: "
              f"predicted={bin_stats['predicted']:.2%}, "
              f"actual={bin_stats['actual']:.2%}, "
              f"n={bin_stats['count']}")

    # Rush type breakdown
    print("\nRush Type Breakdown:")
    type_counts = defaultdict(lambda: {'total': 0, 'goals': 0})
    for rush in model.rush_history:
        type_counts[rush.rush_type.value]['total'] += 1
        if rush.is_goal:
            type_counts[rush.rush_type.value]['goals'] += 1

    for rush_type, stats in type_counts.items():
        goal_rate = stats['goals'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {rush_type}: {stats['total']} rushes, {goal_rate:.1%} goal rate")

    # Transition speed analysis
    print("\nTransition Speed Analysis:")
    speed_analyzer = TransitionSpeedAnalyzer()

    for rush in rushes:
        speed_analyzer.add_transition(rush)

    player_stats = speed_analyzer.get_player_transition_stats("player_0")
    print(f"  Transitions led: {player_stats.get('transitions', 0)}")
    print(f"  Avg transition time: {player_stats.get('avg_transition_time', 0):.2f}s")
    print(f"  Fast breaks: {player_stats.get('fast_breaks', 0)}")
    print(f"  Goals: {player_stats.get('goals', 0)}")

    # Defensive recovery
    print("\nDefensive Recovery Analysis:")
    recovery_model = DefensiveRecoveryModel()

    defenders = [
        {'x': 170, 'y': 30, 'vx': -15, 'vy': 0},
        {'x': 165, 'y': 55, 'vx': -12, 'vy': -3},
    ]
    attackers = [
        {'x': 180, 'y': 40, 'vx': 18, 'vy': 2},
        {'x': 175, 'y': 50, 'vx': 16, 'vy': -1},
    ]
    puck = {'x': 178, 'y': 42}

    recovery = recovery_model.analyze_recovery(defenders, attackers, puck)
    print(f"  Gap: {recovery['gap']:.1f} ft")
    print(f"  Recovery speed: {recovery['avg_recovery_speed']:.1f} ft/s")
    print(f"  Coverage score: {recovery['coverage_score']:.2f}")
    print(f"  Recovery grade: {recovery['recovery_grade']}")
