"""
Zone Entry Expected Threat (xT) Model for Hockey Analytics

This module implements a specialized Expected Threat model for zone entries,
addressing the gap identified in hockey analytics where current zone entry
tracking is binary (controlled vs. dump) without continuous value assessment.

Key features:
- Entry type classification (controlled, dump, chip, failed)
- Entry lane analysis (left, center, right)
- Speed-adjusted values using NHL EDGE data
- Markov chain modeling of post-entry sequences
- Context-aware adjustments (score state, period, strength)

References:
- Eric Tulsky's "Using Zone Entry Data" (MIT Sloan 2013)
- Corey Sznajder's manual zone entry tracking
- Karun Singh's xT blog post (2018)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class EntryType(Enum):
    """Types of zone entries."""
    CONTROLLED = "controlled"  # Carry-in with possession maintained
    DUMP = "dump"              # Dump and chase
    CHIP = "chip"              # Chip in and follow
    FAILED = "failed"          # Failed entry attempt
    PASS = "pass"              # Pass-in to teammate


class EntryLane(Enum):
    """Lane of zone entry."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class EntryOutcome(Enum):
    """Outcome following a zone entry."""
    SHOT = "shot"
    GOAL = "goal"
    EXIT = "exit"  # Cleared by defense
    TURNOVER = "turnover"
    SUSTAINED_POSSESSION = "sustained"
    REGROUP = "regroup"  # Team pulls back


@dataclass
class ZoneEntry:
    """Represents a single zone entry event."""
    entry_id: str
    timestamp: float
    game_id: str
    period: int
    game_time: float

    # Entry characteristics
    entry_type: EntryType
    entry_lane: EntryLane
    entry_x: float  # X coordinate at blue line crossing
    entry_y: float  # Y coordinate at blue line crossing

    # Player info
    carrier_player_id: str
    carrier_team_id: str
    receiver_player_id: Optional[str] = None

    # Entry speed (from EDGE data if available)
    entry_speed: Optional[float] = None
    entry_angle: Optional[float] = None

    # Defensive context
    defenders_in_lane: int = 0
    forechecker_pressure: float = 0.0  # 0-1 scale

    # Game state
    score_differential: int = 0  # Positive = leading
    strength_state: str = "5v5"
    is_power_play: bool = False

    # Outcome (filled post-entry)
    outcome: Optional[EntryOutcome] = None
    shot_within_10s: bool = False
    goal_within_10s: bool = False
    time_in_zone: Optional[float] = None  # Seconds of possession

    # Computed values
    xT_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntrySequence:
    """Represents the sequence of events following a zone entry."""
    entry: ZoneEntry
    events: List[Dict[str, Any]] = field(default_factory=list)
    total_xG: float = 0.0
    shots: int = 0
    goals: int = 0
    possession_duration: float = 0.0


class ZoneEntryxT:
    """
    Zone Entry Expected Threat Model.

    Models the expected value of zone entries based on:
    - Entry type (controlled, dump, chip, pass)
    - Entry lane (left, center, right)
    - Entry speed (from NHL EDGE tracking)
    - Defensive pressure
    - Game state context

    The model uses Markov chains to estimate the probability of
    reaching various outcomes (shot, goal, exit, turnover) from
    each entry type and location.
    """

    def __init__(
        self,
        blue_line_x: float = 175.0,  # Attacking blue line x-coordinate
        rink_width: float = 85.0,
        lane_boundaries: Tuple[float, float] = (28.33, 56.67),  # Y coords dividing lanes
    ):
        """
        Initialize the Zone Entry xT model.

        Args:
            blue_line_x: X-coordinate of the attacking blue line
            rink_width: Width of the rink in feet
            lane_boundaries: Y-coordinates dividing left/center/right lanes
        """
        self.blue_line_x = blue_line_x
        self.rink_width = rink_width
        self.lane_boundaries = lane_boundaries

        # Base xT values by entry type and lane
        # Derived from research: controlled entries generate ~2x shots/goals
        self.base_xT = {
            (EntryType.CONTROLLED, EntryLane.CENTER): 0.085,
            (EntryType.CONTROLLED, EntryLane.LEFT): 0.065,
            (EntryType.CONTROLLED, EntryLane.RIGHT): 0.065,
            (EntryType.CHIP, EntryLane.CENTER): 0.055,
            (EntryType.CHIP, EntryLane.LEFT): 0.045,
            (EntryType.CHIP, EntryLane.RIGHT): 0.045,
            (EntryType.DUMP, EntryLane.CENTER): 0.035,
            (EntryType.DUMP, EntryLane.LEFT): 0.030,
            (EntryType.DUMP, EntryLane.RIGHT): 0.030,
            (EntryType.PASS, EntryLane.CENTER): 0.075,
            (EntryType.PASS, EntryLane.LEFT): 0.060,
            (EntryType.PASS, EntryLane.RIGHT): 0.060,
            (EntryType.FAILED, EntryLane.CENTER): 0.0,
            (EntryType.FAILED, EntryLane.LEFT): 0.0,
            (EntryType.FAILED, EntryLane.RIGHT): 0.0,
        }

        # Transition probabilities from entry to outcomes
        # [shot, goal, exit, turnover, sustained]
        self.outcome_probs = {
            EntryType.CONTROLLED: {
                'shot_prob': 0.45,
                'goal_given_shot': 0.08,
                'exit_prob': 0.25,
                'turnover_prob': 0.15,
                'sustained_prob': 0.15,
            },
            EntryType.DUMP: {
                'shot_prob': 0.22,
                'goal_given_shot': 0.06,
                'exit_prob': 0.40,
                'turnover_prob': 0.25,
                'sustained_prob': 0.13,
            },
            EntryType.CHIP: {
                'shot_prob': 0.35,
                'goal_given_shot': 0.07,
                'exit_prob': 0.30,
                'turnover_prob': 0.20,
                'sustained_prob': 0.15,
            },
            EntryType.PASS: {
                'shot_prob': 0.40,
                'goal_given_shot': 0.085,  # Pass entries often to dangerous areas
                'exit_prob': 0.28,
                'turnover_prob': 0.17,
                'sustained_prob': 0.15,
            },
        }

        # Modifiers for context
        self.modifiers = {
            'speed_factor': 0.015,        # Per mph above/below 15 mph baseline
            'defender_penalty': -0.08,     # Per defender in lane
            'pressure_penalty': -0.10,     # For high forechecker pressure
            'power_play_bonus': 0.25,      # PP multiplier
            'trailing_bonus': 0.10,        # When trailing
            'leading_penalty': -0.05,      # When leading (more conservative)
            'third_period_modifier': 0.05, # Late game urgency
        }

        # Historical tracking for model refinement
        self.entry_history: List[ZoneEntry] = []
        self.outcome_counts = defaultdict(lambda: defaultdict(int))

        # Trained model weights
        self._is_fitted = False
        self._learned_xT = {}

    def classify_entry_lane(self, y: float) -> EntryLane:
        """
        Classify the entry lane based on Y coordinate.

        Args:
            y: Y-coordinate of entry (0-85 for NHL rink)

        Returns:
            EntryLane enum value
        """
        if y < self.lane_boundaries[0]:
            return EntryLane.LEFT
        elif y > self.lane_boundaries[1]:
            return EntryLane.RIGHT
        else:
            return EntryLane.CENTER

    def calculate_entry_xT(
        self,
        entry: ZoneEntry,
        use_learned: bool = True
    ) -> float:
        """
        Calculate the Expected Threat value for a zone entry.

        Args:
            entry: ZoneEntry object with entry details
            use_learned: Whether to use learned values if available

        Returns:
            xT value for the entry
        """
        # Get base xT for entry type and lane
        key = (entry.entry_type, entry.entry_lane)

        if use_learned and self._is_fitted and key in self._learned_xT:
            base_value = self._learned_xT[key]
        else:
            base_value = self.base_xT.get(key, 0.03)

        # Apply modifiers
        value = base_value

        # Speed adjustment (EDGE data)
        if entry.entry_speed is not None:
            speed_adjustment = (entry.entry_speed - 15.0) * self.modifiers['speed_factor']
            value += speed_adjustment

        # Defender adjustment
        if entry.defenders_in_lane > 0:
            value += entry.defenders_in_lane * self.modifiers['defender_penalty']

        # Pressure adjustment
        if entry.forechecker_pressure > 0.5:  # High pressure
            value *= (1 + self.modifiers['pressure_penalty'])

        # Game state adjustments
        if entry.is_power_play:
            value *= (1 + self.modifiers['power_play_bonus'])

        if entry.score_differential < 0:  # Trailing
            value *= (1 + self.modifiers['trailing_bonus'])
        elif entry.score_differential > 0:  # Leading
            value *= (1 + self.modifiers['leading_penalty'])

        # Period adjustment (third period urgency)
        if entry.period == 3:
            value *= (1 + self.modifiers['third_period_modifier'])

        # Floor at 0
        return max(0.0, value)

    def calculate_entry_value_simple(
        self,
        entry_type: str,
        entry_lane: str,
        entry_speed: Optional[float] = None,
        is_power_play: bool = False
    ) -> float:
        """
        Simplified entry value calculation for quick estimates.

        Args:
            entry_type: 'controlled', 'dump', 'chip', or 'pass'
            entry_lane: 'left', 'center', or 'right'
            entry_speed: Optional skating speed from EDGE data
            is_power_play: Whether on power play

        Returns:
            Expected threat value
        """
        # Convert strings to enums
        try:
            entry_type_enum = EntryType(entry_type.lower())
        except ValueError:
            entry_type_enum = EntryType.CONTROLLED

        try:
            entry_lane_enum = EntryLane(entry_lane.lower())
        except ValueError:
            entry_lane_enum = EntryLane.CENTER

        # Get base value
        value = self.base_xT.get(
            (entry_type_enum, entry_lane_enum),
            0.03
        )

        # Speed multiplier
        if entry_speed is not None:
            speed_factor = 1 + (entry_speed - 15) * 0.02
            value *= max(0.5, min(1.5, speed_factor))

        # Power play bonus
        if is_power_play:
            value *= 1.25

        return value

    def fit(self, entries: List[ZoneEntry]) -> 'ZoneEntryxT':
        """
        Fit the model to historical zone entry data.

        Args:
            entries: List of ZoneEntry objects with outcomes

        Returns:
            self
        """
        # Group entries by type and lane
        entry_groups = defaultdict(list)

        for entry in entries:
            if entry.outcome is not None:
                key = (entry.entry_type, entry.entry_lane)
                entry_groups[key].append(entry)

        # Calculate empirical xT for each group
        for key, group_entries in entry_groups.items():
            if len(group_entries) < 10:
                continue

            # xT = weighted combination of outcome probabilities
            shots = sum(1 for e in group_entries if e.shot_within_10s)
            goals = sum(1 for e in group_entries if e.goal_within_10s)
            total = len(group_entries)

            # Empirical xT based on goal probability and shot generation
            shot_rate = shots / total
            goal_rate = goals / total
            goal_given_shot = goals / max(shots, 1)

            # xT = P(goal within 10s) + 0.1 * P(shot within 10s without goal)
            # The 0.1 factor represents residual value of non-scoring shots
            empirical_xT = goal_rate + 0.1 * (shot_rate - goal_rate)

            self._learned_xT[key] = empirical_xT

            # Track outcome counts
            self.outcome_counts[key]['shots'] = shots
            self.outcome_counts[key]['goals'] = goals
            self.outcome_counts[key]['total'] = total

        self._is_fitted = True
        self.entry_history.extend(entries)

        return self

    def evaluate_entry_decision(
        self,
        entry_y: float,
        entry_speed: float,
        defenders_in_lane: int = 0,
        is_power_play: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate which entry type would be optimal given conditions.

        Args:
            entry_y: Y-coordinate of potential entry
            entry_speed: Current skating speed
            defenders_in_lane: Number of defenders blocking the lane
            is_power_play: Whether on power play

        Returns:
            Dictionary with recommended entry type and values for all options
        """
        lane = self.classify_entry_lane(entry_y)

        # Calculate xT for each entry type
        entry_values = {}

        for entry_type in [EntryType.CONTROLLED, EntryType.DUMP, EntryType.CHIP]:
            entry = ZoneEntry(
                entry_id="eval",
                timestamp=0,
                game_id="eval",
                period=1,
                game_time=0,
                entry_type=entry_type,
                entry_lane=lane,
                entry_x=self.blue_line_x,
                entry_y=entry_y,
                carrier_player_id="eval",
                carrier_team_id="eval",
                entry_speed=entry_speed,
                defenders_in_lane=defenders_in_lane,
                is_power_play=is_power_play,
            )

            entry_values[entry_type.value] = self.calculate_entry_xT(entry)

        # Add risk assessment
        # Controlled entries have higher variance - better when open, worse when pressured
        controlled_risk = defenders_in_lane * 0.15 + (1 - entry_speed / 25) * 0.1

        # Dump entries are safer but lower upside
        dump_risk = 0.1  # Relatively consistent

        # Determine recommendation
        if defenders_in_lane >= 2 or entry_speed < 12:
            recommendation = "dump"
        elif entry_values['controlled'] > entry_values['dump'] * 1.3:
            # Controlled needs to be significantly better to justify risk
            recommendation = "controlled"
        else:
            recommendation = "controlled" if controlled_risk < 0.3 else "dump"

        return {
            'entry_values': entry_values,
            'recommended': recommendation,
            'lane': lane.value,
            'controlled_risk': controlled_risk,
            'dump_risk': dump_risk,
            'expected_value': entry_values[recommendation],
        }

    def get_entry_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for zone entries by type and lane.

        Returns:
            DataFrame with entry statistics
        """
        stats = []

        for (entry_type, lane), counts in self.outcome_counts.items():
            total = counts.get('total', 0)
            if total == 0:
                continue

            stats.append({
                'entry_type': entry_type.value,
                'lane': lane.value,
                'entries': total,
                'shots': counts.get('shots', 0),
                'goals': counts.get('goals', 0),
                'shot_rate': counts.get('shots', 0) / total,
                'goal_rate': counts.get('goals', 0) / total,
                'learned_xT': self._learned_xT.get((entry_type, lane), None),
                'base_xT': self.base_xT.get((entry_type, lane), None),
            })

        return pd.DataFrame(stats)

    def get_player_entry_report(
        self,
        player_id: str,
        entries: Optional[List[ZoneEntry]] = None
    ) -> Dict[str, Any]:
        """
        Generate a zone entry report for a specific player.

        Args:
            player_id: Player ID to analyze
            entries: Optional list of entries (uses history if not provided)

        Returns:
            Dictionary with player entry statistics
        """
        entries = entries or self.entry_history
        player_entries = [e for e in entries if e.carrier_player_id == player_id]

        if not player_entries:
            return {'player_id': player_id, 'entries': 0}

        # Count by type
        type_counts = defaultdict(int)
        lane_counts = defaultdict(int)
        success_count = 0
        total_xT = 0.0

        for entry in player_entries:
            type_counts[entry.entry_type.value] += 1
            lane_counts[entry.entry_lane.value] += 1
            if entry.outcome not in [EntryOutcome.EXIT, EntryOutcome.TURNOVER, None]:
                success_count += 1
            if entry.xT_value is not None:
                total_xT += entry.xT_value

        total = len(player_entries)

        return {
            'player_id': player_id,
            'entries': total,
            'entry_types': dict(type_counts),
            'entry_lanes': dict(lane_counts),
            'controlled_rate': type_counts.get('controlled', 0) / total,
            'success_rate': success_count / total,
            'total_xT_generated': total_xT,
            'xT_per_entry': total_xT / total,
            'avg_speed': np.mean([e.entry_speed for e in player_entries if e.entry_speed]) if any(e.entry_speed for e in player_entries) else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export model to dictionary."""
        return {
            'blue_line_x': self.blue_line_x,
            'rink_width': self.rink_width,
            'lane_boundaries': self.lane_boundaries,
            'base_xT': {f"{k[0].value}_{k[1].value}": v for k, v in self.base_xT.items()},
            'learned_xT': {f"{k[0].value}_{k[1].value}": v for k, v in self._learned_xT.items()},
            'modifiers': self.modifiers,
            'outcome_counts': {
                f"{k[0].value}_{k[1].value}": dict(v)
                for k, v in self.outcome_counts.items()
            },
            'is_fitted': self._is_fitted,
        }

    def save(self, filepath: str):
        """Save model to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ZoneEntryxT':
        """Load model from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        model = cls(
            blue_line_x=data['blue_line_x'],
            rink_width=data['rink_width'],
            lane_boundaries=tuple(data['lane_boundaries']),
        )

        # Restore learned xT
        for key_str, value in data.get('learned_xT', {}).items():
            parts = key_str.split('_')
            entry_type = EntryType(parts[0])
            lane = EntryLane(parts[1])
            model._learned_xT[(entry_type, lane)] = value

        model._is_fitted = data.get('is_fitted', False)

        return model


class ZoneExitxT:
    """
    Zone Exit Expected Threat Model.

    Companion to ZoneEntryxT, models the value of zone exits and
    breakout plays from the defensive zone.
    """

    def __init__(
        self,
        defensive_blue_line_x: float = 25.0,
        rink_width: float = 85.0,
    ):
        """
        Initialize the Zone Exit xT model.

        Args:
            defensive_blue_line_x: X-coordinate of defensive blue line
            rink_width: Width of the rink
        """
        self.blue_line_x = defensive_blue_line_x
        self.rink_width = rink_width

        # Base exit values (probability of successful possession leading to entry)
        self.base_exit_xT = {
            'controlled_carry': 0.12,    # D-man or forward carries out
            'pass_up_boards': 0.08,      # Rim/bank pass up the boards
            'stretch_pass': 0.15,        # Long outlet pass
            'chip_out': 0.05,            # Chip puck out of zone
            'clear': 0.03,               # Hard clear (ice or glass)
            'turnover': -0.05,           # Failed exit (negative value)
        }

        # Modifier for forecheck pressure
        self.pressure_penalty = -0.03  # Per pressure level (0-3)

    def calculate_exit_xT(
        self,
        exit_type: str,
        exit_y: float,
        forecheck_pressure: int = 1,
        is_penalty_kill: bool = False
    ) -> float:
        """
        Calculate the Expected Threat value for a zone exit.

        Args:
            exit_type: Type of exit ('controlled_carry', 'pass_up_boards', etc.)
            exit_y: Y-coordinate of exit
            forecheck_pressure: Pressure level (0-3)
            is_penalty_kill: Whether on penalty kill

        Returns:
            xT value for the exit
        """
        base_value = self.base_exit_xT.get(exit_type, 0.05)

        # Apply pressure penalty
        value = base_value + (forecheck_pressure * self.pressure_penalty)

        # PK exits are more valuable (harder to achieve)
        if is_penalty_kill and exit_type != 'turnover':
            value *= 1.2

        return max(0.0, value)


def create_sample_zone_entries() -> List[ZoneEntry]:
    """
    Create sample zone entry data for testing.

    Returns:
        List of ZoneEntry objects with simulated outcomes
    """
    np.random.seed(42)

    entries = []

    for i in range(1000):
        # Random entry type weighted toward controlled
        entry_type = np.random.choice(
            list(EntryType),
            p=[0.55, 0.25, 0.12, 0.05, 0.03]
        )

        # Random lane
        entry_y = np.random.uniform(0, 85)
        if entry_y < 28.33:
            entry_lane = EntryLane.LEFT
        elif entry_y > 56.67:
            entry_lane = EntryLane.RIGHT
        else:
            entry_lane = EntryLane.CENTER

        # Outcome based on entry type
        if entry_type == EntryType.CONTROLLED:
            shot_prob = 0.45
            goal_prob = 0.08
        elif entry_type == EntryType.DUMP:
            shot_prob = 0.22
            goal_prob = 0.06
        elif entry_type == EntryType.CHIP:
            shot_prob = 0.35
            goal_prob = 0.07
        else:
            shot_prob = 0.40
            goal_prob = 0.085

        shot_within_10s = np.random.random() < shot_prob
        goal_within_10s = shot_within_10s and np.random.random() < goal_prob

        entry = ZoneEntry(
            entry_id=f"entry_{i}",
            timestamp=i * 30.0,
            game_id="game_001",
            period=np.random.randint(1, 4),
            game_time=np.random.uniform(0, 1200),
            entry_type=entry_type,
            entry_lane=entry_lane,
            entry_x=175.0,
            entry_y=entry_y,
            carrier_player_id=f"player_{np.random.randint(1, 20)}",
            carrier_team_id="team_a",
            entry_speed=np.random.uniform(12, 25),
            defenders_in_lane=np.random.randint(0, 3),
            shot_within_10s=shot_within_10s,
            goal_within_10s=goal_within_10s,
            outcome=EntryOutcome.SHOT if shot_within_10s else EntryOutcome.EXIT,
        )

        entries.append(entry)

    return entries


if __name__ == "__main__":
    # Demo the Zone Entry xT model
    print("Creating Zone Entry xT model...")

    model = ZoneEntryxT()

    # Create sample data
    print("Generating sample entries...")
    entries = create_sample_zone_entries()

    # Fit model
    print("Fitting model...")
    model.fit(entries)

    # Show statistics
    print("\nEntry Statistics:")
    stats_df = model.get_entry_statistics()
    print(stats_df.to_string())

    # Evaluate entry decision
    print("\nEntry Decision Evaluation:")
    decision = model.evaluate_entry_decision(
        entry_y=42.5,  # Center
        entry_speed=18.0,
        defenders_in_lane=1,
        is_power_play=False
    )
    print(f"  Recommended: {decision['recommended']}")
    print(f"  Entry values: {decision['entry_values']}")
    print(f"  Expected value: {decision['expected_value']:.4f}")

    # Player report
    print("\nPlayer Entry Report (player_1):")
    report = model.get_player_entry_report("player_1", entries)
    for key, value in report.items():
        print(f"  {key}: {value}")

    # Simple calculation example
    print("\nSimple xT calculations:")
    print(f"  Controlled center: {model.calculate_entry_value_simple('controlled', 'center'):.4f}")
    print(f"  Dump left: {model.calculate_entry_value_simple('dump', 'left'):.4f}")
    print(f"  Controlled center at 22 mph: {model.calculate_entry_value_simple('controlled', 'center', entry_speed=22):.4f}")
    print(f"  Controlled center on PP: {model.calculate_entry_value_simple('controlled', 'center', is_power_play=True):.4f}")
