"""
Enhanced Goalie Save Difficulty Model for Hockey Analytics

This module implements an advanced goalie save difficulty model that goes
beyond standard GSAx (Goals Saved Above Expected) by incorporating:

- Goalie positioning pre-shot (square, depth, angle)
- Rebound management (secondary chance prevention)
- Screen quality (bodies in front)
- Shot sequence stress (rapid-fire vs. isolated)
- Movement difficulty (lateral, down-up, butterfly)

Inspired by baseball pitch framing research that revolutionized
catcher evaluation.

References:
- GSAx methodology from MoneyPuck, Evolving Hockey
- Baseball pitch framing research (Baseball Prospectus)
- Goaltending analytics research (LINHAC)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class ShotType(Enum):
    """Types of shots."""
    WRIST = "wrist"
    SLAP = "slap"
    SNAP = "snap"
    BACKHAND = "backhand"
    TIP = "tip"
    DEFLECTION = "deflection"
    WRAP = "wraparound"


class SaveType(Enum):
    """Types of saves a goalie can make."""
    GLOVE = "glove"
    BLOCKER = "blocker"
    PAD = "pad"
    BODY = "body"
    BUTTERFLY = "butterfly"
    SPRAWL = "sprawl"
    POKE_CHECK = "poke_check"
    STACK = "stack"


class MovementType(Enum):
    """Types of goalie movement required."""
    SQUARE = "square"           # No lateral movement
    LATERAL_SHORT = "lateral_short"  # <3 feet
    LATERAL_MEDIUM = "lateral_medium"  # 3-6 feet
    LATERAL_LONG = "lateral_long"    # >6 feet
    DOWN_UP = "down_up"         # Recovery from butterfly
    DEPTH_PUSH = "depth_push"   # Push out to challenge
    DEPTH_RETREAT = "depth_retreat"  # Back into net


@dataclass
class GoaliePosition:
    """Goalie position at time of shot."""
    x: float  # Position on ice (distance from goal line)
    y: float  # Position on ice (lateral)
    depth: float  # Distance from goal line (0 = in net, 6+ = aggressive)
    angle_to_puck: float  # Degrees from center (0 = square to puck)
    stance: str = "ready"  # ready, butterfly, split, sprawled

    # Tracking data (if available)
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    is_moving: bool = False
    movement_direction: Optional[str] = None


@dataclass
class ShotEvent:
    """Enhanced shot event data for goalie analysis."""
    shot_id: str
    timestamp: float
    game_id: str
    period: int

    # Shot characteristics
    shooter_id: str
    shot_type: ShotType
    shot_speed: Optional[float] = None  # mph

    # Shot location
    shot_x: float  # X coordinate of release
    shot_y: float  # Y coordinate of release
    shot_distance: float  # Distance from goal
    shot_angle: float  # Angle from center

    # Target location (where shot is going)
    target_x: float = 0.0  # Left/right in net
    target_z: float = 1.5  # Height (0=ice, 4=crossbar)

    # Goalie state
    goalie_id: str = ""
    goalie_position: Optional[GoaliePosition] = None

    # Pre-shot context
    time_since_last_shot: float = 10.0  # seconds
    shots_in_sequence: int = 1  # Number of rapid shots
    is_rebound: bool = False
    is_one_timer: bool = False

    # Screen/traffic
    screen_quality: float = 0.0  # 0-1 scale
    bodies_in_front: int = 0
    is_screened: bool = False

    # Outcome
    is_goal: bool = False
    is_save: bool = False
    save_type: Optional[SaveType] = None
    rebound_generated: bool = False

    # Computed values
    base_xG: Optional[float] = None
    adjusted_xG: Optional[float] = None
    save_difficulty: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalieProfile:
    """Profile of a goalie's performance."""
    goalie_id: str
    team_id: str

    # Basic stats
    shots_faced: int = 0
    goals_allowed: int = 0
    save_percentage: float = 0.0

    # xG-based metrics
    expected_goals: float = 0.0
    goals_saved_above_expected: float = 0.0  # GSAx

    # Enhanced metrics
    positioning_score: float = 0.0
    rebound_control: float = 0.0
    screen_navigation: float = 0.0
    lateral_movement: float = 0.0
    recovery_speed: float = 0.0
    high_danger_save_pct: float = 0.0

    # Situational
    avg_depth: float = 3.0
    avg_angle_error: float = 0.0
    sequence_save_pct: float = 0.0  # Multi-shot sequences

    # By shot type
    save_pct_by_type: Dict[str, float] = field(default_factory=dict)
    xG_by_movement: Dict[str, float] = field(default_factory=dict)


class SaveDifficultyModel:
    """
    Advanced save difficulty model.

    Calculates the true difficulty of saves by accounting for:
    1. Base xG (shot location, type)
    2. Goalie positioning adjustment
    3. Screen/traffic adjustment
    4. Shot sequence adjustment
    5. Movement requirement adjustment
    """

    def __init__(
        self,
        optimal_depth: float = 4.0,  # Optimal depth from goal line (feet)
        angle_tolerance: float = 5.0,  # Acceptable angle error (degrees)
    ):
        """
        Initialize the save difficulty model.

        Args:
            optimal_depth: Optimal goalie depth for average shot
            angle_tolerance: Acceptable angle deviation
        """
        self.optimal_depth = optimal_depth
        self.angle_tolerance = angle_tolerance

        # Base xG by shot location (simplified)
        self.distance_xG_curve = {
            5: 0.35,   # 5 feet = high danger
            10: 0.25,
            15: 0.15,
            20: 0.10,
            30: 0.06,
            40: 0.04,
            50: 0.02,
        }

        # Modifiers
        self.modifiers = {
            'screen': 0.15,          # Per screen level
            'rebound': 0.10,         # Rebound bonus
            'one_timer': 0.08,       # One-timer bonus
            'sequence_decay': 0.05,  # Per shot in sequence
            'lateral_short': 0.02,
            'lateral_medium': 0.06,
            'lateral_long': 0.12,
            'down_up': 0.08,
        }

        # Shot type multipliers
        self.shot_type_multiplier = {
            ShotType.WRIST: 1.0,
            ShotType.SLAP: 1.1,
            ShotType.SNAP: 1.05,
            ShotType.BACKHAND: 0.85,
            ShotType.TIP: 1.15,
            ShotType.DEFLECTION: 1.20,
            ShotType.WRAP: 0.70,
        }

    def calculate_base_xG(self, shot: ShotEvent) -> float:
        """Calculate base xG from shot characteristics."""
        # Distance-based xG
        base_xG = 0.05  # Default

        for dist, xg in sorted(self.distance_xG_curve.items()):
            if shot.shot_distance <= dist:
                base_xG = xg
                break

        # Angle adjustment (shots from better angles are more dangerous)
        angle_factor = 1.0 - abs(shot.shot_angle) / 90.0 * 0.3
        base_xG *= angle_factor

        # Shot type multiplier
        type_mult = self.shot_type_multiplier.get(shot.shot_type, 1.0)
        base_xG *= type_mult

        return base_xG

    def calculate_positioning_adjustment(
        self,
        shot: ShotEvent,
        goalie_pos: GoaliePosition
    ) -> float:
        """
        Calculate xG adjustment based on goalie positioning.

        Good positioning reduces xG, poor positioning increases it.

        Returns:
            Adjustment factor (negative = good positioning, positive = poor)
        """
        adjustment = 0.0

        # Depth adjustment
        # Optimal depth depends on shot distance
        optimal_depth_for_shot = min(shot.shot_distance * 0.15, 6.0)
        depth_error = abs(goalie_pos.depth - optimal_depth_for_shot)

        if depth_error > 2.0:
            # Too deep or too aggressive
            adjustment += depth_error * 0.02

        # Angle adjustment
        # Goalie should be square to puck (angle_to_puck near 0)
        angle_error = abs(goalie_pos.angle_to_puck)

        if angle_error > self.angle_tolerance:
            adjustment += (angle_error - self.angle_tolerance) * 0.01

        # Movement state adjustment
        if goalie_pos.is_moving:
            adjustment += 0.03  # Harder to save while moving

        if goalie_pos.stance == "butterfly":
            # Low shots easier, high shots harder
            if shot.target_z > 2.5:
                adjustment += 0.05
            else:
                adjustment -= 0.02
        elif goalie_pos.stance == "sprawled":
            adjustment += 0.08  # Very hard to save when sprawled

        return adjustment

    def calculate_screen_adjustment(self, shot: ShotEvent) -> float:
        """Calculate xG adjustment for screens/traffic."""
        if not shot.is_screened and shot.bodies_in_front == 0:
            return 0.0

        adjustment = shot.screen_quality * self.modifiers['screen']

        # Additional adjustment for bodies in front
        adjustment += shot.bodies_in_front * 0.02

        return adjustment

    def calculate_sequence_adjustment(self, shot: ShotEvent) -> float:
        """Calculate xG adjustment for shot sequences."""
        adjustment = 0.0

        # Rapid shots are harder
        if shot.time_since_last_shot < 2.0:
            adjustment += self.modifiers['sequence_decay'] * shot.shots_in_sequence

        # Rebound shots
        if shot.is_rebound:
            adjustment += self.modifiers['rebound']

        # One-timers
        if shot.is_one_timer:
            adjustment += self.modifiers['one_timer']

        return adjustment

    def calculate_movement_adjustment(
        self,
        shot: ShotEvent,
        goalie_pos: GoaliePosition
    ) -> float:
        """Calculate xG adjustment for required goalie movement."""
        adjustment = 0.0

        if goalie_pos.movement_direction:
            direction = goalie_pos.movement_direction

            if direction == "lateral_short":
                adjustment += self.modifiers['lateral_short']
            elif direction == "lateral_medium":
                adjustment += self.modifiers['lateral_medium']
            elif direction == "lateral_long":
                adjustment += self.modifiers['lateral_long']
            elif direction == "down_up":
                adjustment += self.modifiers['down_up']

        return adjustment

    def calculate_save_difficulty(self, shot: ShotEvent) -> Dict[str, float]:
        """
        Calculate comprehensive save difficulty.

        Args:
            shot: ShotEvent with all relevant data

        Returns:
            Dictionary with base_xG, adjustments, and final difficulty
        """
        base_xG = self.calculate_base_xG(shot)

        # Adjustments
        adjustments = {}

        if shot.goalie_position:
            adjustments['positioning'] = self.calculate_positioning_adjustment(
                shot, shot.goalie_position
            )
            adjustments['movement'] = self.calculate_movement_adjustment(
                shot, shot.goalie_position
            )
        else:
            adjustments['positioning'] = 0.0
            adjustments['movement'] = 0.0

        adjustments['screen'] = self.calculate_screen_adjustment(shot)
        adjustments['sequence'] = self.calculate_sequence_adjustment(shot)

        # Total adjustment
        total_adjustment = sum(adjustments.values())

        # Adjusted xG (save difficulty)
        adjusted_xG = base_xG + total_adjustment
        adjusted_xG = np.clip(adjusted_xG, 0.01, 0.95)  # Reasonable bounds

        return {
            'base_xG': base_xG,
            'adjustments': adjustments,
            'total_adjustment': total_adjustment,
            'save_difficulty': adjusted_xG,
        }


class ReboundControlModel:
    """
    Model for evaluating goalie rebound control.

    Measures:
    - Rebound frequency
    - Rebound distance (further is better)
    - Secondary chance prevention
    """

    def __init__(self):
        """Initialize the rebound control model."""
        self.rebound_events: List[Dict[str, Any]] = []

    def add_rebound_event(
        self,
        initial_shot: ShotEvent,
        rebound_distance: float,
        rebound_controlled: bool,
        secondary_chance: bool = False,
        secondary_goal: bool = False
    ):
        """Add a rebound event."""
        self.rebound_events.append({
            'shot_id': initial_shot.shot_id,
            'goalie_id': initial_shot.goalie_id,
            'rebound_distance': rebound_distance,
            'controlled': rebound_controlled,
            'secondary_chance': secondary_chance,
            'secondary_goal': secondary_goal,
        })

    def calculate_rebound_score(self, goalie_id: str) -> Dict[str, float]:
        """Calculate rebound control score for a goalie."""
        goalie_rebounds = [
            r for r in self.rebound_events if r['goalie_id'] == goalie_id
        ]

        if not goalie_rebounds:
            return {'insufficient_data': True}

        n = len(goalie_rebounds)

        # Rebound control rate
        controlled = sum(1 for r in goalie_rebounds if r['controlled'])
        control_rate = controlled / n

        # Secondary chance prevention
        secondary_chances = sum(1 for r in goalie_rebounds if r['secondary_chance'])
        secondary_rate = secondary_chances / n

        # Secondary goal prevention
        secondary_goals = sum(1 for r in goalie_rebounds if r['secondary_goal'])
        secondary_goal_rate = secondary_goals / max(secondary_chances, 1)

        # Average rebound distance
        avg_distance = np.mean([r['rebound_distance'] for r in goalie_rebounds])

        # Composite score (higher = better)
        # Weight: control rate, distance, prevention
        rebound_score = (
            0.4 * control_rate +
            0.3 * (1 - secondary_rate) +
            0.2 * (1 - secondary_goal_rate) +
            0.1 * min(avg_distance / 20.0, 1.0)
        )

        return {
            'rebound_control_rate': control_rate,
            'secondary_chance_rate': secondary_rate,
            'secondary_goal_rate': secondary_goal_rate,
            'avg_rebound_distance': avg_distance,
            'rebound_score': rebound_score,
            'sample_size': n,
        }


class GoalieAnalyzer:
    """
    Comprehensive goalie analysis system.

    Combines:
    - Save difficulty model
    - Rebound control
    - Positioning evaluation
    - Movement analysis
    """

    def __init__(self):
        """Initialize the goalie analyzer."""
        self.save_model = SaveDifficultyModel()
        self.rebound_model = ReboundControlModel()

        self.shots: List[ShotEvent] = []
        self.goalie_profiles: Dict[str, GoalieProfile] = {}

    def add_shot(self, shot: ShotEvent):
        """Add a shot event to the analyzer."""
        # Calculate save difficulty
        difficulty = self.save_model.calculate_save_difficulty(shot)
        shot.base_xG = difficulty['base_xG']
        shot.adjusted_xG = difficulty['save_difficulty']
        shot.save_difficulty = difficulty['save_difficulty']

        self.shots.append(shot)

        # Update goalie profile
        self._update_profile(shot)

    def _update_profile(self, shot: ShotEvent):
        """Update goalie profile with shot data."""
        goalie_id = shot.goalie_id

        if not goalie_id:
            return

        if goalie_id not in self.goalie_profiles:
            self.goalie_profiles[goalie_id] = GoalieProfile(
                goalie_id=goalie_id,
                team_id=""
            )

        profile = self.goalie_profiles[goalie_id]

        # Update basic stats
        profile.shots_faced += 1
        if shot.is_goal:
            profile.goals_allowed += 1

        profile.save_percentage = (
            (profile.shots_faced - profile.goals_allowed) / profile.shots_faced
        )

        # Update xG stats
        if shot.adjusted_xG:
            profile.expected_goals += shot.adjusted_xG

        profile.goals_saved_above_expected = (
            profile.expected_goals - profile.goals_allowed
        )

    def build_full_profile(self, goalie_id: str) -> GoalieProfile:
        """Build comprehensive profile for a goalie."""
        goalie_shots = [s for s in self.shots if s.goalie_id == goalie_id]

        if not goalie_shots:
            return GoalieProfile(goalie_id=goalie_id, team_id="")

        profile = self.goalie_profiles.get(
            goalie_id,
            GoalieProfile(goalie_id=goalie_id, team_id="")
        )

        # Positioning score
        positioning_errors = []
        for shot in goalie_shots:
            if shot.goalie_position:
                error = abs(shot.goalie_position.angle_to_puck)
                positioning_errors.append(error)

        if positioning_errors:
            avg_error = np.mean(positioning_errors)
            profile.avg_angle_error = avg_error
            profile.positioning_score = max(0, 1 - avg_error / 30)  # 30 degrees max

        # High danger saves (shots from <15 feet)
        high_danger = [s for s in goalie_shots if s.shot_distance < 15]
        if high_danger:
            hd_saves = sum(1 for s in high_danger if not s.is_goal)
            profile.high_danger_save_pct = hd_saves / len(high_danger)

        # Save percentage by shot type
        for shot_type in ShotType:
            type_shots = [s for s in goalie_shots if s.shot_type == shot_type]
            if type_shots:
                type_saves = sum(1 for s in type_shots if not s.is_goal)
                profile.save_pct_by_type[shot_type.value] = type_saves / len(type_shots)

        # Sequence saves
        sequence_shots = [s for s in goalie_shots if s.shots_in_sequence > 1]
        if sequence_shots:
            seq_saves = sum(1 for s in sequence_shots if not s.is_goal)
            profile.sequence_save_pct = seq_saves / len(sequence_shots)

        # Rebound control
        rebound_stats = self.rebound_model.calculate_rebound_score(goalie_id)
        if 'rebound_score' in rebound_stats:
            profile.rebound_control = rebound_stats['rebound_score']

        # Average depth
        depths = [s.goalie_position.depth for s in goalie_shots if s.goalie_position]
        if depths:
            profile.avg_depth = np.mean(depths)

        return profile

    def compare_goalies(self, goalie_ids: List[str]) -> pd.DataFrame:
        """Compare multiple goalies."""
        comparisons = []

        for gid in goalie_ids:
            profile = self.build_full_profile(gid)
            comparisons.append({
                'goalie_id': gid,
                'shots_faced': profile.shots_faced,
                'save_pct': profile.save_percentage,
                'expected_goals': profile.expected_goals,
                'GSAx': profile.goals_saved_above_expected,
                'positioning_score': profile.positioning_score,
                'high_danger_sv%': profile.high_danger_save_pct,
                'rebound_control': profile.rebound_control,
                'sequence_sv%': profile.sequence_save_pct,
                'avg_depth': profile.avg_depth,
            })

        return pd.DataFrame(comparisons)

    def get_shot_breakdown(self, goalie_id: str) -> Dict[str, Any]:
        """Get detailed shot breakdown for a goalie."""
        goalie_shots = [s for s in self.shots if s.goalie_id == goalie_id]

        if not goalie_shots:
            return {'error': 'No data'}

        # By distance zone
        distance_zones = {
            'slot': (0, 15),
            'high_slot': (15, 30),
            'point': (30, 50),
            'perimeter': (50, 100),
        }

        breakdown = {}

        for zone_name, (min_d, max_d) in distance_zones.items():
            zone_shots = [
                s for s in goalie_shots
                if min_d <= s.shot_distance < max_d
            ]

            if zone_shots:
                saves = sum(1 for s in zone_shots if not s.is_goal)
                total_xG = sum(s.adjusted_xG or 0 for s in zone_shots)
                goals = sum(1 for s in zone_shots if s.is_goal)

                breakdown[zone_name] = {
                    'shots': len(zone_shots),
                    'saves': saves,
                    'goals': goals,
                    'save_pct': saves / len(zone_shots),
                    'total_xG': total_xG,
                    'gsax': total_xG - goals,
                }

        return breakdown


def create_sample_shots(n_shots: int = 200) -> List[ShotEvent]:
    """Create sample shot data for testing."""
    np.random.seed(42)

    shots = []

    for i in range(n_shots):
        # Random shot characteristics
        distance = np.random.uniform(5, 60)
        angle = np.random.uniform(-60, 60)

        # Goalie position
        goalie_pos = GoaliePosition(
            x=np.random.uniform(0, 6),
            y=np.random.uniform(-3, 3),
            depth=np.random.uniform(1, 6),
            angle_to_puck=np.random.uniform(-15, 15),
            stance=np.random.choice(['ready', 'butterfly', 'split']),
            is_moving=np.random.random() < 0.3,
        )

        # Shot outcome based on distance and randomness
        base_goal_prob = 0.25 - distance * 0.004
        is_goal = np.random.random() < max(base_goal_prob, 0.02)

        shot = ShotEvent(
            shot_id=f"shot_{i}",
            timestamp=i * 30.0,
            game_id="game_001",
            period=np.random.randint(1, 4),
            shooter_id=f"shooter_{np.random.randint(1, 20)}",
            shot_type=np.random.choice(list(ShotType)[:4]),
            shot_speed=np.random.uniform(50, 100),
            shot_x=200 - distance * np.cos(np.radians(angle)),
            shot_y=42.5 + distance * np.sin(np.radians(angle)),
            shot_distance=distance,
            shot_angle=angle,
            target_x=np.random.uniform(-2, 2),
            target_z=np.random.uniform(0.5, 3.5),
            goalie_id="goalie_1",
            goalie_position=goalie_pos,
            time_since_last_shot=np.random.uniform(1, 30),
            shots_in_sequence=np.random.randint(1, 4),
            is_rebound=np.random.random() < 0.15,
            screen_quality=np.random.uniform(0, 0.5),
            bodies_in_front=np.random.randint(0, 3),
            is_screened=np.random.random() < 0.3,
            is_goal=is_goal,
            is_save=not is_goal,
            rebound_generated=np.random.random() < 0.25,
        )

        shots.append(shot)

    return shots


if __name__ == "__main__":
    # Demo the goalie analysis system
    print("Creating Goalie Analyzer...")

    analyzer = GoalieAnalyzer()

    # Create sample data
    print("Generating sample shots...")
    shots = create_sample_shots(200)

    for shot in shots:
        analyzer.add_shot(shot)

    # Build profile
    print("\nGoalie Profile:")
    profile = analyzer.build_full_profile("goalie_1")

    print(f"  Shots faced: {profile.shots_faced}")
    print(f"  Save percentage: {profile.save_percentage:.3f}")
    print(f"  Expected goals: {profile.expected_goals:.2f}")
    print(f"  GSAx: {profile.goals_saved_above_expected:.2f}")
    print(f"  Positioning score: {profile.positioning_score:.3f}")
    print(f"  High danger SV%: {profile.high_danger_save_pct:.3f}")
    print(f"  Average depth: {profile.avg_depth:.2f}")

    # Shot breakdown
    print("\nShot Breakdown by Zone:")
    breakdown = analyzer.get_shot_breakdown("goalie_1")
    for zone, stats in breakdown.items():
        print(f"  {zone}:")
        print(f"    Shots: {stats['shots']}, SV%: {stats['save_pct']:.3f}, GSAx: {stats['gsax']:.2f}")

    # Save by shot type
    print("\nSave % by Shot Type:")
    for shot_type, sv_pct in profile.save_pct_by_type.items():
        print(f"  {shot_type}: {sv_pct:.3f}")

    # Example save difficulty calculation
    print("\nSample Save Difficulty Calculations:")
    for shot in shots[:3]:
        difficulty = analyzer.save_model.calculate_save_difficulty(shot)
        print(f"  Shot from {shot.shot_distance:.0f}ft:")
        print(f"    Base xG: {difficulty['base_xG']:.3f}")
        print(f"    Adjustments: {difficulty['adjustments']}")
        print(f"    Save difficulty: {difficulty['save_difficulty']:.3f}")
