"""
Shot Deception Metrics for Hockey Analytics

This module translates baseball pitch tunneling research to hockey,
measuring how well shooters disguise their shots to deceive goalies.

Key concepts from baseball pitch tunneling (Baseball Prospectus 2017):
- Making different pitches appear identical until "tunnel point"
- Release point consistency
- Trajectory overlap
- Late break differential

Hockey translation:
- Shot type disguise (wrist/slap/snap/backhand)
- Release point consistency
- Trajectory masking until late
- Puck flutter/movement effects

References:
- Baseball Prospectus tunneling research (2017)
- Dr. Barton Smith's seam-shifted wake research
- "Using baseball seams to alter pitch direction" (Smith & Smith 2021)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class ShotType(Enum):
    """Types of hockey shots."""
    WRIST = "wrist"
    SLAP = "slap"
    SNAP = "snap"
    BACKHAND = "backhand"
    TIP = "tip"
    DEFLECTION = "deflection"
    WRAP = "wraparound"


class ShootingHand(Enum):
    """Shooting handedness."""
    LEFT = "left"
    RIGHT = "right"


@dataclass
class ShotTrackingData:
    """
    Tracking data for a single shot.

    Captured from high-speed cameras or NHL EDGE puck tracking.
    """
    shot_id: str
    timestamp: float
    game_id: str

    # Shooter info
    shooter_id: str
    shooter_team: str
    shooting_hand: ShootingHand

    # Shot characteristics
    shot_type: ShotType
    shot_speed: float  # mph

    # Release point (relative to shooter position)
    release_x: float  # Lateral offset from body center
    release_y: float  # Forward offset from skates
    release_z: float  # Height from ice

    # Body mechanics at release
    stick_angle: float  # Degrees from vertical
    hip_rotation: float  # Degrees of rotation
    shoulder_angle: float  # Upper body lean
    weight_transfer: float  # 0-1 scale, back to front

    # Puck trajectory
    initial_velocity_x: float  # Horizontal component
    initial_velocity_y: float  # Forward component
    initial_velocity_z: float  # Vertical component
    launch_angle: float  # Degrees above ice

    # Puck characteristics
    spin_rate: Optional[float] = None  # RPM
    spin_axis: Optional[Tuple[float, float, float]] = None  # Unit vector
    flutter: bool = False  # Knuckling effect

    # Target location
    target_x: float = 0.0  # Net position (left/right)
    target_z: float = 0.0  # Net position (low/high)

    # Outcome
    is_goal: bool = False
    save_type: Optional[str] = None  # If saved
    goalie_reaction_time: Optional[float] = None  # ms

    # Computed metrics (filled by analysis)
    deception_score: Optional[float] = None
    tunnel_distance: Optional[float] = None
    late_break: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShooterProfile:
    """
    Profile of a shooter's tendencies and deception ability.
    """
    player_id: str
    shots_analyzed: int = 0

    # Average release point by shot type
    avg_release_points: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    # Release consistency (lower = more consistent = harder to read)
    release_consistency: Dict[str, float] = field(default_factory=dict)

    # Shot type distribution
    shot_type_distribution: Dict[str, float] = field(default_factory=dict)

    # Deception metrics
    overall_deception_score: float = 0.0
    tunnel_overlap_score: float = 0.0
    late_break_score: float = 0.0

    # Effectiveness
    avg_xG: float = 0.0
    actual_goal_rate: float = 0.0
    deception_xG_boost: float = 0.0  # Added xG from deception


class ShotDeceptionAnalyzer:
    """
    Analyzes shot deception using concepts from baseball pitch tunneling.

    The key insight is that goalies, like batters, have limited reaction time
    and must commit to a save decision before the puck/ball reaches them.
    Deceptive shooters make different shot types appear identical until
    it's too late for the goalie to adjust.

    Metrics computed:
    1. Release Point Consistency - How similar are release points across shot types
    2. Tunnel Distance - How far the puck travels before trajectory diverges
    3. Late Break - Amount of movement in final portion of trajectory
    4. Deception Score - Combined metric of shot disguise quality
    """

    def __init__(
        self,
        decision_distance: float = 35.0,  # ft from goalie where decision is made
        reaction_time: float = 0.175,      # seconds (175ms like baseball)
        min_shots_for_profile: int = 20,
    ):
        """
        Initialize the shot deception analyzer.

        Args:
            decision_distance: Distance from goalie where save decision must be made
            reaction_time: Goalie reaction time in seconds
            min_shots_for_profile: Minimum shots needed for reliable profile
        """
        self.decision_distance = decision_distance
        self.reaction_time = reaction_time
        self.min_shots = min_shots_for_profile

        # Shot data storage
        self.shots: List[ShotTrackingData] = []
        self.shooter_profiles: Dict[str, ShooterProfile] = {}

        # Reference values for "average" shots
        self.avg_release_by_type = {
            ShotType.WRIST: (0.0, 2.5, 1.5),    # x, y, z offsets
            ShotType.SLAP: (-1.0, 4.0, 0.5),
            ShotType.SNAP: (0.5, 2.0, 2.0),
            ShotType.BACKHAND: (-1.5, 1.5, 1.0),
        }

        # Typical shot speeds (mph)
        self.avg_speed_by_type = {
            ShotType.WRIST: 70.0,
            ShotType.SLAP: 95.0,
            ShotType.SNAP: 75.0,
            ShotType.BACKHAND: 55.0,
        }

    def add_shot(self, shot: ShotTrackingData):
        """Add a shot to the analyzer."""
        self.shots.append(shot)

        # Update shooter profile
        if shot.shooter_id not in self.shooter_profiles:
            self.shooter_profiles[shot.shooter_id] = ShooterProfile(
                player_id=shot.shooter_id
            )

    def calculate_tunnel_distance(
        self,
        shot1: ShotTrackingData,
        shot2: ShotTrackingData,
        divergence_threshold: float = 3.0  # inches
    ) -> float:
        """
        Calculate the tunnel distance between two shots.

        Tunnel distance is how far the puck travels before the trajectories
        of two different shots diverge enough for the goalie to distinguish them.

        Args:
            shot1: First shot trajectory
            shot2: Second shot trajectory
            divergence_threshold: Distance (inches) at which trajectories are distinguishable

        Returns:
            Distance in feet before trajectories diverge
        """
        # Simulate trajectory points at 1ms intervals
        dt = 0.001  # seconds
        max_time = 0.5  # seconds (max flight time)

        # Initial positions (release point)
        pos1 = np.array([shot1.release_x, shot1.release_y, shot1.release_z])
        pos2 = np.array([shot2.release_x, shot2.release_y, shot2.release_z])

        # Initial velocities (mph to ft/s)
        mph_to_fps = 1.467
        vel1 = np.array([
            shot1.initial_velocity_x * mph_to_fps,
            shot1.initial_velocity_y * mph_to_fps,
            shot1.initial_velocity_z * mph_to_fps
        ])
        vel2 = np.array([
            shot2.initial_velocity_x * mph_to_fps,
            shot2.initial_velocity_y * mph_to_fps,
            shot2.initial_velocity_z * mph_to_fps
        ])

        # Gravity and drag
        g = 32.2  # ft/s^2
        drag = 0.05  # Simplified drag coefficient

        tunnel_distance = 0.0
        t = 0.0

        while t < max_time:
            # Update positions
            pos1 = pos1 + vel1 * dt
            pos2 = pos2 + vel2 * dt

            # Update velocities (gravity + drag)
            vel1[2] -= g * dt
            vel2[2] -= g * dt
            vel1 *= (1 - drag * dt)
            vel2 *= (1 - drag * dt)

            # Calculate separation
            separation = np.linalg.norm(pos1 - pos2) * 12  # Convert to inches

            # Track distance traveled
            tunnel_distance += np.linalg.norm(vel1) * dt

            # Check if trajectories have diverged
            if separation > divergence_threshold:
                break

            t += dt

        return tunnel_distance

    def calculate_late_break(
        self,
        shot: ShotTrackingData,
        measurement_distance: float = 10.0  # Last 10 feet
    ) -> Tuple[float, float]:
        """
        Calculate the late movement (break) of a shot.

        Late break is the deviation from a straight line trajectory
        in the final portion of the shot's flight.

        Args:
            shot: Shot tracking data
            measurement_distance: Distance from target to measure break

        Returns:
            Tuple of (horizontal_break, vertical_break) in inches
        """
        # Simulate trajectory
        mph_to_fps = 1.467
        velocity = shot.shot_speed * mph_to_fps

        # Time to travel measurement_distance
        travel_time = measurement_distance / velocity

        # Calculate where puck "should" be (straight line)
        # vs where it actually ends up

        # Factors affecting late break:
        # 1. Spin-induced Magnus force
        # 2. Flutter (no-spin knuckling)
        # 3. Gravity drop

        horizontal_break = 0.0
        vertical_break = 0.0

        # Spin-induced movement (Magnus effect)
        if shot.spin_rate:
            # Higher spin = more predictable movement
            magnus_factor = shot.spin_rate / 1500.0  # Normalize
            if shot.spin_axis:
                # Spin axis determines direction of movement
                horizontal_break += magnus_factor * shot.spin_axis[0] * 6.0
                vertical_break += magnus_factor * shot.spin_axis[2] * 4.0

        # Flutter (knuckling effect - unpredictable)
        if shot.flutter:
            # Random-ish late movement
            np.random.seed(hash(shot.shot_id) % 2**32)
            horizontal_break += np.random.uniform(-4.0, 4.0)
            vertical_break += np.random.uniform(-2.0, 2.0)

        # Gravity drop in last segment
        gravity_drop = 0.5 * 32.2 * (travel_time ** 2) * 12  # inches
        vertical_break -= gravity_drop

        return (horizontal_break, vertical_break)

    def calculate_deception_score(
        self,
        shot: ShotTrackingData,
        shooter_profile: Optional[ShooterProfile] = None
    ) -> float:
        """
        Calculate overall deception score for a shot.

        Deception score combines:
        - Release point similarity to other shot types
        - Early trajectory similarity
        - Late break amount
        - Speed differential masking

        Score ranges from 0 (obvious) to 1 (highly deceptive).

        Args:
            shot: Shot to analyze
            shooter_profile: Optional profile for context

        Returns:
            Deception score (0-1)
        """
        deception_components = []

        # 1. Release point similarity (to other shot types)
        # More similar = harder for goalie to read shot type
        release_point = (shot.release_x, shot.release_y, shot.release_z)

        if shooter_profile and shooter_profile.avg_release_points:
            # Compare to other shot types this player uses
            other_types = [st for st in ShotType if st != shot.shot_type]
            min_distance = float('inf')

            for other_type in other_types:
                if other_type.value in shooter_profile.avg_release_points:
                    other_release = shooter_profile.avg_release_points[other_type.value]
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(release_point, other_release)))
                    min_distance = min(min_distance, distance)

            # Convert distance to similarity (closer = more deceptive)
            if min_distance < float('inf'):
                release_similarity = np.exp(-min_distance / 2.0)
                deception_components.append(release_similarity)
        else:
            # Compare to average release points
            for other_type, avg_release in self.avg_release_by_type.items():
                if other_type != shot.shot_type:
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(release_point, avg_release)))
                    similarity = np.exp(-distance / 2.0)
                    deception_components.append(similarity)

        # 2. Body mechanics disguise
        # Consistent mechanics across shot types increases deception
        mechanics_score = 0.5  # Baseline

        # Low hip rotation variance is good (all shots look similar)
        if abs(shot.hip_rotation) < 30:  # Less telegraphed
            mechanics_score += 0.2

        # Weight transfer timing
        if 0.3 < shot.weight_transfer < 0.7:  # Balanced
            mechanics_score += 0.1

        deception_components.append(mechanics_score)

        # 3. Late break contribution
        h_break, v_break = self.calculate_late_break(shot)
        total_break = np.sqrt(h_break ** 2 + v_break ** 2)

        # More break = more deceptive (within reason)
        break_score = min(total_break / 8.0, 1.0)  # Cap at 8 inches
        deception_components.append(break_score)

        # 4. Speed differential masking
        # If shot speed is between two shot types, harder to identify
        avg_speed = self.avg_speed_by_type.get(shot.shot_type, 70.0)
        speed_diff = abs(shot.shot_speed - avg_speed)

        # Unusual speed for shot type can be deceptive
        if speed_diff > 10:  # Significantly different
            speed_deception = min(speed_diff / 20.0, 0.5)
            deception_components.append(0.5 + speed_deception)

        # Combine components
        if deception_components:
            deception_score = np.mean(deception_components)
        else:
            deception_score = 0.5

        return float(np.clip(deception_score, 0.0, 1.0))

    def build_shooter_profile(self, player_id: str) -> ShooterProfile:
        """
        Build a comprehensive shooter profile.

        Args:
            player_id: Player ID to profile

        Returns:
            ShooterProfile with deception metrics
        """
        player_shots = [s for s in self.shots if s.shooter_id == player_id]

        if len(player_shots) < self.min_shots:
            return ShooterProfile(
                player_id=player_id,
                shots_analyzed=len(player_shots)
            )

        profile = ShooterProfile(player_id=player_id)
        profile.shots_analyzed = len(player_shots)

        # Shot type distribution
        type_counts = defaultdict(int)
        for shot in player_shots:
            type_counts[shot.shot_type.value] += 1

        total = len(player_shots)
        profile.shot_type_distribution = {
            st: count / total for st, count in type_counts.items()
        }

        # Average release points by type
        release_by_type = defaultdict(list)
        for shot in player_shots:
            release_by_type[shot.shot_type.value].append(
                (shot.release_x, shot.release_y, shot.release_z)
            )

        for shot_type, releases in release_by_type.items():
            if releases:
                avg_release = tuple(np.mean(releases, axis=0))
                profile.avg_release_points[shot_type] = avg_release

                # Calculate consistency (std dev of release points)
                if len(releases) > 1:
                    std_dev = np.mean(np.std(releases, axis=0))
                    profile.release_consistency[shot_type] = float(std_dev)

        # Calculate deception metrics for each shot
        deception_scores = []
        tunnel_scores = []
        break_scores = []

        for shot in player_shots:
            # Deception score
            dec_score = self.calculate_deception_score(shot, profile)
            deception_scores.append(dec_score)

            # Late break
            h_break, v_break = self.calculate_late_break(shot)
            break_scores.append(np.sqrt(h_break ** 2 + v_break ** 2))

        profile.overall_deception_score = float(np.mean(deception_scores))
        profile.late_break_score = float(np.mean(break_scores))

        # Tunnel overlap score (how similar are different shot types)
        shot_pairs = []
        for i, shot1 in enumerate(player_shots):
            for shot2 in player_shots[i + 1:i + 10]:  # Compare to nearby shots
                if shot1.shot_type != shot2.shot_type:
                    tunnel_dist = self.calculate_tunnel_distance(shot1, shot2)
                    shot_pairs.append(tunnel_dist)

        if shot_pairs:
            # Higher tunnel distance = better deception
            avg_tunnel = np.mean(shot_pairs)
            profile.tunnel_overlap_score = min(avg_tunnel / 30.0, 1.0)  # Normalize to 30ft max

        # Effectiveness
        goals = sum(1 for s in player_shots if s.is_goal)
        profile.actual_goal_rate = goals / total

        # Estimate xG boost from deception
        # High deception + high goal rate = deception is working
        baseline_xG = 0.08  # Typical shot xG
        if profile.actual_goal_rate > baseline_xG:
            profile.deception_xG_boost = (
                (profile.actual_goal_rate - baseline_xG) *
                profile.overall_deception_score
            )
        else:
            profile.deception_xG_boost = 0.0

        # Store in cache
        self.shooter_profiles[player_id] = profile

        return profile

    def compare_shooters(
        self,
        player_ids: List[str]
    ) -> pd.DataFrame:
        """
        Compare deception metrics across multiple shooters.

        Args:
            player_ids: List of player IDs to compare

        Returns:
            DataFrame with comparison metrics
        """
        comparisons = []

        for player_id in player_ids:
            profile = self.build_shooter_profile(player_id)

            comparisons.append({
                'player_id': player_id,
                'shots_analyzed': profile.shots_analyzed,
                'deception_score': profile.overall_deception_score,
                'tunnel_overlap': profile.tunnel_overlap_score,
                'late_break': profile.late_break_score,
                'release_consistency_avg': np.mean(list(profile.release_consistency.values())) if profile.release_consistency else None,
                'goal_rate': profile.actual_goal_rate,
                'deception_xG_boost': profile.deception_xG_boost,
                'primary_shot_type': max(profile.shot_type_distribution.items(), key=lambda x: x[1])[0] if profile.shot_type_distribution else None,
            })

        return pd.DataFrame(comparisons)

    def analyze_shot_sequence(
        self,
        shots: List[ShotTrackingData]
    ) -> Dict[str, Any]:
        """
        Analyze deception within a sequence of shots.

        Identifies:
        - Setup shots (establishing patterns)
        - Tunnel shots (exploiting established patterns)
        - Pattern breaks (changing to deceive)

        Args:
            shots: List of shots in sequence

        Returns:
            Analysis dictionary
        """
        if len(shots) < 3:
            return {'insufficient_data': True}

        analysis = {
            'total_shots': len(shots),
            'shot_types': [s.shot_type.value for s in shots],
            'deception_scores': [],
            'pattern_breaks': [],
            'setup_shots': [],
            'tunnel_shots': [],
        }

        # Analyze each shot in context of previous shots
        for i, shot in enumerate(shots[2:], start=2):
            # Previous shots establish pattern
            prev_shots = shots[max(0, i - 3):i]

            # Calculate how well this shot exploits the pattern
            prev_types = [s.shot_type for s in prev_shots]
            most_common_prev = max(set(prev_types), key=prev_types.count)

            deception_score = self.calculate_deception_score(shot)
            analysis['deception_scores'].append(deception_score)

            # Is this a pattern break?
            if shot.shot_type != most_common_prev and prev_types.count(most_common_prev) >= 2:
                analysis['pattern_breaks'].append({
                    'shot_index': i,
                    'expected_type': most_common_prev.value,
                    'actual_type': shot.shot_type.value,
                    'deception_score': deception_score,
                })

                # High deception pattern break = tunnel shot
                if deception_score > 0.6:
                    analysis['tunnel_shots'].append(i)
                else:
                    analysis['setup_shots'].append(i)

        # Summary statistics
        analysis['avg_deception'] = np.mean(analysis['deception_scores']) if analysis['deception_scores'] else 0
        analysis['pattern_break_rate'] = len(analysis['pattern_breaks']) / max(len(shots) - 2, 1)
        analysis['tunnel_effectiveness'] = (
            sum(1 for s in analysis['tunnel_shots'] if shots[s].is_goal) /
            max(len(analysis['tunnel_shots']), 1)
        )

        return analysis

    def to_dict(self) -> Dict[str, Any]:
        """Export analyzer state to dictionary."""
        return {
            'decision_distance': self.decision_distance,
            'reaction_time': self.reaction_time,
            'min_shots': self.min_shots,
            'num_shots': len(self.shots),
            'num_profiles': len(self.shooter_profiles),
            'profiles': {
                pid: {
                    'shots_analyzed': p.shots_analyzed,
                    'deception_score': p.overall_deception_score,
                    'tunnel_overlap': p.tunnel_overlap_score,
                    'late_break': p.late_break_score,
                    'goal_rate': p.actual_goal_rate,
                }
                for pid, p in self.shooter_profiles.items()
            }
        }


class PuckFlutterAnalyzer:
    """
    Analyzes puck flutter/knuckling effects.

    Similar to baseball's seam-shifted wake research (Dr. Barton Smith),
    this analyzes how puck spin and orientation affect trajectory.

    Key concepts:
    - Spin rate affects predictability
    - Low/no spin creates knuckling effect
    - Orientation at release affects flight
    """

    def __init__(self):
        """Initialize the flutter analyzer."""
        self.flutter_threshold = 300  # RPM below this = flutter potential

    def estimate_flutter_probability(
        self,
        spin_rate: float,
        release_angle: float,
        shot_type: ShotType
    ) -> float:
        """
        Estimate probability of flutter effect.

        Args:
            spin_rate: Puck spin in RPM
            release_angle: Angle of release
            shot_type: Type of shot

        Returns:
            Probability of flutter (0-1)
        """
        # Low spin = high flutter chance
        if spin_rate > self.flutter_threshold:
            spin_factor = 0.1
        else:
            spin_factor = 1.0 - (spin_rate / self.flutter_threshold)

        # Shot type affects flutter probability
        type_factor = {
            ShotType.WRIST: 0.3,    # Moderate spin control
            ShotType.SLAP: 0.5,     # High variability
            ShotType.SNAP: 0.4,     # Quick release, variable spin
            ShotType.BACKHAND: 0.6, # Often low spin
        }.get(shot_type, 0.4)

        # Flat release angles increase flutter
        angle_factor = 1.0 - abs(release_angle) / 30.0

        flutter_prob = spin_factor * type_factor * max(0.3, angle_factor)

        return float(np.clip(flutter_prob, 0.0, 1.0))

    def predict_trajectory_deviation(
        self,
        flutter_prob: float,
        distance: float,
        shot_speed: float
    ) -> Tuple[float, float]:
        """
        Predict trajectory deviation from flutter.

        Args:
            flutter_prob: Probability of flutter effect
            distance: Distance to target (feet)
            shot_speed: Shot speed (mph)

        Returns:
            Tuple of (max_horizontal_deviation, max_vertical_deviation) in inches
        """
        # Flight time
        flight_time = distance / (shot_speed * 1.467)

        # Base deviation scales with flutter probability and flight time
        base_deviation = flutter_prob * flight_time * 8.0  # inches

        # Random direction for horizontal
        horizontal = base_deviation * np.random.uniform(-1, 1)

        # Vertical tends downward (gravity + flutter)
        vertical = base_deviation * np.random.uniform(-1.2, 0.8)

        return (horizontal, vertical)


def create_sample_shots() -> List[ShotTrackingData]:
    """
    Create sample shot tracking data for testing.

    Returns:
        List of ShotTrackingData objects
    """
    np.random.seed(42)

    shots = []

    for i in range(500):
        shot_type = np.random.choice(list(ShotType)[:4])  # Main shot types

        # Release point varies by shot type
        base_release = {
            ShotType.WRIST: (0.0, 2.5, 1.5),
            ShotType.SLAP: (-1.0, 4.0, 0.5),
            ShotType.SNAP: (0.5, 2.0, 2.0),
            ShotType.BACKHAND: (-1.5, 1.5, 1.0),
        }[shot_type]

        # Add some variation
        release = tuple(
            base + np.random.normal(0, 0.3)
            for base in base_release
        )

        # Shot speed varies by type
        base_speed = {
            ShotType.WRIST: 70.0,
            ShotType.SLAP: 95.0,
            ShotType.SNAP: 75.0,
            ShotType.BACKHAND: 55.0,
        }[shot_type]
        speed = base_speed + np.random.normal(0, 5)

        shot = ShotTrackingData(
            shot_id=f"shot_{i}",
            timestamp=i * 60.0,
            game_id="game_001",
            shooter_id=f"player_{np.random.randint(1, 10)}",
            shooter_team="team_a",
            shooting_hand=np.random.choice(list(ShootingHand)),
            shot_type=shot_type,
            shot_speed=speed,
            release_x=release[0],
            release_y=release[1],
            release_z=release[2],
            stick_angle=np.random.uniform(20, 60),
            hip_rotation=np.random.uniform(10, 50),
            shoulder_angle=np.random.uniform(-10, 20),
            weight_transfer=np.random.uniform(0.3, 0.9),
            initial_velocity_x=np.random.uniform(-5, 5),
            initial_velocity_y=speed * 0.95,
            initial_velocity_z=np.random.uniform(-2, 10),
            launch_angle=np.random.uniform(2, 15),
            spin_rate=np.random.uniform(100, 2000),
            flutter=np.random.random() < 0.2,
            target_x=np.random.uniform(-2, 2),
            target_z=np.random.uniform(0.5, 3.5),
            is_goal=np.random.random() < 0.1,
        )

        shots.append(shot)

    return shots


if __name__ == "__main__":
    # Demo the shot deception analyzer
    print("Creating Shot Deception Analyzer...")

    analyzer = ShotDeceptionAnalyzer()

    # Create sample data
    print("Generating sample shots...")
    shots = create_sample_shots()

    for shot in shots:
        analyzer.add_shot(shot)

    # Build profiles for all shooters
    print("\nBuilding shooter profiles...")
    player_ids = list(set(s.shooter_id for s in shots))

    for pid in player_ids[:3]:  # First 3 players
        profile = analyzer.build_shooter_profile(pid)
        print(f"\n{pid}:")
        print(f"  Shots analyzed: {profile.shots_analyzed}")
        print(f"  Deception score: {profile.overall_deception_score:.3f}")
        print(f"  Tunnel overlap: {profile.tunnel_overlap_score:.3f}")
        print(f"  Late break score: {profile.late_break_score:.3f}")
        print(f"  Goal rate: {profile.actual_goal_rate:.3f}")
        print(f"  Shot distribution: {profile.shot_type_distribution}")

    # Compare shooters
    print("\nShooter Comparison:")
    comparison_df = analyzer.compare_shooters(player_ids)
    print(comparison_df.to_string())

    # Analyze shot sequence
    print("\nShot Sequence Analysis:")
    player_shots = [s for s in shots if s.shooter_id == "player_1"][:20]
    sequence_analysis = analyzer.analyze_shot_sequence(player_shots)
    print(f"  Pattern break rate: {sequence_analysis.get('pattern_break_rate', 0):.3f}")
    print(f"  Avg deception: {sequence_analysis.get('avg_deception', 0):.3f}")
    print(f"  Tunnel effectiveness: {sequence_analysis.get('tunnel_effectiveness', 0):.3f}")

    # Flutter analysis
    print("\nPuck Flutter Analysis:")
    flutter_analyzer = PuckFlutterAnalyzer()

    for shot in shots[:5]:
        flutter_prob = flutter_analyzer.estimate_flutter_probability(
            shot.spin_rate or 500,
            shot.launch_angle,
            shot.shot_type
        )
        h_dev, v_dev = flutter_analyzer.predict_trajectory_deviation(
            flutter_prob, 30.0, shot.shot_speed
        )
        print(f"  {shot.shot_type.value}: flutter_prob={flutter_prob:.3f}, "
              f"deviation=({h_dev:.1f}, {v_dev:.1f}) inches")
