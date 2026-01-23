"""
Shooter Gravity Metrics for Hockey Analytics

This module implements gravity/spacing metrics translated from basketball
analytics (RAPTOR, contested shot attempts), measuring how much defensive
attention elite scorers command.

Key concepts:
- Defensive attention drawn by star players
- Space creation for teammates
- Gravity effect on defensive positioning
- Contested vs open shot generation

Basketball parallel:
- Steph Curry gravity opening teammates
- Proxy: contested 3-point attempts weighted by defender distance

Hockey translation:
- Elite shooters pulling defenders
- Goalie attention split
- Opening shooting lanes for linemates

References:
- FiveThirtyEight RAPTOR methodology
- Basketball gravity research
- Defensive attention metrics
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class PlayerFrame:
    """Player position in a single frame."""
    player_id: str
    team_id: str
    x: float
    y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    has_puck: bool = False
    is_goalie: bool = False


@dataclass
class GravityEvent:
    """Event capturing defensive attention on a player."""
    event_id: str
    timestamp: float
    game_id: str
    period: int

    # Player being tracked
    player_id: str
    team_id: str
    player_x: float
    player_y: float

    # Gravity metrics
    defenders_in_range: int = 0
    closest_defender_distance: float = 0.0
    total_defender_attention: float = 0.0  # 0-1 scale

    # Context
    zone: str = "neutral"
    has_puck: bool = False
    puck_carrier_distance: float = 0.0

    # Impact
    space_created_for_teammates: float = 0.0
    shot_lane_opened: bool = False
    teammate_xG_boost: float = 0.0


@dataclass
class ShooterGravityProfile:
    """Gravity profile for a player."""
    player_id: str
    team_id: str

    # Career/season gravity metrics
    avg_defenders_in_range: float = 0.0
    avg_defender_distance: float = 0.0
    gravity_score: float = 0.0

    # Impact metrics
    teammate_xG_with_player: float = 0.0
    teammate_xG_without_player: float = 0.0
    gravity_xG_boost: float = 0.0

    # Context
    offensive_zone_presence: float = 0.0  # % of OZ time
    slot_presence: float = 0.0  # % of time in slot

    # Shot attention
    shots_with_pressure: int = 0
    shots_without_pressure: int = 0
    pressure_rate: float = 0.0

    samples: int = 0


class ShooterGravityModel:
    """
    Model for measuring shooter gravity.

    Gravity measures how much defensive attention a player commands,
    which creates space for teammates even when the player doesn't
    have the puck.

    Components:
    1. Defender proximity when player is in offensive zone
    2. Defender attention when player is near puck
    3. Teammate scoring improvement when player is present
    4. Shot pressure differential (contested vs open)
    """

    def __init__(
        self,
        attention_range: float = 15.0,  # feet
        close_range: float = 8.0,  # feet for "close" attention
        slot_bounds: Tuple[float, float, float, float] = (170, 195, 27, 58),
    ):
        """
        Initialize the gravity model.

        Args:
            attention_range: Max distance for counting defensive attention
            close_range: Distance threshold for close coverage
            slot_bounds: (x_min, x_max, y_min, y_max) for slot area
        """
        self.attention_range = attention_range
        self.close_range = close_range
        self.slot_bounds = slot_bounds

        # Data storage
        self.gravity_events: List[GravityEvent] = []
        self.player_profiles: Dict[str, ShooterGravityProfile] = {}

        # Baseline metrics (league average)
        self.baseline_gravity = 0.5
        self.baseline_teammate_xG = 0.05

    def calculate_defender_attention(
        self,
        player: PlayerFrame,
        defenders: List[PlayerFrame],
        puck_x: float,
        puck_y: float
    ) -> Dict[str, float]:
        """
        Calculate defensive attention on a player.

        Args:
            player: Player position
            defenders: Defender positions
            puck_x, puck_y: Puck location

        Returns:
            Dictionary with attention metrics
        """
        defenders_in_range = 0
        total_attention = 0.0
        closest_distance = float('inf')

        for defender in defenders:
            if defender.is_goalie:
                continue

            distance = np.sqrt(
                (player.x - defender.x) ** 2 +
                (player.y - defender.y) ** 2
            )

            if distance < closest_distance:
                closest_distance = distance

            if distance < self.attention_range:
                defenders_in_range += 1

                # Attention weight based on distance (closer = more attention)
                attention = 1.0 - (distance / self.attention_range)
                total_attention += attention

        # Normalize attention
        total_attention = min(total_attention, 1.5)  # Cap at 150%

        return {
            'defenders_in_range': defenders_in_range,
            'closest_distance': closest_distance if closest_distance < float('inf') else 0,
            'attention_score': total_attention / max(defenders_in_range, 1) if defenders_in_range > 0 else 0,
            'total_attention': total_attention,
        }

    def calculate_gravity_score(
        self,
        attention_metrics: Dict[str, float],
        in_slot: bool,
        has_puck: bool
    ) -> float:
        """
        Calculate overall gravity score.

        Higher gravity = more defensive attention = more space for teammates.

        Args:
            attention_metrics: From calculate_defender_attention
            in_slot: Whether player is in the slot
            has_puck: Whether player has the puck

        Returns:
            Gravity score (0-1 scale, can exceed 1 for elite)
        """
        base_gravity = attention_metrics['attention_score']

        # Multipliers
        if in_slot:
            base_gravity *= 1.3  # More dangerous = more attention
        if has_puck:
            base_gravity *= 1.2  # Puck carrier naturally draws more

        # Bonus for multiple defenders
        if attention_metrics['defenders_in_range'] >= 2:
            base_gravity *= 1.15

        return float(np.clip(base_gravity, 0, 1.5))

    def calculate_space_created(
        self,
        player: PlayerFrame,
        teammates: List[PlayerFrame],
        defenders: List[PlayerFrame],
        player_gravity: float
    ) -> Dict[str, float]:
        """
        Calculate space created for teammates due to player's gravity.

        Args:
            player: Player with gravity
            teammates: Teammate positions
            defenders: Defender positions
            player_gravity: Player's gravity score

        Returns:
            Dictionary with space creation metrics
        """
        # Estimate: higher gravity player draws defenders, creating space
        # for teammates

        teammate_space = 0.0
        xG_boost = 0.0

        for teammate in teammates:
            if teammate.player_id == player.player_id:
                continue

            # Find closest defender to teammate
            min_defender_dist = float('inf')
            for defender in defenders:
                if defender.is_goalie:
                    continue

                dist = np.sqrt(
                    (teammate.x - defender.x) ** 2 +
                    (teammate.y - defender.y) ** 2
                )
                min_defender_dist = min(min_defender_dist, dist)

            # Space is related to defender distance
            if min_defender_dist > 10:
                teammate_space += min_defender_dist

                # xG boost if teammate is in dangerous area with space
                if self._in_slot(teammate.x, teammate.y) and min_defender_dist > 15:
                    xG_boost += 0.02 * player_gravity

        # Scale by gravity (higher gravity = more space created)
        teammate_space *= player_gravity

        return {
            'total_space_created': teammate_space,
            'teammate_xG_boost': xG_boost,
            'avg_teammate_separation': teammate_space / max(len(teammates) - 1, 1),
        }

    def _in_slot(self, x: float, y: float) -> bool:
        """Check if position is in the slot."""
        return (self.slot_bounds[0] <= x <= self.slot_bounds[1] and
                self.slot_bounds[2] <= y <= self.slot_bounds[3])

    def analyze_frame(
        self,
        frame_id: int,
        timestamp: float,
        game_id: str,
        period: int,
        target_player: PlayerFrame,
        teammates: List[PlayerFrame],
        defenders: List[PlayerFrame],
        puck_x: float,
        puck_y: float
    ) -> GravityEvent:
        """
        Analyze a single frame for gravity metrics.

        Args:
            frame_id: Frame identifier
            timestamp: Game time
            game_id: Game identifier
            period: Game period
            target_player: Player to analyze
            teammates: Teammate positions
            defenders: Defender positions
            puck_x, puck_y: Puck location

        Returns:
            GravityEvent with analysis
        """
        # Calculate attention
        attention = self.calculate_defender_attention(
            target_player, defenders, puck_x, puck_y
        )

        # Calculate gravity
        in_slot = self._in_slot(target_player.x, target_player.y)
        gravity = self.calculate_gravity_score(
            attention, in_slot, target_player.has_puck
        )

        # Calculate space created
        space = self.calculate_space_created(
            target_player, teammates, defenders, gravity
        )

        # Determine zone
        if target_player.x > 150:
            zone = "offensive"
        elif target_player.x < 50:
            zone = "defensive"
        else:
            zone = "neutral"

        # Distance to puck carrier
        puck_distance = np.sqrt(
            (target_player.x - puck_x) ** 2 +
            (target_player.y - puck_y) ** 2
        )

        event = GravityEvent(
            event_id=f"grav_{game_id}_{frame_id}",
            timestamp=timestamp,
            game_id=game_id,
            period=period,
            player_id=target_player.player_id,
            team_id=target_player.team_id,
            player_x=target_player.x,
            player_y=target_player.y,
            defenders_in_range=attention['defenders_in_range'],
            closest_defender_distance=attention['closest_distance'],
            total_defender_attention=attention['total_attention'],
            zone=zone,
            has_puck=target_player.has_puck,
            puck_carrier_distance=puck_distance,
            space_created_for_teammates=space['total_space_created'],
            shot_lane_opened=space['teammate_xG_boost'] > 0.01,
            teammate_xG_boost=space['teammate_xG_boost'],
        )

        self.gravity_events.append(event)

        return event

    def build_player_profile(self, player_id: str) -> ShooterGravityProfile:
        """
        Build a gravity profile for a player.

        Args:
            player_id: Player to profile

        Returns:
            ShooterGravityProfile with aggregated metrics
        """
        player_events = [e for e in self.gravity_events if e.player_id == player_id]

        if not player_events:
            return ShooterGravityProfile(
                player_id=player_id,
                team_id=""
            )

        profile = ShooterGravityProfile(
            player_id=player_id,
            team_id=player_events[0].team_id,
            samples=len(player_events),
        )

        # Average metrics
        profile.avg_defenders_in_range = np.mean([e.defenders_in_range for e in player_events])
        profile.avg_defender_distance = np.mean([e.closest_defender_distance for e in player_events])

        # Gravity score (average attention)
        attention_scores = [e.total_defender_attention for e in player_events]
        profile.gravity_score = np.mean(attention_scores) / 1.0  # Normalize

        # Zone presence
        oz_events = [e for e in player_events if e.zone == "offensive"]
        profile.offensive_zone_presence = len(oz_events) / len(player_events)

        slot_events = [e for e in player_events if self._in_slot(e.player_x, e.player_y)]
        profile.slot_presence = len(slot_events) / len(player_events)

        # xG boost
        profile.teammate_xG_with_player = np.mean([e.teammate_xG_boost for e in player_events])

        # Pressure rate (with puck)
        puck_events = [e for e in player_events if e.has_puck]
        if puck_events:
            pressured = sum(1 for e in puck_events if e.defenders_in_range >= 1)
            profile.pressure_rate = pressured / len(puck_events)
            profile.shots_with_pressure = pressured
            profile.shots_without_pressure = len(puck_events) - pressured

        # Calculate gravity xG boost relative to baseline
        profile.gravity_xG_boost = profile.teammate_xG_with_player - self.baseline_teammate_xG

        self.player_profiles[player_id] = profile

        return profile

    def compare_players(
        self,
        player_ids: List[str]
    ) -> pd.DataFrame:
        """
        Compare gravity metrics across players.

        Args:
            player_ids: Players to compare

        Returns:
            DataFrame with comparison
        """
        comparisons = []

        for pid in player_ids:
            profile = self.build_player_profile(pid)
            comparisons.append({
                'player_id': pid,
                'gravity_score': profile.gravity_score,
                'avg_defenders_near': profile.avg_defenders_in_range,
                'avg_defender_dist': profile.avg_defender_distance,
                'oz_presence': profile.offensive_zone_presence,
                'slot_presence': profile.slot_presence,
                'teammate_xG_boost': profile.teammate_xG_with_player,
                'pressure_rate': profile.pressure_rate,
                'samples': profile.samples,
            })

        df = pd.DataFrame(comparisons)
        if not df.empty:
            df = df.sort_values('gravity_score', ascending=False)

        return df

    def get_gravity_impact_on_play(
        self,
        high_gravity_player: PlayerFrame,
        play_result_xG: float,
        baseline_xG: float = 0.05
    ) -> Dict[str, float]:
        """
        Estimate how much a high-gravity player impacted a play.

        Args:
            high_gravity_player: Player with high gravity
            play_result_xG: Actual xG of the resulting shot
            baseline_xG: Expected baseline xG

        Returns:
            Dictionary with impact estimates
        """
        # Get player's gravity profile
        if high_gravity_player.player_id in self.player_profiles:
            profile = self.player_profiles[high_gravity_player.player_id]
        else:
            profile = self.build_player_profile(high_gravity_player.player_id)

        # Estimate gravity contribution
        gravity_contribution = profile.gravity_score * (play_result_xG - baseline_xG)

        return {
            'total_xG': play_result_xG,
            'baseline_xG': baseline_xG,
            'gravity_boost': gravity_contribution,
            'player_gravity_score': profile.gravity_score,
            'pct_attributed_to_gravity': gravity_contribution / max(play_result_xG, 0.01) * 100,
        }


class GoalieAttentionModel:
    """
    Model for measuring how shooters affect goalie attention/positioning.

    Complementary to defender gravity - measures how goalies cheat
    toward or away from certain shooters.
    """

    def __init__(self):
        """Initialize the goalie attention model."""
        self.goalie_tendencies: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)

    def record_goalie_position(
        self,
        goalie_id: str,
        shooter_id: str,
        goalie_x: float,
        goalie_y: float,
        optimal_x: float,
        optimal_y: float,
        shooter_x: float,
        shooter_y: float
    ):
        """
        Record goalie positioning relative to a shooter.

        Args:
            goalie_id: Goalie identifier
            shooter_id: Shooter identifier
            goalie_x, goalie_y: Actual goalie position
            optimal_x, optimal_y: Optimal position (angle bisector)
            shooter_x, shooter_y: Shooter position
        """
        # Calculate deviation from optimal
        deviation_x = goalie_x - optimal_x
        deviation_y = goalie_y - optimal_y

        # Cheating toward shooter?
        to_shooter_x = shooter_x - goalie_x
        to_shooter_y = shooter_y - goalie_y

        # Dot product to see if goalie moved toward shooter
        cheating_toward = (deviation_x * to_shooter_x + deviation_y * to_shooter_y) > 0

        self.goalie_tendencies[(goalie_id, shooter_id)].append({
            'deviation_x': deviation_x,
            'deviation_y': deviation_y,
            'cheating_toward': cheating_toward,
            'deviation_magnitude': np.sqrt(deviation_x ** 2 + deviation_y ** 2),
        })

    def get_shooter_effect_on_goalie(
        self,
        goalie_id: str,
        shooter_id: str
    ) -> Dict[str, float]:
        """
        Get how much a shooter affects a goalie's positioning.

        Args:
            goalie_id: Goalie identifier
            shooter_id: Shooter identifier

        Returns:
            Dictionary with effect metrics
        """
        key = (goalie_id, shooter_id)

        if key not in self.goalie_tendencies or not self.goalie_tendencies[key]:
            return {'insufficient_data': True}

        data = self.goalie_tendencies[key]

        return {
            'avg_deviation': np.mean([d['deviation_magnitude'] for d in data]),
            'cheat_rate': np.mean([d['cheating_toward'] for d in data]),
            'avg_cheat_x': np.mean([d['deviation_x'] for d in data]),
            'avg_cheat_y': np.mean([d['deviation_y'] for d in data]),
            'samples': len(data),
        }


def create_sample_gravity_data() -> Tuple[PlayerFrame, List[PlayerFrame], List[PlayerFrame]]:
    """Create sample data for testing."""
    np.random.seed(42)

    # Star player in offensive zone
    target_player = PlayerFrame(
        player_id="star_player",
        team_id="home",
        x=180,
        y=42,
        velocity_x=5,
        velocity_y=0,
        speed=10,
        has_puck=False,
    )

    # Teammates
    teammates = [
        PlayerFrame(player_id="tm_1", team_id="home", x=175, y=25, has_puck=True),
        PlayerFrame(player_id="tm_2", team_id="home", x=170, y=60),
        PlayerFrame(player_id="tm_3", team_id="home", x=160, y=40),
    ]

    # Defenders (drawn toward star player)
    defenders = [
        PlayerFrame(player_id="def_1", team_id="away", x=178, y=38),  # Close to star
        PlayerFrame(player_id="def_2", team_id="away", x=175, y=45),  # Close to star
        PlayerFrame(player_id="def_3", team_id="away", x=165, y=50),
        PlayerFrame(player_id="goalie", team_id="away", x=195, y=42.5, is_goalie=True),
    ]

    return target_player, teammates, defenders


if __name__ == "__main__":
    # Demo the gravity model
    print("Creating Shooter Gravity Model...")

    model = ShooterGravityModel()

    # Create sample data
    print("Generating sample data...")
    target, teammates, defenders = create_sample_gravity_data()

    # Analyze frame
    print("\nAnalyzing frame...")
    event = model.analyze_frame(
        frame_id=1,
        timestamp=100.0,
        game_id="game_001",
        period=1,
        target_player=target,
        teammates=teammates,
        defenders=defenders,
        puck_x=175,
        puck_y=25,
    )

    print(f"  Player: {event.player_id}")
    print(f"  Defenders in range: {event.defenders_in_range}")
    print(f"  Closest defender: {event.closest_defender_distance:.1f} ft")
    print(f"  Total attention: {event.total_defender_attention:.2f}")
    print(f"  Space created: {event.space_created_for_teammates:.1f} sq ft")
    print(f"  Teammate xG boost: {event.teammate_xG_boost:.3f}")

    # Simulate multiple frames
    print("\nSimulating multiple frames...")
    for i in range(100):
        # Randomize positions slightly
        target.x = 175 + np.random.uniform(-5, 5)
        target.y = 42 + np.random.uniform(-5, 5)

        for defender in defenders:
            if not defender.is_goalie:
                defender.x += np.random.uniform(-2, 2)
                defender.y += np.random.uniform(-2, 2)

        model.analyze_frame(
            frame_id=i + 2,
            timestamp=100 + i * 2,
            game_id="game_001",
            period=1,
            target_player=target,
            teammates=teammates,
            defenders=defenders,
            puck_x=175 + np.random.uniform(-5, 5),
            puck_y=25 + np.random.uniform(-5, 5),
        )

    # Build profile
    print("\nBuilding player profile...")
    profile = model.build_player_profile("star_player")

    print(f"  Gravity score: {profile.gravity_score:.3f}")
    print(f"  Avg defenders near: {profile.avg_defenders_in_range:.2f}")
    print(f"  Avg defender distance: {profile.avg_defender_distance:.1f} ft")
    print(f"  OZ presence: {profile.offensive_zone_presence:.1%}")
    print(f"  Slot presence: {profile.slot_presence:.1%}")
    print(f"  Teammate xG boost: {profile.teammate_xG_with_player:.4f}")
    print(f"  Samples: {profile.samples}")

    # Gravity impact on play
    print("\nGravity Impact on Sample Play:")
    impact = model.get_gravity_impact_on_play(
        target,
        play_result_xG=0.15,
        baseline_xG=0.08
    )
    print(f"  Total xG: {impact['total_xG']:.3f}")
    print(f"  Baseline xG: {impact['baseline_xG']:.3f}")
    print(f"  Gravity boost: {impact['gravity_boost']:.4f}")
    print(f"  % attributed to gravity: {impact['pct_attributed_to_gravity']:.1f}%")
