"""
Off-Ball Scoring Opportunities (OBSO) Model for Hockey Analytics

This module implements a model to quantify scoring opportunities created
by players who never receive the puck, inspired by soccer analytics.

Key concepts:
- Space creation through movement
- Defensive attention drawn (gravity)
- Shooting lanes opened
- Decoy runs that drag defenders

Hockey application:
- Screeners drawing goalie attention
- Players driving the net
- Cycling creating passing lanes
- Defensive collapses on stars opening teammates

References:
- Soccer OBSO models (probabilistic physics-based)
- Spatiotemporal tracking analysis
- Off-puck movement quality metrics
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
class FrameSnapshot:
    """Snapshot of all positions at a moment."""
    timestamp: float
    frame_id: int
    game_id: str
    period: int

    # All player positions
    players: List[PlayerFrame]

    # Puck position
    puck_x: float
    puck_y: float
    puck_carrier_id: Optional[str] = None

    # Game state
    attacking_team: str = ""
    score_home: int = 0
    score_away: int = 0
    strength_state: str = "5v5"


@dataclass
class OBSOEvent:
    """Off-ball scoring opportunity event."""
    player_id: str
    team_id: str
    timestamp: float

    # Opportunity characteristics
    opportunity_value: float  # xG if player received pass
    space_created: float      # Square feet of space opened
    attention_drawn: float    # 0-1 scale
    lane_quality: float       # Quality of shooting lane

    # Movement characteristics
    movement_type: str        # drive, cycle, screen, drift
    start_x: float
    start_y: float
    end_x: float
    end_y: float

    # Impact
    defenders_moved: int = 0
    goalie_affected: bool = False
    shot_resulted: bool = False
    goal_resulted: bool = False


class OBSOModel:
    """
    Off-Ball Scoring Opportunities Model.

    Quantifies the value created by off-puck players through:
    1. Space creation (opening shooting/passing lanes)
    2. Defensive attention (drawing defenders away from puck)
    3. Goalie manipulation (screens, traffic)
    4. Shot opportunity creation
    """

    def __init__(
        self,
        rink_length: float = 200.0,
        rink_width: float = 85.0,
        grid_resolution: int = 10,  # feet per cell
        danger_zone_x: float = 175.0,  # Start of offensive zone
        slot_bounds: Tuple[float, float, float, float] = (170, 195, 27, 58),
    ):
        """
        Initialize the OBSO model.

        Args:
            rink_length: Rink length in feet
            rink_width: Rink width in feet
            grid_resolution: Resolution for space calculations
            danger_zone_x: X coordinate where offensive zone starts
            slot_bounds: (x_min, x_max, y_min, y_max) for slot area
        """
        self.rink_length = rink_length
        self.rink_width = rink_width
        self.grid_resolution = grid_resolution
        self.danger_zone_x = danger_zone_x
        self.slot_bounds = slot_bounds

        # xG values for receiving a pass at each location
        self._build_receiving_xG_map()

        # Track events
        self.obso_events: List[OBSOEvent] = []

    def _build_receiving_xG_map(self):
        """Build map of xG for receiving a pass at each location."""
        nx = int(self.rink_length / self.grid_resolution)
        ny = int(self.rink_width / self.grid_resolution)

        self.receiving_xG = np.zeros((ny, nx))

        for i in range(ny):
            for j in range(nx):
                x = j * self.grid_resolution
                y = i * self.grid_resolution

                # Higher xG in slot
                if self._in_slot(x, y):
                    distance_to_net = np.sqrt((200 - x) ** 2 + (42.5 - y) ** 2)
                    self.receiving_xG[i, j] = max(0.05, 0.35 - distance_to_net * 0.01)
                elif x > 150:  # Offensive zone
                    self.receiving_xG[i, j] = max(0.02, 0.15 - abs(y - 42.5) * 0.002)
                else:
                    self.receiving_xG[i, j] = 0.01

    def _in_slot(self, x: float, y: float) -> bool:
        """Check if position is in the slot."""
        return (self.slot_bounds[0] <= x <= self.slot_bounds[1] and
                self.slot_bounds[2] <= y <= self.slot_bounds[3])

    def calculate_space_created(
        self,
        player: PlayerFrame,
        defenders: List[PlayerFrame],
        prev_defenders: Optional[List[PlayerFrame]] = None
    ) -> float:
        """
        Calculate space created by a player's movement.

        Space is measured as the area opened up by drawing defenders.

        Args:
            player: Current player position
            defenders: Current defender positions
            prev_defenders: Previous defender positions (for delta)

        Returns:
            Space created in square feet
        """
        # Calculate voronoi-like space around player
        space = 0.0

        # Check area around player
        for dx in range(-30, 31, 5):
            for dy in range(-30, 31, 5):
                check_x = player.x + dx
                check_y = player.y + dy

                if check_x < 0 or check_x > 200 or check_y < 0 or check_y > 85:
                    continue

                # Is this point closer to the player than any defender?
                player_dist = np.sqrt(dx ** 2 + dy ** 2)

                closer_to_player = True
                for defender in defenders:
                    def_dist = np.sqrt((check_x - defender.x) ** 2 +
                                       (check_y - defender.y) ** 2)
                    if def_dist < player_dist:
                        closer_to_player = False
                        break

                if closer_to_player:
                    space += 25  # 5x5 = 25 sq ft per cell

        return space

    def calculate_attention_drawn(
        self,
        player: PlayerFrame,
        defenders: List[PlayerFrame],
        puck_x: float,
        puck_y: float
    ) -> Tuple[float, int]:
        """
        Calculate how much defensive attention a player draws.

        Attention is measured by:
        - Number of defenders focused on player (vs puck)
        - Distance of those defenders from the puck

        Args:
            player: Off-puck player position
            defenders: All defender positions
            puck_x, puck_y: Puck location

        Returns:
            Tuple of (attention_score, defenders_drawn)
        """
        attention_score = 0.0
        defenders_drawn = 0

        for defender in defenders:
            if defender.is_goalie:
                continue

            # Distance from defender to player
            dist_to_player = np.sqrt((defender.x - player.x) ** 2 +
                                     (defender.y - player.y) ** 2)

            # Distance from defender to puck
            dist_to_puck = np.sqrt((defender.x - puck_x) ** 2 +
                                   (defender.y - puck_y) ** 2)

            # If defender is closer to off-puck player than puck, they're drawn
            if dist_to_player < dist_to_puck - 5:  # 5ft buffer
                defenders_drawn += 1

                # More attention if defender is very close
                if dist_to_player < 10:
                    attention_score += 0.4
                elif dist_to_player < 20:
                    attention_score += 0.2
                else:
                    attention_score += 0.1

        return min(attention_score, 1.0), defenders_drawn

    def calculate_lane_quality(
        self,
        player: PlayerFrame,
        defenders: List[PlayerFrame],
        puck_x: float,
        puck_y: float
    ) -> float:
        """
        Calculate quality of passing/shooting lane to off-puck player.

        Quality is based on:
        - Clear line of sight
        - Angle quality
        - Distance appropriateness

        Args:
            player: Off-puck player
            defenders: Defender positions
            puck_x, puck_y: Puck location

        Returns:
            Lane quality (0-1)
        """
        # Distance to puck (too far = hard pass)
        distance = np.sqrt((player.x - puck_x) ** 2 + (player.y - puck_y) ** 2)

        if distance > 60:  # Very long pass
            distance_factor = 0.3
        elif distance > 40:
            distance_factor = 0.6
        elif distance > 15:
            distance_factor = 1.0
        else:
            distance_factor = 0.8  # Too close

        # Check for defenders in lane
        lane_blocked = 0.0

        # Line from puck to player
        dx = player.x - puck_x
        dy = player.y - puck_y
        length = np.sqrt(dx * dx + dy * dy)

        if length > 0:
            dx /= length
            dy /= length

            # Check each defender
            for defender in defenders:
                if defender.is_goalie:
                    continue

                # Project defender onto line
                to_def_x = defender.x - puck_x
                to_def_y = defender.y - puck_y

                # Dot product = projection distance along line
                proj = to_def_x * dx + to_def_y * dy

                if 0 < proj < length:  # Defender is between puck and player
                    # Perpendicular distance from defender to line
                    closest_x = puck_x + proj * dx
                    closest_y = puck_y + proj * dy
                    perp_dist = np.sqrt((defender.x - closest_x) ** 2 +
                                        (defender.y - closest_y) ** 2)

                    if perp_dist < 5:
                        lane_blocked += 0.3
                    elif perp_dist < 10:
                        lane_blocked += 0.15

        lane_quality = distance_factor * max(0, 1 - lane_blocked)

        return lane_quality

    def calculate_opportunity_value(
        self,
        player: PlayerFrame,
        lane_quality: float,
        space_created: float
    ) -> float:
        """
        Calculate the xG value if player received a pass.

        Combines positional xG with adjustments for:
        - Lane quality (can pass actually get there)
        - Space available (time to shoot)

        Args:
            player: Player position
            lane_quality: Quality of passing lane
            space_created: Space around player

        Returns:
            Adjusted xG value
        """
        # Get base xG from position
        grid_x = int(player.x / self.grid_resolution)
        grid_y = int(player.y / self.grid_resolution)

        grid_x = min(grid_x, self.receiving_xG.shape[1] - 1)
        grid_y = min(grid_y, self.receiving_xG.shape[0] - 1)

        base_xG = self.receiving_xG[grid_y, grid_x]

        # Adjust for lane quality
        adjusted_xG = base_xG * lane_quality

        # Space bonus (more time = better shot)
        if space_created > 200:  # Good space
            adjusted_xG *= 1.2
        elif space_created > 100:
            adjusted_xG *= 1.1

        return adjusted_xG

    def analyze_frame(
        self,
        frame: FrameSnapshot
    ) -> List[OBSOEvent]:
        """
        Analyze a frame for off-ball scoring opportunities.

        Args:
            frame: Snapshot of positions

        Returns:
            List of OBSO events for off-puck players
        """
        obso_events = []

        # Identify attacking and defending teams
        attacking_players = [p for p in frame.players
                            if p.team_id == frame.attacking_team and not p.has_puck]
        defenders = [p for p in frame.players
                    if p.team_id != frame.attacking_team]

        for player in attacking_players:
            if player.is_goalie:
                continue

            # Skip if player is in defensive zone
            if frame.attacking_team == "home" and player.x < 100:
                continue
            if frame.attacking_team == "away" and player.x > 100:
                continue

            # Calculate OBSO metrics
            space = self.calculate_space_created(player, defenders)
            attention, defenders_drawn = self.calculate_attention_drawn(
                player, defenders, frame.puck_x, frame.puck_y
            )
            lane_quality = self.calculate_lane_quality(
                player, defenders, frame.puck_x, frame.puck_y
            )
            opportunity_value = self.calculate_opportunity_value(
                player, lane_quality, space
            )

            # Determine movement type (simplified)
            if player.speed > 12:
                if player.x > 180:
                    movement_type = "drive"
                else:
                    movement_type = "cycle"
            elif space > 150:
                movement_type = "drift"
            else:
                movement_type = "screen"

            # Only record significant opportunities
            if opportunity_value > 0.03 or attention > 0.3:
                event = OBSOEvent(
                    player_id=player.player_id,
                    team_id=player.team_id,
                    timestamp=frame.timestamp,
                    opportunity_value=opportunity_value,
                    space_created=space,
                    attention_drawn=attention,
                    lane_quality=lane_quality,
                    movement_type=movement_type,
                    start_x=player.x,
                    start_y=player.y,
                    end_x=player.x + player.velocity_x * 0.5,
                    end_y=player.y + player.velocity_y * 0.5,
                    defenders_moved=defenders_drawn,
                )

                obso_events.append(event)
                self.obso_events.append(event)

        return obso_events

    def get_player_obso_summary(
        self,
        player_id: str
    ) -> Dict[str, Any]:
        """
        Get OBSO summary for a player.

        Args:
            player_id: Player to analyze

        Returns:
            Dictionary with OBSO statistics
        """
        player_events = [e for e in self.obso_events if e.player_id == player_id]

        if not player_events:
            return {'player_id': player_id, 'events': 0}

        return {
            'player_id': player_id,
            'events': len(player_events),
            'total_opportunity_value': sum(e.opportunity_value for e in player_events),
            'avg_opportunity_value': np.mean([e.opportunity_value for e in player_events]),
            'total_space_created': sum(e.space_created for e in player_events),
            'avg_attention_drawn': np.mean([e.attention_drawn for e in player_events]),
            'avg_lane_quality': np.mean([e.lane_quality for e in player_events]),
            'total_defenders_moved': sum(e.defenders_moved for e in player_events),
            'movement_types': dict(pd.Series([e.movement_type for e in player_events]).value_counts()),
        }

    def get_team_obso_comparison(self) -> pd.DataFrame:
        """Compare OBSO metrics across all tracked players."""
        player_ids = set(e.player_id for e in self.obso_events)

        summaries = []
        for pid in player_ids:
            summary = self.get_player_obso_summary(pid)
            summaries.append(summary)

        df = pd.DataFrame(summaries)
        if not df.empty:
            df = df.sort_values('total_opportunity_value', ascending=False)

        return df


def create_sample_frame() -> FrameSnapshot:
    """Create sample frame data for testing."""
    np.random.seed(42)

    players = []

    # Home team attacking (in offensive zone)
    home_positions = [
        (180, 42, True),   # Center with puck
        (175, 25, False),  # LW
        (185, 60, False),  # RW (driving net)
        (160, 30, False),  # LD
        (160, 55, False),  # RD
    ]

    for i, (x, y, has_puck) in enumerate(home_positions):
        players.append(PlayerFrame(
            player_id=f"home_{i}",
            team_id="home",
            x=x + np.random.uniform(-2, 2),
            y=y + np.random.uniform(-2, 2),
            velocity_x=np.random.uniform(5, 15) if not has_puck else 0,
            velocity_y=np.random.uniform(-5, 5),
            speed=np.random.uniform(8, 18) if not has_puck else 2,
            has_puck=has_puck,
        ))

    # Away team defending
    away_positions = [
        (175, 40, False),  # Center
        (170, 20, False),  # LW
        (180, 55, False),  # RW (covering net-driver)
        (165, 35, False),  # LD
        (165, 50, False),  # RD
        (195, 42.5, True), # Goalie
    ]

    for i, (x, y, is_goalie) in enumerate(away_positions):
        players.append(PlayerFrame(
            player_id=f"away_{i}",
            team_id="away",
            x=x + np.random.uniform(-2, 2),
            y=y + np.random.uniform(-2, 2),
            velocity_x=np.random.uniform(-5, 5),
            velocity_y=np.random.uniform(-5, 5),
            speed=np.random.uniform(5, 12),
            is_goalie=is_goalie,
        ))

    # Find puck carrier
    puck_carrier = next(p for p in players if p.has_puck)

    return FrameSnapshot(
        timestamp=100.0,
        frame_id=1,
        game_id="game_001",
        period=1,
        players=players,
        puck_x=puck_carrier.x,
        puck_y=puck_carrier.y,
        puck_carrier_id=puck_carrier.player_id,
        attacking_team="home",
    )


if __name__ == "__main__":
    # Demo the OBSO model
    print("Creating OBSO Model...")

    model = OBSOModel()

    # Create sample frame
    print("Generating sample frame...")
    frame = create_sample_frame()

    # Analyze frame
    print("Analyzing off-ball opportunities...")
    obso_events = model.analyze_frame(frame)

    print(f"\nFound {len(obso_events)} OBSO events:")
    for event in obso_events:
        print(f"\n  Player: {event.player_id}")
        print(f"    Movement: {event.movement_type}")
        print(f"    Opportunity value (xG): {event.opportunity_value:.3f}")
        print(f"    Space created: {event.space_created:.0f} sq ft")
        print(f"    Attention drawn: {event.attention_drawn:.2f}")
        print(f"    Lane quality: {event.lane_quality:.2f}")
        print(f"    Defenders moved: {event.defenders_moved}")

    # Simulate multiple frames for summary
    print("\nSimulating multiple frames for summary...")
    for i in range(50):
        frame = create_sample_frame()
        frame.timestamp = i * 2.0
        model.analyze_frame(frame)

    # Player summary
    print("\nPlayer OBSO Summary:")
    for pid in ["home_1", "home_2"]:
        summary = model.get_player_obso_summary(pid)
        print(f"\n  {pid}:")
        print(f"    Events: {summary.get('events', 0)}")
        print(f"    Total xG created: {summary.get('total_opportunity_value', 0):.3f}")
        print(f"    Avg space created: {summary.get('total_space_created', 0) / max(summary.get('events', 1), 1):.0f} sq ft")

    # Team comparison
    print("\nTeam OBSO Comparison:")
    comparison = model.get_team_obso_comparison()
    print(comparison.head(10).to_string())
