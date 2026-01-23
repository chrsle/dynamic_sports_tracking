"""
Defensive Skill Spatial Structure Model

Implements spatial analysis of defensive skill translated from
Franks, Miller, Bornn, and Goldsberry's basketball research.

Key Paper:
- Franks, A., Miller, A., Bornn, L., & Goldsberry, K. (2015).
  "Characterizing the Spatial Structure of Defensive Skill in
  Professional Basketball." Annals of Applied Statistics, 9:194-121.

Key Concepts:
- Spatial analysis of defensive positioning
- Measures influence on opponent shooting performance
- Uses tracking data to quantify "good defense"
- Location-specific defensive impact

Hockey Translation:
- Evaluate defensive positioning relative to shooter
- Credit defensemen for reducing shot quality
- Model "defensive gravity" (attention drawn)
- Zone-specific defensive effectiveness
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class DefensiveZone(Enum):
    """Zones for defensive evaluation."""
    SLOT = "slot"
    HIGH_SLOT = "high_slot"
    LEFT_CIRCLE = "left_circle"
    RIGHT_CIRCLE = "right_circle"
    POINT = "point"
    CORNER_LEFT = "corner_left"
    CORNER_RIGHT = "corner_right"
    BEHIND_NET = "behind_net"
    NEUTRAL = "neutral"


class DefensiveAction(Enum):
    """Types of defensive actions."""
    SHOT_BLOCK = "shot_block"
    PASS_DEFLECTION = "pass_deflection"
    STICK_CHECK = "stick_check"
    BODY_CHECK = "body_check"
    STICK_LIFT = "stick_lift"
    GAP_CONTROL = "gap_control"
    LANE_BLOCK = "lane_block"
    PRESSURE = "pressure"


@dataclass
class DefenderPosition:
    """Position and state of a defender."""
    player_id: str
    x: float                        # feet
    y: float                        # feet
    velocity_x: float = 0.0         # ft/s
    velocity_y: float = 0.0         # ft/s
    stick_angle: float = 0.0        # radians
    facing_direction: float = 0.0   # radians
    is_engaged: bool = False        # In a battle/check

    @property
    def speed(self) -> float:
        """Current speed."""
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2)


@dataclass
class AttackerPosition:
    """Position and state of an attacker."""
    player_id: str
    x: float
    y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    has_puck: bool = False
    shooting_threat: float = 0.0    # 0-1, likelihood to shoot


@dataclass
class DefensiveImpact:
    """Impact of defensive positioning on shot quality."""
    player_id: str
    zone: DefensiveZone
    shot_suppression: float         # Expected xG reduction
    lane_coverage: float            # % of passing lanes blocked
    pressure_rating: float          # 0-1, pressure on puck carrier
    position_quality: float         # 0-1, quality of positioning
    recovery_ability: float         # 0-1, ability to recover


@dataclass
class SpatialDefenseProfile:
    """Player's defensive skill by zone."""
    player_id: str
    zone_effectiveness: Dict[DefensiveZone, float]
    overall_rating: float
    shot_suppression_rate: float    # xG saved per 60
    blocks_per_60: float
    hits_per_60: float
    takeaways_per_60: float


class DefensiveSpatialModel:
    """
    Models spatial structure of defensive skill.

    Evaluates how defender positioning affects opponent shot quality
    and scoring chances, providing location-specific defensive ratings.
    """

    # Rink dimensions
    RINK_LENGTH = 200.0
    RINK_WIDTH = 85.0

    # Zone definitions (from offensive perspective, attacking right goal)
    GOAL_X = 189.0  # Goal line position
    GOAL_Y = 42.5   # Center of rink

    def __init__(
        self,
        grid_x: int = 40,
        grid_y: int = 17,
        base_xg_model: Optional[Any] = None,
    ):
        """
        Initialize defensive spatial model.

        Args:
            grid_x: Grid resolution along rink length
            grid_y: Grid resolution along rink width
            base_xg_model: Expected goals model for baseline
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.base_xg_model = base_xg_model

        # Create coordinate grids for defensive zone
        self.x_coords = np.linspace(150, 200, grid_x)  # Just offensive zone
        self.y_coords = np.linspace(0, self.RINK_WIDTH, grid_y)

        # Pre-compute zone mappings
        self.zone_grid = self._compute_zone_grid()

    def _compute_zone_grid(self) -> np.ndarray:
        """Pre-compute which zone each grid cell belongs to."""
        zones = np.empty((len(self.y_coords), len(self.x_coords)), dtype=object)

        for i, y in enumerate(self.y_coords):
            for j, x in enumerate(self.x_coords):
                zones[i, j] = self._get_zone(x, y)

        return zones

    def _get_zone(self, x: float, y: float) -> DefensiveZone:
        """Determine which defensive zone a point is in."""
        dist_to_goal = np.sqrt((x - self.GOAL_X)**2 + (y - self.GOAL_Y)**2)
        angle_to_goal = np.arctan2(y - self.GOAL_Y, self.GOAL_X - x)

        # Behind net
        if x > self.GOAL_X:
            return DefensiveZone.BEHIND_NET

        # Slot (danger zone)
        if dist_to_goal < 25 and abs(angle_to_goal) < np.pi/4:
            return DefensiveZone.SLOT

        # High slot
        if dist_to_goal < 40 and abs(angle_to_goal) < np.pi/3:
            return DefensiveZone.HIGH_SLOT

        # Circles
        if dist_to_goal < 35:
            if y < self.GOAL_Y:
                return DefensiveZone.LEFT_CIRCLE
            else:
                return DefensiveZone.RIGHT_CIRCLE

        # Point
        if x < 175 and abs(y - self.GOAL_Y) < 25:
            return DefensiveZone.POINT

        # Corners
        if y < 30:
            return DefensiveZone.CORNER_LEFT
        elif y > 55:
            return DefensiveZone.CORNER_RIGHT

        return DefensiveZone.NEUTRAL

    def compute_shot_suppression(
        self,
        shot_location: Tuple[float, float],
        defenders: List[DefenderPosition],
    ) -> float:
        """
        Compute shot suppression effect of defender positioning.

        Returns the expected reduction in xG (0-1 scale).
        """
        if not defenders:
            return 0.0

        shot_x, shot_y = shot_location
        total_suppression = 0.0

        for defender in defenders:
            # Distance from shot location
            dist = np.sqrt(
                (defender.x - shot_x)**2 + (defender.y - shot_y)**2
            )

            # Angle blocking effect
            angle_to_goal = np.arctan2(
                self.GOAL_Y - shot_y, self.GOAL_X - shot_x
            )
            defender_angle = np.arctan2(
                defender.y - shot_y, defender.x - shot_x
            )
            angle_diff = abs(angle_to_goal - defender_angle)

            # In shooting lane?
            in_lane = angle_diff < 0.3 and defender.x > shot_x

            # Pressure effect (close defenders)
            if dist < 5:
                pressure = 0.4  # Very close
            elif dist < 10:
                pressure = 0.25 * (1 - dist / 10)
            elif dist < 20:
                pressure = 0.1 * (1 - dist / 20)
            else:
                pressure = 0.0

            # Lane blocking effect
            if in_lane:
                if dist < 10:
                    lane_block = 0.3
                elif dist < 20:
                    lane_block = 0.15
                else:
                    lane_block = 0.05
            else:
                lane_block = 0.0

            # Stick positioning effect
            stick_threat = self._stick_threat(defender, shot_x, shot_y)

            # Combine effects (diminishing returns)
            defender_suppression = min(
                0.5,  # Max single defender impact
                pressure + lane_block + stick_threat
            )

            total_suppression += defender_suppression

        # Diminishing returns for multiple defenders
        return min(0.8, total_suppression)  # Max 80% suppression

    def _stick_threat(
        self,
        defender: DefenderPosition,
        shot_x: float,
        shot_y: float,
    ) -> float:
        """Calculate threat of stick disrupting shot."""
        dist = np.sqrt(
            (defender.x - shot_x)**2 + (defender.y - shot_y)**2
        )

        # Stick reach is about 5-6 feet
        if dist > 8:
            return 0.0

        # Check if stick is angled toward puck
        stick_end_x = defender.x + 5 * np.cos(defender.stick_angle)
        stick_end_y = defender.y + 5 * np.sin(defender.stick_angle)

        stick_to_shot = np.sqrt(
            (stick_end_x - shot_x)**2 + (stick_end_y - shot_y)**2
        )

        if stick_to_shot < 3:
            return 0.15  # Significant threat
        elif stick_to_shot < 6:
            return 0.05
        return 0.0

    def compute_lane_coverage(
        self,
        puck_location: Tuple[float, float],
        defenders: List[DefenderPosition],
        potential_receivers: List[AttackerPosition],
    ) -> Dict[str, float]:
        """
        Compute how well defenders cover passing lanes.

        Returns coverage rating for each potential receiver.
        """
        puck_x, puck_y = puck_location
        coverage = {}

        for receiver in potential_receivers:
            if receiver.has_puck:
                continue

            # Lane from puck to receiver
            lane_dx = receiver.x - puck_x
            lane_dy = receiver.y - puck_y
            lane_length = np.sqrt(lane_dx**2 + lane_dy**2)

            if lane_length < 1:
                coverage[receiver.player_id] = 1.0
                continue

            # Unit vector along lane
            ux = lane_dx / lane_length
            uy = lane_dy / lane_length

            # Check each defender's coverage of this lane
            max_coverage = 0.0

            for defender in defenders:
                # Vector from puck to defender
                dx = defender.x - puck_x
                dy = defender.y - puck_y

                # Project defender onto lane
                projection = dx * ux + dy * uy

                # Only count if between puck and receiver
                if projection < 0 or projection > lane_length:
                    continue

                # Distance from lane
                closest_x = puck_x + projection * ux
                closest_y = puck_y + projection * uy
                lane_dist = np.sqrt(
                    (defender.x - closest_x)**2 +
                    (defender.y - closest_y)**2
                )

                # Coverage based on distance from lane
                if lane_dist < 3:
                    lane_coverage = 0.9
                elif lane_dist < 6:
                    lane_coverage = 0.6 * (1 - lane_dist / 6)
                elif lane_dist < 10:
                    lane_coverage = 0.2 * (1 - lane_dist / 10)
                else:
                    lane_coverage = 0.0

                max_coverage = max(max_coverage, lane_coverage)

            coverage[receiver.player_id] = max_coverage

        return coverage

    def compute_pressure_rating(
        self,
        puck_carrier: AttackerPosition,
        defenders: List[DefenderPosition],
    ) -> float:
        """
        Compute pressure on puck carrier.

        Returns pressure rating 0-1 (1 = maximum pressure).
        """
        pressure = 0.0

        for defender in defenders:
            dist = np.sqrt(
                (defender.x - puck_carrier.x)**2 +
                (defender.y - puck_carrier.y)**2
            )

            # Velocity toward puck carrier
            dx = puck_carrier.x - defender.x
            dy = puck_carrier.y - defender.y
            dist_norm = max(dist, 0.1)
            ux = dx / dist_norm
            uy = dy / dist_norm

            closing_speed = defender.velocity_x * ux + defender.velocity_y * uy

            # Base pressure from distance
            if dist < 5:
                base_pressure = 0.5
            elif dist < 10:
                base_pressure = 0.3 * (1 - dist / 10)
            elif dist < 15:
                base_pressure = 0.1 * (1 - dist / 15)
            else:
                base_pressure = 0.0

            # Boost for closing speed
            if closing_speed > 0:
                speed_boost = min(0.2, closing_speed / 30)
            else:
                speed_boost = 0.0

            pressure += base_pressure + speed_boost

        return min(1.0, pressure)

    def evaluate_position_quality(
        self,
        defender: DefenderPosition,
        attackers: List[AttackerPosition],
        puck_location: Tuple[float, float],
    ) -> float:
        """
        Evaluate quality of a defender's positioning.

        Good positioning:
        - Between puck and goal
        - Can cover multiple threats
        - Gap control appropriate to situation
        """
        score = 0.0
        puck_x, puck_y = puck_location

        # 1. Between puck and goal?
        to_goal_x = self.GOAL_X - puck_x
        to_goal_y = self.GOAL_Y - puck_y
        to_defender_x = defender.x - puck_x
        to_defender_y = defender.y - puck_y

        # Dot product (positive if defender is toward goal)
        alignment = (to_goal_x * to_defender_x + to_goal_y * to_defender_y)
        if alignment > 0:
            score += 0.25

        # 2. Gap control
        puck_carrier = next(
            (a for a in attackers if a.has_puck), None
        )
        if puck_carrier:
            gap = np.sqrt(
                (defender.x - puck_carrier.x)**2 +
                (defender.y - puck_carrier.y)**2
            )
            # Ideal gap is 8-12 feet in zone
            if 8 <= gap <= 12:
                score += 0.25
            elif 5 <= gap <= 15:
                score += 0.15

        # 3. Coverage of multiple threats
        threats_covered = 0
        for attacker in attackers:
            if attacker.has_puck:
                continue

            dist = np.sqrt(
                (defender.x - attacker.x)**2 +
                (defender.y - attacker.y)**2
            )
            if dist < 15:
                threats_covered += 1

        if threats_covered >= 2:
            score += 0.25
        elif threats_covered == 1:
            score += 0.15

        # 4. Not caught flat-footed
        if defender.speed > 5 or not defender.is_engaged:
            score += 0.1

        # 5. Zone appropriateness
        zone = self._get_zone(defender.x, defender.y)
        if zone in [DefensiveZone.SLOT, DefensiveZone.HIGH_SLOT]:
            # Good to be in dangerous zones
            score += 0.15

        return min(1.0, score)

    def create_defense_profile(
        self,
        player_id: str,
        defensive_events: List[Dict[str, Any]],
    ) -> SpatialDefenseProfile:
        """
        Create spatial defensive profile from historical data.

        Args:
            player_id: Player to profile
            defensive_events: List of defensive events with location

        Returns:
            Complete spatial defensive profile
        """
        zone_stats: Dict[DefensiveZone, List[float]] = defaultdict(list)

        for event in defensive_events:
            if event.get('player_id') != player_id:
                continue

            x = event.get('x', 0)
            y = event.get('y', 0)
            zone = self._get_zone(x, y)

            # Event effectiveness (varies by action type)
            action = event.get('action_type', '')
            if action == DefensiveAction.SHOT_BLOCK.value:
                effectiveness = 0.8
            elif action == DefensiveAction.PASS_DEFLECTION.value:
                effectiveness = 0.6
            elif action == DefensiveAction.STICK_CHECK.value:
                effectiveness = 0.5
            elif action == DefensiveAction.GAP_CONTROL.value:
                effectiveness = event.get('effectiveness', 0.5)
            else:
                effectiveness = 0.4

            zone_stats[zone].append(effectiveness)

        # Compute zone effectiveness
        zone_effectiveness = {}
        for zone in DefensiveZone:
            if zone_stats[zone]:
                zone_effectiveness[zone] = float(np.mean(zone_stats[zone]))
            else:
                zone_effectiveness[zone] = 0.5  # Default

        # Overall rating (weighted by zone importance)
        zone_weights = {
            DefensiveZone.SLOT: 2.0,
            DefensiveZone.HIGH_SLOT: 1.5,
            DefensiveZone.LEFT_CIRCLE: 1.2,
            DefensiveZone.RIGHT_CIRCLE: 1.2,
            DefensiveZone.POINT: 0.8,
            DefensiveZone.CORNER_LEFT: 0.6,
            DefensiveZone.CORNER_RIGHT: 0.6,
            DefensiveZone.BEHIND_NET: 0.7,
            DefensiveZone.NEUTRAL: 0.5,
        }

        weighted_sum = sum(
            zone_effectiveness[z] * zone_weights[z]
            for z in zone_effectiveness
        )
        total_weight = sum(zone_weights.values())
        overall = weighted_sum / total_weight

        return SpatialDefenseProfile(
            player_id=player_id,
            zone_effectiveness=zone_effectiveness,
            overall_rating=overall,
            shot_suppression_rate=0.0,  # Would need xG data
            blocks_per_60=0.0,  # Would need ice time
            hits_per_60=0.0,
            takeaways_per_60=0.0,
        )


class DefensiveGravityModel:
    """
    Models "defensive gravity" - attention drawn by offensive threats.

    Similar to shooter gravity in basketball, this measures how much
    defensive attention each attacker draws.
    """

    def __init__(self, spatial_model: DefensiveSpatialModel):
        """Initialize with spatial model."""
        self.spatial_model = spatial_model

    def compute_attention_allocation(
        self,
        defenders: List[DefenderPosition],
        attackers: List[AttackerPosition],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute how defenders allocate attention to attackers.

        Returns: {defender_id: {attacker_id: attention_weight}}
        """
        allocation: Dict[str, Dict[str, float]] = {}

        for defender in defenders:
            attn = {}
            total = 0.0

            for attacker in attackers:
                # Attention based on:
                # 1. Distance
                dist = np.sqrt(
                    (defender.x - attacker.x)**2 +
                    (defender.y - attacker.y)**2
                )

                # 2. Puck possession
                if attacker.has_puck:
                    puck_mult = 2.5
                else:
                    puck_mult = 1.0

                # 3. Shooting threat
                threat_mult = 1.0 + attacker.shooting_threat

                # 4. Danger zone proximity
                zone = self.spatial_model._get_zone(attacker.x, attacker.y)
                if zone == DefensiveZone.SLOT:
                    zone_mult = 2.0
                elif zone == DefensiveZone.HIGH_SLOT:
                    zone_mult = 1.5
                else:
                    zone_mult = 1.0

                # Attention weight (inverse distance, capped)
                base_attn = 1.0 / max(dist, 5.0)
                weighted_attn = base_attn * puck_mult * threat_mult * zone_mult

                attn[attacker.player_id] = weighted_attn
                total += weighted_attn

            # Normalize to sum to 1
            if total > 0:
                for aid in attn:
                    attn[aid] /= total

            allocation[defender.player_id] = attn

        return allocation

    def compute_attacker_gravity(
        self,
        attacker: AttackerPosition,
        defenders: List[DefenderPosition],
    ) -> float:
        """
        Compute how much attention an attacker draws from defenders.

        Returns: gravity score (0-1)
        """
        total_attention = 0.0

        for defender in defenders:
            dist = np.sqrt(
                (defender.x - attacker.x)**2 +
                (defender.y - attacker.y)**2
            )

            # Attention paid (closer = more attention)
            if dist < 10:
                attention = 0.4
            elif dist < 20:
                attention = 0.2
            elif dist < 30:
                attention = 0.1
            else:
                attention = 0.0

            total_attention += attention

        # Normalize by number of defenders
        if defenders:
            return min(1.0, total_attention / len(defenders))
        return 0.0

    def find_space_creators(
        self,
        attackers: List[AttackerPosition],
        defenders: List[DefenderPosition],
    ) -> List[Dict[str, Any]]:
        """
        Identify attackers creating space for teammates.

        High gravity players create space for others.
        """
        results = []

        for attacker in attackers:
            gravity = self.compute_attacker_gravity(attacker, defenders)

            # Teammates who benefit
            beneficiaries = []
            for other in attackers:
                if other.player_id == attacker.player_id:
                    continue

                # Did this player's gravity create space for other?
                # Check if defenders pulled toward this player
                for defender in defenders:
                    dist_to_attacker = np.sqrt(
                        (defender.x - attacker.x)**2 +
                        (defender.y - attacker.y)**2
                    )
                    dist_to_other = np.sqrt(
                        (defender.x - other.x)**2 +
                        (defender.y - other.y)**2
                    )

                    if dist_to_attacker < dist_to_other and dist_to_attacker < 15:
                        beneficiaries.append(other.player_id)
                        break

            results.append({
                'player_id': attacker.player_id,
                'gravity': gravity,
                'space_created_for': list(set(beneficiaries)),
                'has_puck': attacker.has_puck,
            })

        # Sort by gravity
        results.sort(key=lambda x: x['gravity'], reverse=True)
        return results


class ZoneBreakdownAnalyzer:
    """
    Analyzes defensive breakdowns by zone.

    Identifies patterns in where and how defensive coverage fails.
    """

    def __init__(self, spatial_model: DefensiveSpatialModel):
        """Initialize with spatial model."""
        self.spatial_model = spatial_model

    def analyze_goal_against(
        self,
        shot_location: Tuple[float, float],
        defenders: List[DefenderPosition],
        puck_path: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """
        Analyze defensive breakdown that led to goal.

        Args:
            shot_location: Where goal was scored from
            defenders: Defender positions at time of shot
            puck_path: Path of puck leading to goal

        Returns:
            Analysis of what went wrong
        """
        shot_x, shot_y = shot_location
        zone = self.spatial_model._get_zone(shot_x, shot_y)

        # Compute what suppression should have been
        actual_suppression = self.spatial_model.compute_shot_suppression(
            shot_location, defenders
        )

        # Find closest defender
        min_dist = float('inf')
        closest_defender = None
        for defender in defenders:
            dist = np.sqrt(
                (defender.x - shot_x)**2 + (defender.y - shot_y)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_defender = defender

        # Breakdown type
        if min_dist > 20:
            breakdown_type = "coverage_gap"
        elif min_dist > 10:
            breakdown_type = "loose_coverage"
        elif actual_suppression < 0.2:
            breakdown_type = "poor_positioning"
        else:
            breakdown_type = "execution_failure"

        # Trace back puck path for entry point issues
        path_analysis = self._analyze_puck_path(puck_path, defenders)

        return {
            'shot_zone': zone.value,
            'closest_defender_distance': min_dist,
            'shot_suppression': actual_suppression,
            'breakdown_type': breakdown_type,
            'path_analysis': path_analysis,
            'num_defenders_in_zone': sum(
                1 for d in defenders
                if self.spatial_model._get_zone(d.x, d.y) == zone
            ),
        }

    def _analyze_puck_path(
        self,
        puck_path: List[Tuple[float, float]],
        defenders: List[DefenderPosition],
    ) -> Dict[str, Any]:
        """Analyze puck movement path for defensive issues."""
        if len(puck_path) < 2:
            return {'issue': 'insufficient_data'}

        # Find where puck entered dangerous zone
        entry_point = None
        for i, (x, y) in enumerate(puck_path):
            zone = self.spatial_model._get_zone(x, y)
            if zone in [DefensiveZone.SLOT, DefensiveZone.HIGH_SLOT]:
                entry_point = (x, y)
                break

        if entry_point is None:
            return {'issue': 'no_zone_entry'}

        # Was entry contested?
        entry_x, entry_y = entry_point
        contested = False
        for defender in defenders:
            dist = np.sqrt(
                (defender.x - entry_x)**2 + (defender.y - entry_y)**2
            )
            if dist < 10:
                contested = True
                break

        return {
            'entry_point': entry_point,
            'entry_contested': contested,
            'path_length': len(puck_path),
        }

    def compute_zone_vulnerability(
        self,
        goals_against: List[Dict[str, Any]],
    ) -> Dict[DefensiveZone, float]:
        """
        Compute vulnerability by zone based on goals against.

        Returns zone -> vulnerability score.
        """
        zone_counts: Dict[DefensiveZone, int] = defaultdict(int)
        total_goals = len(goals_against)

        for ga in goals_against:
            zone_str = ga.get('shot_zone', '')
            try:
                zone = DefensiveZone(zone_str)
                zone_counts[zone] += 1
            except ValueError:
                continue

        # Normalize and weight by zone danger
        vulnerability = {}
        for zone in DefensiveZone:
            count = zone_counts[zone]
            if total_goals > 0:
                rate = count / total_goals
            else:
                rate = 0.0

            vulnerability[zone] = rate

        return vulnerability
