"""
Physics-Driven Voronoi Ice Dominance Model

Implements physics-based spatial dominance modeling translated from
Efthimiou's soccer research on Voronoi diagrams.

Key Papers:
- Efthimiou, C.J. (2021). "The Voronoi Diagram in Soccer: A Theoretical
  Study to Measure Dominance Space." arXiv:2107.05714
- Efthimiou, C.J. (2022). "A Physics-Driven Study of Dominance Space
  in Soccer." arXiv:2202.00414

Key Innovations:
- Traditional Voronoi assumes equal speed - incorrect in practice
- Physics extension accounts for velocity, acceleration, direction
- Asymmetric influence (players control more in running direction)
- Includes frictional forces (air resistance, muscle energy)

Hockey Translation:
- Weight by skating speed (faster than soccer running)
- Account for goalie position and crease control
- Model defensive gap management
- Consider momentum when changing direction
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class InfluenceModel(Enum):
    """Types of influence models for Voronoi calculations."""
    STANDARD = "standard"           # Traditional equal-speed Voronoi
    VELOCITY_WEIGHTED = "velocity"  # Accounts for current velocity
    PHYSICS_DRIVEN = "physics"      # Full physics model with acceleration
    ASYMMETRIC = "asymmetric"       # Direction-dependent influence


@dataclass
class SkaterState:
    """Complete physical state of a skater."""
    player_id: str
    team_id: str
    x: float                        # feet
    y: float                        # feet
    velocity_x: float = 0.0         # ft/s
    velocity_y: float = 0.0         # ft/s
    acceleration_x: float = 0.0     # ft/s^2
    acceleration_y: float = 0.0     # ft/s^2
    max_speed: float = 30.0         # ft/s (about 20 mph)
    max_acceleration: float = 15.0  # ft/s^2
    friction_coefficient: float = 0.1  # Ice friction
    is_goalie: bool = False
    fatigue_factor: float = 1.0     # 1.0 = fresh, <1.0 = tired

    @property
    def speed(self) -> float:
        """Current speed in ft/s."""
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2)

    @property
    def direction(self) -> Tuple[float, float]:
        """Unit vector of current direction."""
        speed = self.speed
        if speed < 0.1:
            return (0.0, 0.0)
        return (self.velocity_x / speed, self.velocity_y / speed)


@dataclass
class DominanceCell:
    """Dominance information for a single grid cell."""
    x: float
    y: float
    controlling_team: Optional[str]
    control_probability: float      # 0-1, probability of control
    time_to_reach: Dict[str, float] # player_id -> time to reach
    contested: bool                 # True if close contest
    uncertainty: float              # 0-1, confidence in assignment


@dataclass
class DominanceMap:
    """Full ice dominance map."""
    grid: np.ndarray                # 2D array of control probabilities
    team_a_id: str
    team_b_id: str
    team_a_control: float           # % of ice controlled by team A
    team_b_control: float           # % of ice controlled by team B
    contested_area: float           # % of ice that's contested
    offensive_zone_control: Dict[str, float]
    neutral_zone_control: Dict[str, float]
    defensive_zone_control: Dict[str, float]
    timestamp: float = 0.0


class PhysicsVoronoiModel:
    """
    Physics-driven Voronoi model for ice dominance.

    Unlike traditional Voronoi which assigns regions based on distance,
    this model accounts for:
    - Current velocity (momentum toward/away from point)
    - Maximum acceleration capabilities
    - Direction of travel (asymmetric influence)
    - Fatigue effects on performance
    """

    # Physical constants for hockey
    GRAVITY = 32.174  # ft/s^2
    ICE_FRICTION = 0.03  # coefficient of friction on ice
    AIR_RESISTANCE = 0.001  # drag coefficient
    REACTION_TIME = 0.15  # seconds before direction change

    def __init__(
        self,
        grid_x: int = 100,
        grid_y: int = 43,
        rink_length: float = 200.0,
        rink_width: float = 85.0,
        model_type: InfluenceModel = InfluenceModel.PHYSICS_DRIVEN,
        contest_threshold: float = 0.1,  # seconds difference to be "contested"
    ):
        """
        Initialize physics-based Voronoi model.

        Args:
            grid_x: Grid resolution along rink length
            grid_y: Grid resolution along rink width
            rink_length: Rink length in feet
            rink_width: Rink width in feet
            model_type: Type of influence calculation
            contest_threshold: Time difference threshold for contested areas
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.rink_length = rink_length
        self.rink_width = rink_width
        self.model_type = model_type
        self.contest_threshold = contest_threshold

        # Create coordinate grids
        self.x_coords = np.linspace(0, rink_length, grid_x)
        self.y_coords = np.linspace(0, rink_width, grid_y)
        self.xx, self.yy = np.meshgrid(self.x_coords, self.y_coords)

        # Zone boundaries (NHL standard)
        self.offensive_blue = 175.0  # feet from own goal
        self.defensive_blue = 25.0   # feet from own goal

    def time_to_point_standard(
        self,
        player: SkaterState,
        target_x: float,
        target_y: float,
    ) -> float:
        """
        Standard time calculation (distance / max speed).

        This is what traditional Voronoi uses - incorrect because
        it ignores current velocity and acceleration limits.
        """
        distance = np.sqrt(
            (target_x - player.x)**2 +
            (target_y - player.y)**2
        )
        effective_speed = player.max_speed * player.fatigue_factor
        return distance / max(effective_speed, 0.1)

    def time_to_point_velocity(
        self,
        player: SkaterState,
        target_x: float,
        target_y: float,
    ) -> float:
        """
        Velocity-weighted time calculation.

        Accounts for current velocity - player moving toward target
        arrives faster than one moving away.
        """
        dx = target_x - player.x
        dy = target_y - player.y
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.1:
            return 0.0

        # Unit vector to target
        ux = dx / distance
        uy = dy / distance

        # Component of velocity toward target
        velocity_toward = player.velocity_x * ux + player.velocity_y * uy

        # Effective speed considering momentum
        # If moving toward target, effective speed is higher
        # If moving away, need to slow down first
        effective_speed = player.max_speed * player.fatigue_factor

        if velocity_toward > 0:
            # Moving toward target - use average of current and max
            avg_speed = (player.speed + effective_speed) / 2
        else:
            # Moving away - need reaction time plus deceleration
            # Time to stop + time to accelerate toward target
            decel_time = abs(velocity_toward) / player.max_acceleration
            avg_speed = effective_speed / 2  # Average during acceleration
            return self.REACTION_TIME + decel_time + distance / avg_speed

        return distance / max(avg_speed, 0.1)

    def time_to_point_physics(
        self,
        player: SkaterState,
        target_x: float,
        target_y: float,
    ) -> float:
        """
        Full physics-based time calculation.

        Models realistic skating dynamics including:
        - Reaction time before direction change
        - Deceleration if moving wrong direction
        - Acceleration limits
        - Friction and fatigue effects
        """
        dx = target_x - player.x
        dy = target_y - player.y
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.1:
            return 0.0

        # Direction to target
        ux = dx / distance
        uy = dy / distance

        # Current velocity components
        v_toward = player.velocity_x * ux + player.velocity_y * uy
        v_perp = np.sqrt(
            player.velocity_x**2 + player.velocity_y**2 - v_toward**2
        ) if player.speed > abs(v_toward) else 0.0

        # Effective max speed with fatigue
        v_max = player.max_speed * player.fatigue_factor
        a_max = player.max_acceleration * player.fatigue_factor

        # Friction deceleration
        friction_decel = self.ICE_FRICTION * self.GRAVITY

        total_time = 0.0

        # Phase 1: Reaction time if significant direction change needed
        if v_toward < player.speed * 0.5:  # >60 degree turn
            total_time += self.REACTION_TIME

        # Phase 2: Decelerate perpendicular velocity
        if v_perp > 0:
            perp_time = v_perp / (a_max + friction_decel)
            total_time += perp_time

        # Phase 3: Handle velocity toward/away from target
        remaining_distance = distance
        current_velocity = max(v_toward, 0.0)

        if v_toward < 0:
            # Moving away - need to stop first
            stop_time = abs(v_toward) / (a_max + friction_decel)
            total_time += stop_time
            remaining_distance += abs(v_toward) * stop_time / 2
            current_velocity = 0.0

        # Phase 4: Accelerate toward target
        if current_velocity < v_max:
            # Time to reach max speed
            accel_time = (v_max - current_velocity) / a_max
            # Distance covered during acceleration
            accel_distance = current_velocity * accel_time + 0.5 * a_max * accel_time**2

            if accel_distance >= remaining_distance:
                # Reach target before max speed
                # Solve: d = v0*t + 0.5*a*t^2
                discriminant = current_velocity**2 + 2 * a_max * remaining_distance
                if discriminant >= 0:
                    total_time += (-current_velocity + np.sqrt(discriminant)) / a_max
                    return total_time

            total_time += accel_time
            remaining_distance -= accel_distance
            current_velocity = v_max

        # Phase 5: Coast at max speed
        if remaining_distance > 0:
            total_time += remaining_distance / current_velocity

        return total_time

    def calculate_influence(
        self,
        player: SkaterState,
        target_x: float,
        target_y: float,
    ) -> float:
        """
        Calculate player's influence/control probability at a point.

        Returns time to reach point based on model type.
        Lower time = higher influence.
        """
        if self.model_type == InfluenceModel.STANDARD:
            return self.time_to_point_standard(player, target_x, target_y)
        elif self.model_type == InfluenceModel.VELOCITY_WEIGHTED:
            return self.time_to_point_velocity(player, target_x, target_y)
        elif self.model_type == InfluenceModel.PHYSICS_DRIVEN:
            return self.time_to_point_physics(player, target_x, target_y)
        elif self.model_type == InfluenceModel.ASYMMETRIC:
            return self._asymmetric_influence(player, target_x, target_y)
        else:
            return self.time_to_point_physics(player, target_x, target_y)

    def _asymmetric_influence(
        self,
        player: SkaterState,
        target_x: float,
        target_y: float,
    ) -> float:
        """
        Asymmetric influence model.

        Players have stronger influence in their direction of travel
        due to momentum and anticipation.
        """
        base_time = self.time_to_point_physics(player, target_x, target_y)

        # Check alignment with direction of travel
        if player.speed < 0.5:
            return base_time

        dx = target_x - player.x
        dy = target_y - player.y
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.1:
            return 0.0

        # Unit vector to target
        ux = dx / distance
        uy = dy / distance

        # Dot product with velocity direction
        dir_x, dir_y = player.direction
        alignment = ux * dir_x + uy * dir_y

        # Reduce time for aligned directions (player anticipates)
        # Increase time for opposite directions
        asymmetry_factor = 1.0 - 0.3 * alignment  # +-30% adjustment

        return base_time * asymmetry_factor

    def compute_dominance_map(
        self,
        players: List[SkaterState],
    ) -> DominanceMap:
        """
        Compute full ice dominance map.

        For each grid cell, determines which team controls it
        based on fastest player to reach that point.

        Args:
            players: List of all player states

        Returns:
            DominanceMap with control probabilities
        """
        if not players:
            return self._empty_dominance_map()

        # Get unique teams
        teams = list(set(p.team_id for p in players))
        if len(teams) != 2:
            teams = teams[:2] if len(teams) > 2 else teams + ["unknown"]
        team_a, team_b = teams[0], teams[1]

        # Initialize grid
        dominance_grid = np.zeros((self.grid_y, self.grid_x))

        # For each grid cell
        for i, y in enumerate(self.y_coords):
            for j, x in enumerate(self.x_coords):
                # Calculate time for each player to reach this point
                times_a = []
                times_b = []

                for player in players:
                    time = self.calculate_influence(player, x, y)

                    # Goalies have limited influence outside crease
                    if player.is_goalie:
                        crease_dist = self._distance_from_crease(player, x, y)
                        if crease_dist > 10:  # feet
                            time *= 3.0  # Goalies slow outside crease

                    if player.team_id == team_a:
                        times_a.append(time)
                    else:
                        times_b.append(time)

                # Minimum time for each team
                min_time_a = min(times_a) if times_a else float('inf')
                min_time_b = min(times_b) if times_b else float('inf')

                # Convert to control probability using logistic
                time_diff = min_time_a - min_time_b
                # Positive = team B faster, negative = team A faster
                # Scale so 0.5 second diff = ~88% probability
                prob_a = 1.0 / (1.0 + np.exp(time_diff * 4))

                dominance_grid[i, j] = prob_a

        # Calculate zone control
        team_a_control = np.mean(dominance_grid)
        team_b_control = 1.0 - team_a_control
        contested = np.mean((dominance_grid > 0.4) & (dominance_grid < 0.6))

        # Zone-specific control
        oz_mask = self.xx > self.offensive_blue
        nz_mask = (self.xx >= self.defensive_blue) & (self.xx <= self.offensive_blue)
        dz_mask = self.xx < self.defensive_blue

        oz_control = {
            team_a: float(np.mean(dominance_grid[oz_mask.T])),
            team_b: 1.0 - float(np.mean(dominance_grid[oz_mask.T])),
        }
        nz_control = {
            team_a: float(np.mean(dominance_grid[nz_mask.T])),
            team_b: 1.0 - float(np.mean(dominance_grid[nz_mask.T])),
        }
        dz_control = {
            team_a: float(np.mean(dominance_grid[dz_mask.T])),
            team_b: 1.0 - float(np.mean(dominance_grid[dz_mask.T])),
        }

        return DominanceMap(
            grid=dominance_grid,
            team_a_id=team_a,
            team_b_id=team_b,
            team_a_control=float(team_a_control),
            team_b_control=float(team_b_control),
            contested_area=float(contested),
            offensive_zone_control=oz_control,
            neutral_zone_control=nz_control,
            defensive_zone_control=dz_control,
        )

    def _distance_from_crease(
        self,
        player: SkaterState,
        x: float,
        y: float,
    ) -> float:
        """Calculate distance from goalie's crease center."""
        # Determine which crease based on player position
        if player.x < self.rink_length / 2:
            crease_x = 11.0  # feet from end boards
        else:
            crease_x = self.rink_length - 11.0
        crease_y = self.rink_width / 2

        return np.sqrt((x - crease_x)**2 + (y - crease_y)**2)

    def _empty_dominance_map(self) -> DominanceMap:
        """Return empty dominance map."""
        return DominanceMap(
            grid=np.full((self.grid_y, self.grid_x), 0.5),
            team_a_id="home",
            team_b_id="away",
            team_a_control=0.5,
            team_b_control=0.5,
            contested_area=1.0,
            offensive_zone_control={"home": 0.5, "away": 0.5},
            neutral_zone_control={"home": 0.5, "away": 0.5},
            defensive_zone_control={"home": 0.5, "away": 0.5},
        )

    def compare_models(
        self,
        players: List[SkaterState],
    ) -> Dict[str, DominanceMap]:
        """
        Compare different influence models.

        Useful for demonstrating the importance of physics-based
        calculations over simple distance-based Voronoi.
        """
        results = {}
        original_model = self.model_type

        for model_type in InfluenceModel:
            self.model_type = model_type
            results[model_type.value] = self.compute_dominance_map(players)

        self.model_type = original_model
        return results


class GapAnalyzer:
    """
    Analyzes defensive gaps using physics-based Voronoi.

    Identifies areas of vulnerability and optimal positioning.
    """

    def __init__(self, voronoi_model: PhysicsVoronoiModel):
        """Initialize with Voronoi model."""
        self.model = voronoi_model

    def find_gaps(
        self,
        defenders: List[SkaterState],
        attackers: List[SkaterState],
    ) -> List[Dict[str, Any]]:
        """
        Find dangerous gaps in defensive coverage.

        A gap is dangerous when:
        1. No defender can reach it quickly
        2. An attacker can reach it faster than defenders
        3. It's in a dangerous scoring area
        """
        gaps = []

        # Check grid for gaps
        for i, y in enumerate(self.model.y_coords):
            for j, x in enumerate(self.model.x_coords):
                # Time for closest defender
                def_times = [
                    self.model.calculate_influence(d, x, y)
                    for d in defenders if not d.is_goalie
                ]
                min_def_time = min(def_times) if def_times else float('inf')

                # Time for closest attacker
                att_times = [
                    self.model.calculate_influence(a, x, y)
                    for a in attackers
                ]
                min_att_time = min(att_times) if att_times else float('inf')

                # Gap danger score
                time_advantage = min_def_time - min_att_time

                # Danger zones: slot, high slot, circles
                danger_multiplier = self._danger_zone_multiplier(x, y)

                if time_advantage > 0.3 and danger_multiplier > 0.5:
                    gaps.append({
                        'x': x,
                        'y': y,
                        'time_advantage': time_advantage,
                        'danger_level': time_advantage * danger_multiplier,
                        'defender_time': min_def_time,
                        'attacker_time': min_att_time,
                    })

        # Sort by danger level
        gaps.sort(key=lambda g: g['danger_level'], reverse=True)

        return gaps[:10]  # Top 10 most dangerous gaps

    def _danger_zone_multiplier(self, x: float, y: float) -> float:
        """
        Calculate danger multiplier based on location.

        Slot and crease areas are most dangerous.
        """
        # Assuming attacking goal at x = 200 (right side)
        goal_x = self.model.rink_length - 11
        goal_y = self.model.rink_width / 2

        dist_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

        # Slot is roughly 0-30 feet from goal, 20 feet wide
        if dist_to_goal < 15:  # Crease/doorstep
            return 2.0
        elif dist_to_goal < 30 and abs(y - goal_y) < 10:  # Slot
            return 1.5
        elif dist_to_goal < 40:  # High slot / circles
            return 1.0
        elif dist_to_goal < 60:  # Top of circles
            return 0.5
        else:
            return 0.2

    def optimal_defender_position(
        self,
        current_position: SkaterState,
        attackers: List[SkaterState],
        other_defenders: List[SkaterState],
    ) -> Tuple[float, float]:
        """
        Calculate optimal position for a defender.

        Balances:
        - Covering dangerous gaps
        - Not overlapping with other defenders
        - Ability to reach attacker positions
        """
        best_pos = (current_position.x, current_position.y)
        best_score = float('-inf')

        # Search nearby positions
        search_radius = 15.0  # feet
        search_step = 3.0

        for dx in np.arange(-search_radius, search_radius + search_step, search_step):
            for dy in np.arange(-search_radius, search_radius + search_step, search_step):
                test_x = current_position.x + dx
                test_y = current_position.y + dy

                # Stay on ice
                if test_x < 0 or test_x > self.model.rink_length:
                    continue
                if test_y < 0 or test_y > self.model.rink_width:
                    continue

                # Calculate position score
                score = self._position_score(
                    test_x, test_y,
                    attackers, other_defenders
                )

                if score > best_score:
                    best_score = score
                    best_pos = (test_x, test_y)

        return best_pos

    def _position_score(
        self,
        x: float,
        y: float,
        attackers: List[SkaterState],
        other_defenders: List[SkaterState],
    ) -> float:
        """Score a defensive position."""
        score = 0.0

        # Reward covering dangerous zones
        danger = self._danger_zone_multiplier(x, y)
        score += danger * 10

        # Reward being close enough to attackers to contest
        for attacker in attackers:
            dist = np.sqrt((x - attacker.x)**2 + (y - attacker.y)**2)
            if dist < 20:  # Within contestable range
                score += 5 * (1 - dist / 20)

        # Penalize overlapping with other defenders
        for defender in other_defenders:
            dist = np.sqrt((x - defender.x)**2 + (y - defender.y)**2)
            if dist < 15:  # Too close
                score -= 10 * (1 - dist / 15)

        return score


class MomentumTracker:
    """
    Tracks momentum in ice control over time.

    Identifies when control is shifting between teams.
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize tracker.

        Args:
            window_size: Number of frames to track
        """
        self.window_size = window_size
        self.history: List[DominanceMap] = []

    def update(self, dominance_map: DominanceMap) -> Dict[str, Any]:
        """
        Update tracker with new dominance map.

        Returns momentum analysis.
        """
        self.history.append(dominance_map)

        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

        if len(self.history) < 2:
            return {
                'momentum_direction': 'neutral',
                'momentum_strength': 0.0,
                'control_trend': 0.0,
            }

        # Calculate control trend
        controls = [h.team_a_control for h in self.history]

        # Linear regression for trend
        n = len(controls)
        x = np.arange(n)
        slope = (n * np.sum(x * controls) - np.sum(x) * np.sum(controls)) / \
                (n * np.sum(x**2) - np.sum(x)**2)

        # Recent change
        recent_change = controls[-1] - controls[0] if len(controls) > 1 else 0

        if slope > 0.005:
            direction = 'team_a_gaining'
        elif slope < -0.005:
            direction = 'team_b_gaining'
        else:
            direction = 'neutral'

        return {
            'momentum_direction': direction,
            'momentum_strength': abs(slope) * 100,
            'control_trend': float(recent_change),
            'current_control_a': controls[-1],
            'current_control_b': 1 - controls[-1],
        }

    def detect_shift(self, threshold: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Detect significant momentum shifts.

        Returns details if a shift is detected.
        """
        if len(self.history) < 10:
            return None

        # Compare first half to second half
        mid = len(self.history) // 2
        first_half_avg = np.mean([h.team_a_control for h in self.history[:mid]])
        second_half_avg = np.mean([h.team_a_control for h in self.history[mid:]])

        shift = second_half_avg - first_half_avg

        if abs(shift) > threshold:
            return {
                'shift_detected': True,
                'shift_magnitude': float(shift),
                'benefiting_team': 'team_a' if shift > 0 else 'team_b',
                'pre_shift_control': float(first_half_avg),
                'post_shift_control': float(second_half_avg),
            }

        return None
