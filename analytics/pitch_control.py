"""
Pitch Control Model - Continuous Ice Control Probability

Implements the pitch control methodology from:
- Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan.
- Fernández, J., & Bornn, L. (2018). "Wide Open Spaces: A Statistical Technique
  for Measuring Space Creation in Professional Soccer." MIT Sloan.

This creates probability density functions showing the likelihood of each team
gaining control at any point on the ice, accounting for player positions,
velocities, and ball/puck travel time.

Translated to hockey with:
- Hockey-specific skating speeds and acceleration
- Puck travel speeds for passes
- Goalie crease special handling
- Board physics for bank passes
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime


class ControlModel(Enum):
    """Control model types."""
    SPEARMAN = "spearman"  # Original Spearman model
    FERNANDEZ = "fernandez"  # Fernández-Bornn extension
    HYBRID = "hybrid"  # Combined approach


class PlayerRole(Enum):
    """Player roles affecting control calculations."""
    SKATER = "skater"
    GOALIE = "goalie"


@dataclass
class SkaterPhysics:
    """Physics parameters for skating."""
    max_speed: float = 12.0  # m/s (approximately 27 mph)
    reaction_time: float = 0.7  # seconds
    acceleration: float = 3.5  # m/s^2
    deceleration: float = 4.0  # m/s^2
    direction_change_time: float = 0.5  # seconds to change direction


@dataclass
class PuckPhysics:
    """Physics parameters for puck movement."""
    pass_speed: float = 25.0  # m/s average pass speed
    shot_speed: float = 40.0  # m/s average shot speed
    friction_coefficient: float = 0.02  # ice friction
    board_damping: float = 0.8  # velocity retained after board contact


@dataclass
class PlayerState:
    """Current state of a player."""
    player_id: str
    team_id: str
    x: float  # meters from center ice
    y: float  # meters from center ice
    vx: float = 0.0  # velocity x component
    vy: float = 0.0  # velocity y component
    role: PlayerRole = PlayerRole.SKATER

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)


@dataclass
class ControlPoint:
    """Control probability at a specific point."""
    x: float
    y: float
    home_control: float  # probability [0, 1]
    away_control: float  # probability [0, 1]
    contested: float  # degree of contestation
    nearest_home: Optional[str] = None
    nearest_away: Optional[str] = None
    time_to_home: float = 0.0
    time_to_away: float = 0.0


@dataclass
class ControlGrid:
    """Full control surface for the rink."""
    grid: np.ndarray  # shape (n_x, n_y, 2) for home/away control
    x_coords: np.ndarray
    y_coords: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    home_total_control: float = 0.0
    away_total_control: float = 0.0

    def get_control_at(self, x: float, y: float) -> Tuple[float, float]:
        """Get home/away control at specific point."""
        x_idx = np.argmin(np.abs(self.x_coords - x))
        y_idx = np.argmin(np.abs(self.y_coords - y))
        return self.grid[x_idx, y_idx, 0], self.grid[x_idx, y_idx, 1]


@dataclass
class SpaceCreation:
    """Space creation metrics for a player."""
    player_id: str
    space_created: float  # square meters of space
    space_value: float  # weighted by danger
    space_quality: float  # average control in created space
    movement_efficiency: float  # space created per distance moved


@dataclass
class PassingLane:
    """Passing lane analysis."""
    from_player: str
    to_player: str
    lane_control: float  # probability of successful pass
    intercept_probability: float
    expected_value: float  # value if pass completed


class PitchControlModel:
    """
    Continuous ice control probability model.

    Based on Spearman (2018) with extensions for hockey.
    """

    def __init__(
        self,
        skater_physics: Optional[SkaterPhysics] = None,
        puck_physics: Optional[PuckPhysics] = None,
        grid_resolution: int = 50,
        model_type: ControlModel = ControlModel.HYBRID
    ):
        self.skater_physics = skater_physics or SkaterPhysics()
        self.puck_physics = puck_physics or PuckPhysics()
        self.grid_resolution = grid_resolution
        self.model_type = model_type

        # Rink dimensions (NHL)
        self.rink_length = 60.96  # meters (200 ft)
        self.rink_width = 25.91  # meters (85 ft)

        # Create coordinate grid
        self.x_grid = np.linspace(-self.rink_length/2, self.rink_length/2, grid_resolution)
        self.y_grid = np.linspace(-self.rink_width/2, self.rink_width/2, grid_resolution)

    def time_to_intercept(
        self,
        player: PlayerState,
        target: np.ndarray,
        puck_travel_time: float = 0.0
    ) -> float:
        """
        Calculate time for player to reach target point.

        Uses physics model accounting for reaction time,
        current velocity, and acceleration limits.
        """
        position = player.position
        velocity = player.velocity

        displacement = target - position
        distance = np.linalg.norm(displacement)

        if distance < 0.1:  # Already there
            return 0.0

        # Direction to target
        direction = displacement / distance

        # Current velocity component toward target
        v_toward = np.dot(velocity, direction)

        # Time model based on Spearman
        # T = reaction_time + time to cover distance

        if player.role == PlayerRole.GOALIE:
            # Goalies move differently (shuffles, butterfly)
            reaction = 0.3
            max_speed = 6.0  # Lower than skaters
        else:
            reaction = self.skater_physics.reaction_time
            max_speed = self.skater_physics.max_speed

        # Account for current velocity
        if v_toward > 0:
            # Already moving toward target
            time_to_max = (max_speed - v_toward) / self.skater_physics.acceleration
            dist_during_accel = v_toward * time_to_max + 0.5 * self.skater_physics.acceleration * time_to_max**2

            if dist_during_accel >= distance:
                # Reach target during acceleration
                # d = v0*t + 0.5*a*t^2, solve for t
                a = 0.5 * self.skater_physics.acceleration
                b = v_toward
                c = -distance
                t = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                return reaction + t
            else:
                # Need to accelerate then cruise
                remaining_dist = distance - dist_during_accel
                cruise_time = remaining_dist / max_speed
                return reaction + time_to_max + cruise_time
        else:
            # Moving away from target or stationary
            # Need to decelerate (if moving away), then accelerate toward
            if v_toward < 0:
                # Moving away
                decel_time = abs(v_toward) / self.skater_physics.deceleration
                dist_during_decel = abs(v_toward) * decel_time / 2  # Moving away during decel
                distance += dist_during_decel

            # Now accelerate from rest
            time_to_max = max_speed / self.skater_physics.acceleration
            dist_during_accel = 0.5 * self.skater_physics.acceleration * time_to_max**2

            if dist_during_accel >= distance:
                # Reach during acceleration
                t = np.sqrt(2 * distance / self.skater_physics.acceleration)
                return reaction + (decel_time if v_toward < 0 else 0) + t
            else:
                remaining_dist = distance - dist_during_accel
                cruise_time = remaining_dist / max_speed
                return reaction + (decel_time if v_toward < 0 else 0) + time_to_max + cruise_time

    def control_probability(
        self,
        home_times: List[float],
        away_times: List[float],
        sigma: float = 0.45
    ) -> Tuple[float, float]:
        """
        Calculate control probability using sigmoid model.

        Based on logistic function of time difference.
        """
        if not home_times and not away_times:
            return 0.5, 0.5

        min_home = min(home_times) if home_times else float('inf')
        min_away = min(away_times) if away_times else float('inf')

        if min_home == float('inf') and min_away == float('inf'):
            return 0.5, 0.5

        # Time difference
        dt = min_home - min_away

        # Sigmoid function
        # P(home) = 1 / (1 + exp(dt/sigma))
        # When home arrives first (dt < 0), P(home) > 0.5
        p_home = 1 / (1 + np.exp(dt / sigma))
        p_away = 1 - p_home

        return p_home, p_away

    def compute_control_grid(
        self,
        home_players: List[PlayerState],
        away_players: List[PlayerState],
        puck_position: Optional[np.ndarray] = None
    ) -> ControlGrid:
        """
        Compute full control surface for current frame.

        Returns grid showing home/away control probability at each point.
        """
        grid = np.zeros((len(self.x_grid), len(self.y_grid), 2))

        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                target = np.array([x, y])

                # Calculate puck travel time if position known
                puck_time = 0.0
                if puck_position is not None:
                    puck_dist = np.linalg.norm(target - puck_position)
                    puck_time = puck_dist / self.puck_physics.pass_speed

                # Time for each player to reach point
                home_times = [
                    self.time_to_intercept(p, target, puck_time)
                    for p in home_players
                ]
                away_times = [
                    self.time_to_intercept(p, target, puck_time)
                    for p in away_players
                ]

                p_home, p_away = self.control_probability(home_times, away_times)
                grid[i, j, 0] = p_home
                grid[i, j, 1] = p_away

        # Calculate total control (integrate over surface)
        cell_area = (self.x_grid[1] - self.x_grid[0]) * (self.y_grid[1] - self.y_grid[0])
        home_total = np.sum(grid[:, :, 0]) * cell_area
        away_total = np.sum(grid[:, :, 1]) * cell_area

        return ControlGrid(
            grid=grid,
            x_coords=self.x_grid,
            y_coords=self.y_grid,
            home_total_control=home_total,
            away_total_control=away_total
        )

    def compute_control_at_point(
        self,
        point: np.ndarray,
        home_players: List[PlayerState],
        away_players: List[PlayerState],
        puck_position: Optional[np.ndarray] = None
    ) -> ControlPoint:
        """Compute control probability at a single point."""
        puck_time = 0.0
        if puck_position is not None:
            puck_dist = np.linalg.norm(point - puck_position)
            puck_time = puck_dist / self.puck_physics.pass_speed

        home_times = [
            (p.player_id, self.time_to_intercept(p, point, puck_time))
            for p in home_players
        ]
        away_times = [
            (p.player_id, self.time_to_intercept(p, point, puck_time))
            for p in away_players
        ]

        min_home = min(home_times, key=lambda x: x[1]) if home_times else (None, float('inf'))
        min_away = min(away_times, key=lambda x: x[1]) if away_times else (None, float('inf'))

        p_home, p_away = self.control_probability(
            [t for _, t in home_times],
            [t for _, t in away_times]
        )

        # Contestation: high when probabilities are close
        contested = 1 - abs(p_home - p_away)

        return ControlPoint(
            x=point[0],
            y=point[1],
            home_control=p_home,
            away_control=p_away,
            contested=contested,
            nearest_home=min_home[0],
            nearest_away=min_away[0],
            time_to_home=min_home[1],
            time_to_away=min_away[1]
        )


class SpaceCreationAnalyzer:
    """
    Analyze space creation using pitch control.

    Based on Fernández & Bornn (2018) methodology.
    """

    def __init__(self, control_model: PitchControlModel):
        self.control_model = control_model
        self.danger_weights = self._create_danger_grid()

    def _create_danger_grid(self) -> np.ndarray:
        """Create grid of scoring danger by location."""
        x_grid = self.control_model.x_grid
        y_grid = self.control_model.y_grid

        danger = np.zeros((len(x_grid), len(y_grid)))

        # Goal positions
        goal_x = self.control_model.rink_length / 2

        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                # Distance to offensive goal
                dist_to_goal = np.sqrt((goal_x - x)**2 + y**2)

                # Angle to goal
                angle = np.arctan2(abs(y), goal_x - x)

                # Danger decreases with distance and angle
                danger[i, j] = np.exp(-dist_to_goal / 10) * np.cos(angle)
                danger[i, j] = max(0, danger[i, j])

        # Normalize
        danger = danger / np.max(danger)
        return danger

    def compute_space_created(
        self,
        player: PlayerState,
        teammates: List[PlayerState],
        opponents: List[PlayerState],
        previous_position: np.ndarray
    ) -> SpaceCreation:
        """
        Compute space created by player movement.

        Space = area where team control increased due to movement.
        """
        # Control before movement
        player_before = PlayerState(
            player_id=player.player_id,
            team_id=player.team_id,
            x=previous_position[0],
            y=previous_position[1],
            vx=0, vy=0,
            role=player.role
        )

        all_home_before = teammates + [player_before]
        control_before = self.control_model.compute_control_grid(
            all_home_before, opponents
        )

        # Control after movement
        all_home_after = teammates + [player]
        control_after = self.control_model.compute_control_grid(
            all_home_after, opponents
        )

        # Difference in control
        control_diff = control_after.grid[:, :, 0] - control_before.grid[:, :, 0]

        # Space created where control increased
        space_increase = np.maximum(control_diff, 0)

        # Calculate metrics
        cell_area = (self.control_model.x_grid[1] - self.control_model.x_grid[0]) * \
                    (self.control_model.y_grid[1] - self.control_model.y_grid[0])

        space_created = np.sum(space_increase > 0.1) * cell_area  # Significant increase
        space_value = np.sum(space_increase * self.danger_weights) * cell_area

        # Average control quality in created space
        mask = space_increase > 0.1
        if np.any(mask):
            space_quality = np.mean(control_after.grid[:, :, 0][mask])
        else:
            space_quality = 0.0

        # Movement efficiency
        distance_moved = np.linalg.norm(player.position - previous_position)
        movement_efficiency = space_value / max(distance_moved, 0.1)

        return SpaceCreation(
            player_id=player.player_id,
            space_created=space_created,
            space_value=space_value,
            space_quality=space_quality,
            movement_efficiency=movement_efficiency
        )

    def compute_off_puck_value(
        self,
        player: PlayerState,
        all_home: List[PlayerState],
        all_away: List[PlayerState],
        puck_position: np.ndarray
    ) -> float:
        """
        Compute value of player's off-puck positioning.

        Based on control gained in dangerous areas.
        """
        # Compute control with player
        control_with = self.control_model.compute_control_grid(
            all_home, all_away, puck_position
        )

        # Compute control without player (at center ice)
        teammates = [p for p in all_home if p.player_id != player.player_id]
        neutral_player = PlayerState(
            player_id=player.player_id,
            team_id=player.team_id,
            x=0, y=0, vx=0, vy=0,
            role=player.role
        )
        control_without = self.control_model.compute_control_grid(
            teammates + [neutral_player], all_away, puck_position
        )

        # Value is weighted control difference
        control_diff = control_with.grid[:, :, 0] - control_without.grid[:, :, 0]
        value = np.sum(control_diff * self.danger_weights)

        return value


class PassingLaneAnalyzer:
    """
    Analyze passing lanes using pitch control.

    Integrates control along potential pass paths.
    """

    def __init__(
        self,
        control_model: PitchControlModel,
        n_samples: int = 20
    ):
        self.control_model = control_model
        self.n_samples = n_samples

    def analyze_pass(
        self,
        passer: PlayerState,
        receiver: PlayerState,
        all_home: List[PlayerState],
        all_away: List[PlayerState],
        pass_value: float = 1.0
    ) -> PassingLane:
        """
        Analyze passing lane between two players.

        Samples points along pass path and computes control.
        """
        start = passer.position
        end = receiver.position

        # Sample points along pass path
        path_controls = []
        for t in np.linspace(0, 1, self.n_samples):
            point = start + t * (end - start)

            # Time for puck to reach this point
            puck_time = t * np.linalg.norm(end - start) / self.control_model.puck_physics.pass_speed

            control = self.control_model.compute_control_at_point(
                point, all_home, all_away, start
            )
            path_controls.append(control)

        # Lane control is minimum control along path
        home_controls = [c.home_control for c in path_controls]
        lane_control = min(home_controls)

        # Intercept probability at weakest point
        min_idx = np.argmin(home_controls)
        intercept_probability = path_controls[min_idx].away_control

        # Expected value
        expected_value = lane_control * pass_value

        return PassingLane(
            from_player=passer.player_id,
            to_player=receiver.player_id,
            lane_control=lane_control,
            intercept_probability=intercept_probability,
            expected_value=expected_value
        )

    def find_best_pass(
        self,
        passer: PlayerState,
        teammates: List[PlayerState],
        all_home: List[PlayerState],
        all_away: List[PlayerState],
        pass_values: Optional[Dict[str, float]] = None
    ) -> Optional[PassingLane]:
        """Find the best passing option from current position."""
        if not teammates:
            return None

        pass_values = pass_values or {t.player_id: 1.0 for t in teammates}

        best_pass = None
        best_ev = -float('inf')

        for teammate in teammates:
            value = pass_values.get(teammate.player_id, 1.0)
            lane = self.analyze_pass(
                passer, teammate, all_home, all_away, value
            )

            if lane.expected_value > best_ev:
                best_ev = lane.expected_value
                best_pass = lane

        return best_pass

    def all_passing_options(
        self,
        passer: PlayerState,
        teammates: List[PlayerState],
        all_home: List[PlayerState],
        all_away: List[PlayerState]
    ) -> List[PassingLane]:
        """Analyze all passing options."""
        return [
            self.analyze_pass(passer, t, all_home, all_away)
            for t in teammates
            if t.player_id != passer.player_id
        ]


class ZoneControlAnalyzer:
    """
    Analyze control by zone.

    Breaks rink into zones and computes control in each.
    """

    def __init__(self, control_model: PitchControlModel):
        self.control_model = control_model

        # Define zones
        self.zones = {
            'offensive': (self.control_model.rink_length/6, self.control_model.rink_length/2),
            'neutral': (-self.control_model.rink_length/6, self.control_model.rink_length/6),
            'defensive': (-self.control_model.rink_length/2, -self.control_model.rink_length/6),
            'high_danger': (self.control_model.rink_length/4, self.control_model.rink_length/2),
        }

    def zone_control(
        self,
        control_grid: ControlGrid,
        zone: str
    ) -> Tuple[float, float]:
        """Get average home/away control in zone."""
        if zone not in self.zones:
            return 0.5, 0.5

        x_min, x_max = self.zones[zone]

        # Find grid indices for zone
        x_mask = (control_grid.x_coords >= x_min) & (control_grid.x_coords <= x_max)

        if not np.any(x_mask):
            return 0.5, 0.5

        zone_grid = control_grid.grid[x_mask, :, :]
        home_control = np.mean(zone_grid[:, :, 0])
        away_control = np.mean(zone_grid[:, :, 1])

        return home_control, away_control

    def all_zone_control(
        self,
        control_grid: ControlGrid
    ) -> Dict[str, Tuple[float, float]]:
        """Get control for all zones."""
        return {
            zone: self.zone_control(control_grid, zone)
            for zone in self.zones
        }

    def slot_control(
        self,
        control_grid: ControlGrid
    ) -> Tuple[float, float]:
        """
        Get control specifically in the slot area.

        Slot defined as high-danger area in front of goal.
        """
        # Slot: within ~6 meters of goal, ~3 meters wide
        goal_x = self.control_model.rink_length / 2

        x_mask = (control_grid.x_coords >= goal_x - 6) & (control_grid.x_coords <= goal_x)
        y_mask = (control_grid.y_coords >= -3) & (control_grid.y_coords <= 3)

        slot_home = control_grid.grid[x_mask][:, y_mask, 0].mean()
        slot_away = control_grid.grid[x_mask][:, y_mask, 1].mean()

        return slot_home, slot_away
