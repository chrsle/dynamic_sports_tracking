"""
Ice Control Model for Hockey Analytics

This module implements spatial control models translated from soccer
pitch control research (Fernández & Bornn).

Key concepts:
- Continuous spatial control surfaces
- Real-time ice dominance calculation
- Defensive scheme classification
- Territorial control metrics

Hockey adaptation:
- Account for rink shape and behind-net play
- Blue line as hard boundary
- Faster transitions than soccer
- Different positioning requirements

References:
- "Space and Control in Soccer" (Frontiers 2021)
- Fernández & Bornn pitch control research
- "Machine learning-based analysis of defensive strategies in basketball"
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy.ndimage import gaussian_filter
import json


@dataclass
class PlayerPosition:
    """Position of a player at a moment in time."""
    player_id: str
    team_id: str
    x: float  # Position on ice (0-200)
    y: float  # Position on ice (0-85)
    velocity_x: float = 0.0  # ft/s
    velocity_y: float = 0.0  # ft/s
    speed: float = 0.0  # mph
    has_puck: bool = False
    is_goalie: bool = False


@dataclass
class PuckPosition:
    """Position of the puck."""
    x: float
    y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    carrier_id: Optional[str] = None


class IceControlModel:
    """
    Models spatial control across the ice surface.

    For each point on the ice, calculates which team has control
    based on player positions, velocities, and distances.

    This is the hockey equivalent of soccer pitch control models.
    """

    def __init__(
        self,
        grid_x: int = 100,
        grid_y: int = 43,
        rink_length: float = 200.0,
        rink_width: float = 85.0,
        max_reach: float = 8.0,  # feet - max reach of a player
        reaction_time: float = 0.5,  # seconds
    ):
        """
        Initialize the ice control model.

        Args:
            grid_x: Number of grid points along length
            grid_y: Number of grid points along width
            rink_length: Rink length in feet
            rink_width: Rink width in feet
            max_reach: Maximum reach of a player (stick + arms)
            reaction_time: Average reaction time
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.rink_length = rink_length
        self.rink_width = rink_width
        self.max_reach = max_reach
        self.reaction_time = reaction_time

        # Cell dimensions
        self.cell_width = rink_length / grid_x
        self.cell_height = rink_width / grid_y

        # Create coordinate grids
        x_coords = np.linspace(0, rink_length, grid_x)
        y_coords = np.linspace(0, rink_width, grid_y)
        self.X, self.Y = np.meshgrid(x_coords, y_coords)

        # Control surface (will be filled)
        self.control_surface = np.zeros((grid_y, grid_x))

        # Player parameters
        self.max_speed = 30.0  # mph
        self.acceleration = 15.0  # ft/s^2

    def calculate_time_to_point(
        self,
        player: PlayerPosition,
        target_x: float,
        target_y: float
    ) -> float:
        """
        Calculate time for a player to reach a point on the ice.

        Uses physics model accounting for current velocity and acceleration.

        Args:
            player: Player position with velocity
            target_x, target_y: Target point coordinates

        Returns:
            Time in seconds to reach the point
        """
        # Distance to target
        dx = target_x - player.x
        dy = target_y - player.y
        distance = np.sqrt(dx * dx + dy * dy)

        if distance < self.max_reach:
            return 0.0  # Already in range

        # Account for current velocity direction
        current_speed = player.speed * 1.467  # mph to ft/s

        # Direction to target
        if distance > 0:
            dir_x = dx / distance
            dir_y = dy / distance
        else:
            return 0.0

        # Component of velocity toward target
        velocity_toward = player.velocity_x * dir_x + player.velocity_y * dir_y

        # Time calculation with acceleration
        # d = v0*t + 0.5*a*t^2
        # Simplified: assume constant max speed after brief acceleration

        if current_speed < 1.0:
            # Starting from rest
            time_to_accelerate = (self.max_speed * 1.467) / self.acceleration
            dist_during_accel = 0.5 * self.acceleration * time_to_accelerate ** 2

            if distance <= dist_during_accel:
                # Reach before max speed
                return np.sqrt(2 * distance / self.acceleration) + self.reaction_time
            else:
                remaining = distance - dist_during_accel
                time_at_max = remaining / (self.max_speed * 1.467)
                return time_to_accelerate + time_at_max + self.reaction_time
        else:
            # Already moving
            if velocity_toward > 0:
                # Moving toward target
                return distance / max(velocity_toward, 1.0) + self.reaction_time
            else:
                # Need to turn around
                return distance / (self.max_speed * 0.8 * 1.467) + self.reaction_time * 2

    def calculate_control_probability(
        self,
        time_home: float,
        time_away: float,
        sigma: float = 0.3
    ) -> float:
        """
        Calculate probability of home team controlling a point.

        Based on relative time to reach using logistic function.

        Args:
            time_home: Min time for home team player to reach
            time_away: Min time for away team player to reach
            sigma: Uncertainty parameter

        Returns:
            Probability (0-1) of home team control
        """
        # Logistic function based on time difference
        time_diff = time_away - time_home
        prob = 1.0 / (1.0 + np.exp(-time_diff / sigma))

        return prob

    def compute_control_surface(
        self,
        home_players: List[PlayerPosition],
        away_players: List[PlayerPosition],
        puck: Optional[PuckPosition] = None
    ) -> np.ndarray:
        """
        Compute the control surface for the entire ice.

        Args:
            home_players: List of home team player positions
            away_players: List of away team player positions
            puck: Optional puck position

        Returns:
            2D array where values > 0.5 indicate home control
        """
        self.control_surface = np.zeros((self.grid_y, self.grid_x))

        # For each point on the grid
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                x = j * self.cell_width
                y = i * self.cell_height

                # Find minimum time to reach for each team
                min_time_home = float('inf')
                min_time_away = float('inf')

                for player in home_players:
                    if player.is_goalie:
                        continue
                    time = self.calculate_time_to_point(player, x, y)
                    min_time_home = min(min_time_home, time)

                for player in away_players:
                    if player.is_goalie:
                        continue
                    time = self.calculate_time_to_point(player, x, y)
                    min_time_away = min(min_time_away, time)

                # Calculate control probability
                if min_time_home < float('inf') and min_time_away < float('inf'):
                    prob = self.calculate_control_probability(min_time_home, min_time_away)
                elif min_time_home < float('inf'):
                    prob = 1.0
                elif min_time_away < float('inf'):
                    prob = 0.0
                else:
                    prob = 0.5

                self.control_surface[i, j] = prob

        # Apply Gaussian smoothing for continuity
        self.control_surface = gaussian_filter(self.control_surface, sigma=1.0)

        return self.control_surface

    def get_zone_control(
        self,
        control_surface: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate control percentages by zone.

        Returns:
            Dictionary with control percentages for each zone
        """
        if control_surface is None:
            control_surface = self.control_surface

        # Define zone boundaries (in grid coordinates)
        def_zone_end = int(25 / self.rink_length * self.grid_x)
        off_zone_start = int(175 / self.rink_length * self.grid_x)

        zones = {
            'defensive': (0, def_zone_end),
            'neutral': (def_zone_end, off_zone_start),
            'offensive': (off_zone_start, self.grid_x),
        }

        results = {}

        for zone_name, (start_x, end_x) in zones.items():
            zone_control = control_surface[:, start_x:end_x]

            home_control = np.mean(zone_control)
            away_control = 1 - home_control

            results[zone_name] = {
                'home_control': home_control,
                'away_control': away_control,
            }

        # Overall
        results['total'] = {
            'home_control': np.mean(control_surface),
            'away_control': 1 - np.mean(control_surface),
        }

        return results

    def get_dangerous_space(
        self,
        control_surface: Optional[np.ndarray] = None,
        team: str = 'home'
    ) -> float:
        """
        Calculate amount of dangerous space controlled by a team.

        Dangerous space = high-xT areas that team controls.

        Args:
            control_surface: Control surface array
            team: 'home' or 'away'

        Returns:
            Dangerous space score
        """
        if control_surface is None:
            control_surface = self.control_surface

        # Define danger zones (slot area)
        slot_x_start = int(165 / self.rink_length * self.grid_x)
        slot_x_end = int(195 / self.rink_length * self.grid_x)
        slot_y_start = int(27 / self.rink_width * self.grid_y)
        slot_y_end = int(58 / self.rink_width * self.grid_y)

        slot_control = control_surface[slot_y_start:slot_y_end, slot_x_start:slot_x_end]

        if team == 'home':
            return np.mean(slot_control)
        else:
            return 1 - np.mean(slot_control)


class DefensiveSchemeClassifier:
    """
    Classifies defensive schemes based on player positioning.

    Inspired by basketball defensive scheme classification research
    that achieved 91.4% accuracy using LSTM-CNN hybrid models.

    Hockey schemes:
    - Box (penalty kill)
    - Diamond (penalty kill)
    - Man-to-man
    - Zone defense
    - Trap (neutral zone)
    - Aggressive forecheck
    """

    def __init__(self):
        """Initialize the classifier."""
        self.scheme_patterns = {
            'box': self._check_box_formation,
            'diamond': self._check_diamond_formation,
            'zone': self._check_zone_defense,
            'man_to_man': self._check_man_to_man,
            'trap': self._check_neutral_zone_trap,
            'aggressive_forecheck': self._check_aggressive_forecheck,
        }

    def classify_scheme(
        self,
        defensive_players: List[PlayerPosition],
        offensive_players: List[PlayerPosition],
        puck: PuckPosition
    ) -> Dict[str, Any]:
        """
        Classify the current defensive scheme.

        Args:
            defensive_players: Defending team player positions
            offensive_players: Attacking team player positions
            puck: Puck position

        Returns:
            Dictionary with scheme classification and confidence
        """
        scores = {}

        for scheme_name, check_func in self.scheme_patterns.items():
            scores[scheme_name] = check_func(
                defensive_players, offensive_players, puck
            )

        # Find best match
        best_scheme = max(scores, key=scores.get)
        confidence = scores[best_scheme]

        return {
            'scheme': best_scheme,
            'confidence': confidence,
            'all_scores': scores,
        }

    def _check_box_formation(
        self,
        defenders: List[PlayerPosition],
        attackers: List[PlayerPosition],
        puck: PuckPosition
    ) -> float:
        """Check for box formation (PK)."""
        if len(defenders) != 4:
            return 0.0

        # Box formation: 4 players in rectangle shape
        positions = [(d.x, d.y) for d in defenders if not d.is_goalie]

        if len(positions) < 4:
            return 0.0

        # Calculate bounding box
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        # Box should be roughly square-ish
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0

        # Players should be near corners
        corner_score = 0.0
        # Simplified: just check spread
        if 15 < width < 40 and 15 < height < 35:
            corner_score = 0.6 + 0.4 * aspect_ratio

        return corner_score

    def _check_diamond_formation(
        self,
        defenders: List[PlayerPosition],
        attackers: List[PlayerPosition],
        puck: PuckPosition
    ) -> float:
        """Check for diamond formation (PK)."""
        if len(defenders) != 4:
            return 0.0

        positions = [(d.x, d.y) for d in defenders if not d.is_goalie]

        if len(positions) < 4:
            return 0.0

        # Diamond: one high, one low, two wide
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Check for one player significantly higher/lower than others
        x_spread = max(xs) - min(xs)
        y_center = sum(ys) / len(ys)

        # Check for central alignment
        central_players = sum(1 for _, y in positions if abs(y - y_center) < 10)

        if central_players >= 2 and x_spread > 20:
            return 0.7
        return 0.2

    def _check_zone_defense(
        self,
        defenders: List[PlayerPosition],
        attackers: List[PlayerPosition],
        puck: PuckPosition
    ) -> float:
        """Check for zone defense."""
        # Zone defense: players cover areas, not specific opponents
        # Check if defenders are positioned in zones regardless of attackers

        positions = [(d.x, d.y) for d in defenders if not d.is_goalie]

        if len(positions) < 4:
            return 0.0

        # Check spacing - zone defense typically has even spacing
        xs = sorted([p[0] for p in positions])
        ys = sorted([p[1] for p in positions])

        # Check x-spacing evenness
        x_gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        x_variance = np.var(x_gaps) if x_gaps else float('inf')

        y_gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        y_variance = np.var(y_gaps) if y_gaps else float('inf')

        # Low variance = even spacing = zone defense
        spacing_score = 1.0 / (1.0 + x_variance * 0.01 + y_variance * 0.01)

        return min(spacing_score, 0.9)

    def _check_man_to_man(
        self,
        defenders: List[PlayerPosition],
        attackers: List[PlayerPosition],
        puck: PuckPosition
    ) -> float:
        """Check for man-to-man defense."""
        # Man-to-man: each defender close to an attacker

        if len(defenders) < len(attackers):
            return 0.0

        defender_positions = [(d.x, d.y) for d in defenders if not d.is_goalie]
        attacker_positions = [(a.x, a.y) for a in attackers if not a.is_goalie]

        if not defender_positions or not attacker_positions:
            return 0.0

        # For each attacker, find closest defender
        close_coverage = 0

        for ax, ay in attacker_positions:
            min_dist = float('inf')
            for dx, dy in defender_positions:
                dist = np.sqrt((ax - dx) ** 2 + (ay - dy) ** 2)
                min_dist = min(min_dist, dist)

            if min_dist < 10:  # Within 10 feet
                close_coverage += 1

        coverage_rate = close_coverage / len(attacker_positions)

        return coverage_rate * 0.9

    def _check_neutral_zone_trap(
        self,
        defenders: List[PlayerPosition],
        attackers: List[PlayerPosition],
        puck: PuckPosition
    ) -> float:
        """Check for neutral zone trap."""
        # Trap: defenders packed in neutral zone

        positions = [(d.x, d.y) for d in defenders if not d.is_goalie]

        if len(positions) < 4:
            return 0.0

        # Count players in neutral zone (roughly 75-125 x coordinates)
        neutral_zone_players = sum(
            1 for x, y in positions if 60 < x < 140
        )

        # Trap has 3-4 players clogging neutral zone
        if neutral_zone_players >= 3:
            return 0.8
        elif neutral_zone_players == 2:
            return 0.4

        return 0.1

    def _check_aggressive_forecheck(
        self,
        defenders: List[PlayerPosition],
        attackers: List[PlayerPosition],
        puck: PuckPosition
    ) -> float:
        """Check for aggressive forecheck."""
        # Forecheck: defenders pushing high into offensive zone

        positions = [(d.x, d.y) for d in defenders if not d.is_goalie]

        if len(positions) < 2:
            return 0.0

        # Check how many defenders are in the offensive end
        high_pressure = sum(1 for x, y in positions if x > 150)

        if high_pressure >= 2:
            return 0.85
        elif high_pressure == 1:
            return 0.5

        return 0.1


def create_sample_positions() -> Tuple[List[PlayerPosition], List[PlayerPosition], PuckPosition]:
    """Create sample player positions for testing."""
    np.random.seed(42)

    home_players = []
    away_players = []

    # Home team (defensive positions)
    home_positions = [
        (40, 42, False),   # Center
        (30, 25, False),   # LW
        (30, 60, False),   # RW
        (20, 30, False),   # LD
        (20, 55, False),   # RD
        (5, 42.5, True),   # Goalie
    ]

    for i, (x, y, is_goalie) in enumerate(home_positions):
        home_players.append(PlayerPosition(
            player_id=f"home_{i}",
            team_id="home",
            x=x + np.random.uniform(-3, 3),
            y=y + np.random.uniform(-3, 3),
            velocity_x=np.random.uniform(-10, 10),
            velocity_y=np.random.uniform(-5, 5),
            speed=np.random.uniform(5, 15),
            is_goalie=is_goalie,
        ))

    # Away team (offensive positions)
    away_positions = [
        (60, 42, False),   # Center
        (50, 20, False),   # LW
        (50, 65, False),   # RW
        (35, 15, False),   # LD
        (35, 70, False),   # RD
        (195, 42.5, True), # Goalie
    ]

    for i, (x, y, is_goalie) in enumerate(away_positions):
        away_players.append(PlayerPosition(
            player_id=f"away_{i}",
            team_id="away",
            x=x + np.random.uniform(-3, 3),
            y=y + np.random.uniform(-3, 3),
            velocity_x=np.random.uniform(-10, 10),
            velocity_y=np.random.uniform(-5, 5),
            speed=np.random.uniform(5, 15),
            is_goalie=is_goalie,
        ))

    puck = PuckPosition(x=55, y=40, carrier_id="away_0")

    return home_players, away_players, puck


if __name__ == "__main__":
    # Demo the ice control model
    print("Creating Ice Control Model...")

    model = IceControlModel(grid_x=50, grid_y=25)

    # Create sample positions
    print("Generating sample positions...")
    home_players, away_players, puck = create_sample_positions()

    # Compute control surface
    print("Computing control surface...")
    control_surface = model.compute_control_surface(home_players, away_players, puck)

    # Zone control
    print("\nZone Control:")
    zone_control = model.get_zone_control(control_surface)
    for zone, control in zone_control.items():
        print(f"  {zone}: Home={control['home_control']:.2%}, Away={control['away_control']:.2%}")

    # Dangerous space
    print("\nDangerous Space Control:")
    home_danger = model.get_dangerous_space(control_surface, 'home')
    away_danger = model.get_dangerous_space(control_surface, 'away')
    print(f"  Home: {home_danger:.2%}")
    print(f"  Away: {away_danger:.2%}")

    # Defensive scheme classification
    print("\nDefensive Scheme Classification:")
    classifier = DefensiveSchemeClassifier()

    # Home is defending
    scheme = classifier.classify_scheme(home_players, away_players, puck)
    print(f"  Detected scheme: {scheme['scheme']}")
    print(f"  Confidence: {scheme['confidence']:.2%}")
    print(f"  All scores: {scheme['all_scores']}")

    # Control surface statistics
    print("\nControl Surface Statistics:")
    print(f"  Shape: {control_surface.shape}")
    print(f"  Mean (home bias): {np.mean(control_surface):.3f}")
    print(f"  Max home control: {np.max(control_surface):.3f}")
    print(f"  Max away control: {1 - np.min(control_surface):.3f}")
