"""
K-Nearest Neighbors Ice Ownership Model

Implements flexible ice ownership modeling using KNN approach,
translated from soccer research.

Key Paper:
- "A Neighbor-based Approach to Pitch Ownership Models in Soccer" (2025)
  arXiv:2501.05870

Key Innovations:
- More flexible than traditional Voronoi or Spearman models
- Introduces uncertainty via distance weighting
- Combines multiple Voronoi diagram variants
- Handles contested areas more naturally

Hockey Translation:
- Flexible ice ownership model
- Probabilistic control in contested areas
- Combine with tracking data for real-time visualization
- Useful for zone coverage analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class DistanceMetric(Enum):
    """Distance metrics for KNN calculations."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    TIME_BASED = "time_based"      # Time to reach considering velocity
    INFLUENCE = "influence"        # Weighted by player capabilities


class WeightingScheme(Enum):
    """Weighting schemes for neighbor contributions."""
    UNIFORM = "uniform"            # All k neighbors equal
    INVERSE_DISTANCE = "inverse"   # 1/distance weighting
    GAUSSIAN = "gaussian"          # exp(-distance^2/sigma^2)
    LINEAR = "linear"              # (max_dist - dist) / max_dist


@dataclass
class PlayerState:
    """Player state for ownership calculations."""
    player_id: str
    team_id: str
    x: float                        # feet
    y: float                        # feet
    velocity_x: float = 0.0         # ft/s
    velocity_y: float = 0.0         # ft/s
    max_speed: float = 30.0         # ft/s
    reach_radius: float = 8.0       # feet (stick + arms)
    influence_weight: float = 1.0   # Relative influence
    is_goalie: bool = False

    @property
    def speed(self) -> float:
        """Current speed."""
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2)


@dataclass
class OwnershipCell:
    """Ownership information for a grid cell."""
    x: float
    y: float
    team_probabilities: Dict[str, float]
    controlling_team: str
    confidence: float               # 0-1, certainty of assignment
    nearest_players: List[str]      # IDs of k nearest players
    uncertainty: float              # Measure of contestedness


@dataclass
class OwnershipGrid:
    """Full ice ownership grid."""
    grid: np.ndarray                # team_a probability at each cell
    team_a_id: str
    team_b_id: str
    uncertainty_grid: np.ndarray    # Uncertainty at each cell
    total_ownership: Dict[str, float]
    zone_ownership: Dict[str, Dict[str, float]]


class KNNOwnershipModel:
    """
    K-Nearest Neighbors based ice ownership model.

    Unlike Voronoi which assigns each point to nearest player,
    KNN considers multiple nearby players to calculate ownership
    probability, providing smoother and more realistic control surfaces.
    """

    def __init__(
        self,
        k: int = 5,
        grid_x: int = 100,
        grid_y: int = 43,
        rink_length: float = 200.0,
        rink_width: float = 85.0,
        distance_metric: DistanceMetric = DistanceMetric.TIME_BASED,
        weighting: WeightingScheme = WeightingScheme.GAUSSIAN,
        sigma: float = 15.0,  # For Gaussian weighting
    ):
        """
        Initialize KNN ownership model.

        Args:
            k: Number of neighbors to consider
            grid_x: Grid resolution along length
            grid_y: Grid resolution along width
            rink_length: Rink length in feet
            rink_width: Rink width in feet
            distance_metric: How to measure distance
            weighting: How to weight neighbor contributions
            sigma: Width parameter for Gaussian weighting
        """
        self.k = k
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.rink_length = rink_length
        self.rink_width = rink_width
        self.distance_metric = distance_metric
        self.weighting = weighting
        self.sigma = sigma

        # Create coordinate grids
        self.x_coords = np.linspace(0, rink_length, grid_x)
        self.y_coords = np.linspace(0, rink_width, grid_y)

        # Zone boundaries
        self.offensive_blue = 175.0
        self.defensive_blue = 25.0

    def calculate_distance(
        self,
        player: PlayerState,
        x: float,
        y: float,
    ) -> float:
        """
        Calculate distance/cost from player to point.

        Different metrics capture different aspects of control.
        """
        dx = x - player.x
        dy = y - player.y

        if self.distance_metric == DistanceMetric.EUCLIDEAN:
            return np.sqrt(dx**2 + dy**2)

        elif self.distance_metric == DistanceMetric.MANHATTAN:
            return abs(dx) + abs(dy)

        elif self.distance_metric == DistanceMetric.TIME_BASED:
            # Time to reach considering current velocity
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 0.1:
                return 0.0

            # Component of velocity toward target
            ux = dx / distance
            uy = dy / distance
            v_toward = player.velocity_x * ux + player.velocity_y * uy

            # Effective speed
            if v_toward > 0:
                effective_speed = (player.speed + player.max_speed) / 2
            else:
                # Moving away - need to turn around
                effective_speed = player.max_speed / 2

            return distance / max(effective_speed, 1.0)

        elif self.distance_metric == DistanceMetric.INFLUENCE:
            # Distance weighted by player influence
            base_dist = np.sqrt(dx**2 + dy**2)
            return base_dist / player.influence_weight

        return np.sqrt(dx**2 + dy**2)

    def calculate_weight(self, distance: float, max_distance: float) -> float:
        """
        Calculate weight for a neighbor based on distance.

        Args:
            distance: Distance to neighbor
            max_distance: Maximum distance in the k neighbors

        Returns:
            Weight for this neighbor
        """
        if self.weighting == WeightingScheme.UNIFORM:
            return 1.0

        elif self.weighting == WeightingScheme.INVERSE_DISTANCE:
            return 1.0 / max(distance, 0.1)

        elif self.weighting == WeightingScheme.GAUSSIAN:
            return np.exp(-distance**2 / (2 * self.sigma**2))

        elif self.weighting == WeightingScheme.LINEAR:
            return max(0, (max_distance - distance) / max(max_distance, 0.1))

        return 1.0

    def get_k_nearest(
        self,
        players: List[PlayerState],
        x: float,
        y: float,
    ) -> List[Tuple[PlayerState, float]]:
        """
        Get k nearest players to a point.

        Returns list of (player, distance) tuples.
        """
        distances = []

        for player in players:
            dist = self.calculate_distance(player, x, y)
            distances.append((player, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        return distances[:self.k]

    def compute_cell_ownership(
        self,
        players: List[PlayerState],
        x: float,
        y: float,
    ) -> OwnershipCell:
        """
        Compute ownership for a single cell.

        Uses k nearest neighbors with distance weighting
        to calculate team ownership probabilities.
        """
        if not players:
            return OwnershipCell(
                x=x, y=y,
                team_probabilities={},
                controlling_team="none",
                confidence=0.0,
                nearest_players=[],
                uncertainty=1.0,
            )

        # Get k nearest neighbors
        neighbors = self.get_k_nearest(players, x, y)

        # Calculate weights
        max_dist = neighbors[-1][1] if neighbors else 1.0
        team_weights: Dict[str, float] = defaultdict(float)
        total_weight = 0.0

        player_ids = []
        for player, dist in neighbors:
            weight = self.calculate_weight(dist, max_dist)

            # Adjust weight for goalies outside crease
            if player.is_goalie:
                weight *= self._goalie_weight_adjustment(player, x, y)

            team_weights[player.team_id] += weight
            total_weight += weight
            player_ids.append(player.player_id)

        # Normalize to probabilities
        if total_weight > 0:
            team_probs = {
                team: w / total_weight
                for team, w in team_weights.items()
            }
        else:
            team_probs = {}

        # Determine controlling team
        if team_probs:
            controlling = max(team_probs.keys(), key=lambda t: team_probs[t])
            confidence = team_probs[controlling]
        else:
            controlling = "none"
            confidence = 0.0

        # Calculate uncertainty (entropy-based)
        if len(team_probs) > 1:
            probs = list(team_probs.values())
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            max_entropy = np.log(len(probs))
            uncertainty = entropy / max_entropy if max_entropy > 0 else 0
        else:
            uncertainty = 0.0

        return OwnershipCell(
            x=x,
            y=y,
            team_probabilities=team_probs,
            controlling_team=controlling,
            confidence=confidence,
            nearest_players=player_ids,
            uncertainty=uncertainty,
        )

    def _goalie_weight_adjustment(
        self,
        goalie: PlayerState,
        x: float,
        y: float,
    ) -> float:
        """
        Adjust goalie weight based on distance from crease.

        Goalies have high influence in crease, low outside.
        """
        # Determine crease location
        if goalie.x < self.rink_length / 2:
            crease_x = 11.0
        else:
            crease_x = self.rink_length - 11.0
        crease_y = self.rink_width / 2

        dist_from_crease = np.sqrt(
            (x - crease_x)**2 + (y - crease_y)**2
        )

        if dist_from_crease < 8:  # In crease
            return 2.0  # Increased influence
        elif dist_from_crease < 20:  # Near crease
            return 1.0
        else:  # Far from crease
            return 0.2  # Reduced influence

    def compute_ownership_grid(
        self,
        players: List[PlayerState],
    ) -> OwnershipGrid:
        """
        Compute full ice ownership grid.

        Returns grid of team ownership probabilities.
        """
        if not players:
            return self._empty_grid()

        # Get teams
        teams = list(set(p.team_id for p in players))
        team_a = teams[0] if teams else "home"
        team_b = teams[1] if len(teams) > 1 else "away"

        # Initialize grids
        ownership_grid = np.zeros((self.grid_y, self.grid_x))
        uncertainty_grid = np.zeros((self.grid_y, self.grid_x))

        # Compute ownership for each cell
        for i, y in enumerate(self.y_coords):
            for j, x in enumerate(self.x_coords):
                cell = self.compute_cell_ownership(players, x, y)
                ownership_grid[i, j] = cell.team_probabilities.get(team_a, 0.5)
                uncertainty_grid[i, j] = cell.uncertainty

        # Calculate totals
        total_ownership = {
            team_a: float(np.mean(ownership_grid)),
            team_b: float(1 - np.mean(ownership_grid)),
        }

        # Zone ownership
        zone_ownership = self._calculate_zone_ownership(
            ownership_grid, team_a, team_b
        )

        return OwnershipGrid(
            grid=ownership_grid,
            team_a_id=team_a,
            team_b_id=team_b,
            uncertainty_grid=uncertainty_grid,
            total_ownership=total_ownership,
            zone_ownership=zone_ownership,
        )

    def _calculate_zone_ownership(
        self,
        grid: np.ndarray,
        team_a: str,
        team_b: str,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate zone-by-zone ownership."""
        zones = {}

        # Create zone masks
        x_grid = np.tile(self.x_coords, (self.grid_y, 1))

        # Offensive zone (for team_a attacking right)
        oz_mask = x_grid > self.offensive_blue
        if oz_mask.any():
            oz_val = float(np.mean(grid[oz_mask]))
            zones['offensive'] = {team_a: oz_val, team_b: 1 - oz_val}
        else:
            zones['offensive'] = {team_a: 0.5, team_b: 0.5}

        # Neutral zone
        nz_mask = (x_grid >= self.defensive_blue) & (x_grid <= self.offensive_blue)
        if nz_mask.any():
            nz_val = float(np.mean(grid[nz_mask]))
            zones['neutral'] = {team_a: nz_val, team_b: 1 - nz_val}
        else:
            zones['neutral'] = {team_a: 0.5, team_b: 0.5}

        # Defensive zone
        dz_mask = x_grid < self.defensive_blue
        if dz_mask.any():
            dz_val = float(np.mean(grid[dz_mask]))
            zones['defensive'] = {team_a: dz_val, team_b: 1 - dz_val}
        else:
            zones['defensive'] = {team_a: 0.5, team_b: 0.5}

        return zones

    def _empty_grid(self) -> OwnershipGrid:
        """Return empty ownership grid."""
        return OwnershipGrid(
            grid=np.full((self.grid_y, self.grid_x), 0.5),
            team_a_id="home",
            team_b_id="away",
            uncertainty_grid=np.ones((self.grid_y, self.grid_x)),
            total_ownership={"home": 0.5, "away": 0.5},
            zone_ownership={
                'offensive': {"home": 0.5, "away": 0.5},
                'neutral': {"home": 0.5, "away": 0.5},
                'defensive': {"home": 0.5, "away": 0.5},
            },
        )


class AdaptiveKNNOwnership:
    """
    Adaptive KNN ownership that adjusts k based on player density.

    In crowded areas, uses more neighbors for smoother probability.
    In sparse areas, uses fewer neighbors.
    """

    def __init__(
        self,
        min_k: int = 3,
        max_k: int = 8,
        density_radius: float = 30.0,
        base_model: Optional[KNNOwnershipModel] = None,
    ):
        """
        Initialize adaptive model.

        Args:
            min_k: Minimum neighbors in sparse areas
            max_k: Maximum neighbors in dense areas
            density_radius: Radius for counting nearby players
            base_model: Base KNN model to use
        """
        self.min_k = min_k
        self.max_k = max_k
        self.density_radius = density_radius
        self.base_model = base_model or KNNOwnershipModel()

    def adaptive_k(
        self,
        players: List[PlayerState],
        x: float,
        y: float,
    ) -> int:
        """
        Calculate adaptive k based on local player density.
        """
        # Count players within density radius
        nearby = 0
        for player in players:
            dist = np.sqrt((player.x - x)**2 + (player.y - y)**2)
            if dist < self.density_radius:
                nearby += 1

        # Scale k based on density
        # More players nearby = higher k
        total_players = len(players)
        if total_players == 0:
            return self.min_k

        density_ratio = nearby / min(total_players, 10)
        k = int(self.min_k + (self.max_k - self.min_k) * density_ratio)

        return min(k, len(players))

    def compute_ownership(
        self,
        players: List[PlayerState],
    ) -> OwnershipGrid:
        """
        Compute ownership with adaptive k.
        """
        if not players:
            return self.base_model._empty_grid()

        teams = list(set(p.team_id for p in players))
        team_a = teams[0] if teams else "home"
        team_b = teams[1] if len(teams) > 1 else "away"

        ownership_grid = np.zeros(
            (self.base_model.grid_y, self.base_model.grid_x)
        )
        uncertainty_grid = np.zeros(
            (self.base_model.grid_y, self.base_model.grid_x)
        )

        for i, y in enumerate(self.base_model.y_coords):
            for j, x in enumerate(self.base_model.x_coords):
                # Get adaptive k for this location
                k = self.adaptive_k(players, x, y)

                # Temporarily set k
                old_k = self.base_model.k
                self.base_model.k = k

                # Compute ownership
                cell = self.base_model.compute_cell_ownership(players, x, y)
                ownership_grid[i, j] = cell.team_probabilities.get(team_a, 0.5)
                uncertainty_grid[i, j] = cell.uncertainty

                # Restore k
                self.base_model.k = old_k

        return OwnershipGrid(
            grid=ownership_grid,
            team_a_id=team_a,
            team_b_id=team_b,
            uncertainty_grid=uncertainty_grid,
            total_ownership={
                team_a: float(np.mean(ownership_grid)),
                team_b: float(1 - np.mean(ownership_grid)),
            },
            zone_ownership=self.base_model._calculate_zone_ownership(
                ownership_grid, team_a, team_b
            ),
        )


class OwnershipComparator:
    """
    Compares different ownership models.

    Useful for research and visualization.
    """

    def __init__(self):
        """Initialize comparator with multiple models."""
        self.models = {
            'knn_euclidean': KNNOwnershipModel(
                distance_metric=DistanceMetric.EUCLIDEAN,
                weighting=WeightingScheme.GAUSSIAN,
            ),
            'knn_time': KNNOwnershipModel(
                distance_metric=DistanceMetric.TIME_BASED,
                weighting=WeightingScheme.GAUSSIAN,
            ),
            'knn_inverse': KNNOwnershipModel(
                distance_metric=DistanceMetric.EUCLIDEAN,
                weighting=WeightingScheme.INVERSE_DISTANCE,
            ),
            'knn_uniform': KNNOwnershipModel(
                distance_metric=DistanceMetric.EUCLIDEAN,
                weighting=WeightingScheme.UNIFORM,
            ),
            'adaptive': AdaptiveKNNOwnership(),
        }

    def compare(
        self,
        players: List[PlayerState],
    ) -> Dict[str, OwnershipGrid]:
        """
        Compute ownership with all models.

        Returns dictionary of model_name -> ownership_grid.
        """
        results = {}

        for name, model in self.models.items():
            if isinstance(model, AdaptiveKNNOwnership):
                results[name] = model.compute_ownership(players)
            else:
                results[name] = model.compute_ownership_grid(players)

        return results

    def analyze_differences(
        self,
        results: Dict[str, OwnershipGrid],
    ) -> Dict[str, Any]:
        """
        Analyze differences between model results.
        """
        model_names = list(results.keys())
        if len(model_names) < 2:
            return {}

        # Calculate pairwise differences
        diffs = {}
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                grid1 = results[name1].grid
                grid2 = results[name2].grid

                mae = float(np.mean(np.abs(grid1 - grid2)))
                max_diff = float(np.max(np.abs(grid1 - grid2)))

                diffs[f"{name1}_vs_{name2}"] = {
                    'mean_absolute_difference': mae,
                    'max_difference': max_diff,
                }

        # Overall summary
        grids = [r.grid for r in results.values()]
        variance = np.var(np.stack(grids), axis=0)

        return {
            'pairwise_differences': diffs,
            'average_variance': float(np.mean(variance)),
            'high_disagreement_cells': int(np.sum(variance > 0.1)),
        }


class CoverageAnalyzer:
    """
    Analyzes defensive coverage using KNN ownership.
    """

    def __init__(self, model: KNNOwnershipModel):
        """Initialize with ownership model."""
        self.model = model

    def find_uncovered_areas(
        self,
        defenders: List[PlayerState],
        min_coverage: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Find areas with poor defensive coverage.

        Args:
            defenders: List of defending players
            min_coverage: Minimum coverage threshold (0-1)

        Returns:
            List of uncovered area details
        """
        uncovered = []

        # Create pseudo attackers at each grid point
        for i, y in enumerate(self.model.y_coords):
            for j, x in enumerate(self.model.x_coords):
                cell = self.model.compute_cell_ownership(defenders, x, y)

                # Check if defenders don't control this cell
                team = defenders[0].team_id if defenders else "defense"
                coverage = cell.team_probabilities.get(team, 0)

                if coverage < min_coverage:
                    # Calculate danger level based on location
                    danger = self._location_danger(x, y)

                    if danger > 0.3:  # Only track dangerous areas
                        uncovered.append({
                            'x': x,
                            'y': y,
                            'coverage': coverage,
                            'danger_level': danger,
                            'risk_score': danger * (1 - coverage),
                        })

        # Sort by risk score
        uncovered.sort(key=lambda a: a['risk_score'], reverse=True)

        return uncovered[:20]  # Top 20 risky areas

    def _location_danger(self, x: float, y: float) -> float:
        """
        Calculate inherent danger of a location.

        Based on proximity to goal.
        """
        # Assume attacking right goal
        goal_x = self.model.rink_length - 11
        goal_y = self.model.rink_width / 2

        dist = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

        # Exponential decay from goal
        return np.exp(-dist / 40)

    def zone_coverage_report(
        self,
        defenders: List[PlayerState],
    ) -> Dict[str, Any]:
        """
        Generate zone-by-zone coverage report.
        """
        grid = self.model.compute_ownership_grid(defenders)
        team = defenders[0].team_id if defenders else "defense"

        x_grid = np.tile(self.model.x_coords, (self.model.grid_y, 1))

        report = {}

        # Slot coverage (most critical)
        slot_mask = self._slot_mask(x_grid)
        if slot_mask.any():
            report['slot_coverage'] = float(np.mean(grid.grid[slot_mask]))
        else:
            report['slot_coverage'] = 0.5

        # High slot
        high_slot_mask = self._high_slot_mask(x_grid)
        if high_slot_mask.any():
            report['high_slot_coverage'] = float(np.mean(grid.grid[high_slot_mask]))
        else:
            report['high_slot_coverage'] = 0.5

        # Circles
        circles_mask = self._circles_mask(x_grid)
        if circles_mask.any():
            report['circles_coverage'] = float(np.mean(grid.grid[circles_mask]))
        else:
            report['circles_coverage'] = 0.5

        # Overall defensive zone
        report['defensive_zone'] = grid.zone_ownership.get('defensive', {}).get(team, 0.5)

        # Identify weakest area
        coverages = [
            ('slot', report['slot_coverage']),
            ('high_slot', report['high_slot_coverage']),
            ('circles', report['circles_coverage']),
        ]
        report['weakest_area'] = min(coverages, key=lambda x: x[1])[0]

        return report

    def _slot_mask(self, x_grid: np.ndarray) -> np.ndarray:
        """Create mask for slot area."""
        y_grid = np.tile(
            self.model.y_coords.reshape(-1, 1),
            (1, self.model.grid_x)
        )

        goal_x = self.model.rink_length - 11
        goal_y = self.model.rink_width / 2

        dist_x = abs(x_grid - goal_x)
        dist_y = abs(y_grid - goal_y)

        return (dist_x < 20) & (dist_y < 12)

    def _high_slot_mask(self, x_grid: np.ndarray) -> np.ndarray:
        """Create mask for high slot area."""
        y_grid = np.tile(
            self.model.y_coords.reshape(-1, 1),
            (1, self.model.grid_x)
        )

        goal_x = self.model.rink_length - 11
        goal_y = self.model.rink_width / 2

        dist_x = abs(x_grid - goal_x)
        dist_y = abs(y_grid - goal_y)

        return (dist_x >= 20) & (dist_x < 40) & (dist_y < 15)

    def _circles_mask(self, x_grid: np.ndarray) -> np.ndarray:
        """Create mask for face-off circles area."""
        y_grid = np.tile(
            self.model.y_coords.reshape(-1, 1),
            (1, self.model.grid_x)
        )

        goal_x = self.model.rink_length - 11

        # Left and right circles (roughly)
        left_circle = np.sqrt(
            (x_grid - (goal_x - 20))**2 + (y_grid - 22)**2
        ) < 15
        right_circle = np.sqrt(
            (x_grid - (goal_x - 20))**2 + (y_grid - 63)**2
        ) < 15

        return left_circle | right_circle
