import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from scipy import ndimage
from enum import IntEnum
import warnings
warnings.filterwarnings('ignore')

class RinkZone(IntEnum):
    """Hockey rink zones."""
    OFFENSIVE = 0
    NEUTRAL = 1
    DEFENSIVE = 2
    BEHIND_GOAL_OFF = 3
    BEHIND_GOAL_DEF = 4


class PositionQuality(IntEnum):
    """Position quality ratings."""
    OPTIMAL = 2      # High-danger scoring area, good defensive position
    GOOD = 1         # Reasonable position
    NEUTRAL = 0      # Neither good nor bad
    SUBOPTIMAL = -1  # Out of position, poor coverage
    CRITICAL = -2    # Dangerous gap, odd-man situation


@dataclass
class PositionAnalyticsConfig:
    """Configuration for position analytics."""
    
    # Heatmap settings
    heatmap_resolution: Tuple[int, int] = (100, 43)  # Grid cells (length x width)
    heatmap_decay: float = 0.995  # Per-frame decay for temporal weighting
    heatmap_sigma: float = 2.0  # Gaussian smoothing sigma
    
    # Hotspot detection
    hotspot_threshold: float = 0.7  # Percentile threshold for hotspots
    min_hotspot_size: int = 5  # Minimum cells for valid hotspot
    
    # Zone analysis
    blue_line_x: float = 25.0  # Distance from center to blue line (feet)
    goal_line_x: float = 89.0  # Distance from center to goal line (feet)
    
    # Sliding window for momentum
    momentum_window: int = 300  # Frames (~10 seconds at 30fps)
    
    # Danger zones (scoring areas)
    slot_width: float = 30.0  # Width of slot area
    slot_depth: float = 25.0  # Depth from goal line
    
    # Rink dimensions
    rink_length: float = 200.0
    rink_width: float = 85.0

class PositionHeatmap:
    """
    Generate and maintain position heatmaps for hockey players.
    
    Tracks where players spend time on the ice, with temporal
    weighting to emphasize recent positions.
    """
    
    def __init__(self, config: Optional[PositionAnalyticsConfig] = None):
        self.config = config or PositionAnalyticsConfig()
        
        # Initialize heatmaps for each team
        self.heatmaps = {
            0: np.zeros(self.config.heatmap_resolution, dtype=np.float32),
            1: np.zeros(self.config.heatmap_resolution, dtype=np.float32),
        }
        
        # Combined heatmap
        self.combined_heatmap = np.zeros(self.config.heatmap_resolution, dtype=np.float32)
        
        # Puck heatmap
        self.puck_heatmap = np.zeros(self.config.heatmap_resolution, dtype=np.float32)
        
        self.frame_count = 0
    
    def _rink_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert rink coordinates (feet) to grid cell.
        
        Args:
            x, y: Position in feet (origin at center ice)
            
        Returns:
            (row, col) grid indices
        """
        # Normalize to [0, 1]
        norm_x = (x + self.config.rink_length / 2) / self.config.rink_length
        norm_y = (y + self.config.rink_width / 2) / self.config.rink_width
        
        # Clamp to valid range
        norm_x = np.clip(norm_x, 0, 0.999)
        norm_y = np.clip(norm_y, 0, 0.999)
        
        # Convert to grid indices
        col = int(norm_x * self.config.heatmap_resolution[0])
        row = int(norm_y * self.config.heatmap_resolution[1])
        
        return row, col
    
    def update(
        self, 
        positions: Dict[int, List[Tuple[float, float]]],
        puck_position: Optional[Tuple[float, float]] = None
    ):
        """
        Update heatmaps with new positions.
        
        Args:
            positions: Dict mapping team_id -> list of (x, y) positions
            puck_position: (x, y) of puck if known
        """
        self.frame_count += 1
        
        # Apply temporal decay
        for team_id in self.heatmaps:
            self.heatmaps[team_id] *= self.config.heatmap_decay
        self.combined_heatmap *= self.config.heatmap_decay
        self.puck_heatmap *= self.config.heatmap_decay
        
        # Add new positions
        for team_id, team_positions in positions.items():
            for x, y in team_positions:
                row, col = self._rink_to_grid(x, y)
                if 0 <= row < self.config.heatmap_resolution[1] and \
                   0 <= col < self.config.heatmap_resolution[0]:
                    self.heatmaps[team_id][row, col] += 1.0
                    self.combined_heatmap[row, col] += 1.0
        
        # Update puck heatmap
        if puck_position is not None:
            row, col = self._rink_to_grid(puck_position[0], puck_position[1])
            if 0 <= row < self.config.heatmap_resolution[1] and \
               0 <= col < self.config.heatmap_resolution[0]:
                self.puck_heatmap[row, col] += 1.0
    
    def get_heatmap(
        self, 
        team_id: Optional[int] = None,
        smoothed: bool = True,
        normalized: bool = True
    ) -> np.ndarray:
        """
        Get heatmap for visualization.
        
        Args:
            team_id: Team ID or None for combined
            smoothed: Apply Gaussian smoothing
            normalized: Normalize to [0, 1]
            
        Returns:
            2D heatmap array
        """
        if team_id is not None:
            heatmap = self.heatmaps.get(team_id, self.combined_heatmap).copy()
        else:
            heatmap = self.combined_heatmap.copy()
        
        if smoothed:
            heatmap = ndimage.gaussian_filter(heatmap, sigma=self.config.heatmap_sigma)
        
        if normalized and heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def render_heatmap(
        self,
        rink_image: np.ndarray,
        team_id: Optional[int] = None,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on rink image.
        
        Args:
            rink_image: Base rink image
            team_id: Team ID or None for combined
            alpha: Transparency of overlay
            colormap: OpenCV colormap
            
        Returns:
            Rink image with heatmap overlay
        """
        heatmap = self.get_heatmap(team_id, smoothed=True, normalized=True)
        
        # Resize to match rink image
        heatmap_resized = cv2.resize(
            heatmap, 
            (rink_image.shape[1], rink_image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap
        )
        
        # Blend with rink image
        result = cv2.addWeighted(rink_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return result
    
    def reset(self):
        """Reset all heatmaps."""
        for team_id in self.heatmaps:
            self.heatmaps[team_id].fill(0)
        self.combined_heatmap.fill(0)
        self.puck_heatmap.fill(0)
        self.frame_count = 0

class HotspotDetector:
    """
    Detect hotspots (high-activity areas) on the ice.
    
    Identifies clusters of high player/puck activity that may
    indicate scoring chances, pressure, or strategic focus.
    """
    
    def __init__(self, config: Optional[PositionAnalyticsConfig] = None):
        self.config = config or PositionAnalyticsConfig()
    
    def detect_hotspots(
        self, 
        heatmap: np.ndarray
    ) -> List[Dict]:
        """
        Detect hotspots in a heatmap.
        
        Args:
            heatmap: 2D heatmap array
            
        Returns:
            List of hotspot dictionaries with center, size, intensity
        """
        if heatmap.max() == 0:
            return []
        
        # Normalize
        normalized = heatmap / heatmap.max()
        
        # Threshold
        threshold = np.percentile(normalized[normalized > 0], 
                                   self.config.hotspot_threshold * 100)
        binary = (normalized >= threshold).astype(np.uint8)
        
        # Find connected components
        labeled, num_features = ndimage.label(binary)
        
        hotspots = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            size = np.sum(mask)
            
            if size < self.config.min_hotspot_size:
                continue
            
            # Find center of mass
            cy, cx = ndimage.center_of_mass(normalized, labeled, i)
            
            # Calculate intensity
            intensity = np.mean(normalized[mask])
            
            # Convert grid to rink coordinates
            rink_x = (cx / self.config.heatmap_resolution[0]) * self.config.rink_length - self.config.rink_length / 2
            rink_y = (cy / self.config.heatmap_resolution[1]) * self.config.rink_width - self.config.rink_width / 2
            
            hotspots.append({
                'center': (rink_x, rink_y),
                'grid_center': (cx, cy),
                'size': size,
                'intensity': intensity,
                'zone': self._get_zone(rink_x)
            })
        
        # Sort by intensity
        hotspots.sort(key=lambda h: h['intensity'], reverse=True)
        
        return hotspots
    
    def _get_zone(self, x: float) -> str:
        """Determine zone from x coordinate."""
        if x > self.config.blue_line_x:
            return 'offensive' if x < self.config.goal_line_x else 'behind_goal'
        elif x < -self.config.blue_line_x:
            return 'defensive' if x > -self.config.goal_line_x else 'behind_goal'
        return 'neutral'

class ZoneAnalyzer:
    """
    Analyze time and activity in different rink zones.
    
    Tracks zone possession, time on attack, and territorial control
    for live betting and coaching insights.
    """
    
    def __init__(self, config: Optional[PositionAnalyticsConfig] = None):
        self.config = config or PositionAnalyticsConfig()
        
        # Zone time tracking (in frames)
        self.zone_time = {
            0: defaultdict(int),  # Team 0
            1: defaultdict(int),  # Team 1
        }
        
        # Puck zone time
        self.puck_zone_time = defaultdict(int)
        
        # Recent zone entries (for momentum)
        self.zone_entries = {
            0: deque(maxlen=self.config.momentum_window),
            1: deque(maxlen=self.config.momentum_window),
        }
        
        self.frame_count = 0
    
    def get_zone(self, x: float, team_attacking_right: bool = True) -> RinkZone:
        """
        Determine zone for a position.
        
        Args:
            x: X coordinate (feet from center)
            team_attacking_right: Direction of attack
            
        Returns:
            RinkZone enum
        """
        blue = self.config.blue_line_x
        goal = self.config.goal_line_x
        
        if team_attacking_right:
            if x > goal:
                return RinkZone.BEHIND_GOAL_OFF
            elif x > blue:
                return RinkZone.OFFENSIVE
            elif x < -goal:
                return RinkZone.BEHIND_GOAL_DEF
            elif x < -blue:
                return RinkZone.DEFENSIVE
        else:
            if x < -goal:
                return RinkZone.BEHIND_GOAL_OFF
            elif x < -blue:
                return RinkZone.OFFENSIVE
            elif x > goal:
                return RinkZone.BEHIND_GOAL_DEF
            elif x > blue:
                return RinkZone.DEFENSIVE
        
        return RinkZone.NEUTRAL
    
    def update(
        self,
        team_positions: Dict[int, List[Tuple[float, float]]],
        puck_position: Optional[Tuple[float, float]] = None,
        team_attacking_right: Dict[int, bool] = None
    ):
        """
        Update zone statistics.
        
        Args:
            team_positions: Dict of team_id -> list of positions
            puck_position: Current puck position
            team_attacking_right: Dict of team_id -> attack direction
        """
        self.frame_count += 1
        
        if team_attacking_right is None:
            team_attacking_right = {0: True, 1: False}
        
        # Track player zones
        for team_id, positions in team_positions.items():
            attacking_right = team_attacking_right.get(team_id, team_id == 0)
            
            for x, y in positions:
                zone = self.get_zone(x, attacking_right)
                self.zone_time[team_id][zone] += 1
                
                # Track zone entries
                if zone == RinkZone.OFFENSIVE:
                    self.zone_entries[team_id].append(self.frame_count)
        
        # Track puck zone
        if puck_position is not None:
            puck_zone = self.get_zone(puck_position[0], True)  # Neutral perspective
            self.puck_zone_time[puck_zone] += 1
    
    def get_zone_percentages(self, team_id: int) -> Dict[str, float]:
        """
        Get percentage of time in each zone.
        
        Args:
            team_id: Team ID
            
        Returns:
            Dict mapping zone name to percentage
        """
        total = sum(self.zone_time[team_id].values())
        if total == 0:
            return {z.name.lower(): 0.0 for z in RinkZone}
        
        return {
            zone.name.lower(): (self.zone_time[team_id][zone] / total) * 100
            for zone in RinkZone
        }
    
    def get_offensive_pressure(self, team_id: int, window_frames: int = 300) -> float:
        """
        Calculate offensive pressure (zone entries per window).
        
        Args:
            team_id: Team ID
            window_frames: Analysis window in frames
            
        Returns:
            Offensive pressure score (0-1)
        """
        recent = [e for e in self.zone_entries[team_id] 
                  if e > self.frame_count - window_frames]
        
        # Normalize: ~30 entries per 10 seconds would be very high pressure
        max_entries = window_frames / 10  # Expected max
        pressure = min(len(recent) / max_entries, 1.0)
        
        return pressure
    
    def get_territorial_control(self) -> Dict[str, float]:
        """
        Get territorial control percentage for each team.
        
        Returns:
            Dict with team percentages
        """
        total_0 = sum(self.zone_time[0].values())
        total_1 = sum(self.zone_time[1].values())
        total = total_0 + total_1
        
        if total == 0:
            return {'team_0': 50.0, 'team_1': 50.0}
        
        # Weight offensive zone more heavily
        off_weight = 1.5
        weighted_0 = (self.zone_time[0][RinkZone.OFFENSIVE] * off_weight + 
                      self.zone_time[0][RinkZone.NEUTRAL] +
                      self.zone_time[0][RinkZone.DEFENSIVE] * 0.5)
        weighted_1 = (self.zone_time[1][RinkZone.OFFENSIVE] * off_weight + 
                      self.zone_time[1][RinkZone.NEUTRAL] +
                      self.zone_time[1][RinkZone.DEFENSIVE] * 0.5)
        
        weighted_total = weighted_0 + weighted_1
        if weighted_total == 0:
            return {'team_0': 50.0, 'team_1': 50.0}
        
        return {
            'team_0': (weighted_0 / weighted_total) * 100,
            'team_1': (weighted_1 / weighted_total) * 100
        }
    
    def reset(self):
        """Reset all zone statistics."""
        for team_id in self.zone_time:
            self.zone_time[team_id].clear()
            self.zone_entries[team_id].clear()
        self.puck_zone_time.clear()
        self.frame_count = 0

class PositionQualityAnalyzer:
    """
    Analyze whether player positions are optimal or suboptimal.
    
    Evaluates positions based on:
    - Proximity to high-danger areas (offensive)
    - Coverage of passing lanes (defensive)
    - Support positioning
    - Odd-man situations
    """
    
    def __init__(self, config: Optional[PositionAnalyticsConfig] = None):
        self.config = config or PositionAnalyticsConfig()
    
    def is_in_slot(self, x: float, y: float, attacking_right: bool = True) -> bool:
        """
        Check if position is in the slot (high-danger scoring area).
        
        Args:
            x, y: Position in feet
            attacking_right: Attack direction
            
        Returns:
            True if in slot
        """
        goal_x = self.config.goal_line_x if attacking_right else -self.config.goal_line_x
        
        # Slot is area in front of goal
        if attacking_right:
            in_x_range = (goal_x - self.config.slot_depth) < x < goal_x
        else:
            in_x_range = goal_x < x < (goal_x + self.config.slot_depth)
        
        in_y_range = abs(y) < self.config.slot_width / 2
        
        return in_x_range and in_y_range
    
    def analyze_offensive_position(
        self,
        player_pos: Tuple[float, float],
        puck_pos: Optional[Tuple[float, float]],
        attacking_right: bool = True
    ) -> Tuple[PositionQuality, str]:
        """
        Analyze quality of offensive positioning.
        
        Args:
            player_pos: Player position
            puck_pos: Puck position
            attacking_right: Attack direction
            
        Returns:
            (quality, reason)
        """
        x, y = player_pos
        
        # In the slot = optimal
        if self.is_in_slot(x, y, attacking_right):
            return PositionQuality.OPTIMAL, "In slot (high-danger area)"
        
        # In offensive zone
        if attacking_right and x > self.config.blue_line_x:
            if puck_pos is not None:
                dist_to_puck = np.sqrt((x - puck_pos[0])**2 + (y - puck_pos[1])**2)
                if dist_to_puck < 15:  # Within 15 feet of puck
                    return PositionQuality.GOOD, "Near puck in offensive zone"
            return PositionQuality.NEUTRAL, "In offensive zone"
        elif not attacking_right and x < -self.config.blue_line_x:
            return PositionQuality.NEUTRAL, "In offensive zone"
        
        # In neutral zone when should be attacking
        if abs(x) < self.config.blue_line_x:
            return PositionQuality.SUBOPTIMAL, "In neutral zone during attack"
        
        return PositionQuality.SUBOPTIMAL, "Poor offensive positioning"
    
    def analyze_defensive_position(
        self,
        player_pos: Tuple[float, float],
        opponents: List[Tuple[float, float]],
        puck_pos: Optional[Tuple[float, float]],
        attacking_right: bool = True
    ) -> Tuple[PositionQuality, str]:
        """
        Analyze quality of defensive positioning.
        
        Args:
            player_pos: Defender position
            opponents: List of opponent positions
            puck_pos: Puck position
            attacking_right: Team attack direction (inverted for defense)
            
        Returns:
            (quality, reason)
        """
        x, y = player_pos
        defensive_zone_x = -self.config.blue_line_x if attacking_right else self.config.blue_line_x
        
        # Check if any opponent is in the slot uncovered
        for opp in opponents:
            if self.is_in_slot(opp[0], opp[1], not attacking_right):
                # Is this player covering them?
                dist = np.sqrt((x - opp[0])**2 + (y - opp[1])**2)
                if dist < 8:  # Within 8 feet = covering
                    return PositionQuality.OPTIMAL, "Covering opponent in slot"
        
        # In proper defensive position
        if (attacking_right and x < defensive_zone_x) or \
           (not attacking_right and x > defensive_zone_x):
            return PositionQuality.GOOD, "In defensive zone"
        
        # Caught out of position
        return PositionQuality.SUBOPTIMAL, "Out of defensive position"
    
    def detect_odd_man_rush(
        self,
        attackers: List[Tuple[float, float]],
        defenders: List[Tuple[float, float]],
        puck_pos: Optional[Tuple[float, float]],
        attacking_right: bool = True
    ) -> Optional[Dict]:
        """
        Detect odd-man rush situations (2-on-1, 3-on-2, breakaway).
        
        Args:
            attackers: Attacking team positions
            defenders: Defending team positions (excluding goalie)
            puck_pos: Puck position
            attacking_right: Attack direction
            
        Returns:
            Dict with rush info or None
        """
        if puck_pos is None:
            return None
        
        # Define rush zone (between blue lines and goal)
        if attacking_right:
            rush_zone = (self.config.blue_line_x, self.config.goal_line_x)
        else:
            rush_zone = (-self.config.goal_line_x, -self.config.blue_line_x)
        
        # Count attackers and defenders in rush zone
        def in_zone(pos):
            if attacking_right:
                return rush_zone[0] < pos[0] < rush_zone[1]
            else:
                return rush_zone[0] < pos[0] < rush_zone[1]
        
        attackers_in_zone = [a for a in attackers if in_zone(a)]
        defenders_in_zone = [d for d in defenders if in_zone(d)]
        
        n_att = len(attackers_in_zone)
        n_def = len(defenders_in_zone)
        
        # Detect odd-man situations
        if n_att > 0 and n_def == 0:
            return {
                'type': 'breakaway',
                'attackers': n_att,
                'defenders': n_def,
                'danger': PositionQuality.CRITICAL
            }
        elif n_att > n_def and n_att >= 2:
            return {
                'type': f'{n_att}-on-{n_def}',
                'attackers': n_att,
                'defenders': n_def,
                'danger': PositionQuality.CRITICAL if n_att - n_def >= 2 else PositionQuality.SUBOPTIMAL
            }
        
        return None

class HockeyPositionAnalytics:
    """
    Main class combining all position analytics.
    
    Provides unified interface for:
    - Heatmap generation
    - Hotspot detection
    - Zone analysis
    - Position quality evaluation
    """
    
    def __init__(self, config: Optional[PositionAnalyticsConfig] = None):
        self.config = config or PositionAnalyticsConfig()
        
        self.heatmap = PositionHeatmap(config)
        self.hotspot_detector = HotspotDetector(config)
        self.zone_analyzer = ZoneAnalyzer(config)
        self.position_analyzer = PositionQualityAnalyzer(config)
        
        self.frame_count = 0
    
    def update(
        self,
        team_positions: Dict[int, List[Tuple[float, float]]],
        puck_position: Optional[Tuple[float, float]] = None,
        team_attacking_right: Dict[int, bool] = None
    ):
        """
        Update all analytics with new positions.
        
        Args:
            team_positions: Dict of team_id -> list of (x, y) positions
            puck_position: Current puck position
            team_attacking_right: Dict of team_id -> attack direction
        """
        self.frame_count += 1
        
        # Update all components
        self.heatmap.update(team_positions, puck_position)
        self.zone_analyzer.update(team_positions, puck_position, team_attacking_right)
    
    def get_analytics_summary(self) -> Dict:
        """
        Get comprehensive analytics summary.
        
        Returns:
            Dict with all analytics metrics
        """
        # Get hotspots
        hotspots_0 = self.hotspot_detector.detect_hotspots(
            self.heatmap.get_heatmap(0, smoothed=True, normalized=False)
        )
        hotspots_1 = self.hotspot_detector.detect_hotspots(
            self.heatmap.get_heatmap(1, smoothed=True, normalized=False)
        )
        
        return {
            'frame_count': self.frame_count,
            'zone_percentages': {
                'team_0': self.zone_analyzer.get_zone_percentages(0),
                'team_1': self.zone_analyzer.get_zone_percentages(1),
            },
            'offensive_pressure': {
                'team_0': self.zone_analyzer.get_offensive_pressure(0),
                'team_1': self.zone_analyzer.get_offensive_pressure(1),
            },
            'territorial_control': self.zone_analyzer.get_territorial_control(),
            'hotspots': {
                'team_0': hotspots_0[:3],  # Top 3
                'team_1': hotspots_1[:3],
            }
        }
    
    def reset(self):
        """Reset all analytics."""
        self.heatmap.reset()
        self.zone_analyzer.reset()
        self.frame_count = 0

# Example usage
if __name__ == "__main__":
    # Initialize analytics
    config = PositionAnalyticsConfig(
        heatmap_resolution=(100, 43),
        momentum_window=300
    )
    
    analytics = HockeyPositionAnalytics(config)
    
    print("Hockey Position Analytics Module Ready!")
    print(f"\nFeatures:")
    print(f"  - Position heatmaps (per team and combined)")
    print(f"  - Hotspot detection")
    print(f"  - Zone time analysis")
    print(f"  - Offensive pressure tracking")
    print(f"  - Territorial control metrics")
    print(f"  - Position quality evaluation")
    print(f"  - Odd-man rush detection")
    print(f"\nConfiguration:")
    print(f"  - Heatmap resolution: {config.heatmap_resolution}")
    print(f"  - Momentum window: {config.momentum_window} frames")
    print(f"  - Slot dimensions: {config.slot_width}ft x {config.slot_depth}ft")
