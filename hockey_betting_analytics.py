import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
import json
from datetime import datetime

class BettingSignal(IntEnum):
    """Betting signal strength."""
    STRONG_AWAY = -2
    LEAN_AWAY = -1
    NEUTRAL = 0
    LEAN_HOME = 1
    STRONG_HOME = 2


class DangerLevel(IntEnum):
    """Scoring danger level."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3  # Breakaway, empty net, etc.


@dataclass
class BettingAnalyticsConfig:
    """Configuration for betting analytics."""
    
    # Momentum calculation
    momentum_window_short: int = 150   # ~5 seconds at 30fps
    momentum_window_medium: int = 450  # ~15 seconds
    momentum_window_long: int = 900    # ~30 seconds
    
    # Scoring chance weights
    slot_weight: float = 3.0           # High-danger area
    inner_slot_weight: float = 5.0     # Very high-danger
    point_shot_weight: float = 1.0     # Low-danger
    rebound_bonus: float = 1.5         # Multiplier for rebounds
    
    # Pressure thresholds
    high_pressure_threshold: float = 0.7
    low_pressure_threshold: float = 0.3
    
    # Signal thresholds
    strong_signal_threshold: float = 0.7
    lean_signal_threshold: float = 0.55
    
    # Fatigue detection
    shift_length_warning: int = 45     # Seconds for tired shift
    movement_drop_threshold: float = 0.7  # 30% movement decrease
    
    # Rink dimensions
    rink_length: float = 200.0
    rink_width: float = 85.0
    goal_line_x: float = 89.0
    blue_line_x: float = 25.0
    
    # Slot dimensions
    slot_width: float = 30.0
    slot_depth: float = 25.0
    inner_slot_width: float = 16.0
    inner_slot_depth: float = 12.0

class MomentumTracker:
    """
    Track game momentum for betting signals.
    
    Momentum is calculated from:
    - Zone time
    - Shot attempts
    - Scoring chances
    - Puck possession
    """
    
    def __init__(self, config: Optional[BettingAnalyticsConfig] = None):
        self.config = config or BettingAnalyticsConfig()
        
        # Event history for different windows
        self.zone_events = deque(maxlen=self.config.momentum_window_long)
        self.chance_events = deque(maxlen=self.config.momentum_window_long)
        self.possession_events = deque(maxlen=self.config.momentum_window_long)
        
        self.frame_count = 0
    
    def update(
        self,
        team_in_offensive_zone: Optional[int],
        scoring_chance: Optional[Dict] = None,
        puck_carrier_team: Optional[int] = None
    ):
        """
        Update momentum tracking.
        
        Args:
            team_in_offensive_zone: Which team (0/1) is in offensive zone
            scoring_chance: Dict with chance details if one occurred
            puck_carrier_team: Which team has the puck
        """
        self.frame_count += 1
        
        # Record zone event
        self.zone_events.append({
            'frame': self.frame_count,
            'team': team_in_offensive_zone
        })
        
        # Record scoring chance
        if scoring_chance is not None:
            self.chance_events.append({
                'frame': self.frame_count,
                **scoring_chance
            })
        
        # Record possession
        if puck_carrier_team is not None:
            self.possession_events.append({
                'frame': self.frame_count,
                'team': puck_carrier_team
            })
    
    def get_momentum(
        self, 
        window: str = 'medium'
    ) -> Tuple[float, int]:
        """
        Calculate current momentum.
        
        Args:
            window: 'short', 'medium', or 'long'
            
        Returns:
            (momentum_score, favored_team)
            Score is -1 to 1, negative = team 1, positive = team 0
        """
        window_size = {
            'short': self.config.momentum_window_short,
            'medium': self.config.momentum_window_medium,
            'long': self.config.momentum_window_long
        }.get(window, self.config.momentum_window_medium)
        
        cutoff = self.frame_count - window_size
        
        # Zone time momentum
        zone_team_0 = sum(1 for e in self.zone_events if e['frame'] > cutoff and e['team'] == 0)
        zone_team_1 = sum(1 for e in self.zone_events if e['frame'] > cutoff and e['team'] == 1)
        zone_total = zone_team_0 + zone_team_1
        zone_momentum = (zone_team_0 - zone_team_1) / max(zone_total, 1)
        
        # Scoring chances momentum (weighted by danger)
        chances_0 = sum(e.get('weight', 1) for e in self.chance_events 
                        if e['frame'] > cutoff and e.get('team') == 0)
        chances_1 = sum(e.get('weight', 1) for e in self.chance_events 
                        if e['frame'] > cutoff and e.get('team') == 1)
        chance_total = chances_0 + chances_1
        chance_momentum = (chances_0 - chances_1) / max(chance_total, 1) if chance_total > 0 else 0
        
        # Possession momentum
        poss_0 = sum(1 for e in self.possession_events if e['frame'] > cutoff and e['team'] == 0)
        poss_1 = sum(1 for e in self.possession_events if e['frame'] > cutoff and e['team'] == 1)
        poss_total = poss_0 + poss_1
        poss_momentum = (poss_0 - poss_1) / max(poss_total, 1)
        
        # Weighted combination
        momentum = (
            0.3 * zone_momentum +
            0.5 * chance_momentum +  # Chances weighted more heavily
            0.2 * poss_momentum
        )
        
        favored_team = 0 if momentum >= 0 else 1
        
        return momentum, favored_team
    
    def get_momentum_trend(self) -> str:
        """
        Get momentum trend (comparing short vs long term).
        
        Returns:
            'increasing', 'decreasing', or 'stable'
        """
        short_mom, _ = self.get_momentum('short')
        long_mom, _ = self.get_momentum('long')
        
        diff = short_mom - long_mom
        
        if diff > 0.15:
            return 'increasing_home' if short_mom > 0 else 'increasing_away'
        elif diff < -0.15:
            return 'decreasing_home' if short_mom > 0 else 'decreasing_away'
        return 'stable'
    
    def reset(self):
        """Reset momentum tracking."""
        self.zone_events.clear()
        self.chance_events.clear()
        self.possession_events.clear()
        self.frame_count = 0

class ScoringChanceAnalyzer:
    """
    Analyze scoring chances and expected goals.
    
    Based on position quality, shot location, and situation.
    """
    
    def __init__(self, config: Optional[BettingAnalyticsConfig] = None):
        self.config = config or BettingAnalyticsConfig()
        
        # Expected goal values by area (based on NHL data)
        self.xg_map = self._create_xg_map()
    
    def _create_xg_map(self) -> np.ndarray:
        """
        Create expected goals (xG) heatmap.
        Higher values in slot, lower from distance.
        """
        # 100x43 grid (same as heatmap)
        xg = np.zeros((43, 100), dtype=np.float32)
        
        # Fill based on distance from goal
        for i in range(43):  # y
            for j in range(100):  # x
                # Convert to rink coordinates
                x = (j / 100) * self.config.rink_length - self.config.rink_length / 2
                y = (i / 43) * self.config.rink_width - self.config.rink_width / 2
                
                # Distance to right goal
                goal_x = self.config.goal_line_x
                dist = np.sqrt((x - goal_x)**2 + y**2)
                
                # Base xG from distance
                if x > 0:  # Right side of rink
                    base_xg = max(0, 0.3 - dist * 0.004)
                    
                    # Bonus for slot
                    if abs(y) < self.config.slot_width / 2:
                        if self.config.goal_line_x - x < self.config.slot_depth:
                            base_xg *= 2.0
                            
                            # Extra bonus for inner slot
                            if abs(y) < self.config.inner_slot_width / 2:
                                if self.config.goal_line_x - x < self.config.inner_slot_depth:
                                    base_xg *= 1.5
                    
                    xg[i, j] = min(base_xg, 0.5)  # Cap at 50%
        
        return xg
    
    def evaluate_chance(
        self,
        position: Tuple[float, float],
        is_rebound: bool = False,
        is_one_timer: bool = False,
        defenders_in_lane: int = 0
    ) -> Dict:
        """
        Evaluate a scoring chance.
        
        Args:
            position: (x, y) of shot/chance
            is_rebound: Is this a rebound chance
            is_one_timer: Is this a one-timer
            defenders_in_lane: Number of defenders between shooter and goal
            
        Returns:
            Dict with xG, danger level, and description
        """
        x, y = position
        
        # Get base xG from map
        grid_x = int(((x + self.config.rink_length/2) / self.config.rink_length) * 100)
        grid_y = int(((y + self.config.rink_width/2) / self.config.rink_width) * 43)
        grid_x = np.clip(grid_x, 0, 99)
        grid_y = np.clip(grid_y, 0, 42)
        
        base_xg = self.xg_map[grid_y, grid_x]
        
        # Apply modifiers
        xg = base_xg
        
        if is_rebound:
            xg *= self.config.rebound_bonus
        
        if is_one_timer:
            xg *= 1.3  # One-timers harder to save
        
        # Reduce for defenders in lane
        xg *= (1 - defenders_in_lane * 0.15)
        
        xg = max(0, min(xg, 0.8))  # Clamp
        
        # Determine danger level
        if xg >= 0.3:
            danger = DangerLevel.CRITICAL
        elif xg >= 0.15:
            danger = DangerLevel.HIGH
        elif xg >= 0.08:
            danger = DangerLevel.MEDIUM
        else:
            danger = DangerLevel.LOW
        
        # Description
        dist_to_goal = np.sqrt((x - self.config.goal_line_x)**2 + y**2)
        if dist_to_goal < 15:
            location = "inner slot"
        elif dist_to_goal < 25:
            location = "slot"
        elif dist_to_goal < 40:
            location = "high slot"
        else:
            location = "point"
        
        return {
            'xg': xg,
            'danger': danger,
            'location': location,
            'distance': dist_to_goal,
            'modifiers': {
                'rebound': is_rebound,
                'one_timer': is_one_timer,
                'blocked_lanes': defenders_in_lane
            }
        }

class LiveBettingSignals:
    """
    Generate betting signals from analytics.
    
    Combines momentum, scoring chances, and position analysis
    to generate actionable betting insights.
    """
    
    def __init__(self, config: Optional[BettingAnalyticsConfig] = None):
        self.config = config or BettingAnalyticsConfig()
        
        self.momentum_tracker = MomentumTracker(config)
        self.chance_analyzer = ScoringChanceAnalyzer(config)
        
        # Signal history
        self.signal_history = deque(maxlen=1000)
        
        # Cumulative xG
        self.cumulative_xg = {0: 0.0, 1: 0.0}
        
        self.frame_count = 0
    
    def update(
        self,
        team_positions: Dict[int, List[Tuple[float, float]]],
        puck_position: Optional[Tuple[float, float]],
        puck_carrier_team: Optional[int] = None
    ) -> Dict:
        """
        Update analytics and generate signals.
        
        Args:
            team_positions: Dict of team_id -> positions
            puck_position: Current puck position
            puck_carrier_team: Which team has the puck
            
        Returns:
            Dict with current signals and insights
        """
        self.frame_count += 1
        
        # Determine which team is in offensive zone
        team_in_offense = None
        if puck_position is not None:
            if puck_position[0] > self.config.blue_line_x:
                team_in_offense = 0  # Team 0 attacking right
            elif puck_position[0] < -self.config.blue_line_x:
                team_in_offense = 1  # Team 1 attacking left
        
        # Check for scoring chance
        scoring_chance = None
        if puck_position is not None and team_in_offense is not None:
            # Simple heuristic: chance if puck in dangerous area
            chance_eval = self.chance_analyzer.evaluate_chance(puck_position)
            
            if chance_eval['danger'] >= DangerLevel.MEDIUM:
                scoring_chance = {
                    'team': team_in_offense,
                    'weight': chance_eval['xg'] * 10,  # Scale for momentum
                    **chance_eval
                }
                
                # Add to cumulative xG
                self.cumulative_xg[team_in_offense] += chance_eval['xg'] / 30  # Per frame
        
        # Update momentum
        self.momentum_tracker.update(team_in_offense, scoring_chance, puck_carrier_team)
        
        # Generate signal
        signal = self._generate_signal()
        
        self.signal_history.append({
            'frame': self.frame_count,
            **signal
        })
        
        return signal
    
    def _generate_signal(self) -> Dict:
        """
        Generate betting signal from current state.
        
        Returns:
            Dict with signal, confidence, and reasoning
        """
        # Get momentum at different windows
        short_mom, _ = self.momentum_tracker.get_momentum('short')
        medium_mom, _ = self.momentum_tracker.get_momentum('medium')
        long_mom, _ = self.momentum_tracker.get_momentum('long')
        
        trend = self.momentum_tracker.get_momentum_trend()
        
        # Combine signals
        combined = 0.5 * short_mom + 0.3 * medium_mom + 0.2 * long_mom
        
        # Determine signal strength
        if abs(combined) >= self.config.strong_signal_threshold:
            signal = BettingSignal.STRONG_HOME if combined > 0 else BettingSignal.STRONG_AWAY
            confidence = min(abs(combined) * 1.2, 0.95)
        elif abs(combined) >= self.config.lean_signal_threshold:
            signal = BettingSignal.LEAN_HOME if combined > 0 else BettingSignal.LEAN_AWAY
            confidence = abs(combined)
        else:
            signal = BettingSignal.NEUTRAL
            confidence = 0.5
        
        # Generate reasoning
        reasons = []
        
        if abs(short_mom) > 0.5:
            team = "Home" if short_mom > 0 else "Away"
            reasons.append(f"{team} dominating recent play")
        
        if self.cumulative_xg[0] > self.cumulative_xg[1] * 1.5:
            reasons.append("Home generating quality chances")
        elif self.cumulative_xg[1] > self.cumulative_xg[0] * 1.5:
            reasons.append("Away generating quality chances")
        
        if 'increasing' in trend:
            reasons.append(f"Momentum {trend}")
        
        return {
            'signal': signal,
            'signal_name': signal.name,
            'confidence': confidence,
            'momentum': {
                'short': short_mom,
                'medium': medium_mom,
                'long': long_mom,
                'trend': trend
            },
            'xg': {
                'home': round(self.cumulative_xg[0], 2),
                'away': round(self.cumulative_xg[1], 2)
            },
            'reasons': reasons
        }
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive betting analytics summary.
        
        Returns:
            Dict with all betting-relevant metrics
        """
        current_signal = self._generate_signal()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'frame': self.frame_count,
            'current_signal': current_signal,
            'expected_goals': {
                'home': round(self.cumulative_xg[0], 2),
                'away': round(self.cumulative_xg[1], 2),
                'difference': round(self.cumulative_xg[0] - self.cumulative_xg[1], 2)
            },
            'insights': self._generate_insights()
        }
    
    def _generate_insights(self) -> List[str]:
        """
        Generate human-readable insights.
        
        Returns:
            List of insight strings
        """
        insights = []
        
        short_mom, _ = self.momentum_tracker.get_momentum('short')
        medium_mom, _ = self.momentum_tracker.get_momentum('medium')
        
        # Momentum insights
        if short_mom > 0.6:
            insights.append("🔥 HOME TEAM SURGE - Strong recent pressure")
        elif short_mom < -0.6:
            insights.append("🔥 AWAY TEAM SURGE - Strong recent pressure")
        
        # xG insights
        xg_diff = self.cumulative_xg[0] - self.cumulative_xg[1]
        if xg_diff > 0.5:
            insights.append(f"📊 Home outperforming in xG ({xg_diff:+.2f})")
        elif xg_diff < -0.5:
            insights.append(f"📊 Away outperforming in xG ({xg_diff:+.2f})")
        
        # Trend insights
        trend = self.momentum_tracker.get_momentum_trend()
        if 'increasing_home' in trend:
            insights.append("📈 Home momentum building")
        elif 'increasing_away' in trend:
            insights.append("📈 Away momentum building")
        
        return insights
    
    def reset(self):
        """Reset all tracking."""
        self.momentum_tracker.reset()
        self.signal_history.clear()
        self.cumulative_xg = {0: 0.0, 1: 0.0}
        self.frame_count = 0

class HockeyBettingDashboard:
    """
    Main dashboard combining all betting analytics.
    
    Provides unified interface for video analysis
    and real-time betting signal generation.
    """
    
    def __init__(self, config: Optional[BettingAnalyticsConfig] = None):
        self.config = config or BettingAnalyticsConfig()
        
        self.signals = LiveBettingSignals(config)
        
        # Team info
        self.team_names = {0: "Home", 1: "Away"}
        
        self.frame_count = 0
    
    def set_team_names(self, home: str, away: str):
        """Set team names for display."""
        self.team_names = {0: home, 1: away}
    
    def process_frame(
        self,
        team_positions: Dict[int, List[Tuple[float, float]]],
        puck_position: Optional[Tuple[float, float]] = None,
        puck_carrier_team: Optional[int] = None
    ) -> Dict:
        """
        Process a frame and generate betting signals.
        
        Args:
            team_positions: Player positions by team
            puck_position: Puck position (rink coordinates)
            puck_carrier_team: Which team has the puck
            
        Returns:
            Dict with betting signals and insights
        """
        self.frame_count += 1
        
        # Update signals
        signal_update = self.signals.update(
            team_positions, 
            puck_position, 
            puck_carrier_team
        )
        
        return signal_update
    
    def get_dashboard_data(self) -> Dict:
        """
        Get complete dashboard data for display.
        
        Returns:
            Dict with all dashboard metrics
        """
        summary = self.signals.get_summary()
        
        return {
            'teams': self.team_names,
            'frame': self.frame_count,
            'betting_signal': {
                'recommendation': summary['current_signal']['signal_name'],
                'confidence': f"{summary['current_signal']['confidence']:.1%}",
                'reasons': summary['current_signal']['reasons']
            },
            'momentum': {
                'current': summary['current_signal']['momentum']['medium'],
                'trend': summary['current_signal']['momentum']['trend'],
                'favoring': self.team_names[0] if summary['current_signal']['momentum']['medium'] > 0 else self.team_names[1]
            },
            'expected_goals': summary['expected_goals'],
            'insights': summary['insights']
        }
    
    def render_overlay(
        self,
        frame: np.ndarray,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Render betting analytics overlay on frame.
        
        Args:
            frame: Input frame
            position: Top-left position for overlay
            
        Returns:
            Frame with overlay
        """
        result = frame.copy()
        data = self.get_dashboard_data()
        
        x, y = position
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Background box
        cv2.rectangle(result, (x-5, y-20), (x+300, y+150), (0, 0, 0), -1)
        cv2.rectangle(result, (x-5, y-20), (x+300, y+150), (255, 255, 255), 1)
        
        # Title
        cv2.putText(result, "BETTING ANALYTICS", (x, y), font, 0.6, (0, 255, 255), 2)
        y += line_height
        
        # Signal
        signal = data['betting_signal']['recommendation']
        signal_color = (0, 255, 0) if 'HOME' in signal else (0, 0, 255) if 'AWAY' in signal else (200, 200, 200)
        cv2.putText(result, f"Signal: {signal}", (x, y), font, 0.5, signal_color, 1)
        y += line_height
        
        # Confidence
        cv2.putText(result, f"Confidence: {data['betting_signal']['confidence']}", (x, y), font, 0.5, (255, 255, 255), 1)
        y += line_height
        
        # Momentum
        mom = data['momentum']['current']
        mom_color = (0, 255, 0) if mom > 0.3 else (0, 0, 255) if mom < -0.3 else (200, 200, 200)
        cv2.putText(result, f"Momentum: {data['momentum']['favoring']}", (x, y), font, 0.5, mom_color, 1)
        y += line_height
        
        # xG
        cv2.putText(result, f"xG: {self.team_names[0]} {data['expected_goals']['home']:.2f} - {data['expected_goals']['away']:.2f} {self.team_names[1]}", 
                    (x, y), font, 0.5, (255, 255, 255), 1)
        
        return result
    
    def export_signals(self, filepath: str):
        """
        Export signal history to JSON.
        
        Args:
            filepath: Output file path
        """
        data = {
            'teams': self.team_names,
            'total_frames': self.frame_count,
            'final_xg': self.signals.cumulative_xg,
            'signal_history': list(self.signals.signal_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def reset(self):
        """Reset dashboard."""
        self.signals.reset()
        self.frame_count = 0

# Example usage
if __name__ == "__main__":
    # Initialize dashboard
    config = BettingAnalyticsConfig(
        momentum_window_short=150,
        momentum_window_medium=450,
        momentum_window_long=900
    )
    
    dashboard = HockeyBettingDashboard(config)
    dashboard.set_team_names("Penguins", "Oilers")
    
    print("Hockey Betting Analytics Dashboard Ready!")
    print(f"\nFeatures:")
    print(f"  - Real-time betting signals (STRONG/LEAN HOME/AWAY/NEUTRAL)")
    print(f"  - Momentum tracking (short/medium/long term)")
    print(f"  - Expected goals (xG) calculation")
    print(f"  - Scoring chance quality analysis")
    print(f"  - Position quality insights")
    print(f"\nSignal Thresholds:")
    print(f"  - Strong signal: {config.strong_signal_threshold:.0%}+ confidence")
    print(f"  - Lean signal: {config.lean_signal_threshold:.0%}+ confidence")
    print(f"\nMomentum Windows:")
    print(f"  - Short: {config.momentum_window_short} frames (~{config.momentum_window_short/30:.0f}s)")
    print(f"  - Medium: {config.momentum_window_medium} frames (~{config.momentum_window_medium/30:.0f}s)")
    print(f"  - Long: {config.momentum_window_long} frames (~{config.momentum_window_long/30:.0f}s)")
