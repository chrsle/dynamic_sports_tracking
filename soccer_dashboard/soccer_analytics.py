"""
Soccer Analytics Module
Comprehensive analytics engine for soccer/football match analysis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import deque
import math
import json
from datetime import datetime


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ShotResult(Enum):
    GOAL = "goal"
    SAVED = "saved"
    BLOCKED = "blocked"
    OFF_TARGET = "off_target"
    POST = "post"


class Zone(Enum):
    DEFENSIVE_THIRD = "defensive"
    MIDDLE_THIRD = "middle"
    ATTACKING_THIRD = "attacking"
    BOX = "box"
    SIX_YARD_BOX = "six_yard_box"


class FormationType(Enum):
    ATTACKING = "attacking"
    DEFENSIVE = "defensive"
    BALANCED = "balanced"
    HIGH_PRESS = "high_press"
    LOW_BLOCK = "low_block"


class PressureLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# Soccer pitch dimensions (in meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
BOX_LENGTH = 16.5
BOX_WIDTH = 40.32
SIX_YARD_LENGTH = 5.5
SIX_YARD_WIDTH = 18.32
GOAL_WIDTH = 7.32


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    """2D position on the pitch"""
    x: float  # 0-105m (length)
    y: float  # 0-68m (width)
    timestamp: float = 0.0

    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "timestamp": self.timestamp}


@dataclass
class Player:
    """Player tracking data"""
    player_id: int
    team: str  # "home" or "away"
    position: Position
    jersey_number: Optional[int] = None
    name: Optional[str] = None
    velocity: float = 0.0  # m/s
    acceleration: float = 0.0  # m/s^2
    distance_covered: float = 0.0  # meters
    sprints: int = 0
    high_intensity_runs: int = 0

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "team": self.team,
            "position": self.position.to_dict(),
            "jersey_number": self.jersey_number,
            "name": self.name,
            "velocity": round(self.velocity, 2),
            "acceleration": round(self.acceleration, 2),
            "distance_covered": round(self.distance_covered, 1),
            "sprints": self.sprints,
            "high_intensity_runs": self.high_intensity_runs
        }


@dataclass
class Shot:
    """Shot event data with xG calculation"""
    shot_id: int
    player_id: int
    team: str
    position: Position
    target_position: Position  # Where the shot was aimed
    result: ShotResult
    xg: float
    timestamp: float
    distance: float
    angle: float
    body_part: str = "foot"
    assisted: bool = False
    is_header: bool = False
    is_first_touch: bool = False
    defenders_in_path: int = 0

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "player_id": self.player_id,
            "team": self.team,
            "position": self.position.to_dict(),
            "target_position": self.target_position.to_dict(),
            "result": self.result.value,
            "xg": round(self.xg, 3),
            "timestamp": self.timestamp,
            "distance": round(self.distance, 1),
            "angle": round(self.angle, 1),
            "body_part": self.body_part,
            "is_header": self.is_header,
            "defenders_in_path": self.defenders_in_path
        }


@dataclass
class GoalieMetrics:
    """Goalkeeper analytics"""
    player_id: int
    team: str
    position: Position
    coverage_area: float  # % of goal covered
    positioning_quality: float  # 0-1 score
    reaction_time: float  # seconds
    saves: int = 0
    goals_conceded: int = 0
    xg_prevented: float = 0.0
    distance_from_line: float = 0.0
    angle_to_ball: float = 0.0

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "team": self.team,
            "position": self.position.to_dict(),
            "coverage_area": round(self.coverage_area, 1),
            "positioning_quality": round(self.positioning_quality, 2),
            "reaction_time": round(self.reaction_time, 3),
            "saves": self.saves,
            "goals_conceded": self.goals_conceded,
            "xg_prevented": round(self.xg_prevented, 3),
            "distance_from_line": round(self.distance_from_line, 1)
        }


@dataclass
class TeamState:
    """Current team state and metrics"""
    team: str
    players: List[Player]
    formation: str
    formation_type: FormationType
    possession: float
    pressing_intensity: float
    defensive_line_height: float
    compactness: float
    xg: float = 0.0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    passes: int = 0
    pass_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "formation": self.formation,
            "formation_type": self.formation_type.value,
            "possession": round(self.possession, 1),
            "pressing_intensity": round(self.pressing_intensity, 2),
            "defensive_line_height": round(self.defensive_line_height, 1),
            "compactness": round(self.compactness, 1),
            "xg": round(self.xg, 3),
            "shots": self.shots,
            "shots_on_target": self.shots_on_target,
            "goals": self.goals,
            "passes": self.passes,
            "pass_accuracy": round(self.pass_accuracy, 1)
        }


@dataclass
class ManAdvantageState:
    """Man advantage (red card) situation"""
    is_active: bool = False
    advantage_team: Optional[str] = None
    home_players: int = 11
    away_players: int = 11
    time_remaining: float = 0.0  # seconds since red card
    red_card_player_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "is_active": self.is_active,
            "advantage_team": self.advantage_team,
            "home_players": self.home_players,
            "away_players": self.away_players,
            "time_remaining": round(self.time_remaining, 0),
            "situation": f"{self.home_players}v{self.away_players}" if self.is_active else "11v11"
        }


@dataclass
class FatigueMetrics:
    """Player fatigue analysis"""
    player_id: int
    fatigue_level: float  # 0-1 (1 = exhausted)
    avg_speed_decline: float  # % decline from match start
    sprint_decline: float  # % decline in sprint frequency
    recovery_time_increase: float  # % increase in recovery time
    distance_last_5min: float
    distance_first_5min: float

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "fatigue_level": round(self.fatigue_level, 2),
            "avg_speed_decline": round(self.avg_speed_decline, 1),
            "sprint_decline": round(self.sprint_decline, 1),
            "recovery_time_increase": round(self.recovery_time_increase, 1),
            "distance_last_5min": round(self.distance_last_5min, 0),
            "distance_first_5min": round(self.distance_first_5min, 0)
        }


@dataclass
class PressureMetrics:
    """Team pressing and pressure analytics"""
    team: str
    ppda: float  # Passes allowed per defensive action
    high_press_success_rate: float
    counter_press_intensity: float
    pressing_triggers: int  # Number of press initiations
    regains_in_final_third: int
    time_to_regain: float  # Average seconds to regain possession

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "ppda": round(self.ppda, 2),
            "high_press_success_rate": round(self.high_press_success_rate, 1),
            "counter_press_intensity": round(self.counter_press_intensity, 2),
            "pressing_triggers": self.pressing_triggers,
            "regains_in_final_third": self.regains_in_final_third,
            "time_to_regain": round(self.time_to_regain, 1)
        }


@dataclass
class PatternMetrics:
    """Play pattern analysis"""
    team: str
    build_up_speed: str  # "slow", "medium", "fast", "direct"
    attacking_width: float  # Average width of attacks
    progression_style: str  # "short_passing", "long_balls", "mixed"
    third_man_runs: int
    overlapping_runs: int
    underlapping_runs: int
    switches_of_play: int
    through_balls: int
    key_passes: int

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "build_up_speed": self.build_up_speed,
            "attacking_width": round(self.attacking_width, 1),
            "progression_style": self.progression_style,
            "third_man_runs": self.third_man_runs,
            "overlapping_runs": self.overlapping_runs,
            "underlapping_runs": self.underlapping_runs,
            "switches_of_play": self.switches_of_play,
            "through_balls": self.through_balls,
            "key_passes": self.key_passes
        }


# ============================================================================
# XG CALCULATOR
# ============================================================================

class XGCalculator:
    """Expected Goals calculator based on shot characteristics"""

    # Base xG values by zone
    BASE_XG = {
        "six_yard": 0.65,
        "penalty_box": 0.15,
        "outside_box": 0.04,
    }

    # Modifiers
    HEADER_MODIFIER = 0.75
    FIRST_TOUCH_MODIFIER = 1.2
    FAST_BREAK_MODIFIER = 1.3
    DEFENDER_MODIFIER = 0.85  # Per defender in path
    ANGLE_DECAY = 0.02  # Per degree from center

    @classmethod
    def calculate(cls, shot: Shot, is_fast_break: bool = False) -> float:
        """Calculate xG for a shot"""
        # Get goal center position
        goal_center = Position(PITCH_LENGTH, PITCH_WIDTH / 2)

        # Distance from goal
        distance = shot.position.distance_to(goal_center)

        # Angle to goal (in degrees)
        angle = cls._calculate_angle(shot.position)

        # Determine zone
        if distance <= 6:
            base_xg = cls.BASE_XG["six_yard"]
        elif cls._is_in_box(shot.position):
            base_xg = cls.BASE_XG["penalty_box"]
        else:
            base_xg = cls.BASE_XG["outside_box"]

        # Distance decay
        distance_factor = max(0.1, 1 - (distance / 40) * 0.5)

        # Angle factor
        angle_factor = max(0.3, 1 - abs(90 - angle) * cls.ANGLE_DECAY)

        xg = base_xg * distance_factor * angle_factor

        # Apply modifiers
        if shot.is_header:
            xg *= cls.HEADER_MODIFIER
        if shot.is_first_touch:
            xg *= cls.FIRST_TOUCH_MODIFIER
        if is_fast_break:
            xg *= cls.FAST_BREAK_MODIFIER

        # Defender penalty
        for _ in range(shot.defenders_in_path):
            xg *= cls.DEFENDER_MODIFIER

        return min(0.99, max(0.01, xg))

    @classmethod
    def _calculate_angle(cls, position: Position) -> float:
        """Calculate angle to goal from position"""
        # Goal posts positions
        left_post = Position(PITCH_LENGTH, (PITCH_WIDTH - GOAL_WIDTH) / 2)
        right_post = Position(PITCH_LENGTH, (PITCH_WIDTH + GOAL_WIDTH) / 2)

        # Vectors to posts
        v1 = (left_post.x - position.x, left_post.y - position.y)
        v2 = (right_post.x - position.x, right_post.y - position.y)

        # Angle between vectors
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 * mag2 == 0:
            return 0

        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    @classmethod
    def _is_in_box(cls, position: Position) -> bool:
        """Check if position is inside the penalty box"""
        box_left = (PITCH_WIDTH - BOX_WIDTH) / 2
        box_right = (PITCH_WIDTH + BOX_WIDTH) / 2
        box_start = PITCH_LENGTH - BOX_LENGTH

        return (position.x >= box_start and
                box_left <= position.y <= box_right)


# ============================================================================
# GOALIE ANALYTICS
# ============================================================================

class GoalieAnalytics:
    """Goalkeeper position and performance analysis"""

    def __init__(self):
        self.position_history: Dict[int, List[Position]] = {}
        self.saves_data: Dict[int, List[dict]] = {}

    def analyze_positioning(self, goalie: Player, ball_position: Position,
                           attacking_players: List[Player]) -> GoalieMetrics:
        """Analyze goalkeeper positioning quality"""
        # Goal center
        goal_center = Position(0 if goalie.team == "home" else PITCH_LENGTH, PITCH_WIDTH / 2)

        # Calculate optimal position (bisecting the angle)
        optimal_pos = self._calculate_optimal_position(goal_center, ball_position)

        # Distance from optimal
        dist_from_optimal = goalie.position.distance_to(optimal_pos)
        positioning_quality = max(0, 1 - dist_from_optimal / 5)

        # Coverage calculation
        coverage = self._calculate_coverage(goalie.position, ball_position, goal_center)

        # Distance from goal line
        if goalie.team == "home":
            dist_from_line = goalie.position.x
        else:
            dist_from_line = PITCH_LENGTH - goalie.position.x

        return GoalieMetrics(
            player_id=goalie.player_id,
            team=goalie.team,
            position=goalie.position,
            coverage_area=coverage,
            positioning_quality=positioning_quality,
            reaction_time=0.0,
            distance_from_line=dist_from_line
        )

    def _calculate_optimal_position(self, goal_center: Position,
                                   ball_position: Position) -> Position:
        """Calculate optimal goalkeeper position"""
        # Simple bisector calculation
        dx = ball_position.x - goal_center.x
        dy = ball_position.y - goal_center.y
        dist = math.sqrt(dx**2 + dy**2)

        if dist == 0:
            return goal_center

        # Position 3-5m off the line towards the ball
        offset = min(5, dist * 0.15)
        return Position(
            goal_center.x + (dx / dist) * offset,
            goal_center.y + (dy / dist) * offset
        )

    def _calculate_coverage(self, goalie_pos: Position, ball_pos: Position,
                           goal_center: Position) -> float:
        """Calculate percentage of goal covered by goalkeeper"""
        # Simplified coverage model
        dist_to_ball = goalie_pos.distance_to(ball_pos)
        dist_from_line = abs(goalie_pos.x - goal_center.x)

        # Coverage decreases with distance from line
        base_coverage = 60  # Base coverage at line
        coverage = base_coverage + (dist_from_line * 3)  # Gain coverage by coming out
        coverage = min(95, coverage)  # Cap at 95%

        return coverage


# ============================================================================
# PLAYER ANALYTICS
# ============================================================================

class PlayerAnalytics:
    """Individual player performance tracking"""

    SPRINT_THRESHOLD = 7.0  # m/s (25.2 km/h)
    HIGH_INTENSITY_THRESHOLD = 5.5  # m/s (19.8 km/h)

    def __init__(self):
        self.player_history: Dict[int, deque] = {}
        self.distance_tracking: Dict[int, float] = {}
        self.sprint_tracking: Dict[int, List[float]] = {}
        self.heatmaps: Dict[int, np.ndarray] = {}

    def update_player(self, player: Player, dt: float) -> Player:
        """Update player metrics with new position data"""
        player_id = player.player_id

        # Initialize history if needed
        if player_id not in self.player_history:
            self.player_history[player_id] = deque(maxlen=100)
            self.distance_tracking[player_id] = 0.0
            self.sprint_tracking[player_id] = []
            self.heatmaps[player_id] = np.zeros((PITCH_LENGTH_BINS, PITCH_WIDTH_BINS))

        history = self.player_history[player_id]

        # Calculate velocity and acceleration
        if len(history) > 0:
            last_pos = history[-1]
            distance = player.position.distance_to(last_pos)
            velocity = distance / dt if dt > 0 else 0

            # Update player metrics
            player.velocity = velocity
            player.distance_covered = self.distance_tracking[player_id] + distance
            self.distance_tracking[player_id] = player.distance_covered

            # Check for sprints
            if velocity >= self.SPRINT_THRESHOLD:
                player.sprints += 1
            elif velocity >= self.HIGH_INTENSITY_THRESHOLD:
                player.high_intensity_runs += 1

            # Acceleration
            if len(history) > 1:
                prev_velocity = history[-1].distance_to(history[-2]) / dt if dt > 0 else 0
                player.acceleration = (velocity - prev_velocity) / dt if dt > 0 else 0

        # Update history
        history.append(player.position)

        # Update heatmap
        self._update_heatmap(player_id, player.position)

        return player

    def _update_heatmap(self, player_id: int, position: Position):
        """Update player position heatmap"""
        x_bin = min(int(position.x / PITCH_LENGTH * PITCH_LENGTH_BINS), PITCH_LENGTH_BINS - 1)
        y_bin = min(int(position.y / PITCH_WIDTH * PITCH_WIDTH_BINS), PITCH_WIDTH_BINS - 1)
        self.heatmaps[player_id][x_bin, y_bin] += 1

    def get_player_heatmap(self, player_id: int) -> np.ndarray:
        """Get normalized heatmap for player"""
        if player_id not in self.heatmaps:
            return np.zeros((PITCH_LENGTH_BINS, PITCH_WIDTH_BINS))
        heatmap = self.heatmaps[player_id]
        if heatmap.max() > 0:
            return heatmap / heatmap.max()
        return heatmap

    def calculate_fatigue(self, player: Player, match_time: float) -> FatigueMetrics:
        """Calculate player fatigue metrics"""
        player_id = player.player_id

        if player_id not in self.sprint_tracking:
            return FatigueMetrics(
                player_id=player_id,
                fatigue_level=0.0,
                avg_speed_decline=0.0,
                sprint_decline=0.0,
                recovery_time_increase=0.0,
                distance_last_5min=0.0,
                distance_first_5min=0.0
            )

        # Simple fatigue model based on distance and time
        distance = self.distance_tracking.get(player_id, 0)
        expected_distance = match_time * 0.12  # ~7.2 km/h average

        fatigue_level = min(1.0, distance / (expected_distance * 1.5) if expected_distance > 0 else 0)

        # Speed decline (simplified)
        avg_speed_decline = min(30, fatigue_level * 25)
        sprint_decline = min(50, fatigue_level * 40)

        return FatigueMetrics(
            player_id=player_id,
            fatigue_level=fatigue_level,
            avg_speed_decline=avg_speed_decline,
            sprint_decline=sprint_decline,
            recovery_time_increase=fatigue_level * 20,
            distance_last_5min=distance * 0.1,
            distance_first_5min=distance * 0.15
        )


# Heatmap resolution
PITCH_LENGTH_BINS = 105
PITCH_WIDTH_BINS = 68


# ============================================================================
# PRESSURE ANALYTICS
# ============================================================================

class PressureAnalytics:
    """Team pressing and pressure analysis"""

    def __init__(self):
        self.defensive_actions: Dict[str, List[float]] = {"home": [], "away": []}
        self.passes_allowed: Dict[str, List[int]] = {"home": [], "away": []}
        self.regain_times: Dict[str, List[float]] = {"home": [], "away": []}

    def analyze_press(self, team: str, players: List[Player],
                     opponent_players: List[Player],
                     ball_position: Position) -> PressureMetrics:
        """Analyze team pressing intensity and effectiveness"""
        # Count players in pressing positions
        pressing_players = 0
        for player in players:
            dist_to_ball = player.position.distance_to(ball_position)
            if dist_to_ball < 10:  # Within 10m of ball
                pressing_players += 1

        # Calculate PPDA (simplified)
        passes = len(self.passes_allowed.get(team, [1]))
        actions = len(self.defensive_actions.get(team, [1]))
        ppda = passes / max(1, actions)

        # High press success rate
        regain_times = self.regain_times.get(team, [])
        quick_regains = len([t for t in regain_times if t < 5])  # Under 5 seconds
        success_rate = quick_regains / max(1, len(regain_times)) * 100

        # Counter press intensity
        counter_intensity = pressing_players / 4  # Normalize to ~1.0

        return PressureMetrics(
            team=team,
            ppda=ppda,
            high_press_success_rate=success_rate,
            counter_press_intensity=min(1.5, counter_intensity),
            pressing_triggers=len(self.defensive_actions.get(team, [])),
            regains_in_final_third=quick_regains,
            time_to_regain=np.mean(regain_times) if regain_times else 0
        )


# ============================================================================
# PATTERN ANALYTICS
# ============================================================================

class PatternAnalytics:
    """Play pattern and tactical analysis"""

    def __init__(self):
        self.pass_sequences: Dict[str, List[List[Position]]] = {"home": [], "away": []}
        self.attack_widths: Dict[str, List[float]] = {"home": [], "away": []}

    def analyze_patterns(self, team: str, players: List[Player],
                        recent_passes: List[Position]) -> PatternMetrics:
        """Analyze team play patterns"""
        # Calculate attacking width
        y_positions = [p.position.y for p in players if
                      (team == "home" and p.position.x > PITCH_LENGTH * 0.5) or
                      (team == "away" and p.position.x < PITCH_LENGTH * 0.5)]

        attacking_width = max(y_positions) - min(y_positions) if len(y_positions) > 1 else 0

        # Determine build-up speed based on pass distances
        if recent_passes and len(recent_passes) > 1:
            avg_pass_dist = np.mean([
                recent_passes[i].distance_to(recent_passes[i+1])
                for i in range(len(recent_passes) - 1)
            ])
            if avg_pass_dist > 30:
                build_up = "direct"
            elif avg_pass_dist > 20:
                build_up = "fast"
            elif avg_pass_dist > 10:
                build_up = "medium"
            else:
                build_up = "slow"
        else:
            build_up = "medium"

        # Progression style
        if build_up in ["direct", "fast"]:
            progression = "long_balls"
        elif build_up == "slow":
            progression = "short_passing"
        else:
            progression = "mixed"

        return PatternMetrics(
            team=team,
            build_up_speed=build_up,
            attacking_width=attacking_width,
            progression_style=progression,
            third_man_runs=0,  # Would need event data
            overlapping_runs=0,
            underlapping_runs=0,
            switches_of_play=0,
            through_balls=0,
            key_passes=0
        )


# ============================================================================
# MAN ADVANTAGE DETECTOR
# ============================================================================

class ManAdvantageDetector:
    """Detect and track red card/man advantage situations"""

    def __init__(self):
        self.home_players = 11
        self.away_players = 11
        self.red_cards: List[dict] = []
        self.advantage_start_time: Optional[float] = None

    def update(self, home_player_count: int, away_player_count: int,
               match_time: float) -> ManAdvantageState:
        """Update player counts and detect man advantage"""
        # Detect red cards
        if home_player_count < self.home_players:
            self.red_cards.append({
                "team": "home",
                "time": match_time,
                "from": self.home_players,
                "to": home_player_count
            })
            if self.advantage_start_time is None:
                self.advantage_start_time = match_time

        if away_player_count < self.away_players:
            self.red_cards.append({
                "team": "away",
                "time": match_time,
                "from": self.away_players,
                "to": away_player_count
            })
            if self.advantage_start_time is None:
                self.advantage_start_time = match_time

        self.home_players = home_player_count
        self.away_players = away_player_count

        # Determine advantage
        is_active = home_player_count != away_player_count
        advantage_team = None
        if is_active:
            advantage_team = "home" if home_player_count > away_player_count else "away"

        time_remaining = match_time - self.advantage_start_time if self.advantage_start_time else 0

        return ManAdvantageState(
            is_active=is_active,
            advantage_team=advantage_team,
            home_players=home_player_count,
            away_players=away_player_count,
            time_remaining=time_remaining
        )


# ============================================================================
# MAIN ANALYTICS ENGINE
# ============================================================================

class SoccerAnalyticsEngine:
    """Main analytics engine combining all modules"""

    def __init__(self):
        self.xg_calculator = XGCalculator()
        self.goalie_analytics = GoalieAnalytics()
        self.player_analytics = PlayerAnalytics()
        self.pressure_analytics = PressureAnalytics()
        self.pattern_analytics = PatternAnalytics()
        self.man_advantage_detector = ManAdvantageDetector()

        # State tracking
        self.shots: List[Shot] = []
        self.home_xg = 0.0
        self.away_xg = 0.0
        self.xg_timeline: List[dict] = []
        self.current_frame = 0
        self.match_time = 0.0
        self.ball_position = Position(PITCH_LENGTH / 2, PITCH_WIDTH / 2)

        # Team states
        self.home_state: Optional[TeamState] = None
        self.away_state: Optional[TeamState] = None

    def process_frame(self, frame_data: dict) -> dict:
        """Process a single frame of tracking data"""
        self.current_frame = frame_data.get("frame", self.current_frame + 1)
        self.match_time = frame_data.get("time", self.match_time + 1/30)  # 30 fps default

        # Extract players
        home_players = [Player(**p) if isinstance(p, dict) else p
                       for p in frame_data.get("home_players", [])]
        away_players = [Player(**p) if isinstance(p, dict) else p
                       for p in frame_data.get("away_players", [])]

        # Update ball position
        if "ball" in frame_data:
            ball = frame_data["ball"]
            self.ball_position = Position(ball["x"], ball["y"])

        # Update player analytics
        dt = 1/30  # Assuming 30 fps
        for player in home_players + away_players:
            self.player_analytics.update_player(player, dt)

        # Check for shots
        if "shot" in frame_data:
            shot_data = frame_data["shot"]
            shot = self._create_shot(shot_data)
            self.shots.append(shot)

            if shot.team == "home":
                self.home_xg += shot.xg
            else:
                self.away_xg += shot.xg

            self.xg_timeline.append({
                "time": self.match_time,
                "home_xg": self.home_xg,
                "away_xg": self.away_xg
            })

        # Update man advantage
        man_advantage = self.man_advantage_detector.update(
            len(home_players), len(away_players), self.match_time
        )

        # Goalie analysis
        home_goalie = next((p for p in home_players if p.jersey_number == 1), None)
        away_goalie = next((p for p in away_players if p.jersey_number == 1), None)

        home_goalie_metrics = None
        away_goalie_metrics = None

        if home_goalie:
            home_goalie_metrics = self.goalie_analytics.analyze_positioning(
                home_goalie, self.ball_position, away_players
            )
        if away_goalie:
            away_goalie_metrics = self.goalie_analytics.analyze_positioning(
                away_goalie, self.ball_position, home_players
            )

        # Pressure analysis
        home_pressure = self.pressure_analytics.analyze_press(
            "home", home_players, away_players, self.ball_position
        )
        away_pressure = self.pressure_analytics.analyze_press(
            "away", away_players, home_players, self.ball_position
        )

        # Pattern analysis
        home_patterns = self.pattern_analytics.analyze_patterns("home", home_players, [])
        away_patterns = self.pattern_analytics.analyze_patterns("away", away_players, [])

        # Fatigue metrics for key players
        fatigue_metrics = {}
        for player in home_players[:5] + away_players[:5]:  # Top 5 each team
            fatigue_metrics[player.player_id] = self.player_analytics.calculate_fatigue(
                player, self.match_time
            )

        return {
            "frame": self.current_frame,
            "time": self.match_time,
            "ball": self.ball_position.to_dict(),
            "home_players": [p.to_dict() for p in home_players],
            "away_players": [p.to_dict() for p in away_players],
            "xg": {
                "home": round(self.home_xg, 3),
                "away": round(self.away_xg, 3)
            },
            "shots": [s.to_dict() for s in self.shots[-10:]],  # Last 10 shots
            "xg_timeline": self.xg_timeline[-100:],  # Last 100 points
            "man_advantage": man_advantage.to_dict(),
            "home_goalie": home_goalie_metrics.to_dict() if home_goalie_metrics else None,
            "away_goalie": away_goalie_metrics.to_dict() if away_goalie_metrics else None,
            "pressure": {
                "home": home_pressure.to_dict(),
                "away": away_pressure.to_dict()
            },
            "patterns": {
                "home": home_patterns.to_dict(),
                "away": away_patterns.to_dict()
            },
            "fatigue": {str(k): v.to_dict() for k, v in fatigue_metrics.items()}
        }

    def _create_shot(self, shot_data: dict) -> Shot:
        """Create a shot object from raw data"""
        position = Position(shot_data["x"], shot_data["y"])
        target = Position(
            shot_data.get("target_x", PITCH_LENGTH),
            shot_data.get("target_y", PITCH_WIDTH / 2)
        )

        # Calculate distance and angle
        goal_center = Position(PITCH_LENGTH, PITCH_WIDTH / 2)
        distance = position.distance_to(goal_center)
        angle = XGCalculator._calculate_angle(position)

        shot = Shot(
            shot_id=len(self.shots) + 1,
            player_id=shot_data.get("player_id", 0),
            team=shot_data.get("team", "home"),
            position=position,
            target_position=target,
            result=ShotResult(shot_data.get("result", "off_target")),
            xg=0,  # Will be calculated
            timestamp=self.match_time,
            distance=distance,
            angle=angle,
            body_part=shot_data.get("body_part", "foot"),
            is_header=shot_data.get("is_header", False),
            is_first_touch=shot_data.get("is_first_touch", False),
            defenders_in_path=shot_data.get("defenders", 0)
        )

        shot.xg = XGCalculator.calculate(shot)
        return shot

    def get_full_state(self) -> dict:
        """Get complete analytics state"""
        return {
            "frame": self.current_frame,
            "time": self.match_time,
            "xg": {
                "home": round(self.home_xg, 3),
                "away": round(self.away_xg, 3)
            },
            "shots": [s.to_dict() for s in self.shots],
            "xg_timeline": self.xg_timeline,
            "total_shots": len(self.shots)
        }


# ============================================================================
# VIDEO QUERY INTERFACE
# ============================================================================

class VideoQueryInterface:
    """Interface for video playback and frame-by-frame analysis"""

    def __init__(self, video_path: Optional[str] = None):
        self.video_path = video_path
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        self.is_playing = False
        self.playback_speed = 1.0
        self.frame_cache: Dict[int, dict] = {}
        self.analytics_engine = SoccerAnalyticsEngine()

        # Query filters
        self.filters: Dict[str, any] = {}

    def load_video(self, path: str) -> dict:
        """Load video file for analysis"""
        self.video_path = path
        # In real implementation, would use cv2.VideoCapture
        return {
            "status": "loaded",
            "path": path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": self.total_frames / self.fps if self.fps > 0 else 0
        }

    def seek(self, frame: int) -> dict:
        """Seek to specific frame"""
        self.current_frame = max(0, min(frame, self.total_frames - 1))
        return self.get_frame(self.current_frame)

    def seek_time(self, time_seconds: float) -> dict:
        """Seek to specific time"""
        frame = int(time_seconds * self.fps)
        return self.seek(frame)

    def next_frame(self) -> dict:
        """Advance to next frame"""
        return self.seek(self.current_frame + 1)

    def prev_frame(self) -> dict:
        """Go to previous frame"""
        return self.seek(self.current_frame - 1)

    def skip_frames(self, count: int) -> dict:
        """Skip forward/backward by count frames"""
        return self.seek(self.current_frame + count)

    def get_frame(self, frame: int) -> dict:
        """Get frame data and analytics"""
        # Check cache
        if frame in self.frame_cache:
            return self.frame_cache[frame]

        # In real implementation, would extract frame from video
        frame_data = {
            "frame": frame,
            "time": frame / self.fps,
            "image": None,  # Would be base64 encoded image
            "analytics": {}
        }

        self.frame_cache[frame] = frame_data
        return frame_data

    def set_playback_speed(self, speed: float) -> dict:
        """Set playback speed multiplier"""
        self.playback_speed = max(0.1, min(8.0, speed))
        return {"speed": self.playback_speed}

    def play(self) -> dict:
        """Start playback"""
        self.is_playing = True
        return {"status": "playing"}

    def pause(self) -> dict:
        """Pause playback"""
        self.is_playing = False
        return {"status": "paused"}

    def toggle_play(self) -> dict:
        """Toggle play/pause"""
        if self.is_playing:
            return self.pause()
        return self.play()

    # Query methods
    def query_shots(self, team: Optional[str] = None,
                   result: Optional[ShotResult] = None,
                   min_xg: float = 0.0) -> List[dict]:
        """Query shots with filters"""
        shots = self.analytics_engine.shots

        if team:
            shots = [s for s in shots if s.team == team]
        if result:
            shots = [s for s in shots if s.result == result]
        if min_xg > 0:
            shots = [s for s in shots if s.xg >= min_xg]

        return [s.to_dict() for s in shots]

    def query_player_events(self, player_id: int) -> List[dict]:
        """Query all events for a specific player"""
        events = []
        for shot in self.analytics_engine.shots:
            if shot.player_id == player_id:
                events.append({
                    "type": "shot",
                    "time": shot.timestamp,
                    "frame": int(shot.timestamp * self.fps),
                    "data": shot.to_dict()
                })
        return events

    def query_high_xg_moments(self, threshold: float = 0.3) -> List[dict]:
        """Find high xG moments in the match"""
        moments = []
        for shot in self.analytics_engine.shots:
            if shot.xg >= threshold:
                moments.append({
                    "time": shot.timestamp,
                    "frame": int(shot.timestamp * self.fps),
                    "xg": shot.xg,
                    "team": shot.team,
                    "result": shot.result.value
                })
        return sorted(moments, key=lambda x: x["xg"], reverse=True)

    def export_timeline(self) -> dict:
        """Export complete match timeline"""
        return {
            "xg_timeline": self.analytics_engine.xg_timeline,
            "shots": [s.to_dict() for s in self.analytics_engine.shots],
            "final_xg": {
                "home": self.analytics_engine.home_xg,
                "away": self.analytics_engine.away_xg
            }
        }


if __name__ == "__main__":
    # Test the analytics engine
    engine = SoccerAnalyticsEngine()

    # Simulate some frame data
    test_frame = {
        "frame": 1,
        "time": 0.033,
        "ball": {"x": 80, "y": 34},
        "home_players": [
            {"player_id": 1, "team": "home", "position": {"x": 5, "y": 34, "timestamp": 0}, "jersey_number": 1},
            {"player_id": 2, "team": "home", "position": {"x": 30, "y": 20, "timestamp": 0}},
        ],
        "away_players": [
            {"player_id": 12, "team": "away", "position": {"x": 100, "y": 34, "timestamp": 0}, "jersey_number": 1},
            {"player_id": 13, "team": "away", "position": {"x": 75, "y": 40, "timestamp": 0}},
        ],
        "shot": {
            "x": 85,
            "y": 36,
            "team": "home",
            "player_id": 10,
            "result": "saved"
        }
    }

    result = engine.process_frame(test_frame)
    print(json.dumps(result, indent=2))
