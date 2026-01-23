"""
Interactive Play Designer (Bhostgusters-inspired)

Implements methodology from:
- Seidl, T., et al. (2018). "Bhostgusters: Real-time Interactive Play
  Sketching with Adaptive Agents." MIT Sloan Sports Analytics Conference.

Key concept: Allow coaches to sketch plays and have AI simulate
how defenders would react, providing immediate feedback.

Hockey translation:
- Interactive breakout play design
- Power play setup visualization
- Forechecking system design
- Defensive structure planning
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from datetime import datetime


class PlayType(Enum):
    """Types of hockey plays."""
    BREAKOUT = "breakout"
    ZONE_ENTRY = "zone_entry"
    POWER_PLAY = "power_play"
    PENALTY_KILL = "penalty_kill"
    FORECHECK = "forecheck"
    DEFENSIVE_ZONE = "defensive_zone"
    NEUTRAL_ZONE = "neutral_zone"
    ODD_MAN_RUSH = "odd_man_rush"


class ActionType(Enum):
    """Types of player actions in plays."""
    SKATE = "skate"
    PASS = "pass"
    SHOOT = "shoot"
    CARRY = "carry"
    DUMP = "dump"
    SCREEN = "screen"
    SUPPORT = "support"
    PRESSURE = "pressure"
    BLOCK = "block"


class PlayerSymbol(Enum):
    """Symbols for play diagrams."""
    FORWARD = "F"
    DEFENSEMAN = "D"
    CENTER = "C"
    WINGER = "W"
    GOALIE = "G"
    PUCK = "P"


@dataclass
class Waypoint:
    """Single point in a player's path."""
    x: float
    y: float
    time: float  # seconds into play
    action: ActionType = ActionType.SKATE


@dataclass
class PlayerPath:
    """Complete path for a player in a play."""
    player_id: str
    symbol: PlayerSymbol
    waypoints: List[Waypoint]
    is_puck_carrier: bool = False
    team: str = "offense"  # offense or defense

    @property
    def start_position(self) -> Tuple[float, float]:
        if self.waypoints:
            return (self.waypoints[0].x, self.waypoints[0].y)
        return (0, 0)

    @property
    def end_position(self) -> Tuple[float, float]:
        if self.waypoints:
            return (self.waypoints[-1].x, self.waypoints[-1].y)
        return (0, 0)

    @property
    def duration(self) -> float:
        if self.waypoints:
            return self.waypoints[-1].time - self.waypoints[0].time
        return 0


@dataclass
class PassEvent:
    """Pass between two players."""
    time: float
    from_player: str
    to_player: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    pass_type: str = "tape_to_tape"  # tape_to_tape, saucer, bank, dump


@dataclass
class PlayDiagram:
    """Complete play diagram."""
    name: str
    play_type: PlayType
    offensive_paths: List[PlayerPath]
    defensive_paths: List[PlayerPath]
    passes: List[PassEvent]
    puck_start: Tuple[float, float]
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result of play simulation."""
    success_probability: float
    shot_probability: float
    turnover_probability: float
    time_to_shot: Optional[float]
    defensive_coverage: float  # 0-1, how well defense covers
    key_moments: List[Tuple[float, str]]  # (time, description)
    recommendations: List[str]


class RinkCanvas:
    """
    Virtual rink for play design.

    Provides coordinate system and zones.
    """

    def __init__(
        self,
        length: float = 200,  # feet
        width: float = 85
    ):
        self.length = length
        self.width = width

        # Key locations (in feet from center ice)
        self.offensive_zone_start = 25  # Blue line
        self.defensive_zone_start = -25
        self.goal_line = length / 2 - 11  # 11 feet from end boards

        # Faceoff circles
        self.faceoff_circles = [
            (69, 22),   # Offensive right
            (69, -22),  # Offensive left
            (-69, 22),  # Defensive right
            (-69, -22), # Defensive left
            (20, 22),   # Neutral right
            (20, -22),  # Neutral left
            (-20, 22),  # Neutral right (opp)
            (-20, -22), # Neutral left (opp)
            (0, 0),     # Center ice
        ]

    def in_offensive_zone(self, x: float) -> bool:
        return x > self.offensive_zone_start

    def in_defensive_zone(self, x: float) -> bool:
        return x < self.defensive_zone_start

    def in_neutral_zone(self, x: float) -> bool:
        return self.defensive_zone_start <= x <= self.offensive_zone_start

    def in_slot(self, x: float, y: float) -> bool:
        """Check if position is in the slot (high danger)."""
        goal_x = self.goal_line
        return x > goal_x - 20 and abs(y) < 15

    def get_zone(self, x: float) -> str:
        if self.in_offensive_zone(x):
            return "offensive"
        elif self.in_defensive_zone(x):
            return "defensive"
        return "neutral"


class PathInterpolator:
    """
    Interpolate player paths between waypoints.
    """

    def __init__(self, max_speed: float = 30):  # feet/second
        self.max_speed = max_speed

    def interpolate(
        self,
        path: PlayerPath,
        time_step: float = 0.1
    ) -> List[Tuple[float, float, float]]:
        """
        Interpolate path to regular time intervals.

        Returns: List of (x, y, time) tuples
        """
        if len(path.waypoints) < 2:
            return [(path.waypoints[0].x, path.waypoints[0].y, 0)] if path.waypoints else []

        interpolated = []
        start_time = path.waypoints[0].time
        end_time = path.waypoints[-1].time

        current_time = start_time
        while current_time <= end_time:
            pos = self._position_at_time(path.waypoints, current_time)
            interpolated.append((pos[0], pos[1], current_time))
            current_time += time_step

        return interpolated

    def _position_at_time(
        self,
        waypoints: List[Waypoint],
        time: float
    ) -> Tuple[float, float]:
        """Get position at specific time."""
        if time <= waypoints[0].time:
            return (waypoints[0].x, waypoints[0].y)
        if time >= waypoints[-1].time:
            return (waypoints[-1].x, waypoints[-1].y)

        # Find surrounding waypoints
        for i in range(len(waypoints) - 1):
            if waypoints[i].time <= time <= waypoints[i+1].time:
                # Linear interpolation
                t = (time - waypoints[i].time) / (waypoints[i+1].time - waypoints[i].time)
                x = waypoints[i].x + t * (waypoints[i+1].x - waypoints[i].x)
                y = waypoints[i].y + t * (waypoints[i+1].y - waypoints[i].y)
                return (x, y)

        return (waypoints[-1].x, waypoints[-1].y)


class DefensiveReactor:
    """
    AI model for defensive reactions.

    Simulates how defenders would react to offensive movements.
    """

    def __init__(
        self,
        reaction_time: float = 0.3,  # seconds
        max_speed: float = 28  # feet/second
    ):
        self.reaction_time = reaction_time
        self.max_speed = max_speed

    def generate_reaction(
        self,
        defender_start: Tuple[float, float],
        offensive_paths: List[PlayerPath],
        passes: List[PassEvent],
        puck_start: Tuple[float, float],
        duration: float
    ) -> PlayerPath:
        """
        Generate defensive reaction path.

        Defender tracks puck and nearest threat.
        """
        waypoints = [Waypoint(defender_start[0], defender_start[1], 0)]

        # Simulate at 0.5 second intervals
        time_step = 0.5
        current_pos = np.array(defender_start)
        current_time = 0

        while current_time < duration:
            # Find puck position at this time
            puck_pos = self._get_puck_position(
                current_time, puck_start, passes, offensive_paths
            )

            # Find nearest offensive threat
            threats = []
            for path in offensive_paths:
                if path.team == "offense":
                    pos = self._get_position_at_time(path, current_time)
                    if pos:
                        threats.append(np.array(pos))

            # Determine target position
            target = self._determine_target(
                current_pos, puck_pos, threats
            )

            # Move toward target (with reaction delay)
            if current_time > self.reaction_time:
                direction = target - current_pos
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    move_dist = min(self.max_speed * time_step, dist)
                    current_pos = current_pos + (direction / dist) * move_dist

            current_time += time_step
            waypoints.append(Waypoint(
                current_pos[0], current_pos[1], current_time,
                ActionType.PRESSURE if np.linalg.norm(puck_pos - current_pos) < 10 else ActionType.SKATE
            ))

        return PlayerPath(
            player_id="defender_reaction",
            symbol=PlayerSymbol.DEFENSEMAN,
            waypoints=waypoints,
            team="defense"
        )

    def _get_puck_position(
        self,
        time: float,
        start: Tuple[float, float],
        passes: List[PassEvent],
        paths: List[PlayerPath]
    ) -> np.ndarray:
        """Get puck position at given time."""
        # Check passes
        for p in passes:
            if p.time <= time:
                # Puck is with receiver after pass
                return np.array([p.end_x, p.end_y])

        # Otherwise with carrier
        for path in paths:
            if path.is_puck_carrier:
                pos = self._get_position_at_time(path, time)
                if pos:
                    return np.array(pos)

        return np.array(start)

    def _get_position_at_time(
        self,
        path: PlayerPath,
        time: float
    ) -> Optional[Tuple[float, float]]:
        """Get player position at time."""
        interpolator = PathInterpolator()
        return interpolator._position_at_time(path.waypoints, time)

    def _determine_target(
        self,
        current: np.ndarray,
        puck: np.ndarray,
        threats: List[np.ndarray]
    ) -> np.ndarray:
        """Determine optimal defensive position."""
        # Basic: position between puck and goal
        goal = np.array([-89, 0])  # Defensive goal

        puck_to_goal = goal - puck
        dist = np.linalg.norm(puck_to_goal)
        if dist > 0:
            # Position 1/3 of way from puck to goal
            target = puck + puck_to_goal * 0.33
        else:
            target = puck

        # Adjust for nearest threat
        if threats:
            nearest = min(threats, key=lambda t: np.linalg.norm(t - current))
            # Slight bias toward nearest threat
            target = 0.7 * target + 0.3 * nearest

        return target


class PlaySimulator:
    """
    Simulate play execution with defensive reactions.
    """

    def __init__(self):
        self.canvas = RinkCanvas()
        self.interpolator = PathInterpolator()
        self.reactor = DefensiveReactor()

    def simulate(
        self,
        diagram: PlayDiagram,
        n_simulations: int = 100
    ) -> SimulationResult:
        """
        Simulate play multiple times with randomness.

        Returns aggregate statistics.
        """
        successes = 0
        shots = 0
        turnovers = 0
        shot_times = []
        coverage_scores = []
        key_moments = []

        for _ in range(n_simulations):
            result = self._simulate_once(diagram)
            if result['success']:
                successes += 1
            if result['shot']:
                shots += 1
                shot_times.append(result['shot_time'])
            if result['turnover']:
                turnovers += 1
            coverage_scores.append(result['coverage'])

        success_prob = successes / n_simulations
        shot_prob = shots / n_simulations
        turnover_prob = turnovers / n_simulations
        avg_shot_time = np.mean(shot_times) if shot_times else None
        avg_coverage = np.mean(coverage_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            success_prob, shot_prob, turnover_prob, avg_coverage
        )

        return SimulationResult(
            success_probability=success_prob,
            shot_probability=shot_prob,
            turnover_probability=turnover_prob,
            time_to_shot=avg_shot_time,
            defensive_coverage=avg_coverage,
            key_moments=key_moments,
            recommendations=recommendations
        )

    def _simulate_once(
        self,
        diagram: PlayDiagram
    ) -> Dict:
        """Run single simulation."""
        # Add noise to execution
        noise = 0.1  # 10% execution variance

        # Check pass success
        for pass_event in diagram.passes:
            # Pass success based on distance and coverage
            pass_dist = np.sqrt(
                (pass_event.end_x - pass_event.start_x)**2 +
                (pass_event.end_y - pass_event.start_y)**2
            )
            base_success = 0.95 - pass_dist * 0.005
            if np.random.random() > base_success * (1 - noise * np.random.randn()):
                return {
                    'success': False, 'shot': False, 'turnover': True,
                    'shot_time': None, 'coverage': 0.5
                }

        # Check if play reaches shot
        final_positions = []
        for path in diagram.offensive_paths:
            if path.waypoints:
                final_positions.append(path.end_position)

        # Shot if anyone in slot
        shot = False
        shot_time = None
        for pos in final_positions:
            if self.canvas.in_slot(pos[0], pos[1]):
                shot = True
                shot_time = diagram.offensive_paths[0].duration if diagram.offensive_paths else 3.0
                break

        # Coverage score
        coverage = self._calculate_coverage(diagram)

        return {
            'success': shot,
            'shot': shot,
            'turnover': False,
            'shot_time': shot_time,
            'coverage': coverage
        }

    def _calculate_coverage(self, diagram: PlayDiagram) -> float:
        """Calculate how well defense covers the play."""
        if not diagram.defensive_paths:
            return 0.0

        # Check each offensive player
        covered = 0
        total = 0

        for off_path in diagram.offensive_paths:
            if not off_path.waypoints:
                continue

            total += 1
            off_end = np.array(off_path.end_position)

            # Find nearest defender at end
            min_dist = float('inf')
            for def_path in diagram.defensive_paths:
                if def_path.waypoints:
                    def_end = np.array(def_path.end_position)
                    dist = np.linalg.norm(off_end - def_end)
                    min_dist = min(min_dist, dist)

            # Covered if defender within 8 feet
            if min_dist < 8:
                covered += 1

        return covered / total if total > 0 else 0.0

    def _generate_recommendations(
        self,
        success_prob: float,
        shot_prob: float,
        turnover_prob: float,
        coverage: float
    ) -> List[str]:
        """Generate tactical recommendations."""
        recs = []

        if success_prob < 0.5:
            recs.append("Play has low success rate - consider simplifying")

        if turnover_prob > 0.3:
            recs.append("High turnover risk - reduce cross-ice passes")

        if shot_prob < 0.6:
            recs.append("Low shot generation - add option to attack net")

        if coverage > 0.7:
            recs.append("Defense covers well - add misdirection")

        if not recs:
            recs.append("Play looks effective in simulation")

        return recs


class PlayLibrary:
    """
    Library of saved plays with search.
    """

    def __init__(self):
        self.plays: Dict[str, PlayDiagram] = {}
        self._create_standard_plays()

    def _create_standard_plays(self):
        """Create standard hockey plays."""
        # Basic breakout
        breakout = PlayDiagram(
            name="Standard Breakout Left",
            play_type=PlayType.BREAKOUT,
            offensive_paths=[
                PlayerPath(
                    player_id="LD",
                    symbol=PlayerSymbol.DEFENSEMAN,
                    waypoints=[
                        Waypoint(-80, -20, 0),
                        Waypoint(-70, -25, 1.0),
                        Waypoint(-50, -20, 2.0, ActionType.PASS)
                    ],
                    is_puck_carrier=True
                ),
                PlayerPath(
                    player_id="LW",
                    symbol=PlayerSymbol.WINGER,
                    waypoints=[
                        Waypoint(-60, -35, 0),
                        Waypoint(-40, -30, 1.5),
                        Waypoint(-20, -25, 2.5)
                    ]
                ),
                PlayerPath(
                    player_id="C",
                    symbol=PlayerSymbol.CENTER,
                    waypoints=[
                        Waypoint(-60, 0, 0),
                        Waypoint(-45, -5, 1.0),
                        Waypoint(-30, 0, 2.0)
                    ]
                )
            ],
            defensive_paths=[],
            passes=[
                PassEvent(2.0, "LD", "LW", -50, -20, -35, -28)
            ],
            puck_start=(-85, -15),
            description="D retrieves puck behind net, moves up boards, passes to winger",
            tags=["breakout", "boards", "safe"]
        )
        self.plays["breakout_left"] = breakout

        # Power play setup
        pp_umbrella = PlayDiagram(
            name="PP Umbrella",
            play_type=PlayType.POWER_PLAY,
            offensive_paths=[
                PlayerPath(
                    player_id="QB",
                    symbol=PlayerSymbol.DEFENSEMAN,
                    waypoints=[
                        Waypoint(55, 0, 0),
                        Waypoint(60, 0, 1.0)
                    ],
                    is_puck_carrier=True
                ),
                PlayerPath(
                    player_id="LF",
                    symbol=PlayerSymbol.FORWARD,
                    waypoints=[
                        Waypoint(65, -25, 0),
                        Waypoint(70, -20, 1.5)
                    ]
                ),
                PlayerPath(
                    player_id="RF",
                    symbol=PlayerSymbol.FORWARD,
                    waypoints=[
                        Waypoint(65, 25, 0),
                        Waypoint(70, 20, 1.5)
                    ]
                ),
                PlayerPath(
                    player_id="Net",
                    symbol=PlayerSymbol.FORWARD,
                    waypoints=[
                        Waypoint(82, 0, 0),
                        Waypoint(85, 5, 2.0, ActionType.SCREEN)
                    ]
                ),
                PlayerPath(
                    player_id="Bumper",
                    symbol=PlayerSymbol.FORWARD,
                    waypoints=[
                        Waypoint(72, 0, 0),
                        Waypoint(75, 0, 1.0)
                    ]
                )
            ],
            defensive_paths=[],
            passes=[],
            puck_start=(55, 0),
            description="1-3-1 umbrella power play setup",
            tags=["power_play", "umbrella", "setup"]
        )
        self.plays["pp_umbrella"] = pp_umbrella

    def add_play(self, play: PlayDiagram):
        """Add play to library."""
        self.plays[play.name] = play

    def search(
        self,
        play_type: Optional[PlayType] = None,
        tags: Optional[List[str]] = None
    ) -> List[PlayDiagram]:
        """Search plays by type or tags."""
        results = list(self.plays.values())

        if play_type:
            results = [p for p in results if p.play_type == play_type]

        if tags:
            results = [
                p for p in results
                if any(t in p.tags for t in tags)
            ]

        return results

    def get_similar(
        self,
        play: PlayDiagram,
        top_k: int = 5
    ) -> List[Tuple[PlayDiagram, float]]:
        """Find similar plays."""
        similarities = []

        for name, other in self.plays.items():
            if other.name == play.name:
                continue

            sim = self._calculate_similarity(play, other)
            similarities.append((other, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _calculate_similarity(
        self,
        play1: PlayDiagram,
        play2: PlayDiagram
    ) -> float:
        """Calculate similarity between plays."""
        score = 0.0

        # Same type
        if play1.play_type == play2.play_type:
            score += 0.3

        # Tag overlap
        if play1.tags and play2.tags:
            overlap = len(set(play1.tags) & set(play2.tags))
            score += 0.2 * overlap / max(len(play1.tags), len(play2.tags))

        # Path similarity
        path_sim = self._path_similarity(
            play1.offensive_paths,
            play2.offensive_paths
        )
        score += 0.5 * path_sim

        return score

    def _path_similarity(
        self,
        paths1: List[PlayerPath],
        paths2: List[PlayerPath]
    ) -> float:
        """Calculate similarity between path sets."""
        if not paths1 or not paths2:
            return 0.0

        # Compare start and end positions
        starts1 = [p.start_position for p in paths1]
        starts2 = [p.start_position for p in paths2]

        # Simple matching
        matches = 0
        for s1 in starts1:
            for s2 in starts2:
                dist = np.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)
                if dist < 15:  # Within 15 feet
                    matches += 1
                    break

        return matches / max(len(paths1), len(paths2))
