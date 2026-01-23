"""
Tactical Detection and Formation Analysis for Hockey

This module implements formation and tactical change detection adapted from
SoccerCPD and related research for hockey analytics.

Features:
- Defensive/offensive system classification
- Formation change-point detection
- Power play/penalty kill formation recognition
- Tactical adjustment detection during games

References:
    - Kim, H., et al. (2022). "SoccerCPD: Formation and Role Change-Point Detection
      in Soccer Matches Using Spatiotemporal Tracking Data." ACM SIGKDD.
    - "What Happens to Your Team's Formation During a Match? A Tactical Deep Dive." (2025)
    - "The Principles of Tactical Formation Identification in Association Football." Frontiers (2025)
    - Rein, R., & Memmert, D. (2016). "Big Data and Tactical Analysis in Elite Soccer." SpringerPlus.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class DefensiveSystem(Enum):
    """Defensive systems used in hockey."""
    # Even strength
    MAN_TO_MAN = "man_to_man"
    ZONE = "zone"
    HYBRID = "hybrid"
    TRAP = "trap"
    NEUTRAL_ZONE_TRAP = "nz_trap"
    LEFT_WING_LOCK = "lw_lock"

    # Forechecking
    FORECHECK_1_2_2 = "1-2-2"
    FORECHECK_2_1_2 = "2-1-2"
    FORECHECK_1_3_1 = "1-3-1"
    FORECHECK_2_3 = "2-3"
    AGGRESSIVE_FORECHECK = "aggressive"
    PASSIVE_FORECHECK = "passive"


class OffensiveSystem(Enum):
    """Offensive systems used in hockey."""
    CYCLE = "cycle"
    CRASH_NET = "crash_net"
    PERIMETER = "perimeter"
    QUICK_STRIKE = "quick_strike"
    OVERLOAD = "overload"
    UMBRELLA = "umbrella"  # PP


class PowerPlayFormation(Enum):
    """Power play formations."""
    UMBRELLA_1_3_1 = "umbrella"
    OVERLOAD = "overload"
    ONE_TWO_TWO = "1-2-2"
    ONE_THREE_ONE = "1-3-1"
    SPREAD = "spread"
    HALF_WALL = "half_wall"


class PenaltyKillFormation(Enum):
    """Penalty kill formations."""
    BOX = "box"
    DIAMOND = "diamond"
    TRIANGLE_PLUS_ONE = "triangle_plus_one"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


class GamePhase(Enum):
    """Phases of play."""
    OFFENSIVE_ZONE = "offensive_zone"
    DEFENSIVE_ZONE = "defensive_zone"
    NEUTRAL_ZONE = "neutral_zone"
    TRANSITION_ATTACK = "transition_attack"
    TRANSITION_DEFENSE = "transition_defense"
    FORECHECK = "forecheck"
    BREAKOUT = "breakout"


@dataclass
class PlayerPosition:
    """Player position at a moment in time."""
    player_id: str
    x: float  # -100 to 100 (goal to goal)
    y: float  # -42.5 to 42.5 (boards to boards)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    has_puck: bool = False


@dataclass
class TeamSnapshot:
    """Snapshot of team positions at a moment."""
    timestamp: float
    players: List[PlayerPosition]
    goalie_position: Optional[PlayerPosition] = None
    puck_x: float = 0.0
    puck_y: float = 0.0
    strength_state: str = "5v5"


@dataclass
class FormationTemplate:
    """Template for a known formation."""
    name: str
    positions: List[Tuple[float, float]]  # Normalized positions
    tolerance: float = 15.0  # Match tolerance in feet


@dataclass
class TacticalChangePoint:
    """Detected change point in tactical system."""
    timestamp: float
    period: int
    game_time: str
    previous_system: str
    new_system: str
    confidence: float
    trigger: str  # What likely caused the change


@dataclass
class FormationAnalysis:
    """Analysis of a detected formation."""
    formation: str
    confidence: float
    phase: GamePhase
    player_roles: Dict[str, str]  # player_id -> role
    structure_score: float  # How well-organized
    spacing_score: float  # Quality of spacing
    coverage_score: float  # Ice coverage


class HockeyTacticalDetector:
    """
    Tactical Detection System for Hockey.

    Detects and classifies:
    1. Defensive systems (zone, man-to-man, trap, etc.)
    2. Offensive systems (cycle, crash net, etc.)
    3. Special teams formations
    4. Change points in tactical approach

    Based on SoccerCPD with hockey-specific adaptations.
    """

    # Formation templates for matching
    PP_TEMPLATES = {
        PowerPlayFormation.UMBRELLA_1_3_1: [
            (75, 0),    # Point (center)
            (60, -25),  # Half wall left
            (60, 25),   # Half wall right
            (80, -10),  # Net front left
            (80, 10),   # Net front right
        ],
        PowerPlayFormation.OVERLOAD: [
            (75, -15),  # Point (offset)
            (60, -30),  # Strong side boards
            (70, -10),  # Strong side slot
            (80, 0),    # Net front
            (60, 20),   # Weak side
        ],
        PowerPlayFormation.ONE_THREE_ONE: [
            (55, 0),    # High point
            (70, -25),  # Left circle
            (70, 25),   # Right circle
            (85, 0),    # Net front
            (75, 0),    # Slot
        ],
    }

    PK_TEMPLATES = {
        PenaltyKillFormation.BOX: [
            (70, -15),  # Front left
            (70, 15),   # Front right
            (85, -12),  # Back left
            (85, 12),   # Back right
        ],
        PenaltyKillFormation.DIAMOND: [
            (65, 0),    # Top
            (75, -18),  # Left
            (75, 18),   # Right
            (88, 0),    # Bottom (net front)
        ],
        PenaltyKillFormation.AGGRESSIVE: [
            (55, -10),  # High pressure left
            (55, 10),   # High pressure right
            (80, -15),  # Low left
            (80, 15),   # Low right
        ],
    }

    def __init__(
        self,
        change_point_threshold: float = 0.7,
        min_segment_length: int = 10,
        smoothing_window: int = 5
    ):
        """
        Initialize tactical detector.

        Args:
            change_point_threshold: Threshold for detecting system changes
            min_segment_length: Minimum frames before detecting change
            smoothing_window: Window for smoothing classifications
        """
        self.change_threshold = change_point_threshold
        self.min_segment = min_segment_length
        self.smooth_window = smoothing_window

        # Detection history
        self.classification_history: deque = deque(maxlen=1000)
        self.change_points: List[TacticalChangePoint] = []

        # Current state
        self.current_defensive_system = DefensiveSystem.ZONE
        self.current_offensive_system = OffensiveSystem.CYCLE
        self.frames_in_current = 0

    def classify_defensive_system(
        self,
        team_snapshot: TeamSnapshot,
        opponent_snapshot: TeamSnapshot
    ) -> Tuple[DefensiveSystem, float]:
        """
        Classify the defensive system being used.

        Uses player positions relative to opponents and puck.

        Returns:
            Tuple of (system, confidence)
        """
        if len(team_snapshot.players) < 5:
            return DefensiveSystem.ZONE, 0.5

        # Extract features for classification
        features = self._extract_defensive_features(team_snapshot, opponent_snapshot)

        # Rule-based classification with confidence scores
        scores = {
            DefensiveSystem.MAN_TO_MAN: self._score_man_to_man(features),
            DefensiveSystem.ZONE: self._score_zone(features),
            DefensiveSystem.TRAP: self._score_trap(features),
            DefensiveSystem.NEUTRAL_ZONE_TRAP: self._score_nz_trap(features),
            DefensiveSystem.LEFT_WING_LOCK: self._score_lw_lock(features),
        }

        best_system = max(scores, key=scores.get)
        confidence = scores[best_system]

        return best_system, confidence

    def _extract_defensive_features(
        self,
        team: TeamSnapshot,
        opponent: TeamSnapshot
    ) -> Dict[str, float]:
        """Extract features for defensive system classification."""
        features = {}

        team_positions = np.array([[p.x, p.y] for p in team.players])
        opp_positions = np.array([[p.x, p.y] for p in opponent.players])

        # Average distance from opponents
        min_distances = []
        for tp in team_positions:
            dists = [np.linalg.norm(tp - op) for op in opp_positions]
            min_distances.append(min(dists) if dists else 50.0)
        features['avg_min_distance_to_opponent'] = np.mean(min_distances)

        # Formation spread
        features['team_spread_x'] = np.std(team_positions[:, 0])
        features['team_spread_y'] = np.std(team_positions[:, 1])

        # Center of mass
        team_com = np.mean(team_positions, axis=0)
        features['com_x'] = team_com[0]
        features['com_y'] = team_com[1]

        # Distance from puck
        puck_pos = np.array([team.puck_x, team.puck_y])
        puck_distances = [np.linalg.norm(tp - puck_pos) for tp in team_positions]
        features['avg_puck_distance'] = np.mean(puck_distances)
        features['min_puck_distance'] = min(puck_distances)

        # Players in defensive zone
        features['players_in_dz'] = sum(1 for p in team_positions if p[0] < -25)

        # Players in neutral zone
        features['players_in_nz'] = sum(1 for p in team_positions if -25 <= p[0] <= 25)

        # Tight marking (players within 10ft of opponent)
        tight_marks = sum(1 for d in min_distances if d < 10)
        features['tight_marks'] = tight_marks

        return features

    def _score_man_to_man(self, features: Dict[str, float]) -> float:
        """Score likelihood of man-to-man defense."""
        score = 0.0

        # Man-to-man: tight marking distances
        if features['avg_min_distance_to_opponent'] < 12:
            score += 0.4
        if features['tight_marks'] >= 4:
            score += 0.3

        # Less organized spread (following players)
        if features['team_spread_x'] > 30:
            score += 0.15
        if features['team_spread_y'] > 25:
            score += 0.15

        return min(score, 1.0)

    def _score_zone(self, features: Dict[str, float]) -> float:
        """Score likelihood of zone defense."""
        score = 0.0

        # Zone: consistent structure regardless of opponents
        if 20 <= features['team_spread_x'] <= 40:
            score += 0.25
        if 15 <= features['team_spread_y'] <= 30:
            score += 0.25

        # Moderate opponent distance (defending space, not players)
        if 10 <= features['avg_min_distance_to_opponent'] <= 20:
            score += 0.25

        # Most players in defensive zone when defending
        if features['players_in_dz'] >= 4:
            score += 0.25

        return min(score, 1.0)

    def _score_trap(self, features: Dict[str, float]) -> float:
        """Score likelihood of trap defense."""
        score = 0.0

        # Trap: clogging neutral zone
        if features['players_in_nz'] >= 3:
            score += 0.4

        # Compact structure
        if features['team_spread_x'] < 25:
            score += 0.3

        # Center of mass in neutral zone
        if -10 <= features['com_x'] <= 20:
            score += 0.3

        return min(score, 1.0)

    def _score_nz_trap(self, features: Dict[str, float]) -> float:
        """Score likelihood of neutral zone trap."""
        score = 0.0

        # NZ trap specific: players stacked in neutral zone
        if features['players_in_nz'] >= 4:
            score += 0.5

        # Waiting (not aggressive pressure)
        if features['avg_puck_distance'] > 25:
            score += 0.25

        # Horizontal spread
        if features['team_spread_y'] > 25:
            score += 0.25

        return min(score, 1.0)

    def _score_lw_lock(self, features: Dict[str, float]) -> float:
        """Score likelihood of left wing lock."""
        score = 0.0

        # LW lock: 4 players back, 1 forward pressuring
        if features['players_in_dz'] == 4:
            score += 0.4

        # One player far from others (forechecker)
        if features['min_puck_distance'] < 15 and features['avg_puck_distance'] > 30:
            score += 0.3

        # Asymmetric Y spread
        if features['team_spread_y'] > 20:
            score += 0.3

        return min(score, 1.0)

    def classify_forecheck(
        self,
        team_snapshot: TeamSnapshot
    ) -> Tuple[DefensiveSystem, float]:
        """
        Classify the forechecking system.

        Returns:
            Tuple of (forecheck_system, confidence)
        """
        if len(team_snapshot.players) < 5:
            return DefensiveSystem.FORECHECK_2_1_2, 0.5

        # Count players in each zone vertically
        positions = np.array([[p.x, p.y] for p in team_snapshot.players])

        # Layers for forecheck detection
        # Offensive zone: x > 25
        high_pressure = sum(1 for p in positions if p[0] > 50)  # Deep in OZ
        mid_zone = sum(1 for p in positions if 25 < p[0] <= 50)
        low_support = sum(1 for p in positions if p[0] <= 25)

        # Classify based on player distribution
        if high_pressure == 1 and mid_zone == 2:
            return DefensiveSystem.FORECHECK_1_2_2, 0.8
        elif high_pressure == 2 and mid_zone == 1:
            return DefensiveSystem.FORECHECK_2_1_2, 0.8
        elif high_pressure == 1 and mid_zone == 3:
            return DefensiveSystem.FORECHECK_1_3_1, 0.8
        elif high_pressure >= 2 and mid_zone >= 2:
            return DefensiveSystem.AGGRESSIVE_FORECHECK, 0.7
        elif high_pressure <= 1 and low_support >= 3:
            return DefensiveSystem.PASSIVE_FORECHECK, 0.7
        else:
            return DefensiveSystem.FORECHECK_2_1_2, 0.5  # Default

    def classify_pp_formation(
        self,
        team_snapshot: TeamSnapshot
    ) -> Tuple[PowerPlayFormation, float]:
        """
        Classify power play formation.

        Uses template matching against known PP formations.

        Returns:
            Tuple of (formation, confidence)
        """
        if len(team_snapshot.players) < 4:
            return PowerPlayFormation.UMBRELLA_1_3_1, 0.5

        positions = [(p.x, p.y) for p in team_snapshot.players[:5]]
        best_match = None
        best_score = 0.0

        for formation, template in self.PP_TEMPLATES.items():
            score = self._match_formation_template(positions, template)
            if score > best_score:
                best_score = score
                best_match = formation

        if best_match is None:
            return PowerPlayFormation.UMBRELLA_1_3_1, 0.5

        return best_match, best_score

    def classify_pk_formation(
        self,
        team_snapshot: TeamSnapshot
    ) -> Tuple[PenaltyKillFormation, float]:
        """
        Classify penalty kill formation.

        Returns:
            Tuple of (formation, confidence)
        """
        if len(team_snapshot.players) < 4:
            return PenaltyKillFormation.BOX, 0.5

        positions = [(p.x, p.y) for p in team_snapshot.players[:4]]
        best_match = None
        best_score = 0.0

        for formation, template in self.PK_TEMPLATES.items():
            score = self._match_formation_template(positions, template)
            if score > best_score:
                best_score = score
                best_match = formation

        if best_match is None:
            return PenaltyKillFormation.BOX, 0.5

        return best_match, best_score

    def _match_formation_template(
        self,
        positions: List[Tuple[float, float]],
        template: List[Tuple[float, float]]
    ) -> float:
        """
        Match player positions against a formation template.

        Uses Hungarian algorithm-style optimal assignment.

        Returns:
            Match score (0-1)
        """
        if len(positions) < len(template):
            return 0.0

        # Compute distance matrix
        n_players = len(positions)
        n_template = len(template)
        distances = np.zeros((n_players, n_template))

        for i, pos in enumerate(positions):
            for j, temp in enumerate(template):
                distances[i, j] = np.sqrt(
                    (pos[0] - temp[0])**2 + (pos[1] - temp[1])**2
                )

        # Greedy assignment (simplified Hungarian)
        used_players = set()
        used_templates = set()
        total_distance = 0.0

        for _ in range(n_template):
            best_dist = float('inf')
            best_i, best_j = 0, 0

            for i in range(n_players):
                if i in used_players:
                    continue
                for j in range(n_template):
                    if j in used_templates:
                        continue
                    if distances[i, j] < best_dist:
                        best_dist = distances[i, j]
                        best_i, best_j = i, j

            used_players.add(best_i)
            used_templates.add(best_j)
            total_distance += best_dist

        # Convert distance to similarity score
        avg_distance = total_distance / n_template
        tolerance = 20.0  # Expected position tolerance in feet

        score = max(0.0, 1.0 - avg_distance / tolerance)
        return score

    def detect_change_point(
        self,
        team_snapshot: TeamSnapshot,
        opponent_snapshot: TeamSnapshot,
        timestamp: float,
        period: int
    ) -> Optional[TacticalChangePoint]:
        """
        Detect if a tactical change point occurred.

        Based on SoccerCPD's discrete g-segmentation algorithm.

        Returns:
            TacticalChangePoint if detected, None otherwise
        """
        # Classify current system
        current_system, confidence = self.classify_defensive_system(
            team_snapshot, opponent_snapshot
        )

        # Add to history
        self.classification_history.append({
            'timestamp': timestamp,
            'system': current_system,
            'confidence': confidence
        })

        # Check for change
        if current_system != self.current_defensive_system:
            self.frames_in_current = 0

            # Need minimum segment length to confirm change
            if len(self.classification_history) >= self.min_segment:
                recent = list(self.classification_history)[-self.min_segment:]
                new_system_count = sum(
                    1 for r in recent if r['system'] == current_system
                )

                if new_system_count >= self.min_segment * 0.7:
                    # Confirmed change point
                    change = TacticalChangePoint(
                        timestamp=timestamp,
                        period=period,
                        game_time=self._format_game_time(timestamp, period),
                        previous_system=self.current_defensive_system.value,
                        new_system=current_system.value,
                        confidence=confidence,
                        trigger=self._identify_trigger(team_snapshot)
                    )

                    self.change_points.append(change)
                    self.current_defensive_system = current_system
                    return change
        else:
            self.frames_in_current += 1

        return None

    def _format_game_time(self, timestamp: float, period: int) -> str:
        """Format timestamp as game time string."""
        period_time = timestamp % 1200  # Assuming 20 min periods
        minutes = int(period_time // 60)
        seconds = int(period_time % 60)
        return f"P{period} {20-minutes}:{60-seconds:02d}"

    def _identify_trigger(self, snapshot: TeamSnapshot) -> str:
        """Identify what likely triggered a tactical change."""
        # Simple heuristics - would be more sophisticated in production
        if '5v4' in snapshot.strength_state or '5v3' in snapshot.strength_state:
            return "power_play"
        elif '4v5' in snapshot.strength_state or '3v5' in snapshot.strength_state:
            return "penalty_kill"
        elif snapshot.puck_x < -50:
            return "defensive_zone_pressure"
        elif snapshot.puck_x > 50:
            return "offensive_zone_possession"
        else:
            return "neutral_zone_situation"

    def get_formation_analysis(
        self,
        team_snapshot: TeamSnapshot,
        opponent_snapshot: TeamSnapshot
    ) -> FormationAnalysis:
        """
        Get comprehensive formation analysis.

        Returns:
            FormationAnalysis with system, roles, and quality scores
        """
        # Determine game phase
        phase = self._determine_phase(team_snapshot)

        # Classify system based on phase
        if phase in [GamePhase.DEFENSIVE_ZONE, GamePhase.TRANSITION_DEFENSE]:
            system, confidence = self.classify_defensive_system(
                team_snapshot, opponent_snapshot
            )
            formation = system.value
        elif phase == GamePhase.FORECHECK:
            system, confidence = self.classify_forecheck(team_snapshot)
            formation = system.value
        elif '5v4' in team_snapshot.strength_state or '5v3' in team_snapshot.strength_state:
            pp_form, confidence = self.classify_pp_formation(team_snapshot)
            formation = pp_form.value
        elif '4v5' in team_snapshot.strength_state or '3v5' in team_snapshot.strength_state:
            pk_form, confidence = self.classify_pk_formation(team_snapshot)
            formation = pk_form.value
        else:
            system, confidence = self.classify_defensive_system(
                team_snapshot, opponent_snapshot
            )
            formation = system.value

        # Assign player roles
        player_roles = self._assign_player_roles(team_snapshot, phase)

        # Calculate quality scores
        structure_score = self._calculate_structure_score(team_snapshot)
        spacing_score = self._calculate_spacing_score(team_snapshot)
        coverage_score = self._calculate_coverage_score(team_snapshot)

        return FormationAnalysis(
            formation=formation,
            confidence=confidence,
            phase=phase,
            player_roles=player_roles,
            structure_score=structure_score,
            spacing_score=spacing_score,
            coverage_score=coverage_score
        )

    def _determine_phase(self, snapshot: TeamSnapshot) -> GamePhase:
        """Determine current game phase."""
        puck_x = snapshot.puck_x

        # Calculate team velocity (direction of movement)
        avg_velocity_x = np.mean([p.velocity_x for p in snapshot.players])

        if puck_x > 50:
            if avg_velocity_x > 5:
                return GamePhase.FORECHECK
            return GamePhase.OFFENSIVE_ZONE
        elif puck_x < -50:
            if avg_velocity_x > 5:
                return GamePhase.BREAKOUT
            return GamePhase.DEFENSIVE_ZONE
        else:
            if avg_velocity_x > 10:
                return GamePhase.TRANSITION_ATTACK
            elif avg_velocity_x < -5:
                return GamePhase.TRANSITION_DEFENSE
            return GamePhase.NEUTRAL_ZONE

    def _assign_player_roles(
        self,
        snapshot: TeamSnapshot,
        phase: GamePhase
    ) -> Dict[str, str]:
        """Assign tactical roles to players based on position."""
        roles = {}

        # Sort players by x position
        sorted_players = sorted(snapshot.players, key=lambda p: p.x, reverse=True)

        if phase in [GamePhase.FORECHECK, GamePhase.OFFENSIVE_ZONE]:
            # Offensive roles
            if len(sorted_players) >= 5:
                roles[sorted_players[0].player_id] = "F1_pressure"
                roles[sorted_players[1].player_id] = "F2_support"
                roles[sorted_players[2].player_id] = "F3_high"
                roles[sorted_players[3].player_id] = "D1_point"
                roles[sorted_players[4].player_id] = "D2_point"

        elif phase in [GamePhase.DEFENSIVE_ZONE, GamePhase.BREAKOUT]:
            # Defensive roles
            if len(sorted_players) >= 5:
                roles[sorted_players[0].player_id] = "F_outlet"
                roles[sorted_players[1].player_id] = "F_support"
                roles[sorted_players[2].player_id] = "F_low"
                roles[sorted_players[3].player_id] = "D_strong"
                roles[sorted_players[4].player_id] = "D_weak"

        else:
            # Neutral zone roles
            for i, player in enumerate(sorted_players[:5]):
                roles[player.player_id] = f"position_{i+1}"

        return roles

    def _calculate_structure_score(self, snapshot: TeamSnapshot) -> float:
        """Calculate how well-organized the formation is."""
        if len(snapshot.players) < 5:
            return 0.5

        positions = np.array([[p.x, p.y] for p in snapshot.players])

        # Check for clear layers/structure
        x_sorted = np.sort(positions[:, 0])
        x_gaps = np.diff(x_sorted)

        # Good structure has consistent gaps
        gap_variance = np.var(x_gaps)
        structure = max(0.0, 1.0 - gap_variance / 100)

        return structure

    def _calculate_spacing_score(self, snapshot: TeamSnapshot) -> float:
        """Calculate quality of spacing between players."""
        if len(snapshot.players) < 5:
            return 0.5

        positions = np.array([[p.x, p.y] for p in snapshot.players])

        # Calculate pairwise distances
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)

        avg_distance = np.mean(distances)
        min_distance = min(distances)

        # Good spacing: avg 20-30 ft, no one too close (< 10 ft)
        spacing = 0.0
        if 15 <= avg_distance <= 35:
            spacing += 0.5
        if min_distance >= 8:
            spacing += 0.5

        return spacing

    def _calculate_coverage_score(self, snapshot: TeamSnapshot) -> float:
        """Calculate how well the team covers the ice."""
        if len(snapshot.players) < 5:
            return 0.5

        positions = np.array([[p.x, p.y] for p in snapshot.players])

        # Calculate convex hull area (simplified as bounding box)
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()

        area = x_range * y_range
        max_area = 100 * 85  # Full ice

        # Coverage based on area covered
        coverage = min(area / (max_area * 0.3), 1.0)  # 30% of ice is good coverage

        return coverage


class TacticalTrendAnalyzer:
    """
    Analyze tactical trends over time.

    Identifies patterns in how teams adjust systems during games.
    """

    def __init__(self):
        self.game_analyses: List[Dict] = []

    def add_game_analysis(
        self,
        game_id: str,
        change_points: List[TacticalChangePoint],
        formations: List[FormationAnalysis]
    ):
        """Add analysis from a single game."""
        self.game_analyses.append({
            'game_id': game_id,
            'change_points': change_points,
            'formations': formations,
            'n_changes': len(change_points)
        })

    def get_team_tendencies(self) -> Dict[str, Any]:
        """
        Analyze team tactical tendencies across games.

        Returns:
            Dictionary with tendency analysis
        """
        if not self.game_analyses:
            return {}

        all_changes = []
        for game in self.game_analyses:
            all_changes.extend(game['change_points'])

        # Count system transitions
        transitions = {}
        for change in all_changes:
            key = f"{change.previous_system} -> {change.new_system}"
            transitions[key] = transitions.get(key, 0) + 1

        # Count triggers
        triggers = {}
        for change in all_changes:
            triggers[change.trigger] = triggers.get(change.trigger, 0) + 1

        return {
            'total_games': len(self.game_analyses),
            'avg_changes_per_game': np.mean([g['n_changes'] for g in self.game_analyses]),
            'common_transitions': sorted(transitions.items(), key=lambda x: -x[1])[:5],
            'common_triggers': sorted(triggers.items(), key=lambda x: -x[1])[:5]
        }
