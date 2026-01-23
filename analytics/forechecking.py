"""
Forechecking Valuation for Hockey Analytics

This module implements forechecking and pressing valuation adapted from
soccer's exPress research for hockey analytics.

Features:
- Value individual player contributions to forechecking
- Measure forced turnovers and their quality
- Credit players for disrupting opponent breakouts
- Context-aware valuation (score, time, strength)

References:
    - Stephanos, K. (2025). "exPress: Contextual Valuation of Individual
      Players Within Pressing Situations in Soccer." MIT Sloan.
    - Fernández, J., & Bornn, L. (2018). "Wide Open Spaces." MIT Sloan.
    - Stats Perform. (2021). "Making Offensive Play Predictable." MIT Sloan.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class ForecheckType(Enum):
    """Types of forechecking systems."""
    AGGRESSIVE_1_2_2 = "1-2-2_aggressive"
    CONSERVATIVE_1_2_2 = "1-2-2_conservative"
    TWO_ONE_TWO = "2-1-2"
    ONE_THREE_ONE = "1-3-1"
    TWO_THREE = "2-3"
    FULL_PRESS = "full_press"
    PASSIVE_TRAP = "passive_trap"


class ForecheckOutcome(Enum):
    """Outcomes of forechecking pressure."""
    TURNOVER_WON = "turnover_won"
    CONTROLLED_EXIT = "controlled_exit"
    DUMP_OUT = "dump_out"
    ICING = "icing"
    PENALTY_DRAWN = "penalty_drawn"
    SHOT_ATTEMPT = "shot_attempt"
    POSSESSION_RETAINED = "possession_retained"


class PressureSituation(Enum):
    """Types of pressure situations."""
    FORECHECK = "forecheck"
    BACKCHECK = "backcheck"
    NEUTRAL_ZONE_PRESS = "nz_press"
    DEFENSIVE_ZONE_PRESS = "dz_press"


@dataclass
class PlayerPosition:
    """Player position and movement at a moment."""
    player_id: str
    team: str  # 'pressing' or 'defending'
    x: float  # Ice coordinates
    y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    has_puck: bool = False


@dataclass
class ForecheckEvent:
    """Single forechecking event."""
    event_id: str
    timestamp: float
    period: int
    game_time: float  # Seconds remaining in period

    # Player positions
    pressing_players: List[PlayerPosition]
    defending_players: List[PlayerPosition]
    puck_x: float
    puck_y: float

    # Context
    forecheck_type: ForecheckType
    score_differential: int  # From pressing team perspective
    strength_state: str  # 5v5, pp, pk

    # Outcome (if known)
    outcome: Optional[ForecheckOutcome] = None
    turnover_x: Optional[float] = None
    turnover_y: Optional[float] = None
    time_to_outcome: Optional[float] = None  # Seconds


@dataclass
class PlayerForecheckValue:
    """Forechecking value for a single player in an event."""
    player_id: str
    total_value: float
    pressure_contribution: float
    passing_lane_value: float
    positioning_value: float
    closing_speed_value: float


@dataclass
class ForecheckAnalysis:
    """Complete analysis of a forechecking sequence."""
    event_id: str
    team_forecheck_value: float
    player_values: List[PlayerForecheckValue]
    predicted_outcome: ForecheckOutcome
    actual_outcome: Optional[ForecheckOutcome]
    xTurnover: float  # Expected turnover probability
    breakout_difficulty: float  # How hard for defense to exit
    quality_grade: str  # A, B, C, D, F


class HockeyForecheckValuation:
    """
    Forechecking Valuation Model for Hockey.

    Based on exPress (Stephanos, 2025) with hockey-specific adaptations:
    - Values individual player contributions to forechecking
    - Measures disruption of opponent breakouts
    - Credits F1/F2/F3 and defensive support
    - Context-aware (score, time, strength)

    The model computes:
    1. Pressure value: How much pressure is applied to puck carrier
    2. Passing lane value: How many options are blocked
    3. Positioning value: Optimal positioning for turnover
    4. Closing speed value: Rate of gap closure
    """

    # Turnover probability by zone
    BASE_TURNOVER_PROB = {
        'defensive': 0.15,  # Opponent's DZ (our OZ)
        'neutral': 0.08,
        'behind_net': 0.20,
        'corner': 0.18,
    }

    # Outcome values (from pressing team perspective)
    OUTCOME_VALUES = {
        ForecheckOutcome.TURNOVER_WON: 0.08,  # High value
        ForecheckOutcome.ICING: 0.03,  # Forces faceoff in DZ
        ForecheckOutcome.DUMP_OUT: 0.01,  # Neutral
        ForecheckOutcome.CONTROLLED_EXIT: -0.02,  # Successful breakout
        ForecheckOutcome.SHOT_ATTEMPT: 0.05,  # Created chance
        ForecheckOutcome.PENALTY_DRAWN: 0.10,  # Power play
        ForecheckOutcome.POSSESSION_RETAINED: -0.01,
    }

    def __init__(
        self,
        pressure_radius: float = 15.0,  # Feet
        passing_lane_width: float = 5.0,
        optimal_gap: float = 10.0
    ):
        """
        Initialize forechecking valuation.

        Args:
            pressure_radius: Radius for pressure calculation
            passing_lane_width: Width of passing lanes to block
            optimal_gap: Optimal distance from puck carrier
        """
        self.pressure_radius = pressure_radius
        self.lane_width = passing_lane_width
        self.optimal_gap = optimal_gap

    def value_forecheck_event(
        self,
        event: ForecheckEvent
    ) -> ForecheckAnalysis:
        """
        Value a forechecking event.

        Args:
            event: Forechecking event to analyze

        Returns:
            ForecheckAnalysis with player-level values
        """
        # Find puck carrier
        puck_carrier = self._find_puck_carrier(event)

        # Compute team-level metrics
        total_pressure = self._compute_team_pressure(event, puck_carrier)
        lanes_blocked = self._compute_lanes_blocked(event, puck_carrier)
        breakout_difficulty = self._compute_breakout_difficulty(event, puck_carrier)

        # Expected turnover probability
        xTurnover = self._compute_xTurnover(
            event, total_pressure, lanes_blocked, breakout_difficulty
        )

        # Compute player-level values
        player_values = []
        for player in event.pressing_players:
            value = self._value_player_contribution(
                player, event, puck_carrier, total_pressure, lanes_blocked
            )
            player_values.append(value)

        # Team forecheck value
        team_value = sum(pv.total_value for pv in player_values)

        # Predict outcome
        predicted_outcome = self._predict_outcome(xTurnover, breakout_difficulty)

        # Grade the forecheck
        grade = self._grade_forecheck(team_value, xTurnover, predicted_outcome)

        return ForecheckAnalysis(
            event_id=event.event_id,
            team_forecheck_value=team_value,
            player_values=player_values,
            predicted_outcome=predicted_outcome,
            actual_outcome=event.outcome,
            xTurnover=xTurnover,
            breakout_difficulty=breakout_difficulty,
            quality_grade=grade
        )

    def _find_puck_carrier(self, event: ForecheckEvent) -> Optional[PlayerPosition]:
        """Find the puck carrier from defending players."""
        for player in event.defending_players:
            if player.has_puck:
                return player

        # If no one has puck, find closest to puck position
        closest = None
        min_dist = float('inf')
        for player in event.defending_players:
            dist = np.sqrt((player.x - event.puck_x)**2 + (player.y - event.puck_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest = player

        return closest

    def _compute_team_pressure(
        self,
        event: ForecheckEvent,
        puck_carrier: Optional[PlayerPosition]
    ) -> float:
        """
        Compute total pressure on puck carrier.

        Based on proximity and closing speed of forecheckers.
        """
        if puck_carrier is None:
            return 0.0

        total_pressure = 0.0

        for player in event.pressing_players:
            # Distance to puck carrier
            dist = np.sqrt(
                (player.x - puck_carrier.x)**2 +
                (player.y - puck_carrier.y)**2
            )

            if dist > self.pressure_radius * 2:
                continue

            # Base pressure from distance
            base_pressure = max(0, 1 - dist / self.pressure_radius)

            # Closing speed multiplier
            # Vector from player to puck carrier
            dx = puck_carrier.x - player.x
            dy = puck_carrier.y - player.y
            if dist > 0:
                dx /= dist
                dy /= dist

            # Dot product with velocity (positive = closing)
            closing_speed = player.velocity_x * dx + player.velocity_y * dy
            speed_mult = 1 + max(0, closing_speed) / 20  # Up to 2x for fast closing

            total_pressure += base_pressure * speed_mult

        return min(total_pressure, 3.0)  # Cap at 3.0

    def _compute_lanes_blocked(
        self,
        event: ForecheckEvent,
        puck_carrier: Optional[PlayerPosition]
    ) -> int:
        """
        Compute number of passing lanes blocked.

        A lane is blocked if a forechecker intersects the path
        between puck carrier and a teammate.
        """
        if puck_carrier is None:
            return 0

        blocked_count = 0

        # Get potential pass recipients (defending teammates)
        recipients = [
            p for p in event.defending_players
            if p.player_id != puck_carrier.player_id
        ]

        for recipient in recipients:
            # Vector from carrier to recipient
            lane_dx = recipient.x - puck_carrier.x
            lane_dy = recipient.y - puck_carrier.y
            lane_length = np.sqrt(lane_dx**2 + lane_dy**2)

            if lane_length < 1:
                continue

            # Check if any forechecker blocks this lane
            for checker in event.pressing_players:
                if self._blocks_lane(
                    puck_carrier, recipient, checker, lane_length
                ):
                    blocked_count += 1
                    break

        return blocked_count

    def _blocks_lane(
        self,
        carrier: PlayerPosition,
        recipient: PlayerPosition,
        checker: PlayerPosition,
        lane_length: float
    ) -> bool:
        """Check if checker blocks the passing lane."""
        # Point-to-line distance
        # Line from carrier to recipient

        # Vector from carrier to checker
        cx = checker.x - carrier.x
        cy = checker.y - carrier.y

        # Vector from carrier to recipient (normalized)
        rx = (recipient.x - carrier.x) / lane_length
        ry = (recipient.y - carrier.y) / lane_length

        # Project checker onto lane
        proj = cx * rx + cy * ry

        # Check if projection is between carrier and recipient
        if proj < 0 or proj > lane_length:
            return False

        # Perpendicular distance
        perp_dist = abs(cx * ry - cy * rx)

        return perp_dist < self.lane_width

    def _compute_breakout_difficulty(
        self,
        event: ForecheckEvent,
        puck_carrier: Optional[PlayerPosition]
    ) -> float:
        """
        Compute how difficult it is for defense to break out.

        Based on:
        - Forechecking structure
        - Time and space available
        - Passing options
        """
        if puck_carrier is None:
            return 0.5

        difficulty = 0.5  # Base difficulty

        # Pressure contribution
        pressure = self._compute_team_pressure(event, puck_carrier)
        difficulty += pressure * 0.15

        # Lanes blocked contribution
        total_teammates = len([
            p for p in event.defending_players
            if p.player_id != puck_carrier.player_id
        ])
        blocked = self._compute_lanes_blocked(event, puck_carrier)
        if total_teammates > 0:
            block_ratio = blocked / total_teammates
            difficulty += block_ratio * 0.2

        # Position on ice (harder in corners/behind net)
        if event.puck_x > 70 or event.puck_x < -70:  # Behind net area
            difficulty += 0.1
        if abs(event.puck_y) > 30:  # Corner
            difficulty += 0.1

        return min(difficulty, 1.0)

    def _compute_xTurnover(
        self,
        event: ForecheckEvent,
        pressure: float,
        lanes_blocked: int,
        breakout_difficulty: float
    ) -> float:
        """
        Compute expected turnover probability.

        Combines multiple factors into single probability.
        """
        # Base probability by zone
        if event.puck_x > 60:
            zone = 'defensive'  # Opponent's defensive zone
        elif abs(event.puck_y) > 30 and event.puck_x > 25:
            zone = 'corner'
        elif event.puck_x > 85:
            zone = 'behind_net'
        else:
            zone = 'neutral'

        base_prob = self.BASE_TURNOVER_PROB.get(zone, 0.10)

        # Adjust by pressure
        pressure_mult = 1 + pressure * 0.5

        # Adjust by lanes blocked
        lane_mult = 1 + lanes_blocked * 0.15

        # Adjust by breakout difficulty
        difficulty_mult = 1 + (breakout_difficulty - 0.5) * 0.3

        xTurnover = base_prob * pressure_mult * lane_mult * difficulty_mult

        return min(xTurnover, 0.6)  # Cap at 60%

    def _value_player_contribution(
        self,
        player: PlayerPosition,
        event: ForecheckEvent,
        puck_carrier: Optional[PlayerPosition],
        total_pressure: float,
        lanes_blocked: int
    ) -> PlayerForecheckValue:
        """
        Value individual player's forechecking contribution.
        """
        if puck_carrier is None:
            return PlayerForecheckValue(
                player_id=player.player_id,
                total_value=0.0,
                pressure_contribution=0.0,
                passing_lane_value=0.0,
                positioning_value=0.0,
                closing_speed_value=0.0
            )

        # Distance to puck carrier
        dist = np.sqrt(
            (player.x - puck_carrier.x)**2 +
            (player.y - puck_carrier.y)**2
        )

        # Pressure contribution
        if dist < self.pressure_radius:
            pressure_value = (1 - dist / self.pressure_radius) * 0.03
        else:
            pressure_value = 0.0

        # Passing lane value
        lane_value = 0.0
        for defender in event.defending_players:
            if defender.player_id == puck_carrier.player_id:
                continue
            lane_length = np.sqrt(
                (defender.x - puck_carrier.x)**2 +
                (defender.y - puck_carrier.y)**2
            )
            if lane_length > 1 and self._blocks_lane(puck_carrier, defender, player, lane_length):
                lane_value += 0.02

        # Positioning value (optimal gap)
        gap_error = abs(dist - self.optimal_gap)
        positioning_value = max(0, 0.02 - gap_error * 0.002)

        # Closing speed value
        dx = puck_carrier.x - player.x
        dy = puck_carrier.y - player.y
        if dist > 0:
            dx /= dist
            dy /= dist
        closing_speed = player.velocity_x * dx + player.velocity_y * dy
        closing_value = max(0, closing_speed / 20) * 0.01

        total_value = pressure_value + lane_value + positioning_value + closing_value

        return PlayerForecheckValue(
            player_id=player.player_id,
            total_value=total_value,
            pressure_contribution=pressure_value,
            passing_lane_value=lane_value,
            positioning_value=positioning_value,
            closing_speed_value=closing_value
        )

    def _predict_outcome(
        self,
        xTurnover: float,
        breakout_difficulty: float
    ) -> ForecheckOutcome:
        """Predict most likely outcome."""
        if xTurnover > 0.35:
            return ForecheckOutcome.TURNOVER_WON
        elif breakout_difficulty > 0.7:
            if np.random.random() < 0.3:
                return ForecheckOutcome.ICING
            return ForecheckOutcome.DUMP_OUT
        elif breakout_difficulty > 0.5:
            return ForecheckOutcome.DUMP_OUT
        else:
            return ForecheckOutcome.CONTROLLED_EXIT

    def _grade_forecheck(
        self,
        team_value: float,
        xTurnover: float,
        predicted: ForecheckOutcome
    ) -> str:
        """Grade the quality of the forecheck."""
        # Combined score
        score = team_value * 10 + xTurnover

        if score > 0.5:
            return 'A'
        elif score > 0.35:
            return 'B'
        elif score > 0.2:
            return 'C'
        elif score > 0.1:
            return 'D'
        else:
            return 'F'


class ForecheckTracker:
    """
    Track and accumulate forechecking metrics over time.
    """

    def __init__(self):
        """Initialize tracker."""
        self.model = HockeyForecheckValuation()
        self.player_totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                'total_value': 0.0,
                'pressure_value': 0.0,
                'lane_value': 0.0,
                'events': 0,
                'turnovers_forced': 0
            }
        )
        self.team_events: List[ForecheckAnalysis] = []

    def record_event(self, event: ForecheckEvent) -> ForecheckAnalysis:
        """
        Record and analyze a forechecking event.

        Args:
            event: Forechecking event

        Returns:
            ForecheckAnalysis
        """
        analysis = self.model.value_forecheck_event(event)

        # Accumulate player totals
        for pv in analysis.player_values:
            self.player_totals[pv.player_id]['total_value'] += pv.total_value
            self.player_totals[pv.player_id]['pressure_value'] += pv.pressure_contribution
            self.player_totals[pv.player_id]['lane_value'] += pv.passing_lane_value
            self.player_totals[pv.player_id]['events'] += 1

            if event.outcome == ForecheckOutcome.TURNOVER_WON:
                self.player_totals[pv.player_id]['turnovers_forced'] += 1

        self.team_events.append(analysis)

        return analysis

    def get_player_rankings(self) -> List[Dict[str, Any]]:
        """
        Get player rankings by forechecking value.

        Returns:
            List of player dictionaries sorted by value
        """
        rankings = []
        for player_id, totals in self.player_totals.items():
            if totals['events'] == 0:
                continue

            rankings.append({
                'player_id': player_id,
                'total_forecheck_value': totals['total_value'],
                'value_per_event': totals['total_value'] / totals['events'],
                'pressure_value': totals['pressure_value'],
                'lane_blocking_value': totals['lane_value'],
                'events': totals['events'],
                'turnovers_forced': totals['turnovers_forced'],
                'turnover_rate': totals['turnovers_forced'] / totals['events']
            })

        return sorted(rankings, key=lambda x: -x['total_forecheck_value'])

    def get_team_summary(self) -> Dict[str, Any]:
        """
        Get team-level forechecking summary.

        Returns:
            Dictionary with team metrics
        """
        if not self.team_events:
            return {}

        return {
            'total_events': len(self.team_events),
            'avg_xTurnover': np.mean([e.xTurnover for e in self.team_events]),
            'avg_breakout_difficulty': np.mean([e.breakout_difficulty for e in self.team_events]),
            'grade_distribution': self._grade_distribution(),
            'turnovers_won': sum(
                1 for e in self.team_events
                if e.actual_outcome == ForecheckOutcome.TURNOVER_WON
            ),
            'controlled_exits_allowed': sum(
                1 for e in self.team_events
                if e.actual_outcome == ForecheckOutcome.CONTROLLED_EXIT
            )
        }

    def _grade_distribution(self) -> Dict[str, int]:
        """Get distribution of forecheck grades."""
        grades = defaultdict(int)
        for event in self.team_events:
            grades[event.quality_grade] += 1
        return dict(grades)


class BackcheckValuation:
    """
    Backcheck valuation - defensive transition tracking.

    Measures player contribution to defensive recovery.
    """

    def __init__(self):
        """Initialize backcheck valuation."""
        self.events: List[Dict] = []

    def value_backcheck(
        self,
        player: PlayerPosition,
        puck_carrier: PlayerPosition,
        goal_position: Tuple[float, float]
    ) -> float:
        """
        Value a player's backchecking effort.

        Args:
            player: Backchecking player
            puck_carrier: Opponent with puck
            goal_position: Position of goal being defended

        Returns:
            Backcheck value
        """
        value = 0.0

        # Recovery angle - are they between puck and goal?
        puck_to_goal = (
            goal_position[0] - puck_carrier.x,
            goal_position[1] - puck_carrier.y
        )
        puck_to_player = (
            player.x - puck_carrier.x,
            player.y - puck_carrier.y
        )

        # Dot product (positive = good recovery angle)
        ptg_len = np.sqrt(puck_to_goal[0]**2 + puck_to_goal[1]**2)
        if ptg_len > 0:
            angle_score = (
                puck_to_goal[0] * puck_to_player[0] +
                puck_to_goal[1] * puck_to_player[1]
            ) / (ptg_len * np.sqrt(puck_to_player[0]**2 + puck_to_player[1]**2 + 0.01))

            value += max(0, angle_score) * 0.02

        # Speed toward goal (positive backcheck speed)
        speed_toward_goal = -player.velocity_x  # Assuming goal at negative x
        value += max(0, speed_toward_goal / 25) * 0.02

        # Gap closure
        dist_to_carrier = np.sqrt(
            (player.x - puck_carrier.x)**2 +
            (player.y - puck_carrier.y)**2
        )
        if dist_to_carrier < 20:
            value += (20 - dist_to_carrier) / 20 * 0.01

        return value

    def track_recovery(
        self,
        player_id: str,
        recovery_time: float,
        prevented_chance: bool
    ):
        """
        Track a backcheck recovery event.

        Args:
            player_id: Player who recovered
            recovery_time: Time to recover (seconds)
            prevented_chance: Whether a scoring chance was prevented
        """
        self.events.append({
            'player_id': player_id,
            'recovery_time': recovery_time,
            'prevented_chance': prevented_chance
        })

    def get_player_backcheck_stats(self, player_id: str) -> Dict[str, float]:
        """Get backcheck statistics for a player."""
        player_events = [e for e in self.events if e['player_id'] == player_id]

        if not player_events:
            return {}

        return {
            'events': len(player_events),
            'avg_recovery_time': np.mean([e['recovery_time'] for e in player_events]),
            'chances_prevented': sum(1 for e in player_events if e['prevented_chance']),
            'prevention_rate': sum(1 for e in player_events if e['prevented_chance']) / len(player_events)
        }
