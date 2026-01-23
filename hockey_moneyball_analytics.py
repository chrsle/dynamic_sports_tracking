"""
Hockey Moneyball Analytics - Finding Undervalued Players, Plays & Strategies

This module implements "Moneyball for Hockey" - identifying market inefficiencies
in player valuations, line combinations, and strategic patterns.

Key Features:
1. Player Value Metrics - xG/60, xG above replacement, cost efficiency
2. Line Combination Analysis - Which combinations outperform their salary
3. Shift-by-Shift Efficiency - Fatigue detection and fresh line advantage
4. Forechecking Patterns - Aggressive vs passive system classification
5. Zone Entry Success Rate - Carry-in vs dump-and-chase effectiveness
6. Salary/Cap Efficiency - Contract value vs actual production

Usage:
    from hockey_moneyball_analytics import MoneyballAnalytics

    analytics = MoneyballAnalytics()
    analytics.add_player("87", "Sidney Crosby", team=0, salary=8700000)

    # Track events
    analytics.track_shift_start("87", frame=0)
    analytics.track_zone_entry("87", entry_type="carry", success=True, frame=100)
    analytics.track_scoring_chance("87", xg=0.15, frame=200)
    analytics.track_shift_end("87", frame=900)

    # Get insights
    undervalued = analytics.find_undervalued_players()
    best_lines = analytics.get_best_line_combinations()
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import IntEnum
from datetime import datetime
import json


# ==================== Configuration ====================

@dataclass
class MoneyballConfig:
    """Configuration for moneyball analytics."""

    # Time normalization (per 60 minutes of ice time)
    frames_per_second: float = 30.0
    seconds_per_period: float = 1200.0  # 20 minutes

    # Shift analysis
    optimal_shift_length: float = 45.0   # seconds
    tired_shift_threshold: float = 60.0  # seconds - fatigue kicks in
    fresh_shift_window: float = 15.0     # seconds - "fresh legs" bonus

    # Zone entry thresholds
    entry_success_baseline: float = 0.55  # League average carry-in success
    dump_chase_baseline: float = 0.45     # League average dump-in recovery

    # Forechecking classification
    aggressive_forecheck_threshold: float = 0.65  # High pressure
    passive_forecheck_threshold: float = 0.35     # Trap/passive

    # Value metrics
    replacement_level_xg60: float = 0.35  # Replacement player xG/60
    replacement_level_xa60: float = 0.30  # Replacement player xA/60

    # Salary thresholds (2024 values)
    league_min_salary: int = 775_000
    league_avg_salary: int = 3_500_000
    cap_ceiling: int = 83_500_000

    # Rink dimensions
    rink_length: float = 200.0
    rink_width: float = 85.0
    blue_line_x: float = 25.0
    goal_line_x: float = 89.0


# ==================== Data Classes ====================

class EntryType(IntEnum):
    """Zone entry types."""
    CARRY = 0       # Controlled entry with puck
    DUMP = 1        # Dump and chase
    PASS = 2        # Pass into zone
    DEFLECTION = 3  # Deflected in


class ForecheckStyle(IntEnum):
    """Forechecking system styles."""
    AGGRESSIVE_2_1_2 = 0   # 2 forwards high, 1 mid, 2 back
    STANDARD_1_2_2 = 1     # 1 high, 2 mid, 2 back
    PASSIVE_1_4 = 2        # 1 high, 4 back (trap)
    COLLAPSING = 3         # Everyone collapses to net


@dataclass
class PlayerProfile:
    """Complete player profile for analytics."""
    player_id: str
    name: str
    team: int
    position: str = "F"  # F, D, G
    jersey_number: Optional[int] = None
    salary: int = 0

    # Tracked metrics
    total_ice_time_frames: int = 0
    shifts: List[Dict] = field(default_factory=list)
    scoring_chances: List[Dict] = field(default_factory=list)
    assists: List[Dict] = field(default_factory=list)
    zone_entries: List[Dict] = field(default_factory=list)
    zone_exits: List[Dict] = field(default_factory=list)
    defensive_plays: List[Dict] = field(default_factory=list)

    # Current shift tracking
    current_shift_start: Optional[int] = None
    current_position: Optional[Tuple[float, float]] = None
    position_history: deque = field(default_factory=lambda: deque(maxlen=300))

    # Computed metrics (updated periodically)
    xg_total: float = 0.0
    xa_total: float = 0.0  # Expected assists
    xg_against: float = 0.0  # xG allowed while on ice


@dataclass
class LineCombination:
    """Track line combination performance."""
    player_ids: Tuple[str, ...]
    ice_time_frames: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    zone_entries: int = 0
    zone_entry_successes: int = 0
    scoring_chances: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def combined_salary(self) -> int:
        """Must be set externally after creation."""
        return getattr(self, '_combined_salary', 0)

    @combined_salary.setter
    def combined_salary(self, value: int):
        self._combined_salary = value


@dataclass
class ShiftAnalysis:
    """Analysis of a single shift."""
    player_id: str
    start_frame: int
    end_frame: int
    duration_seconds: float

    # Performance during shift
    xg_generated: float = 0.0
    xg_allowed: float = 0.0
    zone_entries: int = 0
    zone_entry_successes: int = 0
    shots: int = 0

    # Movement metrics
    distance_skated: float = 0.0
    avg_speed: float = 0.0
    time_in_offensive_zone: float = 0.0
    time_in_defensive_zone: float = 0.0

    # Fatigue indicators
    is_tired: bool = False
    speed_dropoff: float = 0.0  # % speed decrease in last 15 seconds


@dataclass
class ZoneEntryEvent:
    """Zone entry event tracking."""
    player_id: str
    frame: int
    entry_type: EntryType
    success: bool  # Did team maintain possession?
    resulted_in_shot: bool = False
    resulted_in_goal: bool = False
    xg_generated: float = 0.0
    time_in_zone: float = 0.0  # Seconds of possession after entry


@dataclass
class ForecheckEvent:
    """Forechecking pressure event."""
    frame: int
    team: int
    players_involved: List[str]
    style: ForecheckStyle
    pressure_duration: float  # Seconds
    caused_turnover: bool = False
    led_to_scoring_chance: bool = False


# ==================== Main Analytics Class ====================

class MoneyballAnalytics:
    """
    Main class for hockey moneyball analytics.

    Tracks players, lines, and events to find undervalued assets.
    """

    def __init__(self, config: Optional[MoneyballConfig] = None):
        self.config = config or MoneyballConfig()

        # Player tracking
        self.players: Dict[str, PlayerProfile] = {}

        # Line combination tracking
        self.line_combinations: Dict[Tuple[str, ...], LineCombination] = {}
        self.current_lines: Dict[int, Set[str]] = {0: set(), 1: set()}

        # Team-level metrics
        self.team_zone_entries: Dict[int, List[ZoneEntryEvent]] = {0: [], 1: []}
        self.team_forecheck_events: Dict[int, List[ForecheckEvent]] = {0: [], 1: []}

        # Frame tracking
        self.frame_count = 0

        # Salary data (can be loaded from external source)
        self.salary_data: Dict[str, int] = {}

    # ==================== Player Management ====================

    def add_player(
        self,
        player_id: str,
        name: str,
        team: int,
        position: str = "F",
        jersey_number: Optional[int] = None,
        salary: int = 0
    ) -> PlayerProfile:
        """
        Add a player to tracking.

        Args:
            player_id: Unique identifier
            name: Player name
            team: Team ID (0 or 1)
            position: F (forward), D (defense), G (goalie)
            jersey_number: Jersey number
            salary: Annual salary in dollars

        Returns:
            PlayerProfile object
        """
        profile = PlayerProfile(
            player_id=player_id,
            name=name,
            team=team,
            position=position,
            jersey_number=jersey_number,
            salary=salary
        )
        self.players[player_id] = profile
        self.salary_data[player_id] = salary
        return profile

    def get_player(self, player_id: str) -> Optional[PlayerProfile]:
        """Get player profile by ID."""
        return self.players.get(player_id)

    # ==================== Shift Tracking ====================

    def track_shift_start(self, player_id: str, frame: int):
        """
        Track a player starting a shift.

        Args:
            player_id: Player identifier
            frame: Current frame number
        """
        player = self.players.get(player_id)
        if not player:
            return

        player.current_shift_start = frame
        player.position_history.clear()

        # Add to current on-ice players
        self.current_lines[player.team].add(player_id)

        # Update line combinations
        self._update_line_tracking(player.team)

    def track_shift_end(self, player_id: str, frame: int) -> Optional[ShiftAnalysis]:
        """
        Track a player ending a shift.

        Args:
            player_id: Player identifier
            frame: Current frame number

        Returns:
            ShiftAnalysis for the completed shift
        """
        player = self.players.get(player_id)
        if not player or player.current_shift_start is None:
            return None

        # Calculate shift duration
        shift_frames = frame - player.current_shift_start
        shift_seconds = shift_frames / self.config.frames_per_second

        # Analyze the shift
        analysis = self._analyze_shift(player, player.current_shift_start, frame)

        # Store shift record
        player.shifts.append({
            'start': player.current_shift_start,
            'end': frame,
            'duration': shift_seconds,
            'analysis': analysis
        })

        # Update ice time
        player.total_ice_time_frames += shift_frames

        # Remove from current on-ice players
        self.current_lines[player.team].discard(player_id)
        player.current_shift_start = None

        return analysis

    def _analyze_shift(
        self,
        player: PlayerProfile,
        start_frame: int,
        end_frame: int
    ) -> ShiftAnalysis:
        """Analyze a completed shift for performance and fatigue."""
        shift_frames = end_frame - start_frame
        shift_seconds = shift_frames / self.config.frames_per_second

        # Calculate xG during shift
        xg_generated = sum(
            c['xg'] for c in player.scoring_chances
            if start_frame <= c['frame'] <= end_frame
        )

        # Calculate zone entries during shift
        entries = [
            e for e in player.zone_entries
            if start_frame <= e['frame'] <= end_frame
        ]
        entry_successes = sum(1 for e in entries if e.get('success', False))

        # Analyze movement for fatigue
        positions = list(player.position_history)
        distance_skated = 0.0
        speeds = []

        for i in range(1, len(positions)):
            if positions[i] and positions[i-1]:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dist = np.sqrt(dx**2 + dy**2)
                distance_skated += dist
                speed = dist * self.config.frames_per_second  # ft/sec
                speeds.append(speed)

        avg_speed = np.mean(speeds) if speeds else 0.0

        # Check for fatigue (speed dropoff in last portion of shift)
        is_tired = shift_seconds > self.config.tired_shift_threshold
        speed_dropoff = 0.0

        if len(speeds) > 20:
            first_third = np.mean(speeds[:len(speeds)//3])
            last_third = np.mean(speeds[-len(speeds)//3:])
            if first_third > 0:
                speed_dropoff = (first_third - last_third) / first_third
                is_tired = is_tired or speed_dropoff > 0.2  # 20% speed drop

        return ShiftAnalysis(
            player_id=player.player_id,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_seconds=shift_seconds,
            xg_generated=xg_generated,
            zone_entries=len(entries),
            zone_entry_successes=entry_successes,
            distance_skated=distance_skated,
            avg_speed=avg_speed,
            is_tired=is_tired,
            speed_dropoff=speed_dropoff
        )

    def _update_line_tracking(self, team: int):
        """Update line combination tracking when roster changes."""
        current = self.current_lines[team]
        if len(current) < 2:
            return

        # Create sorted tuple for consistent key
        line_key = tuple(sorted(current))

        if line_key not in self.line_combinations:
            # Calculate combined salary
            combined_salary = sum(
                self.salary_data.get(pid, 0) for pid in line_key
            )
            self.line_combinations[line_key] = LineCombination(
                player_ids=line_key
            )
            self.line_combinations[line_key].combined_salary = combined_salary

    def update_position(self, player_id: str, position: Tuple[float, float]):
        """
        Update player position for movement tracking.

        Args:
            player_id: Player identifier
            position: (x, y) position in rink coordinates
        """
        player = self.players.get(player_id)
        if player:
            player.current_position = position
            player.position_history.append(position)

    # ==================== Event Tracking ====================

    def track_scoring_chance(
        self,
        player_id: str,
        xg: float,
        frame: int,
        resulted_in_goal: bool = False,
        assisted_by: Optional[List[str]] = None
    ):
        """
        Track a scoring chance.

        Args:
            player_id: Shooter ID
            xg: Expected goals value
            frame: Frame number
            resulted_in_goal: Did it go in?
            assisted_by: List of assisting player IDs
        """
        player = self.players.get(player_id)
        if not player:
            return

        # Record for shooter
        player.scoring_chances.append({
            'frame': frame,
            'xg': xg,
            'goal': resulted_in_goal,
            'assists': assisted_by or []
        })
        player.xg_total += xg

        # Record assists
        if assisted_by:
            xa_per_assist = xg * 0.5  # Split credit
            for assist_id in assisted_by:
                assist_player = self.players.get(assist_id)
                if assist_player:
                    assist_player.assists.append({
                        'frame': frame,
                        'xa': xa_per_assist,
                        'shooter': player_id
                    })
                    assist_player.xa_total += xa_per_assist

        # Update line combination stats
        line_key = tuple(sorted(self.current_lines[player.team]))
        if line_key in self.line_combinations:
            self.line_combinations[line_key].xg_for += xg
            self.line_combinations[line_key].scoring_chances += 1
            if resulted_in_goal:
                self.line_combinations[line_key].goals_for += 1

    def track_zone_entry(
        self,
        player_id: str,
        entry_type: str,  # 'carry', 'dump', 'pass'
        success: bool,
        frame: int,
        resulted_in_shot: bool = False,
        xg_generated: float = 0.0
    ):
        """
        Track a zone entry attempt.

        Args:
            player_id: Player making the entry
            entry_type: Type of entry ('carry', 'dump', 'pass')
            success: Did team maintain possession?
            frame: Frame number
            resulted_in_shot: Did entry lead to a shot?
            xg_generated: xG from subsequent chances
        """
        player = self.players.get(player_id)
        if not player:
            return

        entry_type_enum = {
            'carry': EntryType.CARRY,
            'dump': EntryType.DUMP,
            'pass': EntryType.PASS,
            'deflection': EntryType.DEFLECTION
        }.get(entry_type, EntryType.CARRY)

        entry = ZoneEntryEvent(
            player_id=player_id,
            frame=frame,
            entry_type=entry_type_enum,
            success=success,
            resulted_in_shot=resulted_in_shot,
            xg_generated=xg_generated
        )

        player.zone_entries.append({
            'frame': frame,
            'type': entry_type,
            'success': success,
            'shot': resulted_in_shot,
            'xg': xg_generated
        })

        self.team_zone_entries[player.team].append(entry)

        # Update line combination stats
        line_key = tuple(sorted(self.current_lines[player.team]))
        if line_key in self.line_combinations:
            self.line_combinations[line_key].zone_entries += 1
            if success:
                self.line_combinations[line_key].zone_entry_successes += 1

    def track_forecheck(
        self,
        team: int,
        players_involved: List[str],
        frame: int,
        pressure_duration: float,
        caused_turnover: bool = False,
        led_to_chance: bool = False
    ):
        """
        Track a forechecking pressure event.

        Args:
            team: Team applying forecheck
            players_involved: Players in forecheck
            frame: Frame number
            pressure_duration: Duration of pressure in seconds
            caused_turnover: Did it force a turnover?
            led_to_chance: Did it lead to a scoring chance?
        """
        # Classify forecheck style based on player positions
        style = self._classify_forecheck_style(players_involved)

        event = ForecheckEvent(
            frame=frame,
            team=team,
            players_involved=players_involved,
            style=style,
            pressure_duration=pressure_duration,
            caused_turnover=caused_turnover,
            led_to_scoring_chance=led_to_chance
        )

        self.team_forecheck_events[team].append(event)

    def _classify_forecheck_style(self, players: List[str]) -> ForecheckStyle:
        """
        Classify forechecking style based on player positions.

        Returns predicted style based on how many players are pressuring high.
        """
        high_pressure_count = 0

        for pid in players:
            player = self.players.get(pid)
            if player and player.current_position:
                x = player.current_position[0]
                # Count players pressuring in offensive zone
                if abs(x) > self.config.blue_line_x:
                    high_pressure_count += 1

        if high_pressure_count >= 3:
            return ForecheckStyle.AGGRESSIVE_2_1_2
        elif high_pressure_count >= 2:
            return ForecheckStyle.STANDARD_1_2_2
        elif high_pressure_count == 1:
            return ForecheckStyle.PASSIVE_1_4
        else:
            return ForecheckStyle.COLLAPSING

    # ==================== Value Metrics ====================

    def get_player_value_metrics(self, player_id: str) -> Dict:
        """
        Calculate comprehensive value metrics for a player.

        Returns:
            Dict with xG/60, xA/60, value over replacement, cost efficiency
        """
        player = self.players.get(player_id)
        if not player:
            return {}

        # Calculate ice time in minutes
        ice_time_minutes = (
            player.total_ice_time_frames /
            self.config.frames_per_second / 60.0
        )

        if ice_time_minutes < 1:
            ice_time_minutes = 1  # Avoid division by zero

        # Per-60 metrics
        xg_per_60 = (player.xg_total / ice_time_minutes) * 60
        xa_per_60 = (player.xa_total / ice_time_minutes) * 60
        points_per_60 = xg_per_60 + xa_per_60

        # Value over replacement
        xg_over_replacement = xg_per_60 - self.config.replacement_level_xg60
        xa_over_replacement = xa_per_60 - self.config.replacement_level_xa60
        total_value_over_replacement = xg_over_replacement + xa_over_replacement

        # Zone entry success rate
        entries = player.zone_entries
        carry_entries = [e for e in entries if e.get('type') == 'carry']
        dump_entries = [e for e in entries if e.get('type') == 'dump']

        carry_success_rate = (
            sum(1 for e in carry_entries if e.get('success', False)) /
            max(len(carry_entries), 1)
        )
        dump_success_rate = (
            sum(1 for e in dump_entries if e.get('success', False)) /
            max(len(dump_entries), 1)
        )

        # Shift efficiency
        shift_analyses = [s.get('analysis') for s in player.shifts if s.get('analysis')]
        avg_shift_length = np.mean([s.duration_seconds for s in shift_analyses]) if shift_analyses else 0
        tired_shift_pct = (
            sum(1 for s in shift_analyses if s.is_tired) / max(len(shift_analyses), 1)
        )

        # Fresh legs performance (first 15 seconds of shifts)
        fresh_xg = sum(
            s.xg_generated for s in shift_analyses
            if s.duration_seconds <= self.config.fresh_shift_window
        )

        # Cost efficiency (value per million dollars)
        salary_millions = player.salary / 1_000_000 if player.salary > 0 else 1
        value_per_million = total_value_over_replacement / salary_millions

        # Expected salary based on production (rough estimate)
        # Top players ~0.15 xG/60 = ~$10M, replacement = ~$1M
        production_factor = max(0, points_per_60 - 0.3) / 0.2  # 0-1 scale
        expected_salary = int(
            self.config.league_min_salary +
            production_factor * (10_000_000 - self.config.league_min_salary)
        )

        salary_delta = expected_salary - player.salary
        is_undervalued = salary_delta > 500_000  # $500K+ under expected
        is_overvalued = salary_delta < -500_000

        return {
            'player_id': player_id,
            'name': player.name,
            'team': player.team,
            'position': player.position,
            'salary': player.salary,
            'ice_time_minutes': round(ice_time_minutes, 1),

            # Per-60 metrics
            'xg_per_60': round(xg_per_60, 3),
            'xa_per_60': round(xa_per_60, 3),
            'points_per_60': round(points_per_60, 3),

            # Value over replacement
            'xg_over_replacement': round(xg_over_replacement, 3),
            'xa_over_replacement': round(xa_over_replacement, 3),
            'total_vor': round(total_value_over_replacement, 3),

            # Zone entry
            'carry_entry_success': round(carry_success_rate, 3),
            'dump_entry_success': round(dump_success_rate, 3),
            'entry_differential': round(carry_success_rate - self.config.entry_success_baseline, 3),

            # Shift metrics
            'avg_shift_length': round(avg_shift_length, 1),
            'tired_shift_pct': round(tired_shift_pct, 3),

            # Cost efficiency
            'value_per_million': round(value_per_million, 3),
            'expected_salary': expected_salary,
            'salary_delta': salary_delta,
            'is_undervalued': is_undervalued,
            'is_overvalued': is_overvalued,

            # Raw totals
            'total_xg': round(player.xg_total, 2),
            'total_xa': round(player.xa_total, 2),
            'total_shifts': len(player.shifts),
            'total_zone_entries': len(player.zone_entries)
        }

    def find_undervalued_players(self, min_ice_time: float = 10.0) -> List[Dict]:
        """
        Find players who are producing more than their salary suggests.

        Args:
            min_ice_time: Minimum ice time in minutes to consider

        Returns:
            List of player value metrics, sorted by value/salary ratio
        """
        undervalued = []

        for player_id in self.players:
            metrics = self.get_player_value_metrics(player_id)

            if metrics.get('ice_time_minutes', 0) < min_ice_time:
                continue

            if metrics.get('is_undervalued', False):
                undervalued.append(metrics)

        # Sort by value per million (descending)
        undervalued.sort(key=lambda x: x.get('value_per_million', 0), reverse=True)

        return undervalued

    def find_overvalued_players(self, min_ice_time: float = 10.0) -> List[Dict]:
        """
        Find players who are producing less than their salary suggests.

        Args:
            min_ice_time: Minimum ice time in minutes to consider

        Returns:
            List of player value metrics, sorted by value/salary ratio
        """
        overvalued = []

        for player_id in self.players:
            metrics = self.get_player_value_metrics(player_id)

            if metrics.get('ice_time_minutes', 0) < min_ice_time:
                continue

            if metrics.get('is_overvalued', False):
                overvalued.append(metrics)

        # Sort by value per million (ascending - worst value first)
        overvalued.sort(key=lambda x: x.get('value_per_million', 0))

        return overvalued

    # ==================== Line Combination Analysis ====================

    def get_line_combination_metrics(self, line_key: Tuple[str, ...]) -> Dict:
        """
        Get performance metrics for a line combination.

        Args:
            line_key: Tuple of player IDs

        Returns:
            Dict with line performance metrics
        """
        line = self.line_combinations.get(line_key)
        if not line:
            return {}

        # Calculate ice time in minutes
        ice_time_minutes = (
            line.ice_time_frames /
            self.config.frames_per_second / 60.0
        )

        if ice_time_minutes < 0.1:
            ice_time_minutes = 0.1

        # Per-60 metrics
        xg_for_60 = (line.xg_for / ice_time_minutes) * 60
        xg_against_60 = (line.xg_against / ice_time_minutes) * 60
        xg_differential_60 = xg_for_60 - xg_against_60

        # Zone entry success
        entry_success = (
            line.zone_entry_successes / max(line.zone_entries, 1)
        )

        # Cost efficiency
        salary_millions = line.combined_salary / 1_000_000 if line.combined_salary > 0 else 1
        value_per_million = xg_differential_60 / salary_millions

        # Get player names
        player_names = [
            self.players[pid].name if pid in self.players else pid
            for pid in line_key
        ]

        return {
            'players': player_names,
            'player_ids': line_key,
            'ice_time_minutes': round(ice_time_minutes, 1),
            'combined_salary': line.combined_salary,

            # Performance
            'xg_for_60': round(xg_for_60, 3),
            'xg_against_60': round(xg_against_60, 3),
            'xg_differential_60': round(xg_differential_60, 3),

            # Zone entry
            'zone_entries': line.zone_entries,
            'entry_success_rate': round(entry_success, 3),

            # Goals
            'goals_for': line.goals_for,
            'goals_against': line.goals_against,

            # Cost efficiency
            'value_per_million': round(value_per_million, 3)
        }

    def get_best_line_combinations(
        self,
        min_ice_time: float = 2.0,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Get the best performing line combinations by xG differential.

        Args:
            min_ice_time: Minimum ice time in minutes
            top_n: Number of lines to return

        Returns:
            List of line metrics, sorted by xG differential
        """
        lines = []

        for line_key in self.line_combinations:
            metrics = self.get_line_combination_metrics(line_key)

            if metrics.get('ice_time_minutes', 0) >= min_ice_time:
                lines.append(metrics)

        # Sort by xG differential (best first)
        lines.sort(key=lambda x: x.get('xg_differential_60', 0), reverse=True)

        return lines[:top_n]

    def get_best_value_lines(
        self,
        min_ice_time: float = 2.0,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Get line combinations with best value/salary ratio.

        Args:
            min_ice_time: Minimum ice time in minutes
            top_n: Number of lines to return

        Returns:
            List of line metrics, sorted by value per million
        """
        lines = []

        for line_key in self.line_combinations:
            metrics = self.get_line_combination_metrics(line_key)

            if metrics.get('ice_time_minutes', 0) >= min_ice_time:
                lines.append(metrics)

        # Sort by value per million (best value first)
        lines.sort(key=lambda x: x.get('value_per_million', 0), reverse=True)

        return lines[:top_n]

    # ==================== Shift Efficiency Analysis ====================

    def get_shift_efficiency_report(self, player_id: str) -> Dict:
        """
        Get detailed shift efficiency analysis for a player.

        Args:
            player_id: Player identifier

        Returns:
            Dict with shift-by-shift efficiency breakdown
        """
        player = self.players.get(player_id)
        if not player:
            return {}

        shift_analyses = [s.get('analysis') for s in player.shifts if s.get('analysis')]

        if not shift_analyses:
            return {'player_id': player_id, 'no_data': True}

        # Categorize shifts
        fresh_shifts = [s for s in shift_analyses if s.duration_seconds <= self.config.fresh_shift_window]
        optimal_shifts = [s for s in shift_analyses if self.config.fresh_shift_window < s.duration_seconds <= self.config.optimal_shift_length]
        long_shifts = [s for s in shift_analyses if self.config.optimal_shift_length < s.duration_seconds <= self.config.tired_shift_threshold]
        tired_shifts = [s for s in shift_analyses if s.duration_seconds > self.config.tired_shift_threshold]

        def calc_efficiency(shifts: List[ShiftAnalysis]) -> Dict:
            if not shifts:
                return {'xg_per_shift': 0, 'entry_success': 0, 'count': 0}
            return {
                'xg_per_shift': np.mean([s.xg_generated for s in shifts]),
                'entry_success': np.mean([s.zone_entry_successes / max(s.zone_entries, 1) for s in shifts]),
                'avg_speed': np.mean([s.avg_speed for s in shifts]),
                'count': len(shifts)
            }

        return {
            'player_id': player_id,
            'name': player.name,
            'total_shifts': len(shift_analyses),

            'fresh_shifts': calc_efficiency(fresh_shifts),
            'optimal_shifts': calc_efficiency(optimal_shifts),
            'long_shifts': calc_efficiency(long_shifts),
            'tired_shifts': calc_efficiency(tired_shifts),

            'avg_shift_length': np.mean([s.duration_seconds for s in shift_analyses]),
            'tired_shift_percentage': len(tired_shifts) / max(len(shift_analyses), 1),

            # Fatigue impact
            'avg_speed_dropoff': np.mean([s.speed_dropoff for s in shift_analyses]),

            # Recommendation
            'recommended_shift_length': self._calculate_optimal_shift_length(shift_analyses)
        }

    def _calculate_optimal_shift_length(self, shifts: List[ShiftAnalysis]) -> float:
        """Calculate optimal shift length based on performance data."""
        if not shifts:
            return self.config.optimal_shift_length

        # Find shift length with best xG production
        best_length = self.config.optimal_shift_length
        best_xg_rate = 0

        for target in range(30, 70, 5):  # Test 30-65 second shifts
            nearby = [s for s in shifts if abs(s.duration_seconds - target) < 10]
            if nearby:
                xg_rate = np.mean([s.xg_generated / max(s.duration_seconds, 1) for s in nearby])
                if xg_rate > best_xg_rate:
                    best_xg_rate = xg_rate
                    best_length = target

        return best_length

    # ==================== Forechecking Analysis ====================

    def get_forecheck_analysis(self, team: int) -> Dict:
        """
        Analyze team's forechecking patterns and effectiveness.

        Args:
            team: Team ID

        Returns:
            Dict with forechecking strategy analysis
        """
        events = self.team_forecheck_events[team]

        if not events:
            return {'team': team, 'no_data': True}

        # Count style usage
        style_counts = defaultdict(int)
        style_turnovers = defaultdict(int)
        style_chances = defaultdict(int)

        for event in events:
            style_counts[event.style] += 1
            if event.caused_turnover:
                style_turnovers[event.style] += 1
            if event.led_to_scoring_chance:
                style_chances[event.style] += 1

        # Calculate effectiveness by style
        styles = {}
        for style in ForecheckStyle:
            count = style_counts[style]
            if count > 0:
                styles[style.name] = {
                    'usage_count': count,
                    'usage_pct': count / len(events),
                    'turnover_rate': style_turnovers[style] / count,
                    'chance_rate': style_chances[style] / count
                }

        # Determine dominant style
        dominant_style = max(style_counts, key=style_counts.get)

        # Calculate overall aggression score
        aggression_score = (
            style_counts[ForecheckStyle.AGGRESSIVE_2_1_2] * 1.0 +
            style_counts[ForecheckStyle.STANDARD_1_2_2] * 0.6 +
            style_counts[ForecheckStyle.PASSIVE_1_4] * 0.3 +
            style_counts[ForecheckStyle.COLLAPSING] * 0.1
        ) / max(len(events), 1)

        return {
            'team': team,
            'total_forecheck_events': len(events),
            'dominant_style': dominant_style.name,
            'aggression_score': round(aggression_score, 3),
            'is_aggressive': aggression_score > self.config.aggressive_forecheck_threshold,
            'is_passive': aggression_score < self.config.passive_forecheck_threshold,

            'styles': styles,

            'overall_turnover_rate': sum(1 for e in events if e.caused_turnover) / max(len(events), 1),
            'overall_chance_rate': sum(1 for e in events if e.led_to_scoring_chance) / max(len(events), 1),

            'avg_pressure_duration': np.mean([e.pressure_duration for e in events])
        }

    # ==================== Zone Entry Analysis ====================

    def get_zone_entry_analysis(self, team: int) -> Dict:
        """
        Analyze zone entry effectiveness for a team.

        Args:
            team: Team ID

        Returns:
            Dict with zone entry strategy analysis
        """
        entries = self.team_zone_entries[team]

        if not entries:
            return {'team': team, 'no_data': True}

        # Categorize entries
        carry_ins = [e for e in entries if e.entry_type == EntryType.CARRY]
        dump_ins = [e for e in entries if e.entry_type == EntryType.DUMP]
        pass_ins = [e for e in entries if e.entry_type == EntryType.PASS]

        def analyze_entry_type(entry_list: List[ZoneEntryEvent]) -> Dict:
            if not entry_list:
                return {
                    'count': 0,
                    'success_rate': 0,
                    'shot_rate': 0,
                    'avg_xg': 0
                }
            return {
                'count': len(entry_list),
                'success_rate': sum(1 for e in entry_list if e.success) / len(entry_list),
                'shot_rate': sum(1 for e in entry_list if e.resulted_in_shot) / len(entry_list),
                'avg_xg': np.mean([e.xg_generated for e in entry_list])
            }

        carry_stats = analyze_entry_type(carry_ins)
        dump_stats = analyze_entry_type(dump_ins)
        pass_stats = analyze_entry_type(pass_ins)

        # Calculate overall efficiency
        total_entries = len(entries)
        carry_pct = len(carry_ins) / max(total_entries, 1)

        # Determine if team should carry more or dump more
        carry_value = carry_stats['success_rate'] * carry_stats['avg_xg']
        dump_value = dump_stats['success_rate'] * dump_stats['avg_xg']

        recommendation = "balanced"
        if carry_value > dump_value * 1.5:
            recommendation = "carry_more"
        elif dump_value > carry_value * 1.5:
            recommendation = "dump_more"

        return {
            'team': team,
            'total_entries': total_entries,

            'carry_in': carry_stats,
            'dump_in': dump_stats,
            'pass_in': pass_stats,

            'carry_percentage': round(carry_pct, 3),

            'vs_baseline': {
                'carry_vs_avg': carry_stats['success_rate'] - self.config.entry_success_baseline,
                'dump_vs_avg': dump_stats['success_rate'] - self.config.dump_chase_baseline
            },

            'recommendation': recommendation,
            'carry_value': round(carry_value, 4),
            'dump_value': round(dump_value, 4)
        }

    # ==================== Comprehensive Report ====================

    def get_moneyball_report(self) -> Dict:
        """
        Generate comprehensive moneyball analytics report.

        Returns:
            Dict with all moneyball insights
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'frame_count': self.frame_count,

            # Player value
            'undervalued_players': self.find_undervalued_players(),
            'overvalued_players': self.find_overvalued_players(),

            # Line combinations
            'best_performing_lines': self.get_best_line_combinations(),
            'best_value_lines': self.get_best_value_lines(),

            # Team strategies
            'team_0_forecheck': self.get_forecheck_analysis(0),
            'team_1_forecheck': self.get_forecheck_analysis(1),
            'team_0_zone_entry': self.get_zone_entry_analysis(0),
            'team_1_zone_entry': self.get_zone_entry_analysis(1),

            # Individual player reports
            'player_shift_efficiency': {
                pid: self.get_shift_efficiency_report(pid)
                for pid in self.players
            }
        }

    def export_report(self, filepath: str):
        """Export moneyball report to JSON file."""
        report = self.get_moneyball_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def update_frame(self, frame_num: int):
        """Update frame counter and line ice time."""
        self.frame_count = frame_num

        # Update ice time for current lines
        for team in [0, 1]:
            line_key = tuple(sorted(self.current_lines[team]))
            if line_key in self.line_combinations:
                self.line_combinations[line_key].ice_time_frames += 1


# ==================== Salary Data Integration ====================

# 2024 NHL salary data for common players (sample - expand as needed)
NHL_SALARY_DATA = {
    # Forwards
    "connor_mcdavid": {"name": "Connor McDavid", "salary": 12_500_000, "position": "C"},
    "auston_matthews": {"name": "Auston Matthews", "salary": 13_250_000, "position": "C"},
    "leon_draisaitl": {"name": "Leon Draisaitl", "salary": 8_500_000, "position": "C"},
    "nathan_mackinnon": {"name": "Nathan MacKinnon", "salary": 12_600_000, "position": "C"},
    "nikita_kucherov": {"name": "Nikita Kucherov", "salary": 9_500_000, "position": "RW"},
    "david_pastrnak": {"name": "David Pastrnak", "salary": 11_250_000, "position": "RW"},
    "mitch_marner": {"name": "Mitch Marner", "salary": 10_903_000, "position": "RW"},
    "sidney_crosby": {"name": "Sidney Crosby", "salary": 8_700_000, "position": "C"},
    "alex_ovechkin": {"name": "Alex Ovechkin", "salary": 9_500_000, "position": "LW"},
    "jack_eichel": {"name": "Jack Eichel", "salary": 10_000_000, "position": "C"},

    # Undervalued examples (high performers on cheaper contracts)
    "roope_hintz": {"name": "Roope Hintz", "salary": 3_150_000, "position": "C"},
    "dylan_cozens": {"name": "Dylan Cozens", "salary": 7_100_000, "position": "C"},
    "wyatt_johnston": {"name": "Wyatt Johnston", "salary": 894_167, "position": "C"},
    "matty_beniers": {"name": "Matty Beniers", "salary": 894_167, "position": "C"},
    "mason_mctavish": {"name": "Mason McTavish", "salary": 894_167, "position": "C"},

    # Defensemen
    "cale_makar": {"name": "Cale Makar", "salary": 9_000_000, "position": "D"},
    "roman_josi": {"name": "Roman Josi", "salary": 9_059_000, "position": "D"},
    "adam_fox": {"name": "Adam Fox", "salary": 9_500_000, "position": "D"},
    "erik_karlsson": {"name": "Erik Karlsson", "salary": 11_500_000, "position": "D"},
    "quinn_hughes": {"name": "Quinn Hughes", "salary": 7_850_000, "position": "D"},

    # Goalies
    "igor_shesterkin": {"name": "Igor Shesterkin", "salary": 5_666_667, "position": "G"},
    "andrei_vasilevskiy": {"name": "Andrei Vasilevskiy", "salary": 9_500_000, "position": "G"},
    "connor_hellebuyck": {"name": "Connor Hellebuyck", "salary": 8_500_000, "position": "G"},
}


def load_nhl_salary_data(analytics: MoneyballAnalytics, team_mapping: Dict[str, int]):
    """
    Load NHL salary data into analytics system.

    Args:
        analytics: MoneyballAnalytics instance
        team_mapping: Dict mapping player_id to team (0 or 1)
    """
    for player_id, data in NHL_SALARY_DATA.items():
        if player_id in team_mapping:
            analytics.add_player(
                player_id=player_id,
                name=data["name"],
                team=team_mapping[player_id],
                position=data["position"],
                salary=data["salary"]
            )


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Initialize analytics
    config = MoneyballConfig()
    analytics = MoneyballAnalytics(config)

    # Add sample players
    analytics.add_player("87", "Sidney Crosby", team=0, position="C", salary=8_700_000)
    analytics.add_player("71", "Evgeni Malkin", team=0, position="C", salary=6_100_000)
    analytics.add_player("59", "Jake Guentzel", team=0, position="LW", salary=6_000_000)
    analytics.add_player("97", "Connor McDavid", team=1, position="C", salary=12_500_000)
    analytics.add_player("29", "Leon Draisaitl", team=1, position="C", salary=8_500_000)
    analytics.add_player("93", "Ryan Nugent-Hopkins", team=1, position="LW", salary=5_125_000)

    # Simulate some events
    for frame in range(1000):
        analytics.update_frame(frame)

        # Track shifts
        if frame == 0:
            analytics.track_shift_start("87", frame)
            analytics.track_shift_start("71", frame)
            analytics.track_shift_start("59", frame)
            analytics.track_shift_start("97", frame)
            analytics.track_shift_start("29", frame)
            analytics.track_shift_start("93", frame)

        # Simulate some scoring chances
        if frame == 200:
            analytics.track_scoring_chance("87", xg=0.15, frame=frame, assisted_by=["71", "59"])
        if frame == 400:
            analytics.track_scoring_chance("97", xg=0.25, frame=frame, resulted_in_goal=True)
        if frame == 600:
            analytics.track_zone_entry("87", entry_type="carry", success=True, frame=frame, xg_generated=0.08)

        # End shifts
        if frame == 900:
            analytics.track_shift_end("87", frame)
            analytics.track_shift_end("71", frame)
            analytics.track_shift_end("59", frame)
            analytics.track_shift_end("97", frame)
            analytics.track_shift_end("29", frame)
            analytics.track_shift_end("93", frame)

    print("=" * 60)
    print("HOCKEY MONEYBALL ANALYTICS")
    print("=" * 60)

    # Print player value metrics
    print("\n📊 PLAYER VALUE METRICS:")
    print("-" * 60)

    for player_id in ["87", "97"]:
        metrics = analytics.get_player_value_metrics(player_id)
        print(f"\n{metrics['name']} ({metrics['position']}):")
        print(f"  Salary: ${metrics['salary']:,}")
        print(f"  xG/60: {metrics['xg_per_60']:.3f}")
        print(f"  xA/60: {metrics['xa_per_60']:.3f}")
        print(f"  Value Over Replacement: {metrics['total_vor']:.3f}")
        print(f"  Value/Million: {metrics['value_per_million']:.3f}")
        print(f"  Undervalued: {metrics['is_undervalued']}")

    # Print line combinations
    print("\n🏒 LINE COMBINATION ANALYSIS:")
    print("-" * 60)

    best_lines = analytics.get_best_line_combinations(min_ice_time=0.1)
    for line in best_lines:
        print(f"\n{' - '.join(line['players'])}:")
        print(f"  Ice Time: {line['ice_time_minutes']:.1f} min")
        print(f"  Combined Salary: ${line['combined_salary']:,}")
        print(f"  xG For/60: {line['xg_for_60']:.3f}")
        print(f"  Value/Million: {line['value_per_million']:.3f}")

    print("\n" + "=" * 60)
    print("Moneyball Analytics Ready!")
    print("Use analytics.get_moneyball_report() for full analysis")
    print("=" * 60)
