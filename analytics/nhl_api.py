"""
NHL API Integration for Hockey Analytics

This module provides utilities for fetching data from the NHL API
and NHL EDGE tracking system.

Data available:
- Play-by-play events
- Shift data
- Player statistics
- Game schedules
- NHL EDGE tracking metrics (skating speed, distance, etc.)

References:
- nhl-api-py Python library
- NHL.com/EDGE tracking documentation
- NHL API endpoints
"""

import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import time


class GameType(Enum):
    """NHL game types."""
    PRESEASON = 1
    REGULAR = 2
    PLAYOFFS = 3
    ALLSTAR = 4


class EventType(Enum):
    """NHL play-by-play event types."""
    FACEOFF = "faceoff"
    HIT = "hit"
    SHOT = "shot"
    GOAL = "goal"
    BLOCKED_SHOT = "blocked-shot"
    MISSED_SHOT = "missed-shot"
    GIVEAWAY = "giveaway"
    TAKEAWAY = "takeaway"
    PENALTY = "penalty"
    STOPPAGE = "stoppage"
    PERIOD_START = "period-start"
    PERIOD_END = "period-end"
    GAME_END = "game-end"


@dataclass
class PlayEvent:
    """Represents a play-by-play event."""
    event_id: int
    event_type: str
    period: int
    time_in_period: str
    time_remaining: str
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None

    # Players involved
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    player2_id: Optional[int] = None
    player2_name: Optional[str] = None

    # Team info
    team_id: Optional[int] = None
    team_abbrev: Optional[str] = None

    # Additional details
    description: str = ""
    shot_type: Optional[str] = None
    zone_code: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerEdgeStats:
    """NHL EDGE tracking stats for a player."""
    player_id: int
    player_name: str
    team_id: int
    position: str

    # Skating metrics
    avg_skating_speed: float = 0.0  # mph
    top_skating_speed: float = 0.0  # mph
    total_distance: float = 0.0  # feet

    # Bursts
    speed_bursts: int = 0  # 20+ mph efforts
    top_speed_bursts: int = 0  # 22+ mph efforts

    # Zone time
    offensive_zone_time: float = 0.0  # percentage
    neutral_zone_time: float = 0.0
    defensive_zone_time: float = 0.0

    # Shot metrics
    avg_shot_speed: float = 0.0  # mph
    top_shot_speed: float = 0.0  # mph

    games_played: int = 0


@dataclass
class GoalieEdgeStats:
    """NHL EDGE tracking stats for a goalie."""
    player_id: int
    player_name: str
    team_id: int

    # Movement metrics
    avg_lateral_movement: float = 0.0  # feet per save
    lateral_distance: float = 0.0  # total feet

    # Positioning
    avg_depth: float = 0.0  # feet from goal line

    games_played: int = 0


class NHLAPIClient:
    """
    Client for accessing NHL API data.

    Note: This is a mock/example implementation. In production,
    use the nhl-api-py library or direct API calls.
    """

    def __init__(
        self,
        base_url: str = "https://api-web.nhle.com/v1",
        rate_limit: float = 0.5,  # seconds between requests
    ):
        """
        Initialize the NHL API client.

        Args:
            base_url: Base URL for NHL API
            rate_limit: Minimum time between requests
        """
        self.base_url = base_url
        self.rate_limit = rate_limit
        self._last_request_time = 0

        # Cache for common data
        self._team_cache: Dict[int, Dict] = {}
        self._player_cache: Dict[int, Dict] = {}

    def _rate_limit_wait(self):
        """Wait if needed to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """
        Make an API request.

        In production, this would use requests library.
        Here we provide mock data structure.
        """
        self._rate_limit_wait()

        # Mock response - in production, use:
        # import requests
        # response = requests.get(f"{self.base_url}/{endpoint}")
        # return response.json()

        return {"mock": True, "endpoint": endpoint}

    def get_schedule(
        self,
        date_str: Optional[str] = None,
        team_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get game schedule.

        Args:
            date_str: Date in YYYY-MM-DD format
            team_id: Optional team filter

        Returns:
            List of game dictionaries
        """
        if date_str is None:
            date_str = date.today().isoformat()

        endpoint = f"schedule/{date_str}"

        # Mock response structure
        return [{
            'game_id': 2023020001,
            'date': date_str,
            'home_team': {'id': 1, 'abbrev': 'NJD'},
            'away_team': {'id': 2, 'abbrev': 'NYR'},
            'status': 'scheduled',
        }]

    def get_play_by_play(self, game_id: int) -> List[PlayEvent]:
        """
        Get play-by-play data for a game.

        Args:
            game_id: NHL game ID

        Returns:
            List of PlayEvent objects
        """
        endpoint = f"gamecenter/{game_id}/play-by-play"

        # Mock play-by-play structure
        events = []

        # Example events
        sample_events = [
            {
                'event_id': 1,
                'event_type': 'faceoff',
                'period': 1,
                'time_in_period': '00:00',
                'time_remaining': '20:00',
                'x_coord': 0.0,
                'y_coord': 0.0,
                'description': 'Faceoff won by Home',
            },
            {
                'event_id': 2,
                'event_type': 'shot',
                'period': 1,
                'time_in_period': '05:30',
                'time_remaining': '14:30',
                'x_coord': 78.0,
                'y_coord': 12.0,
                'shot_type': 'wrist',
                'description': 'Wrist shot by Player saved',
            },
        ]

        for event_data in sample_events:
            events.append(PlayEvent(**event_data))

        return events

    def get_shift_data(self, game_id: int) -> List[Dict[str, Any]]:
        """
        Get shift data for a game.

        Args:
            game_id: NHL game ID

        Returns:
            List of shift dictionaries
        """
        endpoint = f"gamecenter/{game_id}/shiftchart"

        # Mock shift data
        return [{
            'player_id': 8478402,
            'player_name': 'Connor McDavid',
            'period': 1,
            'shift_number': 1,
            'start_time': '00:30',
            'end_time': '01:15',
            'duration': 45,
            'events': ['shot'],
        }]

    def get_player_stats(
        self,
        player_id: int,
        season: int = 20232024,
        game_type: GameType = GameType.REGULAR
    ) -> Dict[str, Any]:
        """
        Get player statistics.

        Args:
            player_id: NHL player ID
            season: Season in YYYYYYYY format
            game_type: Type of games

        Returns:
            Player statistics dictionary
        """
        endpoint = f"player/{player_id}/landing"

        # Mock player stats
        return {
            'player_id': player_id,
            'name': 'Sample Player',
            'position': 'C',
            'games_played': 82,
            'goals': 30,
            'assists': 50,
            'points': 80,
            'plus_minus': 15,
            'toi_per_game': '20:30',
        }

    def get_edge_skater_stats(
        self,
        player_id: int,
        season: int = 20232024,
        game_type: GameType = GameType.REGULAR
    ) -> PlayerEdgeStats:
        """
        Get NHL EDGE tracking stats for a skater.

        Args:
            player_id: NHL player ID
            season: Season in YYYYYYYY format
            game_type: Type of games

        Returns:
            PlayerEdgeStats object
        """
        # Mock EDGE data - in production, use actual API
        return PlayerEdgeStats(
            player_id=player_id,
            player_name='Sample Player',
            team_id=1,
            position='C',
            avg_skating_speed=14.5,
            top_skating_speed=23.8,
            total_distance=350000.0,  # feet per season
            speed_bursts=450,
            top_speed_bursts=120,
            offensive_zone_time=35.0,
            neutral_zone_time=30.0,
            defensive_zone_time=35.0,
            avg_shot_speed=72.0,
            top_shot_speed=95.0,
            games_played=82,
        )

    def get_edge_goalie_stats(
        self,
        player_id: int,
        season: int = 20232024
    ) -> GoalieEdgeStats:
        """
        Get NHL EDGE tracking stats for a goalie.

        Args:
            player_id: NHL player ID
            season: Season

        Returns:
            GoalieEdgeStats object
        """
        return GoalieEdgeStats(
            player_id=player_id,
            player_name='Sample Goalie',
            team_id=1,
            avg_lateral_movement=2.5,
            lateral_distance=15000.0,
            avg_depth=4.2,
            games_played=60,
        )

    def get_game_boxscore(self, game_id: int) -> Dict[str, Any]:
        """
        Get game boxscore.

        Args:
            game_id: NHL game ID

        Returns:
            Boxscore dictionary
        """
        return {
            'game_id': game_id,
            'home_team': {
                'id': 1,
                'score': 3,
                'shots': 30,
                'hits': 25,
                'blocked_shots': 12,
            },
            'away_team': {
                'id': 2,
                'score': 2,
                'shots': 28,
                'hits': 22,
                'blocked_shots': 15,
            },
        }


class PlayByPlayParser:
    """
    Parser for converting NHL play-by-play data to analytics format.
    """

    def __init__(self):
        """Initialize the parser."""
        self.coordinate_transform = True  # Transform to standard coordinates

    def parse_events(
        self,
        events: List[PlayEvent]
    ) -> List[Dict[str, Any]]:
        """
        Parse play-by-play events into analytics format.

        Args:
            events: List of PlayEvent objects

        Returns:
            List of parsed event dictionaries
        """
        parsed = []

        for event in events:
            parsed_event = {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'period': event.period,
                'time': event.time_in_period,
                'x': event.x_coord,
                'y': event.y_coord,
                'player_id': event.player_id,
                'team_id': event.team_id,
            }

            # Transform coordinates if needed
            if self.coordinate_transform and event.x_coord is not None:
                parsed_event['x'], parsed_event['y'] = self._transform_coords(
                    event.x_coord, event.y_coord, event.period
                )

            # Add shot-specific fields
            if event.event_type in ['shot', 'goal', 'blocked-shot', 'missed-shot']:
                parsed_event['shot_type'] = event.shot_type
                parsed_event['is_shot'] = True
                parsed_event['is_goal'] = event.event_type == 'goal'

            parsed.append(parsed_event)

        return parsed

    def _transform_coords(
        self,
        x: float,
        y: float,
        period: int
    ) -> Tuple[float, float]:
        """
        Transform NHL coordinates to standard (0-200, 0-85).

        NHL uses center ice as origin, with x in (-100, 100).
        """
        # NHL coords: center ice = (0, 0)
        # Standard coords: (0, 0) at goal line

        # Shift x to 0-200 range
        x_transformed = x + 100

        # Shift y to 0-85 range
        y_transformed = y + 42.5

        return x_transformed, y_transformed

    def extract_shot_data(
        self,
        events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract shot-specific data for xG modeling.

        Args:
            events: Parsed event list

        Returns:
            List of shot dictionaries
        """
        shots = []

        for event in events:
            if event.get('is_shot'):
                shot = {
                    'x': event.get('x', 0),
                    'y': event.get('y', 0),
                    'shot_type': event.get('shot_type'),
                    'is_goal': event.get('is_goal', False),
                    'period': event.get('period'),
                    'player_id': event.get('player_id'),
                    'team_id': event.get('team_id'),
                }

                # Calculate derived features
                shot['distance'] = self._calculate_shot_distance(
                    shot['x'], shot['y']
                )
                shot['angle'] = self._calculate_shot_angle(
                    shot['x'], shot['y']
                )

                shots.append(shot)

        return shots

    def _calculate_shot_distance(self, x: float, y: float) -> float:
        """Calculate distance from shot location to goal."""
        goal_x = 189  # Goal line at 189 feet from far end
        goal_y = 42.5  # Center of rink

        return ((x - goal_x) ** 2 + (y - goal_y) ** 2) ** 0.5

    def _calculate_shot_angle(self, x: float, y: float) -> float:
        """Calculate angle from shot location to goal center."""
        import math

        goal_x = 189
        goal_y = 42.5

        dx = goal_x - x
        dy = goal_y - y

        angle = math.degrees(math.atan2(dy, dx))

        return angle


def fetch_season_data(
    season: int = 2023,
    game_type: GameType = GameType.REGULAR
) -> Dict[str, Any]:
    """
    Fetch data for an entire season.

    This is a high-level utility function that orchestrates
    multiple API calls to gather comprehensive season data.

    Args:
        season: Starting year of season
        game_type: Type of games to fetch

    Returns:
        Dictionary with season data
    """
    client = NHLAPIClient()
    parser = PlayByPlayParser()

    # This would iterate through the season in production
    # Here we provide a template structure

    season_data = {
        'season': f"{season}{season+1}",
        'game_type': game_type.value,
        'games': [],
        'players': {},
        'teams': {},
    }

    return season_data


if __name__ == "__main__":
    # Demo the NHL API client
    print("Initializing NHL API Client...")

    client = NHLAPIClient()
    parser = PlayByPlayParser()

    # Get schedule
    print("\nFetching schedule...")
    schedule = client.get_schedule()
    print(f"  Found {len(schedule)} games")

    # Get play-by-play
    print("\nFetching play-by-play...")
    game_id = 2023020001
    events = client.get_play_by_play(game_id)
    print(f"  Found {len(events)} events")

    # Parse events
    print("\nParsing events...")
    parsed = parser.parse_events(events)
    print(f"  Parsed {len(parsed)} events")

    # Get EDGE stats
    print("\nFetching EDGE stats...")
    edge_stats = client.get_edge_skater_stats(8478402)  # McDavid
    print(f"  Player: {edge_stats.player_name}")
    print(f"  Top speed: {edge_stats.top_skating_speed} mph")
    print(f"  Speed bursts: {edge_stats.speed_bursts}")
    print(f"  Avg shot speed: {edge_stats.avg_shot_speed} mph")

    # Goalie EDGE stats
    print("\nFetching Goalie EDGE stats...")
    goalie_stats = client.get_edge_goalie_stats(8479394)  # Sample goalie
    print(f"  Avg depth: {goalie_stats.avg_depth} ft")
    print(f"  Avg lateral movement: {goalie_stats.avg_lateral_movement} ft")

    # Extract shots
    print("\nExtracting shot data...")
    shots = parser.extract_shot_data(parsed)
    print(f"  Found {len(shots)} shots")
    for shot in shots:
        print(f"    {shot}")


# Type alias for importing
from typing import Tuple
Tuple = Tuple  # Re-export for backwards compatibility
