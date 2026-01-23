"""
Fatigue and Acute:Chronic Workload Ratio (ACWR) Model for Hockey Analytics

This module implements workload monitoring and injury prediction models
translated from soccer/football research.

Key concepts:
- Acute to Chronic Workload Ratio (ACWR)
- Deviation of maximum from average (DEV)
- Exponentially weighted moving averages
- Non-contact injury risk prediction

Hockey application:
- Ice time patterns (back-to-backs, third period fatigue)
- Shift length and recovery time
- Travel schedule impact
- Cumulative hits/blocked shots stress

References:
- "Machine Learning for Understanding and Predicting Injuries in Football" (2022)
- "Machine Learning Outperforms Logistic Regression for NHL Injury Prediction"
- Acute:Chronic Workload Ratio research (Gabbett, 2016)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json


@dataclass
class GameWorkload:
    """Workload data for a single game."""
    game_id: str
    game_date: datetime
    player_id: str
    team_id: str

    # Ice time metrics
    total_ice_time: float  # minutes
    even_strength_time: float
    power_play_time: float
    penalty_kill_time: float
    shift_count: int
    avg_shift_length: float  # seconds

    # EDGE tracking data (if available)
    distance_skated: Optional[float] = None  # feet
    top_speed: Optional[float] = None  # mph
    speed_bursts: Optional[int] = None  # High intensity efforts
    avg_speed: Optional[float] = None  # mph

    # Physical contact
    hits_given: int = 0
    hits_received: int = 0
    blocked_shots: int = 0
    fights: int = 0

    # Performance metrics
    shots: int = 0
    goals: int = 0
    assists: int = 0
    takeaways: int = 0
    giveaways: int = 0
    faceoff_attempts: int = 0

    # Game context
    is_home: bool = True
    is_back_to_back: bool = False
    days_rest: int = 1
    travel_distance: Optional[float] = None  # miles

    # Outcome
    is_injured_after: bool = False
    injury_type: Optional[str] = None

    # Computed workload score (filled by model)
    workload_score: Optional[float] = None
    fatigue_level: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerWorkloadProfile:
    """Workload profile for a player over time."""
    player_id: str
    team_id: str
    position: str = "F"

    # Rolling averages
    acute_workload: float = 0.0  # 7-day average
    chronic_workload: float = 0.0  # 28-day average
    acwr: float = 1.0  # Acute:Chronic ratio

    # Workload metrics
    avg_ice_time: float = 0.0
    avg_shifts: float = 0.0
    avg_distance: float = 0.0
    avg_hits: float = 0.0
    avg_blocked_shots: float = 0.0

    # Risk assessment
    injury_risk_score: float = 0.0
    fatigue_accumulation: float = 0.0
    recommended_rest: int = 0  # days

    # History
    games_played: int = 0
    games_in_acute_window: int = 0
    games_in_chronic_window: int = 0
    injury_history: List[Dict[str, Any]] = field(default_factory=list)


class WorkloadCalculator:
    """
    Calculates workload scores from game data.

    Workload is a composite measure of physical demand including:
    - Ice time
    - Skating distance
    - Physical contact (hits, blocks)
    - High-intensity efforts
    """

    def __init__(
        self,
        ice_time_weight: float = 0.25,
        distance_weight: float = 0.20,
        hits_weight: float = 0.15,
        blocks_weight: float = 0.15,
        intensity_weight: float = 0.15,
        context_weight: float = 0.10,
    ):
        """
        Initialize the workload calculator.

        Args:
            ice_time_weight: Weight for ice time component
            distance_weight: Weight for skating distance
            hits_weight: Weight for physical contact (hits)
            blocks_weight: Weight for blocked shots
            intensity_weight: Weight for high-intensity efforts
            context_weight: Weight for game context (B2B, travel)
        """
        self.weights = {
            'ice_time': ice_time_weight,
            'distance': distance_weight,
            'hits': hits_weight,
            'blocks': blocks_weight,
            'intensity': intensity_weight,
            'context': context_weight,
        }

        # Normalization factors (typical values)
        self.norms = {
            'ice_time': 20.0,    # 20 minutes = average
            'distance': 15000.0, # 15000 feet = ~2.8 miles
            'hits': 3.0,         # 3 hits per game
            'blocks': 1.5,       # 1.5 blocks per game
            'speed_bursts': 15.0, # 15 bursts per game
        }

    def calculate_workload(self, game: GameWorkload) -> float:
        """
        Calculate workload score for a game.

        Args:
            game: GameWorkload object with game data

        Returns:
            Workload score (0-2+ scale, 1.0 = average)
        """
        components = {}

        # Ice time component
        components['ice_time'] = game.total_ice_time / self.norms['ice_time']

        # Distance component (if available)
        if game.distance_skated is not None:
            components['distance'] = game.distance_skated / self.norms['distance']
        else:
            # Estimate from ice time
            components['distance'] = game.total_ice_time / self.norms['ice_time']

        # Physical contact
        total_contact = game.hits_given + game.hits_received
        components['hits'] = total_contact / self.norms['hits']
        components['blocks'] = game.blocked_shots / self.norms['blocks']

        # Intensity (speed bursts or estimate)
        if game.speed_bursts is not None:
            components['intensity'] = game.speed_bursts / self.norms['speed_bursts']
        else:
            # Estimate from shift count (more shifts = more high-intensity)
            components['intensity'] = game.shift_count / 25.0

        # Context modifier
        context_modifier = 1.0
        if game.is_back_to_back:
            context_modifier += 0.15  # Higher effective load
        if game.days_rest == 0:
            context_modifier += 0.10
        if game.travel_distance and game.travel_distance > 1000:
            context_modifier += 0.05  # Long travel

        components['context'] = context_modifier

        # Weighted sum
        workload = 0.0
        for component, value in components.items():
            if component in self.weights:
                workload += self.weights[component] * value

        return float(workload)


class ACWRModel:
    """
    Acute:Chronic Workload Ratio Model.

    ACWR is a key metric for injury risk assessment:
    - Acute workload: Recent (7-day) training load
    - Chronic workload: Longer-term (28-day) training load
    - ACWR = Acute / Chronic

    Risk zones:
    - ACWR < 0.8: Undertraining (detraining risk)
    - ACWR 0.8-1.3: Optimal training zone
    - ACWR 1.3-1.5: Moderate risk
    - ACWR > 1.5: High injury risk (spike)
    """

    def __init__(
        self,
        acute_window: int = 7,    # days
        chronic_window: int = 28,  # days
        ewma_decay: float = 0.1,   # Exponential decay factor
        use_ewma: bool = True,     # Use exponentially weighted averages
    ):
        """
        Initialize the ACWR model.

        Args:
            acute_window: Days for acute workload calculation
            chronic_window: Days for chronic workload calculation
            ewma_decay: Decay factor for exponential weighting
            use_ewma: Whether to use EWMA vs simple rolling average
        """
        self.acute_window = acute_window
        self.chronic_window = chronic_window
        self.ewma_decay = ewma_decay
        self.use_ewma = use_ewma

        self.workload_calculator = WorkloadCalculator()
        self.player_history: Dict[str, List[GameWorkload]] = defaultdict(list)
        self.player_profiles: Dict[str, PlayerWorkloadProfile] = {}

    def add_game(self, game: GameWorkload):
        """
        Add a game to player's workload history.

        Args:
            game: GameWorkload object
        """
        # Calculate workload score
        game.workload_score = self.workload_calculator.calculate_workload(game)

        # Add to history
        self.player_history[game.player_id].append(game)

        # Sort by date
        self.player_history[game.player_id].sort(key=lambda g: g.game_date)

        # Update profile
        self._update_profile(game.player_id)

    def _update_profile(self, player_id: str):
        """Update player's workload profile."""
        history = self.player_history.get(player_id, [])

        if not history:
            return

        if player_id not in self.player_profiles:
            self.player_profiles[player_id] = PlayerWorkloadProfile(
                player_id=player_id,
                team_id=history[-1].team_id
            )

        profile = self.player_profiles[player_id]
        latest_game = history[-1]
        cutoff_date = latest_game.game_date

        # Filter to relevant windows
        acute_games = [
            g for g in history
            if (cutoff_date - g.game_date).days <= self.acute_window
        ]
        chronic_games = [
            g for g in history
            if (cutoff_date - g.game_date).days <= self.chronic_window
        ]

        # Calculate workloads
        if self.use_ewma:
            profile.acute_workload = self._calculate_ewma(acute_games, cutoff_date)
            profile.chronic_workload = self._calculate_ewma(chronic_games, cutoff_date)
        else:
            profile.acute_workload = self._calculate_average(acute_games)
            profile.chronic_workload = self._calculate_average(chronic_games)

        # Calculate ACWR
        if profile.chronic_workload > 0:
            profile.acwr = profile.acute_workload / profile.chronic_workload
        else:
            profile.acwr = 1.0

        # Update counts
        profile.games_played = len(history)
        profile.games_in_acute_window = len(acute_games)
        profile.games_in_chronic_window = len(chronic_games)

        # Calculate averages
        if chronic_games:
            profile.avg_ice_time = np.mean([g.total_ice_time for g in chronic_games])
            profile.avg_shifts = np.mean([g.shift_count for g in chronic_games])
            profile.avg_hits = np.mean([g.hits_given + g.hits_received for g in chronic_games])
            profile.avg_blocked_shots = np.mean([g.blocked_shots for g in chronic_games])

            distances = [g.distance_skated for g in chronic_games if g.distance_skated]
            if distances:
                profile.avg_distance = np.mean(distances)

        # Calculate injury risk
        profile.injury_risk_score = self._calculate_injury_risk(profile)

        # Fatigue accumulation
        profile.fatigue_accumulation = self._calculate_fatigue_accumulation(history, cutoff_date)

        # Recommend rest if needed
        profile.recommended_rest = self._recommend_rest(profile)

    def _calculate_ewma(self, games: List[GameWorkload], reference_date: datetime) -> float:
        """Calculate exponentially weighted moving average."""
        if not games:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for game in games:
            days_ago = (reference_date - game.game_date).days
            weight = np.exp(-self.ewma_decay * days_ago)
            weighted_sum += (game.workload_score or 0) * weight
            total_weight += weight

        return weighted_sum / max(total_weight, 1e-6)

    def _calculate_average(self, games: List[GameWorkload]) -> float:
        """Calculate simple rolling average."""
        if not games:
            return 0.0

        workloads = [g.workload_score or 0 for g in games]
        return np.mean(workloads)

    def _calculate_injury_risk(self, profile: PlayerWorkloadProfile) -> float:
        """
        Calculate injury risk score based on ACWR and other factors.

        Returns:
            Risk score (0-1 scale)
        """
        risk = 0.0

        # ACWR risk (main factor)
        if profile.acwr < 0.8:
            # Undertraining risk
            risk += 0.2 * (0.8 - profile.acwr)
        elif profile.acwr > 1.5:
            # High spike risk
            risk += 0.4 * min((profile.acwr - 1.5), 0.5)
        elif profile.acwr > 1.3:
            # Moderate spike
            risk += 0.2 * (profile.acwr - 1.3)

        # Games played frequency risk
        if profile.games_in_acute_window >= 4:  # 4+ games in 7 days
            risk += 0.1

        # Fatigue accumulation risk
        risk += 0.2 * min(profile.fatigue_accumulation, 1.0)

        # Contact sport risk (hits and blocks)
        if profile.avg_hits > 5 or profile.avg_blocked_shots > 3:
            risk += 0.1

        return min(risk, 1.0)

    def _calculate_fatigue_accumulation(
        self,
        history: List[GameWorkload],
        reference_date: datetime,
        decay_rate: float = 0.15
    ) -> float:
        """
        Calculate accumulated fatigue with decay.

        Args:
            history: Game history
            reference_date: Current date
            decay_rate: Daily fatigue decay rate

        Returns:
            Accumulated fatigue score
        """
        fatigue = 0.0

        for game in history:
            days_ago = (reference_date - game.game_date).days
            if days_ago > 30:  # Only consider last 30 days
                continue

            # Add workload, then decay
            game_fatigue = (game.workload_score or 0)

            # Modifiers
            if game.is_back_to_back:
                game_fatigue *= 1.2

            # Decay based on days since
            game_fatigue *= np.exp(-decay_rate * days_ago)

            fatigue += game_fatigue

        # Normalize to 0-1 scale (rough estimate)
        return min(fatigue / 5.0, 1.0)

    def _recommend_rest(self, profile: PlayerWorkloadProfile) -> int:
        """
        Recommend rest days based on workload profile.

        Returns:
            Recommended rest days
        """
        if profile.acwr > 1.5 or profile.injury_risk_score > 0.7:
            return 3
        elif profile.acwr > 1.3 or profile.injury_risk_score > 0.5:
            return 2
        elif profile.fatigue_accumulation > 0.7:
            return 1
        else:
            return 0

    def get_team_fatigue_report(
        self,
        team_id: str,
        as_of_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate fatigue report for entire team.

        Args:
            team_id: Team identifier
            as_of_date: Date for report (defaults to latest)

        Returns:
            DataFrame with player fatigue metrics
        """
        reports = []

        for player_id, profile in self.player_profiles.items():
            if profile.team_id == team_id:
                reports.append({
                    'player_id': player_id,
                    'games_played': profile.games_played,
                    'acute_workload': profile.acute_workload,
                    'chronic_workload': profile.chronic_workload,
                    'acwr': profile.acwr,
                    'injury_risk': profile.injury_risk_score,
                    'fatigue': profile.fatigue_accumulation,
                    'recommended_rest': profile.recommended_rest,
                    'avg_ice_time': profile.avg_ice_time,
                    'risk_zone': self._classify_risk_zone(profile.acwr),
                })

        df = pd.DataFrame(reports)
        if not df.empty:
            df = df.sort_values('injury_risk', ascending=False)

        return df

    def _classify_risk_zone(self, acwr: float) -> str:
        """Classify ACWR into risk zones."""
        if acwr < 0.8:
            return "undertrained"
        elif acwr <= 1.3:
            return "optimal"
        elif acwr <= 1.5:
            return "moderate_risk"
        else:
            return "high_risk"

    def predict_injury_probability(
        self,
        player_id: str,
        games_ahead: int = 5
    ) -> Dict[str, float]:
        """
        Predict injury probability for upcoming games.

        Uses ACWR, fatigue, and historical patterns.

        Args:
            player_id: Player to predict for
            games_ahead: Number of games to predict

        Returns:
            Dictionary with injury probabilities
        """
        if player_id not in self.player_profiles:
            return {'error': 'No data for player'}

        profile = self.player_profiles[player_id]

        # Base probability from current risk score
        base_prob = profile.injury_risk_score * 0.15  # 15% max base rate

        # Adjust for injury history
        history_multiplier = 1.0 + len(profile.injury_history) * 0.1

        # Predict for each game
        predictions = {}
        cumulative_prob = 0.0

        for game_num in range(1, games_ahead + 1):
            # Probability increases with cumulative games
            game_prob = base_prob * history_multiplier * (1 + game_num * 0.05)
            game_prob = min(game_prob, 0.3)  # Cap at 30%

            predictions[f'game_{game_num}'] = game_prob
            cumulative_prob = 1 - (1 - cumulative_prob) * (1 - game_prob)

        predictions['any_injury_probability'] = cumulative_prob

        return predictions

    def to_dict(self) -> Dict[str, Any]:
        """Export model to dictionary."""
        return {
            'acute_window': self.acute_window,
            'chronic_window': self.chronic_window,
            'ewma_decay': self.ewma_decay,
            'use_ewma': self.use_ewma,
            'num_players': len(self.player_profiles),
            'profiles': {
                pid: {
                    'acwr': p.acwr,
                    'acute_workload': p.acute_workload,
                    'chronic_workload': p.chronic_workload,
                    'injury_risk': p.injury_risk_score,
                    'games_played': p.games_played,
                }
                for pid, p in self.player_profiles.items()
            }
        }


class ThirdPeriodFatigueAnalyzer:
    """
    Analyzes third period fatigue effects on performance.

    Hockey-specific fatigue patterns:
    - Third period performance drop
    - Shift length degradation
    - Speed/skating decline
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.period_performance: Dict[str, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add_period_performance(
        self,
        player_id: str,
        period: int,
        performance_score: float,
        avg_speed: Optional[float] = None,
        shift_length: Optional[float] = None
    ):
        """Add period performance data."""
        self.period_performance[player_id][period].append({
            'score': performance_score,
            'speed': avg_speed,
            'shift_length': shift_length,
        })

    def calculate_fatigue_index(self, player_id: str) -> Dict[str, float]:
        """
        Calculate fatigue index comparing 3rd period to 1st period.

        Returns:
            Dictionary with fatigue metrics
        """
        if player_id not in self.period_performance:
            return {'error': 'No data for player'}

        data = self.period_performance[player_id]

        if not data[1] or not data[3]:
            return {'insufficient_data': True}

        # Average scores by period
        p1_scores = [d['score'] for d in data[1]]
        p3_scores = [d['score'] for d in data[3]]

        p1_avg = np.mean(p1_scores)
        p3_avg = np.mean(p3_scores)

        # Fatigue index: how much performance drops
        fatigue_index = (p1_avg - p3_avg) / max(p1_avg, 0.01)

        # Speed decline
        p1_speeds = [d['speed'] for d in data[1] if d['speed']]
        p3_speeds = [d['speed'] for d in data[3] if d['speed']]

        speed_decline = 0.0
        if p1_speeds and p3_speeds:
            speed_decline = (np.mean(p1_speeds) - np.mean(p3_speeds)) / np.mean(p1_speeds)

        # Shift length decline
        p1_shifts = [d['shift_length'] for d in data[1] if d['shift_length']]
        p3_shifts = [d['shift_length'] for d in data[3] if d['shift_length']]

        shift_decline = 0.0
        if p1_shifts and p3_shifts:
            shift_decline = (np.mean(p1_shifts) - np.mean(p3_shifts)) / np.mean(p1_shifts)

        return {
            'fatigue_index': fatigue_index,
            'performance_p1': p1_avg,
            'performance_p3': p3_avg,
            'speed_decline': speed_decline,
            'shift_length_decline': shift_decline,
            'samples_p1': len(p1_scores),
            'samples_p3': len(p3_scores),
        }


def create_sample_workload_data(n_games: int = 50) -> List[GameWorkload]:
    """
    Create sample workload data for testing.

    Returns:
        List of GameWorkload objects
    """
    np.random.seed(42)

    games = []
    base_date = datetime(2024, 10, 1)

    for i in range(n_games):
        game_date = base_date + timedelta(days=i * 2 + np.random.randint(0, 2))

        # Check for back-to-back
        is_b2b = np.random.random() < 0.2

        game = GameWorkload(
            game_id=f"game_{i}",
            game_date=game_date,
            player_id="player_1",
            team_id="team_a",
            total_ice_time=np.random.uniform(15, 25),
            even_strength_time=np.random.uniform(12, 20),
            power_play_time=np.random.uniform(0, 4),
            penalty_kill_time=np.random.uniform(0, 3),
            shift_count=np.random.randint(18, 30),
            avg_shift_length=np.random.uniform(40, 60),
            distance_skated=np.random.uniform(12000, 18000),
            top_speed=np.random.uniform(20, 26),
            speed_bursts=np.random.randint(10, 25),
            avg_speed=np.random.uniform(12, 16),
            hits_given=np.random.randint(0, 5),
            hits_received=np.random.randint(0, 4),
            blocked_shots=np.random.randint(0, 3),
            shots=np.random.randint(0, 5),
            goals=np.random.randint(0, 2),
            assists=np.random.randint(0, 2),
            is_back_to_back=is_b2b,
            days_rest=0 if is_b2b else np.random.randint(1, 4),
            travel_distance=np.random.uniform(0, 2000),
        )

        games.append(game)

    return games


if __name__ == "__main__":
    # Demo the ACWR model
    print("Creating ACWR Model...")

    model = ACWRModel()

    # Create sample data
    print("Generating sample workload data...")
    games = create_sample_workload_data(50)

    for game in games:
        model.add_game(game)

    # Get player profile
    print("\nPlayer Workload Profile:")
    profile = model.player_profiles.get("player_1")
    if profile:
        print(f"  Games played: {profile.games_played}")
        print(f"  Acute workload: {profile.acute_workload:.3f}")
        print(f"  Chronic workload: {profile.chronic_workload:.3f}")
        print(f"  ACWR: {profile.acwr:.3f}")
        print(f"  Risk zone: {model._classify_risk_zone(profile.acwr)}")
        print(f"  Injury risk score: {profile.injury_risk_score:.3f}")
        print(f"  Fatigue accumulation: {profile.fatigue_accumulation:.3f}")
        print(f"  Recommended rest: {profile.recommended_rest} days")

    # Team fatigue report
    print("\nTeam Fatigue Report:")
    report = model.get_team_fatigue_report("team_a")
    if not report.empty:
        print(report.to_string())

    # Injury prediction
    print("\nInjury Probability Prediction:")
    predictions = model.predict_injury_probability("player_1", games_ahead=5)
    for key, value in predictions.items():
        print(f"  {key}: {value:.3f}")

    # Third period fatigue
    print("\nThird Period Fatigue Analysis:")
    fatigue_analyzer = ThirdPeriodFatigueAnalyzer()

    for i in range(20):
        for period in [1, 2, 3]:
            # Performance drops in 3rd period
            base_score = 1.0 - (period - 1) * 0.1
            score = base_score + np.random.normal(0, 0.1)
            speed = 15 - (period - 1) * 0.5 + np.random.normal(0, 0.5)

            fatigue_analyzer.add_period_performance(
                "player_1", period, score, avg_speed=speed
            )

    fatigue_metrics = fatigue_analyzer.calculate_fatigue_index("player_1")
    print(f"  Fatigue index: {fatigue_metrics.get('fatigue_index', 0):.3f}")
    print(f"  Speed decline: {fatigue_metrics.get('speed_decline', 0):.1%}")
    print(f"  P1 performance: {fatigue_metrics.get('performance_p1', 0):.3f}")
    print(f"  P3 performance: {fatigue_metrics.get('performance_p3', 0):.3f}")
