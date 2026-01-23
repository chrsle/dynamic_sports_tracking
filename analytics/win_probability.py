"""
Win Probability Model for Hockey Analytics

This module implements a Bayesian in-game win probability model adapted from soccer
analytics research (Robberechts et al., 2021) and NFL win probability models.

The model calculates real-time win probability based on:
- Current score differential
- Time remaining
- Strength state (5v5, PP, PK, etc.)
- Recent momentum (xT generated, shots, zone time)
- Goalie pull scenarios

References:
    - Robberechts, P., Van Haaren, J., & Davis, J. (2021). "A Bayesian Approach
      to In-Game Win Probability in Soccer." ACM SIGKDD.
    - Lock, D., & Nettleton, D. (2014). "Using Random Forests to Estimate Win
      Probability Before Each Play of an NFL Game." JQAS.
    - MoneyPuck Live In-Game Model methodology
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.special import expit  # Logistic function


class GameState(Enum):
    """Current state of the game."""
    REGULATION = "regulation"
    OVERTIME = "overtime"
    SHOOTOUT = "shootout"
    FINAL = "final"


class StrengthState(Enum):
    """Manpower situations."""
    FIVE_ON_FIVE = "5v5"
    FIVE_ON_FOUR = "5v4"
    FIVE_ON_THREE = "5v3"
    FOUR_ON_FIVE = "4v5"
    THREE_ON_FIVE = "3v5"
    FOUR_ON_FOUR = "4v4"
    THREE_ON_THREE = "3v3"
    SIX_ON_FIVE = "6v5"  # Goalie pulled
    SIX_ON_FOUR = "6v4"  # Goalie pulled on PP
    FIVE_ON_SIX = "5v6"  # Opponent goalie pulled
    FOUR_ON_SIX = "4v6"


@dataclass
class GameContext:
    """Complete context for win probability calculation."""
    home_score: int
    away_score: int
    period: int
    time_remaining_period: float  # Seconds remaining in period
    strength_state: StrengthState
    home_goalie_pulled: bool = False
    away_goalie_pulled: bool = False
    penalty_time_remaining: float = 0.0  # Seconds

    # Momentum features (last 5 minutes)
    home_xT_5min: float = 0.0
    away_xT_5min: float = 0.0
    home_shots_5min: int = 0
    away_shots_5min: int = 0
    home_zone_time_5min: float = 0.0  # Seconds in offensive zone
    away_zone_time_5min: float = 0.0

    # Game-level features
    home_xG_total: float = 0.0
    away_xG_total: float = 0.0
    home_corsi: int = 0
    away_corsi: int = 0

    # Team strength (pre-game)
    home_rating: float = 0.0  # Elo-style rating difference
    away_rating: float = 0.0


@dataclass
class WinProbabilityResult:
    """Result of win probability calculation."""
    home_win_prob: float
    away_win_prob: float
    tie_prob: float  # Only for OT scenarios
    home_regulation_win: float
    away_regulation_win: float
    overtime_prob: float

    # Decomposition
    score_component: float
    time_component: float
    strength_component: float
    momentum_component: float

    # Uncertainty
    confidence_interval_low: float
    confidence_interval_high: float


@dataclass
class GoalProbability:
    """Probability of scoring in a time window."""
    prob_home_scores: float
    prob_away_scores: float
    expected_home_goals: float
    expected_away_goals: float


class HockeyWinProbability:
    """
    Bayesian Win Probability Model for Hockey.

    Uses a temporal stochastic process to model future goal-scoring
    and compute win probability at any point in the game.

    The model handles:
    - Regulation time scenarios
    - Overtime (sudden death)
    - Shootout probabilities
    - Goalie pull optimization

    Based on Robberechts et al. (2021) with hockey-specific adaptations
    for manpower situations and faster scoring rate.
    """

    # NHL average goals per 60 minutes by strength state
    GOALS_PER_60 = {
        StrengthState.FIVE_ON_FIVE: 2.5,
        StrengthState.FIVE_ON_FOUR: 7.5,  # PP
        StrengthState.FIVE_ON_THREE: 12.0,  # 5v3 PP
        StrengthState.FOUR_ON_FIVE: 0.8,  # PK
        StrengthState.THREE_ON_FIVE: 0.5,  # 3v5 PK
        StrengthState.FOUR_ON_FOUR: 3.0,
        StrengthState.THREE_ON_THREE: 5.0,  # OT 3v3
        StrengthState.SIX_ON_FIVE: 6.0,  # Goalie pulled
        StrengthState.SIX_ON_FOUR: 10.0,  # Goalie pulled on PP
        StrengthState.FIVE_ON_SIX: 8.0,  # EN goal opportunity
        StrengthState.FOUR_ON_SIX: 10.0,
    }

    # Empty net goal rates when goalie pulled
    EN_GOAL_RATE_PER_60 = 15.0  # Goals against when net empty

    # OT and shootout parameters
    OT_LENGTH = 300.0  # 5 minute OT
    SHOOTOUT_HOME_WIN_PROB = 0.50  # Slight home advantage in SO

    def __init__(
        self,
        base_home_advantage: float = 0.04,  # ~54% home win rate in NHL
        momentum_weight: float = 0.15,
        rating_weight: float = 0.25,
        n_simulations: int = 10000
    ):
        """
        Initialize the win probability model.

        Args:
            base_home_advantage: Base home ice advantage (probability boost)
            momentum_weight: Weight for momentum features (0-1)
            rating_weight: Weight for pre-game team ratings (0-1)
            n_simulations: Number of Monte Carlo simulations for complex scenarios
        """
        self.base_home_advantage = base_home_advantage
        self.momentum_weight = momentum_weight
        self.rating_weight = rating_weight
        self.n_simulations = n_simulations

        # Period lengths in seconds
        self.period_length = 1200.0  # 20 minutes
        self.regulation_length = 3600.0  # 60 minutes

    def calculate_win_probability(
        self,
        context: GameContext
    ) -> WinProbabilityResult:
        """
        Calculate win probability given current game context.

        Uses a Bayesian framework combining:
        1. Score differential and time remaining
        2. Current strength state
        3. Recent momentum
        4. Pre-game team ratings

        Args:
            context: Current game context

        Returns:
            WinProbabilityResult with probabilities and decomposition
        """
        # Calculate time remaining in regulation
        time_remaining = self._calculate_time_remaining(context)

        # Handle end of game
        if time_remaining <= 0 and context.period >= 3:
            return self._handle_end_of_regulation(context)

        # Score differential (from home perspective)
        goal_diff = context.home_score - context.away_score

        # Base win probability from score and time (Poisson model)
        score_component = self._score_time_probability(
            goal_diff, time_remaining, context
        )

        # Strength state adjustment
        strength_component = self._strength_adjustment(context, time_remaining)

        # Momentum adjustment
        momentum_component = self._momentum_adjustment(context)

        # Pre-game rating adjustment
        rating_component = self._rating_adjustment(context)

        # Combine components (logistic combination)
        logit_base = self._prob_to_logit(score_component)
        logit_adjusted = (
            logit_base +
            strength_component +
            momentum_component * self.momentum_weight +
            rating_component * self.rating_weight +
            self.base_home_advantage
        )

        # Calculate overtime probability
        overtime_prob = self._overtime_probability(context, time_remaining)

        # Convert back to probability
        home_reg_win = expit(logit_adjusted) * (1 - overtime_prob)
        away_reg_win = (1 - expit(logit_adjusted)) * (1 - overtime_prob)

        # OT win probabilities (approximately 50/50 with slight adjustments)
        home_ot_win = overtime_prob * self._overtime_win_prob(context, home=True)
        away_ot_win = overtime_prob * self._overtime_win_prob(context, home=False)

        home_win_prob = home_reg_win + home_ot_win
        away_win_prob = away_reg_win + away_ot_win

        # Normalize to ensure probabilities sum to 1
        total = home_win_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            away_win_prob /= total

        # Calculate confidence interval using beta distribution
        ci_low, ci_high = self._confidence_interval(home_win_prob, time_remaining)

        return WinProbabilityResult(
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            tie_prob=0.0,  # No ties in NHL
            home_regulation_win=home_reg_win / total if total > 0 else 0.5,
            away_regulation_win=away_reg_win / total if total > 0 else 0.5,
            overtime_prob=overtime_prob,
            score_component=score_component,
            time_component=time_remaining / self.regulation_length,
            strength_component=strength_component,
            momentum_component=momentum_component,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high
        )

    def _calculate_time_remaining(self, context: GameContext) -> float:
        """Calculate total time remaining in regulation."""
        periods_remaining = max(0, 3 - context.period)
        return (
            context.time_remaining_period +
            periods_remaining * self.period_length
        )

    def _score_time_probability(
        self,
        goal_diff: int,
        time_remaining: float,
        context: GameContext
    ) -> float:
        """
        Calculate win probability from score differential and time.

        Uses a Poisson model for goal scoring:
        - Expected goals remaining based on time and strength
        - Probability of overcoming deficit
        """
        if time_remaining <= 0:
            return 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)

        # Minutes remaining
        minutes_remaining = time_remaining / 60.0

        # Expected goals for each team (5v5 baseline)
        base_rate = self.GOALS_PER_60[StrengthState.FIVE_ON_FIVE] / 60.0
        expected_home = base_rate * minutes_remaining
        expected_away = base_rate * minutes_remaining

        # Adjust for xG differential (team quality proxy)
        if context.home_xG_total + context.away_xG_total > 0:
            xg_ratio = context.home_xG_total / (context.home_xG_total + context.away_xG_total)
            expected_home *= (0.5 + xg_ratio)
            expected_away *= (1.5 - xg_ratio)

        # Monte Carlo simulation for complex goal scenarios
        home_wins = 0
        for _ in range(self.n_simulations):
            home_goals = np.random.poisson(expected_home)
            away_goals = np.random.poisson(expected_away)
            final_diff = goal_diff + home_goals - away_goals
            if final_diff > 0:
                home_wins += 1
            elif final_diff == 0:
                home_wins += 0.5  # Tie goes to OT, count as 0.5

        return home_wins / self.n_simulations

    def _strength_adjustment(
        self,
        context: GameContext,
        time_remaining: float
    ) -> float:
        """
        Calculate logit adjustment for current strength state.

        Power plays and penalty kills significantly affect short-term
        win probability.
        """
        if context.penalty_time_remaining <= 0:
            return 0.0

        # Get scoring rates for current strength
        home_rate = self.GOALS_PER_60.get(context.strength_state, 2.5)

        # Determine who has advantage
        if context.strength_state in [
            StrengthState.FIVE_ON_FOUR,
            StrengthState.FIVE_ON_THREE,
            StrengthState.SIX_ON_FOUR
        ]:
            # Home team on PP
            advantage_minutes = min(context.penalty_time_remaining, time_remaining) / 60.0
            expected_advantage = (home_rate - 2.5) / 60.0 * advantage_minutes
            return expected_advantage * 0.1  # Scale to logit
        elif context.strength_state in [
            StrengthState.FOUR_ON_FIVE,
            StrengthState.THREE_ON_FIVE,
            StrengthState.FOUR_ON_SIX
        ]:
            # Home team on PK
            advantage_minutes = min(context.penalty_time_remaining, time_remaining) / 60.0
            pk_rate = self.GOALS_PER_60.get(context.strength_state, 0.8)
            expected_disadvantage = (2.5 - pk_rate) / 60.0 * advantage_minutes
            return -expected_disadvantage * 0.1

        return 0.0

    def _momentum_adjustment(self, context: GameContext) -> float:
        """
        Calculate momentum adjustment based on recent play.

        Uses xT, shots, and zone time from last 5 minutes.
        """
        # xT differential
        xT_diff = context.home_xT_5min - context.away_xT_5min

        # Shot differential
        shot_diff = context.home_shots_5min - context.away_shots_5min

        # Zone time differential (percentage)
        total_zone_time = context.home_zone_time_5min + context.away_zone_time_5min
        if total_zone_time > 0:
            zone_diff = (
                context.home_zone_time_5min - context.away_zone_time_5min
            ) / total_zone_time
        else:
            zone_diff = 0.0

        # Combine into momentum score
        momentum = (
            xT_diff * 0.5 +
            shot_diff * 0.03 +
            zone_diff * 0.2
        )

        return np.clip(momentum, -0.5, 0.5)

    def _rating_adjustment(self, context: GameContext) -> float:
        """
        Adjust for pre-game team strength ratings.

        Uses Elo-style ratings to capture team quality.
        """
        rating_diff = context.home_rating - context.away_rating

        # Elo expected score formula
        # E = 1 / (1 + 10^(-diff/400))
        # Convert to logit adjustment
        return rating_diff / 400.0 * np.log(10)

    def _overtime_probability(
        self,
        context: GameContext,
        time_remaining: float
    ) -> float:
        """
        Calculate probability of game going to overtime.

        Based on current score differential and time remaining.
        """
        goal_diff = abs(context.home_score - context.away_score)

        if goal_diff == 0:
            # Already tied
            if time_remaining <= 0:
                return 1.0
            # Probability of remaining tied
            minutes = time_remaining / 60.0
            expected_goals = self.GOALS_PER_60[StrengthState.FIVE_ON_FIVE] / 60.0 * minutes
            # Probability both teams score same number
            return np.exp(-2 * expected_goals)  # Simplified approximation
        elif goal_diff == 1:
            # One goal game - probability of tying
            minutes = time_remaining / 60.0
            rate = self.GOALS_PER_60[StrengthState.FIVE_ON_FIVE] / 60.0
            # Probability of trailing team scoring and leading team not
            prob_tie = stats.poisson.pmf(1, rate * minutes) * stats.poisson.pmf(0, rate * minutes)
            return prob_tie * 2  # Either team could tie it
        elif goal_diff == 2:
            # Two goal game
            minutes = time_remaining / 60.0
            if minutes < 5:
                return 0.05  # Very unlikely
            rate = self.GOALS_PER_60[StrengthState.FIVE_ON_FIVE] / 60.0
            return stats.poisson.cdf(0, rate * minutes) ** 2 * 0.1
        else:
            return 0.01  # Very unlikely for 3+ goal leads

    def _overtime_win_prob(self, context: GameContext, home: bool) -> float:
        """
        Calculate win probability in overtime/shootout.

        OT is 3v3 sudden death, then shootout.
        """
        # 3v3 OT - higher scoring, slight home advantage
        ot_home_advantage = 0.52

        # Shootout is roughly 50/50 with slight home edge
        so_home_prob = self.SHOOTOUT_HOME_WIN_PROB

        # Probability of scoring in OT (before shootout)
        ot_goal_prob = 0.65  # ~65% of games end in OT, not SO

        if home:
            return ot_goal_prob * ot_home_advantage + (1 - ot_goal_prob) * so_home_prob
        else:
            return ot_goal_prob * (1 - ot_home_advantage) + (1 - ot_goal_prob) * (1 - so_home_prob)

    def _handle_end_of_regulation(self, context: GameContext) -> WinProbabilityResult:
        """Handle win probability at end of regulation."""
        goal_diff = context.home_score - context.away_score

        if goal_diff > 0:
            return WinProbabilityResult(
                home_win_prob=1.0,
                away_win_prob=0.0,
                tie_prob=0.0,
                home_regulation_win=1.0,
                away_regulation_win=0.0,
                overtime_prob=0.0,
                score_component=1.0,
                time_component=0.0,
                strength_component=0.0,
                momentum_component=0.0,
                confidence_interval_low=1.0,
                confidence_interval_high=1.0
            )
        elif goal_diff < 0:
            return WinProbabilityResult(
                home_win_prob=0.0,
                away_win_prob=1.0,
                tie_prob=0.0,
                home_regulation_win=0.0,
                away_regulation_win=1.0,
                overtime_prob=0.0,
                score_component=0.0,
                time_component=0.0,
                strength_component=0.0,
                momentum_component=0.0,
                confidence_interval_low=0.0,
                confidence_interval_high=0.0
            )
        else:
            # Tied - going to OT
            home_ot = self._overtime_win_prob(context, home=True)
            return WinProbabilityResult(
                home_win_prob=home_ot,
                away_win_prob=1 - home_ot,
                tie_prob=0.0,
                home_regulation_win=0.0,
                away_regulation_win=0.0,
                overtime_prob=1.0,
                score_component=0.5,
                time_component=0.0,
                strength_component=0.0,
                momentum_component=0.0,
                confidence_interval_low=home_ot - 0.1,
                confidence_interval_high=home_ot + 0.1
            )

    def _prob_to_logit(self, p: float) -> float:
        """Convert probability to logit (log-odds)."""
        p = np.clip(p, 0.001, 0.999)
        return np.log(p / (1 - p))

    def _confidence_interval(
        self,
        prob: float,
        time_remaining: float
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for win probability.

        Uncertainty increases with time remaining.
        """
        # Uncertainty scales with sqrt of time remaining
        uncertainty_scale = np.sqrt(time_remaining / self.regulation_length) * 0.15

        # Use beta distribution for bounded CI
        ci_low = max(0.0, prob - uncertainty_scale)
        ci_high = min(1.0, prob + uncertainty_scale)

        return ci_low, ci_high

    def calculate_wpa(
        self,
        context_before: GameContext,
        context_after: GameContext
    ) -> float:
        """
        Calculate Win Probability Added for an event.

        WPA = WP_after - WP_before

        Positive WPA means the event helped the home team.

        Args:
            context_before: Game context before the event
            context_after: Game context after the event

        Returns:
            Win Probability Added (from home team perspective)
        """
        wp_before = self.calculate_win_probability(context_before)
        wp_after = self.calculate_win_probability(context_after)

        return wp_after.home_win_prob - wp_before.home_win_prob

    def optimal_goalie_pull_time(
        self,
        context: GameContext
    ) -> Tuple[float, float]:
        """
        Calculate optimal time to pull goalie when trailing.

        Based on expected goal differential with empty net
        vs. probability of tying the game.

        Args:
            context: Current game context

        Returns:
            Tuple of (optimal_time_remaining, expected_wp_gain)
        """
        goal_diff = context.home_score - context.away_score

        if goal_diff >= 0:
            return 0.0, 0.0  # Don't pull if tied or winning

        # Simulate different pull times
        best_time = 0.0
        best_wp = 0.0

        for pull_time in range(30, 180, 10):  # 30 seconds to 3 minutes
            # Expected goals with empty net
            en_time_minutes = pull_time / 60.0

            # Goals for (6v5)
            goals_for_rate = self.GOALS_PER_60[StrengthState.SIX_ON_FIVE] / 60.0
            expected_goals_for = goals_for_rate * en_time_minutes

            # Goals against (EN)
            goals_against_rate = self.EN_GOAL_RATE_PER_60 / 60.0
            expected_goals_against = goals_against_rate * en_time_minutes

            # Net expected goal change
            net_goals = expected_goals_for - expected_goals_against

            # Probability of tying (need exactly |goal_diff| goals for, 0 against)
            prob_tie = (
                stats.poisson.pmf(abs(goal_diff), expected_goals_for) *
                stats.poisson.pmf(0, expected_goals_against)
            )

            # Expected WP gain
            # Tying gives ~50% win prob, otherwise likely lose
            expected_wp = prob_tie * 0.5

            if expected_wp > best_wp:
                best_wp = expected_wp
                best_time = pull_time

        return best_time, best_wp


class WinProbabilityTracker:
    """
    Track win probability throughout a game.

    Records WP at each event for visualization and analysis.
    """

    def __init__(self):
        self.model = HockeyWinProbability()
        self.history: List[Dict[str, Any]] = []

    def record_event(
        self,
        context: GameContext,
        event_type: str,
        event_description: str,
        player_id: Optional[str] = None
    ) -> WinProbabilityResult:
        """
        Record win probability at an event.

        Args:
            context: Current game context
            event_type: Type of event (goal, shot, penalty, etc.)
            event_description: Human-readable description
            player_id: Optional player involved

        Returns:
            WinProbabilityResult
        """
        result = self.model.calculate_win_probability(context)

        game_time = (
            (context.period - 1) * 1200 +
            (1200 - context.time_remaining_period)
        )

        self.history.append({
            'game_time': game_time,
            'period': context.period,
            'time_remaining': context.time_remaining_period,
            'event_type': event_type,
            'description': event_description,
            'player_id': player_id,
            'home_score': context.home_score,
            'away_score': context.away_score,
            'strength_state': context.strength_state.value,
            'home_wp': result.home_win_prob,
            'away_wp': result.away_win_prob,
            'overtime_prob': result.overtime_prob,
            'ci_low': result.confidence_interval_low,
            'ci_high': result.confidence_interval_high
        })

        return result

    def get_leverage_moments(self, threshold: float = 0.1) -> List[Dict]:
        """
        Find high-leverage moments where WP changed significantly.

        Args:
            threshold: Minimum WP change to be considered high-leverage

        Returns:
            List of high-leverage events
        """
        leverage_moments = []

        for i in range(1, len(self.history)):
            wp_change = abs(
                self.history[i]['home_wp'] -
                self.history[i-1]['home_wp']
            )

            if wp_change >= threshold:
                leverage_moments.append({
                    **self.history[i],
                    'wp_change': wp_change,
                    'leverage_index': wp_change / 0.1  # Normalized to average
                })

        return sorted(leverage_moments, key=lambda x: -x['leverage_index'])

    def get_player_wpa(self) -> Dict[str, float]:
        """
        Calculate total WPA for each player in the game.

        Returns:
            Dictionary mapping player_id to cumulative WPA
        """
        player_wpa: Dict[str, float] = {}

        for i in range(1, len(self.history)):
            if self.history[i]['player_id']:
                player_id = self.history[i]['player_id']
                wp_change = (
                    self.history[i]['home_wp'] -
                    self.history[i-1]['home_wp']
                )

                if player_id not in player_wpa:
                    player_wpa[player_id] = 0.0
                player_wpa[player_id] += wp_change

        return player_wpa

    def export_for_visualization(self) -> List[Dict]:
        """Export history in format suitable for charting."""
        return self.history.copy()
