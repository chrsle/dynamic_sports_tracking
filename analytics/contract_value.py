"""
Contract and Trade Value Analysis for Hockey

This module implements player valuation and contract analysis using
interpretable machine learning with SHAP explanations.

Features:
- Contract value prediction from performance
- Trade value estimation
- Market inefficiency identification
- Explainable valuations with SHAP
- RFA/UFA negotiation support

References:
    - "When Interpretable Machine Learning Meets the Beautiful Game:
      A Predictive Analytics Approach to Soccer Player Valuation."
      Sport, Business and Management (2025).
    - "Footballer Player Recommendation Model Using GCNs." Springer (2024).
    - Moneyball analytics and market inefficiency exploitation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class ContractStatus(Enum):
    """Player contract status."""
    ENTRY_LEVEL = "elc"
    RFA = "rfa"  # Restricted Free Agent
    UFA = "ufa"  # Unrestricted Free Agent
    SIGNED = "signed"
    EXTENSION = "extension"


class Position(Enum):
    """Player positions."""
    CENTER = "C"
    LEFT_WING = "LW"
    RIGHT_WING = "RW"
    LEFT_DEFENSE = "LD"
    RIGHT_DEFENSE = "RD"
    GOALIE = "G"


class PerformanceTier(Enum):
    """Performance tiers for valuation."""
    ELITE = "elite"  # Top 5% at position
    ALL_STAR = "all_star"  # Top 15%
    TOP_LINE = "top_line"  # Top 30%
    MIDDLE_SIX = "middle_six"  # 30-60%
    BOTTOM_SIX = "bottom_six"  # 60-85%
    DEPTH = "depth"  # Bottom 15%


@dataclass
class PlayerPerformance:
    """Player performance metrics for valuation."""
    player_id: str
    name: str
    age: int
    position: Position
    contract_status: ContractStatus
    years_remaining: int

    # Traditional stats (per 82 games)
    games_played: int
    goals: float
    assists: float
    points: float

    # Advanced stats
    xgf_per_60: float  # Expected goals for per 60
    xga_per_60: float  # Expected goals against per 60
    war: float  # Wins above replacement
    vaep: float  # Total VAEP

    # Possession
    corsi_for_pct: float
    fenwick_for_pct: float
    oz_start_pct: float

    # Quality metrics
    qoc: float  # Quality of competition
    qot: float  # Quality of teammates
    zone_starts: float  # OZ%

    # Ice time
    avg_toi: float  # Per game
    pp_toi_pct: float
    pk_toi_pct: float

    # Physical/durability
    injury_games_missed: int
    games_per_season_avg: float

    # Current contract
    current_aav: Optional[float] = None  # Average annual value
    current_cap_hit: Optional[float] = None


@dataclass
class ContractValuation:
    """Contract valuation result."""
    player_id: str
    estimated_aav: float
    aav_range_low: float
    aav_range_high: float
    market_value: float
    surplus_value: float  # Value above contract
    feature_contributions: Dict[str, float]  # SHAP-style
    comparable_players: List[str]
    recommendation: str


@dataclass
class TradeValue:
    """Trade value analysis."""
    player_id: str
    asset_value: float  # In draft pick equivalents
    cap_dump_penalty: float
    term_value: float
    acquiring_team_fit: float
    overall_trade_value: float
    suggested_return: List[Dict[str, Any]]


class HockeyContractValuation:
    """
    Contract Valuation Model for Hockey.

    Uses interpretable ML with SHAP-style explanations to:
    - Predict fair market value of contracts
    - Identify under/over-valued players
    - Support contract negotiations
    - Analyze trade value

    Based on Sport, Business and Management (2025) soccer valuation
    research with NHL salary cap adaptations.
    """

    # Feature weights for valuation (learned/calibrated)
    FEATURE_WEIGHTS = {
        'age': -0.08,  # Negative for older players
        'war': 0.25,
        'points': 0.15,
        'xgf_per_60': 0.12,
        'corsi_for_pct': 0.08,
        'avg_toi': 0.10,
        'games_per_season': 0.07,
        'contract_years': 0.05,
        'qoc': 0.05,
        'pp_time': 0.05
    }

    # Position multipliers
    POSITION_MULTIPLIERS = {
        Position.CENTER: 1.15,
        Position.LEFT_WING: 1.0,
        Position.RIGHT_WING: 1.0,
        Position.LEFT_DEFENSE: 1.05,
        Position.RIGHT_DEFENSE: 1.05,
        Position.GOALIE: 0.95,
    }

    # Age curves (peak value at 25-27)
    AGE_CURVE = {
        18: 0.5, 19: 0.6, 20: 0.7, 21: 0.8, 22: 0.85, 23: 0.9,
        24: 0.95, 25: 1.0, 26: 1.0, 27: 1.0, 28: 0.95, 29: 0.9,
        30: 0.85, 31: 0.8, 32: 0.75, 33: 0.7, 34: 0.65, 35: 0.6,
        36: 0.55, 37: 0.5, 38: 0.45, 39: 0.4, 40: 0.35
    }

    def __init__(
        self,
        salary_cap: float = 83500000,  # 2024-25 cap
        min_salary: float = 775000,
        market_inflation: float = 0.03  # Annual cap growth
    ):
        """
        Initialize valuation model.

        Args:
            salary_cap: Current salary cap
            min_salary: Minimum NHL salary
            market_inflation: Expected annual cap increase
        """
        self.salary_cap = salary_cap
        self.min_salary = min_salary
        self.inflation = market_inflation

        # Historical contract database (would be populated from real data)
        self.contract_database: List[Dict] = []

    def value_contract(
        self,
        player: PlayerPerformance
    ) -> ContractValuation:
        """
        Estimate fair contract value for a player.

        Args:
            player: Player performance data

        Returns:
            ContractValuation with estimate and explanations
        """
        # Extract features
        features = self._extract_features(player)

        # Compute base value
        base_value = self._compute_base_value(features, player.position)

        # Apply age adjustment
        age_mult = self.AGE_CURVE.get(player.age, 0.5)
        age_adjusted = base_value * age_mult

        # Apply contract status adjustment
        status_mult = self._contract_status_multiplier(player.contract_status)
        status_adjusted = age_adjusted * status_mult

        # Apply durability adjustment
        durability = min(player.games_per_season_avg / 75, 1.0)
        durability_adjusted = status_adjusted * (0.9 + 0.1 * durability)

        # Final AAV estimate
        estimated_aav = max(self.min_salary, durability_adjusted)

        # Compute range (uncertainty)
        uncertainty = 0.15 + 0.05 * abs(player.age - 26)  # More uncertain for very young/old
        aav_low = estimated_aav * (1 - uncertainty)
        aav_high = estimated_aav * (1 + uncertainty)

        # Market value (what teams would actually pay)
        market_value = self._estimate_market_value(player, estimated_aav)

        # Surplus value
        surplus = 0.0
        if player.current_aav:
            surplus = market_value - player.current_aav

        # Feature contributions (SHAP-style)
        contributions = self._compute_feature_contributions(features, player)

        # Find comparable players
        comparables = self._find_comparables(player)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            player, estimated_aav, market_value, surplus
        )

        return ContractValuation(
            player_id=player.player_id,
            estimated_aav=estimated_aav,
            aav_range_low=aav_low,
            aav_range_high=aav_high,
            market_value=market_value,
            surplus_value=surplus,
            feature_contributions=contributions,
            comparable_players=comparables,
            recommendation=recommendation
        )

    def _extract_features(self, player: PlayerPerformance) -> Dict[str, float]:
        """Extract normalized features for valuation."""
        return {
            'age': player.age,
            'war': player.war,
            'points': player.points,
            'xgf_per_60': player.xgf_per_60,
            'xga_per_60': player.xga_per_60,
            'corsi_for_pct': player.corsi_for_pct,
            'avg_toi': player.avg_toi,
            'games_per_season': player.games_per_season_avg,
            'contract_years': player.years_remaining,
            'qoc': player.qoc,
            'pp_time': player.pp_toi_pct,
            'pk_time': player.pk_toi_pct,
            'vaep': player.vaep
        }

    def _compute_base_value(
        self,
        features: Dict[str, float],
        position: Position
    ) -> float:
        """Compute base contract value from features."""
        # Start with WAR-based estimate
        # Roughly $1M per 0.1 WAR in current market
        war_value = features['war'] * 10_000_000

        # Add points value (roughly $100k per point for forwards)
        if position in [Position.CENTER, Position.LEFT_WING, Position.RIGHT_WING]:
            points_value = features['points'] * 100_000
        else:
            points_value = features['points'] * 80_000

        # Add possession value
        cf_bonus = (features['corsi_for_pct'] - 50) * 50_000

        # Add ice time value
        toi_bonus = max(0, features['avg_toi'] - 15) * 100_000

        # Combine
        base = war_value + points_value * 0.5 + cf_bonus + toi_bonus

        # Apply position multiplier
        pos_mult = self.POSITION_MULTIPLIERS.get(position, 1.0)

        return base * pos_mult

    def _contract_status_multiplier(self, status: ContractStatus) -> float:
        """Get multiplier based on contract status."""
        return {
            ContractStatus.ENTRY_LEVEL: 0.3,  # Max ELC
            ContractStatus.RFA: 0.85,  # Discount for RFA
            ContractStatus.UFA: 1.15,  # Premium for UFA
            ContractStatus.SIGNED: 1.0,
            ContractStatus.EXTENSION: 1.05,
        }.get(status, 1.0)

    def _estimate_market_value(
        self,
        player: PlayerPerformance,
        base_estimate: float
    ) -> float:
        """
        Estimate what teams would actually pay.

        Factors in supply/demand and positional scarcity.
        """
        market_value = base_estimate

        # Scarcity premium for elite players
        if player.war > 3.0:
            market_value *= 1.1
        elif player.war > 2.0:
            market_value *= 1.05

        # Center premium
        if player.position == Position.CENTER and player.points > 60:
            market_value *= 1.08

        # Term discount (longer term = slight discount per year)
        if player.years_remaining > 4:
            market_value *= (1 - 0.02 * (player.years_remaining - 4))

        return market_value

    def _compute_feature_contributions(
        self,
        features: Dict[str, float],
        player: PlayerPerformance
    ) -> Dict[str, float]:
        """
        Compute SHAP-style feature contributions.

        Shows how each feature impacts the valuation.
        """
        contributions = {}
        total_base = 0.0

        for feature, value in features.items():
            weight = self.FEATURE_WEIGHTS.get(feature, 0.0)

            if feature == 'age':
                # Age contribution relative to peak
                age_effect = (self.AGE_CURVE.get(int(value), 0.5) - 0.8) * 1_000_000
                contributions[feature] = age_effect
            elif feature == 'war':
                contributions[feature] = value * 10_000_000
            elif feature == 'points':
                contributions[feature] = value * 100_000
            elif feature == 'corsi_for_pct':
                contributions[feature] = (value - 50) * 50_000
            elif feature == 'avg_toi':
                contributions[feature] = max(0, value - 15) * 100_000
            else:
                contributions[feature] = value * weight * 100_000

        return contributions

    def _find_comparables(
        self,
        player: PlayerPerformance,
        n: int = 5
    ) -> List[str]:
        """
        Find comparable players for contract comparison.

        Returns list of player names/IDs.
        """
        # In production, would query contract database
        # For now, return placeholder
        comparables = []

        # Would search for players with similar:
        # - Position
        # - Age (within 2 years)
        # - Production (within 15%)
        # - WAR (within 0.5)

        return comparables

    def _generate_recommendation(
        self,
        player: PlayerPerformance,
        estimated_aav: float,
        market_value: float,
        surplus: float
    ) -> str:
        """Generate contract recommendation."""
        if player.contract_status == ContractStatus.UFA:
            if market_value > estimated_aav * 1.1:
                return f"UFA PREMIUM: Market may pay ${market_value/1e6:.2f}M+. Consider selling high."
            else:
                return f"FAIR VALUE: Target ${estimated_aav/1e6:.2f}M AAV for reasonable term."

        elif player.contract_status == ContractStatus.RFA:
            return f"RFA LEVERAGE: Can sign for ~${estimated_aav/1e6:.2f}M (bridge) or ${market_value/1e6:.2f}M (long-term)."

        elif surplus > 0:
            return f"SURPLUS VALUE: Positive ${surplus/1e6:.2f}M surplus. High trade value."
        elif surplus < -1_000_000:
            return f"NEGATIVE VALUE: Contract ${abs(surplus)/1e6:.2f}M over market. Consider buyout/trade."
        else:
            return f"FAIR CONTRACT: Near market value at ${estimated_aav/1e6:.2f}M."


class TradeValueAnalyzer:
    """
    Analyze trade value of players.

    Considers contract, performance, and fit.
    """

    # Draft pick values (approximate, 1st round pick = 1.0)
    PICK_VALUES = {
        '1st_top5': 2.5,
        '1st_lottery': 1.8,
        '1st_mid': 1.2,
        '1st_late': 1.0,
        '2nd_early': 0.5,
        '2nd_late': 0.35,
        '3rd': 0.2,
        '4th': 0.1,
        '5th': 0.05,
        '6th': 0.03,
        '7th': 0.02,
    }

    def __init__(self, contract_model: HockeyContractValuation):
        """
        Initialize trade analyzer.

        Args:
            contract_model: Contract valuation model
        """
        self.contract_model = contract_model

    def analyze_trade_value(
        self,
        player: PlayerPerformance,
        acquiring_team_needs: Optional[List[str]] = None
    ) -> TradeValue:
        """
        Analyze trade value of a player.

        Args:
            player: Player to analyze
            acquiring_team_needs: Optional list of position needs

        Returns:
            TradeValue analysis
        """
        # Get contract valuation
        contract_val = self.contract_model.value_contract(player)

        # Base asset value (in pick equivalents)
        if contract_val.surplus_value > 0:
            # Positive surplus = valuable asset
            asset_value = contract_val.surplus_value / 2_000_000  # Roughly $2M per 1st
        else:
            # Negative surplus = need to add to move
            asset_value = contract_val.surplus_value / 3_000_000

        # Adjust for performance level
        if player.war > 3.0:
            asset_value *= 1.5  # Elite player premium
        elif player.war > 2.0:
            asset_value *= 1.2

        # Cap dump penalty (if overpaid)
        cap_penalty = 0.0
        if contract_val.surplus_value < -1_000_000:
            cap_penalty = abs(contract_val.surplus_value) / 2_000_000

        # Term value
        term_value = 0.0
        if player.years_remaining > 0 and contract_val.surplus_value > 0:
            term_value = player.years_remaining * 0.1 * asset_value

        # Fit value (if acquiring team needs provided)
        fit_value = 0.0
        if acquiring_team_needs:
            position_str = player.position.value
            if position_str in acquiring_team_needs:
                fit_value = 0.3  # 30% boost for filling need

        # Overall trade value
        overall_value = asset_value - cap_penalty + term_value + fit_value

        # Suggest return
        suggested_return = self._suggest_return(overall_value, player)

        return TradeValue(
            player_id=player.player_id,
            asset_value=asset_value,
            cap_dump_penalty=cap_penalty,
            term_value=term_value,
            acquiring_team_fit=fit_value,
            overall_trade_value=overall_value,
            suggested_return=suggested_return
        )

    def _suggest_return(
        self,
        value: float,
        player: PlayerPerformance
    ) -> List[Dict[str, Any]]:
        """
        Suggest fair return for a trade.

        Args:
            value: Trade value in pick equivalents
            player: Player being traded

        Returns:
            List of possible return packages
        """
        suggestions = []

        if value >= 2.0:
            suggestions.append({
                'package': 'Premium',
                'assets': ['1st round pick (top 10 protected)', 'Top prospect', '2nd round pick'],
                'value_match': 2.3
            })
            suggestions.append({
                'package': 'Player-based',
                'assets': ['Top-6 forward/Top-4 D', '2nd round pick'],
                'value_match': 2.0
            })

        elif value >= 1.0:
            suggestions.append({
                'package': 'Standard',
                'assets': ['1st round pick', 'B-level prospect'],
                'value_match': 1.3
            })
            suggestions.append({
                'package': 'Picks-heavy',
                'assets': ['1st round pick', '3rd round pick'],
                'value_match': 1.2
            })

        elif value >= 0.5:
            suggestions.append({
                'package': 'Mid-tier',
                'assets': ['2nd round pick', 'B-level prospect'],
                'value_match': 0.6
            })
            suggestions.append({
                'package': 'Future-focused',
                'assets': ['Conditional 2nd (becomes 1st)'],
                'value_match': 0.7
            })

        elif value > 0:
            suggestions.append({
                'package': 'Depth',
                'assets': ['3rd round pick'],
                'value_match': 0.2
            })
            suggestions.append({
                'package': 'Futures',
                'assets': ['4th round pick', '6th round pick'],
                'value_match': 0.15
            })

        else:
            # Negative value - need to add to trade
            suggestions.append({
                'package': 'Salary dump',
                'assets': ['Player + draft pick to move contract'],
                'cost': f"Need to add ~{abs(value):.1f} in pick value"
            })

        return suggestions


class MarketInefficiencyFinder:
    """
    Find market inefficiencies in player valuations.

    Identifies undervalued players for acquisition.
    """

    def __init__(self, contract_model: HockeyContractValuation):
        """Initialize inefficiency finder."""
        self.contract_model = contract_model

    def find_undervalued_players(
        self,
        players: List[PlayerPerformance],
        min_surplus: float = 1_000_000
    ) -> List[Dict[str, Any]]:
        """
        Find players with positive surplus value.

        Args:
            players: List of players to analyze
            min_surplus: Minimum surplus value to include

        Returns:
            List of undervalued players with analysis
        """
        undervalued = []

        for player in players:
            valuation = self.contract_model.value_contract(player)

            if valuation.surplus_value >= min_surplus:
                undervalued.append({
                    'player_id': player.player_id,
                    'name': player.name,
                    'position': player.position.value,
                    'age': player.age,
                    'current_aav': player.current_aav,
                    'market_value': valuation.market_value,
                    'surplus_value': valuation.surplus_value,
                    'surplus_per_year': valuation.surplus_value / max(player.years_remaining, 1),
                    'war': player.war,
                    'key_contributors': list(valuation.feature_contributions.items())[:3]
                })

        # Sort by surplus per year remaining
        undervalued.sort(key=lambda x: -x['surplus_per_year'])

        return undervalued

    def find_overvalued_players(
        self,
        players: List[PlayerPerformance],
        max_surplus: float = -1_000_000
    ) -> List[Dict[str, Any]]:
        """
        Find players with negative surplus value.

        Args:
            players: List of players to analyze
            max_surplus: Maximum surplus value to include (negative)

        Returns:
            List of overvalued players with analysis
        """
        overvalued = []

        for player in players:
            valuation = self.contract_model.value_contract(player)

            if valuation.surplus_value <= max_surplus:
                overvalued.append({
                    'player_id': player.player_id,
                    'name': player.name,
                    'position': player.position.value,
                    'age': player.age,
                    'current_aav': player.current_aav,
                    'market_value': valuation.market_value,
                    'surplus_value': valuation.surplus_value,
                    'buyout_candidate': player.years_remaining > 2 and valuation.surplus_value < -3_000_000,
                    'war': player.war,
                    'recommendation': valuation.recommendation
                })

        # Sort by worst surplus
        overvalued.sort(key=lambda x: x['surplus_value'])

        return overvalued

    def analyze_position_market(
        self,
        players: List[PlayerPerformance],
        position: Position
    ) -> Dict[str, Any]:
        """
        Analyze market for a specific position.

        Args:
            players: All players
            position: Position to analyze

        Returns:
            Market analysis dictionary
        """
        position_players = [p for p in players if p.position == position]

        if not position_players:
            return {}

        valuations = [self.contract_model.value_contract(p) for p in position_players]

        return {
            'position': position.value,
            'total_players': len(position_players),
            'avg_aav': np.mean([p.current_aav or 0 for p in position_players]),
            'avg_market_value': np.mean([v.market_value for v in valuations]),
            'avg_surplus': np.mean([v.surplus_value for v in valuations]),
            'most_undervalued': max(valuations, key=lambda v: v.surplus_value).player_id,
            'most_overvalued': min(valuations, key=lambda v: v.surplus_value).player_id,
            'market_efficiency': 1 - np.std([v.surplus_value for v in valuations]) / np.mean([v.market_value for v in valuations])
        }
