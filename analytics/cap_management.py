"""
Cap Management Analytics (Moneyball 2.0-inspired)

Implements methodology from:
- Brown, M. (2021). "Moneyball 2.0: Market Inefficiency Exploitation
  in Modern Sports Analytics."

Key concept: Identify market inefficiencies in player valuation
to maximize team performance within salary cap constraints.

Hockey translation:
- Salary cap optimization
- Contract structure analysis
- Trade deadline decisions
- Prospect/veteran balance
- Bridge deal vs long-term analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, date


class ContractType(Enum):
    """Types of NHL contracts."""
    ELC = "entry_level"  # Entry-level contract
    BRIDGE = "bridge"  # Short-term bridge deal
    LONG_TERM = "long_term"  # 5+ years
    VETERAN = "veteran"  # 35+ player
    PTO = "pto"  # Professional tryout
    TWO_WAY = "two_way"  # NHL/AHL
    ONE_WAY = "one_way"  # NHL only


class ContractStatus(Enum):
    """Contract status for roster planning."""
    ACTIVE = "active"
    EXPIRING = "expiring"  # Final year
    RFA = "rfa"  # Restricted free agent
    UFA = "ufa"  # Unrestricted free agent
    BUYOUT = "buyout"
    LTIR = "ltir"  # Long-term injured reserve


class SkillTier(Enum):
    """Player skill tier for valuation."""
    ELITE = "elite"  # Top 5% at position
    STAR = "star"  # Top 15%
    TOP_SIX = "top_six"  # Top 30%
    MIDDLE_SIX = "middle_six"
    BOTTOM_SIX = "bottom_six"
    FRINGE = "fringe"
    AHL = "ahl"


@dataclass
class Contract:
    """Player contract details."""
    player_id: str
    aav: float  # Average annual value
    cap_hit: float
    total_value: float
    years_remaining: int
    contract_type: ContractType
    status: ContractStatus
    signing_bonus: float = 0.0
    base_salary: float = 0.0
    performance_bonus: float = 0.0
    ntc: bool = False  # No-trade clause
    nmc: bool = False  # No-movement clause


@dataclass
class PlayerValue:
    """Calculated player value."""
    player_id: str
    war: float  # Wins above replacement
    gaa: float  # Goals above average
    projected_war: float  # Future WAR
    surplus_value: float  # WAR minus cost
    market_value: float  # Estimated market price
    cost_per_war: float  # Cap hit per WAR


@dataclass
class CapSituation:
    """Team's salary cap situation."""
    cap_ceiling: float
    cap_floor: float
    current_cap_hit: float
    cap_space: float
    projected_cap_space: float  # Next season
    ltir_space: float
    bonus_cushion: float
    dead_cap: float  # Buyouts, retained salary


@dataclass
class RosterDecision:
    """Recommended roster decision."""
    decision_type: str  # "sign", "trade", "waive", "buyout", "extend"
    player_id: str
    rationale: str
    financial_impact: float
    war_impact: float
    urgency: str  # "immediate", "soon", "long_term"


@dataclass
class TradeScenario:
    """Trade scenario analysis."""
    players_out: List[str]
    players_in: List[str]
    cap_impact: float
    war_change: float
    surplus_change: float
    draft_picks: Dict[str, int]  # year -> round
    recommendation: str


class MarketValueEstimator:
    """
    Estimate player market value.

    Based on comparable contracts and performance.
    """

    def __init__(self):
        # Market rates by tier (AAV in millions)
        self.market_rates = {
            SkillTier.ELITE: (9.5, 13.0),
            SkillTier.STAR: (6.5, 9.5),
            SkillTier.TOP_SIX: (4.0, 6.5),
            SkillTier.MIDDLE_SIX: (2.0, 4.0),
            SkillTier.BOTTOM_SIX: (0.85, 2.0),
            SkillTier.FRINGE: (0.775, 0.85),
            SkillTier.AHL: (0.775, 0.775),
        }

        # Position adjustments
        self.position_factors = {
            'C': 1.1,
            'LW': 1.0,
            'RW': 1.0,
            'LD': 1.05,
            'RD': 1.05,
            'G': 0.95,  # Goalies often undervalued
        }

        # Age curve (peak at 24-27)
        self.age_curve = self._create_age_curve()

    def _create_age_curve(self) -> Dict[int, float]:
        """Create age-based value multiplier."""
        curve = {}
        for age in range(18, 45):
            if age < 22:
                curve[age] = 0.7 + (age - 18) * 0.075
            elif age < 28:
                curve[age] = 1.0
            elif age < 32:
                curve[age] = 1.0 - (age - 27) * 0.05
            else:
                curve[age] = 0.8 - (age - 31) * 0.08
        return curve

    def estimate_value(
        self,
        player: Dict,
        tier: SkillTier,
        age: int,
        position: str
    ) -> float:
        """Estimate player's market value."""
        # Base range for tier
        min_val, max_val = self.market_rates[tier]

        # WAR-based adjustment within tier
        war = player.get('war', 0)
        tier_percentile = self._war_to_tier_percentile(war, tier)
        base_value = min_val + tier_percentile * (max_val - min_val)

        # Position factor
        pos_factor = self.position_factors.get(position, 1.0)

        # Age factor
        age_factor = self.age_curve.get(age, 0.5)

        return base_value * pos_factor * age_factor

    def _war_to_tier_percentile(self, war: float, tier: SkillTier) -> float:
        """Convert WAR to percentile within tier."""
        # WAR ranges by tier
        war_ranges = {
            SkillTier.ELITE: (4.0, 7.0),
            SkillTier.STAR: (2.5, 4.0),
            SkillTier.TOP_SIX: (1.5, 2.5),
            SkillTier.MIDDLE_SIX: (0.5, 1.5),
            SkillTier.BOTTOM_SIX: (-0.5, 0.5),
            SkillTier.FRINGE: (-1.5, -0.5),
            SkillTier.AHL: (-3.0, -1.5),
        }

        min_war, max_war = war_ranges.get(tier, (0, 1))
        percentile = (war - min_war) / (max_war - min_war + 0.01)
        return np.clip(percentile, 0, 1)


class SurplusValueCalculator:
    """
    Calculate surplus value (WAR minus cost).

    Key metric for identifying undervalued players.
    """

    def __init__(self, dollars_per_war: float = 2.5):
        """
        Args:
            dollars_per_war: Millions of $ per WAR (market estimate)
        """
        self.dollars_per_war = dollars_per_war
        self.market_estimator = MarketValueEstimator()

    def calculate_surplus(
        self,
        player: Dict,
        contract: Contract
    ) -> PlayerValue:
        """Calculate player's surplus value."""
        war = player.get('war', 0)
        projected_war = player.get('projected_war', war * 0.9)

        # Dollar value of WAR
        war_value = war * self.dollars_per_war

        # Surplus = value generated - cost
        surplus = war_value - contract.cap_hit

        # Cost per WAR
        cost_per_war = contract.cap_hit / max(war, 0.1)

        # Estimate market value
        tier = self._classify_tier(war)
        market_value = self.market_estimator.estimate_value(
            player,
            tier,
            player.get('age', 25),
            player.get('position', 'C')
        )

        return PlayerValue(
            player_id=player['id'],
            war=war,
            gaa=player.get('gaa', war * 2.5),  # Approximate
            projected_war=projected_war,
            surplus_value=surplus,
            market_value=market_value,
            cost_per_war=cost_per_war
        )

    def _classify_tier(self, war: float) -> SkillTier:
        """Classify player tier from WAR."""
        if war >= 4.0:
            return SkillTier.ELITE
        elif war >= 2.5:
            return SkillTier.STAR
        elif war >= 1.5:
            return SkillTier.TOP_SIX
        elif war >= 0.5:
            return SkillTier.MIDDLE_SIX
        elif war >= -0.5:
            return SkillTier.BOTTOM_SIX
        elif war >= -1.5:
            return SkillTier.FRINGE
        else:
            return SkillTier.AHL


class CapOptimizer:
    """
    Optimize roster construction within cap constraints.
    """

    def __init__(
        self,
        cap_ceiling: float = 83.5,  # 2023-24 cap
        min_roster: int = 20,
        max_roster: int = 23
    ):
        self.cap_ceiling = cap_ceiling
        self.min_roster = min_roster
        self.max_roster = max_roster
        self.surplus_calc = SurplusValueCalculator()

    def analyze_cap_situation(
        self,
        contracts: List[Contract],
        ltir_players: Optional[List[str]] = None
    ) -> CapSituation:
        """Analyze team's current cap situation."""
        ltir_players = ltir_players or []

        active_contracts = [c for c in contracts if c.status == ContractStatus.ACTIVE]
        ltir_contracts = [c for c in contracts if c.player_id in ltir_players]

        current_cap = sum(c.cap_hit for c in active_contracts)
        ltir_space = sum(c.cap_hit for c in ltir_contracts)
        dead_cap = sum(
            c.cap_hit for c in contracts
            if c.status == ContractStatus.BUYOUT
        )

        # Projected next year (expiring contracts)
        expiring = sum(
            c.cap_hit for c in contracts
            if c.years_remaining == 1
        )
        projected_space = self.cap_ceiling - (current_cap - expiring)

        return CapSituation(
            cap_ceiling=self.cap_ceiling,
            cap_floor=self.cap_ceiling * 0.85,  # Approximate
            current_cap_hit=current_cap,
            cap_space=self.cap_ceiling - current_cap,
            projected_cap_space=projected_space,
            ltir_space=ltir_space,
            bonus_cushion=sum(c.performance_bonus for c in active_contracts),
            dead_cap=dead_cap
        )

    def identify_inefficiencies(
        self,
        players: List[Dict],
        contracts: List[Contract]
    ) -> List[Tuple[str, str, float]]:
        """
        Identify overpaid and underpaid players.

        Returns: List of (player_id, status, surplus_value)
        """
        inefficiencies = []

        for player, contract in zip(players, contracts):
            value = self.surplus_calc.calculate_surplus(player, contract)

            if value.surplus_value > 2.0:
                inefficiencies.append((player['id'], 'underpaid', value.surplus_value))
            elif value.surplus_value < -2.0:
                inefficiencies.append((player['id'], 'overpaid', value.surplus_value))

        # Sort by absolute surplus
        inefficiencies.sort(key=lambda x: abs(x[2]), reverse=True)
        return inefficiencies

    def recommend_actions(
        self,
        cap_situation: CapSituation,
        players: List[Dict],
        contracts: List[Contract]
    ) -> List[RosterDecision]:
        """Generate roster recommendations."""
        recommendations = []

        # Check cap health
        if cap_situation.cap_space < 1.0:
            recommendations.append(RosterDecision(
                decision_type="warning",
                player_id="team",
                rationale="Very limited cap space - consider trades or demotions",
                financial_impact=0,
                war_impact=0,
                urgency="immediate"
            ))

        # Identify worst contracts
        values = []
        for player, contract in zip(players, contracts):
            value = self.surplus_calc.calculate_surplus(player, contract)
            values.append((player, contract, value))

        values.sort(key=lambda x: x[2].surplus_value)

        # Recommend action for worst contracts
        for player, contract, value in values[:3]:
            if value.surplus_value < -1.5:
                if contract.years_remaining <= 1:
                    action = "let_expire"
                    rationale = f"Let contract expire - {value.surplus_value:.1f}M in negative surplus"
                elif contract.years_remaining <= 2 and not contract.ntc:
                    action = "trade"
                    rationale = f"Trade candidate - overpaid by ~{-value.surplus_value:.1f}M/year"
                else:
                    action = "buyout_candidate"
                    rationale = f"Consider buyout if cap crunch - {value.cost_per_war:.1f}M per WAR"

                recommendations.append(RosterDecision(
                    decision_type=action,
                    player_id=player['id'],
                    rationale=rationale,
                    financial_impact=contract.cap_hit,
                    war_impact=value.war,
                    urgency="soon" if contract.years_remaining <= 2 else "long_term"
                ))

        return recommendations


class ContractProjector:
    """
    Project future contract values and team cap situations.
    """

    def __init__(self, cap_growth_rate: float = 0.03):
        self.cap_growth = cap_growth_rate
        self.market_estimator = MarketValueEstimator()

    def project_extension_cost(
        self,
        player: Dict,
        current_war: float,
        term_years: int
    ) -> Dict[str, float]:
        """Project cost of contract extension."""
        age = player.get('age', 25)
        position = player.get('position', 'C')
        tier = self._classify_tier_from_war(current_war)

        # Base value
        base_aav = self.market_estimator.estimate_value(
            player, tier, age, position
        )

        # Term premium/discount
        if term_years >= 7:
            term_factor = 0.95  # Discount for long-term
        elif term_years <= 3:
            term_factor = 1.05  # Premium for short-term
        else:
            term_factor = 1.0

        # Age progression factor
        avg_war = 0
        for y in range(term_years):
            future_age = age + y
            age_factor = self._age_curve_value(future_age)
            avg_war += current_war * age_factor
        avg_war /= term_years

        projected_aav = base_aav * term_factor * (avg_war / current_war)

        return {
            'projected_aav': projected_aav,
            'total_value': projected_aav * term_years,
            'avg_projected_war': avg_war,
            'cap_percentage': projected_aav / 83.5 * 100  # Current cap
        }

    def project_roster_cap(
        self,
        contracts: List[Contract],
        years_ahead: int = 3
    ) -> List[Dict]:
        """Project roster cap situation for future years."""
        projections = []

        for year in range(years_ahead):
            # Active contracts in that year
            active = [
                c for c in contracts
                if c.years_remaining > year
            ]

            cap_hit = sum(c.cap_hit for c in active)
            expiring = sum(
                c.cap_hit for c in active
                if c.years_remaining == year + 1
            )

            # Projected cap ceiling
            projected_ceiling = 83.5 * (1 + self.cap_growth) ** year

            projections.append({
                'year': year + 1,
                'committed_cap': cap_hit,
                'expiring': expiring,
                'projected_ceiling': projected_ceiling,
                'projected_space': projected_ceiling - cap_hit,
                'n_contracts': len(active)
            })

        return projections

    def _classify_tier_from_war(self, war: float) -> SkillTier:
        """Classify tier from WAR."""
        if war >= 4.0:
            return SkillTier.ELITE
        elif war >= 2.5:
            return SkillTier.STAR
        elif war >= 1.5:
            return SkillTier.TOP_SIX
        elif war >= 0.5:
            return SkillTier.MIDDLE_SIX
        else:
            return SkillTier.BOTTOM_SIX

    def _age_curve_value(self, age: int) -> float:
        """Get age curve multiplier."""
        if age < 22:
            return 0.7 + (age - 18) * 0.075
        elif age < 28:
            return 1.0
        elif age < 32:
            return 1.0 - (age - 27) * 0.05
        else:
            return max(0.3, 0.8 - (age - 31) * 0.08)


class TradeAnalyzer:
    """
    Analyze potential trades for cap and value impact.
    """

    def __init__(self):
        self.surplus_calc = SurplusValueCalculator()

    def analyze_trade(
        self,
        outgoing_players: List[Dict],
        outgoing_contracts: List[Contract],
        incoming_players: List[Dict],
        incoming_contracts: List[Contract],
        draft_picks_out: Optional[Dict[str, int]] = None,
        draft_picks_in: Optional[Dict[str, int]] = None
    ) -> TradeScenario:
        """Analyze a potential trade."""
        # Calculate WAR change
        war_out = sum(p.get('war', 0) for p in outgoing_players)
        war_in = sum(p.get('war', 0) for p in incoming_players)
        war_change = war_in - war_out

        # Calculate cap impact
        cap_out = sum(c.cap_hit for c in outgoing_contracts)
        cap_in = sum(c.cap_hit for c in incoming_contracts)
        cap_impact = cap_in - cap_out

        # Calculate surplus change
        surplus_out = sum(
            self.surplus_calc.calculate_surplus(p, c).surplus_value
            for p, c in zip(outgoing_players, outgoing_contracts)
        )
        surplus_in = sum(
            self.surplus_calc.calculate_surplus(p, c).surplus_value
            for p, c in zip(incoming_players, incoming_contracts)
        )
        surplus_change = surplus_in - surplus_out

        # Draft pick value (rough approximation)
        pick_values = {1: 5.0, 2: 2.5, 3: 1.5, 4: 1.0, 5: 0.5, 6: 0.3, 7: 0.2}
        picks_out_value = sum(
            pick_values.get(r, 0.1) for r in (draft_picks_out or {}).values()
        )
        picks_in_value = sum(
            pick_values.get(r, 0.1) for r in (draft_picks_in or {}).values()
        )

        # Generate recommendation
        if surplus_change > 1.0:
            recommendation = "STRONGLY FAVOR - significant surplus gain"
        elif surplus_change > 0:
            recommendation = "FAVOR - positive value"
        elif surplus_change > -1.0:
            recommendation = "NEUTRAL - marginal impact"
        else:
            recommendation = "OPPOSE - negative value"

        # Adjust for cap impact
        if cap_impact < -2.0:
            recommendation += " (cap relief bonus)"
        elif cap_impact > 2.0:
            recommendation += " (cap hit concern)"

        return TradeScenario(
            players_out=[p['id'] for p in outgoing_players],
            players_in=[p['id'] for p in incoming_players],
            cap_impact=cap_impact,
            war_change=war_change,
            surplus_change=surplus_change + picks_in_value - picks_out_value,
            draft_picks=draft_picks_in or {},
            recommendation=recommendation
        )
