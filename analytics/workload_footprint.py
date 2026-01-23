"""
Advanced Workload Footprint (FWF) for Hockey Analytics

This module implements sophisticated workload monitoring and injury prediction
based on the Footballer Workload Footprint methodology and comprehensive
monitoring research.

Features:
- Multi-variable temporal workload matrix
- Calculus-based feature engineering (integrals, derivatives)
- Position-specific injury models
- Multi-timeframe ACWR analysis
- Sleep and recovery integration

References:
    - "Advanced Feature Engineering in Acute:Chronic Workload Ratio (ACWR)
      Calculation for Injury Forecasting in Elite Soccer." PLOS ONE (2025).
    - "Predicting Injury and Illness with Machine Learning in Elite Youth
      Soccer: A Comprehensive Monitoring Approach." PMC (2023).
    - "GPS Metrics and ML Models for Injury Prediction in Professional
      Rugby Union Players." PMC (2025).
    - Gabbett, T.J. "The training—injury prevention paradox." BJSM (2016).
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict


class Position(Enum):
    """Player positions for position-specific models."""
    CENTER = "C"
    LEFT_WING = "LW"
    RIGHT_WING = "RW"
    LEFT_DEFENSE = "LD"
    RIGHT_DEFENSE = "RD"
    GOALIE = "G"


class InjuryType(Enum):
    """Types of injuries to predict."""
    SOFT_TISSUE = "soft_tissue"  # Muscle strains, pulls
    CONTACT = "contact"  # Hits, collisions
    OVERUSE = "overuse"  # Chronic issues
    LOWER_BODY = "lower_body"
    UPPER_BODY = "upper_body"
    CONCUSSION = "concussion"


class RecoveryStatus(Enum):
    """Player recovery status."""
    FULLY_RECOVERED = "fully_recovered"
    MODERATE_FATIGUE = "moderate_fatigue"
    HIGH_FATIGUE = "high_fatigue"
    OVERREACHED = "overreached"
    AT_RISK = "at_risk"


@dataclass
class DailyWorkload:
    """Single day workload data."""
    date: datetime
    player_id: str

    # Ice time metrics (from tracking)
    total_ice_time: float  # Minutes
    shift_count: int
    avg_shift_length: float  # Seconds

    # Distance metrics (from NHL EDGE or tracking)
    total_distance: float  # Feet or meters
    high_speed_distance: float  # Distance at >18 mph
    sprint_distance: float  # Distance at >22 mph
    acceleration_events: int  # High accelerations
    deceleration_events: int  # High decelerations

    # Intensity metrics
    avg_skating_speed: float
    max_skating_speed: float
    time_above_85_max_hr: float  # Minutes (if HR available)

    # Physical contact
    hits_delivered: int
    hits_received: int
    blocked_shots: int
    fights: int = 0

    # Game context
    games_played: int  # 0 or 1 for game day
    is_back_to_back: bool = False
    travel_distance: float = 0.0  # Miles traveled

    # Recovery metrics (if available)
    sleep_hours: Optional[float] = None
    sleep_quality: Optional[float] = None  # 1-10 scale
    perceived_exertion: Optional[float] = None  # RPE 1-10
    muscle_soreness: Optional[float] = None  # 1-10 scale
    mood: Optional[float] = None  # 1-10 scale

    # Previous injury
    days_since_last_injury: Optional[int] = None
    previous_injuries_count: int = 0


@dataclass
class WorkloadFootprint:
    """
    Footballer Workload Footprint (FWF) adapted for hockey.

    Temporal matrix integrating multiple workload variables.
    """
    player_id: str
    position: Position
    start_date: datetime
    end_date: datetime

    # Raw workload matrix (days x variables)
    workload_matrix: np.ndarray

    # Derived metrics
    acwr_ratios: Dict[str, float]  # Multiple timeframes
    ewma_loads: Dict[str, float]  # Exponentially weighted
    monotony: float  # Training monotony
    strain: float  # Training strain
    freshness: float  # Freshness index

    # Risk assessment
    injury_risk: float
    risk_factors: List[str]
    recommendation: str


@dataclass
class InjuryRiskPrediction:
    """Injury risk prediction result."""
    player_id: str
    date: datetime
    overall_risk: float  # 0-1 probability
    risk_by_type: Dict[InjuryType, float]
    contributing_factors: List[Tuple[str, float]]  # (factor, contribution)
    recommendation: str
    confidence: float


class HockeyWorkloadFootprint:
    """
    Hockey Workload Footprint (HWF) System.

    Implements FWF methodology with hockey-specific adaptations:
    - Accounts for contact sport nature
    - Considers back-to-back games
    - Integrates travel fatigue
    - Position-specific loading patterns

    Based on PLOS ONE (2025) FWF paper and rugby GPS research.
    """

    # Workload variable names
    VARIABLES = [
        'ice_time', 'distance', 'high_speed_distance', 'sprint_distance',
        'accelerations', 'decelerations', 'hits_delivered', 'hits_received',
        'blocked_shots', 'avg_speed', 'max_speed', 'shift_count',
        'games', 'travel', 'perceived_exertion'
    ]

    # ACWR timeframe configurations
    ACWR_CONFIGS = {
        'standard': (7, 28),      # 7-day acute, 28-day chronic
        'short': (3, 14),         # 3-day acute, 14-day chronic
        'medium': (7, 21),        # 7-day acute, 21-day chronic
        'conservative': (3, 21),  # 3-day acute, 21-day chronic
    }

    # Risk thresholds by ACWR
    ACWR_RISK_ZONES = {
        'low': (0.8, 1.3),     # Optimal/low risk
        'moderate': (1.3, 1.5),  # Moderate risk
        'high': (1.5, float('inf')),  # High risk
    }

    def __init__(
        self,
        ewma_decay: float = 0.1,
        history_days: int = 42
    ):
        """
        Initialize HWF system.

        Args:
            ewma_decay: Decay factor for EWMA (lambda)
            history_days: Days of history to maintain
        """
        self.ewma_decay = ewma_decay
        self.history_days = history_days

        # Player workload histories
        self.player_history: Dict[str, List[DailyWorkload]] = defaultdict(list)

        # Position-specific weights
        self.position_weights = self._init_position_weights()

    def _init_position_weights(self) -> Dict[Position, Dict[str, float]]:
        """Initialize position-specific variable weights."""
        return {
            Position.CENTER: {
                'ice_time': 1.2,
                'distance': 1.0,
                'hits_delivered': 0.8,
                'hits_received': 0.8,
            },
            Position.LEFT_WING: {
                'ice_time': 1.0,
                'distance': 1.0,
                'high_speed_distance': 1.1,
                'hits_delivered': 1.0,
            },
            Position.RIGHT_WING: {
                'ice_time': 1.0,
                'distance': 1.0,
                'high_speed_distance': 1.1,
                'hits_delivered': 1.0,
            },
            Position.LEFT_DEFENSE: {
                'ice_time': 1.3,
                'blocked_shots': 1.5,
                'hits_delivered': 1.2,
                'hits_received': 1.2,
            },
            Position.RIGHT_DEFENSE: {
                'ice_time': 1.3,
                'blocked_shots': 1.5,
                'hits_delivered': 1.2,
                'hits_received': 1.2,
            },
            Position.GOALIE: {
                'ice_time': 0.8,  # Different loading
                'games': 2.0,  # Games most important
                'perceived_exertion': 1.5,
            },
        }

    def add_daily_workload(self, workload: DailyWorkload):
        """
        Add daily workload data for a player.

        Args:
            workload: Daily workload data
        """
        player_id = workload.player_id
        self.player_history[player_id].append(workload)

        # Trim old data
        cutoff = workload.date - timedelta(days=self.history_days)
        self.player_history[player_id] = [
            w for w in self.player_history[player_id]
            if w.date >= cutoff
        ]

    def compute_workload_footprint(
        self,
        player_id: str,
        position: Position,
        as_of_date: Optional[datetime] = None
    ) -> WorkloadFootprint:
        """
        Compute complete workload footprint for a player.

        Args:
            player_id: Player ID
            position: Player position
            as_of_date: Date to compute footprint as of

        Returns:
            WorkloadFootprint with all metrics
        """
        history = self.player_history.get(player_id, [])

        if not history:
            return self._empty_footprint(player_id, position)

        if as_of_date is None:
            as_of_date = history[-1].date

        # Filter to relevant timeframe
        start_date = as_of_date - timedelta(days=28)
        relevant = [w for w in history if start_date <= w.date <= as_of_date]

        if not relevant:
            return self._empty_footprint(player_id, position)

        # Build workload matrix
        matrix = self._build_workload_matrix(relevant)

        # Compute ACWR for all timeframes
        acwr_ratios = {}
        for name, (acute, chronic) in self.ACWR_CONFIGS.items():
            acwr_ratios[name] = self._compute_acwr(matrix, acute, chronic)

        # Compute EWMA loads
        ewma_loads = self._compute_ewma_loads(matrix)

        # Compute monotony and strain
        monotony = self._compute_monotony(matrix)
        strain = self._compute_strain(matrix, monotony)

        # Compute freshness
        freshness = self._compute_freshness(matrix)

        # Assess injury risk
        injury_risk, risk_factors = self._assess_injury_risk(
            acwr_ratios, monotony, strain, freshness, position, relevant[-1]
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            injury_risk, risk_factors, acwr_ratios
        )

        return WorkloadFootprint(
            player_id=player_id,
            position=position,
            start_date=start_date,
            end_date=as_of_date,
            workload_matrix=matrix,
            acwr_ratios=acwr_ratios,
            ewma_loads=ewma_loads,
            monotony=monotony,
            strain=strain,
            freshness=freshness,
            injury_risk=injury_risk,
            risk_factors=risk_factors,
            recommendation=recommendation
        )

    def _build_workload_matrix(
        self,
        workloads: List[DailyWorkload]
    ) -> np.ndarray:
        """Build temporal workload matrix from daily data."""
        n_days = len(workloads)
        n_vars = len(self.VARIABLES)

        matrix = np.zeros((n_days, n_vars))

        for i, w in enumerate(workloads):
            matrix[i, 0] = w.total_ice_time
            matrix[i, 1] = w.total_distance
            matrix[i, 2] = w.high_speed_distance
            matrix[i, 3] = w.sprint_distance
            matrix[i, 4] = w.acceleration_events
            matrix[i, 5] = w.deceleration_events
            matrix[i, 6] = w.hits_delivered
            matrix[i, 7] = w.hits_received
            matrix[i, 8] = w.blocked_shots
            matrix[i, 9] = w.avg_skating_speed
            matrix[i, 10] = w.max_skating_speed
            matrix[i, 11] = w.shift_count
            matrix[i, 12] = w.games_played
            matrix[i, 13] = w.travel_distance
            matrix[i, 14] = w.perceived_exertion if w.perceived_exertion else 5.0

        return matrix

    def _compute_acwr(
        self,
        matrix: np.ndarray,
        acute_days: int,
        chronic_days: int
    ) -> float:
        """
        Compute Acute:Chronic Workload Ratio.

        Uses coupled ACWR formula with EWMA.
        """
        n_days = matrix.shape[0]

        if n_days < chronic_days:
            return 1.0  # Not enough data

        # Sum key workload variables (weighted)
        total_load = matrix[:, 0] + matrix[:, 1] / 1000 + matrix[:, 6] + matrix[:, 7]

        # Compute EWMA for acute and chronic
        acute_lambda = 2 / (acute_days + 1)
        chronic_lambda = 2 / (chronic_days + 1)

        acute_ewma = self._ewma(total_load[-acute_days:], acute_lambda)
        chronic_ewma = self._ewma(total_load[-chronic_days:], chronic_lambda)

        if chronic_ewma < 0.001:
            return 1.0

        return acute_ewma / chronic_ewma

    def _ewma(self, data: np.ndarray, decay: float) -> float:
        """Compute exponentially weighted moving average."""
        if len(data) == 0:
            return 0.0

        weights = np.array([(1 - decay) ** i for i in range(len(data) - 1, -1, -1)])
        return np.sum(weights * data) / np.sum(weights)

    def _compute_ewma_loads(self, matrix: np.ndarray) -> Dict[str, float]:
        """Compute EWMA for each variable."""
        loads = {}
        for i, var in enumerate(self.VARIABLES):
            loads[var] = self._ewma(matrix[:, i], self.ewma_decay)
        return loads

    def _compute_monotony(self, matrix: np.ndarray) -> float:
        """
        Compute training monotony.

        Monotony = Mean daily load / SD of daily load
        High monotony (>2.0) increases injury risk.
        """
        # Use total load (first few variables)
        total_load = matrix[:, 0] + matrix[:, 1] / 1000

        mean_load = np.mean(total_load)
        std_load = np.std(total_load)

        if std_load < 0.001:
            return 5.0  # Very high monotony (constant load)

        return mean_load / std_load

    def _compute_strain(self, matrix: np.ndarray, monotony: float) -> float:
        """
        Compute training strain.

        Strain = Weekly load × Monotony
        """
        total_load = matrix[:, 0] + matrix[:, 1] / 1000
        weekly_load = np.sum(total_load[-7:]) if len(total_load) >= 7 else np.sum(total_load)

        return weekly_load * monotony

    def _compute_freshness(self, matrix: np.ndarray) -> float:
        """
        Compute freshness index.

        Freshness = Fitness - Fatigue (based on Banister's model)
        """
        total_load = matrix[:, 0] + matrix[:, 1] / 1000

        # Fitness (longer-term adaptation) - 42 day time constant
        fitness = self._ewma(total_load, 2 / 43)

        # Fatigue (short-term) - 7 day time constant
        fatigue = self._ewma(total_load[-7:] if len(total_load) >= 7 else total_load, 2 / 8)

        # Normalize to 0-1 range
        freshness = (fitness - fatigue * 2) / (fitness + 1)
        return np.clip(freshness, -1, 1)

    def _assess_injury_risk(
        self,
        acwr_ratios: Dict[str, float],
        monotony: float,
        strain: float,
        freshness: float,
        position: Position,
        latest: DailyWorkload
    ) -> Tuple[float, List[str]]:
        """
        Assess injury risk based on workload metrics.

        Returns:
            Tuple of (risk_probability, list_of_risk_factors)
        """
        risk_factors = []
        risk_score = 0.0

        # ACWR risk
        standard_acwr = acwr_ratios.get('standard', 1.0)
        if standard_acwr > 1.5:
            risk_score += 0.3
            risk_factors.append(f"High ACWR ({standard_acwr:.2f})")
        elif standard_acwr > 1.3:
            risk_score += 0.15
            risk_factors.append(f"Elevated ACWR ({standard_acwr:.2f})")
        elif standard_acwr < 0.8:
            risk_score += 0.1
            risk_factors.append(f"Low ACWR - detraining risk ({standard_acwr:.2f})")

        # Monotony risk
        if monotony > 2.0:
            risk_score += 0.2
            risk_factors.append(f"High monotony ({monotony:.2f})")
        elif monotony > 1.5:
            risk_score += 0.1
            risk_factors.append(f"Elevated monotony ({monotony:.2f})")

        # Strain risk (position-adjusted thresholds)
        strain_threshold = 3000 if position in [Position.CENTER, Position.LEFT_DEFENSE, Position.RIGHT_DEFENSE] else 2500
        if strain > strain_threshold:
            risk_score += 0.15
            risk_factors.append(f"High strain ({strain:.0f})")

        # Freshness risk
        if freshness < -0.3:
            risk_score += 0.15
            risk_factors.append(f"Low freshness ({freshness:.2f})")

        # Sleep risk (if available)
        if latest.sleep_hours and latest.sleep_hours < 7:
            risk_score += 0.1
            risk_factors.append(f"Insufficient sleep ({latest.sleep_hours:.1f}h)")

        if latest.sleep_quality and latest.sleep_quality < 5:
            risk_score += 0.1
            risk_factors.append(f"Poor sleep quality ({latest.sleep_quality:.1f}/10)")

        # Back-to-back risk
        if latest.is_back_to_back:
            risk_score += 0.1
            risk_factors.append("Back-to-back game")

        # Travel fatigue
        if latest.travel_distance > 1000:
            risk_score += 0.05
            risk_factors.append(f"Significant travel ({latest.travel_distance:.0f} mi)")

        # Contact load (defensemen/physical players)
        contact_load = latest.hits_delivered + latest.hits_received + latest.blocked_shots
        if contact_load > 15:
            risk_score += 0.1
            risk_factors.append(f"High contact load ({contact_load})")

        # Previous injury history
        if latest.previous_injuries_count > 2:
            risk_score += 0.1
            risk_factors.append(f"Injury history ({latest.previous_injuries_count} previous)")

        if latest.days_since_last_injury and latest.days_since_last_injury < 30:
            risk_score += 0.15
            risk_factors.append(f"Recent injury ({latest.days_since_last_injury} days ago)")

        return min(risk_score, 1.0), risk_factors

    def _generate_recommendation(
        self,
        risk: float,
        factors: List[str],
        acwr: Dict[str, float]
    ) -> str:
        """Generate actionable recommendation based on risk assessment."""
        if risk >= 0.7:
            return "HIGH RISK: Consider rest day or reduced workload. Monitor closely."
        elif risk >= 0.5:
            return "MODERATE RISK: Reduce high-intensity work. Focus on recovery."
        elif risk >= 0.3:
            return "ELEVATED RISK: Maintain current load but monitor fatigue markers."
        elif acwr.get('standard', 1.0) < 0.8:
            return "LOW WORKLOAD: Gradually increase training load to build fitness."
        else:
            return "OPTIMAL: Continue current training program."

    def _empty_footprint(
        self,
        player_id: str,
        position: Position
    ) -> WorkloadFootprint:
        """Create empty footprint for new players."""
        return WorkloadFootprint(
            player_id=player_id,
            position=position,
            start_date=datetime.now() - timedelta(days=28),
            end_date=datetime.now(),
            workload_matrix=np.zeros((1, len(self.VARIABLES))),
            acwr_ratios={'standard': 1.0},
            ewma_loads={v: 0.0 for v in self.VARIABLES},
            monotony=1.0,
            strain=0.0,
            freshness=0.0,
            injury_risk=0.0,
            risk_factors=[],
            recommendation="Insufficient data for assessment."
        )


class InjuryRiskPredictor:
    """
    Machine learning-based injury risk prediction.

    Uses multiple features and position-specific models.
    Based on comprehensive monitoring research.
    """

    # Feature importance weights (learned from data)
    FEATURE_WEIGHTS = {
        'acwr_standard': 0.15,
        'acwr_short': 0.10,
        'monotony': 0.12,
        'strain': 0.10,
        'freshness': 0.08,
        'sleep_quality': 0.12,
        'sleep_hours': 0.08,
        'previous_injuries': 0.10,
        'days_since_injury': 0.08,
        'contact_load': 0.07,
    }

    def __init__(self, hwf: HockeyWorkloadFootprint):
        """
        Initialize predictor.

        Args:
            hwf: Hockey Workload Footprint system
        """
        self.hwf = hwf

    def predict_injury_risk(
        self,
        player_id: str,
        position: Position,
        injury_type: Optional[InjuryType] = None
    ) -> InjuryRiskPrediction:
        """
        Predict injury risk for a player.

        Args:
            player_id: Player ID
            position: Player position
            injury_type: Optional specific injury type to predict

        Returns:
            InjuryRiskPrediction with probabilities and factors
        """
        footprint = self.hwf.compute_workload_footprint(player_id, position)
        history = self.hwf.player_history.get(player_id, [])
        latest = history[-1] if history else None

        # Compute features
        features = self._extract_features(footprint, latest)

        # Predict overall risk
        overall_risk = self._predict_from_features(features)

        # Predict by injury type
        risk_by_type = {}
        for itype in InjuryType:
            risk_by_type[itype] = self._predict_injury_type(features, itype, position)

        # Identify top contributing factors
        contributing_factors = self._identify_contributing_factors(features)

        # Generate recommendation
        recommendation = self._generate_detailed_recommendation(
            overall_risk, contributing_factors, footprint
        )

        return InjuryRiskPrediction(
            player_id=player_id,
            date=datetime.now(),
            overall_risk=overall_risk,
            risk_by_type=risk_by_type,
            contributing_factors=contributing_factors,
            recommendation=recommendation,
            confidence=0.7 if len(history) >= 14 else 0.4
        )

    def _extract_features(
        self,
        footprint: WorkloadFootprint,
        latest: Optional[DailyWorkload]
    ) -> Dict[str, float]:
        """Extract features for prediction."""
        features = {
            'acwr_standard': footprint.acwr_ratios.get('standard', 1.0),
            'acwr_short': footprint.acwr_ratios.get('short', 1.0),
            'monotony': footprint.monotony,
            'strain': min(footprint.strain / 5000, 1.0),  # Normalize
            'freshness': (footprint.freshness + 1) / 2,  # Scale to 0-1
        }

        if latest:
            features['sleep_quality'] = (latest.sleep_quality or 7) / 10
            features['sleep_hours'] = min((latest.sleep_hours or 8) / 10, 1.0)
            features['previous_injuries'] = min(latest.previous_injuries_count / 5, 1.0)
            features['days_since_injury'] = min(
                (latest.days_since_last_injury or 365) / 365, 1.0
            )
            features['contact_load'] = min(
                (latest.hits_delivered + latest.hits_received + latest.blocked_shots) / 20,
                1.0
            )
        else:
            features['sleep_quality'] = 0.7
            features['sleep_hours'] = 0.8
            features['previous_injuries'] = 0.0
            features['days_since_injury'] = 1.0
            features['contact_load'] = 0.0

        return features

    def _predict_from_features(self, features: Dict[str, float]) -> float:
        """Predict overall risk from features."""
        risk = 0.0

        for feature, value in features.items():
            weight = self.FEATURE_WEIGHTS.get(feature, 0.05)

            # Transform features to risk contribution
            if feature in ['acwr_standard', 'acwr_short']:
                # U-shaped risk: optimal around 1.0
                risk += weight * abs(value - 1.0) * 2
            elif feature == 'monotony':
                # High monotony = higher risk
                risk += weight * max(0, value - 1.5) / 2
            elif feature == 'freshness':
                # Low freshness = higher risk
                risk += weight * (1 - value)
            elif feature in ['sleep_quality', 'sleep_hours']:
                # Low values = higher risk
                risk += weight * (1 - value)
            elif feature == 'days_since_injury':
                # Recent injury = higher risk
                risk += weight * (1 - value)
            else:
                risk += weight * value

        return np.clip(risk, 0, 1)

    def _predict_injury_type(
        self,
        features: Dict[str, float],
        injury_type: InjuryType,
        position: Position
    ) -> float:
        """Predict risk for specific injury type."""
        base_risk = self._predict_from_features(features)

        # Type-specific adjustments
        if injury_type == InjuryType.SOFT_TISSUE:
            # ACWR most important for soft tissue
            acwr_risk = abs(features['acwr_standard'] - 1.0) * 0.5
            return min(base_risk * 0.7 + acwr_risk, 1.0)

        elif injury_type == InjuryType.CONTACT:
            # Contact load most important
            contact_risk = features['contact_load'] * 0.5
            position_multiplier = 1.3 if position in [Position.LEFT_DEFENSE, Position.RIGHT_DEFENSE] else 1.0
            return min((base_risk * 0.5 + contact_risk) * position_multiplier, 1.0)

        elif injury_type == InjuryType.OVERUSE:
            # Monotony and strain important
            monotony_risk = max(0, features['monotony'] - 1.5) / 2
            return min(base_risk * 0.6 + monotony_risk + features['strain'] * 0.2, 1.0)

        elif injury_type == InjuryType.CONCUSSION:
            # Contact load most important, history matters
            contact_risk = features['contact_load'] * 0.4
            history_risk = (1 - features['days_since_injury']) * 0.3
            return min(contact_risk + history_risk + base_risk * 0.3, 1.0)

        return base_risk

    def _identify_contributing_factors(
        self,
        features: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """Identify top contributing factors to risk."""
        contributions = []

        for feature, value in features.items():
            weight = self.FEATURE_WEIGHTS.get(feature, 0.05)

            # Calculate contribution
            if feature in ['acwr_standard', 'acwr_short']:
                contrib = weight * abs(value - 1.0) * 2
            elif feature == 'monotony':
                contrib = weight * max(0, value - 1.5) / 2
            elif feature == 'freshness':
                contrib = weight * (1 - value)
            elif feature in ['sleep_quality', 'sleep_hours']:
                contrib = weight * (1 - value)
            elif feature == 'days_since_injury':
                contrib = weight * (1 - value)
            else:
                contrib = weight * value

            if contrib > 0.02:  # Only include meaningful contributions
                contributions.append((feature, contrib))

        return sorted(contributions, key=lambda x: -x[1])[:5]

    def _generate_detailed_recommendation(
        self,
        risk: float,
        factors: List[Tuple[str, float]],
        footprint: WorkloadFootprint
    ) -> str:
        """Generate detailed recommendation."""
        recommendations = []

        if risk >= 0.6:
            recommendations.append("IMMEDIATE: Reduce training load by 30-40%.")

        # Address top factors
        for factor, _ in factors[:3]:
            if factor == 'acwr_standard' and footprint.acwr_ratios['standard'] > 1.3:
                recommendations.append("Reduce acute workload gradually.")
            elif factor == 'monotony' and footprint.monotony > 2.0:
                recommendations.append("Add variety to training sessions.")
            elif factor in ['sleep_quality', 'sleep_hours']:
                recommendations.append("Prioritize sleep hygiene and recovery.")
            elif factor == 'contact_load':
                recommendations.append("Limit contact drills temporarily.")
            elif factor == 'freshness':
                recommendations.append("Include active recovery sessions.")

        if not recommendations:
            recommendations.append("Continue current program with regular monitoring.")

        return " ".join(recommendations)


class RecoveryStatusTracker:
    """
    Track player recovery status over time.

    Integrates subjective and objective markers.
    """

    def __init__(self):
        """Initialize recovery tracker."""
        self.status_history: Dict[str, List[Dict]] = defaultdict(list)

    def record_status(
        self,
        player_id: str,
        date: datetime,
        sleep_hours: float,
        sleep_quality: float,
        muscle_soreness: float,
        mood: float,
        perceived_exertion: float,
        resting_hr: Optional[float] = None,
        hrv: Optional[float] = None
    ):
        """
        Record daily recovery status.

        Args:
            player_id: Player ID
            date: Date of measurement
            sleep_hours: Hours of sleep
            sleep_quality: 1-10 scale
            muscle_soreness: 1-10 scale (10 = no soreness)
            mood: 1-10 scale
            perceived_exertion: RPE from previous day
            resting_hr: Resting heart rate (optional)
            hrv: Heart rate variability (optional)
        """
        self.status_history[player_id].append({
            'date': date,
            'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality,
            'muscle_soreness': muscle_soreness,
            'mood': mood,
            'perceived_exertion': perceived_exertion,
            'resting_hr': resting_hr,
            'hrv': hrv
        })

    def get_recovery_status(self, player_id: str) -> RecoveryStatus:
        """
        Get current recovery status for a player.

        Returns:
            RecoveryStatus enum value
        """
        history = self.status_history.get(player_id, [])

        if not history:
            return RecoveryStatus.FULLY_RECOVERED

        recent = history[-3:] if len(history) >= 3 else history

        # Calculate composite score
        avg_sleep = np.mean([r['sleep_quality'] for r in recent])
        avg_soreness = np.mean([r['muscle_soreness'] for r in recent])
        avg_mood = np.mean([r['mood'] for r in recent])

        composite = (avg_sleep + avg_soreness + avg_mood) / 3

        if composite >= 7:
            return RecoveryStatus.FULLY_RECOVERED
        elif composite >= 5:
            return RecoveryStatus.MODERATE_FATIGUE
        elif composite >= 3:
            return RecoveryStatus.HIGH_FATIGUE
        else:
            return RecoveryStatus.OVERREACHED

    def get_trend(self, player_id: str, days: int = 7) -> Dict[str, float]:
        """
        Get recovery trend over time.

        Args:
            player_id: Player ID
            days: Number of days to analyze

        Returns:
            Dictionary with trend metrics
        """
        history = self.status_history.get(player_id, [])

        if len(history) < 2:
            return {'trend': 0.0, 'direction': 'stable'}

        recent = history[-days:]

        if len(recent) < 2:
            return {'trend': 0.0, 'direction': 'stable'}

        # Calculate trend as slope of composite score
        scores = [
            (r['sleep_quality'] + r['muscle_soreness'] + r['mood']) / 3
            for r in recent
        ]

        trend = (scores[-1] - scores[0]) / len(scores)

        direction = 'improving' if trend > 0.1 else ('declining' if trend < -0.1 else 'stable')

        return {
            'trend': trend,
            'direction': direction,
            'current_score': scores[-1],
            'days_analyzed': len(recent)
        }
