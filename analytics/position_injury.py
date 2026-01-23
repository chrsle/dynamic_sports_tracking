"""
Position-Specific Injury Prediction Models

Implements methodology from:
- "Global Positioning System-Derived Metrics and Machine Learning Models for
  Injury Prediction in Professional Rugby Union Players." PMC (2025).

Key features:
- 17 GPS-derived metrics
- Multiple EWMA ACWR ratios (3:14, 3:21, 7:21, 7:28)
- Monotony and strain calculations
- Position-specific models

Hockey translation:
- Position-specific models for forwards vs. defensemen vs. goalies
- Multiple ACWR timeframes for different injury types
- Account for contact (hits, blocks) and non-contact workload
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta


class HockeyPosition(Enum):
    """Hockey positions with different injury profiles."""
    CENTER = "center"
    LEFT_WING = "left_wing"
    RIGHT_WING = "right_wing"
    LEFT_DEFENSE = "left_defense"
    RIGHT_DEFENSE = "right_defense"
    GOALIE = "goalie"


class InjuryType(Enum):
    """Common hockey injury types."""
    LOWER_BODY = "lower_body"
    UPPER_BODY = "upper_body"
    GROIN = "groin"
    CONCUSSION = "concussion"
    BACK = "back"
    SHOULDER = "shoulder"
    KNEE = "knee"
    ANKLE = "ankle"
    HAND_WRIST = "hand_wrist"
    ILLNESS = "illness"


class SeverityLevel(Enum):
    """Injury severity."""
    MINOR = "minor"  # 1-7 days
    MODERATE = "moderate"  # 8-28 days
    SEVERE = "severe"  # 29+ days
    CAREER_ENDING = "career_ending"


@dataclass
class DailyMetrics:
    """Daily workload metrics from tracking."""
    date: datetime
    player_id: str
    position: HockeyPosition

    # Time metrics
    ice_time: float  # minutes
    shifts: int
    avg_shift_length: float  # seconds

    # Distance metrics
    total_distance: float  # meters
    high_speed_distance: float  # meters (>24 km/h)
    sprint_distance: float  # meters (>28 km/h)

    # Acceleration/deceleration
    accelerations: int  # events > 3 m/s^2
    decelerations: int  # events < -3 m/s^2
    high_intensity_actions: int  # combined accel + decel

    # Contact metrics
    hits_delivered: int
    hits_received: int
    blocked_shots: int

    # Intensity metrics
    player_load: float  # arbitrary units from accelerometer
    metabolic_power: float  # watts
    heart_rate_avg: float
    heart_rate_max: float

    # Position-specific
    faceoffs: int = 0  # Centers
    slot_time: float = 0.0  # Forwards
    defensive_zone_time: float = 0.0  # Defense
    saves: int = 0  # Goalies
    high_danger_saves: int = 0  # Goalies

    @property
    def contact_load(self) -> float:
        """Total contact workload."""
        return self.hits_delivered + self.hits_received * 1.2 + self.blocked_shots * 0.8


@dataclass
class ACWRResult:
    """Result of ACWR calculation."""
    ratio: float
    acute_load: float
    chronic_load: float
    risk_zone: str  # low, moderate, high, very_high
    trend: str  # increasing, stable, decreasing


@dataclass
class InjuryRisk:
    """Injury risk prediction."""
    player_id: str
    date: datetime
    overall_risk: float  # 0-1 probability
    risk_by_type: Dict[InjuryType, float]
    risk_factors: List[str]
    recommendations: List[str]
    confidence: float


@dataclass
class TrainingRecommendation:
    """Load management recommendation."""
    player_id: str
    date: datetime
    recommended_ice_time: float  # minutes
    max_high_intensity: float
    avoid_contact: bool
    recovery_focus: List[str]
    rationale: str


class ACWRCalculator:
    """
    Calculate ACWR using various timeframes.

    Supports multiple ratio configurations for different
    injury types and positions.
    """

    def __init__(
        self,
        acute_days: int = 7,
        chronic_days: int = 28,
        smoothing: str = "ewma"  # "ewma" or "rolling"
    ):
        self.acute_days = acute_days
        self.chronic_days = chronic_days
        self.smoothing = smoothing

    def calculate_ewma(
        self,
        values: np.ndarray,
        days: int
    ) -> np.ndarray:
        """Calculate exponentially weighted moving average."""
        alpha = 2 / (days + 1)
        ewma = np.zeros_like(values)
        ewma[0] = values[0]

        for i in range(1, len(values)):
            ewma[i] = alpha * values[i] + (1 - alpha) * ewma[i-1]

        return ewma

    def calculate_rolling(
        self,
        values: np.ndarray,
        days: int
    ) -> np.ndarray:
        """Calculate rolling average."""
        result = np.zeros_like(values)
        for i in range(len(values)):
            start = max(0, i - days + 1)
            result[i] = np.mean(values[start:i+1])
        return result

    def compute_acwr(
        self,
        daily_loads: np.ndarray
    ) -> np.ndarray:
        """
        Compute ACWR time series.

        Returns array of ACWR values, one per day.
        """
        if self.smoothing == "ewma":
            acute = self.calculate_ewma(daily_loads, self.acute_days)
            chronic = self.calculate_ewma(daily_loads, self.chronic_days)
        else:
            acute = self.calculate_rolling(daily_loads, self.acute_days)
            chronic = self.calculate_rolling(daily_loads, self.chronic_days)

        # Avoid division by zero
        chronic = np.maximum(chronic, 0.01)

        return acute / chronic

    def classify_risk_zone(self, acwr: float) -> str:
        """Classify ACWR into risk zones."""
        if acwr < 0.8:
            return "low"  # Undertraining
        elif acwr < 1.3:
            return "optimal"  # Sweet spot
        elif acwr < 1.5:
            return "moderate"
        else:
            return "high"  # Spike

    def compute_result(
        self,
        daily_loads: np.ndarray
    ) -> ACWRResult:
        """Compute full ACWR result for most recent day."""
        acwrs = self.compute_acwr(daily_loads)
        current_acwr = acwrs[-1]

        if self.smoothing == "ewma":
            acute = self.calculate_ewma(daily_loads, self.acute_days)[-1]
            chronic = self.calculate_ewma(daily_loads, self.chronic_days)[-1]
        else:
            acute = self.calculate_rolling(daily_loads, self.acute_days)[-1]
            chronic = self.calculate_rolling(daily_loads, self.chronic_days)[-1]

        # Determine trend
        if len(acwrs) >= 7:
            recent_slope = (acwrs[-1] - acwrs[-7]) / 7
            if recent_slope > 0.02:
                trend = "increasing"
            elif recent_slope < -0.02:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return ACWRResult(
            ratio=current_acwr,
            acute_load=acute,
            chronic_load=chronic,
            risk_zone=self.classify_risk_zone(current_acwr),
            trend=trend
        )


class MonotonyStrainCalculator:
    """
    Calculate training monotony and strain.

    Monotony = mean(load) / std(load) over a week
    Strain = weekly_load * monotony
    """

    def __init__(self, window_days: int = 7):
        self.window_days = window_days

    def calculate_monotony(self, daily_loads: np.ndarray) -> float:
        """Calculate training monotony for recent window."""
        if len(daily_loads) < self.window_days:
            return 0.0

        recent = daily_loads[-self.window_days:]
        mean_load = np.mean(recent)
        std_load = np.std(recent)

        if std_load < 0.01:
            return float('inf')  # All same load = high monotony

        return mean_load / std_load

    def calculate_strain(self, daily_loads: np.ndarray) -> float:
        """Calculate training strain."""
        if len(daily_loads) < self.window_days:
            return 0.0

        recent = daily_loads[-self.window_days:]
        total_load = np.sum(recent)
        monotony = self.calculate_monotony(daily_loads)

        return total_load * monotony

    def assess_monotony_risk(
        self,
        monotony: float
    ) -> Tuple[str, str]:
        """Assess risk level from monotony."""
        if monotony < 1.5:
            return "low", "Good training variation"
        elif monotony < 2.0:
            return "moderate", "Consider adding variety"
        else:
            return "high", "Training too monotonous - increase variation"


class PositionSpecificModel:
    """
    Position-specific injury prediction model.

    Each position has different risk factors and thresholds.
    """

    def __init__(self, position: HockeyPosition):
        self.position = position
        self.risk_factors = self._get_position_risk_factors()
        self.thresholds = self._get_position_thresholds()

    def _get_position_risk_factors(self) -> Dict[str, float]:
        """Get position-specific risk factor weights."""
        base_factors = {
            'acwr_7_28': 1.0,
            'monotony': 0.5,
            'high_speed_distance': 0.3,
            'contact_load': 0.4,
            'previous_injury': 0.6,
        }

        if self.position in [HockeyPosition.CENTER, HockeyPosition.LEFT_WING, HockeyPosition.RIGHT_WING]:
            # Forwards: more groin strain risk from skating
            base_factors['high_speed_distance'] = 0.5
            base_factors['accelerations'] = 0.4
            base_factors['faceoffs'] = 0.2 if self.position == HockeyPosition.CENTER else 0.0

        elif self.position in [HockeyPosition.LEFT_DEFENSE, HockeyPosition.RIGHT_DEFENSE]:
            # Defensemen: more contact-related injuries
            base_factors['contact_load'] = 0.6
            base_factors['blocked_shots'] = 0.4
            base_factors['defensive_zone_time'] = 0.3

        elif self.position == HockeyPosition.GOALIE:
            # Goalies: unique injury profile
            base_factors = {
                'saves': 0.3,
                'high_danger_saves': 0.5,
                'lateral_movement': 0.6,
                'butterfly_stress': 0.7,
                'acwr_7_28': 0.8,
                'previous_injury': 0.7,
            }

        return base_factors

    def _get_position_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get position-specific risk thresholds."""
        if self.position == HockeyPosition.GOALIE:
            return {
                'saves': {'low': 20, 'moderate': 35, 'high': 50},
                'high_danger_saves': {'low': 5, 'moderate': 10, 'high': 15},
                'acwr': {'low': 0.8, 'optimal_low': 0.85, 'optimal_high': 1.2, 'high': 1.4}
            }
        else:
            return {
                'ice_time': {'low': 12, 'moderate': 18, 'high': 22},
                'high_speed_distance': {'low': 300, 'moderate': 500, 'high': 700},
                'contact_load': {'low': 3, 'moderate': 8, 'high': 15},
                'acwr': {'low': 0.8, 'optimal_low': 0.85, 'optimal_high': 1.3, 'high': 1.5}
            }

    def compute_risk_score(
        self,
        metrics: DailyMetrics,
        acwr: ACWRResult,
        monotony: float,
        injury_history: List[Tuple[InjuryType, datetime]]
    ) -> float:
        """Compute position-specific risk score."""
        score = 0.0

        # ACWR contribution
        if acwr.ratio < 0.8 or acwr.ratio > 1.5:
            score += self.risk_factors.get('acwr_7_28', 1.0) * 0.3
        elif acwr.ratio > 1.3:
            score += self.risk_factors.get('acwr_7_28', 1.0) * 0.15

        # Monotony contribution
        if monotony > 2.0:
            score += self.risk_factors.get('monotony', 0.5) * 0.2
        elif monotony > 1.5:
            score += self.risk_factors.get('monotony', 0.5) * 0.1

        # Position-specific metrics
        if self.position == HockeyPosition.GOALIE:
            if metrics.saves > self.thresholds.get('saves', {}).get('high', 50):
                score += self.risk_factors.get('saves', 0.3) * 0.2
            if metrics.high_danger_saves > self.thresholds.get('high_danger_saves', {}).get('high', 15):
                score += self.risk_factors.get('high_danger_saves', 0.5) * 0.25
        else:
            thresholds = self.thresholds
            if metrics.ice_time > thresholds.get('ice_time', {}).get('high', 22):
                score += 0.15
            if metrics.high_speed_distance > thresholds.get('high_speed_distance', {}).get('high', 700):
                score += self.risk_factors.get('high_speed_distance', 0.3) * 0.2
            if metrics.contact_load > thresholds.get('contact_load', {}).get('high', 15):
                score += self.risk_factors.get('contact_load', 0.4) * 0.25

        # Previous injury history
        recent_injuries = [
            inj for inj in injury_history
            if (datetime.now() - inj[1]).days < 365
        ]
        if recent_injuries:
            score += self.risk_factors.get('previous_injury', 0.6) * 0.2 * len(recent_injuries)

        return min(score, 1.0)  # Cap at 1.0


class MultiACWRModel:
    """
    Model using multiple ACWR timeframes.

    Different ratios capture different injury mechanisms:
    - 3:14 - Short-term spikes
    - 7:21 - Medium-term overload
    - 7:28 - Standard ACWR
    - 3:21 - Acute spike in chronic context
    """

    def __init__(self):
        self.calculators = {
            '3:14': ACWRCalculator(3, 14),
            '7:21': ACWRCalculator(7, 21),
            '7:28': ACWRCalculator(7, 28),
            '3:21': ACWRCalculator(3, 21),
        }

        self.injury_type_weights = {
            InjuryType.GROIN: {'3:14': 0.4, '7:21': 0.3, '7:28': 0.2, '3:21': 0.1},
            InjuryType.LOWER_BODY: {'3:14': 0.2, '7:21': 0.3, '7:28': 0.3, '3:21': 0.2},
            InjuryType.UPPER_BODY: {'3:14': 0.3, '7:21': 0.25, '7:28': 0.25, '3:21': 0.2},
            InjuryType.CONCUSSION: {'3:14': 0.1, '7:21': 0.2, '7:28': 0.3, '3:21': 0.4},
        }

    def compute_all_acwrs(
        self,
        daily_loads: np.ndarray
    ) -> Dict[str, ACWRResult]:
        """Compute all ACWR variants."""
        return {
            name: calc.compute_result(daily_loads)
            for name, calc in self.calculators.items()
        }

    def predict_injury_type_risk(
        self,
        daily_loads: np.ndarray,
        injury_type: InjuryType
    ) -> float:
        """Predict risk for specific injury type."""
        acwrs = self.compute_all_acwrs(daily_loads)
        weights = self.injury_type_weights.get(
            injury_type,
            {'3:14': 0.25, '7:21': 0.25, '7:28': 0.25, '3:21': 0.25}
        )

        weighted_risk = 0.0
        for name, weight in weights.items():
            acwr = acwrs[name]
            # Convert ACWR to risk (U-shaped: low and high are risky)
            if acwr.ratio < 0.8:
                risk = (0.8 - acwr.ratio) / 0.8
            elif acwr.ratio > 1.3:
                risk = (acwr.ratio - 1.3) / 0.7
            else:
                risk = 0.0

            weighted_risk += weight * min(risk, 1.0)

        return weighted_risk


class InjuryPredictionSystem:
    """
    Complete injury prediction system.

    Integrates all components for position-specific predictions.
    """

    def __init__(self):
        self.position_models: Dict[HockeyPosition, PositionSpecificModel] = {
            pos: PositionSpecificModel(pos)
            for pos in HockeyPosition
        }
        self.multi_acwr = MultiACWRModel()
        self.monotony_calc = MonotonyStrainCalculator()

    def extract_loads(
        self,
        history: List[DailyMetrics],
        metric: str = "player_load"
    ) -> np.ndarray:
        """Extract load time series from history."""
        return np.array([getattr(m, metric, 0) for m in history])

    def predict_risk(
        self,
        player_id: str,
        position: HockeyPosition,
        history: List[DailyMetrics],
        injury_history: List[Tuple[InjuryType, datetime]]
    ) -> InjuryRisk:
        """Generate comprehensive injury risk prediction."""
        if len(history) < 28:
            return InjuryRisk(
                player_id=player_id,
                date=datetime.now(),
                overall_risk=0.0,
                risk_by_type={},
                risk_factors=["Insufficient data"],
                recommendations=["Continue monitoring"],
                confidence=0.0
            )

        # Extract loads
        loads = self.extract_loads(history, "player_load")
        contact_loads = np.array([m.contact_load for m in history])

        # Compute ACWRs
        acwr_7_28 = ACWRCalculator(7, 28).compute_result(loads)
        all_acwrs = self.multi_acwr.compute_all_acwrs(loads)

        # Monotony and strain
        monotony = self.monotony_calc.calculate_monotony(loads)
        strain = self.monotony_calc.calculate_strain(loads)

        # Position-specific risk
        model = self.position_models[position]
        latest_metrics = history[-1]
        position_risk = model.compute_risk_score(
            latest_metrics, acwr_7_28, monotony, injury_history
        )

        # Risk by injury type
        risk_by_type = {}
        for injury_type in [InjuryType.LOWER_BODY, InjuryType.UPPER_BODY,
                           InjuryType.GROIN, InjuryType.CONCUSSION]:
            risk_by_type[injury_type] = self.multi_acwr.predict_injury_type_risk(
                loads, injury_type
            )

        # Identify risk factors
        risk_factors = []
        if acwr_7_28.ratio > 1.5:
            risk_factors.append(f"High ACWR: {acwr_7_28.ratio:.2f}")
        if acwr_7_28.ratio < 0.8:
            risk_factors.append(f"Low ACWR (undertraining): {acwr_7_28.ratio:.2f}")
        if monotony > 2.0:
            risk_factors.append(f"High monotony: {monotony:.2f}")
        if strain > 5000:
            risk_factors.append(f"High strain: {strain:.0f}")
        if any(m.contact_load > 15 for m in history[-7:]):
            risk_factors.append("High contact load in past week")

        # Generate recommendations
        recommendations = self._generate_recommendations(
            acwr_7_28, monotony, position, risk_factors
        )

        # Overall risk
        overall_risk = min(
            position_risk * 0.4 +
            max(risk_by_type.values()) * 0.3 +
            (0.3 if len(risk_factors) > 2 else len(risk_factors) * 0.1),
            1.0
        )

        return InjuryRisk(
            player_id=player_id,
            date=datetime.now(),
            overall_risk=overall_risk,
            risk_by_type=risk_by_type,
            risk_factors=risk_factors,
            recommendations=recommendations,
            confidence=0.7 if len(history) >= 56 else 0.5
        )

    def _generate_recommendations(
        self,
        acwr: ACWRResult,
        monotony: float,
        position: HockeyPosition,
        risk_factors: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        if acwr.ratio > 1.5:
            recs.append("Reduce training load - consider day off")
        elif acwr.ratio > 1.3:
            recs.append("Monitor closely - maintain current load")
        elif acwr.ratio < 0.8:
            recs.append("Gradually increase training load")

        if monotony > 2.0:
            recs.append("Increase training variation")

        if position == HockeyPosition.GOALIE and any("contact" in rf.lower() for rf in risk_factors):
            recs.append("Limit shootaround volume")

        if not recs:
            recs.append("Continue current program - risk is acceptable")

        return recs

    def generate_load_recommendation(
        self,
        player_id: str,
        position: HockeyPosition,
        history: List[DailyMetrics],
        target_acwr: float = 1.0
    ) -> TrainingRecommendation:
        """Generate specific load recommendations."""
        if len(history) < 28:
            return TrainingRecommendation(
                player_id=player_id,
                date=datetime.now(),
                recommended_ice_time=15.0,
                max_high_intensity=400.0,
                avoid_contact=False,
                recovery_focus=[],
                rationale="Insufficient data - using defaults"
            )

        loads = self.extract_loads(history, "player_load")

        # Calculate what tomorrow's load should be for target ACWR
        chronic = ACWRCalculator(7, 28).calculate_ewma(loads, 28)[-1]
        target_acute = target_acwr * chronic

        # Current acute
        current_acute = ACWRCalculator(7, 28).calculate_ewma(loads, 7)[-1]

        # Recommended load
        # new_acute = 0.25 * new_load + 0.75 * current_acute (EWMA with alpha=0.25)
        target_load = (target_acute - 0.75 * current_acute) / 0.25

        # Convert to ice time (rough approximation)
        avg_load_per_minute = np.mean(loads) / np.mean([m.ice_time for m in history if m.ice_time > 0])
        recommended_time = max(8, min(22, target_load / max(avg_load_per_minute, 1)))

        # High intensity
        recommended_high_intensity = recommended_time * 30  # meters per minute

        # Contact avoidance
        contact_loads = np.array([m.contact_load for m in history[-7:]])
        avoid_contact = np.mean(contact_loads) > 10

        return TrainingRecommendation(
            player_id=player_id,
            date=datetime.now(),
            recommended_ice_time=recommended_time,
            max_high_intensity=recommended_high_intensity,
            avoid_contact=avoid_contact,
            recovery_focus=["Sleep", "Nutrition"] if target_load < loads[-1] else [],
            rationale=f"Targeting ACWR of {target_acwr:.2f} from current {current_acute/chronic:.2f}"
        )
