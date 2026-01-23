"""
Comprehensive Injury/Illness Monitoring System

Implements multi-source monitoring approach translated from
youth soccer research.

Key Paper:
- "Predicting Injury and Illness with Machine Learning in Elite Youth
  Soccer: A Comprehensive Monitoring Approach over 3 Months." PMC (2023).

Key Variables (65 total):
- Training load (EWMA of last 2 sessions)
- ACWR ratios
- Sleep quality and duration
- Jump height (CMJ)
- Blood markers (cfDNA, CRP, ferritin)

Key Findings:
- Sleep quality was most important predictor for injury
- ACWR of tempo runs ranked #2
- Blood variables important for illness prediction

Hockey Translation:
- Integrate multiple data sources (tracking, biometric, subjective)
- Monitor sleep and recovery alongside workload
- Use blood markers for return-to-play decisions
- Position-specific monitoring profiles
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta
import json


class DataSource(Enum):
    """Sources of monitoring data."""
    TRACKING = "tracking"            # GPS/tracking system
    BIOMETRIC = "biometric"          # Heart rate, HRV
    SUBJECTIVE = "subjective"        # Wellness questionnaires
    BLOOD = "blood"                  # Blood markers
    SLEEP = "sleep"                  # Sleep tracking
    PERFORMANCE = "performance"      # Jump tests, etc.
    MEDICAL = "medical"              # Injury history


class InjuryRiskLevel(Enum):
    """Risk levels for injury."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class IllnessRiskLevel(Enum):
    """Risk levels for illness."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Position(Enum):
    """Player positions for position-specific models."""
    CENTER = "C"
    WINGER = "W"
    DEFENSEMAN = "D"
    GOALIE = "G"


@dataclass
class TrackingData:
    """Daily tracking/workload data."""
    date: str
    player_id: str
    total_distance: float           # feet
    high_speed_distance: float      # feet (>18 mph)
    sprint_count: int
    acceleration_count: int         # >3 m/s^2
    deceleration_count: int
    time_on_ice: float              # minutes
    shift_count: int
    avg_shift_length: float         # seconds
    hits_given: int
    hits_taken: int
    blocks: int


@dataclass
class BiometricData:
    """Daily biometric measurements."""
    date: str
    player_id: str
    resting_hr: float               # bpm
    hrv_rmssd: float                # ms
    hrv_sdnn: float                 # ms
    max_hr_session: float           # bpm
    avg_hr_session: float           # bpm
    time_in_red_zone: float         # minutes (>90% max HR)


@dataclass
class SubjectiveData:
    """Daily subjective wellness data."""
    date: str
    player_id: str
    sleep_quality: int              # 1-10
    sleep_hours: float
    fatigue: int                    # 1-10 (10 = very fatigued)
    muscle_soreness: int            # 1-10
    stress: int                     # 1-10
    mood: int                       # 1-10 (10 = excellent)
    motivation: int                 # 1-10
    overall_wellness: int           # 1-10
    notes: str = ""


@dataclass
class BloodMarkers:
    """Blood test results."""
    date: str
    player_id: str
    crp: float                      # mg/L - inflammation
    ferritin: float                 # ng/mL - iron stores
    hemoglobin: float               # g/dL
    testosterone: float             # ng/dL
    cortisol: float                 # μg/dL
    creatine_kinase: float          # U/L - muscle damage
    vitamin_d: float                # ng/mL


@dataclass
class PerformanceTest:
    """Performance test results."""
    date: str
    player_id: str
    cmj_height: float               # cm - countermovement jump
    cmj_power: float                # watts
    reactive_strength: float        # RSI
    sprint_5m: float                # seconds
    sprint_10m: float               # seconds
    agility_time: float             # seconds


@dataclass
class InjuryHistory:
    """Player injury history."""
    player_id: str
    injuries: List[Dict[str, Any]]  # date, type, severity, days_missed
    total_days_missed: int
    injury_prone_areas: List[str]


@dataclass
class MonitoringProfile:
    """Complete monitoring profile for a player."""
    player_id: str
    position: Position
    tracking_history: List[TrackingData]
    biometric_history: List[BiometricData]
    subjective_history: List[SubjectiveData]
    blood_history: List[BloodMarkers]
    performance_history: List[PerformanceTest]
    injury_history: InjuryHistory


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    player_id: str
    date: str
    injury_risk: InjuryRiskLevel
    injury_probability: float
    illness_risk: IllnessRiskLevel
    illness_probability: float
    key_factors: List[Tuple[str, float]]  # (factor, importance)
    recommendations: List[str]
    data_quality: float             # 0-1, completeness of data


class ACWRCalculator:
    """
    Calculates Acute:Chronic Workload Ratios.

    Multiple timeframes as per GPS/rugby research.
    """

    def __init__(self):
        """Initialize calculator."""
        # ACWR timeframes (acute:chronic days)
        self.timeframes = [
            (3, 14),
            (7, 21),
            (7, 28),
        ]

    def calculate_ewma(
        self,
        values: List[float],
        window: int,
    ) -> float:
        """
        Calculate exponentially weighted moving average.

        Uses recommended decay factor for sports science.
        """
        if not values:
            return 0.0

        # Use last 'window' values
        recent = values[-window:] if len(values) >= window else values

        # EWMA decay factor
        decay = 2 / (window + 1)

        ewma = recent[0]
        for val in recent[1:]:
            ewma = decay * val + (1 - decay) * ewma

        return ewma

    def calculate_acwr(
        self,
        workloads: List[float],
        acute_days: int,
        chronic_days: int,
    ) -> float:
        """Calculate ACWR for given timeframe."""
        if len(workloads) < chronic_days:
            return 1.0  # Default to optimal if insufficient data

        acute = self.calculate_ewma(workloads[-acute_days:], acute_days)
        chronic = self.calculate_ewma(workloads[-chronic_days:], chronic_days)

        if chronic < 0.01:
            return 1.0

        return acute / chronic

    def calculate_all_acwr(
        self,
        workloads: List[float],
    ) -> Dict[str, float]:
        """Calculate ACWR for all timeframes."""
        results = {}
        for acute, chronic in self.timeframes:
            key = f"acwr_{acute}_{chronic}"
            results[key] = self.calculate_acwr(workloads, acute, chronic)
        return results

    def get_risk_zone(self, acwr: float) -> str:
        """
        Determine risk zone from ACWR.

        Based on sports science research:
        - <0.8: Under-training (moderate risk)
        - 0.8-1.3: Sweet spot (low risk)
        - 1.3-1.5: Danger zone (high risk)
        - >1.5: Very high risk
        """
        if acwr < 0.8:
            return "under_training"
        elif acwr <= 1.3:
            return "optimal"
        elif acwr <= 1.5:
            return "danger_zone"
        else:
            return "very_high_risk"


class SleepQualityAnalyzer:
    """
    Analyzes sleep quality and its impact on injury risk.

    Sleep quality was #1 predictor in youth soccer study.
    """

    def __init__(self):
        """Initialize analyzer."""
        # Thresholds
        self.min_sleep_hours = 7.0
        self.optimal_sleep_hours = 8.5
        self.min_quality_score = 6

    def analyze_sleep_pattern(
        self,
        sleep_data: List[SubjectiveData],
        window_days: int = 7,
    ) -> Dict[str, Any]:
        """Analyze recent sleep patterns."""
        if not sleep_data:
            return {'status': 'no_data'}

        recent = sleep_data[-window_days:] if len(sleep_data) >= window_days else sleep_data

        avg_hours = np.mean([d.sleep_hours for d in recent])
        avg_quality = np.mean([d.sleep_quality for d in recent])
        variability = np.std([d.sleep_hours for d in recent])

        # Sleep debt
        optimal_total = self.optimal_sleep_hours * len(recent)
        actual_total = sum(d.sleep_hours for d in recent)
        sleep_debt = optimal_total - actual_total

        # Risk factors
        risk_factors = []
        if avg_hours < self.min_sleep_hours:
            risk_factors.append("insufficient_duration")
        if avg_quality < self.min_quality_score:
            risk_factors.append("poor_quality")
        if variability > 1.5:
            risk_factors.append("irregular_schedule")
        if sleep_debt > 7:
            risk_factors.append("significant_debt")

        return {
            'avg_hours': avg_hours,
            'avg_quality': avg_quality,
            'variability': variability,
            'sleep_debt': sleep_debt,
            'risk_factors': risk_factors,
            'risk_score': len(risk_factors) / 4,  # Normalized
        }


class WellnessScoreCalculator:
    """
    Calculates overall wellness score from subjective data.
    """

    def __init__(self):
        """Initialize calculator."""
        # Weights for wellness components
        self.weights = {
            'sleep_quality': 0.20,
            'fatigue': 0.20,
            'muscle_soreness': 0.15,
            'stress': 0.15,
            'mood': 0.15,
            'motivation': 0.15,
        }

    def calculate_score(self, data: SubjectiveData) -> float:
        """
        Calculate wellness score (0-100).

        Higher = better wellness.
        """
        # Normalize each component (some are inverse - higher is worse)
        components = {
            'sleep_quality': data.sleep_quality / 10,
            'fatigue': (10 - data.fatigue) / 10,  # Invert
            'muscle_soreness': (10 - data.muscle_soreness) / 10,  # Invert
            'stress': (10 - data.stress) / 10,  # Invert
            'mood': data.mood / 10,
            'motivation': data.motivation / 10,
        }

        score = sum(
            components[k] * self.weights[k]
            for k in self.weights
        ) * 100

        return score

    def calculate_trend(
        self,
        data_history: List[SubjectiveData],
        window: int = 7,
    ) -> Dict[str, Any]:
        """Calculate wellness trend."""
        if len(data_history) < 2:
            return {'trend': 'stable', 'change': 0.0}

        recent = data_history[-window:] if len(data_history) >= window else data_history
        scores = [self.calculate_score(d) for d in recent]

        # Linear regression for trend
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]

        if slope > 2:
            trend = 'improving'
        elif slope < -2:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'change': float(slope),
            'current_score': scores[-1],
            'avg_score': np.mean(scores),
        }


class BloodMarkerAnalyzer:
    """
    Analyzes blood markers for injury/illness risk.

    Important for illness prediction per the research.
    """

    def __init__(self):
        """Initialize analyzer with reference ranges."""
        # Normal ranges
        self.normal_ranges = {
            'crp': (0, 3.0),           # mg/L
            'ferritin': (30, 300),     # ng/mL (male)
            'hemoglobin': (13.5, 17.5),  # g/dL (male)
            'testosterone': (300, 1000),  # ng/dL
            'cortisol': (5, 25),       # μg/dL
            'creatine_kinase': (30, 200),  # U/L
            'vitamin_d': (30, 100),    # ng/mL
        }

        # Risk thresholds
        self.risk_thresholds = {
            'crp': 5.0,                # Elevated inflammation
            'ferritin': 20,            # Low iron
            'creatine_kinase': 400,    # Muscle damage
            'cortisol_ratio': 0.3,     # T:C ratio
        }

    def analyze_markers(self, markers: BloodMarkers) -> Dict[str, Any]:
        """Analyze blood markers for risk factors."""
        risk_factors = []
        abnormal_values = []

        # Check each marker
        for marker, (low, high) in self.normal_ranges.items():
            value = getattr(markers, marker, None)
            if value is None:
                continue

            if value < low:
                abnormal_values.append((marker, 'low', value))
            elif value > high:
                abnormal_values.append((marker, 'high', value))

        # Specific risk checks
        if markers.crp > self.risk_thresholds['crp']:
            risk_factors.append('elevated_inflammation')

        if markers.ferritin < self.risk_thresholds['ferritin']:
            risk_factors.append('low_iron_stores')

        if markers.creatine_kinase > self.risk_thresholds['creatine_kinase']:
            risk_factors.append('muscle_damage')

        # T:C ratio for overtraining
        if markers.testosterone > 0 and markers.cortisol > 0:
            tc_ratio = markers.testosterone / (markers.cortisol * 10)
            if tc_ratio < self.risk_thresholds['cortisol_ratio']:
                risk_factors.append('overtraining_signal')

        # Illness risk from inflammation + low vitamin D
        illness_risk = 0.0
        if markers.crp > 3.0:
            illness_risk += 0.3
        if markers.vitamin_d < 30:
            illness_risk += 0.2

        return {
            'abnormal_values': abnormal_values,
            'risk_factors': risk_factors,
            'illness_risk_contribution': illness_risk,
            'markers_checked': len(self.normal_ranges),
        }


class ComprehensiveRiskModel:
    """
    Main risk prediction model integrating all data sources.
    """

    def __init__(self):
        """Initialize model components."""
        self.acwr_calc = ACWRCalculator()
        self.sleep_analyzer = SleepQualityAnalyzer()
        self.wellness_calc = WellnessScoreCalculator()
        self.blood_analyzer = BloodMarkerAnalyzer()

        # Feature importance (from research)
        self.feature_importance = {
            'sleep_quality': 0.18,
            'acwr_7_21': 0.15,
            'wellness_trend': 0.12,
            'previous_injury': 0.11,
            'crp': 0.10,
            'muscle_soreness': 0.09,
            'fatigue': 0.08,
            'hrv_change': 0.07,
            'training_monotony': 0.05,
            'other': 0.05,
        }

    def assess_risk(
        self,
        profile: MonitoringProfile,
    ) -> RiskAssessment:
        """
        Assess injury and illness risk.

        Integrates all available data sources.
        """
        risk_contributions = []
        data_available = 0
        data_total = 5  # Number of data sources

        # 1. Workload/ACWR analysis
        if profile.tracking_history:
            data_available += 1
            workloads = [t.total_distance for t in profile.tracking_history]
            acwr_results = self.acwr_calc.calculate_all_acwr(workloads)

            acwr_7_21 = acwr_results.get('acwr_7_21', 1.0)
            acwr_zone = self.acwr_calc.get_risk_zone(acwr_7_21)

            acwr_risk = 0.0
            if acwr_zone == 'danger_zone':
                acwr_risk = 0.3
            elif acwr_zone == 'very_high_risk':
                acwr_risk = 0.5
            elif acwr_zone == 'under_training':
                acwr_risk = 0.15

            risk_contributions.append(('acwr', acwr_risk, 0.15))

        # 2. Sleep analysis
        if profile.subjective_history:
            data_available += 1
            sleep_analysis = self.sleep_analyzer.analyze_sleep_pattern(
                profile.subjective_history
            )
            sleep_risk = sleep_analysis.get('risk_score', 0)
            risk_contributions.append(('sleep', sleep_risk, 0.18))

        # 3. Wellness trend
        if profile.subjective_history:
            wellness_trend = self.wellness_calc.calculate_trend(
                profile.subjective_history
            )
            if wellness_trend['trend'] == 'declining':
                wellness_risk = 0.3
            elif wellness_trend['current_score'] < 60:
                wellness_risk = 0.2
            else:
                wellness_risk = 0.0
            risk_contributions.append(('wellness', wellness_risk, 0.12))

        # 4. Blood markers (illness risk)
        illness_risk_component = 0.0
        if profile.blood_history:
            data_available += 1
            latest_blood = profile.blood_history[-1]
            blood_analysis = self.blood_analyzer.analyze_markers(latest_blood)
            illness_risk_component = blood_analysis['illness_risk_contribution']
            risk_contributions.append(('blood', illness_risk_component, 0.10))

        # 5. Injury history
        if profile.injury_history.injuries:
            data_available += 1
            recent_injuries = sum(
                1 for inj in profile.injury_history.injuries
                if self._is_recent(inj.get('date', ''), 365)
            )
            history_risk = min(recent_injuries * 0.1, 0.3)
            risk_contributions.append(('history', history_risk, 0.11))

        # 6. Biometric analysis (HRV)
        if profile.biometric_history and len(profile.biometric_history) >= 7:
            hrv_values = [b.hrv_rmssd for b in profile.biometric_history[-7:]]
            hrv_baseline = np.mean(hrv_values[:-1]) if len(hrv_values) > 1 else hrv_values[0]
            hrv_current = hrv_values[-1]

            if hrv_baseline > 0:
                hrv_change = (hrv_current - hrv_baseline) / hrv_baseline
                if hrv_change < -0.15:  # 15% decrease
                    hrv_risk = 0.2
                else:
                    hrv_risk = 0.0
                risk_contributions.append(('hrv', hrv_risk, 0.07))

        # Calculate overall injury risk
        total_injury_risk = sum(
            risk * weight for factor, risk, weight in risk_contributions
            if factor != 'blood'  # Blood is mainly for illness
        )
        total_injury_risk = min(total_injury_risk, 1.0)

        # Calculate illness risk
        total_illness_risk = illness_risk_component
        if profile.subjective_history:
            latest_wellness = profile.subjective_history[-1]
            if latest_wellness.stress > 7:
                total_illness_risk += 0.1
        total_illness_risk = min(total_illness_risk, 1.0)

        # Determine risk levels
        injury_level = self._prob_to_level(total_injury_risk, 'injury')
        illness_level = self._prob_to_level(total_illness_risk, 'illness')

        # Key factors
        key_factors = sorted(
            [(f, r) for f, r, _ in risk_contributions],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Recommendations
        recommendations = self._generate_recommendations(
            risk_contributions, injury_level, illness_level
        )

        # Data quality
        data_quality = data_available / data_total

        return RiskAssessment(
            player_id=profile.player_id,
            date=datetime.now().strftime('%Y-%m-%d'),
            injury_risk=injury_level,
            injury_probability=total_injury_risk,
            illness_risk=illness_level,
            illness_probability=total_illness_risk,
            key_factors=key_factors,
            recommendations=recommendations,
            data_quality=data_quality,
        )

    def _is_recent(self, date_str: str, days: int) -> bool:
        """Check if date is within N days."""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            return (datetime.now() - date).days <= days
        except:
            return False

    def _prob_to_level(
        self,
        prob: float,
        risk_type: str,
    ) -> Any:
        """Convert probability to risk level."""
        if risk_type == 'injury':
            if prob < 0.2:
                return InjuryRiskLevel.LOW
            elif prob < 0.4:
                return InjuryRiskLevel.MODERATE
            elif prob < 0.6:
                return InjuryRiskLevel.HIGH
            else:
                return InjuryRiskLevel.CRITICAL
        else:
            if prob < 0.2:
                return IllnessRiskLevel.LOW
            elif prob < 0.4:
                return IllnessRiskLevel.MODERATE
            else:
                return IllnessRiskLevel.HIGH

    def _generate_recommendations(
        self,
        risk_contributions: List[Tuple[str, float, float]],
        injury_level: InjuryRiskLevel,
        illness_level: IllnessRiskLevel,
    ) -> List[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []

        for factor, risk, _ in risk_contributions:
            if risk < 0.1:
                continue

            if factor == 'acwr':
                recommendations.append("Reduce training load intensity")
            elif factor == 'sleep':
                recommendations.append("Prioritize sleep quality and duration")
            elif factor == 'wellness':
                recommendations.append("Monitor wellness closely, consider recovery day")
            elif factor == 'blood':
                recommendations.append("Follow up with team physician")
            elif factor == 'history':
                recommendations.append("Extra warm-up and prehab for injury-prone areas")
            elif factor == 'hrv':
                recommendations.append("Signs of accumulated fatigue - consider rest")

        if injury_level in [InjuryRiskLevel.HIGH, InjuryRiskLevel.CRITICAL]:
            recommendations.insert(0, "⚠️ HIGH INJURY RISK - Modify training")

        if illness_level == IllnessRiskLevel.HIGH:
            recommendations.insert(0, "⚠️ ILLNESS RISK - Monitor symptoms")

        return recommendations[:5]  # Top 5 recommendations
