"""
Talent/Draft Prediction Model

Implements talent prediction from tracking data, translated from
Stats Perform's AutoStats basketball research.

Key Paper:
- Stats Perform. (2021). "Predicting NBA Talent from Enormous Amounts
  of College Basketball Tracking Data." MIT Sloan Sports Analytics
  Conference.

Key Concepts:
- Computer vision generates tracking data from video
- Predicts NHL-level skills from pre-draft data
- Identifies traits not measurable before tracking
- Evaluate prospects without live tracking systems

Hockey Translation:
- Generate tracking data from junior/college hockey video
- Predict NHL success from CHL, NCAA, European league data
- Evaluate international prospects
- Draft analysis and projection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class ProspectLeague(Enum):
    """Leagues for prospect evaluation."""
    CHL_OHL = "ohl"
    CHL_WHL = "whl"
    CHL_QMJHL = "qmjhl"
    USHL = "ushl"
    NCAA = "ncaa"
    SHL = "shl"          # Swedish Hockey League
    LIIGA = "liiga"      # Finnish League
    KHL = "khl"
    MHL = "mhl"          # Russian junior
    CZECH = "czech"
    DEL = "del"          # German League
    NLA = "nla"          # Swiss League
    USDP = "usdp"        # USA Development Program


class Position(Enum):
    """Player positions."""
    CENTER = "C"
    LEFT_WING = "LW"
    RIGHT_WING = "RW"
    LEFT_DEFENSE = "LD"
    RIGHT_DEFENSE = "RD"
    GOALIE = "G"


class SkillCategory(Enum):
    """Categories of skills to evaluate."""
    SKATING = "skating"
    SHOOTING = "shooting"
    PASSING = "passing"
    PUCK_SKILLS = "puck_skills"
    HOCKEY_IQ = "hockey_iq"
    PHYSICAL = "physical"
    DEFENSIVE = "defensive"
    COMPETE = "compete"


@dataclass
class TrackingMetrics:
    """Metrics derived from tracking data."""
    # Skating
    top_speed: float                # mph
    acceleration: float             # ft/s^2
    agility_score: float            # 0-100
    skating_efficiency: float       # Distance covered per minute of TOI

    # With puck
    puck_protection: float          # Seconds before losing puck under pressure
    deception_score: float          # Body fake effectiveness
    zone_entry_success: float       # % successful controlled entries

    # Shooting
    shot_velocity: float            # mph
    release_time: float             # seconds
    shooting_accuracy: float        # % on target

    # Passing
    pass_completion: float          # %
    pass_difficulty: float          # Average pass difficulty attempted
    sauce_accuracy: float           # Saucer pass accuracy

    # Defensive
    gap_control: float              # Average gap distance
    stick_positioning: float        # % time stick in lane
    recovery_speed: float           # Speed returning to position


@dataclass
class ProspectProfile:
    """Complete prospect profile."""
    player_id: str
    name: str
    birth_date: str
    position: Position
    height: float                   # inches
    weight: float                   # lbs
    shoots: str                     # L/R
    league: ProspectLeague
    team: str
    draft_eligible_year: int


@dataclass
class ProspectEvaluation:
    """Evaluation result for a prospect."""
    player_id: str
    overall_score: float            # 0-100
    nhl_probability: float          # Probability of playing NHL games
    top_six_probability: float      # Probability of being top-6 F / top-4 D
    star_probability: float         # Probability of being elite
    projected_role: str             # Projected NHL role
    skill_scores: Dict[SkillCategory, float]
    comparable_players: List[str]   # Similar historical prospects
    strengths: List[str]
    weaknesses: List[str]
    projection_confidence: float    # 0-1


@dataclass
class HistoricalComparable:
    """Historical player comparable."""
    player_id: str
    name: str
    similarity_score: float
    nhl_outcome: str                # What they became in NHL
    games_played: int
    points: int
    draft_position: int


class LeagueEquivalency:
    """
    Converts stats between leagues using equivalency factors.

    Different leagues have different scoring environments,
    so raw stats need adjustment.
    """

    # NHLe factors (points per point in that league)
    # Higher = harder league (more NHL equivalent)
    EQUIVALENCY_FACTORS = {
        ProspectLeague.CHL_OHL: 0.30,
        ProspectLeague.CHL_WHL: 0.29,
        ProspectLeague.CHL_QMJHL: 0.28,
        ProspectLeague.USHL: 0.27,
        ProspectLeague.NCAA: 0.45,
        ProspectLeague.SHL: 0.55,
        ProspectLeague.LIIGA: 0.50,
        ProspectLeague.KHL: 0.60,
        ProspectLeague.MHL: 0.15,
        ProspectLeague.CZECH: 0.40,
        ProspectLeague.DEL: 0.35,
        ProspectLeague.NLA: 0.35,
        ProspectLeague.USDP: 0.25,
    }

    # Age adjustment (points at 18 worth more than at 20)
    AGE_ADJUSTMENT = {
        16: 1.4,
        17: 1.3,
        18: 1.2,
        19: 1.1,
        20: 1.0,
        21: 0.95,
        22: 0.90,
    }

    def convert_to_nhle(
        self,
        points: float,
        games: int,
        league: ProspectLeague,
        age: int,
    ) -> float:
        """
        Convert points to NHL equivalent.

        Args:
            points: Raw points in league
            games: Games played
            league: League played in
            age: Age during season

        Returns:
            NHL equivalent points per 82 games
        """
        if games == 0:
            return 0.0

        ppg = points / games

        # League adjustment
        league_factor = self.EQUIVALENCY_FACTORS.get(league, 0.30)

        # Age adjustment
        age_factor = self.AGE_ADJUSTMENT.get(age, 1.0)

        # Convert to 82-game pace
        nhle = ppg * league_factor * age_factor * 82

        return nhle


class TrackingSkillExtractor:
    """
    Extracts skill ratings from tracking metrics.

    Translates raw tracking numbers into scouting-style grades.
    """

    def __init__(self):
        """Initialize extractor with league benchmarks."""
        # NHL benchmarks for comparison
        self.nhl_benchmarks = {
            'top_speed': 25.0,       # mph
            'acceleration': 18.0,    # ft/s^2
            'shot_velocity': 95.0,   # mph
            'release_time': 0.25,    # seconds
        }

        # Draft-eligible benchmarks (lower than NHL)
        self.draft_benchmarks = {
            'top_speed': 22.0,
            'acceleration': 15.0,
            'shot_velocity': 80.0,
            'release_time': 0.35,
        }

    def extract_skating_score(
        self,
        metrics: TrackingMetrics,
    ) -> float:
        """
        Calculate skating skill score (0-100).
        """
        # Speed component (40%)
        speed_score = min(100, (metrics.top_speed / self.draft_benchmarks['top_speed']) * 70)

        # Acceleration component (30%)
        accel_score = min(100, (metrics.acceleration / self.draft_benchmarks['acceleration']) * 70)

        # Agility component (30%)
        agility_score = metrics.agility_score

        return 0.4 * speed_score + 0.3 * accel_score + 0.3 * agility_score

    def extract_shooting_score(
        self,
        metrics: TrackingMetrics,
    ) -> float:
        """
        Calculate shooting skill score (0-100).
        """
        # Velocity (40%)
        velocity_score = min(100, (metrics.shot_velocity / self.draft_benchmarks['shot_velocity']) * 70)

        # Release time (30%) - lower is better
        release_score = max(0, 100 - (metrics.release_time - 0.15) * 200)

        # Accuracy (30%)
        accuracy_score = metrics.shooting_accuracy

        return 0.4 * velocity_score + 0.3 * release_score + 0.3 * accuracy_score

    def extract_passing_score(
        self,
        metrics: TrackingMetrics,
    ) -> float:
        """
        Calculate passing skill score (0-100).
        """
        # Completion rate (40%)
        completion_score = metrics.pass_completion

        # Difficulty attempted (40%)
        difficulty_score = metrics.pass_difficulty

        # Sauce accuracy (20%)
        sauce_score = metrics.sauce_accuracy

        return 0.4 * completion_score + 0.4 * difficulty_score + 0.2 * sauce_score

    def extract_puck_skills_score(
        self,
        metrics: TrackingMetrics,
    ) -> float:
        """
        Calculate puck handling skill score (0-100).
        """
        # Puck protection (40%)
        protection_score = min(100, metrics.puck_protection * 30)

        # Deception (30%)
        deception_score = metrics.deception_score

        # Zone entry (30%)
        entry_score = metrics.zone_entry_success

        return 0.4 * protection_score + 0.3 * deception_score + 0.3 * entry_score

    def extract_defensive_score(
        self,
        metrics: TrackingMetrics,
    ) -> float:
        """
        Calculate defensive skill score (0-100).
        """
        # Gap control (40%)
        gap_score = max(0, 100 - abs(metrics.gap_control - 10) * 5)

        # Stick positioning (30%)
        stick_score = metrics.stick_positioning

        # Recovery speed (30%)
        recovery_score = min(100, metrics.recovery_speed * 4)

        return 0.4 * gap_score + 0.3 * stick_score + 0.3 * recovery_score

    def extract_all_skills(
        self,
        metrics: TrackingMetrics,
    ) -> Dict[SkillCategory, float]:
        """
        Extract all skill scores from tracking metrics.
        """
        return {
            SkillCategory.SKATING: self.extract_skating_score(metrics),
            SkillCategory.SHOOTING: self.extract_shooting_score(metrics),
            SkillCategory.PASSING: self.extract_passing_score(metrics),
            SkillCategory.PUCK_SKILLS: self.extract_puck_skills_score(metrics),
            SkillCategory.DEFENSIVE: self.extract_defensive_score(metrics),
            # These would need additional data sources
            SkillCategory.HOCKEY_IQ: 50.0,  # Placeholder
            SkillCategory.PHYSICAL: 50.0,   # Placeholder
            SkillCategory.COMPETE: 50.0,    # Placeholder
        }


class TalentPredictor:
    """
    Predicts NHL success from prospect data.

    Combines tracking metrics, production, and comparables
    to project future NHL performance.
    """

    def __init__(self):
        """Initialize predictor."""
        self.skill_extractor = TrackingSkillExtractor()
        self.league_equivalency = LeagueEquivalency()

        # Position-specific skill weights for overall score
        self.position_weights = {
            Position.CENTER: {
                SkillCategory.SKATING: 0.15,
                SkillCategory.SHOOTING: 0.12,
                SkillCategory.PASSING: 0.15,
                SkillCategory.PUCK_SKILLS: 0.12,
                SkillCategory.HOCKEY_IQ: 0.18,
                SkillCategory.DEFENSIVE: 0.12,
                SkillCategory.COMPETE: 0.16,
            },
            Position.LEFT_WING: {
                SkillCategory.SKATING: 0.18,
                SkillCategory.SHOOTING: 0.18,
                SkillCategory.PASSING: 0.10,
                SkillCategory.PUCK_SKILLS: 0.15,
                SkillCategory.HOCKEY_IQ: 0.12,
                SkillCategory.DEFENSIVE: 0.10,
                SkillCategory.COMPETE: 0.17,
            },
            Position.RIGHT_WING: {
                SkillCategory.SKATING: 0.18,
                SkillCategory.SHOOTING: 0.18,
                SkillCategory.PASSING: 0.10,
                SkillCategory.PUCK_SKILLS: 0.15,
                SkillCategory.HOCKEY_IQ: 0.12,
                SkillCategory.DEFENSIVE: 0.10,
                SkillCategory.COMPETE: 0.17,
            },
            Position.LEFT_DEFENSE: {
                SkillCategory.SKATING: 0.20,
                SkillCategory.SHOOTING: 0.08,
                SkillCategory.PASSING: 0.15,
                SkillCategory.PUCK_SKILLS: 0.10,
                SkillCategory.HOCKEY_IQ: 0.17,
                SkillCategory.DEFENSIVE: 0.18,
                SkillCategory.PHYSICAL: 0.12,
            },
            Position.RIGHT_DEFENSE: {
                SkillCategory.SKATING: 0.20,
                SkillCategory.SHOOTING: 0.08,
                SkillCategory.PASSING: 0.15,
                SkillCategory.PUCK_SKILLS: 0.10,
                SkillCategory.HOCKEY_IQ: 0.17,
                SkillCategory.DEFENSIVE: 0.18,
                SkillCategory.PHYSICAL: 0.12,
            },
        }

    def evaluate_prospect(
        self,
        profile: ProspectProfile,
        tracking_metrics: TrackingMetrics,
        season_stats: Dict[str, Any],
    ) -> ProspectEvaluation:
        """
        Comprehensive prospect evaluation.

        Args:
            profile: Prospect biographical info
            tracking_metrics: Tracking-derived metrics
            season_stats: Traditional season statistics

        Returns:
            Complete evaluation with projections
        """
        # Extract skill scores
        skill_scores = self.skill_extractor.extract_all_skills(tracking_metrics)

        # Calculate overall score
        weights = self.position_weights.get(
            profile.position,
            self.position_weights[Position.CENTER]
        )

        overall = sum(
            skill_scores.get(skill, 50) * weight
            for skill, weight in weights.items()
        )

        # NHL probability based on overall score
        nhl_prob = self._score_to_probability(overall, 'nhl')
        top_six_prob = self._score_to_probability(overall, 'top_six')
        star_prob = self._score_to_probability(overall, 'star')

        # Determine projected role
        projected_role = self._project_role(overall, profile.position, skill_scores)

        # Find strengths and weaknesses
        strengths, weaknesses = self._identify_traits(skill_scores)

        # Confidence based on data quality
        confidence = self._calculate_confidence(tracking_metrics, season_stats)

        return ProspectEvaluation(
            player_id=profile.player_id,
            overall_score=overall,
            nhl_probability=nhl_prob,
            top_six_probability=top_six_prob,
            star_probability=star_prob,
            projected_role=projected_role,
            skill_scores=skill_scores,
            comparable_players=[],  # Would need historical database
            strengths=strengths,
            weaknesses=weaknesses,
            projection_confidence=confidence,
        )

    def _score_to_probability(
        self,
        score: float,
        outcome: str,
    ) -> float:
        """Convert overall score to probability of outcome."""
        # Sigmoid function with different centers for different outcomes
        centers = {
            'nhl': 50,      # 50+ score = >50% chance of NHL
            'top_six': 65,  # 65+ score = >50% chance of top-6
            'star': 80,     # 80+ score = >50% chance of star
        }

        center = centers.get(outcome, 50)
        # Steepness parameter
        k = 0.1

        prob = 1.0 / (1.0 + np.exp(-k * (score - center)))
        return float(prob)

    def _project_role(
        self,
        overall: float,
        position: Position,
        skills: Dict[SkillCategory, float],
    ) -> str:
        """Project likely NHL role."""
        if position == Position.GOALIE:
            if overall >= 75:
                return "Starting Goalie"
            elif overall >= 60:
                return "NHL Backup"
            else:
                return "AHL Goalie"

        is_forward = position in [Position.CENTER, Position.LEFT_WING, Position.RIGHT_WING]

        if is_forward:
            if overall >= 80:
                return "First Line Forward"
            elif overall >= 70:
                return "Top-6 Forward"
            elif overall >= 60:
                return "Middle-6 Forward"
            elif overall >= 50:
                return "Bottom-6 Forward"
            else:
                return "AHL Forward"
        else:
            if overall >= 80:
                return "Top Pair Defenseman"
            elif overall >= 70:
                return "Top-4 Defenseman"
            elif overall >= 60:
                return "Second Pair Defenseman"
            elif overall >= 50:
                return "Third Pair/7th D"
            else:
                return "AHL Defenseman"

    def _identify_traits(
        self,
        skills: Dict[SkillCategory, float],
    ) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses."""
        strengths = []
        weaknesses = []

        for skill, score in skills.items():
            if score >= 70:
                strengths.append(f"Elite {skill.value}")
            elif score >= 60:
                strengths.append(f"Strong {skill.value}")

            if score <= 35:
                weaknesses.append(f"Poor {skill.value}")
            elif score <= 45:
                weaknesses.append(f"Below average {skill.value}")

        return strengths[:3], weaknesses[:3]

    def _calculate_confidence(
        self,
        metrics: TrackingMetrics,
        stats: Dict[str, Any],
    ) -> float:
        """Calculate confidence in projection."""
        confidence = 0.5  # Base confidence

        # More games = higher confidence
        games = stats.get('games', 0)
        if games >= 60:
            confidence += 0.2
        elif games >= 40:
            confidence += 0.1

        # Tracking data quality
        if metrics.top_speed > 0 and metrics.shot_velocity > 0:
            confidence += 0.2

        return min(0.95, confidence)


class ComparablesFinder:
    """
    Finds historical comparables for prospects.

    Uses tracking-derived skill profiles to find similar
    historical prospects.
    """

    def __init__(self):
        """Initialize with empty historical database."""
        self.historical_profiles: List[Dict[str, Any]] = []

    def add_historical_player(
        self,
        player_id: str,
        name: str,
        draft_position: int,
        skill_profile: Dict[SkillCategory, float],
        nhl_outcome: Dict[str, Any],
    ):
        """Add historical player to database."""
        self.historical_profiles.append({
            'player_id': player_id,
            'name': name,
            'draft_position': draft_position,
            'skills': skill_profile,
            'outcome': nhl_outcome,
        })

    def find_comparables(
        self,
        prospect_skills: Dict[SkillCategory, float],
        position: Position,
        k: int = 5,
    ) -> List[HistoricalComparable]:
        """
        Find k most similar historical prospects.

        Uses Euclidean distance in skill space.
        """
        if not self.historical_profiles:
            return []

        # Convert prospect skills to vector
        skill_order = list(SkillCategory)
        prospect_vec = np.array([
            prospect_skills.get(s, 50) for s in skill_order
        ])

        similarities = []
        for hist in self.historical_profiles:
            hist_vec = np.array([
                hist['skills'].get(s, 50) for s in skill_order
            ])

            # Euclidean distance (lower = more similar)
            distance = np.linalg.norm(prospect_vec - hist_vec)
            similarity = 1.0 / (1.0 + distance / 50)

            similarities.append({
                'profile': hist,
                'similarity': similarity,
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Convert to HistoricalComparable
        comparables = []
        for s in similarities[:k]:
            profile = s['profile']
            outcome = profile['outcome']
            comparables.append(HistoricalComparable(
                player_id=profile['player_id'],
                name=profile['name'],
                similarity_score=s['similarity'],
                nhl_outcome=outcome.get('role', 'Unknown'),
                games_played=outcome.get('games', 0),
                points=outcome.get('points', 0),
                draft_position=profile['draft_position'],
            ))

        return comparables


class DraftRankingModel:
    """
    Creates draft rankings from prospect evaluations.

    Combines multiple factors for comprehensive ranking.
    """

    def __init__(self, predictor: TalentPredictor):
        """Initialize with talent predictor."""
        self.predictor = predictor

    def create_ranking(
        self,
        prospects: List[Tuple[ProspectProfile, TrackingMetrics, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Create ranked list of prospects.

        Returns list sorted by overall ranking.
        """
        evaluations = []

        for profile, metrics, stats in prospects:
            evaluation = self.predictor.evaluate_prospect(profile, metrics, stats)

            # Calculate composite ranking score
            # Weights: Overall (50%), NHL probability (30%), Upside (20%)
            ranking_score = (
                0.5 * evaluation.overall_score +
                0.3 * (evaluation.nhl_probability * 100) +
                0.2 * (evaluation.star_probability * 100)
            )

            evaluations.append({
                'profile': profile,
                'evaluation': evaluation,
                'ranking_score': ranking_score,
            })

        # Sort by ranking score
        evaluations.sort(key=lambda x: x['ranking_score'], reverse=True)

        # Add rank
        rankings = []
        for rank, e in enumerate(evaluations, 1):
            rankings.append({
                'rank': rank,
                'player_id': e['profile'].player_id,
                'name': e['profile'].name,
                'position': e['profile'].position.value,
                'league': e['profile'].league.value,
                'overall_score': e['evaluation'].overall_score,
                'nhl_probability': e['evaluation'].nhl_probability,
                'projected_role': e['evaluation'].projected_role,
                'strengths': e['evaluation'].strengths,
                'ranking_score': e['ranking_score'],
            })

        return rankings
