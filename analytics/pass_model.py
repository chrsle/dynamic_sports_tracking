"""
Pass Probability and Completion Model for Hockey Analytics

This module implements pass probability models (xPass) adapted from
soccer analytics research using graph neural network approaches.

Features:
- Pass completion probability prediction
- Pass receiver prediction
- Pass difficulty assessment
- Defensive credit for blocked/intercepted passes
- Playmaking valuation

References:
    - Stats Perform AI Team. (2021). "Making Offensive Play Predictable -
      Using a GCN to Understand Defensive Performance in Soccer." MIT Sloan.
    - Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan.
    - Fernández, J., & Bornn, L. (2018). "Wide Open Spaces." MIT Sloan.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class PassType(Enum):
    """Types of passes in hockey."""
    TAPE_TO_TAPE = "tape_to_tape"
    SAUCER = "saucer"
    BANK = "bank"  # Off boards
    CHIP = "chip"
    BACKHAND = "backhand"
    ONE_TOUCH = "one_touch"
    STRETCH = "stretch"  # Long pass
    CROSS_ICE = "cross_ice"
    DROP = "drop"
    BEHIND_BACK = "behind_back"


class PassOutcome(Enum):
    """Outcomes of a pass attempt."""
    COMPLETE = "complete"
    INTERCEPTED = "intercepted"
    BLOCKED = "blocked"
    OFFSIDE = "offside"
    ICING = "icing"
    OUT_OF_PLAY = "out_of_play"
    MISSED = "missed"


class PassContext(Enum):
    """Context of pass attempt."""
    BREAKOUT = "breakout"
    ZONE_ENTRY = "zone_entry"
    CYCLE = "cycle"
    RUSH = "rush"
    POWER_PLAY = "power_play"
    PENALTY_KILL = "penalty_kill"
    NEUTRAL_ZONE = "neutral_zone"


@dataclass
class PlayerPosition:
    """Player position for pass modeling."""
    player_id: str
    team: str  # 'passer', 'receiver_team', 'opponent'
    x: float
    y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0


@dataclass
class PassAttempt:
    """Single pass attempt."""
    pass_id: str
    timestamp: float
    period: int

    # Passer info
    passer_id: str
    passer_x: float
    passer_y: float
    passer_velocity_x: float = 0.0
    passer_velocity_y: float = 0.0

    # Target info
    target_x: float
    target_y: float
    intended_receiver: Optional[str] = None
    actual_receiver: Optional[str] = None

    # Pass details
    pass_type: PassType = PassType.TAPE_TO_TAPE
    pass_speed: Optional[float] = None  # mph

    # Context
    context: PassContext = PassContext.NEUTRAL_ZONE
    strength_state: str = "5v5"
    under_pressure: bool = False

    # All player positions at time of pass
    all_positions: List[PlayerPosition] = field(default_factory=list)

    # Outcome (if known)
    outcome: Optional[PassOutcome] = None


@dataclass
class PassProbability:
    """Pass probability prediction result."""
    pass_id: str
    completion_probability: float
    interception_probability: float
    intended_receiver_prob: float

    # Difficulty factors
    distance_factor: float
    angle_factor: float
    pressure_factor: float
    lane_blockage_factor: float

    # Receiver probabilities (for receiver prediction)
    receiver_probs: Dict[str, float]

    # Value metrics
    pass_difficulty: float  # 0-1 scale
    pass_xT_gain: float  # Expected threat gained if complete


@dataclass
class DefensivePassCredit:
    """Defensive credit for preventing passes."""
    player_id: str
    pass_id: str
    credit_type: str  # 'block', 'interception', 'lane_denial'
    credit_value: float
    lane_coverage: float
    closing_speed: float


class HockeyPassModel:
    """
    Pass Completion Probability Model for Hockey.

    Predicts:
    1. Probability of pass completion (xPass)
    2. Most likely receiver
    3. Interception probability
    4. Pass difficulty rating

    Based on Stats Perform GCN research with hockey-specific adaptations
    for board play, saucer passes, and faster puck movement.
    """

    # Base completion rates by pass type
    BASE_COMPLETION_RATES = {
        PassType.TAPE_TO_TAPE: 0.85,
        PassType.SAUCER: 0.75,
        PassType.BANK: 0.70,
        PassType.CHIP: 0.65,
        PassType.BACKHAND: 0.70,
        PassType.ONE_TOUCH: 0.65,
        PassType.STRETCH: 0.60,
        PassType.CROSS_ICE: 0.70,
        PassType.DROP: 0.90,
        PassType.BEHIND_BACK: 0.55,
    }

    # Context adjustments
    CONTEXT_ADJUSTMENTS = {
        PassContext.BREAKOUT: -0.05,  # Harder
        PassContext.ZONE_ENTRY: -0.08,
        PassContext.CYCLE: 0.05,  # Easier
        PassContext.RUSH: -0.03,
        PassContext.POWER_PLAY: 0.08,
        PassContext.PENALTY_KILL: -0.10,
        PassContext.NEUTRAL_ZONE: 0.0,
    }

    # Puck travel speed (feet per second)
    AVG_PASS_SPEED = 60.0  # ~40 mph

    def __init__(
        self,
        interception_radius: float = 8.0,  # Feet
        lane_width: float = 4.0
    ):
        """
        Initialize pass model.

        Args:
            interception_radius: Radius for interception calculation
            lane_width: Width of passing lanes
        """
        self.interception_radius = interception_radius
        self.lane_width = lane_width

    def predict_pass(self, attempt: PassAttempt) -> PassProbability:
        """
        Predict pass completion probability.

        Args:
            attempt: Pass attempt to analyze

        Returns:
            PassProbability with predictions
        """
        # Calculate pass distance
        pass_distance = np.sqrt(
            (attempt.target_x - attempt.passer_x)**2 +
            (attempt.target_y - attempt.passer_y)**2
        )

        # Calculate pass angle (relative to forward direction)
        pass_angle = np.abs(np.arctan2(
            attempt.target_y - attempt.passer_y,
            attempt.target_x - attempt.passer_x
        ))

        # Base completion probability
        base_prob = self.BASE_COMPLETION_RATES.get(
            attempt.pass_type, 0.75
        )

        # Distance factor (longer = harder)
        distance_factor = self._compute_distance_factor(pass_distance)

        # Angle factor (cross-ice = harder)
        angle_factor = self._compute_angle_factor(pass_angle, pass_distance)

        # Pressure factor
        pressure_factor = self._compute_pressure_factor(attempt)

        # Lane blockage factor
        lane_factor = self._compute_lane_blockage(attempt, pass_distance)

        # Context adjustment
        context_adj = self.CONTEXT_ADJUSTMENTS.get(attempt.context, 0.0)

        # Combine factors
        completion_prob = base_prob
        completion_prob *= distance_factor
        completion_prob *= angle_factor
        completion_prob *= pressure_factor
        completion_prob *= lane_factor
        completion_prob += context_adj

        completion_prob = np.clip(completion_prob, 0.05, 0.98)

        # Interception probability
        interception_prob = self._compute_interception_prob(
            attempt, lane_factor, pass_distance
        )

        # Receiver probabilities
        receiver_probs = self._predict_receiver(attempt)

        # Intended receiver probability
        intended_receiver_prob = 0.0
        if attempt.intended_receiver and attempt.intended_receiver in receiver_probs:
            intended_receiver_prob = receiver_probs[attempt.intended_receiver]

        # Pass difficulty (inverse of completion prob)
        pass_difficulty = 1 - completion_prob

        # Expected threat gain (would integrate with xT model)
        xT_gain = self._estimate_xT_gain(attempt, completion_prob)

        return PassProbability(
            pass_id=attempt.pass_id,
            completion_probability=completion_prob,
            interception_probability=interception_prob,
            intended_receiver_prob=intended_receiver_prob,
            distance_factor=distance_factor,
            angle_factor=angle_factor,
            pressure_factor=pressure_factor,
            lane_blockage_factor=lane_factor,
            receiver_probs=receiver_probs,
            pass_difficulty=pass_difficulty,
            pass_xT_gain=xT_gain
        )

    def _compute_distance_factor(self, distance: float) -> float:
        """Compute completion factor based on pass distance."""
        # Completion drops off with distance
        # ~5% drop per 10 feet after 20 feet
        if distance <= 20:
            return 1.0
        elif distance <= 60:
            return 1.0 - (distance - 20) * 0.005
        else:
            return 0.8 - (distance - 60) * 0.003

    def _compute_angle_factor(self, angle: float, distance: float) -> float:
        """Compute completion factor based on pass angle."""
        # Cross-ice passes (high angle) are harder
        # Normalized angle: 0 = forward, pi/2 = sideways, pi = backward
        if angle < np.pi / 4:
            return 1.0
        elif angle < np.pi / 2:
            return 0.95 - (angle - np.pi / 4) * 0.1
        else:
            return 0.85 - (angle - np.pi / 2) * 0.15

    def _compute_pressure_factor(self, attempt: PassAttempt) -> float:
        """Compute completion factor based on pressure on passer."""
        if not attempt.under_pressure:
            return 1.0

        # Find closest defender to passer
        closest_dist = float('inf')
        for pos in attempt.all_positions:
            if pos.team == 'opponent':
                dist = np.sqrt(
                    (pos.x - attempt.passer_x)**2 +
                    (pos.y - attempt.passer_y)**2
                )
                closest_dist = min(closest_dist, dist)

        if closest_dist < 5:
            return 0.75  # Heavy pressure
        elif closest_dist < 10:
            return 0.85  # Moderate pressure
        elif closest_dist < 15:
            return 0.95  # Light pressure
        else:
            return 1.0

    def _compute_lane_blockage(
        self,
        attempt: PassAttempt,
        pass_distance: float
    ) -> float:
        """Compute factor based on passing lane blockage."""
        if pass_distance < 1:
            return 1.0

        # Pass direction (normalized)
        dx = (attempt.target_x - attempt.passer_x) / pass_distance
        dy = (attempt.target_y - attempt.passer_y) / pass_distance

        # Check for defenders in lane
        lane_blocked = False
        closest_blocker_dist = float('inf')

        for pos in attempt.all_positions:
            if pos.team != 'opponent':
                continue

            # Vector from passer to defender
            def_dx = pos.x - attempt.passer_x
            def_dy = pos.y - attempt.passer_y

            # Project onto pass direction
            proj = def_dx * dx + def_dy * dy

            # Check if in front of passer and before target
            if proj < 0 or proj > pass_distance:
                continue

            # Perpendicular distance from pass line
            perp_dist = abs(def_dx * dy - def_dy * dx)

            if perp_dist < self.lane_width:
                lane_blocked = True
                closest_blocker_dist = min(closest_blocker_dist, perp_dist)

        if not lane_blocked:
            return 1.0
        elif closest_blocker_dist < 2:
            return 0.60  # Direct block likely
        else:
            return 0.75 + closest_blocker_dist * 0.05

    def _compute_interception_prob(
        self,
        attempt: PassAttempt,
        lane_factor: float,
        pass_distance: float
    ) -> float:
        """Compute probability of pass being intercepted."""
        # Base interception probability
        base_prob = 0.05

        # Increase if lane is blocked
        if lane_factor < 0.8:
            base_prob += (1 - lane_factor) * 0.3

        # Increase for longer passes (more time to react)
        if pass_distance > 40:
            base_prob += (pass_distance - 40) * 0.002

        # Decrease for saucer passes (harder to intercept)
        if attempt.pass_type == PassType.SAUCER:
            base_prob *= 0.7

        return min(base_prob, 0.4)

    def _predict_receiver(self, attempt: PassAttempt) -> Dict[str, float]:
        """
        Predict probability of each teammate receiving the pass.

        Uses distance to target and open-ness.
        """
        receiver_probs = {}

        # Find teammates
        teammates = [
            pos for pos in attempt.all_positions
            if pos.team == 'receiver_team' and pos.player_id != attempt.passer_id
        ]

        if not teammates:
            return {}

        # Score each teammate
        scores = []
        for teammate in teammates:
            # Distance from target
            dist_to_target = np.sqrt(
                (teammate.x - attempt.target_x)**2 +
                (teammate.y - attempt.target_y)**2
            )

            # Distance score (closer = higher)
            if dist_to_target < 5:
                dist_score = 1.0
            elif dist_to_target < 15:
                dist_score = 0.8 - (dist_to_target - 5) * 0.05
            else:
                dist_score = 0.3 - (dist_to_target - 15) * 0.01

            # Open-ness score
            min_defender_dist = float('inf')
            for opp in attempt.all_positions:
                if opp.team == 'opponent':
                    opp_dist = np.sqrt(
                        (opp.x - teammate.x)**2 +
                        (opp.y - teammate.y)**2
                    )
                    min_defender_dist = min(min_defender_dist, opp_dist)

            if min_defender_dist > 15:
                open_score = 1.0
            elif min_defender_dist > 8:
                open_score = 0.8
            elif min_defender_dist > 4:
                open_score = 0.5
            else:
                open_score = 0.2

            scores.append((teammate.player_id, dist_score * open_score))

        # Normalize to probabilities
        total = sum(s[1] for s in scores)
        if total > 0:
            for player_id, score in scores:
                receiver_probs[player_id] = score / total

        return receiver_probs

    def _estimate_xT_gain(
        self,
        attempt: PassAttempt,
        completion_prob: float
    ) -> float:
        """Estimate expected threat gain from pass."""
        # Simplified xT gain based on location change
        # In production, would integrate with full xT model

        # xT increases as you move toward opponent goal
        # Assume goal at x=100

        start_xT = self._location_threat(attempt.passer_x, attempt.passer_y)
        end_xT = self._location_threat(attempt.target_x, attempt.target_y)

        # Expected gain = P(complete) * (end_xT - start_xT)
        xT_gain = completion_prob * (end_xT - start_xT)

        return xT_gain

    def _location_threat(self, x: float, y: float) -> float:
        """Estimate threat value of a location (simplified)."""
        # Distance to goal (assuming at x=89)
        dist_to_goal = np.sqrt((x - 89)**2 + y**2)

        # Higher threat closer to goal
        if dist_to_goal < 20:
            return 0.15 - dist_to_goal * 0.005
        elif dist_to_goal < 40:
            return 0.05 - (dist_to_goal - 20) * 0.002
        else:
            return 0.01


class DefensivePassAnalyzer:
    """
    Analyze defensive contribution to preventing passes.

    Credits defenders for:
    - Blocking passing lanes
    - Intercepting passes
    - Forcing bad passes
    """

    def __init__(self, pass_model: HockeyPassModel):
        """Initialize analyzer with pass model."""
        self.pass_model = pass_model
        self.player_credits: Dict[str, List[DefensivePassCredit]] = defaultdict(list)

    def analyze_defensive_impact(
        self,
        attempt: PassAttempt
    ) -> List[DefensivePassCredit]:
        """
        Analyze defensive impact on a pass attempt.

        Args:
            attempt: Pass attempt to analyze

        Returns:
            List of defensive credits
        """
        credits = []
        prediction = self.pass_model.predict_pass(attempt)

        # Find defenders who contributed to difficulty
        for pos in attempt.all_positions:
            if pos.team != 'opponent':
                continue

            # Check if in passing lane
            lane_coverage = self._compute_lane_coverage(attempt, pos)

            # Check closing speed toward passer
            closing_speed = self._compute_closing_speed(attempt, pos)

            # Credit if meaningful contribution
            if lane_coverage > 0.3 or closing_speed > 10:
                credit_value = self._compute_credit_value(
                    lane_coverage, closing_speed, prediction
                )

                credit = DefensivePassCredit(
                    player_id=pos.player_id,
                    pass_id=attempt.pass_id,
                    credit_type=self._determine_credit_type(
                        attempt.outcome, lane_coverage
                    ),
                    credit_value=credit_value,
                    lane_coverage=lane_coverage,
                    closing_speed=closing_speed
                )

                credits.append(credit)
                self.player_credits[pos.player_id].append(credit)

        return credits

    def _compute_lane_coverage(
        self,
        attempt: PassAttempt,
        defender: PlayerPosition
    ) -> float:
        """Compute how much of the passing lane a defender covers."""
        pass_distance = np.sqrt(
            (attempt.target_x - attempt.passer_x)**2 +
            (attempt.target_y - attempt.passer_y)**2
        )

        if pass_distance < 1:
            return 0.0

        # Pass direction
        dx = (attempt.target_x - attempt.passer_x) / pass_distance
        dy = (attempt.target_y - attempt.passer_y) / pass_distance

        # Vector from passer to defender
        def_dx = defender.x - attempt.passer_x
        def_dy = defender.y - attempt.passer_y

        # Project onto pass direction
        proj = def_dx * dx + def_dy * dy

        if proj < 0 or proj > pass_distance:
            return 0.0

        # Perpendicular distance
        perp_dist = abs(def_dx * dy - def_dy * dx)

        # Coverage score
        if perp_dist < 3:
            return 1.0
        elif perp_dist < 6:
            return 0.7
        elif perp_dist < 10:
            return 0.3
        else:
            return 0.0

    def _compute_closing_speed(
        self,
        attempt: PassAttempt,
        defender: PlayerPosition
    ) -> float:
        """Compute defender's closing speed toward passer."""
        # Direction from defender to passer
        dx = attempt.passer_x - defender.x
        dy = attempt.passer_y - defender.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1:
            return 0.0

        dx /= dist
        dy /= dist

        # Dot product with velocity
        closing = defender.velocity_x * dx + defender.velocity_y * dy

        return max(0, closing)

    def _compute_credit_value(
        self,
        lane_coverage: float,
        closing_speed: float,
        prediction: PassProbability
    ) -> float:
        """Compute defensive credit value."""
        # Base credit from difficulty created
        base_credit = prediction.pass_difficulty * 0.01

        # Multiply by contribution
        contribution = lane_coverage * 0.6 + (closing_speed / 25) * 0.4

        return base_credit * contribution

    def _determine_credit_type(
        self,
        outcome: Optional[PassOutcome],
        lane_coverage: float
    ) -> str:
        """Determine type of defensive credit."""
        if outcome == PassOutcome.BLOCKED:
            return 'block'
        elif outcome == PassOutcome.INTERCEPTED:
            return 'interception'
        elif lane_coverage > 0.5:
            return 'lane_denial'
        else:
            return 'pressure'

    def get_player_defensive_stats(self, player_id: str) -> Dict[str, float]:
        """Get defensive passing stats for a player."""
        credits = self.player_credits.get(player_id, [])

        if not credits:
            return {}

        return {
            'total_pass_defense_value': sum(c.credit_value for c in credits),
            'passes_defended': len(credits),
            'blocks': sum(1 for c in credits if c.credit_type == 'block'),
            'interceptions': sum(1 for c in credits if c.credit_type == 'interception'),
            'lane_denials': sum(1 for c in credits if c.credit_type == 'lane_denial'),
            'avg_lane_coverage': np.mean([c.lane_coverage for c in credits]),
            'avg_closing_speed': np.mean([c.closing_speed for c in credits])
        }


class PlaymakingEvaluator:
    """
    Evaluate playmaking ability based on pass metrics.
    """

    def __init__(self, pass_model: HockeyPassModel):
        """Initialize evaluator."""
        self.pass_model = pass_model
        self.player_passes: Dict[str, List[Tuple[PassAttempt, PassProbability]]] = defaultdict(list)

    def record_pass(self, attempt: PassAttempt):
        """Record a pass attempt for evaluation."""
        prediction = self.pass_model.predict_pass(attempt)
        self.player_passes[attempt.passer_id].append((attempt, prediction))

    def evaluate_playmaker(self, player_id: str) -> Dict[str, Any]:
        """
        Evaluate a player's playmaking ability.

        Returns:
            Dictionary with playmaking metrics
        """
        passes = self.player_passes.get(player_id, [])

        if not passes:
            return {}

        # Completion rate
        completed = sum(
            1 for a, _ in passes
            if a.outcome == PassOutcome.COMPLETE
        )
        completion_rate = completed / len(passes)

        # Expected completion (average xPass)
        expected_completion = np.mean([p.completion_probability for _, p in passes])

        # Completion over expected
        coe = completion_rate - expected_completion

        # Average pass difficulty
        avg_difficulty = np.mean([p.pass_difficulty for _, p in passes])

        # Total xT generated
        total_xT = sum(p.pass_xT_gain for _, p in passes if p.pass_xT_gain > 0)

        # Pass type distribution
        pass_types = defaultdict(int)
        for a, _ in passes:
            pass_types[a.pass_type.value] += 1

        return {
            'total_passes': len(passes),
            'completion_rate': completion_rate,
            'expected_completion': expected_completion,
            'completion_over_expected': coe,
            'avg_pass_difficulty': avg_difficulty,
            'total_xT_generated': total_xT,
            'xT_per_pass': total_xT / len(passes),
            'pass_type_distribution': dict(pass_types),
            'playmaking_score': (completion_rate + coe + avg_difficulty) / 3
        }
