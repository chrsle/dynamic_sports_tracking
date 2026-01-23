"""
Puck Trajectory Analysis Model

Implements puck trajectory prediction and analysis, translated from
basketball trajectory research.

Key Paper:
- Shah, R., et al. "Applying Deep Learning to Basketball Trajectories."
  Semantic Scholar.

Key Concepts:
- RNNs predict whether shots will be successful
- Models learn trajectory without physics knowledge
- Uses tracking data (25 fps in basketball, 60+ Hz in hockey)
- Outperforms feature-rich ML models

Hockey Translation:
- Predict shot success based on puck trajectory
- Model deflection probability
- Evaluate passing accuracy predictions
- Analyze puck flutter and aerodynamics
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class ShotType(Enum):
    """Types of hockey shots."""
    WRIST = "wrist"
    SLAP = "slap"
    SNAP = "snap"
    BACKHAND = "backhand"
    TIP = "tip"
    DEFLECTION = "deflection"
    ONE_TIMER = "one_timer"


class ShotOutcome(Enum):
    """Possible shot outcomes."""
    GOAL = "goal"
    SAVE = "save"
    MISS_WIDE = "miss_wide"
    MISS_HIGH = "miss_high"
    BLOCKED = "blocked"
    POST = "post"
    CROSSBAR = "crossbar"


class PassType(Enum):
    """Types of passes."""
    TAPE_TO_TAPE = "tape_to_tape"
    SAUCER = "saucer"
    BANK = "bank"
    CHIP = "chip"
    CROSS_ICE = "cross_ice"
    DROP = "drop"


@dataclass
class PuckState:
    """Complete puck state at a moment."""
    timestamp: float                # seconds
    x: float                        # feet
    y: float                        # feet
    z: float                        # feet (height off ice)
    velocity_x: float               # ft/s
    velocity_y: float               # ft/s
    velocity_z: float               # ft/s
    spin_rate: float = 0.0          # rpm
    spin_axis: Tuple[float, float, float] = (0, 0, 1)

    @property
    def speed(self) -> float:
        """Total speed in ft/s."""
        return np.sqrt(
            self.velocity_x**2 +
            self.velocity_y**2 +
            self.velocity_z**2
        )

    @property
    def speed_mph(self) -> float:
        """Speed in mph."""
        return self.speed * 0.681818  # ft/s to mph


@dataclass
class ShotTrajectory:
    """Complete shot trajectory."""
    shot_id: str
    shooter_id: str
    shot_type: ShotType
    states: List[PuckState]
    outcome: Optional[ShotOutcome] = None
    goalie_id: Optional[str] = None
    release_location: Optional[Tuple[float, float]] = None
    target_location: Optional[Tuple[float, float]] = None


@dataclass
class PassTrajectory:
    """Complete pass trajectory."""
    pass_id: str
    passer_id: str
    receiver_id: str
    pass_type: PassType
    states: List[PuckState]
    completed: bool = False
    intercepted_by: Optional[str] = None


@dataclass
class TrajectoryPrediction:
    """Prediction for trajectory outcome."""
    predicted_outcome: str
    probability: float
    predicted_location: Tuple[float, float, float]
    time_to_target: float
    confidence: float


class PuckPhysicsModel:
    """
    Models puck physics including aerodynamics.

    Hockey puck aerodynamics are complex due to:
    - High speeds (up to 100+ mph)
    - Spin effects (Magnus force)
    - Flutter at certain velocities
    - Drag coefficients varying with orientation
    """

    # Puck properties
    PUCK_MASS = 0.375  # lbs (6 oz)
    PUCK_DIAMETER = 3.0  # inches
    PUCK_THICKNESS = 1.0  # inches

    # Physics constants
    GRAVITY = 32.174  # ft/s^2
    AIR_DENSITY = 0.0023769  # slug/ft^3 at sea level

    # Drag and lift coefficients
    DRAG_COEFFICIENT = 0.5
    LIFT_COEFFICIENT = 0.3  # Magnus effect

    def __init__(self):
        """Initialize physics model."""
        # Cross-sectional area in ft^2
        self.cross_section = np.pi * (self.PUCK_DIAMETER / 24)**2

    def predict_trajectory(
        self,
        initial_state: PuckState,
        duration: float,
        dt: float = 0.001,  # 1ms timestep
    ) -> List[PuckState]:
        """
        Predict puck trajectory from initial state.

        Uses numerical integration with aerodynamic forces.
        """
        states = [initial_state]
        current = initial_state

        t = 0
        while t < duration:
            # Calculate forces
            velocity = np.array([
                current.velocity_x,
                current.velocity_y,
                current.velocity_z
            ])
            speed = np.linalg.norm(velocity)

            if speed < 0.1:
                break

            # Drag force
            drag_magnitude = (
                0.5 * self.AIR_DENSITY *
                speed**2 * self.cross_section *
                self.DRAG_COEFFICIENT
            )
            drag_direction = -velocity / speed
            drag_force = drag_magnitude * drag_direction

            # Magnus force (if spinning)
            if current.spin_rate > 0:
                spin_axis = np.array(current.spin_axis)
                magnus_magnitude = (
                    self.LIFT_COEFFICIENT *
                    current.spin_rate / 60 *  # Convert to Hz
                    speed
                )
                magnus_direction = np.cross(spin_axis, velocity / speed)
                if np.linalg.norm(magnus_direction) > 0:
                    magnus_direction = magnus_direction / np.linalg.norm(magnus_direction)
                magnus_force = magnus_magnitude * magnus_direction
            else:
                magnus_force = np.array([0, 0, 0])

            # Gravity
            gravity_force = np.array([0, 0, -self.GRAVITY * self.PUCK_MASS])

            # Total acceleration
            total_force = drag_force + magnus_force + gravity_force
            acceleration = total_force / self.PUCK_MASS

            # Update velocity
            new_velocity = velocity + acceleration * dt

            # Update position
            new_position = np.array([
                current.x, current.y, current.z
            ]) + velocity * dt

            # Check for ice contact
            if new_position[2] < 0:
                new_position[2] = 0
                new_velocity[2] = -new_velocity[2] * 0.3  # Bounce with energy loss

            # Create new state
            new_state = PuckState(
                timestamp=current.timestamp + dt,
                x=float(new_position[0]),
                y=float(new_position[1]),
                z=float(new_position[2]),
                velocity_x=float(new_velocity[0]),
                velocity_y=float(new_velocity[1]),
                velocity_z=float(new_velocity[2]),
                spin_rate=current.spin_rate * 0.9999,  # Spin decay
                spin_axis=current.spin_axis,
            )

            states.append(new_state)
            current = new_state
            t += dt

        return states

    def calculate_flutter_probability(
        self,
        velocity: float,
        spin_rate: float,
    ) -> float:
        """
        Calculate probability of puck flutter.

        Flutter occurs at certain speed/spin combinations
        causing unpredictable movement.
        """
        # Flutter zone is typically 60-75 mph with low spin
        velocity_mph = velocity * 0.681818

        if velocity_mph < 50 or velocity_mph > 90:
            return 0.1  # Low flutter outside main range

        # Lower spin = more flutter tendency
        spin_factor = 1.0 / (1.0 + spin_rate / 1000)

        # Peak flutter around 70 mph
        velocity_factor = 1.0 - abs(velocity_mph - 70) / 30

        return float(spin_factor * velocity_factor * 0.8)


class ShotSuccessPredictor:
    """
    Predicts shot success using trajectory analysis.

    Uses RNN-inspired approach to analyze puck trajectory
    and predict if shot will be a goal.
    """

    def __init__(self, physics_model: PuckPhysicsModel):
        """Initialize predictor."""
        self.physics = physics_model

        # Model weights (placeholder - would be learned)
        np.random.seed(42)
        self.weights = {
            'trajectory_encoder': np.random.randn(12, 32) * 0.1,
            'context_encoder': np.random.randn(8, 16) * 0.1,
            'output': np.random.randn(48, 2) * 0.1,
        }

    def extract_trajectory_features(
        self,
        trajectory: ShotTrajectory,
    ) -> np.ndarray:
        """
        Extract features from shot trajectory.

        Features include:
        - Release location and velocity
        - Trajectory shape (curvature, flutter)
        - Time characteristics
        """
        if not trajectory.states:
            return np.zeros(12)

        # Release state
        release = trajectory.states[0]

        # Target estimation (last state or predicted)
        target = trajectory.states[-1] if len(trajectory.states) > 1 else release

        # Trajectory statistics
        speeds = [s.speed for s in trajectory.states]
        heights = [s.z for s in trajectory.states]

        features = np.array([
            # Release characteristics
            release.x / 200,  # Normalized x
            release.y / 85,   # Normalized y
            release.speed_mph / 100,  # Normalized speed
            release.z / 5,    # Height at release

            # Trajectory shape
            np.mean(heights) / 5,  # Average height
            np.max(heights) / 5,   # Max height
            np.std(heights) if len(heights) > 1 else 0,  # Height variation

            # Velocity characteristics
            np.mean(speeds) / 100 if speeds else 0,
            np.std(speeds) / 20 if len(speeds) > 1 else 0,

            # Duration and distance
            (target.timestamp - release.timestamp),
            np.sqrt((target.x - release.x)**2 + (target.y - release.y)**2) / 100,

            # Spin
            release.spin_rate / 5000,
        ])

        return features

    def extract_context_features(
        self,
        trajectory: ShotTrajectory,
        goalie_position: Optional[Tuple[float, float]] = None,
        screen_present: bool = False,
        rebound: bool = False,
    ) -> np.ndarray:
        """
        Extract contextual features for shot.
        """
        release = trajectory.states[0] if trajectory.states else PuckState(0, 0, 0, 0, 0, 0, 0)

        # Distance to goal
        goal_x = 189  # Goal line
        goal_y = 42.5  # Center
        distance = np.sqrt((release.x - goal_x)**2 + (release.y - goal_y)**2)

        # Angle to goal
        angle = np.arctan2(release.y - goal_y, goal_x - release.x)

        # Goalie factors
        if goalie_position:
            goalie_x, goalie_y = goalie_position
            goalie_dist = np.sqrt((release.x - goalie_x)**2 + (release.y - goalie_y)**2)
        else:
            goalie_dist = 10  # Default

        features = np.array([
            distance / 100,
            abs(angle) / np.pi,
            goalie_dist / 50,
            1.0 if screen_present else 0.0,
            1.0 if rebound else 0.0,
            1.0 if trajectory.shot_type == ShotType.ONE_TIMER else 0.0,
            1.0 if trajectory.shot_type == ShotType.SLAP else 0.0,
            1.0 if trajectory.shot_type == ShotType.TIP else 0.0,
        ])

        return features

    def predict_success(
        self,
        trajectory: ShotTrajectory,
        goalie_position: Optional[Tuple[float, float]] = None,
        screen_present: bool = False,
        rebound: bool = False,
    ) -> TrajectoryPrediction:
        """
        Predict if shot will be successful.
        """
        # Extract features
        traj_features = self.extract_trajectory_features(trajectory)
        ctx_features = self.extract_context_features(
            trajectory, goalie_position, screen_present, rebound
        )

        # Encode features (simplified forward pass)
        traj_encoded = np.tanh(traj_features @ self.weights['trajectory_encoder'])
        ctx_encoded = np.tanh(ctx_features @ self.weights['context_encoder'])

        # Combine and predict
        combined = np.concatenate([traj_encoded, ctx_encoded])
        logits = combined @ self.weights['output']

        # Softmax for probability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        goal_prob = float(probs[1])

        # Predict target location
        if trajectory.states:
            release = trajectory.states[0]
            predicted_states = self.physics.predict_trajectory(release, 0.5)
            if predicted_states:
                final = predicted_states[-1]
                pred_location = (final.x, final.y, final.z)
                time_to_target = final.timestamp - release.timestamp
            else:
                pred_location = (0, 0, 0)
                time_to_target = 0
        else:
            pred_location = (0, 0, 0)
            time_to_target = 0

        return TrajectoryPrediction(
            predicted_outcome="goal" if goal_prob > 0.5 else "save",
            probability=goal_prob,
            predicted_location=pred_location,
            time_to_target=time_to_target,
            confidence=float(max(probs)),
        )


class PassCompletionPredictor:
    """
    Predicts pass completion probability using trajectory.
    """

    def __init__(self, physics_model: PuckPhysicsModel):
        """Initialize predictor."""
        self.physics = physics_model

    def predict_completion(
        self,
        pass_trajectory: PassTrajectory,
        receiver_position: Tuple[float, float],
        defenders: List[Tuple[float, float]],
    ) -> TrajectoryPrediction:
        """
        Predict if pass will be completed.
        """
        if not pass_trajectory.states:
            return TrajectoryPrediction(
                predicted_outcome="incomplete",
                probability=0.0,
                predicted_location=(0, 0, 0),
                time_to_target=0,
                confidence=0.0,
            )

        release = pass_trajectory.states[0]

        # Predict trajectory
        predicted = self.physics.predict_trajectory(release, 2.0)

        # Find closest point to receiver
        receiver_x, receiver_y = receiver_position
        min_dist = float('inf')
        arrival_state = None

        for state in predicted:
            dist = np.sqrt((state.x - receiver_x)**2 + (state.y - receiver_y)**2)
            if dist < min_dist:
                min_dist = dist
                arrival_state = state

        # Check for interceptions
        intercepted = False
        for def_x, def_y in defenders:
            for state in predicted:
                def_dist = np.sqrt((state.x - def_x)**2 + (state.y - def_y)**2)
                if def_dist < 5:  # Within reach
                    # Check if defender is between passer and receiver
                    pass_dist = np.sqrt(
                        (release.x - receiver_x)**2 +
                        (release.y - receiver_y)**2
                    )
                    def_from_passer = np.sqrt(
                        (release.x - def_x)**2 +
                        (release.y - def_y)**2
                    )
                    if def_from_passer < pass_dist * 0.9:
                        intercepted = True
                        break
            if intercepted:
                break

        # Calculate completion probability
        if intercepted:
            completion_prob = 0.2  # Some chance of avoiding intercept
        elif min_dist < 3:
            completion_prob = 0.95
        elif min_dist < 6:
            completion_prob = 0.8
        elif min_dist < 10:
            completion_prob = 0.5
        else:
            completion_prob = 0.2

        # Adjust for pass type
        if pass_trajectory.pass_type == PassType.SAUCER:
            completion_prob *= 1.1  # Harder to intercept
        elif pass_trajectory.pass_type == PassType.CROSS_ICE:
            completion_prob *= 0.9  # Riskier

        completion_prob = min(1.0, max(0.0, completion_prob))

        return TrajectoryPrediction(
            predicted_outcome="complete" if completion_prob > 0.5 else "incomplete",
            probability=completion_prob,
            predicted_location=(
                arrival_state.x if arrival_state else 0,
                arrival_state.y if arrival_state else 0,
                arrival_state.z if arrival_state else 0,
            ),
            time_to_target=(
                arrival_state.timestamp - release.timestamp
                if arrival_state else 0
            ),
            confidence=abs(completion_prob - 0.5) * 2,
        )


class DeflectionAnalyzer:
    """
    Analyzes deflection and tip probabilities.
    """

    def __init__(self, physics_model: PuckPhysicsModel):
        """Initialize analyzer."""
        self.physics = physics_model

    def analyze_deflection_opportunity(
        self,
        shot_trajectory: ShotTrajectory,
        tipper_position: Tuple[float, float, float],
        tipper_reach: float = 6.0,  # feet
    ) -> Dict[str, Any]:
        """
        Analyze deflection opportunity.

        Returns probability and optimal tip timing/angle.
        """
        if not shot_trajectory.states:
            return {
                'can_reach': False,
                'deflection_probability': 0.0,
            }

        tipper_x, tipper_y, tipper_z = tipper_position

        # Predict trajectory
        release = shot_trajectory.states[0]
        trajectory = self.physics.predict_trajectory(release, 1.0)

        # Find closest approach to tipper
        min_dist = float('inf')
        closest_state = None
        closest_time = 0

        for state in trajectory:
            dist = np.sqrt(
                (state.x - tipper_x)**2 +
                (state.y - tipper_y)**2 +
                (state.z - tipper_z)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_state = state
                closest_time = state.timestamp

        # Can reach?
        can_reach = min_dist < tipper_reach

        if not can_reach:
            return {
                'can_reach': False,
                'deflection_probability': 0.0,
                'closest_distance': min_dist,
            }

        # Deflection difficulty based on:
        # - Puck speed (faster = harder)
        # - Height (at stick height = easier)
        # - Angle (perpendicular = harder)

        speed_factor = 1.0 - min(closest_state.speed_mph / 100, 1.0) * 0.5
        height_factor = 1.0 - abs(closest_state.z - 2.5) / 3  # Optimal at ~2.5 ft
        distance_factor = 1.0 - min_dist / tipper_reach

        deflection_prob = speed_factor * height_factor * distance_factor * 0.6

        # Direction after deflection (toward goal)
        goal_direction = np.array([189 - tipper_x, 42.5 - tipper_y, 0])
        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 0.001)

        return {
            'can_reach': True,
            'deflection_probability': float(deflection_prob),
            'closest_distance': float(min_dist),
            'optimal_contact_time': float(closest_time),
            'puck_speed_at_contact': float(closest_state.speed_mph),
            'recommended_tip_direction': tuple(goal_direction),
        }


class TrajectoryVisualizer:
    """
    Prepares trajectory data for visualization.
    """

    def trajectory_to_points(
        self,
        trajectory: List[PuckState],
        sample_rate: int = 10,
    ) -> List[Dict[str, float]]:
        """
        Convert trajectory to visualization points.

        Args:
            trajectory: List of puck states
            sample_rate: Keep every Nth point

        Returns:
            List of point dictionaries for visualization
        """
        points = []

        for i, state in enumerate(trajectory):
            if i % sample_rate != 0:
                continue

            points.append({
                'x': state.x,
                'y': state.y,
                'z': state.z,
                't': state.timestamp,
                'speed': state.speed_mph,
            })

        return points

    def compare_trajectories(
        self,
        predicted: List[PuckState],
        actual: List[PuckState],
    ) -> Dict[str, float]:
        """
        Compare predicted vs actual trajectories.

        Returns error metrics.
        """
        if not predicted or not actual:
            return {'error': 'insufficient_data'}

        # Align by time
        errors = []
        for pred in predicted:
            # Find closest actual by time
            closest = min(actual, key=lambda a: abs(a.timestamp - pred.timestamp))

            if abs(closest.timestamp - pred.timestamp) < 0.1:  # Within 100ms
                dist_error = np.sqrt(
                    (pred.x - closest.x)**2 +
                    (pred.y - closest.y)**2 +
                    (pred.z - closest.z)**2
                )
                errors.append(dist_error)

        if not errors:
            return {'error': 'no_aligned_points'}

        return {
            'mean_error_feet': float(np.mean(errors)),
            'max_error_feet': float(np.max(errors)),
            'rmse': float(np.sqrt(np.mean(np.array(errors)**2))),
            'num_points': len(errors),
        }
