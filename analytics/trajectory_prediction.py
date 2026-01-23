"""
Trajectory Prediction for Hockey Analytics

This module implements player trajectory prediction adapted from Social LSTM
and CNN-LSTM models used in pedestrian tracking and sports analytics.

Applications:
- Predict player positions 1-3 seconds ahead
- Defensive positioning evaluation ("ghosting")
- Off-puck movement quality assessment
- Goalie positioning prediction

References:
    - Alahi, A., et al. (2016). "Social LSTM: Human Trajectory Prediction in Crowded Spaces." CVPR.
    - Amazon Science. "Prediction of Defensive Player Trajectories in NFL Games."
    - Shah, R., et al. "Applying Deep Learning to Basketball Trajectories."
    - Lucey, P., et al. (2013). "Ghosting: Using Deep Learning to simulate defender trajectories."
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class PlayerType(Enum):
    """Player types for trajectory prediction."""
    FORWARD = "forward"
    DEFENSEMAN = "defenseman"
    GOALIE = "goalie"
    PUCK_CARRIER = "puck_carrier"


@dataclass
class TrackingFrame:
    """Single frame of tracking data."""
    timestamp: float
    player_id: str
    x: float  # Ice coordinates (-100 to 100)
    y: float  # Ice coordinates (-42.5 to 42.5)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    has_puck: bool = False
    player_type: PlayerType = PlayerType.FORWARD


@dataclass
class TrajectoryPrediction:
    """Predicted trajectory for a player."""
    player_id: str
    timestamps: List[float]  # Future timestamps
    positions: List[Tuple[float, float]]  # Predicted (x, y) positions
    velocities: List[Tuple[float, float]]  # Predicted velocities
    confidence: List[float]  # Confidence at each timestep
    prediction_type: str  # Model used


@dataclass
class GhostingResult:
    """Result of ghosting analysis (optimal vs actual positioning)."""
    player_id: str
    timestamp: float
    actual_position: Tuple[float, float]
    optimal_position: Tuple[float, float]
    position_error: float  # Distance from optimal
    coverage_impact: float  # How much coverage was affected
    grade: str  # A, B, C, D, F


class SocialPooling:
    """
    Social pooling layer for modeling player interactions.

    Based on Alahi et al. (2016) Social LSTM architecture.
    Players influence each other's trajectories based on proximity.
    """

    def __init__(
        self,
        grid_size: int = 8,
        neighborhood_size: float = 30.0,  # feet
        hidden_dim: int = 64
    ):
        """
        Initialize social pooling.

        Args:
            grid_size: Size of pooling grid
            neighborhood_size: Radius of neighborhood in feet
            hidden_dim: Hidden state dimension
        """
        self.grid_size = grid_size
        self.neighborhood = neighborhood_size
        self.hidden_dim = hidden_dim

    def compute_social_tensor(
        self,
        target_pos: Tuple[float, float],
        neighbor_states: List[Tuple[Tuple[float, float], np.ndarray]]
    ) -> np.ndarray:
        """
        Compute social tensor encoding nearby players.

        Args:
            target_pos: Position of target player
            neighbor_states: List of (position, hidden_state) for neighbors

        Returns:
            Pooled social tensor
        """
        # Initialize grid
        social_tensor = np.zeros((self.grid_size, self.grid_size, self.hidden_dim))

        cell_size = 2 * self.neighborhood / self.grid_size

        for pos, hidden_state in neighbor_states:
            # Relative position
            rel_x = pos[0] - target_pos[0]
            rel_y = pos[1] - target_pos[1]

            # Check if in neighborhood
            if abs(rel_x) > self.neighborhood or abs(rel_y) > self.neighborhood:
                continue

            # Grid cell
            cell_x = int((rel_x + self.neighborhood) / cell_size)
            cell_y = int((rel_y + self.neighborhood) / cell_size)

            cell_x = np.clip(cell_x, 0, self.grid_size - 1)
            cell_y = np.clip(cell_y, 0, self.grid_size - 1)

            # Pool hidden state (sum pooling)
            social_tensor[cell_y, cell_x] += hidden_state[:self.hidden_dim]

        return social_tensor.flatten()


class HockeyTrajectoryPredictor:
    """
    Player Trajectory Prediction Model for Hockey.

    Combines:
    1. Individual motion model (velocity, acceleration)
    2. Social interaction model (nearby players)
    3. Puck awareness model (reaction to puck movement)
    4. Tactical model (role-based positioning)

    Based on Social LSTM with hockey-specific adaptations.
    """

    # Hockey-specific physics constraints
    MAX_SKATING_SPEED = 35.0  # ft/s (~24 mph)
    MAX_ACCELERATION = 15.0  # ft/s^2
    MAX_TURNING_RATE = 3.0  # radians/s

    # Prediction horizons
    SHORT_HORIZON = 0.5  # seconds
    MEDIUM_HORIZON = 1.5
    LONG_HORIZON = 3.0

    def __init__(
        self,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        sequence_length: int = 10,
        prediction_length: int = 15
    ):
        """
        Initialize trajectory predictor.

        Args:
            hidden_dim: LSTM hidden state dimension
            embedding_dim: Input embedding dimension
            sequence_length: Number of past frames to consider
            prediction_length: Number of future frames to predict
        """
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.seq_length = sequence_length
        self.pred_length = prediction_length

        # Social pooling
        self.social_pooling = SocialPooling(hidden_dim=hidden_dim)

        # Player hidden states (simplified - would be LSTM in full implementation)
        self.player_states: Dict[str, np.ndarray] = {}

        # History buffers
        self.history: Dict[str, deque] = {}

    def update_player_state(
        self,
        frame: TrackingFrame,
        neighbors: List[TrackingFrame]
    ):
        """
        Update player's internal state with new observation.

        Args:
            frame: Current tracking frame for player
            neighbors: Tracking frames for nearby players
        """
        player_id = frame.player_id

        # Initialize history if needed
        if player_id not in self.history:
            self.history[player_id] = deque(maxlen=self.seq_length)
            self.player_states[player_id] = np.zeros(self.hidden_dim)

        # Add to history
        self.history[player_id].append(frame)

        # Update hidden state (simplified - LSTM update in full implementation)
        # Input embedding: position, velocity, acceleration, puck status
        input_vec = np.array([
            frame.x / 100,  # Normalize
            frame.y / 50,
            frame.velocity_x / self.MAX_SKATING_SPEED,
            frame.velocity_y / self.MAX_SKATING_SPEED,
            frame.acceleration_x / self.MAX_ACCELERATION,
            frame.acceleration_y / self.MAX_ACCELERATION,
            1.0 if frame.has_puck else 0.0,
            frame.player_type.value == PlayerType.DEFENSEMAN.value
        ])

        # Social interaction
        neighbor_states = [
            ((n.x, n.y), self.player_states.get(n.player_id, np.zeros(self.hidden_dim)))
            for n in neighbors
        ]
        social_tensor = self.social_pooling.compute_social_tensor(
            (frame.x, frame.y),
            neighbor_states
        )

        # Simple state update (tanh nonlinearity)
        combined = np.concatenate([input_vec, social_tensor[:self.hidden_dim - len(input_vec)]])
        combined = np.pad(combined, (0, max(0, self.hidden_dim - len(combined))))[:self.hidden_dim]

        self.player_states[player_id] = np.tanh(
            0.9 * self.player_states[player_id] + 0.1 * combined
        )

    def predict_trajectory(
        self,
        player_id: str,
        horizon: float = 1.5,
        puck_trajectory: Optional[List[Tuple[float, float]]] = None
    ) -> TrajectoryPrediction:
        """
        Predict future trajectory for a player.

        Args:
            player_id: Player to predict
            horizon: Prediction horizon in seconds
            puck_trajectory: Optional known/predicted puck positions

        Returns:
            TrajectoryPrediction with positions and confidence
        """
        if player_id not in self.history or len(self.history[player_id]) < 2:
            return self._empty_prediction(player_id)

        history = list(self.history[player_id])
        current = history[-1]

        # Number of frames to predict (assuming 30 fps)
        n_frames = int(horizon * 30)
        dt = 1 / 30  # Time step

        # Initialize prediction
        timestamps = []
        positions = []
        velocities = []
        confidences = []

        # Current state
        x, y = current.x, current.y
        vx, vy = current.velocity_x, current.velocity_y
        ax, ay = current.acceleration_x, current.acceleration_y

        # Predict using physics + learned corrections
        for i in range(n_frames):
            t = current.timestamp + (i + 1) * dt

            # Physics-based prediction (constant acceleration)
            new_vx = vx + ax * dt
            new_vy = vy + ay * dt

            # Apply speed constraints
            speed = np.sqrt(new_vx**2 + new_vy**2)
            if speed > self.MAX_SKATING_SPEED:
                scale = self.MAX_SKATING_SPEED / speed
                new_vx *= scale
                new_vy *= scale

            # Update position
            new_x = x + new_vx * dt
            new_y = y + new_vy * dt

            # Apply boundary constraints (ice surface)
            new_x = np.clip(new_x, -100, 100)
            new_y = np.clip(new_y, -42.5, 42.5)

            # Puck awareness adjustment
            if puck_trajectory and i < len(puck_trajectory):
                puck_x, puck_y = puck_trajectory[i]
                puck_influence = self._compute_puck_influence(
                    (new_x, new_y), (puck_x, puck_y), current.player_type
                )
                new_x += puck_influence[0] * dt
                new_y += puck_influence[1] * dt

            # Acceleration decay (players don't accelerate indefinitely)
            ax *= 0.9
            ay *= 0.9

            # Store prediction
            timestamps.append(t)
            positions.append((new_x, new_y))
            velocities.append((new_vx, new_vy))

            # Confidence decreases with time
            confidence = max(0.3, 1.0 - (i / n_frames) * 0.7)
            confidences.append(confidence)

            # Update state for next iteration
            x, y = new_x, new_y
            vx, vy = new_vx, new_vy

        return TrajectoryPrediction(
            player_id=player_id,
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            confidence=confidences,
            prediction_type="physics_social_lstm"
        )

    def _compute_puck_influence(
        self,
        player_pos: Tuple[float, float],
        puck_pos: Tuple[float, float],
        player_type: PlayerType
    ) -> Tuple[float, float]:
        """
        Compute how puck position influences player movement.

        Different player types react differently to puck.
        """
        dx = puck_pos[0] - player_pos[0]
        dy = puck_pos[1] - player_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 1:
            return (0.0, 0.0)

        # Normalize direction
        dx /= distance
        dy /= distance

        # Influence strength based on player type
        if player_type == PlayerType.PUCK_CARRIER:
            # Puck carrier moves with puck
            strength = 5.0
        elif player_type == PlayerType.FORWARD:
            # Forwards attracted to puck area
            strength = 3.0 * max(0, 1 - distance / 50)
        elif player_type == PlayerType.DEFENSEMAN:
            # Defensemen maintain position, less reactive
            strength = 1.5 * max(0, 1 - distance / 40)
        else:
            strength = 1.0

        return (dx * strength, dy * strength)

    def _empty_prediction(self, player_id: str) -> TrajectoryPrediction:
        """Create empty prediction for unknown player."""
        return TrajectoryPrediction(
            player_id=player_id,
            timestamps=[],
            positions=[],
            velocities=[],
            confidence=[],
            prediction_type="empty"
        )


class GhostingAnalyzer:
    """
    Ghosting Analysis for Defensive Evaluation.

    Simulates what optimal defenders would do and compares
    to actual positioning.

    Based on Lucey et al. (2013) "Ghosting" research.
    """

    def __init__(self, predictor: HockeyTrajectoryPredictor):
        """
        Initialize ghosting analyzer.

        Args:
            predictor: Trajectory prediction model
        """
        self.predictor = predictor

        # Optimal positioning weights
        self.weights = {
            'puck_distance': 0.3,
            'passing_lane': 0.25,
            'goal_coverage': 0.25,
            'support_positioning': 0.2
        }

    def compute_optimal_position(
        self,
        defender_id: str,
        puck_pos: Tuple[float, float],
        teammates: List[TrackingFrame],
        opponents: List[TrackingFrame],
        defensive_scheme: str = "zone"
    ) -> Tuple[float, float]:
        """
        Compute optimal position for a defender.

        Args:
            defender_id: Defender to analyze
            puck_pos: Current puck position
            teammates: Teammate tracking data
            opponents: Opponent tracking data
            defensive_scheme: Current defensive scheme

        Returns:
            Optimal (x, y) position
        """
        # Find defender's current position
        defender_frame = None
        for t in teammates:
            if t.player_id == defender_id:
                defender_frame = t
                break

        if defender_frame is None:
            return (0.0, 0.0)

        # Initialize with current position
        optimal_x = defender_frame.x
        optimal_y = defender_frame.y

        # Puck coverage component
        puck_direction = (
            puck_pos[0] - defender_frame.x,
            puck_pos[1] - defender_frame.y
        )
        puck_dist = np.sqrt(puck_direction[0]**2 + puck_direction[1]**2)

        if puck_dist > 0:
            # Move toward puck but maintain gap
            target_gap = 15.0 if defensive_scheme == "zone" else 8.0
            if puck_dist > target_gap:
                move_factor = min(0.3, (puck_dist - target_gap) / puck_dist)
                optimal_x += puck_direction[0] * move_factor
                optimal_y += puck_direction[1] * move_factor

        # Passing lane coverage
        for opp in opponents:
            if opp.has_puck:
                continue

            # Vector from puck to opponent
            to_opp = (opp.x - puck_pos[0], opp.y - puck_pos[1])
            dist_to_opp = np.sqrt(to_opp[0]**2 + to_opp[1]**2)

            if dist_to_opp > 5:
                # Position to intercept pass
                intercept_point = (
                    puck_pos[0] + to_opp[0] * 0.5,
                    puck_pos[1] + to_opp[1] * 0.5
                )

                # Blend toward intercept point
                optimal_x = 0.8 * optimal_x + 0.2 * intercept_point[0]
                optimal_y = 0.8 * optimal_y + 0.2 * intercept_point[1]

        # Goal coverage (stay between puck and goal)
        goal_x = -89.0  # Defensive goal
        goal_y = 0.0

        if puck_pos[0] < 0:  # Defensive zone
            # Position between puck and goal
            optimal_x = max(goal_x + 10, min(optimal_x, puck_pos[0] - 5))

        # Teammate support
        teammate_positions = [(t.x, t.y) for t in teammates if t.player_id != defender_id]
        if teammate_positions:
            avg_teammate_y = np.mean([p[1] for p in teammate_positions])
            # Maintain spacing from teammates
            if abs(optimal_y - avg_teammate_y) < 10:
                # Spread out
                if optimal_y > avg_teammate_y:
                    optimal_y += 5
                else:
                    optimal_y -= 5

        # Apply boundary constraints
        optimal_x = np.clip(optimal_x, -100, 100)
        optimal_y = np.clip(optimal_y, -42.5, 42.5)

        return (optimal_x, optimal_y)

    def evaluate_positioning(
        self,
        defender_id: str,
        actual_pos: Tuple[float, float],
        optimal_pos: Tuple[float, float],
        game_context: Dict[str, Any]
    ) -> GhostingResult:
        """
        Evaluate defender positioning quality.

        Args:
            defender_id: Defender being evaluated
            actual_pos: Actual position
            optimal_pos: Computed optimal position
            game_context: Current game context

        Returns:
            GhostingResult with evaluation
        """
        # Position error
        error = np.sqrt(
            (actual_pos[0] - optimal_pos[0])**2 +
            (actual_pos[1] - optimal_pos[1])**2
        )

        # Coverage impact (how much danger increased due to poor positioning)
        # Simplified - would use xG model in full implementation
        if error < 5:
            coverage_impact = 0.0
        elif error < 10:
            coverage_impact = 0.1
        elif error < 20:
            coverage_impact = 0.3
        else:
            coverage_impact = 0.5

        # Grade
        if error < 5:
            grade = "A"
        elif error < 10:
            grade = "B"
        elif error < 15:
            grade = "C"
        elif error < 25:
            grade = "D"
        else:
            grade = "F"

        return GhostingResult(
            player_id=defender_id,
            timestamp=game_context.get('timestamp', 0.0),
            actual_position=actual_pos,
            optimal_position=optimal_pos,
            position_error=error,
            coverage_impact=coverage_impact,
            grade=grade
        )


class GoalieTrajectoryModel:
    """
    Specialized trajectory model for goalie movement.

    Goalies move differently than skaters - they track the puck
    and maintain crease position.
    """

    # Goalie movement constraints
    MAX_LATERAL_SPEED = 15.0  # ft/s
    MAX_DEPTH_SPEED = 10.0  # ft/s
    CREASE_DEPTH = 6.0  # ft from goal line

    def __init__(self):
        """Initialize goalie model."""
        self.position_history: deque = deque(maxlen=30)

    def predict_goalie_position(
        self,
        current_pos: Tuple[float, float],
        puck_pos: Tuple[float, float],
        puck_velocity: Tuple[float, float],
        shooter_pos: Optional[Tuple[float, float]] = None,
        horizon: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Predict goalie position based on puck movement.

        Args:
            current_pos: Current goalie position
            puck_pos: Current puck position
            puck_velocity: Puck velocity
            shooter_pos: Position of potential shooter
            horizon: Prediction horizon in seconds

        Returns:
            List of predicted positions
        """
        # Goal center (assuming defending left goal at x=-89)
        goal_x = -89.0
        goal_y = 0.0

        predictions = []
        dt = 1 / 30

        x, y = current_pos
        n_frames = int(horizon * 30)

        for i in range(n_frames):
            # Predict puck position
            future_puck_x = puck_pos[0] + puck_velocity[0] * (i + 1) * dt
            future_puck_y = puck_pos[1] + puck_velocity[1] * (i + 1) * dt

            # Angle from goal to puck
            angle = np.arctan2(
                future_puck_y - goal_y,
                future_puck_x - goal_x
            )

            # Optimal depth based on puck distance
            puck_dist = np.sqrt(
                (future_puck_x - goal_x)**2 +
                (future_puck_y - goal_y)**2
            )

            if puck_dist < 30:
                # Close shot - stay deep
                target_depth = 2.0
            elif puck_dist < 50:
                # Medium range - challenge slightly
                target_depth = 4.0
            else:
                # Far - challenge more
                target_depth = self.CREASE_DEPTH

            # Target position
            target_x = goal_x + target_depth * np.cos(angle)
            target_y = goal_y + target_depth * np.sin(angle)

            # Constrain to crease area
            target_y = np.clip(target_y, -4.0, 4.0)

            # Move toward target with speed constraints
            dx = target_x - x
            dy = target_y - y

            # Apply speed limits
            lateral_move = np.clip(dy, -self.MAX_LATERAL_SPEED * dt, self.MAX_LATERAL_SPEED * dt)
            depth_move = np.clip(dx, -self.MAX_DEPTH_SPEED * dt, self.MAX_DEPTH_SPEED * dt)

            x += depth_move
            y += lateral_move

            predictions.append((x, y))

        return predictions

    def evaluate_goalie_positioning(
        self,
        goalie_pos: Tuple[float, float],
        puck_pos: Tuple[float, float],
        shot_location: Optional[Tuple[float, float]] = None
    ) -> Dict[str, float]:
        """
        Evaluate goalie positioning quality.

        Returns:
            Dictionary with positioning metrics
        """
        goal_x = -89.0
        goal_y = 0.0

        # Angle coverage
        angle_to_puck = np.arctan2(
            puck_pos[1] - goal_y,
            puck_pos[0] - goal_x
        )
        goalie_angle = np.arctan2(
            goalie_pos[1] - goal_y,
            goalie_pos[0] - goal_x
        )

        angle_error = abs(angle_to_puck - goalie_angle)

        # Depth appropriateness
        puck_dist = np.sqrt(
            (puck_pos[0] - goal_x)**2 +
            (puck_pos[1] - goal_y)**2
        )
        goalie_depth = goalie_pos[0] - goal_x

        if puck_dist < 30:
            optimal_depth = 2.0
        elif puck_dist < 50:
            optimal_depth = 4.0
        else:
            optimal_depth = 6.0

        depth_error = abs(goalie_depth - optimal_depth)

        # Overall score
        angle_score = max(0, 1 - angle_error / 0.5)  # Radians tolerance
        depth_score = max(0, 1 - depth_error / 3)  # Feet tolerance

        return {
            'angle_coverage': angle_score,
            'depth_score': depth_score,
            'overall_positioning': (angle_score + depth_score) / 2,
            'angle_error_rad': angle_error,
            'depth_error_ft': depth_error
        }


class TrajectoryEvaluator:
    """
    Evaluate trajectory prediction accuracy and player movement quality.
    """

    def __init__(self):
        """Initialize evaluator."""
        self.predictions: List[Dict] = []
        self.actuals: List[Dict] = []

    def add_prediction(
        self,
        prediction: TrajectoryPrediction,
        actual_positions: List[Tuple[float, float]]
    ):
        """
        Add prediction-actual pair for evaluation.

        Args:
            prediction: Predicted trajectory
            actual_positions: Actual positions that occurred
        """
        self.predictions.append({
            'player_id': prediction.player_id,
            'positions': prediction.positions,
            'confidence': prediction.confidence
        })
        self.actuals.append({
            'player_id': prediction.player_id,
            'positions': actual_positions
        })

    def compute_ade(self) -> float:
        """
        Compute Average Displacement Error.

        Standard metric for trajectory prediction.
        """
        total_error = 0.0
        n_points = 0

        for pred, actual in zip(self.predictions, self.actuals):
            min_len = min(len(pred['positions']), len(actual['positions']))

            for i in range(min_len):
                px, py = pred['positions'][i]
                ax, ay = actual['positions'][i]
                error = np.sqrt((px - ax)**2 + (py - ay)**2)
                total_error += error
                n_points += 1

        return total_error / n_points if n_points > 0 else 0.0

    def compute_fde(self) -> float:
        """
        Compute Final Displacement Error.

        Error at the end of prediction horizon.
        """
        total_error = 0.0
        n_predictions = 0

        for pred, actual in zip(self.predictions, self.actuals):
            if pred['positions'] and actual['positions']:
                px, py = pred['positions'][-1]
                ax, ay = actual['positions'][-1]
                error = np.sqrt((px - ax)**2 + (py - ay)**2)
                total_error += error
                n_predictions += 1

        return total_error / n_predictions if n_predictions > 0 else 0.0

    def compute_metrics_by_horizon(self) -> Dict[str, List[float]]:
        """
        Compute error metrics at different prediction horizons.

        Returns:
            Dictionary with errors at each timestep
        """
        # Find max prediction length
        max_len = max(
            len(p['positions']) for p in self.predictions
        ) if self.predictions else 0

        errors_by_step = [[] for _ in range(max_len)]

        for pred, actual in zip(self.predictions, self.actuals):
            min_len = min(len(pred['positions']), len(actual['positions']))

            for i in range(min_len):
                px, py = pred['positions'][i]
                ax, ay = actual['positions'][i]
                error = np.sqrt((px - ax)**2 + (py - ay)**2)
                errors_by_step[i].append(error)

        return {
            'mean_error': [np.mean(e) if e else 0 for e in errors_by_step],
            'std_error': [np.std(e) if e else 0 for e in errors_by_step],
            'n_samples': [len(e) for e in errors_by_step]
        }
