"""
Defender Trajectory CNN-LSTM Model

Implements methodology from:
- Amazon Science. "Prediction of Defensive Player Trajectories in NFL Games
  with Defender CNN-LSTM Model."

Key insight: Individual trajectories are affected by personal assignments,
overall defensive strategy, and surrounding player movements.

Architecture:
- 1D convolutions extract features from multiple players
- LSTM captures temporal dependencies
- Social pooling accounts for nearby player interactions

Translated to hockey for:
- Defenseman reaction prediction
- Backcheck trajectory modeling
- Goalie positioning based on shooter location
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Sequence
import numpy as np
from datetime import datetime


class DefenderRole(Enum):
    """Defensive roles in hockey."""
    LEFT_DEFENSE = "left_defense"
    RIGHT_DEFENSE = "right_defense"
    CENTER = "center"
    LEFT_WING = "left_wing"
    RIGHT_WING = "right_wing"
    GOALIE = "goalie"


class AssignmentType(Enum):
    """Type of defensive assignment."""
    MAN_MARKING = "man_marking"
    ZONE = "zone"
    PUCK_CARRIER = "puck_carrier"
    CREASE = "crease"
    HIGH_SLOT = "high_slot"
    WEAK_SIDE = "weak_side"


@dataclass
class PlayerFrame:
    """Single frame of player position data."""
    player_id: str
    x: float  # meters
    y: float  # meters
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0  # acceleration
    ay: float = 0.0
    orientation: float = 0.0  # facing angle in radians
    role: DefenderRole = DefenderRole.CENTER

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)


@dataclass
class GameFrame:
    """Complete frame with all players and puck."""
    timestamp: float
    home_players: List[PlayerFrame]
    away_players: List[PlayerFrame]
    puck_x: float
    puck_y: float
    puck_vx: float = 0.0
    puck_vy: float = 0.0
    home_has_possession: bool = True


@dataclass
class TrajectoryPrediction:
    """Predicted trajectory for a defender."""
    player_id: str
    predicted_positions: np.ndarray  # shape (n_steps, 2)
    confidence: np.ndarray  # shape (n_steps,)
    predicted_assignment: AssignmentType
    assignment_confidence: float


@dataclass
class SocialPoolingConfig:
    """Configuration for social pooling layer."""
    grid_size: int = 4  # 4x4 pooling grid
    neighborhood_radius: float = 5.0  # meters
    hidden_dim: int = 64


@dataclass
class CNNConfig:
    """Configuration for 1D CNN encoder."""
    input_channels: int = 6  # x, y, vx, vy, ax, ay
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])


@dataclass
class LSTMConfig:
    """Configuration for LSTM decoder."""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class ModelConfig:
    """Full model configuration."""
    cnn: CNNConfig = field(default_factory=CNNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    social: SocialPoolingConfig = field(default_factory=SocialPoolingConfig)
    sequence_length: int = 20  # Input sequence frames
    prediction_horizon: int = 10  # Output prediction frames
    frame_rate: float = 25.0  # Frames per second


class SocialPoolingLayer:
    """
    Social pooling to capture player interactions.

    Pools hidden states of nearby players into a spatial grid
    to capture social forces and defensive assignments.
    """

    def __init__(self, config: SocialPoolingConfig):
        self.config = config
        self.grid_size = config.grid_size
        self.radius = config.neighborhood_radius

    def get_grid_cell(
        self,
        other_pos: np.ndarray,
        player_pos: np.ndarray
    ) -> Tuple[int, int]:
        """Get grid cell for relative position."""
        rel_pos = other_pos - player_pos
        rel_pos = np.clip(rel_pos, -self.radius, self.radius)

        # Map to grid indices
        i = int((rel_pos[0] + self.radius) / (2 * self.radius) * self.grid_size)
        j = int((rel_pos[1] + self.radius) / (2 * self.radius) * self.grid_size)

        i = min(max(i, 0), self.grid_size - 1)
        j = min(max(j, 0), self.grid_size - 1)

        return i, j

    def pool_neighbors(
        self,
        player: PlayerFrame,
        all_players: List[PlayerFrame],
        hidden_states: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Pool hidden states of nearby players.

        Returns spatial grid of pooled features.
        """
        grid = np.zeros((self.grid_size, self.grid_size, self.config.hidden_dim))
        counts = np.zeros((self.grid_size, self.grid_size))

        player_pos = player.position

        for other in all_players:
            if other.player_id == player.player_id:
                continue

            other_pos = other.position
            dist = np.linalg.norm(other_pos - player_pos)

            if dist > self.radius:
                continue

            if other.player_id not in hidden_states:
                continue

            i, j = self.get_grid_cell(other_pos, player_pos)
            grid[i, j] += hidden_states[other.player_id]
            counts[i, j] += 1

        # Average pooling
        mask = counts > 0
        grid[mask] = grid[mask] / counts[mask, np.newaxis]

        return grid.flatten()


class CNNEncoder:
    """
    1D CNN encoder for player trajectory features.

    Extracts temporal features from position/velocity sequences.
    """

    def __init__(self, config: CNNConfig):
        self.config = config
        self.weights = self._initialize_weights()

    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize CNN weights."""
        weights = []
        in_ch = self.config.input_channels

        for out_ch, kernel in zip(
            self.config.hidden_channels,
            self.config.kernel_sizes
        ):
            # Xavier initialization
            scale = np.sqrt(2.0 / (in_ch * kernel))
            w = np.random.randn(out_ch, in_ch, kernel) * scale
            weights.append(w)
            in_ch = out_ch

        return weights

    def conv1d(
        self,
        x: np.ndarray,
        w: np.ndarray
    ) -> np.ndarray:
        """Apply 1D convolution with ReLU."""
        out_ch, in_ch, kernel = w.shape
        seq_len = x.shape[0]

        # Pad sequence
        pad = kernel // 2
        x_padded = np.pad(x, ((pad, pad), (0, 0)), mode='constant')

        # Convolve
        output = np.zeros((seq_len, out_ch))
        for t in range(seq_len):
            for c_out in range(out_ch):
                for c_in in range(in_ch):
                    output[t, c_out] += np.sum(
                        x_padded[t:t+kernel, c_in] * w[c_out, c_in, :]
                    )

        # ReLU activation
        return np.maximum(output, 0)

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        """
        Encode trajectory sequence.

        Args:
            sequence: shape (seq_len, input_channels)

        Returns:
            features: shape (hidden_channels[-1],)
        """
        x = sequence
        for w in self.weights:
            x = self.conv1d(x, w)

        # Global average pooling over time
        return np.mean(x, axis=0)


class LSTMDecoder:
    """
    LSTM decoder for trajectory prediction.

    Generates future positions autoregressively.
    """

    def __init__(self, config: LSTMConfig, input_dim: int):
        self.config = config
        self.input_dim = input_dim
        self.hidden_size = config.hidden_size

        # Initialize weights (simplified single layer)
        self.Wf = np.random.randn(input_dim + config.hidden_size, config.hidden_size) * 0.1
        self.Wi = np.random.randn(input_dim + config.hidden_size, config.hidden_size) * 0.1
        self.Wc = np.random.randn(input_dim + config.hidden_size, config.hidden_size) * 0.1
        self.Wo = np.random.randn(input_dim + config.hidden_size, config.hidden_size) * 0.1

        self.bf = np.zeros(config.hidden_size)
        self.bi = np.zeros(config.hidden_size)
        self.bc = np.zeros(config.hidden_size)
        self.bo = np.zeros(config.hidden_size)

        # Output projection
        self.Wout = np.random.randn(config.hidden_size, 2) * 0.1  # predict (x, y)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def step(
        self,
        x: np.ndarray,
        h: np.ndarray,
        c: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single LSTM step."""
        combined = np.concatenate([x, h])

        f = self.sigmoid(combined @ self.Wf + self.bf)
        i = self.sigmoid(combined @ self.Wi + self.bi)
        c_tilde = np.tanh(combined @ self.Wc + self.bc)
        o = self.sigmoid(combined @ self.Wo + self.bo)

        c_new = f * c + i * c_tilde
        h_new = o * np.tanh(c_new)

        # Output position
        out = h_new @ self.Wout

        return out, h_new, c_new

    def decode(
        self,
        initial_features: np.ndarray,
        n_steps: int,
        initial_position: np.ndarray
    ) -> np.ndarray:
        """
        Decode trajectory autoregressively.

        Args:
            initial_features: Encoded features
            n_steps: Number of future steps
            initial_position: Current position

        Returns:
            predictions: shape (n_steps, 2)
        """
        h = initial_features[:self.hidden_size]
        c = np.zeros(self.hidden_size)

        predictions = []
        current_pos = initial_position.copy()

        for _ in range(n_steps):
            # Input is current position + features
            x = np.concatenate([current_pos, initial_features])
            delta, h, c = self.step(x, h, c)

            # Predict position change
            current_pos = current_pos + delta
            predictions.append(current_pos.copy())

        return np.array(predictions)


class DefenderTrajectoryModel:
    """
    Full defender trajectory prediction model.

    Combines CNN encoder, social pooling, and LSTM decoder.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()

        self.cnn = CNNEncoder(self.config.cnn)
        self.social = SocialPoolingLayer(self.config.social)

        # Feature dimension after CNN + social pooling
        cnn_dim = self.config.cnn.hidden_channels[-1]
        social_dim = self.config.social.grid_size ** 2 * self.config.social.hidden_dim
        combined_dim = cnn_dim + social_dim

        self.lstm = LSTMDecoder(self.config.lstm, combined_dim + 2)

    def prepare_sequence(
        self,
        player_frames: List[PlayerFrame]
    ) -> np.ndarray:
        """Convert player frames to input sequence."""
        seq = np.array([
            [f.x, f.y, f.vx, f.vy, f.ax, f.ay]
            for f in player_frames
        ])
        return seq

    def predict_trajectory(
        self,
        player: PlayerFrame,
        history: List[GameFrame],
        current_frame: GameFrame
    ) -> TrajectoryPrediction:
        """
        Predict future trajectory for a defender.

        Args:
            player: Current player state
            history: Recent game frames
            current_frame: Current game state

        Returns:
            Trajectory prediction with confidence
        """
        # Extract player history
        player_history = []
        for frame in history:
            for p in frame.home_players + frame.away_players:
                if p.player_id == player.player_id:
                    player_history.append(p)
                    break

        if len(player_history) < self.config.sequence_length:
            # Pad with current position
            padding = [player] * (self.config.sequence_length - len(player_history))
            player_history = padding + player_history

        player_history = player_history[-self.config.sequence_length:]

        # Encode trajectory
        seq = self.prepare_sequence(player_history)
        cnn_features = self.cnn.encode(seq)

        # Social pooling
        all_players = current_frame.home_players + current_frame.away_players
        hidden_states = {
            p.player_id: np.random.randn(self.config.social.hidden_dim)
            for p in all_players
        }  # In practice, these would be computed
        social_features = self.social.pool_neighbors(
            player, all_players, hidden_states
        )

        # Combine features
        combined = np.concatenate([cnn_features, social_features])

        # Predict trajectory
        predictions = self.lstm.decode(
            combined,
            self.config.prediction_horizon,
            player.position
        )

        # Estimate confidence (decreases over time)
        confidence = np.exp(-np.arange(self.config.prediction_horizon) * 0.1)

        # Predict assignment based on trajectory
        assignment = self._infer_assignment(player, predictions, current_frame)

        return TrajectoryPrediction(
            player_id=player.player_id,
            predicted_positions=predictions,
            confidence=confidence,
            predicted_assignment=assignment[0],
            assignment_confidence=assignment[1]
        )

    def _infer_assignment(
        self,
        player: PlayerFrame,
        predictions: np.ndarray,
        frame: GameFrame
    ) -> Tuple[AssignmentType, float]:
        """Infer defensive assignment from predicted trajectory."""
        final_pos = predictions[-1]
        puck_pos = np.array([frame.puck_x, frame.puck_y])

        # Distance to puck
        dist_to_puck = np.linalg.norm(final_pos - puck_pos)

        # Distance to crease (assuming goal at x=30)
        goal_pos = np.array([30.0, 0.0])
        dist_to_crease = np.linalg.norm(final_pos - goal_pos)

        # Check man marking
        if frame.home_has_possession:
            opponents = frame.away_players
        else:
            opponents = frame.home_players

        min_dist_to_opponent = float('inf')
        for opp in opponents:
            dist = np.linalg.norm(final_pos - opp.position)
            min_dist_to_opponent = min(min_dist_to_opponent, dist)

        # Assignment logic
        if dist_to_puck < 3.0:
            return AssignmentType.PUCK_CARRIER, 0.9
        elif dist_to_crease < 5.0:
            return AssignmentType.CREASE, 0.85
        elif min_dist_to_opponent < 2.0:
            return AssignmentType.MAN_MARKING, 0.8
        elif abs(final_pos[1]) > 8.0:
            return AssignmentType.WEAK_SIDE, 0.7
        else:
            return AssignmentType.ZONE, 0.6


class BackcheckPredictor:
    """
    Specialized model for backcheck trajectory prediction.

    Predicts how forwards will track back on defensive transitions.
    """

    def __init__(self, base_model: DefenderTrajectoryModel):
        self.base_model = base_model

    def predict_backcheck(
        self,
        forward: PlayerFrame,
        history: List[GameFrame],
        current_frame: GameFrame,
        target_assignment: Optional[PlayerFrame] = None
    ) -> TrajectoryPrediction:
        """
        Predict backcheck trajectory.

        May include target assignment for man-marking situations.
        """
        # Base prediction
        pred = self.base_model.predict_trajectory(forward, history, current_frame)

        if target_assignment is not None:
            # Adjust prediction toward assignment
            target_pos = target_assignment.position

            # Blend predicted positions toward target
            blend_factors = np.linspace(0, 0.3, len(pred.predicted_positions))
            for i, blend in enumerate(blend_factors):
                pred.predicted_positions[i] = (
                    (1 - blend) * pred.predicted_positions[i] +
                    blend * target_pos
                )

            pred.predicted_assignment = AssignmentType.MAN_MARKING
            pred.assignment_confidence = 0.85

        return pred

    def backcheck_quality(
        self,
        actual_positions: np.ndarray,
        optimal_positions: np.ndarray
    ) -> float:
        """
        Evaluate backcheck quality.

        Compares actual trajectory to optimal path.
        """
        errors = np.linalg.norm(actual_positions - optimal_positions, axis=1)
        mean_error = np.mean(errors)

        # Quality score (higher is better)
        quality = np.exp(-mean_error / 5.0)
        return quality


class GoaliePositionPredictor:
    """
    Predict goalie positioning based on game state.

    Accounts for shooter location, screen, and passing threats.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()

        # Goalie physics
        self.reaction_time = 0.15  # seconds
        self.max_shuffle_speed = 4.0  # m/s
        self.butterfly_time = 0.3  # seconds

    def optimal_position(
        self,
        shooter_pos: np.ndarray,
        goal_pos: np.ndarray = np.array([30.0, 0.0]),
        goal_width: float = 1.83  # 6 feet
    ) -> np.ndarray:
        """
        Calculate optimal goalie position for shooter angle.

        Uses angle bisector method.
        """
        # Vector from shooter to goal posts
        left_post = goal_pos + np.array([0, goal_width/2])
        right_post = goal_pos + np.array([0, -goal_width/2])

        # Angle bisector from shooter
        to_left = left_post - shooter_pos
        to_right = right_post - shooter_pos

        to_left_norm = to_left / np.linalg.norm(to_left)
        to_right_norm = to_right / np.linalg.norm(to_right)

        bisector = (to_left_norm + to_right_norm) / 2
        bisector = bisector / np.linalg.norm(bisector)

        # Position along bisector (depth based on distance)
        shooter_dist = np.linalg.norm(shooter_pos - goal_pos)
        depth = min(2.0, shooter_dist * 0.1)  # Closer shooters = tighter

        optimal = goal_pos - bisector * depth

        return optimal

    def predict_position(
        self,
        current_pos: np.ndarray,
        puck_pos: np.ndarray,
        pass_threats: List[np.ndarray],
        time_horizon: float = 0.5
    ) -> Tuple[np.ndarray, float]:
        """
        Predict goalie position accounting for pass threats.

        Returns position and save probability.
        """
        # Optimal for current shooter
        optimal = self.optimal_position(puck_pos)

        # Weight pass threats
        if pass_threats:
            threat_positions = [self.optimal_position(t) for t in pass_threats]

            # Weight by pass probability (simplified)
            weights = [1.0 / (np.linalg.norm(t - puck_pos) + 1) for t in pass_threats]
            total_weight = 1.0 + sum(weights)

            weighted_optimal = optimal / total_weight
            for pos, w in zip(threat_positions, weights):
                weighted_optimal += pos * w / total_weight

            optimal = weighted_optimal

        # Can goalie reach optimal in time?
        dist = np.linalg.norm(optimal - current_pos)
        time_to_optimal = dist / self.max_shuffle_speed + self.reaction_time

        if time_to_optimal <= time_horizon:
            predicted = optimal
            save_prob = 0.92
        else:
            # Partial movement toward optimal
            direction = (optimal - current_pos) / max(dist, 0.001)
            move_dist = self.max_shuffle_speed * max(0, time_horizon - self.reaction_time)
            predicted = current_pos + direction * move_dist
            save_prob = 0.75 * (1 - (time_to_optimal - time_horizon) / time_to_optimal)

        return predicted, save_prob


class TrajectoryEvaluator:
    """
    Evaluate predicted vs actual trajectories.

    Computes metrics for model validation.
    """

    def __init__(self):
        pass

    def average_displacement_error(
        self,
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """Average L2 error across trajectory."""
        errors = np.linalg.norm(predicted - actual, axis=1)
        return np.mean(errors)

    def final_displacement_error(
        self,
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """Error at final predicted position."""
        return np.linalg.norm(predicted[-1] - actual[-1])

    def assignment_accuracy(
        self,
        predicted_assignments: List[AssignmentType],
        actual_assignments: List[AssignmentType]
    ) -> float:
        """Classification accuracy for defensive assignments."""
        correct = sum(p == a for p, a in zip(predicted_assignments, actual_assignments))
        return correct / len(predicted_assignments) if predicted_assignments else 0.0

    def full_evaluation(
        self,
        predictions: List[TrajectoryPrediction],
        actuals: List[np.ndarray],
        actual_assignments: Optional[List[AssignmentType]] = None
    ) -> Dict[str, float]:
        """Full evaluation report."""
        ade = np.mean([
            self.average_displacement_error(p.predicted_positions, a)
            for p, a in zip(predictions, actuals)
        ])

        fde = np.mean([
            self.final_displacement_error(p.predicted_positions, a)
            for p, a in zip(predictions, actuals)
        ])

        results = {
            'average_displacement_error': ade,
            'final_displacement_error': fde,
        }

        if actual_assignments:
            pred_assign = [p.predicted_assignment for p in predictions]
            results['assignment_accuracy'] = self.assignment_accuracy(
                pred_assign, actual_assignments
            )

        return results
