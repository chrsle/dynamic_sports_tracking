"""
Deep Learning Feature Representations from Tracking Data

Implements learned feature representations from raw tracking data,
translated from Horton's football research.

Key Paper:
- Horton, M. (2020). "Learning Feature Representations from Football
  Tracking." 14th MIT Sloan Sports Analytics Conference.

Key Concepts:
- Deep learning on raw tracking data
- Learns meaningful features without hand-engineering
- Generalizes across different plays and situations
- Foundation for multiple downstream tasks

Hockey Translation:
- Learn representations from NHL EDGE tracking data
- Discover patterns not visible to human analysts
- Build foundation models for classification, prediction, evaluation
- Transfer learning across different analysis tasks
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class EncoderType(Enum):
    """Types of sequence encoders."""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CNN_1D = "cnn_1d"
    TEMPORAL_CONV = "temporal_conv"


class PoolingType(Enum):
    """Types of team/player pooling."""
    MEAN = "mean"
    MAX = "max"
    ATTENTION = "attention"
    SET_TRANSFORMER = "set_transformer"


@dataclass
class TrackingFrame:
    """Single frame of tracking data."""
    timestamp: float                # seconds
    player_positions: Dict[str, Tuple[float, float, float, float]]  # id -> (x, y, vx, vy)
    puck_position: Tuple[float, float, float, float]  # x, y, vx, vy
    puck_carrier: Optional[str] = None
    game_state: str = "5v5"


@dataclass
class TrackingSequence:
    """Sequence of tracking frames."""
    frames: List[TrackingFrame]
    sequence_id: str
    label: Optional[str] = None     # For supervised learning
    outcome: Optional[float] = None  # For value prediction

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp


@dataclass
class LearnedFeatures:
    """Learned feature representation."""
    sequence_id: str
    embedding: np.ndarray           # Dense vector representation
    player_embeddings: Dict[str, np.ndarray]
    attention_weights: Optional[np.ndarray] = None


@dataclass
class ModelConfig:
    """Configuration for deep tracking model."""
    input_dim: int = 4              # x, y, vx, vy per player
    hidden_dim: int = 128
    output_dim: int = 64            # Final embedding size
    num_layers: int = 2
    dropout: float = 0.1
    encoder_type: EncoderType = EncoderType.LSTM
    pooling_type: PoolingType = PoolingType.ATTENTION
    max_players: int = 12           # Max players per team
    sequence_length: int = 50       # Frames to process


class TrackingPreprocessor:
    """
    Preprocesses raw tracking data for deep learning.

    Handles:
    - Coordinate normalization
    - Velocity scaling
    - Missing data imputation
    - Player ordering consistency
    """

    RINK_LENGTH = 200.0
    RINK_WIDTH = 85.0
    MAX_SPEED = 40.0  # ft/s

    def __init__(self, config: ModelConfig):
        """Initialize preprocessor."""
        self.config = config

    def preprocess_frame(
        self,
        frame: TrackingFrame,
        team_a_players: List[str],
        team_b_players: List[str],
    ) -> np.ndarray:
        """
        Convert frame to fixed-size tensor.

        Output shape: (2 * max_players + 1, input_dim)
        - Team A players
        - Team B players
        - Puck
        """
        output = np.zeros((
            2 * self.config.max_players + 1,
            self.config.input_dim
        ))

        # Process Team A
        for i, player_id in enumerate(team_a_players[:self.config.max_players]):
            if player_id in frame.player_positions:
                x, y, vx, vy = frame.player_positions[player_id]
                output[i] = self._normalize_state(x, y, vx, vy)

        # Process Team B
        offset = self.config.max_players
        for i, player_id in enumerate(team_b_players[:self.config.max_players]):
            if player_id in frame.player_positions:
                x, y, vx, vy = frame.player_positions[player_id]
                output[offset + i] = self._normalize_state(x, y, vx, vy)

        # Process puck
        px, py, pvx, pvy = frame.puck_position
        output[-1] = self._normalize_state(px, py, pvx, pvy)

        return output

    def _normalize_state(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
    ) -> np.ndarray:
        """Normalize position and velocity."""
        return np.array([
            x / self.RINK_LENGTH,          # 0-1
            y / self.RINK_WIDTH,           # 0-1
            vx / self.MAX_SPEED,           # ~-1 to 1
            vy / self.MAX_SPEED,           # ~-1 to 1
        ])

    def preprocess_sequence(
        self,
        sequence: TrackingSequence,
        team_a_players: List[str],
        team_b_players: List[str],
    ) -> np.ndarray:
        """
        Convert sequence to tensor.

        Output shape: (seq_length, 2 * max_players + 1, input_dim)
        """
        # Pad or truncate to fixed length
        target_len = self.config.sequence_length

        frames = sequence.frames
        if len(frames) > target_len:
            # Subsample
            indices = np.linspace(0, len(frames) - 1, target_len, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < target_len:
            # Pad with last frame
            while len(frames) < target_len:
                frames.append(frames[-1] if frames else TrackingFrame(
                    timestamp=0.0,
                    player_positions={},
                    puck_position=(0, 0, 0, 0),
                ))

        # Process each frame
        output = np.stack([
            self.preprocess_frame(f, team_a_players, team_b_players)
            for f in frames
        ])

        return output


class DeepTrackingEncoder:
    """
    Deep learning encoder for tracking sequences.

    Learns to extract meaningful features from raw tracking data
    without hand-engineered features.
    """

    def __init__(self, config: ModelConfig):
        """Initialize encoder."""
        self.config = config
        self.preprocessor = TrackingPreprocessor(config)

        # Model weights (in practice, these would be learned)
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        np.random.seed(42)

        # Entity encoder (per-player/puck)
        entity_input = self.config.input_dim
        hidden = self.config.hidden_dim

        self.entity_encoder = {
            'W1': np.random.randn(entity_input, hidden) * 0.1,
            'b1': np.zeros(hidden),
            'W2': np.random.randn(hidden, hidden) * 0.1,
            'b2': np.zeros(hidden),
        }

        # Temporal encoder (LSTM-like)
        self.temporal_weights = {
            'Wf': np.random.randn(hidden * 2, hidden) * 0.1,
            'Wi': np.random.randn(hidden * 2, hidden) * 0.1,
            'Wc': np.random.randn(hidden * 2, hidden) * 0.1,
            'Wo': np.random.randn(hidden * 2, hidden) * 0.1,
        }

        # Attention weights
        self.attention_weights = {
            'Wq': np.random.randn(hidden, hidden // 4) * 0.1,
            'Wk': np.random.randn(hidden, hidden // 4) * 0.1,
            'Wv': np.random.randn(hidden, hidden) * 0.1,
        }

        # Output projection
        self.output_proj = {
            'W': np.random.randn(hidden, self.config.output_dim) * 0.1,
            'b': np.zeros(self.config.output_dim),
        }

    def encode_entity(self, entity_state: np.ndarray) -> np.ndarray:
        """Encode single entity state to hidden representation."""
        # Two-layer MLP with ReLU
        h1 = np.maximum(0, entity_state @ self.entity_encoder['W1'] + self.entity_encoder['b1'])
        h2 = np.maximum(0, h1 @ self.entity_encoder['W2'] + self.entity_encoder['b2'])
        return h2

    def encode_frame(self, frame_tensor: np.ndarray) -> np.ndarray:
        """
        Encode all entities in a frame.

        Args:
            frame_tensor: (num_entities, input_dim)

        Returns:
            (num_entities, hidden_dim)
        """
        encoded = np.stack([
            self.encode_entity(frame_tensor[i])
            for i in range(frame_tensor.shape[0])
        ])
        return encoded

    def temporal_encode(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply temporal encoding across sequence.

        Simplified LSTM-like processing.

        Args:
            sequence: (seq_len, num_entities, hidden_dim)

        Returns:
            (num_entities, hidden_dim) - final hidden state
        """
        seq_len, num_entities, hidden = sequence.shape

        # Initialize hidden state
        h = np.zeros((num_entities, hidden))
        c = np.zeros((num_entities, hidden))

        for t in range(seq_len):
            x = sequence[t]  # (num_entities, hidden)

            # Concatenate input and hidden
            combined = np.concatenate([x, h], axis=-1)

            # LSTM gates (simplified)
            f = self._sigmoid(combined @ self.temporal_weights['Wf'])
            i = self._sigmoid(combined @ self.temporal_weights['Wi'])
            c_tilde = np.tanh(combined @ self.temporal_weights['Wc'])
            o = self._sigmoid(combined @ self.temporal_weights['Wo'])

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

        return h

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def attention_pool(
        self,
        entity_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pool entity embeddings using attention.

        Args:
            entity_embeddings: (num_entities, hidden_dim)

        Returns:
            pooled: (hidden_dim,) - single vector representation
            weights: (num_entities,) - attention weights
        """
        # Compute queries, keys, values
        Q = entity_embeddings @ self.attention_weights['Wq']
        K = entity_embeddings @ self.attention_weights['Wk']
        V = entity_embeddings @ self.attention_weights['Wv']

        # Self-attention scores
        scores = Q @ K.T / np.sqrt(Q.shape[-1])
        weights = self._softmax(scores.mean(axis=0))  # Average over queries

        # Weighted sum
        pooled = weights @ V

        return pooled, weights

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def encode_sequence(
        self,
        sequence: TrackingSequence,
        team_a_players: List[str],
        team_b_players: List[str],
    ) -> LearnedFeatures:
        """
        Encode full tracking sequence to learned features.

        Args:
            sequence: Raw tracking sequence
            team_a_players: List of team A player IDs
            team_b_players: List of team B player IDs

        Returns:
            LearnedFeatures with embeddings
        """
        # Preprocess
        tensor = self.preprocessor.preprocess_sequence(
            sequence, team_a_players, team_b_players
        )  # (seq_len, num_entities, input_dim)

        # Encode each frame
        encoded_frames = np.stack([
            self.encode_frame(tensor[t])
            for t in range(tensor.shape[0])
        ])  # (seq_len, num_entities, hidden_dim)

        # Temporal encoding
        final_hidden = self.temporal_encode(encoded_frames)

        # Attention pooling
        pooled, attention_weights = self.attention_pool(final_hidden)

        # Output projection
        embedding = pooled @ self.output_proj['W'] + self.output_proj['b']

        # Extract player-level embeddings
        all_players = team_a_players + team_b_players
        player_embeddings = {}
        for i, player_id in enumerate(all_players[:2 * self.config.max_players]):
            if i < final_hidden.shape[0]:
                player_emb = final_hidden[i] @ self.output_proj['W'] + self.output_proj['b']
                player_embeddings[player_id] = player_emb

        return LearnedFeatures(
            sequence_id=sequence.sequence_id,
            embedding=embedding,
            player_embeddings=player_embeddings,
            attention_weights=attention_weights,
        )


class FeatureExtractor:
    """
    Extracts interpretable features from learned representations.

    Bridges deep learned features to interpretable analytics.
    """

    def __init__(self, encoder: DeepTrackingEncoder):
        """Initialize with trained encoder."""
        self.encoder = encoder

    def extract_play_features(
        self,
        features: LearnedFeatures,
    ) -> Dict[str, float]:
        """
        Extract interpretable play-level features.

        Uses the learned embedding to compute high-level metrics.
        """
        embedding = features.embedding

        # These would be learned mappings in practice
        # Here we use simple projections as placeholders

        # Offensive threat (higher = more dangerous)
        offensive_threat = float(np.tanh(embedding[:16].mean()))

        # Possession quality
        possession_quality = float(self._sigmoid_scalar(embedding[16:32].mean()))

        # Transition speed (based on velocity features)
        transition_speed = float(np.abs(embedding[32:48]).mean())

        # Team structure (based on positional features)
        structure_score = float(1.0 - np.var(embedding[48:]))

        return {
            'offensive_threat': offensive_threat,
            'possession_quality': possession_quality,
            'transition_speed': transition_speed,
            'structure_score': structure_score,
        }

    def _sigmoid_scalar(self, x: float) -> float:
        """Sigmoid for single value."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def extract_player_features(
        self,
        features: LearnedFeatures,
        player_id: str,
    ) -> Dict[str, float]:
        """
        Extract player-specific features from embeddings.
        """
        if player_id not in features.player_embeddings:
            return {}

        emb = features.player_embeddings[player_id]

        # Attention weight (how important was this player)
        attention = 0.0
        if features.attention_weights is not None:
            # Find player index
            all_players = list(features.player_embeddings.keys())
            if player_id in all_players:
                idx = all_players.index(player_id)
                if idx < len(features.attention_weights):
                    attention = float(features.attention_weights[idx])

        return {
            'involvement': attention,
            'activity_level': float(np.abs(emb).mean()),
            'unique_contribution': float(np.var(emb)),
        }

    def compare_sequences(
        self,
        features1: LearnedFeatures,
        features2: LearnedFeatures,
    ) -> float:
        """
        Compute similarity between two sequence embeddings.

        Uses cosine similarity.
        """
        emb1 = features1.embedding
        emb2 = features2.embedding

        # Cosine similarity
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        return float(dot / (norm1 * norm2))


class PlayClassifier:
    """
    Classifies plays using learned features.

    Can be trained to recognize play types, outcomes, etc.
    """

    def __init__(
        self,
        encoder: DeepTrackingEncoder,
        num_classes: int = 10,
    ):
        """Initialize classifier."""
        self.encoder = encoder
        self.num_classes = num_classes

        # Classification head weights
        np.random.seed(43)
        self.classifier_weights = {
            'W': np.random.randn(encoder.config.output_dim, num_classes) * 0.1,
            'b': np.zeros(num_classes),
        }

        self.class_names: Dict[int, str] = {}

    def set_class_names(self, names: Dict[int, str]):
        """Set human-readable class names."""
        self.class_names = names

    def classify(
        self,
        sequence: TrackingSequence,
        team_a_players: List[str],
        team_b_players: List[str],
    ) -> Dict[str, Any]:
        """
        Classify a tracking sequence.

        Returns predicted class and probabilities.
        """
        features = self.encoder.encode_sequence(
            sequence, team_a_players, team_b_players
        )

        # Classification logits
        logits = features.embedding @ self.classifier_weights['W'] + self.classifier_weights['b']

        # Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Predicted class
        pred_idx = int(np.argmax(probs))
        pred_name = self.class_names.get(pred_idx, f"class_{pred_idx}")

        # Top-k predictions
        top_k = 3
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_predictions = [
            {
                'class': self.class_names.get(int(i), f"class_{i}"),
                'probability': float(probs[i]),
            }
            for i in top_indices
        ]

        return {
            'predicted_class': pred_name,
            'confidence': float(probs[pred_idx]),
            'top_predictions': top_predictions,
            'embedding': features.embedding,
        }


class SimilaritySearch:
    """
    Finds similar plays using learned embeddings.

    Enables "query by example" play retrieval.
    """

    def __init__(self, encoder: DeepTrackingEncoder):
        """Initialize search."""
        self.encoder = encoder
        self.index: List[Tuple[str, np.ndarray]] = []

    def add_to_index(
        self,
        sequence: TrackingSequence,
        team_a_players: List[str],
        team_b_players: List[str],
    ):
        """Add sequence to search index."""
        features = self.encoder.encode_sequence(
            sequence, team_a_players, team_b_players
        )
        self.index.append((sequence.sequence_id, features.embedding))

    def search(
        self,
        query: TrackingSequence,
        team_a_players: List[str],
        team_b_players: List[str],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find k most similar sequences to query.
        """
        if not self.index:
            return []

        # Encode query
        query_features = self.encoder.encode_sequence(
            query, team_a_players, team_b_players
        )
        query_emb = query_features.embedding

        # Compute similarities
        similarities = []
        for seq_id, emb in self.index:
            if seq_id == query.sequence_id:
                continue

            # Cosine similarity
            dot = np.dot(query_emb, emb)
            norm_q = np.linalg.norm(query_emb)
            norm_e = np.linalg.norm(emb)

            if norm_q > 1e-6 and norm_e > 1e-6:
                sim = dot / (norm_q * norm_e)
            else:
                sim = 0.0

            similarities.append({
                'sequence_id': seq_id,
                'similarity': float(sim),
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:k]

    def build_index_from_sequences(
        self,
        sequences: List[TrackingSequence],
        team_a_players: List[str],
        team_b_players: List[str],
    ):
        """Build search index from list of sequences."""
        for seq in sequences:
            self.add_to_index(seq, team_a_players, team_b_players)


class TransferLearner:
    """
    Applies transfer learning from pretrained tracking model.

    Fine-tunes for specific downstream tasks.
    """

    def __init__(self, pretrained_encoder: DeepTrackingEncoder):
        """Initialize with pretrained encoder."""
        self.encoder = pretrained_encoder

    def create_value_predictor(self) -> Dict[str, np.ndarray]:
        """
        Create a value prediction head.

        Predicts expected value (like EPV) from embeddings.
        """
        np.random.seed(44)
        output_dim = self.encoder.config.output_dim

        return {
            'W1': np.random.randn(output_dim, 32) * 0.1,
            'b1': np.zeros(32),
            'W2': np.random.randn(32, 1) * 0.1,
            'b2': np.zeros(1),
        }

    def predict_value(
        self,
        features: LearnedFeatures,
        value_head: Dict[str, np.ndarray],
    ) -> float:
        """
        Predict value from learned features.
        """
        h1 = np.maximum(0, features.embedding @ value_head['W1'] + value_head['b1'])
        output = h1 @ value_head['W2'] + value_head['b2']
        return float(output[0])

    def create_outcome_classifier(
        self,
        outcomes: List[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[int, str]]:
        """
        Create outcome classification head.

        E.g., classifying if possession ends in shot, turnover, etc.
        """
        np.random.seed(45)
        output_dim = self.encoder.config.output_dim
        num_outcomes = len(outcomes)

        head = {
            'W': np.random.randn(output_dim, num_outcomes) * 0.1,
            'b': np.zeros(num_outcomes),
        }

        outcome_map = {i: o for i, o in enumerate(outcomes)}

        return head, outcome_map
