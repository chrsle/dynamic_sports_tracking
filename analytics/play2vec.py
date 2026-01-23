"""
Play2Vec: Sports Play Embedding and Retrieval

Implements play embedding and retrieval system translated from
soccer research.

Key Paper:
- Wang, Z., et al. (2020). "Effective and Efficient Sports Play Retrieval
  with Deep Representation Learning." KDD.
- github.com/zhengwang125/play2vec

Key Concepts:
- Maps tracking sequences to grid-based segment matrices
- Builds "sports corpus" using Jaccard similarity
- Skip-gram model learns segment embeddings
- Denoising Sequence Encoder-Decoder (DSED) combines segments

Hockey Translation:
- Retrieve similar plays from historical database
- "Find me plays that look like this power play setup"
- Opponent tendencies analysis
- Teaching tool for player development
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class PlayType(Enum):
    """Types of hockey plays."""
    BREAKOUT = "breakout"
    ZONE_ENTRY = "zone_entry"
    OFFENSIVE_CYCLE = "cycle"
    NET_DRIVE = "net_drive"
    POINT_SHOT = "point_shot"
    RUSH = "rush"
    FORECHECK = "forecheck"
    BACKCHECK = "backcheck"
    PENALTY_KILL = "pk"
    POWER_PLAY = "pp"
    FACEOFF = "faceoff"


@dataclass
class TrackingPoint:
    """Single point in tracking data."""
    timestamp: float
    player_id: str
    team_id: str
    x: float
    y: float
    has_puck: bool = False


@dataclass
class PlaySequence:
    """A sequence of tracking data representing a play."""
    play_id: str
    frames: List[List[TrackingPoint]]  # List of frames, each frame has all players
    puck_path: List[Tuple[float, float, float]]  # x, y, timestamp
    play_type: Optional[PlayType] = None
    outcome: Optional[str] = None  # 'goal', 'shot', 'turnover', etc.
    team_id: str = ""
    game_id: str = ""


@dataclass
class PlayEmbedding:
    """Embedding representation of a play."""
    play_id: str
    embedding: np.ndarray
    segment_embeddings: List[np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class SimilarPlay:
    """Result of similarity search."""
    play_id: str
    similarity: float
    play_type: Optional[PlayType]
    outcome: Optional[str]
    metadata: Dict[str, Any]


class GridEncoder:
    """
    Encodes tracking data to grid-based segments.

    Creates spatial-temporal segments that capture play patterns.
    """

    def __init__(
        self,
        grid_x: int = 10,
        grid_y: int = 5,
        rink_length: float = 200.0,
        rink_width: float = 85.0,
        time_segments: int = 5,
    ):
        """Initialize encoder."""
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.rink_length = rink_length
        self.rink_width = rink_width
        self.time_segments = time_segments

        self.cell_width = rink_length / grid_x
        self.cell_height = rink_width / grid_y

    def position_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert position to grid cell."""
        cell_x = int(np.clip(x / self.cell_width, 0, self.grid_x - 1))
        cell_y = int(np.clip(y / self.cell_height, 0, self.grid_y - 1))
        return cell_x, cell_y

    def encode_frame(
        self,
        frame: List[TrackingPoint],
        team_id: str,
    ) -> np.ndarray:
        """
        Encode single frame to grid.

        Returns grid with player counts per cell.
        """
        grid = np.zeros((self.grid_y, self.grid_x, 3))
        # Channel 0: Own team
        # Channel 1: Opponent
        # Channel 2: Puck

        for point in frame:
            cell_x, cell_y = self.position_to_cell(point.x, point.y)

            if point.team_id == team_id:
                grid[cell_y, cell_x, 0] += 1
            else:
                grid[cell_y, cell_x, 1] += 1

            if point.has_puck:
                grid[cell_y, cell_x, 2] = 1

        return grid

    def encode_sequence(
        self,
        sequence: PlaySequence,
    ) -> np.ndarray:
        """
        Encode play sequence to segment matrix.

        Returns 3D array: (time_segments, grid_y, grid_x, channels)
        """
        n_frames = len(sequence.frames)
        if n_frames == 0:
            return np.zeros((self.time_segments, self.grid_y, self.grid_x, 3))

        # Divide into time segments
        frames_per_segment = max(1, n_frames // self.time_segments)

        segment_matrices = []
        for t in range(self.time_segments):
            start_idx = t * frames_per_segment
            end_idx = min((t + 1) * frames_per_segment, n_frames)

            # Average over frames in segment
            segment_grid = np.zeros((self.grid_y, self.grid_x, 3))
            count = 0

            for idx in range(start_idx, end_idx):
                if idx < len(sequence.frames):
                    frame_grid = self.encode_frame(
                        sequence.frames[idx], sequence.team_id
                    )
                    segment_grid += frame_grid
                    count += 1

            if count > 0:
                segment_grid /= count

            segment_matrices.append(segment_grid)

        return np.array(segment_matrices)

    def compute_segment_signature(
        self,
        segment: np.ndarray,
    ) -> str:
        """
        Compute signature for a segment.

        Used for building corpus and Jaccard similarity.
        """
        # Threshold to binary
        binary = (segment > 0.5).astype(int)

        # Create string signature
        signature_parts = []
        for c in range(segment.shape[2]):
            channel_sig = ''.join(
                str(binary[i, j, c])
                for i in range(segment.shape[0])
                for j in range(segment.shape[1])
            )
            signature_parts.append(channel_sig)

        return '_'.join(signature_parts)


class SegmentVocabulary:
    """
    Builds vocabulary of segment patterns.

    Similar to word vocabulary in NLP.
    """

    def __init__(self, min_count: int = 2):
        """Initialize vocabulary."""
        self.min_count = min_count
        self.signature_to_idx: Dict[str, int] = {}
        self.idx_to_signature: Dict[int, str] = {}
        self.signature_counts: Dict[str, int] = defaultdict(int)
        self.vocab_size = 0

    def add_signature(self, signature: str):
        """Add signature to vocabulary."""
        self.signature_counts[signature] += 1

    def build(self):
        """Build vocabulary from collected signatures."""
        idx = 0
        for sig, count in self.signature_counts.items():
            if count >= self.min_count:
                self.signature_to_idx[sig] = idx
                self.idx_to_signature[idx] = sig
                idx += 1

        self.vocab_size = idx

    def get_idx(self, signature: str) -> int:
        """Get index for signature, -1 if unknown."""
        return self.signature_to_idx.get(signature, -1)


class SkipGramEmbedder:
    """
    Skip-gram model for learning segment embeddings.

    Similar to Word2Vec but for play segments.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        window_size: int = 2,
        learning_rate: float = 0.01,
    ):
        """Initialize embedder."""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate

        # Initialize embeddings
        np.random.seed(42)
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.context_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1

    def train_step(
        self,
        center_idx: int,
        context_idx: int,
        negative_samples: List[int],
    ) -> float:
        """
        One training step with negative sampling.

        Returns loss.
        """
        # Get embeddings
        center_emb = self.embeddings[center_idx]
        context_emb = self.context_embeddings[context_idx]

        # Positive sample
        pos_score = np.dot(center_emb, context_emb)
        pos_prob = 1.0 / (1.0 + np.exp(-np.clip(pos_score, -500, 500)))
        pos_loss = -np.log(pos_prob + 1e-10)

        # Gradient for positive
        pos_grad = (pos_prob - 1) * context_emb
        context_grad = (pos_prob - 1) * center_emb

        # Negative samples
        neg_loss = 0
        for neg_idx in negative_samples:
            neg_emb = self.context_embeddings[neg_idx]
            neg_score = np.dot(center_emb, neg_emb)
            neg_prob = 1.0 / (1.0 + np.exp(-np.clip(neg_score, -500, 500)))
            neg_loss -= np.log(1 - neg_prob + 1e-10)

            # Gradients
            pos_grad += neg_prob * neg_emb
            self.context_embeddings[neg_idx] -= self.learning_rate * neg_prob * center_emb

        # Update embeddings
        self.embeddings[center_idx] -= self.learning_rate * pos_grad
        self.context_embeddings[context_idx] -= self.learning_rate * context_grad

        return pos_loss + neg_loss

    def get_embedding(self, idx: int) -> np.ndarray:
        """Get embedding for segment."""
        if 0 <= idx < self.vocab_size:
            return self.embeddings[idx]
        return np.zeros(self.embedding_dim)


class SequenceEncoder:
    """
    Encodes sequence of segment embeddings to play embedding.

    Uses LSTM-like aggregation.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        """Initialize encoder."""
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        np.random.seed(43)
        self.W_f = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.1
        self.W_i = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.1
        self.W_c = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.1
        self.W_o = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.1

    def encode(self, segment_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Encode sequence of segment embeddings.

        Returns single play embedding.
        """
        if not segment_embeddings:
            return np.zeros(self.hidden_dim)

        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        for seg_emb in segment_embeddings:
            # Ensure correct dimension
            if len(seg_emb) != self.embedding_dim:
                seg_emb = np.pad(seg_emb, (0, max(0, self.embedding_dim - len(seg_emb))))
                seg_emb = seg_emb[:self.embedding_dim]

            combined = np.concatenate([seg_emb, h])

            # LSTM gates
            f = self._sigmoid(combined @ self.W_f)
            i = self._sigmoid(combined @ self.W_i)
            c_tilde = np.tanh(combined @ self.W_c)
            o = self._sigmoid(combined @ self.W_o)

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

        return h

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class Play2VecModel:
    """
    Complete Play2Vec model for play embedding and retrieval.
    """

    def __init__(
        self,
        grid_x: int = 10,
        grid_y: int = 5,
        time_segments: int = 5,
        embedding_dim: int = 32,
        play_embedding_dim: int = 64,
    ):
        """Initialize model."""
        self.grid_encoder = GridEncoder(
            grid_x=grid_x,
            grid_y=grid_y,
            time_segments=time_segments,
        )
        self.vocabulary = SegmentVocabulary()
        self.segment_embedder: Optional[SkipGramEmbedder] = None
        self.sequence_encoder = SequenceEncoder(embedding_dim, play_embedding_dim)

        self.embedding_dim = embedding_dim
        self.play_embedding_dim = play_embedding_dim

        # Index of encoded plays
        self.play_index: Dict[str, PlayEmbedding] = {}

    def build_vocabulary(self, sequences: List[PlaySequence]):
        """Build segment vocabulary from training data."""
        for sequence in sequences:
            matrix = self.grid_encoder.encode_sequence(sequence)

            for t in range(matrix.shape[0]):
                signature = self.grid_encoder.compute_segment_signature(matrix[t])
                self.vocabulary.add_signature(signature)

        self.vocabulary.build()

        # Initialize embedder with vocabulary
        self.segment_embedder = SkipGramEmbedder(
            vocab_size=max(1, self.vocabulary.vocab_size),
            embedding_dim=self.embedding_dim,
        )

    def train(
        self,
        sequences: List[PlaySequence],
        epochs: int = 10,
        negative_samples: int = 5,
    ):
        """Train segment embeddings."""
        if self.segment_embedder is None:
            self.build_vocabulary(sequences)

        # Collect training pairs
        for epoch in range(epochs):
            total_loss = 0
            count = 0

            for sequence in sequences:
                matrix = self.grid_encoder.encode_sequence(sequence)
                indices = []

                for t in range(matrix.shape[0]):
                    signature = self.grid_encoder.compute_segment_signature(matrix[t])
                    idx = self.vocabulary.get_idx(signature)
                    if idx >= 0:
                        indices.append(idx)

                # Skip-gram pairs
                for i, center_idx in enumerate(indices):
                    window_start = max(0, i - self.segment_embedder.window_size)
                    window_end = min(len(indices), i + self.segment_embedder.window_size + 1)

                    for j in range(window_start, window_end):
                        if i != j:
                            context_idx = indices[j]

                            # Generate negative samples
                            negatives = []
                            while len(negatives) < negative_samples:
                                neg = np.random.randint(0, self.vocabulary.vocab_size)
                                if neg != center_idx and neg != context_idx:
                                    negatives.append(neg)

                            loss = self.segment_embedder.train_step(
                                center_idx, context_idx, negatives
                            )
                            total_loss += loss
                            count += 1

    def encode_play(self, sequence: PlaySequence) -> PlayEmbedding:
        """Encode a play to embedding."""
        if self.segment_embedder is None:
            # Create dummy embedder if not trained
            self.segment_embedder = SkipGramEmbedder(
                vocab_size=100,
                embedding_dim=self.embedding_dim,
            )

        matrix = self.grid_encoder.encode_sequence(sequence)
        segment_embeddings = []

        for t in range(matrix.shape[0]):
            signature = self.grid_encoder.compute_segment_signature(matrix[t])
            idx = self.vocabulary.get_idx(signature)

            if idx >= 0:
                seg_emb = self.segment_embedder.get_embedding(idx)
            else:
                # Unknown segment - use zero embedding
                seg_emb = np.zeros(self.embedding_dim)

            segment_embeddings.append(seg_emb)

        # Encode full sequence
        play_embedding = self.sequence_encoder.encode(segment_embeddings)

        return PlayEmbedding(
            play_id=sequence.play_id,
            embedding=play_embedding,
            segment_embeddings=segment_embeddings,
            metadata={
                'play_type': sequence.play_type.value if sequence.play_type else None,
                'outcome': sequence.outcome,
                'team_id': sequence.team_id,
                'game_id': sequence.game_id,
            },
        )

    def add_to_index(self, sequence: PlaySequence):
        """Add play to search index."""
        embedding = self.encode_play(sequence)
        self.play_index[sequence.play_id] = embedding

    def search_similar(
        self,
        query: PlaySequence,
        k: int = 5,
        play_type_filter: Optional[PlayType] = None,
    ) -> List[SimilarPlay]:
        """
        Search for similar plays.

        Args:
            query: Query play sequence
            k: Number of results
            play_type_filter: Optional filter by play type

        Returns:
            List of similar plays with similarity scores
        """
        query_embedding = self.encode_play(query)

        results = []
        for play_id, indexed in self.play_index.items():
            if play_id == query.play_id:
                continue

            # Filter by play type
            if play_type_filter:
                if indexed.metadata.get('play_type') != play_type_filter.value:
                    continue

            # Cosine similarity
            similarity = self._cosine_similarity(
                query_embedding.embedding,
                indexed.embedding,
            )

            results.append(SimilarPlay(
                play_id=play_id,
                similarity=similarity,
                play_type=PlayType(indexed.metadata['play_type'])
                    if indexed.metadata.get('play_type') else None,
                outcome=indexed.metadata.get('outcome'),
                metadata=indexed.metadata,
            ))

        # Sort by similarity
        results.sort(key=lambda x: x.similarity, reverse=True)

        return results[:k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))


class PlayRetriever:
    """
    High-level interface for play retrieval.
    """

    def __init__(self, model: Play2VecModel):
        """Initialize retriever."""
        self.model = model

    def find_plays_like(
        self,
        query_play: PlaySequence,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find plays similar to query.

        User-friendly interface for coaches.
        """
        similar = self.model.search_similar(query_play, k)

        results = []
        for play in similar:
            results.append({
                'play_id': play.play_id,
                'similarity_score': f"{play.similarity:.1%}",
                'play_type': play.play_type.value if play.play_type else 'unknown',
                'outcome': play.outcome or 'unknown',
                'game_id': play.metadata.get('game_id', ''),
            })

        return results

    def find_opponent_tendencies(
        self,
        opponent_plays: List[PlaySequence],
        situation: str,
    ) -> Dict[str, Any]:
        """
        Analyze opponent tendencies.

        Groups similar plays to identify patterns.
        """
        # Encode all plays
        embeddings = [self.model.encode_play(p) for p in opponent_plays]

        # Simple clustering by similarity
        clusters: List[List[int]] = []
        assigned = set()

        for i, emb in enumerate(embeddings):
            if i in assigned:
                continue

            cluster = [i]
            assigned.add(i)

            for j, other in enumerate(embeddings[i+1:], i+1):
                if j in assigned:
                    continue

                sim = self.model._cosine_similarity(emb.embedding, other.embedding)
                if sim > 0.8:  # High similarity threshold
                    cluster.append(j)
                    assigned.add(j)

            if len(cluster) >= 2:
                clusters.append(cluster)

        # Analyze clusters
        tendencies = []
        for cluster in clusters:
            plays = [opponent_plays[i] for i in cluster]
            outcomes = [p.outcome for p in plays if p.outcome]
            play_types = [p.play_type.value for p in plays if p.play_type]

            most_common_type = max(set(play_types), key=play_types.count) if play_types else 'unknown'

            tendencies.append({
                'pattern_count': len(cluster),
                'play_type': most_common_type,
                'success_rate': outcomes.count('goal') / len(outcomes) if outcomes else 0,
                'sample_plays': [plays[0].play_id] if plays else [],
            })

        return {
            'situation': situation,
            'patterns_found': len(tendencies),
            'tendencies': sorted(tendencies, key=lambda x: x['pattern_count'], reverse=True),
        }
