"""
NHL2Vec: Player Embeddings for Hockey Analytics

This module implements dense player representations (embeddings) adapted from
NBA2Vec and Football2Vec research. These embeddings capture player style,
position, and effectiveness in a continuous vector space.

Applications:
- Player similarity search (find comparable players)
- Line chemistry prediction
- Trade/draft analysis
- Style clustering and archetypes

References:
    - "NBA2Vec: Dense Feature Representations of NBA Players." arXiv:2302.13386
    - Magdaci, O. (2023). "Football2Vec: Embedding the Language of Football Using NLP."
    - Wang, Z., et al. (2020). "Play2Vec: Sports Play Retrieval." KDD.
    - "Footballer Player Recommendation Model Using Graph Convolutional Networks." Springer 2024.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class Position(Enum):
    """Player positions."""
    CENTER = "C"
    LEFT_WING = "LW"
    RIGHT_WING = "RW"
    LEFT_DEFENSE = "LD"
    RIGHT_DEFENSE = "RD"
    GOALIE = "G"


class PlayerRole(Enum):
    """Tactical roles (derived from play style)."""
    # Forwards
    SNIPER = "sniper"
    PLAYMAKER = "playmaker"
    POWER_FORWARD = "power_forward"
    TWO_WAY_FORWARD = "two_way_forward"
    GRINDER = "grinder"
    ENERGY = "energy"

    # Defensemen
    OFFENSIVE_DMAN = "offensive_dman"
    TWO_WAY_DMAN = "two_way_dman"
    SHUTDOWN_DMAN = "shutdown_dman"
    PHYSICAL_DMAN = "physical_dman"

    # Goalies
    BUTTERFLY = "butterfly"
    HYBRID = "hybrid"
    STANDUP = "standup"


@dataclass
class PlayerStats:
    """Comprehensive player statistics for embedding computation."""
    player_id: str
    name: str
    position: Position
    team_id: str

    # Counting stats (per 60 minutes)
    goals_per_60: float = 0.0
    assists_per_60: float = 0.0
    points_per_60: float = 0.0
    shots_per_60: float = 0.0
    hits_per_60: float = 0.0
    blocks_per_60: float = 0.0
    takeaways_per_60: float = 0.0
    giveaways_per_60: float = 0.0
    penalties_drawn_per_60: float = 0.0
    penalties_taken_per_60: float = 0.0

    # Advanced stats
    xg_per_60: float = 0.0
    xga_per_60: float = 0.0  # xG against
    corsi_for_pct: float = 50.0
    fenwick_for_pct: float = 50.0
    oz_start_pct: float = 50.0  # Offensive zone faceoff %

    # On-ice impact
    on_ice_goals_for_per_60: float = 0.0
    on_ice_goals_against_per_60: float = 0.0
    on_ice_xgf_per_60: float = 0.0
    on_ice_xga_per_60: float = 0.0

    # Time on ice distribution
    avg_toi_per_game: float = 0.0
    pp_toi_pct: float = 0.0  # % of TOI on power play
    pk_toi_pct: float = 0.0  # % of TOI on penalty kill
    ev_toi_pct: float = 100.0  # % of TOI at even strength

    # Passing/playmaking
    pass_completion_pct: float = 0.0
    zone_entries_per_60: float = 0.0
    zone_exits_per_60: float = 0.0
    controlled_entries_pct: float = 0.0

    # Physical
    avg_skating_speed: float = 0.0
    distance_per_60: float = 0.0
    high_speed_distance_per_60: float = 0.0

    # Context
    quality_of_competition: float = 0.0  # Average opponent xGF%
    quality_of_teammates: float = 0.0  # Average teammate xGF%


@dataclass
class PlayerEmbedding:
    """Player embedding with metadata."""
    player_id: str
    name: str
    position: Position
    embedding: np.ndarray  # Dense vector representation
    derived_role: PlayerRole
    style_scores: Dict[str, float]  # Interpretable style dimensions


@dataclass
class ShiftContext:
    """Context for a single shift (used for context-based embeddings)."""
    shift_id: str
    player_ids: List[str]  # All players on ice
    opponent_ids: List[str]
    strength_state: str
    zone_start: str  # 'offensive', 'defensive', 'neutral'
    outcome: str  # 'goal_for', 'goal_against', 'no_goal'
    xg_for: float
    xg_against: float
    duration_seconds: float


class NHL2Vec:
    """
    NHL2Vec: Dense player embeddings for hockey.

    Creates continuous vector representations of players that capture:
    - Playing style (scoring, physical, defensive)
    - Positional tendencies
    - Effectiveness metrics
    - Line chemistry potential

    Based on skip-gram architecture where players appearing together
    (same shift, same line) learn similar representations.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        style_dim: int = 8,
        learning_rate: float = 0.01,
        negative_samples: int = 5
    ):
        """
        Initialize NHL2Vec model.

        Args:
            embedding_dim: Dimension of player embeddings
            style_dim: Dimension of interpretable style features
            learning_rate: Learning rate for embedding updates
            negative_samples: Number of negative samples per positive
        """
        self.embedding_dim = embedding_dim
        self.style_dim = style_dim
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples

        # Player embeddings (initialized when players are added)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.context_embeddings: Dict[str, np.ndarray] = {}

        # Player metadata
        self.player_stats: Dict[str, PlayerStats] = {}
        self.player_positions: Dict[str, Position] = {}

        # Training data
        self.shift_data: List[ShiftContext] = []
        self.co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        # Style dimensions for interpretability
        self.style_dimensions = [
            'offensive', 'defensive', 'physical', 'playmaking',
            'speed', 'shot_quality', 'two_way', 'energy'
        ]

    def add_player(self, stats: PlayerStats):
        """
        Add a player to the model.

        Initializes embeddings and stores stats.
        """
        player_id = stats.player_id

        # Initialize embedding (Xavier initialization)
        self.embeddings[player_id] = np.random.randn(self.embedding_dim) * np.sqrt(2.0 / self.embedding_dim)
        self.context_embeddings[player_id] = np.random.randn(self.embedding_dim) * np.sqrt(2.0 / self.embedding_dim)

        # Store metadata
        self.player_stats[player_id] = stats
        self.player_positions[player_id] = stats.position

    def add_shift(self, shift: ShiftContext):
        """
        Add shift data for training.

        Records co-occurrences between players.
        """
        self.shift_data.append(shift)

        # Record co-occurrences (players on same team, same shift)
        for i, p1 in enumerate(shift.player_ids):
            for p2 in shift.player_ids[i+1:]:
                pair = tuple(sorted([p1, p2]))
                self.co_occurrence[pair] += 1

    def train(self, epochs: int = 10, batch_size: int = 256):
        """
        Train embeddings using skip-gram with negative sampling.

        Players that appear together learn similar embeddings.
        """
        all_players = list(self.embeddings.keys())
        if len(all_players) < 2:
            return

        for epoch in range(epochs):
            total_loss = 0.0
            n_updates = 0

            # Sample positive pairs from co-occurrences
            for (p1, p2), count in self.co_occurrence.items():
                if p1 not in self.embeddings or p2 not in self.embeddings:
                    continue

                # Weight by co-occurrence count
                weight = np.log(count + 1)

                # Positive sample
                loss = self._train_pair(p1, p2, positive=True, weight=weight)
                total_loss += loss
                n_updates += 1

                # Negative samples
                for _ in range(self.negative_samples):
                    neg_player = np.random.choice(all_players)
                    if neg_player != p1 and neg_player != p2:
                        loss = self._train_pair(p1, neg_player, positive=False, weight=1.0)
                        total_loss += loss
                        n_updates += 1

            if n_updates > 0:
                avg_loss = total_loss / n_updates
                # Could log progress here

    def _train_pair(
        self,
        target: str,
        context: str,
        positive: bool,
        weight: float
    ) -> float:
        """
        Train on a single pair using skip-gram objective.

        Args:
            target: Target player ID
            context: Context player ID
            positive: Whether this is a positive (True) or negative (False) sample
            weight: Sample weight

        Returns:
            Loss for this sample
        """
        target_emb = self.embeddings[target]
        context_emb = self.context_embeddings[context]

        # Compute dot product
        dot = np.dot(target_emb, context_emb)
        sigmoid = 1 / (1 + np.exp(-np.clip(dot, -10, 10)))

        # Compute loss and gradient
        if positive:
            loss = -weight * np.log(sigmoid + 1e-10)
            grad = weight * (sigmoid - 1)
        else:
            loss = -weight * np.log(1 - sigmoid + 1e-10)
            grad = weight * sigmoid

        # Update embeddings
        target_grad = grad * context_emb
        context_grad = grad * target_emb

        self.embeddings[target] -= self.learning_rate * target_grad
        self.context_embeddings[context] -= self.learning_rate * context_grad

        return loss

    def compute_stat_embedding(self, player_id: str) -> np.ndarray:
        """
        Compute embedding from player statistics.

        Creates a feature vector from normalized statistics.
        """
        if player_id not in self.player_stats:
            return np.zeros(self.style_dim)

        stats = self.player_stats[player_id]

        # Normalize stats to style dimensions
        style_vector = np.array([
            # Offensive: goals, shots, xG
            (stats.goals_per_60 / 1.5 + stats.xg_per_60 / 1.0) / 2,

            # Defensive: blocks, takeaways, low GA
            (stats.blocks_per_60 / 3.0 + stats.takeaways_per_60 / 1.5 +
             (1 - stats.on_ice_xga_per_60 / 3.0)) / 3,

            # Physical: hits
            stats.hits_per_60 / 10.0,

            # Playmaking: assists, pass completion, controlled entries
            (stats.assists_per_60 / 2.0 + stats.pass_completion_pct / 100 +
             stats.controlled_entries_pct / 100) / 3,

            # Speed: skating speed, distance
            (stats.avg_skating_speed / 25.0 + stats.high_speed_distance_per_60 / 5000) / 2,

            # Shot quality: xG per shot (implied)
            stats.xg_per_60 / max(stats.shots_per_60, 1) * 10,

            # Two-way: Corsi%, zone starts balance
            (stats.corsi_for_pct / 100 + (1 - abs(stats.oz_start_pct - 50) / 50)) / 2,

            # Energy: penalties drawn vs taken
            (stats.penalties_drawn_per_60 - stats.penalties_taken_per_60) / 2 + 0.5
        ])

        return np.clip(style_vector, 0, 1)

    def get_embedding(self, player_id: str) -> PlayerEmbedding:
        """
        Get full player embedding with metadata.

        Combines learned embedding with style features.
        """
        if player_id not in self.embeddings:
            raise ValueError(f"Player {player_id} not found")

        stats = self.player_stats.get(player_id)
        learned_emb = self.embeddings[player_id]
        style_emb = self.compute_stat_embedding(player_id)

        # Combine learned and statistical embeddings
        combined = np.concatenate([learned_emb, style_emb])

        # Derive role from style
        role = self._derive_role(style_emb, stats.position if stats else Position.CENTER)

        # Create style scores dictionary
        style_scores = {
            dim: float(style_emb[i])
            for i, dim in enumerate(self.style_dimensions)
        }

        return PlayerEmbedding(
            player_id=player_id,
            name=stats.name if stats else player_id,
            position=stats.position if stats else Position.CENTER,
            embedding=combined,
            derived_role=role,
            style_scores=style_scores
        )

    def _derive_role(self, style_emb: np.ndarray, position: Position) -> PlayerRole:
        """
        Derive tactical role from style embedding.
        """
        offensive = style_emb[0]
        defensive = style_emb[1]
        physical = style_emb[2]
        playmaking = style_emb[3]
        two_way = style_emb[6]

        if position in [Position.CENTER, Position.LEFT_WING, Position.RIGHT_WING]:
            # Forward roles
            if offensive > 0.7 and playmaking < 0.5:
                return PlayerRole.SNIPER
            elif playmaking > 0.7:
                return PlayerRole.PLAYMAKER
            elif physical > 0.6 and offensive > 0.5:
                return PlayerRole.POWER_FORWARD
            elif two_way > 0.6:
                return PlayerRole.TWO_WAY_FORWARD
            elif physical > 0.7:
                return PlayerRole.GRINDER
            else:
                return PlayerRole.ENERGY

        elif position in [Position.LEFT_DEFENSE, Position.RIGHT_DEFENSE]:
            # Defenseman roles
            if offensive > 0.6:
                return PlayerRole.OFFENSIVE_DMAN
            elif defensive > 0.7:
                return PlayerRole.SHUTDOWN_DMAN
            elif physical > 0.6:
                return PlayerRole.PHYSICAL_DMAN
            else:
                return PlayerRole.TWO_WAY_DMAN

        else:
            return PlayerRole.HYBRID  # Default for goalies

    def find_similar_players(
        self,
        player_id: str,
        n: int = 10,
        same_position: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Find most similar players by embedding distance.

        Args:
            player_id: Query player
            n: Number of similar players to return
            same_position: Whether to filter by same position

        Returns:
            List of (player_id, similarity_score) tuples
        """
        if player_id not in self.embeddings:
            return []

        query_emb = self.get_embedding(player_id)
        query_pos = self.player_positions.get(player_id)

        similarities = []
        for pid in self.embeddings:
            if pid == player_id:
                continue

            if same_position and self.player_positions.get(pid) != query_pos:
                continue

            candidate_emb = self.get_embedding(pid)
            sim = self._cosine_similarity(query_emb.embedding, candidate_emb.embedding)
            similarities.append((pid, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:n]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def predict_line_chemistry(
        self,
        player_ids: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Predict chemistry score for a line combination.

        Uses embedding similarity and role complementarity.

        Args:
            player_ids: List of player IDs (typically 3 forwards or 2 D)

        Returns:
            Tuple of (chemistry_score, analysis_dict)
        """
        if len(player_ids) < 2:
            return 0.5, {"error": "Need at least 2 players"}

        embeddings = []
        roles = []

        for pid in player_ids:
            if pid not in self.embeddings:
                continue
            emb = self.get_embedding(pid)
            embeddings.append(emb)
            roles.append(emb.derived_role)

        if len(embeddings) < 2:
            return 0.5, {"error": "Players not found"}

        # Similarity score (average pairwise similarity)
        sim_scores = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(
                    embeddings[i].embedding,
                    embeddings[j].embedding
                )
                sim_scores.append(sim)

        avg_similarity = np.mean(sim_scores)

        # Role complementarity (diverse roles = better)
        unique_roles = len(set(roles))
        role_diversity = unique_roles / len(roles)

        # Style balance (need offense + defense)
        avg_offensive = np.mean([e.style_scores['offensive'] for e in embeddings])
        avg_defensive = np.mean([e.style_scores['defensive'] for e in embeddings])
        avg_playmaking = np.mean([e.style_scores['playmaking'] for e in embeddings])

        # Chemistry formula
        # Good chemistry = similar enough to work together + diverse enough to complement
        chemistry = (
            0.3 * avg_similarity +
            0.2 * role_diversity +
            0.2 * avg_offensive +
            0.15 * avg_defensive +
            0.15 * avg_playmaking
        )

        analysis = {
            'similarity': avg_similarity,
            'role_diversity': role_diversity,
            'roles': [r.value for r in roles],
            'avg_offensive': avg_offensive,
            'avg_defensive': avg_defensive,
            'avg_playmaking': avg_playmaking,
            'style_balance': 1 - abs(avg_offensive - avg_defensive)
        }

        return float(chemistry), analysis

    def cluster_players(
        self,
        n_clusters: int = 8
    ) -> Dict[int, List[str]]:
        """
        Cluster players into archetypes using K-means.

        Args:
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster ID to list of player IDs
        """
        if len(self.embeddings) < n_clusters:
            return {}

        # Gather embeddings
        player_ids = list(self.embeddings.keys())
        embeddings = np.array([
            self.get_embedding(pid).embedding for pid in player_ids
        ])

        # Simple K-means implementation
        centroids = embeddings[np.random.choice(len(embeddings), n_clusters, replace=False)]

        for _ in range(50):  # Max iterations
            # Assign to clusters
            distances = np.array([
                [np.linalg.norm(e - c) for c in centroids]
                for e in embeddings
            ])
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = []
            for k in range(n_clusters):
                cluster_points = embeddings[assignments == k]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    new_centroids.append(centroids[k])

            centroids = np.array(new_centroids)

        # Build cluster dictionary
        clusters: Dict[int, List[str]] = defaultdict(list)
        for pid, cluster_id in zip(player_ids, assignments):
            clusters[int(cluster_id)].append(pid)

        return dict(clusters)


class Action2Vec:
    """
    Action embeddings using Word2Vec-style approach.

    Treats sequences of hockey actions as "sentences" and learns
    embeddings for action types.
    """

    def __init__(self, embedding_dim: int = 16, window_size: int = 3):
        """
        Initialize Action2Vec.

        Args:
            embedding_dim: Dimension of action embeddings
            window_size: Context window for skip-gram
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.action_embeddings: Dict[str, np.ndarray] = {}

    def add_action_type(self, action_type: str):
        """Add an action type with random initialization."""
        if action_type not in self.action_embeddings:
            self.action_embeddings[action_type] = np.random.randn(self.embedding_dim) * 0.1

    def train_on_sequence(self, actions: List[str], learning_rate: float = 0.01):
        """
        Train on a sequence of actions.

        Args:
            actions: List of action type strings
            learning_rate: Learning rate
        """
        for action in actions:
            self.add_action_type(action)

        for i, center in enumerate(actions):
            start = max(0, i - self.window_size)
            end = min(len(actions), i + self.window_size + 1)

            for j in range(start, end):
                if j != i:
                    context = actions[j]
                    self._update_pair(center, context, learning_rate)

    def _update_pair(self, center: str, context: str, lr: float):
        """Update embeddings for a center-context pair."""
        center_emb = self.action_embeddings[center]
        context_emb = self.action_embeddings[context]

        dot = np.dot(center_emb, context_emb)
        sigmoid = 1 / (1 + np.exp(-np.clip(dot, -10, 10)))

        grad = lr * (1 - sigmoid)
        self.action_embeddings[center] += grad * context_emb
        self.action_embeddings[context] += grad * center_emb

    def get_embedding(self, action_type: str) -> np.ndarray:
        """Get embedding for an action type."""
        return self.action_embeddings.get(action_type, np.zeros(self.embedding_dim))

    def find_similar_actions(self, action_type: str, n: int = 5) -> List[Tuple[str, float]]:
        """Find similar action types."""
        if action_type not in self.action_embeddings:
            return []

        query = self.action_embeddings[action_type]
        similarities = []

        for at, emb in self.action_embeddings.items():
            if at != action_type:
                sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-10)
                similarities.append((at, float(sim)))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:n]


class PlayerRecommendationSystem:
    """
    GCN-inspired player recommendation system.

    Based on "Footballer Player Recommendation Model Using Graph Convolutional Networks"
    (Springer, 2024).

    Uses player similarity graphs and embedding propagation.
    """

    def __init__(self, nhl2vec: NHL2Vec):
        """
        Initialize recommendation system.

        Args:
            nhl2vec: Trained NHL2Vec model
        """
        self.nhl2vec = nhl2vec
        self.similarity_threshold = 0.5

    def recommend_replacement(
        self,
        player_id: str,
        budget_constraint: Optional[float] = None,
        team_context: Optional[List[str]] = None,
        n_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recommend replacement players for a given player.

        Args:
            player_id: Player to replace
            budget_constraint: Optional salary cap constraint
            team_context: Current team players (for chemistry fit)
            n_recommendations: Number of recommendations

        Returns:
            List of recommendation dictionaries
        """
        if player_id not in self.nhl2vec.embeddings:
            return []

        # Find similar players
        similar = self.nhl2vec.find_similar_players(
            player_id,
            n=n_recommendations * 2,
            same_position=True
        )

        recommendations = []
        for candidate_id, similarity in similar:
            rec = {
                'player_id': candidate_id,
                'similarity': similarity,
                'style_match': self._compute_style_match(player_id, candidate_id),
                'role': self.nhl2vec.get_embedding(candidate_id).derived_role.value
            }

            # Add chemistry fit if team context provided
            if team_context:
                chemistry, _ = self.nhl2vec.predict_line_chemistry(
                    [candidate_id] + team_context[:2]
                )
                rec['team_chemistry'] = chemistry

            recommendations.append(rec)

        # Sort by combined score
        recommendations.sort(
            key=lambda x: x['similarity'] * 0.6 + x.get('team_chemistry', 0.5) * 0.4,
            reverse=True
        )

        return recommendations[:n_recommendations]

    def _compute_style_match(self, player1: str, player2: str) -> float:
        """Compute style match between two players."""
        emb1 = self.nhl2vec.get_embedding(player1)
        emb2 = self.nhl2vec.get_embedding(player2)

        style1 = np.array(list(emb1.style_scores.values()))
        style2 = np.array(list(emb2.style_scores.values()))

        return float(1 - np.mean(np.abs(style1 - style2)))

    def build_depth_chart(
        self,
        team_players: List[str],
        optimize_chemistry: bool = True
    ) -> Dict[str, List[List[str]]]:
        """
        Build optimal depth chart for a team.

        Args:
            team_players: All players on team
            optimize_chemistry: Whether to optimize for line chemistry

        Returns:
            Dictionary with forward lines and defensive pairings
        """
        # Separate by position
        forwards = []
        defensemen = []

        for pid in team_players:
            if pid not in self.nhl2vec.player_positions:
                continue
            pos = self.nhl2vec.player_positions[pid]
            if pos in [Position.CENTER, Position.LEFT_WING, Position.RIGHT_WING]:
                forwards.append(pid)
            elif pos in [Position.LEFT_DEFENSE, Position.RIGHT_DEFENSE]:
                defensemen.append(pid)

        result = {
            'forward_lines': [],
            'defensive_pairs': []
        }

        # Build forward lines (groups of 3)
        if optimize_chemistry and len(forwards) >= 3:
            # Simple greedy approach - would use optimization in production
            used = set()
            while len(forwards) - len(used) >= 3:
                best_line = None
                best_chemistry = -1

                available = [f for f in forwards if f not in used]
                for i in range(len(available)):
                    for j in range(i + 1, len(available)):
                        for k in range(j + 1, len(available)):
                            line = [available[i], available[j], available[k]]
                            chem, _ = self.nhl2vec.predict_line_chemistry(line)
                            if chem > best_chemistry:
                                best_chemistry = chem
                                best_line = line

                if best_line:
                    result['forward_lines'].append(best_line)
                    used.update(best_line)
                else:
                    break
        else:
            # Just group by 3
            for i in range(0, len(forwards), 3):
                result['forward_lines'].append(forwards[i:i+3])

        # Build defensive pairs
        if optimize_chemistry and len(defensemen) >= 2:
            used = set()
            while len(defensemen) - len(used) >= 2:
                best_pair = None
                best_chemistry = -1

                available = [d for d in defensemen if d not in used]
                for i in range(len(available)):
                    for j in range(i + 1, len(available)):
                        pair = [available[i], available[j]]
                        chem, _ = self.nhl2vec.predict_line_chemistry(pair)
                        if chem > best_chemistry:
                            best_chemistry = chem
                            best_pair = pair

                if best_pair:
                    result['defensive_pairs'].append(best_pair)
                    used.update(best_pair)
                else:
                    break
        else:
            for i in range(0, len(defensemen), 2):
                result['defensive_pairs'].append(defensemen[i:i+2])

        return result
