"""
Complex Multiplex Passing Network (CMPN)

Implements multi-layer network analysis for passing, translated from
soccer research.

Key Paper:
- "CMPN: Modeling and Analysis of Soccer Teams Using Complex Multiplex
  Passing Network." Chaos, Solitons & Fractals (2023).

Key Concepts:
- Multiple layers representing specific pass types
- Achieves >90% accuracy predicting attacking play outcomes
- Combines topological features with machine learning
- Each layer captures different passing dynamics

Hockey Translation:
- Separate layers for: tape-to-tape, saucer, dump-in, chip, bank passes
- Predict zone entry success based on passing patterns
- Identify which pass types lead to high-danger chances
- Team-level and line-level passing fingerprints
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class PassLayer(Enum):
    """Layers in the multiplex network (pass types)."""
    TAPE_TO_TAPE = "tape_to_tape"    # Direct passes along ice
    SAUCER = "saucer"                # Passes over sticks
    BANK = "bank"                    # Bank passes off boards
    CHIP = "chip"                    # Chip out passes
    DUMP = "dump"                    # Dump-ins and dump-outs
    DROP = "drop"                    # Drop passes
    CROSS_ICE = "cross_ice"          # Passes across zone
    BEHIND_NET = "behind_net"        # Passes using boards behind net


class ZoneType(Enum):
    """Zones for pass analysis."""
    DEFENSIVE = "dz"
    NEUTRAL = "nz"
    OFFENSIVE = "oz"


@dataclass
class PassEdge:
    """An edge in the passing network."""
    passer_id: str
    receiver_id: str
    pass_type: PassLayer
    timestamp: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    successful: bool
    zone: ZoneType
    game_id: str = ""


@dataclass
class NetworkNode:
    """A node (player) in the network."""
    player_id: str
    position: str                    # C, LW, RW, LD, RD
    passes_sent: int = 0
    passes_received: int = 0
    betweenness: float = 0.0
    clustering: float = 0.0
    eigenvector_centrality: float = 0.0


@dataclass
class LayerMetrics:
    """Metrics for a single network layer."""
    layer_type: PassLayer
    edge_count: int
    node_count: int
    density: float
    avg_clustering: float
    avg_path_length: float
    success_rate: float


@dataclass
class MultiplexMetrics:
    """Overall multiplex network metrics."""
    layer_metrics: Dict[PassLayer, LayerMetrics]
    inter_layer_correlation: Dict[Tuple[PassLayer, PassLayer], float]
    overall_density: float
    layer_participation: Dict[str, Dict[PassLayer, float]]  # player -> layer -> participation


class MultiplexPassingNetwork:
    """
    Multi-layer passing network for hockey analysis.

    Each layer represents a different pass type, allowing
    analysis of team passing patterns by style.
    """

    def __init__(self):
        """Initialize multiplex network."""
        # Adjacency matrices for each layer
        self.layers: Dict[PassLayer, Dict[Tuple[str, str], List[PassEdge]]] = {
            layer: defaultdict(list) for layer in PassLayer
        }

        # Node information
        self.nodes: Dict[str, NetworkNode] = {}

        # All edges for reference
        self.all_edges: List[PassEdge] = []

    def add_pass(self, edge: PassEdge):
        """Add a pass to the network."""
        # Add edge to appropriate layer
        self.layers[edge.pass_type][(edge.passer_id, edge.receiver_id)].append(edge)
        self.all_edges.append(edge)

        # Update node info
        if edge.passer_id not in self.nodes:
            self.nodes[edge.passer_id] = NetworkNode(player_id=edge.passer_id, position="")
        if edge.receiver_id not in self.nodes:
            self.nodes[edge.receiver_id] = NetworkNode(player_id=edge.receiver_id, position="")

        self.nodes[edge.passer_id].passes_sent += 1
        self.nodes[edge.receiver_id].passes_received += 1

    def build_from_events(self, pass_events: List[Dict[str, Any]]):
        """
        Build network from list of pass events.

        Events should have: passer_id, receiver_id, pass_type,
        start_x, start_y, end_x, end_y, successful, timestamp
        """
        for event in pass_events:
            edge = PassEdge(
                passer_id=event['passer_id'],
                receiver_id=event['receiver_id'],
                pass_type=PassLayer(event.get('pass_type', 'tape_to_tape')),
                timestamp=event.get('timestamp', 0.0),
                start_x=event.get('start_x', 0.0),
                start_y=event.get('start_y', 0.0),
                end_x=event.get('end_x', 0.0),
                end_y=event.get('end_y', 0.0),
                successful=event.get('successful', True),
                zone=ZoneType(event.get('zone', 'nz')),
                game_id=event.get('game_id', ''),
            )
            self.add_pass(edge)

    def get_adjacency_matrix(
        self,
        layer: PassLayer,
        weighted: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get adjacency matrix for a specific layer.

        Returns matrix and list of player IDs (row/col order).
        """
        player_ids = sorted(self.nodes.keys())
        n = len(player_ids)
        id_to_idx = {pid: i for i, pid in enumerate(player_ids)}

        matrix = np.zeros((n, n))

        for (passer, receiver), edges in self.layers[layer].items():
            if passer in id_to_idx and receiver in id_to_idx:
                i = id_to_idx[passer]
                j = id_to_idx[receiver]
                if weighted:
                    matrix[i, j] = len(edges)
                else:
                    matrix[i, j] = 1 if edges else 0

        return matrix, player_ids

    def compute_layer_metrics(self, layer: PassLayer) -> LayerMetrics:
        """Compute metrics for a single layer."""
        layer_edges = self.layers[layer]

        # Get unique nodes in this layer
        nodes_in_layer: Set[str] = set()
        successful_count = 0
        total_count = 0

        for (passer, receiver), edges in layer_edges.items():
            nodes_in_layer.add(passer)
            nodes_in_layer.add(receiver)
            for edge in edges:
                total_count += 1
                if edge.successful:
                    successful_count += 1

        n_nodes = len(nodes_in_layer)
        n_edges = sum(len(edges) for edges in layer_edges.values())

        # Density
        max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
        density = n_edges / max_edges if max_edges > 0 else 0

        # Success rate
        success_rate = successful_count / total_count if total_count > 0 else 0

        # Clustering coefficient (simplified)
        matrix, _ = self.get_adjacency_matrix(layer, weighted=False)
        clustering = self._compute_avg_clustering(matrix)

        # Path length (simplified)
        path_length = self._compute_avg_path_length(matrix)

        return LayerMetrics(
            layer_type=layer,
            edge_count=n_edges,
            node_count=n_nodes,
            density=density,
            avg_clustering=clustering,
            avg_path_length=path_length,
            success_rate=success_rate,
        )

    def _compute_avg_clustering(self, adj_matrix: np.ndarray) -> float:
        """Compute average clustering coefficient."""
        n = adj_matrix.shape[0]
        if n < 3:
            return 0.0

        clustering_sum = 0.0
        count = 0

        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                continue

            # Count triangles
            triangles = 0
            for j in neighbors:
                for l in neighbors:
                    if j < l and adj_matrix[j, l] > 0:
                        triangles += 1

            possible = k * (k - 1) / 2
            if possible > 0:
                clustering_sum += triangles / possible
                count += 1

        return clustering_sum / count if count > 0 else 0.0

    def _compute_avg_path_length(self, adj_matrix: np.ndarray) -> float:
        """Compute average shortest path length using BFS."""
        n = adj_matrix.shape[0]
        if n < 2:
            return 0.0

        total_length = 0
        pairs = 0

        for start in range(n):
            # BFS from start
            distances = [-1] * n
            distances[start] = 0
            queue = [start]
            head = 0

            while head < len(queue):
                current = queue[head]
                head += 1

                for neighbor in range(n):
                    if adj_matrix[current, neighbor] > 0 and distances[neighbor] == -1:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)

            for end in range(n):
                if distances[end] > 0:
                    total_length += distances[end]
                    pairs += 1

        return total_length / pairs if pairs > 0 else 0.0

    def compute_all_metrics(self) -> MultiplexMetrics:
        """Compute metrics for entire multiplex network."""
        layer_metrics = {}
        for layer in PassLayer:
            layer_metrics[layer] = self.compute_layer_metrics(layer)

        # Inter-layer correlation
        correlations = self._compute_inter_layer_correlations()

        # Overall density
        total_edges = len(self.all_edges)
        n_nodes = len(self.nodes)
        max_edges = n_nodes * (n_nodes - 1) * len(PassLayer)
        overall_density = total_edges / max_edges if max_edges > 0 else 0

        # Layer participation per player
        participation = self._compute_layer_participation()

        return MultiplexMetrics(
            layer_metrics=layer_metrics,
            inter_layer_correlation=correlations,
            overall_density=overall_density,
            layer_participation=participation,
        )

    def _compute_inter_layer_correlations(
        self,
    ) -> Dict[Tuple[PassLayer, PassLayer], float]:
        """Compute correlation between layer adjacency matrices."""
        correlations = {}
        layers = list(PassLayer)

        for i, layer1 in enumerate(layers):
            for layer2 in layers[i+1:]:
                matrix1, _ = self.get_adjacency_matrix(layer1)
                matrix2, _ = self.get_adjacency_matrix(layer2)

                # Flatten and compute correlation
                flat1 = matrix1.flatten()
                flat2 = matrix2.flatten()

                if np.std(flat1) > 0 and np.std(flat2) > 0:
                    corr = np.corrcoef(flat1, flat2)[0, 1]
                else:
                    corr = 0.0

                correlations[(layer1, layer2)] = float(corr)

        return correlations

    def _compute_layer_participation(
        self,
    ) -> Dict[str, Dict[PassLayer, float]]:
        """Compute how much each player participates in each layer."""
        participation: Dict[str, Dict[PassLayer, float]] = {}

        for player_id in self.nodes:
            player_part: Dict[PassLayer, float] = {}
            total_passes = 0

            for layer in PassLayer:
                layer_passes = 0
                for (passer, receiver), edges in self.layers[layer].items():
                    if passer == player_id or receiver == player_id:
                        layer_passes += len(edges)
                player_part[layer] = layer_passes
                total_passes += layer_passes

            # Normalize
            if total_passes > 0:
                for layer in PassLayer:
                    player_part[layer] /= total_passes

            participation[player_id] = player_part

        return participation


class PassingStyleClassifier:
    """
    Classifies team/line passing style using multiplex features.
    """

    def __init__(self):
        """Initialize classifier."""
        # Style archetypes
        self.style_profiles = {
            'possession': {
                PassLayer.TAPE_TO_TAPE: 0.4,
                PassLayer.DROP: 0.2,
                PassLayer.CROSS_ICE: 0.15,
                PassLayer.SAUCER: 0.1,
                'density': 0.7,
                'success_rate': 0.85,
            },
            'dump_and_chase': {
                PassLayer.DUMP: 0.35,
                PassLayer.CHIP: 0.2,
                PassLayer.BANK: 0.2,
                PassLayer.TAPE_TO_TAPE: 0.15,
                'density': 0.4,
                'success_rate': 0.7,
            },
            'transition': {
                PassLayer.CHIP: 0.25,
                PassLayer.CROSS_ICE: 0.25,
                PassLayer.TAPE_TO_TAPE: 0.25,
                PassLayer.SAUCER: 0.15,
                'density': 0.5,
                'success_rate': 0.75,
            },
            'cycle': {
                PassLayer.BANK: 0.3,
                PassLayer.BEHIND_NET: 0.25,
                PassLayer.TAPE_TO_TAPE: 0.25,
                PassLayer.DROP: 0.1,
                'density': 0.6,
                'success_rate': 0.8,
            },
        }

    def classify_style(
        self,
        metrics: MultiplexMetrics,
    ) -> Dict[str, Any]:
        """
        Classify team passing style.

        Returns style with confidence scores.
        """
        # Compute layer distribution
        total_edges = sum(m.edge_count for m in metrics.layer_metrics.values())
        if total_edges == 0:
            return {'style': 'unknown', 'confidence': 0.0}

        layer_dist = {
            layer: m.edge_count / total_edges
            for layer, m in metrics.layer_metrics.items()
        }

        # Compute similarity to each profile
        similarities = {}
        for style, profile in self.style_profiles.items():
            sim = 0.0
            weight_sum = 0.0

            for key, target_val in profile.items():
                if isinstance(key, PassLayer):
                    actual_val = layer_dist.get(key, 0)
                    sim += (1 - abs(target_val - actual_val)) * 0.8
                    weight_sum += 0.8
                elif key == 'density':
                    sim += (1 - abs(target_val - metrics.overall_density)) * 0.1
                    weight_sum += 0.1
                elif key == 'success_rate':
                    avg_success = np.mean([
                        m.success_rate for m in metrics.layer_metrics.values()
                    ])
                    sim += (1 - abs(target_val - avg_success)) * 0.1
                    weight_sum += 0.1

            similarities[style] = sim / weight_sum if weight_sum > 0 else 0

        # Best match
        best_style = max(similarities, key=similarities.get)
        confidence = similarities[best_style]

        return {
            'style': best_style,
            'confidence': confidence,
            'all_similarities': similarities,
            'layer_distribution': {l.value: v for l, v in layer_dist.items()},
        }


class PlayOutcomePredictor:
    """
    Predicts play outcomes from network features.

    Based on CMPN paper achieving >90% accuracy.
    """

    def __init__(self, network: MultiplexPassingNetwork):
        """Initialize predictor."""
        self.network = network

        # Feature weights (would be learned in practice)
        np.random.seed(44)
        self.weights = np.random.randn(20) * 0.1

    def extract_sequence_features(
        self,
        edges: List[PassEdge],
    ) -> np.ndarray:
        """Extract features from a passing sequence."""
        if not edges:
            return np.zeros(20)

        features = []

        # Pass count by type
        type_counts = defaultdict(int)
        for edge in edges:
            type_counts[edge.pass_type] += 1

        for layer in PassLayer:
            features.append(type_counts[layer] / max(len(edges), 1))

        # Success rate
        successes = sum(1 for e in edges if e.successful)
        features.append(successes / max(len(edges), 1))

        # Sequence length
        features.append(min(len(edges) / 10, 1.0))

        # Zone progression
        oz_passes = sum(1 for e in edges if e.zone == ZoneType.OFFENSIVE)
        features.append(oz_passes / max(len(edges), 1))

        # Unique passers
        unique_passers = len(set(e.passer_id for e in edges))
        features.append(unique_passers / 6)  # Normalize by max skaters

        # Cross-ice movement
        cross_ice = sum(1 for e in edges if abs(e.end_y - e.start_y) > 30)
        features.append(cross_ice / max(len(edges), 1))

        # Pad to 20 features
        while len(features) < 20:
            features.append(0.0)

        return np.array(features[:20])

    def predict_outcome(
        self,
        sequence: List[PassEdge],
    ) -> Dict[str, float]:
        """
        Predict outcome of passing sequence.

        Outcomes: shot, turnover, zone_exit, continued_possession
        """
        features = self.extract_sequence_features(sequence)

        # Simple logistic regression (in practice, use proper model)
        logits = features @ self.weights[:len(features)]

        # Convert to probabilities
        base_probs = {
            'shot': 0.15,
            'turnover': 0.25,
            'zone_exit': 0.20,
            'continued_possession': 0.40,
        }

        # Adjust based on features
        shot_boost = features[7] * 0.1  # OZ passes
        turnover_penalty = features[8] * 0.05  # Success rate inverse

        probs = {
            'shot': base_probs['shot'] + shot_boost,
            'turnover': base_probs['turnover'] - features[8] * 0.1,
            'zone_exit': base_probs['zone_exit'],
            'continued_possession': base_probs['continued_possession'],
        }

        # Normalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        return probs


class LineChemistryAnalyzer:
    """
    Analyzes line chemistry using multiplex network.
    """

    def __init__(self, network: MultiplexPassingNetwork):
        """Initialize analyzer."""
        self.network = network

    def analyze_line(
        self,
        player_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Analyze chemistry for a specific line.

        Args:
            player_ids: List of player IDs on the line

        Returns:
            Chemistry analysis
        """
        if len(player_ids) < 2:
            return {'error': 'Need at least 2 players'}

        # Get passes within line
        line_set = set(player_ids)
        line_passes: Dict[PassLayer, int] = defaultdict(int)
        successful_passes = 0
        total_passes = 0

        for layer in PassLayer:
            for (passer, receiver), edges in self.network.layers[layer].items():
                if passer in line_set and receiver in line_set:
                    line_passes[layer] += len(edges)
                    for edge in edges:
                        total_passes += 1
                        if edge.successful:
                            successful_passes += 1

        # Chemistry score based on:
        # 1. Pass volume
        # 2. Success rate
        # 3. Pass type diversity
        # 4. Reciprocity

        pass_volume_score = min(total_passes / 50, 1.0)  # 50 passes = max score
        success_rate = successful_passes / total_passes if total_passes > 0 else 0
        diversity = len([l for l, c in line_passes.items() if c > 0]) / len(PassLayer)

        # Reciprocity
        reciprocity = self._compute_reciprocity(player_ids)

        chemistry_score = (
            pass_volume_score * 0.2 +
            success_rate * 0.3 +
            diversity * 0.2 +
            reciprocity * 0.3
        )

        return {
            'chemistry_score': chemistry_score,
            'pass_volume': total_passes,
            'success_rate': success_rate,
            'pass_type_diversity': diversity,
            'reciprocity': reciprocity,
            'primary_pass_types': self._get_primary_types(line_passes),
            'player_connections': self._get_player_connections(player_ids),
        }

    def _compute_reciprocity(self, player_ids: List[str]) -> float:
        """Compute reciprocity among line players."""
        connections = defaultdict(lambda: defaultdict(int))

        for layer in PassLayer:
            for (passer, receiver), edges in self.network.layers[layer].items():
                if passer in player_ids and receiver in player_ids:
                    connections[passer][receiver] += len(edges)

        # Check for reciprocal connections
        reciprocal_pairs = 0
        total_pairs = 0

        for p1 in player_ids:
            for p2 in player_ids:
                if p1 < p2:  # Each pair once
                    total_pairs += 1
                    if connections[p1][p2] > 0 and connections[p2][p1] > 0:
                        reciprocal_pairs += 1

        return reciprocal_pairs / total_pairs if total_pairs > 0 else 0

    def _get_primary_types(
        self,
        pass_counts: Dict[PassLayer, int],
    ) -> List[str]:
        """Get top 3 pass types."""
        sorted_types = sorted(pass_counts.items(), key=lambda x: x[1], reverse=True)
        return [t[0].value for t in sorted_types[:3] if t[1] > 0]

    def _get_player_connections(
        self,
        player_ids: List[str],
    ) -> Dict[str, Dict[str, int]]:
        """Get pass connections between players."""
        connections: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for layer in PassLayer:
            for (passer, receiver), edges in self.network.layers[layer].items():
                if passer in player_ids and receiver in player_ids:
                    connections[passer][receiver] += len(edges)

        # Convert to regular dict
        return {k: dict(v) for k, v in connections.items()}

    def find_optimal_linemates(
        self,
        center_id: str,
        available_wingers: List[str],
        n_combinations: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find optimal linemates for a center.

        Returns top N line combinations by chemistry.
        """
        from itertools import combinations

        if len(available_wingers) < 2:
            return []

        results = []

        for winger_pair in combinations(available_wingers, 2):
            line = [center_id, winger_pair[0], winger_pair[1]]
            analysis = self.analyze_line(line)

            results.append({
                'line': line,
                'chemistry_score': analysis['chemistry_score'],
                'analysis': analysis,
            })

        # Sort by chemistry
        results.sort(key=lambda x: x['chemistry_score'], reverse=True)

        return results[:n_combinations]
