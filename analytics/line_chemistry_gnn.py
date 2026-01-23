"""
Line Chemistry Graph Neural Network for Hockey Analytics

This module implements Graph Neural Networks to analyze line chemistry
and passing network effectiveness, inspired by soccer network analysis.

Key concepts from soccer research:
- Pass networks as graph representations
- Adjacency matrices for player importance
- Flow motifs for identifying playing styles
- Complex Multiplex Passing Network (CMPN) for attack prediction

Hockey translation:
- Line combination effectiveness through passing patterns
- Centrality metrics for playmakers vs. finishers
- Recurring 3-4 player patterns leading to high-danger chances
- Chemistry scores for predicting line performance

References:
- "Using Network Science to Analyse Football Passing Networks" (Frontiers 2018)
- "CMPN: Modeling and analysis using Complex Multiplex Passing Network" (2023)
- Graph Neural Network literature for sports analytics
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class PassType(Enum):
    """Types of passes in hockey."""
    TAPE_TO_TAPE = "tape_to_tape"  # Clean pass to stick
    SAUCER = "saucer"              # Aerial pass over sticks
    BANK = "bank"                  # Off boards
    SHOT_PASS = "shot_pass"        # One-timer setup
    DROP = "drop"                  # Drop pass
    CROSS_ICE = "cross_ice"        # Wide horizontal pass
    STRETCH = "stretch"            # Long outlet pass


class PassOutcome(Enum):
    """Outcome of a pass."""
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    INTERCEPTED = "intercepted"
    SHOT = "shot"              # Led directly to shot
    GOAL = "goal"              # Led directly to goal


@dataclass
class PassEvent:
    """Represents a single pass event."""
    pass_id: str
    timestamp: float
    game_id: str
    period: int

    # Players involved
    passer_id: str
    receiver_id: str
    team_id: str

    # Pass characteristics
    pass_type: PassType
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    pass_distance: float
    pass_speed: Optional[float] = None

    # Outcome
    outcome: PassOutcome = PassOutcome.COMPLETED

    # Context
    zone: str = "neutral"  # offensive, neutral, defensive
    strength_state: str = "5v5"
    time_since_entry: Optional[float] = None
    pressure_level: float = 0.0  # 0-1 scale

    # Value
    xT_value: Optional[float] = None  # Expected Threat change

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerNode:
    """Represents a player in the passing network graph."""
    player_id: str
    team_id: str
    position: str = "F"  # F, D, G
    jersey_number: Optional[int] = None

    # Network metrics
    passes_made: int = 0
    passes_received: int = 0
    pass_completion_rate: float = 0.0
    avg_xT_per_pass: float = 0.0

    # Centrality metrics
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0

    # Role classification
    playmaker_score: float = 0.0
    finisher_score: float = 0.0
    connector_score: float = 0.0


@dataclass
class PassingEdge:
    """Represents a passing connection between two players."""
    passer_id: str
    receiver_id: str

    # Edge attributes
    pass_count: int = 0
    completion_rate: float = 0.0
    avg_distance: float = 0.0
    total_xT: float = 0.0
    shots_generated: int = 0
    goals_generated: int = 0

    # Pattern attributes
    dominant_zones: List[str] = field(default_factory=list)
    dominant_pass_types: List[str] = field(default_factory=list)


class PassingNetwork:
    """
    Graph representation of a hockey team's passing network.

    Nodes represent players, edges represent passing connections
    with attributes like frequency, completion rate, and xT value.
    """

    def __init__(self, team_id: str):
        """
        Initialize the passing network.

        Args:
            team_id: Team identifier
        """
        self.team_id = team_id
        self.nodes: Dict[str, PlayerNode] = {}
        self.edges: Dict[Tuple[str, str], PassingEdge] = {}
        self.passes: List[PassEvent] = []

        # Adjacency matrix (filled when computed)
        self._adjacency_matrix: Optional[np.ndarray] = None
        self._player_index: Dict[str, int] = {}

    def add_player(self, player: PlayerNode):
        """Add a player node to the network."""
        self.nodes[player.player_id] = player

    def add_pass(self, pass_event: PassEvent):
        """Add a pass event to the network."""
        self.passes.append(pass_event)

        # Ensure nodes exist
        if pass_event.passer_id not in self.nodes:
            self.nodes[pass_event.passer_id] = PlayerNode(
                player_id=pass_event.passer_id,
                team_id=pass_event.team_id
            )
        if pass_event.receiver_id not in self.nodes:
            self.nodes[pass_event.receiver_id] = PlayerNode(
                player_id=pass_event.receiver_id,
                team_id=pass_event.team_id
            )

        # Update or create edge
        edge_key = (pass_event.passer_id, pass_event.receiver_id)
        if edge_key not in self.edges:
            self.edges[edge_key] = PassingEdge(
                passer_id=pass_event.passer_id,
                receiver_id=pass_event.receiver_id
            )

        edge = self.edges[edge_key]
        edge.pass_count += 1

        if pass_event.outcome in [PassOutcome.COMPLETED, PassOutcome.SHOT, PassOutcome.GOAL]:
            # Update completion rate
            completed = edge.pass_count * edge.completion_rate + 1
            edge.completion_rate = completed / edge.pass_count
        else:
            completed = edge.pass_count * edge.completion_rate
            edge.completion_rate = completed / edge.pass_count

        # Update distance
        edge.avg_distance = (
            (edge.avg_distance * (edge.pass_count - 1) + pass_event.pass_distance) /
            edge.pass_count
        )

        # Update xT
        if pass_event.xT_value is not None:
            edge.total_xT += pass_event.xT_value

        # Track shots/goals
        if pass_event.outcome == PassOutcome.SHOT:
            edge.shots_generated += 1
        elif pass_event.outcome == PassOutcome.GOAL:
            edge.goals_generated += 1
            edge.shots_generated += 1

        # Track zones and pass types
        if pass_event.zone not in edge.dominant_zones:
            edge.dominant_zones.append(pass_event.zone)
        if pass_event.pass_type.value not in edge.dominant_pass_types:
            edge.dominant_pass_types.append(pass_event.pass_type.value)

        # Invalidate cached adjacency matrix
        self._adjacency_matrix = None

    def build_adjacency_matrix(self) -> np.ndarray:
        """
        Build the adjacency matrix for the network.

        Returns:
            Adjacency matrix (n_players x n_players)
        """
        players = list(self.nodes.keys())
        n = len(players)
        self._player_index = {p: i for i, p in enumerate(players)}

        # Create matrix
        matrix = np.zeros((n, n))

        for (passer, receiver), edge in self.edges.items():
            if passer in self._player_index and receiver in self._player_index:
                i = self._player_index[passer]
                j = self._player_index[receiver]
                matrix[i, j] = edge.pass_count

        self._adjacency_matrix = matrix
        return matrix

    def compute_centrality_metrics(self):
        """Compute centrality metrics for all nodes."""
        if self._adjacency_matrix is None:
            self.build_adjacency_matrix()

        n = len(self.nodes)
        if n == 0:
            return

        adj = self._adjacency_matrix

        # Normalize adjacency for calculations
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adj_norm = adj / row_sums

        players = list(self.nodes.keys())

        for i, player_id in enumerate(players):
            node = self.nodes[player_id]

            # Degree centrality (normalized pass count)
            out_degree = adj[i, :].sum()
            in_degree = adj[:, i].sum()
            total_degree = out_degree + in_degree
            max_degree = adj.sum() / n if n > 1 else 1
            node.degree_centrality = total_degree / max(max_degree, 1)

            # Closeness centrality (inverse of average path length)
            # Simplified: average distance to other players
            distances = self._compute_distances_from(i, adj)
            reachable = distances[distances < float('inf')]
            if len(reachable) > 1:
                node.closeness_centrality = (len(reachable) - 1) / sum(reachable[reachable > 0])
            else:
                node.closeness_centrality = 0

            # Betweenness centrality (how often player is on shortest path)
            node.betweenness_centrality = self._compute_betweenness(i, adj)

            # Update pass stats
            node.passes_made = int(out_degree)
            node.passes_received = int(in_degree)

            # Completion rate from edges
            total_passes = 0
            completed_passes = 0
            for (passer, _), edge in self.edges.items():
                if passer == player_id:
                    total_passes += edge.pass_count
                    completed_passes += edge.pass_count * edge.completion_rate
            node.pass_completion_rate = completed_passes / max(total_passes, 1)

        # Eigenvector centrality (importance based on neighbor importance)
        self._compute_eigenvector_centrality()

    def _compute_distances_from(self, source: int, adj: np.ndarray) -> np.ndarray:
        """Compute shortest path distances from source node."""
        n = adj.shape[0]
        distances = np.full(n, float('inf'))
        distances[source] = 0

        # BFS-like approach on weighted graph
        visited = set()
        current = [source]

        while current:
            next_level = []
            for node in current:
                if node in visited:
                    continue
                visited.add(node)

                for neighbor in range(n):
                    if adj[node, neighbor] > 0:
                        new_dist = distances[node] + 1
                        if new_dist < distances[neighbor]:
                            distances[neighbor] = new_dist
                            next_level.append(neighbor)

            current = next_level

        return distances

    def _compute_betweenness(self, node: int, adj: np.ndarray) -> float:
        """Compute betweenness centrality for a node (simplified)."""
        n = adj.shape[0]
        if n < 3:
            return 0.0

        # Count paths through this node
        paths_through = 0
        total_paths = 0

        for source in range(n):
            if source == node:
                continue
            for target in range(n):
                if target == node or target == source:
                    continue

                # Check if path from source to target goes through node
                # Simplified: check if direct edges exist
                if adj[source, node] > 0 and adj[node, target] > 0:
                    paths_through += 1

                if adj[source, target] > 0 or (adj[source, node] > 0 and adj[node, target] > 0):
                    total_paths += 1

        return paths_through / max(total_paths, 1)

    def _compute_eigenvector_centrality(self, iterations: int = 100, tol: float = 1e-6):
        """Compute eigenvector centrality using power iteration."""
        if self._adjacency_matrix is None:
            return

        adj = self._adjacency_matrix.copy()
        n = adj.shape[0]

        if n == 0:
            return

        # Make symmetric for undirected interpretation
        adj_sym = adj + adj.T

        # Normalize
        row_sums = adj_sym.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adj_norm = adj_sym / row_sums

        # Power iteration
        x = np.ones(n) / n

        for _ in range(iterations):
            x_new = adj_norm @ x
            norm = np.linalg.norm(x_new)
            if norm > 0:
                x_new = x_new / norm

            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new

        # Assign to nodes
        players = list(self.nodes.keys())
        for i, player_id in enumerate(players):
            self.nodes[player_id].eigenvector_centrality = x[i]

    def classify_player_roles(self):
        """
        Classify players into roles based on network metrics.

        Roles:
        - Playmaker: High assist potential, high centrality
        - Finisher: Receives passes in dangerous areas
        - Connector: Links different parts of the network
        """
        # First compute centrality if not done
        if any(node.degree_centrality == 0 for node in self.nodes.values()):
            self.compute_centrality_metrics()

        for player_id, node in self.nodes.items():
            # Playmaker score: high out-degree, high betweenness
            playmaker_factors = [
                node.degree_centrality * 0.3,
                node.betweenness_centrality * 0.3,
                node.pass_completion_rate * 0.2,
                node.passes_made / max(node.passes_received, 1) * 0.2 * 0.5
            ]
            node.playmaker_score = sum(playmaker_factors)

            # Finisher score: receives passes in offensive zone, near goal
            # Using xT as proxy for dangerous area
            finish_xT = sum(
                edge.total_xT for (_, receiver), edge in self.edges.items()
                if receiver == player_id
            )
            shots_from_passes = sum(
                edge.shots_generated for (_, receiver), edge in self.edges.items()
                if receiver == player_id
            )
            finish_factors = [
                min(finish_xT / 10.0, 0.5),
                min(shots_from_passes / 20.0, 0.5),
            ]
            node.finisher_score = sum(finish_factors)

            # Connector score: high betweenness, connects different clusters
            connector_factors = [
                node.betweenness_centrality * 0.5,
                node.closeness_centrality * 0.3,
                (1 - abs(node.passes_made - node.passes_received) /
                 max(node.passes_made + node.passes_received, 1)) * 0.2
            ]
            node.connector_score = sum(connector_factors)

    def find_passing_motifs(
        self,
        min_occurrences: int = 3,
        max_players: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Find recurring passing motifs (patterns).

        Motifs are common sequences of passes between specific players
        that lead to scoring chances.

        Args:
            min_occurrences: Minimum times pattern must occur
            max_players: Maximum players in a motif

        Returns:
            List of motif dictionaries
        """
        # Group passes into possessions (simplified by timestamp gaps)
        possessions = []
        current_possession = []

        sorted_passes = sorted(self.passes, key=lambda p: p.timestamp)

        for pass_event in sorted_passes:
            if current_possession:
                time_gap = pass_event.timestamp - current_possession[-1].timestamp
                if time_gap > 10.0:  # New possession if >10s gap
                    if len(current_possession) >= 2:
                        possessions.append(current_possession)
                    current_possession = []

            current_possession.append(pass_event)

        if len(current_possession) >= 2:
            possessions.append(current_possession)

        # Extract player sequences from possessions
        sequences = []
        for poss in possessions:
            players = [poss[0].passer_id]
            for p in poss:
                if p.receiver_id not in players:
                    players.append(p.receiver_id)
                if len(players) > max_players:
                    break

            # Track outcome
            final_outcome = poss[-1].outcome

            sequences.append({
                'players': tuple(players[:max_players]),
                'shot': final_outcome in [PassOutcome.SHOT, PassOutcome.GOAL],
                'goal': final_outcome == PassOutcome.GOAL,
            })

        # Count motif occurrences
        motif_counts = defaultdict(lambda: {'count': 0, 'shots': 0, 'goals': 0})

        for seq in sequences:
            key = seq['players']
            motif_counts[key]['count'] += 1
            if seq['shot']:
                motif_counts[key]['shots'] += 1
            if seq['goal']:
                motif_counts[key]['goals'] += 1

        # Filter and format motifs
        motifs = []
        for players, stats in motif_counts.items():
            if stats['count'] >= min_occurrences:
                motifs.append({
                    'players': list(players),
                    'occurrences': stats['count'],
                    'shots': stats['shots'],
                    'goals': stats['goals'],
                    'shot_rate': stats['shots'] / stats['count'],
                    'goal_rate': stats['goals'] / stats['count'],
                })

        # Sort by effectiveness
        motifs.sort(key=lambda m: m['goal_rate'], reverse=True)

        return motifs

    def get_chemistry_score(
        self,
        player_ids: List[str]
    ) -> Dict[str, float]:
        """
        Calculate chemistry score for a group of players.

        Chemistry is measured by:
        - Pass completion rate between players
        - xT generation from passes
        - Shots generated from combinations

        Args:
            player_ids: List of player IDs to evaluate

        Returns:
            Dictionary with chemistry metrics
        """
        # Get all edges between these players
        relevant_edges = []
        for (passer, receiver), edge in self.edges.items():
            if passer in player_ids and receiver in player_ids:
                relevant_edges.append(edge)

        if not relevant_edges:
            return {
                'chemistry_score': 0.0,
                'avg_completion_rate': 0.0,
                'total_xT': 0.0,
                'shots_generated': 0,
                'goals_generated': 0,
            }

        # Aggregate metrics
        total_passes = sum(e.pass_count for e in relevant_edges)
        weighted_completion = sum(
            e.pass_count * e.completion_rate for e in relevant_edges
        )
        total_xT = sum(e.total_xT for e in relevant_edges)
        shots = sum(e.shots_generated for e in relevant_edges)
        goals = sum(e.goals_generated for e in relevant_edges)

        avg_completion = weighted_completion / max(total_passes, 1)

        # Chemistry score: weighted combination
        chemistry_score = (
            0.3 * avg_completion +
            0.3 * min(total_xT / (total_passes * 0.05 + 1), 1.0) +  # xT per pass
            0.2 * min(shots / (total_passes * 0.1 + 1), 1.0) +       # Shot generation
            0.2 * min(goals / (shots * 0.1 + 1), 1.0)               # Conversion
        )

        return {
            'chemistry_score': chemistry_score,
            'avg_completion_rate': avg_completion,
            'total_xT': total_xT,
            'total_passes': total_passes,
            'shots_generated': shots,
            'goals_generated': goals,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export network to dictionary."""
        return {
            'team_id': self.team_id,
            'nodes': {
                pid: {
                    'player_id': node.player_id,
                    'position': node.position,
                    'passes_made': node.passes_made,
                    'passes_received': node.passes_received,
                    'degree_centrality': node.degree_centrality,
                    'betweenness_centrality': node.betweenness_centrality,
                    'playmaker_score': node.playmaker_score,
                    'finisher_score': node.finisher_score,
                    'connector_score': node.connector_score,
                }
                for pid, node in self.nodes.items()
            },
            'edges': {
                f"{passer}_{receiver}": {
                    'pass_count': edge.pass_count,
                    'completion_rate': edge.completion_rate,
                    'total_xT': edge.total_xT,
                    'shots_generated': edge.shots_generated,
                    'goals_generated': edge.goals_generated,
                }
                for (passer, receiver), edge in self.edges.items()
            },
            'adjacency_matrix': self._adjacency_matrix.tolist() if self._adjacency_matrix is not None else None,
        }


class LineChemistryPredictor:
    """
    Predicts line chemistry using network analysis and optional GNN features.

    Uses historical passing network data to predict how well a line
    combination will perform together.
    """

    def __init__(self):
        """Initialize the chemistry predictor."""
        self.networks: Dict[str, PassingNetwork] = {}
        self.historical_chemistry: Dict[Tuple[str, ...], Dict[str, float]] = {}

    def add_network(self, network: PassingNetwork):
        """Add a passing network for analysis."""
        self.networks[network.team_id] = network

    def predict_chemistry(
        self,
        player_ids: List[str],
        team_id: str
    ) -> Dict[str, Any]:
        """
        Predict chemistry for a proposed line combination.

        Args:
            player_ids: List of player IDs for the line (3-5 players)
            team_id: Team identifier

        Returns:
            Predicted chemistry metrics
        """
        if team_id not in self.networks:
            return {'error': 'No network data for team'}

        network = self.networks[team_id]

        # Get observed chemistry if players have played together
        observed = network.get_chemistry_score(player_ids)

        # Get individual player metrics
        player_metrics = []
        for pid in player_ids:
            if pid in network.nodes:
                node = network.nodes[pid]
                player_metrics.append({
                    'player_id': pid,
                    'playmaker': node.playmaker_score,
                    'finisher': node.finisher_score,
                    'connector': node.connector_score,
                })

        # Predict synergy based on role balance
        if player_metrics:
            avg_playmaker = np.mean([p['playmaker'] for p in player_metrics])
            avg_finisher = np.mean([p['finisher'] for p in player_metrics])
            avg_connector = np.mean([p['connector'] for p in player_metrics])

            # Best lines have role diversity
            role_diversity = np.std([avg_playmaker, avg_finisher, avg_connector])

            # Synergy score: high diversity + observed chemistry
            synergy = (
                0.3 * role_diversity +
                0.7 * observed.get('chemistry_score', 0.5)
            )
        else:
            synergy = 0.5

        return {
            'predicted_chemistry': synergy,
            'observed_chemistry': observed.get('chemistry_score', None),
            'role_balance': {
                'playmaker_avg': avg_playmaker if player_metrics else 0,
                'finisher_avg': avg_finisher if player_metrics else 0,
                'connector_avg': avg_connector if player_metrics else 0,
            },
            'players': player_metrics,
            'recommendation': 'good' if synergy > 0.6 else 'fair' if synergy > 0.4 else 'poor',
        }

    def suggest_line_combinations(
        self,
        available_players: List[str],
        team_id: str,
        line_size: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Suggest optimal line combinations from available players.

        Args:
            available_players: List of available player IDs
            team_id: Team identifier
            line_size: Number of players per line (default 3 for forward line)

        Returns:
            List of suggested combinations sorted by predicted chemistry
        """
        from itertools import combinations

        suggestions = []

        for combo in combinations(available_players, line_size):
            prediction = self.predict_chemistry(list(combo), team_id)
            suggestions.append({
                'players': list(combo),
                'predicted_chemistry': prediction['predicted_chemistry'],
                'role_balance': prediction['role_balance'],
                'recommendation': prediction['recommendation'],
            })

        # Sort by chemistry
        suggestions.sort(key=lambda x: x['predicted_chemistry'], reverse=True)

        return suggestions[:10]  # Top 10 combinations


def create_sample_passing_data(n_passes: int = 1000) -> List[PassEvent]:
    """
    Create sample passing data for testing.

    Returns:
        List of PassEvent objects
    """
    np.random.seed(42)

    passes = []

    # Define some players
    players = [f"player_{i}" for i in range(12)]

    for i in range(n_passes):
        passer = np.random.choice(players[:6])  # Forward group
        # Bias receiver selection based on passer
        if np.random.random() < 0.7:
            # Pass within forward group
            receiver = np.random.choice([p for p in players[:6] if p != passer])
        else:
            # Pass to defense
            receiver = np.random.choice(players[6:])

        # Generate pass characteristics
        start_x = np.random.uniform(0, 200)
        start_y = np.random.uniform(0, 85)
        end_x = start_x + np.random.uniform(-30, 40)
        end_y = start_y + np.random.uniform(-20, 20)
        end_x = np.clip(end_x, 0, 200)
        end_y = np.clip(end_y, 0, 85)

        distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

        # Outcome based on distance and zone
        completion_prob = 0.85 - distance * 0.003
        outcome = (PassOutcome.COMPLETED if np.random.random() < completion_prob
                   else PassOutcome.INCOMPLETE)

        # Chance of shot if in offensive zone
        if end_x > 150 and outcome == PassOutcome.COMPLETED and np.random.random() < 0.15:
            outcome = PassOutcome.SHOT
            if np.random.random() < 0.1:
                outcome = PassOutcome.GOAL

        # Determine zone
        if end_x > 150:
            zone = "offensive"
        elif end_x < 50:
            zone = "defensive"
        else:
            zone = "neutral"

        pass_event = PassEvent(
            pass_id=f"pass_{i}",
            timestamp=i * 5.0,  # 5 seconds between passes on average
            game_id="game_001",
            period=np.random.randint(1, 4),
            passer_id=passer,
            receiver_id=receiver,
            team_id="team_a",
            pass_type=np.random.choice(list(PassType)),
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            pass_distance=distance,
            outcome=outcome,
            zone=zone,
            xT_value=np.random.uniform(-0.02, 0.1) if outcome != PassOutcome.INCOMPLETE else 0,
        )

        passes.append(pass_event)

    return passes


if __name__ == "__main__":
    # Demo the Line Chemistry GNN
    print("Creating Passing Network...")

    # Create network
    network = PassingNetwork(team_id="team_a")

    # Add sample data
    print("Generating sample passes...")
    passes = create_sample_passing_data(1000)

    for pass_event in passes:
        network.add_pass(pass_event)

    # Compute metrics
    print("Computing network metrics...")
    network.compute_centrality_metrics()
    network.classify_player_roles()

    # Display player metrics
    print("\nPlayer Network Metrics:")
    for pid, node in sorted(network.nodes.items(), key=lambda x: x[1].degree_centrality, reverse=True)[:6]:
        print(f"  {pid}:")
        print(f"    Passes made/received: {node.passes_made}/{node.passes_received}")
        print(f"    Degree centrality: {node.degree_centrality:.3f}")
        print(f"    Betweenness: {node.betweenness_centrality:.3f}")
        print(f"    Playmaker score: {node.playmaker_score:.3f}")
        print(f"    Finisher score: {node.finisher_score:.3f}")

    # Find motifs
    print("\nPassing Motifs:")
    motifs = network.find_passing_motifs(min_occurrences=3)
    for motif in motifs[:5]:
        print(f"  {' -> '.join(motif['players'])}: "
              f"{motif['occurrences']} times, "
              f"{motif['shot_rate']:.2%} shot rate, "
              f"{motif['goal_rate']:.2%} goal rate")

    # Chemistry scores
    print("\nLine Chemistry Scores:")
    forward_lines = [
        ["player_0", "player_1", "player_2"],
        ["player_3", "player_4", "player_5"],
    ]

    for line in forward_lines:
        chemistry = network.get_chemistry_score(line)
        print(f"  {', '.join(line)}:")
        print(f"    Chemistry score: {chemistry['chemistry_score']:.3f}")
        print(f"    Completion rate: {chemistry['avg_completion_rate']:.2%}")
        print(f"    xT generated: {chemistry['total_xT']:.3f}")

    # Line suggestions
    print("\nLine Combination Suggestions:")
    predictor = LineChemistryPredictor()
    predictor.add_network(network)

    suggestions = predictor.suggest_line_combinations(
        available_players=[f"player_{i}" for i in range(6)],
        team_id="team_a",
        line_size=3
    )

    for i, suggestion in enumerate(suggestions[:3]):
        print(f"  #{i+1}: {', '.join(suggestion['players'])}")
        print(f"      Chemistry: {suggestion['predicted_chemistry']:.3f}")
        print(f"      Recommendation: {suggestion['recommendation']}")
