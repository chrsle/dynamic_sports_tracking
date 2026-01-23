"""
Play Pattern Recognition for Hockey Analytics

This module implements group activity recognition for identifying hockey plays
and tactical patterns from tracking data.

Recognizes:
- Offensive plays: cycle, give-and-go, dump-and-chase, crash the net
- Defensive schemes: man-to-man, zone, box-out
- Special teams patterns: PP setups, PK formations
- Transition plays: odd-man rushes, breakouts

References:
    - Hauri, S., & Vucetic, S. (2022). "NETS: Neural Embeddings in Team Sports."
      arXiv:2209.00451.
    - "Multi-task Learning for Sports Visual Tracking." arXiv:2401.09942
    - Wang, Z., et al. (2020). "Play2Vec: Sports Play Retrieval." KDD.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class OffensivePlay(Enum):
    """Offensive play types."""
    CYCLE = "cycle"
    GIVE_AND_GO = "give_and_go"
    DUMP_AND_CHASE = "dump_and_chase"
    CONTROLLED_ENTRY = "controlled_entry"
    CRASH_NET = "crash_net"
    PERIMETER_PASSING = "perimeter_passing"
    SCREEN_SHOT = "screen_shot"
    DEFLECTION_ATTEMPT = "deflection_attempt"
    ONE_TIMER_SETUP = "one_timer_setup"
    RUSH_ATTACK = "rush_attack"
    BREAKAWAY = "breakaway"
    TWO_ON_ONE = "2_on_1"
    THREE_ON_TWO = "3_on_2"
    ODD_MAN_RUSH = "odd_man_rush"


class DefensivePlay(Enum):
    """Defensive play types."""
    GAP_CONTROL = "gap_control"
    PUCK_PRESSURE = "puck_pressure"
    PASSING_LANE_BLOCK = "passing_lane_block"
    BOX_OUT = "box_out"
    SHOT_BLOCK = "shot_block"
    STICK_CHECK = "stick_check"
    BODY_CHECK = "body_check"
    CLEAR_ATTEMPT = "clear_attempt"
    BREAKOUT_START = "breakout_start"


class SpecialTeamsPlay(Enum):
    """Special teams play types."""
    PP_UMBRELLA = "pp_umbrella"
    PP_OVERLOAD = "pp_overload"
    PP_ONE_THREE_ONE = "pp_1_3_1"
    PP_HALF_WALL = "pp_half_wall"
    PK_BOX = "pk_box"
    PK_DIAMOND = "pk_diamond"
    PK_AGGRESSIVE = "pk_aggressive"
    PK_PASSIVE = "pk_passive"


@dataclass
class PlayerFrame:
    """Single player observation in a frame."""
    player_id: str
    team: str  # 'home' or 'away'
    x: float
    y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    has_puck: bool = False


@dataclass
class GameFrame:
    """Complete game state at a moment."""
    timestamp: float
    period: int
    game_time: float  # Seconds into period
    players: List[PlayerFrame]
    puck_x: float
    puck_y: float
    puck_velocity_x: float = 0.0
    puck_velocity_y: float = 0.0
    strength_state: str = "5v5"
    home_score: int = 0
    away_score: int = 0


@dataclass
class PlayRecognition:
    """Recognized play with metadata."""
    play_type: str
    category: str  # 'offensive', 'defensive', 'special_teams'
    confidence: float
    start_time: float
    end_time: Optional[float]
    involved_players: List[str]
    key_events: List[Dict[str, Any]]
    success: Optional[bool] = None
    quality_score: float = 0.0


@dataclass
class PlaySequence:
    """Sequence of frames comprising a play."""
    frames: List[GameFrame]
    play_recognition: Optional[PlayRecognition] = None


class PlayPatternMatcher:
    """
    Template-based play pattern matching.

    Matches observed sequences against known play patterns
    using spatial and temporal features.
    """

    # Pattern templates defined as movement signatures
    PATTERNS = {
        OffensivePlay.CYCLE: {
            'zone': 'offensive',
            'puck_path': 'circular',
            'min_passes': 2,
            'behind_net': True,
            'duration_range': (3.0, 15.0)
        },
        OffensivePlay.GIVE_AND_GO: {
            'zone': 'any',
            'pass_return': True,
            'player_movement': 'forward_after_pass',
            'duration_range': (1.0, 4.0)
        },
        OffensivePlay.DUMP_AND_CHASE: {
            'zone_transition': 'nz_to_oz',
            'dump_event': True,
            'chase_pattern': True,
            'duration_range': (2.0, 8.0)
        },
        OffensivePlay.TWO_ON_ONE: {
            'attackers': 2,
            'defenders': 1,
            'rush_speed': 'high',
            'zone': 'offensive',
            'duration_range': (2.0, 6.0)
        }
    }

    def __init__(self, match_threshold: float = 0.6):
        """
        Initialize pattern matcher.

        Args:
            match_threshold: Minimum confidence for pattern match
        """
        self.threshold = match_threshold

    def match_pattern(
        self,
        sequence: PlaySequence,
        pattern_type: OffensivePlay
    ) -> Tuple[bool, float]:
        """
        Check if sequence matches a specific pattern.

        Args:
            sequence: Frame sequence to analyze
            pattern_type: Pattern to match against

        Returns:
            Tuple of (matches, confidence)
        """
        if pattern_type not in self.PATTERNS:
            return False, 0.0

        template = self.PATTERNS[pattern_type]
        scores = []

        # Check each template criterion
        if 'zone' in template:
            zone_score = self._check_zone(sequence, template['zone'])
            scores.append(zone_score)

        if 'puck_path' in template:
            path_score = self._check_puck_path(sequence, template['puck_path'])
            scores.append(path_score)

        if 'min_passes' in template:
            pass_score = self._check_passes(sequence, template['min_passes'])
            scores.append(pass_score)

        if 'behind_net' in template:
            behind_score = self._check_behind_net(sequence)
            scores.append(behind_score)

        if 'attackers' in template and 'defenders' in template:
            advantage_score = self._check_man_advantage(
                sequence, template['attackers'], template['defenders']
            )
            scores.append(advantage_score)

        if not scores:
            return False, 0.0

        confidence = np.mean(scores)
        return confidence >= self.threshold, confidence

    def _check_zone(self, sequence: PlaySequence, expected_zone: str) -> float:
        """Check if play occurs in expected zone."""
        if not sequence.frames:
            return 0.0

        zone_frames = 0
        for frame in sequence.frames:
            puck_x = frame.puck_x

            in_zone = False
            if expected_zone == 'offensive' and puck_x > 25:
                in_zone = True
            elif expected_zone == 'defensive' and puck_x < -25:
                in_zone = True
            elif expected_zone == 'neutral' and -25 <= puck_x <= 25:
                in_zone = True
            elif expected_zone == 'any':
                in_zone = True

            if in_zone:
                zone_frames += 1

        return zone_frames / len(sequence.frames)

    def _check_puck_path(self, sequence: PlaySequence, path_type: str) -> float:
        """Check puck movement pattern."""
        if len(sequence.frames) < 3:
            return 0.0

        puck_positions = [(f.puck_x, f.puck_y) for f in sequence.frames]

        if path_type == 'circular':
            # Check for circular motion (cycle play)
            return self._detect_circular_motion(puck_positions)
        elif path_type == 'linear':
            return self._detect_linear_motion(puck_positions)

        return 0.5

    def _detect_circular_motion(self, positions: List[Tuple[float, float]]) -> float:
        """Detect circular puck movement pattern."""
        if len(positions) < 5:
            return 0.0

        # Calculate angle changes
        angles = []
        for i in range(1, len(positions) - 1):
            v1 = (positions[i][0] - positions[i-1][0],
                  positions[i][1] - positions[i-1][1])
            v2 = (positions[i+1][0] - positions[i][0],
                  positions[i+1][1] - positions[i][1])

            # Angle between vectors
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)

            if mag1 > 0.1 and mag2 > 0.1:
                cos_angle = np.clip(dot / (mag1 * mag2), -1, 1)
                angles.append(np.arccos(cos_angle))

        if not angles:
            return 0.0

        # Circular motion has consistent turning
        avg_angle = np.mean(angles)
        angle_variance = np.var(angles)

        # Good circular motion: avg angle ~0.3-0.8 rad, low variance
        if 0.2 <= avg_angle <= 1.0 and angle_variance < 0.3:
            return 0.8
        elif angle_variance < 0.5:
            return 0.5

        return 0.2

    def _detect_linear_motion(self, positions: List[Tuple[float, float]]) -> float:
        """Detect linear puck movement."""
        if len(positions) < 2:
            return 0.0

        # Fit line and calculate R-squared
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        if np.std(xs) < 0.1:
            return 0.3  # Vertical line

        # Simple linear fit
        m = np.cov(xs, ys)[0, 1] / np.var(xs)
        b = np.mean(ys) - m * np.mean(xs)

        # Calculate residuals
        residuals = [abs(y - (m * x + b)) for x, y in positions]
        avg_residual = np.mean(residuals)

        # Low residuals = linear motion
        return max(0, 1 - avg_residual / 20)

    def _check_passes(self, sequence: PlaySequence, min_passes: int) -> float:
        """Check for minimum number of passes."""
        # Detect passes by puck carrier changes
        pass_count = 0
        prev_carrier = None

        for frame in sequence.frames:
            carrier = None
            for player in frame.players:
                if player.has_puck:
                    carrier = player.player_id
                    break

            if carrier and prev_carrier and carrier != prev_carrier:
                pass_count += 1

            prev_carrier = carrier

        if pass_count >= min_passes:
            return 1.0
        elif pass_count > 0:
            return pass_count / min_passes

        return 0.0

    def _check_behind_net(self, sequence: PlaySequence) -> float:
        """Check if play involves behind-the-net area."""
        behind_net_frames = 0

        for frame in sequence.frames:
            # Behind net: x > 89 or x < -89
            if frame.puck_x > 89 or frame.puck_x < -89:
                behind_net_frames += 1

        return behind_net_frames / len(sequence.frames) if sequence.frames else 0.0

    def _check_man_advantage(
        self,
        sequence: PlaySequence,
        expected_attackers: int,
        expected_defenders: int
    ) -> float:
        """Check for man advantage situation (odd-man rush)."""
        if not sequence.frames:
            return 0.0

        advantage_frames = 0

        for frame in sequence.frames:
            # Determine zone
            puck_x = frame.puck_x

            # Count players in zone
            home_in_zone = 0
            away_in_zone = 0

            for player in frame.players:
                if puck_x > 0:  # Offensive zone for one team
                    if player.x > 25:
                        if player.team == 'home':
                            home_in_zone += 1
                        else:
                            away_in_zone += 1

            # Check for expected advantage
            puck_carrier_team = None
            for player in frame.players:
                if player.has_puck:
                    puck_carrier_team = player.team
                    break

            if puck_carrier_team == 'home':
                if home_in_zone >= expected_attackers and away_in_zone <= expected_defenders:
                    advantage_frames += 1
            elif puck_carrier_team == 'away':
                if away_in_zone >= expected_attackers and home_in_zone <= expected_defenders:
                    advantage_frames += 1

        return advantage_frames / len(sequence.frames)


class NETSPlayRecognizer:
    """
    Neural Embeddings in Team Sports (NETS) inspired play recognizer.

    Uses sequence modeling to recognize plays from tracking data.
    Based on Hauri & Vucetic (2022).
    """

    def __init__(
        self,
        sequence_length: int = 30,  # ~1 second at 30fps
        embedding_dim: int = 64,
        n_play_types: int = 20
    ):
        """
        Initialize NETS recognizer.

        Args:
            sequence_length: Number of frames per sequence
            embedding_dim: Dimension of play embeddings
            n_play_types: Number of play types to classify
        """
        self.seq_length = sequence_length
        self.embedding_dim = embedding_dim
        self.n_plays = n_play_types

        # Play type mappings
        self.play_types = list(OffensivePlay) + list(DefensivePlay)

        # Simplified feature extraction (would be neural network in full impl)
        self.feature_dim = 50  # Players * features per player

    def extract_features(self, frame: GameFrame) -> np.ndarray:
        """
        Extract feature vector from a game frame.

        Creates a fixed-size representation of the game state.
        """
        features = []

        # Puck features
        features.extend([
            frame.puck_x / 100,
            frame.puck_y / 50,
            frame.puck_velocity_x / 50,
            frame.puck_velocity_y / 50
        ])

        # Player features (sorted by x position for consistency)
        home_players = sorted(
            [p for p in frame.players if p.team == 'home'],
            key=lambda p: p.x
        )
        away_players = sorted(
            [p for p in frame.players if p.team == 'away'],
            key=lambda p: p.x
        )

        # Extract features for up to 5 players per team
        for players in [home_players, away_players]:
            for i in range(5):
                if i < len(players):
                    p = players[i]
                    features.extend([
                        p.x / 100,
                        p.y / 50,
                        p.velocity_x / 30,
                        p.velocity_y / 30,
                        1.0 if p.has_puck else 0.0
                    ])
                else:
                    features.extend([0.0] * 5)

        # Pad or truncate to feature_dim
        features = features[:self.feature_dim]
        while len(features) < self.feature_dim:
            features.append(0.0)

        return np.array(features)

    def extract_sequence_features(
        self,
        sequence: PlaySequence
    ) -> np.ndarray:
        """
        Extract features from a sequence of frames.

        Returns:
            Feature matrix of shape (seq_length, feature_dim)
        """
        features = []

        for frame in sequence.frames[-self.seq_length:]:
            features.append(self.extract_features(frame))

        # Pad if needed
        while len(features) < self.seq_length:
            features.insert(0, np.zeros(self.feature_dim))

        return np.array(features)

    def compute_play_scores(
        self,
        sequence: PlaySequence
    ) -> Dict[str, float]:
        """
        Compute confidence scores for each play type.

        Uses simplified rule-based scoring (would be neural network in production).

        Args:
            sequence: Play sequence to classify

        Returns:
            Dictionary mapping play types to confidence scores
        """
        features = self.extract_sequence_features(sequence)
        scores = {}

        # Compute features for classification
        avg_puck_x = np.mean([f.puck_x for f in sequence.frames])
        avg_puck_y = np.mean([f.puck_y for f in sequence.frames])
        puck_x_std = np.std([f.puck_x for f in sequence.frames])
        puck_y_std = np.std([f.puck_y for f in sequence.frames])

        # Count pass events
        pass_count = 0
        prev_carrier = None
        for frame in sequence.frames:
            carrier = next((p.player_id for p in frame.players if p.has_puck), None)
            if carrier and prev_carrier and carrier != prev_carrier:
                pass_count += 1
            prev_carrier = carrier

        # Compute speeds
        speeds = []
        for frame in sequence.frames:
            for player in frame.players:
                speed = np.sqrt(player.velocity_x**2 + player.velocity_y**2)
                speeds.append(speed)
        avg_speed = np.mean(speeds) if speeds else 0

        # Score each play type
        for play in OffensivePlay:
            if play == OffensivePlay.CYCLE:
                # Cycle: in OZ, circular movement, multiple passes
                score = 0.0
                if avg_puck_x > 50:
                    score += 0.3
                if puck_x_std < 20 and puck_y_std > 10:
                    score += 0.3
                if pass_count >= 2:
                    score += 0.4
                scores[play.value] = score

            elif play == OffensivePlay.DUMP_AND_CHASE:
                # Dump: starts in NZ, ends in OZ, high speed
                frames = sequence.frames
                if frames:
                    start_x = frames[0].puck_x
                    end_x = frames[-1].puck_x
                    if start_x < 25 < end_x and avg_speed > 15:
                        scores[play.value] = 0.7
                    else:
                        scores[play.value] = 0.2
                else:
                    scores[play.value] = 0.0

            elif play == OffensivePlay.TWO_ON_ONE:
                # 2-on-1: 2 attackers vs 1 defender in OZ
                score = 0.0
                for frame in sequence.frames:
                    if frame.puck_x > 50:
                        attackers = sum(1 for p in frame.players
                                       if p.team == 'home' and p.x > 50)
                        defenders = sum(1 for p in frame.players
                                       if p.team == 'away' and p.x > 50)
                        if attackers == 2 and defenders == 1:
                            score = max(score, 0.8)
                scores[play.value] = score

            elif play == OffensivePlay.RUSH_ATTACK:
                # Rush: high speed, forward movement
                if avg_speed > 20 and puck_x_std > 30:
                    scores[play.value] = 0.7
                else:
                    scores[play.value] = 0.2

            else:
                scores[play.value] = 0.3  # Default

        return scores

    def recognize_play(
        self,
        sequence: PlaySequence
    ) -> PlayRecognition:
        """
        Recognize the play type from a sequence.

        Args:
            sequence: Play sequence

        Returns:
            PlayRecognition with best match
        """
        scores = self.compute_play_scores(sequence)

        # Find best match
        best_play = max(scores, key=scores.get)
        confidence = scores[best_play]

        # Get involved players
        involved = set()
        for frame in sequence.frames:
            for player in frame.players:
                if player.has_puck:
                    involved.add(player.player_id)

        return PlayRecognition(
            play_type=best_play,
            category='offensive',
            confidence=confidence,
            start_time=sequence.frames[0].timestamp if sequence.frames else 0,
            end_time=sequence.frames[-1].timestamp if sequence.frames else None,
            involved_players=list(involved),
            key_events=[],
            quality_score=confidence
        )


class PlaySimilaritySearch:
    """
    Play2Vec-inspired play retrieval system.

    Find similar plays from a database of historical plays.
    Based on Wang et al. (2020).
    """

    def __init__(self, embedding_dim: int = 32):
        """
        Initialize similarity search.

        Args:
            embedding_dim: Dimension of play embeddings
        """
        self.embedding_dim = embedding_dim
        self.play_database: List[Dict] = []
        self.embeddings: List[np.ndarray] = []

    def embed_play(self, sequence: PlaySequence) -> np.ndarray:
        """
        Create embedding vector for a play.

        Args:
            sequence: Play sequence

        Returns:
            Play embedding vector
        """
        if not sequence.frames:
            return np.zeros(self.embedding_dim)

        # Extract summary statistics as embedding
        features = []

        # Spatial features
        puck_positions = [(f.puck_x, f.puck_y) for f in sequence.frames]
        features.extend([
            np.mean([p[0] for p in puck_positions]) / 100,
            np.mean([p[1] for p in puck_positions]) / 50,
            np.std([p[0] for p in puck_positions]) / 50,
            np.std([p[1] for p in puck_positions]) / 25,
            (puck_positions[-1][0] - puck_positions[0][0]) / 100,  # Net x movement
            (puck_positions[-1][1] - puck_positions[0][1]) / 50,   # Net y movement
        ])

        # Temporal features
        duration = sequence.frames[-1].timestamp - sequence.frames[0].timestamp
        features.append(min(duration / 10, 1.0))

        # Team features
        home_positions = []
        away_positions = []
        for frame in sequence.frames:
            for player in frame.players:
                if player.team == 'home':
                    home_positions.append((player.x, player.y))
                else:
                    away_positions.append((player.x, player.y))

        if home_positions:
            features.extend([
                np.mean([p[0] for p in home_positions]) / 100,
                np.std([p[0] for p in home_positions]) / 50
            ])
        else:
            features.extend([0.0, 0.0])

        if away_positions:
            features.extend([
                np.mean([p[0] for p in away_positions]) / 100,
                np.std([p[0] for p in away_positions]) / 50
            ])
        else:
            features.extend([0.0, 0.0])

        # Pass count
        pass_count = 0
        prev_carrier = None
        for frame in sequence.frames:
            carrier = next((p.player_id for p in frame.players if p.has_puck), None)
            if carrier and prev_carrier and carrier != prev_carrier:
                pass_count += 1
            prev_carrier = carrier
        features.append(min(pass_count / 5, 1.0))

        # Pad or truncate to embedding_dim
        features = features[:self.embedding_dim]
        while len(features) < self.embedding_dim:
            features.append(0.0)

        return np.array(features)

    def add_play(
        self,
        sequence: PlaySequence,
        metadata: Dict[str, Any]
    ):
        """
        Add a play to the database.

        Args:
            sequence: Play sequence
            metadata: Additional metadata (game_id, result, etc.)
        """
        embedding = self.embed_play(sequence)

        self.play_database.append({
            'sequence': sequence,
            'metadata': metadata
        })
        self.embeddings.append(embedding)

    def find_similar(
        self,
        query_sequence: PlaySequence,
        n: int = 5
    ) -> List[Dict]:
        """
        Find similar plays in database.

        Args:
            query_sequence: Query play
            n: Number of results to return

        Returns:
            List of similar plays with similarity scores
        """
        query_emb = self.embed_play(query_sequence)

        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_emb, emb)
            similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])

        results = []
        for idx, sim in similarities[:n]:
            results.append({
                'play': self.play_database[idx],
                'similarity': sim
            })

        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class PlayEffectivenessAnalyzer:
    """
    Analyze play effectiveness and success rates.
    """

    def __init__(self):
        """Initialize analyzer."""
        self.play_outcomes: Dict[str, List[Dict]] = {}

    def record_outcome(
        self,
        play_type: str,
        success: bool,
        xg_generated: float,
        goals_scored: int,
        context: Dict[str, Any]
    ):
        """
        Record outcome of a play.

        Args:
            play_type: Type of play
            success: Whether play was successful
            xg_generated: Expected goals generated
            goals_scored: Actual goals scored
            context: Additional context (strength, score, etc.)
        """
        if play_type not in self.play_outcomes:
            self.play_outcomes[play_type] = []

        self.play_outcomes[play_type].append({
            'success': success,
            'xg': xg_generated,
            'goals': goals_scored,
            'context': context
        })

    def get_play_statistics(self, play_type: str) -> Dict[str, float]:
        """
        Get statistics for a play type.

        Args:
            play_type: Play type to analyze

        Returns:
            Dictionary with success rate, avg xG, etc.
        """
        if play_type not in self.play_outcomes:
            return {}

        outcomes = self.play_outcomes[play_type]
        n = len(outcomes)

        if n == 0:
            return {}

        return {
            'count': n,
            'success_rate': sum(1 for o in outcomes if o['success']) / n,
            'avg_xg': np.mean([o['xg'] for o in outcomes]),
            'goals_per_play': sum(o['goals'] for o in outcomes) / n,
            'conversion_rate': sum(o['goals'] for o in outcomes) / sum(o['xg'] for o in outcomes) if sum(o['xg'] for o in outcomes) > 0 else 0
        }

    def compare_plays(self) -> List[Dict]:
        """
        Compare effectiveness across all play types.

        Returns:
            Sorted list of play type statistics
        """
        stats = []
        for play_type in self.play_outcomes:
            play_stats = self.get_play_statistics(play_type)
            play_stats['play_type'] = play_type
            stats.append(play_stats)

        # Sort by xG per play
        stats.sort(key=lambda x: -x.get('avg_xg', 0))
        return stats
