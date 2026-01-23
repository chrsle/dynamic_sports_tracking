"""
CNN Phase of Play Detection

Implements phase of play classification using CNN architecture,
translated from soccer research.

Key Paper:
- "What Happens to Your Team's Formation During a Match? A Tactical
  Deep Dive." (2025). Bundesliga tracking data analysis.

Key Concepts:
- CNN classifies phases of play (build-up, mid-block, high-block, etc.)
- Formation detected within each phase
- Hierarchical clustering for formation archetypes
- 7 seasons of tracking data analysis

Hockey Translation:
- Phase detection: offensive zone, neutral zone, defensive zone, transition
- Formation within each zone (forecheck, neutral zone trap, box, etc.)
- Forecheck vs. trap identification
- Breakout pattern classification
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class GamePhase(Enum):
    """High-level game phases."""
    OFFENSIVE_ZONE = "oz"
    NEUTRAL_ZONE = "nz"
    DEFENSIVE_ZONE = "dz"
    OFFENSIVE_TRANSITION = "o_trans"
    DEFENSIVE_TRANSITION = "d_trans"
    FACEOFF = "faceoff"
    SPECIAL_TEAMS = "special"


class OffensiveZonePhase(Enum):
    """Sub-phases in offensive zone."""
    CYCLE = "cycle"
    HIGH_POSSESSION = "high_possession"
    POINT_PLAY = "point_play"
    NET_FRONT = "net_front"
    RUSH_ATTACK = "rush_attack"
    SCRAMBLE = "scramble"


class DefensiveZonePhase(Enum):
    """Sub-phases in defensive zone."""
    BOX = "box"
    DIAMOND = "diamond"
    ROTATION = "rotation"
    CLEAR_ATTEMPT = "clear_attempt"
    UNDER_PRESSURE = "under_pressure"
    CONTROLLED = "controlled"


class NeutralZonePhase(Enum):
    """Sub-phases in neutral zone."""
    FORECHECK_PRESSURE = "forecheck_pressure"
    TRAP = "trap"
    CONTROLLED_ENTRY = "controlled_entry"
    DUMP_CHASE = "dump_chase"
    REGROUP = "regroup"


class ForecheckSystem(Enum):
    """Forechecking systems."""
    ONE_TWO_TWO = "1-2-2"
    TWO_ONE_TWO = "2-1-2"
    ONE_THREE_ONE = "1-3-1"
    TWO_THREE = "2-3"
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"


@dataclass
class TrackingSnapshot:
    """Single frame of tracking data."""
    timestamp: float
    home_positions: List[Tuple[str, float, float]]  # id, x, y
    away_positions: List[Tuple[str, float, float]]
    puck_x: float
    puck_y: float
    puck_carrier_team: str  # 'home' or 'away'
    game_state: str  # '5v5', 'pp', 'pk', etc.


@dataclass
class PhaseDetection:
    """Result of phase detection."""
    primary_phase: GamePhase
    sub_phase: Optional[str]
    confidence: float
    formation: Optional[str]
    phase_duration: float
    transition_probability: float


@dataclass
class PhaseSequence:
    """Sequence of detected phases."""
    phases: List[PhaseDetection]
    total_duration: float
    phase_distribution: Dict[GamePhase, float]


class PhaseClassifierCNN:
    """
    CNN-based phase classifier for hockey.

    Uses player positions to classify current phase of play.
    """

    def __init__(
        self,
        grid_size: int = 20,
        hidden_dim: int = 64,
    ):
        """Initialize classifier."""
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim

        # Initialize weights
        np.random.seed(42)
        self.conv_weights = {
            'W1': np.random.randn(3, 3, 6, 16) * 0.1,
            'W2': np.random.randn(3, 3, 16, 32) * 0.1,
        }
        self.fc_weights = {
            'W1': np.random.randn(32 * 5 * 5, hidden_dim) * 0.1,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, len(GamePhase)) * 0.1,
            'b2': np.zeros(len(GamePhase)),
        }

        # Phase to index mapping
        self.phase_to_idx = {p: i for i, p in enumerate(GamePhase)}
        self.idx_to_phase = {i: p for p, i in self.phase_to_idx.items()}

    def snapshot_to_grid(
        self,
        snapshot: TrackingSnapshot,
    ) -> np.ndarray:
        """
        Convert snapshot to grid representation.

        Channels:
        0: Home player positions
        1: Away player positions
        2: Puck position
        3: Puck carrier indicator
        4: Home velocity (simplified)
        5: Away velocity (simplified)
        """
        grid = np.zeros((self.grid_size, self.grid_size, 6))

        def to_grid(x, y):
            gx = int(np.clip(x / 200 * self.grid_size, 0, self.grid_size - 1))
            gy = int(np.clip(y / 85 * self.grid_size, 0, self.grid_size - 1))
            return gx, gy

        # Home positions (channel 0)
        for _, x, y in snapshot.home_positions:
            gx, gy = to_grid(x, y)
            grid[gy, gx, 0] = 1.0

        # Away positions (channel 1)
        for _, x, y in snapshot.away_positions:
            gx, gy = to_grid(x, y)
            grid[gy, gx, 1] = 1.0

        # Puck (channel 2)
        px, py = to_grid(snapshot.puck_x, snapshot.puck_y)
        grid[py, px, 2] = 1.0

        # Carrier team (channel 3)
        if snapshot.puck_carrier_team == 'home':
            grid[py, px, 3] = 1.0
        else:
            grid[py, px, 4] = -1.0

        return grid

    def forward(self, grid: np.ndarray) -> np.ndarray:
        """Forward pass through CNN (simplified)."""
        # Global average pooling as simplified CNN
        pooled = grid.mean(axis=(0, 1))  # (6,)

        # Add zone features
        puck_zone = self._get_puck_zone(grid)

        features = np.concatenate([pooled, puck_zone])

        # FC layers
        h1 = np.maximum(0, features @ self.fc_weights['W1'][:len(features)] +
                        self.fc_weights['b1'])
        logits = h1 @ self.fc_weights['W2'] + self.fc_weights['b2']

        return logits

    def _get_puck_zone(self, grid: np.ndarray) -> np.ndarray:
        """Determine puck zone from grid."""
        puck_channel = grid[:, :, 2]
        puck_pos = np.argmax(puck_channel)
        puck_x = puck_pos % self.grid_size

        zone_features = np.zeros(3)
        if puck_x < self.grid_size * 0.25:
            zone_features[0] = 1.0  # Defensive
        elif puck_x > self.grid_size * 0.75:
            zone_features[2] = 1.0  # Offensive
        else:
            zone_features[1] = 1.0  # Neutral

        return zone_features

    def classify(
        self,
        snapshot: TrackingSnapshot,
    ) -> PhaseDetection:
        """Classify phase from snapshot."""
        grid = self.snapshot_to_grid(snapshot)
        logits = self.forward(grid)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Get primary phase
        primary_idx = np.argmax(probs)
        primary_phase = self.idx_to_phase[primary_idx]
        confidence = float(probs[primary_idx])

        # Detect sub-phase
        sub_phase = self._detect_sub_phase(snapshot, primary_phase)

        # Detect formation
        formation = self._detect_formation(snapshot, primary_phase)

        # Transition probability
        trans_prob = 1.0 - confidence

        return PhaseDetection(
            primary_phase=primary_phase,
            sub_phase=sub_phase,
            confidence=confidence,
            formation=formation,
            phase_duration=0.0,  # Will be set by sequence analyzer
            transition_probability=trans_prob,
        )

    def _detect_sub_phase(
        self,
        snapshot: TrackingSnapshot,
        primary_phase: GamePhase,
    ) -> Optional[str]:
        """Detect sub-phase within primary phase."""
        if primary_phase == GamePhase.OFFENSIVE_ZONE:
            return self._detect_oz_subphase(snapshot)
        elif primary_phase == GamePhase.DEFENSIVE_ZONE:
            return self._detect_dz_subphase(snapshot)
        elif primary_phase == GamePhase.NEUTRAL_ZONE:
            return self._detect_nz_subphase(snapshot)
        return None

    def _detect_oz_subphase(self, snapshot: TrackingSnapshot) -> str:
        """Detect offensive zone sub-phase."""
        # Simplified detection based on puck and player positions
        puck_x, puck_y = snapshot.puck_x, snapshot.puck_y

        # Behind net?
        if puck_x > 189:
            return OffensiveZonePhase.CYCLE.value

        # In slot?
        if puck_x > 170 and 25 < puck_y < 60:
            return OffensiveZonePhase.NET_FRONT.value

        # At point?
        if puck_x < 175:
            return OffensiveZonePhase.POINT_PLAY.value

        return OffensiveZonePhase.HIGH_POSSESSION.value

    def _detect_dz_subphase(self, snapshot: TrackingSnapshot) -> str:
        """Detect defensive zone sub-phase."""
        # Count players in dangerous areas
        home_in_slot = sum(
            1 for _, x, y in snapshot.home_positions
            if x < 30 and 25 < y < 60
        )

        if home_in_slot >= 3:
            return DefensiveZonePhase.BOX.value
        elif home_in_slot >= 2:
            return DefensiveZonePhase.DIAMOND.value
        else:
            return DefensiveZonePhase.ROTATION.value

    def _detect_nz_subphase(self, snapshot: TrackingSnapshot) -> str:
        """Detect neutral zone sub-phase."""
        # Check player distribution
        carrier_team = snapshot.puck_carrier_team

        if carrier_team == 'home':
            # Home has puck - check if facing pressure
            away_in_nz = sum(
                1 for _, x, y in snapshot.away_positions
                if 75 < x < 125
            )
            if away_in_nz >= 3:
                return NeutralZonePhase.TRAP.value
            else:
                return NeutralZonePhase.CONTROLLED_ENTRY.value
        else:
            home_in_nz = sum(
                1 for _, x, y in snapshot.home_positions
                if 75 < x < 125
            )
            if home_in_nz >= 3:
                return NeutralZonePhase.TRAP.value
            else:
                return NeutralZonePhase.FORECHECK_PRESSURE.value

    def _detect_formation(
        self,
        snapshot: TrackingSnapshot,
        primary_phase: GamePhase,
    ) -> Optional[str]:
        """Detect formation/system being used."""
        if primary_phase in [GamePhase.OFFENSIVE_ZONE, GamePhase.OFFENSIVE_TRANSITION]:
            return self._detect_forecheck(snapshot)
        elif primary_phase == GamePhase.NEUTRAL_ZONE:
            return self._detect_nz_system(snapshot)
        return None

    def _detect_forecheck(self, snapshot: TrackingSnapshot) -> str:
        """Detect forechecking system."""
        # Simplified: count players at different depths
        away_positions = snapshot.away_positions  # Forechecking team

        deep = sum(1 for _, x, _ in away_positions if x > 175)
        mid = sum(1 for _, x, _ in away_positions if 150 < x <= 175)
        high = sum(1 for _, x, _ in away_positions if x <= 150)

        if deep >= 2:
            return ForecheckSystem.AGGRESSIVE.value
        elif deep == 1 and mid >= 2:
            return ForecheckSystem.ONE_TWO_TWO.value
        elif deep >= 2 and mid == 1:
            return ForecheckSystem.TWO_ONE_TWO.value
        else:
            return ForecheckSystem.PASSIVE.value

    def _detect_nz_system(self, snapshot: TrackingSnapshot) -> str:
        """Detect neutral zone system."""
        # Check defensive team positioning
        home_spread = np.std([y for _, _, y in snapshot.home_positions])

        if home_spread > 25:
            return "spread"
        else:
            return "compact"


class PhaseSequenceAnalyzer:
    """
    Analyzes sequences of phases over time.
    """

    def __init__(self, classifier: PhaseClassifierCNN):
        """Initialize analyzer."""
        self.classifier = classifier

    def analyze_sequence(
        self,
        snapshots: List[TrackingSnapshot],
        min_phase_duration: float = 1.0,
    ) -> PhaseSequence:
        """
        Analyze sequence of snapshots.

        Merges consecutive same phases and computes statistics.
        """
        if not snapshots:
            return PhaseSequence(
                phases=[],
                total_duration=0.0,
                phase_distribution={},
            )

        # Classify each snapshot
        raw_phases = [self.classifier.classify(s) for s in snapshots]

        # Merge consecutive same phases
        merged_phases = []
        current_phase = raw_phases[0]
        phase_start = snapshots[0].timestamp
        phase_count = 1

        for i, (phase, snapshot) in enumerate(zip(raw_phases[1:], snapshots[1:]), 1):
            if phase.primary_phase == current_phase.primary_phase:
                phase_count += 1
            else:
                # Save current phase
                duration = snapshot.timestamp - phase_start
                if duration >= min_phase_duration:
                    current_phase.phase_duration = duration
                    merged_phases.append(current_phase)

                # Start new phase
                current_phase = phase
                phase_start = snapshot.timestamp
                phase_count = 1

        # Don't forget last phase
        if snapshots:
            duration = snapshots[-1].timestamp - phase_start
            current_phase.phase_duration = duration
            merged_phases.append(current_phase)

        # Calculate distribution
        total_duration = sum(p.phase_duration for p in merged_phases)
        distribution: Dict[GamePhase, float] = defaultdict(float)

        for phase in merged_phases:
            distribution[phase.primary_phase] += phase.phase_duration

        if total_duration > 0:
            distribution = {k: v / total_duration for k, v in distribution.items()}

        return PhaseSequence(
            phases=merged_phases,
            total_duration=total_duration,
            phase_distribution=dict(distribution),
        )

    def detect_transition_points(
        self,
        sequence: PhaseSequence,
    ) -> List[Dict[str, Any]]:
        """
        Detect phase transition points.

        Returns list of transitions with context.
        """
        transitions = []

        for i in range(len(sequence.phases) - 1):
            current = sequence.phases[i]
            next_phase = sequence.phases[i + 1]

            transitions.append({
                'from_phase': current.primary_phase.value,
                'to_phase': next_phase.primary_phase.value,
                'from_duration': current.phase_duration,
                'transition_confidence': 1 - current.transition_probability,
                'from_formation': current.formation,
                'to_formation': next_phase.formation,
            })

        return transitions


class PhaseBasedAnalytics:
    """
    Analytics based on phase detection.
    """

    def __init__(self, analyzer: PhaseSequenceAnalyzer):
        """Initialize analytics."""
        self.analyzer = analyzer
        self.phase_stats: Dict[str, List[Dict]] = defaultdict(list)

    def record_sequence(
        self,
        sequence: PhaseSequence,
        outcome: str,  # 'goal_for', 'goal_against', 'no_goal'
        team: str,
    ):
        """Record a sequence for analysis."""
        for phase in sequence.phases:
            self.phase_stats[phase.primary_phase.value].append({
                'duration': phase.phase_duration,
                'sub_phase': phase.sub_phase,
                'formation': phase.formation,
                'outcome': outcome,
                'team': team,
            })

    def get_phase_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Get effectiveness metrics by phase."""
        results = {}

        for phase, events in self.phase_stats.items():
            if not events:
                continue

            n = len(events)
            goals_for = sum(1 for e in events if e['outcome'] == 'goal_for')
            goals_against = sum(1 for e in events if e['outcome'] == 'goal_against')

            results[phase] = {
                'total_events': n,
                'avg_duration': np.mean([e['duration'] for e in events]),
                'goal_for_rate': goals_for / n,
                'goal_against_rate': goals_against / n,
                'net_rate': (goals_for - goals_against) / n,
            }

        return results

    def get_formation_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Get effectiveness by formation."""
        formation_stats: Dict[str, List[Dict]] = defaultdict(list)

        for events in self.phase_stats.values():
            for event in events:
                if event['formation']:
                    formation_stats[event['formation']].append(event)

        results = {}
        for formation, events in formation_stats.items():
            if not events:
                continue

            n = len(events)
            goals_for = sum(1 for e in events if e['outcome'] == 'goal_for')
            goals_against = sum(1 for e in events if e['outcome'] == 'goal_against')

            results[formation] = {
                'count': n,
                'goal_for_rate': goals_for / n,
                'goal_against_rate': goals_against / n,
            }

        return results

    def get_transition_analysis(
        self,
        sequences: List[PhaseSequence],
    ) -> Dict[str, Any]:
        """Analyze transitions across sequences."""
        transition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        transition_outcomes: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for sequence in sequences:
            for i in range(len(sequence.phases) - 1):
                from_phase = sequence.phases[i].primary_phase.value
                to_phase = sequence.phases[i + 1].primary_phase.value
                transition_counts[(from_phase, to_phase)] += 1

        # Find most common transitions
        sorted_transitions = sorted(
            transition_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'most_common': [
                {'from': t[0][0], 'to': t[0][1], 'count': t[1]}
                for t in sorted_transitions[:10]
            ],
            'total_transitions': sum(transition_counts.values()),
        }
