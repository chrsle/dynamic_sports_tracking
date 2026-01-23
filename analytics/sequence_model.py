"""
Sequence Modeling for Hockey Events

Implements methodology from:
- Wei, X., et al. (2016). "Rally Analyzer: Sequential Point Modeling
  for Tennis Match Analytics."

Key concept: Model sequences of events as Markov chains or
neural sequence models to predict outcomes and find patterns.

Hockey translation:
- Faceoff sequence modeling
- Shift event sequences
- Power play progression modeling
- Zone entry sequences
- Shot sequence analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Sequence
import numpy as np
from datetime import datetime


class EventType(Enum):
    """Types of hockey events."""
    FACEOFF_WIN = "faceoff_win"
    FACEOFF_LOSS = "faceoff_loss"
    SHOT = "shot"
    SHOT_BLOCKED = "shot_blocked"
    SHOT_MISSED = "shot_missed"
    GOAL = "goal"
    PASS_COMPLETE = "pass_complete"
    PASS_INCOMPLETE = "pass_incomplete"
    CARRY = "carry"
    DUMP_IN = "dump_in"
    DUMP_OUT = "dump_out"
    HIT = "hit"
    TAKEAWAY = "takeaway"
    GIVEAWAY = "giveaway"
    PENALTY = "penalty"
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    ICING = "icing"
    OFFSIDE = "offside"


class ZoneState(Enum):
    """Zone possession states."""
    OZ_CONTROLLED = "oz_controlled"  # Offensive zone control
    OZ_CONTESTED = "oz_contested"
    NZ_OFFENSE = "nz_offense"  # Neutral zone moving forward
    NZ_DEFENSE = "nz_defense"  # Neutral zone moving back
    DZ_CONTROLLED = "dz_controlled"  # Defensive zone control
    DZ_CONTESTED = "dz_contested"


@dataclass
class SequenceEvent:
    """Single event in a sequence."""
    event_type: EventType
    timestamp: float
    player_id: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    zone: Optional[ZoneState] = None
    outcome_value: float = 0.0  # Calculated value


@dataclass
class EventSequence:
    """Complete sequence of events."""
    events: List[SequenceEvent]
    start_time: float
    end_time: float
    terminal_event: EventType  # How sequence ended
    outcome: str  # "goal", "shot", "turnover", "stoppage"
    total_value: float = 0.0


@dataclass
class TransitionMatrix:
    """Markov chain transition matrix."""
    states: List[str]
    matrix: np.ndarray  # shape (n_states, n_states)
    absorbing_states: List[str]


@dataclass
class SequencePattern:
    """Identified pattern in sequences."""
    pattern: Tuple[EventType, ...]
    frequency: int
    success_rate: float  # Rate of positive outcomes
    avg_value: float
    example_sequences: List[int]  # Indices of example sequences


class MarkovSequenceModel:
    """
    Markov chain model for event sequences.

    Models probability of transitioning between states.
    """

    def __init__(self):
        self.states = [e.value for e in EventType]
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

        # Initialize uniform transition matrix
        self.transition_matrix = np.ones((self.n_states, self.n_states))
        self.transition_matrix /= self.n_states

        # Absorbing states (sequence terminators)
        self.absorbing = {EventType.GOAL.value, EventType.ICING.value,
                         EventType.OFFSIDE.value, EventType.PENALTY.value}

    def fit(self, sequences: List[EventSequence]):
        """Learn transition probabilities from data."""
        # Count transitions
        counts = np.zeros((self.n_states, self.n_states))

        for seq in sequences:
            for i in range(len(seq.events) - 1):
                curr = seq.events[i].event_type.value
                next_event = seq.events[i + 1].event_type.value

                if curr in self.state_to_idx and next_event in self.state_to_idx:
                    counts[self.state_to_idx[curr], self.state_to_idx[next_event]] += 1

        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = counts / row_sums

    def transition_probability(
        self,
        from_state: EventType,
        to_state: EventType
    ) -> float:
        """Get probability of transitioning between states."""
        i = self.state_to_idx.get(from_state.value, 0)
        j = self.state_to_idx.get(to_state.value, 0)
        return self.transition_matrix[i, j]

    def most_likely_next(
        self,
        current_state: EventType,
        n: int = 3
    ) -> List[Tuple[EventType, float]]:
        """Get n most likely next events."""
        i = self.state_to_idx.get(current_state.value, 0)
        probs = self.transition_matrix[i]

        # Get top n
        top_indices = np.argsort(probs)[-n:][::-1]
        results = []

        for idx in top_indices:
            event = EventType(self.states[idx])
            results.append((event, probs[idx]))

        return results

    def simulate_sequence(
        self,
        start_state: EventType,
        max_length: int = 20
    ) -> List[EventType]:
        """Simulate a sequence from starting state."""
        sequence = [start_state]
        current = start_state

        for _ in range(max_length - 1):
            if current.value in self.absorbing:
                break

            i = self.state_to_idx[current.value]
            probs = self.transition_matrix[i]
            next_idx = np.random.choice(self.n_states, p=probs)
            current = EventType(self.states[next_idx])
            sequence.append(current)

        return sequence

    def sequence_probability(
        self,
        sequence: List[EventType]
    ) -> float:
        """Calculate probability of observing sequence."""
        if len(sequence) < 2:
            return 1.0

        prob = 1.0
        for i in range(len(sequence) - 1):
            prob *= self.transition_probability(sequence[i], sequence[i + 1])

        return prob


class FaceoffSequenceAnalyzer:
    """
    Specialized analyzer for faceoff sequences.

    Models what happens after faceoff wins/losses.
    """

    def __init__(self):
        self.win_model = MarkovSequenceModel()
        self.loss_model = MarkovSequenceModel()

        # Zone-specific models
        self.zone_models = {
            'offensive': MarkovSequenceModel(),
            'defensive': MarkovSequenceModel(),
            'neutral': MarkovSequenceModel()
        }

    def fit(
        self,
        win_sequences: List[EventSequence],
        loss_sequences: List[EventSequence]
    ):
        """Train models on faceoff outcome sequences."""
        self.win_model.fit(win_sequences)
        self.loss_model.fit(loss_sequences)

    def analyze_faceoff_impact(
        self,
        zone: str
    ) -> Dict[str, float]:
        """Analyze impact of faceoff win vs loss in zone."""
        # Simulate many sequences
        n_sims = 1000

        # Win outcomes
        win_goals = 0
        win_shots = 0

        for _ in range(n_sims):
            seq = self.win_model.simulate_sequence(EventType.FACEOFF_WIN)
            if EventType.GOAL in seq:
                win_goals += 1
            if EventType.SHOT in seq:
                win_shots += 1

        # Loss outcomes
        loss_goals = 0
        loss_shots = 0

        for _ in range(n_sims):
            seq = self.loss_model.simulate_sequence(EventType.FACEOFF_LOSS)
            if EventType.GOAL in seq:
                loss_goals += 1
            if EventType.SHOT in seq:
                loss_shots += 1

        return {
            'win_goal_rate': win_goals / n_sims,
            'loss_goal_rate': loss_goals / n_sims,
            'win_shot_rate': win_shots / n_sims,
            'loss_shot_rate': loss_shots / n_sims,
            'goal_rate_difference': (win_goals - loss_goals) / n_sims,
            'zone': zone
        }


class LSTMSequenceModel:
    """
    LSTM-based sequence model for event prediction.

    More sophisticated than Markov for capturing long-range dependencies.
    """

    def __init__(
        self,
        n_events: int = len(EventType),
        hidden_dim: int = 64,
        embedding_dim: int = 32
    ):
        self.n_events = n_events
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Embedding layer
        self.embedding = np.random.randn(n_events, embedding_dim) * 0.1

        # LSTM weights
        input_dim = embedding_dim
        self.W_lstm = np.random.randn(input_dim, hidden_dim * 4) * 0.1
        self.U_lstm = np.random.randn(hidden_dim, hidden_dim * 4) * 0.1
        self.b_lstm = np.zeros(hidden_dim * 4)

        # Output layer
        self.W_out = np.random.randn(hidden_dim, n_events) * 0.1
        self.b_out = np.zeros(n_events)

    def encode_sequence(
        self,
        events: List[EventType]
    ) -> np.ndarray:
        """Encode sequence to hidden state."""
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        event_types = list(EventType)

        for event in events:
            idx = event_types.index(event)
            x = self.embedding[idx]

            # LSTM step
            gates = x @ self.W_lstm + h @ self.U_lstm + self.b_lstm
            i, f, o, g = np.split(gates, 4)

            i = 1 / (1 + np.exp(-np.clip(i, -500, 500)))
            f = 1 / (1 + np.exp(-np.clip(f, -500, 500)))
            o = 1 / (1 + np.exp(-np.clip(o, -500, 500)))
            g = np.tanh(g)

            c = f * c + i * g
            h = o * np.tanh(c)

        return h

    def predict_next(
        self,
        events: List[EventType]
    ) -> Tuple[EventType, float]:
        """Predict next event in sequence."""
        h = self.encode_sequence(events)

        # Output probabilities
        logits = h @ self.W_out + self.b_out
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        event_types = list(EventType)
        idx = np.argmax(probs)

        return event_types[idx], probs[idx]

    def sequence_embedding(
        self,
        events: List[EventType]
    ) -> np.ndarray:
        """Get embedding for full sequence."""
        return self.encode_sequence(events)


class PatternMiner:
    """
    Mine frequent patterns from event sequences.
    """

    def __init__(
        self,
        min_support: int = 10,
        max_pattern_length: int = 5
    ):
        self.min_support = min_support
        self.max_length = max_pattern_length

    def find_patterns(
        self,
        sequences: List[EventSequence]
    ) -> List[SequencePattern]:
        """Find frequent patterns in sequences."""
        # Count all subsequences
        pattern_counts: Dict[Tuple, List[int]] = {}

        for seq_idx, seq in enumerate(sequences):
            event_types = tuple(e.event_type for e in seq.events)

            # Generate all subsequences
            for length in range(2, min(len(event_types) + 1, self.max_length + 1)):
                for start in range(len(event_types) - length + 1):
                    pattern = event_types[start:start + length]

                    if pattern not in pattern_counts:
                        pattern_counts[pattern] = []
                    pattern_counts[pattern].append(seq_idx)

        # Filter by support
        patterns = []
        for pattern, indices in pattern_counts.items():
            if len(indices) >= self.min_support:
                # Calculate success rate
                successes = sum(
                    1 for i in indices
                    if sequences[i].outcome in ['goal', 'shot']
                )
                success_rate = successes / len(indices)

                # Average value
                avg_value = np.mean([
                    sequences[i].total_value for i in indices
                ])

                patterns.append(SequencePattern(
                    pattern=pattern,
                    frequency=len(indices),
                    success_rate=success_rate,
                    avg_value=avg_value,
                    example_sequences=indices[:5]
                ))

        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns

    def find_successful_patterns(
        self,
        sequences: List[EventSequence],
        min_success_rate: float = 0.3
    ) -> List[SequencePattern]:
        """Find patterns that lead to successful outcomes."""
        all_patterns = self.find_patterns(sequences)
        return [
            p for p in all_patterns
            if p.success_rate >= min_success_rate
        ]


class SequenceValueCalculator:
    """
    Calculate value of event sequences.

    Uses outcome probabilities and position values.
    """

    def __init__(self):
        # Base values by event type
        self.event_values = {
            EventType.GOAL: 1.0,
            EventType.SHOT: 0.08,
            EventType.SHOT_BLOCKED: 0.03,
            EventType.SHOT_MISSED: 0.02,
            EventType.PASS_COMPLETE: 0.02,
            EventType.PASS_INCOMPLETE: -0.01,
            EventType.CARRY: 0.01,
            EventType.ZONE_ENTRY: 0.04,
            EventType.ZONE_EXIT: -0.03,
            EventType.TAKEAWAY: 0.05,
            EventType.GIVEAWAY: -0.05,
            EventType.FACEOFF_WIN: 0.015,
            EventType.FACEOFF_LOSS: -0.01,
            EventType.HIT: 0.005,
            EventType.DUMP_IN: 0.01,
            EventType.DUMP_OUT: -0.02,
            EventType.PENALTY: -0.1,
            EventType.ICING: -0.02,
            EventType.OFFSIDE: -0.01,
        }

    def calculate_sequence_value(
        self,
        sequence: EventSequence
    ) -> float:
        """Calculate total value of sequence."""
        total = 0.0

        for event in sequence.events:
            base_value = self.event_values.get(event.event_type, 0)

            # Position modifier
            if event.x is not None:
                # Higher value in offensive zone
                position_mult = 1 + event.x / 200  # Normalized
                base_value *= position_mult

            total += base_value

        sequence.total_value = total
        return total

    def calculate_expected_value(
        self,
        model: MarkovSequenceModel,
        current_state: EventType,
        horizon: int = 10
    ) -> float:
        """Calculate expected value from current state."""
        # Monte Carlo simulation
        n_sims = 500
        total_value = 0

        for _ in range(n_sims):
            seq = model.simulate_sequence(current_state, max_length=horizon)
            for event in seq:
                total_value += self.event_values.get(event, 0)

        return total_value / n_sims


class PowerPlaySequenceAnalyzer:
    """
    Specialized analyzer for power play sequences.
    """

    def __init__(self):
        self.sequence_model = MarkovSequenceModel()
        self.pattern_miner = PatternMiner(min_support=5)
        self.value_calc = SequenceValueCalculator()

    def analyze_power_play(
        self,
        sequences: List[EventSequence]
    ) -> Dict:
        """Comprehensive power play sequence analysis."""
        # Fit model
        self.sequence_model.fit(sequences)

        # Find patterns
        all_patterns = self.pattern_miner.find_patterns(sequences)
        successful_patterns = self.pattern_miner.find_successful_patterns(sequences)

        # Calculate values
        for seq in sequences:
            self.value_calc.calculate_sequence_value(seq)

        avg_value = np.mean([seq.total_value for seq in sequences])

        # Entry analysis
        entry_outcomes = {}
        for seq in sequences:
            first_event = seq.events[0].event_type if seq.events else None
            if first_event:
                if first_event not in entry_outcomes:
                    entry_outcomes[first_event] = {'count': 0, 'goals': 0}
                entry_outcomes[first_event]['count'] += 1
                if seq.outcome == 'goal':
                    entry_outcomes[first_event]['goals'] += 1

        return {
            'total_sequences': len(sequences),
            'avg_value': avg_value,
            'top_patterns': all_patterns[:10],
            'successful_patterns': successful_patterns[:5],
            'entry_analysis': {
                k: {
                    'count': v['count'],
                    'goal_rate': v['goals'] / v['count'] if v['count'] > 0 else 0
                }
                for k, v in entry_outcomes.items()
            },
            'goal_rate': sum(1 for s in sequences if s.outcome == 'goal') / len(sequences),
            'shot_rate': sum(1 for s in sequences if s.outcome in ['goal', 'shot']) / len(sequences)
        }


class ShiftSequenceAnalyzer:
    """
    Analyze event sequences within player shifts.
    """

    def __init__(self):
        self.value_calc = SequenceValueCalculator()

    def analyze_shift(
        self,
        events: List[SequenceEvent],
        player_id: str
    ) -> Dict:
        """Analyze sequence within a shift."""
        player_events = [e for e in events if e.player_id == player_id]

        if not player_events:
            return {'value': 0, 'events': 0}

        # Create sequence
        seq = EventSequence(
            events=player_events,
            start_time=player_events[0].timestamp,
            end_time=player_events[-1].timestamp,
            terminal_event=player_events[-1].event_type,
            outcome='shift_end'
        )

        value = self.value_calc.calculate_sequence_value(seq)

        # Event breakdown
        event_counts = {}
        for e in player_events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1

        return {
            'value': value,
            'n_events': len(player_events),
            'duration': seq.end_time - seq.start_time,
            'event_breakdown': event_counts,
            'value_per_minute': value / max((seq.end_time - seq.start_time) / 60, 0.01)
        }
