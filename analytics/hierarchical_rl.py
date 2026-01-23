"""
Hierarchical Deep RL for Tactical Optimization

Implements methodology from:
- Meng, X. (2025). "AI-powered Tactical Optimization in Dynamic Team Sports:
  A Hierarchical Reinforcement Learning Approach." ScienceDirect.

Key results from paper:
- 34.7% improvement in tactical accuracy
- 28.3% improvement in decision-making speed
- 41.2% improvement in computational efficiency
- 23.6% improvement in scoring efficiency (NBA G-League)

Hockey translation:
- Multi-level decision hierarchy (game → period → shift → play)
- Real-time tactical suggestions
- Human-AI collaboration in coaching
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from datetime import datetime


class DecisionLevel(Enum):
    """Hierarchical decision levels."""
    GAME = "game"  # Game-level strategy
    PERIOD = "period"  # Period-level adjustments
    SHIFT = "shift"  # Shift-level deployment
    PLAY = "play"  # Play-level actions


class GameStrategy(Enum):
    """High-level game strategies."""
    AGGRESSIVE = "aggressive"  # Push tempo, take risks
    BALANCED = "balanced"  # Standard approach
    CONSERVATIVE = "conservative"  # Protect lead, limit risks
    COMEBACK = "comeback"  # All-out attack when trailing


class PeriodTactic(Enum):
    """Period-level tactical choices."""
    HIGH_PRESSURE = "high_pressure"  # Aggressive forecheck
    NEUTRAL_ZONE_TRAP = "neutral_zone_trap"
    TRANSITION_FOCUS = "transition_focus"
    DEFENSIVE_SHELL = "defensive_shell"
    MATCHUP_HUNTING = "matchup_hunting"


class ShiftAction(Enum):
    """Shift-level deployment decisions."""
    TOP_LINE = "top_line"
    SCORING_LINE = "scoring_line"
    CHECKING_LINE = "checking_line"
    ENERGY_LINE = "energy_line"
    TOP_PAIR = "top_pair"
    SHUTDOWN_PAIR = "shutdown_pair"
    OFFENSIVE_PAIR = "offensive_pair"


class PlayAction(Enum):
    """Play-level action choices."""
    CYCLE = "cycle"
    CRASH_NET = "crash_net"
    POINT_SHOT = "point_shot"
    CROSS_ICE_PASS = "cross_ice_pass"
    DUMP_AND_CHASE = "dump_and_chase"
    CONTROLLED_ENTRY = "controlled_entry"
    STRETCH_PASS = "stretch_pass"


@dataclass
class GameState:
    """Complete game state for decision making."""
    home_score: int
    away_score: int
    period: int
    time_remaining: float  # seconds in period
    home_shots: int
    away_shots: int
    home_hits: int
    away_hits: int
    home_faceoff_pct: float
    home_powerplay: bool = False
    away_powerplay: bool = False
    home_players_on_ice: int = 5
    away_players_on_ice: int = 5
    zone: str = "neutral"  # offensive, defensive, neutral
    possession: str = "home"  # home, away, contested
    momentum: float = 0.0  # -1 to 1 scale

    @property
    def score_differential(self) -> int:
        return self.home_score - self.away_score

    @property
    def total_time_remaining(self) -> float:
        """Total game time remaining in seconds."""
        periods_left = max(0, 3 - self.period)
        return self.time_remaining + periods_left * 1200  # 20 min periods

    @property
    def strength_state(self) -> str:
        if self.home_players_on_ice > self.away_players_on_ice:
            return "pp"
        elif self.home_players_on_ice < self.away_players_on_ice:
            return "pk"
        return "even"


@dataclass
class HierarchicalPolicy:
    """Policy outputs at each level."""
    game_strategy: GameStrategy
    period_tactic: PeriodTactic
    shift_action: ShiftAction
    play_action: PlayAction
    confidence: Dict[DecisionLevel, float] = field(default_factory=dict)


@dataclass
class Experience:
    """Experience tuple for learning."""
    state: GameState
    action: HierarchicalPolicy
    reward: float
    next_state: GameState
    done: bool
    level: DecisionLevel


@dataclass
class OptionFramework:
    """
    Options framework for temporal abstraction.

    Options are extended actions that span multiple time steps.
    """
    name: str
    initiation_set: Callable[[GameState], bool]  # When can option start
    termination_condition: Callable[[GameState], float]  # Probability of ending
    internal_policy: Callable[[GameState], PlayAction]  # Policy while active


class HighLevelPolicy:
    """
    Game and period level policy network.

    Makes strategic decisions that persist over longer time scales.
    """

    def __init__(
        self,
        state_dim: int = 20,
        hidden_dim: int = 128,
        learning_rate: float = 0.001
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate

        # Weights for game strategy
        self.W1_game = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1_game = np.zeros(hidden_dim)
        self.W2_game = np.random.randn(hidden_dim, len(GameStrategy)) * 0.1
        self.b2_game = np.zeros(len(GameStrategy))

        # Weights for period tactics
        self.W1_period = np.random.randn(state_dim + len(GameStrategy), hidden_dim) * 0.1
        self.b1_period = np.zeros(hidden_dim)
        self.W2_period = np.random.randn(hidden_dim, len(PeriodTactic)) * 0.1
        self.b2_period = np.zeros(len(PeriodTactic))

    def state_to_features(self, state: GameState) -> np.ndarray:
        """Convert game state to feature vector."""
        features = [
            state.score_differential / 5.0,  # Normalize
            state.period / 3.0,
            state.time_remaining / 1200.0,
            (state.home_shots - state.away_shots) / 20.0,
            (state.home_hits - state.away_hits) / 20.0,
            state.home_faceoff_pct - 0.5,
            1.0 if state.home_powerplay else 0.0,
            1.0 if state.away_powerplay else 0.0,
            state.home_players_on_ice / 6.0,
            state.away_players_on_ice / 6.0,
            1.0 if state.zone == "offensive" else 0.0,
            1.0 if state.zone == "defensive" else 0.0,
            1.0 if state.possession == "home" else 0.0,
            state.momentum,
            state.total_time_remaining / 3600.0,
            # Additional context
            0.0, 0.0, 0.0, 0.0, 0.0  # Padding
        ]
        return np.array(features[:self.state_dim])

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def select_game_strategy(
        self,
        state: GameState,
        temperature: float = 1.0
    ) -> Tuple[GameStrategy, float]:
        """Select high-level game strategy."""
        features = self.state_to_features(state)

        # Forward pass
        h = np.maximum(0, features @ self.W1_game + self.b1_game)  # ReLU
        logits = h @ self.W2_game + self.b2_game
        probs = self.softmax(logits / temperature)

        # Sample action
        strategies = list(GameStrategy)
        idx = np.random.choice(len(strategies), p=probs)

        return strategies[idx], probs[idx]

    def select_period_tactic(
        self,
        state: GameState,
        strategy: GameStrategy,
        temperature: float = 1.0
    ) -> Tuple[PeriodTactic, float]:
        """Select period-level tactics conditioned on game strategy."""
        features = self.state_to_features(state)

        # One-hot encode strategy
        strategy_vec = np.zeros(len(GameStrategy))
        strategy_vec[list(GameStrategy).index(strategy)] = 1.0

        # Concatenate
        combined = np.concatenate([features, strategy_vec])

        # Forward pass
        h = np.maximum(0, combined @ self.W1_period + self.b1_period)
        logits = h @ self.W2_period + self.b2_period
        probs = self.softmax(logits / temperature)

        tactics = list(PeriodTactic)
        idx = np.random.choice(len(tactics), p=probs)

        return tactics[idx], probs[idx]


class MidLevelPolicy:
    """
    Shift-level policy for line deployment.

    Decides which lines/pairings to use given tactical context.
    """

    def __init__(
        self,
        state_dim: int = 24,
        hidden_dim: int = 64,
        learning_rate: float = 0.001
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate

        # Weights
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, len(ShiftAction)) * 0.1
        self.b2 = np.zeros(len(ShiftAction))

    def state_to_features(
        self,
        state: GameState,
        strategy: GameStrategy,
        tactic: PeriodTactic
    ) -> np.ndarray:
        """Convert state + high-level decisions to features."""
        base_features = [
            state.score_differential / 5.0,
            state.time_remaining / 1200.0,
            1.0 if state.home_powerplay else 0.0,
            1.0 if state.away_powerplay else 0.0,
            1.0 if state.zone == "offensive" else 0.0,
            1.0 if state.zone == "defensive" else 0.0,
            state.momentum,
            # Line fatigue (placeholder)
            0.5, 0.5, 0.5, 0.5,  # Fatigue for 4 lines
            # Strategy one-hot
        ]

        strategy_vec = np.zeros(len(GameStrategy))
        strategy_vec[list(GameStrategy).index(strategy)] = 1.0

        tactic_vec = np.zeros(len(PeriodTactic))
        tactic_vec[list(PeriodTactic).index(tactic)] = 1.0

        features = np.concatenate([
            base_features, strategy_vec, tactic_vec
        ])

        # Pad to state_dim
        if len(features) < self.state_dim:
            features = np.concatenate([features, np.zeros(self.state_dim - len(features))])

        return features[:self.state_dim]

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def select_shift(
        self,
        state: GameState,
        strategy: GameStrategy,
        tactic: PeriodTactic,
        temperature: float = 1.0
    ) -> Tuple[ShiftAction, float]:
        """Select shift/deployment decision."""
        features = self.state_to_features(state, strategy, tactic)

        h = np.maximum(0, features @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = self.softmax(logits / temperature)

        actions = list(ShiftAction)
        idx = np.random.choice(len(actions), p=probs)

        return actions[idx], probs[idx]


class LowLevelPolicy:
    """
    Play-level policy for in-game actions.

    Makes real-time tactical decisions during play.
    """

    def __init__(
        self,
        state_dim: int = 30,
        hidden_dim: int = 64,
        learning_rate: float = 0.001
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate

        # Weights
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, len(PlayAction)) * 0.1
        self.b2 = np.zeros(len(PlayAction))

    def state_to_features(
        self,
        state: GameState,
        strategy: GameStrategy,
        tactic: PeriodTactic,
        shift: ShiftAction
    ) -> np.ndarray:
        """Full context features for play decisions."""
        base_features = [
            state.score_differential / 5.0,
            state.time_remaining / 60.0,  # More granular
            1.0 if state.zone == "offensive" else 0.0,
            1.0 if state.zone == "defensive" else 0.0,
            1.0 if state.zone == "neutral" else 0.0,
            1.0 if state.possession == "home" else 0.0,
            state.momentum,
        ]

        strategy_vec = np.zeros(len(GameStrategy))
        strategy_vec[list(GameStrategy).index(strategy)] = 1.0

        tactic_vec = np.zeros(len(PeriodTactic))
        tactic_vec[list(PeriodTactic).index(tactic)] = 1.0

        shift_vec = np.zeros(len(ShiftAction))
        shift_vec[list(ShiftAction).index(shift)] = 1.0

        features = np.concatenate([
            base_features, strategy_vec, tactic_vec, shift_vec
        ])

        if len(features) < self.state_dim:
            features = np.concatenate([features, np.zeros(self.state_dim - len(features))])

        return features[:self.state_dim]

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def select_play(
        self,
        state: GameState,
        strategy: GameStrategy,
        tactic: PeriodTactic,
        shift: ShiftAction,
        temperature: float = 1.0
    ) -> Tuple[PlayAction, float]:
        """Select play-level action."""
        features = self.state_to_features(state, strategy, tactic, shift)

        h = np.maximum(0, features @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = self.softmax(logits / temperature)

        actions = list(PlayAction)
        idx = np.random.choice(len(actions), p=probs)

        return actions[idx], probs[idx]


class HierarchicalRLAgent:
    """
    Full hierarchical RL agent for hockey.

    Coordinates policies at all levels of the hierarchy.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        high_level_frequency: int = 20,  # Update every N shifts
        mid_level_frequency: int = 1,  # Update every shift
    ):
        self.gamma = gamma
        self.high_freq = high_level_frequency
        self.mid_freq = mid_level_frequency

        # Initialize policies
        self.high_policy = HighLevelPolicy()
        self.mid_policy = MidLevelPolicy()
        self.low_policy = LowLevelPolicy()

        # Experience buffers
        self.high_buffer: List[Experience] = []
        self.mid_buffer: List[Experience] = []
        self.low_buffer: List[Experience] = []

        # Current decisions (persist between calls)
        self.current_strategy: Optional[GameStrategy] = None
        self.current_tactic: Optional[PeriodTactic] = None
        self.current_shift: Optional[ShiftAction] = None

        self.shift_count = 0

    def select_action(
        self,
        state: GameState,
        force_high_update: bool = False,
        temperature: float = 1.0
    ) -> HierarchicalPolicy:
        """
        Select actions at all levels of hierarchy.

        Higher-level decisions persist unless explicitly updated.
        """
        confidences = {}

        # High-level: update periodically or at period start
        if (self.current_strategy is None or
            force_high_update or
            self.shift_count % self.high_freq == 0):

            self.current_strategy, conf = self.high_policy.select_game_strategy(
                state, temperature
            )
            confidences[DecisionLevel.GAME] = conf

            self.current_tactic, conf = self.high_policy.select_period_tactic(
                state, self.current_strategy, temperature
            )
            confidences[DecisionLevel.PERIOD] = conf

        # Mid-level: update every shift
        if (self.current_shift is None or
            self.shift_count % self.mid_freq == 0):

            self.current_shift, conf = self.mid_policy.select_shift(
                state, self.current_strategy, self.current_tactic, temperature
            )
            confidences[DecisionLevel.SHIFT] = conf
            self.shift_count += 1

        # Low-level: update every play
        play_action, conf = self.low_policy.select_play(
            state, self.current_strategy, self.current_tactic,
            self.current_shift, temperature
        )
        confidences[DecisionLevel.PLAY] = conf

        return HierarchicalPolicy(
            game_strategy=self.current_strategy,
            period_tactic=self.current_tactic,
            shift_action=self.current_shift,
            play_action=play_action,
            confidence=confidences
        )

    def store_experience(
        self,
        experience: Experience
    ):
        """Store experience in appropriate buffer."""
        if experience.level == DecisionLevel.GAME or experience.level == DecisionLevel.PERIOD:
            self.high_buffer.append(experience)
        elif experience.level == DecisionLevel.SHIFT:
            self.mid_buffer.append(experience)
        else:
            self.low_buffer.append(experience)

    def compute_hierarchical_reward(
        self,
        state: GameState,
        action: HierarchicalPolicy,
        next_state: GameState,
        outcome: Dict[str, float]
    ) -> Dict[DecisionLevel, float]:
        """
        Compute rewards at each level of hierarchy.

        Higher levels care about long-term outcomes,
        lower levels care about immediate results.
        """
        rewards = {}

        # Game-level: final score differential change
        score_change = (next_state.score_differential - state.score_differential)
        rewards[DecisionLevel.GAME] = score_change * 0.5 + outcome.get('win_prob_change', 0)

        # Period-level: shot differential + territorial control
        shot_diff = ((next_state.home_shots - next_state.away_shots) -
                     (state.home_shots - state.away_shots))
        zone_value = 1.0 if next_state.zone == "offensive" else (
            -1.0 if next_state.zone == "defensive" else 0.0
        )
        rewards[DecisionLevel.PERIOD] = shot_diff * 0.1 + zone_value * 0.2

        # Shift-level: possession and zone time
        poss_value = 1.0 if next_state.possession == "home" else -0.5
        rewards[DecisionLevel.SHIFT] = poss_value * 0.3 + zone_value * 0.1

        # Play-level: immediate action success
        rewards[DecisionLevel.PLAY] = outcome.get('action_success', 0) * 0.5

        return rewards


class OptionsManager:
    """
    Manage temporal abstraction with options.

    Options allow extended actions that span multiple time steps.
    """

    def __init__(self):
        self.options = self._create_hockey_options()
        self.active_option: Optional[OptionFramework] = None

    def _create_hockey_options(self) -> Dict[str, OptionFramework]:
        """Create hockey-specific options."""

        def cycle_initiation(state: GameState) -> bool:
            return state.zone == "offensive" and state.possession == "home"

        def cycle_termination(state: GameState) -> float:
            if state.zone != "offensive" or state.possession != "home":
                return 1.0
            return 0.1  # Small probability of ending

        def cycle_policy(state: GameState) -> PlayAction:
            return PlayAction.CYCLE

        return {
            "power_play_cycle": OptionFramework(
                name="power_play_cycle",
                initiation_set=cycle_initiation,
                termination_condition=cycle_termination,
                internal_policy=cycle_policy
            ),
            "dump_and_forecheck": OptionFramework(
                name="dump_and_forecheck",
                initiation_set=lambda s: s.zone == "neutral",
                termination_condition=lambda s: 0.3 if s.zone == "offensive" else 0.05,
                internal_policy=lambda s: PlayAction.DUMP_AND_CHASE
            ),
            "trap_and_counter": OptionFramework(
                name="trap_and_counter",
                initiation_set=lambda s: s.zone == "neutral" and s.possession != "home",
                termination_condition=lambda s: 0.5 if s.possession == "home" else 0.1,
                internal_policy=lambda s: PlayAction.STRETCH_PASS
            )
        }

    def select_option(self, state: GameState) -> Optional[OptionFramework]:
        """Select an available option."""
        available = [
            opt for opt in self.options.values()
            if opt.initiation_set(state)
        ]

        if not available:
            return None

        return np.random.choice(available)

    def execute_option(
        self,
        state: GameState
    ) -> Tuple[Optional[PlayAction], bool]:
        """
        Execute current option or select new one.

        Returns action and whether option terminated.
        """
        # Check if current option should terminate
        if self.active_option is not None:
            term_prob = self.active_option.termination_condition(state)
            if np.random.random() < term_prob:
                self.active_option = None

        # Select new option if needed
        if self.active_option is None:
            self.active_option = self.select_option(state)

        if self.active_option is None:
            return None, True

        action = self.active_option.internal_policy(state)
        return action, False


class TacticalAdvisor:
    """
    Human-AI collaboration interface.

    Provides explanations and recommendations to coaches.
    """

    def __init__(self, agent: HierarchicalRLAgent):
        self.agent = agent

    def get_recommendation(
        self,
        state: GameState
    ) -> Dict[str, str]:
        """Get tactical recommendation with explanation."""
        policy = self.agent.select_action(state, temperature=0.5)

        explanation = self._explain_decision(state, policy)

        return {
            'strategy': policy.game_strategy.value,
            'tactic': policy.period_tactic.value,
            'deployment': policy.shift_action.value,
            'play': policy.play_action.value,
            'explanation': explanation,
            'confidence': {
                k.value: v for k, v in policy.confidence.items()
            }
        }

    def _explain_decision(
        self,
        state: GameState,
        policy: HierarchicalPolicy
    ) -> str:
        """Generate human-readable explanation."""
        reasons = []

        # Strategy reasoning
        if state.score_differential > 0:
            if policy.game_strategy == GameStrategy.CONSERVATIVE:
                reasons.append("Protecting lead with conservative approach")
            elif policy.game_strategy == GameStrategy.AGGRESSIVE:
                reasons.append("Pushing to extend lead")
        elif state.score_differential < 0:
            if policy.game_strategy == GameStrategy.COMEBACK:
                reasons.append("Trailing - switching to comeback mode")

        # Tactical reasoning
        if state.momentum > 0.3:
            reasons.append(f"Momentum favorable ({state.momentum:.2f})")
        elif state.momentum < -0.3:
            reasons.append(f"Momentum against ({state.momentum:.2f})")

        # Time context
        if state.time_remaining < 300 and state.period == 3:
            reasons.append("Late game situation")

        # Zone context
        if state.zone == "offensive":
            reasons.append(f"Offensive zone possession - {policy.play_action.value}")

        return "; ".join(reasons) if reasons else "Standard tactical decision"

    def suggest_adjustment(
        self,
        current_state: GameState,
        goal_state: Dict[str, float]
    ) -> Dict[str, str]:
        """Suggest tactical adjustments to achieve goal state."""
        suggestions = []

        # Analyze gaps
        if 'shot_differential' in goal_state:
            current_diff = current_state.home_shots - current_state.away_shots
            target = goal_state['shot_differential']

            if current_diff < target:
                suggestions.append("Increase shot volume - consider high-pressure forecheck")

        if 'zone_time' in goal_state:
            if current_state.zone != "offensive":
                suggestions.append("Need offensive zone time - controlled entries recommended")

        return {
            'suggestions': suggestions,
            'priority': 'high' if len(suggestions) > 2 else 'medium'
        }
