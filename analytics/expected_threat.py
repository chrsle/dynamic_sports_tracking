"""
Expected Threat (xT) Framework for Hockey Analytics

This module implements the Expected Threat model adapted from soccer analytics
(Karun Singh's xT framework, Sarah Rudd's Markov chain approach).

The xT model values every location on the ice based on the probability of
scoring from actions starting at that location. Unlike xG which only values
shots, xT values passes, carries, and all puck movements.

Formula:
    xT(x,y) = s(x,y) × g(x,y) + m(x,y) × Σ T(x,y → z,w) × xT(z,w)

Where:
    - s(x,y) = Probability of taking a shot from zone (x,y)
    - g(x,y) = Probability of scoring given a shot from zone (x,y) [xG]
    - m(x,y) = Probability of moving the puck (pass/carry) instead of shooting
    - T(x,y → z,w) = Transition probability from zone (x,y) to zone (z,w)

References:
    - Karun Singh's xT blog post (2018): https://karun.in/blog/expected-threat.html
    - Sarah Rudd's original Markov chain framework (2011)
    - ML-KULeuven/socceraction Python library
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class ActionType(Enum):
    """Types of hockey actions for xT calculation."""
    PASS = "pass"
    CARRY = "carry"
    SHOT = "shot"
    DUMP = "dump"
    CHIP = "chip"
    TURNOVER = "turnover"
    FACEOFF = "faceoff"


class StrengthState(Enum):
    """Game strength states."""
    FIVE_ON_FIVE = "5v5"
    POWER_PLAY = "pp"
    PENALTY_KILL = "pk"
    FOUR_ON_FOUR = "4v4"
    THREE_ON_THREE = "3v3"
    EMPTY_NET = "en"


@dataclass
class HockeyRinkDimensions:
    """NHL standard rink dimensions in feet."""
    length: float = 200.0
    width: float = 85.0
    blue_line_distance: float = 25.0  # From goal line
    center_line: float = 100.0
    goal_line_distance: float = 11.0  # From end boards
    crease_radius: float = 6.0
    faceoff_circle_radius: float = 15.0
    slot_width: float = 30.0
    slot_depth: float = 25.0


@dataclass
class ZoneDefinition:
    """Definition of a zone on the ice."""
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    base_xg_multiplier: float = 1.0


@dataclass
class ActionEvent:
    """Represents a single action event."""
    event_id: str
    timestamp: float
    event_type: ActionType
    player_id: str
    team_id: str
    x: float
    y: float
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    success: bool = True
    strength_state: StrengthState = StrengthState.FIVE_ON_FIVE
    period: int = 1
    game_time: float = 0.0
    velocity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HockeyExpectedThreat:
    """
    Expected Threat model adapted for hockey.

    Based on Karun Singh's soccer xT framework with hockey-specific adaptations:
    - 20x10 grid (200 zones) to account for rink shape
    - Behind-net zones for cycle play
    - Dump-in action type
    - Strength state differentiation

    Attributes:
        grid_length: Number of zones along the length of the ice (goal to goal)
        grid_width: Number of zones across the width of the ice
        eps: Convergence threshold for iterative xT calculation
        xT: The computed Expected Threat matrix
    """

    def __init__(
        self,
        grid_length: int = 20,
        grid_width: int = 10,
        eps: float = 1e-5,
        max_iterations: int = 100,
        rink_dims: Optional[HockeyRinkDimensions] = None
    ):
        """
        Initialize the xT model.

        Args:
            grid_length: Number of zones along the length of the ice (goal to goal)
            grid_width: Number of zones across the width of the ice
            eps: Convergence threshold
            max_iterations: Maximum iterations for convergence
            rink_dims: Optional custom rink dimensions
        """
        self.l = grid_length
        self.w = grid_width
        self.eps = eps
        self.max_iterations = max_iterations
        self.rink = rink_dims or HockeyRinkDimensions()

        # Zone dimensions in feet
        self.zone_length = self.rink.length / self.l
        self.zone_width = self.rink.width / self.w

        # Initialize probability matrices
        self.xT = np.zeros((self.w, self.l))
        self.shot_prob = np.zeros((self.w, self.l))
        self.goal_prob = np.zeros((self.w, self.l))
        self.move_prob = np.zeros((self.w, self.l))
        self.turnover_prob = np.zeros((self.w, self.l))
        self.transition_matrix = np.zeros((self.w * self.l, self.w * self.l))

        # Separate matrices for different strength states
        self.xT_by_strength: Dict[StrengthState, np.ndarray] = {}

        # Track if model has been fit
        self._is_fitted = False
        self._convergence_iterations = 0

        # Define special zones
        self._define_special_zones()

    def _define_special_zones(self):
        """Define hockey-specific zones for analysis."""
        # Offensive zone (attacking end)
        off_zone_start = self.rink.length - self.rink.blue_line_distance - self.rink.goal_line_distance

        self.zones = {
            'behind_net': ZoneDefinition(
                'behind_net',
                x_min=self.rink.length - self.rink.goal_line_distance,
                x_max=self.rink.length,
                y_min=0,
                y_max=self.rink.width,
                base_xg_multiplier=0.5
            ),
            'slot': ZoneDefinition(
                'slot',
                x_min=self.rink.length - self.rink.goal_line_distance - self.rink.slot_depth,
                x_max=self.rink.length - self.rink.goal_line_distance,
                y_min=(self.rink.width - self.rink.slot_width) / 2,
                y_max=(self.rink.width + self.rink.slot_width) / 2,
                base_xg_multiplier=2.5
            ),
            'high_slot': ZoneDefinition(
                'high_slot',
                x_min=self.rink.length - self.rink.goal_line_distance - self.rink.slot_depth - 15,
                x_max=self.rink.length - self.rink.goal_line_distance - self.rink.slot_depth,
                y_min=(self.rink.width - self.rink.slot_width) / 2,
                y_max=(self.rink.width + self.rink.slot_width) / 2,
                base_xg_multiplier=1.5
            ),
            'point_left': ZoneDefinition(
                'point_left',
                x_min=off_zone_start,
                x_max=off_zone_start + 15,
                y_min=0,
                y_max=self.rink.width / 2,
                base_xg_multiplier=0.8
            ),
            'point_right': ZoneDefinition(
                'point_right',
                x_min=off_zone_start,
                x_max=off_zone_start + 15,
                y_min=self.rink.width / 2,
                y_max=self.rink.width,
                base_xg_multiplier=0.8
            ),
        }

    def coords_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert rink coordinates to grid indices."""
        grid_x = int(x / self.rink.length * self.l)
        grid_y = int(y / self.rink.width * self.w)
        return (
            np.clip(grid_x, 0, self.l - 1),
            np.clip(grid_y, 0, self.w - 1)
        )

    def grid_to_coords(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to rink coordinates (center of zone)."""
        x = (grid_x + 0.5) * self.zone_length
        y = (grid_y + 0.5) * self.zone_width
        return (x, y)

    def grid_to_flat_index(self, grid_x: int, grid_y: int) -> int:
        """Convert 2D grid coordinates to flat index."""
        return grid_y * self.l + grid_x

    def flat_to_grid(self, flat_idx: int) -> Tuple[int, int]:
        """Convert flat index to 2D grid coordinates."""
        grid_y = flat_idx // self.l
        grid_x = flat_idx % self.l
        return (grid_x, grid_y)

    def fit(
        self,
        events_df: pd.DataFrame,
        strength_state: Optional[StrengthState] = None
    ) -> 'HockeyExpectedThreat':
        """
        Fit the xT model to event data.

        Args:
            events_df: DataFrame with columns:
                - x, y: start coordinates (0-200, 0-85 for NHL rink)
                - end_x, end_y: end coordinates
                - event_type: 'pass', 'carry', 'shot', 'goal', 'dump', 'turnover'
                - success: boolean indicating if action was successful

        Returns:
            self
        """
        # Make a copy to avoid modifying original
        df = events_df.copy()

        # Filter by strength state if specified
        if strength_state and 'strength_state' in df.columns:
            df = df[df['strength_state'] == strength_state.value]

        # Normalize coordinates to grid
        df['grid_x'] = (df['x'] / self.rink.length * self.l).astype(int).clip(0, self.l - 1)
        df['grid_y'] = (df['y'] / self.rink.width * self.w).astype(int).clip(0, self.w - 1)

        if 'end_x' in df.columns and 'end_y' in df.columns:
            df['grid_end_x'] = (df['end_x'] / self.rink.length * self.l).astype(int).clip(0, self.l - 1)
            df['grid_end_y'] = (df['end_y'] / self.rink.width * self.w).astype(int).clip(0, self.w - 1)

        # Calculate probability matrices
        self._calculate_shot_probability(df)
        self._calculate_goal_probability(df)
        self._calculate_move_probability(df)
        self._calculate_turnover_probability(df)
        self._calculate_transition_matrix(df)

        # Iterate to convergence
        self._solve_xT()

        self._is_fitted = True

        # Store for this strength state if specified
        if strength_state:
            self.xT_by_strength[strength_state] = self.xT.copy()

        return self

    def fit_all_strength_states(self, events_df: pd.DataFrame) -> 'HockeyExpectedThreat':
        """
        Fit separate xT models for each strength state.

        Args:
            events_df: DataFrame with 'strength_state' column

        Returns:
            self
        """
        for state in StrengthState:
            state_df = events_df[events_df.get('strength_state', StrengthState.FIVE_ON_FIVE.value) == state.value]
            if len(state_df) > 100:  # Need sufficient data
                self.fit(events_df, strength_state=state)

        # Default to 5v5 for main xT matrix
        if StrengthState.FIVE_ON_FIVE in self.xT_by_strength:
            self.xT = self.xT_by_strength[StrengthState.FIVE_ON_FIVE]

        return self

    def _calculate_shot_probability(self, df: pd.DataFrame):
        """Calculate P(shot) for each zone."""
        self.shot_prob = np.zeros((self.w, self.l))

        for y in range(self.w):
            for x in range(self.l):
                zone_events = df[(df['grid_x'] == x) & (df['grid_y'] == y)]
                if len(zone_events) > 0:
                    shots = zone_events[zone_events['event_type'].isin(['shot', 'goal'])]
                    self.shot_prob[y, x] = len(shots) / len(zone_events)

    def _calculate_goal_probability(self, df: pd.DataFrame):
        """Calculate P(goal | shot) for each zone - essentially xG."""
        self.goal_prob = np.zeros((self.w, self.l))

        for y in range(self.w):
            for x in range(self.l):
                shots = df[
                    (df['grid_x'] == x) &
                    (df['grid_y'] == y) &
                    (df['event_type'].isin(['shot', 'goal']))
                ]
                if len(shots) > 0:
                    goals = shots[shots['event_type'] == 'goal']
                    self.goal_prob[y, x] = len(goals) / len(shots)

    def _calculate_move_probability(self, df: pd.DataFrame):
        """Calculate P(move) for each zone."""
        self.move_prob = np.zeros((self.w, self.l))

        for y in range(self.w):
            for x in range(self.l):
                zone_events = df[(df['grid_x'] == x) & (df['grid_y'] == y)]
                if len(zone_events) > 0:
                    moves = zone_events[zone_events['event_type'].isin(['pass', 'carry', 'dump', 'chip'])]
                    self.move_prob[y, x] = len(moves) / len(zone_events)

    def _calculate_turnover_probability(self, df: pd.DataFrame):
        """Calculate P(turnover) for each zone."""
        self.turnover_prob = np.zeros((self.w, self.l))

        for y in range(self.w):
            for x in range(self.l):
                zone_events = df[(df['grid_x'] == x) & (df['grid_y'] == y)]
                if len(zone_events) > 0:
                    turnovers = zone_events[zone_events['event_type'] == 'turnover']
                    self.turnover_prob[y, x] = len(turnovers) / len(zone_events)

    def _calculate_transition_matrix(self, df: pd.DataFrame):
        """Calculate transition probabilities between zones."""
        self.transition_matrix = np.zeros((self.w * self.l, self.w * self.l))

        # Filter to successful moves only
        moves = df[
            (df['event_type'].isin(['pass', 'carry', 'dump', 'chip'])) &
            (df.get('success', True) == True)
        ]

        if 'grid_end_x' not in moves.columns:
            return

        for _, row in moves.iterrows():
            start_idx = self.grid_to_flat_index(int(row['grid_x']), int(row['grid_y']))
            end_idx = self.grid_to_flat_index(int(row['grid_end_x']), int(row['grid_end_y']))
            self.transition_matrix[start_idx, end_idx] += 1

        # Normalize rows
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = self.transition_matrix / row_sums

    def _solve_xT(self):
        """Iterate to solve for xT values using Bellman equation."""
        # Initialize with shot value (xG)
        self.xT = self.shot_prob * self.goal_prob

        iteration = 0

        while iteration < self.max_iterations:
            xT_old = self.xT.copy()

            # Calculate expected value from moving
            move_value = np.zeros((self.w, self.l))

            for y in range(self.w):
                for x in range(self.l):
                    start_idx = self.grid_to_flat_index(x, y)

                    for end_y in range(self.w):
                        for end_x in range(self.l):
                            end_idx = self.grid_to_flat_index(end_x, end_y)
                            move_value[y, x] += (
                                self.transition_matrix[start_idx, end_idx] *
                                self.xT[end_y, end_x]
                            )

            # Update xT using Bellman equation
            # xT = P(shot) * P(goal|shot) + P(move) * E[xT of destination]
            self.xT = (
                self.shot_prob * self.goal_prob +
                self.move_prob * move_value
            )

            # Check convergence
            max_diff = np.max(np.abs(self.xT - xT_old))
            if max_diff < self.eps:
                break

            iteration += 1

        self._convergence_iterations = iteration

    def get_xT(self, x: float, y: float, strength_state: Optional[StrengthState] = None) -> float:
        """
        Get the xT value at a specific location.

        Args:
            x, y: Rink coordinates
            strength_state: Optional strength state for state-specific xT

        Returns:
            xT value at the location
        """
        grid_x, grid_y = self.coords_to_grid(x, y)

        if strength_state and strength_state in self.xT_by_strength:
            return self.xT_by_strength[strength_state][grid_y, grid_x]

        return self.xT[grid_y, grid_x]

    def get_action_value(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        strength_state: Optional[StrengthState] = None
    ) -> float:
        """
        Get the xT value added by an action.

        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            strength_state: Optional strength state

        Returns:
            xT value added (can be negative for backwards passes)
        """
        start_xT = self.get_xT(start_x, start_y, strength_state)
        end_xT = self.get_xT(end_x, end_y, strength_state)

        return end_xT - start_xT

    def get_player_threat_added(
        self,
        events: List[ActionEvent],
        player_id: str
    ) -> Dict[str, float]:
        """
        Calculate total threat added by a player.

        Args:
            events: List of action events
            player_id: Player ID to analyze

        Returns:
            Dictionary with threat added statistics
        """
        player_events = [e for e in events if e.player_id == player_id]

        total_threat = 0.0
        pass_threat = 0.0
        carry_threat = 0.0

        for event in player_events:
            if event.end_x is not None and event.end_y is not None:
                threat = self.get_action_value(
                    event.x, event.y,
                    event.end_x, event.end_y,
                    event.strength_state
                )

                total_threat += threat

                if event.event_type == ActionType.PASS:
                    pass_threat += threat
                elif event.event_type == ActionType.CARRY:
                    carry_threat += threat

        return {
            'total_threat_added': total_threat,
            'pass_threat_added': pass_threat,
            'carry_threat_added': carry_threat,
            'actions': len(player_events),
            'threat_per_action': total_threat / max(len(player_events), 1)
        }

    def get_zone_breakdown(self) -> pd.DataFrame:
        """
        Get xT breakdown by zone.

        Returns:
            DataFrame with zone-level xT statistics
        """
        data = []

        for y in range(self.w):
            for x in range(self.l):
                center_x, center_y = self.grid_to_coords(x, y)

                # Determine zone name
                zone_name = 'neutral'
                if center_x > self.rink.length - self.rink.blue_line_distance - self.rink.goal_line_distance:
                    zone_name = 'offensive'
                elif center_x < self.rink.blue_line_distance + self.rink.goal_line_distance:
                    zone_name = 'defensive'

                # Check for special zones
                for special_zone_name, zone_def in self.zones.items():
                    if (zone_def.x_min <= center_x <= zone_def.x_max and
                        zone_def.y_min <= center_y <= zone_def.y_max):
                        zone_name = special_zone_name
                        break

                data.append({
                    'grid_x': x,
                    'grid_y': y,
                    'center_x': center_x,
                    'center_y': center_y,
                    'zone': zone_name,
                    'xT': self.xT[y, x],
                    'shot_prob': self.shot_prob[y, x],
                    'goal_prob': self.goal_prob[y, x],
                    'move_prob': self.move_prob[y, x],
                    'turnover_prob': self.turnover_prob[y, x]
                })

        return pd.DataFrame(data)

    def to_dict(self) -> Dict[str, Any]:
        """Export model to dictionary for serialization."""
        return {
            'grid_length': self.l,
            'grid_width': self.w,
            'xT': self.xT.tolist(),
            'shot_prob': self.shot_prob.tolist(),
            'goal_prob': self.goal_prob.tolist(),
            'move_prob': self.move_prob.tolist(),
            'turnover_prob': self.turnover_prob.tolist(),
            'transition_matrix': self.transition_matrix.tolist(),
            'convergence_iterations': self._convergence_iterations,
            'is_fitted': self._is_fitted
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HockeyExpectedThreat':
        """Load model from dictionary."""
        model = cls(
            grid_length=data['grid_length'],
            grid_width=data['grid_width']
        )
        model.xT = np.array(data['xT'])
        model.shot_prob = np.array(data['shot_prob'])
        model.goal_prob = np.array(data['goal_prob'])
        model.move_prob = np.array(data['move_prob'])
        model.turnover_prob = np.array(data['turnover_prob'])
        model.transition_matrix = np.array(data['transition_matrix'])
        model._convergence_iterations = data.get('convergence_iterations', 0)
        model._is_fitted = data.get('is_fitted', True)
        return model

    def save(self, filepath: str):
        """Save model to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, filepath: str) -> 'HockeyExpectedThreat':
        """Load model from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))

    def plot_xT_surface(self, ax=None, cmap='RdYlGn', title='Expected Threat (xT) Surface'):
        """
        Plot the xT surface as a heatmap.

        Args:
            ax: Matplotlib axes (creates new if None)
            cmap: Colormap to use
            title: Plot title

        Returns:
            Matplotlib axes object
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))

        # Plot xT heatmap
        im = ax.imshow(
            self.xT,
            origin='lower',
            aspect='auto',
            cmap=cmap,
            extent=[0, self.rink.length, 0, self.rink.width]
        )

        # Add rink markings
        # Blue lines
        ax.axvline(x=self.rink.blue_line_distance + self.rink.goal_line_distance,
                   color='blue', linewidth=2, linestyle='--', alpha=0.7)
        ax.axvline(x=self.rink.length - self.rink.blue_line_distance - self.rink.goal_line_distance,
                   color='blue', linewidth=2, linestyle='--', alpha=0.7)

        # Center line
        ax.axvline(x=self.rink.center_line, color='red', linewidth=2, linestyle='--', alpha=0.7)

        # Goal lines
        ax.axvline(x=self.rink.goal_line_distance, color='red', linewidth=1, alpha=0.5)
        ax.axvline(x=self.rink.length - self.rink.goal_line_distance, color='red', linewidth=1, alpha=0.5)

        ax.set_xlabel('Rink Length (ft)')
        ax.set_ylabel('Rink Width (ft)')
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label='xT Value')

        return ax


class xTActionValuer:
    """
    Utility class for valuing individual actions using xT.

    Provides a simple interface for valuing passes, carries, and shots
    without needing to manage the full xT model directly.
    """

    def __init__(self, xT_model: HockeyExpectedThreat):
        """
        Initialize the action valuer.

        Args:
            xT_model: Fitted HockeyExpectedThreat model
        """
        self.model = xT_model

    def value_pass(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        successful: bool = True,
        strength_state: Optional[StrengthState] = None
    ) -> float:
        """
        Value a pass action.

        Args:
            start_x, start_y: Pass origin coordinates
            end_x, end_y: Pass destination coordinates
            successful: Whether pass was completed
            strength_state: Game strength state

        Returns:
            xT value of the pass (negative if turnover)
        """
        if not successful:
            # Failed pass = loss of xT at start location
            return -self.model.get_xT(start_x, start_y, strength_state)

        return self.model.get_action_value(start_x, start_y, end_x, end_y, strength_state)

    def value_carry(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        velocity: Optional[float] = None,
        strength_state: Optional[StrengthState] = None
    ) -> float:
        """
        Value a carry/skate action.

        Args:
            start_x, start_y: Carry start coordinates
            end_x, end_y: Carry end coordinates
            velocity: Optional skating velocity for bonus
            strength_state: Game strength state

        Returns:
            xT value of the carry
        """
        base_value = self.model.get_action_value(start_x, start_y, end_x, end_y, strength_state)

        # Add velocity bonus for fast carries (higher xT for speed)
        if velocity is not None:
            # Baseline ~15 mph, bonus for faster
            velocity_multiplier = 1.0 + (velocity - 15) * 0.01
            velocity_multiplier = np.clip(velocity_multiplier, 0.8, 1.3)
            base_value *= velocity_multiplier

        return base_value

    def value_shot(
        self,
        x: float,
        y: float,
        xG: Optional[float] = None,
        strength_state: Optional[StrengthState] = None
    ) -> float:
        """
        Value a shot action.

        Args:
            x, y: Shot location coordinates
            xG: Optional pre-calculated xG (uses model's if not provided)
            strength_state: Game strength state

        Returns:
            xT value consumed by the shot
        """
        if xG is not None:
            return xG

        # Use the model's goal probability as xG approximation
        grid_x, grid_y = self.model.coords_to_grid(x, y)
        return self.model.goal_prob[grid_y, grid_x]

    def value_dump_in(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        recovered: bool = True,
        strength_state: Optional[StrengthState] = None
    ) -> float:
        """
        Value a dump-in action.

        Args:
            start_x, start_y: Dump origin coordinates
            end_x, end_y: Dump destination coordinates
            recovered: Whether team recovered the puck
            strength_state: Game strength state

        Returns:
            xT value of the dump (discounted vs carry)
        """
        if not recovered:
            # Lost dump = loss of xT
            return -self.model.get_xT(start_x, start_y, strength_state)

        # Dump-ins are typically less valuable than controlled entries
        base_value = self.model.get_action_value(start_x, start_y, end_x, end_y, strength_state)

        # Apply discount factor for dump vs carry
        dump_discount = 0.5  # Dump generates ~half the threat of carry

        return base_value * dump_discount


def create_sample_xT_model() -> HockeyExpectedThreat:
    """
    Create a sample xT model with synthetic data for testing.

    Returns:
        Fitted HockeyExpectedThreat model
    """
    np.random.seed(42)

    n_events = 50000

    # Generate realistic hockey event distribution
    # More events in offensive/defensive zones, fewer in neutral
    x_coords = np.concatenate([
        np.random.uniform(0, 50, int(n_events * 0.2)),      # Defensive zone
        np.random.uniform(50, 150, int(n_events * 0.3)),    # Neutral zone
        np.random.uniform(150, 200, int(n_events * 0.5)),   # Offensive zone
    ])

    y_coords = np.random.uniform(0, 85, n_events)

    # Generate end coordinates (mostly forward movement in offensive zone)
    end_x = x_coords + np.random.uniform(-20, 30, n_events)
    end_x = np.clip(end_x, 0, 200)
    end_y = y_coords + np.random.uniform(-15, 15, n_events)
    end_y = np.clip(end_y, 0, 85)

    # Event types - more shots in offensive zone
    event_types = []
    for x in x_coords:
        if x > 175:  # Deep offensive
            event_types.append(np.random.choice(
                ['pass', 'carry', 'shot', 'goal', 'turnover'],
                p=[0.35, 0.2, 0.25, 0.08, 0.12]
            ))
        elif x > 150:  # Offensive zone
            event_types.append(np.random.choice(
                ['pass', 'carry', 'shot', 'goal', 'turnover'],
                p=[0.4, 0.25, 0.18, 0.05, 0.12]
            ))
        else:
            event_types.append(np.random.choice(
                ['pass', 'carry', 'shot', 'goal', 'dump', 'turnover'],
                p=[0.45, 0.3, 0.05, 0.01, 0.1, 0.09]
            ))

    sample_events = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'end_x': end_x,
        'end_y': end_y,
        'event_type': event_types,
        'success': np.random.choice([True, False], n_events, p=[0.85, 0.15])
    })

    # Fit model
    model = HockeyExpectedThreat(grid_length=20, grid_width=10)
    model.fit(sample_events)

    return model


if __name__ == "__main__":
    # Demo the xT model
    print("Creating sample xT model...")
    model = create_sample_xT_model()

    print(f"Model converged in {model._convergence_iterations} iterations")
    print(f"\nxT Statistics:")
    print(f"  Max xT: {model.xT.max():.4f}")
    print(f"  Mean xT: {model.xT.mean():.4f}")
    print(f"  Min xT: {model.xT.min():.4f}")

    # Example action values
    valuer = xTActionValuer(model)

    print("\nSample Action Values:")
    print(f"  Center ice to slot pass: {valuer.value_pass(100, 42, 180, 42):.4f}")
    print(f"  Slot to crease pass: {valuer.value_pass(180, 42, 195, 42):.4f}")
    print(f"  Defensive zone dump-in: {valuer.value_dump_in(60, 42, 180, 20, recovered=True):.4f}")
    print(f"  Failed dump-in: {valuer.value_dump_in(60, 42, 180, 20, recovered=False):.4f}")

    # Zone breakdown
    zone_df = model.get_zone_breakdown()
    print("\nZone xT Summary:")
    print(zone_df.groupby('zone')['xT'].agg(['mean', 'max', 'min']).round(4))
