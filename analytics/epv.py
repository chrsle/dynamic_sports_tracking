"""
Expected Possession Value (EPV) Framework for Hockey Analytics

This module implements a continuous Expected Possession Value model,
adapted from basketball (Cervone et al.) and soccer (Fernández & Bornn)
EPV frameworks.

Key concepts:
- Frame-by-frame likelihood of scoring/conceding
- Decomposed into action components (pass, carry, shot)
- Continuous valuation of every moment in possession
- Spatiotemporal tracking integration

"Despite many recent innovations, most advanced metrics remain based on
simple tallies relating to the terminal states of possessions... While
these have shed light on the game, they are akin to analyzing a chess
match based only on the move that resulted in checkmate."
— Cervone et al., Basketball EPV paper

References:
- "Decomposing the Immeasurable Sport" (Sloan 2019)
- "A framework for fine-grained evaluation of soccer possessions" (2021)
- Basketball EPV (Cervone et al.)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class PossessionState(Enum):
    """States a possession can be in."""
    ATTACKING = "attacking"
    DEFENDING = "defending"
    LOOSE_PUCK = "loose_puck"
    DEAD_BALL = "dead_ball"


class ActionType(Enum):
    """Types of actions that change EPV."""
    PASS = "pass"
    CARRY = "carry"
    SHOT = "shot"
    RECEPTION = "reception"
    TURNOVER = "turnover"
    RECOVERY = "recovery"
    FACEOFF = "faceoff"
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"


@dataclass
class PossessionFrame:
    """Single frame of possession data."""
    frame_id: int
    timestamp: float
    game_id: str
    period: int

    # Possession state
    possessing_team: Optional[str] = None
    possession_state: PossessionState = PossessionState.LOOSE_PUCK

    # Puck position
    puck_x: float = 100.0
    puck_y: float = 42.5
    puck_carrier_id: Optional[str] = None

    # All player positions: {player_id: (x, y, vx, vy)}
    home_positions: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    away_positions: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)

    # Computed EPV
    epv_home: Optional[float] = None
    epv_away: Optional[float] = None

    # Action that just occurred (if any)
    action_type: Optional[ActionType] = None
    action_player_id: Optional[str] = None
    action_value: Optional[float] = None


@dataclass
class PossessionSequence:
    """A complete possession sequence."""
    sequence_id: str
    game_id: str
    possessing_team: str

    # Frames in this possession
    frames: List[PossessionFrame] = field(default_factory=list)

    # Sequence characteristics
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    # Zone information
    start_zone: str = "neutral"
    zones_entered: List[str] = field(default_factory=list)

    # Outcome
    outcome: str = "turnover"  # goal, shot, turnover, penalty
    outcome_xG: Optional[float] = None

    # EPV trajectory
    initial_epv: float = 0.0
    final_epv: float = 0.0
    max_epv: float = 0.0
    epv_trajectory: List[float] = field(default_factory=list)


class EPVModel:
    """
    Expected Possession Value Model.

    Calculates the expected value of a possession at each moment,
    accounting for:
    1. Current puck location
    2. Player positions (offense and defense)
    3. Game state (score, period, strength)
    4. Recent action history

    EPV is decomposed into:
    - Shot value (probability and quality of taking a shot)
    - Pass value (expected value from passing)
    - Carry value (expected value from skating with puck)
    - Turnover risk (probability and cost of losing possession)
    """

    def __init__(
        self,
        grid_x: int = 40,
        grid_y: int = 20,
        rink_length: float = 200.0,
        rink_width: float = 85.0,
    ):
        """
        Initialize the EPV model.

        Args:
            grid_x: Number of grid cells along length
            grid_y: Number of grid cells along width
            rink_length: Rink length in feet
            rink_width: Rink width in feet
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.rink_length = rink_length
        self.rink_width = rink_width

        self.cell_width = rink_length / grid_x
        self.cell_height = rink_width / grid_y

        # Build component models
        self._build_shot_model()
        self._build_pass_model()
        self._build_carry_model()
        self._build_turnover_model()

        # Possession tracking
        self.possession_sequences: List[PossessionSequence] = []
        self.current_sequence: Optional[PossessionSequence] = None

    def _build_shot_model(self):
        """Build the shot probability and xG models."""
        self.shot_prob = np.zeros((self.grid_y, self.grid_x))
        self.shot_xG = np.zeros((self.grid_y, self.grid_x))

        for i in range(self.grid_y):
            for j in range(self.grid_x):
                x = j * self.cell_width
                y = i * self.cell_height

                # Shot probability increases in offensive zone
                if x > 150:
                    distance_to_goal = np.sqrt((200 - x) ** 2 + (42.5 - y) ** 2)
                    self.shot_prob[i, j] = max(0, 0.3 - distance_to_goal * 0.005)

                    # xG based on distance and angle
                    angle = abs(np.arctan2(y - 42.5, 200 - x))
                    self.shot_xG[i, j] = max(0.01, 0.35 - distance_to_goal * 0.008 - angle * 0.1)

    def _build_pass_model(self):
        """Build pass transition and success models."""
        # Simplified: pass success rate by distance
        self.pass_success_base = 0.85
        self.pass_distance_penalty = 0.005  # Per foot

    def _build_carry_model(self):
        """Build carry value model."""
        # Value of advancing the puck
        self.carry_value_per_foot = 0.001  # In offensive zone
        self.carry_turnover_risk = 0.02  # Per second of carrying

    def _build_turnover_model(self):
        """Build turnover probability model."""
        self.turnover_base = np.zeros((self.grid_y, self.grid_x))

        for i in range(self.grid_y):
            for j in range(self.grid_x):
                x = j * self.cell_width

                # Higher turnover risk in offensive zone (more pressure)
                if x > 175:
                    self.turnover_base[i, j] = 0.08
                elif x > 150:
                    self.turnover_base[i, j] = 0.05
                else:
                    self.turnover_base[i, j] = 0.03

    def calculate_epv(
        self,
        frame: PossessionFrame,
        include_decomposition: bool = False
    ) -> Dict[str, float]:
        """
        Calculate Expected Possession Value for a frame.

        Args:
            frame: PossessionFrame with all positions
            include_decomposition: Whether to include value breakdown

        Returns:
            Dictionary with EPV and optional decomposition
        """
        if frame.possessing_team is None:
            return {'epv': 0.0}

        # Get puck location in grid
        grid_x = int(frame.puck_x / self.cell_width)
        grid_y = int(frame.puck_y / self.cell_height)
        grid_x = np.clip(grid_x, 0, self.grid_x - 1)
        grid_y = np.clip(grid_y, 0, self.grid_y - 1)

        # Calculate EPV components

        # 1. Shot value = P(shot) * xG
        shot_prob = self.shot_prob[grid_y, grid_x]
        shot_xG = self.shot_xG[grid_y, grid_x]
        shot_value = shot_prob * shot_xG

        # 2. Pass value (simplified)
        # Would integrate over all possible passes in full model
        pass_value = self._estimate_pass_value(frame, grid_x, grid_y)

        # 3. Carry value
        carry_value = self._estimate_carry_value(frame, grid_x, grid_y)

        # 4. Turnover cost
        turnover_prob = self._estimate_turnover_prob(frame, grid_x, grid_y)
        turnover_cost = turnover_prob * 0.05  # Expected value lost on turnover

        # Total EPV = weighted sum minus turnover cost
        # Weights could be learned from data
        epv = shot_value + 0.4 * pass_value + 0.3 * carry_value - turnover_cost
        epv = max(0, min(epv, 1.0))  # Bound to [0, 1]

        result = {'epv': float(epv)}

        if include_decomposition:
            result['decomposition'] = {
                'shot_value': shot_value,
                'pass_value': pass_value,
                'carry_value': carry_value,
                'turnover_cost': turnover_cost,
                'shot_prob': shot_prob,
                'shot_xG': shot_xG,
                'turnover_prob': turnover_prob,
            }

        return result

    def _estimate_pass_value(
        self,
        frame: PossessionFrame,
        grid_x: int,
        grid_y: int
    ) -> float:
        """Estimate expected value from passing."""
        # Simplified: look at available passing lanes
        if frame.possessing_team is None:
            return 0.0

        positions = (frame.home_positions if frame.possessing_team == "home"
                     else frame.away_positions)

        pass_values = []

        for player_id, (px, py, _, _) in positions.items():
            if player_id == frame.puck_carrier_id:
                continue

            # Distance to teammate
            dist = np.sqrt((frame.puck_x - px) ** 2 + (frame.puck_y - py) ** 2)

            # Pass success probability
            success_prob = max(0.5, self.pass_success_base - dist * self.pass_distance_penalty)

            # Value at destination
            dest_grid_x = int(px / self.cell_width)
            dest_grid_y = int(py / self.cell_height)
            dest_grid_x = np.clip(dest_grid_x, 0, self.grid_x - 1)
            dest_grid_y = np.clip(dest_grid_y, 0, self.grid_y - 1)

            dest_shot_value = self.shot_prob[dest_grid_y, dest_grid_x] * self.shot_xG[dest_grid_y, dest_grid_x]

            pass_values.append(success_prob * dest_shot_value)

        return max(pass_values) if pass_values else 0.0

    def _estimate_carry_value(
        self,
        frame: PossessionFrame,
        grid_x: int,
        grid_y: int
    ) -> float:
        """Estimate expected value from carrying/skating."""
        # Value of advancing toward goal
        current_x = frame.puck_x
        potential_advance = min(20, 200 - current_x)  # Max 20 feet advance

        if current_x > 150:  # In offensive zone
            return potential_advance * self.carry_value_per_foot
        else:
            return potential_advance * self.carry_value_per_foot * 0.5

    def _estimate_turnover_prob(
        self,
        frame: PossessionFrame,
        grid_x: int,
        grid_y: int
    ) -> float:
        """Estimate turnover probability."""
        base_turnover = self.turnover_base[grid_y, grid_x]

        # Adjust for defensive pressure (count nearby defenders)
        if frame.possessing_team == "home":
            defenders = frame.away_positions
        else:
            defenders = frame.home_positions

        pressure = 0
        for _, (dx, dy, _, _) in defenders.items():
            dist = np.sqrt((frame.puck_x - dx) ** 2 + (frame.puck_y - dy) ** 2)
            if dist < 10:  # Within 10 feet
                pressure += 0.02

        return min(base_turnover + pressure, 0.3)

    def calculate_action_value(
        self,
        frame_before: PossessionFrame,
        frame_after: PossessionFrame
    ) -> Dict[str, float]:
        """
        Calculate the value added by an action.

        Value added = EPV(after) - EPV(before)

        Args:
            frame_before: Frame before action
            frame_after: Frame after action

        Returns:
            Dictionary with action value and details
        """
        epv_before = self.calculate_epv(frame_before)['epv']
        epv_after = self.calculate_epv(frame_after)['epv']

        value_added = epv_after - epv_before

        return {
            'value_added': value_added,
            'epv_before': epv_before,
            'epv_after': epv_after,
            'action_type': frame_after.action_type.value if frame_after.action_type else None,
        }

    def track_possession(self, frame: PossessionFrame):
        """
        Track a possession frame.

        Args:
            frame: Current possession frame
        """
        # Calculate EPV
        epv_result = self.calculate_epv(frame)

        if frame.possessing_team == "home":
            frame.epv_home = epv_result['epv']
            frame.epv_away = 0.0
        else:
            frame.epv_home = 0.0
            frame.epv_away = epv_result['epv']

        # Check if this is a new possession
        if self.current_sequence is None:
            self._start_new_possession(frame)
        elif frame.possessing_team != self.current_sequence.possessing_team:
            # Possession changed - end current and start new
            self._end_possession(frame)
            self._start_new_possession(frame)
        else:
            # Continue current possession
            self.current_sequence.frames.append(frame)
            self.current_sequence.epv_trajectory.append(
                frame.epv_home if frame.possessing_team == "home" else frame.epv_away
            )

    def _start_new_possession(self, frame: PossessionFrame):
        """Start tracking a new possession."""
        self.current_sequence = PossessionSequence(
            sequence_id=f"poss_{frame.game_id}_{frame.timestamp}",
            game_id=frame.game_id,
            possessing_team=frame.possessing_team or "unknown",
            start_time=frame.timestamp,
        )
        self.current_sequence.frames.append(frame)
        epv = frame.epv_home if frame.possessing_team == "home" else frame.epv_away
        self.current_sequence.initial_epv = epv or 0.0
        self.current_sequence.epv_trajectory.append(epv or 0.0)

    def _end_possession(self, frame: PossessionFrame):
        """End current possession tracking."""
        if self.current_sequence is None:
            return

        self.current_sequence.end_time = frame.timestamp
        self.current_sequence.duration = (
            self.current_sequence.end_time - self.current_sequence.start_time
        )

        if self.current_sequence.epv_trajectory:
            self.current_sequence.final_epv = self.current_sequence.epv_trajectory[-1]
            self.current_sequence.max_epv = max(self.current_sequence.epv_trajectory)

        self.possession_sequences.append(self.current_sequence)
        self.current_sequence = None

    def get_possession_summary(self) -> pd.DataFrame:
        """Get summary of all tracked possessions."""
        summaries = []

        for poss in self.possession_sequences:
            summaries.append({
                'sequence_id': poss.sequence_id,
                'team': poss.possessing_team,
                'duration': poss.duration,
                'initial_epv': poss.initial_epv,
                'final_epv': poss.final_epv,
                'max_epv': poss.max_epv,
                'epv_change': poss.final_epv - poss.initial_epv,
                'outcome': poss.outcome,
                'frames': len(poss.frames),
            })

        return pd.DataFrame(summaries)

    def get_player_epv_contribution(
        self,
        player_id: str
    ) -> Dict[str, float]:
        """
        Calculate a player's EPV contribution.

        Args:
            player_id: Player to analyze

        Returns:
            Dictionary with EPV contribution metrics
        """
        total_epv_added = 0.0
        actions = 0

        for poss in self.possession_sequences:
            for i, frame in enumerate(poss.frames[1:], 1):
                if frame.action_player_id == player_id and frame.action_value:
                    total_epv_added += frame.action_value
                    actions += 1

        return {
            'player_id': player_id,
            'total_epv_added': total_epv_added,
            'actions': actions,
            'epv_per_action': total_epv_added / max(actions, 1),
        }


def create_sample_possession() -> List[PossessionFrame]:
    """Create sample possession data for testing."""
    np.random.seed(42)

    frames = []

    # Simulate a 10-second possession moving toward goal
    for i in range(50):  # 50 frames at ~5 fps
        timestamp = i * 0.2

        # Puck advances toward goal
        puck_x = 100 + i * 2 + np.random.uniform(-2, 2)
        puck_x = min(195, puck_x)
        puck_y = 42.5 + np.random.uniform(-5, 5)

        # Home team positions (attacking)
        home_positions = {}
        for j in range(5):
            px = puck_x - np.random.uniform(5, 20)
            py = 10 + j * 15 + np.random.uniform(-3, 3)
            home_positions[f"home_{j}"] = (px, py, np.random.uniform(5, 15), 0)

        # Away team positions (defending)
        away_positions = {}
        for j in range(5):
            px = puck_x + np.random.uniform(5, 15)
            py = 10 + j * 15 + np.random.uniform(-3, 3)
            away_positions[f"away_{j}"] = (px, py, np.random.uniform(-10, 0), 0)

        # Determine action
        action_type = None
        if i > 0 and np.random.random() < 0.1:
            action_type = np.random.choice([ActionType.PASS, ActionType.CARRY])

        frame = PossessionFrame(
            frame_id=i,
            timestamp=timestamp,
            game_id="game_001",
            period=1,
            possessing_team="home",
            possession_state=PossessionState.ATTACKING,
            puck_x=puck_x,
            puck_y=puck_y,
            puck_carrier_id="home_0",
            home_positions=home_positions,
            away_positions=away_positions,
            action_type=action_type,
            action_player_id="home_0" if action_type else None,
        )

        frames.append(frame)

    return frames


if __name__ == "__main__":
    # Demo the EPV model
    print("Creating EPV Model...")

    model = EPVModel(grid_x=40, grid_y=20)

    # Create sample possession
    print("Generating sample possession...")
    frames = create_sample_possession()

    # Track possession
    print("Tracking possession...")
    for frame in frames:
        model.track_possession(frame)

    # Force end possession
    model._end_possession(frames[-1])

    # Analyze single frame
    print("\nSingle Frame EPV Analysis:")
    sample_frame = frames[25]
    epv_result = model.calculate_epv(sample_frame, include_decomposition=True)
    print(f"  EPV: {epv_result['epv']:.4f}")
    if 'decomposition' in epv_result:
        decomp = epv_result['decomposition']
        print(f"  Shot value: {decomp['shot_value']:.4f}")
        print(f"  Pass value: {decomp['pass_value']:.4f}")
        print(f"  Carry value: {decomp['carry_value']:.4f}")
        print(f"  Turnover cost: {decomp['turnover_cost']:.4f}")

    # Action value
    print("\nAction Value Analysis:")
    action_value = model.calculate_action_value(frames[20], frames[25])
    print(f"  Value added: {action_value['value_added']:.4f}")
    print(f"  EPV before: {action_value['epv_before']:.4f}")
    print(f"  EPV after: {action_value['epv_after']:.4f}")

    # Possession summary
    print("\nPossession Summary:")
    summary = model.get_possession_summary()
    if not summary.empty:
        for _, row in summary.iterrows():
            print(f"  Duration: {row['duration']:.1f}s")
            print(f"  Initial EPV: {row['initial_epv']:.4f}")
            print(f"  Max EPV: {row['max_epv']:.4f}")
            print(f"  Final EPV: {row['final_epv']:.4f}")
            print(f"  EPV change: {row['epv_change']:.4f}")

    # EPV trajectory
    if model.possession_sequences:
        trajectory = model.possession_sequences[0].epv_trajectory
        print(f"\nEPV Trajectory (first 10 frames):")
        for i, epv in enumerate(trajectory[:10]):
            print(f"  Frame {i}: {epv:.4f}")
