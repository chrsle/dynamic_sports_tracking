"""
Multi-Agent Strategy Optimization

Implements methodology from:
- Chu, D., et al. (2020). "At the Helm: Learning Multi-Agent Strategy."
  (Sailing optimization adapted to team sports)

Key concept: Optimize coordinated strategies for multiple agents
(players) simultaneously, considering interactions and constraints.

Hockey translation:
- Line deployment optimization
- Power play unit coordination
- Forechecking system coordination
- Defensive coverage optimization
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from datetime import datetime


class AgentRole(Enum):
    """Roles for agents in hockey context."""
    # Forward roles
    PUCK_CARRIER = "puck_carrier"
    SUPPORT_STRONG = "support_strong"
    SUPPORT_WEAK = "support_weak"
    NET_FRONT = "net_front"
    HIGH_SLOT = "high_slot"

    # Defensive roles
    PRESSURE = "pressure"
    CONTAIN = "contain"
    COVER_SLOT = "cover_slot"
    COVER_POINT = "cover_point"
    BACKSIDE = "backside"

    # Neutral
    TRANSITION = "transition"


class TeamObjective(Enum):
    """High-level team objectives."""
    SCORE = "score"  # Maximize scoring chances
    DEFEND = "defend"  # Minimize opponent chances
    POSSESS = "possess"  # Maintain possession
    TRANSITION_FAST = "transition_fast"  # Quick counters
    GRIND = "grind"  # Physical play, wear down opponent


@dataclass
class AgentState:
    """State of a single agent."""
    agent_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    energy: float = 1.0  # Fatigue level
    has_puck: bool = False
    current_role: Optional[AgentRole] = None


@dataclass
class TeamState:
    """State of the entire team."""
    agents: List[AgentState]
    puck_position: Tuple[float, float]
    puck_possession: str  # "home", "away", or "contested"
    score_differential: int
    time_remaining: float
    strength_state: str = "5v5"


@dataclass
class AgentAction:
    """Action for a single agent."""
    agent_id: str
    target_x: float
    target_y: float
    action_type: str  # "skate", "pass", "shoot", "check"
    target_agent: Optional[str] = None  # For passes/checks


@dataclass
class TeamAction:
    """Coordinated actions for all agents."""
    actions: List[AgentAction]
    objective: TeamObjective
    expected_value: float


@dataclass
class CoordinationConstraint:
    """Constraint on agent coordination."""
    constraint_type: str  # "spacing", "coverage", "support"
    involved_agents: List[str]
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class AgentValueFunction:
    """
    Value function for individual agent.

    Evaluates position/action quality given team context.
    """

    def __init__(self, rink_length: float = 200, rink_width: float = 85):
        self.rink_length = rink_length
        self.rink_width = rink_width

        # Position value grid (offensive end)
        self.position_values = self._create_value_grid()

    def _create_value_grid(self) -> np.ndarray:
        """Create grid of position values."""
        grid = np.zeros((20, 10))

        for i in range(20):
            for j in range(10):
                x = (i / 20) * self.rink_length - self.rink_length / 2
                y = (j / 10) * self.rink_width - self.rink_width / 2

                # Higher value near offensive goal
                goal_x = self.rink_length / 2 - 11
                dist_to_goal = np.sqrt((x - goal_x)**2 + y**2)
                grid[i, j] = np.exp(-dist_to_goal / 30)

                # Slot bonus
                if x > goal_x - 30 and abs(y) < 15:
                    grid[i, j] *= 1.5

        return grid / np.max(grid)

    def evaluate_position(
        self,
        agent: AgentState,
        team_state: TeamState
    ) -> float:
        """Evaluate value of agent's position."""
        # Grid lookup
        i = int((agent.x + self.rink_length/2) / self.rink_length * 19)
        j = int((agent.y + self.rink_width/2) / self.rink_width * 9)
        i = np.clip(i, 0, 19)
        j = np.clip(j, 0, 9)

        base_value = self.position_values[i, j]

        # Puck carrier bonus
        if agent.has_puck:
            base_value *= 1.3

        # Energy factor
        base_value *= agent.energy

        return base_value

    def evaluate_action(
        self,
        agent: AgentState,
        action: AgentAction,
        team_state: TeamState
    ) -> float:
        """Evaluate value of potential action."""
        # Simulate action result
        new_state = AgentState(
            agent_id=agent.agent_id,
            x=action.target_x,
            y=action.target_y,
            vx=0, vy=0,
            energy=agent.energy - 0.02,  # Action cost
            has_puck=action.action_type == "carry",
            current_role=agent.current_role
        )

        return self.evaluate_position(new_state, team_state)


class CoordinationOptimizer:
    """
    Optimize coordinated actions for multiple agents.

    Uses constrained optimization to find best team strategy.
    """

    def __init__(
        self,
        spacing_constraint: float = 15,  # Minimum feet between teammates
        max_iterations: int = 100
    ):
        self.spacing = spacing_constraint
        self.max_iter = max_iterations
        self.value_fn = AgentValueFunction()

    def optimize(
        self,
        team_state: TeamState,
        objective: TeamObjective,
        constraints: Optional[List[CoordinationConstraint]] = None
    ) -> TeamAction:
        """
        Find optimal coordinated actions.

        Uses gradient-free optimization with constraints.
        """
        constraints = constraints or []

        best_actions = None
        best_value = -float('inf')

        # Random search with constraint filtering
        for _ in range(self.max_iter):
            actions = self._generate_candidate_actions(team_state, objective)

            if self._satisfies_constraints(actions, team_state, constraints):
                value = self._evaluate_team_action(actions, team_state, objective)

                if value > best_value:
                    best_value = value
                    best_actions = actions

        if best_actions is None:
            # Fallback: simple actions without constraints
            best_actions = self._generate_default_actions(team_state)
            best_value = self._evaluate_team_action(best_actions, team_state, objective)

        return TeamAction(
            actions=best_actions,
            objective=objective,
            expected_value=best_value
        )

    def _generate_candidate_actions(
        self,
        team_state: TeamState,
        objective: TeamObjective
    ) -> List[AgentAction]:
        """Generate candidate actions based on objective."""
        actions = []

        for agent in team_state.agents:
            if objective == TeamObjective.SCORE:
                # Move toward offensive zone
                target_x = agent.x + np.random.uniform(5, 20)
                target_y = agent.y + np.random.uniform(-10, 10)
            elif objective == TeamObjective.DEFEND:
                # Move toward defensive zone
                target_x = agent.x - np.random.uniform(5, 20)
                target_y = agent.y + np.random.uniform(-10, 10)
            else:
                # Random movement
                target_x = agent.x + np.random.uniform(-15, 15)
                target_y = agent.y + np.random.uniform(-10, 10)

            # Clamp to rink bounds
            target_x = np.clip(target_x, -100, 100)
            target_y = np.clip(target_y, -42.5, 42.5)

            actions.append(AgentAction(
                agent_id=agent.agent_id,
                target_x=target_x,
                target_y=target_y,
                action_type="skate"
            ))

        return actions

    def _generate_default_actions(
        self,
        team_state: TeamState
    ) -> List[AgentAction]:
        """Generate simple default actions."""
        return [
            AgentAction(
                agent_id=agent.agent_id,
                target_x=agent.x,
                target_y=agent.y,
                action_type="skate"
            )
            for agent in team_state.agents
        ]

    def _satisfies_constraints(
        self,
        actions: List[AgentAction],
        team_state: TeamState,
        constraints: List[CoordinationConstraint]
    ) -> bool:
        """Check if actions satisfy all constraints."""
        # Build position map
        positions = {a.agent_id: (a.target_x, a.target_y) for a in actions}

        for constraint in constraints:
            if constraint.constraint_type == "spacing":
                # Check minimum spacing
                involved = constraint.involved_agents
                for i, a1 in enumerate(involved):
                    for a2 in involved[i+1:]:
                        if a1 in positions and a2 in positions:
                            dist = np.sqrt(
                                (positions[a1][0] - positions[a2][0])**2 +
                                (positions[a1][1] - positions[a2][1])**2
                            )
                            min_dist = constraint.min_value or self.spacing
                            if dist < min_dist:
                                return False

        return True

    def _evaluate_team_action(
        self,
        actions: List[AgentAction],
        team_state: TeamState,
        objective: TeamObjective
    ) -> float:
        """Evaluate total value of team actions."""
        total_value = 0.0

        for action in actions:
            agent = next(
                (a for a in team_state.agents if a.agent_id == action.agent_id),
                None
            )
            if agent:
                value = self.value_fn.evaluate_action(agent, action, team_state)
                total_value += value

        # Objective bonuses
        if objective == TeamObjective.SCORE:
            # Bonus for concentration near goal
            goal_x = 89  # Offensive goal
            avg_x = np.mean([a.target_x for a in actions])
            total_value += max(0, (avg_x - 50) / 40)

        return total_value


class RoleAssigner:
    """
    Assign roles to agents dynamically.
    """

    def __init__(self):
        self.role_priorities = {
            TeamObjective.SCORE: [
                AgentRole.PUCK_CARRIER,
                AgentRole.NET_FRONT,
                AgentRole.HIGH_SLOT,
                AgentRole.SUPPORT_STRONG,
                AgentRole.SUPPORT_WEAK
            ],
            TeamObjective.DEFEND: [
                AgentRole.PRESSURE,
                AgentRole.COVER_SLOT,
                AgentRole.CONTAIN,
                AgentRole.COVER_POINT,
                AgentRole.BACKSIDE
            ]
        }

    def assign_roles(
        self,
        team_state: TeamState,
        objective: TeamObjective
    ) -> Dict[str, AgentRole]:
        """Assign roles to all agents."""
        assignments = {}
        roles = self.role_priorities.get(
            objective,
            list(AgentRole)[:len(team_state.agents)]
        )

        # Sort agents by position
        sorted_agents = sorted(
            team_state.agents,
            key=lambda a: a.x,
            reverse=(objective == TeamObjective.SCORE)
        )

        # Assign puck carrier first
        puck_carrier = next(
            (a for a in team_state.agents if a.has_puck),
            None
        )
        if puck_carrier:
            assignments[puck_carrier.agent_id] = AgentRole.PUCK_CARRIER
            sorted_agents = [a for a in sorted_agents if a.agent_id != puck_carrier.agent_id]
            roles = [r for r in roles if r != AgentRole.PUCK_CARRIER]

        # Assign remaining roles
        for agent, role in zip(sorted_agents, roles):
            assignments[agent.agent_id] = role

        return assignments


class LineOptimizer:
    """
    Optimize line combinations using multi-agent framework.
    """

    def __init__(self):
        self.coord_optimizer = CoordinationOptimizer()
        self.role_assigner = RoleAssigner()

    def optimize_deployment(
        self,
        available_players: List[Dict],
        game_state: Dict,
        opponent_on_ice: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Optimize which players should be on ice.

        Returns recommended line combinations.
        """
        # Score players for current situation
        player_scores = {}

        objective = self._determine_objective(game_state)

        for player in available_players:
            score = self._score_player(player, objective, game_state)
            player_scores[player['id']] = score

        # Select top players by position
        forwards = [p for p in available_players if p['position'] in ['C', 'LW', 'RW']]
        defensemen = [p for p in available_players if p['position'] in ['LD', 'RD']]

        # Sort by score
        forwards.sort(key=lambda p: player_scores[p['id']], reverse=True)
        defensemen.sort(key=lambda p: player_scores[p['id']], reverse=True)

        return {
            'forwards': [f['id'] for f in forwards[:3]],
            'defensemen': [d['id'] for d in defensemen[:2]],
            'objective': objective.value
        }

    def _determine_objective(self, game_state: Dict) -> TeamObjective:
        """Determine team objective from game state."""
        score_diff = game_state.get('score_differential', 0)
        time_remaining = game_state.get('time_remaining', 1200)

        if score_diff < 0 and time_remaining < 300:
            return TeamObjective.SCORE
        elif score_diff > 0 and time_remaining < 300:
            return TeamObjective.DEFEND
        elif time_remaining < 60:
            return TeamObjective.SCORE if score_diff < 0 else TeamObjective.DEFEND
        else:
            return TeamObjective.POSSESS

    def _score_player(
        self,
        player: Dict,
        objective: TeamObjective,
        game_state: Dict
    ) -> float:
        """Score player for current objective."""
        base_score = player.get('overall_rating', 70) / 100

        # Objective-specific bonuses
        if objective == TeamObjective.SCORE:
            base_score += player.get('offensive_rating', 70) / 200
        elif objective == TeamObjective.DEFEND:
            base_score += player.get('defensive_rating', 70) / 200

        # Fatigue penalty
        fatigue = player.get('fatigue', 0)
        base_score *= (1 - fatigue * 0.3)

        return base_score


class PowerPlayOptimizer:
    """
    Optimize power play unit coordination.
    """

    def __init__(self):
        self.coord_optimizer = CoordinationOptimizer(spacing_constraint=12)

        # Standard PP formations
        self.formations = {
            'umbrella': [
                (60, 0),   # Point
                (75, 22),  # Left flank
                (75, -22), # Right flank
                (85, 0),   # Net front
                (72, 0),   # Bumper
            ],
            'overload': [
                (60, -10),  # Point
                (70, -25), # Low left
                (80, -15), # Net front
                (85, -5),  # Crease
                (72, -18), # Middle
            ],
            '1-3-1': [
                (55, 0),   # High point
                (75, 25),  # Right wing
                (75, -25), # Left wing
                (75, 0),   # Middle
                (88, 0),   # Net front
            ]
        }

    def recommend_formation(
        self,
        players: List[Dict],
        opponent_pk: str = "box"
    ) -> Dict:
        """Recommend PP formation based on personnel and opponent."""
        # Score each formation
        formation_scores = {}

        for name, positions in self.formations.items():
            score = self._score_formation(name, players, opponent_pk)
            formation_scores[name] = score

        best = max(formation_scores, key=formation_scores.get)

        return {
            'formation': best,
            'positions': self.formations[best],
            'player_assignments': self._assign_pp_roles(players, best),
            'score': formation_scores[best]
        }

    def _score_formation(
        self,
        formation: str,
        players: List[Dict],
        opponent_pk: str
    ) -> float:
        """Score formation for given personnel."""
        score = 0.5

        # Formation-specific scoring
        if formation == 'umbrella':
            # Good with strong point shot
            has_point_shot = any(p.get('shot_power', 0) > 80 for p in players)
            if has_point_shot:
                score += 0.2

        elif formation == 'overload':
            # Good with strong playmakers
            playmakers = sum(1 for p in players if p.get('passing', 0) > 80)
            score += playmakers * 0.1

        elif formation == '1-3-1':
            # Good against aggressive PK
            if opponent_pk == "aggressive":
                score += 0.3

        return score

    def _assign_pp_roles(
        self,
        players: List[Dict],
        formation: str
    ) -> Dict[str, str]:
        """Assign players to PP positions."""
        positions = self.formations[formation]
        assignments = {}

        # Sort players by attributes relevant to each position
        available = list(players)

        # Assign point (needs shot/passing)
        point_scores = [(p, p.get('shot_power', 0) + p.get('passing', 0)) for p in available]
        point_scores.sort(key=lambda x: x[1], reverse=True)
        if point_scores:
            assignments['point'] = point_scores[0][0]['id']
            available.remove(point_scores[0][0])

        # Assign net front (needs size/strength)
        net_scores = [(p, p.get('strength', 0) + p.get('size', 0)) for p in available]
        net_scores.sort(key=lambda x: x[1], reverse=True)
        if net_scores:
            assignments['net_front'] = net_scores[0][0]['id']
            available.remove(net_scores[0][0])

        # Assign remaining
        for i, p in enumerate(available[:3]):
            assignments[f'wing_{i+1}'] = p['id']

        return assignments
