"""
Ghosting Model - Optimal Defender Trajectory Simulation

Implements methodology from:
- Lucey, P., et al. (2013). "Representing and Discovering Adversarial Team
  Behaviors using Player Roles." MIT Sloan Sports Analytics Conference.

Key concept: "Ghosting" simulates what optimal defenders SHOULD have done
given the offensive situation, allowing comparison to actual performance.

Hockey translation:
- Simulate optimal defensive positioning
- Evaluate defensive decision quality
- Generate coaching feedback on positioning
- Identify defensive breakdowns in real-time
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from datetime import datetime


class DefensiveRole(Enum):
    """Defensive roles/assignments."""
    PUCK_CARRIER = "puck_carrier"  # Pressure the puck
    STRONG_SIDE_SUPPORT = "strong_side_support"  # Support pressure
    WEAK_SIDE_COVERAGE = "weak_side_coverage"  # Cover weak side
    SLOT_PROTECTION = "slot_protection"  # Protect high danger area
    NET_FRONT = "net_front"  # Crease coverage
    POINT_COVERAGE = "point_coverage"  # Cover point shooters


class DefensiveScheme(Enum):
    """Defensive system types."""
    MAN_TO_MAN = "man_to_man"
    ZONE = "zone"
    HYBRID = "hybrid"
    COLLAPSING = "collapsing"
    AGGRESSIVE = "aggressive"


@dataclass
class PlayerState:
    """Current state of a player."""
    player_id: str
    x: float  # meters
    y: float  # meters
    vx: float = 0.0
    vy: float = 0.0
    is_goalie: bool = False

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])


@dataclass
class PuckState:
    """Current state of the puck."""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    carrier_id: Optional[str] = None

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class GameSnapshot:
    """Single frame of game state."""
    timestamp: float
    offensive_players: List[PlayerState]
    defensive_players: List[PlayerState]
    goalie: PlayerState
    puck: PuckState


@dataclass
class GhostPosition:
    """Optimal "ghost" position for a defender."""
    player_id: str
    optimal_x: float
    optimal_y: float
    actual_x: float
    actual_y: float
    role: DefensiveRole
    positioning_error: float  # Distance from optimal
    urgency: float  # How critical this position is

    @property
    def optimal_position(self) -> np.ndarray:
        return np.array([self.optimal_x, self.optimal_y])

    @property
    def actual_position(self) -> np.ndarray:
        return np.array([self.actual_x, self.actual_y])


@dataclass
class GhostingResult:
    """Result of ghosting analysis for a frame."""
    timestamp: float
    ghost_positions: List[GhostPosition]
    team_positioning_score: float  # 0-1, higher is better
    vulnerable_areas: List[Tuple[float, float]]  # Unprotected dangerous spots
    scheme_detected: DefensiveScheme
    breakdown_risk: float  # Risk of allowing high-danger chance


@dataclass
class RoleAssignment:
    """Defensive role assignment."""
    defender_id: str
    role: DefensiveRole
    target_player_id: Optional[str]  # For man coverage
    target_zone: Optional[Tuple[float, float, float, float]]  # For zone coverage


class OptimalPositionCalculator:
    """
    Calculate optimal defensive positions.

    Uses geometric principles and threat analysis.
    """

    def __init__(
        self,
        rink_length: float = 60.96,
        rink_width: float = 25.91
    ):
        self.rink_length = rink_length
        self.rink_width = rink_width

        # Goal position (defensive end)
        self.goal_x = -rink_length / 2
        self.goal_y = 0.0
        self.goal_width = 1.83  # 6 feet

        # Danger zones
        self.slot_bounds = (-rink_length/2, -rink_length/2 + 10, -5, 5)
        self.high_danger_radius = 8.0

    def shooting_lane_position(
        self,
        shooter: PlayerState,
        defender: PlayerState
    ) -> np.ndarray:
        """
        Calculate optimal position to block shooting lane.

        Position between shooter and goal center.
        """
        shooter_pos = shooter.position
        goal_pos = np.array([self.goal_x, self.goal_y])

        # Vector from shooter to goal
        to_goal = goal_pos - shooter_pos
        dist = np.linalg.norm(to_goal)

        if dist < 0.1:
            return shooter_pos

        # Position 1/3 of the way from shooter to goal
        optimal = shooter_pos + to_goal * 0.33

        return optimal

    def passing_lane_position(
        self,
        passer: PlayerState,
        receiver: PlayerState,
        defender: PlayerState
    ) -> np.ndarray:
        """
        Calculate optimal position to intercept pass.

        Position between passer and receiver, weighted by danger.
        """
        passer_pos = passer.position
        receiver_pos = receiver.position

        # Midpoint
        midpoint = (passer_pos + receiver_pos) / 2

        # Adjust toward more dangerous player
        receiver_danger = self._position_danger(receiver_pos)
        passer_danger = self._position_danger(passer_pos)

        total_danger = receiver_danger + passer_danger + 0.01
        weight = receiver_danger / total_danger

        # Weighted position toward more dangerous player
        optimal = passer_pos * (1 - weight) + receiver_pos * weight

        # But stay in passing lane
        lane_dir = receiver_pos - passer_pos
        lane_dir = lane_dir / (np.linalg.norm(lane_dir) + 1e-8)

        # Project optimal onto lane
        proj = np.dot(optimal - passer_pos, lane_dir)
        proj = np.clip(proj, 0, np.linalg.norm(receiver_pos - passer_pos))
        optimal = passer_pos + lane_dir * proj

        return optimal

    def slot_protection_position(
        self,
        offensive_players: List[PlayerState]
    ) -> np.ndarray:
        """
        Calculate optimal slot protection position.

        Guard the slot area while tracking threats.
        """
        # Center of slot
        slot_center = np.array([self.goal_x + 6, 0])

        # Adjust based on offensive player positions
        threats_in_slot = []
        for player in offensive_players:
            if self._in_slot(player.position):
                threats_in_slot.append(player.position)

        if threats_in_slot:
            # Move toward average threat position
            avg_threat = np.mean(threats_in_slot, axis=0)
            optimal = 0.7 * slot_center + 0.3 * avg_threat
        else:
            optimal = slot_center

        return optimal

    def _position_danger(self, pos: np.ndarray) -> float:
        """Calculate danger level of a position."""
        goal_pos = np.array([self.goal_x, self.goal_y])
        dist_to_goal = np.linalg.norm(pos - goal_pos)

        # Danger decreases with distance
        danger = np.exp(-dist_to_goal / 10)

        # Higher danger in slot
        if self._in_slot(pos):
            danger *= 1.5

        return danger

    def _in_slot(self, pos: np.ndarray) -> bool:
        """Check if position is in the slot."""
        x_min, x_max, y_min, y_max = self.slot_bounds
        return x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max


class RoleAssigner:
    """
    Assign defensive roles to players.

    Uses Hungarian algorithm for optimal assignment.
    """

    def __init__(self, position_calculator: OptimalPositionCalculator):
        self.calc = position_calculator

    def assign_roles(
        self,
        defenders: List[PlayerState],
        offenders: List[PlayerState],
        puck: PuckState,
        scheme: DefensiveScheme = DefensiveScheme.ZONE
    ) -> List[RoleAssignment]:
        """Assign defensive roles based on scheme."""
        if scheme == DefensiveScheme.MAN_TO_MAN:
            return self._assign_man_to_man(defenders, offenders, puck)
        elif scheme == DefensiveScheme.ZONE:
            return self._assign_zone(defenders, offenders, puck)
        else:
            return self._assign_hybrid(defenders, offenders, puck)

    def _assign_man_to_man(
        self,
        defenders: List[PlayerState],
        offenders: List[PlayerState],
        puck: PuckState
    ) -> List[RoleAssignment]:
        """Assign each defender to mark an offensive player."""
        assignments = []

        # Find puck carrier
        carrier = None
        for off in offenders:
            if off.player_id == puck.carrier_id:
                carrier = off
                break

        # Rank offenders by danger
        offender_danger = [
            (off, self.calc._position_danger(off.position))
            for off in offenders
        ]
        offender_danger.sort(key=lambda x: x[1], reverse=True)

        # Greedy assignment (in practice use Hungarian)
        assigned_defenders = set()
        assigned_offenders = set()

        for off, _ in offender_danger:
            best_defender = None
            best_dist = float('inf')

            for defender in defenders:
                if defender.player_id in assigned_defenders:
                    continue

                dist = np.linalg.norm(defender.position - off.position)
                if dist < best_dist:
                    best_dist = dist
                    best_defender = defender

            if best_defender:
                role = DefensiveRole.PUCK_CARRIER if off == carrier else DefensiveRole.STRONG_SIDE_SUPPORT
                assignments.append(RoleAssignment(
                    defender_id=best_defender.player_id,
                    role=role,
                    target_player_id=off.player_id,
                    target_zone=None
                ))
                assigned_defenders.add(best_defender.player_id)
                assigned_offenders.add(off.player_id)

        return assignments

    def _assign_zone(
        self,
        defenders: List[PlayerState],
        offenders: List[PlayerState],
        puck: PuckState
    ) -> List[RoleAssignment]:
        """Assign defenders to zones."""
        assignments = []

        # Define zones
        zones = {
            DefensiveRole.PUCK_CARRIER: None,  # Follow puck
            DefensiveRole.SLOT_PROTECTION: (-30, -22, -4, 4),
            DefensiveRole.WEAK_SIDE_COVERAGE: (-30, -15, 4, 12),
            DefensiveRole.STRONG_SIDE_SUPPORT: (-30, -15, -12, -4),
            DefensiveRole.POINT_COVERAGE: (-15, -5, -10, 10),
        }

        # Assign based on current position proximity
        assigned = set()
        roles_needed = list(zones.keys())

        for role in roles_needed:
            zone = zones[role]
            if zone is None:
                # Puck carrier - assign closest to puck
                best_defender = None
                best_dist = float('inf')

                for d in defenders:
                    if d.player_id in assigned:
                        continue
                    dist = np.linalg.norm(d.position - puck.position)
                    if dist < best_dist:
                        best_dist = dist
                        best_defender = d

                if best_defender:
                    assignments.append(RoleAssignment(
                        defender_id=best_defender.player_id,
                        role=role,
                        target_player_id=puck.carrier_id,
                        target_zone=None
                    ))
                    assigned.add(best_defender.player_id)
            else:
                # Zone assignment
                zone_center = np.array([
                    (zone[0] + zone[1]) / 2,
                    (zone[2] + zone[3]) / 2
                ])

                best_defender = None
                best_dist = float('inf')

                for d in defenders:
                    if d.player_id in assigned:
                        continue
                    dist = np.linalg.norm(d.position - zone_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_defender = d

                if best_defender:
                    assignments.append(RoleAssignment(
                        defender_id=best_defender.player_id,
                        role=role,
                        target_player_id=None,
                        target_zone=zone
                    ))
                    assigned.add(best_defender.player_id)

        return assignments

    def _assign_hybrid(
        self,
        defenders: List[PlayerState],
        offenders: List[PlayerState],
        puck: PuckState
    ) -> List[RoleAssignment]:
        """Hybrid zone + man coverage."""
        # Man coverage on puck carrier and nearest threat
        # Zone for others
        assignments = []

        carrier = None
        for off in offenders:
            if off.player_id == puck.carrier_id:
                carrier = off
                break

        if carrier:
            # Assign defender to carrier (man)
            closest_to_carrier = min(
                defenders,
                key=lambda d: np.linalg.norm(d.position - carrier.position)
            )
            assignments.append(RoleAssignment(
                defender_id=closest_to_carrier.player_id,
                role=DefensiveRole.PUCK_CARRIER,
                target_player_id=carrier.player_id,
                target_zone=None
            ))

            # Rest play zone
            remaining = [d for d in defenders if d.player_id != closest_to_carrier.player_id]
            zone_assignments = self._assign_zone(remaining, offenders, puck)
            assignments.extend(zone_assignments)

        return assignments


class GhostingModel:
    """
    Main ghosting model for defensive analysis.

    Generates "ghost" optimal positions and evaluates defense.
    """

    def __init__(
        self,
        scheme: DefensiveScheme = DefensiveScheme.ZONE,
        max_speed: float = 10.0  # m/s
    ):
        self.calc = OptimalPositionCalculator()
        self.assigner = RoleAssigner(self.calc)
        self.scheme = scheme
        self.max_speed = max_speed

    def generate_ghosts(
        self,
        snapshot: GameSnapshot
    ) -> GhostingResult:
        """
        Generate ghost positions for all defenders.

        Returns optimal positions and evaluation.
        """
        # Assign roles
        assignments = self.assigner.assign_roles(
            snapshot.defensive_players,
            snapshot.offensive_players,
            snapshot.puck,
            self.scheme
        )

        # Calculate optimal position for each assignment
        ghost_positions = []

        for assignment in assignments:
            defender = next(
                (d for d in snapshot.defensive_players if d.player_id == assignment.defender_id),
                None
            )
            if defender is None:
                continue

            optimal = self._calculate_optimal_position(
                assignment,
                snapshot.offensive_players,
                snapshot.puck
            )

            error = np.linalg.norm(defender.position - optimal)
            urgency = self._calculate_urgency(
                optimal, snapshot.offensive_players, snapshot.puck
            )

            ghost_positions.append(GhostPosition(
                player_id=defender.player_id,
                optimal_x=optimal[0],
                optimal_y=optimal[1],
                actual_x=defender.x,
                actual_y=defender.y,
                role=assignment.role,
                positioning_error=error,
                urgency=urgency
            ))

        # Calculate team score
        team_score = self._calculate_team_score(ghost_positions)

        # Find vulnerable areas
        vulnerable = self._find_vulnerable_areas(
            snapshot.defensive_players,
            snapshot.offensive_players,
            snapshot.puck
        )

        # Breakdown risk
        breakdown_risk = self._calculate_breakdown_risk(
            ghost_positions, vulnerable
        )

        return GhostingResult(
            timestamp=snapshot.timestamp,
            ghost_positions=ghost_positions,
            team_positioning_score=team_score,
            vulnerable_areas=vulnerable,
            scheme_detected=self.scheme,
            breakdown_risk=breakdown_risk
        )

    def _calculate_optimal_position(
        self,
        assignment: RoleAssignment,
        offenders: List[PlayerState],
        puck: PuckState
    ) -> np.ndarray:
        """Calculate optimal position for role."""
        if assignment.role == DefensiveRole.PUCK_CARRIER:
            # Pressure the puck
            return puck.position + np.array([2, 0])  # Offset toward defensive zone

        elif assignment.role == DefensiveRole.SLOT_PROTECTION:
            return self.calc.slot_protection_position(offenders)

        elif assignment.target_player_id:
            # Man coverage - position between player and goal
            target = next(
                (o for o in offenders if o.player_id == assignment.target_player_id),
                None
            )
            if target:
                return self.calc.shooting_lane_position(target, PlayerState("", 0, 0))

        elif assignment.target_zone:
            # Zone coverage - center of zone
            zone = assignment.target_zone
            return np.array([
                (zone[0] + zone[1]) / 2,
                (zone[2] + zone[3]) / 2
            ])

        return np.array([0, 0])

    def _calculate_urgency(
        self,
        position: np.ndarray,
        offenders: List[PlayerState],
        puck: PuckState
    ) -> float:
        """Calculate urgency of reaching position."""
        # Distance to puck
        puck_dist = np.linalg.norm(position - puck.position)

        # Nearest offensive threat
        min_threat_dist = float('inf')
        for off in offenders:
            dist = np.linalg.norm(position - off.position)
            min_threat_dist = min(min_threat_dist, dist)

        # High urgency if close to puck or threat
        urgency = np.exp(-min(puck_dist, min_threat_dist) / 5)
        return urgency

    def _calculate_team_score(
        self,
        ghost_positions: List[GhostPosition]
    ) -> float:
        """Calculate team positioning score."""
        if not ghost_positions:
            return 0.5

        # Weighted average of individual errors
        total_weight = 0.0
        weighted_score = 0.0

        for ghost in ghost_positions:
            # Score based on error distance
            individual_score = np.exp(-ghost.positioning_error / 3)

            # Weight by urgency
            weight = ghost.urgency
            weighted_score += individual_score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_score / total_weight
        return 0.5

    def _find_vulnerable_areas(
        self,
        defenders: List[PlayerState],
        offenders: List[PlayerState],
        puck: PuckState
    ) -> List[Tuple[float, float]]:
        """Find unprotected dangerous areas."""
        vulnerable = []

        # Check slot area
        slot_covered = False
        for d in defenders:
            if self.calc._in_slot(d.position):
                slot_covered = True
                break

        if not slot_covered:
            # Find offensive players in slot
            for off in offenders:
                if self.calc._in_slot(off.position):
                    vulnerable.append((off.x, off.y))

        # Check passing lanes
        for off in offenders:
            if off.player_id == puck.carrier_id:
                continue

            # Is passing lane covered?
            lane_covered = False
            for d in defenders:
                # Check if defender is between puck and receiver
                puck_to_off = off.position - puck.position
                puck_to_def = d.position - puck.position

                if np.linalg.norm(puck_to_off) > 0.1:
                    proj = np.dot(puck_to_def, puck_to_off) / np.dot(puck_to_off, puck_to_off)
                    if 0 < proj < 1:
                        closest_on_lane = puck.position + proj * puck_to_off
                        if np.linalg.norm(d.position - closest_on_lane) < 2:
                            lane_covered = True
                            break

            if not lane_covered and self.calc._position_danger(off.position) > 0.3:
                vulnerable.append((off.x, off.y))

        return vulnerable

    def _calculate_breakdown_risk(
        self,
        ghost_positions: List[GhostPosition],
        vulnerable: List[Tuple[float, float]]
    ) -> float:
        """Calculate risk of defensive breakdown."""
        # Base risk from positioning errors
        avg_error = np.mean([g.positioning_error for g in ghost_positions]) if ghost_positions else 0

        # Additional risk from vulnerable areas
        vulnerable_risk = len(vulnerable) * 0.15

        # Combine
        risk = 1 - np.exp(-avg_error / 5) + vulnerable_risk
        return min(risk, 1.0)


class DefensiveEvaluator:
    """
    Evaluate defensive performance using ghosting.
    """

    def __init__(self, ghosting_model: GhostingModel):
        self.model = ghosting_model

    def evaluate_sequence(
        self,
        snapshots: List[GameSnapshot]
    ) -> Dict[str, float]:
        """Evaluate defense over a sequence of frames."""
        all_results = [self.model.generate_ghosts(s) for s in snapshots]

        # Aggregate metrics
        avg_team_score = np.mean([r.team_positioning_score for r in all_results])
        avg_breakdown_risk = np.mean([r.breakdown_risk for r in all_results])
        max_breakdown_risk = max(r.breakdown_risk for r in all_results)

        # Per-player analysis
        player_errors = {}
        for result in all_results:
            for ghost in result.ghost_positions:
                if ghost.player_id not in player_errors:
                    player_errors[ghost.player_id] = []
                player_errors[ghost.player_id].append(ghost.positioning_error)

        player_avg_errors = {
            pid: np.mean(errors) for pid, errors in player_errors.items()
        }

        return {
            'avg_team_score': avg_team_score,
            'avg_breakdown_risk': avg_breakdown_risk,
            'max_breakdown_risk': max_breakdown_risk,
            'player_errors': player_avg_errors,
            'n_frames': len(snapshots)
        }

    def generate_feedback(
        self,
        result: GhostingResult
    ) -> List[str]:
        """Generate coaching feedback from ghosting result."""
        feedback = []

        if result.team_positioning_score < 0.6:
            feedback.append("Team defensive structure is compromised")

        for ghost in result.ghost_positions:
            if ghost.positioning_error > 5 and ghost.urgency > 0.5:
                feedback.append(
                    f"Player {ghost.player_id}: {ghost.positioning_error:.1f}m out of position "
                    f"({ghost.role.value})"
                )

        if result.vulnerable_areas:
            areas = ", ".join([f"({x:.1f}, {y:.1f})" for x, y in result.vulnerable_areas[:3]])
            feedback.append(f"Vulnerable areas: {areas}")

        if result.breakdown_risk > 0.7:
            feedback.append("HIGH DANGER: Defensive breakdown imminent")

        return feedback if feedback else ["Defensive positioning is adequate"]
