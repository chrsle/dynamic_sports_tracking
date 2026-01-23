"""
Multi-Task Vision Tracking

Implements methodology from:
- "Multi-task Learning for Joint Re-identification, Team Affiliation, and
  Role Classification for Sports Visual Tracking." arXiv:2401.09942 (2024).

Key features:
- Single backbone for three tasks
- Part-based player representations (PRTReID)
- Team clustering without predefined classes
- Role classification: player, goalkeeper, referee, staff

Hockey translation:
- Jersey number recognition in hockey broadcasts
- Team identification from video
- Role classification including officials
- Enhanced tracking through occlusions
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from datetime import datetime


class PlayerRole(Enum):
    """Role classification for detected persons."""
    HOME_SKATER = "home_skater"
    AWAY_SKATER = "away_skater"
    HOME_GOALIE = "home_goalie"
    AWAY_GOALIE = "away_goalie"
    REFEREE = "referee"
    LINESMAN = "linesman"
    COACH = "coach"
    STAFF = "staff"
    FAN = "fan"
    UNKNOWN = "unknown"


class TrackingState(Enum):
    """State of a tracked object."""
    ACTIVE = "active"
    LOST = "lost"
    RECOVERED = "recovered"
    TERMINATED = "terminated"


@dataclass
class BoundingBox:
    """Bounding box for detected object."""
    x: float  # Top-left x
    y: float  # Top-left y
    width: float
    height: float
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: 'BoundingBox') -> float:
        """Intersection over Union with another box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class PlayerDetection:
    """Single player detection in a frame."""
    bbox: BoundingBox
    embedding: np.ndarray  # Re-ID feature vector
    role: PlayerRole
    role_confidence: float
    team_id: Optional[int] = None
    jersey_number: Optional[int] = None
    jersey_confidence: float = 0.0
    pose_keypoints: Optional[np.ndarray] = None


@dataclass
class Track:
    """Tracked object across frames."""
    track_id: int
    detections: List[Tuple[int, PlayerDetection]]  # (frame_id, detection)
    state: TrackingState = TrackingState.ACTIVE
    player_id: Optional[str] = None  # Matched to known player
    smoothed_embedding: Optional[np.ndarray] = None

    @property
    def last_detection(self) -> Optional[Tuple[int, PlayerDetection]]:
        return self.detections[-1] if self.detections else None

    @property
    def frames_lost(self) -> int:
        if self.state != TrackingState.LOST:
            return 0
        return 0  # Would need current frame to compute


@dataclass
class PartFeatures:
    """Part-based features for re-identification."""
    head: np.ndarray
    torso_upper: np.ndarray
    torso_lower: np.ndarray
    jersey: np.ndarray  # Specific region for number
    legs: np.ndarray
    global_feature: np.ndarray

    def concatenate(self) -> np.ndarray:
        """Combine all parts into single vector."""
        return np.concatenate([
            self.head, self.torso_upper, self.torso_lower,
            self.jersey, self.legs, self.global_feature
        ])


@dataclass
class ModelConfig:
    """Configuration for multi-task model."""
    embedding_dim: int = 256
    part_features_dim: int = 64
    num_parts: int = 5
    hidden_dim: int = 512
    num_teams: int = 2
    num_roles: int = len(PlayerRole)
    backbone: str = "resnet50"
    use_attention: bool = True


class BackboneEncoder:
    """
    Shared backbone encoder for all tasks.

    In practice, this would be a CNN like ResNet-50.
    Here we simulate with random projections.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # Simulated weights
        self.conv_weights = np.random.randn(3, config.hidden_dim) * 0.1
        self.global_pool_dim = config.hidden_dim

    def extract_features(
        self,
        image_crop: np.ndarray  # (H, W, 3)
    ) -> np.ndarray:
        """Extract backbone features from image crop."""
        # Simulate CNN forward pass
        # In reality: ResNet-50 → global average pool

        # Simple simulation: flatten and project
        if image_crop.ndim == 3:
            flat = image_crop.reshape(-1, 3)
            features = np.mean(flat, axis=0) @ self.conv_weights
        else:
            features = np.random.randn(self.config.hidden_dim)

        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        return features


class PartBasedEncoder:
    """
    Part-based representation for re-identification.

    PRTReID: Divides body into parts for robust matching
    even with partial occlusions.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.part_dims = config.part_features_dim
        self.num_parts = config.num_parts

        # Part-specific projections
        self.part_projections = [
            np.random.randn(config.hidden_dim, config.part_features_dim) * 0.1
            for _ in range(config.num_parts)
        ]

        # Global projection
        self.global_projection = np.random.randn(
            config.hidden_dim, config.embedding_dim
        ) * 0.1

    def extract_parts(
        self,
        backbone_features: np.ndarray,
        bbox: BoundingBox
    ) -> PartFeatures:
        """Extract part-based features."""
        # In reality: use spatial attention or fixed regions
        # Here: simulate with different projections

        part_features = []
        for proj in self.part_projections:
            feat = backbone_features @ proj
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            part_features.append(feat)

        global_feat = backbone_features @ self.global_projection
        global_feat = global_feat / (np.linalg.norm(global_feat) + 1e-8)

        return PartFeatures(
            head=part_features[0],
            torso_upper=part_features[1],
            torso_lower=part_features[2],
            jersey=part_features[3],
            legs=part_features[4],
            global_feature=global_feat
        )

    def compute_similarity(
        self,
        features1: PartFeatures,
        features2: PartFeatures
    ) -> float:
        """Compute similarity between two part-based representations."""
        # Part-wise cosine similarity
        similarities = []

        for f1, f2 in [
            (features1.head, features2.head),
            (features1.torso_upper, features2.torso_upper),
            (features1.torso_lower, features2.torso_lower),
            (features1.jersey, features2.jersey),
            (features1.legs, features2.legs),
        ]:
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
            similarities.append(sim)

        # Global similarity
        global_sim = np.dot(features1.global_feature, features2.global_feature)
        global_sim /= (np.linalg.norm(features1.global_feature) *
                      np.linalg.norm(features2.global_feature) + 1e-8)

        # Weighted combination
        part_weight = 0.4
        global_weight = 0.6

        return part_weight * np.mean(similarities) + global_weight * global_sim


class TeamClassifier:
    """
    Team affiliation classifier.

    Uses clustering + learned features to identify team
    without predefined jersey colors.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # Classifier weights
        self.W = np.random.randn(config.embedding_dim, config.num_teams) * 0.1
        self.b = np.zeros(config.num_teams)

        # Running cluster centers
        self.team_centers: Optional[np.ndarray] = None

    def classify(
        self,
        embedding: np.ndarray
    ) -> Tuple[int, float]:
        """Classify team from embedding."""
        logits = embedding @ self.W + self.b
        probs = np.exp(logits) / np.sum(np.exp(logits))

        team_id = np.argmax(probs)
        confidence = probs[team_id]

        return team_id, confidence

    def update_clusters(
        self,
        embeddings: List[np.ndarray],
        team_ids: List[int]
    ):
        """Update running cluster centers."""
        for team in range(self.config.num_teams):
            team_embeddings = [e for e, t in zip(embeddings, team_ids) if t == team]
            if team_embeddings:
                center = np.mean(team_embeddings, axis=0)
                if self.team_centers is None:
                    self.team_centers = np.zeros((self.config.num_teams, self.config.embedding_dim))
                self.team_centers[team] = 0.9 * self.team_centers[team] + 0.1 * center

    def classify_by_clustering(
        self,
        embedding: np.ndarray
    ) -> Tuple[int, float]:
        """Classify using cluster distance."""
        if self.team_centers is None:
            return 0, 0.5

        distances = [
            np.linalg.norm(embedding - center)
            for center in self.team_centers
        ]

        team_id = np.argmin(distances)
        min_dist = distances[team_id]
        max_dist = max(distances)

        confidence = 1 - (min_dist / (max_dist + 1e-8))
        return team_id, confidence


class RoleClassifier:
    """
    Role classification: skater, goalie, referee, etc.

    Uses appearance + context for classification.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # Classification head
        self.W = np.random.randn(config.embedding_dim, config.num_roles) * 0.1
        self.b = np.zeros(config.num_roles)

    def classify(
        self,
        embedding: np.ndarray,
        bbox: BoundingBox,
        ice_position: Optional[Tuple[float, float]] = None
    ) -> Tuple[PlayerRole, float]:
        """Classify role from appearance and context."""
        # Base classification from appearance
        logits = embedding @ self.W + self.b
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        # Context adjustments
        if ice_position is not None:
            # Goalie: typically near goal
            if abs(ice_position[0]) > 25:  # Near end boards
                if abs(ice_position[1]) < 5:  # Near center
                    probs[list(PlayerRole).index(PlayerRole.HOME_GOALIE)] *= 2
                    probs[list(PlayerRole).index(PlayerRole.AWAY_GOALIE)] *= 2

        # Re-normalize
        probs = probs / np.sum(probs)

        role_idx = np.argmax(probs)
        return list(PlayerRole)[role_idx], probs[role_idx]


class JerseyNumberRecognizer:
    """
    Recognize jersey numbers from video.

    Uses specialized CNN for digit recognition.
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim

        # Digit classifier (0-99 for hockey)
        self.W = np.random.randn(embedding_dim, 100) * 0.1
        self.b = np.zeros(100)

    def recognize(
        self,
        jersey_features: np.ndarray
    ) -> Tuple[Optional[int], float]:
        """Recognize jersey number from features."""
        logits = jersey_features @ self.W + self.b
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        best_idx = np.argmax(probs)
        confidence = probs[best_idx]

        if confidence < 0.3:
            return None, confidence

        return best_idx, confidence

    def recognize_with_priors(
        self,
        jersey_features: np.ndarray,
        valid_numbers: List[int]
    ) -> Tuple[Optional[int], float]:
        """Recognize with known valid numbers as prior."""
        logits = jersey_features @ self.W + self.b

        # Mask invalid numbers
        mask = np.full(100, -np.inf)
        for num in valid_numbers:
            if 0 <= num < 100:
                mask[num] = 0
        logits = logits + mask

        probs = np.exp(logits - np.max(logits))
        probs = probs / (np.sum(probs) + 1e-8)

        best_idx = np.argmax(probs)
        confidence = probs[best_idx]

        if confidence < 0.3 or best_idx not in valid_numbers:
            return None, confidence

        return best_idx, confidence


class MultiTaskTracker:
    """
    Full multi-task tracking system.

    Combines detection, re-ID, team classification,
    and role classification.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()

        self.backbone = BackboneEncoder(self.config)
        self.part_encoder = PartBasedEncoder(self.config)
        self.team_classifier = TeamClassifier(self.config)
        self.role_classifier = RoleClassifier(self.config)
        self.jersey_recognizer = JerseyNumberRecognizer(self.config.part_features_dim)

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0

        # Matching thresholds
        self.matching_threshold = 0.5
        self.max_frames_lost = 30

    def process_frame(
        self,
        frame_id: int,
        image: np.ndarray,
        detections: List[BoundingBox]
    ) -> List[PlayerDetection]:
        """Process a single frame."""
        results = []

        for bbox in detections:
            # Extract crop (simulated)
            crop = np.random.randn(128, 64, 3)  # Would be actual image crop

            # Backbone features
            backbone_feat = self.backbone.extract_features(crop)

            # Part-based features
            part_features = self.part_encoder.extract_parts(backbone_feat, bbox)

            # Team classification
            team_id, team_conf = self.team_classifier.classify(part_features.global_feature)

            # Role classification
            role, role_conf = self.role_classifier.classify(
                part_features.global_feature, bbox
            )

            # Jersey number
            jersey_num, jersey_conf = self.jersey_recognizer.recognize(part_features.jersey)

            detection = PlayerDetection(
                bbox=bbox,
                embedding=part_features.global_feature,
                role=role,
                role_confidence=role_conf,
                team_id=team_id,
                jersey_number=jersey_num,
                jersey_confidence=jersey_conf
            )
            results.append(detection)

        # Update tracks
        self._update_tracks(frame_id, results)

        return results

    def _update_tracks(
        self,
        frame_id: int,
        detections: List[PlayerDetection]
    ):
        """Update tracks with new detections."""
        if not detections:
            # Mark all tracks as lost
            for track in self.tracks.values():
                if track.state == TrackingState.ACTIVE:
                    track.state = TrackingState.LOST
            return

        # Compute similarity matrix
        active_tracks = [t for t in self.tracks.values() if t.state == TrackingState.ACTIVE]

        if not active_tracks:
            # All new tracks
            for det in detections:
                self._create_track(frame_id, det)
            return

        similarity_matrix = np.zeros((len(active_tracks), len(detections)))

        for i, track in enumerate(active_tracks):
            last_det = track.last_detection[1] if track.last_detection else None
            if last_det is None:
                continue

            for j, det in enumerate(detections):
                # Embedding similarity
                emb_sim = np.dot(last_det.embedding, det.embedding)

                # IoU
                iou = last_det.bbox.iou(det.bbox)

                similarity_matrix[i, j] = 0.7 * emb_sim + 0.3 * iou

        # Hungarian matching (simplified greedy here)
        matched_tracks = set()
        matched_dets = set()

        while True:
            # Find best unmatched pair
            best_i, best_j = -1, -1
            best_sim = self.matching_threshold

            for i in range(len(active_tracks)):
                if i in matched_tracks:
                    continue
                for j in range(len(detections)):
                    if j in matched_dets:
                        continue
                    if similarity_matrix[i, j] > best_sim:
                        best_sim = similarity_matrix[i, j]
                        best_i, best_j = i, j

            if best_i < 0:
                break

            # Match
            track = active_tracks[best_i]
            track.detections.append((frame_id, detections[best_j]))
            matched_tracks.add(best_i)
            matched_dets.add(best_j)

        # Unmatched tracks → lost
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track.state = TrackingState.LOST

        # Unmatched detections → new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(frame_id, det)

    def _create_track(
        self,
        frame_id: int,
        detection: PlayerDetection
    ):
        """Create new track."""
        track = Track(
            track_id=self.next_track_id,
            detections=[(frame_id, detection)],
            state=TrackingState.ACTIVE
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def get_active_tracks(self) -> List[Track]:
        """Get all active tracks."""
        return [t for t in self.tracks.values() if t.state == TrackingState.ACTIVE]

    def match_to_roster(
        self,
        home_roster: Dict[int, str],  # jersey → name
        away_roster: Dict[int, str]
    ) -> Dict[int, str]:
        """Match tracks to known players."""
        matches = {}

        for track in self.tracks.values():
            if not track.detections:
                continue

            # Aggregate jersey predictions
            jersey_votes = {}
            team_votes = {0: 0, 1: 0}

            for _, det in track.detections:
                if det.jersey_number is not None:
                    jersey_votes[det.jersey_number] = jersey_votes.get(det.jersey_number, 0) + 1
                if det.team_id is not None:
                    team_votes[det.team_id] += 1

            # Most common team
            team_id = 0 if team_votes[0] >= team_votes[1] else 1
            roster = home_roster if team_id == 0 else away_roster

            # Most common jersey
            if jersey_votes:
                jersey = max(jersey_votes, key=jersey_votes.get)
                if jersey in roster:
                    matches[track.track_id] = roster[jersey]
                    track.player_id = roster[jersey]

        return matches


class OcclusionHandler:
    """
    Handle tracking through occlusions.

    Uses re-ID to recover tracks after temporary loss.
    """

    def __init__(
        self,
        tracker: MultiTaskTracker,
        max_frames: int = 60
    ):
        self.tracker = tracker
        self.max_frames = max_frames

        # Store features of lost tracks
        self.lost_track_features: Dict[int, np.ndarray] = {}

    def on_track_lost(self, track: Track):
        """Store features when track is lost."""
        if track.detections:
            # Average recent embeddings
            recent = track.detections[-10:]
            avg_embedding = np.mean([det.embedding for _, det in recent], axis=0)
            self.lost_track_features[track.track_id] = avg_embedding

    def attempt_recovery(
        self,
        detection: PlayerDetection
    ) -> Optional[int]:
        """Try to match detection to lost track."""
        best_track_id = None
        best_similarity = 0.6  # Recovery threshold

        for track_id, features in self.lost_track_features.items():
            similarity = np.dot(features, detection.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_track_id = track_id

        if best_track_id is not None:
            del self.lost_track_features[best_track_id]

        return best_track_id

    def cleanup_old_tracks(self, current_frame: int):
        """Remove very old lost tracks."""
        to_remove = []

        for track_id in self.lost_track_features:
            track = self.tracker.tracks.get(track_id)
            if track:
                last_frame = track.last_detection[0] if track.last_detection else 0
                if current_frame - last_frame > self.max_frames:
                    to_remove.append(track_id)
                    track.state = TrackingState.TERMINATED

        for track_id in to_remove:
            del self.lost_track_features[track_id]
