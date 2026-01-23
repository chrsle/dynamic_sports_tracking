"""
Video Action Spotting and Analysis

Implements video-based action spotting translated from SoccerNet
research.

Key Papers:
- Deliège, A., et al. (2021). "SoccerNet-v2: A Dataset and Benchmarks
  for Holistic Understanding of Broadcast Soccer Videos." CVPR Workshop.

Key Concepts:
- 500+ games, 764 hours of video
- 17 action classes, 300,000+ annotations
- Camera calibration, player tracking
- Game state reconstruction

Hockey Translation:
- Foundation for hockey video understanding
- Action spotting: goals, saves, hits, fights
- Camera calibration for broadcast analytics
- Player tracking from video
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class HockeyAction(Enum):
    """Hockey actions for spotting."""
    GOAL = "goal"
    SHOT = "shot"
    SAVE = "save"
    HIT = "hit"
    BLOCK = "block"
    TAKEAWAY = "takeaway"
    GIVEAWAY = "giveaway"
    FACEOFF_WIN = "faceoff_win"
    FACEOFF_LOSS = "faceoff_loss"
    ICING = "icing"
    OFFSIDE = "offside"
    PENALTY = "penalty"
    FIGHT = "fight"
    LINE_CHANGE = "line_change"
    GOALIE_CHANGE = "goalie_change"
    POWER_PLAY_START = "pp_start"
    POWER_PLAY_END = "pp_end"


class CameraView(Enum):
    """Camera view types in hockey broadcast."""
    CENTER_ICE = "center_ice"        # Main broadcast view
    END_ZONE = "end_zone"            # Behind goal view
    CORNER = "corner"                # Corner camera
    OVERHEAD = "overhead"            # Top-down view
    HANDHELD = "handheld"            # Roaming camera
    REPLAY = "replay"                # Replay footage


@dataclass
class ActionSpot:
    """A spotted action in video."""
    action_type: HockeyAction
    timestamp: float                 # Video timestamp in seconds
    confidence: float                # 0-1
    game_time: Optional[str] = None  # Period + clock time
    player_id: Optional[str] = None
    team_id: Optional[str] = None
    location: Optional[Tuple[float, float]] = None  # x, y on ice
    camera_view: Optional[CameraView] = None


@dataclass
class VideoFrame:
    """Single video frame data."""
    frame_number: int
    timestamp: float                 # Seconds from video start
    image_features: np.ndarray       # Extracted CNN features
    detected_players: List[Dict[str, Any]]  # Bounding boxes
    detected_puck: Optional[Tuple[float, float, float, float]] = None  # bbox
    camera_view: CameraView = CameraView.CENTER_ICE


@dataclass
class GameSegment:
    """Segment of game video."""
    start_time: float
    end_time: float
    actions: List[ActionSpot]
    period: int
    game_clock_start: str
    game_clock_end: str


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result."""
    video_id: str
    duration: float                  # Total video length
    segments: List[GameSegment]
    all_actions: List[ActionSpot]
    action_counts: Dict[HockeyAction, int]
    camera_view_distribution: Dict[CameraView, float]


class FeatureExtractor:
    """
    Extracts visual features from video frames.

    Uses CNN-based feature extraction (simulated here).
    """

    def __init__(self, feature_dim: int = 512):
        """Initialize extractor."""
        self.feature_dim = feature_dim

        # Simulated CNN weights
        np.random.seed(42)
        self.conv_weights = np.random.randn(feature_dim, 3, 3, 3) * 0.1

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from image frame.

        In practice, would use pre-trained CNN like ResNet.
        """
        # Simulated feature extraction
        if frame is None:
            return np.zeros(self.feature_dim)

        # Global average pool + projection
        if len(frame.shape) == 3:
            pooled = frame.mean(axis=(0, 1))
        else:
            pooled = frame.flatten()

        # Project to feature dim
        if len(pooled) > self.feature_dim:
            features = pooled[:self.feature_dim]
        else:
            features = np.pad(pooled, (0, self.feature_dim - len(pooled)))

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features


class ActionClassifier:
    """
    Classifies hockey actions from video features.
    """

    def __init__(self, feature_dim: int = 512):
        """Initialize classifier."""
        self.feature_dim = feature_dim
        self.n_classes = len(HockeyAction)

        # Initialize weights
        np.random.seed(43)
        self.W1 = np.random.randn(feature_dim, 128) * 0.1
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, self.n_classes) * 0.1
        self.b2 = np.zeros(self.n_classes)

        # Class to action mapping
        self.idx_to_action = {i: action for i, action in enumerate(HockeyAction)}

    def classify(
        self,
        features: np.ndarray,
        context_features: Optional[np.ndarray] = None,
    ) -> List[Tuple[HockeyAction, float]]:
        """
        Classify action from features.

        Returns list of (action, probability) sorted by probability.
        """
        # Forward pass
        h1 = np.maximum(0, features @ self.W1 + self.b1)

        if context_features is not None:
            # Add context (previous frames)
            h1 = h1 + 0.1 * context_features[:128] if len(context_features) >= 128 else h1

        logits = h1 @ self.W2 + self.b2

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Sort by probability
        results = [
            (self.idx_to_action[i], float(p))
            for i, p in enumerate(probs)
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results


class TemporalActionDetector:
    """
    Detects actions across temporal sequence of frames.

    Uses sliding window approach with NMS.
    """

    def __init__(
        self,
        classifier: ActionClassifier,
        window_size: int = 30,      # frames
        stride: int = 5,
        confidence_threshold: float = 0.5,
    ):
        """Initialize detector."""
        self.classifier = classifier
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold

    def detect_actions(
        self,
        frames: List[VideoFrame],
    ) -> List[ActionSpot]:
        """
        Detect actions in sequence of frames.

        Uses temporal pooling over windows.
        """
        if not frames:
            return []

        detections = []

        # Slide window over frames
        for start in range(0, len(frames) - self.window_size + 1, self.stride):
            end = start + self.window_size
            window_frames = frames[start:end]

            # Pool features over window
            features = np.stack([f.image_features for f in window_frames])
            pooled = np.mean(features, axis=0)

            # Also create context from temporal sequence
            context = features.flatten()[:512]

            # Classify
            results = self.classifier.classify(pooled, context)

            # Get top prediction
            top_action, confidence = results[0]

            if confidence > self.confidence_threshold:
                # Get center timestamp
                center_frame = window_frames[len(window_frames) // 2]

                detections.append(ActionSpot(
                    action_type=top_action,
                    timestamp=center_frame.timestamp,
                    confidence=confidence,
                    camera_view=center_frame.camera_view,
                ))

        # Non-max suppression
        detections = self._nms(detections)

        return detections

    def _nms(
        self,
        detections: List[ActionSpot],
        time_threshold: float = 2.0,  # seconds
    ) -> List[ActionSpot]:
        """
        Non-maximum suppression to remove duplicate detections.
        """
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        kept = []
        for det in detections:
            # Check overlap with kept detections
            overlap = False
            for kept_det in kept:
                if det.action_type == kept_det.action_type:
                    time_diff = abs(det.timestamp - kept_det.timestamp)
                    if time_diff < time_threshold:
                        overlap = True
                        break

            if not overlap:
                kept.append(det)

        return kept


class CameraCalibration:
    """
    Camera calibration for broadcast video.

    Maps image coordinates to ice coordinates.
    """

    def __init__(self):
        """Initialize calibration."""
        # Rink dimensions (NHL)
        self.rink_length = 200.0  # feet
        self.rink_width = 85.0

        # Default homography (identity-like)
        self.homography = np.eye(3)

    def set_homography(self, H: np.ndarray):
        """Set homography matrix."""
        self.homography = H

    def estimate_from_keypoints(
        self,
        image_points: List[Tuple[float, float]],
        ice_points: List[Tuple[float, float]],
    ):
        """
        Estimate homography from corresponding points.

        Keypoints could be: goal lines, blue lines, circles, etc.
        """
        if len(image_points) < 4 or len(ice_points) < 4:
            return

        # Build system of equations
        A = []
        for (ix, iy), (rx, ry) in zip(image_points, ice_points):
            A.append([-ix, -iy, -1, 0, 0, 0, ix*rx, iy*rx, rx])
            A.append([0, 0, 0, -ix, -iy, -1, ix*ry, iy*ry, ry])

        A = np.array(A)

        # SVD solution
        _, _, Vh = np.linalg.svd(A)
        H = Vh[-1].reshape(3, 3)

        # Normalize
        H = H / H[2, 2]

        self.homography = H

    def image_to_ice(
        self,
        image_x: float,
        image_y: float,
    ) -> Tuple[float, float]:
        """
        Convert image coordinates to ice coordinates.
        """
        point = np.array([image_x, image_y, 1.0])
        transformed = self.homography @ point
        transformed = transformed / transformed[2]

        return float(transformed[0]), float(transformed[1])

    def ice_to_image(
        self,
        ice_x: float,
        ice_y: float,
    ) -> Tuple[float, float]:
        """
        Convert ice coordinates to image coordinates.
        """
        H_inv = np.linalg.inv(self.homography)
        point = np.array([ice_x, ice_y, 1.0])
        transformed = H_inv @ point
        transformed = transformed / transformed[2]

        return float(transformed[0]), float(transformed[1])


class PlayerTracker:
    """
    Tracks players across video frames.

    Uses simple IoU-based tracking (in practice, would use
    more sophisticated methods like DeepSORT).
    """

    def __init__(self, iou_threshold: float = 0.3):
        """Initialize tracker."""
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, List[Dict]] = {}
        self.next_track_id = 0

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_number: int,
    ) -> Dict[int, Dict]:
        """
        Update tracks with new detections.

        Returns current track assignments.
        """
        if not detections:
            return {}

        # Get active tracks
        active_tracks = {
            tid: track[-1]
            for tid, track in self.tracks.items()
            if track and frame_number - track[-1]['frame'] < 30  # Active within 30 frames
        }

        # Match detections to tracks
        assignments = {}
        used_detections = set()

        for tid, last_det in active_tracks.items():
            best_iou = 0
            best_det_idx = -1

            for i, det in enumerate(detections):
                if i in used_detections:
                    continue

                iou = self._compute_iou(last_det['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            if best_iou > self.iou_threshold:
                det = detections[best_det_idx]
                det['frame'] = frame_number
                self.tracks[tid].append(det)
                assignments[tid] = det
                used_detections.add(best_det_idx)

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                det['frame'] = frame_number
                self.tracks[self.next_track_id] = [det]
                assignments[self.next_track_id] = det
                self.next_track_id += 1

        return assignments

    def _compute_iou(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_track(self, track_id: int) -> List[Dict]:
        """Get full track history."""
        return self.tracks.get(track_id, [])


class VideoAnalyzer:
    """
    Main video analysis pipeline.
    """

    def __init__(self):
        """Initialize analyzer."""
        self.feature_extractor = FeatureExtractor()
        self.classifier = ActionClassifier()
        self.detector = TemporalActionDetector(self.classifier)
        self.calibration = CameraCalibration()
        self.tracker = PlayerTracker()

    def analyze_video(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
    ) -> VideoAnalysisResult:
        """
        Analyze complete video.

        Args:
            frames: List of image frames (numpy arrays)
            fps: Frames per second

        Returns:
            Complete analysis result
        """
        video_frames = []

        # Process each frame
        for i, frame in enumerate(frames):
            timestamp = i / fps
            features = self.feature_extractor.extract_features(frame)

            video_frame = VideoFrame(
                frame_number=i,
                timestamp=timestamp,
                image_features=features,
                detected_players=[],  # Would come from object detector
                camera_view=self._detect_camera_view(frame),
            )
            video_frames.append(video_frame)

        # Detect actions
        actions = self.detector.detect_actions(video_frames)

        # Count actions
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action.action_type] += 1

        # Camera view distribution
        view_counts = defaultdict(int)
        for frame in video_frames:
            view_counts[frame.camera_view] += 1

        total_frames = len(video_frames)
        view_dist = {
            view: count / total_frames
            for view, count in view_counts.items()
        }

        return VideoAnalysisResult(
            video_id="analysis_" + str(hash(tuple(f.timestamp for f in video_frames[:10]))),
            duration=len(frames) / fps,
            segments=[],  # Would segment by period
            all_actions=actions,
            action_counts=dict(action_counts),
            camera_view_distribution=view_dist,
        )

    def _detect_camera_view(self, frame: np.ndarray) -> CameraView:
        """
        Detect camera view from frame.

        Simplified - would use CNN classifier in practice.
        """
        # Random for simulation
        return CameraView.CENTER_ICE

    def generate_highlight_timestamps(
        self,
        result: VideoAnalysisResult,
        highlight_actions: List[HockeyAction] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate timestamps for highlight reel.
        """
        if highlight_actions is None:
            highlight_actions = [
                HockeyAction.GOAL,
                HockeyAction.SAVE,
                HockeyAction.HIT,
                HockeyAction.FIGHT,
            ]

        highlights = []
        for action in result.all_actions:
            if action.action_type in highlight_actions:
                highlights.append({
                    'timestamp': action.timestamp,
                    'action': action.action_type.value,
                    'confidence': action.confidence,
                    'start_time': max(0, action.timestamp - 5),  # 5 sec before
                    'end_time': action.timestamp + 5,  # 5 sec after
                })

        # Sort by timestamp
        highlights.sort(key=lambda x: x['timestamp'])

        return highlights
