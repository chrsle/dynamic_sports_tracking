"""
Unified player tracking module for sports analytics.

Consolidates ByteTrack and SAM2 tracking approaches from
hockey and basketball modules.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import supervision as sv
import torch

from shared_utils import BaseTracker, BaseConfig, get_device, bgr_to_rgb


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrackingConfig(BaseConfig):
    """Configuration for player tracking."""

    # ByteTrack settings
    track_buffer: int = 30
    match_thresh: float = 0.8
    track_thresh: float = 0.25
    frame_rate: int = 30

    # Trail settings
    trail_length: int = 30

    # Ball tracking
    ball_track_buffer: int = 60
    ball_velocity_smoothing: int = 5

    # SAM2 settings
    use_sam2: bool = False
    sam2_checkpoint: str = 'facebook/sam2-hiera-large'


# =============================================================================
# Tracked Object Container
# =============================================================================

@dataclass
class TrackedObject:
    """Container for tracked object data."""
    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int = 0
    team_id: int = -1
    jersey_number: Optional[str] = None
    player_name: Optional[str] = None
    mask: Optional[np.ndarray] = None

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return (
            int((self.bbox[0] + self.bbox[2]) / 2),
            int((self.bbox[1] + self.bbox[3]) / 2)
        )

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Get bottom center point (ground position)."""
        return (
            int((self.bbox[0] + self.bbox[2]) / 2),
            int(self.bbox[3])
        )


# =============================================================================
# ByteTrack-based Player Tracker
# =============================================================================

class PlayerTracker(BaseTracker):
    """
    Multi-object tracker using ByteTrack algorithm.

    ByteTrack excels at:
    - High-speed movement tracking
    - Re-identification after occlusions
    - Maintaining consistent IDs
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()

        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.config.track_thresh,
            lost_track_buffer=self.config.track_buffer,
            minimum_matching_threshold=self.config.match_thresh,
            frame_rate=self.config.frame_rate
        )

        self.position_history: Dict[int, deque] = {}
        self.frame_count = 0

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracking with new detections.

        Args:
            detections: New frame detections

        Returns:
            Tracked detections with consistent IDs
        """
        self.frame_count += 1

        tracked = self._tracker.update_with_detections(detections)

        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                if track_id not in self.position_history:
                    self.position_history[track_id] = deque(
                        maxlen=self.config.trail_length
                    )

                bbox = tracked.xyxy[i]
                center_x = (bbox[0] + bbox[2]) / 2
                bottom_y = bbox[3]

                self.position_history[track_id].append(
                    (center_x, bottom_y, self.frame_count)
                )

        return tracked

    def get_trail(self, track_id: int) -> List[Tuple[float, float]]:
        """Get position trail for a tracked player."""
        if track_id not in self.position_history:
            return []
        return [(p[0], p[1]) for p in self.position_history[track_id]]

    def get_velocity(
        self,
        track_id: int,
        window: int = 5
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate player velocity from recent positions.

        Args:
            track_id: Tracker ID
            window: Number of frames to use

        Returns:
            (vx, vy) velocity in pixels per frame, or None
        """
        if track_id not in self.position_history:
            return None

        history = list(self.position_history[track_id])
        if len(history) < 2:
            return None

        recent = history[-min(window, len(history)):]

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = recent[-1][2] - recent[0][2]

        if dt == 0:
            return (0.0, 0.0)

        return (dx / dt, dy / dt)

    def reset(self):
        """Reset tracker state."""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.config.track_thresh,
            lost_track_buffer=self.config.track_buffer,
            minimum_matching_threshold=self.config.match_thresh,
            frame_rate=self.config.frame_rate
        )
        self.position_history.clear()
        self.frame_count = 0


# =============================================================================
# Ball/Puck Tracker
# =============================================================================

class BallTracker:
    """
    Specialized tracker for small, fast-moving objects (ball/puck).

    Handles:
    - Very small objects
    - High speed (puck can exceed 100 mph)
    - Frequent occlusions
    - Motion blur
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()

        self.history: deque = deque(maxlen=self.config.ball_track_buffer)
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.frame_count = 0
        self.last_detection_frame = 0

    def update(
        self,
        detection: Optional[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Update tracking with new detection.

        Args:
            detection: (x, y, confidence) or None if not detected

        Returns:
            (x, y) position (detected or interpolated) or None
        """
        self.frame_count += 1

        if detection is not None:
            x, y, conf = detection
            self.history.append((x, y, self.frame_count, conf))
            self.last_detection_frame = self.frame_count
            self._update_velocity()
            return (x, y)

        return self._interpolate()

    def _update_velocity(self):
        """Update velocity estimate from recent detections."""
        if len(self.history) < 2:
            return

        window = min(self.config.ball_velocity_smoothing, len(self.history))
        recent = list(self.history)[-window:]

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = recent[-1][2] - recent[0][2]

        if dt > 0:
            self.velocity = (dx / dt, dy / dt)

    def _interpolate(self) -> Optional[Tuple[float, float]]:
        """Interpolate position when not detected."""
        if len(self.history) == 0:
            return None

        frames_missed = self.frame_count - self.last_detection_frame
        max_interpolate = 10

        if frames_missed > max_interpolate:
            return None

        last = self.history[-1]
        pred_x = last[0] + self.velocity[0] * frames_missed
        pred_y = last[1] + self.velocity[1] * frames_missed

        return (pred_x, pred_y)

    def get_trail(self, length: int = 15) -> List[Tuple[float, float]]:
        """Get ball trail for visualization."""
        recent = list(self.history)[-length:]
        return [(p[0], p[1]) for p in recent]

    def get_speed(self, fps: float = 30.0) -> float:
        """Get ball speed in pixels per second."""
        vx, vy = self.velocity
        return np.sqrt(vx**2 + vy**2) * fps

    def reset(self):
        """Reset tracker state."""
        self.history.clear()
        self.velocity = (0.0, 0.0)
        self.frame_count = 0
        self.last_detection_frame = 0


# =============================================================================
# SAM2 Tracker (Optional)
# =============================================================================

class SAM2Tracker:
    """
    Optional tracker using SAM2 for segmentation masks.

    Provides:
    - Pixel-perfect segmentation
    - Temporal memory for re-identification
    - Better handling of partial occlusions
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig(use_sam2=True)
        self._predictor = None
        self._video_predictor = None
        self._inference_state = None
        self.active_tracks: Dict[int, Any] = {}

    def _load_model(self):
        """Load SAM2 model."""
        if self._predictor is not None:
            return

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError:
            raise ImportError("Please install SAM2: pip install segment-anything-2")

        self._predictor = SAM2ImagePredictor.from_pretrained(self.config.sam2_checkpoint)
        self._video_predictor_class = SAM2VideoPredictor
        print(f"Loaded SAM2: {self.config.sam2_checkpoint}")

    def initialize_video(self, video_path: str):
        """Initialize tracking for a new video."""
        self._load_model()

        self._video_predictor = self._video_predictor_class.from_pretrained(
            self.config.sam2_checkpoint
        )

        self._inference_state = self._video_predictor.init_state(
            video_path=video_path
        )

        self.active_tracks.clear()

    def add_track(self, frame_idx: int, track_id: int, bbox: np.ndarray):
        """Add a new track using bounding box prompt."""
        if self._video_predictor is None:
            raise RuntimeError("Call initialize_video first")

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        point = np.array([[center_x, center_y]])
        label = np.array([1])

        self._video_predictor.add_new_points_or_box(
            inference_state=self._inference_state,
            frame_idx=frame_idx,
            obj_id=track_id,
            points=point,
            labels=label,
            box=bbox
        )

        self.active_tracks[track_id] = {'start_frame': frame_idx, 'bbox': bbox}

    def segment_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> List[np.ndarray]:
        """Segment detected objects in a single frame."""
        self._load_model()

        rgb_frame = bgr_to_rgb(frame)
        self._predictor.set_image(rgb_frame)

        masks = []
        for bbox in detections.xyxy:
            mask_output, scores, _ = self._predictor.predict(
                box=bbox,
                multimask_output=False
            )
            masks.append(mask_output[0])

        return masks


# =============================================================================
# Combined Tracker
# =============================================================================

class MultiObjectTracker:
    """
    Combined tracker using ByteTrack for speed and optional
    SAM2 for segmentation masks.
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()

        self.player_tracker = PlayerTracker(self.config)
        self.ball_tracker = BallTracker(self.config)

        if self.config.use_sam2:
            self.sam2_tracker = SAM2Tracker(self.config)
        else:
            self.sam2_tracker = None

    def update(
        self,
        frame: np.ndarray,
        player_detections: sv.Detections,
        ball_detection: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[sv.Detections, Optional[Tuple[float, float]]]:
        """
        Update tracking with new detections.

        Args:
            frame: Current frame
            player_detections: Player detections
            ball_detection: Ball/puck detection (x, y, confidence)

        Returns:
            Tuple of (tracked player detections, ball position)
        """
        tracked_players = self.player_tracker.update(player_detections)
        ball_position = self.ball_tracker.update(ball_detection)

        if self.sam2_tracker and len(tracked_players) > 0:
            masks = self.sam2_tracker.segment_frame(frame, tracked_players)
            tracked_players.mask = np.array(masks)

        return tracked_players, ball_position

    def get_tracked_objects(
        self,
        tracked_detections: sv.Detections,
        team_ids: Optional[np.ndarray] = None,
        jersey_numbers: Optional[Dict[int, str]] = None,
        player_names: Optional[Dict[int, str]] = None
    ) -> List[TrackedObject]:
        """Convert detections to TrackedObject instances."""
        objects = []

        for i in range(len(tracked_detections)):
            track_id = int(tracked_detections.tracker_id[i]) if tracked_detections.tracker_id is not None else i

            obj = TrackedObject(
                track_id=track_id,
                bbox=tracked_detections.xyxy[i],
                confidence=float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 1.0,
                class_id=int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0,
                team_id=int(team_ids[i]) if team_ids is not None else -1,
                jersey_number=jersey_numbers.get(track_id) if jersey_numbers else None,
                player_name=player_names.get(track_id) if player_names else None,
                mask=tracked_detections.mask[i] if tracked_detections.mask is not None else None
            )

            objects.append(obj)

        return objects

    def reset(self):
        """Reset all trackers."""
        self.player_tracker.reset()
        self.ball_tracker.reset()


# =============================================================================
# Visualization
# =============================================================================

def draw_tracks(
    frame: np.ndarray,
    tracked_objects: List[TrackedObject],
    track_history: Dict[int, List],
    team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw tracking visualization on frame.

    Args:
        frame: Input frame
        tracked_objects: List of tracked objects
        track_history: Position history per track
        team_colors: Colors per team (BGR)

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    if team_colors is None:
        team_colors = {
            0: (255, 0, 0),
            1: (0, 0, 255),
            -1: (128, 128, 128)
        }

    for obj in tracked_objects:
        color = team_colors.get(obj.team_id, (128, 128, 128))

        x1, y1, x2, y2 = map(int, obj.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if obj.player_name:
            label = f"{obj.player_name} #{obj.jersey_number}"
        elif obj.jersey_number:
            label = f"#{obj.jersey_number}"
        else:
            label = f"ID: {obj.track_id}"

        cv2.putText(
            annotated, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

        history = track_history.get(obj.track_id, [])
        if len(history) > 1:
            points = np.array(history, dtype=np.int32)
            cv2.polylines(annotated, [points], False, color, 2)

        if obj.mask is not None:
            mask_overlay = np.zeros_like(annotated)
            mask_overlay[obj.mask > 0] = color
            annotated = cv2.addWeighted(annotated, 1.0, mask_overlay, 0.3, 0)

    return annotated


def draw_ball_trail(
    frame: np.ndarray,
    ball_tracker: BallTracker,
    current_position: Optional[Tuple[float, float]] = None,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draw ball/puck trail on frame."""
    annotated = frame.copy()

    trail = ball_tracker.get_trail()

    if len(trail) >= 2:
        for i in range(1, len(trail)):
            pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))

            alpha = i / len(trail)
            trail_color = tuple(int(c * alpha) for c in color)

            cv2.line(annotated, pt1, pt2, trail_color, 2)

    if current_position is not None:
        px, py = int(current_position[0]), int(current_position[1])
        cv2.circle(annotated, (px, py), 8, color, 2)
        cv2.circle(annotated, (px, py), 3, color, -1)

    return annotated


if __name__ == "__main__":
    config = TrackingConfig(
        track_buffer=30,
        match_thresh=0.8,
        use_sam2=False
    )

    tracker = MultiObjectTracker(config)

    print("Player Tracking Module Ready!")
    print(f"Track buffer: {config.track_buffer} frames")
    print(f"SAM2 enabled: {config.use_sam2}")
