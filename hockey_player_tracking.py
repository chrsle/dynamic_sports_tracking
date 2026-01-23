import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import supervision as sv
import torch

@dataclass
class HockeyTrackingConfig:
    """Configuration for hockey tracking."""
    
    # Player tracking (ByteTrack)
    player_track_buffer: int = 45  # Frames to keep lost tracks
    player_match_thresh: float = 0.75  # Matching threshold
    
    # Puck tracking
    puck_track_buffer: int = 60  # Longer buffer for puck
    puck_match_thresh: float = 0.5  # More lenient for small object
    puck_velocity_smoothing: int = 5  # Frames for velocity estimation
    
    # Trail visualization
    trail_length: int = 30  # Frames of trail to display
    
    # SAM2 settings (optional)
    use_sam2: bool = False
    sam2_checkpoint: str = "facebook/sam2-hiera-large"
    
    # Device
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

class HockeyPlayerTracker:
    """
    Multi-object tracker optimized for hockey players.
    
    Uses ByteTrack which excels at:
    - Handling high-speed movement
    - Re-identifying players after occlusions
    - Maintaining consistent IDs through scrums
    """
    
    def __init__(self, config: Optional[HockeyTrackingConfig] = None):
        """
        Initialize the tracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config or HockeyTrackingConfig()
        
        # ByteTrack for players
        self.player_tracker = sv.ByteTrack(
            lost_track_buffer=self.config.player_track_buffer,
            minimum_matching_threshold=self.config.player_match_thresh
        )
        
        # Position history for trails
        self.position_history: Dict[int, deque] = {}
        
        # Frame counter
        self.frame_count = 0
    
    def track(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracking with new detections.
        
        Args:
            detections: Player detections for current frame
            
        Returns:
            Tracked detections with consistent tracker_ids
        """
        self.frame_count += 1
        
        # Update ByteTrack
        tracked = self.player_tracker.update_with_detections(detections)
        
        # Update position history
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                if track_id not in self.position_history:
                    self.position_history[track_id] = deque(
                        maxlen=self.config.trail_length
                    )
                
                # Store bottom center position
                bbox = tracked.xyxy[i]
                center_x = (bbox[0] + bbox[2]) / 2
                bottom_y = bbox[3]
                
                self.position_history[track_id].append(
                    (center_x, bottom_y, self.frame_count)
                )
        
        return tracked
    
    def get_trail(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Get position trail for a tracked player.
        
        Args:
            track_id: Tracker ID
            
        Returns:
            List of (x, y) positions
        """
        if track_id not in self.position_history:
            return []
        
        return [(p[0], p[1]) for p in self.position_history[track_id]]
    
    def get_velocity(self, track_id: int, window: int = 5) -> Optional[Tuple[float, float]]:
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
        
        # Use recent positions
        recent = history[-min(window, len(history)):]
        
        # Average velocity
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = recent[-1][2] - recent[0][2]
        
        if dt == 0:
            return (0.0, 0.0)
        
        return (dx / dt, dy / dt)
    
    def reset(self):
        """Reset tracker state."""
        self.player_tracker = sv.ByteTrack(
            lost_track_buffer=self.config.player_track_buffer,
            minimum_matching_threshold=self.config.player_match_thresh
        )
        self.position_history.clear()
        self.frame_count = 0

class PuckTracker:
    """
    Specialized tracker for hockey puck.
    
    Handles challenges specific to puck tracking:
    - Very small object
    - Extremely high speed (100+ mph)
    - Frequent occlusions by players/boards
    - Motion blur
    """
    
    def __init__(self, config: Optional[HockeyTrackingConfig] = None):
        """
        Initialize puck tracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config or HockeyTrackingConfig()
        
        # Position history for interpolation
        self.history: deque = deque(maxlen=60)
        
        # Velocity estimation
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        
        # Frame counter
        self.frame_count = 0
        self.last_detection_frame = 0
    
    def update(
        self, 
        puck_detection: Optional[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Update puck tracking with new detection.
        
        Args:
            puck_detection: (x, y, confidence) or None if not detected
            
        Returns:
            (x, y) position (detected or interpolated) or None
        """
        self.frame_count += 1
        
        if puck_detection is not None:
            x, y, conf = puck_detection
            
            # Store in history
            self.history.append((x, y, self.frame_count, conf))
            self.last_detection_frame = self.frame_count
            
            # Update velocity estimate
            self._update_velocity()
            
            return (x, y)
        
        # Try to interpolate if no detection
        return self._interpolate()
    
    def _update_velocity(self):
        """Update velocity estimate from recent detections."""
        if len(self.history) < 2:
            return
        
        # Use smoothing window
        window = min(self.config.puck_velocity_smoothing, len(self.history))
        recent = list(self.history)[-window:]
        
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = recent[-1][2] - recent[0][2]
        
        if dt > 0:
            self.velocity = (dx / dt, dy / dt)
    
    def _interpolate(self) -> Optional[Tuple[float, float]]:
        """
        Interpolate puck position when not detected.
        
        Returns:
            Interpolated (x, y) or None if too many frames missed
        """
        if len(self.history) == 0:
            return None
        
        # Only interpolate for a few frames
        frames_missed = self.frame_count - self.last_detection_frame
        max_interpolate = 10  # Don't interpolate beyond 10 frames
        
        if frames_missed > max_interpolate:
            return None
        
        # Linear extrapolation from last position and velocity
        last = self.history[-1]
        pred_x = last[0] + self.velocity[0] * frames_missed
        pred_y = last[1] + self.velocity[1] * frames_missed
        
        return (pred_x, pred_y)
    
    def get_trail(self, length: int = 15) -> List[Tuple[float, float]]:
        """
        Get puck trail for visualization.
        
        Args:
            length: Number of positions to return
            
        Returns:
            List of (x, y) positions
        """
        recent = list(self.history)[-length:]
        return [(p[0], p[1]) for p in recent]
    
    def get_speed_estimate(self, fps: float = 30.0) -> float:
        """
        Estimate puck speed in pixels per second.
        
        Args:
            fps: Video frame rate
            
        Returns:
            Speed in pixels/second
        """
        vx, vy = self.velocity
        return np.sqrt(vx**2 + vy**2) * fps
    
    def reset(self):
        """Reset tracker state."""
        self.history.clear()
        self.velocity = (0.0, 0.0)
        self.frame_count = 0
        self.last_detection_frame = 0

class SAM2HockeyTracker:
    """
    Optional SAM2-based tracker for hockey.
    
    SAM2 provides:
    - Pixel-perfect segmentation masks
    - Temporal memory for consistent tracking
    - Better handling of partial occlusions
    """
    
    def __init__(self, config: Optional[HockeyTrackingConfig] = None):
        """
        Initialize SAM2 tracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config or HockeyTrackingConfig()
        self._model = None
        self._predictor = None
    
    def _load_model(self):
        """Load SAM2 model."""
        if self._model is not None:
            return
        
        try:
            from sam2.build_sam import build_sam2_video_predictor
            
            self._predictor = build_sam2_video_predictor(
                self.config.sam2_checkpoint,
                device=self.config.device
            )
            print(f"✓ Loaded SAM2: {self.config.sam2_checkpoint}")
            
        except ImportError:
            print("⚠ SAM2 not available. Install with: pip install segment-anything-2")
    
    def initialize_tracks(
        self, 
        video_path: str, 
        initial_detections: sv.Detections
    ):
        """
        Initialize SAM2 tracking for a video.
        
        Args:
            video_path: Path to video file
            initial_detections: Detections from first frame
        """
        self._load_model()
        
        if self._predictor is None:
            return
        
        # Initialize video predictor
        self._predictor.init_state(video_path)
        
        # Add initial points from detections
        for i, bbox in enumerate(initial_detections.xyxy):
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            self._predictor.add_new_points_or_box(
                frame_idx=0,
                obj_id=i,
                points=[[center_x, center_y]],
                labels=[1]
            )
    
    def track_frame(self, frame_idx: int) -> Dict[int, np.ndarray]:
        """
        Get segmentation masks for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Dictionary mapping object IDs to masks
        """
        if self._predictor is None:
            return {}
        
        out_frame_idx, obj_ids, masks = self._predictor.propagate_in_video()
        
        result = {}
        for obj_id, mask in zip(obj_ids, masks):
            result[obj_id] = mask.cpu().numpy()
        
        return result

def draw_trails(
    frame: np.ndarray,
    tracker: HockeyPlayerTracker,
    tracked_detections: sv.Detections,
    team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw player trails on frame.
    
    Args:
        frame: Input frame
        tracker: Player tracker
        tracked_detections: Current detections with tracker_ids
        team_colors: Optional team color mapping
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    default_colors = {
        0: (0, 0, 200),    # Home team (red)
        1: (200, 200, 200)  # Away team (white)
    }
    colors = team_colors or default_colors
    
    if tracked_detections.tracker_id is None:
        return annotated
    
    for i, track_id in enumerate(tracked_detections.tracker_id):
        trail = tracker.get_trail(track_id)
        
        if len(trail) < 2:
            continue
        
        # Get team color (default to home)
        team_id = 0  # Would come from team classification
        color = colors.get(team_id, (128, 128, 128))
        
        # Draw trail with fading effect
        for j in range(1, len(trail)):
            pt1 = (int(trail[j-1][0]), int(trail[j-1][1]))
            pt2 = (int(trail[j][0]), int(trail[j][1]))
            
            alpha = j / len(trail)
            thickness = max(1, int(3 * alpha))
            
            cv2.line(annotated, pt1, pt2, color, thickness)
    
    return annotated


def draw_puck_trail(
    frame: np.ndarray,
    puck_tracker: PuckTracker,
    current_position: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Draw puck trail on frame.
    
    Args:
        frame: Input frame
        puck_tracker: Puck tracker
        current_position: Current puck position
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    trail = puck_tracker.get_trail()
    
    # Draw trail
    if len(trail) >= 2:
        for i in range(1, len(trail)):
            pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            
            alpha = i / len(trail)
            color = (0, int(255 * alpha), 0)
            
            cv2.line(annotated, pt1, pt2, color, 2)
    
    # Draw current position
    if current_position is not None:
        px, py = int(current_position[0]), int(current_position[1])
        cv2.circle(annotated, (px, py), 8, (0, 255, 0), 2)
        cv2.circle(annotated, (px, py), 3, (0, 255, 0), -1)
    
    return annotated

# Example usage
if __name__ == "__main__":
    # Initialize trackers
    config = HockeyTrackingConfig(
        player_track_buffer=45,
        player_match_thresh=0.75,
        use_sam2=False
    )
    
    player_tracker = HockeyPlayerTracker(config)
    puck_tracker = PuckTracker(config)
    
    print("Hockey Tracking Module Ready!")
    print(f"\nConfiguration:")
    print(f"  - Player track buffer: {config.player_track_buffer} frames")
    print(f"  - Player match threshold: {config.player_match_thresh}")
    print(f"  - Puck track buffer: {config.puck_track_buffer} frames")
    print(f"  - SAM2 enabled: {config.use_sam2}")
    print(f"  - Trail length: {config.trail_length} frames")
