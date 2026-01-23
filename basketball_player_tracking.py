import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import supervision as sv
import torch

# Tracking configuration
TRACK_THRESH = 0.25
TRACK_BUFFER = 30
MATCH_THRESH = 0.8
FRAME_RATE = 30

# SAM2 configuration
SAM2_CHECKPOINT = "facebook/sam2-hiera-large"
SAM2_POINTS_PER_SIDE = 32

@dataclass
class TrackedPlayer:
    """Container for tracked player data."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    team_id: int = -1
    jersey_number: Optional[str] = None
    player_name: Optional[str] = None
    mask: Optional[np.ndarray] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x = int((self.bbox[0] + self.bbox[2]) / 2)
        y = int((self.bbox[1] + self.bbox[3]) / 2)
        return (x, y)
    
    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Get bottom center point (for court position mapping)."""
        x = int((self.bbox[0] + self.bbox[2]) / 2)
        y = int(self.bbox[3])
        return (x, y)

class ByteTrackTracker:
    """
    Multi-object tracker using ByteTrack algorithm via supervision.
    
    ByteTrack is fast and accurate, making it suitable for real-time
    basketball tracking where players move quickly and frequently occlude.
    """
    
    def __init__(
        self,
        track_thresh: float = TRACK_THRESH,
        track_buffer: int = TRACK_BUFFER,
        match_thresh: float = MATCH_THRESH,
        frame_rate: int = FRAME_RATE
    ):
        """
        Initialize ByteTrack tracker.
        
        Args:
            track_thresh: Detection confidence threshold
            track_buffer: Frames to keep lost tracks
            match_thresh: IoU threshold for matching
            frame_rate: Video frame rate
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        self.track_history: Dict[int, List[np.ndarray]] = {}
    
    def update(
        self,
        detections: sv.Detections
    ) -> sv.Detections:
        """
        Update tracker with new detections.
        
        Args:
            detections: New frame detections
            
        Returns:
            Detections with assigned tracker IDs
        """
        tracked = self.tracker.update_with_detections(detections)
        
        # Update track history
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                center = tracked.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER
                )[i]
                self.track_history[track_id].append(center)
                
                # Keep only recent history
                if len(self.track_history[track_id]) > 60:
                    self.track_history[track_id] = self.track_history[track_id][-60:]
        
        return tracked
    
    def get_track_history(self, track_id: int) -> List[np.ndarray]:
        """
        Get position history for a track.
        
        Args:
            track_id: Track ID to query
            
        Returns:
            List of position coordinates
        """
        return self.track_history.get(track_id, [])
    
    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()
        self.track_history.clear()

class SAM2Tracker:
    """
    Tracking using SAM2 (Segment Anything Model 2).
    
    SAM2 provides segmentation and tracking in videos.
    It uses a temporal memory bank for re-identification
    after occlusions - crucial for basketball.
    """
    
    def __init__(
        self,
        model_id: str = SAM2_CHECKPOINT,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize SAM2 tracker.
        
        Args:
            model_id: SAM2 model checkpoint
            device: Device to run inference on
        """
        self.model_id = model_id
        self.device = device
        self._predictor = None
        self._video_predictor = None
        self._inference_state = None
        self.active_tracks: Dict[int, Any] = {}
    
    def _load_model(self):
        """Lazy load SAM2 model."""
        if self._predictor is None:
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                from sam2.sam2_video_predictor import SAM2VideoPredictor
                from sam2.build_sam import build_sam2, build_sam2_video_predictor
            except ImportError:
                raise ImportError(
                    "Please install SAM2: pip install segment-anything-2"
                )
            
            # Load image predictor for initial prompting
            self._predictor = SAM2ImagePredictor.from_pretrained(self.model_id)
            
            # Video predictor will be initialized per video
            self._video_predictor_class = SAM2VideoPredictor
    
    def initialize_video(self, video_path: str):
        """
        Initialize tracking for a new video.
        
        Args:
            video_path: Path to video file
        """
        self._load_model()
        
        # Create video predictor
        self._video_predictor = self._video_predictor_class.from_pretrained(
            self.model_id
        )
        
        # Initialize inference state
        self._inference_state = self._video_predictor.init_state(
            video_path=video_path
        )
        
        self.active_tracks.clear()
    
    def add_track(
        self,
        frame_idx: int,
        track_id: int,
        bbox: np.ndarray
    ):
        """
        Add a new track to follow using bounding box prompt.
        
        Args:
            frame_idx: Frame index where object appears
            track_id: Unique ID for this track
            bbox: Bounding box [x1, y1, x2, y2]
        """
        if self._video_predictor is None:
            raise RuntimeError("Call initialize_video first")
        
        # Convert bbox to points (center)
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        point = np.array([[center_x, center_y]])
        label = np.array([1])  # Foreground
        
        # Add to tracker
        _, out_obj_ids, out_mask_logits = self._video_predictor.add_new_points_or_box(
            inference_state=self._inference_state,
            frame_idx=frame_idx,
            obj_id=track_id,
            points=point,
            labels=label,
            box=bbox
        )
        
        self.active_tracks[track_id] = {
            'start_frame': frame_idx,
            'bbox': bbox
        }
    
    def propagate(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Propagate tracking through all frames.
        
        Returns:
            Dictionary mapping frame_idx -> {track_id: mask}
        """
        if self._video_predictor is None:
            raise RuntimeError("Call initialize_video first")
        
        results = {}
        
        # Propagate through video
        for frame_idx, obj_ids, mask_logits in self._video_predictor.propagate_in_video(
            self._inference_state
        ):
            results[frame_idx] = {}
            
            for obj_id, mask_logit in zip(obj_ids, mask_logits):
                mask = (mask_logit[0] > 0).cpu().numpy().astype(np.uint8)
                results[frame_idx][obj_id] = mask
        
        return results
    
    def segment_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> List[np.ndarray]:
        """
        Segment detected objects in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            detections: Object detections with bounding boxes
            
        Returns:
            List of segmentation masks
        """
        self._load_model()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set image
        self._predictor.set_image(rgb_frame)
        
        masks = []
        
        for bbox in detections.xyxy:
            # Predict mask using bounding box prompt
            mask_output, scores, _ = self._predictor.predict(
                box=bbox,
                multimask_output=False
            )
            
            masks.append(mask_output[0])
        
        return masks

class MultiObjectTracker:
    """
    Combined tracker using ByteTrack for speed and optional
    SAM2 for segmentation masks.
    
    This is the recommended tracker for basketball analysis.
    """
    
    def __init__(
        self,
        use_sam2: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize combined tracker.
        
        Args:
            use_sam2: Whether to use SAM2 for segmentation
            device: Device for SAM2 inference
        """
        self.byte_tracker = ByteTrackTracker()
        self.use_sam2 = use_sam2
        
        if use_sam2:
            self.sam2_tracker = SAM2Tracker(device=device)
        else:
            self.sam2_tracker = None
    
    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> sv.Detections:
        """
        Update tracking with new detections.
        
        Args:
            frame: Current frame
            detections: New detections
            
        Returns:
            Tracked detections with IDs and optionally masks
        """
        # ByteTrack for ID assignment
        tracked = self.byte_tracker.update(detections)
        
        # Optional SAM2 segmentation
        if self.use_sam2 and self.sam2_tracker and len(tracked) > 0:
            masks = self.sam2_tracker.segment_frame(frame, tracked)
            tracked.mask = np.array(masks)
        
        return tracked
    
    def get_tracked_players(
        self,
        tracked_detections: sv.Detections,
        team_ids: Optional[np.ndarray] = None,
        jersey_numbers: Optional[Dict[int, str]] = None,
        player_names: Optional[Dict[int, str]] = None
    ) -> List[TrackedPlayer]:
        """
        Convert detections to TrackedPlayer objects.
        
        Args:
            tracked_detections: Tracked detections with IDs
            team_ids: Team assignments per detection
            jersey_numbers: Map of track_id to jersey number
            player_names: Map of track_id to player name
            
        Returns:
            List of TrackedPlayer objects
        """
        players = []
        
        for i in range(len(tracked_detections)):
            track_id = tracked_detections.tracker_id[i] if tracked_detections.tracker_id is not None else i
            
            player = TrackedPlayer(
                track_id=int(track_id),
                bbox=tracked_detections.xyxy[i],
                confidence=tracked_detections.confidence[i] if tracked_detections.confidence is not None else 1.0,
                class_id=tracked_detections.class_id[i] if tracked_detections.class_id is not None else 0,
                team_id=int(team_ids[i]) if team_ids is not None else -1,
                jersey_number=jersey_numbers.get(track_id) if jersey_numbers else None,
                player_name=player_names.get(track_id) if player_names else None,
                mask=tracked_detections.mask[i] if tracked_detections.mask is not None else None
            )
            
            players.append(player)
        
        return players
    
    def reset(self):
        """Reset tracker state."""
        self.byte_tracker.reset()

def draw_tracks(
    frame: np.ndarray,
    players: List[TrackedPlayer],
    track_history: Dict[int, List[np.ndarray]],
    team_colors: Dict[int, Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw tracking visualization on frame.
    
    Args:
        frame: Input frame
        players: List of tracked players
        track_history: Position history per track
        team_colors: Colors per team (BGR format)
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    # Default team colors
    if team_colors is None:
        team_colors = {
            0: (255, 0, 0),    # Blue
            1: (0, 0, 255),    # Red
            -1: (128, 128, 128)  # Gray for unknown
        }
    
    for player in players:
        color = team_colors.get(player.team_id, (128, 128, 128))
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, player.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if player.player_name:
            label = f"{player.player_name} #{player.jersey_number}"
        elif player.jersey_number:
            label = f"#{player.jersey_number}"
        else:
            label = f"ID: {player.track_id}"
        
        cv2.putText(
            annotated, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2
        )
        
        # Draw track trail
        history = track_history.get(player.track_id, [])
        if len(history) > 1:
            points = np.array(history, dtype=np.int32)
            cv2.polylines(
                annotated, [points],
                False, color, 2
            )
        
        # Draw mask if available
        if player.mask is not None:
            mask_overlay = np.zeros_like(annotated)
            mask_overlay[player.mask > 0] = color
            annotated = cv2.addWeighted(annotated, 1.0, mask_overlay, 0.3, 0)
    
    return annotated

# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = MultiObjectTracker(use_sam2=False)  # Set True for segmentation masks
    
    # For each frame:
    # 1. Run detection
    # detections = detector.detect(frame)
    
    # 2. Update tracker
    # tracked = tracker.update(frame, detections)
    
    # 3. Get player objects
    # players = tracker.get_tracked_players(tracked, team_ids, jersey_numbers)
    
    # 4. Draw visualization
    # annotated = draw_tracks(frame, players, tracker.byte_tracker.track_history)
    
    print("Tracking module initialized!")
    print("ByteTrack ready for fast multi-object tracking")
