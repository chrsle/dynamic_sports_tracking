import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import supervision as sv
from ultralytics import YOLO
from inference import get_model
import torch

# Class IDs for basketball detection
BALL_CLASS_ID = 0
PLAYER_CLASS_ID = 1
REFEREE_CLASS_ID = 2
RIM_CLASS_ID = 3

# Model paths - can be Roboflow model IDs or local paths
PLAYER_DETECTION_MODEL = "basketball-players-fy4c2/1"  # Roboflow Universe model
JERSEY_NUMBER_MODEL = "basketball-player-detection-3-ycjdo/6"  # For jersey number detection

# Detection confidence thresholds
PLAYER_CONFIDENCE_THRESHOLD = 0.3
BALL_CONFIDENCE_THRESHOLD = 0.5
JERSEY_CONFIDENCE_THRESHOLD = 0.4

@dataclass
class DetectionResult:
    """Container for detection results."""
    detections: sv.Detections
    frame: np.ndarray
    frame_number: int
    
    @property
    def player_detections(self) -> sv.Detections:
        """Filter detections to only include players."""
        mask = self.detections.class_id == PLAYER_CLASS_ID
        return self.detections[mask]
    
    @property
    def ball_detections(self) -> sv.Detections:
        """Filter detections to only include the ball."""
        mask = self.detections.class_id == BALL_CLASS_ID
        return self.detections[mask]
    
    @property
    def referee_detections(self) -> sv.Detections:
        """Filter detections to only include referees."""
        mask = self.detections.class_id == REFEREE_CLASS_ID
        return self.detections[mask]

class BasketballPlayerDetector:
    """
    Basketball player detection using YOLO or RF-DETR models.
    
    Based on Roboflow's approach for detecting players, referees,
    and basketball objects in game footage.
    """
    
    def __init__(
        self,
        model_id: str = PLAYER_DETECTION_MODEL,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = PLAYER_CONFIDENCE_THRESHOLD,
        use_roboflow: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize the basketball player detector.
        
        Args:
            model_id: Model identifier (Roboflow model ID or local path)
            device: Device to run inference on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
            use_roboflow: Whether to use Roboflow inference API
            api_key: Roboflow API key (required if use_roboflow=True)
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_roboflow = use_roboflow
        
        if use_roboflow:
            self.model = get_model(model_id=model_id, api_key=api_key)
        else:
            # Load local YOLO model
            self.model = YOLO(model_id)
            self.model.to(device)
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run detection on a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            Detections object containing bounding boxes, confidence scores, and class IDs
        """
        if self.use_roboflow:
            result = self.model.infer(frame)[0]
            detections = sv.Detections.from_inference(result)
        else:
            result = self.model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
        
        # Filter by confidence threshold
        detections = detections[detections.confidence >= self.confidence_threshold]
        
        return detections
    
    def detect_video(
        self,
        video_path: str,
        stride: int = 1
    ) -> List[DetectionResult]:
        """
        Run detection on a video file.
        
        Args:
            video_path: Path to the video file
            stride: Process every nth frame (default: 1 = all frames)
            
        Returns:
            List of DetectionResult objects for each processed frame
        """
        results = []
        frame_generator = sv.get_video_frames_generator(video_path, stride=stride)
        
        for frame_number, frame in enumerate(frame_generator):
            detections = self.detect(frame)
            results.append(DetectionResult(
                detections=detections,
                frame=frame,
                frame_number=frame_number * stride
            ))
        
        return results

class JerseyNumberDetector:
    """
    Detect jersey number regions on players.
    
    This detector finds bounding boxes around jersey numbers,
    which are then passed to the OCR module for recognition.
    """
    
    def __init__(
        self,
        model_id: str = JERSEY_NUMBER_MODEL,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = JERSEY_CONFIDENCE_THRESHOLD,
        api_key: Optional[str] = None
    ):
        """
        Initialize the jersey number detector.
        
        Args:
            model_id: Roboflow model ID for jersey number detection
            device: Device to run inference on
            confidence_threshold: Minimum confidence for detections
            api_key: Roboflow API key
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = get_model(model_id=model_id, api_key=api_key)
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect jersey number regions in a frame.
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            Detections containing jersey number bounding boxes
        """
        result = self.model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections[detections.confidence >= self.confidence_threshold]
        return detections
    
    def get_jersey_crops(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> List[np.ndarray]:
        """
        Extract cropped images of jersey number regions.
        
        Args:
            frame: Full frame image
            detections: Jersey number detections
            
        Returns:
            List of cropped jersey number images
        """
        crops = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        return crops

def get_player_crops(
    frame: np.ndarray,
    detections: sv.Detections,
    margin: int = 0
) -> List[np.ndarray]:
    """
    Extract cropped images of detected players.
    
    Args:
        frame: Full frame image
        detections: Player detections
        margin: Additional margin around bounding boxes
        
    Returns:
        List of cropped player images
    """
    h, w = frame.shape[:2]
    crops = []
    
    for bbox in detections.xyxy:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Apply margin with boundary checks
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
    
    return crops

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_ios(box_small: np.ndarray, box_large: np.ndarray) -> float:
    """
    Calculate Intersection over Smaller Area.
    
    Used to associate jersey numbers with players - IoS measures
    how much of the smaller box (jersey number) is contained within
    the larger box (player).
    
    Args:
        box_small: Smaller bounding box (jersey number) [x1, y1, x2, y2]
        box_large: Larger bounding box (player) [x1, y1, x2, y2]
        
    Returns:
        IoS score between 0 and 1 (1.0 = fully contained)
    """
    x1 = max(box_small[0], box_large[0])
    y1 = max(box_small[1], box_large[1])
    x2 = min(box_small[2], box_large[2])
    y2 = min(box_small[3], box_large[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_small = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])
    
    return intersection / area_small if area_small > 0 else 0

def associate_jerseys_with_players(
    player_detections: sv.Detections,
    jersey_detections: sv.Detections,
    ios_threshold: float = 0.9
) -> Dict[int, int]:
    """
    Associate jersey number detections with player detections.
    
    Uses Intersection over Smaller Area (IoS) to match jersey numbers
    to players. A jersey is associated with a player if the IoS >= threshold.
    
    Args:
        player_detections: Detections of players
        jersey_detections: Detections of jersey numbers
        ios_threshold: Minimum IoS to consider a match (default: 0.9)
        
    Returns:
        Dictionary mapping player index to jersey detection index
    """
    associations = {}
    
    for jersey_idx, jersey_bbox in enumerate(jersey_detections.xyxy):
        best_ios = 0
        best_player_idx = None
        
        for player_idx, player_bbox in enumerate(player_detections.xyxy):
            ios = calculate_ios(jersey_bbox, player_bbox)
            
            if ios >= ios_threshold and ios > best_ios:
                best_ios = ios
                best_player_idx = player_idx
        
        if best_player_idx is not None:
            associations[best_player_idx] = jersey_idx
    
    return associations

# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize detector
    # Note: Set your Roboflow API key as an environment variable
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    
    detector = BasketballPlayerDetector(
        model_id=PLAYER_DETECTION_MODEL,
        api_key=api_key,
        confidence_threshold=0.3
    )
    
    # Example: Process a single frame
    # frame = cv2.imread("basketball_frame.jpg")
    # detections = detector.detect(frame)
    # print(f"Detected {len(detections)} objects")
    
    # Example: Process a video
    # video_path = "basketball_game.mp4"
    # results = detector.detect_video(video_path, stride=5)
    # print(f"Processed {len(results)} frames")
