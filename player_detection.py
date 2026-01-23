"""
Unified player detection module for multiple sports.

Consolidates hockey and basketball detection using a sport-agnostic
approach with configurable class definitions.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import supervision as sv
import torch

from shared_utils import (
    BaseDetector, BaseConfig, get_device, bgr_to_rgb,
    compute_iou, filter_overlapping_detections, extract_crops
)


# =============================================================================
# Sport-Specific Class Definitions
# =============================================================================

HOCKEY_CLASSES = {
    0: 'player',
    1: 'goalie',
    2: 'referee',
    3: 'puck'
}

BASKETBALL_CLASSES = {
    0: 'ball',
    1: 'player',
    2: 'referee',
    3: 'rim'
}

SPORT_CLASSES = {
    'hockey': HOCKEY_CLASSES,
    'basketball': BASKETBALL_CLASSES
}


# =============================================================================
# Detection Configuration
# =============================================================================

@dataclass
class DetectionConfig(BaseConfig):
    """Configuration for player detection."""

    # Sport selection
    sport: str = 'hockey'

    # Model selection
    model_type: str = 'rf_detr'  # 'rf_detr', 'yolo', 'roboflow'
    model_path: str = 'rf-detr-medium'

    # Confidence thresholds (per class type)
    player_confidence: float = 0.35
    goalie_confidence: float = 0.40
    referee_confidence: float = 0.35
    ball_confidence: float = 0.25  # puck/basketball

    # NMS threshold
    nms_threshold: float = 0.5

    # Roboflow settings (if using roboflow)
    roboflow_api_key: Optional[str] = None

    @property
    def classes(self) -> Dict[int, str]:
        """Get class definitions for the selected sport."""
        return SPORT_CLASSES.get(self.sport, HOCKEY_CLASSES)

    def get_confidence(self, class_name: str) -> float:
        """Get confidence threshold for a class."""
        thresholds = {
            'player': self.player_confidence,
            'goalie': self.goalie_confidence,
            'referee': self.referee_confidence,
            'puck': self.ball_confidence,
            'ball': self.ball_confidence,
            'rim': 0.5
        }
        return thresholds.get(class_name, 0.3)


# =============================================================================
# Unified Player Detector
# =============================================================================

class PlayerDetector(BaseDetector):
    """
    Multi-sport player detector supporting RF-DETR, YOLO, and Roboflow models.

    Features:
    - Lazy model loading
    - Per-class confidence filtering
    - Sport-specific class handling
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        super().__init__()
        self.config = config or DetectionConfig()
        self._model_type = None

    def _load_model(self):
        """Load the detection model based on configuration."""
        if self._model is not None:
            return self._model

        if self.config.model_type == 'rf_detr':
            self._load_rf_detr()
        elif self.config.model_type == 'yolo':
            self._load_yolo()
        else:
            self._load_roboflow()

        return self._model

    def _load_rf_detr(self):
        """Load RF-DETR model."""
        try:
            from rfdetr import RFDETRBase, RFDETRSmall, RFDETRMedium, RFDETRLarge

            model_classes = {
                'rf-detr-nano': RFDETRBase,
                'rf-detr-small': RFDETRSmall,
                'rf-detr-medium': RFDETRMedium,
                'rf-detr-large': RFDETRLarge
            }

            model_class = model_classes.get(self.config.model_path, RFDETRMedium)
            self._model = model_class()
            self._model_type = 'rf_detr'
            print(f"Loaded RF-DETR ({self.config.model_path})")

        except ImportError:
            print("RF-DETR not available, falling back to Roboflow")
            self._load_roboflow()

    def _load_yolo(self):
        """Load YOLO model."""
        from ultralytics import YOLO

        self._model = YOLO(self.config.model_path)
        self._model.to(self.config.device)
        self._model_type = 'yolo'
        print(f"Loaded YOLO model: {self.config.model_path}")

    def _load_roboflow(self):
        """Load Roboflow model."""
        from inference import get_model

        api_key = self.config.roboflow_api_key or os.environ.get('ROBOFLOW_API_KEY')
        self._model = get_model(model_id=self.config.model_path, api_key=api_key)
        self._model_type = 'roboflow'
        print(f"Loaded Roboflow model: {self.config.model_path}")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run detection on a frame.

        Args:
            frame: Input image (BGR format)

        Returns:
            Detections with all detected objects
        """
        model = self._load_model()

        if self._model_type == 'rf_detr':
            rgb_frame = bgr_to_rgb(frame)
            detections = model.predict(rgb_frame, threshold=0.2)
        elif self._model_type == 'yolo':
            result = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
        else:
            result = model.infer(frame)[0]
            detections = sv.Detections.from_inference(result)

        return detections

    def detect_and_filter(self, frame: np.ndarray) -> Dict[str, sv.Detections]:
        """
        Detect and filter by class with per-class confidence thresholds.

        Args:
            frame: Input image (BGR format)

        Returns:
            Dictionary mapping class names to filtered detections
        """
        all_detections = self.detect(frame)

        results = {}
        for class_id, class_name in self.config.classes.items():
            class_mask = all_detections.class_id == class_id
            threshold = self.config.get_confidence(class_name)
            conf_mask = all_detections.confidence >= threshold

            results[class_name] = all_detections[class_mask & conf_mask]

        return results

    def detect_players(
        self,
        frame: np.ndarray,
        include_goalies: bool = True
    ) -> sv.Detections:
        """
        Detect only players (and optionally goalies).

        Args:
            frame: Input image
            include_goalies: Whether to include goalies (hockey)

        Returns:
            Player detections
        """
        detections = self.detect_and_filter(frame)

        players = detections.get('player', sv.Detections.empty())

        if include_goalies and 'goalie' in detections and len(detections['goalie']) > 0:
            goalies = detections['goalie']
            if len(players) > 0:
                players = sv.Detections(
                    xyxy=np.vstack([players.xyxy, goalies.xyxy]),
                    confidence=np.concatenate([players.confidence, goalies.confidence]),
                    class_id=np.concatenate([players.class_id, goalies.class_id])
                )
            else:
                players = goalies

        return players

    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Detect ball/puck and return position.

        Args:
            frame: Input image

        Returns:
            Tuple of (center_x, center_y, confidence) or None
        """
        detections = self.detect_and_filter(frame)

        ball_key = 'puck' if self.config.sport == 'hockey' else 'ball'
        balls = detections.get(ball_key, sv.Detections.empty())

        if len(balls) == 0:
            return None

        best_idx = np.argmax(balls.confidence)
        bbox = balls.xyxy[best_idx]
        confidence = balls.confidence[best_idx]

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        return (center_x, center_y, float(confidence))


# =============================================================================
# Detection Result Container
# =============================================================================

@dataclass
class DetectionResult:
    """Container for detection results."""
    detections: sv.Detections
    frame: np.ndarray
    frame_number: int

    def get_class_detections(self, class_id: int) -> sv.Detections:
        """Filter detections by class ID."""
        mask = self.detections.class_id == class_id
        return self.detections[mask]

    @property
    def player_detections(self) -> sv.Detections:
        """Get player detections (class_id=1 for most sports)."""
        return self.get_class_detections(1)


# =============================================================================
# Convenience Functions
# =============================================================================

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
    return extract_crops(frame, detections, margin)


def associate_objects(
    primary_detections: sv.Detections,
    secondary_detections: sv.Detections,
    ios_threshold: float = 0.9
) -> Dict[int, int]:
    """
    Associate secondary detections with primary detections using IoS.

    Useful for associating jersey numbers with players.

    Args:
        primary_detections: Primary objects (e.g., players)
        secondary_detections: Secondary objects (e.g., jersey numbers)
        ios_threshold: Minimum IoS for association

    Returns:
        Dictionary mapping primary index to secondary index
    """
    from shared_utils import compute_ios

    associations = {}

    for sec_idx, sec_bbox in enumerate(secondary_detections.xyxy):
        best_ios = 0
        best_primary_idx = None

        for pri_idx, pri_bbox in enumerate(primary_detections.xyxy):
            ios = compute_ios(sec_bbox, pri_bbox)

            if ios >= ios_threshold and ios > best_ios:
                best_ios = ios
                best_primary_idx = pri_idx

        if best_primary_idx is not None:
            associations[best_primary_idx] = sec_idx

    return associations


# =============================================================================
# Model Training Utilities
# =============================================================================

def fine_tune_rf_detr(
    dataset_path: str,
    sport: str = 'hockey',
    model_variant: str = 'rf-detr-medium',
    epochs: int = 50,
    batch_size: int = 8,
    output_dir: str = './fine_tuned_model',
    device: str = None
):
    """
    Fine-tune RF-DETR on a sports dataset.

    Args:
        dataset_path: Path to COCO-format dataset
        sport: Sport type for class configuration
        model_variant: RF-DETR model variant
        epochs: Training epochs
        batch_size: Training batch size
        output_dir: Output directory for model
        device: Training device
    """
    try:
        from rfdetr import RFDETRBase, RFDETRSmall, RFDETRMedium, RFDETRLarge
    except ImportError:
        print("RF-DETR not installed. Install with: pip install rf-detr")
        return

    device = device or get_device()

    model_classes = {
        'rf-detr-nano': RFDETRBase,
        'rf-detr-small': RFDETRSmall,
        'rf-detr-medium': RFDETRMedium,
        'rf-detr-large': RFDETRLarge
    }

    model = model_classes.get(model_variant, RFDETRMedium)()

    print(f"\nFine-tuning RF-DETR for {sport.title()} Detection")
    print(f"Model: {model_variant}")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")

    model.train(
        dataset_dir=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        device=device
    )

    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    # Example: Hockey detection
    hockey_config = DetectionConfig(
        sport='hockey',
        model_type='rf_detr',
        model_path='rf-detr-medium'
    )
    hockey_detector = PlayerDetector(hockey_config)

    # Example: Basketball detection
    basketball_config = DetectionConfig(
        sport='basketball',
        model_type='roboflow',
        model_path='basketball-players-fy4c2/1'
    )
    basketball_detector = PlayerDetector(basketball_config)

    print("Player Detection Module Ready!")
    print(f"Hockey config: {hockey_config.classes}")
    print(f"Basketball config: {basketball_config.classes}")
