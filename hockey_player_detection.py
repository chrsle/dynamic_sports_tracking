import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import supervision as sv
import torch
import os

# Hockey detection class definitions
HOCKEY_CLASSES = {
    0: 'player',
    1: 'goalie',
    2: 'referee',
    3: 'puck'
}

HOCKEY_CLASS_IDS = {v: k for k, v in HOCKEY_CLASSES.items()}

@dataclass
class HockeyDetectionConfig:
    """Configuration for hockey detection."""
    
    # Model settings
    use_rf_detr: bool = True
    rf_detr_model: str = "rf-detr-medium"  # nano, small, medium, large
    roboflow_model: str = "hockey-players-puck/1"  # Fallback
    
    # Confidence thresholds (per class)
    player_confidence: float = 0.35
    goalie_confidence: float = 0.40
    referee_confidence: float = 0.35
    puck_confidence: float = 0.25  # Lower for small object
    
    # NMS settings (for fallback models)
    nms_threshold: float = 0.5
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_confidence(self, class_name: str) -> float:
        """Get confidence threshold for a class."""
        thresholds = {
            'player': self.player_confidence,
            'goalie': self.goalie_confidence,
            'referee': self.referee_confidence,
            'puck': self.puck_confidence
        }
        return thresholds.get(class_name, 0.3)

class RFDETRHockeyDetector:
    """
    Hockey object detector using RF-DETR with Meta's DINOv2 backbone.
    
    RF-DETR is particularly well-suited for hockey because:
    1. Better handling of overlapping players (no NMS artifacts)
    2. Superior small object detection (pucks)
    3. Fast fine-tuning on custom hockey datasets
    4. Real-time inference for live game analysis
    """
    
    def __init__(self, config: Optional[HockeyDetectionConfig] = None):
        """
        Initialize the detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config or HockeyDetectionConfig()
        self._model = None
        self._is_rf_detr = False
        
    def _load_model(self):
        """Load the detection model (lazy loading)."""
        if self._model is not None:
            return self._model
        
        if self.config.use_rf_detr:
            try:
                # Try to load RF-DETR
                from rfdetr import RFDETRBase, RFDETRSmall, RFDETRMedium, RFDETRLarge
                
                model_classes = {
                    'rf-detr-nano': RFDETRBase,
                    'rf-detr-small': RFDETRSmall,
                    'rf-detr-medium': RFDETRMedium,
                    'rf-detr-large': RFDETRLarge
                }
                
                model_class = model_classes.get(
                    self.config.rf_detr_model, 
                    RFDETRMedium
                )
                
                self._model = model_class()
                self._is_rf_detr = True
                
                print(f"✓ Loaded RF-DETR ({self.config.rf_detr_model})")
                print(f"  Backbone: Meta DINOv2 (self-supervised)")
                print(f"  Architecture: Transformer-based, no anchors/NMS")
                
            except ImportError as e:
                print(f"⚠ RF-DETR not available: {e}")
                print("  Falling back to Roboflow inference...")
                self._load_fallback()
        else:
            self._load_fallback()
        
        return self._model
    
    def _load_fallback(self):
        """Load fallback Roboflow model."""
        from inference import get_model
        
        api_key = os.environ.get('ROBOFLOW_API_KEY')
        self._model = get_model(
            model_id=self.config.roboflow_model,
            api_key=api_key
        )
        self._is_rf_detr = False
        print(f"✓ Loaded Roboflow model: {self.config.roboflow_model}")
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run detection on a frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            Detections with all hockey objects
        """
        model = self._load_model()
        
        if self._is_rf_detr:
            # RF-DETR expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb_frame, threshold=0.2)
        else:
            # Roboflow inference
            result = model.infer(frame)[0]
            detections = sv.Detections.from_inference(result)
        
        return detections
    
    def detect_and_filter(
        self, 
        frame: np.ndarray
    ) -> Dict[str, sv.Detections]:
        """
        Detect and filter by class with per-class confidence thresholds.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            Dictionary mapping class names to filtered detections
        """
        all_detections = self.detect(frame)
        
        results = {}
        for class_id, class_name in HOCKEY_CLASSES.items():
            # Filter by class
            class_mask = all_detections.class_id == class_id
            
            # Filter by confidence
            threshold = self.config.get_confidence(class_name)
            conf_mask = all_detections.confidence >= threshold
            
            results[class_name] = all_detections[class_mask & conf_mask]
        
        return results
    
    def detect_players_only(
        self, 
        frame: np.ndarray,
        include_goalies: bool = True
    ) -> sv.Detections:
        """
        Detect only players (and optionally goalies).
        
        Args:
            frame: Input image
            include_goalies: Whether to include goalies
            
        Returns:
            Player detections
        """
        detections = self.detect_and_filter(frame)
        
        players = detections['player']
        
        if include_goalies and len(detections['goalie']) > 0:
            # Merge player and goalie detections
            goalies = detections['goalie']
            players = sv.Detections(
                xyxy=np.vstack([players.xyxy, goalies.xyxy]) if len(players) > 0 else goalies.xyxy,
                confidence=np.concatenate([players.confidence, goalies.confidence]) if len(players) > 0 else goalies.confidence,
                class_id=np.concatenate([players.class_id, goalies.class_id]) if len(players) > 0 else goalies.class_id
            )
        
        return players
    
    def detect_puck(
        self, 
        frame: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """
        Detect the puck and return its position and confidence.
        
        Args:
            frame: Input image
            
        Returns:
            Tuple of (center_x, center_y, confidence) or None
        """
        detections = self.detect_and_filter(frame)
        pucks = detections['puck']
        
        if len(pucks) == 0:
            return None
        
        # Use highest confidence detection
        best_idx = np.argmax(pucks.confidence)
        bbox = pucks.xyxy[best_idx]
        confidence = pucks.confidence[best_idx]
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        return (center_x, center_y, confidence)

def fine_tune_rf_detr_hockey(
    dataset_path: str,
    model_variant: str = "rf-detr-medium",
    epochs: int = 50,
    batch_size: int = 8,
    output_dir: str = "./hockey_rf_detr",
    device: str = "cuda"
):
    """
    Fine-tune RF-DETR on a hockey dataset.
    
    RF-DETR converges faster than YOLO models due to the pre-trained
    DINOv2 backbone, typically requiring only 30-50 epochs.
    
    Dataset should be in COCO format with classes:
    - player (id: 0)
    - goalie (id: 1)
    - referee (id: 2)
    - puck (id: 3)
    
    Args:
        dataset_path: Path to COCO-format dataset
        model_variant: RF-DETR model size
        epochs: Number of training epochs
        batch_size: Training batch size
        output_dir: Directory to save model
        device: Training device
    """
    try:
        from rfdetr import RFDETRBase, RFDETRSmall, RFDETRMedium, RFDETRLarge
    except ImportError:
        print("RF-DETR not installed. Install with: pip install rf-detr")
        return
    
    model_classes = {
        'rf-detr-nano': RFDETRBase,
        'rf-detr-small': RFDETRSmall,
        'rf-detr-medium': RFDETRMedium,
        'rf-detr-large': RFDETRLarge
    }
    
    model_class = model_classes.get(model_variant, RFDETRMedium)
    model = model_class()
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning RF-DETR for Hockey Detection")
    print(f"{'='*60}")
    print(f"Model: {model_variant}")
    print(f"Backbone: Meta DINOv2 (self-supervised)")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Fine-tune
    model.train(
        dataset_dir=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        device=device
    )
    
    print(f"\n✓ Model saved to: {output_dir}")

# Utility functions for detection analysis

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in [x1, y1, x2, y2] format
        
    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def filter_overlapping_detections(
    detections: sv.Detections,
    iou_threshold: float = 0.5
) -> sv.Detections:
    """
    Filter overlapping detections keeping highest confidence.
    
    Note: RF-DETR doesn't need this as it has no NMS, but useful for
    fallback models.
    
    Args:
        detections: Input detections
        iou_threshold: IoU threshold for considering overlap
        
    Returns:
        Filtered detections
    """
    if len(detections) == 0:
        return detections
    
    # Sort by confidence
    indices = np.argsort(-detections.confidence)
    keep = []
    
    for i in indices:
        should_keep = True
        for j in keep:
            if compute_iou(detections.xyxy[i], detections.xyxy[j]) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(i)
    
    return detections[keep]

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    config = HockeyDetectionConfig(
        use_rf_detr=True,
        rf_detr_model="rf-detr-medium",
        player_confidence=0.35,
        puck_confidence=0.25
    )
    
    detector = RFDETRHockeyDetector(config)
    
    print("Hockey Detection Module Ready!")
    print(f"\nConfiguration:")
    print(f"  - Model: {'RF-DETR' if config.use_rf_detr else 'Roboflow YOLO'}")
    print(f"  - Player confidence: {config.player_confidence}")
    print(f"  - Goalie confidence: {config.goalie_confidence}")
    print(f"  - Puck confidence: {config.puck_confidence}")
    print(f"  - Device: {config.device}")
    
    # Example detection (uncomment with actual image)
    # frame = cv2.imread("hockey_frame.jpg")
    # detections = detector.detect_and_filter(frame)
    # print(f"\nDetected:")
    # for class_name, dets in detections.items():
    #     print(f"  - {class_name}: {len(dets)}")
