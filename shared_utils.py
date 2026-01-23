"""
Shared utilities for sports tracking modules.

This module consolidates common functions and base classes following
the DRY (Don't Repeat Yourself) principle from Clean Code and
The Pragmatic Programmer.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, TypeVar, Iterable, Generator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from PIL import Image
import supervision as sv

# Type variable for generic batching
T = TypeVar('T')


# =============================================================================
# Device Selection
# =============================================================================

def get_device() -> str:
    """Get the best available device for PyTorch inference."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Image Conversion Utilities
# =============================================================================

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB format."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB Image to OpenCV BGR format."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# =============================================================================
# Bounding Box Utilities
# =============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

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

    return intersection / union if union > 0 else 0.0


def compute_ios(box_small: np.ndarray, box_large: np.ndarray) -> float:
    """
    Compute Intersection over Smaller Area (IoS).

    Used to associate smaller objects (e.g., jersey numbers) with
    larger objects (e.g., players).

    Args:
        box_small: Smaller bounding box [x1, y1, x2, y2]
        box_large: Larger bounding box [x1, y1, x2, y2]

    Returns:
        IoS score between 0 and 1 (1.0 = fully contained)
    """
    x1 = max(box_small[0], box_large[0])
    y1 = max(box_small[1], box_large[1])
    x2 = min(box_small[2], box_large[2])
    y2 = min(box_small[3], box_large[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_small = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])

    return intersection / area_small if area_small > 0 else 0.0


def get_box_center(bbox: np.ndarray) -> Tuple[float, float]:
    """Get center point of a bounding box."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def get_box_bottom_center(bbox: np.ndarray) -> Tuple[float, float]:
    """Get bottom center point of a bounding box (useful for ground position)."""
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


# =============================================================================
# Crop Extraction
# =============================================================================

def extract_crops(
    frame: np.ndarray,
    detections: sv.Detections,
    margin: int = 0
) -> List[np.ndarray]:
    """
    Extract image crops from detections.

    Args:
        frame: Full frame image
        detections: Detection results with bounding boxes
        margin: Additional margin around bounding boxes (default: 0)

    Returns:
        List of cropped images
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


def extract_jersey_region(
    crop: np.ndarray,
    top_ratio: float = 0.15,
    bottom_ratio: float = 0.65
) -> np.ndarray:
    """
    Extract jersey region from a player crop.

    Args:
        crop: Player crop image
        top_ratio: Start of jersey region (ratio from top)
        bottom_ratio: End of jersey region (ratio from top)

    Returns:
        Cropped jersey region
    """
    h = crop.shape[0]
    top = int(h * top_ratio)
    bottom = int(h * bottom_ratio)
    return crop[top:bottom, :]


# =============================================================================
# Batching Utilities
# =============================================================================

def create_batches(
    sequence: Iterable[T],
    batch_size: int
) -> Generator[List[T], None, None]:
    """
    Generate batches from a sequence.

    Args:
        sequence: Input sequence to batch
        batch_size: Size of each batch

    Yields:
        Lists of elements with size up to batch_size
    """
    batch_size = max(batch_size, 1)
    current_batch = []

    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)

    if current_batch:
        yield current_batch


# =============================================================================
# NMS and Filtering
# =============================================================================

def filter_overlapping_detections(
    detections: sv.Detections,
    iou_threshold: float = 0.5
) -> sv.Detections:
    """
    Filter overlapping detections, keeping highest confidence.

    Args:
        detections: Input detections
        iou_threshold: IoU threshold for overlap

    Returns:
        Filtered detections
    """
    if len(detections) == 0:
        return detections

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


# =============================================================================
# Base Configuration
# =============================================================================

@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    device: str = field(default_factory=get_device)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# Base Classes for ML Components
# =============================================================================

class BaseDetector(ABC):
    """Abstract base class for object detectors."""

    def __init__(self):
        self._model = None

    @abstractmethod
    def _load_model(self):
        """Load the detection model (lazy loading)."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run detection on a frame.

        Args:
            frame: Input image (BGR format)

        Returns:
            Detection results
        """
        pass


class BaseTracker(ABC):
    """Abstract base class for object trackers."""

    @abstractmethod
    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracking with new detections.

        Args:
            detections: New frame detections

        Returns:
            Tracked detections with IDs
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset tracker state."""
        pass


class BaseClassifier(ABC):
    """Abstract base class for classifiers with fit/predict interface."""

    def __init__(self):
        self._fitted = False

    @abstractmethod
    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract features from image crops."""
        pass

    @abstractmethod
    def fit(self, crops: List[np.ndarray]):
        """Fit the classifier on training data."""
        pass

    @abstractmethod
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Predict labels for new data."""
        pass

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class BaseOCR(ABC):
    """Abstract base class for OCR/text recognition."""

    def __init__(self):
        self._model = None

    @abstractmethod
    def _load_model(self):
        """Load the OCR model."""
        pass

    @abstractmethod
    def read(self, image: np.ndarray) -> Optional[str]:
        """
        Read text from an image.

        Args:
            image: Input image (BGR format)

        Returns:
            Recognized text or None
        """
        pass

    def read_batch(self, images: List[np.ndarray]) -> List[Optional[str]]:
        """Read text from multiple images."""
        return [self.read(img) for img in images]


# =============================================================================
# Color Utilities
# =============================================================================

def extract_dominant_colors(
    image: np.ndarray,
    n_colors: int = 3,
    region: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """
    Extract dominant colors from an image using KMeans.

    Args:
        image: Input image (BGR format)
        n_colors: Number of dominant colors to extract
        region: Optional (top, bottom, left, right) ratios to crop

    Returns:
        Array of dominant colors in BGR format
    """
    from sklearn.cluster import KMeans

    if region:
        h, w = image.shape[:2]
        top, bottom, left, right = region
        image = image[int(h*top):int(h*bottom), int(w*left):int(w*right)]

    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    sorted_indices = np.argsort(-counts)

    return kmeans.cluster_centers_[sorted_indices]


def compute_color_histogram(
    image: np.ndarray,
    bins_h: int = 30,
    bins_s: int = 32,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute HSV color histogram for an image.

    Args:
        image: Input image (BGR format)
        bins_h: Number of hue bins
        bins_s: Number of saturation bins
        normalize: Whether to normalize the histogram

    Returns:
        Flattened histogram array
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [bins_h, bins_s],
        [0, 180, 0, 256]
    )

    if normalize:
        hist = cv2.normalize(hist, hist)

    return hist.flatten()
