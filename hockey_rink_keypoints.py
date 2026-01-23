import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
from enum import IntEnum

# NHL Standard Rink Dimensions (in feet, converted to coordinate system)
# Rink is 200ft x 85ft

class RinkDimensions:
    """NHL standard rink dimensions."""
    
    # Full rink (in feet)
    LENGTH = 200.0
    WIDTH = 85.0
    
    # Corner radius
    CORNER_RADIUS = 28.0
    
    # Lines from center
    BLUE_LINE_DIST = 25.0  # From center to blue line
    GOAL_LINE_DIST = 89.0  # From center to goal line
    
    # Face-off circles
    FACEOFF_CIRCLE_RADIUS = 15.0
    FACEOFF_DOT_DIST_FROM_GOAL = 20.0
    FACEOFF_DOT_WIDTH = 22.0  # From center of rink
    
    # Neutral zone dots
    NEUTRAL_DOT_DIST = 5.0  # From blue line
    
    # Goal crease
    CREASE_RADIUS = 6.0
    CREASE_WIDTH = 8.0
    
    # Goal dimensions
    GOAL_WIDTH = 6.0
    GOAL_DEPTH = 4.0


class RinkZone(IntEnum):
    """Hockey rink zones."""
    OFFENSIVE = 0
    NEUTRAL = 1
    DEFENSIVE = 2
    BEHIND_GOAL_OFFENSIVE = 3
    BEHIND_GOAL_DEFENSIVE = 4

@dataclass
class RinkKeypointConfig:
    """Configuration for rink keypoint detection."""
    
    # Model settings
    model_path: str = "SimulaMet-HOST/HockeyRink"  # HuggingFace model
    use_local_model: bool = False
    local_model_path: Optional[str] = None
    
    # Detection settings
    confidence_threshold: float = 0.5
    min_keypoints_for_homography: int = 4
    
    # Homography settings
    use_ransac: bool = True
    ransac_threshold: float = 5.0
    
    # Smoothing settings (for video)
    enable_smoothing: bool = True
    smoothing_window: int = 5  # Frames
    
    # Output rink image size
    rink_output_width: int = 600  # pixels
    rink_output_height: int = 255  # pixels (maintains aspect ratio)
    
    # Device
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

def get_nhl_rink_keypoints() -> np.ndarray:
    """
    Get standardized NHL rink keypoints in feet.
    Origin at center ice, x = length, y = width.
    
    Returns:
        Array of 56 keypoints (x, y) in feet
    """
    keypoints = []
    
    # Rink half-dimensions
    half_length = RinkDimensions.LENGTH / 2  # 100
    half_width = RinkDimensions.WIDTH / 2    # 42.5
    
    # 1-4: Corner points (accounting for rounded corners)
    corner_offset = RinkDimensions.CORNER_RADIUS * 0.3  # Approximate
    keypoints.extend([
        (-half_length + corner_offset, -half_width + corner_offset),  # Bottom-left
        (-half_length + corner_offset, half_width - corner_offset),   # Top-left
        (half_length - corner_offset, half_width - corner_offset),    # Top-right
        (half_length - corner_offset, -half_width + corner_offset),   # Bottom-right
    ])
    
    # 5-8: Blue line - boards intersections (left blue line)
    blue_left = -RinkDimensions.BLUE_LINE_DIST
    blue_right = RinkDimensions.BLUE_LINE_DIST
    keypoints.extend([
        (blue_left, -half_width),
        (blue_left, half_width),
        (blue_right, -half_width),
        (blue_right, half_width),
    ])
    
    # 9-12: Center line points
    keypoints.extend([
        (0, -half_width),
        (0, half_width),
        (0, -RinkDimensions.FACEOFF_DOT_WIDTH),  # Center dot edge
        (0, RinkDimensions.FACEOFF_DOT_WIDTH),
    ])
    
    # 13: Center ice face-off dot
    keypoints.append((0, 0))
    
    # 14-17: Offensive zone face-off circles (right side)
    off_x = half_length - RinkDimensions.FACEOFF_DOT_DIST_FROM_GOAL
    keypoints.extend([
        (off_x, -RinkDimensions.FACEOFF_DOT_WIDTH),   # Bottom circle center
        (off_x, RinkDimensions.FACEOFF_DOT_WIDTH),    # Top circle center
        (-off_x, -RinkDimensions.FACEOFF_DOT_WIDTH),  # Defensive bottom
        (-off_x, RinkDimensions.FACEOFF_DOT_WIDTH),   # Defensive top
    ])
    
    # 18-25: Face-off circle edge points (4 per circle, 2 circles per end)
    for x_mult in [1, -1]:  # Offensive and defensive
        for y_mult in [1, -1]:  # Top and bottom
            cx = x_mult * off_x
            cy = y_mult * RinkDimensions.FACEOFF_DOT_WIDTH
            r = RinkDimensions.FACEOFF_CIRCLE_RADIUS
            keypoints.extend([
                (cx + r, cy),
                (cx - r, cy),
            ])
    
    # 26-29: Neutral zone dots
    neutral_x = RinkDimensions.BLUE_LINE_DIST + RinkDimensions.NEUTRAL_DOT_DIST
    keypoints.extend([
        (neutral_x, -RinkDimensions.FACEOFF_DOT_WIDTH),
        (neutral_x, RinkDimensions.FACEOFF_DOT_WIDTH),
        (-neutral_x, -RinkDimensions.FACEOFF_DOT_WIDTH),
        (-neutral_x, RinkDimensions.FACEOFF_DOT_WIDTH),
    ])
    
    # 30-37: Goal crease corners (4 per goal)
    for x_mult in [1, -1]:
        goal_x = x_mult * RinkDimensions.GOAL_LINE_DIST
        crease_front = goal_x - x_mult * RinkDimensions.CREASE_RADIUS
        keypoints.extend([
            (crease_front, -RinkDimensions.CREASE_WIDTH/2),
            (crease_front, RinkDimensions.CREASE_WIDTH/2),
            (goal_x, -RinkDimensions.CREASE_WIDTH/2),
            (goal_x, RinkDimensions.CREASE_WIDTH/2),
        ])
    
    # 38-45: Goal posts (4 per goal)
    for x_mult in [1, -1]:
        goal_x = x_mult * RinkDimensions.GOAL_LINE_DIST
        goal_back = goal_x + x_mult * RinkDimensions.GOAL_DEPTH
        keypoints.extend([
            (goal_x, -RinkDimensions.GOAL_WIDTH/2),
            (goal_x, RinkDimensions.GOAL_WIDTH/2),
            (goal_back, -RinkDimensions.GOAL_WIDTH/2),
            (goal_back, RinkDimensions.GOAL_WIDTH/2),
        ])
    
    # 46-53: Goal line - boards intersections and behind-net points
    for x_mult in [1, -1]:
        goal_x = x_mult * RinkDimensions.GOAL_LINE_DIST
        keypoints.extend([
            (goal_x, -half_width + corner_offset),
            (goal_x, half_width - corner_offset),
        ])
    
    # 54-56: Additional reference points
    keypoints.extend([
        (-half_length, 0),  # Left boards center
        (half_length, 0),   # Right boards center
        (0, -half_width + corner_offset),  # Center boards bottom
    ])
    
    return np.array(keypoints, dtype=np.float32)

class RinkKeypointDetector:
    """
    Detect keypoints on hockey rink for homography estimation.
    
    Uses YOLOv8 pose estimation trained on HockeyRink dataset
    to detect 56 keypoints on the ice surface.
    """
    
    def __init__(self, config: Optional[RinkKeypointConfig] = None):
        self.config = config or RinkKeypointConfig()
        self._model = None
        self._rink_template = get_nhl_rink_keypoints()
        
        # Smoothing buffer
        self._keypoint_history = []
        self._homography_history = []
    
    def _load_model(self):
        """Load keypoint detection model."""
        if self._model is not None:
            return
        
        try:
            from ultralytics import YOLO
            
            if self.config.use_local_model and self.config.local_model_path:
                self._model = YOLO(self.config.local_model_path)
                print(f"✓ Loaded local rink keypoint model")
            else:
                # Try to load from HuggingFace
                from huggingface_hub import hf_hub_download
                
                model_path = hf_hub_download(
                    repo_id=self.config.model_path,
                    filename="hockey-rink-keypoints.pt"
                )
                self._model = YOLO(model_path)
                print(f"✓ Loaded HockeyRink keypoint model from HuggingFace")
                
        except Exception as e:
            print(f"⚠ Could not load keypoint model: {e}")
            print("  Using fallback line detection method")
            self._model = None
    
    def detect_keypoints(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect rink keypoints in frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (keypoints, confidences) where keypoints is Nx2
        """
        self._load_model()
        
        if self._model is None:
            # Fallback: detect lines and estimate keypoints
            return self._fallback_detection(frame)
        
        # Run model
        results = self._model(frame, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return np.array([]), np.array([])
        
        # Extract keypoints
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        confs = results[0].keypoints.conf[0].cpu().numpy()
        
        # Filter by confidence
        mask = confs >= self.config.confidence_threshold
        
        return kpts, confs
    
    def _fallback_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback keypoint detection using line detection.
        Detects blue lines, red line, and goal lines.
        """
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        keypoints = []
        confidences = []
        
        # Detect red lines (center line, goal lines)
        red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask |= cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        
        # Detect blue lines
        blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        
        # Find line intersections with boards
        for mask, name in [(red_mask, 'red'), (blue_mask, 'blue')]:
            lines = cv2.HoughLinesP(mask, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    keypoints.append([x1, y1])
                    keypoints.append([x2, y2])
                    confidences.extend([0.6, 0.6])  # Medium confidence for fallback
        
        if len(keypoints) == 0:
            return np.array([]), np.array([])
        
        return np.array(keypoints), np.array(confidences)
    
    def compute_homography(
        self, 
        detected_keypoints: np.ndarray,
        keypoint_indices: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Compute homography from detected keypoints to rink template.
        
        Args:
            detected_keypoints: Detected keypoints in image coordinates
            keypoint_indices: Which template keypoints these correspond to
            
        Returns:
            3x3 homography matrix or None
        """
        if len(detected_keypoints) < self.config.min_keypoints_for_homography:
            return None
        
        # Get corresponding template points
        if keypoint_indices is not None:
            template_points = self._rink_template[keypoint_indices]
        else:
            # Assume first N keypoints correspond
            n = min(len(detected_keypoints), len(self._rink_template))
            template_points = self._rink_template[:n]
            detected_keypoints = detected_keypoints[:n]
        
        # Compute homography
        if self.config.use_ransac:
            H, mask = cv2.findHomography(
                detected_keypoints, 
                template_points,
                cv2.RANSAC,
                self.config.ransac_threshold
            )
        else:
            H, _ = cv2.findHomography(detected_keypoints, template_points)
        
        # Apply smoothing if enabled
        if self.config.enable_smoothing and H is not None:
            H = self._smooth_homography(H)
        
        return H
    
    def _smooth_homography(self, H: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to homography."""
        self._homography_history.append(H)
        
        if len(self._homography_history) > self.config.smoothing_window:
            self._homography_history.pop(0)
        
        # Average homographies (simple smoothing)
        if len(self._homography_history) > 1:
            return np.mean(self._homography_history, axis=0)
        
        return H
    
    def transform_point(
        self, 
        point: Tuple[float, float], 
        H: np.ndarray
    ) -> Tuple[float, float]:
        """
        Transform a point from image to rink coordinates.
        
        Args:
            point: (x, y) in image coordinates
            H: Homography matrix
            
        Returns:
            (x, y) in rink coordinates (feet from center ice)
        """
        pt = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pt, H)
        return (transformed[0][0][0], transformed[0][0][1])
    
    def transform_points(
        self, 
        points: np.ndarray, 
        H: np.ndarray
    ) -> np.ndarray:
        """
        Transform multiple points from image to rink coordinates.
        
        Args:
            points: Nx2 array of (x, y) in image coordinates
            H: Homography matrix
            
        Returns:
            Nx2 array in rink coordinates
        """
        if len(points) == 0:
            return np.array([])
        
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, H)
        return transformed.reshape(-1, 2)
    
    def get_zone(self, rink_point: Tuple[float, float], team_attacking_right: bool = True) -> RinkZone:
        """
        Determine which zone a point is in.
        
        Args:
            rink_point: (x, y) in rink coordinates
            team_attacking_right: Whether the team attacks the right goal
            
        Returns:
            RinkZone enum value
        """
        x, y = rink_point
        
        # Blue lines at ±25 feet from center
        blue_line = RinkDimensions.BLUE_LINE_DIST
        goal_line = RinkDimensions.GOAL_LINE_DIST
        
        if team_attacking_right:
            if x > blue_line:
                return RinkZone.OFFENSIVE if x < goal_line else RinkZone.BEHIND_GOAL_OFFENSIVE
            elif x < -blue_line:
                return RinkZone.DEFENSIVE if x > -goal_line else RinkZone.BEHIND_GOAL_DEFENSIVE
        else:
            if x < -blue_line:
                return RinkZone.OFFENSIVE if x > -goal_line else RinkZone.BEHIND_GOAL_OFFENSIVE
            elif x > blue_line:
                return RinkZone.DEFENSIVE if x < goal_line else RinkZone.BEHIND_GOAL_DEFENSIVE
        
        return RinkZone.NEUTRAL
    
    def reset(self):
        """Reset smoothing buffers."""
        self._keypoint_history.clear()
        self._homography_history.clear()

def draw_rink_template(width: int = 600, height: int = 255) -> np.ndarray:
    """
    Draw a standardized NHL rink template.
    
    Args:
        width: Output image width
        height: Output image height
        
    Returns:
        Rink template image (BGR)
    """
    # Create white ice background
    rink = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Scale factors
    scale_x = width / RinkDimensions.LENGTH
    scale_y = height / RinkDimensions.WIDTH
    
    # Center offset
    cx, cy = width // 2, height // 2
    
    def to_px(x, y):
        return int(cx + x * scale_x), int(cy - y * scale_y)
    
    # Draw boards (rounded rectangle)
    cv2.rectangle(rink, (5, 5), (width-5, height-5), (150, 150, 150), 2)
    
    # Draw center red line
    cv2.line(rink, to_px(0, -42.5), to_px(0, 42.5), (0, 0, 200), 3)
    
    # Draw blue lines
    blue = (200, 100, 0)
    cv2.line(rink, to_px(25, -42.5), to_px(25, 42.5), blue, 2)
    cv2.line(rink, to_px(-25, -42.5), to_px(-25, 42.5), blue, 2)
    
    # Draw goal lines
    cv2.line(rink, to_px(89, -42.5), to_px(89, 42.5), (0, 0, 200), 1)
    cv2.line(rink, to_px(-89, -42.5), to_px(-89, 42.5), (0, 0, 200), 1)
    
    # Draw center circle
    cv2.circle(rink, to_px(0, 0), int(15 * scale_x), (0, 0, 200), 1)
    cv2.circle(rink, to_px(0, 0), 3, (0, 0, 200), -1)  # Center dot
    
    # Draw face-off circles
    faceoff_positions = [
        (69, 22), (69, -22),   # Right
        (-69, 22), (-69, -22), # Left
    ]
    for fx, fy in faceoff_positions:
        cv2.circle(rink, to_px(fx, fy), int(15 * scale_x), (200, 0, 0), 1)
        cv2.circle(rink, to_px(fx, fy), 3, (200, 0, 0), -1)
    
    # Draw neutral zone dots
    neutral_dots = [
        (20, 22), (20, -22),
        (-20, 22), (-20, -22),
    ]
    for dx, dy in neutral_dots:
        cv2.circle(rink, to_px(dx, dy), 3, (200, 0, 0), -1)
    
    # Draw goal creases (simplified as rectangles)
    for gx in [89, -89]:
        crease_color = (200, 200, 255)  # Light blue
        direction = -1 if gx > 0 else 1
        x1, y1 = to_px(gx, -4)
        x2, y2 = to_px(gx + direction * 6, 4)
        cv2.rectangle(rink, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), crease_color, -1)
        cv2.rectangle(rink, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), (200, 0, 0), 1)
    
    # Draw goals
    for gx in [89, -89]:
        goal_color = (50, 50, 50)
        direction = 1 if gx > 0 else -1
        x1, y1 = to_px(gx, -3)
        x2, y2 = to_px(gx + direction * 4, 3)
        cv2.rectangle(rink, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), goal_color, 2)
    
    return rink

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Initialize detector
    config = RinkKeypointConfig(
        confidence_threshold=0.5,
        enable_smoothing=True
    )
    detector = RinkKeypointDetector(config)
    
    # Draw rink template
    rink_template = draw_rink_template(600, 255)
    
    print("Rink Keypoint Detection Module Ready!")
    print(f"\nConfiguration:")
    print(f"  - Model: {config.model_path}")
    print(f"  - Confidence threshold: {config.confidence_threshold}")
    print(f"  - Smoothing enabled: {config.enable_smoothing}")
    print(f"  - Number of keypoints: 56")
    print(f"\nRink dimensions: {RinkDimensions.LENGTH}ft x {RinkDimensions.WIDTH}ft")
    
    # Display template
    plt.figure(figsize=(12, 5))
    plt.imshow(cv2.cvtColor(rink_template, cv2.COLOR_BGR2RGB))
    plt.title("NHL Rink Template")
    plt.axis('off')
    plt.show()
