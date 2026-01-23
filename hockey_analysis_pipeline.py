import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import supervision as sv
from tqdm import tqdm
import torch
import os

# Import custom modules (from this project)
# These would be imported from the respective notebooks when run as scripts
# For notebook usage, cells from those notebooks can be copied or run via %run magic

# %run hockey_player_detection.ipynb
# %run hockey_team_classification.ipynb
# %run hockey_jersey_recognition.ipynb
# %run hockey_player_tracking.ipynb

# Hockey-specific class labels
class HockeyClasses:
    """Class labels for hockey detection."""
    PLAYER = 0
    GOALIE = 1
    REFEREE = 2
    PUCK = 3
    
    @classmethod
    def get_name(cls, class_id: int) -> str:
        names = {0: 'player', 1: 'goalie', 2: 'referee', 3: 'puck'}
        return names.get(class_id, 'unknown')
    
    @classmethod
    def is_skater(cls, class_id: int) -> bool:
        """Check if class is a skater (player or goalie)."""
        return class_id in [cls.PLAYER, cls.GOALIE]

# Pipeline Configuration
@dataclass
class HockeyPipelineConfig:
    """Configuration for the hockey analysis pipeline."""
    
    # RF-DETR Detection settings
    use_rf_detr: bool = True  # Use RF-DETR (Meta DINOv2) instead of YOLO
    rf_detr_model: str = "rf-detr-medium"  # Options: rf-detr-nano, rf-detr-small, rf-detr-medium
    player_detection_model: str = "hockey-players-puck/1"  # Roboflow model as fallback
    player_confidence: float = 0.35
    puck_confidence: float = 0.25  # Lower for small, fast puck
    goalie_confidence: float = 0.4
    
    # Tracking settings
    track_buffer: int = 45  # Longer buffer for hockey (fast movements)
    match_thresh: float = 0.75
    use_sam2: bool = False
    
    # Team classification settings
    use_siglip: bool = True  # False for color-based classification
    team_classifier_batch_size: int = 32
    
    # Jersey recognition settings
    use_vlm_ocr: bool = True  # SmolVLM2 for OCR
    jersey_sample_interval: int = 8  # Sample less frequently (motion blur)
    stabilization_threshold: int = 4  # More confirmations needed
    
    # Hockey-specific settings
    enable_puck_tracking: bool = True
    enable_goalie_tracking: bool = True
    rink_keypoint_model: Optional[str] = None  # For bird's-eye view
    
    # General settings
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    roboflow_api_key: Optional[str] = None
    
    def __post_init__(self):
        # Try to get API key from environment
        if self.roboflow_api_key is None:
            self.roboflow_api_key = os.environ.get('ROBOFLOW_API_KEY')

@dataclass
class HockeyFrameResult:
    """Results for a single frame of hockey analysis."""
    frame_number: int
    frame: np.ndarray
    
    # Player detections and tracking
    player_detections: sv.Detections
    tracked_players: sv.Detections
    team_assignments: np.ndarray
    player_labels: Dict[int, str]
    
    # Goalie detections
    goalie_detections: Optional[sv.Detections] = None
    
    # Puck tracking
    puck_detection: Optional[sv.Detections] = None
    puck_position: Optional[Tuple[float, float]] = None
    
    # Visualization
    annotated_frame: Optional[np.ndarray] = None

class RFDETRDetector:
    """
    RF-DETR detector wrapper using Meta's DINOv2 backbone.
    
    RF-DETR is a state-of-the-art transformer-based object detector that:
    - Uses Meta's DINOv2 self-supervised vision backbone
    - Eliminates anchor boxes and NMS for true end-to-end detection
    - Achieves 54.7% mAP on COCO at real-time speeds
    - Excels at fine-tuning on custom datasets with fewer epochs
    """
    
    def __init__(
        self,
        model_variant: str = "rf-detr-medium",
        device: str = "cuda",
        confidence_threshold: float = 0.35
    ):
        """
        Initialize RF-DETR detector.
        
        Args:
            model_variant: Model size (rf-detr-nano, rf-detr-small, rf-detr-medium)
            device: Device to run inference on
            confidence_threshold: Detection confidence threshold
        """
        self.model_variant = model_variant
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model = None
        
    def _load_model(self):
        """Lazy load RF-DETR model."""
        if self._model is None:
            try:
                from rfdetr import RFDETRBase, RFDETRSmall, RFDETRMedium, RFDETRLarge
                
                model_map = {
                    "rf-detr-nano": RFDETRBase,  # Nano uses base architecture
                    "rf-detr-small": RFDETRSmall,
                    "rf-detr-medium": RFDETRMedium,
                    "rf-detr-large": RFDETRLarge,
                }
                
                model_class = model_map.get(self.model_variant, RFDETRMedium)
                self._model = model_class()
                
                print(f"Loaded RF-DETR model: {self.model_variant}")
                print(f"  - Backbone: Meta DINOv2 (self-supervised)")
                print(f"  - Architecture: Transformer (no anchors, no NMS)")
                print(f"  - Device: {self.device}")
                
            except ImportError:
                print("RF-DETR not installed. Install with: pip install rf-detr")
                print("Falling back to Roboflow inference...")
                self._model = None
                
        return self._model
    
    def predict(self, image: np.ndarray) -> sv.Detections:
        """
        Run detection on an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Detections object
        """
        model = self._load_model()
        
        if model is None:
            # Return empty detections if model failed to load
            return sv.Detections.empty()
        
        # RF-DETR expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        detections = model.predict(rgb_image, threshold=self.confidence_threshold)
        
        return detections
    
    def fine_tune(
        self,
        dataset_path: str,
        epochs: int = 50,
        batch_size: int = 8,
        output_dir: str = "./hockey_rf_detr_finetuned"
    ):
        """
        Fine-tune RF-DETR on hockey dataset.
        
        RF-DETR converges faster than YOLO on custom datasets due to
        the pre-trained DINOv2 backbone.
        
        Args:
            dataset_path: Path to dataset in COCO format
            epochs: Number of training epochs
            batch_size: Training batch size
            output_dir: Directory to save fine-tuned model
        """
        model = self._load_model()
        
        if model is None:
            raise RuntimeError("RF-DETR model not loaded. Cannot fine-tune.")
        
        print(f"Fine-tuning RF-DETR on hockey dataset...")
        print(f"  - Dataset: {dataset_path}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        
        # Fine-tuning code
        model.train(
            dataset_dir=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            output_dir=output_dir,
            device=self.device
        )
        
        print(f"Fine-tuned model saved to: {output_dir}")

class HockeyAnalysisPipeline:
    """
    Complete pipeline for hockey player detection, tracking,
    and identification using RF-DETR.
    
    Key features:
    - RF-DETR with Meta's DINOv2 backbone for superior detection
    - Puck tracking optimized for small, fast-moving objects
    - Goalie-specific handling
    - Team classification via SigLIP embeddings
    - Jersey OCR with stabilization heuristics
    """
    
    def __init__(self, config: Optional[HockeyPipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or HockeyPipelineConfig()
        
        # Initialize components (lazy loading)
        self._rf_detr_detector = None
        self._fallback_detector = None
        self._player_tracker = None
        self._puck_tracker = None
        self._team_classifier = None
        self._jersey_ocr = None
        self._identity_manager = None
        
        # State
        self._team_classifier_fitted = False
        self._frame_count = 0
        self._puck_history = []  # Track puck positions for trajectory
    
    @property
    def detector(self):
        """Get the primary detector (RF-DETR or fallback)."""
        if self.config.use_rf_detr:
            if self._rf_detr_detector is None:
                self._rf_detr_detector = RFDETRDetector(
                    model_variant=self.config.rf_detr_model,
                    device=self.config.device,
                    confidence_threshold=self.config.player_confidence
                )
            return self._rf_detr_detector
        else:
            if self._fallback_detector is None:
                from inference import get_model
                self._fallback_detector = get_model(
                    model_id=self.config.player_detection_model,
                    api_key=self.config.roboflow_api_key
                )
            return self._fallback_detector
    
    @property
    def player_tracker(self):
        """Lazy load player tracker."""
        if self._player_tracker is None:
            self._player_tracker = sv.ByteTrack(
                lost_track_buffer=self.config.track_buffer,
                minimum_matching_threshold=self.config.match_thresh
            )
        return self._player_tracker
    
    @property
    def puck_tracker(self):
        """Lazy load puck tracker (separate tracker for small object)."""
        if self._puck_tracker is None:
            self._puck_tracker = sv.ByteTrack(
                lost_track_buffer=60,  # Longer buffer for puck occlusions
                minimum_matching_threshold=0.5  # More lenient matching
            )
        return self._puck_tracker
    
    @property
    def team_classifier(self):
        """Lazy load team classifier."""
        if self._team_classifier is None:
            if self.config.use_siglip:
                # SigLIP-based classifier
                from transformers import AutoProcessor, SiglipVisionModel
                import umap
                from sklearn.cluster import KMeans
                
                class SigLIPTeamClassifier:
                    def __init__(self, device, batch_size):
                        self.device = device
                        self.batch_size = batch_size
                        self.model = SiglipVisionModel.from_pretrained(
                            'google/siglip-base-patch16-224'
                        ).to(device)
                        self.processor = AutoProcessor.from_pretrained(
                            'google/siglip-base-patch16-224'
                        )
                        self.reducer = umap.UMAP(n_components=3)
                        self.cluster_model = KMeans(n_clusters=2, n_init=10)
                        self._fitted = False
                    
                    def extract_features(self, crops):
                        from PIL import Image
                        pil_crops = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
                        embeddings = []
                        
                        with torch.no_grad():
                            for i in range(0, len(pil_crops), self.batch_size):
                                batch = pil_crops[i:i+self.batch_size]
                                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                                outputs = self.model(**inputs)
                                emb = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                                embeddings.append(emb)
                        
                        return np.concatenate(embeddings)
                    
                    def fit(self, crops):
                        features = self.extract_features(crops)
                        projections = self.reducer.fit_transform(features)
                        self.cluster_model.fit(projections)
                        self._fitted = True
                    
                    def predict(self, crops):
                        if len(crops) == 0:
                            return np.array([])
                        features = self.extract_features(crops)
                        projections = self.reducer.transform(features)
                        return self.cluster_model.predict(projections)
                
                self._team_classifier = SigLIPTeamClassifier(
                    self.config.device,
                    self.config.team_classifier_batch_size
                )
            else:
                # Color-based classifier (hockey jerseys often have distinct colors)
                from sklearn.cluster import KMeans
                
                class ColorTeamClassifier:
                    def __init__(self):
                        self.cluster_model = KMeans(n_clusters=2, n_init=10)
                        self._fitted = False
                    
                    def extract_features(self, crops):
                        features = []
                        for crop in crops:
                            h = crop.shape[0]
                            # Hockey jerseys: focus on torso area
                            jersey = crop[int(h*0.2):int(h*0.7), :]
                            hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
                            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                            features.append(cv2.normalize(hist, hist).flatten())
                        return np.array(features)
                    
                    def fit(self, crops):
                        features = self.extract_features(crops)
                        self.cluster_model.fit(features)
                        self._fitted = True
                    
                    def predict(self, crops):
                        if len(crops) == 0:
                            return np.array([])
                        features = self.extract_features(crops)
                        return self.cluster_model.predict(features)
                
                self._team_classifier = ColorTeamClassifier()
        
        return self._team_classifier
    
    @property
    def identity_manager(self):
        """Lazy load identity manager."""
        if self._identity_manager is None:
            from collections import Counter, defaultdict
            
            class IdentityManager:
                def __init__(self, stabilization_threshold):
                    self.stabilization_threshold = stabilization_threshold
                    self.predictions = defaultdict(list)
                    self.confirmed = {}
                    self.roster = {}
                
                def add_prediction(self, track_id, number, team_id=-1):
                    if track_id in self.confirmed:
                        return True
                    
                    self.predictions[track_id].append(number)
                    
                    recent = self.predictions[track_id][-self.stabilization_threshold:]
                    if len(recent) >= self.stabilization_threshold and len(set(recent)) == 1:
                        self.confirmed[track_id] = {
                            'number': recent[0],
                            'team_id': team_id
                        }
                        return True
                    return False
                
                def get_label(self, track_id):
                    if track_id in self.confirmed:
                        num = self.confirmed[track_id]['number']
                        team = self.confirmed[track_id]['team_id']
                        name = self.roster.get(str(team), {}).get(num)
                        if name:
                            return f"{name} #{num}"
                        return f"#{num}"
                    elif track_id in self.predictions and self.predictions[track_id]:
                        counter = Counter(self.predictions[track_id])
                        best = counter.most_common(1)[0][0]
                        return f"#{best}?"
                    return f"ID:{track_id}"
                
                def set_roster(self, roster):
                    self.roster = roster
            
            self._identity_manager = IdentityManager(
                self.config.stabilization_threshold
            )
        
        return self._identity_manager
    
    def detect(self, frame: np.ndarray) -> Tuple[sv.Detections, sv.Detections, sv.Detections]:
        """
        Detect all objects in frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (player_detections, goalie_detections, puck_detections)
        """
        if self.config.use_rf_detr:
            # RF-DETR detection
            all_detections = self.detector.predict(frame)
        else:
            # Roboflow inference
            result = self.detector.infer(frame)[0]
            all_detections = sv.Detections.from_inference(result)
        
        # Separate by class
        player_mask = all_detections.class_id == HockeyClasses.PLAYER
        goalie_mask = all_detections.class_id == HockeyClasses.GOALIE
        puck_mask = all_detections.class_id == HockeyClasses.PUCK
        
        # Apply confidence thresholds
        player_conf_mask = all_detections.confidence >= self.config.player_confidence
        goalie_conf_mask = all_detections.confidence >= self.config.goalie_confidence
        puck_conf_mask = all_detections.confidence >= self.config.puck_confidence
        
        players = all_detections[player_mask & player_conf_mask]
        goalies = all_detections[goalie_mask & goalie_conf_mask]
        pucks = all_detections[puck_mask & puck_conf_mask]
        
        return players, goalies, pucks
    
    def get_player_crops(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> List[np.ndarray]:
        """
        Extract player image crops.
        
        Args:
            frame: Input frame
            detections: Player detections
            
        Returns:
            List of player crop images
        """
        h, w = frame.shape[:2]
        crops = []
        
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        
        return crops
    
    def classify_teams(
        self,
        crops: List[np.ndarray],
        fit: bool = False
    ) -> np.ndarray:
        """
        Classify players into teams.
        
        Args:
            crops: Player image crops
            fit: Whether to fit the classifier (first frames)
            
        Returns:
            Array of team labels (0 or 1)
        """
        if len(crops) < 2:
            return np.zeros(len(crops), dtype=int)
        
        if fit and not self._team_classifier_fitted:
            self.team_classifier.fit(crops)
            self._team_classifier_fitted = True
        
        return self.team_classifier.predict(crops)
    
    def track_puck(self, puck_detections: sv.Detections) -> Optional[Tuple[float, float]]:
        """
        Track the puck position with interpolation for missed frames.
        
        Args:
            puck_detections: Puck detection for current frame
            
        Returns:
            Puck position (x, y) or None if not found
        """
        if len(puck_detections) > 0:
            # Get puck position (use highest confidence if multiple)
            best_idx = np.argmax(puck_detections.confidence)
            bbox = puck_detections.xyxy[best_idx]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            self._puck_history.append((center_x, center_y, self._frame_count))
            
            # Keep only recent history
            if len(self._puck_history) > 60:
                self._puck_history = self._puck_history[-60:]
            
            return (center_x, center_y)
        
        # Interpolate if puck not detected but we have history
        if len(self._puck_history) >= 2:
            # Simple linear extrapolation
            last = self._puck_history[-1]
            prev = self._puck_history[-2]
            
            dt = self._frame_count - last[2]
            if dt <= 5:  # Only extrapolate for 5 frames
                vx = last[0] - prev[0]
                vy = last[1] - prev[1]
                pred_x = last[0] + vx * dt
                pred_y = last[1] + vy * dt
                return (pred_x, pred_y)
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> HockeyFrameResult:
        """
        Process a single frame through the full pipeline.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            HockeyFrameResult with all analysis results
        """
        self._frame_count += 1
        
        # 1. Detect players, goalies, and puck
        players, goalies, pucks = self.detect(frame)
        
        # 2. Track players
        tracked_players = self.player_tracker.update_with_detections(players)
        
        # 3. Track puck
        puck_position = None
        if self.config.enable_puck_tracking:
            puck_position = self.track_puck(pucks)
        
        # 4. Get player crops for team classification
        crops = self.get_player_crops(frame, tracked_players)
        
        # 5. Classify teams (fit on first frames with enough players)
        fit_classifier = not self._team_classifier_fitted and len(crops) >= 4
        team_assignments = self.classify_teams(crops, fit=fit_classifier)
        
        # 6. Get player labels
        player_labels = {}
        if tracked_players.tracker_id is not None:
            for i, track_id in enumerate(tracked_players.tracker_id):
                player_labels[track_id] = self.identity_manager.get_label(track_id)
        
        return HockeyFrameResult(
            frame_number=self._frame_count,
            frame=frame,
            player_detections=players,
            tracked_players=tracked_players,
            team_assignments=team_assignments,
            player_labels=player_labels,
            goalie_detections=goalies if self.config.enable_goalie_tracking else None,
            puck_detection=pucks if self.config.enable_puck_tracking else None,
            puck_position=puck_position
        )
    
    def annotate_frame(self, result: HockeyFrameResult) -> np.ndarray:
        """
        Create annotated visualization of frame.
        
        Args:
            result: Frame processing result
            
        Returns:
            Annotated frame image
        """
        annotated = result.frame.copy()
        
        # Team colors (typical hockey colors)
        team_colors = {
            0: sv.Color(0, 0, 200),   # Red for team 0 (home)
            1: sv.Color(200, 200, 200),  # White for team 1 (away)
        }
        
        # Create annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_LEFT,
            text_thickness=1,
            text_scale=0.5
        )
        
        # Annotate players
        if len(result.tracked_players) > 0:
            labels = [result.player_labels.get(tid, f"ID:{tid}") 
                      for tid in (result.tracked_players.tracker_id or [])]
            
            annotated = box_annotator.annotate(
                scene=annotated,
                detections=result.tracked_players
            )
            
            if labels:
                annotated = label_annotator.annotate(
                    scene=annotated,
                    detections=result.tracked_players,
                    labels=labels
                )
        
        # Annotate goalies with distinct style
        if result.goalie_detections is not None and len(result.goalie_detections) > 0:
            goalie_annotator = sv.BoxAnnotator(thickness=3, color=sv.Color(0, 255, 255))
            annotated = goalie_annotator.annotate(
                scene=annotated,
                detections=result.goalie_detections
            )
        
        # Annotate puck
        if result.puck_position is not None:
            px, py = int(result.puck_position[0]), int(result.puck_position[1])
            cv2.circle(annotated, (px, py), 8, (0, 255, 0), 2)
            cv2.circle(annotated, (px, py), 3, (0, 255, 0), -1)
            
            # Draw puck trail
            if len(self._puck_history) >= 2:
                for i in range(1, min(len(self._puck_history), 10)):
                    pt1 = (int(self._puck_history[-i][0]), int(self._puck_history[-i][1]))
                    pt2 = (int(self._puck_history[-i-1][0]), int(self._puck_history[-i-1][1]))
                    alpha = 1.0 - (i / 10)
                    cv2.line(annotated, pt1, pt2, (0, int(255*alpha), 0), 2)
        
        # Add frame info
        info_text = f"Frame: {result.frame_number} | Players: {len(result.tracked_players)}"
        if result.puck_position:
            info_text += " | Puck: Tracked"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        return annotated
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        stride: int = 1
    ) -> List[HockeyFrameResult]:
        """
        Process a complete video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            display: Whether to show live preview
            stride: Process every Nth frame
            
        Returns:
            List of HockeyFrameResult objects
        """
        # Get video info
        video_info = sv.VideoInfo.from_video_path(video_path)
        
        results = []
        
        # Setup video writer if output requested
        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                video_info.fps // stride,
                (video_info.width, video_info.height)
            )
        
        # Process frames
        frame_generator = sv.get_video_frames_generator(
            video_path, stride=stride
        )
        
        total_frames = video_info.total_frames // stride
        
        for frame in tqdm(frame_generator, total=total_frames, desc="Processing Hockey Video"):
            # Process frame
            result = self.process_frame(frame)
            
            # Annotate
            annotated = self.annotate_frame(result)
            result.annotated_frame = annotated
            
            results.append(result)
            
            # Write output
            if writer:
                writer.write(annotated)
            
            # Display preview
            if display:
                cv2.imshow('Hockey Analysis', annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
        
        # Cleanup
        if writer:
            writer.release()
        
        if display:
            cv2.destroyAllWindows()
        
        return results
    
    def reset(self):
        """Reset pipeline state for new video."""
        self._player_tracker = None
        self._puck_tracker = None
        self._team_classifier_fitted = False
        self._identity_manager = None
        self._frame_count = 0
        self._puck_history = []

# Rink visualization utilities

def create_rink_view(
    frame: np.ndarray,
    detections: sv.Detections,
    team_assignments: np.ndarray,
    puck_position: Optional[Tuple[float, float]] = None,
    rink_keypoints: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create bird's-eye view of player positions on hockey rink.
    
    Args:
        frame: Input frame
        detections: Player detections
        team_assignments: Team labels
        puck_position: Puck position (x, y)
        rink_keypoints: Rink corner keypoints for homography
        
    Returns:
        Rink view image
    """
    # NHL rink dimensions (200ft x 85ft, scaled to pixels)
    RINK_WIDTH = 400  # pixels (200ft)
    RINK_HEIGHT = 170  # pixels (85ft)
    
    # Create rink canvas
    rink = np.ones((RINK_HEIGHT + 40, RINK_WIDTH + 40, 3), dtype=np.uint8) * 240
    
    # Draw rink outline
    pts = np.array([
        [20, 40], [RINK_WIDTH + 20, 40],
        [RINK_WIDTH + 20, RINK_HEIGHT], [20, RINK_HEIGHT]
    ], np.int32)
    cv2.polylines(rink, [pts], True, (0, 0, 0), 2)
    
    # Draw center line (red)
    center_x = RINK_WIDTH // 2 + 20
    cv2.line(rink, (center_x, 40), (center_x, RINK_HEIGHT), (0, 0, 200), 2)
    
    # Draw blue lines
    blue1_x = RINK_WIDTH // 3 + 20
    blue2_x = 2 * RINK_WIDTH // 3 + 20
    cv2.line(rink, (blue1_x, 40), (blue1_x, RINK_HEIGHT), (200, 0, 0), 2)
    cv2.line(rink, (blue2_x, 40), (blue2_x, RINK_HEIGHT), (200, 0, 0), 2)
    
    # Draw center circle
    center_y = (40 + RINK_HEIGHT) // 2
    cv2.circle(rink, (center_x, center_y), 30, (0, 0, 200), 2)
    
    # Draw goal creases
    cv2.rectangle(rink, (25, center_y - 15), (45, center_y + 15), (0, 0, 200), 2)
    cv2.rectangle(rink, (RINK_WIDTH - 5, center_y - 15), (RINK_WIDTH + 15, center_y + 15), (0, 0, 200), 2)
    
    # If we have keypoints, compute homography and transform positions
    if rink_keypoints is not None and len(rink_keypoints) >= 4:
        target_pts = np.array([
            [20, 40],
            [RINK_WIDTH + 20, 40],
            [RINK_WIDTH + 20, RINK_HEIGHT],
            [20, RINK_HEIGHT]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(rink_keypoints[:4], target_pts)
        
        if H is not None:
            # Transform player positions
            positions = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            transformed = cv2.perspectiveTransform(
                positions.reshape(-1, 1, 2).astype(np.float32), H
            ).reshape(-1, 2)
            
            # Draw players
            colors = [(0, 0, 200), (200, 200, 200)]  # Home (red), Away (white)
            for i, pos in enumerate(transformed):
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < RINK_WIDTH + 40 and 0 <= y < RINK_HEIGHT + 40:
                    team = team_assignments[i] if i < len(team_assignments) else 0
                    cv2.circle(rink, (x, y), 8, colors[team], -1)
                    cv2.circle(rink, (x, y), 8, (0, 0, 0), 1)
            
            # Draw puck
            if puck_position is not None:
                puck_trans = cv2.perspectiveTransform(
                    np.array([[puck_position]], dtype=np.float32), H
                )[0][0]
                px, py = int(puck_trans[0]), int(puck_trans[1])
                if 0 <= px < RINK_WIDTH + 40 and 0 <= py < RINK_HEIGHT + 40:
                    cv2.circle(rink, (px, py), 5, (0, 0, 0), -1)
    
    return rink

# Example usage
if __name__ == "__main__":
    # Initialize pipeline with RF-DETR (Meta DINOv2 backbone)
    config = HockeyPipelineConfig(
        use_rf_detr=True,
        rf_detr_model="rf-detr-medium",
        use_siglip=True,
        use_sam2=False,
        enable_puck_tracking=True,
        jersey_sample_interval=8
    )
    
    pipeline = HockeyAnalysisPipeline(config)
    
    # Set roster for player name lookup (example NHL teams)
    roster = {
        "0": {  # Home team
            "87": "Sidney Crosby",
            "71": "Evgeni Malkin",
            "58": "Kris Letang"
        },
        "1": {  # Away team
            "97": "Connor McDavid",
            "29": "Leon Draisaitl",
            "93": "Ryan Nugent-Hopkins"
        }
    }
    pipeline.identity_manager.set_roster(roster)
    
    # Process video
    # results = pipeline.process_video(
    #     video_path="hockey_game.mp4",
    #     output_path="analyzed_hockey.mp4",
    #     display=True
    # )
    
    print("Hockey Analysis Pipeline Ready!")
    print(f"\nConfiguration:")
    print(f"  - Detector: {'RF-DETR (' + config.rf_detr_model + ')' if config.use_rf_detr else 'Roboflow YOLO'}")
    print(f"  - Backbone: Meta DINOv2 (self-supervised)" if config.use_rf_detr else "")
    print(f"  - Team classifier: {'SigLIP' if config.use_siglip else 'Color-based'}")
    print(f"  - Tracking: ByteTrack" + (" + SAM2" if config.use_sam2 else ""))
    print(f"  - Puck tracking: {'Enabled' if config.enable_puck_tracking else 'Disabled'}")
    print(f"  - Device: {config.device}")
