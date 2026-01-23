import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import supervision as sv
from tqdm import tqdm
import torch
import os

# Import custom modules (from this project)
# These would be imported from the respective notebooks when run as scripts
# For notebook usage, cells from those notebooks can be copied or run via %run magic

# %run basketball_player_detection.ipynb
# %run basketball_team_classification.ipynb
# %run basketball_jersey_recognition.ipynb
# %run basketball_player_tracking.ipynb

# Pipeline Configuration
@dataclass
class PipelineConfig:
    """Configuration for the basketball analysis pipeline."""
    
    # Detection settings
    player_detection_model: str = "basketball-players-fy4c2/1"
    jersey_detection_model: str = "basketball-player-detection-3-ycjdo/6"
    player_confidence: float = 0.3
    jersey_confidence: float = 0.4
    
    # Tracking settings
    track_buffer: int = 30
    match_thresh: float = 0.8
    use_sam2: bool = False
    
    # Team classification settings
    use_siglip: bool = True  # False for color-based classification
    team_classifier_batch_size: int = 32
    
    # Jersey recognition settings
    use_vlm_ocr: bool = True  # False for ResNet classifier
    jersey_sample_interval: int = 5  # Sample every N frames
    stabilization_threshold: int = 3
    
    # General settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    roboflow_api_key: Optional[str] = None
    
    def __post_init__(self):
        # Try to get API key from environment
        if self.roboflow_api_key is None:
            self.roboflow_api_key = os.environ.get('ROBOFLOW_API_KEY')

@dataclass
class FrameResult:
    """Results for a single frame."""
    frame_number: int
    frame: np.ndarray
    detections: sv.Detections
    tracked_detections: sv.Detections
    team_assignments: np.ndarray
    player_labels: Dict[int, str]
    annotated_frame: Optional[np.ndarray] = None

class BasketballAnalysisPipeline:
    """
    Complete pipeline for basketball player detection, tracking,
    and identification.
    
    Combines multiple models to achieve robust player identification
    even with motion blur, occlusions, and similar uniforms.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        
        # Initialize components (lazy loading)
        self._player_detector = None
        self._jersey_detector = None
        self._tracker = None
        self._team_classifier = None
        self._jersey_ocr = None
        self._identity_manager = None
        
        # State
        self._team_classifier_fitted = False
        self._frame_count = 0
    
    @property
    def player_detector(self):
        """Lazy load player detector."""
        if self._player_detector is None:
            from inference import get_model
            self._player_detector = get_model(
                model_id=self.config.player_detection_model,
                api_key=self.config.roboflow_api_key
            )
        return self._player_detector
    
    @property
    def jersey_detector(self):
        """Lazy load jersey detector."""
        if self._jersey_detector is None:
            from inference import get_model
            self._jersey_detector = get_model(
                model_id=self.config.jersey_detection_model,
                api_key=self.config.roboflow_api_key
            )
        return self._jersey_detector
    
    @property
    def tracker(self):
        """Lazy load tracker."""
        if self._tracker is None:
            self._tracker = sv.ByteTrack(
                lost_track_buffer=self.config.track_buffer,
                minimum_matching_threshold=self.config.match_thresh
            )
        return self._tracker
    
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
                # Color-based classifier
                from sklearn.cluster import KMeans
                
                class ColorTeamClassifier:
                    def __init__(self):
                        self.cluster_model = KMeans(n_clusters=2, n_init=10)
                        self._fitted = False
                    
                    def extract_features(self, crops):
                        features = []
                        for crop in crops:
                            h = crop.shape[0]
                            jersey = crop[:int(h*0.6), :]
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
    
    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect players in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Player detections
        """
        result = self.player_detector.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        
        # Filter by confidence
        mask = detections.confidence >= self.config.player_confidence
        return detections[mask]
    
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
    
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single frame through the full pipeline.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            FrameResult with all analysis results
        """
        self._frame_count += 1
        
        # 1. Detect players
        detections = self.detect_players(frame)
        
        # 2. Track players
        tracked = self.tracker.update_with_detections(detections)
        
        # 3. Get player crops
        crops = self.get_player_crops(frame, tracked)
        
        # 4. Classify teams (fit on first frames with enough players)
        fit_classifier = not self._team_classifier_fitted and len(crops) >= 4
        team_assignments = self.classify_teams(crops, fit=fit_classifier)
        
        # 5. Jersey recognition (sample every N frames)
        player_labels = {}
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                player_labels[track_id] = self.identity_manager.get_label(track_id)
        
        return FrameResult(
            frame_number=self._frame_count,
            frame=frame,
            detections=detections,
            tracked_detections=tracked,
            team_assignments=team_assignments,
            player_labels=player_labels
        )
    
    def annotate_frame(self, result: FrameResult) -> np.ndarray:
        """
        Create annotated visualization of frame.
        
        Args:
            result: Frame processing result
            
        Returns:
            Annotated frame image
        """
        annotated = result.frame.copy()
        
        # Team colors
        team_colors = {
            0: sv.Color(0, 0, 255),   # Red for team 0
            1: sv.Color(255, 0, 0),   # Blue for team 1
        }
        
        # Create annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_LEFT,
            text_thickness=1,
            text_scale=0.5
        )
        
        # Assign colors based on team
        if len(result.team_assignments) > 0:
            colors = [team_colors.get(t, sv.Color(128, 128, 128)) for t in result.team_assignments]
        else:
            colors = None
        
        # Get labels
        labels = [result.player_labels.get(tid, f"ID:{tid}") 
                  for tid in (result.tracked_detections.tracker_id or [])]
        
        # Draw boxes and labels
        annotated = box_annotator.annotate(
            scene=annotated,
            detections=result.tracked_detections
        )
        
        if labels:
            annotated = label_annotator.annotate(
                scene=annotated,
                detections=result.tracked_detections,
                labels=labels
            )
        
        return annotated
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        stride: int = 1
    ) -> List[FrameResult]:
        """
        Process a complete video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            display: Whether to show live preview
            stride: Process every Nth frame
            
        Returns:
            List of FrameResult objects
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
        
        for frame in tqdm(frame_generator, total=total_frames, desc="Processing"):
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
                cv2.imshow('Basketball Analysis', annotated)
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
        self._tracker = None
        self._team_classifier_fitted = False
        self._identity_manager = None
        self._frame_count = 0

# Visualization utilities

def create_court_view(
    frame: np.ndarray,
    detections: sv.Detections,
    team_assignments: np.ndarray,
    court_keypoints: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create bird's-eye view of player positions on court.
    
    Args:
        frame: Input frame
        detections: Player detections
        team_assignments: Team labels
        court_keypoints: Court corner keypoints for homography
        
    Returns:
        Court view image
    """
    # Standard NBA court dimensions (in feet, scaled)
    COURT_WIDTH = 500  # pixels
    COURT_HEIGHT = 940  # pixels
    
    # Create court canvas
    court = np.ones((COURT_HEIGHT, COURT_WIDTH, 3), dtype=np.uint8) * 200
    
    # Draw court lines
    cv2.rectangle(court, (10, 10), (COURT_WIDTH-10, COURT_HEIGHT-10), (0, 0, 0), 2)
    cv2.line(court, (10, COURT_HEIGHT//2), (COURT_WIDTH-10, COURT_HEIGHT//2), (0, 0, 0), 2)
    
    # Draw center circle
    cv2.circle(court, (COURT_WIDTH//2, COURT_HEIGHT//2), 60, (0, 0, 0), 2)
    
    # If we have keypoints, compute homography
    if court_keypoints is not None and len(court_keypoints) >= 4:
        # Define target court corners
        target_pts = np.array([
            [10, 10],
            [COURT_WIDTH-10, 10],
            [COURT_WIDTH-10, COURT_HEIGHT-10],
            [10, COURT_HEIGHT-10]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(court_keypoints[:4], target_pts)
        
        # Transform player positions
        if H is not None:
            positions = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            transformed = cv2.perspectiveTransform(
                positions.reshape(-1, 1, 2).astype(np.float32), H
            ).reshape(-1, 2)
            
            # Draw players
            colors = [(0, 0, 255), (255, 0, 0)]  # Red, Blue
            for i, pos in enumerate(transformed):
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < COURT_WIDTH and 0 <= y < COURT_HEIGHT:
                    team = team_assignments[i] if i < len(team_assignments) else 0
                    cv2.circle(court, (x, y), 15, colors[team], -1)
    
    return court

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    config = PipelineConfig(
        use_siglip=True,
        use_sam2=False,
        jersey_sample_interval=5
    )
    
    pipeline = BasketballAnalysisPipeline(config)
    
    # Set roster for player name lookup
    roster = {
        "0": {"23": "LeBron James", "3": "Anthony Davis"},
        "1": {"30": "Stephen Curry", "11": "Klay Thompson"}
    }
    pipeline.identity_manager.set_roster(roster)
    
    # Process video
    # results = pipeline.process_video(
    #     video_path="basketball_game.mp4",
    #     output_path="analyzed_game.mp4",
    #     display=True
    # )
    
    print("Basketball Analysis Pipeline Ready!")
    print(f"Device: {config.device}")
    print(f"Team classifier: {'SigLIP' if config.use_siglip else 'Color-based'}")
    print(f"Tracking: ByteTrack" + (" + SAM2" if config.use_sam2 else ""))
