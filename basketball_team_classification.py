import cv2
import numpy as np
from typing import List, Generator, TypeVar, Iterable, Optional, Tuple
from dataclasses import dataclass
import supervision as sv
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from PIL import Image

# SigLIP model for visual embeddings
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

# Team classification constants
NUM_TEAMS = 2
UMAP_COMPONENTS = 3

# Type variable for generic batching
V = TypeVar('V')

def create_batches(
    sequence: Iterable[V],
    batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.
    
    Args:
        sequence: The input sequence to be batched
        batch_size: The size of each batch
        
    Yields:
        Batches of the input sequence
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

def cv2_to_pillow(image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image (BGR) to PIL Image (RGB).
    
    Args:
        image: OpenCV image in BGR format
        
    Returns:
        PIL Image in RGB format
    """
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

class TeamClassifier:
    """
    A classifier using pre-trained SigLIP vision model for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    
    This approach captures visual similarity between player appearances,
    enabling robust team classification even with similar uniform colors.
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32,
        n_clusters: int = NUM_TEAMS
    ):
        """
        Initialize the TeamClassifier.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Batch size for embedding extraction
            n_clusters: Number of teams/clusters (default: 2)
        """
        self.device = device
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        
        # Lazy loading of models
        self._features_model = None
        self._processor = None
        self._reducer = None
        self._cluster_model = None
        self._fitted = False
    
    def _load_models(self):
        """Lazy load the SigLIP model and processor."""
        if self._features_model is None:
            try:
                from transformers import AutoProcessor, SiglipVisionModel
                import umap
            except ImportError:
                raise ImportError(
                    "Please install transformers and umap-learn: "
                    "pip install transformers umap-learn"
                )
            
            self._features_model = SiglipVisionModel.from_pretrained(
                SIGLIP_MODEL_PATH
            ).to(self.device)
            self._processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
            self._reducer = umap.UMAP(n_components=UMAP_COMPONENTS)
            self._cluster_model = KMeans(n_clusters=self.n_clusters, n_init=10)
    
    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract visual features from image crops using SigLIP.
        
        Args:
            crops: List of player crop images (BGR format)
            
        Returns:
            Feature embeddings as numpy array
        """
        self._load_models()
        
        # Convert crops to PIL images
        pil_crops = [cv2_to_pillow(crop) for crop in crops]
        
        # Process in batches
        batches = create_batches(pil_crops, self.batch_size)
        embeddings_list = []
        
        with torch.no_grad():
            for batch in tqdm(batches, desc='Extracting embeddings'):
                inputs = self._processor(
                    images=batch,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self._features_model(**inputs)
                
                # Mean pooling over sequence dimension
                embeddings = torch.mean(
                    outputs.last_hidden_state, dim=1
                ).cpu().numpy()
                
                embeddings_list.append(embeddings)
        
        return np.concatenate(embeddings_list)
    
    def fit(self, crops: List[np.ndarray]) -> 'TeamClassifier':
        """
        Fit the classifier model on player image crops.
        
        Args:
            crops: List of player crop images to train on
            
        Returns:
            Self for method chaining
        """
        self._load_models()
        
        if len(crops) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} samples to fit {self.n_clusters} clusters"
            )
        
        # Extract features
        data = self.extract_features(crops)
        
        # Reduce dimensionality
        projections = self._reducer.fit_transform(data)
        
        # Cluster
        self._cluster_model.fit(projections)
        self._fitted = True
        
        return self
    
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict team labels for player image crops.
        
        Args:
            crops: List of player crop images
            
        Returns:
            Array of team labels (0 or 1)
        """
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before predicting")
        
        if len(crops) == 0:
            return np.array([])
        
        # Extract features
        data = self.extract_features(crops)
        
        # Project to lower dimension
        projections = self._reducer.transform(data)
        
        # Predict cluster labels
        return self._cluster_model.predict(projections)
    
    def fit_predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Fit the model and predict labels in one step.
        
        Args:
            crops: List of player crop images
            
        Returns:
            Array of team labels
        """
        self.fit(crops)
        return self.predict(crops)

class ColorBasedTeamClassifier:
    """
    Simple team classifier based on dominant jersey colors.
    
    This is a faster alternative to the SigLIP-based approach,
    useful for games with clearly different uniform colors.
    """
    
    def __init__(self, n_clusters: int = NUM_TEAMS):
        """
        Initialize the color-based classifier.
        
        Args:
            n_clusters: Number of teams (default: 2)
        """
        self.n_clusters = n_clusters
        self.cluster_model = KMeans(n_clusters=n_clusters, n_init=10)
        self._fitted = False
    
    def extract_color_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract color histogram features from player crops.
        
        Focuses on the upper body (jersey area) for more accurate
        color extraction.
        
        Args:
            crops: List of player crop images
            
        Returns:
            Color feature vectors
        """
        features = []
        
        for crop in crops:
            # Focus on upper 60% (jersey area)
            h = crop.shape[0]
            jersey_region = crop[:int(h * 0.6), :]
            
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms
            hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            # Normalize histograms
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            # Concatenate features
            feature_vector = np.concatenate([hist_h, hist_s, hist_v])
            features.append(feature_vector)
        
        return np.array(features)
    
    def fit(self, crops: List[np.ndarray]) -> 'ColorBasedTeamClassifier':
        """
        Fit the classifier on player crops.
        
        Args:
            crops: List of player crop images
            
        Returns:
            Self for method chaining
        """
        features = self.extract_color_features(crops)
        self.cluster_model.fit(features)
        self._fitted = True
        return self
    
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict team labels for player crops.
        
        Args:
            crops: List of player crop images
            
        Returns:
            Array of team labels
        """
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before predicting")
        
        if len(crops) == 0:
            return np.array([])
        
        features = self.extract_color_features(crops)
        return self.cluster_model.predict(features)

def resolve_team_assignments(
    player_detections: sv.Detections,
    team_labels: np.ndarray,
    referee_class_id: int = 2
) -> sv.Detections:
    """
    Add team assignments to player detections.
    
    Args:
        player_detections: Detection results
        team_labels: Array of team labels from classifier
        referee_class_id: Class ID for referees (excluded from team assignment)
        
    Returns:
        Detections with team_id in data dictionary
    """
    # Initialize team_id array
    team_ids = np.full(len(player_detections), -1, dtype=int)
    
    # Filter out referees
    player_mask = player_detections.class_id != referee_class_id
    player_indices = np.where(player_mask)[0]
    
    # Assign team labels to players
    for idx, label in zip(player_indices, team_labels):
        team_ids[idx] = label
    
    # Add to detections data
    if player_detections.data is None:
        player_detections.data = {}
    
    player_detections.data['team_id'] = team_ids
    
    return player_detections

def get_team_colors(
    team_labels: np.ndarray,
    team_0_color: Tuple[int, int, int] = (255, 0, 0),  # Blue (BGR)
    team_1_color: Tuple[int, int, int] = (0, 0, 255),  # Red (BGR)
    unknown_color: Tuple[int, int, int] = (128, 128, 128)  # Gray
) -> List[Tuple[int, int, int]]:
    """
    Map team labels to colors for visualization.
    
    Args:
        team_labels: Array of team labels
        team_0_color: Color for team 0 (BGR format)
        team_1_color: Color for team 1 (BGR format)
        unknown_color: Color for unknown/referee (BGR format)
        
    Returns:
        List of colors corresponding to each label
    """
    color_map = {
        0: team_0_color,
        1: team_1_color,
        -1: unknown_color
    }
    
    return [color_map.get(label, unknown_color) for label in team_labels]

# Example usage
if __name__ == "__main__":
    # Example: Using SigLIP-based classifier
    classifier = TeamClassifier(device='cpu', batch_size=16)
    
    # Example: Using color-based classifier (faster, simpler)
    color_classifier = ColorBasedTeamClassifier()
    
    # To use:
    # 1. Get player crops from detection module
    # crops = get_player_crops(frame, detections)
    
    # 2. Fit classifier on initial frames
    # classifier.fit(crops)
    
    # 3. Predict team labels
    # labels = classifier.predict(new_crops)
    
    print("Team classifiers initialized successfully!")
