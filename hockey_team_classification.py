import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
from sklearn.cluster import KMeans
from PIL import Image

@dataclass
class TeamClassificationConfig:
    """Configuration for team classification."""
    
    # Method selection
    use_siglip: bool = True  # False for color-based
    
    # SigLIP settings
    siglip_model: str = "google/siglip-base-patch16-224"
    batch_size: int = 32
    
    # UMAP settings for dimensionality reduction
    umap_components: int = 3
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    
    # Clustering settings
    n_teams: int = 2  # Usually 2 teams
    kmeans_init: int = 10  # Number of KMeans initializations
    
    # Color-based settings
    jersey_crop_top: float = 0.15  # Start of jersey (below helmet)
    jersey_crop_bottom: float = 0.65  # End of jersey (above pants)
    hist_bins_h: int = 30  # Hue bins
    hist_bins_s: int = 32  # Saturation bins
    
    # Device
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

class SigLIPTeamClassifier:
    """
    Team classifier using Google's SigLIP vision model.
    
    SigLIP provides robust visual embeddings that capture uniform
    appearance beyond just color, including patterns, logos, and style.
    """
    
    def __init__(self, config: Optional[TeamClassificationConfig] = None):
        """
        Initialize the classifier.
        
        Args:
            config: Classification configuration
        """
        self.config = config or TeamClassificationConfig()
        
        self._model = None
        self._processor = None
        self._reducer = None
        self._cluster_model = None
        self._fitted = False
    
    def _load_model(self):
        """Load SigLIP model and processor."""
        if self._model is not None:
            return
        
        from transformers import AutoProcessor, SiglipVisionModel
        import umap
        
        print(f"Loading SigLIP model: {self.config.siglip_model}")
        
        self._model = SiglipVisionModel.from_pretrained(
            self.config.siglip_model
        ).to(self.config.device)
        
        self._processor = AutoProcessor.from_pretrained(
            self.config.siglip_model
        )
        
        self._reducer = umap.UMAP(
            n_components=self.config.umap_components,
            n_neighbors=self.config.umap_neighbors,
            min_dist=self.config.umap_min_dist
        )
        
        self._cluster_model = KMeans(
            n_clusters=self.config.n_teams,
            n_init=self.config.kmeans_init
        )
        
        print(f"✓ SigLIP loaded on {self.config.device}")
    
    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract SigLIP embeddings from player crops.
        
        Args:
            crops: List of player image crops (BGR)
            
        Returns:
            Feature embeddings array
        """
        self._load_model()
        
        # Convert to PIL
        pil_crops = [
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            for c in crops
        ]
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(pil_crops), self.config.batch_size):
                batch = pil_crops[i:i + self.config.batch_size]
                
                inputs = self._processor(
                    images=batch,
                    return_tensors="pt"
                ).to(self.config.device)
                
                outputs = self._model(**inputs)
                
                # Mean pool over spatial dimensions
                emb = torch.mean(
                    outputs.last_hidden_state, dim=1
                ).cpu().numpy()
                
                embeddings.append(emb)
        
        return np.concatenate(embeddings)
    
    def fit(self, crops: List[np.ndarray]):
        """
        Fit the classifier on player crops.
        
        Args:
            crops: List of player image crops
        """
        if len(crops) < self.config.n_teams:
            print(f"Warning: Need at least {self.config.n_teams} crops to fit")
            return
        
        # Extract features
        features = self.extract_features(crops)
        
        # Reduce dimensionality
        projections = self._reducer.fit_transform(features)
        
        # Cluster
        self._cluster_model.fit(projections)
        
        self._fitted = True
        print(f"✓ Classifier fitted on {len(crops)} samples")
    
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict team labels for player crops.
        
        Args:
            crops: List of player image crops
            
        Returns:
            Array of team labels (0 or 1)
        """
        if not self._fitted:
            print("Warning: Classifier not fitted, returning zeros")
            return np.zeros(len(crops), dtype=int)
        
        if len(crops) == 0:
            return np.array([], dtype=int)
        
        # Extract features
        features = self.extract_features(crops)
        
        # Reduce dimensionality
        projections = self._reducer.transform(features)
        
        # Predict
        return self._cluster_model.predict(projections)
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted

class ColorTeamClassifier:
    """
    Team classifier using color histograms.
    
    Simpler and faster than SigLIP, works well when teams
    have distinctly different jersey colors.
    """
    
    def __init__(self, config: Optional[TeamClassificationConfig] = None):
        """
        Initialize the classifier.
        
        Args:
            config: Classification configuration
        """
        self.config = config or TeamClassificationConfig()
        
        self._cluster_model = KMeans(
            n_clusters=self.config.n_teams,
            n_init=self.config.kmeans_init
        )
        self._fitted = False
    
    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract color histogram features from player crops.
        
        Args:
            crops: List of player image crops (BGR)
            
        Returns:
            Feature array
        """
        features = []
        
        for crop in crops:
            # Extract jersey region
            h = crop.shape[0]
            top = int(h * self.config.jersey_crop_top)
            bottom = int(h * self.config.jersey_crop_bottom)
            jersey = crop[top:bottom, :]
            
            # Convert to HSV
            hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
            
            # Compute histogram
            hist = cv2.calcHist(
                [hsv], [0, 1], None,
                [self.config.hist_bins_h, self.config.hist_bins_s],
                [0, 180, 0, 256]
            )
            
            # Normalize and flatten
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
        
        return np.array(features)
    
    def fit(self, crops: List[np.ndarray]):
        """
        Fit the classifier on player crops.
        
        Args:
            crops: List of player image crops
        """
        if len(crops) < self.config.n_teams:
            print(f"Warning: Need at least {self.config.n_teams} crops to fit")
            return
        
        features = self.extract_features(crops)
        self._cluster_model.fit(features)
        self._fitted = True
        
        print(f"✓ Color classifier fitted on {len(crops)} samples")
    
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict team labels for player crops.
        
        Args:
            crops: List of player image crops
            
        Returns:
            Array of team labels (0 or 1)
        """
        if not self._fitted:
            print("Warning: Classifier not fitted, returning zeros")
            return np.zeros(len(crops), dtype=int)
        
        if len(crops) == 0:
            return np.array([], dtype=int)
        
        features = self.extract_features(crops)
        return self._cluster_model.predict(features)
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted

def get_team_colors(
    crops: List[np.ndarray],
    team_labels: np.ndarray
) -> Dict[int, Tuple[int, int, int]]:
    """
    Extract dominant color for each team.
    
    Args:
        crops: Player image crops
        team_labels: Team assignments
        
    Returns:
        Dictionary mapping team ID to BGR color
    """
    team_colors = {}
    
    for team_id in [0, 1]:
        team_crops = [c for c, l in zip(crops, team_labels) if l == team_id]
        
        if not team_crops:
            team_colors[team_id] = (128, 128, 128)  # Gray default
            continue
        
        # Combine jersey regions
        jersey_pixels = []
        for crop in team_crops:
            h = crop.shape[0]
            jersey = crop[int(h*0.15):int(h*0.65), :]
            jersey_pixels.append(jersey.reshape(-1, 3))
        
        all_pixels = np.vstack(jersey_pixels)
        
        # Find dominant color using KMeans
        kmeans = KMeans(n_clusters=3, n_init=5)
        kmeans.fit(all_pixels)
        
        # Use the most common cluster as team color
        counts = np.bincount(kmeans.labels_)
        dominant_idx = np.argmax(counts)
        dominant_color = kmeans.cluster_centers_[dominant_idx]
        
        team_colors[team_id] = tuple(map(int, dominant_color))
    
    return team_colors

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    config = TeamClassificationConfig(
        use_siglip=True,
        batch_size=32,
        n_teams=2
    )
    
    if config.use_siglip:
        classifier = SigLIPTeamClassifier(config)
    else:
        classifier = ColorTeamClassifier(config)
    
    print("Team Classification Module Ready!")
    print(f"\nConfiguration:")
    print(f"  - Method: {'SigLIP' if config.use_siglip else 'Color-based'}")
    print(f"  - Number of teams: {config.n_teams}")
    print(f"  - Device: {config.device}")
