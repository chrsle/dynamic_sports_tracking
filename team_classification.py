"""
Unified team classification module for sports tracking.

Consolidates SigLIP-based and color-based team classification
approaches that were duplicated across hockey and basketball modules.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
from sklearn.cluster import KMeans
import supervision as sv

from shared_utils import (
    BaseClassifier, BaseConfig, get_device, bgr_to_pil,
    create_batches, extract_jersey_region, compute_color_histogram
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TeamClassificationConfig(BaseConfig):
    """Configuration for team classification."""

    # Method selection
    method: str = 'siglip'  # 'siglip' or 'color'

    # SigLIP settings
    siglip_model: str = 'google/siglip-base-patch16-224'
    batch_size: int = 32

    # UMAP settings
    umap_components: int = 3
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1

    # Clustering settings
    n_teams: int = 2
    kmeans_init: int = 10

    # Color-based settings
    jersey_crop_top: float = 0.15
    jersey_crop_bottom: float = 0.65
    hist_bins_h: int = 30
    hist_bins_s: int = 32


# =============================================================================
# SigLIP-based Team Classifier
# =============================================================================

class SigLIPTeamClassifier(BaseClassifier):
    """
    Team classifier using Google's SigLIP vision model.

    Uses SigLIP embeddings + UMAP dimensionality reduction + KMeans clustering
    to classify players into teams based on visual appearance.
    """

    def __init__(self, config: Optional[TeamClassificationConfig] = None):
        super().__init__()
        self.config = config or TeamClassificationConfig(method='siglip')
        self._model = None
        self._processor = None
        self._reducer = None
        self._cluster_model = None

    def _load_model(self):
        """Load SigLIP model and initialize clustering."""
        if self._model is not None:
            return

        from transformers import AutoProcessor, SiglipVisionModel
        import umap

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

        print(f"Loaded SigLIP on {self.config.device}")

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract SigLIP embeddings from player crops.

        Args:
            crops: List of player image crops (BGR)

        Returns:
            Feature embeddings array
        """
        self._load_model()

        pil_crops = [bgr_to_pil(c) for c in crops]
        embeddings = []

        with torch.no_grad():
            for batch in create_batches(pil_crops, self.config.batch_size):
                inputs = self._processor(
                    images=batch,
                    return_tensors="pt"
                ).to(self.config.device)

                outputs = self._model(**inputs)
                emb = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
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

        features = self.extract_features(crops)
        projections = self._reducer.fit_transform(features)
        self._cluster_model.fit(projections)
        self._fitted = True

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict team labels for player crops.

        Args:
            crops: List of player image crops

        Returns:
            Array of team labels (0, 1, etc.)
        """
        if not self._fitted:
            return np.zeros(len(crops), dtype=int)

        if len(crops) == 0:
            return np.array([], dtype=int)

        features = self.extract_features(crops)
        projections = self._reducer.transform(features)
        return self._cluster_model.predict(projections)


# =============================================================================
# Color-based Team Classifier
# =============================================================================

class ColorTeamClassifier(BaseClassifier):
    """
    Team classifier using color histograms.

    Faster and simpler than SigLIP, works well when teams
    have distinctly different jersey colors.
    """

    def __init__(self, config: Optional[TeamClassificationConfig] = None):
        super().__init__()
        self.config = config or TeamClassificationConfig(method='color')
        self._cluster_model = KMeans(
            n_clusters=self.config.n_teams,
            n_init=self.config.kmeans_init
        )

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
            jersey = extract_jersey_region(
                crop,
                self.config.jersey_crop_top,
                self.config.jersey_crop_bottom
            )

            hist = compute_color_histogram(
                jersey,
                self.config.hist_bins_h,
                self.config.hist_bins_s
            )
            features.append(hist)

        return np.array(features)

    def fit(self, crops: List[np.ndarray]):
        """Fit the classifier on player crops."""
        if len(crops) < self.config.n_teams:
            print(f"Warning: Need at least {self.config.n_teams} crops to fit")
            return

        features = self.extract_features(crops)
        self._cluster_model.fit(features)
        self._fitted = True

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Predict team labels for player crops."""
        if not self._fitted:
            return np.zeros(len(crops), dtype=int)

        if len(crops) == 0:
            return np.array([], dtype=int)

        features = self.extract_features(crops)
        return self._cluster_model.predict(features)


# =============================================================================
# Factory Function
# =============================================================================

def create_team_classifier(
    config: Optional[TeamClassificationConfig] = None
) -> BaseClassifier:
    """
    Create a team classifier based on configuration.

    Args:
        config: Classification configuration

    Returns:
        Team classifier instance
    """
    config = config or TeamClassificationConfig()

    if config.method == 'siglip':
        return SigLIPTeamClassifier(config)
    else:
        return ColorTeamClassifier(config)


# =============================================================================
# Utility Functions
# =============================================================================

def get_team_colors(
    crops: List[np.ndarray],
    team_labels: np.ndarray,
    n_teams: int = 2
) -> Dict[int, Tuple[int, int, int]]:
    """
    Extract dominant color for each team.

    Args:
        crops: Player image crops
        team_labels: Team assignments
        n_teams: Number of teams

    Returns:
        Dictionary mapping team ID to BGR color
    """
    team_colors = {}

    for team_id in range(n_teams):
        team_crops = [c for c, l in zip(crops, team_labels) if l == team_id]

        if not team_crops:
            team_colors[team_id] = (128, 128, 128)
            continue

        jersey_pixels = []
        for crop in team_crops:
            jersey = extract_jersey_region(crop)
            jersey_pixels.append(jersey.reshape(-1, 3))

        all_pixels = np.vstack(jersey_pixels)

        kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
        kmeans.fit(all_pixels)

        counts = np.bincount(kmeans.labels_)
        dominant_idx = np.argmax(counts)
        dominant_color = kmeans.cluster_centers_[dominant_idx]

        team_colors[team_id] = tuple(map(int, dominant_color))

    return team_colors


def resolve_team_assignments(
    detections: sv.Detections,
    team_labels: np.ndarray,
    referee_class_id: int = 2
) -> sv.Detections:
    """
    Add team assignments to detections.

    Args:
        detections: Detection results
        team_labels: Array of team labels
        referee_class_id: Class ID for referees (excluded)

    Returns:
        Detections with team_id in data dictionary
    """
    team_ids = np.full(len(detections), -1, dtype=int)

    player_mask = detections.class_id != referee_class_id
    player_indices = np.where(player_mask)[0]

    for idx, label in zip(player_indices[:len(team_labels)], team_labels):
        team_ids[idx] = label

    if detections.data is None:
        detections.data = {}

    detections.data['team_id'] = team_ids

    return detections


def map_labels_to_colors(
    team_labels: np.ndarray,
    team_0_color: Tuple[int, int, int] = (255, 0, 0),
    team_1_color: Tuple[int, int, int] = (0, 0, 255),
    unknown_color: Tuple[int, int, int] = (128, 128, 128)
) -> List[Tuple[int, int, int]]:
    """
    Map team labels to visualization colors.

    Args:
        team_labels: Array of team labels
        team_0_color: Color for team 0 (BGR)
        team_1_color: Color for team 1 (BGR)
        unknown_color: Color for unknown/referee (BGR)

    Returns:
        List of colors for each label
    """
    color_map = {
        0: team_0_color,
        1: team_1_color,
        -1: unknown_color
    }

    return [color_map.get(int(label), unknown_color) for label in team_labels]


if __name__ == "__main__":
    # Example: SigLIP-based classifier
    siglip_config = TeamClassificationConfig(method='siglip')
    siglip_classifier = create_team_classifier(siglip_config)

    # Example: Color-based classifier
    color_config = TeamClassificationConfig(method='color')
    color_classifier = create_team_classifier(color_config)

    print("Team Classification Module Ready!")
    print(f"Available methods: siglip, color")
