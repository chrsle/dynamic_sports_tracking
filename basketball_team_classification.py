"""
Basketball team classification module.

DEPRECATED: This module is maintained for backwards compatibility.
Use team_classification.py for new code.
"""

from team_classification import (
    TeamClassificationConfig,
    SigLIPTeamClassifier as TeamClassifier,
    ColorTeamClassifier as ColorBasedTeamClassifier,
    create_team_classifier,
    get_team_colors,
    resolve_team_assignments,
    map_labels_to_colors,
)
from shared_utils import create_batches, bgr_to_pil as cv2_to_pillow

# Re-export constants for backwards compatibility
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
NUM_TEAMS = 2
UMAP_COMPONENTS = 3

# Re-export for backwards compatibility
__all__ = [
    'TeamClassifier',
    'ColorBasedTeamClassifier',
    'create_batches',
    'cv2_to_pillow',
    'get_team_colors',
    'resolve_team_assignments',
]

if __name__ == "__main__":
    classifier = TeamClassifier()
    color_classifier = ColorBasedTeamClassifier()

    print("Basketball Team Classification (using consolidated team_classification)")
    print("Classifiers initialized successfully!")
