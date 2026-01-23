"""
Hockey team classification module.

DEPRECATED: This module is maintained for backwards compatibility.
Use team_classification.py for new code.
"""

from team_classification import (
    TeamClassificationConfig,
    SigLIPTeamClassifier,
    ColorTeamClassifier,
    create_team_classifier,
    get_team_colors,
    resolve_team_assignments,
    map_labels_to_colors,
)

# Re-export for backwards compatibility
__all__ = [
    'TeamClassificationConfig',
    'SigLIPTeamClassifier',
    'ColorTeamClassifier',
    'get_team_colors',
]

if __name__ == "__main__":
    config = TeamClassificationConfig(method='siglip')
    classifier = create_team_classifier(config)

    print("Hockey Team Classification (using consolidated team_classification)")
    print(f"Method: {config.method}")
