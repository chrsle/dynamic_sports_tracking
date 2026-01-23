"""
Basketball jersey recognition module.

DEPRECATED: This module is maintained for backwards compatibility.
Use jersey_recognition.py for new code.
"""

from jersey_recognition import (
    JerseyRecognitionConfig,
    JerseyPrediction,
    VLMJerseyOCR as SmolVLMJerseyOCR,
    ResNetJerseyClassifier,
    PlayerIdentity,
    PlayerIdentityManager,
    create_jersey_ocr,
)

# Re-export constants for backwards compatibility
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM2-Instruct"
JERSEY_OCR_PROMPT = "What is the jersey number shown in this image? Reply with only the number, nothing else."
STABILIZATION_THRESHOLD = 3
SAMPLING_INTERVAL = 5

# Re-export for backwards compatibility
__all__ = [
    'JerseyPrediction',
    'PlayerIdentity',
    'SmolVLMJerseyOCR',
    'ResNetJerseyClassifier',
    'PlayerIdentityManager',
]

# Example roster format
EXAMPLE_ROSTER = {
    "0": {
        "23": "LeBron James",
        "3": "Anthony Davis",
        "15": "Austin Reaves",
    },
    "1": {
        "30": "Stephen Curry",
        "11": "Klay Thompson",
        "23": "Draymond Green",
    }
}

if __name__ == "__main__":
    identity_manager = PlayerIdentityManager(
        roster=EXAMPLE_ROSTER,
    )

    print("Basketball Jersey Recognition (using consolidated jersey_recognition)")
    print(f"Sample roster: {EXAMPLE_ROSTER}")
