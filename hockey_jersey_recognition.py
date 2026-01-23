"""
Hockey jersey recognition module.

DEPRECATED: This module is maintained for backwards compatibility.
Use jersey_recognition.py for new code.
"""

from jersey_recognition import (
    JerseyRecognitionConfig as JerseyOCRConfig,
    JerseyPrediction,
    VLMJerseyOCR as SmolVLM2JerseyOCR,
    ResNetJerseyClassifier,
    PlayerIdentity,
    PlayerIdentityManager,
    create_jersey_ocr,
)

# Re-export for backwards compatibility
__all__ = [
    'JerseyOCRConfig',
    'SmolVLM2JerseyOCR',
    'ResNetJerseyClassifier',
    'PlayerIdentityManager',
]

if __name__ == "__main__":
    config = JerseyOCRConfig(method='vlm')
    ocr = create_jersey_ocr(config)
    identity_manager = PlayerIdentityManager(config)

    roster = {
        "0": {87: "Sidney Crosby", 71: "Evgeni Malkin"},
        "1": {97: "Connor McDavid", 29: "Leon Draisaitl"}
    }
    identity_manager.roster = roster

    print("Hockey Jersey Recognition (using consolidated jersey_recognition)")
    print(f"Method: {config.method}")
