"""
Hockey player detection module.

DEPRECATED: This module is maintained for backwards compatibility.
Use player_detection.py with sport='hockey' for new code.
"""

from player_detection import (
    PlayerDetector,
    DetectionConfig,
    DetectionResult,
    get_player_crops,
    fine_tune_rf_detr,
    HOCKEY_CLASSES,
)
from shared_utils import compute_iou, filter_overlapping_detections

# Re-export for backwards compatibility
HOCKEY_CLASS_IDS = {v: k for k, v in HOCKEY_CLASSES.items()}


class HockeyDetectionConfig(DetectionConfig):
    """Hockey-specific detection configuration (backwards compatible)."""

    def __init__(
        self,
        use_rf_detr: bool = True,
        rf_detr_model: str = "rf-detr-medium",
        roboflow_model: str = "hockey-players-puck/1",
        player_confidence: float = 0.35,
        goalie_confidence: float = 0.40,
        referee_confidence: float = 0.35,
        puck_confidence: float = 0.25,
        nms_threshold: float = 0.5,
        device: str = None,
    ):
        super().__init__(
            sport='hockey',
            model_type='rf_detr' if use_rf_detr else 'roboflow',
            model_path=rf_detr_model if use_rf_detr else roboflow_model,
            player_confidence=player_confidence,
            goalie_confidence=goalie_confidence,
            referee_confidence=referee_confidence,
            ball_confidence=puck_confidence,
            nms_threshold=nms_threshold,
        )
        self.use_rf_detr = use_rf_detr
        self.rf_detr_model = rf_detr_model
        self.roboflow_model = roboflow_model
        self.puck_confidence = puck_confidence


class RFDETRHockeyDetector(PlayerDetector):
    """Hockey detector using RF-DETR (backwards compatible wrapper)."""

    def __init__(self, config: HockeyDetectionConfig = None):
        if config is None:
            config = HockeyDetectionConfig()
        elif not isinstance(config, DetectionConfig):
            # Convert old-style config
            config = HockeyDetectionConfig(
                use_rf_detr=getattr(config, 'use_rf_detr', True),
                rf_detr_model=getattr(config, 'rf_detr_model', 'rf-detr-medium'),
                player_confidence=getattr(config, 'player_confidence', 0.35),
                puck_confidence=getattr(config, 'puck_confidence', 0.25),
            )
        super().__init__(config)

    def detect_players_only(self, frame, include_goalies: bool = True):
        """Backwards compatible method name."""
        return self.detect_players(frame, include_goalies)

    def detect_puck(self, frame):
        """Backwards compatible method name."""
        return self.detect_ball(frame)


def fine_tune_rf_detr_hockey(
    dataset_path: str,
    model_variant: str = "rf-detr-medium",
    epochs: int = 50,
    batch_size: int = 8,
    output_dir: str = "./hockey_rf_detr",
    device: str = "cuda"
):
    """Fine-tune RF-DETR for hockey (backwards compatible)."""
    fine_tune_rf_detr(
        dataset_path=dataset_path,
        sport='hockey',
        model_variant=model_variant,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        device=device
    )


if __name__ == "__main__":
    config = HockeyDetectionConfig(
        use_rf_detr=True,
        rf_detr_model="rf-detr-medium",
        player_confidence=0.35,
        puck_confidence=0.25
    )

    detector = RFDETRHockeyDetector(config)

    print("Hockey Detection Module (using consolidated player_detection)")
    print(f"Config: sport={config.sport}, model={config.model_type}")
