"""
Basketball player detection module.

DEPRECATED: This module is maintained for backwards compatibility.
Use player_detection.py with sport='basketball' for new code.
"""

from player_detection import (
    PlayerDetector,
    DetectionConfig,
    DetectionResult,
    get_player_crops,
    associate_objects,
    BASKETBALL_CLASSES,
)
from shared_utils import compute_iou, compute_ios

# Re-export class IDs for backwards compatibility
BALL_CLASS_ID = 0
PLAYER_CLASS_ID = 1
REFEREE_CLASS_ID = 2
RIM_CLASS_ID = 3

PLAYER_DETECTION_MODEL = "basketball-players-fy4c2/1"
JERSEY_NUMBER_MODEL = "basketball-player-detection-3-ycjdo/6"

PLAYER_CONFIDENCE_THRESHOLD = 0.3
BALL_CONFIDENCE_THRESHOLD = 0.5
JERSEY_CONFIDENCE_THRESHOLD = 0.4

# Backwards compatible alias
calculate_iou = compute_iou
calculate_ios = compute_ios


class BasketballPlayerDetector(PlayerDetector):
    """Basketball player detector (backwards compatible wrapper)."""

    def __init__(
        self,
        model_id: str = PLAYER_DETECTION_MODEL,
        device: str = None,
        confidence_threshold: float = PLAYER_CONFIDENCE_THRESHOLD,
        use_roboflow: bool = True,
        api_key: str = None
    ):
        config = DetectionConfig(
            sport='basketball',
            model_type='roboflow' if use_roboflow else 'yolo',
            model_path=model_id,
            player_confidence=confidence_threshold,
            roboflow_api_key=api_key,
        )
        if device:
            config.device = device
        super().__init__(config)

    def detect_video(self, video_path: str, stride: int = 1):
        """Run detection on a video file."""
        import supervision as sv

        results = []
        frame_generator = sv.get_video_frames_generator(video_path, stride=stride)

        for frame_number, frame in enumerate(frame_generator):
            detections = self.detect(frame)
            results.append(DetectionResult(
                detections=detections,
                frame=frame,
                frame_number=frame_number * stride
            ))

        return results


class JerseyNumberDetector(PlayerDetector):
    """Jersey number region detector (backwards compatible wrapper)."""

    def __init__(
        self,
        model_id: str = JERSEY_NUMBER_MODEL,
        device: str = None,
        confidence_threshold: float = JERSEY_CONFIDENCE_THRESHOLD,
        api_key: str = None
    ):
        config = DetectionConfig(
            sport='basketball',
            model_type='roboflow',
            model_path=model_id,
            player_confidence=confidence_threshold,
            roboflow_api_key=api_key,
        )
        if device:
            config.device = device
        super().__init__(config)

    def get_jersey_crops(self, frame, detections):
        """Extract jersey number crops."""
        return get_player_crops(frame, detections, margin=0)


def associate_jerseys_with_players(
    player_detections,
    jersey_detections,
    ios_threshold: float = 0.9
):
    """Associate jersey detections with players (backwards compatible)."""
    return associate_objects(player_detections, jersey_detections, ios_threshold)


if __name__ == "__main__":
    detector = BasketballPlayerDetector(
        model_id=PLAYER_DETECTION_MODEL,
        confidence_threshold=0.3
    )

    print("Basketball Detection Module (using consolidated player_detection)")
    print(f"Config: sport={detector.config.sport}, model={detector.config.model_type}")
