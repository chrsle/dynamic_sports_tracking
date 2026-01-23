"""
Basketball player tracking module.

DEPRECATED: This module is maintained for backwards compatibility.
Use player_tracking.py for new code.
"""

from player_tracking import (
    TrackingConfig,
    TrackedObject as TrackedPlayer,
    PlayerTracker as ByteTrackTracker,
    SAM2Tracker,
    MultiObjectTracker,
    draw_tracks,
)

# Re-export constants for backwards compatibility
TRACK_THRESH = 0.25
TRACK_BUFFER = 30
MATCH_THRESH = 0.8
FRAME_RATE = 30
SAM2_CHECKPOINT = "facebook/sam2-hiera-large"
SAM2_POINTS_PER_SIDE = 32

# Re-export for backwards compatibility
__all__ = [
    'TrackedPlayer',
    'ByteTrackTracker',
    'SAM2Tracker',
    'MultiObjectTracker',
    'draw_tracks',
]

if __name__ == "__main__":
    tracker = MultiObjectTracker(use_sam2=False)

    print("Basketball Tracking (using consolidated player_tracking)")
    print("ByteTrack ready for fast multi-object tracking")
