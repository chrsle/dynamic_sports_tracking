"""
Hockey player tracking module.

DEPRECATED: This module is maintained for backwards compatibility.
Use player_tracking.py for new code.
"""

from player_tracking import (
    TrackingConfig as HockeyTrackingConfig,
    TrackedObject,
    PlayerTracker as HockeyPlayerTracker,
    BallTracker as PuckTracker,
    SAM2Tracker as SAM2HockeyTracker,
    MultiObjectTracker,
    draw_tracks as draw_trails,
    draw_ball_trail as draw_puck_trail,
)

# Re-export for backwards compatibility
__all__ = [
    'HockeyTrackingConfig',
    'HockeyPlayerTracker',
    'PuckTracker',
    'SAM2HockeyTracker',
    'draw_trails',
    'draw_puck_trail',
]

if __name__ == "__main__":
    config = HockeyTrackingConfig(
        track_buffer=45,
        match_thresh=0.75,
        use_sam2=False
    )

    player_tracker = HockeyPlayerTracker(config)
    puck_tracker = PuckTracker(config)

    print("Hockey Tracking (using consolidated player_tracking)")
    print(f"Track buffer: {config.track_buffer} frames")
