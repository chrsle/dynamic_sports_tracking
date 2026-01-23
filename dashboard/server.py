"""
Hockey Analytics Dashboard Server

FastAPI backend that serves real-time analytics to the web dashboard.
Connects to video analysis pipeline and streams updates via WebSocket.
"""

import asyncio
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Hockey Analytics Dashboard",
    description="Real-time hockey analytics and betting insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (dashboard)
dashboard_path = Path(__file__).parent
app.mount("/static", StaticFiles(directory=dashboard_path), name="static")


# ==================== Data Models ====================

@dataclass
class BettingSignal:
    text: str
    confidence: float
    reasons: List[str]


@dataclass
class Momentum:
    value: float
    short: float
    medium: float
    long: float
    trend: str


@dataclass
class ExpectedGoals:
    home: float
    away: float


@dataclass
class ZoneTime:
    offensive: float
    neutral: float
    defensive: float


@dataclass
class PositionQuality:
    optimal: int
    good: int
    suboptimal: int
    critical: int


@dataclass
class Shot:
    """Individual shot data."""
    x: float
    y: float
    team: str  # 'home' or 'away'
    xg: float
    result: str  # 'goal', 'save', 'miss', 'block'
    time: float  # game time in seconds
    period: int
    shooter_id: Optional[int] = None


@dataclass
class GoaliePosition:
    """Goalie position tracking."""
    x: float
    y: float
    team: str
    depth: float  # distance from goal line
    angle: float  # angle coverage
    quality: str  # 'optimal', 'good', 'vulnerable', 'out_of_position'


@dataclass
class PowerPlayState:
    """Power play tracking."""
    is_power_play: bool
    team_with_advantage: Optional[str]  # 'home', 'away', or None
    home_players: int
    away_players: int
    time_remaining: float  # seconds remaining in PP
    type: str  # '5v4', '5v3', '4v3', 'even'


@dataclass
class XGTimelinePoint:
    """Point in xG timeline."""
    time: float
    home_xg: float
    away_xg: float


@dataclass
class AnalyticsState:
    """Current state of all analytics."""
    teams: Dict[str, str]
    signal: BettingSignal
    momentum: Momentum
    xg: ExpectedGoals
    zones: Dict[str, ZoneTime]
    positions: PositionQuality
    insights: List[Dict]
    frame: int = 0
    # New fields for enhanced analytics
    shots: List[Shot] = field(default_factory=list)
    goalie_positions: Dict[str, GoaliePosition] = field(default_factory=dict)
    power_play: PowerPlayState = field(default_factory=lambda: PowerPlayState(
        is_power_play=False,
        team_with_advantage=None,
        home_players=5,
        away_players=5,
        time_remaining=0,
        type='even'
    ))
    xg_timeline: List[XGTimelinePoint] = field(default_factory=list)


# Global state
class DashboardState:
    def __init__(self):
        self.analytics = AnalyticsState(
            teams={"home": "Home", "away": "Away"},
            signal=BettingSignal("NEUTRAL", 0.5, []),
            momentum=Momentum(0, 0, 0, 0, "stable"),
            xg=ExpectedGoals(0, 0),
            zones={
                "home": ZoneTime(33, 34, 33),
                "away": ZoneTime(33, 34, 33)
            },
            positions=PositionQuality(0, 0, 0, 0),
            insights=[],
            shots=[],
            goalie_positions={
                "home": GoaliePosition(-85, 0, "home", 5, 30, "optimal"),
                "away": GoaliePosition(85, 0, "away", 5, 30, "optimal")
            },
            power_play=PowerPlayState(False, None, 5, 5, 0, "even"),
            xg_timeline=[]
        )
        self.connections: List[WebSocket] = []
        self.is_processing = False
        self.video_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert state to dictionary for JSON serialization."""
        return {
            "teams": self.analytics.teams,
            "signal": asdict(self.analytics.signal),
            "momentum": asdict(self.analytics.momentum),
            "xg": asdict(self.analytics.xg),
            "zones": {
                k: asdict(v) for k, v in self.analytics.zones.items()
            },
            "positions": asdict(self.analytics.positions),
            "insights": self.analytics.insights,
            "frame": self.analytics.frame,
            # New fields
            "shots": [asdict(s) for s in self.analytics.shots],
            "goaliePositions": {
                k: asdict(v) for k, v in self.analytics.goalie_positions.items()
            },
            "powerPlay": asdict(self.analytics.power_play),
            "xgTimeline": [asdict(p) for p in self.analytics.xg_timeline]
        }


state = DashboardState()


# ==================== WebSocket Manager ====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        """Send message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML."""
    html_path = dashboard_path / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/state")
async def get_state():
    """Get current analytics state."""
    return JSONResponse(content=state.to_dict())


@app.post("/api/teams")
async def set_teams(home: str, away: str):
    """Set team names."""
    state.analytics.teams = {"home": home, "away": away}
    await manager.broadcast(state.to_dict())
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video for analysis."""
    # Save uploaded file
    upload_path = dashboard_path / "uploads" / file.filename
    upload_path.parent.mkdir(exist_ok=True)

    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    state.video_path = str(upload_path)

    return {
        "status": "ok",
        "filename": file.filename,
        "path": str(upload_path)
    }


@app.post("/api/start")
async def start_analysis():
    """Start video analysis."""
    if state.is_processing:
        return {"status": "error", "message": "Already processing"}

    if not state.video_path:
        return {"status": "error", "message": "No video uploaded"}

    state.is_processing = True

    # Start processing in background
    asyncio.create_task(process_video(state.video_path))

    return {"status": "ok", "message": "Analysis started"}


@app.post("/api/stop")
async def stop_analysis():
    """Stop video analysis."""
    state.is_processing = False
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        # Send initial state
        await websocket.send_json(state.to_dict())

        while True:
            # Wait for messages from client
            data = await websocket.receive_text()

            # Handle client commands
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ==================== Video Processing ====================

async def process_video(video_path: str):
    """
    Process video and stream analytics to dashboard.

    This function integrates with the hockey analysis pipeline
    and streams updates via WebSocket.
    """
    import cv2
    import sys

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        # Import hockey analysis modules
        # Note: These would be imported from the notebooks
        # For now, we simulate the processing

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            state.is_processing = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / fps

        frame_count = 0

        while state.is_processing:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            state.analytics.frame = frame_count

            # Process frame (simulated - replace with actual pipeline)
            analytics = await process_frame_analytics(frame, frame_count)

            # Update state
            update_state_from_analytics(analytics)

            # Broadcast to connected clients
            await manager.broadcast(state.to_dict())

            # Control frame rate
            await asyncio.sleep(frame_delay)

        cap.release()

    except Exception as e:
        print(f"Error processing video: {e}")

    finally:
        state.is_processing = False


async def process_frame_analytics(frame, frame_number: int) -> Dict:
    """
    Process a single frame and return analytics.

    Replace this with actual pipeline integration:

    ```python
    from hockey_analysis_pipeline import HockeyAnalysisPipeline
    from hockey_rink_keypoints import RinkKeypointDetector
    from hockey_betting_analytics import HockeyBettingDashboard

    # Initialize once
    pipeline = HockeyAnalysisPipeline(config)
    keypoint_detector = RinkKeypointDetector()
    betting = HockeyBettingDashboard()

    # Process frame
    result = pipeline.process_frame(frame)
    keypoints, _ = keypoint_detector.detect_keypoints(frame)
    H = keypoint_detector.compute_homography(keypoints)

    # Transform positions
    if H is not None:
        positions = keypoint_detector.transform_points(
            result.tracked_players.get_anchors_coordinates(...),
            H
        )

    # Get betting signal
    signal = betting.process_frame(team_positions, puck_position)

    return signal
    ```
    """
    import random
    import math

    # Simulated analytics - replace with actual pipeline
    momentum = (random.random() - 0.5) * 2
    game_time = frame_number / 30  # Approximate seconds

    # Calculate cumulative xG
    home_xg = frame_number * 0.001 * (1 + momentum)
    away_xg = frame_number * 0.001 * (1 - momentum)

    # Generate shot events (occasionally)
    new_shot = None
    if random.random() > 0.98:  # ~2% chance per frame
        shot_team = "home" if random.random() > 0.45 else "away"
        # Shot position in offensive zone
        if shot_team == "home":
            shot_x = 50 + random.random() * 40  # Right side (attacking)
            shot_y = (random.random() - 0.5) * 60
        else:
            shot_x = -50 - random.random() * 40  # Left side (attacking)
            shot_y = (random.random() - 0.5) * 60

        # Calculate xG based on distance to goal
        goal_x = 89 if shot_team == "home" else -89
        dist = math.sqrt((shot_x - goal_x)**2 + shot_y**2)
        shot_xg = max(0.02, min(0.5, 0.4 - dist * 0.008))

        # Determine result
        rand = random.random()
        if rand < shot_xg:
            result = "goal"
        elif rand < shot_xg + 0.6:
            result = "save"
        elif rand < shot_xg + 0.75:
            result = "miss"
        else:
            result = "block"

        new_shot = {
            "x": shot_x,
            "y": shot_y,
            "team": shot_team,
            "xg": shot_xg,
            "result": result,
            "time": game_time,
            "period": min(3, int(game_time / 1200) + 1)
        }

    # Goalie positions (simulated movement)
    goalie_home = {
        "x": -85 + random.uniform(-3, 3),
        "y": random.uniform(-8, 8),
        "team": "home",
        "depth": 5 + random.uniform(-2, 5),
        "angle": 28 + random.uniform(-5, 5),
        "quality": random.choice(["optimal", "optimal", "good", "good", "vulnerable"])
    }
    goalie_away = {
        "x": 85 + random.uniform(-3, 3),
        "y": random.uniform(-8, 8),
        "team": "away",
        "depth": 5 + random.uniform(-2, 5),
        "angle": 28 + random.uniform(-5, 5),
        "quality": random.choice(["optimal", "optimal", "good", "good", "vulnerable"])
    }

    # Power play state (simulate occasional power plays)
    is_pp = random.random() > 0.92
    pp_state = {
        "is_power_play": is_pp,
        "team_with_advantage": random.choice(["home", "away"]) if is_pp else None,
        "home_players": 5 if not is_pp or random.random() > 0.5 else 4,
        "away_players": 5 if not is_pp or random.random() > 0.5 else 4,
        "time_remaining": random.uniform(30, 120) if is_pp else 0,
        "type": "5v4" if is_pp else "even"
    }
    # Fix player counts based on advantage
    if is_pp:
        if pp_state["team_with_advantage"] == "home":
            pp_state["home_players"] = 5
            pp_state["away_players"] = 4
        else:
            pp_state["home_players"] = 4
            pp_state["away_players"] = 5
        pp_state["type"] = f"{pp_state['home_players']}v{pp_state['away_players']}"

    return {
        "signal": {
            "text": "LEAN_HOME" if momentum > 0.2 else "LEAN_AWAY" if momentum < -0.2 else "NEUTRAL",
            "confidence": 0.5 + abs(momentum) * 0.4,
            "reasons": ["Processing frame " + str(frame_number)]
        },
        "momentum": {
            "value": momentum,
            "short": momentum,
            "medium": momentum * 0.8,
            "long": momentum * 0.6,
            "trend": "increasing_home" if momentum > 0.3 else "increasing_away" if momentum < -0.3 else "stable"
        },
        "xg": {
            "home": home_xg,
            "away": away_xg
        },
        "positions": {
            "optimal": random.randint(2, 4),
            "good": random.randint(3, 5),
            "suboptimal": random.randint(1, 3),
            "critical": random.randint(0, 2)
        },
        # New analytics fields
        "new_shot": new_shot,
        "goalie_positions": {
            "home": goalie_home,
            "away": goalie_away
        },
        "power_play": pp_state,
        "xg_timeline_point": {
            "time": game_time,
            "home_xg": home_xg,
            "away_xg": away_xg
        }
    }


def update_state_from_analytics(analytics: Dict):
    """Update global state from analytics results."""
    if "signal" in analytics:
        state.analytics.signal = BettingSignal(**analytics["signal"])

    if "momentum" in analytics:
        state.analytics.momentum = Momentum(**analytics["momentum"])

    if "xg" in analytics:
        state.analytics.xg = ExpectedGoals(**analytics["xg"])

    if "positions" in analytics:
        state.analytics.positions = PositionQuality(**analytics["positions"])

    # New analytics fields
    if "new_shot" in analytics and analytics["new_shot"] is not None:
        shot_data = analytics["new_shot"]
        new_shot = Shot(
            x=shot_data["x"],
            y=shot_data["y"],
            team=shot_data["team"],
            xg=shot_data["xg"],
            result=shot_data["result"],
            time=shot_data["time"],
            period=shot_data["period"]
        )
        state.analytics.shots.append(new_shot)
        # Keep last 100 shots
        if len(state.analytics.shots) > 100:
            state.analytics.shots = state.analytics.shots[-100:]

    if "goalie_positions" in analytics:
        for team, pos in analytics["goalie_positions"].items():
            state.analytics.goalie_positions[team] = GoaliePosition(**pos)

    if "power_play" in analytics:
        state.analytics.power_play = PowerPlayState(**analytics["power_play"])

    if "xg_timeline_point" in analytics:
        point = analytics["xg_timeline_point"]
        state.analytics.xg_timeline.append(XGTimelinePoint(
            time=point["time"],
            home_xg=point["home_xg"],
            away_xg=point["away_xg"]
        ))
        # Keep last 500 points (for timeline chart)
        if len(state.analytics.xg_timeline) > 500:
            state.analytics.xg_timeline = state.analytics.xg_timeline[-500:]


# ==================== Run Server ====================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the dashboard server."""
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║           Hockey Analytics Dashboard Server               ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Dashboard:  http://{host}:{port}                          ║
    ║  API Docs:   http://{host}:{port}/docs                     ║
    ║  WebSocket:  ws://{host}:{port}/ws                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
