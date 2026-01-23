"""
Hockey Analytics Dashboard Server

FastAPI backend that serves real-time analytics to the web dashboard.
Connects to video analysis pipeline and streams updates via WebSocket.
"""

import asyncio
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
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
            insights=[]
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
            "frame": self.analytics.frame
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

    # Simulated analytics - replace with actual pipeline
    momentum = (random.random() - 0.5) * 2

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
            "home": frame_number * 0.001 * (1 + momentum),
            "away": frame_number * 0.001 * (1 - momentum)
        },
        "positions": {
            "optimal": random.randint(2, 4),
            "good": random.randint(3, 5),
            "suboptimal": random.randint(1, 3),
            "critical": random.randint(0, 2)
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
