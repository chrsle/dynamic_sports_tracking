"""
Soccer Analytics Dashboard Server
FastAPI backend with WebSocket support for real-time analytics
"""

import asyncio
import json
import os
import random
import math
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from soccer_analytics import (
    SoccerAnalyticsEngine, VideoQueryInterface, Position, Player,
    Shot, ShotResult, XGCalculator, PITCH_LENGTH, PITCH_WIDTH
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TeamConfig(BaseModel):
    home_name: str = "Home"
    away_name: str = "Away"
    home_color: str = "#3b82f6"
    away_color: str = "#ef4444"


class PlaybackCommand(BaseModel):
    action: str  # play, pause, seek, next, prev, speed
    value: Optional[float] = None


class QueryRequest(BaseModel):
    query_type: str  # shots, player, xg_moments, timeline
    filters: Optional[Dict[str, Any]] = None


class AnalyticsState(BaseModel):
    frame: int = 0
    time: float = 0.0
    home_team: str = "Home"
    away_team: str = "Away"
    score: Dict[str, int] = {"home": 0, "away": 0}
    xg: Dict[str, float] = {"home": 0.0, "away": 0.0}
    possession: Dict[str, float] = {"home": 50.0, "away": 50.0}
    shots: List[Dict] = []
    is_playing: bool = False
    playback_speed: float = 1.0


# ============================================================================
# APPLICATION STATE
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


class DemoDataGenerator:
    """Generate realistic demo data for the dashboard"""

    def __init__(self):
        self.frame = 0
        self.match_time = 0.0
        self.home_xg = 0.0
        self.away_xg = 0.0
        self.home_score = 0
        self.away_score = 0
        self.shots: List[Dict] = []
        self.xg_timeline: List[Dict] = []
        self.home_players: List[Dict] = []
        self.away_players: List[Dict] = []
        self.ball_position = {"x": PITCH_LENGTH / 2, "y": PITCH_WIDTH / 2}
        self.possession = {"home": 50.0, "away": 50.0}
        self.man_advantage = {"is_active": False, "home_players": 11, "away_players": 11}
        self.momentum = 0.5  # 0 = away, 1 = home

        # Initialize players
        self._init_players()

    def _init_players(self):
        """Initialize player positions for both teams"""
        # Home team (4-3-3)
        home_formation = [
            (5, 34, 1),  # GK
            (25, 10, 2), (25, 25, 4), (25, 43, 5), (25, 58, 3),  # Defense
            (45, 20, 6), (45, 34, 8), (45, 48, 10),  # Midfield
            (70, 15, 7), (75, 34, 9), (70, 53, 11),  # Attack
        ]

        self.home_players = [
            {
                "player_id": i + 1,
                "team": "home",
                "position": {"x": x, "y": y, "timestamp": 0},
                "jersey_number": num,
                "velocity": 0,
                "distance_covered": 0,
                "sprints": 0
            }
            for i, (x, y, num) in enumerate(home_formation)
        ]

        # Away team (4-4-2)
        away_formation = [
            (100, 34, 1),  # GK
            (80, 10, 2), (80, 25, 4), (80, 43, 5), (80, 58, 3),  # Defense
            (60, 8, 7), (60, 27, 8), (60, 41, 6), (60, 60, 11),  # Midfield
            (35, 25, 9), (35, 43, 10),  # Attack
        ]

        self.away_players = [
            {
                "player_id": i + 12,
                "team": "away",
                "position": {"x": x, "y": y, "timestamp": 0},
                "jersey_number": num,
                "velocity": 0,
                "distance_covered": 0,
                "sprints": 0
            }
            for i, (x, y, num) in enumerate(away_formation)
        ]

    def generate_frame(self) -> Dict:
        """Generate a single frame of demo data"""
        self.frame += 1
        self.match_time += 1 / 30  # 30 fps

        # Update ball position with momentum
        self._update_ball()

        # Update player positions
        self._update_players()

        # Maybe generate a shot
        shot = self._maybe_generate_shot()

        # Update possession
        self._update_possession()

        # Update momentum
        self._update_momentum()

        # Calculate fatigue for players
        fatigue = self._calculate_fatigue()

        # Calculate pressure metrics
        pressure = self._calculate_pressure()

        # Calculate patterns
        patterns = self._calculate_patterns()

        return {
            "type": "frame",
            "frame": self.frame,
            "time": round(self.match_time, 2),
            "match_time_formatted": self._format_time(self.match_time),
            "ball": self.ball_position,
            "home_players": self.home_players,
            "away_players": self.away_players,
            "xg": {
                "home": round(self.home_xg, 3),
                "away": round(self.away_xg, 3)
            },
            "score": {
                "home": self.home_score,
                "away": self.away_score
            },
            "possession": self.possession,
            "shots": self.shots[-20:],  # Last 20 shots
            "xg_timeline": self.xg_timeline[-100:],
            "momentum": round(self.momentum, 2),
            "man_advantage": self.man_advantage,
            "home_goalie": self._get_goalie_metrics("home"),
            "away_goalie": self._get_goalie_metrics("away"),
            "fatigue": fatigue,
            "pressure": pressure,
            "patterns": patterns,
            "new_shot": shot
        }

    def _update_ball(self):
        """Update ball position with realistic movement"""
        # Ball moves towards attacking third based on momentum
        target_x = 80 if self.momentum > 0.5 else 25
        target_y = PITCH_WIDTH / 2 + random.uniform(-20, 20)

        # Smooth movement
        self.ball_position["x"] += (target_x - self.ball_position["x"]) * 0.02 + random.uniform(-2, 2)
        self.ball_position["y"] += (target_y - self.ball_position["y"]) * 0.02 + random.uniform(-2, 2)

        # Keep in bounds
        self.ball_position["x"] = max(0, min(PITCH_LENGTH, self.ball_position["x"]))
        self.ball_position["y"] = max(0, min(PITCH_WIDTH, self.ball_position["y"]))

    def _update_players(self):
        """Update player positions based on ball and tactics"""
        ball_x = self.ball_position["x"]
        ball_y = self.ball_position["y"]

        for player in self.home_players + self.away_players:
            pos = player["position"]
            is_home = player["team"] == "home"

            # Base position attraction
            if is_home:
                base_x = pos["x"] + (ball_x - 52.5) * 0.3
            else:
                base_x = pos["x"] + (ball_x - 52.5) * 0.3

            # Random movement
            dx = random.uniform(-0.5, 0.5)
            dy = random.uniform(-0.5, 0.5)

            # Ball attraction for nearby players
            dist_to_ball = math.sqrt((pos["x"] - ball_x)**2 + (pos["y"] - ball_y)**2)
            if dist_to_ball < 20:
                dx += (ball_x - pos["x"]) * 0.05
                dy += (ball_y - pos["y"]) * 0.05

            pos["x"] += dx
            pos["y"] += dy

            # Keep in bounds (with team-specific zones)
            if is_home:
                pos["x"] = max(0, min(95, pos["x"]))
            else:
                pos["x"] = max(10, min(105, pos["x"]))
            pos["y"] = max(2, min(66, pos["y"]))

            # Update velocity
            player["velocity"] = math.sqrt(dx**2 + dy**2) * 30  # Convert to m/s
            player["distance_covered"] += player["velocity"] / 30

            # Track sprints
            if player["velocity"] > 7:
                player["sprints"] += 1

    def _maybe_generate_shot(self) -> Optional[Dict]:
        """Occasionally generate a shot"""
        if random.random() > 0.002:  # ~0.2% chance per frame
            return None

        # Determine shooting team based on ball position
        team = "home" if self.ball_position["x"] > 70 else "away" if self.ball_position["x"] < 35 else None
        if team is None:
            return None

        # Shot position near ball
        shot_x = self.ball_position["x"] + random.uniform(-5, 5)
        shot_y = self.ball_position["y"] + random.uniform(-5, 5)

        if team == "home":
            shot_x = max(70, min(100, shot_x))
        else:
            shot_x = max(5, min(35, shot_x))

        # Calculate xG
        goal_x = PITCH_LENGTH if team == "home" else 0
        goal_y = PITCH_WIDTH / 2
        distance = math.sqrt((shot_x - goal_x)**2 + (shot_y - goal_y)**2)

        # Simple xG model
        if distance < 6:
            base_xg = 0.6
        elif distance < 12:
            base_xg = 0.25
        elif distance < 20:
            base_xg = 0.1
        else:
            base_xg = 0.03

        # Angle modifier
        angle = abs(math.atan2(shot_y - goal_y, abs(shot_x - goal_x)))
        angle_factor = max(0.3, 1 - angle / math.pi)
        xg = base_xg * angle_factor

        # Determine result
        rand = random.random()
        if rand < xg * 0.8:
            result = "goal"
            if team == "home":
                self.home_score += 1
            else:
                self.away_score += 1
        elif rand < xg + 0.2:
            result = "saved"
        elif rand < xg + 0.4:
            result = "blocked"
        elif rand < xg + 0.5:
            result = "post"
        else:
            result = "off_target"

        # Update xG totals
        if team == "home":
            self.home_xg += xg
        else:
            self.away_xg += xg

        # Create shot record
        shot = {
            "shot_id": len(self.shots) + 1,
            "team": team,
            "player_id": random.choice([p["player_id"] for p in
                                       (self.home_players if team == "home" else self.away_players)]),
            "position": {"x": shot_x, "y": shot_y},
            "result": result,
            "xg": round(xg, 3),
            "distance": round(distance, 1),
            "time": self.match_time,
            "time_formatted": self._format_time(self.match_time)
        }

        self.shots.append(shot)
        self.xg_timeline.append({
            "time": self.match_time,
            "home_xg": round(self.home_xg, 3),
            "away_xg": round(self.away_xg, 3)
        })

        return shot

    def _update_possession(self):
        """Update possession based on ball position"""
        # Possession trends towards team with ball
        if self.ball_position["x"] > 52.5:
            target_home_poss = 60
        else:
            target_home_poss = 40

        current = self.possession["home"]
        new_poss = current + (target_home_poss - current) * 0.01
        self.possession["home"] = round(new_poss, 1)
        self.possession["away"] = round(100 - new_poss, 1)

    def _update_momentum(self):
        """Update match momentum"""
        # Momentum based on recent events and ball position
        ball_factor = (self.ball_position["x"] - 52.5) / 52.5 * 0.3
        random_factor = random.uniform(-0.02, 0.02)

        self.momentum += ball_factor * 0.01 + random_factor
        self.momentum = max(0, min(1, self.momentum))

    def _calculate_fatigue(self) -> Dict:
        """Calculate fatigue metrics for players"""
        fatigue = {}
        for player in self.home_players + self.away_players:
            player_id = player["player_id"]
            distance = player["distance_covered"]
            time_played = self.match_time / 60  # minutes

            # Fatigue increases with distance and time
            if time_played > 0:
                fatigue_level = min(1.0, (distance / 1000) * (time_played / 45) * 0.5)
            else:
                fatigue_level = 0

            fatigue[str(player_id)] = {
                "player_id": player_id,
                "fatigue_level": round(fatigue_level, 2),
                "distance_covered": round(distance, 0),
                "avg_speed_decline": round(fatigue_level * 20, 1),
                "sprint_decline": round(fatigue_level * 35, 1)
            }
        return fatigue

    def _calculate_pressure(self) -> Dict:
        """Calculate pressure metrics"""
        return {
            "home": {
                "team": "home",
                "ppda": round(8 + random.uniform(-2, 2), 2),
                "high_press_success_rate": round(35 + random.uniform(-10, 10), 1),
                "counter_press_intensity": round(0.7 + random.uniform(-0.2, 0.2), 2),
                "pressing_triggers": int(self.match_time / 60 * 15),
                "regains_in_final_third": int(self.match_time / 60 * 3)
            },
            "away": {
                "team": "away",
                "ppda": round(10 + random.uniform(-2, 2), 2),
                "high_press_success_rate": round(30 + random.uniform(-10, 10), 1),
                "counter_press_intensity": round(0.6 + random.uniform(-0.2, 0.2), 2),
                "pressing_triggers": int(self.match_time / 60 * 12),
                "regains_in_final_third": int(self.match_time / 60 * 2)
            }
        }

    def _calculate_patterns(self) -> Dict:
        """Calculate pattern metrics"""
        return {
            "home": {
                "team": "home",
                "build_up_speed": random.choice(["slow", "medium", "fast"]),
                "attacking_width": round(45 + random.uniform(-10, 10), 1),
                "progression_style": "short_passing",
                "third_man_runs": int(self.match_time / 60 * 8),
                "overlapping_runs": int(self.match_time / 60 * 5),
                "key_passes": int(self.match_time / 60 * 2)
            },
            "away": {
                "team": "away",
                "build_up_speed": random.choice(["medium", "fast", "direct"]),
                "attacking_width": round(40 + random.uniform(-10, 10), 1),
                "progression_style": "mixed",
                "third_man_runs": int(self.match_time / 60 * 6),
                "overlapping_runs": int(self.match_time / 60 * 4),
                "key_passes": int(self.match_time / 60 * 1.5)
            }
        }

    def _get_goalie_metrics(self, team: str) -> Dict:
        """Get goalkeeper metrics"""
        players = self.home_players if team == "home" else self.away_players
        goalie = next((p for p in players if p["jersey_number"] == 1), None)

        if not goalie:
            return {}

        pos = goalie["position"]
        ball = self.ball_position

        # Calculate positioning quality
        goal_x = 0 if team == "home" else PITCH_LENGTH
        dist_from_line = abs(pos["x"] - goal_x)

        # Coverage based on position
        coverage = 60 + dist_from_line * 2

        return {
            "player_id": goalie["player_id"],
            "team": team,
            "position": pos,
            "coverage_area": round(min(95, coverage), 1),
            "positioning_quality": round(0.7 + random.uniform(-0.1, 0.2), 2),
            "distance_from_line": round(dist_from_line, 1),
            "saves": len([s for s in self.shots if s["team"] != team and s["result"] == "saved"]),
            "goals_conceded": self.away_score if team == "home" else self.home_score
        }

    def _format_time(self, seconds: float) -> str:
        """Format time as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Soccer Analytics Dashboard", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Managers and state
manager = ConnectionManager()
demo_generator = DemoDataGenerator()
analytics_engine = SoccerAnalyticsEngine()
video_interface = VideoQueryInterface()

# Global state
app_state = {
    "is_playing": False,
    "playback_speed": 1.0,
    "team_config": TeamConfig(),
    "current_frame": 0,
    "video_loaded": False
}


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard"""
    dashboard_path = Path(__file__).parent / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return HTMLResponse("<h1>Dashboard not found</h1>")


@app.get("/api/state")
async def get_state():
    """Get current analytics state"""
    return {
        "state": app_state,
        "analytics": demo_generator.generate_frame()
    }


@app.post("/api/teams")
async def set_teams(config: TeamConfig):
    """Set team names and colors"""
    app_state["team_config"] = config
    await manager.broadcast({
        "type": "team_config",
        "config": config.model_dump()
    })
    return {"status": "ok"}


@app.post("/api/playback")
async def control_playback(command: PlaybackCommand):
    """Control video playback"""
    action = command.action
    value = command.value

    if action == "play":
        app_state["is_playing"] = True
    elif action == "pause":
        app_state["is_playing"] = False
    elif action == "toggle":
        app_state["is_playing"] = not app_state["is_playing"]
    elif action == "speed":
        if value is not None:
            app_state["playback_speed"] = max(0.1, min(8.0, value))
    elif action == "seek":
        if value is not None:
            video_interface.seek(int(value))
            app_state["current_frame"] = int(value)
    elif action == "next":
        video_interface.next_frame()
        app_state["current_frame"] = video_interface.current_frame
    elif action == "prev":
        video_interface.prev_frame()
        app_state["current_frame"] = video_interface.current_frame
    elif action == "skip":
        if value is not None:
            video_interface.skip_frames(int(value))
            app_state["current_frame"] = video_interface.current_frame

    await manager.broadcast({
        "type": "playback_state",
        "is_playing": app_state["is_playing"],
        "speed": app_state["playback_speed"],
        "frame": app_state["current_frame"]
    })

    return {"status": "ok", "state": app_state}


@app.post("/api/query")
async def query_data(request: QueryRequest):
    """Query analytics data"""
    query_type = request.query_type
    filters = request.filters or {}

    if query_type == "shots":
        return {
            "data": video_interface.query_shots(
                team=filters.get("team"),
                min_xg=filters.get("min_xg", 0)
            )
        }
    elif query_type == "player":
        player_id = filters.get("player_id")
        if player_id:
            return {"data": video_interface.query_player_events(player_id)}
    elif query_type == "xg_moments":
        threshold = filters.get("threshold", 0.3)
        return {"data": video_interface.query_high_xg_moments(threshold)}
    elif query_type == "timeline":
        return {"data": video_interface.export_timeline()}

    return {"data": []}


@app.get("/api/shots")
async def get_shots():
    """Get all shots"""
    return {"shots": demo_generator.shots}


@app.get("/api/xg-timeline")
async def get_xg_timeline():
    """Get xG timeline data"""
    return {"timeline": demo_generator.xg_timeline}


@app.get("/api/players/{team}")
async def get_players(team: str):
    """Get players for a team"""
    if team == "home":
        return {"players": demo_generator.home_players}
    elif team == "away":
        return {"players": demo_generator.away_players}
    return {"players": []}


@app.get("/api/player/{player_id}/fatigue")
async def get_player_fatigue(player_id: int):
    """Get fatigue metrics for a player"""
    fatigue = demo_generator._calculate_fatigue()
    return fatigue.get(str(player_id), {})


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video for analysis"""
    # Save file
    upload_dir = Path(__file__).parent / "uploads"
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Load into video interface
    result = video_interface.load_video(str(file_path))
    app_state["video_loaded"] = True

    return {"status": "uploaded", "path": str(file_path), **result}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    # Send initial state
    await websocket.send_json({
        "type": "init",
        "state": app_state,
        "team_config": app_state["team_config"].model_dump()
    })

    try:
        # Start sending frames
        asyncio.create_task(send_frames(websocket))

        # Listen for commands
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "playback":
                command = PlaybackCommand(**data)
                await control_playback(command)
            elif data.get("action") == "query":
                request = QueryRequest(**data)
                result = await query_data(request)
                await websocket.send_json({"type": "query_result", **result})

    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def send_frames(websocket: WebSocket):
    """Send frames at regular intervals"""
    while True:
        try:
            if app_state["is_playing"]:
                frame = demo_generator.generate_frame()
                await websocket.send_json(frame)

            # Adjust delay based on playback speed
            delay = 1 / (30 * app_state["playback_speed"])
            await asyncio.sleep(delay)

        except Exception:
            break


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
