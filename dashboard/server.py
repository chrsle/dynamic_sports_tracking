"""
Hockey Analytics Dashboard Server

FastAPI backend that serves real-time analytics to the web dashboard.
Connects to video analysis pipeline and streams updates via WebSocket.

Includes Moneyball Analytics for finding undervalued players, plays & strategies.

Security Features:
- API key authentication
- CORS restriction
- Input validation
- Rate limiting
- Secure file uploads
- Structured logging
- Request correlation IDs
- Sentry error monitoring (optional)
"""

import asyncio
import json
import logging
import os
import re
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Annotated, Dict, List, Optional

from dataclasses import dataclass, asdict, field
from pathlib import Path

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File,
    HTTPException, Depends, Security, Request, Response, status
)
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Request correlation ID context variable
correlation_id_ctx: ContextVar[str] = ContextVar('correlation_id', default='')


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record):
        record.correlation_id = correlation_id_ctx.get('')
        return True


# Configure structured logging with correlation ID
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "correlation_id": "%(correlation_id)s", "message": "%(message)s", "module": "%(module)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.addFilter(CorrelationIdFilter())

# Initialize Sentry for error monitoring (optional)
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[
                StarletteIntegration(transaction_style="endpoint"),
                FastApiIntegration(transaction_style="endpoint"),
            ],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            environment=os.getenv("ENVIRONMENT", "production"),
            release=os.getenv("APP_VERSION", "2.0.0"),
        )
        logger.info("Sentry error monitoring initialized")
    except ImportError:
        logger.warning("sentry-sdk not installed, error monitoring disabled")
else:
    logger.info("SENTRY_DSN not set, error monitoring disabled")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hockey_moneyball_analytics import MoneyballAnalytics, MoneyballConfig, NHL_SALARY_DATA
    MONEYBALL_AVAILABLE = True
except ImportError:
    MONEYBALL_AVAILABLE = False
    logger.warning("hockey_moneyball_analytics not found, moneyball features disabled")


# ==================== Configuration ====================

class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        self.api_key = os.getenv("API_KEY", self._generate_default_key())
        self.secret_key = os.getenv("API_SECRET_KEY", secrets.token_hex(32))
        self.cors_origins = self._parse_cors_origins()
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.max_upload_size = 500 * 1024 * 1024  # 500MB
        self.allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        if self.debug_mode:
            logger.warning("Running in DEBUG mode - not suitable for production!")

    def _generate_default_key(self) -> str:
        key = secrets.token_hex(32)
        logger.warning(f"No API_KEY set, using generated key. Set API_KEY environment variable for production.")
        return key

    def _parse_cors_origins(self) -> List[str]:
        origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
        if not origins_str:
            logger.warning("No CORS_ALLOWED_ORIGINS set, defaulting to localhost only")
            return ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"]
        return [origin.strip() for origin in origins_str.split(",")]


settings = Settings()


# ==================== Rate Limiting ====================

class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests: int, window: int):
        self.requests = requests
        self.window = window
        self.clients: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.clients:
            self.clients[client_id] = []

        # Remove old requests outside window
        self.clients[client_id] = [
            t for t in self.clients[client_id]
            if now - t < self.window
        ]

        if len(self.clients[client_id]) >= self.requests:
            return False

        self.clients[client_id].append(now)
        return True

    def cleanup(self):
        """Remove stale entries."""
        now = time.time()
        for client_id in list(self.clients.keys()):
            self.clients[client_id] = [
                t for t in self.clients[client_id]
                if now - t < self.window
            ]
            if not self.clients[client_id]:
                del self.clients[client_id]


rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window)


# ==================== Security ====================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key for protected endpoints."""
    if settings.debug_mode and api_key is None:
        return "debug-client"

    if api_key is None:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )

    if not secrets.compare_digest(api_key, settings.api_key):
        logger.warning("Invalid API key attempted")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return api_key


async def rate_limit_check(request: Request):
    """Check rate limit for client."""
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for client: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )


# ==================== Pydantic Models for Input Validation ====================

class TeamNamesRequest(BaseModel):
    """Request model for setting team names."""
    home: str = Field(..., min_length=1, max_length=100, description="Home team name")
    away: str = Field(..., min_length=1, max_length=100, description="Away team name")

    @field_validator('home', 'away')
    @classmethod
    def sanitize_team_name(cls, v: str) -> str:
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&"\']', '', v.strip())
        if not sanitized:
            raise ValueError("Team name cannot be empty after sanitization")
        return sanitized


class AddPlayerRequest(BaseModel):
    """Request model for adding a player."""
    player_id: str = Field(..., min_length=1, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$')
    name: str = Field(..., min_length=1, max_length=100)
    team: int = Field(..., ge=0, le=1)
    position: str = Field(default="F", pattern=r'^[FDGC]$')
    jersey_number: Optional[int] = Field(default=None, ge=1, le=99)
    salary: int = Field(default=0, ge=0, le=50_000_000)

    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        sanitized = re.sub(r'[<>&"\']', '', v.strip())
        if not sanitized:
            raise ValueError("Name cannot be empty after sanitization")
        return sanitized


class TeamMappingRequest(BaseModel):
    """Request model for team mapping."""
    team_mapping: Dict[str, int] = Field(..., description="Player ID to team mapping")

    @field_validator('team_mapping')
    @classmethod
    def validate_mapping(cls, v: Dict[str, int]) -> Dict[str, int]:
        for player_id, team in v.items():
            if not re.match(r'^[a-zA-Z0-9_-]+$', player_id):
                raise ValueError(f"Invalid player ID format: {player_id}")
            if team not in (0, 1):
                raise ValueError(f"Team must be 0 or 1, got {team}")
        return v


# ==================== Lifespan Management ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Hockey Analytics Dashboard Server")
    logger.info(f"CORS origins: {settings.cors_origins}")
    logger.info(f"Rate limiting: {settings.rate_limit_requests} requests per {settings.rate_limit_window} seconds")

    # Startup
    yield

    # Shutdown
    logger.info("Shutting down server...")
    rate_limiter.cleanup()


# ==================== FastAPI App ====================

app = FastAPI(
    title="Hockey Analytics Dashboard",
    description="Real-time hockey analytics and betting insights",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug_mode else None,  # Disable docs in production
    redoc_url="/redoc" if settings.debug_mode else None,
)

# ==================== Middleware ====================

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Add correlation ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next):
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        correlation_id_ctx.set(correlation_id)

        # Log request
        logger.info(f"Request started: {request.method} {request.url.path}")

        start_time = time.time()

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Add correlation ID and timing to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"

            logger.info(f"Request completed: {request.method} {request.url.path} - {response.status_code} ({process_time:.4f}s)")

            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request failed: {request.method} {request.url.path} - {str(e)} ({process_time:.4f}s)")
            raise


# Add correlation ID middleware first
app.add_middleware(CorrelationIdMiddleware)

# CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type", "X-Correlation-ID"],
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
    team: str
    xg: float
    result: str
    time: float
    period: int
    shooter_id: Optional[int] = None


@dataclass
class GoaliePosition:
    """Goalie position tracking."""
    x: float
    y: float
    team: str
    depth: float
    angle: float
    quality: str


@dataclass
class PowerPlayState:
    """Power play tracking."""
    is_power_play: bool
    team_with_advantage: Optional[str]
    home_players: int
    away_players: int
    time_remaining: float
    type: str


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


class DashboardState:
    """Thread-safe dashboard state management."""

    def __init__(self):
        self._lock = asyncio.Lock()
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

    async def update_teams(self, home: str, away: str):
        async with self._lock:
            self.analytics.teams = {"home": home, "away": away}

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
            "shots": [asdict(s) for s in self.analytics.shots],
            "goaliePositions": {
                k: asdict(v) for k, v in self.analytics.goalie_positions.items()
            },
            "powerPlay": asdict(self.analytics.power_play),
            "xgTimeline": [asdict(p) for p in self.analytics.xg_timeline]
        }


state = DashboardState()

# Initialize Moneyball Analytics
if MONEYBALL_AVAILABLE:
    moneyball = MoneyballAnalytics(MoneyballConfig())
else:
    moneyball = None


# ==================== WebSocket Manager ====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        """Send message to all connected clients with error handling."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# ==================== Health Check Endpoints ====================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint."""
    checks = {
        "moneyball_available": MONEYBALL_AVAILABLE,
        "websocket_manager": len(manager.active_connections),
        "processing_active": state.is_processing
    }
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }


# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML."""
    html_path = dashboard_path / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/state", dependencies=[Depends(rate_limit_check)])
async def get_state():
    """Get current analytics state."""
    return JSONResponse(content=state.to_dict())


@app.post("/api/teams", dependencies=[Depends(rate_limit_check)])
async def set_teams(
    request: TeamNamesRequest,
    api_key: str = Depends(verify_api_key)
):
    """Set team names with validation."""
    logger.info(f"Setting teams: home={request.home}, away={request.away}")
    await state.update_teams(request.home, request.away)
    await manager.broadcast(state.to_dict())
    return {"status": "ok"}


@app.post("/api/upload", dependencies=[Depends(rate_limit_check)])
async def upload_video(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Upload video for analysis with security checks."""
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        logger.warning(f"Rejected upload with invalid extension: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.allowed_extensions)}"
        )

    # Validate content type
    allowed_content_types = {
        "video/mp4", "video/avi", "video/quicktime",
        "video/x-matroska", "video/webm", "application/octet-stream"
    }
    if file.content_type and file.content_type not in allowed_content_types:
        logger.warning(f"Rejected upload with invalid content type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid content type")

    # Generate safe filename (UUID + extension)
    safe_filename = f"{uuid.uuid4()}{file_ext}"
    upload_dir = dashboard_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    upload_path = upload_dir / safe_filename

    # Read and validate file size
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size // (1024*1024)}MB"
        )

    # Write file
    try:
        with open(upload_path, "wb") as f:
            f.write(content)
        logger.info(f"Video uploaded: {safe_filename} ({len(content)} bytes)")
    except IOError as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

    state.video_path = str(upload_path)

    return {
        "status": "ok",
        "filename": safe_filename,
        "size": len(content)
    }


@app.post("/api/start", dependencies=[Depends(rate_limit_check)])
async def start_analysis(api_key: str = Depends(verify_api_key)):
    """Start video analysis."""
    if state.is_processing:
        return {"status": "error", "message": "Already processing"}

    if not state.video_path:
        return {"status": "error", "message": "No video uploaded"}

    # Validate video path exists and is within uploads directory
    video_path = Path(state.video_path)
    uploads_dir = dashboard_path / "uploads"

    try:
        video_path.resolve().relative_to(uploads_dir.resolve())
    except ValueError:
        logger.error(f"Path traversal attempt detected: {state.video_path}")
        raise HTTPException(status_code=400, detail="Invalid video path")

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    state.is_processing = True
    logger.info(f"Starting video analysis: {state.video_path}")

    # Start processing in background
    asyncio.create_task(process_video(state.video_path))

    return {"status": "ok", "message": "Analysis started"}


@app.post("/api/stop", dependencies=[Depends(rate_limit_check)])
async def stop_analysis(api_key: str = Depends(verify_api_key)):
    """Stop video analysis."""
    state.is_processing = False
    logger.info("Video analysis stopped")
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
            else:
                logger.debug(f"Received WebSocket message: {data[:100]}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ==================== Moneyball Analytics Endpoints ====================

@app.get("/api/moneyball/status", dependencies=[Depends(rate_limit_check)])
async def get_moneyball_status():
    """Check if moneyball analytics is available."""
    return JSONResponse(content={
        "available": MONEYBALL_AVAILABLE,
        "players_tracked": len(moneyball.players) if moneyball else 0,
        "lines_tracked": len(moneyball.line_combinations) if moneyball else 0
    })


@app.post("/api/moneyball/add_player", dependencies=[Depends(rate_limit_check)])
async def add_player(
    request: AddPlayerRequest,
    api_key: str = Depends(verify_api_key)
):
    """Add a player to moneyball tracking with validation."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    moneyball.add_player(
        request.player_id,
        request.name,
        request.team,
        request.position,
        request.jersey_number,
        request.salary
    )
    logger.info(f"Added player: {request.player_id} ({request.name})")
    return {"status": "ok", "player_id": request.player_id}


@app.post("/api/moneyball/load_nhl_salaries", dependencies=[Depends(rate_limit_check)])
async def load_nhl_salaries(
    request: TeamMappingRequest,
    api_key: str = Depends(verify_api_key)
):
    """Load NHL salary data for players on specified teams."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    loaded_count = 0
    for player_id, data in NHL_SALARY_DATA.items():
        if player_id in request.team_mapping:
            moneyball.add_player(
                player_id=player_id,
                name=data["name"],
                team=request.team_mapping[player_id],
                position=data["position"],
                salary=data["salary"]
            )
            loaded_count += 1

    logger.info(f"Loaded {loaded_count} player salaries")
    return {"status": "ok", "players_loaded": loaded_count}


@app.get("/api/moneyball/undervalued", dependencies=[Depends(rate_limit_check)])
async def get_undervalued_players(min_ice_time: float = 5.0):
    """Get list of undervalued players (producing more than salary suggests)."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    # Validate parameter
    if min_ice_time < 0 or min_ice_time > 100:
        raise HTTPException(status_code=400, detail="min_ice_time must be between 0 and 100")

    undervalued = moneyball.find_undervalued_players(min_ice_time)
    return JSONResponse(content={"undervalued_players": undervalued})


@app.get("/api/moneyball/overvalued", dependencies=[Depends(rate_limit_check)])
async def get_overvalued_players(min_ice_time: float = 5.0):
    """Get list of overvalued players (producing less than salary suggests)."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    if min_ice_time < 0 or min_ice_time > 100:
        raise HTTPException(status_code=400, detail="min_ice_time must be between 0 and 100")

    overvalued = moneyball.find_overvalued_players(min_ice_time)
    return JSONResponse(content={"overvalued_players": overvalued})


@app.get("/api/moneyball/player/{player_id}", dependencies=[Depends(rate_limit_check)])
async def get_player_value(player_id: str):
    """Get comprehensive value metrics for a specific player."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    # Validate player_id format
    if not re.match(r'^[a-zA-Z0-9_-]+$', player_id):
        raise HTTPException(status_code=400, detail="Invalid player ID format")

    metrics = moneyball.get_player_value_metrics(player_id)
    if not metrics:
        return JSONResponse(content={"error": "Player not found"}, status_code=404)

    return JSONResponse(content=metrics)


@app.get("/api/moneyball/player/{player_id}/shifts", dependencies=[Depends(rate_limit_check)])
async def get_player_shift_efficiency(player_id: str):
    """Get shift-by-shift efficiency analysis for a player."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    if not re.match(r'^[a-zA-Z0-9_-]+$', player_id):
        raise HTTPException(status_code=400, detail="Invalid player ID format")

    report = moneyball.get_shift_efficiency_report(player_id)
    if not report or report.get('no_data'):
        return JSONResponse(content={"error": "No shift data available"}, status_code=404)

    return JSONResponse(content=report)


@app.get("/api/moneyball/lines/best", dependencies=[Depends(rate_limit_check)])
async def get_best_lines(min_ice_time: float = 2.0, top_n: int = 10):
    """Get best performing line combinations by xG differential."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    if min_ice_time < 0 or min_ice_time > 100:
        raise HTTPException(status_code=400, detail="min_ice_time must be between 0 and 100")
    if top_n < 1 or top_n > 100:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 100")

    lines = moneyball.get_best_line_combinations(min_ice_time, top_n)
    return JSONResponse(content={"best_lines": lines})


@app.get("/api/moneyball/lines/value", dependencies=[Depends(rate_limit_check)])
async def get_best_value_lines(min_ice_time: float = 2.0, top_n: int = 10):
    """Get line combinations with best value/salary ratio."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    if min_ice_time < 0 or min_ice_time > 100:
        raise HTTPException(status_code=400, detail="min_ice_time must be between 0 and 100")
    if top_n < 1 or top_n > 100:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 100")

    lines = moneyball.get_best_value_lines(min_ice_time, top_n)
    return JSONResponse(content={"best_value_lines": lines})


@app.get("/api/moneyball/team/{team_id}/forecheck", dependencies=[Depends(rate_limit_check)])
async def get_team_forecheck(team_id: int):
    """Get forechecking pattern analysis for a team."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    if team_id not in (0, 1):
        raise HTTPException(status_code=400, detail="team_id must be 0 or 1")

    analysis = moneyball.get_forecheck_analysis(team_id)
    return JSONResponse(content=analysis)


@app.get("/api/moneyball/team/{team_id}/zone_entries", dependencies=[Depends(rate_limit_check)])
async def get_team_zone_entries(team_id: int):
    """Get zone entry effectiveness analysis for a team."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    if team_id not in (0, 1):
        raise HTTPException(status_code=400, detail="team_id must be 0 or 1")

    analysis = moneyball.get_zone_entry_analysis(team_id)
    return JSONResponse(content=analysis)


@app.get("/api/moneyball/report", dependencies=[Depends(rate_limit_check)])
async def get_full_moneyball_report(api_key: str = Depends(verify_api_key)):
    """Get comprehensive moneyball analytics report."""
    if not moneyball:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    report = moneyball.get_moneyball_report()
    return JSONResponse(content=report)


@app.get("/api/moneyball/salaries", dependencies=[Depends(rate_limit_check)])
async def get_nhl_salary_data():
    """Get available NHL salary data."""
    if not MONEYBALL_AVAILABLE:
        return JSONResponse(content={"error": "Moneyball analytics not available"}, status_code=503)

    return JSONResponse(content={"salary_data": NHL_SALARY_DATA})


# ==================== Video Processing ====================

async def process_video(video_path: str):
    """
    Process video and stream analytics to dashboard.
    """
    import cv2

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            state.is_processing = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / fps
        frame_count = 0

        logger.info(f"Processing video at {fps} FPS")

        while state.is_processing:
            ret, frame = cap.read()

            if not ret:
                logger.info("Video processing completed")
                break

            frame_count += 1
            state.analytics.frame = frame_count

            # Process frame
            analytics = await process_frame_analytics(frame, frame_count)

            # Update state
            update_state_from_analytics(analytics)

            # Broadcast to connected clients
            await manager.broadcast(state.to_dict())

            # Control frame rate
            await asyncio.sleep(frame_delay)

        cap.release()

    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)

    finally:
        state.is_processing = False
        logger.info("Video processing stopped")


async def process_frame_analytics(frame, frame_number: int) -> Dict:
    """Process a single frame and return analytics."""
    import random
    import math

    # Simulated analytics - replace with actual pipeline
    momentum = (random.random() - 0.5) * 2
    game_time = frame_number / 30

    home_xg = frame_number * 0.001 * (1 + momentum)
    away_xg = frame_number * 0.001 * (1 - momentum)

    new_shot = None
    if random.random() > 0.98:
        shot_team = "home" if random.random() > 0.45 else "away"
        if shot_team == "home":
            shot_x = 50 + random.random() * 40
            shot_y = (random.random() - 0.5) * 60
        else:
            shot_x = -50 - random.random() * 40
            shot_y = (random.random() - 0.5) * 60

        goal_x = 89 if shot_team == "home" else -89
        dist = math.sqrt((shot_x - goal_x)**2 + shot_y**2)
        shot_xg = max(0.02, min(0.5, 0.4 - dist * 0.008))

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

    is_pp = random.random() > 0.92
    pp_state = {
        "is_power_play": is_pp,
        "team_with_advantage": random.choice(["home", "away"]) if is_pp else None,
        "home_players": 5 if not is_pp or random.random() > 0.5 else 4,
        "away_players": 5 if not is_pp or random.random() > 0.5 else 4,
        "time_remaining": random.uniform(30, 120) if is_pp else 0,
        "type": "5v4" if is_pp else "even"
    }
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
            "reasons": [f"Processing frame {frame_number}"]
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
        if len(state.analytics.shots) > 100:
            state.analytics.shots = state.analytics.shots[-100:]

        if moneyball and shot_data.get("shooter_id"):
            moneyball.track_scoring_chance(
                player_id=shot_data["shooter_id"],
                xg=shot_data["xg"],
                frame=state.analytics.frame,
                resulted_in_goal=(shot_data["result"] == "goal")
            )

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
        if len(state.analytics.xg_timeline) > 500:
            state.analytics.xg_timeline = state.analytics.xg_timeline[-500:]

    if moneyball:
        moneyball.update_frame(state.analytics.frame)

    if moneyball and "zone_entry" in analytics:
        entry = analytics["zone_entry"]
        moneyball.track_zone_entry(
            player_id=entry["player_id"],
            entry_type=entry["type"],
            success=entry["success"],
            frame=state.analytics.frame,
            resulted_in_shot=entry.get("resulted_in_shot", False),
            xg_generated=entry.get("xg_generated", 0)
        )


# ==================== Run Server ====================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the dashboard server."""
    moneyball_status = "ENABLED" if MONEYBALL_AVAILABLE else "DISABLED"
    logger.info(f"""
    Hockey Analytics Dashboard Server v2.0
    ======================================
    Dashboard:    http://{host}:{port}
    Health:       http://{host}:{port}/health
    API Docs:     http://{host}:{port}/docs (debug mode only)
    WebSocket:    ws://{host}:{port}/ws

    Security:
    - API Key Auth: ENABLED
    - CORS Origins: {settings.cors_origins}
    - Rate Limiting: {settings.rate_limit_requests} req/{settings.rate_limit_window}s

    Moneyball Analytics: {moneyball_status}
    """)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
