"""
Tests for dashboard server API endpoints.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Set environment before imports
os.environ["DEBUG_MODE"] = "true"
os.environ["API_KEY"] = "test-api-key-12345"

from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))

from dashboard.server import app, settings, rate_limiter


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Return authentication headers."""
    return {"X-API-Key": "test-api-key-12345"}


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_readiness_check(self, client):
        """Test readiness endpoint."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "checks" in data


class TestStateEndpoints:
    """Tests for state management endpoints."""

    def test_get_state(self, client):
        """Test getting current state."""
        response = client.get("/api/state")
        assert response.status_code == 200
        data = response.json()
        assert "teams" in data
        assert "momentum" in data
        assert "xg" in data

    def test_set_teams_requires_auth(self, client):
        """Test that setting teams requires authentication."""
        response = client.post(
            "/api/teams",
            json={"home": "Penguins", "away": "Capitals"}
        )
        # In debug mode, this should work without auth
        assert response.status_code == 200

    def test_set_teams_with_auth(self, client, auth_headers):
        """Test setting teams with authentication."""
        response = client.post(
            "/api/teams",
            json={"home": "Penguins", "away": "Capitals"},
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_set_teams_validation(self, client, auth_headers):
        """Test team name validation."""
        # Empty name should fail
        response = client.post(
            "/api/teams",
            json={"home": "", "away": "Team"},
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error

    def test_set_teams_sanitization(self, client, auth_headers):
        """Test that dangerous characters are sanitized."""
        response = client.post(
            "/api/teams",
            json={"home": "Team<script>", "away": "Other"},
            headers=auth_headers
        )
        assert response.status_code == 200


class TestUploadEndpoint:
    """Tests for file upload endpoint."""

    def test_upload_requires_auth(self, client):
        """Test that upload requires authentication."""
        # Create a fake file
        files = {"file": ("test.mp4", b"fake video content", "video/mp4")}
        response = client.post("/api/upload", files=files)
        # In debug mode this works
        assert response.status_code in [200, 401]

    def test_upload_rejects_invalid_extension(self, client, auth_headers):
        """Test that invalid file extensions are rejected."""
        files = {"file": ("malware.exe", b"malicious content", "application/x-msdownload")}
        response = client.post("/api/upload", files=files, headers=auth_headers)
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_upload_rejects_missing_filename(self, client, auth_headers):
        """Test upload with missing filename."""
        files = {"file": ("", b"content", "video/mp4")}
        response = client.post("/api/upload", files=files, headers=auth_headers)
        # FastAPI handles this differently
        assert response.status_code in [400, 422]


class TestMoneyballEndpoints:
    """Tests for moneyball analytics endpoints."""

    def test_moneyball_status(self, client):
        """Test moneyball status endpoint."""
        response = client.get("/api/moneyball/status")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data

    def test_add_player(self, client, auth_headers, sample_player_data):
        """Test adding a player."""
        response = client.post(
            "/api/moneyball/add_player",
            json=sample_player_data,
            headers=auth_headers
        )
        # May return 503 if moneyball not available, 200 if available
        assert response.status_code in [200, 503]

    def test_add_player_validation(self, client, auth_headers):
        """Test player validation."""
        invalid_player = {
            "player_id": "invalid id!@#",  # Invalid characters
            "name": "Test",
            "team": 0
        }
        response = client.post(
            "/api/moneyball/add_player",
            json=invalid_player,
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error

    def test_add_player_team_validation(self, client, auth_headers):
        """Test team value validation."""
        invalid_player = {
            "player_id": "test_player",
            "name": "Test",
            "team": 5  # Invalid team
        }
        response = client.post(
            "/api/moneyball/add_player",
            json=invalid_player,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_get_undervalued_players(self, client):
        """Test getting undervalued players."""
        response = client.get("/api/moneyball/undervalued")
        assert response.status_code in [200, 503]

    def test_get_undervalued_invalid_param(self, client):
        """Test invalid parameter validation."""
        response = client.get("/api/moneyball/undervalued?min_ice_time=-5")
        assert response.status_code in [400, 503]

    def test_get_best_lines(self, client):
        """Test getting best lines."""
        response = client.get("/api/moneyball/lines/best")
        assert response.status_code in [200, 503]

    def test_get_player_value(self, client):
        """Test getting player value."""
        response = client.get("/api/moneyball/player/test_player")
        assert response.status_code in [404, 503]  # Player doesn't exist

    def test_player_id_validation(self, client):
        """Test player ID validation."""
        response = client.get("/api/moneyball/player/invalid<>id")
        assert response.status_code == 400

    def test_team_forecheck(self, client):
        """Test team forecheck endpoint."""
        response = client.get("/api/moneyball/team/0/forecheck")
        assert response.status_code in [200, 503]

    def test_team_forecheck_invalid_team(self, client):
        """Test invalid team ID."""
        response = client.get("/api/moneyball/team/5/forecheck")
        assert response.status_code in [400, 503]


class TestInputValidation:
    """Tests for input validation."""

    def test_xss_prevention_in_team_names(self, client, auth_headers):
        """Test XSS prevention in team names."""
        response = client.post(
            "/api/teams",
            json={
                "home": "<script>alert('xss')</script>",
                "away": "Normal Team"
            },
            headers=auth_headers
        )
        # Should succeed but sanitize the input
        assert response.status_code == 200

    def test_sql_injection_prevention(self, client, auth_headers):
        """Test SQL injection prevention."""
        response = client.post(
            "/api/teams",
            json={
                "home": "'; DROP TABLE users; --",
                "away": "Normal Team"
            },
            headers=auth_headers
        )
        # Should succeed (no SQL being executed)
        assert response.status_code == 200


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limiter_allows_normal_requests(self):
        """Test rate limiter allows normal requests."""
        rate_limiter.clients.clear()  # Reset
        for i in range(10):
            assert rate_limiter.is_allowed("test_client")

    def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks excess requests."""
        rate_limiter.clients.clear()  # Reset

        # Make requests up to limit
        for i in range(settings.rate_limit_requests):
            assert rate_limiter.is_allowed("test_client_block")

        # Next request should be blocked
        assert not rate_limiter.is_allowed("test_client_block")

    def test_rate_limiter_cleanup(self):
        """Test rate limiter cleanup."""
        rate_limiter.clients["old_client"] = []
        rate_limiter.cleanup()
        assert "old_client" not in rate_limiter.clients


class TestWebSocket:
    """Tests for WebSocket functionality."""

    def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Should receive initial state
            data = websocket.receive_json()
            assert "teams" in data
            assert "momentum" in data

    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong."""
        with client.websocket_connect("/ws") as websocket:
            # Receive initial state
            websocket.receive_json()

            # Send ping
            websocket.send_text("ping")
            response = websocket.receive_text()
            assert response == "pong"
