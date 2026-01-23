"""
Pytest configuration and fixtures.
"""
import os
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables
os.environ["DEBUG_MODE"] = "true"
os.environ["API_KEY"] = "test-api-key-12345"
os.environ["CORS_ALLOWED_ORIGINS"] = "http://localhost:3000,http://localhost:8000"


@pytest.fixture
def api_key():
    """Return test API key."""
    return "test-api-key-12345"


@pytest.fixture
def sample_player_data():
    """Sample player data for testing."""
    return {
        "player_id": "test_player_1",
        "name": "Test Player",
        "team": 0,
        "position": "F",
        "jersey_number": 99,
        "salary": 5_000_000
    }


@pytest.fixture
def sample_team_names():
    """Sample team names for testing."""
    return {
        "home": "Home Team",
        "away": "Away Team"
    }
