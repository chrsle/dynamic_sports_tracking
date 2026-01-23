#!/bin/bash
#
# Production Server Script (without systemd)
#
# Usage: ./scripts/run_prod.sh
#
# For production with systemd, use install.sh instead.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env if it exists
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Validate required environment variables
if [ -z "$API_KEY" ]; then
    echo "ERROR: API_KEY environment variable is required"
    echo ""
    echo "Create a .env file with:"
    echo "  API_KEY=your-secure-api-key"
    echo "  CORS_ALLOWED_ORIGINS=https://your-domain.com"
    exit 1
fi

# Set production defaults
export DEBUG_MODE="${DEBUG_MODE:-false}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export PORT="${PORT:-8000}"
export WORKERS="${WORKERS:-4}"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Create required directories
mkdir -p "$PROJECT_DIR/dashboard/uploads"
mkdir -p "$PROJECT_DIR/logs"

echo "Starting production server..."
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Log Level: $LOG_LEVEL"
echo ""

exec gunicorn \
    -c "$PROJECT_DIR/gunicorn.conf.py" \
    dashboard.server:app
