#!/bin/bash
#
# Development Server Script
#
# Usage: ./scripts/run_dev.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env if it exists
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Set development defaults
export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_LEVEL="${LOG_LEVEL:-debug}"
export PORT="${PORT:-8000}"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

echo "Starting development server..."
echo "  Port: $PORT"
echo "  Debug Mode: $DEBUG_MODE"
echo "  API Docs: http://localhost:$PORT/docs"
echo ""

python -m uvicorn dashboard.server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --reload \
    --log-level "${LOG_LEVEL:-info}"
