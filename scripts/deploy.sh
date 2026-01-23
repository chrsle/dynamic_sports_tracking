#!/bin/bash
#
# Production Deployment Script for Hockey Analytics Dashboard
#
# Usage: ./scripts/deploy.sh [--install-deps] [--create-venv]
#
# Environment variables required:
#   API_KEY - API key for authentication
#   CORS_ALLOWED_ORIGINS - Comma-separated list of allowed origins
#
# Optional environment variables:
#   SENTRY_DSN - Sentry DSN for error monitoring
#   PORT - Server port (default: 8000)
#   WORKERS - Number of gunicorn workers (default: auto)
#   LOG_LEVEL - Logging level (default: info)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required environment variables
check_env() {
    local missing=0

    if [ -z "$API_KEY" ]; then
        log_error "API_KEY environment variable is required"
        missing=1
    fi

    if [ -z "$CORS_ALLOWED_ORIGINS" ]; then
        log_warn "CORS_ALLOWED_ORIGINS not set, using localhost defaults"
    fi

    if [ $missing -eq 1 ]; then
        echo ""
        echo "Required environment variables:"
        echo "  export API_KEY='your-secure-api-key'"
        echo "  export CORS_ALLOWED_ORIGINS='https://your-domain.com'"
        echo ""
        echo "Optional environment variables:"
        echo "  export SENTRY_DSN='https://...@sentry.io/...'"
        echo "  export PORT=8000"
        echo "  export WORKERS=4"
        echo "  export LOG_LEVEL=info"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    log_info "Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/venv"
    source "$PROJECT_DIR/venv/bin/activate"
    pip install --upgrade pip
}

# Install dependencies
install_deps() {
    log_info "Installing dependencies..."
    if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
    fi
    pip install -r "$PROJECT_DIR/requirements.txt"
}

# Create uploads directory
create_dirs() {
    log_info "Creating required directories..."
    mkdir -p "$PROJECT_DIR/dashboard/uploads"
    mkdir -p "$PROJECT_DIR/logs"
}

# Run database migrations (placeholder for future use)
run_migrations() {
    log_info "Running migrations (if any)..."
    # Add migration commands here when database is used
}

# Start the server
start_server() {
    log_info "Starting Hockey Analytics Dashboard..."
    echo ""
    echo "Server Configuration:"
    echo "  Port: ${PORT:-8000}"
    echo "  Workers: ${WORKERS:-auto}"
    echo "  Log Level: ${LOG_LEVEL:-info}"
    echo "  CORS Origins: ${CORS_ALLOWED_ORIGINS:-localhost}"
    echo "  Sentry: ${SENTRY_DSN:+enabled}"
    echo ""

    cd "$PROJECT_DIR"

    if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
    fi

    # Use gunicorn for production
    exec gunicorn \
        -c gunicorn.conf.py \
        dashboard.server:app
}

# Main
main() {
    log_info "Hockey Analytics Dashboard - Production Deployment"
    echo ""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-deps)
                INSTALL_DEPS=1
                shift
                ;;
            --create-venv)
                CREATE_VENV=1
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Check environment
    check_env

    # Create venv if requested
    if [ "$CREATE_VENV" = "1" ]; then
        create_venv
    fi

    # Install deps if requested
    if [ "$INSTALL_DEPS" = "1" ]; then
        install_deps
    fi

    # Create directories
    create_dirs

    # Run migrations
    run_migrations

    # Start server
    start_server
}

main "$@"
