#!/bin/bash
#
# Production Installation Script for Hockey Analytics Dashboard
#
# This script installs the application as a systemd service.
# Run with sudo on a production server.
#
# Usage: sudo ./scripts/install.sh
#

set -e

# Configuration
INSTALL_DIR="/opt/hockey-analytics"
SERVICE_USER="www-data"
SERVICE_GROUP="www-data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run as root (sudo ./scripts/install.sh)"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check Python 3.10+
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Python version: $PYTHON_VERSION"

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."

    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y \
            python3-venv \
            python3-dev \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev
    elif command -v yum &> /dev/null; then
        yum install -y \
            python3-devel \
            mesa-libGL \
            glib2
    else
        log_warn "Unknown package manager, skipping system dependencies"
    fi
}

# Create installation directory
create_install_dir() {
    log_info "Creating installation directory: $INSTALL_DIR"

    mkdir -p "$INSTALL_DIR"
    mkdir -p "$INSTALL_DIR/dashboard/uploads"
    mkdir -p "$INSTALL_DIR/logs"
}

# Copy application files
copy_files() {
    log_info "Copying application files..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

    # Copy all files except venv, __pycache__, etc.
    rsync -av --exclude='venv' \
              --exclude='__pycache__' \
              --exclude='*.pyc' \
              --exclude='.git' \
              --exclude='.ipynb_checkpoints' \
              --exclude='dashboard/uploads/*' \
              --exclude='logs/*' \
              "$PROJECT_DIR/" "$INSTALL_DIR/"
}

# Create virtual environment and install dependencies
setup_venv() {
    log_info "Creating virtual environment..."

    python3 -m venv "$INSTALL_DIR/venv"
    source "$INSTALL_DIR/venv/bin/activate"

    log_info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r "$INSTALL_DIR/requirements.txt"

    deactivate
}

# Create environment file
create_env_file() {
    if [ ! -f "$INSTALL_DIR/.env" ]; then
        log_info "Creating environment file..."

        # Generate a random API key
        API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

        cat > "$INSTALL_DIR/.env" << EOF
# Hockey Analytics Dashboard Configuration
# Generated on $(date)

# REQUIRED: API Key for authentication
API_KEY=$API_KEY

# REQUIRED: Allowed CORS origins (comma-separated)
CORS_ALLOWED_ORIGINS=http://localhost:8000

# Server configuration
PORT=8000
WORKERS=4
LOG_LEVEL=info
DEBUG_MODE=false

# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Optional: Sentry error monitoring
# SENTRY_DSN=https://...@sentry.io/...

# Optional: Environment name for Sentry
ENVIRONMENT=production
EOF

        log_warn "Environment file created at $INSTALL_DIR/.env"
        log_warn "Please edit this file to configure CORS_ALLOWED_ORIGINS and other settings"
        log_warn "Generated API_KEY: $API_KEY"
    else
        log_info "Environment file already exists, skipping..."
    fi
}

# Set permissions
set_permissions() {
    log_info "Setting permissions..."

    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
    chmod 750 "$INSTALL_DIR"
    chmod 640 "$INSTALL_DIR/.env"
    chmod -R 755 "$INSTALL_DIR/dashboard/uploads"
    chmod -R 755 "$INSTALL_DIR/logs"
}

# Install systemd service
install_service() {
    log_info "Installing systemd service..."

    cp "$INSTALL_DIR/scripts/hockey-analytics.service" /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable hockey-analytics
}

# Print completion message
print_completion() {
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Installation Complete!${NC}"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Edit the environment file:"
    echo "   sudo nano $INSTALL_DIR/.env"
    echo ""
    echo "2. Configure CORS_ALLOWED_ORIGINS with your domain"
    echo ""
    echo "3. Start the service:"
    echo "   sudo systemctl start hockey-analytics"
    echo ""
    echo "4. Check service status:"
    echo "   sudo systemctl status hockey-analytics"
    echo ""
    echo "5. View logs:"
    echo "   sudo journalctl -u hockey-analytics -f"
    echo ""
    echo "The API will be available at:"
    echo "   http://localhost:8000"
    echo ""
    echo "Health check endpoint:"
    echo "   http://localhost:8000/health"
    echo ""
}

# Main
main() {
    log_info "Hockey Analytics Dashboard - Production Installation"
    echo ""

    check_root
    check_requirements
    install_system_deps
    create_install_dir
    copy_files
    setup_venv
    create_env_file
    set_permissions
    install_service
    print_completion
}

main "$@"
