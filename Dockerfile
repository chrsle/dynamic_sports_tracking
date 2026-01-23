# Multi-stage build for Hockey Analytics Dashboard
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM python:3.10-slim as production

# Labels
LABEL maintainer="Hockey Analytics Team"
LABEL version="2.0.0"
LABEL description="Hockey Analytics Dashboard with Moneyball Features"

# Create non-root user for security
RUN groupadd -r analytics && useradd -r -g analytics analytics

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=analytics:analytics . .

# Create uploads directory with proper permissions
RUN mkdir -p /app/dashboard/uploads && \
    chown -R analytics:analytics /app/dashboard/uploads

# Switch to non-root user
USER analytics

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBUG_MODE=false
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "-m", "uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]
