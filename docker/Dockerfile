# QuantFlow — Python application image
# Used for: FastAPI API server AND Celery workers (same image, different CMD)
#
# Multi-stage build:
#   builder  — installs dependencies into a venv
#   runtime  — copies venv, runs as non-root user

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip + install wheel
RUN pip install --upgrade pip wheel

# Install Python dependencies first (layer cached unless pyproject.toml changes)
COPY pyproject.toml ./
# Install the package in editable mode so quantflow is importable
# Use --no-build-isolation to reuse already-installed build tools
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd -r quantflow && useradd -r -g quantflow quantflow

WORKDIR /app

# Runtime OS dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY quantflow/ ./quantflow/
COPY pyproject.toml ./

# Reinstall package (no build deps needed — venv already has everything)
RUN pip install --no-deps -e . --quiet

# Create directories for logs and temp files
RUN mkdir -p /app/logs /app/tmp && chown -R quantflow:quantflow /app

USER quantflow

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: FastAPI server
# Override with celery command for worker containers
CMD ["uvicorn", "quantflow.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
