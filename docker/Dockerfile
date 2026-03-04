# QuantFlow — Python application image
# Used for: FastAPI API server AND Celery workers (same image, different CMD)
#
# Multi-stage build:
#   builder  — installs dependencies into a venv (uses uv for fast resolution)
#   runtime  — copies venv + source, runs as non-root user

# ---------------------------------------------------------------------------
# Stage 1: builder — install all Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# System build dependencies (gcc for C extensions, libpq-dev for asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv so we can copy it cleanly into runtime
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# Use uv for dependency resolution — orders of magnitude faster than pip
# on large dependency trees (80+ packages).
RUN pip install --upgrade pip uv

# --- Dependency caching layer ---
# Copy only metadata files + a package stub so uv can resolve deps.
# This layer is cached until pyproject.toml changes.
COPY pyproject.toml README.md ./
RUN mkdir -p quantflow && echo '__version__ = "0.1.0"' > quantflow/__init__.py

# Install production dependencies only (no [dev] extras).
RUN uv pip install --no-cache .

# --- Real source layer ---
# Copy the actual package source and reinstall (deps already cached above).
COPY quantflow/ ./quantflow/
RUN uv pip install --no-cache --no-deps --reinstall .

# ---------------------------------------------------------------------------
# Stage 2: runtime — minimal image with just the venv + source
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd -r quantflow && useradd -r -g quantflow quantflow

WORKDIR /app

# Runtime OS dependencies only (no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application source
COPY quantflow/ ./quantflow/
COPY pyproject.toml README.md ./

# Create directories for logs and temp files
RUN mkdir -p /app/logs /app/tmp && chown -R quantflow:quantflow /app

USER quantflow

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: FastAPI server (override CMD for celery workers)
CMD ["uvicorn", "quantflow.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
