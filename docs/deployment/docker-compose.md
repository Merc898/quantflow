# Docker Compose — Local Development

The full QuantFlow development stack runs via a single `docker compose up -d` command.

## Services

| Service | Port | Description |
|---------|------|-------------|
| `db` | 5432 | TimescaleDB (PostgreSQL 15 + TimescaleDB extension) |
| `redis` | 6379 | Redis — cache, Celery broker, rate-limit store |
| `api` | 8000 | FastAPI application server (hot-reload) |
| `celery-worker` | — | Background task worker |
| `celery-beat` | — | Periodic task scheduler |
| `frontend` | 3000 | Next.js SaaS frontend |
| `nginx` | 80 | Reverse proxy entry point |
| `mlflow` | 5000 | MLflow experiment tracking |
| `prometheus` | 9090 | Prometheus metrics collection |
| `alertmanager` | 9093 | Alert routing (Slack, PagerDuty) |
| `grafana` | 3001 | Metrics dashboards |

### Optional (--profile tools)

| Service | Port | Description |
|---------|------|-------------|
| `redis-commander` | 8081 | Redis web UI |
| `flower` | 5555 | Celery task monitor |

## Environment Setup

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your keys
```

## Starting the Stack

```bash
# Core services only
docker compose up -d

# With monitoring UIs
docker compose --profile tools up -d

# View logs
docker compose logs -f api

# Stop without wiping data
docker compose down

# Full reset (WARNING: deletes all data)
docker compose down -v
```

## Database Migrations

```bash
# Run Alembic migrations after starting services
docker compose exec api alembic upgrade head
```
