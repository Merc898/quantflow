# Monitoring — Prometheus + Grafana + Alertmanager

## Metrics Exposed

The FastAPI application exposes metrics at `GET /metrics` (Prometheus text format).

### HTTP Metrics (via `PrometheusMetricsMiddleware`)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_requests_total` | Counter | method, endpoint, http_status | Total request count |
| `http_request_duration_seconds` | Histogram | method, endpoint | Request latency |
| `http_requests_in_progress` | Gauge | method, endpoint | Active requests |

### Business Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `quantflow_signals_generated_total` | Counter | symbol, recommendation | Signals computed |
| `quantflow_active_websocket_connections` | Gauge | — | Live WS connections |
| `quantflow_celery_tasks_total` | Counter | task_name, status | Celery throughput |

## Alerting Rules

Alerts are defined in `docker/prometheus/alert_rules.yml`:

| Alert | Severity | Condition |
|-------|----------|-----------|
| `APIDown` | critical | API unreachable > 1 min |
| `APIHighErrorRate` | warning | 5xx rate > 5% for 5 min |
| `APIHighLatencyP95` | warning | p95 > 2s for 10 min |
| `APIHighLatencyP99` | critical | p99 > 10s for 5 min |
| `NoSignalsGenerated` | warning | Zero signals for 30 min |
| `RedisDown` | critical | Redis unreachable > 2 min |
| `CeleryWorkerDown` | critical | No worker metrics for 5 min |

## Alertmanager Routing

```
critical → PagerDuty (on-call wake-up) + #quantflow-alerts Slack
warning  → #quantflow-warnings Slack + email
platform → platform-team@quantflow.io
```

## Accessing Dashboards

| Service | URL | Default Login |
|---------|-----|---------------|
| Grafana | http://localhost:3001 | admin / quantflow |
| Prometheus | http://localhost:9090 | — |
| Alertmanager | http://localhost:9093 | — |
| MLflow | http://localhost:5000 | — |

## Adding Custom Metrics

```python
from quantflow.api.middleware.metrics import record_signal_generated

# In your signal generation code:
record_signal_generated(symbol="AAPL", recommendation="BUY")
```
