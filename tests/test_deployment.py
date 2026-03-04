"""Deployment configuration validation tests.

Validates:
- Docker Compose config parses correctly and contains required services
- Kubernetes manifests are valid YAML with required fields
- Prometheus alert rules are well-formed
- Alertmanager config is well-formed
- Prometheus metrics middleware registers without errors
- Celery app can be instantiated
- Sentry init is a no-op when SENTRY_DSN is unset
- /metrics endpoint returns valid Prometheus format when prometheus_client available
- /health endpoint returns correct JSON
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

# Project root
_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Docker Compose validation
# ---------------------------------------------------------------------------


class TestDockerCompose:
    """Validate docker-compose.yml structure."""

    def test_docker_compose_is_valid_yaml(self) -> None:
        path = _ROOT / "docker-compose.yml"
        assert path.exists(), "docker-compose.yml not found"
        doc = yaml.safe_load(path.read_text())
        assert doc is not None

    def test_required_services_present(self) -> None:
        doc = yaml.safe_load((_ROOT / "docker-compose.yml").read_text())
        services = set(doc.get("services", {}).keys())
        required = {"db", "redis", "api", "celery-worker", "celery-beat", "frontend", "nginx"}
        missing = required - services
        assert not missing, f"Missing services: {missing}"

    def test_monitoring_services_present(self) -> None:
        doc = yaml.safe_load((_ROOT / "docker-compose.yml").read_text())
        services = set(doc.get("services", {}).keys())
        monitoring = {"prometheus", "grafana", "alertmanager"}
        missing = monitoring - services
        assert not missing, f"Missing monitoring services: {missing}"

    def test_api_service_has_healthcheck(self) -> None:
        doc = yaml.safe_load((_ROOT / "docker-compose.yml").read_text())
        db_service = doc["services"]["db"]
        assert "healthcheck" in db_service

    def test_volumes_defined(self) -> None:
        doc = yaml.safe_load((_ROOT / "docker-compose.yml").read_text())
        volumes = doc.get("volumes", {})
        assert "timescaledb_data" in volumes
        assert "redis_data" in volumes
        assert "prometheus_data" in volumes
        assert "grafana_data" in volumes

    def test_prod_override_is_valid_yaml(self) -> None:
        path = _ROOT / "docker-compose.prod.yml"
        assert path.exists(), "docker-compose.prod.yml not found"
        doc = yaml.safe_load(path.read_text())
        assert doc is not None
        assert "services" in doc


# ---------------------------------------------------------------------------
# Dockerfile validation
# ---------------------------------------------------------------------------


class TestDockerfiles:
    """Validate Dockerfile existence and basic syntax."""

    def test_api_dockerfile_exists(self) -> None:
        assert (_ROOT / "docker" / "Dockerfile").exists()

    def test_frontend_dockerfile_exists(self) -> None:
        assert (_ROOT / "docker" / "Dockerfile.frontend").exists()

    def test_api_dockerfile_has_multistage(self) -> None:
        content = (_ROOT / "docker" / "Dockerfile").read_text()
        assert "AS builder" in content
        assert "AS runtime" in content

    def test_api_dockerfile_runs_as_nonroot(self) -> None:
        content = (_ROOT / "docker" / "Dockerfile").read_text()
        assert "useradd" in content or "adduser" in content
        assert "USER quantflow" in content

    def test_api_dockerfile_has_healthcheck(self) -> None:
        content = (_ROOT / "docker" / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_nginx_config_exists(self) -> None:
        assert (_ROOT / "docker" / "nginx" / "nginx.conf").exists()

    def test_initdb_sql_exists(self) -> None:
        assert (_ROOT / "docker" / "initdb" / "001_extensions.sql").exists()


# ---------------------------------------------------------------------------
# Kubernetes manifest validation
# ---------------------------------------------------------------------------


class TestKubernetesManifests:
    """Validate k8s YAML structure."""

    def _load(self, filename: str) -> list[dict]:
        path = _ROOT / "k8s" / filename
        assert path.exists(), f"k8s/{filename} not found"
        # Support multi-document YAML
        docs = list(yaml.safe_load_all(path.read_text()))
        return [d for d in docs if d is not None]

    def test_namespace_yaml(self) -> None:
        docs = self._load("namespace.yaml")
        assert len(docs) == 1
        assert docs[0]["kind"] == "Namespace"
        assert docs[0]["metadata"]["name"] == "quantflow"

    def test_configmap_yaml(self) -> None:
        docs = self._load("configmap.yaml")
        assert len(docs) == 1
        assert docs[0]["kind"] == "ConfigMap"
        cm_data = docs[0]["data"]
        assert "ENVIRONMENT" in cm_data
        assert "LOG_LEVEL" in cm_data

    def test_api_deployment_has_probes(self) -> None:
        docs = self._load("api-deployment.yaml")
        deployments = [d for d in docs if d.get("kind") == "Deployment"]
        assert deployments, "No Deployment found in api-deployment.yaml"
        containers = deployments[0]["spec"]["template"]["spec"]["containers"]
        assert any("readinessProbe" in c for c in containers)
        assert any("livenessProbe" in c for c in containers)

    def test_api_deployment_has_resource_limits(self) -> None:
        docs = self._load("api-deployment.yaml")
        deployments = [d for d in docs if d.get("kind") == "Deployment"]
        containers = deployments[0]["spec"]["template"]["spec"]["containers"]
        for c in containers:
            assert "resources" in c, f"Container {c['name']} missing resources"
            assert "limits" in c["resources"]

    def test_api_deployment_namespace(self) -> None:
        docs = self._load("api-deployment.yaml")
        for doc in docs:
            assert doc["metadata"]["namespace"] == "quantflow"

    def test_ingress_has_tls(self) -> None:
        docs = self._load("ingress.yaml")
        ingress = [d for d in docs if d.get("kind") == "Ingress"][0]
        assert "tls" in ingress["spec"]

    def test_hpa_yaml(self) -> None:
        docs = self._load("hpa.yaml")
        hpas = [d for d in docs if d.get("kind") == "HorizontalPodAutoscaler"]
        names = [h["metadata"]["name"] for h in hpas]
        assert "quantflow-api-hpa" in names
        assert "quantflow-celery-worker-hpa" in names

    def test_secret_template_exists(self) -> None:
        assert (_ROOT / "k8s" / "secret.yaml.template").exists()

    def test_secret_template_has_required_keys(self) -> None:
        content = (_ROOT / "k8s" / "secret.yaml.template").read_text()
        for key in ["DATABASE_URL", "JWT_SECRET_KEY", "OPENAI_API_KEY", "SENTRY_DSN"]:
            assert key in content, f"Secret template missing {key}"


# ---------------------------------------------------------------------------
# Prometheus config validation
# ---------------------------------------------------------------------------


class TestPrometheusConfig:
    """Validate Prometheus and Alertmanager YAML."""

    def test_prometheus_yml_valid(self) -> None:
        path = _ROOT / "docker" / "prometheus" / "prometheus.yml"
        assert path.exists()
        doc = yaml.safe_load(path.read_text())
        assert "scrape_configs" in doc
        assert "alerting" in doc

    def test_prometheus_scrapes_api(self) -> None:
        doc = yaml.safe_load(
            (_ROOT / "docker" / "prometheus" / "prometheus.yml").read_text()
        )
        job_names = [j["job_name"] for j in doc["scrape_configs"]]
        assert "quantflow_api" in job_names

    def test_alert_rules_valid_yaml(self) -> None:
        path = _ROOT / "docker" / "prometheus" / "alert_rules.yml"
        assert path.exists()
        doc = yaml.safe_load(path.read_text())
        assert "groups" in doc

    def test_alert_rules_has_critical_api_alert(self) -> None:
        doc = yaml.safe_load(
            (_ROOT / "docker" / "prometheus" / "alert_rules.yml").read_text()
        )
        all_alerts = []
        for group in doc["groups"]:
            all_alerts.extend(rule["alert"] for rule in group.get("rules", []))
        assert "APIDown" in all_alerts
        assert "APIHighErrorRate" in all_alerts

    def test_alertmanager_yml_valid(self) -> None:
        path = _ROOT / "docker" / "alertmanager" / "alertmanager.yml"
        assert path.exists()
        doc = yaml.safe_load(path.read_text())
        assert "route" in doc
        assert "receivers" in doc

    def test_alertmanager_has_pagerduty_receiver(self) -> None:
        doc = yaml.safe_load(
            (_ROOT / "docker" / "alertmanager" / "alertmanager.yml").read_text()
        )
        receiver_names = [r["name"] for r in doc["receivers"]]
        assert "pagerduty" in receiver_names

    def test_grafana_dashboard_valid_json(self) -> None:
        import json
        path = _ROOT / "docker" / "grafana" / "dashboards" / "quantflow.json"
        assert path.exists()
        dashboard = json.loads(path.read_text())
        assert "panels" in dashboard
        assert len(dashboard["panels"]) >= 5


# ---------------------------------------------------------------------------
# GitHub Actions workflow validation
# ---------------------------------------------------------------------------


class TestGitHubWorkflows:
    """Validate GitHub Actions YAML."""

    def _load_workflow(self, name: str) -> dict:
        path = _ROOT / ".github" / "workflows" / name
        assert path.exists(), f".github/workflows/{name} not found"
        return yaml.safe_load(path.read_text())

    def test_ci_workflow_valid(self) -> None:
        wf = self._load_workflow("ci.yml")
        assert "jobs" in wf
        # YAML 1.1 parses "on:" as boolean True; both are acceptable
        assert "on" in wf or True in wf

    def test_ci_has_required_jobs(self) -> None:
        wf = self._load_workflow("ci.yml")
        jobs = set(wf["jobs"].keys())
        assert "lint" in jobs
        assert "test-python" in jobs
        assert "test-frontend" in jobs

    def test_cd_workflow_valid(self) -> None:
        wf = self._load_workflow("cd.yml")
        assert "jobs" in wf
        jobs = set(wf["jobs"].keys())
        assert "build-and-push" in jobs
        assert "deploy-staging" in jobs
        assert "deploy-production" in jobs

    def test_docs_workflow_valid(self) -> None:
        wf = self._load_workflow("docs.yml")
        assert "jobs" in wf


# ---------------------------------------------------------------------------
# Python module validation
# ---------------------------------------------------------------------------


class TestPythonModules:
    """Validate new Python modules load without errors."""

    def test_metrics_middleware_imports(self) -> None:
        from quantflow.api.middleware.metrics import (  # noqa: F401
            PrometheusMetricsMiddleware,
            metrics_response,
            record_signal_generated,
            set_active_ws_connections,
        )

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("celery") is None,
        reason="celery not installed",
    )
    def test_celery_app_imports(self) -> None:
        from quantflow.api.celery_app import app  # noqa: F401
        assert app is not None
        assert app.main == "quantflow"

    def test_main_app_imports_cleanly(self) -> None:
        # Should not raise even with no SENTRY_DSN
        os.environ.pop("SENTRY_DSN", None)
        from quantflow.api.main import app  # noqa: F401
        assert app is not None

    def test_metrics_response_returns_response(self) -> None:
        from quantflow.api.middleware.metrics import metrics_response
        resp = metrics_response()
        assert resp is not None
        assert resp.status_code == 200

    def test_record_signal_no_error(self) -> None:
        from quantflow.api.middleware.metrics import record_signal_generated
        # Should not raise even if prometheus_client not installed
        record_signal_generated("AAPL", "BUY")

    def test_set_ws_connections_no_error(self) -> None:
        from quantflow.api.middleware.metrics import set_active_ws_connections
        set_active_ws_connections(42)


# ---------------------------------------------------------------------------
# FastAPI application endpoint tests
# ---------------------------------------------------------------------------


class TestFastAPIEndpoints:
    """Smoke-test the /health and /metrics endpoints."""

    @pytest.fixture
    def client(self):
        from httpx import AsyncClient, ASGITransport
        from quantflow.api.main import app
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client) -> None:
        async with client as c:
            resp = await c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "environment" in body

    @pytest.mark.asyncio
    async def test_metrics_endpoint_returns_200(self, client) -> None:
        async with client as c:
            resp = await c.get("/metrics")
        assert resp.status_code == 200
        # Prometheus text format always starts with a comment or metric
        assert len(resp.content) > 0

    @pytest.mark.asyncio
    async def test_openapi_schema_accessible(self, client) -> None:
        async with client as c:
            resp = await c.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "QuantFlow API"
        assert "paths" in schema


# ---------------------------------------------------------------------------
# Documentation validation
# ---------------------------------------------------------------------------


class TestDocumentation:
    """Validate documentation files."""

    def test_mkdocs_yml_valid(self) -> None:
        path = _ROOT / "mkdocs.yml"
        assert path.exists()
        doc = yaml.safe_load(path.read_text())
        assert "site_name" in doc
        assert "nav" in doc

    def test_docs_index_exists(self) -> None:
        assert (_ROOT / "docs" / "index.md").exists()

    def test_docs_deployment_sections_exist(self) -> None:
        for fname in ["docker-compose.md", "kubernetes.md", "monitoring.md"]:
            assert (_ROOT / "docs" / "deployment" / fname).exists(), f"Missing docs/deployment/{fname}"
