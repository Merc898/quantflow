# Kubernetes — Production Deployment

## Prerequisites

- Kubernetes 1.28+
- `kubectl` configured against your cluster
- NGINX Ingress Controller installed
- cert-manager installed (for TLS)
- Container registry access (GHCR)

## Manifests

| File | Description |
|------|-------------|
| `k8s/namespace.yaml` | `quantflow` namespace |
| `k8s/configmap.yaml` | Non-secret configuration |
| `k8s/secret.yaml.template` | Secret template (inject real values via CI) |
| `k8s/api-deployment.yaml` | FastAPI Deployment + Service + ServiceAccount |
| `k8s/celery-deployment.yaml` | Celery worker + beat Deployments + PVC |
| `k8s/frontend-deployment.yaml` | Next.js Deployment + Service |
| `k8s/ingress.yaml` | NGINX Ingress with TLS |
| `k8s/hpa.yaml` | Horizontal Pod Autoscalers |

## Initial Deployment

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Create secrets (do NOT commit real values)
kubectl create secret generic quantflow-secrets \
  --namespace=quantflow \
  --from-literal=DATABASE_URL="postgresql+asyncpg://..." \
  --from-literal=JWT_SECRET_KEY="..." \
  # ... see k8s/secret.yaml.template for full list

# 3. Apply all manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/celery-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Watch rollout
kubectl rollout status deployment/quantflow-api -n quantflow
```

## Scaling

HPA handles automatic scaling based on CPU/memory:

| Component | Min Replicas | Max Replicas | Scale Trigger |
|-----------|-------------|-------------|---------------|
| API | 2 | 10 | CPU > 70% |
| Celery Worker | 2 | 8 | CPU > 75% |
| Frontend | 2 | 6 | CPU > 70% |

## Health Checks

```bash
# Check pod status
kubectl get pods -n quantflow

# View API logs
kubectl logs -n quantflow deployment/quantflow-api -f

# Check HPA status
kubectl get hpa -n quantflow
```
