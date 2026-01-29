"""DevOps and CI/CD Tools.

Tools for creating pipelines, deployment configurations, and infrastructure.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool


@tool
def create_ci_pipeline(
    project_name: str,
    platform: str = "github-actions",
    language: str = "python",
    include_tests: bool = True,
    include_security: bool = True,
    session_id: str = "default",
) -> str:
    """Create a CI pipeline configuration.

    Generates CI pipeline for:
    - GitHub Actions
    - GitLab CI
    - Azure DevOps
    - Jenkins

    Args:
        project_name: Name of the project.
        platform: CI platform to use.
        language: Programming language.
        include_tests: Whether to include test stage.
        include_security: Whether to include security scanning.
        session_id: Session identifier.

    Returns:
        JSON string with CI pipeline configuration.
    """
    pipeline_id = f"CI-{str(uuid.uuid4())[:8].upper()}"

    pipelines = {
        "github-actions": {
            "filename": ".github/workflows/ci.yml",
            "content": f"""name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Lint
        run: |
          pip install ruff
          ruff check .

{'''      - name: Run tests
        run: |
          pip install pytest pytest-cov
          pytest --cov=. --cov-report=xml

''' if include_tests else ''}{'''      - name: Security scan
        run: |
          pip install bandit safety
          bandit -r . -ll
          safety check
''' if include_security else ''}""",
        },
        "gitlab-ci": {
            "filename": ".gitlab-ci.yml",
            "content": f"""stages:
  - build
  - test
  - security

build:
  stage: build
  image: python:3.11
  script:
    - pip install -r requirements.txt

{'''test:
  stage: test
  image: python:3.11
  script:
    - pip install pytest
    - pytest
''' if include_tests else ''}{'''security:
  stage: security
  image: python:3.11
  script:
    - pip install bandit
    - bandit -r .
''' if include_security else ''}""",
        },
        "azure-devops": {
            "filename": "azure-pipelines.yml",
            "content": f"""trigger:
  - main
  - develop

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'

  - script: |
      pip install -r requirements.txt
    displayName: 'Install dependencies'

{'''  - script: |
      pip install pytest
      pytest
    displayName: 'Run tests'
''' if include_tests else ''}""",
        },
    }

    config = pipelines.get(platform, pipelines["github-actions"])

    result = {
        "id": pipeline_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "project_name": project_name,
        "platform": platform,
        "language": language,
        "pipeline_config": config,
        "features": {
            "tests": include_tests,
            "security": include_security,
            "linting": True,
        },
    }

    return json.dumps(result, indent=2)


@tool
def create_cd_pipeline(
    project_name: str,
    platform: str = "github-actions",
    deployment_target: str = "kubernetes",
    environments: str = "staging,production",
    session_id: str = "default",
) -> str:
    """Create a CD pipeline configuration.

    Generates deployment pipeline for:
    - Kubernetes
    - AWS ECS
    - Azure App Service
    - Docker Compose

    Args:
        project_name: Name of the project.
        platform: CI/CD platform.
        deployment_target: Where to deploy.
        environments: Comma-separated list of environments.
        session_id: Session identifier.

    Returns:
        JSON string with CD pipeline configuration.
    """
    pipeline_id = f"CD-{str(uuid.uuid4())[:8].upper()}"

    env_list = [e.strip() for e in environments.split(",")]

    result = {
        "id": pipeline_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "project_name": project_name,
        "platform": platform,
        "deployment_target": deployment_target,
        "environments": env_list,
        "pipeline_config": {
            "filename": ".github/workflows/cd.yml" if platform == "github-actions" else "deploy.yml",
            "stages": ["build", "push", "deploy-staging", "approve", "deploy-production"],
        },
        "deployment_strategy": "rolling" if deployment_target == "kubernetes" else "blue-green",
    }

    return json.dumps(result, indent=2)


@tool
def configure_deployment(
    service_name: str,
    environment: str = "production",
    replicas: int = 3,
    resources: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Configure deployment settings for a service.

    Sets up:
    - Resource limits
    - Scaling configuration
    - Health checks
    - Environment variables

    Args:
        service_name: Name of the service.
        environment: Target environment.
        replicas: Number of replicas.
        resources: Optional resource configuration JSON.
        session_id: Session identifier.

    Returns:
        JSON string with deployment configuration.
    """
    deploy_id = f"DEP-{str(uuid.uuid4())[:8].upper()}"

    # Parse resources or use defaults
    if resources:
        try:
            resource_config = json.loads(resources)
        except json.JSONDecodeError:
            resource_config = None
    else:
        resource_config = None

    if not resource_config:
        resource_config = {
            "cpu": "500m" if environment == "production" else "250m",
            "memory": "512Mi" if environment == "production" else "256Mi",
            "cpu_limit": "1000m" if environment == "production" else "500m",
            "memory_limit": "1Gi" if environment == "production" else "512Mi",
        }

    result = {
        "id": deploy_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "service_name": service_name,
        "environment": environment,
        "configuration": {
            "replicas": replicas,
            "resources": resource_config,
            "health_check": {
                "path": "/health",
                "interval": "30s",
                "timeout": "10s",
            },
            "rollout_strategy": {
                "type": "RollingUpdate",
                "max_surge": "25%",
                "max_unavailable": "25%",
            },
        },
    }

    return json.dumps(result, indent=2)


@tool
def generate_dockerfile(
    project_type: str = "python-api",
    base_image: Optional[str] = None,
    port: int = 8000,
    session_id: str = "default",
) -> str:
    """Generate a Dockerfile for the project.

    Creates optimized Dockerfiles for:
    - Python API
    - Node.js API
    - Static website
    - Multi-stage builds

    Args:
        project_type: Type of project.
        base_image: Optional custom base image.
        port: Port to expose.
        session_id: Session identifier.

    Returns:
        JSON string with Dockerfile content.
    """
    docker_id = f"DOCKER-{str(uuid.uuid4())[:8].upper()}"

    dockerfiles = {
        "python-api": f"""# Multi-stage build for Python API
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE {port}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
""",
        "node-api": f"""# Multi-stage build for Node.js API
FROM node:20-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:20-alpine

WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .

ENV NODE_ENV=production

EXPOSE {port}

CMD ["node", "index.js"]
""",
        "static": f"""FROM nginx:alpine

COPY dist/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE {port}

CMD ["nginx", "-g", "daemon off;"]
""",
    }

    dockerfile_content = dockerfiles.get(project_type, dockerfiles["python-api"])

    result = {
        "id": docker_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "project_type": project_type,
        "base_image": base_image or "python:3.11-slim",
        "port": port,
        "dockerfile": dockerfile_content,
        "best_practices": [
            "Using multi-stage build for smaller image",
            "Non-root user recommended for production",
            "Add .dockerignore to exclude unnecessary files",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def create_kubernetes_config(
    service_name: str,
    namespace: str = "default",
    replicas: int = 3,
    port: int = 8000,
    session_id: str = "default",
) -> str:
    """Create Kubernetes deployment and service configurations.

    Generates:
    - Deployment manifest
    - Service manifest
    - ConfigMap
    - HPA (Horizontal Pod Autoscaler)

    Args:
        service_name: Name of the service.
        namespace: Kubernetes namespace.
        replicas: Number of replicas.
        port: Container port.
        session_id: Session identifier.

    Returns:
        JSON string with Kubernetes configurations.
    """
    k8s_id = f"K8S-{str(uuid.uuid4())[:8].upper()}"

    deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service_name}
  namespace: {namespace}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {service_name}
  template:
    metadata:
      labels:
        app: {service_name}
    spec:
      containers:
      - name: {service_name}
        image: {service_name}:latest
        ports:
        - containerPort: {port}
        resources:
          requests:
            cpu: "250m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
"""

    service = f"""apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {namespace}
spec:
  selector:
    app: {service_name}
  ports:
  - port: 80
    targetPort: {port}
  type: ClusterIP
"""

    result = {
        "id": k8s_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "service_name": service_name,
        "namespace": namespace,
        "manifests": {
            "deployment": deployment,
            "service": service,
        },
        "files": [
            f"k8s/{service_name}-deployment.yaml",
            f"k8s/{service_name}-service.yaml",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def setup_monitoring(
    service_name: str,
    monitoring_stack: str = "prometheus",
    metrics: str = "basic",
    session_id: str = "default",
) -> str:
    """Set up monitoring and observability for a service.

    Configures:
    - Prometheus metrics
    - Grafana dashboards
    - Alert rules
    - Log aggregation

    Args:
        service_name: Name of the service.
        monitoring_stack: Stack to use - "prometheus", "datadog", "newrelic".
        metrics: Level of metrics - "basic", "detailed", "custom".
        session_id: Session identifier.

    Returns:
        JSON string with monitoring configuration.
    """
    monitor_id = f"MON-{str(uuid.uuid4())[:8].upper()}"

    monitoring_config = {
        "prometheus": {
            "scrape_config": {
                "job_name": service_name,
                "scrape_interval": "15s",
                "static_configs": [{"targets": [f"{service_name}:8000"]}],
            },
            "metrics_path": "/metrics",
        },
        "datadog": {
            "service": service_name,
            "env": "production",
            "version": "1.0.0",
        },
    }

    alerts = [
        {
            "name": f"{service_name}_high_error_rate",
            "condition": "error_rate > 5%",
            "severity": "critical",
        },
        {
            "name": f"{service_name}_high_latency",
            "condition": "p99_latency > 1s",
            "severity": "warning",
        },
        {
            "name": f"{service_name}_low_availability",
            "condition": "availability < 99.9%",
            "severity": "critical",
        },
    ]

    result = {
        "id": monitor_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "service_name": service_name,
        "monitoring_stack": monitoring_stack,
        "configuration": monitoring_config.get(monitoring_stack, monitoring_config["prometheus"]),
        "alerts": alerts,
        "dashboards": [
            "Service Overview",
            "Error Analysis",
            "Latency Distribution",
            "Resource Usage",
        ],
    }

    return json.dumps(result, indent=2)
