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
        base_image: Optional custom base image (e.g., "python:3.12-slim", "node:22-alpine").
        port: Port to expose.
        session_id: Session identifier.

    Returns:
        JSON string with Dockerfile content.
    """
    docker_id = f"DOCKER-{str(uuid.uuid4())[:8].upper()}"

    # Default base images for each project type
    default_base_images = {
        "python-api": "python:3.11-slim",
        "node-api": "node:20-alpine",
        "static": "nginx:alpine",
    }

    # Determine the actual base image to use
    actual_base_image = base_image or default_base_images.get(project_type, "python:3.11-slim")

    # Extract image family for multi-stage builds (e.g., "python:3.12-slim" -> "python")
    image_family = actual_base_image.split(":")[0].split("/")[-1]

    # Generate Dockerfile based on project type with custom base image support
    if project_type == "python-api":
        # For Python, use the custom base image if provided
        dockerfile_content = f"""# Multi-stage build for Python API
FROM {actual_base_image} as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM {actual_base_image}

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE {port}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
"""

    elif project_type == "node-api":
        dockerfile_content = f"""# Multi-stage build for Node.js API
FROM {actual_base_image} as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM {actual_base_image}

WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .

ENV NODE_ENV=production

EXPOSE {port}

CMD ["node", "index.js"]
"""

    elif project_type == "static":
        dockerfile_content = f"""FROM {actual_base_image}

COPY dist/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE {port}

CMD ["nginx", "-g", "daemon off;"]
"""

    elif project_type == "go-api":
        # Support for Go applications with custom base image
        go_base = actual_base_image if "go" in actual_base_image.lower() else "golang:1.22-alpine"
        runtime_base = "alpine:latest" if base_image is None else actual_base_image
        dockerfile_content = f"""# Multi-stage build for Go API
FROM {go_base} as builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

FROM {runtime_base}

WORKDIR /app
COPY --from=builder /app/main .

EXPOSE {port}

CMD ["./main"]
"""

    elif project_type == "java-api":
        # Support for Java applications with custom base image
        java_base = actual_base_image if "java" in actual_base_image.lower() or "openjdk" in actual_base_image.lower() else "eclipse-temurin:17-jdk-alpine"
        runtime_base = actual_base_image if "jre" in actual_base_image.lower() else "eclipse-temurin:17-jre-alpine"
        dockerfile_content = f"""# Multi-stage build for Java API
FROM {java_base} as builder

WORKDIR /app
COPY . .
RUN ./gradlew build -x test || ./mvnw package -DskipTests

FROM {runtime_base}

WORKDIR /app
COPY --from=builder /app/build/libs/*.jar app.jar
# Or for Maven: COPY --from=builder /app/target/*.jar app.jar

EXPOSE {port}

ENTRYPOINT ["java", "-jar", "app.jar"]
"""

    else:
        # Default to Python if project type is unknown, but still honor custom base image
        if base_image:
            dockerfile_content = f"""# Custom Dockerfile
FROM {actual_base_image}

WORKDIR /app
COPY . .

EXPOSE {port}

# Update CMD based on your application requirements
CMD ["echo", "Update this CMD for your application"]
"""
        else:
            dockerfile_content = f"""# Multi-stage build for Python API
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
"""

    result = {
        "id": docker_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "project_type": project_type,
        "base_image": actual_base_image,
        "custom_base_image": base_image is not None,
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

    # Add ConfigMap for application configuration
    configmap = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {service_name}-config
  namespace: {namespace}
data:
  APP_NAME: "{service_name}"
  APP_PORT: "{port}"
  LOG_LEVEL: "info"
  # Add application-specific configuration here
  # Example:
  # DATABASE_HOST: "db-service"
  # CACHE_TTL: "3600"
"""

    # Add HorizontalPodAutoscaler for auto-scaling
    hpa = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {service_name}-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {service_name}
  minReplicas: {max(1, replicas // 2)}
  maxReplicas: {replicas * 3}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
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
            "configmap": configmap,
            "hpa": hpa,
        },
        "files": [
            f"k8s/{service_name}-deployment.yaml",
            f"k8s/{service_name}-service.yaml",
            f"k8s/{service_name}-configmap.yaml",
            f"k8s/{service_name}-hpa.yaml",
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
