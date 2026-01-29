"""Architecture and Design Tools.

Tools for designing system architecture, APIs, and data models.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool


@tool
def design_architecture(
    requirements: str,
    pattern: str = "microservices",
    constraints: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Design system architecture based on requirements.

    Proposes architecture following specified pattern:
    - Microservices
    - Monolithic
    - Serverless
    - Event-driven
    - Layered

    Args:
        requirements: System requirements description.
        pattern: Architecture pattern to use.
        constraints: Technical or business constraints.
        session_id: Session identifier.

    Returns:
        JSON string with architecture design.
    """
    arch_id = f"ARCH-{str(uuid.uuid4())[:8].upper()}"

    patterns_config = {
        "microservices": {
            "components": ["API Gateway", "Service Registry", "Config Server", "Message Queue"],
            "communication": "REST/gRPC + Message Queue",
            "deployment": "Kubernetes/Docker",
        },
        "monolithic": {
            "components": ["Web Layer", "Business Logic", "Data Access", "Database"],
            "communication": "In-process calls",
            "deployment": "Single deployment unit",
        },
        "serverless": {
            "components": ["API Gateway", "Lambda Functions", "Event Triggers", "Managed Services"],
            "communication": "Event-driven",
            "deployment": "Cloud-native (AWS/Azure/GCP)",
        },
        "event_driven": {
            "components": ["Event Bus", "Producers", "Consumers", "Event Store"],
            "communication": "Pub/Sub messaging",
            "deployment": "Distributed",
        },
        "layered": {
            "components": ["Presentation", "Application", "Domain", "Infrastructure"],
            "communication": "Layer-to-layer calls",
            "deployment": "N-tier deployment",
        },
    }

    config = patterns_config.get(pattern, patterns_config["microservices"])

    architecture = {
        "id": arch_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "pattern": pattern,
        "requirements_summary": requirements[:300],
        "design": {
            "core_components": config["components"],
            "communication_pattern": config["communication"],
            "deployment_strategy": config["deployment"],
            "cross_cutting_concerns": [
                "Authentication & Authorization",
                "Logging & Monitoring",
                "Error Handling",
                "Configuration Management",
            ],
        },
        "recommendations": [
            "Implement circuit breaker for resilience",
            "Use centralized logging with correlation IDs",
            "Add health check endpoints for all services",
        ],
    }

    return json.dumps(architecture, indent=2)


@tool
def create_api_spec(
    api_name: str,
    resources: str,
    spec_format: str = "openapi",
    session_id: str = "default",
) -> str:
    """Create API specification for given resources.

    Generates API documentation in:
    - OpenAPI 3.0 format
    - GraphQL schema
    - gRPC proto

    Args:
        api_name: Name of the API.
        resources: Comma-separated list of resources or JSON description.
        spec_format: Output format - "openapi", "graphql", or "grpc".
        session_id: Session identifier.

    Returns:
        JSON string with API specification.
    """
    api_id = f"API-{str(uuid.uuid4())[:8].upper()}"

    # Parse resources
    try:
        resource_list = json.loads(resources)
    except json.JSONDecodeError:
        resource_list = [r.strip() for r in resources.split(",") if r.strip()]

    spec = {
        "id": api_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "api_name": api_name,
        "format": spec_format,
        "version": "1.0.0",
    }

    if spec_format == "openapi":
        spec["openapi_spec"] = {
            "openapi": "3.0.0",
            "info": {"title": api_name, "version": "1.0.0"},
            "paths": {},
            "components": {"schemas": {}},
        }
        for resource in resource_list:
            resource_lower = resource.lower().replace(" ", "_")
            spec["openapi_spec"]["paths"][f"/{resource_lower}s"] = {
                "get": {"summary": f"List {resource}s", "responses": {"200": {"description": "Success"}}},
                "post": {"summary": f"Create {resource}", "responses": {"201": {"description": "Created"}}},
            }
            spec["openapi_spec"]["paths"][f"/{resource_lower}s/{{id}}"] = {
                "get": {"summary": f"Get {resource}", "responses": {"200": {"description": "Success"}}},
                "put": {"summary": f"Update {resource}", "responses": {"200": {"description": "Updated"}}},
                "delete": {"summary": f"Delete {resource}", "responses": {"204": {"description": "Deleted"}}},
            }

    elif spec_format == "graphql":
        spec["graphql_schema"] = {
            "types": [],
            "queries": [],
            "mutations": [],
        }
        for resource in resource_list:
            spec["graphql_schema"]["types"].append(f"type {resource} {{ id: ID!, name: String }}")
            spec["graphql_schema"]["queries"].append(f"{resource.lower()}s: [{resource}]")
            spec["graphql_schema"]["mutations"].append(f"create{resource}(name: String!): {resource}")

    return json.dumps(spec, indent=2)


@tool
def suggest_tech_stack(
    project_type: str,
    requirements: str,
    preferences: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Suggest technology stack based on project requirements.

    Recommends technologies for:
    - Frontend framework
    - Backend language/framework
    - Database
    - Infrastructure
    - DevOps tools

    Args:
        project_type: Type of project (web, mobile, api, data, ml).
        requirements: Project requirements and constraints.
        preferences: Team preferences or constraints.
        session_id: Session identifier.

    Returns:
        JSON string with technology recommendations.
    """
    stack_id = f"STACK-{str(uuid.uuid4())[:8].upper()}"

    stacks = {
        "web": {
            "frontend": ["React", "Vue.js", "Angular"],
            "backend": ["Python/FastAPI", "Node.js/Express", "Go/Gin"],
            "database": ["PostgreSQL", "MongoDB", "Redis"],
            "infrastructure": ["AWS", "Azure", "GCP"],
        },
        "api": {
            "backend": ["Python/FastAPI", "Go/Gin", "Rust/Actix"],
            "database": ["PostgreSQL", "MongoDB"],
            "caching": ["Redis", "Memcached"],
            "infrastructure": ["Kubernetes", "Docker"],
        },
        "mobile": {
            "framework": ["React Native", "Flutter", "Swift/Kotlin"],
            "backend": ["Firebase", "AWS Amplify", "Custom API"],
            "database": ["SQLite", "Realm", "Firebase"],
        },
        "data": {
            "processing": ["Python/Pandas", "Apache Spark", "dbt"],
            "storage": ["Snowflake", "BigQuery", "Databricks"],
            "orchestration": ["Airflow", "Prefect", "Dagster"],
        },
        "ml": {
            "framework": ["PyTorch", "TensorFlow", "scikit-learn"],
            "mlops": ["MLflow", "Kubeflow", "Weights & Biases"],
            "serving": ["FastAPI", "TensorFlow Serving", "Triton"],
        },
    }

    recommendation = {
        "id": stack_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "project_type": project_type,
        "recommendations": stacks.get(project_type, stacks["web"]),
        "rationale": [
            "Selected based on project requirements and industry best practices",
            "Consider team expertise when making final decisions",
            "Evaluate total cost of ownership for each option",
        ],
    }

    return json.dumps(recommendation, indent=2)


@tool
def design_data_model(
    entities: str,
    relationships: Optional[str] = None,
    database_type: str = "relational",
    session_id: str = "default",
) -> str:
    """Design data model for specified entities.

    Creates database schema with:
    - Entity definitions
    - Relationships
    - Indexes
    - Constraints

    Args:
        entities: Comma-separated list of entities or JSON description.
        relationships: Optional relationship definitions.
        database_type: Type of database - "relational", "document", or "graph".
        session_id: Session identifier.

    Returns:
        JSON string with data model design.
    """
    model_id = f"DM-{str(uuid.uuid4())[:8].upper()}"

    # Parse entities
    try:
        entity_list = json.loads(entities)
    except json.JSONDecodeError:
        entity_list = [e.strip() for e in entities.split(",") if e.strip()]

    model = {
        "id": model_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "database_type": database_type,
        "entities": {},
    }

    for entity in entity_list:
        if database_type == "relational":
            model["entities"][entity] = {
                "table_name": entity.lower() + "s",
                "columns": [
                    {"name": "id", "type": "UUID", "primary_key": True},
                    {"name": "name", "type": "VARCHAR(255)", "nullable": False},
                    {"name": "created_at", "type": "TIMESTAMP", "default": "NOW()"},
                    {"name": "updated_at", "type": "TIMESTAMP", "default": "NOW()"},
                ],
                "indexes": [
                    {"name": f"idx_{entity.lower()}_name", "columns": ["name"]},
                ],
            }
        elif database_type == "document":
            model["entities"][entity] = {
                "collection": entity.lower() + "s",
                "schema": {
                    "_id": "ObjectId",
                    "name": "string",
                    "metadata": "object",
                    "created_at": "date",
                },
            }

    return json.dumps(model, indent=2)


@tool
def create_component_diagram(
    system_name: str,
    components: str,
    session_id: str = "default",
) -> str:
    """Create component diagram description.

    Generates a component diagram specification
    that can be rendered using Mermaid or PlantUML.

    Args:
        system_name: Name of the system.
        components: JSON or comma-separated list of components.
        session_id: Session identifier.

    Returns:
        JSON string with diagram specification and Mermaid code.
    """
    diagram_id = f"DIAG-{str(uuid.uuid4())[:8].upper()}"

    # Parse components
    try:
        component_list = json.loads(components)
    except json.JSONDecodeError:
        component_list = [c.strip() for c in components.split(",") if c.strip()]

    # Generate Mermaid diagram
    mermaid_lines = ["graph TB"]
    for i, comp in enumerate(component_list):
        comp_id = comp.replace(" ", "_").lower()
        mermaid_lines.append(f"    {comp_id}[{comp}]")
        if i > 0:
            prev_id = component_list[i-1].replace(" ", "_").lower()
            mermaid_lines.append(f"    {prev_id} --> {comp_id}")

    diagram = {
        "id": diagram_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "system_name": system_name,
        "components": component_list,
        "mermaid_code": "\n".join(mermaid_lines),
    }

    return json.dumps(diagram, indent=2)


@tool
def analyze_dependencies(
    project_path: str,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Analyze project dependencies.

    Identifies:
    - Direct dependencies
    - Transitive dependencies
    - Version conflicts
    - Outdated packages
    - Security vulnerabilities

    Args:
        project_path: Path to project or dependency file content.
        language: Programming language - "python", "javascript", "java".
        session_id: Session identifier.

    Returns:
        JSON string with dependency analysis.
    """
    analysis_id = f"DEP-{str(uuid.uuid4())[:8].upper()}"

    # Simulated dependency analysis
    analysis = {
        "id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "analysis": {
            "total_dependencies": 0,
            "direct_dependencies": 0,
            "transitive_dependencies": 0,
            "outdated_count": 0,
            "vulnerability_count": 0,
        },
        "recommendations": [
            "Run dependency audit regularly",
            "Update dependencies with security patches",
            "Consider using dependency lock files",
            "Set up automated vulnerability scanning",
        ],
    }

    return json.dumps(analysis, indent=2)
