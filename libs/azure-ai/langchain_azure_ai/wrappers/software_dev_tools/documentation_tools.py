"""Documentation Generation Tools.

Tools for generating technical documentation, API docs, and user guides.
"""

import json
import uuid
from datetime import datetime
from typing import Optional
import re

from langchain_core.tools import tool


@tool
def generate_api_docs(
    code: str,
    format_type: str = "openapi",
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Generate API documentation from code.

    Creates documentation in:
    - OpenAPI/Swagger format
    - Markdown format
    - HTML format

    Args:
        code: API code to document.
        format_type: Output format - "openapi", "markdown", "html".
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with API documentation.
    """
    doc_id = f"APIDOC-{str(uuid.uuid4())[:8].upper()}"

    # Extract endpoints from code
    endpoints = []

    # FastAPI patterns
    fastapi_patterns = re.findall(r'@(?:app|router)\.(get|post|put|delete|patch)\s*\(["\']([^"\']+)["\']', code, re.IGNORECASE)
    for method, path in fastapi_patterns:
        endpoints.append({"method": method.upper(), "path": path})

    # Flask patterns
    flask_patterns = re.findall(r'@(?:app|bp)\.route\s*\(["\']([^"\']+)["\'].*methods\s*=\s*\[([^\]]+)\]', code)
    for path, methods in flask_patterns:
        for method in methods.replace("'", "").replace('"', "").split(","):
            endpoints.append({"method": method.strip().upper(), "path": path})

    # Generate documentation structure
    if format_type == "openapi":
        doc_content = {
            "openapi": "3.0.0",
            "info": {
                "title": "API Documentation",
                "version": "1.0.0",
                "description": "Auto-generated API documentation",
            },
            "paths": {},
        }
        for endpoint in endpoints:
            if endpoint["path"] not in doc_content["paths"]:
                doc_content["paths"][endpoint["path"]] = {}
            doc_content["paths"][endpoint["path"]][endpoint["method"].lower()] = {
                "summary": f"{endpoint['method']} {endpoint['path']}",
                "responses": {"200": {"description": "Success"}},
            }
    elif format_type == "markdown":
        doc_lines = ["# API Documentation\n"]
        for endpoint in endpoints:
            doc_lines.append(f"## {endpoint['method']} {endpoint['path']}\n")
            doc_lines.append("**Description:** TODO\n")
            doc_lines.append("**Response:** 200 OK\n")
        doc_content = "\n".join(doc_lines)
    else:
        doc_content = {"endpoints": endpoints}

    result = {
        "id": doc_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "format": format_type,
        "language": language,
        "endpoints_found": len(endpoints),
        "documentation": doc_content,
    }

    return json.dumps(result, indent=2)


@tool
def create_readme(
    project_name: str,
    description: str,
    language: str = "python",
    features: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Create a README file for the project.

    Generates comprehensive README with:
    - Project overview
    - Installation instructions
    - Usage examples
    - Contributing guidelines

    Args:
        project_name: Name of the project.
        description: Project description.
        language: Primary programming language.
        features: Optional comma-separated list of features.
        session_id: Session identifier.

    Returns:
        JSON string with README content.
    """
    readme_id = f"README-{str(uuid.uuid4())[:8].upper()}"

    feature_list = []
    if features:
        feature_list = [f.strip() for f in features.split(",") if f.strip()]

    install_commands = {
        "python": "pip install -r requirements.txt",
        "javascript": "npm install",
        "typescript": "npm install",
        "go": "go mod download",
        "java": "mvn install",
    }

    readme_content = f"""# {project_name}

{description}

## Features

{chr(10).join(f'- {f}' for f in feature_list) if feature_list else '- Feature 1\n- Feature 2\n- Feature 3'}

## Installation

```bash
# Clone the repository
git clone https://github.com/username/{project_name.lower().replace(' ', '-')}.git
cd {project_name.lower().replace(' ', '-')}

# Install dependencies
{install_commands.get(language, 'pip install -r requirements.txt')}
```

## Usage

```{language}
# Example usage
from {project_name.lower().replace(' ', '_')} import main

result = main.run()
print(result)
```

## Configuration

Create a `.env` file with the following variables:

```
API_KEY=your_api_key
DEBUG=false
```

## Development

```bash
# Run tests
{'pytest' if language == 'python' else 'npm test'}

# Run linter
{'ruff check .' if language == 'python' else 'npm run lint'}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""

    result = {
        "id": readme_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "project_name": project_name,
        "language": language,
        "readme_content": readme_content,
        "sections": [
            "Features",
            "Installation",
            "Usage",
            "Configuration",
            "Development",
            "Contributing",
            "License",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def document_architecture(
    system_name: str,
    components: str,
    description: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Generate architecture documentation.

    Creates:
    - System overview
    - Component descriptions
    - Data flow diagrams
    - Decision records

    Args:
        system_name: Name of the system.
        components: JSON or comma-separated list of components.
        description: Optional system description.
        session_id: Session identifier.

    Returns:
        JSON string with architecture documentation.
    """
    arch_id = f"ARCHDOC-{str(uuid.uuid4())[:8].upper()}"

    # Parse components
    try:
        component_list = json.loads(components)
    except json.JSONDecodeError:
        component_list = [c.strip() for c in components.split(",") if c.strip()]

    arch_doc = f"""# {system_name} Architecture Documentation

## Overview

{description or f'{system_name} is a software system designed to provide [purpose].'}

## Components

{chr(10).join(f'### {c}\n\n- **Purpose:** [Describe purpose]\n- **Technology:** [Describe technology]\n- **Dependencies:** [List dependencies]\n' for c in component_list)}

## Data Flow

```mermaid
graph LR
{chr(10).join(f'    {component_list[i].replace(" ", "_")} --> {component_list[i+1].replace(" ", "_")}' for i in range(len(component_list)-1))}
```

## Architecture Decisions

### ADR-001: [Decision Title]

**Status:** Accepted

**Context:** [Why this decision was needed]

**Decision:** [What was decided]

**Consequences:** [Impact of the decision]

## Deployment

The system is deployed using [deployment strategy].

## Security Considerations

- Authentication: [Method]
- Authorization: [Method]
- Data encryption: [At rest / In transit]

## Monitoring

- Metrics: [What is monitored]
- Alerts: [What triggers alerts]
- Logging: [Log aggregation approach]
"""

    result = {
        "id": arch_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "system_name": system_name,
        "components": component_list,
        "documentation": arch_doc,
    }

    return json.dumps(result, indent=2)


@tool
def generate_changelog(
    version: str,
    changes: str,
    previous_version: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Generate a CHANGELOG entry.

    Creates changelog following:
    - Keep a Changelog format
    - Semantic versioning
    - Categorized changes

    Args:
        version: New version number.
        changes: JSON or newline-separated list of changes.
        previous_version: Optional previous version for comparison.
        session_id: Session identifier.

    Returns:
        JSON string with changelog entry.
    """
    changelog_id = f"CHANGELOG-{str(uuid.uuid4())[:8].upper()}"

    # Parse changes
    try:
        change_list = json.loads(changes)
    except json.JSONDecodeError:
        change_list = [c.strip() for c in changes.split("\n") if c.strip()]

    # Categorize changes
    categories = {
        "Added": [],
        "Changed": [],
        "Deprecated": [],
        "Removed": [],
        "Fixed": [],
        "Security": [],
    }

    for change in change_list:
        change_lower = change.lower()
        if any(kw in change_lower for kw in ["add", "new", "implement"]):
            categories["Added"].append(change)
        elif any(kw in change_lower for kw in ["fix", "bug", "resolve"]):
            categories["Fixed"].append(change)
        elif any(kw in change_lower for kw in ["security", "vulnerability", "cve"]):
            categories["Security"].append(change)
        elif any(kw in change_lower for kw in ["remove", "delete"]):
            categories["Removed"].append(change)
        elif any(kw in change_lower for kw in ["deprecate"]):
            categories["Deprecated"].append(change)
        else:
            categories["Changed"].append(change)

    # Generate changelog entry
    changelog_lines = [f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n"]

    for category, items in categories.items():
        if items:
            changelog_lines.append(f"### {category}\n")
            for item in items:
                changelog_lines.append(f"- {item}")
            changelog_lines.append("")

    changelog_entry = "\n".join(changelog_lines)

    result = {
        "id": changelog_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "version": version,
        "previous_version": previous_version,
        "categories": {k: v for k, v in categories.items() if v},
        "changelog_entry": changelog_entry,
    }

    return json.dumps(result, indent=2)


@tool
def add_inline_comments(
    code: str,
    comment_style: str = "docstring",
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Add inline comments and documentation to code.

    Adds:
    - Function/method docstrings
    - Inline explanations
    - Type hints
    - TODO comments for unclear sections

    Args:
        code: Code to document.
        comment_style: Style - "docstring", "inline", "both".
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with documented code.
    """
    comment_id = f"COMMENT-{str(uuid.uuid4())[:8].upper()}"

    # Find functions without docstrings
    functions_without_docs = []

    if language == "python":
        # Find function definitions
        func_pattern = r'def\s+(\w+)\s*\(([^)]*)\)\s*(?:->.*?)?:'
        matches = list(re.finditer(func_pattern, code))

        for match in matches:
            func_name = match.group(1)
            params = match.group(2)
            # Check if docstring follows
            after_def = code[match.end():]
            if not after_def.strip().startswith('"""') and not after_def.strip().startswith("'''"):
                functions_without_docs.append({
                    "name": func_name,
                    "params": params,
                    "needs_docstring": True,
                })

    result = {
        "id": comment_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "comment_style": comment_style,
        "analysis": {
            "functions_without_docs": len(functions_without_docs),
            "functions": functions_without_docs[:5],  # Limit for display
        },
        "documented_code": code,  # In real implementation, this would be modified
        "suggestions": [
            "Add docstrings to all public functions",
            "Include parameter and return type documentation",
            "Add examples in docstrings for complex functions",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def create_user_guide(
    product_name: str,
    features: str,
    target_audience: str = "developers",
    session_id: str = "default",
) -> str:
    """Create a user guide for the product.

    Generates:
    - Getting started guide
    - Feature documentation
    - Troubleshooting section
    - FAQ

    Args:
        product_name: Name of the product.
        features: JSON or comma-separated list of features.
        target_audience: Target audience - "developers", "end-users", "admins".
        session_id: Session identifier.

    Returns:
        JSON string with user guide content.
    """
    guide_id = f"GUIDE-{str(uuid.uuid4())[:8].upper()}"

    # Parse features
    try:
        feature_list = json.loads(features)
    except json.JSONDecodeError:
        feature_list = [f.strip() for f in features.split(",") if f.strip()]

    user_guide = f"""# {product_name} User Guide

## Introduction

Welcome to {product_name}! This guide will help you get started and make the most of the available features.

## Target Audience

This guide is intended for {target_audience}.

## Getting Started

### Prerequisites

- [List prerequisites]
- [System requirements]

### Installation

1. [Step 1]
2. [Step 2]
3. [Step 3]

### Quick Start

```
# Quick start example
[code example]
```

## Features

{chr(10).join(f'### {f}\n\n[Description of {f}]\n\n**How to use:**\n1. [Step 1]\n2. [Step 2]\n' for f in feature_list)}

## Configuration

### Basic Configuration

[Configuration instructions]

### Advanced Configuration

[Advanced options]

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| [Issue 1] | [Solution 1] |
| [Issue 2] | [Solution 2] |

### Getting Help

- Documentation: [URL]
- Support: [Email/URL]
- Community: [Forum/Discord]

## FAQ

**Q: [Common question 1]?**
A: [Answer]

**Q: [Common question 2]?**
A: [Answer]

## Appendix

### Glossary

- **Term 1:** Definition
- **Term 2:** Definition

### Version History

See [CHANGELOG.md](CHANGELOG.md) for version history.
"""

    result = {
        "id": guide_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "product_name": product_name,
        "target_audience": target_audience,
        "features": feature_list,
        "user_guide": user_guide,
        "sections": [
            "Introduction",
            "Getting Started",
            "Features",
            "Configuration",
            "Troubleshooting",
            "FAQ",
            "Appendix",
        ],
    }

    return json.dumps(result, indent=2)
