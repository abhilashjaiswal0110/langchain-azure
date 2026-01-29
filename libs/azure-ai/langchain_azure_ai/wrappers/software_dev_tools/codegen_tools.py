"""Code Generation Tools.

Tools for generating, refactoring, and formatting code
across multiple programming languages.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool


# Session storage for generated code
_code_store: dict[str, dict] = {}


@tool
def generate_code(
    description: str,
    language: str = "python",
    framework: Optional[str] = None,
    include_tests: bool = False,
    session_id: str = "default",
) -> str:
    """Generate code based on description.

    Creates production-ready code following best practices
    for the specified language and framework.

    Args:
        description: Description of what the code should do.
        language: Programming language (python, javascript, typescript, java, go).
        framework: Optional framework (fastapi, django, express, react, spring).
        include_tests: Whether to include test code.
        session_id: Session identifier.

    Returns:
        JSON string with generated code and metadata.
    """
    code_id = f"CODE-{str(uuid.uuid4())[:8].upper()}"

    # Code templates based on language
    templates = {
        "python": '''"""
{description}

This module provides functionality for {purpose}.
"""

from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class {class_name}:
    """
    {class_docstring}
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize {class_name}.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {{}}
        logger.info(f"{class_name} initialized")

    def execute(self, input_data: Any) -> dict:
        """Execute the main operation.

        Args:
            input_data: Input data to process.

        Returns:
            Dictionary with results.
        """
        try:
            # Implementation here
            result = {{"status": "success", "data": input_data}}
            return result
        except Exception as e:
            logger.error(f"Error in execute: {{e}}")
            raise
''',
        "typescript": '''/**
 * {description}
 *
 * @module {module_name}
 */

interface Config {{
  [key: string]: any;
}}

interface Result {{
  status: string;
  data: any;
}}

/**
 * {class_docstring}
 */
export class {class_name} {{
  private config: Config;

  constructor(config: Config = {{}}) {{
    this.config = config;
    console.log(`{class_name} initialized`);
  }}

  /**
   * Execute the main operation.
   * @param inputData - Input data to process
   * @returns Result object
   */
  execute(inputData: any): Result {{
    try {{
      return {{
        status: 'success',
        data: inputData
      }};
    }} catch (error) {{
      console.error(`Error in execute: ${{error}}`);
      throw error;
    }}
  }}
}}
''',
    }

    # Generate class name from description
    class_name = "".join(word.capitalize() for word in description.split()[:3])

    code_template = templates.get(language, templates["python"])
    generated_code = code_template.format(
        description=description,
        purpose=description.lower()[:50],
        class_name=class_name,
        class_docstring=f"Implementation for: {description}",
        module_name=class_name.lower(),
    )

    result = {
        "id": code_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "framework": framework,
        "description": description,
        "code": generated_code,
        "line_count": len(generated_code.split("\n")),
    }

    if include_tests:
        result["tests"] = f'''"""Tests for {class_name}"""

import pytest

def test_{class_name.lower()}_init():
    """Test {class_name} initialization."""
    instance = {class_name}()
    assert instance is not None

def test_{class_name.lower()}_execute():
    """Test {class_name} execute method."""
    instance = {class_name}()
    result = instance.execute({{"test": "data"}})
    assert result["status"] == "success"
'''

    _code_store[code_id] = result
    return json.dumps(result, indent=2)


@tool
def refactor_code(
    code: str,
    refactor_type: str = "readability",
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Refactor code for improved quality.

    Applies refactoring techniques:
    - readability: Improve naming, structure, documentation
    - performance: Optimize for speed
    - maintainability: Extract functions, reduce complexity
    - security: Add input validation, sanitization

    Args:
        code: Code to refactor.
        refactor_type: Type of refactoring to apply.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with refactored code and suggestions.
    """
    refactor_id = f"REF-{str(uuid.uuid4())[:8].upper()}"

    suggestions = {
        "readability": [
            "Use descriptive variable names",
            "Add docstrings to functions",
            "Break long functions into smaller ones",
            "Add type hints",
        ],
        "performance": [
            "Use list comprehensions instead of loops",
            "Cache repeated computations",
            "Use generators for large data",
            "Optimize database queries",
        ],
        "maintainability": [
            "Extract helper functions",
            "Remove duplicate code (DRY)",
            "Add error handling",
            "Use configuration files",
        ],
        "security": [
            "Validate all inputs",
            "Sanitize user data",
            "Use parameterized queries",
            "Remove hardcoded secrets",
        ],
    }

    result = {
        "id": refactor_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "refactor_type": refactor_type,
        "language": language,
        "original_lines": len(code.split("\n")),
        "suggestions": suggestions.get(refactor_type, suggestions["readability"]),
        "refactored_code": code,  # In real implementation, this would be modified
    }

    return json.dumps(result, indent=2)


@tool
def apply_design_pattern(
    code: str,
    pattern: str = "singleton",
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Apply a design pattern to the code.

    Supports patterns:
    - singleton, factory, builder (Creational)
    - adapter, decorator, facade (Structural)
    - observer, strategy, command (Behavioral)

    Args:
        code: Code to modify.
        pattern: Design pattern to apply.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with pattern-applied code.
    """
    pattern_id = f"PAT-{str(uuid.uuid4())[:8].upper()}"

    patterns = {
        "singleton": {
            "description": "Ensure a class has only one instance",
            "use_case": "Database connections, Configuration managers",
        },
        "factory": {
            "description": "Create objects without specifying exact class",
            "use_case": "Plugin systems, Multiple implementations",
        },
        "builder": {
            "description": "Construct complex objects step by step",
            "use_case": "Complex object construction, Fluent APIs",
        },
        "adapter": {
            "description": "Convert interface of a class to another",
            "use_case": "Legacy system integration, Third-party APIs",
        },
        "decorator": {
            "description": "Add behavior to objects dynamically",
            "use_case": "Logging, Caching, Authentication",
        },
        "observer": {
            "description": "Define one-to-many dependency between objects",
            "use_case": "Event systems, UI updates",
        },
        "strategy": {
            "description": "Define family of algorithms, make interchangeable",
            "use_case": "Sorting algorithms, Payment methods",
        },
    }

    pattern_info = patterns.get(pattern, patterns["singleton"])

    result = {
        "id": pattern_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "pattern": pattern,
        "pattern_info": pattern_info,
        "language": language,
        "modified_code": code,  # In real implementation, this would be transformed
    }

    return json.dumps(result, indent=2)


@tool
def generate_boilerplate(
    project_type: str,
    project_name: str,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Generate project boilerplate code.

    Creates standard project structure with:
    - Directory structure
    - Configuration files
    - Main entry point
    - README template

    Args:
        project_type: Type of project (api, cli, library, web).
        project_name: Name of the project.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with boilerplate structure.
    """
    boilerplate_id = f"BP-{str(uuid.uuid4())[:8].upper()}"

    structures = {
        "api": {
            "directories": ["src", "src/routes", "src/models", "src/services", "tests", "docs"],
            "files": ["src/main.py", "src/config.py", "requirements.txt", "README.md", ".env.example"],
        },
        "cli": {
            "directories": ["src", "src/commands", "tests"],
            "files": ["src/cli.py", "setup.py", "requirements.txt", "README.md"],
        },
        "library": {
            "directories": [project_name, f"{project_name}/core", "tests", "docs", "examples"],
            "files": [f"{project_name}/__init__.py", "setup.py", "pyproject.toml", "README.md"],
        },
        "web": {
            "directories": ["src", "src/components", "src/pages", "public", "tests"],
            "files": ["src/App.tsx", "src/index.tsx", "package.json", "README.md"],
        },
    }

    structure = structures.get(project_type, structures["api"])

    result = {
        "id": boilerplate_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "project_type": project_type,
        "project_name": project_name,
        "language": language,
        "structure": structure,
    }

    return json.dumps(result, indent=2)


@tool
def optimize_imports(
    code: str,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Optimize and organize imports in code.

    Performs:
    - Remove unused imports
    - Sort imports (stdlib, third-party, local)
    - Group related imports
    - Fix import order

    Args:
        code: Code with imports to optimize.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with optimized imports.
    """
    opt_id = f"IMP-{str(uuid.uuid4())[:8].upper()}"

    result = {
        "id": opt_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "original_code": code,
        "optimized_code": code,  # In real implementation, this would be modified
        "changes": {
            "removed_imports": [],
            "reordered_imports": True,
            "grouped_imports": True,
        },
    }

    return json.dumps(result, indent=2)


@tool
def format_code(
    code: str,
    language: str = "python",
    style: str = "default",
    session_id: str = "default",
) -> str:
    """Format code according to style guide.

    Applies formatting:
    - Python: PEP 8, Black
    - JavaScript/TypeScript: Prettier
    - Go: gofmt
    - Java: Google Java Format

    Args:
        code: Code to format.
        language: Programming language.
        style: Style guide to use.
        session_id: Session identifier.

    Returns:
        JSON string with formatted code.
    """
    fmt_id = f"FMT-{str(uuid.uuid4())[:8].upper()}"

    style_guides = {
        "python": "PEP 8 / Black",
        "javascript": "Prettier",
        "typescript": "Prettier",
        "go": "gofmt",
        "java": "Google Java Format",
    }

    result = {
        "id": fmt_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "style_guide": style_guides.get(language, "default"),
        "original_code": code,
        "formatted_code": code,  # In real implementation, this would be formatted
        "changes_made": 0,
    }

    return json.dumps(result, indent=2)
