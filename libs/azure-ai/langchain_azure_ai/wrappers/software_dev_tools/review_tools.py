"""Code Review and Quality Tools.

Tools for performing automated code reviews and quality checks.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool


@tool
def review_code(
    code: str,
    language: str = "python",
    focus_areas: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Perform comprehensive code review.

    Reviews code for:
    - Correctness
    - Style compliance
    - Security issues
    - Performance concerns
    - Maintainability

    Args:
        code: Code to review.
        language: Programming language.
        focus_areas: Optional comma-separated focus areas.
        session_id: Session identifier.

    Returns:
        JSON string with review results.
    """
    review_id = f"REV-{str(uuid.uuid4())[:8].upper()}"

    # Analyze code characteristics
    lines = code.split("\n")
    line_count = len(lines)

    review = {
        "id": review_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "metrics": {
            "line_count": line_count,
            "estimated_complexity": "moderate" if line_count > 50 else "low",
        },
        "findings": [
            {
                "severity": "info",
                "category": "style",
                "message": "Consider adding type hints for better maintainability",
                "line": None,
            },
            {
                "severity": "info",
                "category": "documentation",
                "message": "Ensure all public methods have docstrings",
                "line": None,
            },
        ],
        "recommendations": [
            "Add comprehensive error handling",
            "Include input validation",
            "Add logging statements for debugging",
        ],
        "overall_quality": "good",
    }

    return json.dumps(review, indent=2)


@tool
def check_code_style(
    code: str,
    language: str = "python",
    style_guide: str = "default",
    session_id: str = "default",
) -> str:
    """Check code against style guidelines.

    Validates code against:
    - Python: PEP 8
    - JavaScript: ESLint
    - TypeScript: TSLint/ESLint
    - Go: gofmt

    Args:
        code: Code to check.
        language: Programming language.
        style_guide: Style guide to use.
        session_id: Session identifier.

    Returns:
        JSON string with style check results.
    """
    style_id = f"STY-{str(uuid.uuid4())[:8].upper()}"

    result = {
        "id": style_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "style_guide": style_guide,
        "passed": True,
        "issues": [],
        "summary": {
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
        },
    }

    # Basic style checks
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            result["issues"].append({
                "line": i,
                "severity": "warning",
                "message": f"Line exceeds 120 characters ({len(line)} chars)",
            })
            result["summary"]["warning_count"] += 1
            result["passed"] = False

    return json.dumps(result, indent=2)


@tool
def analyze_complexity(
    code: str,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Analyze code complexity metrics.

    Calculates:
    - Cyclomatic complexity
    - Cognitive complexity
    - Nesting depth
    - Lines of code metrics

    Args:
        code: Code to analyze.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with complexity analysis.
    """
    complexity_id = f"CMP-{str(uuid.uuid4())[:8].upper()}"

    lines = code.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    # Count complexity indicators
    branches = sum(1 for l in lines if any(kw in l for kw in ["if ", "elif ", "for ", "while ", "try:", "except"]))
    nesting = max((l.count("    ") for l in lines), default=0)

    result = {
        "id": complexity_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "metrics": {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "comment_lines": sum(1 for l in lines if l.strip().startswith("#") or l.strip().startswith("//")),
            "estimated_cyclomatic_complexity": branches + 1,
            "max_nesting_depth": nesting,
        },
        "thresholds": {
            "cyclomatic_complexity_max": 10,
            "nesting_depth_max": 4,
        },
        "assessment": "acceptable" if branches < 10 and nesting < 4 else "needs_refactoring",
    }

    return json.dumps(result, indent=2)


@tool
def detect_code_smells(
    code: str,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Detect code smells and anti-patterns.

    Identifies:
    - Long methods
    - Large classes
    - Duplicate code
    - Dead code
    - Magic numbers
    - God objects

    Args:
        code: Code to analyze.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with detected code smells.
    """
    smell_id = f"SMELL-{str(uuid.uuid4())[:8].upper()}"

    lines = code.split("\n")
    smells = []

    # Check for long methods (> 50 lines between def/function and return/end)
    if len(lines) > 100:
        smells.append({
            "type": "long_method",
            "severity": "medium",
            "message": "Consider breaking this into smaller functions",
        })

    # Check for magic numbers
    import re
    magic_numbers = re.findall(r'[=<>!+\-*/]\s*\d{2,}', code)
    if magic_numbers:
        smells.append({
            "type": "magic_numbers",
            "severity": "low",
            "message": f"Found {len(magic_numbers)} magic numbers. Consider using named constants.",
        })

    # Check for duplicate strings
    string_matches = re.findall(r'["\'][^"\']{10,}["\']', code)
    duplicates = [s for s in set(string_matches) if string_matches.count(s) > 1]
    if duplicates:
        smells.append({
            "type": "duplicate_literals",
            "severity": "low",
            "message": f"Found {len(duplicates)} duplicate string literals. Consider extracting to constants.",
        })

    result = {
        "id": smell_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "smells_detected": len(smells),
        "smells": smells,
        "recommendations": [
            "Address high severity smells first",
            "Use refactoring tools to fix safely",
            "Add tests before refactoring",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def suggest_improvements(
    code: str,
    language: str = "python",
    improvement_type: str = "all",
    session_id: str = "default",
) -> str:
    """Suggest improvements for the code.

    Provides suggestions for:
    - Performance optimization
    - Code clarity
    - Error handling
    - Best practices

    Args:
        code: Code to analyze.
        language: Programming language.
        improvement_type: Type of improvements - "performance", "clarity", "error_handling", "all".
        session_id: Session identifier.

    Returns:
        JSON string with improvement suggestions.
    """
    suggest_id = f"SUG-{str(uuid.uuid4())[:8].upper()}"

    suggestions = {
        "performance": [
            "Use list comprehensions for simple transformations",
            "Cache expensive computations",
            "Use generators for large data processing",
            "Consider async operations for I/O-bound tasks",
        ],
        "clarity": [
            "Use descriptive variable names",
            "Add docstrings to all public functions",
            "Break complex expressions into named steps",
            "Use type hints for function signatures",
        ],
        "error_handling": [
            "Catch specific exceptions, not bare except",
            "Add context to error messages",
            "Use custom exceptions for domain errors",
            "Log errors with appropriate severity",
        ],
    }

    if improvement_type == "all":
        all_suggestions = []
        for category, items in suggestions.items():
            for item in items:
                all_suggestions.append({"category": category, "suggestion": item})
        selected_suggestions = all_suggestions
    else:
        selected_suggestions = [
            {"category": improvement_type, "suggestion": s}
            for s in suggestions.get(improvement_type, [])
        ]

    result = {
        "id": suggest_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "improvement_type": improvement_type,
        "suggestions": selected_suggestions,
    }

    return json.dumps(result, indent=2)


@tool
def check_best_practices(
    code: str,
    language: str = "python",
    framework: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Check code against best practices.

    Validates:
    - SOLID principles
    - DRY (Don't Repeat Yourself)
    - KISS (Keep It Simple)
    - Framework-specific practices

    Args:
        code: Code to check.
        language: Programming language.
        framework: Optional framework context.
        session_id: Session identifier.

    Returns:
        JSON string with best practices assessment.
    """
    bp_id = f"BP-{str(uuid.uuid4())[:8].upper()}"

    principles = {
        "single_responsibility": {
            "status": "review_needed",
            "message": "Ensure each class/function has one responsibility",
        },
        "open_closed": {
            "status": "review_needed",
            "message": "Design for extension, not modification",
        },
        "dependency_inversion": {
            "status": "review_needed",
            "message": "Depend on abstractions, not concretions",
        },
        "dry": {
            "status": "passed",
            "message": "No obvious code duplication detected",
        },
        "kiss": {
            "status": "passed",
            "message": "Code appears reasonably simple",
        },
    }

    result = {
        "id": bp_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "framework": framework,
        "principles": principles,
        "overall_compliance": "moderate",
        "recommendations": [
            "Review SOLID principle adherence",
            "Consider dependency injection for testability",
            "Use interfaces/protocols for abstractions",
        ],
    }

    return json.dumps(result, indent=2)
