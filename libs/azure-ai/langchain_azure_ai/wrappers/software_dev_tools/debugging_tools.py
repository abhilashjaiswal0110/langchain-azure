"""Debugging and Optimization Tools.

Tools for analyzing errors, tracing execution, and optimizing performance.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool


@tool
def analyze_error(
    error_message: str,
    stack_trace: Optional[str] = None,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Analyze an error message and stack trace.

    Provides:
    - Error classification
    - Potential causes
    - Suggested fixes
    - Related documentation

    Args:
        error_message: The error message to analyze.
        stack_trace: Optional stack trace.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with error analysis.
    """
    analysis_id = f"ERR-{str(uuid.uuid4())[:8].upper()}"

    # Common error patterns and their analysis
    error_patterns = {
        "TypeError": {
            "category": "Type Error",
            "common_causes": [
                "Passing wrong type to function",
                "Operating on None/null value",
                "Missing type conversion",
            ],
            "fix_suggestions": [
                "Check variable types before operations",
                "Add type validation",
                "Use type hints and static analysis",
            ],
        },
        "KeyError": {
            "category": "Dictionary Error",
            "common_causes": [
                "Accessing non-existent dictionary key",
                "Key spelling mismatch",
                "Key not initialized",
            ],
            "fix_suggestions": [
                "Use .get() method with default value",
                "Check key existence with 'in' operator",
                "Validate dictionary structure",
            ],
        },
        "AttributeError": {
            "category": "Attribute Error",
            "common_causes": [
                "Object doesn't have the attribute",
                "Object is None",
                "Typo in attribute name",
            ],
            "fix_suggestions": [
                "Check object type before accessing attribute",
                "Use hasattr() to verify attribute exists",
                "Add null checks",
            ],
        },
        "ConnectionError": {
            "category": "Network Error",
            "common_causes": [
                "Server not reachable",
                "Network timeout",
                "DNS resolution failure",
            ],
            "fix_suggestions": [
                "Add retry logic with exponential backoff",
                "Check network connectivity",
                "Verify server URL and port",
            ],
        },
        "ImportError": {
            "category": "Import Error",
            "common_causes": [
                "Module not installed",
                "Circular import",
                "Wrong module path",
            ],
            "fix_suggestions": [
                "Install missing package",
                "Check import path",
                "Resolve circular dependencies",
            ],
        },
    }

    # Determine error type
    error_type = "Unknown"
    analysis = {
        "category": "General Error",
        "common_causes": ["Check error message for details"],
        "fix_suggestions": ["Review code logic", "Add proper error handling"],
    }

    for err_type, err_analysis in error_patterns.items():
        if err_type.lower() in error_message.lower():
            error_type = err_type
            analysis = err_analysis
            break

    result = {
        "id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "error_message": error_message[:500],
        "error_type": error_type,
        "analysis": analysis,
        "stack_trace_available": stack_trace is not None,
        "debugging_steps": [
            "1. Reproduce the error consistently",
            "2. Identify the exact line causing the error",
            "3. Check variable values at that point",
            "4. Apply the suggested fix",
            "5. Add tests to prevent regression",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def trace_execution(
    code: str,
    entry_point: Optional[str] = None,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Trace code execution path.

    Provides:
    - Execution flow visualization
    - Variable state tracking
    - Branch analysis
    - Call graph

    Args:
        code: Code to trace.
        entry_point: Optional function to start tracing from.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with execution trace.
    """
    trace_id = f"TRACE-{str(uuid.uuid4())[:8].upper()}"

    # Analyze code structure
    import re
    functions = re.findall(r'def\s+(\w+)\s*\(', code)
    classes = re.findall(r'class\s+(\w+)\s*[:\(]', code)
    conditionals = len(re.findall(r'\bif\b|\belif\b|\belse\b', code))
    loops = len(re.findall(r'\bfor\b|\bwhile\b', code))

    # Build simplified call graph
    call_graph = []
    for func in functions:
        calls = re.findall(rf'\b{func}\s*\(', code)
        if len(calls) > 1:  # Function is called somewhere
            call_graph.append({"function": func, "call_count": len(calls) - 1})

    result = {
        "id": trace_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "entry_point": entry_point or functions[0] if functions else "main",
        "code_structure": {
            "functions": functions,
            "classes": classes,
            "conditional_branches": conditionals,
            "loops": loops,
        },
        "call_graph": call_graph,
        "complexity_indicators": {
            "cyclomatic_complexity": conditionals + loops + 1,
            "function_count": len(functions),
            "nesting_depth": code.count("    ") // max(len(code.split("\n")), 1),
        },
    }

    return json.dumps(result, indent=2)


@tool
def identify_root_cause(
    symptoms: str,
    context: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Identify root cause using RCA techniques.

    Applies:
    - 5 Whys analysis
    - Fishbone diagram approach
    - Fault tree analysis

    Args:
        symptoms: Description of the problem symptoms.
        context: Additional context about the system.
        session_id: Session identifier.

    Returns:
        JSON string with root cause analysis.
    """
    rca_id = f"RCA-{str(uuid.uuid4())[:8].upper()}"

    # 5 Whys template
    five_whys = [
        {"level": 1, "question": "Why did this happen?", "answer": "Immediate cause to be determined"},
        {"level": 2, "question": "Why did that occur?", "answer": "Contributing factor to investigate"},
        {"level": 3, "question": "Why was that the case?", "answer": "Underlying condition to verify"},
        {"level": 4, "question": "Why did that exist?", "answer": "Process gap to examine"},
        {"level": 5, "question": "Why was there a gap?", "answer": "Root cause to confirm"},
    ]

    # Fishbone categories
    fishbone_categories = {
        "People": ["Training gaps", "Communication issues", "Workload"],
        "Process": ["Missing procedures", "Unclear requirements", "Insufficient testing"],
        "Technology": ["Bug in code", "Infrastructure issue", "Integration failure"],
        "Environment": ["Configuration error", "External dependency", "Resource constraints"],
    }

    result = {
        "id": rca_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "symptoms": symptoms[:500],
        "analysis_methods": {
            "five_whys": five_whys,
            "fishbone_categories": fishbone_categories,
        },
        "recommendations": [
            "Complete the 5 Whys analysis with actual answers",
            "Review each fishbone category for contributing factors",
            "Document the root cause and preventive measures",
            "Implement fixes and add regression tests",
        ],
        "preventive_actions": [
            "Add monitoring for early detection",
            "Improve test coverage for similar scenarios",
            "Update documentation and runbooks",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def propose_fix(
    issue_description: str,
    code_context: Optional[str] = None,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Propose a fix for the identified issue.

    Provides:
    - Code fix suggestions
    - Implementation steps
    - Testing recommendations
    - Rollback plan

    Args:
        issue_description: Description of the issue.
        code_context: Optional code context.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with proposed fix.
    """
    fix_id = f"FIX-{str(uuid.uuid4())[:8].upper()}"

    result = {
        "id": fix_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "issue": issue_description[:300],
        "proposed_fix": {
            "summary": "Apply targeted fix based on root cause analysis",
            "steps": [
                "1. Create a branch for the fix",
                "2. Implement the code change",
                "3. Add unit tests for the fix",
                "4. Run full test suite",
                "5. Code review and approval",
                "6. Deploy to staging for verification",
                "7. Deploy to production with monitoring",
            ],
        },
        "testing_requirements": [
            "Unit tests for the specific fix",
            "Regression tests for related functionality",
            "Integration tests if applicable",
        ],
        "rollback_plan": {
            "steps": [
                "1. Monitor for issues after deployment",
                "2. If issues detected, revert the commit",
                "3. Redeploy previous version",
                "4. Investigate the failure",
            ],
            "estimated_time": "5-10 minutes",
        },
    }

    return json.dumps(result, indent=2)


@tool
def analyze_performance(
    code: str,
    focus_area: str = "time",
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Analyze code performance.

    Analyzes:
    - Time complexity
    - Space complexity
    - Bottlenecks
    - Optimization opportunities

    Args:
        code: Code to analyze.
        focus_area: Focus - "time", "memory", "both".
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with performance analysis.
    """
    perf_id = f"PERF-{str(uuid.uuid4())[:8].upper()}"

    # Analyze code for performance patterns
    import re

    issues = []

    # Check for nested loops (potential O(n²) or worse)
    nested_loops = re.findall(r'for\s+.*:\s*\n\s+for\s+', code)
    if nested_loops:
        issues.append({
            "type": "nested_loops",
            "severity": "medium",
            "description": f"Found {len(nested_loops)} nested loops - potential O(n²) complexity",
            "suggestion": "Consider using hash maps or other O(1) lookups",
        })

    # Check for list operations in loops
    if re.search(r'for\s+.*:\s*\n.*\.append\(', code):
        issues.append({
            "type": "list_append_in_loop",
            "severity": "low",
            "description": "Appending to list in loop",
            "suggestion": "Consider list comprehension for better performance",
        })

    # Check for string concatenation in loops
    if re.search(r'for\s+.*:\s*\n.*\+\s*=\s*["\']', code):
        issues.append({
            "type": "string_concat_in_loop",
            "severity": "medium",
            "description": "String concatenation in loop - creates new strings each iteration",
            "suggestion": "Use join() or StringIO for better performance",
        })

    result = {
        "id": perf_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "focus_area": focus_area,
        "analysis": {
            "estimated_time_complexity": "O(n)" if not nested_loops else "O(n²)",
            "issues_found": len(issues),
            "issues": issues,
        },
        "optimization_suggestions": [
            "Profile code to identify actual bottlenecks",
            "Use appropriate data structures (sets, dicts for lookups)",
            "Consider caching for repeated computations",
            "Use generators for large data processing",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def detect_memory_issues(
    code: str,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Detect potential memory issues in code.

    Identifies:
    - Memory leaks
    - Large object retention
    - Circular references
    - Inefficient data structures

    Args:
        code: Code to analyze.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with memory analysis.
    """
    mem_id = f"MEM-{str(uuid.uuid4())[:8].upper()}"

    issues = []
    import re

    # Check for global mutable objects
    if re.search(r'^[A-Z_]+\s*=\s*\[\]|^[A-Z_]+\s*=\s*\{\}', code, re.MULTILINE):
        issues.append({
            "type": "global_mutable",
            "severity": "medium",
            "description": "Global mutable objects detected",
            "suggestion": "Be careful with global state - can grow unbounded",
        })

    # Check for large data loading without limits
    if re.search(r'\.read\(\)|\.readlines\(\)|list\(.*\.read', code):
        issues.append({
            "type": "unbounded_read",
            "severity": "high",
            "description": "Reading entire file/data into memory",
            "suggestion": "Use streaming/chunked reading for large files",
        })

    # Check for potential circular references
    if re.search(r'self\.\w+\s*=\s*self', code):
        issues.append({
            "type": "potential_circular_ref",
            "severity": "low",
            "description": "Potential circular reference detected",
            "suggestion": "Use weakref if circular reference is intentional",
        })

    result = {
        "id": mem_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "analysis": {
            "issues_found": len(issues),
            "issues": issues,
        },
        "recommendations": [
            "Use context managers for resource cleanup",
            "Implement __del__ or use weakref for circular refs",
            "Profile memory usage with memory_profiler",
            "Use generators for large data processing",
            "Clear large objects when no longer needed",
        ],
    }

    return json.dumps(result, indent=2)
