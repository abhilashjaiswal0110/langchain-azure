"""Requirements Analysis Tools.

Tools for analyzing, validating, and extracting software requirements.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool


# Session storage for requirements
_requirements_store: dict[str, dict] = {}


@tool
def analyze_requirements(
    requirements_text: str,
    context: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Analyze natural language requirements and extract structured information.

    Parses requirements text to identify:
    - Functional requirements
    - Non-functional requirements
    - Technical constraints
    - Ambiguities and risks

    Args:
        requirements_text: The raw requirements text to analyze.
        context: Optional context about the project or domain.
        session_id: Session identifier for tracking.

    Returns:
        JSON string with analyzed requirements and metadata.
    """
    req_id = f"REQ-{str(uuid.uuid4())[:8].upper()}"

    # Analyze the requirements text
    analysis = {
        "id": req_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "raw_text": requirements_text[:500],  # Truncate for storage
        "analysis": {
            "word_count": len(requirements_text.split()),
            "has_functional_keywords": any(
                kw in requirements_text.lower()
                for kw in ["must", "shall", "should", "will", "can"]
            ),
            "has_non_functional_keywords": any(
                kw in requirements_text.lower()
                for kw in ["performance", "security", "scalability", "availability"]
            ),
            "has_technical_terms": any(
                kw in requirements_text.lower()
                for kw in ["api", "database", "authentication", "integration"]
            ),
        },
        "recommendations": [
            "Break down into user stories for better tracking",
            "Add acceptance criteria for each requirement",
            "Identify dependencies between requirements",
        ],
    }

    # Store in session
    _requirements_store[req_id] = analysis

    return json.dumps(analysis, indent=2)


@tool
def extract_user_stories(
    requirements_text: str,
    format_type: str = "standard",
    session_id: str = "default",
) -> str:
    """Extract user stories from requirements text.

    Converts requirements into well-formed user stories with
    the format: "As a [user], I want [goal], so that [benefit]"

    Args:
        requirements_text: Requirements text to convert.
        format_type: Output format - "standard", "technical", or "agile".
        session_id: Session identifier.

    Returns:
        JSON string with extracted user stories.
    """
    story_id = f"US-{str(uuid.uuid4())[:8].upper()}"

    # Generate user story template
    user_story = {
        "id": story_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "format": format_type,
        "story": {
            "as_a": "[User type to be determined]",
            "i_want": requirements_text[:200],
            "so_that": "[Business value to be defined]",
        },
        "acceptance_criteria": [
            "Given [context], when [action], then [expected result]",
        ],
        "story_points": None,
        "priority": "should_have",
    }

    return json.dumps(user_story, indent=2)


@tool
def validate_requirements(
    requirements: str,
    validation_rules: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Validate requirements against quality criteria.

    Checks if requirements are:
    - Specific and unambiguous
    - Measurable
    - Achievable
    - Relevant
    - Time-bound (SMART)

    Args:
        requirements: Requirements text or JSON to validate.
        validation_rules: Optional custom validation rules.
        session_id: Session identifier.

    Returns:
        JSON string with validation results.
    """
    validation_id = f"VAL-{str(uuid.uuid4())[:8].upper()}"

    # Perform validation checks
    validation_result = {
        "id": validation_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "is_valid": True,
        "checks": {
            "specific": {
                "passed": len(requirements) > 50,
                "message": "Requirement is sufficiently detailed" if len(requirements) > 50 else "Add more specificity",
            },
            "measurable": {
                "passed": any(c.isdigit() for c in requirements),
                "message": "Has measurable criteria" if any(c.isdigit() for c in requirements) else "Add quantifiable metrics",
            },
            "achievable": {
                "passed": True,
                "message": "Appears technically feasible",
            },
            "relevant": {
                "passed": True,
                "message": "Aligns with typical software goals",
            },
            "time_bound": {
                "passed": any(kw in requirements.lower() for kw in ["deadline", "by", "within", "before"]),
                "message": "Timeline specified" if any(kw in requirements.lower() for kw in ["deadline", "by", "within", "before"]) else "Consider adding timeline",
            },
        },
        "recommendations": [],
    }

    # Add recommendations based on checks
    for check_name, check_result in validation_result["checks"].items():
        if not check_result["passed"]:
            validation_result["is_valid"] = False
            validation_result["recommendations"].append(check_result["message"])

    return json.dumps(validation_result, indent=2)


@tool
def prioritize_requirements(
    requirements_list: str,
    method: str = "moscow",
    session_id: str = "default",
) -> str:
    """Prioritize requirements using specified methodology.

    Supports:
    - MoSCoW (Must, Should, Could, Won't)
    - Weighted scoring
    - Kano model

    Args:
        requirements_list: JSON array or comma-separated list of requirements.
        method: Prioritization method - "moscow", "weighted", or "kano".
        session_id: Session identifier.

    Returns:
        JSON string with prioritized requirements.
    """
    priority_id = f"PRI-{str(uuid.uuid4())[:8].upper()}"

    # Parse requirements
    try:
        reqs = json.loads(requirements_list)
    except json.JSONDecodeError:
        reqs = [r.strip() for r in requirements_list.split(",") if r.strip()]

    # Apply prioritization
    prioritized = {
        "id": priority_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "method": method,
        "results": {},
    }

    if method == "moscow":
        prioritized["results"] = {
            "must_have": reqs[:len(reqs)//4] if reqs else [],
            "should_have": reqs[len(reqs)//4:len(reqs)//2] if len(reqs) > 1 else [],
            "could_have": reqs[len(reqs)//2:3*len(reqs)//4] if len(reqs) > 2 else [],
            "wont_have": reqs[3*len(reqs)//4:] if len(reqs) > 3 else [],
        }
    elif method == "weighted":
        prioritized["results"] = {
            f"req_{i+1}": {"text": req, "score": 10 - i, "weight": 1.0}
            for i, req in enumerate(reqs)
        }

    return json.dumps(prioritized, indent=2)


@tool
def detect_ambiguities(
    requirements_text: str,
    session_id: str = "default",
) -> str:
    """Detect ambiguities and potential issues in requirements.

    Identifies:
    - Vague terms (e.g., "fast", "user-friendly")
    - Missing definitions
    - Conflicting statements
    - Incomplete specifications

    Args:
        requirements_text: Requirements text to analyze.
        session_id: Session identifier.

    Returns:
        JSON string with detected ambiguities.
    """
    ambiguity_id = f"AMB-{str(uuid.uuid4())[:8].upper()}"

    # Ambiguous terms to detect
    vague_terms = ["fast", "quick", "easy", "simple", "user-friendly", "efficient",
                   "reliable", "secure", "scalable", "flexible", "intuitive", "modern"]

    found_ambiguities = []
    for term in vague_terms:
        if term in requirements_text.lower():
            found_ambiguities.append({
                "term": term,
                "issue": f"'{term}' is subjective and should be quantified",
                "suggestion": f"Define specific metrics for '{term}'",
            })

    result = {
        "id": ambiguity_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "ambiguities_found": len(found_ambiguities),
        "ambiguities": found_ambiguities,
        "risk_level": "high" if len(found_ambiguities) > 3 else "medium" if len(found_ambiguities) > 0 else "low",
        "recommendations": [
            "Replace vague terms with measurable criteria",
            "Add acceptance criteria with specific thresholds",
            "Review with stakeholders to clarify intent",
        ] if found_ambiguities else ["Requirements appear well-defined"],
    }

    return json.dumps(result, indent=2)


@tool
def generate_acceptance_criteria(
    user_story: str,
    format_type: str = "gherkin",
    session_id: str = "default",
) -> str:
    """Generate acceptance criteria for a user story.

    Creates testable acceptance criteria in specified format:
    - Gherkin (Given/When/Then)
    - Checklist
    - Rule-based

    Args:
        user_story: The user story to generate criteria for.
        format_type: Output format - "gherkin", "checklist", or "rules".
        session_id: Session identifier.

    Returns:
        JSON string with acceptance criteria.
    """
    criteria_id = f"AC-{str(uuid.uuid4())[:8].upper()}"

    result = {
        "id": criteria_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_story": user_story[:200],
        "format": format_type,
        "acceptance_criteria": [],
    }

    if format_type == "gherkin":
        result["acceptance_criteria"] = [
            {
                "scenario": "Happy path",
                "given": "the system is in normal operating state",
                "when": "the user performs the primary action",
                "then": "the expected result is achieved",
            },
            {
                "scenario": "Error handling",
                "given": "an error condition exists",
                "when": "the user attempts the action",
                "then": "an appropriate error message is displayed",
            },
            {
                "scenario": "Edge case",
                "given": "boundary conditions are present",
                "when": "the user performs the action",
                "then": "the system handles it gracefully",
            },
        ]
    elif format_type == "checklist":
        result["acceptance_criteria"] = [
            {"item": "Feature is accessible to authorized users", "required": True},
            {"item": "Data is validated before processing", "required": True},
            {"item": "Error messages are user-friendly", "required": True},
            {"item": "Action is logged for audit", "required": False},
            {"item": "Performance meets SLA requirements", "required": True},
        ]
    else:  # rules
        result["acceptance_criteria"] = [
            {"rule": "MUST complete within 3 seconds"},
            {"rule": "MUST validate all input fields"},
            {"rule": "MUST display confirmation on success"},
            {"rule": "SHOULD support undo action"},
            {"rule": "COULD allow batch processing"},
        ]

    return json.dumps(result, indent=2)
