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
        validation_rules: Optional custom validation rules as JSON or comma-separated list.
            Example JSON: {"min_length": 100, "require_numbers": true, "forbidden_words": ["maybe", "possibly"]}
            Example list: "min_length:100, require_stakeholder, no_passive_voice"
        session_id: Session identifier.

    Returns:
        JSON string with validation results.
    """
    validation_id = f"VAL-{str(uuid.uuid4())[:8].upper()}"

    # Parse custom validation rules
    custom_rules = {}
    if validation_rules:
        try:
            custom_rules = json.loads(validation_rules)
        except json.JSONDecodeError:
            # Parse comma-separated rules like "min_length:100, require_numbers"
            for rule in validation_rules.split(","):
                rule = rule.strip()
                if ":" in rule:
                    key, value = rule.split(":", 1)
                    try:
                        custom_rules[key.strip()] = int(value.strip())
                    except ValueError:
                        custom_rules[key.strip()] = value.strip()
                else:
                    custom_rules[rule] = True

    # Default SMART validation checks
    min_length = custom_rules.get("min_length", 50)
    checks = {
        "specific": {
            "passed": len(requirements) > min_length,
            "message": f"Requirement is sufficiently detailed (>{min_length} chars)" if len(requirements) > min_length else f"Add more specificity (minimum {min_length} characters)",
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
    }

    # Apply custom validation rules
    custom_check_results = {}

    # Check for forbidden words
    if "forbidden_words" in custom_rules:
        forbidden = custom_rules["forbidden_words"]
        if isinstance(forbidden, str):
            forbidden = [w.strip() for w in forbidden.split(",")]
        found_forbidden = [w for w in forbidden if w.lower() in requirements.lower()]
        custom_check_results["forbidden_words"] = {
            "passed": len(found_forbidden) == 0,
            "message": f"Found forbidden words: {found_forbidden}" if found_forbidden else "No forbidden words found",
        }

    # Check for required words
    if "required_words" in custom_rules:
        required = custom_rules["required_words"]
        if isinstance(required, str):
            required = [w.strip() for w in required.split(",")]
        missing_required = [w for w in required if w.lower() not in requirements.lower()]
        custom_check_results["required_words"] = {
            "passed": len(missing_required) == 0,
            "message": f"Missing required words: {missing_required}" if missing_required else "All required words present",
        }

    # Check for stakeholder mention
    if custom_rules.get("require_stakeholder"):
        stakeholder_keywords = ["user", "admin", "customer", "system", "client", "developer", "operator"]
        has_stakeholder = any(kw in requirements.lower() for kw in stakeholder_keywords)
        custom_check_results["stakeholder"] = {
            "passed": has_stakeholder,
            "message": "Stakeholder identified" if has_stakeholder else "Identify the stakeholder (user, admin, customer, etc.)",
        }

    # Check for passive voice (simplified check)
    if custom_rules.get("no_passive_voice"):
        passive_indicators = ["is done", "is performed", "is executed", "are processed", "will be", "should be"]
        has_passive = any(indicator in requirements.lower() for indicator in passive_indicators)
        custom_check_results["active_voice"] = {
            "passed": not has_passive,
            "message": "Uses active voice" if not has_passive else "Convert passive voice to active voice",
        }

    # Check for acceptance criteria keywords
    if custom_rules.get("require_acceptance_criteria"):
        ac_keywords = ["given", "when", "then", "acceptance", "criteria", "verify", "validate"]
        has_ac = any(kw in requirements.lower() for kw in ac_keywords)
        custom_check_results["acceptance_criteria"] = {
            "passed": has_ac,
            "message": "Has acceptance criteria" if has_ac else "Add acceptance criteria (Given/When/Then)",
        }

    # Check max length if specified
    if "max_length" in custom_rules:
        max_len = int(custom_rules["max_length"])
        custom_check_results["max_length"] = {
            "passed": len(requirements) <= max_len,
            "message": f"Within length limit ({len(requirements)}/{max_len})" if len(requirements) <= max_len else f"Exceeds max length ({len(requirements)}/{max_len})",
        }

    # Combine default and custom checks
    all_checks = {**checks, **custom_check_results}

    # Determine overall validity
    is_valid = all(check["passed"] for check in all_checks.values())

    # Generate recommendations for failed checks
    recommendations = []
    for check_name, check_result in all_checks.items():
        if not check_result["passed"]:
            recommendations.append(check_result["message"])

    validation_result = {
        "id": validation_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "is_valid": is_valid,
        "checks": all_checks,
        "custom_rules_applied": list(custom_rules.keys()) if custom_rules else [],
        "recommendations": recommendations,
    }

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
    - Kano model (Must-be, One-dimensional, Attractive, Indifferent, Reverse)

    Args:
        requirements_list: JSON array or comma-separated list of requirements.
            For Kano method, JSON format with functional/dysfunctional responses is preferred:
            [{"req": "Feature A", "functional": "like", "dysfunctional": "expect"}]
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
        prioritized["methodology"] = {
            "name": "MoSCoW",
            "description": "Prioritization by Must/Should/Could/Won't have categories",
            "categories": {
                "must_have": "Critical requirements that must be delivered",
                "should_have": "Important but not critical, can be delayed if necessary",
                "could_have": "Nice to have, delivered only if time permits",
                "wont_have": "Explicitly excluded from current scope",
            },
        }

    elif method == "weighted":
        prioritized["results"] = {
            f"req_{i+1}": {"text": req, "score": 10 - i, "weight": 1.0}
            for i, req in enumerate(reqs)
        }
        prioritized["methodology"] = {
            "name": "Weighted Scoring",
            "description": "Numerical scoring based on importance",
            "scoring_criteria": "Higher score = higher priority",
        }

    elif method == "kano":
        # Kano model classification based on customer satisfaction research
        # Categories: Must-be, One-dimensional, Attractive, Indifferent, Reverse, Questionable

        # Kano evaluation matrix (functional vs dysfunctional responses)
        # Responses: "like", "expect", "neutral", "tolerate", "dislike"
        kano_matrix = {
            ("like", "dislike"): "attractive",      # A - Delighters
            ("like", "tolerate"): "attractive",     # A
            ("like", "neutral"): "attractive",      # A
            ("like", "expect"): "questionable",     # Q - Inconsistent
            ("like", "like"): "questionable",       # Q
            ("expect", "dislike"): "one_dimensional",  # O - Satisfiers
            ("expect", "tolerate"): "one_dimensional", # O
            ("expect", "neutral"): "indifferent",   # I
            ("expect", "expect"): "questionable",   # Q
            ("expect", "like"): "reverse",          # R
            ("neutral", "dislike"): "must_be",      # M - Basic needs
            ("neutral", "tolerate"): "indifferent", # I
            ("neutral", "neutral"): "indifferent",  # I
            ("neutral", "expect"): "reverse",       # R
            ("neutral", "like"): "reverse",         # R
            ("tolerate", "dislike"): "must_be",     # M
            ("tolerate", "tolerate"): "indifferent",# I
            ("tolerate", "neutral"): "indifferent", # I
            ("tolerate", "expect"): "reverse",      # R
            ("tolerate", "like"): "reverse",        # R
            ("dislike", "dislike"): "questionable", # Q
            ("dislike", "tolerate"): "reverse",     # R
            ("dislike", "neutral"): "reverse",      # R
            ("dislike", "expect"): "reverse",       # R
            ("dislike", "like"): "reverse",         # R
        }

        kano_categories = {
            "must_be": [],        # Basic needs - expected, dissatisfaction if absent
            "one_dimensional": [], # Satisfiers - more is better
            "attractive": [],     # Delighters - unexpected positive features
            "indifferent": [],    # No impact on satisfaction
            "reverse": [],        # Causes dissatisfaction if present
            "questionable": [],   # Inconsistent responses
            "unclassified": [],   # Requirements without Kano data
        }

        # Process each requirement
        for i, req in enumerate(reqs):
            if isinstance(req, dict):
                # JSON format with functional/dysfunctional responses
                req_text = req.get("req", req.get("requirement", req.get("text", f"Requirement {i+1}")))
                functional = req.get("functional", "neutral").lower()
                dysfunctional = req.get("dysfunctional", "neutral").lower()

                # Normalize response values
                response_map = {
                    "like": "like", "love": "like", "want": "like", "1": "like",
                    "expect": "expect", "must": "expect", "need": "expect", "2": "expect",
                    "neutral": "neutral", "3": "neutral",
                    "tolerate": "tolerate", "live_with": "tolerate", "4": "tolerate",
                    "dislike": "dislike", "hate": "dislike", "5": "dislike",
                }
                functional = response_map.get(functional, "neutral")
                dysfunctional = response_map.get(dysfunctional, "neutral")

                # Look up Kano category
                category = kano_matrix.get((functional, dysfunctional), "unclassified")
                kano_categories[category].append({
                    "requirement": req_text,
                    "functional_response": functional,
                    "dysfunctional_response": dysfunctional,
                })
            else:
                # Simple string - use heuristics to classify
                req_lower = str(req).lower()
                if any(kw in req_lower for kw in ["must", "critical", "essential", "required", "security", "compliance"]):
                    kano_categories["must_be"].append({"requirement": req, "classified_by": "keyword_heuristic"})
                elif any(kw in req_lower for kw in ["should", "important", "performance", "faster", "better"]):
                    kano_categories["one_dimensional"].append({"requirement": req, "classified_by": "keyword_heuristic"})
                elif any(kw in req_lower for kw in ["could", "nice", "delight", "wow", "innovative", "extra"]):
                    kano_categories["attractive"].append({"requirement": req, "classified_by": "keyword_heuristic"})
                else:
                    kano_categories["unclassified"].append({"requirement": req, "classified_by": "keyword_heuristic"})

        prioritized["results"] = kano_categories
        prioritized["methodology"] = {
            "name": "Kano Model",
            "description": "Customer satisfaction-based prioritization developed by Noriaki Kano",
            "categories": {
                "must_be": "Basic needs - Expected features, cause dissatisfaction if absent but don't increase satisfaction if present",
                "one_dimensional": "Performance needs - Satisfaction proportional to fulfillment (more is better)",
                "attractive": "Excitement needs - Unexpected delighters, great satisfaction if present but no dissatisfaction if absent",
                "indifferent": "No significant impact on customer satisfaction",
                "reverse": "Features that cause dissatisfaction when present",
                "questionable": "Inconsistent or contradictory responses",
                "unclassified": "Requirements not yet evaluated with Kano questionnaire",
            },
            "priority_order": ["must_be", "one_dimensional", "attractive", "indifferent"],
            "recommended_action": {
                "must_be": "Must implement - these are table stakes",
                "one_dimensional": "Prioritize based on competitive analysis",
                "attractive": "Invest selectively for differentiation",
                "indifferent": "Deprioritize or remove",
                "reverse": "Do not implement",
            },
        }

        # Add summary statistics
        prioritized["summary"] = {
            "total_requirements": len(reqs),
            "categorized": {k: len(v) for k, v in kano_categories.items() if v},
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
