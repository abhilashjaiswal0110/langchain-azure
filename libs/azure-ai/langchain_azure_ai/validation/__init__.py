"""Input validation and sanitization framework.

This module provides comprehensive input validation to prevent:
- SQL injection attacks
- Prompt injection attacks
- XSS (Cross-Site Scripting) attacks
- Path traversal attacks
- Command injection attacks

Usage:
    from langchain_azure_ai.validation import (
        InputValidator,
        ValidationConfig,
        ValidationLevel,
        SafePromptTemplate,
    )

    # Initialize validator
    validator = InputValidator(
        config=ValidationConfig(
            level=ValidationLevel.STRICT,
            max_string_length=5000,
        )
    )

    # Validate user input
    result = validator.validate_string(user_input)
    if result.is_valid:
        safe_input = result.sanitized_value
    else:
        print(f"Threats: {result.threats_detected}")

    # Validate prompts
    try:
        safe_prompt = validator.validate_prompt(user_prompt)
    except ValueError as e:
        print(f"Invalid prompt: {e}")
"""

from langchain_azure_ai.validation.validators import (
    InputValidator,
    SafePromptTemplate,
    ThreatType,
    ValidationConfig,
    ValidationLevel,
    ValidationResult,
    sanitize_filename,
    validate_prompt,
    validate_string,
)

__all__ = [
    "InputValidator",
    "ValidationConfig",
    "ValidationLevel",
    "ValidationResult",
    "ThreatType",
    "SafePromptTemplate",
    "validate_string",
    "validate_prompt",
    "sanitize_filename",
]
