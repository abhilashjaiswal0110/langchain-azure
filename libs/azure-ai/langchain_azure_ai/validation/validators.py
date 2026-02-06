"""Input validation and sanitization framework.

Provides comprehensive input validation to prevent security vulnerabilities
including SQL injection, prompt injection, XSS, and other attacks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Union

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels."""

    STRICT = "strict"  # Reject anything suspicious
    MODERATE = "moderate"  # Allow common patterns
    PERMISSIVE = "permissive"  # Minimal validation


class ThreatType(str, Enum):
    """Types of security threats."""

    SQL_INJECTION = "sql_injection"
    PROMPT_INJECTION = "prompt_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"


@dataclass
class ValidationResult:
    """Result of input validation.

    Attributes:
        is_valid: Whether the input passed validation.
        sanitized_value: The sanitized input value.
        original_value: The original input value.
        threats_detected: List of detected threats.
        warnings: List of validation warnings.
    """

    is_valid: bool
    sanitized_value: Any
    original_value: Any
    threats_detected: List[ThreatType] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return validity status."""
        return self.is_valid


@dataclass
class ValidationConfig:
    """Configuration for input validation.

    Attributes:
        level: Validation strictness level.
        max_string_length: Maximum allowed string length.
        max_list_length: Maximum allowed list length.
        max_dict_depth: Maximum allowed dictionary nesting depth.
        allow_unicode: Allow Unicode characters.
        allow_html: Allow HTML tags.
        log_threats: Log detected threats.
        raise_on_threat: Raise exception on threat detection.
    """

    level: ValidationLevel = ValidationLevel.MODERATE
    max_string_length: int = 10000
    max_list_length: int = 1000
    max_dict_depth: int = 10
    allow_unicode: bool = True
    allow_html: bool = False
    log_threats: bool = True
    raise_on_threat: bool = True


class InputValidator:
    """Centralized input validation and sanitization.

    Prevents SQL injection, prompt injection, XSS, and other attacks
    through pattern-based detection and sanitization.

    Example:
        >>> from langchain_azure_ai.validation import InputValidator, ValidationConfig
        >>>
        >>> validator = InputValidator(
        ...     config=ValidationConfig(
        ...         level=ValidationLevel.STRICT,
        ...         max_string_length=5000,
        ...     )
        ... )
        >>>
        >>> # Validate user input
        >>> result = validator.validate_string(user_input)
        >>> if result.is_valid:
        ...     safe_input = result.sanitized_value
        ... else:
        ...     print(f"Threats detected: {result.threats_detected}")
        >>>
        >>> # Validate prompt
        >>> try:
        ...     safe_prompt = validator.validate_prompt(user_prompt)
        ... except ValueError as e:
        ...     print(f"Invalid prompt: {e}")
    """

    # SQL injection patterns
    SQL_INJECTION_PATTERNS: List[Pattern] = [
        re.compile(r"(\s|^)(union|select|insert|update|delete|drop|create|alter|exec|execute)(\s|$|;)", re.IGNORECASE),
        re.compile(r"--", re.IGNORECASE),  # SQL comment
        re.compile(r"/\*.*?\*/", re.IGNORECASE | re.DOTALL),  # SQL block comment
        re.compile(r";\s*(union|select|insert|update|delete|drop)", re.IGNORECASE),
        re.compile(r"xp_\w+", re.IGNORECASE),  # Extended stored procedures
        re.compile(r"'(\s|;)*or\s+", re.IGNORECASE),  # OR injection
        re.compile(r"'(\s|;)*and\s+", re.IGNORECASE),  # AND injection
        re.compile(r"\b(or|and)\s+\d+\s*=\s*\d+", re.IGNORECASE),  # Numeric tautology
        re.compile(r"\b(or|and)\s+'[^']*'\s*=\s*'[^']*'", re.IGNORECASE),  # String tautology
    ]

    # Prompt injection patterns
    PROMPT_INJECTION_PATTERNS: List[Pattern] = [
        re.compile(r"ignore\s+(previous|above|all|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
        re.compile(r"disregard\s+(previous|above|all|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
        re.compile(r"forget\s+(previous|above|all|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(a|an|the)", re.IGNORECASE),
        re.compile(r"new\s+(instructions|role|persona|identity)\s*:", re.IGNORECASE),
        re.compile(r"system\s*:", re.IGNORECASE),
        re.compile(r"<\|im_start\|>", re.IGNORECASE),  # ChatML tags
        re.compile(r"<\|im_end\|>", re.IGNORECASE),
        re.compile(r"\[INST\]", re.IGNORECASE),  # Llama format
        re.compile(r"\[/INST\]", re.IGNORECASE),
        re.compile(r"<<SYS>>", re.IGNORECASE),  # System prompt markers
        re.compile(r"<</SYS>>", re.IGNORECASE),
        re.compile(r"###\s*(System|Human|Assistant)\s*:", re.IGNORECASE),  # Role markers
        re.compile(r"pretend\s+(you|to\s+be|that\s+you)", re.IGNORECASE),
        re.compile(r"act\s+as\s+(if|a|an|the)", re.IGNORECASE),
        re.compile(r"roleplay\s+as", re.IGNORECASE),
        re.compile(r"do\s+not\s+follow\s+(your|the|any)\s+", re.IGNORECASE),
    ]

    # XSS patterns
    XSS_PATTERNS: List[Pattern] = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"on\w+\s*=\s*[\"']", re.IGNORECASE),  # Event handlers
        re.compile(r"on\w+\s*=\s*[^\"'\s>]+", re.IGNORECASE),  # Unquoted handlers
        re.compile(r"<iframe[^>]*>", re.IGNORECASE),
        re.compile(r"<object[^>]*>", re.IGNORECASE),
        re.compile(r"<embed[^>]*>", re.IGNORECASE),
        re.compile(r"<img[^>]+onerror\s*=", re.IGNORECASE),
        re.compile(r"expression\s*\(", re.IGNORECASE),  # CSS expression
        re.compile(r"url\s*\(\s*[\"']?\s*javascript:", re.IGNORECASE),
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS: List[Pattern] = [
        re.compile(r"[;&|`$]", re.IGNORECASE),
        re.compile(r"\$\([^)]+\)", re.IGNORECASE),  # Command substitution
        re.compile(r"`[^`]+`", re.IGNORECASE),  # Backtick execution
        re.compile(r"\|{1,2}", re.IGNORECASE),  # Pipe
        re.compile(r"\n", re.IGNORECASE),  # Newline injection
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS: List[Pattern] = [
        re.compile(r"\.\./", re.IGNORECASE),
        re.compile(r"\.\.\\", re.IGNORECASE),
        re.compile(r"%2e%2e[/\\]", re.IGNORECASE),  # URL encoded
        re.compile(r"\.\.%2f", re.IGNORECASE),
        re.compile(r"\.\.%5c", re.IGNORECASE),
    ]

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize input validator.

        Args:
            config: Validation configuration.
        """
        self.config = config or ValidationConfig()

        logger.info(
            f"InputValidator initialized: level={self.config.level.value}"
        )

    def validate_string(
        self,
        value: str,
        *,
        min_length: int = 0,
        max_length: Optional[int] = None,
        allow_special_chars: bool = True,
        custom_pattern: Optional[str] = None,
    ) -> ValidationResult:
        """Validate and sanitize string input.

        Args:
            value: String to validate.
            min_length: Minimum length.
            max_length: Maximum length (overrides config).
            allow_special_chars: Allow special characters.
            custom_pattern: Custom regex pattern to match.

        Returns:
            ValidationResult with validation status and sanitized value.
        """
        threats: List[ThreatType] = []
        warnings: List[str] = []

        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(value),
                original_value=value,
                warnings=["Input is not a string"],
            )

        max_len = max_length or self.config.max_string_length

        # Length check
        if len(value) < min_length:
            warnings.append(f"String too short (min: {min_length})")
        if len(value) > max_len:
            value = value[:max_len]
            warnings.append(f"String truncated to {max_len} chars")

        # Special character check
        if not allow_special_chars:
            if not re.match(r"^[a-zA-Z0-9\s\-_.]*$", value):
                warnings.append("Special characters removed")
                value = re.sub(r"[^a-zA-Z0-9\s\-_.]", "", value)

        # SQL injection check
        if self.config.level in [ValidationLevel.STRICT, ValidationLevel.MODERATE]:
            for pattern in self.SQL_INJECTION_PATTERNS:
                if pattern.search(value):
                    threats.append(ThreatType.SQL_INJECTION)
                    if self.config.log_threats:
                        logger.warning(f"SQL injection pattern detected: {pattern.pattern}")
                    break

        # XSS check
        if self.config.level == ValidationLevel.STRICT:
            for pattern in self.XSS_PATTERNS:
                if pattern.search(value):
                    threats.append(ThreatType.XSS)
                    if self.config.log_threats:
                        logger.warning(f"XSS pattern detected: {pattern.pattern}")
                    break

        # Custom pattern validation
        if custom_pattern:
            if not re.match(custom_pattern, value):
                warnings.append("Does not match required pattern")

        # Sanitize
        sanitized = value.strip()

        # Determine validity
        is_valid = len(threats) == 0

        if not is_valid and self.config.raise_on_threat:
            msg = f"Security threat detected: {threats}"
            raise ValueError(msg)

        return ValidationResult(
            is_valid=is_valid,
            sanitized_value=sanitized,
            original_value=value,
            threats_detected=threats,
            warnings=warnings,
        )

    def validate_prompt(
        self,
        prompt: str,
        *,
        max_length: Optional[int] = None,
        check_injection: bool = True,
    ) -> str:
        """Validate LLM prompt for injection attempts.

        Args:
            prompt: User-provided prompt.
            max_length: Maximum prompt length.
            check_injection: Check for prompt injection patterns.

        Returns:
            Validated and sanitized prompt.

        Raises:
            ValueError: If validation fails.
        """
        max_len = max_length or self.config.max_string_length

        if len(prompt) > max_len:
            msg = f"Prompt too long (max: {max_len})"
            raise ValueError(msg)

        if check_injection:
            for pattern in self.PROMPT_INJECTION_PATTERNS:
                if pattern.search(prompt):
                    if self.config.log_threats:
                        logger.warning(
                            f"Prompt injection pattern detected: {pattern.pattern}"
                        )
                    msg = (
                        "Prompt contains potentially malicious content. "
                        "Please rephrase your request."
                    )
                    raise ValueError(msg)

        return prompt.strip()

    def validate_metadata_filter(
        self,
        filter_dict: Dict[str, Any],
        *,
        allowed_fields: Optional[List[str]] = None,
        allowed_operators: Optional[List[str]] = None,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """Validate metadata filter dictionary for vector search.

        Prevents SQL injection through filter parameters.

        Args:
            filter_dict: Filter dictionary for vector search.
            allowed_fields: Whitelist of allowed field names.
            allowed_operators: Whitelist of allowed operators.
            max_depth: Maximum nesting depth.

        Returns:
            Validated filter dictionary.

        Raises:
            ValueError: If validation fails.
        """
        default_operators = {
            "$eq", "$ne", "$lt", "$lte", "$gt", "$gte",
            "$in", "$nin", "$like", "$between", "$and", "$or",
        }
        valid_operators = set(allowed_operators) if allowed_operators else default_operators

        def validate_recursive(obj: Any, depth: int = 0) -> Any:
            if depth > max_depth:
                msg = f"Filter nesting too deep (max: {max_depth})"
                raise ValueError(msg)

            if isinstance(obj, dict):
                validated = {}
                for key, value in obj.items():
                    # Validate key format
                    if not re.match(r"^[a-zA-Z0-9_.$]+$", key):
                        msg = f"Invalid filter key: {key}"
                        raise ValueError(msg)

                    # Check if it's an operator
                    if key.startswith("$"):
                        if key not in valid_operators:
                            msg = f"Operator not allowed: {key}"
                            raise ValueError(msg)
                    elif allowed_fields and key not in allowed_fields:
                        msg = f"Field not allowed: {key}"
                        raise ValueError(msg)

                    validated[key] = validate_recursive(value, depth + 1)

                return validated

            elif isinstance(obj, list):
                return [validate_recursive(item, depth + 1) for item in obj]

            elif isinstance(obj, (str, int, float, bool, type(None))):
                if isinstance(obj, str):
                    # Check for SQL injection in string values
                    for pattern in self.SQL_INJECTION_PATTERNS:
                        if pattern.search(obj):
                            msg = "Filter value contains malicious content"
                            raise ValueError(msg)
                return obj

            else:
                msg = f"Unsupported filter value type: {type(obj)}"
                raise ValueError(msg)

        return validate_recursive(filter_dict)

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal.

        Args:
            filename: Filename to sanitize.

        Returns:
            Safe filename.
        """
        # Check for path traversal
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if pattern.search(filename):
                if self.config.log_threats:
                    logger.warning(f"Path traversal attempt detected: {filename}")
                # Remove traversal sequences
                filename = pattern.sub("", filename)

        # Remove path separators
        filename = filename.replace("/", "_").replace("\\", "_")

        # Remove null bytes
        filename = filename.replace("\x00", "")

        # Whitelist safe characters
        filename = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)

        # Limit length
        if len(filename) > 255:
            name, _, ext = filename.rpartition(".")
            if ext:
                filename = name[:250] + "." + ext
            else:
                filename = filename[:255]

        return filename

    def validate_url(
        self,
        url: str,
        *,
        allowed_schemes: Optional[List[str]] = None,
        allowed_domains: Optional[List[str]] = None,
    ) -> str:
        """Validate and sanitize URL.

        Args:
            url: URL to validate.
            allowed_schemes: Allowed URL schemes (default: https, http).
            allowed_domains: Allowed domains (None = allow all).

        Returns:
            Validated URL.

        Raises:
            ValueError: If URL is invalid or not allowed.
        """
        from urllib.parse import urlparse

        schemes = allowed_schemes or ["https", "http"]

        try:
            parsed = urlparse(url)
        except Exception as e:
            msg = f"Invalid URL format: {e}"
            raise ValueError(msg) from e

        if not parsed.scheme:
            msg = "URL must have a scheme (http/https)"
            raise ValueError(msg)

        if parsed.scheme not in schemes:
            msg = f"URL scheme not allowed: {parsed.scheme}"
            raise ValueError(msg)

        if allowed_domains and parsed.netloc not in allowed_domains:
            msg = f"Domain not allowed: {parsed.netloc}"
            raise ValueError(msg)

        # Check for JavaScript URLs
        if "javascript:" in url.lower():
            msg = "JavaScript URLs not allowed"
            raise ValueError(msg)

        return url

    def validate_email(self, email: str) -> str:
        """Validate email address format.

        Args:
            email: Email address to validate.

        Returns:
            Validated email.

        Raises:
            ValueError: If email is invalid.
        """
        email_pattern = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )

        email = email.strip().lower()

        if not email_pattern.match(email):
            msg = "Invalid email format"
            raise ValueError(msg)

        if len(email) > 254:  # RFC 5321
            msg = "Email too long"
            raise ValueError(msg)

        return email

    def validate_json(
        self,
        json_str: str,
        *,
        max_size: int = 1_000_000,  # 1MB default
    ) -> Dict[str, Any]:
        """Validate and parse JSON string.

        Args:
            json_str: JSON string to validate.
            max_size: Maximum size in bytes.

        Returns:
            Parsed JSON object.

        Raises:
            ValueError: If JSON is invalid or too large.
        """
        import json

        if len(json_str.encode("utf-8")) > max_size:
            msg = f"JSON too large (max: {max_size} bytes)"
            raise ValueError(msg)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON: {e}"
            raise ValueError(msg) from e

        return data


class SafePromptTemplate:
    """Template system with automatic input sanitization.

    Example:
        >>> template = SafePromptTemplate(
        ...     template="Answer the question: {question}\\nContext: {context}",
        ...     validators={
        ...         "question": lambda x: validator.validate_prompt(x, max_length=500),
        ...     }
        ... )
        >>>
        >>> safe_prompt = template.format(
        ...     question=user_question,
        ...     context=retrieved_context
        ... )
    """

    def __init__(
        self,
        template: str,
        *,
        validators: Optional[Dict[str, Callable[[str], str]]] = None,
        default_validator: Optional[InputValidator] = None,
    ):
        """Initialize safe template.

        Args:
            template: Prompt template with {placeholders}.
            validators: Custom validators for each placeholder.
            default_validator: Default validator for unspecified placeholders.
        """
        self.template = template
        self.validators = validators or {}
        self.default_validator = default_validator or InputValidator()

    def format(self, **kwargs: Any) -> str:
        """Format template with validated inputs.

        Args:
            **kwargs: Placeholder values.

        Returns:
            Formatted prompt with validated inputs.

        Raises:
            ValueError: If validation fails.
        """
        validated_kwargs = {}

        for key, value in kwargs.items():
            if key in self.validators:
                validated_kwargs[key] = self.validators[key](value)
            elif isinstance(value, str):
                validated_kwargs[key] = self.default_validator.validate_prompt(value)
            else:
                validated_kwargs[key] = value

        return self.template.format(**validated_kwargs)

    def format_safe(self, **kwargs: Any) -> tuple[str, List[str]]:
        """Format template with validation, returning errors instead of raising.

        Args:
            **kwargs: Placeholder values.

        Returns:
            Tuple of (formatted prompt or empty string, list of errors).
        """
        errors: List[str] = []
        validated_kwargs = {}

        for key, value in kwargs.items():
            try:
                if key in self.validators:
                    validated_kwargs[key] = self.validators[key](value)
                elif isinstance(value, str):
                    validated_kwargs[key] = self.default_validator.validate_prompt(value)
                else:
                    validated_kwargs[key] = value
            except ValueError as e:
                errors.append(f"{key}: {e}")

        if errors:
            return "", errors

        return self.template.format(**validated_kwargs), []


# Convenience functions
def validate_string(
    value: str,
    level: ValidationLevel = ValidationLevel.MODERATE,
) -> ValidationResult:
    """Quick string validation.

    Args:
        value: String to validate.
        level: Validation strictness.

    Returns:
        ValidationResult.
    """
    validator = InputValidator(ValidationConfig(level=level))
    return validator.validate_string(value)


def validate_prompt(
    prompt: str,
    max_length: int = 10000,
) -> str:
    """Quick prompt validation.

    Args:
        prompt: Prompt to validate.
        max_length: Maximum length.

    Returns:
        Validated prompt.

    Raises:
        ValueError: If validation fails.
    """
    validator = InputValidator()
    return validator.validate_prompt(prompt, max_length=max_length)


def sanitize_filename(filename: str) -> str:
    """Quick filename sanitization.

    Args:
        filename: Filename to sanitize.

    Returns:
        Safe filename.
    """
    validator = InputValidator()
    return validator.sanitize_filename(filename)
