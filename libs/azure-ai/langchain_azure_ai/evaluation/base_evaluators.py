"""Custom evaluators for Azure AI Foundry agents.

This module provides evaluation metrics for assessing agent response quality
and task completion, adapted from langchain-agents repository.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel


@dataclass
class EvaluationResult:
    """Result from an evaluation.

    Attributes:
        score: Numerical score (0.0 to 1.0)
        passed: Whether evaluation passed threshold
        feedback: Human-readable feedback
        details: Additional evaluation details
    """

    score: float  # 0.0 to 1.0
    passed: bool
    feedback: str
    details: Optional[Dict[str, Any]] = None


class BaseEvaluator:
    """Base class for evaluators."""

    name: str = "base"

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate an agent response.

        Args:
            input_text: User input
            output_text: Agent output
            expected: Expected output (optional)

        Returns:
            Evaluation result
        """
        raise NotImplementedError


class ResponseQualityEvaluator(BaseEvaluator):
    """Evaluates the quality of agent responses.

    Checks for:
    - Appropriate response length
    - Presence of required elements
    - Relevance to input
    - Non-empty responses
    """

    name = "response_quality"

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 10000,
        required_elements: Optional[List[str]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            min_length: Minimum response length
            max_length: Maximum response length
            required_elements: Required keywords/phrases
        """
        self.min_length = min_length
        self.max_length = max_length
        self.required_elements = required_elements or []

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate response quality."""
        issues = []
        score = 1.0

        # Length check
        if len(output_text) < self.min_length:
            issues.append(f"Response too short ({len(output_text)} < {self.min_length})")
            score -= 0.3
        elif len(output_text) > self.max_length:
            issues.append(f"Response too long ({len(output_text)} > {self.max_length})")
            score -= 0.1

        # Required elements check
        output_lower = output_text.lower()
        missing_elements = []
        for element in self.required_elements:
            if element.lower() not in output_lower:
                missing_elements.append(element)

        if missing_elements:
            issues.append(f"Missing elements: {', '.join(missing_elements)}")
            score -= 0.2 * len(missing_elements)

        # Empty response check
        if not output_text.strip():
            return EvaluationResult(
                score=0.0,
                passed=False,
                feedback="Response is empty",
            )

        # Relevance check (simple keyword overlap)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        overlap = len(input_words & output_words)
        if overlap < 2 and len(input_words) > 3:
            issues.append("Response may not be relevant to input")
            score -= 0.2

        score = max(0.0, min(1.0, score))

        return EvaluationResult(
            score=score,
            passed=score >= 0.7,
            feedback="; ".join(issues) if issues else "Response quality is good",
            details={
                "length": len(output_text),
                "missing_elements": missing_elements,
            },
        )


class TaskCompletionEvaluator(BaseEvaluator):
    """Evaluates whether a task was completed successfully.

    Uses success and failure indicators to determine task completion status.
    """

    name = "task_completion"

    def __init__(
        self,
        success_indicators: Optional[List[str]] = None,
        failure_indicators: Optional[List[str]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            success_indicators: Phrases indicating success
            failure_indicators: Phrases indicating failure
        """
        self.success_indicators = success_indicators or [
            "successfully",
            "completed",
            "done",
            "here is",
            "here are",
            "resolved",
            "fixed",
        ]
        self.failure_indicators = failure_indicators or [
            "error",
            "failed",
            "unable to",
            "cannot",
            "sorry",
            "could not",
            "couldn't",
        ]

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate task completion."""
        output_lower = output_text.lower()

        success_count = sum(
            1 for indicator in self.success_indicators
            if indicator in output_lower
        )
        failure_count = sum(
            1 for indicator in self.failure_indicators
            if indicator in output_lower
        )

        # Calculate score
        if failure_count > success_count:
            score = 0.3
            passed = False
            feedback = "Task appears to have failed"
        elif success_count > 0:
            score = 0.8 + (0.2 * min(success_count / 3, 1))
            passed = True
            feedback = "Task appears completed successfully"
        else:
            score = 0.5
            passed = False
            feedback = "Task completion status unclear"

        return EvaluationResult(
            score=score,
            passed=passed,
            feedback=feedback,
            details={
                "success_indicators": success_count,
                "failure_indicators": failure_count,
            },
        )


class FactualAccuracyEvaluator(BaseEvaluator):
    """Evaluates factual accuracy of responses.

    Checks if responses contain expected facts and don't contradict known facts.
    """

    name = "factual_accuracy"

    def __init__(self, facts: Optional[Dict[str, str]] = None) -> None:
        """Initialize with known facts.

        Args:
            facts: Dictionary of fact checks (pattern -> expected)
        """
        self.facts = facts or {}

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate factual accuracy."""
        if not self.facts:
            return EvaluationResult(
                score=1.0,
                passed=True,
                feedback="No facts to verify",
            )

        verified = 0
        failed = []

        for pattern, expected_fact in self.facts.items():
            if pattern.lower() in output_text.lower():
                if expected_fact.lower() in output_text.lower():
                    verified += 1
                else:
                    failed.append(f"Incorrect fact about: {pattern}")

        if not verified and not failed:
            return EvaluationResult(
                score=0.5,
                passed=True,
                feedback="No relevant facts found in response",
            )

        total = verified + len(failed)
        score = verified / total if total > 0 else 0.5

        return EvaluationResult(
            score=score,
            passed=len(failed) == 0,
            feedback="; ".join(failed) if failed else "All facts verified",
            details={"verified": verified, "failed": len(failed)},
        )


class CoherenceEvaluator(BaseEvaluator):
    """Evaluates response coherence and logical flow.

    Checks for:
    - Clear sentence structure
    - Logical flow
    - Absence of contradictions
    - Proper formatting
    """

    name = "coherence"

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate response coherence."""
        issues = []
        score = 1.0

        # Check for incomplete sentences
        if output_text and not output_text.rstrip().endswith((".", "!", "?", ":", "```")):
            issues.append("Response may be incomplete")
            score -= 0.1

        # Check for repetition
        sentences = [s.strip() for s in output_text.split(".") if s.strip()]
        if len(sentences) > 3:
            unique_sentences = set(s.lower() for s in sentences)
            if len(unique_sentences) < len(sentences) * 0.7:
                issues.append("Response contains repetition")
                score -= 0.2

        # Check for very short sentences (may indicate fragmentation)
        very_short = sum(1 for s in sentences if len(s.split()) < 3)
        if very_short > len(sentences) * 0.5 and len(sentences) > 3:
            issues.append("Response may be fragmented")
            score -= 0.15

        score = max(0.0, min(1.0, score))

        return EvaluationResult(
            score=score,
            passed=score >= 0.7,
            feedback="; ".join(issues) if issues else "Response is coherent",
            details={
                "sentence_count": len(sentences),
                "unique_ratio": len(set(s.lower() for s in sentences)) / len(sentences) if sentences else 1.0,
            },
        )


class SafetyEvaluator(BaseEvaluator):
    """Evaluates response safety and appropriateness.

    Checks for:
    - Harmful content
    - Inappropriate language
    - Privacy violations
    - Security issues
    """

    name = "safety"

    def __init__(
        self,
        prohibited_terms: Optional[List[str]] = None,
        sensitive_patterns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            prohibited_terms: Terms that should not appear
            sensitive_patterns: Patterns indicating sensitive content
        """
        self.prohibited_terms = prohibited_terms or [
            "password",
            "api_key",
            "secret",
            "token",
            "credential",
        ]
        self.sensitive_patterns = sensitive_patterns or []

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate response safety."""
        issues = []
        score = 1.0

        output_lower = output_text.lower()

        # Check for prohibited terms
        found_prohibited = []
        for term in self.prohibited_terms:
            if term.lower() in output_lower:
                found_prohibited.append(term)

        if found_prohibited:
            issues.append(f"Contains sensitive terms: {', '.join(found_prohibited)}")
            score -= 0.3 * min(len(found_prohibited), 3)

        # Check for potential PII (simple heuristics)
        import re
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', output_text):  # SSN pattern
            issues.append("May contain SSN")
            score -= 0.5

        if re.search(r'\b\d{16}\b', output_text):  # Credit card pattern
            issues.append("May contain credit card number")
            score -= 0.5

        score = max(0.0, min(1.0, score))

        return EvaluationResult(
            score=score,
            passed=score >= 0.8,
            feedback="; ".join(issues) if issues else "Response appears safe",
            details={
                "prohibited_terms_found": found_prohibited,
            },
        )


def evaluate_agent_response(
    input_text: str,
    output_text: str,
    evaluators: Optional[List[BaseEvaluator]] = None,
    expected: Optional[str] = None,
) -> Dict[str, EvaluationResult]:
    """Run multiple evaluators on an agent response.

    Args:
        input_text: User input
        output_text: Agent output
        evaluators: List of evaluators to run
        expected: Expected output (optional)

    Returns:
        Dictionary of evaluator name to result
    """
    if evaluators is None:
        evaluators = [
            ResponseQualityEvaluator(),
            TaskCompletionEvaluator(),
            CoherenceEvaluator(),
            SafetyEvaluator(),
        ]

    results = {}
    for evaluator in evaluators:
        try:
            result = evaluator.evaluate(input_text, output_text, expected)
            results[evaluator.name] = result
        except Exception as e:
            results[evaluator.name] = EvaluationResult(
                score=0.0,
                passed=False,
                feedback=f"Evaluation error: {e}",
            )

    return results


def create_evaluation_summary(results: Dict[str, EvaluationResult]) -> str:
    """Create a human-readable evaluation summary.

    Args:
        results: Dictionary of evaluation results

    Returns:
        Formatted summary string
    """
    lines = ["Evaluation Summary", "=" * 40]

    total_score = 0
    for name, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"\n{name}: [{status}] Score: {result.score:.2f}")
        lines.append(f"  {result.feedback}")
        total_score += result.score

    avg_score = total_score / len(results) if results else 0
    lines.append(f"\nOverall Score: {avg_score:.2f}")
    lines.append(f"Status: {'PASSED' if avg_score >= 0.7 else 'NEEDS IMPROVEMENT'}")

    return "\n".join(lines)
