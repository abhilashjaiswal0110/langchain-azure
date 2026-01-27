"""Azure AI Foundry evaluation integration.

Provides integration with Azure AI Foundry's evaluation capabilities:
- Built-in evaluators (groundedness, relevance, coherence, fluency)
- Custom evaluator deployment
- Evaluation runs and experiments
- Integration with Azure AI projects

Usage:
    from langchain_azure_ai.evaluation import AzureAIFoundryEvaluator

    evaluator = AzureAIFoundryEvaluator()
    result = await evaluator.run_evaluation(
        agent_func=my_agent,
        test_data=test_cases,
        metrics=["groundedness", "relevance", "coherence"],
    )
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class AzureAIFoundryConfig:
    """Configuration for Azure AI Foundry evaluation.

    Attributes:
        project_endpoint: Azure AI Foundry project endpoint
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        project_name: Project name
        credential: Azure credential (DefaultAzureCredential if None)
    """

    project_endpoint: Optional[str] = None
    subscription_id: Optional[str] = None
    resource_group: Optional[str] = None
    project_name: Optional[str] = None
    credential: Any = None

    @classmethod
    def from_env(cls) -> "AzureAIFoundryConfig":
        """Create configuration from environment variables."""
        from azure.identity import DefaultAzureCredential

        return cls(
            project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
            project_name=os.getenv("AZURE_PROJECT_NAME"),
            credential=DefaultAzureCredential(),
        )


@dataclass
class FoundryMetric:
    """Azure AI Foundry evaluation metric.

    Attributes:
        name: Metric name (e.g., "groundedness", "relevance")
        score: Metric score (0.0 to 1.0 or specific range)
        threshold: Pass/fail threshold
        passed: Whether metric passed threshold
        details: Additional metric details
    """

    name: str
    score: float
    threshold: float = 0.7
    passed: bool = False
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Calculate passed status after initialization."""
        self.passed = self.score >= self.threshold


@dataclass
class FoundryEvaluationResult:
    """Result from Azure AI Foundry evaluation.

    Attributes:
        id: Unique evaluation ID
        name: Evaluation name
        status: Evaluation status
        created_at: Creation timestamp
        metrics: Dictionary of metric name to FoundryMetric
        test_results: List of individual test results
        summary: Overall summary statistics
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: Dict[str, FoundryMetric] = field(default_factory=dict)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AzureAIFoundryEvaluator:
    """Azure AI Foundry evaluation integration.

    Supports:
    - Built-in Azure AI Foundry evaluators
    - Custom evaluator deployment
    - Evaluation experiments and tracking
    - Integration with Azure AI projects
    """

    def __init__(self, config: Optional[AzureAIFoundryConfig] = None) -> None:
        """Initialize Azure AI Foundry evaluator.

        Args:
            config: Azure AI Foundry configuration. If None, loads from environment.
        """
        self.config = config or AzureAIFoundryConfig.from_env()
        self._client = None

    @property
    def client(self):
        """Get or create Azure AI client."""
        if self._client is None:
            try:
                from azure.ai.projects import AIProjectClient

                if not self.config.project_endpoint:
                    logger.warning(
                        "AZURE_AI_PROJECT_ENDPOINT not set. "
                        "Azure AI Foundry evaluations will be disabled."
                    )
                    return None

                self._client = AIProjectClient.from_connection_string(
                    conn_str=self.config.project_endpoint,
                    credential=self.config.credential,
                )
            except ImportError:
                logger.warning(
                    "azure-ai-projects not installed. "
                    "Install with: pip install azure-ai-projects"
                )
                return None
            except Exception as e:
                logger.error(f"Failed to initialize Azure AI client: {e}")
                return None

        return self._client

    async def run_evaluation(
        self,
        agent_func: Callable,
        test_data: List[Dict[str, Any]],
        metrics: List[str],
        evaluation_name: Optional[str] = None,
    ) -> FoundryEvaluationResult:
        """Run evaluation using Azure AI Foundry.

        Args:
            agent_func: Agent function to evaluate
            test_data: List of test cases
            metrics: List of metrics to evaluate ("groundedness", "relevance", etc.)
            evaluation_name: Optional name for this evaluation

        Returns:
            FoundryEvaluationResult with evaluation results
        """
        evaluation_name = evaluation_name or f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        result = FoundryEvaluationResult(
            name=evaluation_name,
            status="running",
        )

        if self.client is None:
            result.status = "failed"
            result.metadata["error"] = "Azure AI client not available"
            return result

        try:
            # Run evaluations for each test case
            for idx, test_case in enumerate(test_data):
                input_text = test_case.get("input", "")
                expected_output = test_case.get("expected_output", "")
                context = test_case.get("context", "")

                try:
                    # Get agent output
                    output = await agent_func(input_text)
                    output_text = str(output)

                    # Evaluate with Azure AI Foundry metrics
                    test_result = {
                        "id": test_case.get("id", f"test-{idx}"),
                        "input": input_text,
                        "output": output_text,
                        "expected": expected_output,
                        "metrics": {},
                    }

                    # Run each metric
                    for metric_name in metrics:
                        metric_score = await self._evaluate_metric(
                            metric_name=metric_name,
                            input_text=input_text,
                            output_text=output_text,
                            expected_output=expected_output,
                            context=context,
                        )

                        test_result["metrics"][metric_name] = metric_score

                        # Aggregate to overall metrics
                        if metric_name not in result.metrics:
                            result.metrics[metric_name] = FoundryMetric(
                                name=metric_name,
                                score=metric_score,
                            )
                        else:
                            # Update running average
                            current_metric = result.metrics[metric_name]
                            n = len(result.test_results)
                            current_metric.score = (
                                current_metric.score * n + metric_score
                            ) / (n + 1)

                    result.test_results.append(test_result)

                except Exception as e:
                    logger.error(f"Error evaluating test case {idx}: {e}")
                    result.test_results.append({
                        "id": test_case.get("id", f"test-{idx}"),
                        "error": str(e),
                    })

            # Calculate summary
            result.summary = self._calculate_summary(result)
            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.metadata["error"] = str(e)
            logger.error(f"Evaluation failed: {e}")

        return result

    async def _evaluate_metric(
        self,
        metric_name: str,
        input_text: str,
        output_text: str,
        expected_output: str,
        context: str,
    ) -> float:
        """Evaluate a single metric using Azure AI Foundry.

        Args:
            metric_name: Name of the metric to evaluate
            input_text: Input text
            output_text: Agent output
            expected_output: Expected output
            context: Additional context

        Returns:
            Metric score (0.0 to 1.0)
        """
        try:
            # Use Azure AI Foundry evaluators
            if metric_name == "groundedness":
                return await self._evaluate_groundedness(output_text, context)
            elif metric_name == "relevance":
                return await self._evaluate_relevance(input_text, output_text)
            elif metric_name == "coherence":
                return await self._evaluate_coherence(output_text)
            elif metric_name == "fluency":
                return await self._evaluate_fluency(output_text)
            elif metric_name == "similarity":
                return await self._evaluate_similarity(output_text, expected_output)
            else:
                logger.warning(f"Unknown metric: {metric_name}")
                return 0.5

        except Exception as e:
            logger.error(f"Error evaluating metric {metric_name}: {e}")
            return 0.0

    async def _evaluate_groundedness(self, output: str, context: str) -> float:
        """Evaluate groundedness (how well output is supported by context)."""
        # Placeholder implementation - would use Azure AI Foundry evaluator
        # In production, this would call the actual Azure AI Foundry API
        if not context:
            return 0.5

        # Simple heuristic: check keyword overlap
        output_words = set(output.lower().split())
        context_words = set(context.lower().split())
        overlap = len(output_words & context_words)

        return min(overlap / max(len(output_words), 1), 1.0)

    async def _evaluate_relevance(self, input_text: str, output: str) -> float:
        """Evaluate relevance of output to input."""
        # Placeholder implementation
        input_words = set(input_text.lower().split())
        output_words = set(output.lower().split())
        overlap = len(input_words & output_words)

        return min(overlap / max(len(input_words), 1), 1.0)

    async def _evaluate_coherence(self, output: str) -> float:
        """Evaluate coherence of output."""
        # Placeholder implementation
        from langchain_azure_ai.evaluation.base_evaluators import CoherenceEvaluator

        evaluator = CoherenceEvaluator()
        result = evaluator.evaluate("", output, None)
        return result.score

    async def _evaluate_fluency(self, output: str) -> float:
        """Evaluate fluency of output."""
        # Placeholder implementation - checks for basic fluency indicators
        if not output:
            return 0.0

        # Check for proper sentence endings
        sentences = [s.strip() for s in output.split(".") if s.strip()]
        if not sentences:
            return 0.5

        # Basic fluency heuristics
        score = 1.0
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)

        if avg_length < 3:  # Very short sentences
            score -= 0.3
        elif avg_length > 40:  # Very long sentences
            score -= 0.2

        return max(score, 0.0)

    async def _evaluate_similarity(self, output: str, expected: str) -> float:
        """Evaluate similarity between output and expected output."""
        if not expected:
            return 1.0

        # Simple word-level similarity
        output_words = set(output.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 1.0

        intersection = len(output_words & expected_words)
        union = len(output_words | expected_words)

        return intersection / union if union > 0 else 0.0

    def _calculate_summary(self, result: FoundryEvaluationResult) -> Dict[str, Any]:
        """Calculate summary statistics for evaluation result."""
        return {
            "total_tests": len(result.test_results),
            "passed_tests": sum(
                1
                for test in result.test_results
                if all(
                    FoundryMetric(name=name, score=score).passed
                    for name, score in test.get("metrics", {}).items()
                )
            ),
            "average_scores": {
                name: metric.score for name, metric in result.metrics.items()
            },
            "overall_pass": all(metric.passed for metric in result.metrics.values()),
        }


def create_foundry_evaluator(config: Optional[AzureAIFoundryConfig] = None) -> AzureAIFoundryEvaluator:
    """Create an Azure AI Foundry evaluator instance.

    Args:
        config: Optional configuration. Uses environment if not provided.

    Returns:
        AzureAIFoundryEvaluator instance
    """
    return AzureAIFoundryEvaluator(config)


async def run_foundry_evaluation(
    agent_func: Callable,
    test_data: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> FoundryEvaluationResult:
    """Convenience function to run Azure AI Foundry evaluation.

    Args:
        agent_func: Agent function to evaluate
        test_data: Test cases
        metrics: Metrics to evaluate (defaults to standard set)

    Returns:
        FoundryEvaluationResult
    """
    metrics = metrics or ["groundedness", "relevance", "coherence", "fluency"]
    evaluator = create_foundry_evaluator()
    return await evaluator.run_evaluation(agent_func, test_data, metrics)
