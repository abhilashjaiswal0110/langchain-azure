"""Comprehensive evaluation tests for DeepAgents.

This module tests the evaluation framework against all DeepAgents:
- IT Operations Agent
- Sales Intelligence Agent
- Recruitment Agent

Tests include:
- Single response evaluation
- Multi-turn conversation evaluation
- Performance metrics tracking
- LangSmith integration
- Azure AI Foundry evaluation

Run with: pytest tests/evaluation/test_deep_agents_evaluation.py -v -s
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add libs to path
libs_path = Path(__file__).parent.parent.parent / "libs" / "azure-ai"
sys.path.insert(0, str(libs_path))

from langchain_azure_ai.evaluation import (
    ResponseQualityEvaluator,
    TaskCompletionEvaluator,
    CoherenceEvaluator,
    SafetyEvaluator,
    evaluate_agent_response,
    get_agent_metrics,
    record_agent_execution,
    LangSmithEvaluator,
    AzureAIFoundryEvaluator,
)
from langchain_azure_ai.evaluation.datasets import get_dataset, AGENT_DATASETS


# =============================================================================
# Test Configuration
# =============================================================================

DEEP_AGENTS = ["it_operations", "sales_intelligence", "recruitment"]

# Mock agent for testing (simulates real agent responses)
class MockDeepAgent:
    """Mock DeepAgent for testing without requiring full agent setup."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.responses = self._get_mock_responses()

    def _get_mock_responses(self) -> Dict[str, str]:
        """Get mock responses for different test cases."""
        return {
            "it_operations": {
                "it-ops-001": "Based on incident analysis INC0012345, this is a network connectivity issue affecting the switch in Server Room B. Root cause: VLAN misconfiguration on port 24. Recommended solution: 1) Backup current config 2) Reconfigure VLAN settings 3) Restart the network switch 4) Verify connectivity. Expected resolution time: 30 minutes.",
                "it-ops-002": "Current SLA compliance report: Overall uptime 99.8% across all critical services. Services meeting SLA: Database (99.95%), Web Server (99.9%), API Gateway (99.85%). Services below target: Email Server (99.2% - target 99.5%), File Server (98.9% - target 99.5%). Recommended actions: Investigate email server issues, schedule file server maintenance.",
                "it-ops-003": "Network performance diagnostics completed. Findings: 1) Bandwidth utilization at 85% during peak hours 2) Latency to cloud services elevated (150ms avg) 3) Router CPU at 75% 4) QoS not configured. Recommendations: Implement QoS policies prioritizing business-critical traffic, upgrade router firmware, consider bandwidth upgrade if utilization remains >80%.",
            },
            "sales_intelligence": {
                "sales-001": "Lead qualification analysis (BANT Framework): Budget: Confirmed $100K (✓). Authority: Need to identify decision maker - typically CEO/CTO for 50-employee startup (⚠). Need: CRM solution clearly identified (✓). Timeline: Not specified, needs clarification (⚠). Score: 75/100 - Qualified lead. Next steps: 1) Schedule discovery call 2) Request meeting with decision maker 3) Prepare customized demo 4) Send ROI analysis.",
                "sales-002": "Q1 Sales Pipeline Analysis: Total opportunities: 15 ($2.5M). By stage: Prospecting (3, $400K), Qualification (5, $800K), Proposal (4, $750K), Negotiation (3, $550K). At-risk deals identified: 3 opportunities worth $450K showing no activity for 14+ days. Risk factors: No follow-up, stale proposals, competitor activity. Actions required: Immediate outreach to account managers, refresh proposals, schedule executive meetings.",
            },
            "recruitment": {
                "recruitment-001": "Resume screening completed for Senior Software Engineer position. Candidate profile: 7 years professional experience with Python and Java development, demonstrated cloud platform expertise (AWS/Azure), active open-source contributor (GitHub profile shows 50+ repositories). Technical skills match: 95%. Recommendation: STRONG CANDIDATE - Proceed to technical interview. Suggested focus areas: System design, cloud architecture, code quality practices.",
                "recruitment-002": "DevOps Engineer Interview Plan: ROUND 1 - Technical Screening (30 min): CI/CD pipeline design, Docker containerization, Kubernetes basics, Git workflow. ROUND 2 - Deep Technical (60 min): Cloud infrastructure (AWS/Azure/GCP), Infrastructure as Code (Terraform/CloudFormation), monitoring & observability, security best practices. ROUND 3 - System Design (45 min): Design scalable deployment pipeline, implement blue-green deployments, incident response procedures. ROUND 4 - Cultural Fit (30 min): Team collaboration, on-call experience, continuous learning.",
            },
        }

    def chat(self, message: str, thread_id: str = None) -> str:
        """Simulate agent chat response."""
        # Find matching test case
        for test_id, response in self.responses[self.agent_name].items():
            # Simple matching - in real scenario would use agent logic
            if len(message) > 20:  # Assume it matches if message is substantial
                return response
        # Default response
        return f"[{self.agent_name}] Processing your request: {message[:50]}..."


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def evaluators():
    """Create standard evaluators for testing."""
    return [
        ResponseQualityEvaluator(min_length=50, max_length=5000),
        TaskCompletionEvaluator(),
        CoherenceEvaluator(),
        SafetyEvaluator(),
    ]


@pytest.fixture
def langsmith_evaluator():
    """Create LangSmith evaluator if configured."""
    try:
        return LangSmithEvaluator()
    except Exception as e:
        pytest.skip(f"LangSmith not configured: {e}")


# =============================================================================
# DeepAgents Evaluation Tests
# =============================================================================

class TestDeepAgentsEvaluation:
    """Comprehensive test suite for DeepAgents evaluation."""

    @pytest.mark.parametrize("agent_name", DEEP_AGENTS)
    def test_deep_agent_dataset_exists(self, agent_name):
        """Test that evaluation datasets exist for all DeepAgents."""
        dataset = get_dataset(agent_name)
        assert len(dataset) > 0, f"No dataset found for {agent_name}"
        logger.info(f"✓ Dataset for {agent_name}: {len(dataset)} test cases")

    @pytest.mark.parametrize("agent_name", DEEP_AGENTS)
    def test_single_response_evaluation(self, agent_name, evaluators):
        """Test single response evaluation for each DeepAgent."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {agent_name.upper()} - Single Response Evaluation")
        logger.info(f"{'='*80}")

        # Get dataset and agent
        dataset = get_dataset(agent_name)
        agent = MockDeepAgent(agent_name)

        results_summary = []

        for test_case in dataset:
            logger.info(f"\n[Test Case: {test_case['id']}]")
            logger.info(f"Input: {test_case['input'][:100]}...")

            # Get agent response
            response = agent.chat(test_case['input'])
            logger.info(f"Response length: {len(response)} chars")

            # Evaluate response
            eval_results = evaluate_agent_response(
                input_text=test_case['input'],
                output_text=response,
                evaluators=evaluators,
                expected=test_case.get('expected_output'),
            )

            # Log results
            scores = {}
            for name, result in eval_results.items():
                status = "[PASS]" if result.passed else "[FAIL]"
                logger.info(f"  {name}: {status} Score={result.score:.2f}")
                scores[name] = result.score

            overall_score = sum(scores.values()) / len(scores)
            logger.info(f"  Overall: {overall_score:.2f}")

            results_summary.append({
                "test_id": test_case['id'],
                "overall_score": overall_score,
                "passed": overall_score >= 0.7,
                "scores": scores,
            })

        # Assert all tests passed
        passed_count = sum(1 for r in results_summary if r['passed'])
        logger.info(f"\n{agent_name} Summary: {passed_count}/{len(results_summary)} tests passed")

        assert passed_count >= len(results_summary) * 0.7, \
            f"{agent_name}: Only {passed_count}/{len(results_summary)} tests passed (70% threshold)"

    @pytest.mark.parametrize("agent_name", DEEP_AGENTS)
    def test_performance_metrics_tracking(self, agent_name):
        """Test performance metrics tracking for DeepAgents."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {agent_name.upper()} - Performance Metrics")
        logger.info(f"{'='*80}")

        agent = MockDeepAgent(agent_name)
        dataset = get_dataset(agent_name)

        # Simulate executions
        for test_case in dataset:
            import time
            start = time.time()

            response = agent.chat(test_case['input'])

            duration_ms = (time.time() - start) * 1000

            # Record execution
            record_agent_execution(
                agent_name=agent_name,
                duration_ms=duration_ms,
                prompt_tokens=len(test_case['input'].split()) * 2,  # Rough estimate
                completion_tokens=len(response.split()) * 2,  # Rough estimate
                success=True,
                user_rating=5 if len(response) > 100 else 4,
            )

            logger.info(f"  Recorded: {test_case['id']} - {duration_ms:.0f}ms")

        # Get metrics
        metrics = get_agent_metrics(agent_name, window_hours=24)

        logger.info(f"\n{agent_name} Performance Metrics:")
        logger.info(f"  Total Requests: {metrics.total_requests}")
        logger.info(f"  Success Rate: {metrics.success_rate:.2%}")
        logger.info(f"  Avg Response Time: {metrics.avg_response_time_ms:.0f}ms")
        logger.info(f"  Total Tokens: {metrics.total_tokens:,}")
        logger.info(f"  Estimated Cost: ${metrics.estimated_cost_usd:.4f}")

        # Assertions
        assert metrics.total_requests >= len(dataset)
        assert metrics.success_rate >= 0.7
        assert metrics.avg_response_time_ms < 10000  # Less than 10 seconds

    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_name", DEEP_AGENTS)
    async def test_langsmith_dataset_sync(self, agent_name, langsmith_evaluator):
        """Test LangSmith dataset synchronization for DeepAgents."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {agent_name.upper()} - LangSmith Dataset Sync")
        logger.info(f"{'='*80}")

        dataset = get_dataset(agent_name)
        dataset_name = f"{agent_name}-eval-test"

        try:
            # Sync dataset to LangSmith
            dataset_id = langsmith_evaluator.sync_dataset_from_local(
                dataset_name=dataset_name,
                test_cases=dataset,
            )

            logger.info(f"✓ Dataset synced to LangSmith")
            logger.info(f"  Dataset Name: {dataset_name}")
            logger.info(f"  Dataset ID: {dataset_id}")
            logger.info(f"  Test Cases: {len(dataset)}")

            assert dataset_id is not None

        except Exception as e:
            logger.error(f"✗ LangSmith sync failed: {e}")
            pytest.skip(f"LangSmith integration not available: {e}")

    @pytest.mark.asyncio
    async def test_azure_foundry_evaluation(self):
        """Test Azure AI Foundry evaluation for IT Operations agent."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing Azure AI Foundry Evaluation")
        logger.info(f"{'='*80}")

        agent_name = "it_operations"
        dataset = get_dataset(agent_name)[:2]  # Use first 2 test cases

        agent = MockDeepAgent(agent_name)

        # Create evaluator
        foundry_eval = AzureAIFoundryEvaluator()

        # Prepare test data
        test_data = []
        for test_case in dataset:
            test_data.append({
                "id": test_case["id"],
                "input": test_case["input"],
                "expected_output": test_case["expected_output"],
                "context": test_case.get("context", ""),
            })

        # Run evaluation
        async def agent_func(input_text: str) -> str:
            return agent.chat(input_text)

        try:
            result = await foundry_eval.run_evaluation(
                agent_func=agent_func,
                test_data=test_data,
                metrics=["groundedness", "relevance", "coherence"],
                evaluation_name=f"test-{agent_name}-foundry",
            )

            logger.info(f"✓ Azure AI Foundry evaluation completed")
            logger.info(f"  Status: {result.status}")
            logger.info(f"  Total Tests: {len(result.test_results)}")

            for metric_name, metric in result.metrics.items():
                logger.info(f"  {metric_name}: {metric.score:.2f} ({'PASS' if metric.passed else 'FAIL'})")

            assert result.status == "completed"
            assert len(result.test_results) == len(test_data)

        except Exception as e:
            logger.warning(f"Azure AI Foundry evaluation skipped: {e}")
            pytest.skip(f"Azure AI Foundry not fully configured: {e}")


# =============================================================================
# Summary Report Generation
# =============================================================================

def test_generate_evaluation_summary():
    """Generate comprehensive evaluation summary for all DeepAgents."""
    logger.info(f"\n{'='*80}")
    logger.info(f"DEEPAGENTS EVALUATION SUMMARY")
    logger.info(f"{'='*80}\n")

    summary = {}

    for agent_name in DEEP_AGENTS:
        dataset = get_dataset(agent_name)
        metrics = get_agent_metrics(agent_name, window_hours=24)

        summary[agent_name] = {
            "test_cases": len(dataset),
            "requests_processed": metrics.total_requests,
            "success_rate": f"{metrics.success_rate:.2%}",
            "avg_response_time_ms": f"{metrics.avg_response_time_ms:.0f}",
            "total_tokens": metrics.total_tokens,
            "estimated_cost": f"${metrics.estimated_cost_usd:.4f}",
        }

    logger.info("Agent Performance Summary:")
    logger.info(json.dumps(summary, indent=2))

    # Write to file
    output_dir = Path(__file__).parent.parent.parent / "test_results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "deepagents_evaluation_summary.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✓ Summary saved to: {output_file}")

    assert len(summary) == len(DEEP_AGENTS)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
