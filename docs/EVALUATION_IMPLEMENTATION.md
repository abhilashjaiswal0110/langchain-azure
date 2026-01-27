# Evaluation Implementation for LangChain Azure AI Agents

## Overview

This document outlines the comprehensive evaluation framework implemented for Azure AI Foundry agents, including integration with both LangSmith and Azure AI Foundry evaluation services.

## Architecture

```
langchain_azure_ai/
└── evaluation/
    ├── __init__.py                     # Main evaluation API exports
    ├── base_evaluators.py              # Core evaluation metrics
    ├── multi_turn_evaluators.py        # Multi-turn conversation evaluation
    ├── langsmith_evaluator.py          # LangSmith integration
    ├── azure_foundry_evaluator.py      # Azure AI Foundry integration
    └── agent_metrics.py                # Agent performance tracking
```

## Components Implemented

### 1. Base Evaluators (`base_evaluators.py`)

**Core Evaluators:**
- **ResponseQualityEvaluator**: Evaluates response length, relevance, and required elements
- **TaskCompletionEvaluator**: Determines if tasks were successfully completed
- **FactualAccuracyEvaluator**: Checks factual correctness against known facts
- **CoherenceEvaluator**: Evaluates logical flow and sentence structure
- **SafetyEvaluator**: Detects potentially harmful or sensitive content

**Usage:**
```python
from langchain_azure_ai.evaluation import (
    ResponseQualityEvaluator,
    TaskCompletionEvaluator,
    evaluate_agent_response,
)

evaluators = [
    ResponseQualityEvaluator(min_length=50, max_length=2000),
    TaskCompletionEvaluator(),
]

results = evaluate_agent_response(
    input_text="How do I reset my password?",
    output_text="To reset your password, go to Settings > Security > Reset Password...",
    evaluators=evaluators,
)

for name, result in results.items():
    print(f"{name}: Score={result.score:.2f}, Passed={result.passed}")
    print(f"  Feedback: {result.feedback}")
```

### 2. Multi-Turn Evaluators (`multi_turn_evaluators.py`)

**Multi-Turn Evaluation Components:**
- **IntentCompletionEvaluator**: Tracks if user intents are fulfilled across turns
- **ContextCoherenceEvaluator**: Evaluates context maintenance across conversation
- **ToolSequenceEvaluator**: Validates tool calling sequences
- **ConversationFlowEvaluator**: Assesses natural conversation flow
- **MultiTurnEvaluator**: Comprehensive multi-turn conversation evaluator

**Usage:**
```python
from langchain_azure_ai.evaluation import (
    evaluate_multi_turn_conversation,
    MultiTurnTestCase,
)

messages = [
    HumanMessage(content="I need to reset my password"),
    AIMessage(content="I can help you reset your password. What's your username?"),
    HumanMessage(content="john.doe@company.com"),
    AIMessage(content="Password reset link sent to john.doe@company.com"),
]

result = evaluate_multi_turn_conversation(
    messages=messages,
    expected_intents=["password reset", "verification"],
    context_requirements=["username", "email"],
)

print(f"Overall Score: {result.overall_score:.2f}")
print(f"Intent Completion: {result.intent_completion['passed']}")
print(f"Context Coherence: {result.context_coherence['passed']}")
```

### 3. LangSmith Integration (`langsmith_evaluator.py`)

**Features:**
- Offline evaluation against LangSmith datasets
- Online feedback submission
- Experiment tracking and management
- SDK-compatible evaluator wrappers
- Tracing diagnostics and verification

**Configuration:**
```bash
# Environment variables
export LANGCHAIN_API_KEY=your_langsmith_api_key
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=azure-ai-agents-eval
export EVAL_PROJECT_NAME=azure-agents-eval
export EVAL_DATASET_NAME=it-helpdesk-dataset
```

**Usage:**
```python
from langchain_azure_ai.evaluation import (
    LangSmithEvaluator,
    ResponseQualityEvaluator,
    TaskCompletionEvaluator,
)

# Initialize evaluator
langsmith_eval = LangSmithEvaluator()

# Register custom evaluators
langsmith_eval.register_evaluator(ResponseQualityEvaluator())
langsmith_eval.register_evaluator(TaskCompletionEvaluator())

# Create dataset
dataset_id = langsmith_eval.sync_dataset_from_local(
    dataset_name="it-helpdesk-dataset",
    test_cases=[
        {
            "id": "test-1",
            "input": "How do I reset my password?",
            "expected_output": "Go to Settings > Security > Reset Password",
            "tags": ["password", "authentication"],
        },
        {
            "id": "test-2",
            "input": "My account is locked",
            "expected_output": "I'll help you unlock your account...",
            "tags": ["account", "security"],
        },
    ],
)

# Run offline evaluation
experiment = await langsmith_eval.run_offline_evaluation(
    agent_func=my_agent.chat,
    dataset_name="it-helpdesk-dataset",
)

print(f"Experiment: {experiment.name}")
print(f"Results: {len(experiment.results)} tests")
print(f"Metrics: {experiment.metrics}")
```

**LangSmith SDK Integration:**
```python
from langsmith.evaluation import evaluate
from langchain_azure_ai.evaluation import (
    create_langsmith_evaluator_wrapper,
    ResponseQualityEvaluator,
)

# Create SDK-compatible evaluator
quality_eval = create_langsmith_evaluator_wrapper(ResponseQualityEvaluator())

# Run evaluation with LangSmith SDK
results = evaluate(
    agent_func,
    data="it-helpdesk-dataset",
    evaluators=[quality_eval],
    experiment_prefix="helpdesk-eval",
)
```

### 4. Azure AI Foundry Integration (`azure_foundry_evaluator.py`)

**Built-in Metrics:**
- **Groundedness**: How well output is supported by context
- **Relevance**: Relevance of output to input
- **Coherence**: Logical flow and structure
- **Fluency**: Natural language quality
- **Similarity**: Comparison with expected output

**Configuration:**
```bash
# Environment variables
export AZURE_AI_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
export AZURE_SUBSCRIPTION_ID=your-subscription-id
export AZURE_RESOURCE_GROUP=your-resource-group
export AZURE_PROJECT_NAME=your-project-name
```

**Usage:**
```python
from langchain_azure_ai.evaluation import (
    AzureAIFoundryEvaluator,
    run_foundry_evaluation,
)

# Initialize evaluator
foundry_eval = AzureAIFoundryEvaluator()

# Run evaluation
result = await foundry_eval.run_evaluation(
    agent_func=my_agent.chat,
    test_data=[
        {
            "input": "What is Azure AI Foundry?",
            "expected_output": "Azure AI Foundry is a platform for building AI applications...",
            "context": "Azure AI Foundry documentation...",
        },
    ],
    metrics=["groundedness", "relevance", "coherence", "fluency"],
)

print(f"Status: {result.status}")
print(f"Overall Pass: {result.summary['overall_pass']}")
for name, metric in result.metrics.items():
    print(f"  {name}: {metric.score:.2f} ({'PASS' if metric.passed else 'FAIL'})")
```

### 5. Agent Performance Metrics (`agent_metrics.py`)

**Tracked Metrics:**
- Response time (avg, p50, p95, p99)
- Token usage (prompt, completion, total)
- Success/error rates
- User satisfaction ratings
- Estimated costs
- Session statistics

**Usage:**
```python
from langchain_azure_ai.evaluation import (
    AgentPerformanceTracker,
    get_agent_metrics,
    record_agent_execution,
)

# Track performance
tracker = AgentPerformanceTracker("it-helpdesk")

# Record executions
tracker.record_execution(
    duration_ms=1500,
    prompt_tokens=250,
    completion_tokens=150,
    success=True,
    user_rating=5,
    session_id="session-123",
)

# Get metrics
metrics = tracker.get_metrics()
print(f"Agent: {metrics.agent_name}")
print(f"Total Requests: {metrics.total_requests}")
print(f"Success Rate: {metrics.success_rate:.2%}")
print(f"Avg Response Time: {metrics.avg_response_time_ms:.0f}ms")
print(f"P95 Response Time: {metrics.p95_response_time_ms:.0f}ms")
print(f"Avg Tokens: {metrics.avg_tokens_per_request:.0f}")
print(f"Estimated Cost: ${metrics.estimated_cost_usd:.4f}")

# Convenience function
record_agent_execution(
    agent_name="it-helpdesk",
    duration_ms=1200,
    prompt_tokens=200,
    completion_tokens=100,
    success=True,
)

# Get metrics for any agent
metrics = get_agent_metrics("it-helpdesk", window_hours=24)
```

## Integration with Existing Observability

The evaluation framework integrates seamlessly with the existing Azure Monitor/OpenTelemetry observability:

```python
from langchain_azure_ai.observability import AgentTelemetry
from langchain_azure_ai.evaluation import (
    evaluate_agent_response,
    record_agent_execution,
)

# Agent telemetry tracks execution metrics
telemetry = AgentTelemetry("it-helpdesk", "it")

with telemetry.track_execution() as metrics:
    response = agent.chat(message)
    metrics.prompt_tokens = 250
    metrics.completion_tokens = 150

# Evaluation assesses response quality
eval_results = evaluate_agent_response(
    input_text=message,
    output_text=response,
)

# Performance tracking monitors long-term trends
record_agent_execution(
    agent_name="it-helpdesk",
    duration_ms=metrics.duration_ms,
    prompt_tokens=metrics.prompt_tokens,
    completion_tokens=metrics.completion_tokens,
    success=metrics.success,
)
```

## Next Steps

### 1. Create Evaluation Datasets

Create datasets for each agent type:

```python
# IT Agents Dataset
it_helpdesk_dataset = [
    {
        "id": "it-001",
        "input": "How do I reset my password?",
        "expected_output": "To reset your password, follow these steps...",
        "expected_keywords": ["Settings", "Security", "Reset Password"],
        "tags": ["password", "authentication"],
        "difficulty": "easy",
    },
    # Add more test cases...
]

# Enterprise Agents Dataset
research_agent_dataset = [
    {
        "id": "research-001",
        "input": "Find information about quantum computing",
        "expected_output": "Quantum computing is a field...",
        "expected_keywords": ["quantum", "qubits", "superposition"],
        "tags": ["research", "technology"],
        "difficulty": "medium",
    },
    # Add more test cases...
]

# Sync to LangSmith
langsmith_eval.sync_dataset_from_local(
    dataset_name="it-helpdesk-v1",
    test_cases=it_helpdesk_dataset,
)
```

### 2. Add Evaluation API Endpoints

Add evaluation endpoints to the FastAPI server:

```python
# libs/azure-ai/langchain_azure_ai/server/__init__.py

from langchain_azure_ai.evaluation import (
    evaluate_agent_response,
    get_agent_metrics,
    LangSmithEvaluator,
)

@app.post("/api/evaluate/{agent_name}")
async def evaluate_agent(agent_name: str, request: EvaluationRequest):
    """Evaluate agent response quality."""
    agent = registry.get_agent(agent_name)
    response = agent.chat(request.message)

    results = evaluate_agent_response(
        input_text=request.message,
        output_text=response,
    )

    return {"results": results, "response": response}

@app.get("/api/metrics/{agent_name}")
async def get_metrics(agent_name: str):
    """Get agent performance metrics."""
    metrics = get_agent_metrics(agent_name)
    return metrics

@app.post("/api/evaluation/run-offline")
async def run_offline_evaluation(request: OfflineEvalRequest):
    """Run offline evaluation against dataset."""
    langsmith_eval = LangSmithEvaluator()
    experiment = await langsmith_eval.run_offline_evaluation(
        agent_func=get_agent_func(request.agent_name),
        dataset_name=request.dataset_name,
    )
    return experiment
```

### 3. Write Unit Tests

Create comprehensive unit tests:

```python
# tests/unit_tests/evaluation/test_base_evaluators.py
def test_response_quality_evaluator():
    evaluator = ResponseQualityEvaluator(min_length=50)
    result = evaluator.evaluate(
        input_text="Test input",
        output_text="This is a good quality response with sufficient length.",
    )
    assert result.score > 0.7
    assert result.passed

# tests/unit_tests/evaluation/test_langsmith.py
def test_langsmith_dataset_sync():
    evaluator = LangSmithEvaluator()
    dataset_id = evaluator.sync_dataset_from_local(
        dataset_name="test-dataset",
        test_cases=[{"input": "test", "expected_output": "result"}],
    )
    assert dataset_id is not None
```

### 4. Write Integration Tests

Create integration tests for Azure and LangSmith:

```python
# tests/integration_tests/evaluation/test_azure_foundry_eval.py
@pytest.mark.integration
async def test_azure_foundry_evaluation():
    evaluator = AzureAIFoundryEvaluator()
    result = await evaluator.run_evaluation(
        agent_func=test_agent,
        test_data=[{"input": "test", "expected_output": "result"}],
        metrics=["relevance", "coherence"],
    )
    assert result.status == "completed"
    assert "relevance" in result.metrics
```

### 5. Create Documentation

Document evaluation usage in `docs/evaluation.md`:
- Setup and configuration
- Creating custom evaluators
- Running evaluations
- Interpreting results
- Best practices

### 6. Setup Continuous Evaluation

Create scheduled evaluation pipeline:

```python
# scripts/run_evaluations.py
import asyncio
from langchain_azure_ai.evaluation import (
    LangSmithEvaluator,
    AzureAIFoundryEvaluator,
)

async def run_continuous_evaluation():
    """Run evaluations on schedule."""
    langsmith_eval = LangSmithEvaluator()
    foundry_eval = AzureAIFoundryEvaluator()

    # Run LangSmith evaluation
    ls_experiment = await langsmith_eval.run_offline_evaluation(
        agent_func=helpdesk_agent.chat,
        dataset_name="it-helpdesk-v1",
    )

    # Run Azure AI Foundry evaluation
    foundry_result = await foundry_eval.run_evaluation(
        agent_func=helpdesk_agent.chat,
        test_data=load_test_data(),
        metrics=["groundedness", "relevance", "coherence"],
    )

    # Report results
    print(f"LangSmith: {ls_experiment.metrics}")
    print(f"Azure AI Foundry: {foundry_result.summary}")

if __name__ == "__main__":
    asyncio.run(run_continuous_evaluation())
```

## Environment Variables Summary

```bash
# Azure AI Foundry
export AZURE_AI_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
export AZURE_SUBSCRIPTION_ID=your-subscription-id
export AZURE_RESOURCE_GROUP=your-resource-group
export AZURE_PROJECT_NAME=your-project-name

# LangSmith
export LANGCHAIN_API_KEY=your_langsmith_api_key
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=azure-ai-agents-eval

# Evaluation Configuration
export EVAL_PROJECT_NAME=azure-agents-eval
export EVAL_DATASET_NAME=it-helpdesk-dataset
export EVAL_AUTO_FEEDBACK=true
export EVAL_ONLINE_SAMPLING_RATE=0.1

# Azure Monitor (already configured)
export APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=xxx
export ENABLE_AZURE_MONITOR=true
export OTEL_SERVICE_NAME=azure-ai-agents
```

## Benefits

1. **Comprehensive Evaluation**: Multiple evaluation dimensions (quality, completion, coherence, safety)
2. **Dual Integration**: Both LangSmith and Azure AI Foundry evaluation services
3. **Performance Tracking**: Long-term agent performance monitoring
4. **Observability Integration**: Seamless integration with existing Azure Monitor/OpenTelemetry
5. **Flexible Framework**: Easy to add custom evaluators
6. **Production Ready**: Built for enterprise deployment with proper error handling

## References

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Azure AI Foundry Documentation](https://learn.microsoft.com/en-us/azure/ai-studio/)
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation)
- [Azure Monitor OpenTelemetry](https://learn.microsoft.com/en-us/azure/azure-monitor/app/opentelemetry-overview)

---

**Last Updated**: 2026-01-27
**Implementation Status**: Core framework complete, integration pending
