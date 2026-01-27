"""Evaluation framework for Azure AI Foundry agents.

This module provides comprehensive evaluation capabilities:
- Custom evaluators for agent response quality
- Multi-turn conversation evaluation
- LangSmith integration for offline/online evaluation
- Azure AI Foundry evaluation integration
- Metrics collection and reporting

Usage:
    from langchain_azure_ai.evaluation import (
        ResponseQualityEvaluator,
        TaskCompletionEvaluator,
        LangSmithEvaluator,
        AzureAIFoundryEvaluator,
        evaluate_agent_response,
    )

    # Evaluate a single response
    evaluators = [ResponseQualityEvaluator(), TaskCompletionEvaluator()]
    results = evaluate_agent_response(
        input_text="How do I reset my password?",
        output_text="To reset your password, go to Settings...",
        evaluators=evaluators,
    )

    # Run offline evaluation with LangSmith
    langsmith_eval = LangSmithEvaluator()
    experiment = await langsmith_eval.run_offline_evaluation(
        agent_func=my_agent,
        dataset_name="it-helpdesk-dataset",
        evaluators=evaluators,
    )
"""

from langchain_azure_ai.evaluation.base_evaluators import (
    BaseEvaluator,
    EvaluationResult,
    ResponseQualityEvaluator,
    TaskCompletionEvaluator,
    FactualAccuracyEvaluator,
    CoherenceEvaluator,
    SafetyEvaluator,
    evaluate_agent_response,
    create_evaluation_summary,
)

from langchain_azure_ai.evaluation.multi_turn_evaluators import (
    ConversationTurn,
    MultiTurnTestCase,
    MultiTurnEvaluationResult,
    IntentCompletionEvaluator,
    ContextCoherenceEvaluator,
    ToolSequenceEvaluator,
    ConversationFlowEvaluator,
    MultiTurnEvaluator,
    evaluate_multi_turn_conversation,
)

from langchain_azure_ai.evaluation.langsmith_evaluator import (
    LangSmithConfig,
    EvaluationExperiment,
    LangSmithEvaluator,
    get_langsmith_evaluator,
    reset_langsmith_evaluator,
    submit_online_feedback,
    evaluate_agent_offline,
    create_langsmith_evaluator_wrapper,
    create_playground_compatible_evaluator,
    run_langsmith_sdk_evaluation,
    verify_tracing_config,
    test_langsmith_connection,
    get_recent_traces,
    ensure_tracing_enabled,
)

from langchain_azure_ai.evaluation.azure_foundry_evaluator import (
    AzureAIFoundryConfig,
    AzureAIFoundryEvaluator,
    FoundryEvaluationResult,
    FoundryMetric,
    create_foundry_evaluator,
    run_foundry_evaluation,
)

from langchain_azure_ai.evaluation.agent_metrics import (
    AgentMetrics,
    AgentPerformanceTracker,
    ExecutionRecord,
    get_agent_metrics,
    calculate_agent_benchmarks,
    record_agent_execution,
)

__all__ = [
    # Base evaluators
    "BaseEvaluator",
    "EvaluationResult",
    "ResponseQualityEvaluator",
    "TaskCompletionEvaluator",
    "FactualAccuracyEvaluator",
    "CoherenceEvaluator",
    "SafetyEvaluator",
    "evaluate_agent_response",
    "create_evaluation_summary",
    # Multi-turn evaluators
    "ConversationTurn",
    "MultiTurnTestCase",
    "MultiTurnEvaluationResult",
    "IntentCompletionEvaluator",
    "ContextCoherenceEvaluator",
    "ToolSequenceEvaluator",
    "ConversationFlowEvaluator",
    "MultiTurnEvaluator",
    "evaluate_multi_turn_conversation",
    # LangSmith integration
    "LangSmithConfig",
    "EvaluationExperiment",
    "LangSmithEvaluator",
    "get_langsmith_evaluator",
    "reset_langsmith_evaluator",
    "submit_online_feedback",
    "evaluate_agent_offline",
    "create_langsmith_evaluator_wrapper",
    "create_playground_compatible_evaluator",
    "run_langsmith_sdk_evaluation",
    "verify_tracing_config",
    "test_langsmith_connection",
    "get_recent_traces",
    "ensure_tracing_enabled",
    # Azure AI Foundry integration
    "AzureAIFoundryConfig",
    "AzureAIFoundryEvaluator",
    "FoundryEvaluationResult",
    "FoundryMetric",
    "create_foundry_evaluator",
    "run_foundry_evaluation",
    # Agent metrics
    "AgentMetrics",
    "AgentPerformanceTracker",
    "ExecutionRecord",
    "get_agent_metrics",
    "calculate_agent_benchmarks",
    "record_agent_execution",
]
