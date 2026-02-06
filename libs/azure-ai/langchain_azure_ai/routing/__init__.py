"""Cost optimization, intelligent routing, and resilience for Azure AI.

This module provides:
- Cost-aware model selection and routing
- Budget tracking and enforcement
- Advanced rate limiting with token bucket
- Retry strategies with exponential backoff
- Circuit breaker pattern

Usage:
    from langchain_azure_ai.routing import (
        # Cost optimization
        CostOptimizingRouter,
        RoutingConfig,
        ModelTier,
        # Resilience
        RetryHandler,
        RetryConfig,
        RetryStrategy,
        CircuitBreaker,
        CircuitBreakerConfig,
        AdvancedRateLimiter,
        with_resilience,
    )

    # Initialize router with budget
    router = CostOptimizingRouter(
        config=RoutingConfig(
            default_tier=ModelTier.STANDARD,
            daily_budget=100.0,
        )
    )

    # Resilient API call
    @with_resilience(
        retry_config=RetryConfig(max_retries=3),
        rate_limit_rpm=60,
    )
    async def call_llm(prompt):
        return await llm.ainvoke(prompt)
"""

from langchain_azure_ai.routing.cost_optimizer import (
    CostOptimizingRouter,
    ModelCost,
    ModelTier,
    RoutingConfig,
    UsageMetrics,
    estimate_query_cost,
)
from langchain_azure_ai.routing.resilience import (
    AdvancedRateLimiter,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    RateLimitBucket,
    RateLimitExceeded,
    RetryConfig,
    RetryHandler,
    RetryStats,
    RetryStrategy,
    with_resilience,
)

__all__ = [
    # Cost optimization
    "CostOptimizingRouter",
    "RoutingConfig",
    "ModelTier",
    "ModelCost",
    "UsageMetrics",
    "estimate_query_cost",
    # Retry handling
    "RetryHandler",
    "RetryConfig",
    "RetryStrategy",
    "RetryStats",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    # Rate limiting
    "AdvancedRateLimiter",
    "RateLimitBucket",
    "RateLimitExceeded",
    # Convenience
    "with_resilience",
]
