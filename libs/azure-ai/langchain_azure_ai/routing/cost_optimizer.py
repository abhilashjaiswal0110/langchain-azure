"""Cost optimization routing for LLM operations.

Provides intelligent model selection and cost tracking to minimize
Azure AI costs while maintaining quality.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tiers by cost and capability."""

    PREMIUM = "premium"  # GPT-4o, Claude Opus, o1
    STANDARD = "standard"  # GPT-4o-mini, Claude Sonnet
    ECONOMY = "economy"  # GPT-3.5, DeepSeek
    REASONING = "reasoning"  # o1, DeepSeek-R1 (high cost but specialized)


@dataclass
class ModelCost:
    """Cost information for a model.

    Attributes:
        name: Model identifier.
        tier: Model tier classification.
        cost_per_1k_input_tokens: Cost in USD per 1000 input tokens.
        cost_per_1k_output_tokens: Cost in USD per 1000 output tokens.
        max_tokens: Maximum context window size.
        provider: Model provider (azure_openai, openai, anthropic, etc.).
        supports_streaming: Whether model supports streaming.
        supports_functions: Whether model supports function calling.
        supports_vision: Whether model supports vision/images.
    """

    name: str
    tier: ModelTier
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    max_tokens: int
    provider: str = "azure_openai"
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_vision: bool = False


# Azure OpenAI pricing (as of 2026-02)
MODEL_COSTS: Dict[str, ModelCost] = {
    # Premium tier
    "gpt-4o": ModelCost(
        name="gpt-4o",
        tier=ModelTier.PREMIUM,
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        max_tokens=128000,
        supports_vision=True,
    ),
    "gpt-4o-2024-11-20": ModelCost(
        name="gpt-4o-2024-11-20",
        tier=ModelTier.PREMIUM,
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        max_tokens=128000,
        supports_vision=True,
    ),
    "gpt-4-turbo": ModelCost(
        name="gpt-4-turbo",
        tier=ModelTier.PREMIUM,
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.03,
        max_tokens=128000,
        supports_vision=True,
    ),
    # Standard tier
    "gpt-4o-mini": ModelCost(
        name="gpt-4o-mini",
        tier=ModelTier.STANDARD,
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        max_tokens=128000,
        supports_vision=True,
    ),
    "gpt-4o-mini-2024-07-18": ModelCost(
        name="gpt-4o-mini-2024-07-18",
        tier=ModelTier.STANDARD,
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        max_tokens=128000,
        supports_vision=True,
    ),
    # Economy tier
    "gpt-35-turbo": ModelCost(
        name="gpt-35-turbo",
        tier=ModelTier.ECONOMY,
        cost_per_1k_input_tokens=0.0005,
        cost_per_1k_output_tokens=0.0015,
        max_tokens=16384,
        supports_functions=True,
    ),
    "deepseek-r1": ModelCost(
        name="deepseek-r1",
        tier=ModelTier.ECONOMY,
        cost_per_1k_input_tokens=0.00014,
        cost_per_1k_output_tokens=0.00028,
        max_tokens=64000,
        provider="github_models",
    ),
    # Reasoning tier (specialized for complex reasoning)
    "o1": ModelCost(
        name="o1",
        tier=ModelTier.REASONING,
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.06,
        max_tokens=200000,
        supports_streaming=False,
    ),
    "o1-mini": ModelCost(
        name="o1-mini",
        tier=ModelTier.REASONING,
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.012,
        max_tokens=128000,
        supports_streaming=False,
    ),
}

# Tier defaults for model selection
TIER_DEFAULTS: Dict[ModelTier, str] = {
    ModelTier.PREMIUM: "gpt-4o",
    ModelTier.STANDARD: "gpt-4o-mini",
    ModelTier.ECONOMY: "gpt-35-turbo",
    ModelTier.REASONING: "o1-mini",
}


@dataclass
class UsageMetrics:
    """Token usage and cost metrics.

    Attributes:
        total_input_tokens: Total input tokens used.
        total_output_tokens: Total output tokens used.
        total_cost: Total cost in USD.
        request_count: Number of requests made.
        cache_hits: Number of cache hits.
        model_usage: Usage breakdown by model.
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0
    cache_hits: int = 0
    model_usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        cached: bool = False,
    ) -> None:
        """Record usage metrics.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cost: Cost in USD.
            cached: Whether result was cached.
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.request_count += 1

        if cached:
            self.cache_hits += 1

        if model not in self.model_usage:
            self.model_usage[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "requests": 0,
            }

        self.model_usage[model]["input_tokens"] += input_tokens
        self.model_usage[model]["output_tokens"] += output_tokens
        self.model_usage[model]["cost"] += cost
        self.model_usage[model]["requests"] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": round(self.total_cost, 6),
            "request_count": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": (
                self.cache_hits / self.request_count
                if self.request_count > 0
                else 0.0
            ),
            "average_cost_per_request": (
                self.total_cost / self.request_count
                if self.request_count > 0
                else 0.0
            ),
            "model_usage": self.model_usage,
        }


@dataclass
class RoutingConfig:
    """Configuration for cost-optimizing router.

    Attributes:
        default_tier: Default model tier to use.
        enable_automatic_routing: Automatically select model based on query.
        max_budget_per_request: Maximum cost per request in USD.
        daily_budget: Daily budget limit in USD.
        prefer_streaming: Prefer models that support streaming.
        complexity_thresholds: Word count thresholds for complexity detection.
        complexity_keywords: Keywords indicating high complexity.
    """

    default_tier: ModelTier = ModelTier.STANDARD
    enable_automatic_routing: bool = True
    max_budget_per_request: Optional[float] = None
    daily_budget: Optional[float] = None
    prefer_streaming: bool = True
    complexity_thresholds: Dict[str, int] = field(
        default_factory=lambda: {"high": 50, "medium": 20}
    )
    complexity_keywords: tuple[str, ...] = (
        "analyze",
        "compare",
        "evaluate",
        "reason",
        "explain why",
        "complex",
        "detailed",
        "comprehensive",
        "step by step",
        "thoroughly",
    )


class CostOptimizingRouter:
    """Route requests to cost-appropriate models based on query complexity.

    Automatically selects the most cost-effective model tier based on
    query characteristics while respecting budget constraints.

    Example:
        >>> from langchain_azure_ai.routing import CostOptimizingRouter, RoutingConfig
        >>>
        >>> router = CostOptimizingRouter(
        ...     config=RoutingConfig(
        ...         default_tier=ModelTier.STANDARD,
        ...         enable_automatic_routing=True,
        ...         daily_budget=100.0,
        ...     )
        ... )
        >>>
        >>> # Simple query -> economy model
        >>> model = router.select_model("What is 2+2?")
        >>> print(model)  # "gpt-35-turbo"
        >>>
        >>> # Complex query -> premium model
        >>> model = router.select_model("Analyze the economic implications...")
        >>> print(model)  # "gpt-4o"
        >>>
        >>> # Track costs
        >>> router.record_usage(model, input_tokens=100, output_tokens=50)
        >>> print(router.get_usage_summary())
    """

    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        custom_models: Optional[Dict[str, ModelCost]] = None,
    ):
        """Initialize cost optimizing router.

        Args:
            config: Routing configuration.
            custom_models: Custom model definitions to add.
        """
        self.config = config or RoutingConfig()
        self.models = {**MODEL_COSTS}

        if custom_models:
            self.models.update(custom_models)

        self._metrics = UsageMetrics()
        self._daily_spend = 0.0
        self._daily_reset_time = time.time()

        logger.info(
            f"CostOptimizingRouter initialized: "
            f"default_tier={self.config.default_tier.value}, "
            f"auto_routing={self.config.enable_automatic_routing}"
        )

    def select_model(
        self,
        query: str,
        *,
        require_reasoning: bool = False,
        require_vision: bool = False,
        require_functions: bool = False,
        require_streaming: bool = False,
        force_tier: Optional[ModelTier] = None,
        max_tokens_needed: Optional[int] = None,
    ) -> str:
        """Select most cost-effective model for the query.

        Args:
            query: User query text.
            require_reasoning: Whether advanced reasoning is required.
            require_vision: Whether vision capability is required.
            require_functions: Whether function calling is required.
            require_streaming: Whether streaming is required.
            force_tier: Force a specific tier.
            max_tokens_needed: Maximum tokens needed for context.

        Returns:
            Model name to use.

        Raises:
            ValueError: If no suitable model is found.
        """
        # Check budget constraints
        if not self._check_budget():
            logger.warning("Daily budget exceeded, using economy tier")
            return self._get_model_for_tier(ModelTier.ECONOMY)

        # Force tier if specified
        if force_tier:
            return self._get_model_for_tier(force_tier)

        # Reasoning requirement overrides other considerations
        if require_reasoning:
            logger.debug("Using reasoning tier for complex analysis")
            return self._get_model_for_tier(ModelTier.REASONING)

        # Automatic routing based on complexity
        if self.config.enable_automatic_routing:
            complexity = self._assess_complexity(query)
            tier = self._complexity_to_tier(complexity)
        else:
            tier = self.config.default_tier

        # Find suitable model in tier
        model = self._find_suitable_model(
            tier=tier,
            require_vision=require_vision,
            require_functions=require_functions,
            require_streaming=require_streaming,
            max_tokens_needed=max_tokens_needed,
        )

        logger.debug(f"Selected model: {model} (tier: {tier.value})")
        return model

    def estimate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for model usage.

        Args:
            model_name: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        if model_name not in self.models:
            logger.warning(f"Unknown model for cost estimation: {model_name}")
            return 0.0

        cost_info = self.models[model_name]

        input_cost = (input_tokens / 1000) * cost_info.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * cost_info.cost_per_1k_output_tokens

        total_cost = input_cost + output_cost

        logger.debug(
            f"Estimated cost for {model_name}: "
            f"${total_cost:.6f} ({input_tokens} in, {output_tokens} out)"
        )

        return total_cost

    def record_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
    ) -> float:
        """Record model usage and return cost.

        Args:
            model_name: Model used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cached: Whether result was from cache.

        Returns:
            Cost of this usage in USD.
        """
        cost = self.estimate_cost(model_name, input_tokens, output_tokens)

        self._metrics.record(
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            cached=cached,
        )

        # Update daily spend
        self._check_daily_reset()
        self._daily_spend += cost

        return cost

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary.

        Returns:
            Dictionary with usage metrics.
        """
        return {
            **self._metrics.to_dict(),
            "daily_spend": round(self._daily_spend, 6),
            "daily_budget": self.config.daily_budget,
            "budget_remaining": (
                round(self.config.daily_budget - self._daily_spend, 6)
                if self.config.daily_budget
                else None
            ),
        }

    def reset_metrics(self) -> None:
        """Reset all usage metrics."""
        self._metrics = UsageMetrics()
        self._daily_spend = 0.0
        self._daily_reset_time = time.time()
        logger.info("Usage metrics reset")

    def compare_models(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> List[Dict[str, Any]]:
        """Compare costs across all models.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            List of models with estimated costs, sorted by cost.
        """
        comparisons = []

        for name, cost_info in self.models.items():
            cost = self.estimate_cost(name, input_tokens, output_tokens)
            comparisons.append(
                {
                    "model": name,
                    "tier": cost_info.tier.value,
                    "cost": round(cost, 6),
                    "cost_per_1k_input": cost_info.cost_per_1k_input_tokens,
                    "cost_per_1k_output": cost_info.cost_per_1k_output_tokens,
                }
            )

        return sorted(comparisons, key=lambda x: x["cost"])

    def get_model_info(self, model_name: str) -> Optional[ModelCost]:
        """Get information about a model.

        Args:
            model_name: Model name.

        Returns:
            ModelCost info or None if not found.
        """
        return self.models.get(model_name)

    def _assess_complexity(
        self,
        query: str,
    ) -> Literal["low", "medium", "high"]:
        """Assess query complexity.

        Args:
            query: User query.

        Returns:
            Complexity level.
        """
        query_lower = query.lower()
        word_count = len(query.split())

        # Check for complexity keywords
        has_complexity_keywords = any(
            keyword in query_lower for keyword in self.config.complexity_keywords
        )

        # High complexity
        if has_complexity_keywords or word_count > self.config.complexity_thresholds["high"]:
            return "high"

        # Medium complexity
        if word_count > self.config.complexity_thresholds["medium"]:
            return "medium"

        return "low"

    def _complexity_to_tier(
        self,
        complexity: Literal["low", "medium", "high"],
    ) -> ModelTier:
        """Map complexity to model tier.

        Args:
            complexity: Complexity level.

        Returns:
            Appropriate model tier.
        """
        mapping = {
            "high": ModelTier.PREMIUM,
            "medium": ModelTier.STANDARD,
            "low": ModelTier.ECONOMY,
        }
        return mapping[complexity]

    def _get_model_for_tier(self, tier: ModelTier) -> str:
        """Get default model for tier.

        Args:
            tier: Model tier.

        Returns:
            Model name.
        """
        return TIER_DEFAULTS.get(tier, TIER_DEFAULTS[ModelTier.STANDARD])

    def _find_suitable_model(
        self,
        tier: ModelTier,
        require_vision: bool = False,
        require_functions: bool = False,
        require_streaming: bool = False,
        max_tokens_needed: Optional[int] = None,
    ) -> str:
        """Find a suitable model meeting all requirements.

        Args:
            tier: Preferred tier.
            require_vision: Whether vision is required.
            require_functions: Whether functions are required.
            require_streaming: Whether streaming is required.
            max_tokens_needed: Minimum context window size.

        Returns:
            Model name.

        Raises:
            ValueError: If no suitable model is found.
        """
        # Get candidates in the tier
        candidates = [
            (name, info)
            for name, info in self.models.items()
            if info.tier == tier
        ]

        # Filter by requirements
        suitable = []
        for name, info in candidates:
            if require_vision and not info.supports_vision:
                continue
            if require_functions and not info.supports_functions:
                continue
            if require_streaming and not info.supports_streaming:
                continue
            if max_tokens_needed and info.max_tokens < max_tokens_needed:
                continue
            suitable.append((name, info))

        if suitable:
            # Return cheapest option
            suitable.sort(key=lambda x: x[1].cost_per_1k_input_tokens)
            return suitable[0][0]

        # Fallback to default for tier
        default = self._get_model_for_tier(tier)
        if default in self.models:
            return default

        # Ultimate fallback
        return "gpt-4o-mini"

    def _check_budget(self) -> bool:
        """Check if within budget constraints.

        Returns:
            True if within budget.
        """
        if self.config.daily_budget is None:
            return True

        self._check_daily_reset()
        return self._daily_spend < self.config.daily_budget

    def _check_daily_reset(self) -> None:
        """Reset daily spend if new day."""
        current_time = time.time()
        # Reset after 24 hours
        if current_time - self._daily_reset_time > 86400:
            self._daily_spend = 0.0
            self._daily_reset_time = current_time
            logger.info("Daily budget reset")


# Convenience function for quick cost estimation
def estimate_query_cost(
    query: str,
    expected_output_tokens: int = 500,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Estimate cost for a query.

    Args:
        query: Query text.
        expected_output_tokens: Expected output tokens.
        model: Specific model to estimate for (if None, compares all).

    Returns:
        Cost estimation dictionary.
    """
    # Rough token estimate (4 chars per token average)
    input_tokens = len(query) // 4

    router = CostOptimizingRouter()

    if model:
        cost = router.estimate_cost(model, input_tokens, expected_output_tokens)
        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": expected_output_tokens,
            "estimated_cost": round(cost, 6),
        }

    comparisons = router.compare_models(input_tokens, expected_output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": expected_output_tokens,
        "model_comparisons": comparisons,
        "recommended": comparisons[0]["model"] if comparisons else None,
    }
