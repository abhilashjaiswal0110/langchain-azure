"""Enhanced distributed tracing with Azure Monitor integration.

Provides comprehensive tracing capabilities for LangChain operations
with automatic span creation, context propagation, and metrics.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(str, Enum):
    """Type of operation being traced."""

    LLM = "llm"
    EMBEDDING = "embedding"
    VECTORSTORE = "vectorstore"
    RETRIEVAL = "retrieval"
    AGENT = "agent"
    TOOL = "tool"
    CHAIN = "chain"
    HTTP = "http"
    DATABASE = "database"


@dataclass
class SpanContext:
    """Context information for a span.

    Attributes:
        trace_id: Unique trace identifier.
        span_id: Unique span identifier.
        parent_span_id: Parent span ID if nested.
        baggage: Key-value pairs propagated across process boundaries.
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class SpanAttributes:
    """Standard attributes for spans.

    Attributes:
        service_name: Name of the service.
        operation_name: Name of the operation.
        span_kind: Type of span.
        model_name: LLM model name (if applicable).
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        duration_ms: Operation duration in milliseconds.
        status: Operation status (ok, error).
        error_message: Error message if failed.
        custom: Additional custom attributes.
    """

    service_name: str = "langchain-azure-ai"
    operation_name: str = ""
    span_kind: SpanKind = SpanKind.CHAIN
    model_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    duration_ms: Optional[float] = None
    status: str = "ok"
    error_message: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        result = {
            "service.name": self.service_name,
            "operation.name": self.operation_name,
            "span.kind": self.span_kind.value,
            "status": self.status,
        }

        if self.model_name:
            result["llm.model"] = self.model_name
        if self.input_tokens is not None:
            result["llm.input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            result["llm.output_tokens"] = self.output_tokens
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.error_message:
            result["error.message"] = self.error_message

        result.update(self.custom)
        return result


class EnhancedTracer:
    """Enhanced distributed tracing with Azure Monitor and OpenTelemetry.

    Provides decorator-based tracing, context propagation, and integration
    with both Azure Monitor (Application Insights) and LangSmith.

    Example:
        >>> from langchain_azure_ai.observability import EnhancedTracer
        >>>
        >>> tracer = EnhancedTracer(
        ...     service_name="my-agent-app",
        ...     enable_azure_monitor=True,
        ... )
        >>>
        >>> @tracer.trace_function(span_name="process_query")
        ... async def process_query(query: str) -> str:
        ...     return await agent.ainvoke(query)
        >>>
        >>> # Specialized decorators
        >>> @tracer.trace_llm_call(model="gpt-4o")
        ... async def call_llm(prompt: str):
        ...     return await llm.ainvoke(prompt)
        >>>
        >>> @tracer.trace_vector_search(vectorstore="azure_search")
        ... async def search(query: str, k: int = 5):
        ...     return await vectorstore.asimilarity_search(query, k=k)
    """

    def __init__(
        self,
        service_name: str = "langchain-azure-ai",
        enable_azure_monitor: bool = True,
        enable_console: bool = False,
        connection_string: Optional[str] = None,
        sample_rate: float = 1.0,
    ):
        """Initialize enhanced tracer.

        Args:
            service_name: Service name for telemetry.
            enable_azure_monitor: Enable Azure Monitor/App Insights export.
            enable_console: Enable console output for debugging.
            connection_string: App Insights connection string.
            sample_rate: Sampling rate (0.0 to 1.0, 1.0 = 100%).
        """
        self.service_name = service_name
        self.sample_rate = sample_rate
        self._tracer: Optional[Any] = None
        self._meter: Optional[Any] = None

        # Initialize OpenTelemetry
        self._setup_opentelemetry(
            enable_azure_monitor=enable_azure_monitor,
            enable_console=enable_console,
            connection_string=connection_string,
        )

        logger.info(
            f"EnhancedTracer initialized: service={service_name}, "
            f"azure_monitor={enable_azure_monitor}"
        )

    def _setup_opentelemetry(
        self,
        enable_azure_monitor: bool,
        enable_console: bool,
        connection_string: Optional[str],
    ) -> None:
        """Set up OpenTelemetry instrumentation."""
        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource

            # Create resource
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": "1.0.0",
                }
            )

            # Set up tracer provider
            tracer_provider = TracerProvider(resource=resource)

            # Add exporters
            if enable_azure_monitor:
                try:
                    from azure.monitor.opentelemetry.exporter import (
                        AzureMonitorTraceExporter,
                    )
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor

                    conn_str = connection_string or os.getenv(
                        "APPLICATIONINSIGHTS_CONNECTION_STRING"
                    )
                    if conn_str:
                        exporter = AzureMonitorTraceExporter(
                            connection_string=conn_str
                        )
                        tracer_provider.add_span_processor(
                            BatchSpanProcessor(exporter)
                        )
                        logger.info("Azure Monitor exporter configured")
                except ImportError:
                    logger.warning(
                        "azure-monitor-opentelemetry-exporter not installed"
                    )

            if enable_console:
                try:
                    from opentelemetry.sdk.trace.export import (
                        ConsoleSpanExporter,
                        SimpleSpanProcessor,
                    )

                    tracer_provider.add_span_processor(
                        SimpleSpanProcessor(ConsoleSpanExporter())
                    )
                except ImportError:
                    pass

            trace.set_tracer_provider(tracer_provider)
            self._tracer = trace.get_tracer(self.service_name)

            # Set up meter provider
            meter_provider = MeterProvider(resource=resource)
            metrics.set_meter_provider(meter_provider)
            self._meter = metrics.get_meter(self.service_name)

        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. Tracing will be limited. "
                "Install with: pip install opentelemetry-sdk"
            )

    def trace_function(
        self,
        *,
        span_name: Optional[str] = None,
        span_kind: SpanKind = SpanKind.CHAIN,
        attributes: Optional[Dict[str, Any]] = None,
        capture_args: bool = False,
        capture_result: bool = False,
    ) -> Callable[[F], F]:
        """Decorator to trace function execution.

        Args:
            span_name: Custom span name (defaults to function name).
            span_kind: Type of operation.
            attributes: Additional span attributes.
            capture_args: Include function arguments in span.
            capture_result: Include return value metadata in span.

        Returns:
            Decorated function.
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                name = span_name or f"{func.__module__}.{func.__name__}"
                return await self._trace_execution(
                    func,
                    name,
                    span_kind,
                    attributes or {},
                    capture_args,
                    capture_result,
                    args,
                    kwargs,
                    is_async=True,
                )

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                name = span_name or f"{func.__module__}.{func.__name__}"

                if self._tracer is None:
                    return func(*args, **kwargs)

                try:
                    from opentelemetry.trace import Status, StatusCode

                    with self._tracer.start_as_current_span(name) as span:
                        self._add_attributes(span, span_kind, attributes or {})

                        if capture_args:
                            self._add_args_to_span(span, args, kwargs)

                        start_time = time.perf_counter()

                        try:
                            result = func(*args, **kwargs)

                            if capture_result:
                                self._add_result_to_span(span, result)

                            span.set_status(Status(StatusCode.OK))
                            return result

                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                            raise

                        finally:
                            duration_ms = (time.perf_counter() - start_time) * 1000
                            span.set_attribute("duration_ms", duration_ms)

                except ImportError:
                    return func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator

    async def _trace_execution(
        self,
        func: Callable,
        span_name: str,
        span_kind: SpanKind,
        attributes: Dict[str, Any],
        capture_args: bool,
        capture_result: bool,
        args: tuple,
        kwargs: dict,
        is_async: bool,
    ) -> Any:
        """Execute function with tracing."""
        if self._tracer is None:
            if is_async:
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        try:
            from opentelemetry.trace import Status, StatusCode

            with self._tracer.start_as_current_span(span_name) as span:
                self._add_attributes(span, span_kind, attributes)

                if capture_args:
                    self._add_args_to_span(span, args, kwargs)

                start_time = time.perf_counter()

                try:
                    if is_async:
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    if capture_result:
                        self._add_result_to_span(span, result)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)

        except ImportError:
            if is_async:
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

    def trace_llm_call(
        self,
        model: str,
        provider: str = "azure_openai",
    ) -> Callable[[F], F]:
        """Specialized decorator for LLM calls.

        Args:
            model: Model name (e.g., "gpt-4o").
            provider: LLM provider.

        Returns:
            Decorated function.
        """
        return self.trace_function(
            span_name=f"llm.{provider}.{model}",
            span_kind=SpanKind.LLM,
            attributes={
                "llm.model": model,
                "llm.provider": provider,
            },
            capture_args=False,
            capture_result=False,
        )

    def trace_vector_search(
        self,
        vectorstore: str,
    ) -> Callable[[F], F]:
        """Specialized decorator for vector search operations.

        Args:
            vectorstore: Vector store name.

        Returns:
            Decorated function.
        """
        return self.trace_function(
            span_name=f"vectorstore.{vectorstore}.search",
            span_kind=SpanKind.VECTORSTORE,
            attributes={
                "vectorstore.type": vectorstore,
            },
            capture_args=True,
            capture_result=True,
        )

    def trace_agent_execution(
        self,
        agent_name: str,
    ) -> Callable[[F], F]:
        """Specialized decorator for agent execution.

        Args:
            agent_name: Agent name.

        Returns:
            Decorated function.
        """
        return self.trace_function(
            span_name=f"agent.{agent_name}.execute",
            span_kind=SpanKind.AGENT,
            attributes={
                "agent.name": agent_name,
            },
            capture_args=True,
            capture_result=True,
        )

    def trace_tool_call(
        self,
        tool_name: str,
    ) -> Callable[[F], F]:
        """Specialized decorator for tool calls.

        Args:
            tool_name: Tool name.

        Returns:
            Decorated function.
        """
        return self.trace_function(
            span_name=f"tool.{tool_name}",
            span_kind=SpanKind.TOOL,
            attributes={
                "tool.name": tool_name,
            },
            capture_args=True,
            capture_result=True,
        )

    @contextmanager
    def span(
        self,
        name: str,
        span_kind: SpanKind = SpanKind.CHAIN,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """Create a manual span context.

        Args:
            name: Span name.
            span_kind: Type of operation.
            attributes: Span attributes.

        Yields:
            Span object (or None if tracing unavailable).
        """
        if self._tracer is None:
            yield None
            return

        try:
            from opentelemetry.trace import Status, StatusCode

            with self._tracer.start_as_current_span(name) as span:
                self._add_attributes(span, span_kind, attributes or {})
                start_time = time.perf_counter()

                try:
                    yield span
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)

        except ImportError:
            yield None

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        span_kind: SpanKind = SpanKind.CHAIN,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Create an async span context.

        Args:
            name: Span name.
            span_kind: Type of operation.
            attributes: Span attributes.

        Yields:
            Span object (or None if tracing unavailable).
        """
        if self._tracer is None:
            yield None
            return

        try:
            from opentelemetry.trace import Status, StatusCode

            with self._tracer.start_as_current_span(name) as span:
                self._add_attributes(span, span_kind, attributes or {})
                start_time = time.perf_counter()

                try:
                    yield span
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)

        except ImportError:
            yield None

    def _add_attributes(
        self,
        span: Any,
        span_kind: SpanKind,
        attributes: Dict[str, Any],
    ) -> None:
        """Add attributes to span."""
        span.set_attribute("span.kind", span_kind.value)
        span.set_attribute("service.name", self.service_name)

        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)

    def _add_args_to_span(
        self,
        span: Any,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Add function arguments to span attributes."""
        span.set_attribute("function.args_count", len(args))
        span.set_attribute("function.kwargs_count", len(kwargs))

        # Safe kwargs to log
        safe_kwargs = ["query", "k", "score_threshold", "filter", "model"]
        for key in safe_kwargs:
            if key in kwargs:
                value = kwargs[key]
                str_value = str(value)[:500]
                span.set_attribute(f"function.kwargs.{key}", str_value)

    def _add_result_to_span(
        self,
        span: Any,
        result: Any,
    ) -> None:
        """Add result metadata to span."""
        span.set_attribute("function.result_type", type(result).__name__)

        if hasattr(result, "__len__"):
            try:
                span.set_attribute("function.result_length", len(result))
            except TypeError:
                pass

    def record_token_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float] = None,
    ) -> None:
        """Record token usage metrics.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cost: Optional cost in USD.
        """
        if self._meter is None:
            return

        try:
            # Get or create counters
            input_counter = self._meter.create_counter(
                "llm.token.input",
                description="Input tokens consumed",
                unit="tokens",
            )
            output_counter = self._meter.create_counter(
                "llm.token.output",
                description="Output tokens generated",
                unit="tokens",
            )

            labels = {"model": model}
            input_counter.add(input_tokens, labels)
            output_counter.add(output_tokens, labels)

            if cost is not None:
                cost_counter = self._meter.create_counter(
                    "llm.cost",
                    description="LLM cost in USD",
                    unit="usd",
                )
                cost_counter.add(cost, labels)

        except Exception as e:
            logger.debug(f"Failed to record metrics: {e}")


# Global tracer instance
_global_tracer: Optional[EnhancedTracer] = None


def get_tracer(
    service_name: Optional[str] = None,
) -> EnhancedTracer:
    """Get or create global tracer instance.

    Args:
        service_name: Optional service name override.

    Returns:
        EnhancedTracer instance.
    """
    global _global_tracer

    if _global_tracer is None:
        _global_tracer = EnhancedTracer(
            service_name=service_name
            or os.getenv("OTEL_SERVICE_NAME", "langchain-azure-ai"),
            enable_azure_monitor=os.getenv("ENABLE_AZURE_MONITOR", "true").lower()
            == "true",
            connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
        )

    return _global_tracer


def trace_function(
    span_name: Optional[str] = None,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Convenience decorator using global tracer.

    Args:
        span_name: Span name.
        **kwargs: Additional arguments for trace_function.

    Returns:
        Decorated function.
    """
    return get_tracer().trace_function(span_name=span_name, **kwargs)
