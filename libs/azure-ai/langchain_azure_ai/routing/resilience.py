"""Advanced rate limiting and retry strategy implementation.

Provides sophisticated rate limiting, retry logic with exponential backoff,
and circuit breaker patterns for resilient API operations.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RetryStrategy(str, Enum):
    """Retry strategy types."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        strategy: Retry delay strategy.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter: Add randomness to prevent thundering herd.
        jitter_factor: Jitter percentage (0.0 to 1.0).
        retry_on_exceptions: Exception types to retry on.
        retry_on_status_codes: HTTP status codes to retry on.
        dont_retry_on_exceptions: Exception types to never retry.
    """

    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retry_on_exceptions: Tuple[type, ...] = (Exception,)
    retry_on_status_codes: Set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )
    dont_retry_on_exceptions: Tuple[type, ...] = (
        KeyboardInterrupt,
        SystemExit,
        ValueError,
        TypeError,
    )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Consecutive failures before opening.
        success_threshold: Successes needed to close from half-open.
        recovery_timeout: Seconds before attempting recovery.
        half_open_max_calls: Max calls allowed in half-open state.
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


@dataclass
class RetryStats:
    """Statistics for retry operations.

    Attributes:
        total_attempts: Total retry attempts.
        successful_retries: Retries that succeeded.
        failed_retries: Retries that ultimately failed.
        total_delay: Total delay time across all retries.
    """

    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_delay: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_attempts": self.total_attempts,
            "successful_retries": self.successful_retries,
            "failed_retries": self.failed_retries,
            "total_delay_seconds": round(self.total_delay, 2),
            "success_rate": (
                self.successful_retries / self.total_attempts
                if self.total_attempts > 0
                else 0.0
            ),
        }


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to failing services.
    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing recovery)

    Example:
        >>> breaker = CircuitBreaker(config=CircuitBreakerConfig(
        ...     failure_threshold=5,
        ...     recovery_timeout=30.0,
        ... ))
        >>>
        >>> @breaker.protect
        ... async def call_external_service():
        ...     return await http_client.get(url)
        >>>
        >>> try:
        ...     result = await call_external_service()
        ... except CircuitBreakerOpenError:
        ...     result = get_fallback_result()
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration.
        """
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = Lock()

        logger.info(
            f"CircuitBreaker initialized: threshold={self.config.failure_threshold}, "
            f"recovery={self.config.recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        self._check_state_transition()
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        self._success_count = 0
                        logger.info("Circuit breaker entering HALF_OPEN state")

    def can_execute(self) -> bool:
        """Check if request can be executed.

        Returns:
            True if request should proceed.
        """
        state = self.state

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        # Half-open: limit concurrent calls
        with self._lock:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker recovered - entering CLOSED state")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed operation.

        Args:
            exception: The exception that caused the failure.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker returning to OPEN state after failure")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(
                        f"Circuit breaker OPEN after {self._failure_count} failures"
                    )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            logger.info("Circuit breaker reset to CLOSED")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
            },
        }

    def protect(self, func: F) -> F:
        """Decorator to protect function with circuit breaker.

        Args:
            func: Function to protect.

        Returns:
            Protected function.
        """

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.can_execute():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is {self._state.value}"
                )

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.can_execute():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is {self._state.value}"
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class RetryHandler:
    """Advanced retry handler with multiple strategies.

    Provides configurable retry logic with exponential backoff,
    jitter, and circuit breaker integration.

    Example:
        >>> retry = RetryHandler(config=RetryConfig(
        ...     max_retries=3,
        ...     strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        ...     base_delay=1.0,
        ...     jitter=True,
        ... ))
        >>>
        >>> @retry.with_retry
        ... async def call_api():
        ...     return await api_client.request()
        >>>
        >>> # Or use directly
        >>> result = await retry.execute(call_api)
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """Initialize retry handler.

        Args:
            config: Retry configuration.
            circuit_breaker: Optional circuit breaker.
        """
        self.config = config or RetryConfig()
        self.circuit_breaker = circuit_breaker
        self._stats = RetryStats()
        self._fibonacci_cache: Dict[int, int] = {0: 0, 1: 1}

        logger.info(
            f"RetryHandler initialized: max_retries={self.config.max_retries}, "
            f"strategy={self.config.strategy.value}"
        )

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry logic.

        Args:
            func: Function to execute.
            *args, **kwargs: Function arguments.

        Returns:
            Function result.

        Raises:
            Exception: After all retries exhausted.
        """
        last_exception: Optional[Exception] = None
        total_delay = 0.0

        for attempt in range(self.config.max_retries + 1):
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.can_execute():
                raise CircuitBreakerOpenError("Circuit breaker is open")

            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success
                if attempt > 0:
                    self._stats.successful_retries += 1
                    self._stats.total_delay += total_delay
                    logger.info(f"Succeeded after {attempt} retries")

                if self.circuit_breaker:
                    self.circuit_breaker.record_success()

                return result

            except self.config.dont_retry_on_exceptions as e:
                # Don't retry these
                logger.debug(f"Not retrying {type(e).__name__}")
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(e)
                raise

            except Exception as e:
                last_exception = e
                self._stats.total_attempts += 1

                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(e)

                # Check if should retry
                if not self._should_retry(e, attempt):
                    logger.warning(f"Not retrying: {type(e).__name__}")
                    break

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    total_delay += delay

                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)
                else:
                    self._stats.failed_retries += 1
                    self._stats.total_delay += total_delay
                    logger.error(f"All {self.config.max_retries} retries exhausted")

        # All retries failed
        if last_exception:
            raise last_exception

        msg = "Retry logic failed without exception"
        raise RuntimeError(msg)

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if exception should trigger retry.

        Args:
            exception: Exception that occurred.
            attempt: Current attempt number.

        Returns:
            True if should retry.
        """
        # Check if max retries reached
        if attempt >= self.config.max_retries:
            return False

        # Check for HTTP status code
        if hasattr(exception, "status_code"):
            if exception.status_code in self.config.retry_on_status_codes:
                return True

        # Check if exception type is retryable
        if isinstance(exception, self.config.retry_on_exceptions):
            # But not in the don't retry list
            if not isinstance(exception, self.config.dont_retry_on_exceptions):
                return True

        return False

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(
                self.config.base_delay * (2 ** attempt),
                self.config.max_delay,
            )

        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(
                self.config.base_delay * (attempt + 1),
                self.config.max_delay,
            )

        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            fib = self._fibonacci(attempt + 1)
            delay = min(
                self.config.base_delay * fib,
                self.config.max_delay,
            )

        else:  # FIXED_DELAY
            delay = self.config.base_delay

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.0, delay)

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number with memoization.

        Args:
            n: Position in Fibonacci sequence.

        Returns:
            Fibonacci number.
        """
        if n not in self._fibonacci_cache:
            self._fibonacci_cache[n] = self._fibonacci(n - 1) + self._fibonacci(n - 2)
        return self._fibonacci_cache[n]

    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self._stats = RetryStats()

    def with_retry(self, func: F) -> F:
        """Decorator to add retry logic to function.

        Args:
            func: Function to wrap.

        Returns:
            Wrapped function.
        """

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.execute(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.execute(func, *args, **kwargs)
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting.

    Attributes:
        capacity: Maximum tokens in bucket.
        tokens: Current token count.
        refill_rate: Tokens added per second.
        last_update: Last update timestamp.
    """

    capacity: float
    tokens: float
    refill_rate: float
    last_update: float = field(default_factory=time.monotonic)

    def consume(self, tokens: float = 1.0) -> Tuple[bool, float, float]:
        """Attempt to consume tokens.

        Args:
            tokens: Tokens to consume.

        Returns:
            Tuple of (allowed, remaining, retry_after).
        """
        now = time.monotonic()
        elapsed = now - self.last_update

        # Refill tokens
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, self.tokens, 0.0
        else:
            tokens_needed = tokens - self.tokens
            retry_after = tokens_needed / self.refill_rate if self.refill_rate > 0 else 60.0
            return False, self.tokens, retry_after


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple dimensions and strategies.

    Supports per-user, per-endpoint, and global rate limiting
    with token bucket algorithm.

    Example:
        >>> limiter = AdvancedRateLimiter(
        ...     requests_per_minute=60,
        ...     burst_size=10,
        ... )
        >>>
        >>> # Check if request is allowed
        >>> allowed, remaining, retry_after = limiter.check("user123")
        >>> if not allowed:
        ...     raise RateLimitExceeded(retry_after)
        >>>
        >>> # Or use as decorator
        >>> @limiter.limit(key_func=lambda req: req.user_id)
        ... async def handle_request(req):
        ...     return await process(req)
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
        enable_per_endpoint: bool = True,
        endpoint_limits: Optional[Dict[str, int]] = None,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Base rate limit per minute.
            requests_per_hour: Rate limit per hour.
            burst_size: Maximum burst capacity.
            enable_per_endpoint: Enable per-endpoint limits.
            endpoint_limits: Custom limits per endpoint.
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.enable_per_endpoint = enable_per_endpoint
        self.endpoint_limits = endpoint_limits or {}

        self._buckets: Dict[str, RateLimitBucket] = {}
        self._lock = Lock()
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 300  # 5 minutes
        self._bucket_ttl = 3600  # 1 hour
        self._bucket_last_access: Dict[str, float] = {}

        logger.info(
            f"AdvancedRateLimiter initialized: "
            f"{requests_per_minute} req/min, burst={burst_size}"
        )

    def check(
        self,
        key: str,
        endpoint: Optional[str] = None,
        tokens: float = 1.0,
    ) -> Tuple[bool, float, float]:
        """Check if request is allowed.

        Args:
            key: Client identifier.
            endpoint: Optional endpoint for per-endpoint limits.
            tokens: Tokens to consume.

        Returns:
            Tuple of (allowed, remaining, retry_after).
        """
        with self._lock:
            # Periodic cleanup
            self._maybe_cleanup()

            # Get or create bucket
            bucket_key = self._make_bucket_key(key, endpoint)
            bucket = self._get_or_create_bucket(bucket_key, endpoint)

            # Update access time
            self._bucket_last_access[bucket_key] = time.monotonic()

            # Try to consume tokens
            return bucket.consume(tokens)

    def _make_bucket_key(
        self,
        key: str,
        endpoint: Optional[str],
    ) -> str:
        """Create bucket key."""
        if endpoint and self.enable_per_endpoint:
            return f"{key}:{endpoint}"
        return key

    def _get_or_create_bucket(
        self,
        bucket_key: str,
        endpoint: Optional[str],
    ) -> RateLimitBucket:
        """Get or create rate limit bucket."""
        if bucket_key not in self._buckets:
            # Determine rate limit
            if endpoint and endpoint in self.endpoint_limits:
                rate_per_minute = self.endpoint_limits[endpoint]
            else:
                rate_per_minute = self.requests_per_minute

            rate_per_second = rate_per_minute / 60.0

            self._buckets[bucket_key] = RateLimitBucket(
                capacity=self.burst_size,
                tokens=self.burst_size,
                refill_rate=rate_per_second,
            )

        return self._buckets[bucket_key]

    def _maybe_cleanup(self) -> None:
        """Clean up old buckets if interval has passed."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._cleanup_old_buckets()
        self._last_cleanup = now

    def _cleanup_old_buckets(self) -> None:
        """Remove stale buckets."""
        now = time.monotonic()
        stale_keys = [
            key for key, last_access in self._bucket_last_access.items()
            if now - last_access > self._bucket_ttl
        ]

        for key in stale_keys:
            self._buckets.pop(key, None)
            self._bucket_last_access.pop(key, None)

        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limit buckets")

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "active_buckets": len(self._buckets),
            "requests_per_minute": self.requests_per_minute,
            "burst_size": self.burst_size,
            "endpoint_limits": self.endpoint_limits,
        }

    def limit(
        self,
        key_func: Optional[Callable[..., str]] = None,
        endpoint: Optional[str] = None,
    ) -> Callable[[F], F]:
        """Decorator to apply rate limiting.

        Args:
            key_func: Function to extract rate limit key from args.
            endpoint: Endpoint name for per-endpoint limits.

        Returns:
            Decorated function.
        """

        def decorator(func: F) -> F:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Get rate limit key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    key = "default"

                allowed, remaining, retry_after = self.check(key, endpoint)

                if not allowed:
                    raise RateLimitExceeded(
                        retry_after=retry_after,
                        remaining=remaining,
                    )

                return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    key = "default"

                allowed, remaining, retry_after = self.check(key, endpoint)

                if not allowed:
                    raise RateLimitExceeded(
                        retry_after=retry_after,
                        remaining=remaining,
                    )

                return func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        retry_after: float,
        remaining: float = 0.0,
        message: str = "Rate limit exceeded",
    ):
        """Initialize exception.

        Args:
            retry_after: Seconds until retry is allowed.
            remaining: Remaining tokens.
            message: Error message.
        """
        self.retry_after = retry_after
        self.remaining = remaining
        super().__init__(f"{message}. Retry after {retry_after:.1f}s")


# Convenience function for creating a resilient wrapper
def with_resilience(
    retry_config: Optional[RetryConfig] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    rate_limit_rpm: Optional[int] = None,
) -> Callable[[F], F]:
    """Create a resilient wrapper with retry, circuit breaker, and rate limiting.

    Args:
        retry_config: Retry configuration.
        circuit_config: Circuit breaker configuration.
        rate_limit_rpm: Rate limit in requests per minute.

    Returns:
        Decorator function.

    Example:
        >>> @with_resilience(
        ...     retry_config=RetryConfig(max_retries=3),
        ...     circuit_config=CircuitBreakerConfig(failure_threshold=5),
        ...     rate_limit_rpm=60,
        ... )
        ... async def call_external_api():
        ...     return await api.request()
    """
    circuit_breaker = None
    if circuit_config:
        circuit_breaker = CircuitBreaker(circuit_config)

    retry_handler = RetryHandler(
        config=retry_config or RetryConfig(),
        circuit_breaker=circuit_breaker,
    )

    rate_limiter = None
    if rate_limit_rpm:
        rate_limiter = AdvancedRateLimiter(requests_per_minute=rate_limit_rpm)

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check rate limit
            if rate_limiter:
                allowed, _, retry_after = rate_limiter.check("global")
                if not allowed:
                    raise RateLimitExceeded(retry_after)

            # Execute with retry
            return await retry_handler.execute(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if rate_limiter:
                allowed, _, retry_after = rate_limiter.check("global")
                if not allowed:
                    raise RateLimitExceeded(retry_after)

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                retry_handler.execute(func, *args, **kwargs)
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
