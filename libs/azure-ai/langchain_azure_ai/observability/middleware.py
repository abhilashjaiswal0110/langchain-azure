"""FastAPI middleware for observability, security, and request tracking.

This module provides middleware components for:
- Request/response logging
- Execution time tracking
- OpenTelemetry context propagation
- Error tracking and alerting
- Rate limiting (token bucket algorithm)

Usage:
    from fastapi import FastAPI
    from langchain_azure_ai.observability.middleware import (
        TracingMiddleware,
        RequestLoggingMiddleware,
        RateLimitMiddleware,
    )
    
    app = FastAPI()
    app.add_middleware(TracingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.
    
    Attributes:
        requests_per_minute: Maximum requests per minute per client.
        requests_per_hour: Maximum requests per hour per client.
        burst_size: Maximum burst size (token bucket capacity).
        by_ip: Rate limit by IP address.
        by_user: Rate limit by user ID (from header or token).
        by_api_key: Rate limit by API key.
        user_header: Header name for user identification.
        api_key_header: Header name for API key.
        exempt_paths: Paths exempt from rate limiting.
        endpoint_limits: Custom limits per endpoint (path -> requests_per_minute).
    """
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    by_ip: bool = True
    by_user: bool = True
    by_api_key: bool = True
    user_header: str = "X-User-ID"
    api_key_header: str = "X-API-Key"
    exempt_paths: Tuple[str, ...] = ("/health", "/metrics", "/docs", "/openapi.json")
    endpoint_limits: Dict[str, int] = field(default_factory=dict)


class TokenBucket:
    """Token bucket implementation for rate limiting.
    
    The token bucket algorithm allows:
    - Steady rate limiting at configured rate
    - Burst handling up to bucket capacity
    - Graceful degradation under load
    
    Thread-safe for concurrent access.
    """
    
    def __init__(
        self,
        rate: float,
        capacity: int,
        initial_tokens: Optional[float] = None,
    ):
        """Initialize the token bucket.
        
        Args:
            rate: Token refill rate per second.
            capacity: Maximum bucket capacity (burst size).
            initial_tokens: Initial token count (defaults to capacity).
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_update = time.monotonic()
        self._lock = Lock()
    
    def consume(self, tokens: int = 1) -> Tuple[bool, float, float]:
        """Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume.
            
        Returns:
            Tuple of (allowed, remaining_tokens, retry_after_seconds).
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            
            # Refill tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, self.tokens, 0.0
            else:
                # Calculate retry-after time
                tokens_needed = tokens - self.tokens
                retry_after = tokens_needed / self.rate if self.rate > 0 else 60.0
                return False, self.tokens, retry_after


class RateLimiter:
    """Rate limiter using token bucket algorithm with multiple dimensions.
    
    Supports rate limiting by:
    - IP address
    - User ID
    - API key
    - Endpoint-specific limits
    
    Automatically cleans up stale entries to prevent memory leaks.
    """
    
    def __init__(self, config: RateLimitConfig):
        """Initialize the rate limiter.
        
        Args:
            config: Rate limiting configuration.
        """
        self.config = config
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = Lock()
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 300  # 5 minutes
        self._bucket_ttl = 3600  # 1 hour
        self._bucket_last_access: Dict[str, float] = {}
    
    def _get_client_key(self, request: Request) -> str:
        """Generate a unique key for the client.
        
        Args:
            request: The incoming request.
            
        Returns:
            Unique client identifier string.
        """
        parts = []
        
        if self.config.by_ip:
            client_ip = self._get_client_ip(request)
            parts.append(f"ip:{client_ip}")
        
        if self.config.by_user:
            user_id = request.headers.get(self.config.user_header)
            if user_id:
                parts.append(f"user:{user_id}")
        
        if self.config.by_api_key:
            api_key = request.headers.get(self.config.api_key_header)
            if api_key:
                # Hash API key deterministically for privacy using SHA-256
                import hashlib
                api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
                parts.append(f"key:{api_key_hash}")
        
        return "|".join(parts) if parts else "anonymous"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, handling proxies.
        
        Args:
            request: The incoming request.
            
        Returns:
            Client IP address string.
        """
        # Check X-Forwarded-For header (from load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_bucket(self, key: str, endpoint: str) -> TokenBucket:
        """Get or create a token bucket for the given key.
        
        Args:
            key: Client identifier.
            endpoint: Request endpoint path.
            
        Returns:
            Token bucket for rate limiting.
        """
        bucket_key = f"{key}:{endpoint}"
        
        with self._lock:
            # Periodic cleanup
            now = time.monotonic()
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_stale_buckets()
                self._last_cleanup = now
            
            # Get or create bucket
            if bucket_key not in self._buckets:
                # Check for endpoint-specific limits
                rate_per_minute = self.config.endpoint_limits.get(
                    endpoint, self.config.requests_per_minute
                )
                rate_per_second = rate_per_minute / 60.0
                
                self._buckets[bucket_key] = TokenBucket(
                    rate=rate_per_second,
                    capacity=self.config.burst_size,
                )
            
            self._bucket_last_access[bucket_key] = now
            return self._buckets[bucket_key]
    
    def _cleanup_stale_buckets(self) -> None:
        """Remove stale bucket entries to prevent memory leaks."""
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
    
    def check(self, request: Request) -> Tuple[bool, float, float, str]:
        """Check if request is allowed under rate limits.
        
        Args:
            request: The incoming request.
            
        Returns:
            Tuple of (allowed, remaining, retry_after, client_key).
        """
        # Skip exempt paths
        if request.url.path in self.config.exempt_paths:
            return True, float('inf'), 0.0, "exempt"
        
        client_key = self._get_client_key(request)
        endpoint = request.url.path
        
        bucket = self._get_bucket(client_key, endpoint)
        allowed, remaining, retry_after = bucket.consume(1)
        
        return allowed, remaining, retry_after, client_key


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for API rate limiting using token bucket algorithm.
    
    Features:
    - Token bucket algorithm for smooth rate limiting
    - Per-client tracking (IP, user, API key)
    - Endpoint-specific rate limits
    - Automatic stale entry cleanup
    - Standard rate limit headers (RateLimit-*, Retry-After)
    
    Example:
        >>> from fastapi import FastAPI
        >>> from langchain_azure_ai.observability.middleware import (
        ...     RateLimitMiddleware,
        ...     RateLimitConfig,
        ... )
        >>> 
        >>> app = FastAPI()
        >>> 
        >>> # Basic usage with defaults (60 req/min)
        >>> app.add_middleware(RateLimitMiddleware)
        >>> 
        >>> # Custom configuration
        >>> config = RateLimitConfig(
        ...     requests_per_minute=100,
        ...     burst_size=20,
        ...     endpoint_limits={
        ...         "/api/deepagent": 30,  # Lower limit for expensive endpoints
        ...     },
        ... )
        >>> app.add_middleware(RateLimitMiddleware, config=config)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[RateLimitConfig] = None,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        enabled: bool = True,
    ):
        """Initialize the rate limit middleware.
        
        Args:
            app: The ASGI application.
            config: Full rate limit configuration (overrides other params).
            requests_per_minute: Requests per minute per client (if no config).
            burst_size: Burst capacity (if no config).
            enabled: Whether rate limiting is enabled.
        """
        super().__init__(app)
        
        # Allow environment variable override
        self.enabled = enabled and os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        
        if config is not None:
            self.config = config
        else:
            # Read from environment with fallbacks
            env_rpm = os.getenv("RATE_LIMIT_RPM")
            env_burst = os.getenv("RATE_LIMIT_BURST")
            
            self.config = RateLimitConfig(
                requests_per_minute=int(env_rpm) if env_rpm else requests_per_minute,
                burst_size=int(env_burst) if env_burst else burst_size,
            )
        
        self.limiter = RateLimiter(self.config)
        
        logger.info(
            f"Rate limiting {'enabled' if self.enabled else 'disabled'}: "
            f"{self.config.requests_per_minute} req/min, burst={self.config.burst_size}"
        )
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request with rate limiting."""
        if not self.enabled:
            return await call_next(request)
        
        allowed, remaining, retry_after, client_key = self.limiter.check(request)
        
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_key} on {request.url.path}. "
                f"Retry after {retry_after:.1f}s"
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please retry after {int(retry_after) + 1} seconds.",
                    "retry_after": int(retry_after) + 1,
                },
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "RateLimit-Limit": str(self.config.requests_per_minute),
                    "RateLimit-Remaining": "0",
                    "RateLimit-Reset": str(int(time.time() + retry_after)),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["RateLimit-Limit"] = str(self.config.requests_per_minute)
        response.headers["RateLimit-Remaining"] = str(int(remaining))
        response.headers["RateLimit-Reset"] = str(int(time.time() + 60))
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses.
    
    This middleware logs:
    - Request method, path, and client IP
    - Response status code and execution time
    - Request/response body (configurable)
    
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(
        ...     RequestLoggingMiddleware,
        ...     log_request_body=True,
        ...     log_response_body=False,
        ... )
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[list] = None,
        max_body_log_size: int = 1000,
    ):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application.
            log_request_body: Whether to log request bodies.
            log_response_body: Whether to log response bodies.
            exclude_paths: List of paths to exclude from logging.
            max_body_log_size: Maximum body size to log (truncates beyond this).
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self.max_body_log_size = max_body_log_size
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request and log details."""
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Log request
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "unknown"),
        }
        
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                body_str = body.decode("utf-8")[:self.max_body_log_size]
                log_data["request_body"] = body_str
                # Reset body for downstream handlers
                request._body = body
            except Exception as e:
                log_data["request_body_error"] = str(e)
        
        logger.info(f"[{request_id}] → {request.method} {request.url.path}", extra=log_data)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log response
            response_log = {
                **log_data,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            }
            
            log_level = logging.INFO if response.status_code < 400 else logging.WARNING
            logger.log(
                log_level,
                f"[{request_id}] ← {response.status_code} ({duration_ms:.2f}ms)",
                extra=response_log,
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] ✗ Error: {str(e)} ({duration_ms:.2f}ms)",
                extra={
                    **log_data,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                },
                exc_info=True,
            )
            raise


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware for OpenTelemetry distributed tracing.
    
    This middleware:
    - Creates spans for HTTP requests
    - Propagates trace context
    - Records request/response attributes
    - Captures errors as span events
    
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(TracingMiddleware)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "azure-ai-agents",
        exclude_paths: Optional[list] = None,
    ):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application.
            service_name: Name of the service for tracing.
            exclude_paths: List of paths to exclude from tracing.
        """
        super().__init__(app)
        self.service_name = service_name
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self._tracer = None
    
    def _get_tracer(self):
        """Lazy load the tracer."""
        if self._tracer is None:
            try:
                from opentelemetry import trace
                self._tracer = trace.get_tracer(self.service_name)
            except ImportError:
                return None
        return self._tracer
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request with tracing."""
        # Skip tracing for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        tracer = self._get_tracer()
        if tracer is None:
            return await call_next(request)
        
        # Extract trace context from headers
        try:
            from opentelemetry.propagate import extract
            from opentelemetry.trace import SpanKind
            
            context = extract(dict(request.headers))
            
            with tracer.start_as_current_span(
                f"{request.method} {request.url.path}",
                context=context,
                kind=SpanKind.SERVER,
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.route": request.url.path,
                    "http.scheme": request.url.scheme,
                    "http.host": request.url.hostname or "",
                    "http.client_ip": request.client.host if request.client else "",
                    "http.user_agent": request.headers.get("user-agent", ""),
                },
            ) as span:
                try:
                    response = await call_next(request)
                    
                    span.set_attribute("http.status_code", response.status_code)
                    
                    if response.status_code >= 400:
                        from opentelemetry.trace import StatusCode
                        span.set_status(
                            StatusCode.ERROR,
                            f"HTTP {response.status_code}",
                        )
                    
                    return response
                    
                except Exception as e:
                    from opentelemetry.trace import StatusCode
                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise
                    
        except ImportError:
            return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for recording HTTP metrics.
    
    This middleware records:
    - Request count by endpoint and status
    - Request duration histogram
    - Active request gauge
    
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(MetricsMiddleware)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "azure-ai-agents",
        exclude_paths: Optional[list] = None,
    ):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application.
            service_name: Name of the service for metrics.
            exclude_paths: List of paths to exclude from metrics.
        """
        super().__init__(app)
        self.service_name = service_name
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self._meter = None
        self._request_counter = None
        self._duration_histogram = None
        self._active_requests = None
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Set up metrics instruments."""
        try:
            from opentelemetry import metrics
            
            self._meter = metrics.get_meter(self.service_name)
            
            self._request_counter = self._meter.create_counter(
                name="http.server.requests",
                description="Total HTTP requests",
                unit="requests",
            )
            
            self._duration_histogram = self._meter.create_histogram(
                name="http.server.duration",
                description="HTTP request duration",
                unit="ms",
            )
            
            self._active_requests = self._meter.create_up_down_counter(
                name="http.server.active_requests",
                description="Number of active HTTP requests",
                unit="requests",
            )
            
        except ImportError:
            logger.debug("OpenTelemetry metrics not available")
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process the request and record metrics."""
        # Skip metrics for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        if self._meter is None:
            return await call_next(request)
        
        labels = {
            "method": request.method,
            "route": request.url.path,
        }
        
        # Track active requests
        if self._active_requests:
            self._active_requests.add(1, labels)
        
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            
            labels["status_code"] = str(response.status_code)
            
            return response
            
        except Exception:
            labels["status_code"] = "500"
            raise
            
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            if self._duration_histogram:
                self._duration_histogram.record(duration_ms, labels)
            
            if self._request_counter:
                self._request_counter.add(1, labels)
            
            if self._active_requests:
                self._active_requests.add(-1, {"method": request.method, "route": request.url.path})


# Export public API
__all__ = [
    "RequestLoggingMiddleware",
    "TracingMiddleware",
    "MetricsMiddleware",
    "RateLimitMiddleware",
    "RateLimitConfig",
    "RateLimiter",
    "TokenBucket",
]
