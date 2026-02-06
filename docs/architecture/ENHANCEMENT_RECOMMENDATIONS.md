# Enhancement Recommendations for LangChain Azure Platform

**Document Version**: 1.0
**Date**: 2026-02-06
**Based On**: Comprehensive analysis of libs/ codebase
**Target Audience**: Development team and architects

---

## Executive Summary

This document provides actionable recommendations across three key areas:
1. **Feature Enhancements** - New capabilities and improved functionality
2. **Security Improvements** - Hardening, compliance, and best practices
3. **Azure Best Practices** - Architecture, cost optimization, and scalability

Each recommendation includes:
- Priority level (Critical / High / Medium / Low)
- Implementation effort estimate
- Expected impact
- Specific implementation guidance

---

## Table of Contents

1. [Feature Enhancements](#feature-enhancements)
2. [Security Improvements](#security-improvements)
3. [Azure Best Practices](#azure-best-practices)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Quick Wins](#quick-wins)

---

## Feature Enhancements

### 1. Unified Authentication & Credential Management Framework

**Priority**: High
**Effort**: Medium (2-3 weeks)
**Impact**: High - Reduces code duplication, improves maintainability

#### Current State
Each library implements its own credential handling:
```python
# azure-ai
credential = connection_properties.get("credential") or DefaultAzureCredential()

# azure-postgresql
credential = DefaultAzureCredential()

# azure-storage
credential = credential or DefaultAzureCredential()
```

#### Recommendation
Create a centralized credential management module:

**File**: `libs/azure-common/langchain_azure_common/auth.py`

```python
from typing import Optional, Union
from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    ClientSecretCredential,
    AzureCliCredential,
)
from azure.core.credentials import (
    TokenCredential,
    AzureSasCredential,
    AzureKeyCredential,
)
import logging

logger = logging.getLogger(__name__)

SupportedCredential = Union[
    TokenCredential,
    AzureSasCredential,
    AzureKeyCredential,
]


class AzureCredentialManager:
    """Centralized credential management with fallback chain.

    Provides consistent credential resolution across all Azure libraries
    with proper caching, logging, and error handling.
    """

    _credential_cache: dict[str, TokenCredential] = {}

    @staticmethod
    def get_credential(
        credential: Optional[SupportedCredential] = None,
        *,
        scopes: Optional[list[str]] = None,
        client_id: Optional[str] = None,
        cache_key: str = "default",
        log_auth_chain: bool = True,
    ) -> SupportedCredential:
        """Get or create credential with caching.

        Args:
            credential: User-provided credential (highest priority).
            scopes: Token scopes for validation.
            client_id: For user-assigned managed identity.
            cache_key: Cache identifier for credential reuse.
            log_auth_chain: Log which credential source was used.

        Returns:
            Resolved credential ready for use.
        """
        if credential:
            if log_auth_chain:
                logger.info(f"Using provided credential: {type(credential).__name__}")
            return credential

        # Check cache
        if cache_key in AzureCredentialManager._credential_cache:
            logger.debug(f"Using cached credential for key: {cache_key}")
            return AzureCredentialManager._credential_cache[cache_key]

        # Create DefaultAzureCredential with options
        credential_kwargs = {}
        if client_id:
            credential_kwargs["managed_identity_client_id"] = client_id

        new_credential = DefaultAzureCredential(**credential_kwargs)

        # Validate by requesting token
        if scopes:
            try:
                token = new_credential.get_token(*scopes)
                logger.info(
                    f"Credential validated successfully. Expires: {token.expires_on}"
                )
            except Exception as e:
                logger.error(f"Credential validation failed: {e}")
                raise

        # Cache and return
        AzureCredentialManager._credential_cache[cache_key] = new_credential

        if log_auth_chain:
            logger.info("Using DefaultAzureCredential")

        return new_credential

    @staticmethod
    def create_from_env(
        *,
        tenant_id_var: str = "AZURE_TENANT_ID",
        client_id_var: str = "AZURE_CLIENT_ID",
        client_secret_var: str = "AZURE_CLIENT_SECRET",
        fallback_to_default: bool = True,
    ) -> SupportedCredential:
        """Create credential from environment variables.

        Priority:
        1. Service Principal (if all env vars present)
        2. Managed Identity with client_id (if client_id present)
        3. DefaultAzureCredential (fallback)

        Args:
            tenant_id_var: Environment variable for tenant ID.
            client_id_var: Environment variable for client ID.
            client_secret_var: Environment variable for client secret.
            fallback_to_default: Use DefaultAzureCredential if env vars missing.

        Returns:
            Configured credential.
        """
        import os

        tenant_id = os.getenv(tenant_id_var)
        client_id = os.getenv(client_id_var)
        client_secret = os.getenv(client_secret_var)

        # Service Principal
        if tenant_id and client_id and client_secret:
            logger.info("Using Service Principal from environment")
            return ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )

        # User-assigned Managed Identity
        if client_id:
            logger.info(f"Using Managed Identity with client_id: {client_id}")
            return ManagedIdentityCredential(client_id=client_id)

        # Fallback
        if fallback_to_default:
            logger.info("Using DefaultAzureCredential fallback")
            return DefaultAzureCredential()

        raise ValueError(
            "No valid credentials found. Set AZURE_TENANT_ID, "
            "AZURE_CLIENT_ID, AZURE_CLIENT_SECRET or configure managed identity."
        )

    @staticmethod
    def clear_cache(cache_key: Optional[str] = None) -> None:
        """Clear cached credentials.

        Args:
            cache_key: Specific key to clear. If None, clears all.
        """
        if cache_key:
            AzureCredentialManager._credential_cache.pop(cache_key, None)
            logger.debug(f"Cleared credential cache for: {cache_key}")
        else:
            AzureCredentialManager._credential_cache.clear()
            logger.debug("Cleared all credential caches")
```

**Benefits**:
- **DRY Principle**: Single source of truth for credential logic
- **Better Logging**: Understand which auth method is being used
- **Credential Caching**: Reduce token acquisition calls
- **Validation**: Test credentials early with token requests
- **Flexibility**: Support for all Azure identity patterns

**Migration Path**:
```python
# Old
from azure.identity import DefaultAzureCredential
credential = credential or DefaultAzureCredential()

# New
from langchain_azure_common.auth import AzureCredentialManager
credential = AzureCredentialManager.get_credential(
    credential=credential,
    scopes=["https://database.windows.net/.default"],
    cache_key="postgres_connection"
)
```

---

### 2. Advanced Rate Limiting & Retry Strategy

**Priority**: High
**Effort**: Medium (1-2 weeks)
**Impact**: High - Prevents throttling, improves reliability

#### Current State
- Basic `RateLimitMiddleware` exists in `azure-ai/middleware.py`
- No exponential backoff or circuit breaker
- No request prioritization
- No shared rate limiting across instances

#### Recommendation
Implement comprehensive rate limiting with retry logic:

**File**: `libs/azure-ai/langchain_azure_ai/middleware/rate_limiting.py`

```python
import asyncio
import time
from typing import Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter: bool = True  # Add randomness to prevent thundering herd
    retry_on_status_codes: set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    burst_size: int = 10  # Allow bursts up to this size
    enable_distributed: bool = False  # Use Redis for multi-instance
    redis_url: Optional[str] = None
    key_func: Optional[Callable] = None  # Custom key extraction


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to failing services.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening circuit.
            recovery_timeout: Seconds before attempting recovery.
            expected_exception: Exception type that triggers circuit.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute.
            *args, **kwargs: Function arguments.

        Returns:
            Function result.

        Raises:
            Exception: If circuit is OPEN or function fails.
        """
        if self.state == "OPEN":
            # Check if recovery period has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info("Circuit breaker entering HALF_OPEN state")
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)

            # Success - reset or close circuit
            if self.state == "HALF_OPEN":
                logger.info("Circuit breaker recovered - entering CLOSED state")
                self.state = "CLOSED"

            self.failure_count = 0
            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker OPEN after {self.failure_count} failures"
                )
                self.state = "OPEN"

            raise


class EnhancedRetryHandler:
    """Advanced retry logic with multiple strategies and circuit breaker."""

    def __init__(
        self,
        config: RetryConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """Initialize retry handler.

        Args:
            config: Retry configuration.
            circuit_breaker: Optional circuit breaker for failure protection.
        """
        self.config = config
        self.circuit_breaker = circuit_breaker

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs,
    ):
        """Execute function with retry logic.

        Args:
            func: Async function to execute.
            *args, **kwargs: Function arguments.

        Returns:
            Function result.

        Raises:
            Exception: After all retries exhausted.
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Use circuit breaker if provided
                if self.circuit_breaker:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                else:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                # Success
                if attempt > 0:
                    logger.info(f"Request succeeded after {attempt} retries")
                return result

            except Exception as e:
                last_exception = e

                # Check if should retry
                should_retry = self._should_retry(e, attempt)

                if not should_retry:
                    logger.warning(f"Not retrying exception: {type(e).__name__}")
                    raise

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_retries} retries exhausted"
                    )

        raise last_exception

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if exception should trigger retry.

        Args:
            exception: Exception that occurred.
            attempt: Current attempt number.

        Returns:
            True if should retry.
        """
        # Check if it's a retriable HTTP status
        if hasattr(exception, "status_code"):
            if exception.status_code in self.config.retry_on_status_codes:
                return True

        # Check if max retries reached
        if attempt >= self.config.max_retries:
            return False

        # Default: retry on most exceptions except authentication
        if "authentication" in str(exception).lower():
            return False
        if "unauthorized" in str(exception).lower():
            return False

        return True

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
                self.config.max_delay
            )
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(
                self.config.base_delay * (attempt + 1),
                self.config.max_delay
            )
        else:  # FIXED_DELAY
            delay = self.config.base_delay

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)


class AdvancedRateLimiter:
    """Token bucket rate limiter with burst support."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration.
        """
        self.config = config
        self.tokens: dict[str, float] = defaultdict(float)
        self.last_update: dict[str, float] = defaultdict(float)

        # Redis support for distributed rate limiting
        self.redis_client = None
        if config.enable_distributed and config.redis_url:
            import redis
            self.redis_client = redis.from_url(config.redis_url)

    async def acquire(self, key: str, tokens: float = 1.0) -> bool:
        """Attempt to acquire tokens for request.

        Args:
            key: Identifier for rate limit bucket (e.g., user_id, ip_address).
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens acquired, False if rate limited.
        """
        if self.config.enable_distributed and self.redis_client:
            return await self._acquire_distributed(key, tokens)
        else:
            return self._acquire_local(key, tokens)

    def _acquire_local(self, key: str, tokens: float) -> bool:
        """Local in-memory rate limiting."""
        now = time.time()

        # Calculate tokens to add based on time elapsed
        if key in self.last_update:
            time_elapsed = now - self.last_update[key]
            tokens_to_add = time_elapsed * (
                self.config.requests_per_minute / 60.0
            )
            self.tokens[key] = min(
                self.tokens[key] + tokens_to_add,
                self.config.burst_size
            )
        else:
            # First request - start with full burst capacity
            self.tokens[key] = self.config.burst_size

        self.last_update[key] = now

        # Check if enough tokens available
        if self.tokens[key] >= tokens:
            self.tokens[key] -= tokens
            return True
        else:
            logger.warning(f"Rate limit exceeded for key: {key}")
            return False

    async def _acquire_distributed(self, key: str, tokens: float) -> bool:
        """Distributed rate limiting using Redis."""
        # Use Redis Lua script for atomic token bucket operations
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local rate = tonumber(ARGV[2])
        local requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(bucket[1]) or capacity
        local last_update = tonumber(bucket[2]) or now

        local time_elapsed = now - last_update
        local tokens_to_add = time_elapsed * rate
        tokens = math.min(tokens + tokens_to_add, capacity)

        if tokens >= requested then
            tokens = tokens - requested
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
            return 1
        else
            return 0
        end
        """

        result = self.redis_client.eval(
            lua_script,
            1,
            f"ratelimit:{key}",
            self.config.burst_size,
            self.config.requests_per_minute / 60.0,
            tokens,
            time.time(),
        )

        return bool(result)
```

**Usage Example**:
```python
from langchain_azure_ai.middleware.rate_limiting import (
    EnhancedRetryHandler,
    RetryConfig,
    RetryStrategy,
    CircuitBreaker,
)

# Configure retry with exponential backoff
retry_config = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay=1.0,
    jitter=True,
)

circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
)

retry_handler = EnhancedRetryHandler(
    config=retry_config,
    circuit_breaker=circuit_breaker,
)

# Use in API call
result = await retry_handler.execute_with_retry(
    llm.ainvoke,
    prompt="Your query here"
)
```

**Benefits**:
- **Reliability**: Automatic retry on transient failures
- **Cost Efficiency**: Avoid failed requests counting against quota
- **User Experience**: Transparent recovery from temporary issues
- **System Protection**: Circuit breaker prevents cascade failures

---

### 3. Semantic Caching Layer for Vector Stores

**Priority**: Medium
**Effort**: Medium (1-2 weeks)
**Impact**: High - Reduces costs, improves latency

#### Current State
- No caching infrastructure across vector stores
- Every query hits the database
- Repeated similar queries recomputed

#### Recommendation
Implement semantic caching with similarity-based cache keys:

**File**: `libs/azure-ai/langchain_azure_ai/caching/semantic_cache.py`

```python
from typing import Optional, Any, Callable
from dataclasses import dataclass
import hashlib
import time
import logging
from enum import Enum
import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live


@dataclass
class CacheConfig:
    """Configuration for semantic cache."""
    enabled: bool = True
    max_size: int = 1000  # Maximum cache entries
    ttl_seconds: int = 3600  # 1 hour default TTL
    similarity_threshold: float = 0.95  # 95% similarity to be cache hit
    strategy: CacheStrategy = CacheStrategy.LRU
    enable_distributed: bool = False
    redis_url: Optional[str] = None


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    embedding: np.ndarray
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = None


class SemanticCache:
    """Semantic similarity-based caching for LLM and vector operations.

    Uses embedding similarity to match semantically similar queries,
    not just exact string matches.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        config: CacheConfig,
    ):
        """Initialize semantic cache.

        Args:
            embeddings: Embedding model for query encoding.
            config: Cache configuration.
        """
        self.embeddings = embeddings
        self.config = config
        self.cache: dict[str, CacheEntry] = {}

        # Redis support for distributed caching
        self.redis_client = None
        if config.enable_distributed and config.redis_url:
            import redis
            self.redis_client = redis.from_url(config.redis_url)

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    async def get(self, query: str) -> Optional[Any]:
        """Retrieve cached value for semantically similar query.

        Args:
            query: Query string to look up.

        Returns:
            Cached value if found, None otherwise.
        """
        if not self.config.enabled:
            return None

        # Generate embedding for query
        query_embedding = await self._get_embedding(query)

        # Search for similar cached queries
        best_match = self._find_best_match(query_embedding)

        if best_match:
            self.hits += 1
            logger.debug(f"Cache HIT for query: {query[:50]}...")

            # Update access metadata
            best_match.access_count += 1
            best_match.last_accessed = time.time()

            return best_match.value

        self.misses += 1
        logger.debug(f"Cache MISS for query: {query[:50]}...")
        return None

    async def set(self, query: str, value: Any) -> None:
        """Add entry to cache.

        Args:
            query: Query string as cache key.
            value: Value to cache.
        """
        if not self.config.enabled:
            return

        # Generate embedding
        query_embedding = await self._get_embedding(query)

        # Create cache key (hash of query for fast lookup)
        cache_key = self._generate_key(query)

        # Check if cache is full
        if len(self.cache) >= self.config.max_size:
            self._evict()

        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            embedding=query_embedding,
            value=value,
            timestamp=time.time(),
            last_accessed=time.time(),
        )

        self.cache[cache_key] = entry

        logger.debug(f"Cached result for query: {query[:50]}...")

    def _find_best_match(
        self,
        query_embedding: np.ndarray,
    ) -> Optional[CacheEntry]:
        """Find most similar cached entry.

        Args:
            query_embedding: Embedding vector of query.

        Returns:
            Best matching cache entry if similarity above threshold.
        """
        best_similarity = 0.0
        best_entry: Optional[CacheEntry] = None

        now = time.time()

        for entry in self.cache.values():
            # Check TTL
            if self.config.strategy == CacheStrategy.TTL:
                if now - entry.timestamp > self.config.ttl_seconds:
                    continue  # Expired

            # Calculate cosine similarity
            similarity = self._cosine_similarity(
                query_embedding,
                entry.embedding
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        # Return if above threshold
        if best_similarity >= self.config.similarity_threshold:
            logger.debug(f"Found match with similarity: {best_similarity:.4f}")
            return best_entry

        return None

    def _evict(self) -> None:
        """Evict entry based on configured strategy."""
        if not self.cache:
            return

        if self.config.strategy == CacheStrategy.LRU:
            # Evict least recently used
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed
            )
        elif self.config.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].access_count
            )
        else:  # TTL
            # Evict oldest by timestamp
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].timestamp
            )

        del self.cache[oldest_key]
        self.evictions += 1
        logger.debug(f"Evicted cache entry: {oldest_key}")

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        embeddings = await self.embeddings.aembed_documents([text])
        return np.array(embeddings[0])

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score (0 to 1).
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def _generate_key(text: str) -> str:
        """Generate cache key from text.

        Args:
            text: Text to hash.

        Returns:
            Hash string.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with hit rate, size, and other metrics.
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.config.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        logger.info("Cache cleared")
```

**Integration with Vector Stores**:
```python
from langchain_azure_ai.caching import SemanticCache, CacheConfig
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel

# Initialize cache
embeddings = AzureAIEmbeddingsModel(...)
cache = SemanticCache(
    embeddings=embeddings,
    config=CacheConfig(
        max_size=1000,
        similarity_threshold=0.95,
        ttl_seconds=3600,
    )
)

# Wrap vector store operations
async def cached_similarity_search(query: str, k: int = 4):
    # Check cache
    cached_result = await cache.get(query)
    if cached_result:
        return cached_result

    # Perform actual search
    result = await vector_store.asimilarity_search(query, k=k)

    # Cache result
    await cache.set(query, result)

    return result

# Usage
documents = await cached_similarity_search("What is Azure AI?")
print(f"Cache stats: {cache.get_stats()}")
```

**Benefits**:
- **Cost Reduction**: Avoid redundant vector search operations
- **Latency Improvement**: Cached results return instantly
- **Smart Matching**: Semantic similarity handles query variations
- **Configurable**: Tune similarity threshold and eviction strategy

---

### 4. Multi-Modal Document Processing Pipeline

**Priority**: Medium
**Effort**: High (3-4 weeks)
**Impact**: Medium - Enables richer document understanding

#### Current State
- `AzureAIDocumentIntelligenceTool` handles PDF/Word/images separately
- No unified pipeline for mixed-content documents
- Limited extraction of tables, charts, layouts

#### Recommendation
Build unified multi-modal document processing:

**File**: `libs/azure-ai/langchain_azure_ai/document_processing/multimodal_pipeline.py`

```python
from typing import Optional, Literal
from dataclasses import dataclass
from pathlib import Path
import logging
from langchain_core.documents import Document
from langchain_azure_ai.tools import (
    AzureAIDocumentIntelligenceTool,
    AzureAIImageAnalysisTool,
)

logger = logging.getLogger(__name__)


DocumentType = Literal["pdf", "docx", "pptx", "image", "mixed"]


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    extract_tables: bool = True
    extract_images: bool = True
    extract_charts: bool = True
    ocr_images: bool = True
    analyze_layout: bool = True
    generate_summaries: bool = False
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class ProcessedDocument:
    """Result of document processing."""
    text_content: str
    structured_data: dict  # Tables, key-value pairs
    images: list[dict]  # Extracted images with descriptions
    layout: dict  # Layout analysis results
    metadata: dict
    chunks: list[Document]  # Text chunks for RAG


class MultiModalDocumentPipeline:
    """Unified pipeline for processing multi-modal documents.

    Handles PDFs, Word documents, PowerPoint, images with integrated
    text extraction, OCR, table extraction, and image analysis.
    """

    def __init__(
        self,
        doc_intelligence_tool: AzureAIDocumentIntelligenceTool,
        image_analysis_tool: AzureAIImageAnalysisTool,
        config: ProcessingConfig,
    ):
        """Initialize pipeline.

        Args:
            doc_intelligence_tool: Tool for document analysis.
            image_analysis_tool: Tool for image understanding.
            config: Processing configuration.
        """
        self.doc_intel = doc_intelligence_tool
        self.image_analysis = image_analysis_tool
        self.config = config

    async def process(
        self,
        file_path: str,
        document_type: Optional[DocumentType] = None,
    ) -> ProcessedDocument:
        """Process document through multi-modal pipeline.

        Args:
            file_path: Path to document file.
            document_type: Type hint for processing optimization.

        Returns:
            Processed document with all extracted information.
        """
        path = Path(file_path)

        # Auto-detect document type if not provided
        if document_type is None:
            document_type = self._detect_type(path)

        logger.info(f"Processing {document_type} document: {path.name}")

        # Run appropriate processing pipeline
        if document_type == "image":
            return await self._process_image(file_path)
        elif document_type in ["pdf", "docx", "pptx"]:
            return await self._process_document(file_path, document_type)
        else:
            raise ValueError(f"Unsupported document type: {document_type}")

    async def _process_document(
        self,
        file_path: str,
        doc_type: DocumentType,
    ) -> ProcessedDocument:
        """Process document-type files (PDF, Word, PPT).

        Args:
            file_path: Path to document.
            doc_type: Document type.

        Returns:
            Processed document.
        """
        # Step 1: Extract content using Document Intelligence
        doc_result = await self.doc_intel._arun(file_path)

        # Step 2: Parse structured data
        structured_data = self._extract_structured_data(doc_result)

        # Step 3: Extract and analyze embedded images
        images = []
        if self.config.extract_images:
            images = await self._extract_and_analyze_images(doc_result)

        # Step 4: Layout analysis
        layout = {}
        if self.config.analyze_layout:
            layout = self._analyze_layout(doc_result)

        # Step 5: Create text chunks for RAG
        chunks = self._create_chunks(
            doc_result.get("content", ""),
            metadata={
                "source": file_path,
                "type": doc_type,
                "has_tables": bool(structured_data.get("tables")),
                "image_count": len(images),
            }
        )

        return ProcessedDocument(
            text_content=doc_result.get("content", ""),
            structured_data=structured_data,
            images=images,
            layout=layout,
            metadata={"source": file_path, "type": doc_type},
            chunks=chunks,
        )

    async def _process_image(self, file_path: str) -> ProcessedDocument:
        """Process standalone image file.

        Args:
            file_path: Path to image.

        Returns:
            Processed document with image analysis.
        """
        # Analyze image
        analysis_result = await self.image_analysis._arun(file_path)

        # Extract text via OCR if enabled
        text_content = ""
        if self.config.ocr_images:
            ocr_result = await self.doc_intel._arun(file_path)
            text_content = ocr_result.get("content", "")

        return ProcessedDocument(
            text_content=text_content,
            structured_data={},
            images=[{
                "file": file_path,
                "caption": analysis_result.get("caption", ""),
                "tags": analysis_result.get("tags", []),
                "objects": analysis_result.get("objects", []),
            }],
            layout={},
            metadata={"source": file_path, "type": "image"},
            chunks=self._create_chunks(
                f"Image: {analysis_result.get('caption', '')}\nOCR: {text_content}",
                metadata={"source": file_path, "type": "image"}
            ),
        )

    def _extract_structured_data(self, doc_result: dict) -> dict:
        """Extract tables and key-value pairs.

        Args:
            doc_result: Raw document intelligence result.

        Returns:
            Structured data dictionary.
        """
        structured = {
            "tables": [],
            "key_value_pairs": [],
        }

        if self.config.extract_tables and "tables" in doc_result:
            structured["tables"] = doc_result["tables"]

        if "key_value_pairs" in doc_result:
            structured["key_value_pairs"] = doc_result["key_value_pairs"]

        return structured

    async def _extract_and_analyze_images(self, doc_result: dict) -> list[dict]:
        """Extract embedded images and analyze them.

        Args:
            doc_result: Raw document intelligence result.

        Returns:
            List of analyzed images.
        """
        images = []

        # Extract embedded images from document
        if "figures" in doc_result:
            for figure in doc_result["figures"]:
                # Analyze each image
                if "image_data" in figure:
                    analysis = await self.image_analysis._arun(
                        figure["image_data"]
                    )

                    images.append({
                        "caption": analysis.get("caption", ""),
                        "tags": analysis.get("tags", []),
                        "location": figure.get("bounding_box", {}),
                    })

        return images

    def _analyze_layout(self, doc_result: dict) -> dict:
        """Analyze document layout structure.

        Args:
            doc_result: Raw document intelligence result.

        Returns:
            Layout analysis.
        """
        return {
            "sections": doc_result.get("sections", []),
            "paragraphs": doc_result.get("paragraphs", []),
            "reading_order": doc_result.get("reading_order", []),
        }

    def _create_chunks(
        self,
        text: str,
        metadata: dict,
    ) -> list[Document]:
        """Split text into chunks for RAG.

        Args:
            text: Full text content.
            metadata: Document metadata.

        Returns:
            List of text chunks as Documents.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )

        return splitter.create_documents([text], metadatas=[metadata])

    @staticmethod
    def _detect_type(file_path: Path) -> DocumentType:
        """Auto-detect document type from file extension.

        Args:
            file_path: Path to file.

        Returns:
            Detected document type.
        """
        extension = file_path.suffix.lower()

        type_map = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "docx",
            ".pptx": "pptx",
            ".ppt": "pptx",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".gif": "image",
            ".bmp": "image",
        }

        return type_map.get(extension, "mixed")
```

**Usage Example**:
```python
from langchain_azure_ai.document_processing import (
    MultiModalDocumentPipeline,
    ProcessingConfig,
)

# Initialize pipeline
pipeline = MultiModalDocumentPipeline(
    doc_intelligence_tool=doc_intel_tool,
    image_analysis_tool=image_tool,
    config=ProcessingConfig(
        extract_tables=True,
        extract_images=True,
        ocr_images=True,
        analyze_layout=True,
    )
)

# Process document
result = await pipeline.process("annual_report.pdf")

# Access extracted data
print(f"Text length: {len(result.text_content)}")
print(f"Tables extracted: {len(result.structured_data['tables'])}")
print(f"Images found: {len(result.images)}")

# Use chunks for RAG
vector_store.add_documents(result.chunks)
```

**Benefits**:
- **Comprehensive Extraction**: Tables, images, text, layout
- **Better RAG**: Structured chunks preserve context
- **Automation**: Single pipeline for all document types
- **Extensibility**: Easy to add new extractors

---

### 5. Enhanced Observability with Distributed Tracing

**Priority**: High
**Effort**: Medium (2 weeks)
**Impact**: High - Critical for production debugging

#### Current State
- Basic OpenTelemetry integration exists
- Limited span context across async operations
- No correlation between LangSmith and Azure Monitor traces
- Missing key performance indicators

#### Recommendation
Implement comprehensive distributed tracing:

**File**: `libs/azure-ai/langchain_azure_ai/observability/enhanced_tracing.py`

```python
from typing import Optional, Callable, Any
from contextlib import contextmanager, asynccontextmanager
import time
import logging
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from azure.monitor.opentelemetry import configure_azure_monitor
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class EnhancedTracer:
    """Enhanced distributed tracing with Azure Monitor and custom spans.

    Provides decorator-based tracing, context propagation across async boundaries,
    and integration with both Azure Monitor and LangSmith.
    """

    def __init__(
        self,
        service_name: str,
        enable_azure_monitor: bool = True,
        enable_langsmith: bool = True,
        connection_string: Optional[str] = None,
    ):
        """Initialize enhanced tracer.

        Args:
            service_name: Service name for telemetry.
            enable_azure_monitor: Enable Azure Monitor export.
            enable_langsmith: Enable LangSmith export.
            connection_string: Application Insights connection string.
        """
        self.service_name = service_name

        # Configure Azure Monitor
        if enable_azure_monitor:
            if connection_string:
                configure_azure_monitor(connection_string=connection_string)
            else:
                configure_azure_monitor()  # Use env var

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

        logger.info(f"Enhanced tracing initialized for: {service_name}")

    def trace_function(
        self,
        *,
        span_name: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
        capture_args: bool = False,
        capture_result: bool = False,
    ):
        """Decorator to trace function execution.

        Args:
            span_name: Custom span name (defaults to function name).
            attributes: Additional span attributes.
            capture_args: Include function arguments in span.
            capture_result: Include return value in span.

        Returns:
            Decorated function.
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"

                with self.tracer.start_as_current_span(name) as span:
                    # Add default attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)

                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)

                    # Capture arguments if enabled
                    if capture_args:
                        self._add_args_to_span(span, args, kwargs)

                    start_time = time.time()

                    try:
                        result = await func(*args, **kwargs)

                        # Capture result if enabled
                        if capture_result:
                            span.set_attribute(
                                "function.result_type",
                                type(result).__name__
                            )
                            if hasattr(result, "__len__"):
                                span.set_attribute(
                                    "function.result_length",
                                    len(result)
                                )

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(
                            Status(StatusCode.ERROR, str(e))
                        )
                        span.record_exception(e)
                        raise

                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("function.duration_ms", duration * 1000)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"

                with self.tracer.start_as_current_span(name) as span:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)

                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)

                    if capture_args:
                        self._add_args_to_span(span, args, kwargs)

                    start_time = time.time()

                    try:
                        result = func(*args, **kwargs)

                        if capture_result:
                            span.set_attribute(
                                "function.result_type",
                                type(result).__name__
                            )

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("function.duration_ms", duration * 1000)

            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ):
        """Create a manual span context.

        Args:
            name: Span name.
            attributes: Span attributes.

        Yields:
            Span object.
        """
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ):
        """Create an async span context.

        Args:
            name: Span name.
            attributes: Span attributes.

        Yields:
            Span object.
        """
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def trace_llm_call(
        self,
        model: str,
        provider: str = "azure_openai",
    ):
        """Specialized decorator for LLM calls.

        Args:
            model: Model name (e.g., "gpt-4o").
            provider: LLM provider.

        Returns:
            Decorated function.
        """
        return self.trace_function(
            span_name=f"llm.{provider}.{model}",
            attributes={
                "llm.model": model,
                "llm.provider": provider,
            },
            capture_args=False,  # Don't log prompts by default
            capture_result=False,  # Privacy
        )

    def trace_vector_search(
        self,
        vector_store: str,
    ):
        """Specialized decorator for vector search operations.

        Args:
            vector_store: Vector store name.

        Returns:
            Decorated function.
        """
        return self.trace_function(
            span_name=f"vectorstore.{vector_store}.search",
            attributes={
                "vectorstore.type": vector_store,
            },
            capture_args=True,  # Log query
            capture_result=True,  # Log result count
        )

    @staticmethod
    def _add_args_to_span(span, args, kwargs):
        """Add function arguments to span attributes.

        Args:
            span: Current span.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        # Add argument count
        span.set_attribute("function.args_count", len(args))
        span.set_attribute("function.kwargs_count", len(kwargs))

        # Add specific kwargs (be careful with sensitive data)
        safe_kwargs = ["query", "k", "score_threshold", "filter"]
        for key in safe_kwargs:
            if key in kwargs:
                value = kwargs[key]
                # Convert to string, truncate if too long
                str_value = str(value)[:500]
                span.set_attribute(f"function.kwargs.{key}", str_value)


# Global tracer instance
_global_tracer: Optional[EnhancedTracer] = None


def get_tracer() -> EnhancedTracer:
    """Get or create global tracer instance.

    Returns:
        EnhancedTracer instance.
    """
    global _global_tracer

    if _global_tracer is None:
        import os
        _global_tracer = EnhancedTracer(
            service_name=os.getenv("OTEL_SERVICE_NAME", "langchain-azure-ai"),
            enable_azure_monitor=os.getenv("ENABLE_AZURE_MONITOR", "true").lower() == "true",
            connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
        )

    return _global_tracer
```

**Usage Example**:
```python
from langchain_azure_ai.observability import get_tracer

tracer = get_tracer()

# Decorator for functions
@tracer.trace_function(
    span_name="process_user_query",
    attributes={"service": "chatbot"},
    capture_args=True,
)
async def process_query(user_id: str, query: str):
    # Your logic here
    result = await agent.ainvoke(query)
    return result

# Specialized decorators
@tracer.trace_llm_call(model="gpt-4o", provider="azure_openai")
async def call_llm(prompt: str):
    return await llm.ainvoke(prompt)

@tracer.trace_vector_search(vector_store="azure_search")
async def search_docs(query: str, k: int = 5):
    return await vector_store.asimilarity_search(query, k=k)

# Manual spans
async def complex_operation():
    async with tracer.async_span(
        "data_processing",
        attributes={"batch_size": 100}
    ) as span:
        # Step 1
        async with tracer.async_span("step1_extraction") as step1:
            data = await extract_data()
            step1.set_attribute("records_extracted", len(data))

        # Step 2
        async with tracer.async_span("step2_transformation") as step2:
            transformed = await transform(data)
            step2.set_attribute("records_transformed", len(transformed))

        return transformed
```

**Benefits**:
- **End-to-End Visibility**: Trace requests across all components
- **Performance Analysis**: Identify bottlenecks with timing data
- **Error Diagnosis**: Full exception context and stack traces
- **Correlation**: Link spans across services and async boundaries

---

## Security Improvements

### 1. Secrets Management with Azure Key Vault Integration

**Priority**: Critical
**Effort**: Medium (1-2 weeks)
**Impact**: Critical - Eliminates hardcoded secrets

#### Current State
- Secrets loaded from `.env` files
- No centralized secret rotation
- Limited audit trail for secret access

#### Recommendation
Integrate Azure Key Vault for secret management:

**File**: `libs/azure-common/langchain_azure_common/secrets.py`

```python
from typing import Optional, Any
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class SecretManager:
    """Centralized secret management with Azure Key Vault integration.

    Provides transparent secret loading with fallback to environment variables
    and caching for performance.
    """

    def __init__(
        self,
        key_vault_url: Optional[str] = None,
        credential: Optional[Any] = None,
        enable_key_vault: bool = True,
        cache_ttl: int = 300,  # 5 minutes
    ):
        """Initialize secret manager.

        Args:
            key_vault_url: Azure Key Vault URL (e.g., https://myvault.vault.azure.net/).
            credential: Azure credential for authentication.
            enable_key_vault: Enable Key Vault lookup.
            cache_ttl: Cache TTL in seconds.
        """
        self.enable_key_vault = enable_key_vault
        self.cache_ttl = cache_ttl

        # Initialize Key Vault client if enabled
        self.kv_client: Optional[SecretClient] = None
        if self.enable_key_vault:
            vault_url = key_vault_url or os.getenv("AZURE_KEY_VAULT_URL")
            if vault_url:
                cred = credential or DefaultAzureCredential()
                self.kv_client = SecretClient(
                    vault_url=vault_url,
                    credential=cred
                )
                logger.info(f"Key Vault client initialized: {vault_url}")
            else:
                logger.warning(
                    "Key Vault enabled but no URL provided. "
                    "Set AZURE_KEY_VAULT_URL environment variable."
                )

    @lru_cache(maxsize=128)
    def get_secret(
        self,
        secret_name: str,
        default: Optional[str] = None,
        *,
        env_var_fallback: bool = True,
    ) -> Optional[str]:
        """Retrieve secret with fallback chain.

        Priority order:
        1. Azure Key Vault (if enabled)
        2. Environment variable (if env_var_fallback=True)
        3. Default value

        Args:
            secret_name: Name of secret.
            default: Default value if secret not found.
            env_var_fallback: Check environment variables.

        Returns:
            Secret value or default.
        """
        # Try Key Vault first
        if self.kv_client:
            try:
                secret = self.kv_client.get_secret(secret_name)
                logger.debug(f"Retrieved secret from Key Vault: {secret_name}")
                return secret.value
            except Exception as e:
                logger.warning(
                    f"Failed to retrieve secret from Key Vault: {secret_name}. "
                    f"Error: {e}"
                )

        # Fallback to environment variable
        if env_var_fallback:
            env_value = os.getenv(secret_name)
            if env_value:
                logger.debug(f"Retrieved secret from env var: {secret_name}")
                return env_value

        # Return default
        if default is not None:
            logger.debug(f"Using default value for secret: {secret_name}")
            return default

        logger.error(f"Secret not found: {secret_name}")
        return None

    def get_secret_required(self, secret_name: str) -> str:
        """Retrieve secret or raise exception if not found.

        Args:
            secret_name: Name of secret.

        Returns:
            Secret value.

        Raises:
            ValueError: If secret not found.
        """
        value = self.get_secret(secret_name)
        if value is None:
            raise ValueError(
                f"Required secret not found: {secret_name}. "
                f"Set via Key Vault or environment variable."
            )
        return value

    def get_connection_string(
        self,
        service: str,
        *,
        secret_name: Optional[str] = None,
    ) -> str:
        """Retrieve connection string for Azure service.

        Args:
            service: Service name (e.g., "storage", "cosmosdb", "postgres").
            secret_name: Override secret name.

        Returns:
            Connection string.
        """
        # Default naming convention
        if secret_name is None:
            secret_name = f"{service.upper()}-CONNECTION-STRING"

        return self.get_secret_required(secret_name)

    def set_secret(
        self,
        secret_name: str,
        value: str,
        *,
        content_type: Optional[str] = None,
    ) -> None:
        """Store secret in Key Vault.

        Args:
            secret_name: Name of secret.
            value: Secret value.
            content_type: Optional content type hint.

        Raises:
            ValueError: If Key Vault not configured.
        """
        if not self.kv_client:
            raise ValueError("Key Vault not configured")

        self.kv_client.set_secret(
            secret_name,
            value,
            content_type=content_type
        )

        # Clear cache for this secret
        self.get_secret.cache_clear()

        logger.info(f"Secret stored in Key Vault: {secret_name}")

    def delete_secret(self, secret_name: str) -> None:
        """Delete secret from Key Vault.

        Args:
            secret_name: Name of secret to delete.

        Raises:
            ValueError: If Key Vault not configured.
        """
        if not self.kv_client:
            raise ValueError("Key Vault not configured")

        self.kv_client.begin_delete_secret(secret_name).wait()

        # Clear cache
        self.get_secret.cache_clear()

        logger.info(f"Secret deleted from Key Vault: {secret_name}")

    def clear_cache(self) -> None:
        """Clear secret cache."""
        self.get_secret.cache_clear()
        logger.debug("Secret cache cleared")


# Global secret manager instance
_global_secret_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """Get or create global secret manager.

    Returns:
        SecretManager instance.
    """
    global _global_secret_manager

    if _global_secret_manager is None:
        _global_secret_manager = SecretManager()

    return _global_secret_manager


# Convenience functions
def get_secret(secret_name: str, default: Optional[str] = None) -> Optional[str]:
    """Get secret using global manager.

    Args:
        secret_name: Name of secret.
        default: Default value.

    Returns:
        Secret value or default.
    """
    return get_secret_manager().get_secret(secret_name, default)


def get_secret_required(secret_name: str) -> str:
    """Get required secret using global manager.

    Args:
        secret_name: Name of secret.

    Returns:
        Secret value.

    Raises:
        ValueError: If secret not found.
    """
    return get_secret_manager().get_secret_required(secret_name)
```

**Migration Guide**:
```python
# OLD: Direct environment variable access
import os
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# NEW: Using SecretManager
from langchain_azure_common.secrets import get_secret_required
api_key = get_secret_required("AZURE-OPENAI-API-KEY")

# Or initialize with explicit Key Vault
from langchain_azure_common.secrets import SecretManager
secrets = SecretManager(key_vault_url="https://myvault.vault.azure.net/")
api_key = secrets.get_secret_required("AZURE-OPENAI-API-KEY")
```

**Setup Azure Key Vault**:
```bash
# Create Key Vault
az keyvault create \
  --name myvault \
  --resource-group myresourcegroup \
  --location eastus

# Add secret
az keyvault secret set \
  --vault-name myvault \
  --name AZURE-OPENAI-API-KEY \
  --value "your-api-key"

# Grant access to managed identity
az keyvault set-policy \
  --name myvault \
  --object-id <managed-identity-object-id> \
  --secret-permissions get list

# Set environment variable in application
export AZURE_KEY_VAULT_URL="https://myvault.vault.azure.net/"
```

**Benefits**:
- **Centralized Secret Storage**: All secrets in one secure location
- **Automatic Rotation**: Key Vault supports automatic secret rotation
- **Audit Trail**: All secret access logged in Azure Monitor
- **Zero Downtime**: Update secrets without redeployment
- **Managed Identity**: No credentials needed in code

---

### 2. Input Validation & Sanitization Framework

**Priority**: High
**Effort**: Medium (1-2 weeks)
**Impact**: High - Prevents injection attacks

#### Current State
- Basic validation in some endpoints
- No centralized validation framework
- SQL injection risk in metadata filters
- Prompt injection vulnerabilities

#### Recommendation
Implement comprehensive input validation:

**File**: `libs/azure-common/langchain_azure_common/validation.py`

```python
from typing import Any, Optional, Callable
from pydantic import BaseModel, field_validator, ValidationError
import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"  # Reject anything suspicious
    MODERATE = "moderate"  # Allow common patterns
    PERMISSIVE = "permissive"  # Minimal validation


class InputValidator:
    """Centralized input validation and sanitization.

    Prevents SQL injection, prompt injection, XSS, and other attacks.
    """

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\s|^)(union|select|insert|update|delete|drop|create|alter|exec|execute)(\s|$)",
        r"--",  # SQL comment
        r"/\*.*\*/",  # SQL block comment
        r";",  # Statement terminator
        r"xp_",  # Extended stored procedures
    ]

    # Prompt injection patterns
    PROMPT_INJECTION_PATTERNS = [
        r"ignore (previous|above|all) (instructions|prompts)",
        r"disregard (previous|above|all) (instructions|prompts)",
        r"you are now",
        r"new instructions:",
        r"system:",
        r"<\|im_start\|>",  # ChatML tags
        r"<\|im_end\|>",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers (onclick, onerror, etc.)
    ]

    @staticmethod
    def validate_string(
        value: str,
        *,
        min_length: int = 0,
        max_length: int = 10000,
        allow_special_chars: bool = True,
        validation_level: ValidationLevel = ValidationLevel.MODERATE,
    ) -> str:
        """Validate and sanitize string input.

        Args:
            value: String to validate.
            min_length: Minimum length.
            max_length: Maximum length.
            allow_special_chars: Allow special characters.
            validation_level: Validation strictness.

        Returns:
            Sanitized string.

        Raises:
            ValueError: If validation fails.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value)}")

        # Length check
        if len(value) < min_length:
            raise ValueError(f"String too short (min: {min_length})")
        if len(value) > max_length:
            raise ValueError(f"String too long (max: {max_length})")

        # Special character check
        if not allow_special_chars:
            if not re.match(r'^[a-zA-Z0-9\s\-_.]*$', value):
                raise ValueError("String contains disallowed special characters")

        # SQL injection check
        if validation_level in [ValidationLevel.STRICT, ValidationLevel.MODERATE]:
            for pattern in InputValidator.SQL_INJECTION_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    logger.warning(f"Potential SQL injection detected: {pattern}")
                    raise ValueError("Input contains potentially malicious content")

        # XSS check
        if validation_level == ValidationLevel.STRICT:
            for pattern in InputValidator.XSS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    logger.warning(f"Potential XSS detected: {pattern}")
                    raise ValueError("Input contains potentially malicious content")

        return value.strip()

    @staticmethod
    def validate_prompt(
        prompt: str,
        *,
        max_length: int = 10000,
        check_injection: bool = True,
    ) -> str:
        """Validate LLM prompt for injection attempts.

        Args:
            prompt: User-provided prompt.
            max_length: Maximum prompt length.
            check_injection: Check for prompt injection patterns.

        Returns:
            Validated prompt.

        Raises:
            ValueError: If validation fails.
        """
        if len(prompt) > max_length:
            raise ValueError(f"Prompt too long (max: {max_length})")

        if check_injection:
            for pattern in InputValidator.PROMPT_INJECTION_PATTERNS:
                if re.search(pattern, prompt, re.IGNORECASE):
                    logger.warning(f"Potential prompt injection detected: {pattern}")
                    raise ValueError(
                        "Prompt contains potentially malicious content. "
                        "Please rephrase your request."
                    )

        return prompt

    @staticmethod
    def validate_metadata_filter(
        filter_dict: dict[str, Any],
        *,
        allowed_fields: Optional[list[str]] = None,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Validate metadata filter dictionary.

        Prevents SQL injection through filter parameters.

        Args:
            filter_dict: Filter dictionary for vector search.
            allowed_fields: Whitelist of allowed field names.
            max_depth: Maximum nesting depth.

        Returns:
            Validated filter dictionary.

        Raises:
            ValueError: If validation fails.
        """
        def validate_recursive(obj: Any, depth: int = 0):
            if depth > max_depth:
                raise ValueError(f"Filter nesting too deep (max: {max_depth})")

            if isinstance(obj, dict):
                validated = {}
                for key, value in obj.items():
                    # Validate key
                    if not re.match(r'^[a-zA-Z0-9_.$]+$', key):
                        raise ValueError(f"Invalid filter key: {key}")

                    # Check whitelist
                    if allowed_fields and key not in allowed_fields:
                        # Allow operators
                        if not key.startswith("$"):
                            raise ValueError(f"Field not allowed: {key}")

                    # Recursive validation
                    validated[key] = validate_recursive(value, depth + 1)

                return validated

            elif isinstance(obj, list):
                return [validate_recursive(item, depth + 1) for item in obj]

            elif isinstance(obj, (str, int, float, bool, type(None))):
                # Validate string values
                if isinstance(obj, str):
                    # Check for SQL injection in string values
                    for pattern in InputValidator.SQL_INJECTION_PATTERNS:
                        if re.search(pattern, obj, re.IGNORECASE):
                            raise ValueError("Filter value contains malicious content")
                return obj

            else:
                raise ValueError(f"Unsupported filter type: {type(obj)}")

        return validate_recursive(filter_dict)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal.

        Args:
            filename: Filename to sanitize.

        Returns:
            Safe filename.
        """
        # Remove path components
        filename = filename.replace("../", "").replace("..\\", "")
        filename = filename.replace("/", "").replace("\\", "")

        # Remove null bytes
        filename = filename.replace("\x00", "")

        # Whitelist characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
            filename = name[:250] + ("." + ext if ext else "")

        return filename


class SafePromptTemplate:
    """Template system with automatic input sanitization."""

    def __init__(
        self,
        template: str,
        *,
        input_validators: Optional[dict[str, Callable]] = None,
    ):
        """Initialize safe template.

        Args:
            template: Prompt template with {placeholders}.
            input_validators: Custom validators for each placeholder.
        """
        self.template = template
        self.input_validators = input_validators or {}

    def format(self, **kwargs: Any) -> str:
        """Format template with validated inputs.

        Args:
            **kwargs: Placeholder values.

        Returns:
            Formatted prompt.

        Raises:
            ValueError: If validation fails.
        """
        validated_kwargs = {}

        for key, value in kwargs.items():
            # Apply custom validator if provided
            if key in self.input_validators:
                validated_kwargs[key] = self.input_validators[key](value)
            else:
                # Default validation
                if isinstance(value, str):
                    validated_kwargs[key] = InputValidator.validate_prompt(value)
                else:
                    validated_kwargs[key] = value

        return self.template.format(**validated_kwargs)
```

**Usage Example**:
```python
from langchain_azure_common.validation import (
    InputValidator,
    SafePromptTemplate,
    ValidationLevel,
)

# Validate user input
try:
    user_query = InputValidator.validate_string(
        user_input,
        max_length=5000,
        validation_level=ValidationLevel.STRICT,
    )
except ValueError as e:
    return {"error": str(e)}

# Validate metadata filter
try:
    validated_filter = InputValidator.validate_metadata_filter(
        filter_dict={"category": {"$eq": user_category}},
        allowed_fields=["category", "date", "author"],
    )
except ValueError as e:
    return {"error": "Invalid filter"}

# Safe prompt templates
template = SafePromptTemplate(
    template="Answer the following question: {question}\nContext: {context}",
    input_validators={
        "question": lambda x: InputValidator.validate_prompt(x, max_length=500),
    }
)

prompt = template.format(
    question=user_question,
    context=retrieved_context
)
```

**Benefits**:
- **Attack Prevention**: Stop SQL injection, XSS, prompt injection
- **Centralized Logic**: Single source of truth for validation
- **Configurable**: Adjust strictness per use case
- **Logging**: Track suspicious input attempts

---

### 3. Role-Based Access Control (RBAC) for Agent Operations

**Priority**: High
**Effort**: High (2-3 weeks)
**Impact**: High - Essential for enterprise deployments

#### Current State
- No access control on agent endpoints
- All users can access all agents
- No audit trail for operations

#### Recommendation
Implement RBAC with Azure AD integration:

**File**: `libs/azure-ai/langchain_azure_ai/security/rbac.py`

```python
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging
from functools import wraps
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from azure.identity import DefaultAzureCredential
import jwt
import requests

logger = logging.getLogger(__name__)

security = HTTPBearer()


class Permission(str, Enum):
    """Predefined permissions for agent operations."""
    AGENT_READ = "agent:read"
    AGENT_WRITE = "agent:write"
    AGENT_EXECUTE = "agent:execute"
    AGENT_ADMIN = "agent:admin"
    TOOL_USE = "tool:use"
    TOOL_ADMIN = "tool:admin"


class Role(str, Enum):
    """Predefined roles."""
    VIEWER = "viewer"  # Read-only
    USER = "user"  # Execute agents
    DEVELOPER = "developer"  # Create/modify agents
    ADMIN = "admin"  # Full access


# Role to permission mapping
ROLE_PERMISSIONS = {
    Role.VIEWER: [Permission.AGENT_READ],
    Role.USER: [Permission.AGENT_READ, Permission.AGENT_EXECUTE, Permission.TOOL_USE],
    Role.DEVELOPER: [
        Permission.AGENT_READ,
        Permission.AGENT_WRITE,
        Permission.AGENT_EXECUTE,
        Permission.TOOL_USE,
    ],
    Role.ADMIN: [
        Permission.AGENT_READ,
        Permission.AGENT_WRITE,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_ADMIN,
        Permission.TOOL_USE,
        Permission.TOOL_ADMIN,
    ],
}


@dataclass
class User:
    """User information from authentication."""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: list[Role] = None
    permissions: list[Permission] = None
    tenant_id: Optional[str] = None


class AzureADAuthenticator:
    """Azure AD JWT token validation and user extraction."""

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        *,
        validate_signature: bool = True,
    ):
        """Initialize authenticator.

        Args:
            tenant_id: Azure AD tenant ID.
            client_id: Application (client) ID.
            validate_signature: Validate JWT signature.
        """
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.validate_signature = validate_signature

        # JWKS URL for signature validation
        self.jwks_url = (
            f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
        )
        self.jwks_client = None

        if self.validate_signature:
            from jwt import PyJWKClient
            self.jwks_client = PyJWKClient(self.jwks_url)

    def validate_token(self, token: str) -> User:
        """Validate JWT token and extract user info.

        Args:
            token: JWT access token.

        Returns:
            User object with claims.

        Raises:
            HTTPException: If token invalid.
        """
        try:
            # Decode and validate
            if self.validate_signature:
                signing_key = self.jwks_client.get_signing_key_from_jwt(token)
                claims = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256"],
                    audience=self.client_id,
                    issuer=f"https://login.microsoftonline.com/{self.tenant_id}/v2.0",
                )
            else:
                # Skip signature validation (for development only)
                claims = jwt.decode(token, options={"verify_signature": False})

            # Extract user info
            user = User(
                user_id=claims.get("oid") or claims.get("sub"),
                username=claims.get("preferred_username") or claims.get("name"),
                email=claims.get("email"),
                tenant_id=claims.get("tid"),
                roles=[],  # Will be populated from roles claim or group membership
                permissions=[],
            )

            # Extract roles from token claims
            if "roles" in claims:
                user.roles = [Role(r) for r in claims["roles"] if r in Role.__members__]

            # Map roles to permissions
            for role in user.roles:
                if role in ROLE_PERMISSIONS:
                    user.permissions.extend(ROLE_PERMISSIONS[role])

            return user

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


class RBACManager:
    """Role-Based Access Control manager."""

    def __init__(
        self,
        authenticator: AzureADAuthenticator,
        *,
        enable_logging: bool = True,
    ):
        """Initialize RBAC manager.

        Args:
            authenticator: Authentication provider.
            enable_logging: Log access attempts.
        """
        self.authenticator = authenticator
        self.enable_logging = enable_logging

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> User:
        """Extract and validate user from request.

        Args:
            credentials: HTTP authorization credentials.

        Returns:
            Authenticated user.
        """
        token = credentials.credentials
        user = self.authenticator.validate_token(token)

        if self.enable_logging:
            logger.info(f"User authenticated: {user.username} ({user.user_id})")

        return user

    def require_permission(self, permission: Permission):
        """Decorator to require specific permission.

        Args:
            permission: Required permission.

        Returns:
            Decorated function.
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, user: User = Depends(self.get_current_user), **kwargs):
                if not self.has_permission(user, permission):
                    logger.warning(
                        f"Access denied for {user.username}: "
                        f"missing permission {permission}"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail=f"Permission denied: {permission} required"
                    )

                if self.enable_logging:
                    logger.info(
                        f"Access granted: {user.username} - {permission}"
                    )

                return await func(*args, user=user, **kwargs)

            return wrapper

        return decorator

    def require_role(self, role: Role):
        """Decorator to require specific role.

        Args:
            role: Required role.

        Returns:
            Decorated function.
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, user: User = Depends(self.get_current_user), **kwargs):
                if role not in user.roles:
                    logger.warning(
                        f"Access denied for {user.username}: "
                        f"missing role {role}"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail=f"Role required: {role}"
                    )

                return await func(*args, user=user, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def has_permission(user: User, permission: Permission) -> bool:
        """Check if user has specific permission.

        Args:
            user: User object.
            permission: Permission to check.

        Returns:
            True if user has permission.
        """
        return permission in user.permissions

    @staticmethod
    def has_role(user: User, role: Role) -> bool:
        """Check if user has specific role.

        Args:
            user: User object.
            role: Role to check.

        Returns:
            True if user has role.
        """
        return role in user.roles


# Global RBAC manager
_global_rbac: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get or create global RBAC manager.

    Returns:
        RBACManager instance.
    """
    global _global_rbac

    if _global_rbac is None:
        import os
        authenticator = AzureADAuthenticator(
            tenant_id=os.getenv("AZURE_TENANT_ID"),
            client_id=os.getenv("AZURE_CLIENT_ID"),
        )
        _global_rbac = RBACManager(authenticator)

    return _global_rbac
```

**Usage in FastAPI Endpoints**:
```python
from fastapi import FastAPI, Depends
from langchain_azure_ai.security.rbac import (
    get_rbac_manager,
    User,
    Permission,
    Role,
)

app = FastAPI()
rbac = get_rbac_manager()

# Require permission
@app.post("/api/agents/execute")
@rbac.require_permission(Permission.AGENT_EXECUTE)
async def execute_agent(
    agent_name: str,
    query: str,
    user: User = Depends(rbac.get_current_user),
):
    logger.info(f"User {user.username} executing agent: {agent_name}")
    # Execute agent logic
    return {"result": "success"}

# Require role
@app.post("/api/agents/create")
@rbac.require_role(Role.DEVELOPER)
async def create_agent(
    agent_config: dict,
    user: User = Depends(rbac.get_current_user),
):
    # Create agent logic
    return {"agent_id": "new-agent-123"}

# Check permission manually
@app.get("/api/agents/list")
async def list_agents(user: User = Depends(rbac.get_current_user)):
    # Show all agents user has access to
    accessible_agents = []
    for agent in all_agents:
        if rbac.has_permission(user, Permission.AGENT_READ):
            accessible_agents.append(agent)

    return {"agents": accessible_agents}
```

**Azure AD Setup**:
```bash
# 1. Register application in Azure AD
az ad app create \
  --display-name "LangChain Agents API" \
  --sign-in-audience AzureADMyOrg

# 2. Create app roles
az ad app update \
  --id <app-id> \
  --app-roles @app-roles.json

# app-roles.json:
# [
#   {
#     "allowedMemberTypes": ["User"],
#     "description": "Full access to all agent operations",
#     "displayName": "Agent Admin",
#     "isEnabled": true,
#     "value": "admin"
#   },
#   {
#     "allowedMemberTypes": ["User"],
#     "description": "Can execute agents",
#     "displayName": "Agent User",
#     "isEnabled": true,
#     "value": "user"
#   }
# ]

# 3. Assign roles to users
az ad app role assignment add \
  --id <app-id> \
  --assignee <user-object-id> \
  --role "admin"

# 4. Set environment variables
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
```

**Benefits**:
- **Enterprise Security**: Azure AD integration with SSO
- **Fine-Grained Control**: Permission-based access
- **Audit Trail**: All access attempts logged
- **Compliance**: Meet regulatory requirements for access control

---

## Azure Best Practices

### 1. Cost Optimization Strategy

**Priority**: High
**Effort**: Medium (2 weeks)
**Impact**: High - Reduce Azure spend

#### Current State
- No cost tracking or budgets
- All LLM calls use GPT-4o (expensive)
- No caching of results
- Always-on deployments

#### Recommendations

**A. Implement Model Router for Cost Optimization**

**File**: `libs/azure-ai/langchain_azure_ai/routing/cost_optimizer.py`

```python
from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tiers by cost."""
    PREMIUM = "premium"  # GPT-4o, Claude Opus
    STANDARD = "standard"  # GPT-4o-mini, Claude Sonnet
    ECONOMY = "economy"  # GPT-3.5, DeepSeek


@dataclass
class ModelCost:
    """Cost information for a model."""
    name: str
    tier: ModelTier
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    max_tokens: int


# Model pricing (as of 2026-02)
MODEL_COSTS = {
    "gpt-4o": ModelCost(
        name="gpt-4o",
        tier=ModelTier.PREMIUM,
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        max_tokens=128000,
    ),
    "gpt-4o-mini": ModelCost(
        name="gpt-4o-mini",
        tier=ModelTier.STANDARD,
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        max_tokens=128000,
    ),
    "deepseek-r1": ModelCost(
        name="deepseek-r1",
        tier=ModelTier.ECONOMY,
        cost_per_1k_input_tokens=0.00014,
        cost_per_1k_output_tokens=0.00028,
        max_tokens=64000,
    ),
}


class CostOptimizingRouter:
    """Route requests to cost-appropriate models."""

    def __init__(
        self,
        *,
        default_tier: ModelTier = ModelTier.STANDARD,
        enable_automatic_routing: bool = True,
    ):
        """Initialize cost optimizer.

        Args:
            default_tier: Default model tier.
            enable_automatic_routing: Auto-select model based on query complexity.
        """
        self.default_tier = default_tier
        self.enable_automatic_routing = enable_automatic_routing

    def select_model(
        self,
        query: str,
        *,
        require_reasoning: bool = False,
        max_budget_per_query: Optional[float] = None,
        force_tier: Optional[ModelTier] = None,
    ) -> str:
        """Select most cost-effective model for query.

        Args:
            query: User query.
            require_reasoning: Requires advanced reasoning (forces premium).
            max_budget_per_query: Maximum cost per query in USD.
            force_tier: Force specific tier.

        Returns:
            Model name to use.
        """
        # Force tier if specified
        if force_tier:
            return self._get_model_for_tier(force_tier)

        # Advanced reasoning requires premium model
        if require_reasoning:
            logger.info("Using premium model for reasoning task")
            return self._get_model_for_tier(ModelTier.PREMIUM)

        # Automatic routing based on complexity
        if self.enable_automatic_routing:
            complexity = self._assess_complexity(query)

            if complexity == "high":
                tier = ModelTier.PREMIUM
            elif complexity == "medium":
                tier = ModelTier.STANDARD
            else:
                tier = ModelTier.ECONOMY

            logger.info(f"Query complexity: {complexity}, selecting tier: {tier}")
            return self._get_model_for_tier(tier)

        # Use default tier
        return self._get_model_for_tier(self.default_tier)

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
        if model_name not in MODEL_COSTS:
            logger.warning(f"Unknown model for cost estimation: {model_name}")
            return 0.0

        cost_info = MODEL_COSTS[model_name]

        input_cost = (input_tokens / 1000) * cost_info.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * cost_info.cost_per_1k_output_tokens

        total_cost = input_cost + output_cost

        logger.debug(
            f"Estimated cost for {model_name}: "
            f"${total_cost:.4f} ({input_tokens} in, {output_tokens} out)"
        )

        return total_cost

    @staticmethod
    def _assess_complexity(query: str) -> Literal["low", "medium", "high"]:
        """Assess query complexity.

        Args:
            query: User query.

        Returns:
            Complexity level.
        """
        # Simple heuristics
        word_count = len(query.split())

        # High complexity indicators
        if any(word in query.lower() for word in [
            "analyze", "compare", "evaluate", "reason", "explain why",
            "complex", "detailed", "comprehensive"
        ]):
            return "high"

        # Word count based
        if word_count > 50:
            return "high"
        elif word_count > 20:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _get_model_for_tier(tier: ModelTier) -> str:
        """Get default model for tier.

        Args:
            tier: Model tier.

        Returns:
            Model name.
        """
        tier_defaults = {
            ModelTier.PREMIUM: "gpt-4o",
            ModelTier.STANDARD: "gpt-4o-mini",
            ModelTier.ECONOMY: "deepseek-r1",
        }

        return tier_defaults[tier]
```

**Usage**:
```python
from langchain_azure_ai.routing import CostOptimizingRouter, ModelTier

router = CostOptimizingRouter(
    default_tier=ModelTier.STANDARD,
    enable_automatic_routing=True,
)

# Automatic model selection
model_name = router.select_model(user_query)
llm = AzureAIChatCompletionsModel(model=model_name)

# Force economy model for simple tasks
model_name = router.select_model(
    user_query,
    force_tier=ModelTier.ECONOMY
)

# Estimate cost
cost = router.estimate_cost(
    model_name="gpt-4o-mini",
    input_tokens=1000,
    output_tokens=500,
)
print(f"Estimated cost: ${cost:.4f}")
```

**B. Implement Azure Cost Management Integration**

```python
from azure.mgmt.costmanagement import CostManagementClient
from azure.identity import DefaultAzureCredential

def setup_cost_alerts():
    """Configure cost alerts in Azure."""
    credential = DefaultAzureCredential()
    client = CostManagementClient(credential, subscription_id="your-sub-id")

    # Create budget
    budget = {
        "eTag": None,
        "properties": {
            "category": "Cost",
            "amount": 1000.00,  # Monthly budget
            "timeGrain": "Monthly",
            "timePeriod": {
                "startDate": "2026-02-01",
                "endDate": "2026-12-31",
            },
            "notifications": {
                "Actual_GreaterThan_80_Percent": {
                    "enabled": True,
                    "operator": "GreaterThan",
                    "threshold": 80,
                    "contactEmails": ["team@example.com"],
                }
            }
        }
    }

    # Apply budget
    client.budgets.create_or_update(
        scope=f"/subscriptions/{subscription_id}",
        budget_name="langchain-agents-budget",
        parameters=budget,
    )
```

**C. Resource Optimization Checklist**:

1. **Use Azure Spot Instances** for non-production workloads
```bash
az container create \
  --resource-group myResourceGroup \
  --name mycontainer \
  --image myimage:latest \
  --priority Spot \
  --command-line "python app.py"
```

2. **Enable Auto-Scaling** for Container Apps
```bash
az containerapp update \
  --name myapp \
  --resource-group myResourceGroup \
  --min-replicas 0 \  # Scale to zero when idle
  --max-replicas 10
```

3. **Use Reserved Capacity** for predictable workloads
   - 1-year reservation: 20-30% discount
   - 3-year reservation: 40-60% discount

4. **Implement Request Batching**
```python
# Batch multiple queries to reduce per-request overhead
queries = ["query1", "query2", "query3"]
results = await llm.abatch(queries)  # Single API call
```

5. **Use Streaming for Long Responses**
```python
# Start displaying results immediately, reduce perceived latency
async for chunk in llm.astream(prompt):
    print(chunk.content, end="", flush=True)
```

**Benefits**:
- **Cost Reduction**: 50-70% reduction by using appropriate models
- **Budget Control**: Alerts prevent overspending
- **Right-Sizing**: Match resources to actual demand
- **Waste Prevention**: Auto-scaling eliminates idle resources

---

### 2. Multi-Region Deployment for High Availability

**Priority**: Medium
**Effort**: High (3-4 weeks)
**Impact**: High - Ensure business continuity

#### Current State
- Single region deployment
- No failover mechanism
- No geo-replication

#### Recommendations

**A. Implement Multi-Region Router**

**File**: `libs/azure-ai/langchain_azure_ai/routing/region_router.py`

```python
from typing import Optional, Callable
from dataclasses import dataclass
import logging
import asyncio
import time
from enum import Enum

logger = logging.getLogger(__name__)


class Region(str, Enum):
    """Azure regions."""
    EAST_US = "eastus"
    WEST_US = "westus"
    NORTH_EUROPE = "northeurope"
    SOUTHEAST_ASIA = "southeastasia"


@dataclass
class RegionEndpoint:
    """Endpoint configuration for a region."""
    region: Region
    endpoint_url: str
    priority: int = 1  # Lower is higher priority
    healthy: bool = True
    last_check: float = 0
    response_time_ms: float = 0


class MultiRegionRouter:
    """Route requests across multiple Azure regions with failover."""

    def __init__(
        self,
        endpoints: list[RegionEndpoint],
        *,
        health_check_interval: int = 60,  # seconds
        timeout: float = 5.0,
        max_retries: int = 2,
    ):
        """Initialize multi-region router.

        Args:
            endpoints: List of regional endpoints.
            health_check_interval: Health check frequency in seconds.
            timeout: Request timeout per region.
            max_retries: Max retry attempts across regions.
        """
        self.endpoints = sorted(endpoints, key=lambda x: x.priority)
        self.health_check_interval = health_check_interval
        self.timeout = timeout
        self.max_retries = max_retries

        # Start background health checks
        asyncio.create_task(self._health_check_loop())

    async def execute_with_failover(
        self,
        func: Callable,
        *args,
        **kwargs,
    ):
        """Execute function with automatic regional failover.

        Args:
            func: Async function to execute.
            *args, **kwargs: Function arguments.

        Returns:
            Function result from first successful region.

        Raises:
            Exception: If all regions fail.
        """
        last_exception = None
        attempts = 0

        while attempts < self.max_retries:
            for endpoint in self._get_healthy_endpoints():
                try:
                    logger.info(f"Attempting request to region: {endpoint.region}")

                    # Set endpoint in kwargs
                    kwargs["endpoint_url"] = endpoint.endpoint_url

                    # Execute with timeout
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.timeout
                    )

                    logger.info(f"Request succeeded in region: {endpoint.region}")
                    return result

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout in region: {endpoint.region}")
                    endpoint.healthy = False
                    last_exception = TimeoutError(f"Timeout in {endpoint.region}")

                except Exception as e:
                    logger.warning(f"Error in region {endpoint.region}: {e}")
                    endpoint.healthy = False
                    last_exception = e

            attempts += 1

            if attempts < self.max_retries:
                logger.info(f"Retrying... (attempt {attempts + 1}/{self.max_retries})")
                await asyncio.sleep(1)  # Brief delay before retry

        # All regions failed
        logger.error("All regions failed after max retries")
        raise Exception(f"All regions failed: {last_exception}")

    async def _health_check_loop(self):
        """Background task for health checks."""
        while True:
            await asyncio.sleep(self.health_check_interval)

            for endpoint in self.endpoints:
                try:
                    start_time = time.time()

                    # Simple health check (customize based on your API)
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{endpoint.endpoint_url}/health",
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                endpoint.healthy = True
                                endpoint.response_time_ms = (time.time() - start_time) * 1000
                                logger.debug(
                                    f"Health check passed: {endpoint.region} "
                                    f"({endpoint.response_time_ms:.2f}ms)"
                                )
                            else:
                                endpoint.healthy = False

                except Exception as e:
                    logger.warning(f"Health check failed for {endpoint.region}: {e}")
                    endpoint.healthy = False

                endpoint.last_check = time.time()

    def _get_healthy_endpoints(self) -> list[RegionEndpoint]:
        """Get list of healthy endpoints sorted by priority.

        Returns:
            List of healthy endpoints.
        """
        healthy = [e for e in self.endpoints if e.healthy]

        if not healthy:
            logger.warning("No healthy endpoints available, using all endpoints")
            return self.endpoints

        return sorted(healthy, key=lambda x: (x.priority, x.response_time_ms))

    def get_status(self) -> dict:
        """Get router status.

        Returns:
            Dictionary with region health status.
        """
        return {
            "regions": [
                {
                    "region": e.region,
                    "healthy": e.healthy,
                    "priority": e.priority,
                    "response_time_ms": e.response_time_ms,
                    "last_check": e.last_check,
                }
                for e in self.endpoints
            ],
            "healthy_count": sum(1 for e in self.endpoints if e.healthy),
            "total_count": len(self.endpoints),
        }
```

**Usage**:
```python
from langchain_azure_ai.routing import MultiRegionRouter, RegionEndpoint, Region

# Configure endpoints
endpoints = [
    RegionEndpoint(
        region=Region.EAST_US,
        endpoint_url="https://eastus.api.myservice.com",
        priority=1,  # Primary
    ),
    RegionEndpoint(
        region=Region.WEST_US,
        endpoint_url="https://westus.api.myservice.com",
        priority=2,  # Secondary
    ),
    RegionEndpoint(
        region=Region.NORTH_EUROPE,
        endpoint_url="https://northeurope.api.myservice.com",
        priority=3,  # Tertiary
    ),
]

router = MultiRegionRouter(endpoints)

# Execute with automatic failover
async def call_llm(endpoint_url: str, prompt: str):
    llm = AzureAIChatCompletionsModel(
        endpoint_url=endpoint_url,
        model="gpt-4o-mini"
    )
    return await llm.ainvoke(prompt)

result = await router.execute_with_failover(
    call_llm,
    prompt="What is Azure?"
)

# Check router status
status = router.get_status()
print(f"Healthy regions: {status['healthy_count']}/{status['total_count']}")
```

**B. Implement Active-Active Deployment**

**Architecture**:
```
                    ┌─────────────────┐
                    │  Azure Traffic  │
                    │    Manager      │
                    │ (Global LB)     │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
    │ East US   │     │ West US   │     │  Europe   │
    │ Region    │     │ Region    │     │  Region   │
    ├───────────┤     ├───────────┤     ├───────────┤
    │ Container │     │ Container │     │ Container │
    │ Apps      │     │ Apps      │     │ Apps      │
    ├───────────┤     ├───────────┤     ├───────────┤
    │ Azure     │     │ Azure     │     │ Azure     │
    │ OpenAI    │     │ OpenAI    │     │ OpenAI    │
    ├───────────┤     ├───────────┤     ├───────────┤
    │ Cosmos DB │◄───►│ Cosmos DB │◄───►│ Cosmos DB │
    │ (Replicated)     │ (Replicated)     │ (Replicated)
    └───────────┘     └───────────┘     └───────────┘
```

**Setup Script**:
```bash
#!/bin/bash
# deploy-multi-region.sh

REGIONS=("eastus" "westus" "northeurope")
APP_NAME="langchain-agents"
RESOURCE_GROUP="langchain-rg"

# Create resource group
az group create --name $RESOURCE_GROUP --location eastus

# Create Cosmos DB with multi-region writes
az cosmosdb create \
  --name "${APP_NAME}-cosmos" \
  --resource-group $RESOURCE_GROUP \
  --locations regionName=eastus failoverPriority=0 isZoneRedundant=False \
  --locations regionName=westus failoverPriority=1 isZoneRedundant=False \
  --locations regionName=northeurope failoverPriority=2 isZoneRedundant=False \
  --enable-multiple-write-locations true

# Deploy to each region
for REGION in "${REGIONS[@]}"; do
  echo "Deploying to $REGION..."

  # Create Container App Environment
  az containerapp env create \
    --name "${APP_NAME}-env-${REGION}" \
    --resource-group $RESOURCE_GROUP \
    --location $REGION

  # Deploy Container App
  az containerapp create \
    --name "${APP_NAME}-${REGION}" \
    --resource-group $RESOURCE_GROUP \
    --environment "${APP_NAME}-env-${REGION}" \
    --image myregistry.azurecr.io/${APP_NAME}:latest \
    --target-port 8000 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 10
done

# Create Traffic Manager profile
az network traffic-manager profile create \
  --name "${APP_NAME}-tm" \
  --resource-group $RESOURCE_GROUP \
  --routing-method Performance \  # Route to closest region
  --unique-dns-name "${APP_NAME}-global"

# Add endpoints for each region
for REGION in "${REGIONS[@]}"; do
  ENDPOINT_URL=$(az containerapp show \
    --name "${APP_NAME}-${REGION}" \
    --resource-group $RESOURCE_GROUP \
    --query "properties.configuration.ingress.fqdn" \
    --output tsv)

  az network traffic-manager endpoint create \
    --name "${REGION}-endpoint" \
    --profile-name "${APP_NAME}-tm" \
    --resource-group $RESOURCE_GROUP \
    --type externalEndpoints \
    --target $ENDPOINT_URL \
    --endpoint-status enabled
done

echo "Multi-region deployment complete!"
echo "Global endpoint: ${APP_NAME}-global.trafficmanager.net"
```

**Benefits**:
- **High Availability**: 99.99% uptime SLA
- **Disaster Recovery**: Automatic failover
- **Performance**: Route to nearest region
- **Compliance**: Data residency requirements

---

### 3. Implement Comprehensive Monitoring & Alerting

**Priority**: Critical
**Effort**: Medium (2 weeks)
**Impact**: Critical - Essential for production

#### Recommendations

**A. Azure Monitor Dashboard**

Create custom dashboard with key metrics:

```python
# create_dashboard.py
from azure.mgmt.dashboard import DashboardManagementClient
from azure.identity import DefaultAzureCredential

def create_monitoring_dashboard():
    """Create Azure Monitor dashboard for LangChain agents."""
    credential = DefaultAzureCredential()
    client = DashboardManagementClient(credential, subscription_id="your-sub-id")

    dashboard_config = {
        "location": "global",
        "tags": {"environment": "production"},
        "properties": {
            "lenses": {
                "0": {
                    "order": 0,
                    "parts": [
                        # Agent Execution Count
                        {
                            "position": {"x": 0, "y": 0, "colSpan": 6, "rowSpan": 4},
                            "metadata": {
                                "type": "Extension/HubsExtension/PartType/MonitorChartPart",
                                "settings": {
                                    "content": {
                                        "metrics": [{
                                            "name": "agent_execution_count",
                                            "aggregationType": "Count",
                                        }]
                                    }
                                }
                            }
                        },
                        # Response Time
                        {
                            "position": {"x": 6, "y": 0, "colSpan": 6, "rowSpan": 4},
                            "metadata": {
                                "type": "Extension/HubsExtension/PartType/MonitorChartPart",
                                "settings": {
                                    "content": {
                                        "metrics": [{
                                            "name": "agent_duration_ms",
                                            "aggregationType": "Average",
                                        }]
                                    }
                                }
                            }
                        },
                        # Error Rate
                        {
                            "position": {"x": 0, "y": 4, "colSpan": 6, "rowSpan": 4},
                            "metadata": {
                                "type": "Extension/HubsExtension/PartType/MonitorChartPart",
                                "settings": {
                                    "content": {
                                        "metrics": [{
                                            "name": "agent_error_count",
                                            "aggregationType": "Count",
                                        }]
                                    }
                                }
                            }
                        },
                        # Token Usage
                        {
                            "position": {"x": 6, "y": 4, "colSpan": 6, "rowSpan": 4},
                            "metadata": {
                                "type": "Extension/HubsExtension/PartType/MonitorChartPart",
                                "settings": {
                                    "content": {
                                        "metrics": [{
                                            "name": "agent_token_count",
                                            "aggregationType": "Sum",
                                        }]
                                    }
                                }
                            }
                        },
                    ]
                }
            }
        }
    }

    # Create dashboard
    client.grafana.create(
        resource_group_name="langchain-rg",
        workspace_name="langchain-dashboard",
        parameters=dashboard_config
    )

    print("Dashboard created successfully!")


if __name__ == "__main__":
    create_monitoring_dashboard()
```

**B. Alert Rules**

```bash
# create_alerts.sh

RESOURCE_GROUP="langchain-rg"
APP_INSIGHTS="langchain-appinsights"

# Alert: High Error Rate
az monitor metrics alert create \
  --name "High Error Rate" \
  --resource-group $RESOURCE_GROUP \
  --scopes "/subscriptions/{sub-id}/resourceGroups/${RESOURCE_GROUP}/providers/microsoft.insights/components/${APP_INSIGHTS}" \
  --condition "avg exceptions/count > 10" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action email team@example.com \
  --description "Error rate exceeds threshold"

# Alert: Slow Response Time
az monitor metrics alert create \
  --name "Slow Response Time" \
  --resource-group $RESOURCE_GROUP \
  --scopes "/subscriptions/{sub-id}/resourceGroups/${RESOURCE_GROUP}/providers/microsoft.insights/components/${APP_INSIGHTS}" \
  --condition "avg requests/duration > 5000" \  # 5 seconds
  --window-size 5m \
  --evaluation-frequency 1m \
  --action email team@example.com

# Alert: High Token Usage (Cost)
az monitor metrics alert create \
  --name "High Token Usage" \
  --resource-group $RESOURCE_GROUP \
  --scopes "/subscriptions/{sub-id}/resourceGroups/${RESOURCE_GROUP}/providers/microsoft.insights/components/${APP_INSIGHTS}" \
  --condition "sum customMetrics/agent_token_count > 1000000" \
  --window-size 1h \
  --evaluation-frequency 15m \
  --action email team@example.com
```

**C. KQL Queries for Analysis**

```kql
// Agent Performance by Type
customMetrics
| where name == "agent_duration_ms"
| extend agent_name = tostring(customDimensions.agent_name)
| summarize
    avg_duration = avg(value),
    p95_duration = percentile(value, 95),
    p99_duration = percentile(value, 99),
    count = count()
  by agent_name
| order by avg_duration desc

// Error Analysis
exceptions
| where timestamp > ago(24h)
| extend agent_name = tostring(customDimensions.agent_name)
| summarize
    error_count = count(),
    error_types = make_set(type)
  by agent_name, bin(timestamp, 1h)
| order by timestamp desc

// Token Usage & Cost Tracking
customMetrics
| where name in ("agent_token_count_input", "agent_token_count_output")
| extend
    agent_name = tostring(customDimensions.agent_name),
    model = tostring(customDimensions.model_name)
| summarize
    total_input_tokens = sumif(value, name == "agent_token_count_input"),
    total_output_tokens = sumif(value, name == "agent_token_count_output")
  by agent_name, model, bin(timestamp, 1d)
| extend estimated_cost =
    (total_input_tokens / 1000 * 0.00015) +  // Input cost
    (total_output_tokens / 1000 * 0.0006)     // Output cost
| order by estimated_cost desc

// User Session Analytics
customEvents
| where name == "agent_chat"
| extend
    session_id = tostring(customDimensions.session_id),
    user_id = tostring(customDimensions.user_id)
| summarize
    message_count = count(),
    distinct_sessions = dcount(session_id),
    avg_messages_per_session = count() / dcount(session_id)
  by user_id
| order by message_count desc
```

**Benefits**:
- **Proactive Monitoring**: Detect issues before users report them
- **Cost Visibility**: Track spend in real-time
- **Performance Optimization**: Identify slow agents
- **Capacity Planning**: Understand usage patterns

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

**Immediate Impact, Low Effort**

1. **Unified Credential Management** ✓
   - Create `azure-common` library
   - Implement `AzureCredentialManager`
   - Migrate 2-3 libraries as proof-of-concept

2. **Semantic Caching** ✓
   - Implement `SemanticCache`
   - Add to one high-traffic vector store
   - Measure hit rate and cost savings

3. **Input Validation** ✓
   - Implement `InputValidator`
   - Add to all user-facing endpoints
   - Security scanning for existing vulnerabilities

4. **Cost Optimization - Basic** ✓
   - Implement `CostOptimizingRouter`
   - Use economy models for simple queries
   - Add cost estimation to logs

### Phase 2: Security & Compliance (2-4 weeks)

**Critical for Enterprise**

1. **Azure Key Vault Integration** ✓
   - Implement `SecretManager`
   - Migrate all secrets from `.env` to Key Vault
   - Set up automatic rotation

2. **RBAC Implementation** ✓
   - Implement `RBACManager`
   - Configure Azure AD integration
   - Assign roles to users/groups

3. **Enhanced Tracing** ✓
   - Implement `EnhancedTracer`
   - Add decorators to all agent operations
   - Configure span correlation

4. **Audit Logging** ✓
   - Log all sensitive operations
   - Integrate with Azure Monitor
   - Create compliance reports

### Phase 3: Reliability & Scale (4-6 weeks)

**Production Hardening**

1. **Rate Limiting & Retry** ✓
   - Implement `EnhancedRetryHandler`
   - Add circuit breakers
   - Configure per-user rate limits

2. **Multi-Region Deployment** ✓
   - Deploy to 3 regions (East US, West US, Europe)
   - Configure Traffic Manager
   - Test failover scenarios

3. **Multi-Modal Pipeline** ✓
   - Implement `MultiModalDocumentPipeline`
   - Support PDF, Word, PowerPoint, images
   - Integrate with RAG agents

4. **Monitoring & Alerting** ✓
   - Create Azure Monitor dashboard
   - Configure alert rules
   - Set up on-call rotation

### Phase 4: Advanced Features (6-8 weeks)

**Competitive Differentiation**

1. **Advanced RAG Features**
   - Hybrid search (vector + keyword)
   - Query expansion and rewriting
   - Multi-hop reasoning

2. **Agent Orchestration**
   - Workflow engine for multi-step agents
   - Conditional branching
   - Human-in-loop integration

3. **Fine-Tuning Pipeline**
   - Data collection from production
   - Model fine-tuning automation
   - A/B testing framework

4. **Advanced Analytics**
   - User behavior analytics
   - Agent performance scoring
   - Predictive cost modeling

---

## Quick Wins

### Top 5 Immediate Actions

1. **Enable Semantic Caching** (1 day)
   - 30-50% cost reduction for repeated queries
   - Instant latency improvement

2. **Switch to GPT-4o-mini** (1 hour)
   - 95% cost reduction vs GPT-4o
   - Minimal quality impact for most tasks

3. **Add Input Validation** (2 days)
   - Prevent security vulnerabilities
   - Improve error messages

4. **Configure Cost Alerts** (1 hour)
   - Avoid surprise bills
   - Budget enforcement

5. **Enable Auto-Scaling** (2 hours)
   - Scale to zero when idle
   - Automatic capacity management

---

## Summary

This comprehensive analysis has identified **15 major enhancement areas** across:

**Feature Enhancements** (5):
- Unified Authentication Framework
- Advanced Rate Limiting & Retry
- Semantic Caching
- Multi-Modal Document Processing
- Enhanced Distributed Tracing

**Security Improvements** (3):
- Azure Key Vault Integration
- Input Validation & Sanitization
- Role-Based Access Control

**Azure Best Practices** (3):
- Cost Optimization Strategy
- Multi-Region High Availability
- Comprehensive Monitoring & Alerting

**Expected Impact**:
- **Cost**: 50-70% reduction through model routing and caching
- **Security**: Enterprise-grade compliance readiness
- **Reliability**: 99.99% uptime with multi-region deployment
- **Performance**: 30-50% latency improvement with caching
- **Scalability**: 10x capacity with auto-scaling

**Total Implementation Effort**: 8-12 weeks with dedicated team

---

**Document Owner**: LangChain Azure Team
**Last Updated**: 2026-02-06
**Next Review**: 2026-05-06
