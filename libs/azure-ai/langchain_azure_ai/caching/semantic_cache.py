"""Semantic cache implementation using embedding similarity.

Provides intelligent caching that matches semantically similar queries,
not just exact string matches.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, List, Optional, TypeVar, Union

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheConfig:
    """Configuration for semantic cache.

    Attributes:
        enabled: Whether caching is enabled.
        max_size: Maximum number of entries in cache.
        ttl_seconds: Time to live for cache entries (0 = no expiry).
        similarity_threshold: Minimum similarity score for cache hit (0.0-1.0).
        strategy: Cache eviction strategy.
        enable_distributed: Use Redis for distributed caching.
        redis_url: Redis connection URL.
        namespace: Cache namespace for key prefixing.
        warm_up_on_start: Pre-warm cache on initialization.
        persist_to_disk: Persist cache to disk.
        disk_cache_path: Path for disk persistence.
    """

    enabled: bool = True
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour default
    similarity_threshold: float = 0.95
    strategy: CacheStrategy = CacheStrategy.LRU
    enable_distributed: bool = False
    redis_url: Optional[str] = None
    namespace: str = "semantic_cache"
    warm_up_on_start: bool = False
    persist_to_disk: bool = False
    disk_cache_path: str = ".cache/semantic_cache"


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry with metadata.

    Attributes:
        key: Cache key (hash of query).
        query: Original query string.
        embedding: Query embedding vector.
        value: Cached value.
        timestamp: Creation timestamp.
        access_count: Number of times accessed.
        last_accessed: Last access timestamp.
        ttl: Time to live in seconds.
    """

    key: str
    query: str
    embedding: np.ndarray
    value: T
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    ttl: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl <= 0:
            return False
        return time.time() - self.timestamp > self.ttl


@dataclass
class CacheStats:
    """Cache performance statistics.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of evicted entries.
        size: Current cache size.
        max_size: Maximum cache size.
        avg_similarity: Average similarity score for hits.
        avg_latency_ms: Average cache lookup latency.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    avg_similarity: float = 0.0
    avg_latency_ms: float = 0.0
    _similarity_sum: float = field(default=0.0, repr=False)
    _latency_sum: float = field(default=0.0, repr=False)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_hit(self, similarity: float, latency_ms: float) -> None:
        """Record a cache hit."""
        self.hits += 1
        self._similarity_sum += similarity
        self._latency_sum += latency_ms
        self.avg_similarity = self._similarity_sum / self.hits
        self.avg_latency_ms = self._latency_sum / (self.hits + self.misses)

    def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss."""
        self.misses += 1
        self._latency_sum += latency_ms
        self.avg_latency_ms = self._latency_sum / (self.hits + self.misses)

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "avg_similarity": round(self.avg_similarity, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class SemanticCache(Generic[T]):
    """Semantic similarity-based caching for LLM and vector operations.

    Uses embedding similarity to match semantically similar queries,
    not just exact string matches. This enables cache hits even when
    queries are phrased differently but have the same meaning.

    Example:
        >>> from langchain_azure_ai.caching import SemanticCache, CacheConfig
        >>> from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
        >>>
        >>> embeddings = AzureAIEmbeddingsModel(...)
        >>> cache = SemanticCache(
        ...     embeddings=embeddings,
        ...     config=CacheConfig(
        ...         max_size=1000,
        ...         similarity_threshold=0.95,
        ...         ttl_seconds=3600,
        ...     ),
        ... )
        >>>
        >>> # Cache a result
        >>> await cache.set("What is Azure AI?", result)
        >>>
        >>> # Similar query will hit cache
        >>> cached = await cache.get("Tell me about Azure AI")
        >>> assert cached == result  # Semantic match!

    Thread Safety:
        All cache operations are thread-safe through internal locking.

    Attributes:
        config: Cache configuration.
        stats: Cache performance statistics.
    """

    def __init__(
        self,
        embeddings: Any,
        config: Optional[CacheConfig] = None,
    ):
        """Initialize semantic cache.

        Args:
            embeddings: Embedding model for query encoding.
            config: Cache configuration.
        """
        self.embeddings = embeddings
        self.config = config or CacheConfig()
        self.stats = CacheStats(max_size=self.config.max_size)

        # Internal storage
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()

        # Redis client for distributed caching
        self._redis: Optional[Any] = None
        if self.config.enable_distributed and self.config.redis_url:
            self._init_redis()

        logger.info(
            f"SemanticCache initialized: max_size={self.config.max_size}, "
            f"threshold={self.config.similarity_threshold}, "
            f"strategy={self.config.strategy.value}"
        )

    def _init_redis(self) -> None:
        """Initialize Redis client for distributed caching."""
        try:
            import redis

            self._redis = redis.from_url(
                self.config.redis_url,
                decode_responses=False,
            )
            self._redis.ping()
            logger.info(f"Redis connected: {self.config.redis_url}")
        except ImportError:
            logger.warning("redis package not installed, falling back to local cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to local cache")
            self._redis = None

    async def get(self, query: str) -> Optional[T]:
        """Retrieve cached value for semantically similar query.

        Args:
            query: Query string to look up.

        Returns:
            Cached value if found with sufficient similarity, None otherwise.
        """
        if not self.config.enabled:
            return None

        start_time = time.perf_counter()

        try:
            # Generate embedding for query
            query_embedding = await self._get_embedding(query)

            # Search for similar cached queries
            best_match, similarity = self._find_best_match(query_embedding)

            latency_ms = (time.perf_counter() - start_time) * 1000

            if best_match is not None:
                self.stats.record_hit(similarity, latency_ms)
                logger.debug(
                    f"Cache HIT: similarity={similarity:.4f}, "
                    f"query='{query[:50]}...'"
                )

                # Update access metadata
                with self._lock:
                    best_match.access_count += 1
                    best_match.last_accessed = time.time()

                    # Move to end for LRU
                    if self.config.strategy == CacheStrategy.LRU:
                        self._cache.move_to_end(best_match.key)

                return best_match.value

            self.stats.record_miss(latency_ms)
            logger.debug(f"Cache MISS: query='{query[:50]}...'")
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        query: str,
        value: T,
        ttl: Optional[int] = None,
    ) -> None:
        """Add entry to cache.

        Args:
            query: Query string as cache key.
            value: Value to cache.
            ttl: Optional TTL override in seconds.
        """
        if not self.config.enabled:
            return

        try:
            # Generate embedding
            query_embedding = await self._get_embedding(query)

            # Create cache key
            cache_key = self._generate_key(query)

            # Check if cache is full and evict if necessary
            with self._lock:
                if len(self._cache) >= self.config.max_size:
                    self._evict()

                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    query=query,
                    embedding=query_embedding,
                    value=value,
                    ttl=ttl if ttl is not None else self.config.ttl_seconds,
                )

                self._cache[cache_key] = entry
                self.stats.size = len(self._cache)

            logger.debug(f"Cached: key={cache_key[:16]}..., query='{query[:50]}...'")

        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def _find_best_match(
        self,
        query_embedding: np.ndarray,
    ) -> tuple[Optional[CacheEntry[T]], float]:
        """Find most similar cached entry.

        Args:
            query_embedding: Embedding vector of query.

        Returns:
            Tuple of (best matching entry or None, similarity score).
        """
        best_similarity = 0.0
        best_entry: Optional[CacheEntry[T]] = None

        with self._lock:
            # Remove expired entries during search
            expired_keys: List[str] = []

            for key, entry in self._cache.items():
                # Check expiry
                if entry.is_expired():
                    expired_keys.append(key)
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, entry.embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            # Clean up expired entries
            for key in expired_keys:
                del self._cache[key]
                self.stats.evictions += 1
            self.stats.size = len(self._cache)

        # Return if above threshold
        if best_similarity >= self.config.similarity_threshold:
            return best_entry, best_similarity

        return None, 0.0

    def _evict(self) -> None:
        """Evict entry based on configured strategy."""
        if not self._cache:
            return

        with self._lock:
            if self.config.strategy == CacheStrategy.LRU:
                # Evict least recently used (first item in OrderedDict)
                self._cache.popitem(last=False)

            elif self.config.strategy == CacheStrategy.LFU:
                # Evict least frequently used
                min_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].access_count,
                )
                del self._cache[min_key]

            elif self.config.strategy == CacheStrategy.TTL:
                # Evict oldest by timestamp
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].timestamp,
                )
                del self._cache[oldest_key]

            elif self.config.strategy == CacheStrategy.FIFO:
                # Evict first item
                self._cache.popitem(last=False)

            self.stats.evictions += 1
            self.stats.size = len(self._cache)
            logger.debug(f"Evicted cache entry (strategy={self.config.strategy.value})")

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        # Check if embeddings support async
        if hasattr(self.embeddings, "aembed_documents"):
            embeddings = await self.embeddings.aembed_documents([text])
        elif hasattr(self.embeddings, "embed_documents"):
            embeddings = self.embeddings.embed_documents([text])
        else:
            msg = "Embeddings must have embed_documents or aembed_documents method"
            raise ValueError(msg)

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

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _generate_key(text: str) -> str:
        """Generate cache key from text.

        Args:
            text: Text to hash.

        Returns:
            Hash string as cache key.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache metrics.
        """
        self.stats.size = len(self._cache)
        return self.stats.to_dict()

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats = CacheStats(max_size=self.config.max_size)
        logger.info("Cache cleared")

    def invalidate(self, query: str) -> bool:
        """Invalidate specific cache entry.

        Args:
            query: Query string to invalidate.

        Returns:
            True if entry was found and removed.
        """
        cache_key = self._generate_key(query)
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self.stats.size = len(self._cache)
                logger.debug(f"Invalidated cache entry: {cache_key[:16]}...")
                return True
        return False

    def contains(self, query: str) -> bool:
        """Check if query exists in cache (exact match).

        Args:
            query: Query string to check.

        Returns:
            True if exact match exists in cache.
        """
        cache_key = self._generate_key(query)
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if not entry.is_expired():
                    return True
                # Remove expired entry
                del self._cache[cache_key]
                self.stats.size = len(self._cache)
        return False

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)

    def __contains__(self, query: str) -> bool:
        """Check if query exists in cache."""
        return self.contains(query)


class SemanticCacheWrapper:
    """Decorator for adding semantic caching to functions.

    Example:
        >>> @SemanticCacheWrapper(cache)
        ... async def expensive_llm_call(query: str) -> str:
        ...     return await llm.ainvoke(query)
        >>>
        >>> # First call - cache miss, executes function
        >>> result = await expensive_llm_call("What is AI?")
        >>>
        >>> # Similar call - cache hit!
        >>> result = await expensive_llm_call("Tell me about AI")
    """

    def __init__(
        self,
        cache: SemanticCache,
        key_func: Optional[Callable[..., str]] = None,
    ):
        """Initialize wrapper.

        Args:
            cache: SemanticCache instance.
            key_func: Optional function to extract cache key from args.
        """
        self.cache = cache
        self.key_func = key_func

    def __call__(self, func: Callable) -> Callable:
        """Wrap function with caching."""
        import functools

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract cache key
            if self.key_func:
                cache_key = self.key_func(*args, **kwargs)
            elif args:
                cache_key = str(args[0])
            else:
                cache_key = str(kwargs)

            # Check cache
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await self.cache.set(cache_key, result)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # For sync functions, run in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
