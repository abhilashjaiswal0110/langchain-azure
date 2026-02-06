"""Semantic caching layer for vector stores and LLM operations.

This module provides advanced semantic caching capabilities:
- Similarity-based cache matching (not just exact string matches)
- Multiple eviction strategies (LRU, LFU, TTL)
- Distributed caching with Redis support
- Cache statistics and monitoring

Usage:
    from langchain_azure_ai.caching import (
        SemanticCache,
        CacheConfig,
        CacheStrategy,
    )

    # Initialize cache
    cache = SemanticCache(
        embeddings=embedding_model,
        config=CacheConfig(
            max_size=1000,
            similarity_threshold=0.95,
        ),
    )

    # Use cache
    result = await cache.get(query)
    if result is None:
        result = await expensive_operation()
        await cache.set(query, result)
"""

from langchain_azure_ai.caching.semantic_cache import (
    CacheConfig,
    CacheEntry,
    CacheStats,
    CacheStrategy,
    SemanticCache,
)
from langchain_azure_ai.caching.vectorstore_cache import (
    CachedVectorStore,
    VectorStoreCacheConfig,
)

__all__ = [
    "SemanticCache",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "CacheStrategy",
    "CachedVectorStore",
    "VectorStoreCacheConfig",
]
