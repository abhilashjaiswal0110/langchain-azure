"""Cached vector store wrapper with semantic caching.

Provides transparent caching for vector store operations to reduce
costs and improve latency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_azure_ai.caching.semantic_cache import (
    CacheConfig,
    CacheStrategy,
    SemanticCache,
)

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreCacheConfig(CacheConfig):
    """Configuration for vector store caching.

    Extends CacheConfig with vector store specific options.

    Attributes:
        cache_similarity_search: Cache similarity_search results.
        cache_mmr_search: Cache max_marginal_relevance_search results.
        cache_by_k: Include k parameter in cache key.
        cache_by_filter: Include filter in cache key.
        bypass_on_empty: Bypass cache when results are empty.
    """

    cache_similarity_search: bool = True
    cache_mmr_search: bool = True
    cache_by_k: bool = True
    cache_by_filter: bool = True
    bypass_on_empty: bool = False


class CachedVectorStore(VectorStore):
    """Vector store wrapper with transparent semantic caching.

    Wraps any VectorStore implementation with semantic caching layer
    to reduce costs and improve latency for repeated or similar queries.

    Example:
        >>> from langchain_azure_ai.caching import (
        ...     CachedVectorStore,
        ...     VectorStoreCacheConfig,
        ... )
        >>> from langchain_azure_ai.vectorstores import AzureSearch
        >>>
        >>> # Create base vector store
        >>> base_store = AzureSearch(...)
        >>>
        >>> # Wrap with caching
        >>> cached_store = CachedVectorStore(
        ...     vectorstore=base_store,
        ...     embeddings=embeddings,
        ...     config=VectorStoreCacheConfig(
        ...         max_size=500,
        ...         similarity_threshold=0.92,
        ...         ttl_seconds=1800,
        ...     ),
        ... )
        >>>
        >>> # Use like normal vector store
        >>> results = await cached_store.asimilarity_search("How do I reset password?")
        >>>
        >>> # Similar query hits cache
        >>> results = await cached_store.asimilarity_search("Password reset steps")
        >>>
        >>> # Check cache stats
        >>> print(cached_store.get_cache_stats())
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        embeddings: Embeddings,
        config: Optional[VectorStoreCacheConfig] = None,
    ):
        """Initialize cached vector store.

        Args:
            vectorstore: Underlying vector store to wrap.
            embeddings: Embedding model for cache key generation.
            config: Cache configuration.
        """
        self._vectorstore = vectorstore
        self._embeddings = embeddings
        self._config = config or VectorStoreCacheConfig()

        # Initialize semantic cache
        self._cache: SemanticCache[List[Document]] = SemanticCache(
            embeddings=embeddings,
            config=self._config,
        )

        logger.info(
            f"CachedVectorStore initialized: underlying={type(vectorstore).__name__}"
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Return embedding model."""
        return self._embeddings

    def _generate_cache_key(
        self,
        query: str,
        k: int,
        filter: Optional[dict] = None,
        operation: str = "similarity",
    ) -> str:
        """Generate cache key including operation parameters.

        Args:
            query: Search query.
            k: Number of results requested.
            filter: Optional metadata filter.
            operation: Type of search operation.

        Returns:
            Cache key string.
        """
        parts = [query]

        if self._config.cache_by_k:
            parts.append(f"k={k}")

        if self._config.cache_by_filter and filter:
            # Sort filter for consistent keys
            import json
            filter_str = json.dumps(filter, sort_keys=True)
            parts.append(f"filter={filter_str}")

        parts.append(f"op={operation}")

        return "|".join(parts)

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Async similarity search with caching.

        Args:
            query: Search query.
            k: Number of results to return.
            **kwargs: Additional arguments passed to underlying store.

        Returns:
            List of similar documents.
        """
        if not self._config.cache_similarity_search or not self._config.enabled:
            return await self._vectorstore.asimilarity_search(query, k=k, **kwargs)

        # Generate cache key
        cache_key = self._generate_cache_key(
            query=query,
            k=k,
            filter=kwargs.get("filter"),
            operation="similarity",
        )

        # Check cache
        cached_result = await self._cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Execute search
        results = await self._vectorstore.asimilarity_search(query, k=k, **kwargs)

        # Cache results (unless empty and bypass enabled)
        if results or not self._config.bypass_on_empty:
            await self._cache.set(cache_key, results)

        return results

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Similarity search with caching.

        Args:
            query: Search query.
            k: Number of results to return.
            **kwargs: Additional arguments passed to underlying store.

        Returns:
            List of similar documents.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.asimilarity_search(query, k=k, **kwargs)
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Async similarity search with scores and caching.

        Args:
            query: Search query.
            k: Number of results to return.
            **kwargs: Additional arguments.

        Returns:
            List of (document, score) tuples.
        """
        if not self._config.cache_similarity_search or not self._config.enabled:
            return await self._vectorstore.asimilarity_search_with_score(
                query, k=k, **kwargs
            )

        cache_key = self._generate_cache_key(
            query=query,
            k=k,
            filter=kwargs.get("filter"),
            operation="similarity_with_score",
        )

        cached_result = await self._cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        results = await self._vectorstore.asimilarity_search_with_score(
            query, k=k, **kwargs
        )

        if results or not self._config.bypass_on_empty:
            await self._cache.set(cache_key, results)

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Similarity search with scores and caching."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.asimilarity_search_with_score(query, k=k, **kwargs)
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Async MMR search with caching.

        Args:
            query: Search query.
            k: Number of results to return.
            fetch_k: Number of docs to fetch before reranking.
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance).
            **kwargs: Additional arguments.

        Returns:
            List of diverse, relevant documents.
        """
        if not self._config.cache_mmr_search or not self._config.enabled:
            return await self._vectorstore.amax_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
            )

        cache_key = self._generate_cache_key(
            query=query,
            k=k,
            filter=kwargs.get("filter"),
            operation=f"mmr_fk{fetch_k}_lm{lambda_mult}",
        )

        cached_result = await self._cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        results = await self._vectorstore.amax_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

        if results or not self._config.bypass_on_empty:
            await self._cache.set(cache_key, results)

        return results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """MMR search with caching."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.amax_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
            )
        )

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to underlying store and invalidate cache.

        When adding new texts, we clear the cache to ensure fresh results.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
            **kwargs: Additional arguments.

        Returns:
            List of IDs for added texts.
        """
        # Invalidate cache on write
        self._cache.clear()
        logger.debug("Cache cleared due to add_texts operation")

        return self._vectorstore.add_texts(texts, metadatas=metadatas, **kwargs)

    async def aadd_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async add texts to underlying store."""
        self._cache.clear()
        logger.debug("Cache cleared due to aadd_texts operation")

        if hasattr(self._vectorstore, "aadd_texts"):
            return await self._vectorstore.aadd_texts(
                texts, metadatas=metadatas, **kwargs
            )
        return self.add_texts(texts, metadatas=metadatas, **kwargs)

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to underlying store."""
        self._cache.clear()
        logger.debug("Cache cleared due to add_documents operation")

        return self._vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Async add documents to underlying store."""
        self._cache.clear()
        logger.debug("Cache cleared due to aadd_documents operation")

        if hasattr(self._vectorstore, "aadd_documents"):
            return await self._vectorstore.aadd_documents(documents, **kwargs)
        return self.add_documents(documents, **kwargs)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents from underlying store."""
        self._cache.clear()
        logger.debug("Cache cleared due to delete operation")

        if hasattr(self._vectorstore, "delete"):
            return self._vectorstore.delete(ids=ids, **kwargs)
        return None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache metrics.
        """
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def invalidate_query(self, query: str) -> bool:
        """Invalidate cache entries for a specific query.

        Args:
            query: Query to invalidate.

        Returns:
            True if entries were found and invalidated.
        """
        # Invalidate all variations with this query
        cache_key = self._generate_cache_key(query, k=4, operation="similarity")
        return self._cache.invalidate(cache_key)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        vectorstore_cls: Type[VectorStore],
        metadatas: Optional[List[dict]] = None,
        cache_config: Optional[VectorStoreCacheConfig] = None,
        **kwargs: Any,
    ) -> "CachedVectorStore":
        """Create cached vector store from texts.

        Args:
            texts: Texts to add.
            embedding: Embedding model.
            vectorstore_cls: VectorStore class to instantiate.
            metadatas: Optional metadata for each text.
            cache_config: Cache configuration.
            **kwargs: Arguments passed to vectorstore_cls.

        Returns:
            CachedVectorStore instance.
        """
        base_store = vectorstore_cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            **kwargs,
        )

        return cls(
            vectorstore=base_store,
            embeddings=embedding,
            config=cache_config,
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        vectorstore_cls: Type[VectorStore],
        cache_config: Optional[VectorStoreCacheConfig] = None,
        **kwargs: Any,
    ) -> "CachedVectorStore":
        """Create cached vector store from documents.

        Args:
            documents: Documents to add.
            embedding: Embedding model.
            vectorstore_cls: VectorStore class to instantiate.
            cache_config: Cache configuration.
            **kwargs: Arguments passed to vectorstore_cls.

        Returns:
            CachedVectorStore instance.
        """
        base_store = vectorstore_cls.from_documents(
            documents=documents,
            embedding=embedding,
            **kwargs,
        )

        return cls(
            vectorstore=base_store,
            embeddings=embedding,
            config=cache_config,
        )
