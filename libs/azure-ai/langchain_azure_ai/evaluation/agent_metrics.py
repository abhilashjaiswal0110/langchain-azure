"""Agent performance metrics and tracking.

Provides comprehensive metrics for agent performance monitoring:
- Response time tracking
- Token usage monitoring
- Success rate calculation
- User satisfaction scores
- Cost estimation

Usage:
    from langchain_azure_ai.evaluation import AgentPerformanceTracker

    tracker = AgentPerformanceTracker("it-helpdesk")
    tracker.record_execution(
        duration_ms=1500,
        tokens=500,
        success=True,
        user_rating=5,
    )

    metrics = tracker.get_metrics()
    print(f"Average response time: {metrics.avg_response_time_ms}ms")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Comprehensive agent performance metrics.

    Attributes:
        agent_name: Name of the agent
        total_requests: Total number of requests processed
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        avg_response_time_ms: Average response time in milliseconds
        p50_response_time_ms: 50th percentile response time
        p95_response_time_ms: 95th percentile response time
        p99_response_time_ms: 99th percentile response time
        total_tokens: Total tokens used
        avg_tokens_per_request: Average tokens per request
        avg_prompt_tokens: Average prompt tokens per request
        avg_completion_tokens: Average completion tokens per request
        success_rate: Success rate (0.0 to 1.0)
        error_rate: Error rate (0.0 to 1.0)
        avg_user_rating: Average user satisfaction rating
        estimated_cost_usd: Estimated cost in USD
        time_period: Time period for metrics
    """

    agent_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    total_tokens: int = 0
    avg_tokens_per_request: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    avg_user_rating: Optional[float] = None
    estimated_cost_usd: float = 0.0
    time_period: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionRecord:
    """Record of a single agent execution.

    Attributes:
        timestamp: Execution timestamp
        duration_ms: Execution duration in milliseconds
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        success: Whether execution was successful
        error: Error message if failed
        user_rating: Optional user satisfaction rating (1-5)
        session_id: Optional session identifier
        metadata: Additional metadata
    """

    timestamp: datetime
    duration_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    user_rating: Optional[int] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentPerformanceTracker:
    """Tracks agent performance metrics over time.

    Maintains a sliding window of execution records and calculates
    aggregate metrics for monitoring agent performance.
    """

    def __init__(
        self,
        agent_name: str,
        window_hours: int = 24,
        max_records: int = 10000,
    ) -> None:
        """Initialize performance tracker.

        Args:
            agent_name: Name of the agent to track
            window_hours: Sliding window in hours for metrics
            max_records: Maximum number of records to keep
        """
        self.agent_name = agent_name
        self.window_hours = window_hours
        self.max_records = max_records
        self._records: List[ExecutionRecord] = []

    def record_execution(
        self,
        duration_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        user_rating: Optional[int] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an agent execution.

        Args:
            duration_ms: Execution duration in milliseconds
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens generated
            success: Whether execution was successful
            error: Error message if failed
            user_rating: User satisfaction rating (1-5)
            session_id: Session identifier
            metadata: Additional metadata
        """
        record = ExecutionRecord(
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
            error=error,
            user_rating=user_rating,
            session_id=session_id,
            metadata=metadata or {},
        )

        self._records.append(record)

        # Trim old records if exceeding max
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

        # Remove records outside window
        self._trim_old_records()

    def _trim_old_records(self) -> None:
        """Remove records outside the sliding window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.window_hours)
        self._records = [r for r in self._records if r.timestamp >= cutoff]

    def get_metrics(self) -> AgentMetrics:
        """Calculate current agent metrics.

        Returns:
            AgentMetrics with aggregate statistics
        """
        self._trim_old_records()

        if not self._records:
            return AgentMetrics(
                agent_name=self.agent_name,
                time_period=f"Last {self.window_hours} hours",
            )

        # Calculate metrics
        successful = [r for r in self._records if r.success]
        failed = [r for r in self._records if not r.success]

        durations = [r.duration_ms for r in self._records]
        durations_sorted = sorted(durations)

        total_prompt_tokens = sum(r.prompt_tokens for r in self._records)
        total_completion_tokens = sum(r.completion_tokens for r in self._records)
        total_tokens = total_prompt_tokens + total_completion_tokens

        # Calculate percentiles
        n = len(durations_sorted)
        p50_idx = int(n * 0.5)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        # User ratings
        ratings = [r.user_rating for r in self._records if r.user_rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        # Estimate cost (using GPT-4 pricing as example)
        # Prompt: $0.03/1K tokens, Completion: $0.06/1K tokens
        estimated_cost = (total_prompt_tokens / 1000 * 0.03) + (
            total_completion_tokens / 1000 * 0.06
        )

        return AgentMetrics(
            agent_name=self.agent_name,
            total_requests=len(self._records),
            successful_requests=len(successful),
            failed_requests=len(failed),
            avg_response_time_ms=sum(durations) / len(durations),
            p50_response_time_ms=durations_sorted[p50_idx] if n > 0 else 0.0,
            p95_response_time_ms=durations_sorted[p95_idx] if n > 0 else 0.0,
            p99_response_time_ms=durations_sorted[p99_idx] if n > 0 else 0.0,
            total_tokens=total_tokens,
            avg_tokens_per_request=total_tokens / len(self._records),
            avg_prompt_tokens=total_prompt_tokens / len(self._records),
            avg_completion_tokens=total_completion_tokens / len(self._records),
            success_rate=len(successful) / len(self._records),
            error_rate=len(failed) / len(self._records),
            avg_user_rating=avg_rating,
            estimated_cost_usd=estimated_cost,
            time_period=f"Last {self.window_hours} hours",
        )

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors by type.

        Returns:
            Dictionary mapping error messages to counts
        """
        self._trim_old_records()

        error_counts: Dict[str, int] = {}
        for record in self._records:
            if record.error:
                error_counts[record.error] = error_counts.get(record.error, 0) + 1

        return error_counts

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics grouped by session.

        Returns:
            Dictionary with session-level statistics
        """
        self._trim_old_records()

        sessions: Dict[str, List[ExecutionRecord]] = {}
        for record in self._records:
            if record.session_id:
                if record.session_id not in sessions:
                    sessions[record.session_id] = []
                sessions[record.session_id].append(record)

        return {
            "total_sessions": len(sessions),
            "avg_requests_per_session": (
                sum(len(records) for records in sessions.values()) / len(sessions)
                if sessions
                else 0
            ),
            "sessions_with_errors": sum(
                1 for records in sessions.values() if any(not r.success for r in records)
            ),
        }


# Global tracker registry
_trackers: Dict[str, AgentPerformanceTracker] = {}


def get_agent_metrics(
    agent_name: str,
    window_hours: int = 24,
) -> AgentMetrics:
    """Get metrics for an agent.

    Args:
        agent_name: Name of the agent
        window_hours: Time window in hours

    Returns:
        AgentMetrics for the specified agent
    """
    if agent_name not in _trackers:
        _trackers[agent_name] = AgentPerformanceTracker(agent_name, window_hours)

    return _trackers[agent_name].get_metrics()


def calculate_agent_benchmarks(
    metrics_list: List[AgentMetrics],
) -> Dict[str, Any]:
    """Calculate benchmark statistics across multiple agents.

    Args:
        metrics_list: List of AgentMetrics to analyze

    Returns:
        Dictionary with benchmark statistics
    """
    if not metrics_list:
        return {}

    return {
        "avg_response_time_ms": sum(m.avg_response_time_ms for m in metrics_list)
        / len(metrics_list),
        "avg_success_rate": sum(m.success_rate for m in metrics_list) / len(metrics_list),
        "avg_tokens_per_request": sum(m.avg_tokens_per_request for m in metrics_list)
        / len(metrics_list),
        "total_requests": sum(m.total_requests for m in metrics_list),
        "total_cost_usd": sum(m.estimated_cost_usd for m in metrics_list),
        "agents_analyzed": len(metrics_list),
    }


def record_agent_execution(
    agent_name: str,
    duration_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    success: bool = True,
    error: Optional[str] = None,
    user_rating: Optional[int] = None,
    session_id: Optional[str] = None,
) -> None:
    """Convenience function to record agent execution.

    Args:
        agent_name: Name of the agent
        duration_ms: Execution duration in milliseconds
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        success: Whether execution was successful
        error: Error message if failed
        user_rating: User satisfaction rating (1-5)
        session_id: Session identifier
    """
    if agent_name not in _trackers:
        _trackers[agent_name] = AgentPerformanceTracker(agent_name)

    _trackers[agent_name].record_execution(
        duration_ms=duration_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        success=success,
        error=error,
        user_rating=user_rating,
        session_id=session_id,
    )
