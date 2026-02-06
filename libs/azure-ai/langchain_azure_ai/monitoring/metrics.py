"""Comprehensive monitoring and alerting for Azure AI applications.

Provides metrics collection, health checks, dashboards, and alerting
capabilities for production monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status."""

    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class Alert:
    """Represents an alert condition.

    Attributes:
        name: Alert name.
        severity: Alert severity level.
        status: Current alert status.
        message: Alert message.
        timestamp: When alert was triggered.
        labels: Key-value labels for the alert.
        value: Current metric value that triggered alert.
        threshold: Threshold value that was exceeded.
    """

    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class HealthCheck:
    """Represents a health check result.

    Attributes:
        name: Health check name.
        status: Current health status.
        message: Status message.
        latency_ms: Check latency in milliseconds.
        timestamp: When check was performed.
        metadata: Additional check metadata.
    """

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MetricPoint:
    """Single metric data point.

    Attributes:
        name: Metric name.
        value: Metric value.
        timestamp: When metric was recorded.
        labels: Metric labels.
        unit: Metric unit.
    """

    name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class AlertRule:
    """Rule for triggering alerts.

    Attributes:
        name: Rule name.
        metric_name: Metric to monitor.
        condition: Comparison operator (gt, lt, gte, lte, eq, ne).
        threshold: Threshold value.
        severity: Alert severity.
        for_duration: How long condition must be true before alerting.
        labels: Rule labels.
        annotations: Additional rule annotations.
    """

    name: str
    metric_name: str
    condition: str  # gt, lt, gte, lte, eq, ne
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    for_duration: int = 60  # seconds
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics.

    Example:
        >>> collector = MetricsCollector(namespace="langchain")
        >>> collector.increment("requests_total", labels={"agent": "it_helpdesk"})
        >>> collector.histogram("request_duration_ms", 150.5)
        >>> summary = collector.get_summary()
    """

    def __init__(
        self,
        namespace: str = "langchain_azure_ai",
        retention_minutes: int = 60,
    ):
        """Initialize metrics collector.

        Args:
            namespace: Metrics namespace prefix.
            retention_minutes: How long to retain metrics.
        """
        self.namespace = namespace
        self.retention_minutes = retention_minutes
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._timestamps: Dict[str, datetime] = {}

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name.
            value: Increment value.
            labels: Metric labels.
        """
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0.0) + value
        self._timestamps[key] = datetime.now(timezone.utc)

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric.

        Args:
            name: Metric name.
            value: Gauge value.
            labels: Metric labels.
        """
        key = self._make_key(name, labels)
        self._gauges[key] = value
        self._timestamps[key] = datetime.now(timezone.utc)

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation.

        Args:
            name: Metric name.
            value: Observation value.
            labels: Metric labels.
        """
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        self._timestamps[key] = datetime.now(timezone.utc)

        # Limit stored observations
        if len(self._histograms[key]) > 10000:
            self._histograms[key] = self._histograms[key][-5000:]

    def get_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get current counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    def get_gauge(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)

    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get histogram statistics.

        Returns:
            Dictionary with count, sum, avg, min, max, p50, p95, p99.
        """
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "count": count,
            "sum": sum(sorted_values),
            "avg": sum(sorted_values) / count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get full metrics summary.

        Returns:
            Dictionary with all metrics.
        """
        return {
            "namespace": self.namespace,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                key: self.get_histogram_stats(key.split("{")[0])
                for key in self._histograms.keys()
            },
        }

    def _make_key(
        self,
        name: str,
        labels: Optional[Dict[str, str]],
    ) -> str:
        """Create metric key from name and labels."""
        full_name = f"{self.namespace}_{name}"
        if labels:
            sorted_labels = sorted(labels.items())
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted_labels)
            return f"{full_name}{{{label_str}}}"
        return full_name

    def cleanup_old_metrics(self) -> int:
        """Remove metrics older than retention period.

        Returns:
            Number of metrics removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.retention_minutes)
        removed = 0

        old_keys = [
            key for key, ts in self._timestamps.items() if ts < cutoff
        ]

        for key in old_keys:
            self._counters.pop(key, None)
            self._gauges.pop(key, None)
            self._histograms.pop(key, None)
            self._timestamps.pop(key, None)
            removed += 1

        return removed


class HealthChecker:
    """Manages health checks for application components.

    Example:
        >>> checker = HealthChecker()
        >>> checker.register_check("database", db_health_check)
        >>> checker.register_check("llm", llm_health_check)
        >>> results = await checker.run_checks()
        >>> overall = checker.get_overall_health()
    """

    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable] = {}
        self._last_results: Dict[str, HealthCheck] = {}

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], Union[bool, HealthCheck]],
        timeout: float = 5.0,
    ) -> None:
        """Register a health check.

        Args:
            name: Check name.
            check_fn: Function that returns True/False or HealthCheck.
            timeout: Check timeout in seconds.
        """
        self._checks[name] = (check_fn, timeout)
        logger.debug(f"Registered health check: {name}")

    def unregister_check(self, name: str) -> bool:
        """Unregister a health check.

        Args:
            name: Check name.

        Returns:
            True if check was removed.
        """
        if name in self._checks:
            del self._checks[name]
            return True
        return False

    async def run_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks.

        Returns:
            Dictionary of check results.
        """
        results: Dict[str, HealthCheck] = {}

        for name, (check_fn, timeout) in self._checks.items():
            results[name] = await self._run_single_check(name, check_fn, timeout)

        self._last_results = results
        return results

    async def _run_single_check(
        self,
        name: str,
        check_fn: Callable,
        timeout: float,
    ) -> HealthCheck:
        """Run a single health check with timeout.

        Args:
            name: Check name.
            check_fn: Check function.
            timeout: Timeout in seconds.

        Returns:
            HealthCheck result.
        """
        start_time = time.perf_counter()

        try:
            # Run with timeout
            if asyncio.iscoroutinefunction(check_fn):
                result = await asyncio.wait_for(check_fn(), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(check_fn),
                    timeout=timeout,
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Handle different return types
            if isinstance(result, HealthCheck):
                result.latency_ms = latency_ms
                return result
            elif isinstance(result, bool):
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="Check passed" if result else "Check failed",
                    latency_ms=latency_ms,
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    latency_ms=latency_ms,
                )

        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {timeout}s",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency_ms,
            )

    def get_overall_health(self) -> HealthCheck:
        """Get overall application health based on all checks.

        Returns:
            Aggregate HealthCheck.
        """
        if not self._last_results:
            return HealthCheck(
                name="overall",
                status=HealthStatus.HEALTHY,
                message="No checks registered",
            )

        unhealthy = [
            name for name, check in self._last_results.items()
            if check.status == HealthStatus.UNHEALTHY
        ]
        degraded = [
            name for name, check in self._last_results.items()
            if check.status == HealthStatus.DEGRADED
        ]

        if unhealthy:
            return HealthCheck(
                name="overall",
                status=HealthStatus.UNHEALTHY,
                message=f"Unhealthy checks: {', '.join(unhealthy)}",
                metadata={
                    "unhealthy_checks": unhealthy,
                    "degraded_checks": degraded,
                    "total_checks": len(self._last_results),
                },
            )
        elif degraded:
            return HealthCheck(
                name="overall",
                status=HealthStatus.DEGRADED,
                message=f"Degraded checks: {', '.join(degraded)}",
                metadata={
                    "degraded_checks": degraded,
                    "total_checks": len(self._last_results),
                },
            )
        else:
            return HealthCheck(
                name="overall",
                status=HealthStatus.HEALTHY,
                message=f"All {len(self._last_results)} checks passed",
                metadata={"total_checks": len(self._last_results)},
            )

    def get_last_results(self) -> Dict[str, HealthCheck]:
        """Get last health check results."""
        return dict(self._last_results)


class AlertManager:
    """Manages alert rules and notifications.

    Example:
        >>> manager = AlertManager()
        >>> manager.add_rule(AlertRule(
        ...     name="high_error_rate",
        ...     metric_name="errors_total",
        ...     condition="gt",
        ...     threshold=10,
        ...     severity=AlertSeverity.CRITICAL,
        ... ))
        >>> alerts = manager.evaluate(metrics_collector)
    """

    def __init__(
        self,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ):
        """Initialize alert manager.

        Args:
            on_alert: Callback for new alerts.
        """
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._condition_start: Dict[str, datetime] = {}
        self._on_alert = on_alert

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule to add.
        """
        self._rules[rule.name] = rule
        logger.debug(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule.

        Args:
            name: Rule name.

        Returns:
            True if rule was removed.
        """
        if name in self._rules:
            del self._rules[name]
            return True
        return False

    def evaluate(
        self,
        metrics: MetricsCollector,
    ) -> List[Alert]:
        """Evaluate all rules against current metrics.

        Args:
            metrics: Metrics collector with current values.

        Returns:
            List of new alerts.
        """
        new_alerts: List[Alert] = []
        now = datetime.now(timezone.utc)

        for rule_name, rule in self._rules.items():
            # Get current metric value - check if it exists as counter or gauge
            key = metrics._make_key(rule.metric_name, rule.labels)

            if key in metrics._counters:
                value = metrics._counters[key]
            elif key in metrics._gauges:
                value = metrics._gauges[key]
            else:
                # Metric doesn't exist, skip this rule
                continue

            # Check condition
            condition_met = self._check_condition(value, rule.condition, rule.threshold)

            if condition_met:
                # Track when condition started
                if rule_name not in self._condition_start:
                    self._condition_start[rule_name] = now

                # Check if condition has been true for required duration
                duration = (now - self._condition_start[rule_name]).total_seconds()

                if duration >= rule.for_duration and rule_name not in self._active_alerts:
                    # Create alert
                    alert = Alert(
                        name=rule_name,
                        severity=rule.severity,
                        status=AlertStatus.FIRING,
                        message=f"{rule.metric_name} {rule.condition} {rule.threshold}",
                        labels=rule.labels,
                        value=value,
                        threshold=rule.threshold,
                    )

                    self._active_alerts[rule_name] = alert
                    self._alert_history.append(alert)
                    new_alerts.append(alert)

                    # Call callback
                    if self._on_alert:
                        try:
                            self._on_alert(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")

                    logger.warning(f"Alert fired: {rule_name} - {alert.message}")

            else:
                # Condition no longer met
                self._condition_start.pop(rule_name, None)

                # Resolve active alert
                if rule_name in self._active_alerts:
                    alert = self._active_alerts.pop(rule_name)
                    resolved_alert = Alert(
                        name=rule_name,
                        severity=alert.severity,
                        status=AlertStatus.RESOLVED,
                        message=f"Resolved: {alert.message}",
                        labels=alert.labels,
                        value=value,
                        threshold=alert.threshold,
                    )
                    self._alert_history.append(resolved_alert)
                    logger.info(f"Alert resolved: {rule_name}")

        return new_alerts

    def _check_condition(
        self,
        value: float,
        condition: str,
        threshold: float,
    ) -> bool:
        """Check if condition is met.

        Args:
            value: Current value.
            condition: Comparison operator.
            threshold: Threshold value.

        Returns:
            True if condition is met.
        """
        conditions = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
            "ne": lambda v, t: v != t,
        }

        check_fn = conditions.get(condition)
        if check_fn:
            return check_fn(value, threshold)

        logger.warning(f"Unknown condition: {condition}")
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return list(self._active_alerts.values())

    def get_alert_history(
        self,
        limit: int = 100,
    ) -> List[Alert]:
        """Get recent alert history.

        Args:
            limit: Maximum alerts to return.

        Returns:
            List of recent alerts.
        """
        return self._alert_history[-limit:]

    def acknowledge_alert(self, name: str) -> bool:
        """Acknowledge an active alert.

        Args:
            name: Alert name.

        Returns:
            True if alert was acknowledged.
        """
        if name in self._active_alerts:
            self._active_alerts[name].status = AlertStatus.ACKNOWLEDGED
            return True
        return False


class MonitoringDashboard:
    """Central dashboard for monitoring data.

    Combines metrics, health checks, and alerts into a unified view.

    Example:
        >>> dashboard = MonitoringDashboard()
        >>> dashboard.register_health_check("llm", check_llm_health)
        >>> dashboard.add_alert_rule(error_rate_rule)
        >>> status = await dashboard.get_status()
    """

    def __init__(
        self,
        namespace: str = "langchain_azure_ai",
    ):
        """Initialize monitoring dashboard.

        Args:
            namespace: Metrics namespace.
        """
        self.namespace = namespace
        self.metrics = MetricsCollector(namespace=namespace)
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()

        logger.info(f"MonitoringDashboard initialized: {namespace}")

    def record_request(
        self,
        agent_name: str,
        duration_ms: float,
        success: bool = True,
        tokens: int = 0,
    ) -> None:
        """Record an agent request.

        Args:
            agent_name: Name of the agent.
            duration_ms: Request duration.
            success: Whether request succeeded.
            tokens: Tokens used.
        """
        labels = {"agent": agent_name}

        # Record both labeled (for detailed analysis) and unlabeled (for alert evaluation)
        self.metrics.increment("requests_total", labels=labels)
        self.metrics.increment("requests_total")  # Unlabeled aggregate for alerts
        self.metrics.histogram("request_duration_ms", duration_ms, labels=labels)

        if not success:
            self.metrics.increment("errors_total", labels=labels)
            self.metrics.increment("errors_total")  # Unlabeled aggregate for alerts

        if tokens > 0:
            self.metrics.increment("tokens_total", value=tokens, labels=labels)
            self.metrics.increment("tokens_total", value=tokens)  # Unlabeled aggregate

    def record_cache_access(
        self,
        hit: bool,
        latency_ms: float,
    ) -> None:
        """Record a cache access.

        Args:
            hit: Whether it was a cache hit.
            latency_ms: Access latency.
        """
        if hit:
            self.metrics.increment("cache_hits_total")
        else:
            self.metrics.increment("cache_misses_total")

        self.metrics.histogram("cache_latency_ms", latency_ms)

    def register_health_check(
        self,
        name: str,
        check_fn: Callable,
        timeout: float = 5.0,
    ) -> None:
        """Register a health check.

        Args:
            name: Check name.
            check_fn: Health check function.
            timeout: Check timeout.
        """
        self.health_checker.register_check(name, check_fn, timeout)

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule.
        """
        self.alert_manager.add_rule(rule)

    async def get_status(self) -> Dict[str, Any]:
        """Get full monitoring status.

        Returns:
            Dictionary with health, metrics, and alerts.
        """
        # Run health checks
        health_results = await self.health_checker.run_checks()
        overall_health = self.health_checker.get_overall_health()

        # Evaluate alerts
        new_alerts = self.alert_manager.evaluate(self.metrics)

        # Get metrics summary
        metrics_summary = self.metrics.get_summary()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": {
                "overall": overall_health.to_dict(),
                "checks": {
                    name: check.to_dict()
                    for name, check in health_results.items()
                },
            },
            "metrics": metrics_summary,
            "alerts": {
                "active": [a.to_dict() for a in self.alert_manager.get_active_alerts()],
                "new": [a.to_dict() for a in new_alerts],
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get quick summary of monitoring status.

        Returns:
            Brief status summary.
        """
        overall = self.health_checker.get_overall_health()
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            "health_status": overall.status.value,
            "active_alerts": len(active_alerts),
            "critical_alerts": sum(
                1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL
            ),
            "total_requests": self.metrics.get_counter("requests_total"),
            "total_errors": self.metrics.get_counter("errors_total"),
        }


# Health check factory functions
def create_llm_health_check(llm: Any) -> Callable:
    """Create health check for LLM.

    Args:
        llm: LLM instance.

    Returns:
        Health check function.
    """

    async def check() -> HealthCheck:
        try:
            start = time.perf_counter()
            # Simple ping with minimal prompt
            await llm.ainvoke("Hi")
            latency = (time.perf_counter() - start) * 1000

            return HealthCheck(
                name="llm",
                status=HealthStatus.HEALTHY if latency < 5000 else HealthStatus.DEGRADED,
                message=f"LLM responding in {latency:.0f}ms",
                latency_ms=latency,
            )
        except Exception as e:
            return HealthCheck(
                name="llm",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    return check


def create_vectorstore_health_check(vectorstore: Any) -> Callable:
    """Create health check for vector store.

    Args:
        vectorstore: Vector store instance.

    Returns:
        Health check function.
    """

    async def check() -> HealthCheck:
        try:
            start = time.perf_counter()
            # Simple query
            await vectorstore.asimilarity_search("test", k=1)
            latency = (time.perf_counter() - start) * 1000

            return HealthCheck(
                name="vectorstore",
                status=HealthStatus.HEALTHY,
                message=f"Vector store responding in {latency:.0f}ms",
                latency_ms=latency,
            )
        except Exception as e:
            return HealthCheck(
                name="vectorstore",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    return check


# Default alert rules
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="high_error_rate",
        metric_name="errors_total",
        condition="gt",
        threshold=10,
        severity=AlertSeverity.CRITICAL,
        for_duration=60,
        annotations={"description": "Error rate exceeds threshold"},
    ),
    AlertRule(
        name="slow_responses",
        metric_name="request_duration_ms",
        condition="gt",
        threshold=5000,
        severity=AlertSeverity.WARNING,
        for_duration=120,
        annotations={"description": "Response time exceeds 5 seconds"},
    ),
    AlertRule(
        name="low_cache_hit_rate",
        metric_name="cache_hits_total",
        condition="lt",
        threshold=10,
        severity=AlertSeverity.INFO,
        for_duration=300,
        annotations={"description": "Cache hit rate is too low"},
    ),
]
