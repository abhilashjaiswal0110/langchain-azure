"""Comprehensive monitoring and alerting for Azure AI applications.

This module provides production-ready monitoring capabilities:
- Metrics collection and aggregation
- Health checks for all components
- Alert rules and notifications
- Unified monitoring dashboard

Usage:
    from langchain_azure_ai.monitoring import (
        MonitoringDashboard,
        AlertRule,
        AlertSeverity,
        HealthChecker,
        MetricsCollector,
    )

    # Initialize dashboard
    dashboard = MonitoringDashboard(namespace="my-app")

    # Register health checks
    dashboard.register_health_check("llm", check_llm_health)
    dashboard.register_health_check("vectorstore", check_vs_health)

    # Add alert rules
    dashboard.add_alert_rule(AlertRule(
        name="high_error_rate",
        metric_name="errors_total",
        condition="gt",
        threshold=10,
        severity=AlertSeverity.CRITICAL,
    ))

    # Record metrics
    dashboard.record_request("agent_name", duration_ms=150.5, success=True)

    # Get full status
    status = await dashboard.get_status()
"""

from langchain_azure_ai.monitoring.metrics import (
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    DEFAULT_ALERT_RULES,
    HealthCheck,
    HealthChecker,
    HealthStatus,
    MetricPoint,
    MetricsCollector,
    MonitoringDashboard,
    create_llm_health_check,
    create_vectorstore_health_check,
)

__all__ = [
    # Dashboard
    "MonitoringDashboard",
    # Metrics
    "MetricsCollector",
    "MetricPoint",
    # Health
    "HealthChecker",
    "HealthCheck",
    "HealthStatus",
    # Alerts
    "AlertManager",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "DEFAULT_ALERT_RULES",
    # Factory functions
    "create_llm_health_check",
    "create_vectorstore_health_check",
]
