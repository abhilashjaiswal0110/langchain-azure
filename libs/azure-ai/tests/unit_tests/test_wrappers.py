"""Comprehensive unit tests for the Azure AI Foundry wrapper system.

This module tests:
- Base wrapper class functionality
- IT agent wrappers
- Enterprise agent wrappers
- DeepAgent wrappers
- Server endpoints with mocked agents
- Observability components
"""

import asyncio
import os
import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Test fixtures and mocks
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-12-01-preview")
    monkeypatch.setenv("USE_AZURE_FOUNDRY", "false")
    monkeypatch.setenv("ENABLE_TRACING", "false")


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    from langchain_core.messages import AIMessage
    
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Test response")
    return mock


@pytest.fixture
def mock_agent():
    """Create a mock compiled agent graph."""
    from langchain_core.messages import AIMessage, HumanMessage
    
    mock = MagicMock()
    mock.invoke.return_value = {
        "messages": [
            HumanMessage(content="Test input"),
            AIMessage(content="Test response from agent"),
        ]
    }
    mock.ainvoke = AsyncMock(return_value={
        "messages": [
            HumanMessage(content="Test input"),
            AIMessage(content="Async test response"),
        ]
    })
    return mock


class TestWrapperConfig:
    """Tests for WrapperConfig dataclass."""

    def test_from_env_defaults(self, mock_env_vars):
        """Test loading config from environment with defaults."""
        from langchain_azure_ai.wrappers.base import WrapperConfig
        
        config = WrapperConfig.from_env()
        
        assert config.use_azure_foundry is False
        assert config.enable_tracing is False
        assert config.langsmith_enabled is True

    def test_from_env_azure_foundry_enabled(self, mock_env_vars, monkeypatch):
        """Test loading config with Azure Foundry enabled."""
        from langchain_azure_ai.wrappers.base import WrapperConfig
        
        monkeypatch.setenv("USE_AZURE_FOUNDRY", "true")
        monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://test.ai.azure.com")
        
        config = WrapperConfig.from_env()
        
        assert config.use_azure_foundry is True
        assert config.project_endpoint == "https://test.ai.azure.com"

    def test_validate_missing_endpoint(self, mock_env_vars, monkeypatch):
        """Test validation catches missing endpoint when Foundry enabled."""
        from langchain_azure_ai.wrappers.base import WrapperConfig
        
        monkeypatch.setenv("USE_AZURE_FOUNDRY", "true")
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        
        config = WrapperConfig.from_env()
        issues = config.validate()
        
        assert len(issues) == 1
        assert "AZURE_AI_PROJECT_ENDPOINT not set" in issues[0]


class TestFoundryAgentWrapper:
    """Tests for the base FoundryAgentWrapper class."""

    def test_init_with_existing_agent(self, mock_env_vars, mock_agent):
        """Test initializing wrapper with an existing agent."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper, AgentType
        
        # Create a concrete subclass for testing
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-agent"
        assert wrapper._agent == mock_agent
        assert wrapper.agent_type == AgentType.CUSTOM

    def test_invoke_returns_response(self, mock_env_vars, mock_agent):
        """Test that invoke returns agent response."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        from langchain_core.messages import HumanMessage
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        result = wrapper.invoke({"messages": [HumanMessage(content="Hello")]})
        
        assert "messages" in result
        assert len(result["messages"]) == 2

    def test_chat_returns_string(self, mock_env_vars, mock_agent):
        """Test that chat returns a string response."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        response = wrapper.chat("Hello, agent!")
        
        assert isinstance(response, str)
        assert response == "Test response from agent"

    @pytest.mark.asyncio
    async def test_ainvoke_async(self, mock_env_vars, mock_agent):
        """Test async invocation."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        from langchain_core.messages import HumanMessage
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        result = await wrapper.ainvoke({"messages": [HumanMessage(content="Hello")]})
        
        assert "messages" in result

    def test_is_foundry_enabled_false_by_default(self, mock_env_vars, mock_agent):
        """Test that Foundry is disabled by default."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        wrapper = TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        )
        
        assert wrapper.is_foundry_enabled is False

    def test_context_manager(self, mock_env_vars, mock_agent):
        """Test using wrapper as context manager."""
        from langchain_azure_ai.wrappers.base import FoundryAgentWrapper
        
        class TestWrapper(FoundryAgentWrapper):
            def _create_agent_impl(self, llm, tools):
                return mock_agent
        
        with TestWrapper(
            name="test-agent",
            instructions="Test instructions",
            existing_agent=mock_agent,
        ) as wrapper:
            assert wrapper.name == "test-agent"


class TestITAgentWrappers:
    """Tests for IT agent wrappers."""

    def test_helpdesk_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating IT Helpdesk wrapper."""
        from langchain_azure_ai.wrappers import ITHelpdeskWrapper
        
        wrapper = ITHelpdeskWrapper(
            name="test-helpdesk",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-helpdesk"
        assert wrapper.agent_subtype == "helpdesk"

    def test_servicenow_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating ServiceNow wrapper."""
        from langchain_azure_ai.wrappers import ServiceNowWrapper
        
        wrapper = ServiceNowWrapper(
            name="test-servicenow",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-servicenow"
        assert wrapper.agent_subtype == "servicenow"

    def test_hitl_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating HITL Support wrapper."""
        from langchain_azure_ai.wrappers import HITLSupportWrapper
        
        wrapper = HITLSupportWrapper(
            name="test-hitl",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-hitl"
        assert wrapper.agent_subtype == "hitl"


class TestEnterpriseAgentWrappers:
    """Tests for Enterprise agent wrappers."""

    def test_research_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Research Agent wrapper."""
        from langchain_azure_ai.wrappers import ResearchAgentWrapper
        
        wrapper = ResearchAgentWrapper(
            name="test-research",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-research"
        assert wrapper.agent_subtype == "research"

    def test_content_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Content Agent wrapper."""
        from langchain_azure_ai.wrappers import ContentAgentWrapper
        
        wrapper = ContentAgentWrapper(
            name="test-content",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-content"
        assert wrapper.agent_subtype == "content"

    def test_data_analyst_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Data Analyst wrapper."""
        from langchain_azure_ai.wrappers import DataAnalystWrapper
        
        wrapper = DataAnalystWrapper(
            name="test-data-analyst",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-data-analyst"
        assert wrapper.agent_subtype == "data_analyst"

    def test_code_assistant_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Code Assistant wrapper."""
        from langchain_azure_ai.wrappers import CodeAssistantWrapper
        
        wrapper = CodeAssistantWrapper(
            name="test-code-assistant",
            existing_agent=mock_agent,
        )
        
        assert wrapper.name == "test-code-assistant"
        assert wrapper.agent_subtype == "code_assistant"


class TestDeepAgentWrappers:
    """Tests for DeepAgent wrappers."""

    def test_it_operations_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating IT Operations wrapper."""
        from langchain_azure_ai.wrappers import ITOperationsWrapper

        wrapper = ITOperationsWrapper(
            name="test-it-ops",
            existing_agent=mock_agent,
        )

        assert wrapper.name == "test-it-ops"
        assert wrapper.agent_subtype == "it_operations"

    def test_sales_intelligence_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Sales Intelligence wrapper."""
        from langchain_azure_ai.wrappers import SalesIntelligenceWrapper

        wrapper = SalesIntelligenceWrapper(
            name="test-sales",
            existing_agent=mock_agent,
        )

        assert wrapper.name == "test-sales"
        assert wrapper.agent_subtype == "sales_intelligence"

    def test_recruitment_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Recruitment wrapper."""
        from langchain_azure_ai.wrappers import RecruitmentWrapper

        wrapper = RecruitmentWrapper(
            name="test-recruitment",
            existing_agent=mock_agent,
        )

        assert wrapper.name == "test-recruitment"
        assert wrapper.agent_subtype == "recruitment"


class TestSoftwareDevelopmentWrapper:
    """Tests for SoftwareDevelopmentWrapper (#9)."""

    def test_software_development_wrapper_creation(self, mock_env_vars, mock_agent):
        """Test creating Software Development wrapper."""
        from langchain_azure_ai.wrappers import SoftwareDevelopmentWrapper

        wrapper = SoftwareDevelopmentWrapper(
            name="test-software-dev",
            existing_agent=mock_agent,
        )

        assert wrapper.name == "test-software-dev"
        assert wrapper.agent_subtype == "software_development"

    def test_software_development_wrapper_has_instructions(self, mock_env_vars, mock_agent):
        """Test that Software Development wrapper has proper instructions."""
        from langchain_azure_ai.wrappers import SoftwareDevelopmentWrapper

        wrapper = SoftwareDevelopmentWrapper(
            name="test-software-dev",
            existing_agent=mock_agent,
        )

        assert "SDLC" in wrapper.instructions
        assert "Requirements Phase" in wrapper.instructions

    def test_software_development_wrapper_default_subagents(self, mock_env_vars):
        """Test that default subagents are created with tools."""
        from langchain_azure_ai.wrappers import SoftwareDevelopmentWrapper

        subagents = SoftwareDevelopmentWrapper._get_default_subagents()

        assert len(subagents) == 9

        # Verify each subagent has tools
        subagent_names = [sa.name for sa in subagents]
        assert "requirements-intelligence" in subagent_names
        assert "architecture-design" in subagent_names
        assert "code-generator" in subagent_names
        assert "code-reviewer" in subagent_names
        assert "testing-automation" in subagent_names
        assert "debugging-optimization" in subagent_names
        assert "security-compliance" in subagent_names
        assert "devops-integration" in subagent_names
        assert "documentation" in subagent_names

    def test_software_development_subagents_have_tools(self, mock_env_vars):
        """Test that subagents have tools wired (#14)."""
        from langchain_azure_ai.wrappers import SoftwareDevelopmentWrapper

        subagents = SoftwareDevelopmentWrapper._get_default_subagents()

        # Each subagent should have tools
        for subagent in subagents:
            assert len(subagent.tools) > 0, f"{subagent.name} has no tools"
            assert len(subagent.tools) == 6, f"{subagent.name} should have 6 tools"

    def test_software_development_custom_subagents(self, mock_env_vars, mock_agent):
        """Test that custom subagents can be provided."""
        from langchain_azure_ai.wrappers import SoftwareDevelopmentWrapper, SubAgentConfig

        custom_subagents = [
            SubAgentConfig(
                name="custom-agent",
                instructions="Custom instructions",
                tools=[],
            )
        ]

        wrapper = SoftwareDevelopmentWrapper(
            name="test-software-dev",
            existing_agent=mock_agent,
            sub_agents=custom_subagents,
        )

        assert len(wrapper.sub_agents) == 1
        assert wrapper.sub_agents[0].name == "custom-agent"


class TestSoftwareDevTools:
    """Tests for Software Development Tools (#9, #13)."""

    def test_get_all_software_dev_tools(self):
        """Test that all tools are exported (#13)."""
        from langchain_azure_ai.wrappers.software_dev_tools import get_all_software_dev_tools

        tools = get_all_software_dev_tools()

        # Should have 54 tools (6 per category * 9 categories)
        assert len(tools) == 54

    def test_design_architecture_uses_constraints(self):
        """Test design_architecture uses constraints parameter (#7)."""
        from langchain_azure_ai.wrappers.software_dev_tools import design_architecture
        import json

        result = design_architecture.invoke({
            "requirements": "Build a web application",
            "pattern": "microservices",
            "constraints": "budget constraints, high performance required",
        })

        data = json.loads(result)
        assert "constraints_applied" in data
        assert len(data["constraints_applied"]) > 0
        assert "cost_considerations" in data["design"] or "performance_optimizations" in data["design"]

    def test_create_api_spec_grpc(self):
        """Test create_api_spec generates gRPC specs (#12)."""
        from langchain_azure_ai.wrappers.software_dev_tools import create_api_spec
        import json

        result = create_api_spec.invoke({
            "api_name": "UserService",
            "resources": "User,Order",
            "spec_format": "grpc",
        })

        data = json.loads(result)
        assert "grpc_spec" in data
        assert "proto_file_content" in data["grpc_spec"]
        assert "syntax" in data["grpc_spec"]["proto_file_content"]
        assert "service" in data["grpc_spec"]["proto_file_content"].lower()

    def test_trace_execution_nesting_depth(self):
        """Test trace_execution calculates nesting depth correctly (#8)."""
        from langchain_azure_ai.wrappers.software_dev_tools import trace_execution
        import json

        code = """
def outer():
    if True:
        for i in range(10):
            if i > 5:
                print(i)
"""
        result = trace_execution.invoke({"code": code})
        data = json.loads(result)

        # Should have nesting depth of at least 4 (function -> if -> for -> if)
        assert data["complexity_indicators"]["nesting_depth"] >= 4

    def test_analyze_test_coverage_parses_results(self):
        """Test analyze_test_coverage parses actual results (#10)."""
        from langchain_azure_ai.wrappers.software_dev_tools import analyze_test_coverage
        import json

        # Test with JSON coverage report
        coverage_report = json.dumps({
            "coverage": {
                "line_coverage": 85.5,
                "branch_coverage": 72.0,
                "function_coverage": 90.0,
            }
        })

        result = analyze_test_coverage.invoke({"test_results": coverage_report})
        data = json.loads(result)

        assert data["coverage"]["line_coverage"] == 85.5
        assert data["coverage"]["branch_coverage"] == 72.0
        assert data["meets_threshold"] is True

    def test_generate_test_data_honors_count(self):
        """Test generate_test_data honors count parameter (#11)."""
        from langchain_azure_ai.wrappers.software_dev_tools import generate_test_data
        import json

        schema = json.dumps({"properties": {"name": {"type": "string"}}})

        result = generate_test_data.invoke({
            "schema": schema,
            "count": 15,
            "data_type": "realistic",
        })

        data = json.loads(result)
        assert data["count_requested"] == 15
        assert data["count_generated"] == 15
        assert len(data["data"]) == 15

    def test_generate_unit_tests_framework_specific(self):
        """Test generate_unit_tests generates framework-specific code (#15)."""
        from langchain_azure_ai.wrappers.software_dev_tools import generate_unit_tests
        import json

        code = "def add(a, b): return a + b"

        # Test pytest
        result_pytest = generate_unit_tests.invoke({"code": code, "framework": "pytest"})
        data_pytest = json.loads(result_pytest)
        assert "import pytest" in data_pytest["test_code"]

        # Test unittest
        result_unittest = generate_unit_tests.invoke({"code": code, "framework": "unittest"})
        data_unittest = json.loads(result_unittest)
        assert "import unittest" in data_unittest["test_code"]
        assert "class TestModule" in data_unittest["test_code"]

        # Test jest
        result_jest = generate_unit_tests.invoke({"code": code, "framework": "jest"})
        data_jest = json.loads(result_jest)
        assert "describe(" in data_jest["test_code"]
        assert "test(" in data_jest["test_code"]

    def test_create_kubernetes_config_has_configmap_hpa(self):
        """Test create_kubernetes_config includes ConfigMap and HPA (#16)."""
        from langchain_azure_ai.wrappers.software_dev_tools import create_kubernetes_config
        import json

        result = create_kubernetes_config.invoke({
            "service_name": "my-service",
            "namespace": "production",
            "replicas": 3,
        })

        data = json.loads(result)
        assert "configmap" in data["manifests"]
        assert "hpa" in data["manifests"]
        assert "HorizontalPodAutoscaler" in data["manifests"]["hpa"]
        assert "ConfigMap" in data["manifests"]["configmap"]

    def test_validate_requirements_uses_rules(self):
        """Test validate_requirements uses validation_rules (#17)."""
        from langchain_azure_ai.wrappers.software_dev_tools import validate_requirements
        import json

        result = validate_requirements.invoke({
            "requirements": "The system should be fast and user-friendly",
            "validation_rules": json.dumps({
                "forbidden_words": ["fast", "user-friendly"],
                "min_length": 10,
            }),
        })

        data = json.loads(result)
        assert "forbidden_words" in data["checks"]
        assert data["checks"]["forbidden_words"]["passed"] is False
        assert "custom_rules_applied" in data

    def test_generate_dockerfile_uses_base_image(self):
        """Test generate_dockerfile uses base_image parameter (#18)."""
        from langchain_azure_ai.wrappers.software_dev_tools import generate_dockerfile
        import json

        result = generate_dockerfile.invoke({
            "project_type": "python-api",
            "base_image": "python:3.12-alpine",
            "port": 8080,
        })

        data = json.loads(result)
        assert data["base_image"] == "python:3.12-alpine"
        assert data["custom_base_image"] is True
        assert "python:3.12-alpine" in data["dockerfile"]

    def test_prioritize_requirements_kano_method(self):
        """Test prioritize_requirements implements Kano method (#19)."""
        from langchain_azure_ai.wrappers.software_dev_tools import prioritize_requirements
        import json

        # Test with Kano-formatted input
        requirements = json.dumps([
            {"req": "Security authentication", "functional": "expect", "dysfunctional": "dislike"},
            {"req": "Dark mode theme", "functional": "like", "dysfunctional": "neutral"},
            {"req": "Slow loading", "functional": "dislike", "dysfunctional": "like"},
        ])

        result = prioritize_requirements.invoke({
            "requirements_list": requirements,
            "method": "kano",
        })

        data = json.loads(result)
        assert data["method"] == "kano"
        assert "methodology" in data
        assert data["methodology"]["name"] == "Kano Model"
        assert "must_be" in data["results"]
        assert "attractive" in data["results"]
        assert "reverse" in data["results"]

    def test_generate_security_report_dynamic_values(self):
        """Test generate_security_report uses actual scan data (#21)."""
        from langchain_azure_ai.wrappers.software_dev_tools import generate_security_report
        import json

        scan_results = json.dumps({
            "total_issues": 5,
            "issues_by_severity": {
                "critical": 1,
                "high": 2,
                "medium": 1,
                "low": 1,
            },
            "issues": [
                {"severity": "critical", "owasp_id": "A03"},
                {"severity": "high", "owasp_id": "A01"},
            ],
        })

        result = generate_security_report.invoke({
            "scan_results": scan_results,
            "format_type": "detailed",
        })

        data = json.loads(result)
        assert data["summary"]["total_findings"] == 5
        assert data["summary"]["critical_findings"] == 1
        assert data["summary"]["high_findings"] == 2
        assert data["summary"]["risk_level"] == "critical"
        assert "URGENT" in data["recommendations"][0]


class TestWrapperRegistration:
    """Tests for wrapper registration (#13)."""

    def test_all_wrappers_exported(self):
        """Test that all wrappers are properly exported."""
        from langchain_azure_ai.wrappers import (
            FoundryAgentWrapper,
            WrapperConfig,
            AgentType,
            ITAgentWrapper,
            ITHelpdeskWrapper,
            ServiceNowWrapper,
            HITLSupportWrapper,
            EnterpriseAgentWrapper,
            ResearchAgentWrapper,
            ContentAgentWrapper,
            DataAnalystWrapper,
            DocumentAgentWrapper,
            CodeAssistantWrapper,
            RAGAgentWrapper,
            DocumentIntelligenceWrapper,
            DeepAgentWrapper,
            SubAgentConfig,
            DeepAgentState,
            ITOperationsWrapper,
            SalesIntelligenceWrapper,
            RecruitmentWrapper,
            SoftwareDevelopmentWrapper,
        )

        # Verify all classes are importable
        assert FoundryAgentWrapper is not None
        assert SoftwareDevelopmentWrapper is not None

    def test_software_dev_tools_all_exported(self):
        """Test that all software dev tools are in __all__."""
        from langchain_azure_ai.wrappers import software_dev_tools

        # Verify __all__ contains expected tools
        assert "analyze_requirements" in software_dev_tools.__all__
        assert "design_architecture" in software_dev_tools.__all__
        assert "generate_code" in software_dev_tools.__all__
        assert "review_code" in software_dev_tools.__all__
        assert "generate_unit_tests" in software_dev_tools.__all__
        assert "scan_security_issues" in software_dev_tools.__all__
        assert "create_ci_pipeline" in software_dev_tools.__all__
        assert "analyze_error" in software_dev_tools.__all__
        assert "generate_api_docs" in software_dev_tools.__all__
        assert "get_all_software_dev_tools" in software_dev_tools.__all__


class TestObservability:
    """Tests for observability module."""

    def test_telemetry_config_from_env(self, mock_env_vars, monkeypatch):
        """Test TelemetryConfig creation from environment."""
        from langchain_azure_ai.observability import TelemetryConfig
        
        monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "test-connection")
        monkeypatch.setenv("ENABLE_AZURE_MONITOR", "true")
        
        config = TelemetryConfig.from_env()
        
        assert config.app_insights_connection == "test-connection"
        assert config.enable_azure_monitor is True

    def test_execution_metrics_finalize(self):
        """Test ExecutionMetrics finalization."""
        from langchain_azure_ai.observability import ExecutionMetrics
        import time
        
        metrics = ExecutionMetrics(agent_name="test", agent_type="custom")
        metrics.prompt_tokens = 100
        metrics.completion_tokens = 50
        
        time.sleep(0.01)  # Small delay to ensure measurable duration
        metrics.finalize()
        
        assert metrics.end_time is not None
        assert metrics.duration_ms > 0
        assert metrics.total_tokens == 150

    def test_agent_telemetry_track_execution(self, mock_env_vars):
        """Test AgentTelemetry context manager."""
        from langchain_azure_ai.observability import AgentTelemetry
        
        telemetry = AgentTelemetry("test-agent", "enterprise")
        
        with telemetry.track_execution("invoke") as metrics:
            metrics.prompt_tokens = 50
            metrics.completion_tokens = 25
        
        assert metrics.success is True
        assert metrics.duration_ms > 0
        assert metrics.total_tokens == 75

    def test_agent_telemetry_error_tracking(self, mock_env_vars):
        """Test that errors are tracked correctly."""
        from langchain_azure_ai.observability import AgentTelemetry
        
        telemetry = AgentTelemetry("test-agent", "enterprise")
        
        try:
            with telemetry.track_execution("invoke") as metrics:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        assert metrics.success is False
        assert "Test error" in metrics.error


class TestServerEndpoints:
    """Tests for FastAPI server endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create a test client for the FastAPI app."""
        from fastapi.testclient import TestClient
        from langchain_azure_ai.server import app
        
        return TestClient(app)

    def test_health_endpoint(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "agents_loaded" in data

    def test_list_agents_endpoint(self, test_client):
        """Test the agents listing endpoint."""
        response = test_client.get("/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_chat_ui_endpoint(self, test_client):
        """Test the chat UI endpoint returns HTML."""
        response = test_client.get("/chat")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestMiddleware:
    """Tests for observability middleware."""

    def test_request_logging_middleware_init(self):
        """Test RequestLoggingMiddleware initialization."""
        from langchain_azure_ai.observability.middleware import RequestLoggingMiddleware
        from starlette.applications import Starlette
        
        app = Starlette()
        middleware = RequestLoggingMiddleware(
            app,
            log_request_body=True,
            exclude_paths=["/health"],
        )
        
        assert middleware.log_request_body is True
        assert "/health" in middleware.exclude_paths

    def test_tracing_middleware_init(self):
        """Test TracingMiddleware initialization."""
        from langchain_azure_ai.observability.middleware import TracingMiddleware
        from starlette.applications import Starlette
        
        app = Starlette()
        middleware = TracingMiddleware(
            app,
            service_name="test-service",
        )
        
        assert middleware.service_name == "test-service"

    def test_metrics_middleware_init(self):
        """Test MetricsMiddleware initialization."""
        from langchain_azure_ai.observability.middleware import MetricsMiddleware
        from starlette.applications import Starlette
        
        app = Starlette()
        middleware = MetricsMiddleware(
            app,
            service_name="test-service",
        )
        
        assert middleware.service_name == "test-service"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
