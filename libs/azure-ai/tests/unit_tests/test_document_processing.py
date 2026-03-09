"""Unit tests for the Copilot Studio document processing endpoint.

Covers:
- DocumentOperation enum validation
- CopilotDocumentRequest model validation
- _process_document_content: input validation, SSRF, DoS limits
- _process_document_content: summarize / analyze / extract operations
- copilot_process_document endpoint (happy path + error paths)
- AgentTelemetry.record_custom_metric counter caching
"""

import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


SMALL_PDF_B64 = _b64(b"%PDF-1.4 tiny test document content")


class _ChatOnlyAgent:
    """Test stub: exposes ``chat()`` but **not** ``process_document()``.

    Used as a ``MagicMock(spec=...)`` template so that
    ``hasattr(mock, 'process_document')`` returns ``False`` while
    ``mock.chat`` remains accessible.
    """

    def chat(self, message: str) -> str:  # noqa: D102
        ...


# ===========================================================================
# DocumentOperation enum
# ===========================================================================

class TestDocumentOperation:
    """Validate the DocumentOperation enum."""

    def test_all_operations_present(self):
        from langchain_azure_ai.server.copilot_studio import DocumentOperation

        expected = {"summarize", "extract_text", "extract_tables", "extract_key_values", "analyze"}
        assert {op.value for op in DocumentOperation} == expected

    def test_enum_is_string(self):
        from langchain_azure_ai.server.copilot_studio import DocumentOperation

        assert isinstance(DocumentOperation.SUMMARIZE, str)
        assert DocumentOperation.SUMMARIZE == "summarize"


# ===========================================================================
# CopilotDocumentRequest model
# ===========================================================================

class TestCopilotDocumentRequest:
    """Pydantic model validation for CopilotDocumentRequest."""

    def test_default_operation_is_summarize(self):
        from langchain_azure_ai.server.copilot_studio import (
            CopilotDocumentRequest,
            DocumentOperation,
        )

        req = CopilotDocumentRequest(documentContent=SMALL_PDF_B64)
        assert req.operation == DocumentOperation.SUMMARIZE

    def test_valid_operation_enum_value(self):
        from langchain_azure_ai.server.copilot_studio import (
            CopilotDocumentRequest,
            DocumentOperation,
        )

        req = CopilotDocumentRequest(
            documentContent=SMALL_PDF_B64,
            operation="extract_tables",
        )
        assert req.operation == DocumentOperation.EXTRACT_TABLES

    def test_invalid_operation_raises(self):
        from langchain_azure_ai.server.copilot_studio import CopilotDocumentRequest
        import pydantic

        with pytest.raises((pydantic.ValidationError, ValueError)):
            CopilotDocumentRequest(
                documentContent=SMALL_PDF_B64,
                operation="INVALID_OP",
            )

    def test_url_field_accepted(self):
        from langchain_azure_ai.server.copilot_studio import CopilotDocumentRequest

        req = CopilotDocumentRequest(documentUrl="https://example.com/doc.pdf")
        assert req.documentUrl == "https://example.com/doc.pdf"
        assert req.documentContent is None

    def test_optional_fields_default_to_none(self):
        from langchain_azure_ai.server.copilot_studio import CopilotDocumentRequest

        req = CopilotDocumentRequest(documentContent=SMALL_PDF_B64)
        assert req.conversationId is None
        assert req.userId is None
        assert req.options is None
        assert req.documentName is None


# ===========================================================================
# _process_document_content — input validation
# ===========================================================================

class TestProcessDocumentValidation:
    """Input validation and security checks in _process_document_content."""

    @pytest.fixture(autouse=True)
    def _patch_registry(self):
        """Patch the registry so no real agent is loaded."""
        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg:
            mock_reg.return_value.get_enterprise_agent.return_value = None
            yield

    # --- missing content --------------------------------------------------

    def test_no_content_no_url_returns_error(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=None,
                url=None,
                filename=None,
                operation="summarize",
                options=None,
                telemetry=None,
            )
        )
        assert result["success"] is False
        assert "No document content provided" in result["error"]

    # --- base64 size limit ------------------------------------------------

    def test_oversized_base64_rejected(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        # base64 encodes 3 bytes as 4 chars, so the pre-decode estimated size
        # is (len * 3) // 4.  To exceed the 50 MB limit the encoded string must
        # be longer than ceil(50 * 1024 * 1024 * 4 / 3) ≈ 66.7 MB chars.
        _50MB = 50 * 1024 * 1024
        min_chars = (_50MB * 4 + 2) // 3   # minimum chars to exceed 50 MB decoded
        min_chars = (min_chars + 3) & ~3    # align to 4 for valid base64 input
        huge_b64 = "A" * min_chars
        result = asyncio.run(
            _process_document_content(
                content=huge_b64,
                url=None,
                filename="big.pdf",
                operation="summarize",
                options=None,
                telemetry=None,
            )
        )
        assert result["success"] is False
        assert "50" in result["error"]  # mentions 50MB

    # --- malformed base64 -------------------------------------------------

    def test_invalid_base64_returns_error(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content="!!!NOT_VALID_BASE64!!!",
                url=None,
                filename="bad.pdf",
                operation="summarize",
                options=None,
                telemetry=None,
            )
        )
        assert result["success"] is False
        assert "base64" in result["error"].lower()

    # --- SSRF: invalid scheme ---------------------------------------------

    def test_ftp_url_rejected(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=None,
                url="ftp://example.com/doc.pdf",
                filename=None,
                operation="summarize",
                options=None,
                telemetry=None,
            )
        )
        assert result["success"] is False
        assert "scheme" in result["error"].lower()

    def test_file_url_scheme_rejected(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=None,
                url="file:///etc/passwd",
                filename=None,
                operation="summarize",
                options=None,
                telemetry=None,
            )
        )
        assert result["success"] is False
        assert "scheme" in result["error"].lower()

    # --- SSRF: localhost --------------------------------------------------

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "0.0.0.0"])
    def test_localhost_url_rejected(self, host):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=None,
                url=f"http://{host}/internal.pdf",
                filename=None,
                operation="summarize",
                options=None,
                telemetry=None,
            )
        )
        assert result["success"] is False
        assert "localhost" in result["error"].lower() or "not allowed" in result["error"].lower()

    # --- SSRF: private IPs -----------------------------------------------

    @pytest.mark.parametrize(
        "private_url",
        [
            "http://192.168.1.100/doc.pdf",
            "http://10.0.0.1/doc.pdf",
            "http://172.16.0.1/doc.pdf",
        ],
    )
    def test_private_ip_url_rejected(self, private_url):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=None,
                url=private_url,
                filename=None,
                operation="summarize",
                options=None,
                telemetry=None,
            )
        )
        assert result["success"] is False
        assert "private" in result["error"].lower() or "not allowed" in result["error"].lower()

    # --- valid HTTPS URL accepted ------------------------------------------

    def test_valid_https_url_proceeds_past_validation(self):
        """A public HTTPS URL should pass validation (pipeline may not be available in unit tests)."""
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=None,
                url="https://example.com/report.pdf",
                filename="report.pdf",
                operation="summarize",
                options=None,
                telemetry=None,
            )
        )
        # Without a pipeline or agent it returns an error, but NOT an SSRF/validation error
        assert "scheme" not in result.get("error", "").lower()
        assert "private" not in result.get("error", "").lower()
        assert "localhost" not in result.get("error", "").lower()

    # --- unknown operation ------------------------------------------------

    def test_unknown_operation_returns_error(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=SMALL_PDF_B64,
                url=None,
                filename="test.pdf",
                operation="do_magic",
                options=None,
                telemetry=None,
            )
        )
        assert result["success"] is False
        assert "do_magic" in result["error"]


# ===========================================================================
# _process_document_content — summarize with doc_agent
# ===========================================================================

class TestProcessDocumentSummarize:
    """Happy-path tests for the summarize operation."""

    def test_summarize_with_process_document_method(self, tmp_path):
        """doc_agent.process_document() is called when the method exists."""
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        mock_agent = MagicMock()
        mock_agent.process_document.return_value = {"summary": "A great summary."}

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg:
            mock_reg.return_value.get_enterprise_agent.return_value = mock_agent

            result = asyncio.run(
                _process_document_content(
                    content=SMALL_PDF_B64,
                    url=None,
                    filename="test.pdf",
                    operation="summarize",
                    options=None,
                    telemetry=None,
                )
            )

        assert result["success"] is True
        assert result["response"] == "A great summary."
        mock_agent.process_document.assert_called_once()

    def test_summarize_fallback_to_chat(self):
        """When doc_agent has no process_document, chat() is called with file text."""
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        mock_agent = MagicMock(spec=_ChatOnlyAgent)
        mock_agent.chat.return_value = "Chat-based summary."

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg:
            mock_reg.return_value.get_enterprise_agent.return_value = mock_agent

            result = asyncio.run(
                _process_document_content(
                    content=SMALL_PDF_B64,
                    url=None,
                    filename="test.pdf",
                    operation="summarize",
                    options=None,
                    telemetry=None,
                )
            )

        assert result["success"] is True
        assert result["response"] == "Chat-based summary."
        call_args = mock_agent.chat.call_args[0][0]
        assert "Document:" in call_args  # file text included in prompt

    def test_summarize_url_via_agent_chat(self):
        """URL-based summarize calls doc_agent.chat() with the URL in the prompt."""
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        mock_agent = MagicMock(spec=_ChatOnlyAgent)
        mock_agent.chat.return_value = "URL summary."

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg:
            mock_reg.return_value.get_enterprise_agent.return_value = mock_agent

            result = asyncio.run(
                _process_document_content(
                    content=None,
                    url="https://example.com/annual_report.pdf",
                    filename=None,
                    operation="summarize",
                    options=None,
                    telemetry=None,
                )
            )

        assert result["success"] is True
        assert result["response"] == "URL summary."
        call_args = mock_agent.chat.call_args[0][0]
        assert "https://example.com/annual_report.pdf" in call_args

    def test_summarize_options_propagated(self):
        """Summarize prompt should incorporate max_length and format options."""
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        mock_agent = MagicMock(spec=_ChatOnlyAgent)
        mock_agent.chat.return_value = "Bullet summary."

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg:
            mock_reg.return_value.get_enterprise_agent.return_value = mock_agent

            result = asyncio.run(
                _process_document_content(
                    content=None,
                    url="https://example.com/doc.pdf",
                    filename=None,
                    operation="summarize",
                    options={"max_length": 200, "format": "bullet_points"},
                    telemetry=None,
                )
            )

        prompt = mock_agent.chat.call_args[0][0]
        assert "200" in prompt
        assert "bullet" in prompt.lower()


# ===========================================================================
# _process_document_content — analyze operation
# ===========================================================================

class TestProcessDocumentAnalyze:
    """Tests for the analyze operation."""

    def test_analyze_with_agent(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        mock_agent = MagicMock(spec=_ChatOnlyAgent)
        mock_agent.chat.return_value = "Comprehensive analysis result."

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg:
            mock_reg.return_value.get_enterprise_agent.return_value = mock_agent

            result = asyncio.run(
                _process_document_content(
                    content=None,
                    url="https://example.com/report.pdf",
                    filename="report.pdf",
                    operation="analyze",
                    options=None,
                    telemetry=None,
                )
            )

        assert result["success"] is True
        assert result["response"] == "Comprehensive analysis result."
        prompt = mock_agent.chat.call_args[0][0]
        assert "comprehensively" in prompt.lower() or "Analyze" in prompt

    def test_analyze_without_agent_or_pipeline_returns_error(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg, patch(
            "langchain_azure_ai.server.copilot_studio._get_document_pipeline",
            return_value=None,
        ):
            mock_reg.return_value.get_enterprise_agent.return_value = None

            result = asyncio.run(
                _process_document_content(
                    content=None,
                    url="https://example.com/report.pdf",
                    filename=None,
                    operation="analyze",
                    options=None,
                    telemetry=None,
                )
            )

        assert result["success"] is False


# ===========================================================================
# _process_document_content — pipeline-backed operations
# ===========================================================================

class TestProcessDocumentPipelineOperations:
    """Tests for extract_text, extract_tables, extract_key_values using the pipeline."""

    def _make_pipeline_result(self):
        """Return a mock ProcessedDocument-like object."""
        table1 = MagicMock()
        table1.markdown = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"

        processed = MagicMock()
        processed.text_content = "Hello world. This is extracted text."
        processed.tables = [table1]
        processed.key_value_pairs = [("Invoice No", "INV-001"), ("Date", "2026-03-09")]
        processed.layout = MagicMock()
        processed.layout.page_count = 2
        return processed

    @pytest.fixture
    def patched_pipeline(self):
        processed = self._make_pipeline_result()
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(return_value=processed)

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg, patch(
            "langchain_azure_ai.server.copilot_studio._get_document_pipeline",
            return_value=mock_pipeline,
        ):
            mock_reg.return_value.get_enterprise_agent.return_value = None
            yield mock_pipeline, processed

    def test_extract_text_returns_full_text(self, patched_pipeline):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=None,
                url="https://example.com/doc.pdf",
                filename=None,
                operation="extract_text",
                options=None,
                telemetry=None,
            )
        )

        assert result["success"] is True
        assert result["response"] == "Hello world. This is extracted text."
        assert result["metadata"]["text_length"] == len("Hello world. This is extracted text.")

    def test_extract_tables_returns_markdown(self, patched_pipeline):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=None,
                url="https://example.com/doc.pdf",
                filename=None,
                operation="extract_tables",
                options=None,
                telemetry=None,
            )
        )

        assert result["success"] is True
        assert "Col1" in result["response"]
        assert result["metadata"]["tables_found"] == 1

    def test_extract_tables_no_tables_message(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        processed = MagicMock()
        processed.tables = []
        processed.text_content = "Some text"
        processed.layout = MagicMock()
        processed.layout.page_count = 1
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(return_value=processed)

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg, patch(
            "langchain_azure_ai.server.copilot_studio._get_document_pipeline",
            return_value=mock_pipeline,
        ):
            mock_reg.return_value.get_enterprise_agent.return_value = None

            result = asyncio.run(
                _process_document_content(
                    content=None,
                    url="https://example.com/doc.pdf",
                    filename=None,
                    operation="extract_tables",
                    options=None,
                    telemetry=None,
                )
            )

        assert result["success"] is True
        assert "No tables found" in result["response"]

    def test_extract_key_values(self, patched_pipeline):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        result = asyncio.run(
            _process_document_content(
                content=None,
                url="https://example.com/doc.pdf",
                filename=None,
                operation="extract_key_values",
                options=None,
                telemetry=None,
            )
        )

        assert result["success"] is True
        assert "Invoice No" in result["response"]
        assert "INV-001" in result["response"]
        assert result["metadata"]["pairs_found"] == 2

    def test_extract_key_values_empty(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        processed = MagicMock()
        processed.key_value_pairs = []
        processed.text_content = ""
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(return_value=processed)

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg, patch(
            "langchain_azure_ai.server.copilot_studio._get_document_pipeline",
            return_value=mock_pipeline,
        ):
            mock_reg.return_value.get_enterprise_agent.return_value = None

            result = asyncio.run(
                _process_document_content(
                    content=None,
                    url="https://example.com/doc.pdf",
                    filename=None,
                    operation="extract_key_values",
                    options=None,
                    telemetry=None,
                )
            )

        assert result["success"] is True
        assert "No key-value pairs" in result["response"]


# ===========================================================================
# _process_document_content — telemetry recording
# ===========================================================================

class TestProcessDocumentTelemetry:
    """Telemetry is recorded after successful processing."""

    def test_telemetry_metrics_recorded_on_success(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        mock_telemetry = MagicMock()
        mock_agent = MagicMock(spec=_ChatOnlyAgent)
        mock_agent.chat.return_value = "summary text"

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg:
            mock_reg.return_value.get_enterprise_agent.return_value = mock_agent

            asyncio.run(
                _process_document_content(
                    content=None,
                    url="https://example.com/doc.pdf",
                    filename=None,
                    operation="summarize",
                    options=None,
                    telemetry=mock_telemetry,
                )
            )

        # At least processing_time_ms should be recorded
        calls = [c[0][0] for c in mock_telemetry.record_custom_metric.call_args_list]
        assert "document_processing_time_ms" in calls

    def test_telemetry_error_recorded_on_exception(self):
        from langchain_azure_ai.server.copilot_studio import _process_document_content

        mock_telemetry = MagicMock()

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg:
            mock_reg.return_value.get_enterprise_agent.side_effect = RuntimeError("boom")

            result = asyncio.run(
                _process_document_content(
                    content=SMALL_PDF_B64,
                    url=None,
                    filename="test.pdf",
                    operation="summarize",
                    options=None,
                    telemetry=mock_telemetry,
                )
            )

        assert result["success"] is False
        mock_telemetry.record_error.assert_called_once()


# ===========================================================================
# AgentTelemetry.record_custom_metric — counter caching
# ===========================================================================

class TestAgentTelemetryMetricCache:
    """The instrument for a given metric name must only be created once."""

    def _make_telemetry(self):
        from langchain_azure_ai.observability import AgentTelemetry

        # Clear class-level cache to prevent cross-test contamination.
        AgentTelemetry._metric_instrument_cache.clear()

        telemetry = AgentTelemetry.__new__(AgentTelemetry)
        # __new__ bypasses __init__, so set the attributes that record_custom_metric needs.
        telemetry.agent_name = "test-agent"
        telemetry.agent_type = "test"
        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_meter.create_up_down_counter.return_value = mock_counter
        telemetry._meter = mock_meter
        telemetry._tracer = None
        return telemetry, mock_meter, mock_counter

    def test_counter_created_only_once_for_same_name(self):
        telemetry, mock_meter, mock_counter = self._make_telemetry()

        telemetry.record_custom_metric("my_metric", 1.0)
        telemetry.record_custom_metric("my_metric", 2.0)
        telemetry.record_custom_metric("my_metric", 3.0)

        # Instrument should be created exactly once
        assert mock_meter.create_up_down_counter.call_count == 1
        # add() should be called three times
        assert mock_counter.add.call_count == 3

    def test_different_names_create_separate_counters(self):
        telemetry, mock_meter, _ = self._make_telemetry()

        telemetry.record_custom_metric("metric_a", 1.0)
        telemetry.record_custom_metric("metric_b", 2.0)
        telemetry.record_custom_metric("metric_a", 3.0)

        # Two distinct instruments, one per unique metric name
        assert mock_meter.create_up_down_counter.call_count == 2
        names_created = {
            c.kwargs.get("name") or c.args[0]
            for c in mock_meter.create_up_down_counter.call_args_list
        }
        # Names are prefixed as "agent.custom.<metric_name>"
        assert any("metric_a" in n for n in names_created)
        assert any("metric_b" in n for n in names_created)

    def test_string_metric_does_not_create_counter(self):
        """String values should be logged as attributes, not as counters."""
        telemetry, mock_meter, _ = self._make_telemetry()

        telemetry.record_custom_metric("operation_name", "summarize")

        mock_meter.create_up_down_counter.assert_not_called()

    def test_no_meter_is_a_noop(self):
        """If no meter is configured, record_custom_metric should not raise."""
        from langchain_azure_ai.observability import AgentTelemetry

        telemetry = AgentTelemetry.__new__(AgentTelemetry)
        telemetry._meter = None
        telemetry._tracer = None

        # Should complete without raising
        telemetry.record_custom_metric("some_metric", 42.0)


# ===========================================================================
# copilot_process_document endpoint (FastAPI)
# ===========================================================================

class TestCopilotProcessDocumentEndpoint:
    """FastAPI endpoint integration tests using httpx.AsyncClient + TestClient."""

    @pytest.fixture
    def client(self):
        """Create a TestClient with auth dependency overridden for testing."""
        try:
            from fastapi.testclient import TestClient
            from langchain_azure_ai.server.copilot_studio import router, verify_api_key
            from fastapi import FastAPI

            app = FastAPI()
            # router already has prefix="/api/copilot"; do not add a second prefix.
            app.include_router(router)

            # Override the auth dependency so tests never fail due to API key
            # configuration being cached at module import time.
            app.dependency_overrides[verify_api_key] = lambda: "test_auth"

            return TestClient(app, raise_server_exceptions=False)
        except Exception:
            pytest.skip("FastAPI TestClient not available in this environment")

    def test_missing_document_returns_400(self, client):
        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg, patch(
            "langchain_azure_ai.server.copilot_studio.get_copilot_telemetry",
            return_value=None,
        ):
            mock_reg.return_value.get_enterprise_agent.return_value = None
            resp = client.post(
                "/api/copilot/document",
                json={"operation": "summarize"},
            )
        # Either 400 (validation) or 422 (pydantic) is acceptable
        assert resp.status_code in (400, 422)

    def test_invalid_base64_returns_400(self, client):
        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg, patch(
            "langchain_azure_ai.server.copilot_studio.get_copilot_telemetry",
            return_value=None,
        ):
            mock_reg.return_value.get_enterprise_agent.return_value = None

            resp = client.post(
                "/api/copilot/document",
                json={"documentContent": "!!!invalid-b64!!!", "operation": "summarize"},
            )

        assert resp.status_code == 400
        assert "base64" in resp.json().get("detail", "").lower()

    def test_ssrf_localhost_returns_400(self, client):
        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg, patch(
            "langchain_azure_ai.server.copilot_studio.get_copilot_telemetry",
            return_value=None,
        ):
            mock_reg.return_value.get_enterprise_agent.return_value = None

            resp = client.post(
                "/api/copilot/document",
                json={"documentUrl": "http://localhost/secret", "operation": "summarize"},
            )

        assert resp.status_code == 400
        assert "localhost" in resp.json().get("detail", "").lower() or "not allowed" in resp.json().get("detail", "").lower()

    def test_successful_summarize_returns_200(self, client):
        mock_agent = MagicMock(spec=_ChatOnlyAgent)
        mock_agent.chat.return_value = "This document is about quarterly results."

        with patch(
            "langchain_azure_ai.server.copilot_studio.get_registry"
        ) as mock_reg, patch(
            "langchain_azure_ai.server.copilot_studio.get_copilot_telemetry",
            return_value=None,
        ):
            mock_reg.return_value.get_enterprise_agent.return_value = mock_agent

            resp = client.post(
                "/api/copilot/document",
                json={
                    "documentUrl": "https://example.com/report.pdf",
                    "operation": "summarize",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "This document is about quarterly results."
        assert body["operation"] == "summarize"
