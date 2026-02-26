"""Copilot Studio Integration API Routes.

This module provides REST API endpoints for Microsoft Copilot Studio integration.
It exposes agent capabilities as OpenAPI-compliant endpoints that can be consumed
by Copilot Studio as custom connectors or plugins.

Key Features:
- OpenAPI 2.0 (Swagger) specification for Copilot Studio compatibility
- Plugin manifest (ai-plugin.json) for Microsoft 365 Copilot
- OAuth 2.0 and API Key authentication support
- Health monitoring and observability
- Multi-agent routing with conversation context

References:
- https://learn.microsoft.com/microsoft-copilot-studio/agent-extend-action-rest-api
- https://learn.microsoft.com/connectors/custom-connectors/define-openapi-definition
- https://learn.microsoft.com/microsoft-365-copilot/extensibility/overview-api-plugins
"""

import base64
import json
import logging
import os
import secrets
import tempfile
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# Observability Integration
# ============================================================================
try:
    from langchain_azure_ai.observability import AgentTelemetry
    OBSERVABILITY_AVAILABLE = True
    logger.info("Copilot Studio observability enabled")
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    logger.info("Copilot Studio observability not available (optional)")


def get_copilot_telemetry(agent_id: str = "copilot-studio") -> Optional["AgentTelemetry"]:
    """Get telemetry instance for Copilot Studio operations."""
    if not OBSERVABILITY_AVAILABLE:
        return None
    telemetry = AgentTelemetry(agent_name=agent_id)
    # Attach custom dimensions after initialization
    if hasattr(telemetry, "custom_dimensions") and isinstance(telemetry.custom_dimensions, dict):
        telemetry.custom_dimensions.update({
            "source": "copilot_studio",
            "integration": "custom_connector"
        })
    return telemetry


# ============================================================================
# Security Configuration
# ============================================================================

# API Key header scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_SCHEME = HTTPBearer(auto_error=False)


class SecurityConfig:
    """Security configuration for Copilot Studio endpoints."""

    def __init__(self):
        self.api_key_enabled = os.getenv("COPILOT_API_KEY_ENABLED", "true").lower() == "true"
        self.api_key = os.getenv("COPILOT_API_KEY", "")
        self.oauth_enabled = os.getenv("COPILOT_OAUTH_ENABLED", "false").lower() == "true"
        self.tenant_id = os.getenv("AZURE_TENANT_ID", "")
        self.client_id = os.getenv("COPILOT_CLIENT_ID", "")
        self.allowed_origins = os.getenv("COPILOT_ALLOWED_ORIGINS", "").split(",")
        self.rate_limit_rpm = int(os.getenv("COPILOT_RATE_LIMIT_RPM", "60"))

    def validate_api_key(self, provided_key: str) -> bool:
        """Validate API key using constant-time comparison."""
        if not self.api_key:
            logger.warning("COPILOT_API_KEY not configured - rejecting all requests")
            return False
        return secrets.compare_digest(provided_key, self.api_key)


security_config = SecurityConfig()


async def verify_api_key(
    api_key: Optional[str] = Depends(API_KEY_HEADER),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(BEARER_SCHEME),
) -> str:
    """Verify API key or Bearer token for authentication.

    Supports both API Key (X-API-Key header) and Bearer token authentication.
    """
    if not security_config.api_key_enabled:
        return "anonymous"

    # Check API Key first
    if api_key and security_config.validate_api_key(api_key):
        return "api_key_auth"

    # Check Bearer token
    if bearer and bearer.credentials:
        # For OAuth, we'd validate the JWT here
        # For now, treat as API key
        if security_config.validate_api_key(bearer.credentials):
            return "bearer_auth"

    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ============================================================================
# Request/Response Models (Copilot Studio Compatible)
# ============================================================================

class CopilotChatRequest(BaseModel):
    """Chat request model compatible with Copilot Studio.

    This model follows the expected format for Copilot Studio custom connectors.
    """
    message: str = Field(..., description="The user message to process")
    conversationId: Optional[str] = Field(None, description="Conversation ID for context continuity")
    userId: Optional[str] = Field(None, description="User identifier")
    locale: str = Field("en-US", description="User locale for response formatting")
    channelId: str = Field("copilot", description="Channel identifier (copilot, teams, web)")
    agentHint: Optional[str] = Field(None, description="Preferred agent ID (optional hint for routing)")

    class Config:
        schema_extra = {
            "example": {
                "message": "Help me reset my password",
                "conversationId": "conv-12345",
                "userId": "user@contoso.com",
                "locale": "en-US",
                "channelId": "copilot"
            }
        }


class CopilotChatResponse(BaseModel):
    """Chat response model for Copilot Studio."""
    response: str = Field(..., description="The agent's response text")
    conversationId: str = Field(..., description="Conversation ID for follow-up")
    agentId: str = Field(..., description="The agent that processed the request")
    agentType: str = Field(..., description="Type of agent (IT, Enterprise, DeepAgent)")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "response": "I can help you reset your password. Please follow these steps...",
                "conversationId": "conv-12345",
                "agentId": "helpdesk",
                "agentType": "IT",
                "timestamp": "2026-02-13T10:30:00Z",
                "suggestions": ["Check password requirements", "Contact IT support"],
                "metadata": {"confidence": 0.95}
            }
        }


class AgentCapability(BaseModel):
    """Describes an agent's capability for Copilot discovery."""
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Agent description for LLM understanding")
    type: str = Field(..., description="Agent type (IT, Enterprise, DeepAgent)")
    capabilities: List[str] = Field(..., description="Supported operations")
    triggerPhrases: List[str] = Field(default_factory=list, description="Phrases that invoke this agent")


class AgentListResponse(BaseModel):
    """Response model for agent list endpoint."""
    agents: List[AgentCapability] = Field(..., description="List of available agents")
    total: int = Field(..., description="Total number of agents")


class PluginManifest(BaseModel):
    """Microsoft 365 Copilot plugin manifest."""
    schema_version: str = Field("v2.1", alias="$schema_version")
    name_for_human: str
    name_for_model: str
    description_for_human: str
    description_for_model: str
    api: Dict[str, Any]
    auth: Dict[str, Any]
    logo_url: str
    contact_email: str
    legal_info_url: str


class HealthStatus(BaseModel):
    """Health check response for monitoring."""
    status: str
    timestamp: str
    version: str
    agents_available: int
    copilot_integration: bool
    authentication: str


# ============================================================================
# Document Processing Models (Copilot Studio Compatible)
# ============================================================================

class DocumentOperation(str, Enum):
    """Supported document operations."""
    SUMMARIZE = "summarize"
    EXTRACT_TEXT = "extract_text"
    EXTRACT_TABLES = "extract_tables"
    EXTRACT_KEY_VALUES = "extract_key_values"
    ANALYZE = "analyze"


class CopilotDocumentRequest(BaseModel):
    """Document processing request model for Copilot Studio.

    Supports multiple input methods: base64 content, URL, or file path.
    """
    documentContent: Optional[str] = Field(
        None,
        description="Base64-encoded document content"
    )
    documentUrl: Optional[str] = Field(
        None,
        description="URL to the document (must be publicly accessible or pre-authenticated)"
    )
    documentName: Optional[str] = Field(
        None,
        description="Original filename for type detection"
    )
    operation: str = Field(
        "summarize",
        description="Operation to perform: summarize, extract_text, extract_tables, extract_key_values, analyze"
    )
    conversationId: Optional[str] = Field(
        None,
        description="Conversation ID for context continuity"
    )
    userId: Optional[str] = Field(
        None,
        description="User identifier"
    )
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional processing options (max_length, language, format)"
    )

    class Config:
        schema_extra = {
            "example": {
                "documentContent": "JVBERi0xLjQK...",  # Base64 PDF
                "documentName": "quarterly_report.pdf",
                "operation": "summarize",
                "conversationId": "conv-12345",
                "options": {"max_length": 500, "format": "bullet_points"}
            }
        }


class CopilotDocumentResponse(BaseModel):
    """Document processing response for Copilot Studio."""
    response: str = Field(..., description="Processed result (summary, extracted text, etc.)")
    conversationId: str = Field(..., description="Conversation ID for follow-up")
    operation: str = Field(..., description="Operation that was performed")
    documentName: Optional[str] = Field(None, description="Processed document name")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")

    class Config:
        schema_extra = {
            "example": {
                "response": "## Executive Summary\n\nThe quarterly report shows...",
                "conversationId": "conv-12345",
                "operation": "summarize",
                "documentName": "quarterly_report.pdf",
                "timestamp": "2026-02-26T10:30:00Z",
                "metadata": {
                    "pages_processed": 12,
                    "tables_found": 3,
                    "processing_time_ms": 2500
                },
                "suggestions": [
                    "Extract all tables from this document",
                    "Analyze key financial metrics",
                    "Compare with previous quarter"
                ]
            }
        }


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(
    prefix="/api/copilot",
    tags=["copilot-studio"],
    responses={
        401: {"description": "Authentication required"},
        429: {"description": "Rate limit exceeded"},
    },
)


# ============================================================================
# Plugin Discovery Endpoints
# ============================================================================

@router.get(
    "/plugin-manifest",
    response_model=Dict[str, Any],
    summary="Get Plugin Manifest",
    description="Returns the ai-plugin.json manifest for Microsoft 365 Copilot integration.",
)
async def get_plugin_manifest(request: Request):
    """Get the plugin manifest for Copilot Studio registration.

    This manifest describes the plugin to Microsoft 365 Copilot and enables
    automatic discovery and invocation.
    """
    base_url = str(request.base_url).rstrip("/")

    manifest = {
        "$schema": "https://developer.microsoft.com/json-schemas/copilot/plugin/v2.1/schema.json",
        "schema_version": "v2.1",
        "name_for_human": "Azure AI Foundry Agents",
        "name_for_model": "azure_ai_foundry_agents",
        "description_for_human": "Enterprise AI agents for IT support, business intelligence, and automation powered by Azure AI Foundry and LangChain.",
        "description_for_model": (
            "Use this plugin to interact with enterprise AI agents and process documents. "
            "Available capabilities: "
            "1) DOCUMENT PROCESSING - Summarize documents, extract text/tables/key-values, analyze PDFs/Word/PowerPoint. "
            "2) IT SUPPORT - Helpdesk (password resets, technical support), ServiceNow (ticket creation, incident management). "
            "3) BUSINESS PRODUCTIVITY - Research (information gathering), Content (document creation), "
            "Data Analyst (data analysis, visualization), Code Assistant (code review, debugging). "
            "4) SPECIALIZED AGENTS - IT Operations (infrastructure), Sales Intelligence (deal analysis), "
            "Recruitment (candidate screening), Software Development (full SDLC support). "
            "For document tasks, use the /document endpoint. For chat tasks, use /chat endpoint."
        ),
        "api": {
            "type": "openapi",
            "url": f"{base_url}/api/copilot/openapi.json"
        },
        "auth": {
            "type": "api_key" if security_config.api_key_enabled else "none",
            "api_key_header": "X-API-Key"
        },
        "logo_url": os.getenv("COPILOT_LOGO_URL", f"{base_url}/static/logo.png"),
        "contact_email": os.getenv("COPILOT_CONTACT_EMAIL", "support@azure.microsoft.com"),
        "legal_info_url": os.getenv("COPILOT_LEGAL_URL", "https://azure.microsoft.com/terms/"),
        "capabilities": {
            "conversation": True,
            "localization": {
                "supported_locales": ["en-US", "en-GB", "de-DE", "fr-FR", "es-ES"]
            }
        }
    }

    return JSONResponse(content=manifest)


@router.get(
    "/openapi.json",
    response_model=Dict[str, Any],
    summary="Get OpenAPI Specification",
    description="Returns OpenAPI 2.0 (Swagger) specification for Copilot Studio custom connector.",
)
async def get_openapi_spec(request: Request):
    """Get OpenAPI 2.0 specification for Copilot Studio.

    Copilot Studio requires OpenAPI 2.0 format (Swagger) for custom connectors.
    This endpoint provides a compatible specification.
    """
    base_url = str(request.base_url).rstrip("/")
    host = request.url.netloc

    openapi_spec = {
        "swagger": "2.0",
        "info": {
            "title": "Azure AI Foundry Agents API",
            "description": (
                "Enterprise AI agents for IT support, business intelligence, and automation. "
                "Powered by Azure AI Foundry and LangChain."
            ),
            "version": "2.0.0",
            "contact": {
                "name": "Azure AI Foundry Team",
                "url": "https://github.com/microsoft/langchain-azure"
            }
        },
        "host": host,
        "basePath": "/api/copilot",
        "schemes": ["https"] if request.url.scheme == "https" else ["http", "https"],
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "securityDefinitions": {
            "apiKey": {
                "type": "apiKey",
                "name": "X-API-Key",
                "in": "header"
            }
        },
        "security": [{"apiKey": []}] if security_config.api_key_enabled else [],
        "paths": {
            "/chat": {
                "post": {
                    "operationId": "chat",
                    "summary": "Chat with AI Agent",
                    "description": (
                        "Send a message to the AI agent system. The system will automatically "
                        "route the request to the most appropriate agent based on the message content."
                    ),
                    "x-ms-visibility": "important",
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "required": ["message"],
                                "properties": {
                                    "message": {
                                        "type": "string",
                                        "description": "The user message to process",
                                        "x-ms-summary": "Message"
                                    },
                                    "conversationId": {
                                        "type": "string",
                                        "description": "Conversation ID for context continuity",
                                        "x-ms-summary": "Conversation ID"
                                    },
                                    "agentHint": {
                                        "type": "string",
                                        "description": "Optional hint for agent selection (helpdesk, servicenow, research, etc.)",
                                        "x-ms-summary": "Agent Hint",
                                        "enum": [
                                            "helpdesk", "servicenow", "research", "content",
                                            "data_analyst", "code_assistant", "it_operations",
                                            "sales_intelligence", "recruitment", "software_development"
                                        ]
                                    }
                                }
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "response": {
                                        "type": "string",
                                        "description": "The agent's response"
                                    },
                                    "conversationId": {
                                        "type": "string",
                                        "description": "Conversation ID for follow-up"
                                    },
                                    "agentId": {
                                        "type": "string",
                                        "description": "Agent that processed the request"
                                    },
                                    "agentType": {
                                        "type": "string",
                                        "description": "Type of agent"
                                    },
                                    "suggestions": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Follow-up suggestions"
                                    }
                                }
                            }
                        },
                        "401": {
                            "description": "Authentication required"
                        },
                        "500": {
                            "description": "Server error"
                        }
                    }
                }
            },
            "/chat/{agentId}": {
                "post": {
                    "operationId": "chatWithAgent",
                    "summary": "Chat with Specific Agent",
                    "description": "Send a message directly to a specific agent.",
                    "x-ms-visibility": "advanced",
                    "parameters": [
                        {
                            "name": "agentId",
                            "in": "path",
                            "required": True,
                            "type": "string",
                            "description": "The agent identifier",
                            "enum": [
                                "helpdesk", "servicenow", "hitl_support",
                                "research", "content", "data_analyst", "document",
                                "code_assistant", "rag", "document_intelligence",
                                "it_operations", "sales_intelligence", "recruitment",
                                "software_development"
                            ]
                        },
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "required": ["message"],
                                "properties": {
                                    "message": {
                                        "type": "string",
                                        "description": "The user message to process"
                                    },
                                    "conversationId": {
                                        "type": "string",
                                        "description": "Conversation ID for context continuity"
                                    }
                                }
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "schema": {
                                "$ref": "#/definitions/ChatResponse"
                            }
                        }
                    }
                }
            },
            "/agents": {
                "get": {
                    "operationId": "listAgents",
                    "summary": "List Available Agents",
                    "description": "Get a list of all available agents and their capabilities.",
                    "x-ms-visibility": "advanced",
                    "responses": {
                        "200": {
                            "description": "List of agents with total count",
                            "schema": {
                                "$ref": "#/definitions/AgentListResponse"
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "operationId": "healthCheck",
                    "summary": "Health Check",
                    "description": "Check the health status of the API.",
                    "x-ms-visibility": "internal",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "schema": {
                                "$ref": "#/definitions/HealthStatus"
                            }
                        }
                    }
                }
            },
            "/document": {
                "post": {
                    "operationId": "processDocument",
                    "summary": "Process Document",
                    "description": (
                        "Process a document for summarization, text extraction, table extraction, "
                        "or comprehensive analysis. Supports PDF, Word, PowerPoint, and images."
                    ),
                    "x-ms-visibility": "important",
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "documentContent": {
                                        "type": "string",
                                        "description": "Base64-encoded document content",
                                        "x-ms-summary": "Document Content (Base64)"
                                    },
                                    "documentUrl": {
                                        "type": "string",
                                        "description": "URL to the document",
                                        "x-ms-summary": "Document URL"
                                    },
                                    "documentName": {
                                        "type": "string",
                                        "description": "Original filename for type detection",
                                        "x-ms-summary": "Document Name"
                                    },
                                    "operation": {
                                        "type": "string",
                                        "description": "Operation to perform on the document",
                                        "x-ms-summary": "Operation",
                                        "enum": ["summarize", "extract_text", "extract_tables", "extract_key_values", "analyze"],
                                        "default": "summarize"
                                    },
                                    "conversationId": {
                                        "type": "string",
                                        "description": "Conversation ID for context continuity",
                                        "x-ms-summary": "Conversation ID"
                                    },
                                    "options": {
                                        "type": "object",
                                        "description": "Additional processing options",
                                        "x-ms-summary": "Options"
                                    }
                                }
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Document processed successfully",
                            "schema": {
                                "$ref": "#/definitions/DocumentResponse"
                            }
                        },
                        "400": {
                            "description": "Invalid request or document"
                        },
                        "401": {
                            "description": "Authentication required"
                        },
                        "500": {
                            "description": "Processing error"
                        }
                    }
                }
            }
        },
        "definitions": {
            "ChatResponse": {
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "conversationId": {"type": "string"},
                    "agentId": {"type": "string"},
                    "agentType": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "AgentInfo": {
                "type": "object",
                "required": ["id", "name", "description", "type", "capabilities"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Agent ID"
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable name"
                    },
                    "description": {
                        "type": "string",
                        "description": "Agent description for LLM understanding"
                    },
                    "type": {
                        "type": "string",
                        "description": "Agent type (IT, Enterprise, DeepAgent)"
                    },
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Supported operations"
                    },
                    "triggerPhrases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Phrases that invoke this agent",
                        "default": []
                    }
                }
            },
            "AgentListResponse": {
                "type": "object",
                "required": ["agents", "total"],
                "properties": {
                    "agents": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/AgentInfo"},
                        "description": "List of available agents"
                    },
                    "total": {
                        "type": "integer",
                        "description": "Total number of agents"
                    }
                }
            },
            "HealthStatus": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "version": {"type": "string"},
                    "agents_available": {"type": "integer"}
                }
            },
            "DocumentResponse": {
                "type": "object",
                "required": ["response", "conversationId", "operation", "timestamp"],
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Processed result (summary, extracted text, etc.)"
                    },
                    "conversationId": {
                        "type": "string",
                        "description": "Conversation ID for follow-up"
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation that was performed"
                    },
                    "documentName": {
                        "type": "string",
                        "description": "Processed document name"
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "ISO 8601 timestamp"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Processing metadata"
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Follow-up suggestions"
                    }
                }
            }
        }
    }

    return JSONResponse(content=openapi_spec)


# ============================================================================
# Main Chat Endpoint
# ============================================================================

# Import registry from main server (will be injected)
_registry = None


def set_registry(registry):
    """Set the agent registry for the Copilot routes."""
    global _registry
    _registry = registry


def get_registry():
    """Get the agent registry."""
    if _registry is None:
        raise HTTPException(
            status_code=503,
            detail="Agent registry not initialized"
        )
    return _registry


@router.post(
    "/chat",
    response_model=CopilotChatResponse,
    summary="Chat with AI Agent",
    description=(
        "Send a message to the AI agent system. The system will automatically "
        "route the request to the most appropriate agent based on the message content."
    ),
)
async def copilot_chat(
    request: CopilotChatRequest,
    auth: str = Depends(verify_api_key),
):
    """Main chat endpoint for Copilot Studio integration.

    This endpoint:
    1. Routes messages to the appropriate agent based on content or agentHint
    2. Maintains conversation context via conversationId
    3. Returns structured responses with suggestions
    """
    registry = get_registry()

    conversation_id = request.conversationId or str(uuid.uuid4())
    start_time = time.time()

    # Prefer agentHint if provided and valid, otherwise use keyword routing
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None

    if request.agentHint:
        # Try to resolve the agent hint
        if registry.get_it_agent(request.agentHint):
            agent_id = request.agentHint
            agent_type = "IT"
        elif registry.get_enterprise_agent(request.agentHint):
            agent_id = request.agentHint
            agent_type = "Enterprise"
        elif registry.get_deep_agent(request.agentHint):
            agent_id = request.agentHint
            agent_type = "DeepAgent"

    # Fall back to keyword-based routing if hint doesn't resolve
    if not agent_id or not agent_type:
        agent_id, agent_type = _route_to_agent(request.message, registry)

    # Initialize telemetry for this request
    telemetry = get_copilot_telemetry(f"copilot-{agent_id}")

    try:
        # Get the agent
        if agent_type == "IT":
            agent = registry.get_it_agent(agent_id)
        elif agent_type == "Enterprise":
            agent = registry.get_enterprise_agent(agent_id)
        elif agent_type == "DeepAgent":
            agent = registry.get_deep_agent(agent_id)
        else:
            agent = registry.get_it_agent("helpdesk")  # Fallback
            agent_id = "helpdesk"
            agent_type = "IT"

        if not agent:
            # Fallback to helpdesk
            agent = registry.get_it_agent("helpdesk")
            agent_id = "helpdesk"
            agent_type = "IT"

        # Execute chat with telemetry tracking
        if telemetry:
            with telemetry.track_execution() as span:
                if span:
                    span.set_attribute("copilot.conversation_id", conversation_id)
                    span.set_attribute("copilot.agent_id", agent_id)
                    span.set_attribute("copilot.agent_type", agent_type)
                    span.set_attribute("copilot.channel", request.channelId or "unknown")
                response_text = agent.chat(request.message, thread_id=conversation_id)
        else:
            response_text = agent.chat(request.message, thread_id=conversation_id)

        # Generate suggestions based on agent type
        suggestions = _generate_suggestions(agent_id, response_text)

        duration_ms = (time.time() - start_time) * 1000

        # Record telemetry metrics
        if telemetry:
            telemetry.record_custom_metric("copilot_chat_duration_ms", duration_ms)

        logger.info(
            f"Copilot chat completed: agent={agent_id}, "
            f"conversation={conversation_id}, duration={duration_ms:.2f}ms"
        )

        return CopilotChatResponse(
            response=response_text,
            conversationId=conversation_id,
            agentId=agent_id,
            agentType=agent_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            suggestions=suggestions,
            metadata={
                "duration_ms": duration_ms,
                "channel": request.channelId,
                "locale": request.locale,
            }
        )

    except Exception as e:
        if telemetry:
            telemetry.record_error(str(e))
        # Log full error details internally
        logger.error(f"Error in Copilot chat: {e}", exc_info=True)
        # Return generic error to client (don't expose internal details)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again or contact support."
        )


@router.post(
    "/chat/{agent_id}",
    response_model=CopilotChatResponse,
    summary="Chat with Specific Agent",
    description="Send a message directly to a specific agent.",
)
async def copilot_chat_specific(
    agent_id: str,
    request: CopilotChatRequest,
    auth: str = Depends(verify_api_key),
):
    """Chat with a specific agent by ID."""
    registry = get_registry()

    conversation_id = request.conversationId or str(uuid.uuid4())

    # Look up agent across all registries
    agent = None
    agent_type = "Unknown"

    if registry.get_it_agent(agent_id):
        agent = registry.get_it_agent(agent_id)
        agent_type = "IT"
    elif registry.get_enterprise_agent(agent_id):
        agent = registry.get_enterprise_agent(agent_id)
        agent_type = "Enterprise"
    elif registry.get_deep_agent(agent_id):
        agent = registry.get_deep_agent(agent_id)
        agent_type = "DeepAgent"

    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found"
        )

    try:
        response_text = agent.chat(request.message, thread_id=conversation_id)
        suggestions = _generate_suggestions(agent_id, response_text)

        return CopilotChatResponse(
            response=response_text,
            conversationId=conversation_id,
            agentId=agent_id,
            agentType=agent_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            suggestions=suggestions,
        )

    except Exception as e:
        # Log full details internally
        logger.error(f"Error in Copilot chat with {agent_id}: {e}", exc_info=True)
        # Return generic error
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again or contact support."
        )


# ============================================================================
# Document Processing Endpoints
# ============================================================================

# Document processing pipeline (lazy loaded)
_document_pipeline = None


def _get_document_pipeline():
    """Get or create the document processing pipeline."""
    global _document_pipeline
    if _document_pipeline is None:
        try:
            from langchain_azure_ai.document_processing import (
                MultiModalDocumentPipeline,
                ProcessingConfig,
            )
            _document_pipeline = MultiModalDocumentPipeline(
                config=ProcessingConfig(
                    extract_tables=True,
                    extract_images=False,  # Disable for faster processing
                    generate_summaries=True,
                    chunk_size=2000,
                )
            )
            logger.info("Document processing pipeline initialized")
        except ImportError as e:
            logger.warning(f"Document processing not available: {e}")
    return _document_pipeline


async def _process_document_content(
    content: Optional[str],
    url: Optional[str],
    filename: Optional[str],
    operation: str,
    options: Optional[Dict[str, Any]],
    telemetry: Optional["AgentTelemetry"],
) -> Dict[str, Any]:
    """Process document and return results.

    Args:
        content: Base64-encoded document content.
        url: URL to fetch document from.
        filename: Original filename for type detection.
        operation: Processing operation to perform.
        options: Additional processing options.
        telemetry: Telemetry instance for tracking.

    Returns:
        Dictionary with processing results.
    """
    registry = get_registry()
    start_time = time.time()
    result_metadata: Dict[str, Any] = {}

    # Get document intelligence agent for AI-powered operations
    doc_agent = registry.get_enterprise_agent("document_intelligence")

    try:
        # Determine source type and prepare document
        temp_file_path = None
        source_type = "local"

        if content:
            # Decode base64 content to temp file
            decoded = base64.b64decode(content)
            suffix = Path(filename).suffix if filename else ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(decoded)
                temp_file_path = tmp.name
            source_type = "local"
            result_metadata["source"] = "base64_upload"
            result_metadata["document_size_bytes"] = len(decoded)

        elif url:
            # Use URL directly
            temp_file_path = url
            source_type = "url"
            result_metadata["source"] = "url"

        else:
            return {
                "error": "No document content provided. Supply either documentContent (base64) or documentUrl.",
                "success": False,
            }

        # Perform operation based on type
        response_text = ""

        if operation == "summarize":
            # Use Document Intelligence agent for AI summarization
            if doc_agent:
                prompt = f"Please summarize this document concisely. "
                if options:
                    if options.get("max_length"):
                        prompt += f"Keep the summary under {options['max_length']} words. "
                    if options.get("format") == "bullet_points":
                        prompt += "Use bullet points. "
                    if options.get("focus"):
                        prompt += f"Focus on: {options['focus']}. "

                # Process with agent
                if temp_file_path and source_type == "local":
                    # Upload file to agent first if it supports it
                    if hasattr(doc_agent, "process_document"):
                        doc_result = doc_agent.process_document(
                            temp_file_path,
                            operation="summarize",
                            options=options,
                        )
                        response_text = doc_result.get("summary", doc_result.get("text", ""))
                    else:
                        # Fallback: read and send content
                        with open(temp_file_path, "rb") as f:
                            file_content = f.read()
                        response_text = doc_agent.chat(
                            f"{prompt}\n\n[Document: {filename or 'uploaded document'}]"
                        )
                else:
                    response_text = doc_agent.chat(
                        f"{prompt}\n\nDocument URL: {url}"
                    )
            else:
                # Fallback to pipeline
                pipeline = _get_document_pipeline()
                if pipeline:
                    processed = await pipeline.process(temp_file_path, source_type=source_type)
                    response_text = _generate_summary(processed.text_content, options)
                    result_metadata["pages_processed"] = processed.layout.page_count if processed.layout else 1
                    result_metadata["tables_found"] = len(processed.tables)
                else:
                    return {"error": "Document processing not available", "success": False}

        elif operation == "extract_text":
            pipeline = _get_document_pipeline()
            if pipeline:
                processed = await pipeline.process(temp_file_path, source_type=source_type)
                response_text = processed.text_content
                result_metadata["text_length"] = len(response_text)
            else:
                return {"error": "Document processing not available", "success": False}

        elif operation == "extract_tables":
            pipeline = _get_document_pipeline()
            if pipeline:
                processed = await pipeline.process(temp_file_path, source_type=source_type)
                tables_markdown = "\n\n".join([t.markdown for t in processed.tables])
                response_text = tables_markdown or "No tables found in the document."
                result_metadata["tables_found"] = len(processed.tables)
            else:
                return {"error": "Document processing not available", "success": False}

        elif operation == "extract_key_values":
            pipeline = _get_document_pipeline()
            if pipeline:
                processed = await pipeline.process(temp_file_path, source_type=source_type)
                if processed.key_value_pairs:
                    kv_text = "\n".join([f"- **{k}**: {v}" for k, v in processed.key_value_pairs])
                    response_text = kv_text
                else:
                    response_text = "No key-value pairs found in the document."
                result_metadata["pairs_found"] = len(processed.key_value_pairs)
            else:
                return {"error": "Document processing not available", "success": False}

        elif operation == "analyze":
            # Comprehensive analysis using Document Intelligence agent
            if doc_agent:
                response_text = doc_agent.chat(
                    f"Analyze this document comprehensively. Extract key insights, "
                    f"main topics, and important data points.\n\n"
                    f"[Document: {filename or url or 'uploaded document'}]"
                )
            else:
                pipeline = _get_document_pipeline()
                if pipeline:
                    processed = await pipeline.process(temp_file_path, source_type=source_type)
                    response_text = _generate_analysis(processed, options)
                    result_metadata["pages_processed"] = processed.layout.page_count if processed.layout else 1
                else:
                    return {"error": "Document processing not available", "success": False}

        else:
            return {
                "error": f"Unknown operation: {operation}. Supported: summarize, extract_text, extract_tables, extract_key_values, analyze",
                "success": False,
            }

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        result_metadata["processing_time_ms"] = round(processing_time_ms, 2)

        # Record telemetry
        if telemetry:
            telemetry.record_custom_metric("document_processing_time_ms", processing_time_ms)
            telemetry.record_custom_metric("document_operation", operation)

        return {
            "response": response_text,
            "success": True,
            "metadata": result_metadata,
        }

    except Exception as e:
        logger.error(f"Document processing error: {e}", exc_info=True)
        if telemetry:
            telemetry.record_error(str(e))
        return {
            "error": f"Failed to process document: {str(e)}",
            "success": False,
        }

    finally:
        # Clean up temp file
        if temp_file_path and source_type == "local" and content:
            try:
                Path(temp_file_path).unlink(missing_ok=True)
            except Exception:
                pass


def _generate_summary(text: str, options: Optional[Dict[str, Any]] = None) -> str:
    """Generate a basic summary from extracted text."""
    if not text:
        return "No text content found in the document."

    max_length = 500
    if options and options.get("max_length"):
        max_length = options["max_length"]

    # Simple extractive summary - take first paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    summary_parts = []
    current_length = 0

    for para in paragraphs[:10]:  # Consider first 10 paragraphs
        if current_length + len(para.split()) > max_length:
            break
        summary_parts.append(para)
        current_length += len(para.split())

    if not summary_parts:
        # Fallback: truncate text
        words = text.split()[:max_length]
        return " ".join(words) + "..."

    return "\n\n".join(summary_parts)


def _generate_analysis(processed_doc, options: Optional[Dict[str, Any]] = None) -> str:
    """Generate comprehensive analysis of processed document."""
    analysis_parts = ["## Document Analysis\n"]

    # Document overview
    if processed_doc.layout:
        analysis_parts.append(f"**Pages:** {processed_doc.layout.page_count}")

    # Text summary
    text_length = len(processed_doc.text_content) if processed_doc.text_content else 0
    analysis_parts.append(f"**Text Length:** {text_length:,} characters")

    # Tables
    if processed_doc.tables:
        analysis_parts.append(f"\n### Tables Found: {len(processed_doc.tables)}")
        for i, table in enumerate(processed_doc.tables[:3], 1):  # Show first 3
            analysis_parts.append(f"\n**Table {i}:**\n{table.markdown[:500]}")

    # Key-value pairs
    if processed_doc.key_value_pairs:
        analysis_parts.append(f"\n### Key Information: {len(processed_doc.key_value_pairs)} items")
        for k, v in processed_doc.key_value_pairs[:10]:  # Show first 10
            analysis_parts.append(f"- **{k}:** {v}")

    # Content preview
    if processed_doc.text_content:
        preview = processed_doc.text_content[:1000]
        analysis_parts.append(f"\n### Content Preview:\n{preview}...")

    return "\n".join(analysis_parts)


@router.post(
    "/document",
    response_model=CopilotDocumentResponse,
    summary="Process Document",
    description=(
        "Process a document for summarization, text extraction, table extraction, "
        "or comprehensive analysis. Supports PDF, Word, PowerPoint, and images. "
        "Send document as base64-encoded content or provide a URL."
    ),
)
async def copilot_process_document(
    request: CopilotDocumentRequest,
    auth: str = Depends(verify_api_key),
):
    """Process a document and return the requested analysis.

    This endpoint supports:
    - Document summarization (AI-powered)
    - Text extraction (full text content)
    - Table extraction (markdown format)
    - Key-value pair extraction
    - Comprehensive analysis

    Documents can be provided as:
    - Base64-encoded content (documentContent)
    - URL to publicly accessible document (documentUrl)
    """
    conversation_id = request.conversationId or str(uuid.uuid4())

    # Initialize telemetry with dual tracking
    telemetry = get_copilot_telemetry(f"copilot-document-{request.operation}")

    # Enable LangSmith tracing if configured
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    if langsmith_enabled:
        logger.debug("LangSmith tracing enabled for document processing")

    try:
        # Track with telemetry (Azure App Insights)
        if telemetry:
            with telemetry.track_execution() as span:
                if span:
                    span.set_attribute("copilot.conversation_id", conversation_id)
                    span.set_attribute("copilot.operation", request.operation)
                    span.set_attribute("copilot.document_name", request.documentName or "unknown")
                    span.set_attribute("copilot.source", "base64" if request.documentContent else "url")

                result = await _process_document_content(
                    content=request.documentContent,
                    url=request.documentUrl,
                    filename=request.documentName,
                    operation=request.operation,
                    options=request.options,
                    telemetry=telemetry,
                )
        else:
            result = await _process_document_content(
                content=request.documentContent,
                url=request.documentUrl,
                filename=request.documentName,
                operation=request.operation,
                options=request.options,
                telemetry=None,
            )

        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Document processing failed")
            )

        # Generate suggestions based on operation
        suggestions = _get_document_suggestions(request.operation)

        logger.info(
            f"Copilot document processing completed: operation={request.operation}, "
            f"conversation={conversation_id}, "
            f"time_ms={result.get('metadata', {}).get('processing_time_ms', 0)}"
        )

        return CopilotDocumentResponse(
            response=result["response"],
            conversationId=conversation_id,
            operation=request.operation,
            documentName=request.documentName,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=result.get("metadata"),
            suggestions=suggestions,
        )

    except HTTPException:
        raise
    except Exception as e:
        if telemetry:
            telemetry.record_error(str(e))
        logger.error(f"Error in Copilot document processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your document. Please try again."
        )


def _get_document_suggestions(operation: str) -> List[str]:
    """Get follow-up suggestions based on document operation."""
    suggestions_map = {
        "summarize": [
            "Extract all tables from this document",
            "Get key information and data points",
            "Analyze document in detail",
        ],
        "extract_text": [
            "Summarize the extracted text",
            "Find specific information in the text",
            "Extract tables separately",
        ],
        "extract_tables": [
            "Summarize the document content",
            "Analyze the data in these tables",
            "Extract key-value pairs",
        ],
        "extract_key_values": [
            "Summarize the document",
            "Extract full text content",
            "Get detailed analysis",
        ],
        "analyze": [
            "Get a shorter summary",
            "Extract specific sections",
            "Ask questions about the content",
        ],
    }
    return suggestions_map.get(operation, ["Process another document", "Ask a question"])


# ============================================================================
# Agent Discovery Endpoints
# ============================================================================

@router.get(
    "/agents",
    response_model=AgentListResponse,
    summary="List Available Agents",
    description="Get a list of all available agents with their capabilities.",
)
async def list_copilot_agents(auth: str = Depends(verify_api_key)):
    """List all agents available for Copilot Studio."""
    registry = get_registry()

    agents = []

    # IT Agents
    it_agent_configs = {
        "helpdesk": {
            "name": "IT Helpdesk",
            "description": "General IT support for password resets, software issues, hardware problems, and technical questions.",
            "triggerPhrases": ["IT help", "password reset", "computer issue", "tech support", "VPN problem"]
        },
        "servicenow": {
            "name": "ServiceNow Integration",
            "description": "Create and manage ServiceNow tickets, incidents, and service requests.",
            "triggerPhrases": ["create ticket", "ServiceNow", "incident", "service request", "ticket status"]
        },
        "hitl_support": {
            "name": "Human Escalation",
            "description": "Escalate complex issues to human support agents.",
            "triggerPhrases": ["talk to human", "escalate", "agent transfer", "supervisor"]
        },
    }

    for agent_id, config in it_agent_configs.items():
        if registry.get_it_agent(agent_id):
            agents.append(AgentCapability(
                id=agent_id,
                name=config["name"],
                description=config["description"],
                type="IT",
                capabilities=["password_reset", "software_request", "ticket_status"],
                triggerPhrases=config["triggerPhrases"]
            ))

    # Enterprise Agents
    enterprise_agent_configs = {
        "research": {
            "name": "Research Assistant",
            "description": "Research topics, gather information, and provide analysis.",
            "triggerPhrases": ["research", "find information", "look up", "analyze topic"]
        },
        "content": {
            "name": "Content Creator",
            "description": "Create, edit, and improve written content, documents, and communications.",
            "triggerPhrases": ["write content", "create document", "edit text", "improve writing"]
        },
        "data_analyst": {
            "name": "Data Analyst",
            "description": "Analyze data, create visualizations, and provide statistical insights.",
            "triggerPhrases": ["analyze data", "create chart", "statistics", "data insights"]
        },
        "code_assistant": {
            "name": "Code Assistant",
            "description": "Help with coding, debugging, code review, and programming questions.",
            "triggerPhrases": ["write code", "debug", "programming help", "code review"]
        },
    }

    for agent_id, config in enterprise_agent_configs.items():
        if registry.get_enterprise_agent(agent_id):
            agents.append(AgentCapability(
                id=agent_id,
                name=config["name"],
                description=config["description"],
                type="Enterprise",
                capabilities=["content_generation", "data_analysis", "code_review"],
                triggerPhrases=config["triggerPhrases"]
            ))

    # DeepAgents
    deep_agent_configs = {
        "it_operations": {
            "name": "IT Operations",
            "description": "Comprehensive IT operations management including incidents, changes, and infrastructure.",
            "triggerPhrases": ["system status", "infrastructure", "deployment", "monitoring"]
        },
        "sales_intelligence": {
            "name": "Sales Intelligence",
            "description": "Sales analytics, deal qualification, competitive analysis, and pipeline insights.",
            "triggerPhrases": ["sales data", "deal analysis", "pipeline", "customer insights"]
        },
        "recruitment": {
            "name": "Recruitment Assistant",
            "description": "Support hiring processes including resume screening, interview preparation, and candidate evaluation.",
            "triggerPhrases": ["candidates", "resume", "interview questions", "hiring"]
        },
        "software_development": {
            "name": "Software Development",
            "description": "Full software development lifecycle support from requirements to deployment.",
            "triggerPhrases": ["requirements", "architecture", "testing", "deployment", "SDLC"]
        },
    }

    for agent_id, config in deep_agent_configs.items():
        if registry.get_deep_agent(agent_id):
            agents.append(AgentCapability(
                id=agent_id,
                name=config["name"],
                description=config["description"],
                type="DeepAgent",
                capabilities=["requirements", "design", "coding", "testing", "deployment"],
                triggerPhrases=config["triggerPhrases"]
            ))

    return AgentListResponse(agents=agents, total=len(agents))


# ============================================================================
# Health & Monitoring
# ============================================================================

@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Health Check",
    description="Check the health status of the Copilot Studio integration.",
)
async def copilot_health():
    """Health check endpoint for monitoring."""
    try:
        registry = get_registry()
        agents_count = registry.total_agents
    except Exception:
        agents_count = 0

    return HealthStatus(
        status="healthy" if agents_count > 0 else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="2.0.0",
        agents_available=agents_count,
        copilot_integration=True,
        authentication="api_key" if security_config.api_key_enabled else "none",
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _route_to_agent(message: str, registry) -> tuple:
    """Route a message to the most appropriate agent.

    Uses keyword matching to determine the best agent for the request.
    Returns (agent_id, agent_type) tuple.
    """
    message_lower = message.lower()

    # IT-related keywords
    it_keywords = {
        "helpdesk": ["password", "reset", "vpn", "wifi", "internet", "computer", "laptop",
                     "email", "outlook", "teams", "software", "install", "access", "login", "account"],
        "servicenow": ["ticket", "incident", "service request", "servicenow", "change request",
                       "priority", "escalate ticket"],
    }

    # Enterprise keywords
    enterprise_keywords = {
        "research": ["research", "find out", "look up", "investigate", "analyze information",
                     "what is", "tell me about", "learn about"],
        "content": ["write", "create document", "draft", "edit", "proofread", "summarize",
                    "blog post", "email draft", "report"],
        "data_analyst": ["data", "statistics", "chart", "graph", "analyze numbers",
                         "trends", "metrics", "dashboard", "excel"],
        "code_assistant": ["code", "programming", "debug", "function", "api", "script",
                          "python", "javascript", "error", "compile"],
    }

    # DeepAgent keywords
    deep_keywords = {
        "it_operations": ["infrastructure", "deployment", "server", "monitoring", "incident management",
                          "change management", "cmdb", "sla"],
        "sales_intelligence": ["sales", "deal", "pipeline", "revenue", "forecast", "customer",
                               "competitive", "rfp", "proposal"],
        "recruitment": ["hire", "candidate", "resume", "interview", "job", "recruiting",
                        "screening", "position"],
        "software_development": ["requirements", "architecture", "testing", "sprint", "user story",
                                 "ci/cd", "devops"],
    }

    # Check IT agents first
    for agent_id, keywords in it_keywords.items():
        if registry.get_it_agent(agent_id):
            if any(kw in message_lower for kw in keywords):
                return (agent_id, "IT")

    # Check Enterprise agents
    for agent_id, keywords in enterprise_keywords.items():
        if registry.get_enterprise_agent(agent_id):
            if any(kw in message_lower for kw in keywords):
                return (agent_id, "Enterprise")

    # Check DeepAgents
    for agent_id, keywords in deep_keywords.items():
        if registry.get_deep_agent(agent_id):
            if any(kw in message_lower for kw in keywords):
                return (agent_id, "DeepAgent")

    # Default to helpdesk
    return ("helpdesk", "IT")


def _generate_suggestions(agent_id: str, response: str) -> List[str]:
    """Generate follow-up suggestions based on agent and response.

    Returns a list of suggested follow-up questions or actions.
    """
    suggestions_map = {
        "helpdesk": [
            "Check my ticket status",
            "Other IT issues",
            "Talk to human support"
        ],
        "servicenow": [
            "Create another ticket",
            "View my open tickets",
            "Check ticket priority"
        ],
        "research": [
            "Dig deeper into this topic",
            "Find related information",
            "Generate a summary report"
        ],
        "content": [
            "Revise this content",
            "Create a different version",
            "Make it more concise"
        ],
        "data_analyst": [
            "Visualize this data",
            "Perform another analysis",
            "Export results"
        ],
        "code_assistant": [
            "Explain this code",
            "Optimize performance",
            "Add tests"
        ],
        "it_operations": [
            "Check system status",
            "View recent incidents",
            "Schedule maintenance"
        ],
        "sales_intelligence": [
            "Analyze another deal",
            "Get competitive insights",
            "Generate proposal"
        ],
        "recruitment": [
            "Screen more candidates",
            "Generate interview questions",
            "Create job posting"
        ],
        "software_development": [
            "Create user stories",
            "Review code quality",
            "Plan next sprint"
        ],
    }

    return suggestions_map.get(agent_id, ["Ask another question", "Get more help"])


# ============================================================================
# Well-Known Endpoint (for plugin discovery)
# ============================================================================

well_known_router = APIRouter(tags=["discovery"])


@well_known_router.get("/.well-known/ai-plugin.json")
async def well_known_plugin(request: Request):
    """Serve the ai-plugin.json at the well-known location.

    This enables automatic plugin discovery by Microsoft 365 Copilot.
    """
    return await get_plugin_manifest(request)
