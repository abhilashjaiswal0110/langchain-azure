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

import json
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
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
            "Use this plugin to interact with enterprise AI agents. "
            "Available agent types: IT Helpdesk (password resets, technical support), "
            "ServiceNow (ticket creation, incident management), "
            "Research (information gathering, analysis), "
            "Content (document creation, editing), "
            "Data Analyst (data analysis, visualization), "
            "Code Assistant (code review, debugging), "
            "IT Operations (infrastructure monitoring), "
            "Sales Intelligence (deal analysis, forecasting), "
            "Recruitment (candidate screening, interview questions), "
            "Software Development (full SDLC support). "
            "Route user requests to the appropriate agent based on their needs."
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
