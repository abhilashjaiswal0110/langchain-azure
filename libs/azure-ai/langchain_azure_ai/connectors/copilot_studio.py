"""Microsoft Copilot Studio Connector for exporting agents to Microsoft 365 Copilot.

This module provides the ability to:
- Export LangChain agents as Copilot Studio custom connectors
- Create agent manifests compatible with Microsoft 365 Copilot
- Define topics and actions for Copilot interactions
- Manage Copilot Studio bot configurations

References:
- https://learn.microsoft.com/en-us/microsoft-copilot-studio/
- https://learn.microsoft.com/en-us/connectors/custom-connectors/
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Types of Copilot actions."""
    
    INVOKE = "invoke"
    STREAM = "stream"
    UPLOAD = "upload"
    WORKFLOW = "workflow"


class TopicTrigger(str, Enum):
    """Trigger types for Copilot topics."""
    
    PHRASE = "phrase"
    EVENT = "event"
    REDIRECT = "redirect"
    ACTIVITY = "activity"


@dataclass
class CopilotAction:
    """Represents an action that can be invoked from Copilot.
    
    Attributes:
        name: Unique action name
        display_name: Human-readable name shown in Copilot
        description: Detailed description for LLM understanding
        operation_id: API operation identifier
        method: HTTP method (POST, GET, etc.)
        path: API endpoint path
        parameters: Action parameters schema
        response_schema: Expected response schema
    """
    
    name: str
    display_name: str
    description: str
    operation_id: str
    method: str = "POST"
    path: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    response_schema: Dict[str, Any] = field(default_factory=dict)
    action_type: ActionType = ActionType.INVOKE
    
    def to_openapi_operation(self) -> Dict[str, Any]:
        """Convert to OpenAPI operation spec."""
        return {
            "operationId": self.operation_id,
            "summary": self.display_name,
            "description": self.description,
            "parameters": self._build_parameters(),
            "requestBody": self._build_request_body(),
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": self.response_schema or {"type": "object"}
                        }
                    }
                }
            }
        }
    
    def _build_parameters(self) -> List[Dict[str, Any]]:
        """Build OpenAPI parameters from action parameters."""
        params = []
        for name, schema in self.parameters.items():
            if schema.get("in") == "path":
                params.append({
                    "name": name,
                    "in": "path",
                    "required": True,
                    "schema": {"type": schema.get("type", "string")}
                })
        return params
    
    def _build_request_body(self) -> Dict[str, Any]:
        """Build OpenAPI request body spec."""
        body_params = {
            k: v for k, v in self.parameters.items()
            if v.get("in") != "path"
        }
        
        if not body_params:
            return {}
        
        properties = {}
        required = []
        
        for name, schema in body_params.items():
            properties[name] = {
                "type": schema.get("type", "string"),
                "description": schema.get("description", ""),
            }
            if schema.get("required", False):
                required.append(name)
        
        return {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
        }


@dataclass
class CopilotTopic:
    """Represents a Copilot topic for conversation flow.
    
    Topics define how Copilot responds to specific triggers and
    orchestrate the conversation flow.
    """
    
    name: str
    display_name: str
    description: str
    trigger_type: TopicTrigger = TopicTrigger.PHRASE
    trigger_phrases: List[str] = field(default_factory=list)
    actions: List[CopilotAction] = field(default_factory=list)
    system_prompt: str = ""
    
    def to_manifest(self) -> Dict[str, Any]:
        """Convert topic to manifest format."""
        return {
            "name": self.name,
            "displayName": self.display_name,
            "description": self.description,
            "trigger": {
                "type": self.trigger_type.value,
                "phrases": self.trigger_phrases
            },
            "actions": [a.operation_id for a in self.actions],
            "systemPrompt": self.system_prompt
        }


@dataclass
class AgentManifest:
    """Manifest describing an agent for Copilot Studio.
    
    This manifest is used to register agents with Copilot Studio
    and define their capabilities.
    """
    
    id: str
    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    publisher: str = "Azure AI Foundry"
    icon_url: str = ""
    privacy_url: str = ""
    terms_url: str = ""
    actions: List[CopilotAction] = field(default_factory=list)
    topics: List[CopilotTopic] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "$schema": "https://developer.microsoft.com/json-schemas/copilot/plugin/v2.1/schema.json",
            "schema_version": "v2.1",
            "name_for_human": self.display_name,
            "name_for_model": self.name,
            "description_for_human": self.description,
            "description_for_model": f"Agent: {self.description}. Use this plugin to interact with {self.display_name}.",
            "api": {
                "type": "openapi",
                "url": f"/api/agents/{self.id}/openapi.json"
            },
            "auth": {
                "type": "none"  # Will be configured per deployment
            },
            "logo_url": self.icon_url or f"/api/agents/{self.id}/icon.png",
            "contact_email": "support@azure.microsoft.com",
            "legal_info_url": self.terms_url,
            "capabilities": {
                "conversation": True,
                "streaming": True,
                "file_upload": "file_upload" in self.capabilities,
                "multi_turn": True
            }
        }
    
    def to_openapi_spec(self, base_url: str = "") -> Dict[str, Any]:
        """Generate OpenAPI specification for the agent."""
        paths = {}
        
        for action in self.actions:
            path = action.path or f"/api/agents/{self.id}/{action.operation_id}"
            if path not in paths:
                paths[path] = {}
            paths[path][action.method.lower()] = action.to_openapi_operation()
        
        return {
            "openapi": "3.0.0",
            "info": {
                "title": self.display_name,
                "description": self.description,
                "version": self.version
            },
            "servers": [
                {"url": base_url or "https://api.azure.com"}
            ],
            "paths": paths,
            "components": {
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer"
                    }
                }
            }
        }
    
    def save(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Save manifest and OpenAPI spec to files.
        
        Returns:
            Dictionary with paths to created files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Save plugin manifest
        manifest_path = output_dir / "ai-plugin.json"
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        files["manifest"] = manifest_path
        
        # Save OpenAPI spec
        openapi_path = output_dir / "openapi.json"
        with open(openapi_path, "w") as f:
            json.dump(self.to_openapi_spec(), f, indent=2)
        files["openapi"] = openapi_path
        
        # Save topics
        if self.topics:
            topics_path = output_dir / "topics.json"
            with open(topics_path, "w") as f:
                json.dump([t.to_manifest() for t in self.topics], f, indent=2)
            files["topics"] = topics_path
        
        logger.info(f"Saved agent manifest to {output_dir}")
        return files


@dataclass
class CopilotStudioConfig:
    """Configuration for Copilot Studio connection.
    
    Attributes:
        environment_id: Power Platform environment ID
        bot_id: Copilot Studio bot ID (optional, for existing bots)
        tenant_id: Azure AD tenant ID
        client_id: App registration client ID
        client_secret: App registration client secret
        region: Copilot Studio region (default: "unitedstates")
    """
    
    environment_id: str
    bot_id: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    region: str = "unitedstates"
    
    @classmethod
    def from_env(cls) -> "CopilotStudioConfig":
        """Load configuration from environment variables."""
        return cls(
            environment_id=os.getenv("COPILOT_ENVIRONMENT_ID", ""),
            bot_id=os.getenv("COPILOT_BOT_ID"),
            tenant_id=os.getenv("AZURE_TENANT_ID"),
            client_id=os.getenv("COPILOT_CLIENT_ID"),
            client_secret=os.getenv("COPILOT_CLIENT_SECRET"),
            region=os.getenv("COPILOT_REGION", "unitedstates"),
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        if not self.environment_id:
            issues.append("COPILOT_ENVIRONMENT_ID is required")
        if not self.tenant_id:
            issues.append("AZURE_TENANT_ID is required for authentication")
        if not self.client_id:
            issues.append("COPILOT_CLIENT_ID is required for authentication")
        return issues


class CopilotStudioConnector:
    """Connector for exporting agents to Microsoft Copilot Studio.
    
    This connector enables:
    - Exporting LangChain agents as Copilot Studio custom connectors
    - Creating agent manifests for Microsoft 365 Copilot
    - Managing Copilot Studio bot configurations
    - Syncing agent updates to Copilot Studio
    
    Example:
        >>> config = CopilotStudioConfig.from_env()
        >>> connector = CopilotStudioConnector(config)
        >>> 
        >>> # Export a wrapper as a Copilot connector
        >>> manifest = connector.export_agent(
        ...     wrapper=my_agent_wrapper,
        ...     name="my-agent",
        ...     description="My helpful agent"
        ... )
        >>> 
        >>> # Save manifest files
        >>> manifest.save("./copilot-export")
    """
    
    def __init__(self, config: CopilotStudioConfig):
        """Initialize the Copilot Studio connector.
        
        Args:
            config: Copilot Studio configuration
        """
        self.config = config
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        
        # Validate config
        issues = config.validate()
        if issues:
            logger.warning(f"Configuration issues: {issues}")
    
    def export_agent(
        self,
        wrapper: Any,
        name: str,
        description: str = "",
        display_name: str = "",
        include_streaming: bool = True,
        include_upload: bool = False,
        custom_topics: Optional[List[CopilotTopic]] = None,
    ) -> AgentManifest:
        """Export a wrapper as a Copilot Studio agent manifest.
        
        Args:
            wrapper: Agent wrapper instance
            name: Agent name (used as identifier)
            description: Agent description
            display_name: Human-readable display name
            include_streaming: Include streaming action
            include_upload: Include file upload action
            custom_topics: Custom topics for conversation flow
        
        Returns:
            AgentManifest ready for export
        """
        agent_id = str(uuid.uuid4())
        display_name = display_name or name.replace("-", " ").title()
        
        # Extract agent metadata
        agent_type = getattr(wrapper, "agent_type", "CUSTOM").name
        agent_subtype = getattr(wrapper, "agent_subtype", "general")
        instructions = getattr(wrapper, "instructions", description)
        
        if not description:
            description = instructions[:500] if instructions else f"A {agent_subtype} agent"
        
        # Build actions
        actions = self._build_actions(
            agent_id=agent_id,
            agent_name=name,
            agent_type=agent_type,
            include_streaming=include_streaming,
            include_upload=include_upload,
        )
        
        # Build topics
        topics = custom_topics or self._build_default_topics(
            agent_name=name,
            agent_subtype=agent_subtype,
            actions=actions,
        )
        
        # Determine capabilities
        capabilities = ["conversation", "multi_turn"]
        if include_streaming:
            capabilities.append("streaming")
        if include_upload:
            capabilities.append("file_upload")
        
        manifest = AgentManifest(
            id=agent_id,
            name=name,
            display_name=display_name,
            description=description,
            actions=actions,
            topics=topics,
            capabilities=capabilities,
        )
        
        logger.info(f"Created manifest for agent: {name}")
        return manifest
    
    def _build_actions(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        include_streaming: bool,
        include_upload: bool,
    ) -> List[CopilotAction]:
        """Build Copilot actions for the agent."""
        actions = []
        
        # Chat action (always included)
        chat_action = CopilotAction(
            name="chat",
            display_name="Chat",
            description=f"Send a message to the {agent_name} agent and get a response",
            operation_id=f"{agent_name}_chat",
            method="POST",
            path=f"/api/agents/{agent_id}/chat",
            parameters={
                "message": {
                    "type": "string",
                    "description": "The user message to send",
                    "required": True
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID for conversation continuity",
                    "required": False
                }
            },
            response_schema={
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "session_id": {"type": "string"},
                    "agent_type": {"type": "string"}
                }
            }
        )
        actions.append(chat_action)
        
        # Streaming action
        if include_streaming:
            stream_action = CopilotAction(
                name="stream",
                display_name="Stream Chat",
                description=f"Stream a response from the {agent_name} agent",
                operation_id=f"{agent_name}_stream",
                method="POST",
                path=f"/api/agents/{agent_id}/stream",
                action_type=ActionType.STREAM,
                parameters={
                    "message": {
                        "type": "string",
                        "description": "The user message to send",
                        "required": True
                    }
                },
                response_schema={
                    "type": "string",
                    "description": "Server-Sent Events stream"
                }
            )
            actions.append(stream_action)
        
        # Upload action
        if include_upload:
            upload_action = CopilotAction(
                name="upload",
                display_name="Upload File",
                description=f"Upload a file for the {agent_name} agent to analyze",
                operation_id=f"{agent_name}_upload",
                method="POST",
                path=f"/api/agents/{agent_id}/upload",
                action_type=ActionType.UPLOAD,
                parameters={
                    "file": {
                        "type": "string",
                        "format": "binary",
                        "description": "File to upload",
                        "required": True
                    }
                },
                response_schema={
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                        "analysis": {"type": "string"}
                    }
                }
            )
            actions.append(upload_action)
        
        return actions
    
    def _build_default_topics(
        self,
        agent_name: str,
        agent_subtype: str,
        actions: List[CopilotAction],
    ) -> List[CopilotTopic]:
        """Build default topics based on agent type."""
        topics = []
        
        # Main conversation topic
        main_topic = CopilotTopic(
            name=f"{agent_name}_main",
            display_name=f"Talk to {agent_name.replace('-', ' ').title()}",
            description=f"Start a conversation with the {agent_subtype} agent",
            trigger_type=TopicTrigger.PHRASE,
            trigger_phrases=self._get_trigger_phrases(agent_subtype),
            actions=actions,
            system_prompt=f"You are connected to the {agent_name} agent. Forward all user messages to this agent and relay the responses."
        )
        topics.append(main_topic)
        
        # Help topic
        help_topic = CopilotTopic(
            name=f"{agent_name}_help",
            display_name="Get Help",
            description=f"Learn about {agent_name}'s capabilities",
            trigger_type=TopicTrigger.PHRASE,
            trigger_phrases=["help", "what can you do", "capabilities"],
            actions=[],
            system_prompt=f"Explain the capabilities of the {agent_name} {agent_subtype} agent and how users can interact with it."
        )
        topics.append(help_topic)
        
        return topics
    
    def _get_trigger_phrases(self, agent_subtype: str) -> List[str]:
        """Get trigger phrases based on agent subtype."""
        phrases_map = {
            "helpdesk": ["IT help", "tech support", "computer issue", "password reset"],
            "servicenow": ["create ticket", "ServiceNow", "incident", "service request"],
            "hitl": ["escalate", "talk to human", "human support", "agent transfer"],
            "research": ["research", "find information", "look up", "analyze data"],
            "content": ["write content", "create document", "generate text", "edit content"],
            "data_analyst": ["analyze data", "create chart", "statistics", "data insights"],
            "code_assistant": ["write code", "debug", "programming help", "code review"],
            "it_operations": ["system status", "monitor", "infrastructure", "deployment"],
            "sales_intelligence": ["sales data", "customer insights", "revenue", "pipeline"],
            "recruitment": ["job posting", "candidates", "hiring", "interview"],
        }
        
        default = ["help me", "assist me", "I need help"]
        return phrases_map.get(agent_subtype, default)
    
    async def publish_to_copilot_studio(
        self,
        manifest: AgentManifest,
        base_url: str,
    ) -> Dict[str, Any]:
        """Publish an agent manifest to Copilot Studio.
        
        Note: This requires Copilot Studio API access and proper authentication.
        Currently exports files locally that can be manually imported.
        
        Args:
            manifest: Agent manifest to publish
            base_url: Base URL where the agent API is hosted
        
        Returns:
            Publication result with bot_id and status
        """
        # For now, we'll create the export package
        # Full Copilot Studio API integration requires Power Platform APIs
        
        logger.info(f"Publishing agent {manifest.name} to Copilot Studio...")
        
        export_dir = Path(f"./copilot-export/{manifest.name}")
        files = manifest.save(export_dir)
        
        # Create deployment instructions
        instructions = {
            "steps": [
                "1. Go to https://copilotstudio.microsoft.com",
                "2. Create a new custom connector or copilot",
                "3. Import the ai-plugin.json manifest",
                "4. Import the openapi.json specification",
                f"5. Configure the API URL: {base_url}",
                "6. Set up authentication as needed",
                "7. Test the connector and publish"
            ],
            "files": {k: str(v) for k, v in files.items()},
            "base_url": base_url
        }
        
        instructions_path = export_dir / "DEPLOYMENT_INSTRUCTIONS.json"
        with open(instructions_path, "w") as f:
            json.dump(instructions, f, indent=2)
        
        return {
            "status": "exported",
            "export_dir": str(export_dir),
            "files": list(files.keys()),
            "next_steps": "Import to Copilot Studio manually"
        }
    
    def create_m365_copilot_plugin(
        self,
        wrapper: Any,
        name: str,
        description: str = "",
        api_base_url: str = "",
    ) -> Dict[str, Any]:
        """Create a Microsoft 365 Copilot plugin package.
        
        This creates a plugin package that can be deployed to
        Microsoft 365 Copilot via the admin center.
        
        Args:
            wrapper: Agent wrapper instance
            name: Plugin name
            description: Plugin description
            api_base_url: Base URL for the API
        
        Returns:
            Dictionary with paths to plugin files
        """
        manifest = self.export_agent(
            wrapper=wrapper,
            name=name,
            description=description,
            include_streaming=True,
            include_upload=True,
        )
        
        # Create M365 Copilot plugin structure
        plugin_dir = Path(f"./m365-copilot-plugins/{name}")
        plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plugin manifest for M365
        m365_manifest = {
            "$schema": "https://developer.microsoft.com/json-schemas/copilot/plugin/v2.1/schema.json",
            "schema_version": "v2.1",
            "name_for_human": manifest.display_name,
            "name_for_model": name,
            "description_for_human": description or manifest.description,
            "description_for_model": f"Use this plugin to interact with {manifest.display_name}. {manifest.description}",
            "api": {
                "type": "openapi",
                "url": f"{api_base_url}/openapi.json" if api_base_url else "./openapi.json"
            },
            "auth": {
                "type": "oauth2",
                "authorization_url": f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/authorize",
                "token_url": f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token",
                "scope": "https://graph.microsoft.com/.default"
            },
            "logo_url": f"{api_base_url}/icon.png" if api_base_url else "./icon.png",
            "contact_email": "support@contoso.com",
            "legal_info_url": "https://contoso.com/legal"
        }
        
        manifest_path = plugin_dir / "ai-plugin.json"
        with open(manifest_path, "w") as f:
            json.dump(m365_manifest, f, indent=2)
        
        # Save OpenAPI spec
        openapi_path = plugin_dir / "openapi.json"
        with open(openapi_path, "w") as f:
            json.dump(manifest.to_openapi_spec(api_base_url), f, indent=2)
        
        # Create Teams app manifest for M365
        teams_manifest = {
            "$schema": "https://developer.microsoft.com/json-schemas/teams/v1.16/MicrosoftTeams.schema.json",
            "manifestVersion": "1.16",
            "version": "1.0.0",
            "id": manifest.id,
            "packageName": f"com.azure.{name.replace('-', '')}",
            "developer": {
                "name": "Azure AI Foundry",
                "websiteUrl": "https://azure.microsoft.com",
                "privacyUrl": "https://privacy.microsoft.com",
                "termsOfUseUrl": "https://azure.microsoft.com/terms"
            },
            "name": {
                "short": manifest.display_name[:30],
                "full": manifest.display_name
            },
            "description": {
                "short": description[:80] if description else manifest.description[:80],
                "full": manifest.description
            },
            "icons": {
                "outline": "outline.png",
                "color": "color.png"
            },
            "accentColor": "#0078D4",
            "copilotExtensions": {
                "plugins": [
                    {
                        "file": "ai-plugin.json",
                        "id": name
                    }
                ]
            }
        }
        
        teams_manifest_path = plugin_dir / "manifest.json"
        with open(teams_manifest_path, "w") as f:
            json.dump(teams_manifest, f, indent=2)
        
        logger.info(f"Created M365 Copilot plugin at {plugin_dir}")
        
        return {
            "plugin_dir": str(plugin_dir),
            "files": {
                "ai_plugin": str(manifest_path),
                "openapi": str(openapi_path),
                "teams_manifest": str(teams_manifest_path)
            },
            "deployment_steps": [
                "1. Package the plugin directory as a .zip file",
                "2. Go to Microsoft 365 admin center",
                "3. Navigate to Settings > Integrated apps",
                "4. Upload the plugin package",
                "5. Assign users and groups",
                "6. Test in Microsoft 365 Copilot"
            ]
        }
