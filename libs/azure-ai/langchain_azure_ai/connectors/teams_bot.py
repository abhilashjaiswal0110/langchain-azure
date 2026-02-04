"""Microsoft Teams Bot Connector for agent integration.

This module provides:
- Teams bot framework integration for LangChain agents
- Adaptive card support for rich responses
- Proactive messaging capabilities
- Multi-tenant bot support

References:
- https://learn.microsoft.com/en-us/microsoftteams/platform/bots/
- https://learn.microsoft.com/en-us/azure/bot-service/
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ActivityType(str, Enum):
    """Types of Teams activities."""
    
    MESSAGE = "message"
    CONVERSATION_UPDATE = "conversationUpdate"
    INVOKE = "invoke"
    EVENT = "event"
    MESSAGE_REACTION = "messageReaction"


class CardActionType(str, Enum):
    """Types of adaptive card actions."""
    
    OPEN_URL = "Action.OpenUrl"
    SUBMIT = "Action.Submit"
    SHOW_CARD = "Action.ShowCard"
    TOGGLE_VISIBILITY = "Action.ToggleVisibility"


@dataclass
class TeamsActivity:
    """Represents a Teams activity/message.
    
    Attributes:
        type: Activity type (message, event, etc.)
        id: Activity ID
        text: Message text content
        from_id: Sender ID
        from_name: Sender display name
        conversation_id: Conversation/thread ID
        channel_id: Teams channel ID
        service_url: Bot service URL
        timestamp: Activity timestamp
        attachments: List of attachments (adaptive cards, etc.)
    """
    
    type: ActivityType = ActivityType.MESSAGE
    id: str = ""
    text: str = ""
    from_id: str = ""
    from_name: str = ""
    conversation_id: str = ""
    channel_id: str = "msteams"
    service_url: str = ""
    timestamp: str = ""
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    value: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_request(cls, data: Dict[str, Any]) -> "TeamsActivity":
        """Create activity from incoming webhook request."""
        return cls(
            type=ActivityType(data.get("type", "message")),
            id=data.get("id", ""),
            text=data.get("text", ""),
            from_id=data.get("from", {}).get("id", ""),
            from_name=data.get("from", {}).get("name", ""),
            conversation_id=data.get("conversation", {}).get("id", ""),
            channel_id=data.get("channelId", "msteams"),
            service_url=data.get("serviceUrl", ""),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            attachments=data.get("attachments", []),
            entities=data.get("entities", []),
            value=data.get("value", {}),
        )
    
    def to_response(self, reply_text: str, attachments: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Create a response activity."""
        return {
            "type": "message",
            "text": reply_text,
            "from": {
                "id": "bot",
                "name": "Azure AI Agent"
            },
            "conversation": {
                "id": self.conversation_id
            },
            "replyToId": self.id,
            "attachments": attachments or [],
        }


@dataclass
class TeamsAdaptiveCard:
    """Builder for Teams Adaptive Cards.
    
    Adaptive Cards provide rich, interactive UI elements in Teams messages.
    """
    
    title: str = ""
    body: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    version: str = "1.4"
    
    def add_text(self, text: str, size: str = "default", weight: str = "default", wrap: bool = True) -> "TeamsAdaptiveCard":
        """Add a text block."""
        self.body.append({
            "type": "TextBlock",
            "text": text,
            "size": size,
            "weight": weight,
            "wrap": wrap
        })
        return self
    
    def add_header(self, text: str) -> "TeamsAdaptiveCard":
        """Add a header text block."""
        return self.add_text(text, size="large", weight="bolder")
    
    def add_fact_set(self, facts: Dict[str, str]) -> "TeamsAdaptiveCard":
        """Add a fact set (key-value pairs)."""
        self.body.append({
            "type": "FactSet",
            "facts": [{"title": k, "value": v} for k, v in facts.items()]
        })
        return self
    
    def add_image(self, url: str, alt_text: str = "", size: str = "auto") -> "TeamsAdaptiveCard":
        """Add an image."""
        self.body.append({
            "type": "Image",
            "url": url,
            "altText": alt_text,
            "size": size
        })
        return self
    
    def add_column_set(self, columns: List[Dict[str, Any]]) -> "TeamsAdaptiveCard":
        """Add a column set for layout."""
        self.body.append({
            "type": "ColumnSet",
            "columns": columns
        })
        return self
    
    def add_action_submit(self, title: str, data: Dict[str, Any]) -> "TeamsAdaptiveCard":
        """Add a submit action button."""
        self.actions.append({
            "type": CardActionType.SUBMIT.value,
            "title": title,
            "data": data
        })
        return self
    
    def add_action_url(self, title: str, url: str) -> "TeamsAdaptiveCard":
        """Add an open URL action button."""
        self.actions.append({
            "type": CardActionType.OPEN_URL.value,
            "title": title,
            "url": url
        })
        return self
    
    def add_input_text(self, id: str, placeholder: str = "", label: str = "", is_multiline: bool = False) -> "TeamsAdaptiveCard":
        """Add a text input field."""
        self.body.append({
            "type": "Input.Text",
            "id": id,
            "placeholder": placeholder,
            "label": label,
            "isMultiline": is_multiline
        })
        return self
    
    def add_input_choice(self, id: str, choices: List[Dict[str, str]], label: str = "") -> "TeamsAdaptiveCard":
        """Add a choice input (dropdown/radio)."""
        self.body.append({
            "type": "Input.ChoiceSet",
            "id": id,
            "label": label,
            "choices": choices
        })
        return self
    
    def to_attachment(self) -> Dict[str, Any]:
        """Convert to Teams attachment format."""
        return {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": self.version,
                "body": self.body,
                "actions": self.actions
            }
        }
    
    @classmethod
    def from_agent_response(cls, response: str, agent_name: str = "Agent") -> "TeamsAdaptiveCard":
        """Create a card from an agent response."""
        card = cls()
        card.add_header(f"🤖 {agent_name}")
        card.add_text(response)
        return card
    
    @classmethod
    def error_card(cls, error_message: str) -> "TeamsAdaptiveCard":
        """Create an error notification card."""
        card = cls()
        card.add_header("⚠️ Error")
        card.add_text(error_message)
        card.add_action_submit("Retry", {"action": "retry"})
        return card
    
    @classmethod
    def thinking_card(cls, message: str = "Processing your request...") -> "TeamsAdaptiveCard":
        """Create a 'thinking' indicator card."""
        card = cls()
        card.add_text("⏳ " + message)
        return card


@dataclass
class TeamsBotConfig:
    """Configuration for Teams Bot.
    
    Attributes:
        app_id: Microsoft App ID (from Azure Bot registration)
        app_password: Microsoft App Password
        tenant_id: Azure AD tenant ID (for single-tenant bots)
        endpoint: Bot messaging endpoint
        multi_tenant: Whether this is a multi-tenant bot
    """
    
    app_id: str
    app_password: str
    tenant_id: Optional[str] = None
    endpoint: str = "/api/messages"
    multi_tenant: bool = True
    
    @classmethod
    def from_env(cls) -> "TeamsBotConfig":
        """Load configuration from environment variables."""
        return cls(
            app_id=os.getenv("MICROSOFT_APP_ID", ""),
            app_password=os.getenv("MICROSOFT_APP_PASSWORD", ""),
            tenant_id=os.getenv("AZURE_TENANT_ID"),
            endpoint=os.getenv("BOT_ENDPOINT", "/api/messages"),
            multi_tenant=os.getenv("BOT_MULTI_TENANT", "true").lower() == "true",
        )
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        issues = []
        if not self.app_id:
            issues.append("MICROSOFT_APP_ID is required")
        if not self.app_password:
            issues.append("MICROSOFT_APP_PASSWORD is required")
        return issues


class TeamsBotConnector:
    """Connector for Microsoft Teams bot integration.
    
    This connector enables:
    - Receiving messages from Teams
    - Sending responses with rich formatting
    - Adaptive card interactions
    - Proactive messaging
    
    Example:
        >>> config = TeamsBotConfig.from_env()
        >>> connector = TeamsBotConnector(config)
        >>> 
        >>> # Register agent handlers
        >>> connector.register_agent("helpdesk", helpdesk_wrapper)
        >>> 
        >>> # Process incoming activity
        >>> response = await connector.process_activity(activity, "helpdesk")
    """
    
    def __init__(self, config: TeamsBotConfig):
        """Initialize the Teams bot connector.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        self._agents: Dict[str, Any] = {}
        self._conversation_refs: Dict[str, Dict[str, Any]] = {}
        
        issues = config.validate()
        if issues:
            logger.warning(f"Bot configuration issues: {issues}")
    
    def register_agent(self, name: str, wrapper: Any) -> None:
        """Register an agent wrapper for handling messages.
        
        Args:
            name: Agent name (used for routing)
            wrapper: Agent wrapper instance
        """
        self._agents[name] = wrapper
        logger.info(f"Registered Teams agent: {name}")
    
    def get_agent(self, name: str) -> Optional[Any]:
        """Get a registered agent by name."""
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())
    
    async def process_activity(
        self,
        activity: TeamsActivity,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process an incoming Teams activity.
        
        Args:
            activity: Incoming Teams activity
            agent_name: Specific agent to use (optional, auto-detects)
        
        Returns:
            Response activity data
        """
        # Store conversation reference for proactive messaging
        self._store_conversation_ref(activity)
        
        if activity.type == ActivityType.MESSAGE:
            return await self._handle_message(activity, agent_name)
        elif activity.type == ActivityType.CONVERSATION_UPDATE:
            return await self._handle_conversation_update(activity)
        elif activity.type == ActivityType.INVOKE:
            return await self._handle_invoke(activity)
        else:
            logger.debug(f"Unhandled activity type: {activity.type}")
            return {}
    
    async def _handle_message(
        self,
        activity: TeamsActivity,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle incoming message activity."""
        text = activity.text.strip()
        
        # Handle @mentions (remove bot mention)
        for entity in activity.entities:
            if entity.get("type") == "mention":
                mentioned = entity.get("mentioned", {})
                if mentioned.get("id") == self.config.app_id:
                    # Remove the mention from the text
                    mention_text = entity.get("text", "")
                    text = text.replace(mention_text, "").strip()
        
        # Detect agent from command or use default
        agent_name = agent_name or self._detect_agent(text)
        
        agent = self._agents.get(agent_name)
        if not agent:
            # Fall back to first registered agent
            if self._agents:
                agent_name = list(self._agents.keys())[0]
                agent = self._agents[agent_name]
            else:
                return activity.to_response(
                    "No agents are currently available. Please try again later.",
                    [TeamsAdaptiveCard.error_card("No agents available").to_attachment()]
                )
        
        try:
            # Get response from agent
            response = agent.chat(text, thread_id=activity.conversation_id)
            
            # Create adaptive card response
            card = TeamsAdaptiveCard.from_agent_response(
                response,
                agent_name=agent_name.replace("-", " ").title()
            )
            
            # Add helpful actions
            card.add_action_submit("Ask Follow-up", {"action": "followup", "agent": agent_name})
            card.add_action_submit("New Conversation", {"action": "new", "agent": agent_name})
            
            return activity.to_response(response, [card.to_attachment()])
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return activity.to_response(
                f"I encountered an error: {str(e)}",
                [TeamsAdaptiveCard.error_card(str(e)).to_attachment()]
            )
    
    async def _handle_conversation_update(self, activity: TeamsActivity) -> Dict[str, Any]:
        """Handle conversation update (member added/removed)."""
        # Send welcome message when bot is added
        welcome_card = TeamsAdaptiveCard()
        welcome_card.add_header("👋 Welcome to Azure AI Agents!")
        welcome_card.add_text("I can connect you with specialized AI agents for various tasks:")
        
        if self._agents:
            agents_list = "\n".join([f"• **{name}**" for name in self._agents.keys()])
            welcome_card.add_text(agents_list)
        
        welcome_card.add_text("\nJust type your question or say 'help' to learn more.")
        
        return activity.to_response(
            "Welcome! I'm your Azure AI Agent assistant.",
            [welcome_card.to_attachment()]
        )
    
    async def _handle_invoke(self, activity: TeamsActivity) -> Dict[str, Any]:
        """Handle invoke activities (card actions, etc.)."""
        action = activity.value.get("action")
        
        if action == "retry":
            # Retry last message
            return {"status": 200, "body": {"message": "Retrying..."}}
        elif action == "new":
            # Start new conversation
            return {"status": 200, "body": {"message": "Starting new conversation"}}
        elif action == "followup":
            # Follow-up action
            return {"status": 200, "body": {"message": "Ready for follow-up"}}
        
        return {"status": 200, "body": {}}
    
    def _detect_agent(self, text: str) -> str:
        """Detect which agent should handle the message."""
        text_lower = text.lower()
        
        # Check for explicit agent mentions
        for agent_name in self._agents.keys():
            if agent_name.lower() in text_lower:
                return agent_name
        
        # Keyword-based detection
        keyword_map = {
            "helpdesk": ["password", "reset", "computer", "laptop", "IT", "tech support"],
            "servicenow": ["ticket", "incident", "service request", "servicenow"],
            "research": ["research", "analyze", "find", "look up", "information"],
            "content": ["write", "create", "document", "content", "article"],
            "code": ["code", "programming", "debug", "function", "class"],
            "data": ["data", "chart", "statistics", "analysis", "metrics"],
        }
        
        for agent_prefix, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                # Find matching agent
                for agent_name in self._agents.keys():
                    if agent_prefix in agent_name.lower():
                        return agent_name
        
        # Default to first agent
        return list(self._agents.keys())[0] if self._agents else "default"
    
    def _store_conversation_ref(self, activity: TeamsActivity) -> None:
        """Store conversation reference for proactive messaging."""
        self._conversation_refs[activity.conversation_id] = {
            "conversation": {"id": activity.conversation_id},
            "serviceUrl": activity.service_url,
            "channelId": activity.channel_id,
        }
    
    async def send_proactive_message(
        self,
        conversation_id: str,
        message: str,
        card: Optional[TeamsAdaptiveCard] = None,
    ) -> bool:
        """Send a proactive message to a conversation.
        
        Args:
            conversation_id: Target conversation ID
            message: Message text
            card: Optional adaptive card
        
        Returns:
            True if message was sent successfully
        """
        conv_ref = self._conversation_refs.get(conversation_id)
        if not conv_ref:
            logger.warning(f"No conversation reference for: {conversation_id}")
            return False
        
        # Build proactive message
        proactive_activity = {
            "type": "message",
            "text": message,
            "conversation": conv_ref["conversation"],
            "channelId": conv_ref["channelId"],
        }
        
        if card:
            proactive_activity["attachments"] = [card.to_attachment()]
        
        # In production, this would use the Bot Framework SDK
        # to send via the serviceUrl with proper authentication
        logger.info(f"Proactive message queued for: {conversation_id}")
        
        return True
    
    def create_fastapi_routes(self) -> Any:
        """Create FastAPI routes for the bot endpoint.
        
        Returns:
            FastAPI router with bot endpoints
        """
        from fastapi import APIRouter, Request, Response
        
        router = APIRouter(tags=["teams-bot"])
        
        @router.post(self.config.endpoint)
        async def messages_endpoint(request: Request):
            """Handle incoming bot messages."""
            try:
                body = await request.json()
                activity = TeamsActivity.from_request(body)
                
                response = await self.process_activity(activity)
                
                return response
            except Exception as e:
                logger.error(f"Bot endpoint error: {e}")
                return {"error": str(e)}
        
        @router.get("/api/bot/health")
        async def bot_health():
            """Bot health check."""
            return {
                "status": "healthy",
                "agents": self.list_agents(),
                "conversations": len(self._conversation_refs)
            }
        
        return router
    
    def generate_manifest(self, base_url: str) -> Dict[str, Any]:
        """Generate Teams app manifest for the bot.
        
        Args:
            base_url: Base URL where the bot is hosted
        
        Returns:
            Teams app manifest dictionary
        """
        manifest = {
            "$schema": "https://developer.microsoft.com/json-schemas/teams/v1.16/MicrosoftTeams.schema.json",
            "manifestVersion": "1.16",
            "version": "1.0.0",
            "id": self.config.app_id or str(uuid.uuid4()),
            "packageName": "com.azure.aiagents.bot",
            "developer": {
                "name": "Azure AI Foundry",
                "websiteUrl": "https://azure.microsoft.com",
                "privacyUrl": "https://privacy.microsoft.com",
                "termsOfUseUrl": "https://azure.microsoft.com/terms"
            },
            "name": {
                "short": "Azure AI Agents",
                "full": "Azure AI Foundry Agent Bot"
            },
            "description": {
                "short": "Chat with AI agents for IT help, research, and more",
                "full": "Connect with specialized AI agents powered by Azure AI Foundry. Get help with IT issues, research, content creation, data analysis, and more."
            },
            "icons": {
                "outline": "outline.png",
                "color": "color.png"
            },
            "accentColor": "#0078D4",
            "bots": [
                {
                    "botId": self.config.app_id,
                    "scopes": ["personal", "team", "groupChat"],
                    "supportsFiles": True,
                    "isNotificationOnly": False,
                    "commandLists": [
                        {
                            "scopes": ["personal", "team", "groupChat"],
                            "commands": [
                                {"title": "help", "description": "Get help with available agents"},
                                {"title": "agents", "description": "List available agents"},
                            ] + [
                                {"title": name, "description": f"Chat with {name} agent"}
                                for name in self._agents.keys()
                            ]
                        }
                    ]
                }
            ],
            "permissions": ["identity", "messageTeamMembers"],
            "validDomains": [base_url.replace("https://", "").replace("http://", "").split("/")[0]]
        }
        
        return manifest
    
    def save_manifest(self, output_dir: Union[str, Path], base_url: str) -> Path:
        """Save Teams app manifest to a file.
        
        Args:
            output_dir: Directory to save manifest
            base_url: Base URL for the bot
        
        Returns:
            Path to saved manifest
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = self.generate_manifest(base_url)
        
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved Teams manifest to {manifest_path}")
        return manifest_path
