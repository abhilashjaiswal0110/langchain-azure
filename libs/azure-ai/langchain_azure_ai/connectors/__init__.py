"""Connectors for external platforms and services.

This module provides connectors for integrating LangChain agents with:
- Microsoft 365 Copilot (via Copilot Studio)
- Microsoft Teams bots
- Azure Functions serverless deployment

Example:
    >>> from langchain_azure_ai.connectors import CopilotStudioConnector
    >>> connector = CopilotStudioConnector(environment_id="...", bot_id="...")
    >>> connector.export_agent(wrapper, "my-agent")
"""

from langchain_azure_ai.connectors.copilot_studio import (
    CopilotStudioConnector,
    CopilotStudioConfig,
    AgentManifest,
    CopilotAction,
    CopilotTopic,
)

from langchain_azure_ai.connectors.teams_bot import (
    TeamsBotConnector,
    TeamsBotConfig,
    TeamsActivity,
    TeamsAdaptiveCard,
)

from langchain_azure_ai.connectors.azure_functions import (
    AzureFunctionsDeployer,
    FunctionAppConfig,
    FunctionTrigger,
    ScalingConfig,
)

__all__ = [
    # Copilot Studio
    "CopilotStudioConnector",
    "CopilotStudioConfig",
    "AgentManifest",
    "CopilotAction",
    "CopilotTopic",
    # Teams Bot
    "TeamsBotConnector",
    "TeamsBotConfig",
    "TeamsActivity",
    "TeamsAdaptiveCard",
    # Azure Functions
    "AzureFunctionsDeployer",
    "FunctionAppConfig",
    "FunctionTrigger",
    "ScalingConfig",
]
