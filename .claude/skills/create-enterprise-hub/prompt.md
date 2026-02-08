# Create Enterprise Hub Skill - Execution Instructions

You are executing the `create-enterprise-hub` skill to build an end-to-end multi-agent enterprise system.

## Purpose

Create a production-ready enterprise hub that combines multiple LangChain agents, RAG knowledge bases, and enterprise connectors (Teams, Copilot Studio, Azure Functions) into a unified system.

## Parameters

- **hub_name** (required): Hub name (kebab-case, e.g., "it-support-hub")
- **use_case** (optional, default: "it-support"): Predefined template
  - `it-support`: IT Helpdesk + Knowledge Base + ServiceNow integration
  - `customer-service`: Customer Support + FAQ + Ticketing
  - `doc-intelligence`: Document processing + Analysis + Q&A
  - `custom`: Custom agent configuration
- **agents** (optional): Comma-separated agent list
- **deploy_to** (optional, default: "all"): Deployment targets
- **with_rag** (optional, default: true): Include RAG
- **with_monitoring** (optional, default: true): Include observability

## Pre-Execution Planning

### Step 1: Determine Agent Configuration

Based on `use_case` parameter:

#### IT Support Hub (use_case="it-support")
```python
agents = [
    {
        "name": "helpdesk",
        "role": "IT Helpdesk agent that handles common IT issues",
        "model": "gpt-4o",
        "tools": ["search_kb", "reset_password", "check_ticket_status"],
        "instructions": "You are an IT helpdesk agent..."
    },
    {
        "name": "knowledge_base",
        "role": "RAG agent for IT documentation",
        "type": "rag",
        "storage": "search",
        "embeddings": "azure"
    },
    {
        "name": "servicenow",
        "role": "ServiceNow integration for ticket management",
        "model": "gpt-4o-mini",
        "tools": ["create_incident", "update_incident", "search_cmdb"],
        "instructions": "You handle ServiceNow operations..."
    }
]

orchestrator_logic = "Route to helpdesk for general queries, knowledge_base for documentation lookup, servicenow for ticket operations"
```

#### Customer Service Hub (use_case="customer-service")
```python
agents = [
    {
        "name": "support",
        "role": "Customer support agent",
        "model": "gpt-4o",
        "tools": ["search_orders", "check_status", "initiate_return"],
        "instructions": "You are a customer support agent..."
    },
    {
        "name": "faq",
        "role": "FAQ knowledge base",
        "type": "rag",
        "storage": "cosmos",
        "embeddings": "azure"
    },
    {
        "name": "ticketing",
        "role": "Ticket creation and management",
        "model": "gpt-4o-mini",
        "tools": ["create_ticket", "escalate_ticket", "resolve_ticket"],
        "instructions": "You manage customer tickets..."
    }
]

orchestrator_logic = "Route to support for active conversations, faq for questions, ticketing for issues needing escalation"
```

#### Document Intelligence Hub (use_case="doc-intelligence")
```python
agents = [
    {
        "name": "processor",
        "role": "Document processing agent",
        "model": "gpt-4o",
        "tools": ["extract_text", "analyze_layout", "extract_tables"],
        "instructions": "You process documents using Azure Document Intelligence..."
    },
    {
        "name": "summarizer",
        "role": "Document summarization agent",
        "model": "gpt-4o",
        "tools": ["summarize", "extract_key_points"],
        "instructions": "You create concise summaries..."
    },
    {
        "name": "qa",
        "role": "Q&A over documents",
        "type": "rag",
        "storage": "postgresql",
        "embeddings": "azure"
    }
]

orchestrator_logic = "Process documents with processor, summarize with summarizer, answer questions with qa"
```

## Execution Steps

### Step 1: Create Hub Directory Structure

```bash
mkdir -p samples/enterprise-{{hub_name}}
cd samples/enterprise-{{hub_name}}

# Create subdirectories
mkdir -p agents
mkdir -p shared
mkdir -p deploy/{teams,copilot,functions}
mkdir -p tests
mkdir -p docs
```

### Step 2: Create Individual Agents

For each agent in the configuration:

#### A. For Standard Agents (non-RAG)

Create `agents/{{agent_name}}_agent.py`:

```python
"""{{Agent Display Name}} Agent.

This agent is part of the {{Hub Name}} enterprise system.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool
from langchain_azure_ai.agents import AgentServiceFactory
from azure.identity import DefaultAzureCredential


# Define agent-specific tools
@tool
def {{tool_name}}(param: str) -> str:
    """{{Tool description}}.

    Args:
        param: {{Parameter description}}.

    Returns:
        {{Return description}}.
    """
    # Tool implementation
    return f"Result: {param}"


def create_{{agent_name}}_agent(factory: AgentServiceFactory) -> Any:
    """Create {{Agent Display Name}} agent.

    Args:
        factory: AgentServiceFactory instance.

    Returns:
        Compiled agent graph.
    """
    tools = [{{tool_name}}]  # Add all agent tools

    agent = factory.create_prompt_agent(
        name="{{agent_name}}",
        model="{{model}}",
        instructions='''{{instructions}}''',
        tools=tools,
        trace=True,
    )

    return agent


__all__ = ["create_{{agent_name}}_agent"]
```

#### B. For RAG Agents

Create `agents/{{agent_name}}_rag.py`:

```python
"""{{Agent Display Name}} RAG Agent.

Knowledge base for {{Hub Name}}.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import appropriate vectorstore based on storage choice
{% if storage == "search" %}
from langchain_azure_ai.vectorstores import AzureAISearchVectorStore
{% elif storage == "cosmos" %}
from langchain_azure_ai.vectorstores import AzureCosmosDBVectorStore
{% elif storage == "postgresql" %}
from langchain_azure_postgresql.vectorstores import AzurePostgreSQLVectorStore
{% endif %}

from langchain_azure_ai.embeddings import AzureOpenAIEmbeddings


class {{PascalCase}}RAG:
    """RAG system for {{Agent Display Name}}."""

    def __init__(self) -> None:
        """Initialize RAG system."""
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-large"
        )

        self.vectorstore = {% if storage == "search" %}AzureAISearchVectorStore{% elif storage == "cosmos" %}AzureCosmosDBVectorStore{% elif storage == "postgresql" %}AzurePostgreSQLVectorStore{% endif %}(
            {% if storage == "search" %}index_name="{{hub_name}}_{{agent_name}}",{% elif storage == "cosmos" %}collection_name="{{hub_name}}_{{agent_name}}",{% elif storage == "postgresql" %}collection_name="{{hub_name}}_{{agent_name}}",{% endif %}
            embedding{% if storage == "search" %}_function{% endif %}=self.embeddings,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def ingest_documents(self, documents: list[Document]) -> list[str]:
        """Ingest documents into knowledge base."""
        chunks = self.text_splitter.split_documents(documents)
        ids = self.vectorstore.add_documents(chunks)
        return ids

    def query(self, query: str, k: int = 4) -> list[Document]:
        """Query the knowledge base."""
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def as_retriever(self, **kwargs: Any) -> Any:
        """Get as LangChain retriever."""
        return self.vectorstore.as_retriever(**kwargs)


def create_{{agent_name}}_rag() -> {{PascalCase}}RAG:
    """Create {{Agent Display Name}} RAG system."""
    return {{PascalCase}}RAG()


__all__ = ["{{PascalCase}}RAG", "create_{{agent_name}}_rag"]
```

### Step 3: Create Multi-Agent Orchestrator

Create `orchestrator.py`:

```python
"""Multi-Agent Orchestrator for {{Hub Name}}.

This orchestrator manages multiple specialized agents and routes requests
to the appropriate agent based on user intent and context.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_azure_ai.agents import AgentServiceFactory
from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer
from azure.identity import DefaultAzureCredential
from azure.monitor.opentelemetry import configure_azure_monitor

# Import created agents
{% for agent in agents %}
{% if agent.type != "rag" %}
from agents.{{agent.name}}_agent import create_{{agent.name}}_agent
{% else %}
from agents.{{agent.name}}_rag import create_{{agent.name}}_rag
{% endif %}
{% endfor %}

load_dotenv()

# Configure monitoring if enabled
{% if with_monitoring %}
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()
{% endif %}


class HubState(TypedDict):
    """State for {{Hub Name}} orchestrator."""

    messages: Annotated[List[BaseMessage], add_messages]
    user_request: str
    current_agent: str
    session_id: str
    context: Dict[str, Any]


class {{PascalCase}}Hub:
    """{{Hub Name}} Multi-Agent Hub.

    This hub orchestrates multiple specialized agents:
    {% for agent in agents %}
    - {{agent.name}}: {{agent.role}}
    {% endfor %}
    """

    def __init__(self) -> None:
        """Initialize the hub."""
        # Initialize agent factory
        self.factory = AgentServiceFactory(
            project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            credential=DefaultAzureCredential(),
        )

        # Create individual agents
        {% for agent in agents %}
        {% if agent.type != "rag" %}
        self.{{agent.name}} = create_{{agent.name}}_agent(self.factory)
        {% else %}
        self.{{agent.name}} = create_{{agent.name}}_rag()
        {% endif %}
        {% endfor %}

        # Create orchestrator graph
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the orchestration graph."""
        builder = StateGraph(HubState)

        # Add router node
        builder.add_node("router", self._route_request)

        # Add agent nodes
        {% for agent in agents %}
        builder.add_node("{{agent.name}}", self._{{agent.name}}_node)
        {% endfor %}

        # Add edges
        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            self._determine_next_agent,
            {% raw %}{{% endraw %}
            {% for agent in agents %}
                "{{agent.name}}": "{{agent.name}}",
            {% endfor %}
                END: END,
            {% raw %}}{% endraw %}
        )

        {% for agent in agents %}
        builder.add_edge("{{agent.name}}", END)
        {% endfor %}

        # Compile with checkpointer for memory
        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    def _route_request(self, state: HubState) -> HubState:
        """Route user request to appropriate agent."""
        user_message = state["messages"][-1].content if state["messages"] else state["user_request"]

        # Use LLM to determine routing
        router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        routing_prompt = f"""Given the user request, determine which agent should handle it.

Available agents:
{% for agent in agents %}
- {{agent.name}}: {{agent.role}}
{% endfor %}

User request: {user_message}

{{Orchestrator routing logic}}

Respond with ONLY the agent name, nothing else."""

        result = router_llm.invoke([HumanMessage(content=routing_prompt)])
        chosen_agent = result.content.strip().lower()

        state["current_agent"] = chosen_agent
        return state

    def _determine_next_agent(self, state: HubState) -> str:
        """Determine next agent based on routing."""
        agent = state.get("current_agent", "").lower()

        {% for agent in agents %}
        if agent == "{{agent.name}}":
            return "{{agent.name}}"
        {% endfor %}

        return END

    {% for agent in agents %}
    def _{{agent.name}}_node(self, state: HubState) -> HubState:
        """Execute {{agent.name}} agent."""
        {% if agent.type != "rag" %}
        # Standard agent execution
        result = self.{{agent.name}}.invoke({"messages": state["messages"]})
        state["messages"] = result["messages"]
        {% else %}
        # RAG agent execution
        query = state["messages"][-1].content if state["messages"] else state["user_request"]
        docs = self.{{agent.name}}.query(query)

        # Create response with retrieved context
        context = "\n\n".join([doc.page_content for doc in docs])
        response = f"Based on the knowledge base:\n\n{context}"
        state["messages"].append(AIMessage(content=response))
        {% endif %}

        return state
    {% endfor %}

    def invoke(self, request: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Invoke the hub with a user request.

        Args:
            request: User request text.
            session_id: Optional session ID for conversation continuity.

        Returns:
            Hub response with messages and metadata.
        """
        import uuid

        session_id = session_id or str(uuid.uuid4())

        config = {"configurable": {"thread_id": session_id}}

        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=request)],
                "user_request": request,
                "session_id": session_id,
                "context": {},
            },
            config=config,
        )

        return result

    async def ainvoke(self, request: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Async invoke the hub."""
        import uuid

        session_id = session_id or str(uuid.uuid4())

        config = {"configurable": {"thread_id": session_id}}

        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=request)],
                "user_request": request,
                "session_id": session_id,
                "context": {},
            },
            config=config,
        )

        return result


def create_hub() -> {{PascalCase}}Hub:
    """Create {{Hub Name}} instance."""
    return {{PascalCase}}Hub()


if __name__ == "__main__":
    # Example usage
    hub = create_hub()

    # Test request
    response = hub.invoke("Hello, I need help with my laptop")

    print("\\n=== Response ===")
    for msg in response["messages"]:
        print(f"{msg.type}: {msg.content}\\n")
```

### Step 4: Create Deployment Configurations

#### A. Teams Bot Deployment

Create `deploy/teams/deploy_teams.py`:

```python
"""Deploy {{Hub Name}} to Microsoft Teams."""

from langchain_azure_ai.connectors import TeamsBotConnector, TeamsBotConfig
from fastapi import FastAPI
import sys
sys.path.append("../..")

from orchestrator import create_hub

# Create FastAPI app
app = FastAPI(title="{{Hub Name}} Teams Bot")

# Configure Teams bot
config = TeamsBotConfig.from_env()
bot = TeamsBotConnector(config)

# Register hub
hub = create_hub()
bot.register_agent("{{hub_name}}", hub)

# Add Teams routes
app.include_router(bot.create_fastapi_routes())

# Generate Teams manifest
manifest = bot.generate_manifest(
    name="{{Hub Display Name}}",
    description="{{Hub description}}",
    base_url=config.bot_endpoint,
)

# Save manifest
import json
with open("teams_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Teams bot configured!")
print(f"📄 Manifest saved to: teams_manifest.json")
print(f"🚀 Start server: uvicorn deploy_teams:app --host 0.0.0.0 --port 8000")
```

#### B. Copilot Studio Deployment

Create `deploy/copilot/deploy_copilot.py`:

```python
"""Deploy {{Hub Name}} to Microsoft 365 Copilot Studio."""

from langchain_azure_ai.connectors import CopilotStudioConnector, CopilotStudioConfig
import sys
sys.path.append("../..")

from orchestrator import create_hub

# Configure Copilot Studio
config = CopilotStudioConfig.from_env()
connector = CopilotStudioConnector(config)

# Create hub
hub = create_hub()

# Export as M365 Copilot plugin
connector.create_m365_copilot_plugin(
    wrapper=hub,
    name="{{Hub Display Name}}",
    description="{{Hub description}}",
    api_base_url=config.api_base_url or "https://your-api.example.com",
    output_dir="copilot_plugin",
)

print(f"✅ Copilot Studio plugin created!")
print(f"📁 Plugin files in: copilot_plugin/")
print(f"📖 See copilot_plugin/README.md for import instructions")
```

#### C. Azure Functions Deployment

Create `deploy/functions/deploy_functions.py`:

```python
"""Deploy {{Hub Name}} to Azure Functions."""

from langchain_azure_ai.connectors import (
    AzureFunctionsDeployer,
    FunctionAppConfig,
    ScalingConfig,
)
import sys
sys.path.append("../..")

from orchestrator import create_hub

# Configure Azure Functions
config = FunctionAppConfig(
    name="{{hub_name}}-functions",
    resource_group=os.getenv("AZURE_RESOURCE_GROUP", "rg-{{hub_name}}"),
    scaling=ScalingConfig(min_instances=1, max_instances=10),
)

# Create deployer
deployer = AzureFunctionsDeployer(config)

# Create hub
hub = create_hub()

# Generate function app scaffold
deployer.generate_scaffold(
    output_dir="function_app",
    wrappers={"{{hub_name}}": hub},
)

print(f"✅ Azure Functions app generated!")
print(f"📁 Function app in: function_app/")
print(f"🚀 Deploy with: cd function_app && func azure functionapp publish {{hub_name}}-functions")
```

### Step 5: Create Configuration Files

#### .env.example

```bash
# Azure AI Project
AZURE_AI_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key

# Azure Monitor (optional)
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...

# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT={{hub_name}}

# Teams Bot Configuration
TEAMS_APP_ID=your-teams-app-id
TEAMS_APP_PASSWORD=your-teams-app-password
TEAMS_BOT_ENDPOINT=https://your-bot.azurewebsites.net

# M365 Copilot Configuration
COPILOT_STUDIO_ENVIRONMENT=prod
COPILOT_API_BASE_URL=https://your-api.example.com

# Vector Store Configuration
{% if with_rag %}
{% if storage == "search" %}
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your-key
{% elif storage == "cosmos" %}
COSMOS_CONNECTION_STRING=your-connection-string
COSMOS_DATABASE_NAME=your-db
{% elif storage == "postgresql" %}
POSTGRESQL_CONNECTION_STRING=your-connection-string
{% endif %}
{% endif %}
```

### Step 6: Create Tests

Create `tests/test_orchestrator.py`:

```python
"""Tests for {{Hub Name}} orchestrator."""

import pytest
from unittest.mock import MagicMock, patch
import sys
sys.path.append("..")

from orchestrator import create_hub, {{PascalCase}}Hub


class Test{{PascalCase}}Hub:
    """Test suite for {{Hub Name}}."""

    @pytest.fixture
    def hub(self):
        """Create hub instance."""
        with patch("orchestrator.AgentServiceFactory"):
            return create_hub()

    def test_hub_creation(self, hub):
        """Test hub can be created."""
        assert isinstance(hub, {{PascalCase}}Hub)
        assert hub.graph is not None

    def test_routing(self, hub):
        """Test request routing."""
        # Mock agent executions
        hub.graph.invoke = MagicMock(return_value={
            "messages": [],
            "current_agent": "helpdesk",
        })

        response = hub.invoke("I need help with my laptop")
        assert response is not None

    {% for agent in agents %}
    def test_{{agent.name}}_node(self, hub):
        """Test {{agent.name}} node execution."""
        state = {
            "messages": [],
            "user_request": "Test request",
            "current_agent": "{{agent.name}}",
            "session_id": "test",
            "context": {},
        }

        result = hub._{{agent.name}}_node(state)
        assert result is not None
        assert "messages" in result
    {% endfor %}
```

### Step 7: Create Documentation

Create `README.md`:

```markdown
# {{Hub Display Name}}

Enterprise multi-agent hub built with LangChain Azure AI.

## Overview

{{Hub description and purpose}}

## Architecture

This hub consists of multiple specialized agents:

{% for agent in agents %}
### {{agent.display_name}}
- **Role**: {{agent.role}}
- **Model**: {{agent.model}}
- **Type**: {{"RAG" if agent.type == "rag" else "Generative"}}
{% endfor %}

## Setup

### Prerequisites

- Python 3.10+
- Azure AI Project
- Azure credentials configured

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Running Locally

```bash
python orchestrator.py
```

## Deployment

### Microsoft Teams

```bash
cd deploy/teams
python deploy_teams.py
uvicorn deploy_teams:app --host 0.0.0.0 --port 8000
```

### Microsoft 365 Copilot

```bash
cd deploy/copilot
python deploy_copilot.py
# Follow instructions in copilot_plugin/README.md
```

### Azure Functions

```bash
cd deploy/functions
python deploy_functions.py
cd function_app
func azure functionapp publish {{hub_name}}-functions
```

## Usage Examples

### Basic Usage

```python
from orchestrator import create_hub

hub = create_hub()
response = hub.invoke("Your request here")

for msg in response["messages"]:
    print(f"{msg.type}: {msg.content}")
```

### With Session Continuity

```python
session_id = "user-123"
response = hub.invoke("First request", session_id=session_id)
response = hub.invoke("Follow-up request", session_id=session_id)
```

## Testing

```bash
pytest tests/ -v
```

## Monitoring

- **Azure Monitor**: Configured via APPLICATIONINSIGHTS_CONNECTION_STRING
- **LangSmith**: Configured via LANGCHAIN_TRACING_V2

## Architecture Diagram

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│   Router/            │
│   Orchestrator       │
└──────┬───────────────┘
       │
       ├──► {{agents[0].name}}
       ├──► {{agents[1].name}}
       {% if agents|length > 2 %}└──► {{agents[2].name}}{% endif %}
```

## License

MIT
```

### Step 8: Create requirements.txt

```txt
langchain-azure-ai>=0.1.0
langchain-openai>=0.1.0
langchain-core>=0.3.0
langgraph>=0.2.0
{% if with_rag %}
{% if storage == "search" %}
langchain-azure-search>=0.1.0
{% elif storage == "cosmos" %}
langchain-azure-cosmos>=0.1.0
{% elif storage == "postgresql" %}
langchain-azure-postgresql>=0.1.0
{% endif %}
{% endif %}
azure-identity>=1.15.0
azure-ai-projects>=1.0.0
azure-monitor-opentelemetry>=1.0.0
python-dotenv>=1.0.0
fastapi>=0.115.0
uvicorn>=0.30.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

## Post-Execution Summary

Provide this summary to the user:

```
✅ Successfully created {{Hub Display Name}}!

📁 Project Structure:
samples/enterprise-{{hub_name}}/
├── agents/                    # Individual agent implementations
│   {% for agent in agents %}├── {{agent.name}}_{{"rag" if agent.type == "rag" else "agent"}}.py
│   {% endfor %}
├── orchestrator.py           # Multi-agent orchestration
├── deploy/
│   ├── teams/               # Teams bot deployment
│   ├── copilot/             # Copilot Studio deployment
│   └── functions/           # Azure Functions deployment
├── tests/                   # Test suite
├── docs/                    # Documentation
├── .env.example             # Configuration template
├── requirements.txt         # Dependencies
└── README.md                # Project documentation

🤖 Agents Created:
{% for agent in agents %}
- {{agent.name}}: {{agent.role}}
{% endfor %}

🚀 Quick Start:
1. cd samples/enterprise-{{hub_name}}
2. cp .env.example .env
3. # Edit .env with your Azure credentials
4. pip install -r requirements.txt
5. python orchestrator.py

📦 Deployment Options:
- Teams: cd deploy/teams && python deploy_teams.py
- Copilot: cd deploy/copilot && python deploy_copilot.py
- Functions: cd deploy/functions && python deploy_functions.py

📊 Monitoring:
{% if with_monitoring %}
- Azure Monitor: Configured (set APPLICATIONINSIGHTS_CONNECTION_STRING)
- LangSmith: Configured (set LANGCHAIN_TRACING_V2=true)
{% else %}
- Monitoring: Not configured (use --with_monitoring true)
{% endif %}

📚 Documentation:
- samples/enterprise-{{hub_name}}/README.md

🧪 Run Tests:
pytest samples/enterprise-{{hub_name}}/tests/ -v

Need help? Check the README.md for detailed instructions!
```

## Success Criteria

- [ ] All agent files created successfully
- [ ] Orchestrator implements proper routing
- [ ] Deployment configs for all targets
- [ ] Tests created and structure validated
- [ ] Documentation complete
- [ ] Configuration templates provided
- [ ] All templates properly filled with parameters
- [ ] Code follows repository standards

## Error Handling

### If hub already exists:
Ask user to choose different name or confirm overwrite.

### If Azure credentials missing:
Warn that deployment configurations require Azure credentials but continue with code generation.

### If invalid use_case:
List available use cases and ask user to choose.

---

**Remember**: This skill creates a complete, production-ready enterprise system. Take time to generate high-quality code with proper error handling, logging, and documentation.
