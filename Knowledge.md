# Knowledge Base: langchain-azure Repository

## Repository Overview

This is the **langchain-azure** repository - an official LangChain integration package providing first-class support for Azure AI Foundry capabilities in the LangChain and LangGraph ecosystem.

### Repository Structure

```
langchain-azure/
├── libs/
│   ├── azure-ai/              # Core Azure AI Foundry integration package
│   ├── azure-dynamic-sessions/ # Azure Dynamic Sessions integration
│   ├── azure-postgresql/      # Azure PostgreSQL vector store
│   ├── azure-storage/         # Azure Blob Storage document loaders
│   └── sqlserver/            # SQL Server vector store
├── samples/
│   ├── react-agent-docintelligence/  # Sample ReAct agent with Document Intelligence
│   ├── multi-agent-travel-planner/   # Multi-agent sample
│   └── rag-storage-document-loaders/ # RAG with Azure Storage
└── README.md
```

## Core Package: langchain-azure-ai (v2.0.0)

### Key Features

1. **Azure AI Agent Service Integration**
   - `AgentServiceFactory` for creating prompt-based agents
   - Support for declarative agents created in Azure AI Foundry
   - LangGraph integration for complex agent workflows
   - Tool calling and structured outputs

2. **Azure AI Foundry Models**
   - `AzureAIChatCompletionsModel` - Chat completions using Azure AI Inference API
   - `AzureAIEmbeddingsModel` - Embeddings generation
   - Support for Azure OpenAI and GitHub Models endpoints
   - Support for models like GPT-4o, DeepSeek-R1, etc.

3. **Vector Stores**
   - Azure AI Search (formerly Cognitive Search)
   - Azure Cosmos DB (NoSQL and MongoDB vCore)
   - Semantic caching capabilities

4. **Azure AI Services Tools**
   - `AzureAIDocumentIntelligenceTool` - Document parsing and analysis
   - `AzureAITextAnalyticsHealthTool` - Health text analytics
   - `AzureAIImageAnalysisTool` - Image analysis
   - `AIServicesToolkit` - Unified access to all tools

5. **Enterprise Connectors** (New in v2.0.0)
   - `CopilotStudioConnector` - Export agents to Microsoft 365 Copilot
   - `TeamsBotConnector` - Microsoft Teams bot integration
   - `AzureFunctionsDeployer` - Serverless deployment to Azure Functions

5. **Enterprise Observability & Monitoring**
   - `AzureAIOpenTelemetryTracer` - OpenTelemetry tracing integration
   - Azure Monitor (Application Insights) integration
   - LangSmith tracing support
   - Session and user tracking
   - Custom telemetry dimensions (agent_name, session_id, user_id, message/response lengths)
   - AgentTelemetry wrapper for execution metrics
   - Dual observability stack (Azure Monitor + LangSmith)
   - Production-ready monitoring middleware
   - Graceful degradation if telemetry unavailable

6. **Additional Features**
   - Chat message histories with Cosmos DB
   - Azure Logic Apps integration
   - Retrievers for Azure AI Search

### Dependencies

**Core Dependencies:**
- `langchain>=1.0.0,<2.0.0`
- `langchain-openai>=1.0.0,<2.0.0`
- `azure-ai-agents==1.2.0b5`
- `azure-ai-inference[opentelemetry]>=1.0.0b9,<2.0`
- `azure-ai-projects~=1.0`
- `azure-identity~=1.15`
- `azure-search-documents~=11.4`
- `azure-cosmos>=4.14.0b1,<5.0`

**Optional Dependencies:**
- `[opentelemetry]` - For tracing capabilities
- `[tools]` - For Azure AI Services tools

## Integration with LangChain/LangGraph/LangSmith

### LangChain Integration
- Full compatibility with LangChain 1.0+
- Implements standard LangChain interfaces (`BaseChatModel`, `BaseEmbeddings`, `VectorStore`, etc.)
- Works seamlessly with LangChain chains, agents, and tools

### LangGraph Integration
- Native support for building agent workflows
- `AgentServiceFactory.create_prompt_agent()` returns a `CompiledStateGraph`
- Can be deployed using LangGraph CLI (`langgraph dev`, `langgraph deploy`)
- Support for `langgraph.json` configuration files
- Integration with LangGraph Cloud/Platform

### LangSmith Integration
- Compatible with LangSmith tracing via OpenTelemetry
- `AzureAIOpenTelemetryTracer` can export traces to LangSmith
- Environment variables for LangSmith configuration work as expected

## Azure AI Foundry Agent Service

### What is Azure AI Agent Service?
Azure AI Agent Service is a managed service in Azure AI Foundry that provides:
- Declarative agent creation through the portal
- Managed agent execution and lifecycle
- Built-in conversation threading
- Tool integration and execution
- State management

### Integration Patterns

**Pattern 1: Prompt-Based Agents (Recommended)**
```python
from langchain_azure_ai.agents import AgentServiceFactory

factory = AgentServiceFactory(
    project_endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>",
    credential=DefaultAzureCredential()
)

agent = factory.create_prompt_agent(
    name="my-agent",
    model="gpt-4.1",
    instructions="You are a helpful assistant...",
    tools=[tool1, tool2],
    trace=True
)

# Returns a CompiledStateGraph (LangGraph)
result = agent.invoke({"messages": [HumanMessage(content="Hello")]})
```

**Pattern 2: Declarative Agents from Azure Portal**
- Create agents through Azure AI Foundry portal
- Import and use them in LangGraph workflows
- Combines portal-managed agents with custom LangGraph logic

## Deployment Options

### Option 1: Local Development with LangGraph CLI
```bash
langgraph dev
```
- Runs agent locally with hot reload
- Interactive Studio UI
- Best for development and testing

### Option 2: Azure Container Apps (via LangGraph Platform)
- Deploy LangGraph applications to Azure
- Managed hosting with autoscaling
- Built-in observability

### Option 3: Azure AI Foundry Deployment
- Deploy agents through Azure AI Foundry portal
- Managed endpoint with API keys
- Built-in monitoring and logging

### Option 4: Custom Azure Deployment
- Deploy as Azure Container Instance
- Deploy as Azure App Service
- Deploy as Azure Kubernetes Service (AKS)

## Authentication

The package supports multiple authentication methods:
1. **DefaultAzureCredential** (Recommended for production)
   - Uses Azure Managed Identity, Azure CLI, Visual Studio, etc.
   - No secrets in code
2. **API Key Authentication**
   - Using `credential="your-api-key"`
   - Simpler for development

## Observability & Monitoring

### Built-in Observability Features

The server implementation includes comprehensive observability:

**1. Azure Monitor (Application Insights)**
- OpenTelemetry instrumentation enabled by default
- Custom dimensions on all requests:
  - `agent_name`, `agent_type`
  - `session_id`, `user_id`
  - `message_length`, `response_length`
  - Custom metadata fields
- Request/response tracking with duration
- Exception tracking with stack traces
- Live Metrics support

**Configuration:**
```bash
APPLICATIONSIGHTS_CONNECTION_STRING="InstrumentationKey=..."
ENABLE_AZURE_MONITOR=true  # Default: true
```

**2. LangSmith Tracing**
- Native LangChain/LangGraph tracing
- Full execution graphs
- LLM call inspection
- Tool invocation tracking
- Session continuity

**Configuration:**
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="your-api-key"
LANGCHAIN_PROJECT="your-project-name"
```

**3. Session & User Tracking**
- Unique session IDs per conversation
- Session continuity across requests
- User ID tracking (anonymous support)
- User-scoped analytics
- Metadata propagation through telemetry stack

**Usage Example:**
```python
from langchain_azure_ai.server import ChatRequest

request = ChatRequest(
    message="Help me with this issue",
    session_id="session-123",
    user_id="user-456",
    metadata={
        "environment": "production",
        "source": "web-app",
        "priority": "high"
    }
)
```

**4. AgentTelemetry Wrapper**
- Execution duration tracking
- Success/failure metrics
- Custom attributes per execution
- Error handling and reporting

**Server Middleware:**
- RequestLoggingMiddleware: HTTP logging with request IDs
- TracingMiddleware: OpenTelemetry span creation
- MetricsMiddleware: Request count, duration, active requests

### Querying Observability Data

**Application Insights (KQL):**
```kusto
// Agent execution times
requests
| where timestamp > ago(1h)
| where customDimensions.agent_name != ""
| project timestamp, duration, 
    agent_name = customDimensions.agent_name,
    session_id = customDimensions.session_id,
    user_id = customDimensions.user_id
| order by timestamp desc

// Session analytics
requests
| where customDimensions.session_id != ""
| summarize 
    request_count = count(),
    avg_duration = avg(duration)
    by session_id = tostring(customDimensions.session_id)
```

**LangSmith:**
- Navigate to https://smith.langchain.com
- Select your project
- Filter by session_id, user_id, or time range
- View full execution traces with LLM inputs/outputs

## Environment Variables

Key environment variables used:

**Core Configuration:**
- `AZURE_AI_PROJECT_ENDPOINT` - Azure AI Foundry project endpoint
- `AZURE_OPENAI_API_KEY` - API key for Azure OpenAI
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint
- `AZURE_OPENAI_DEPLOYMENT_NAME` - Model deployment name

**Observability Configuration:**
- `APPLICATIONINSIGHTS_CONNECTION_STRING` - Application Insights connection
- `ENABLE_AZURE_MONITOR` - Enable Azure Monitor (default: true)
- `LANGCHAIN_API_KEY` - For LangSmith integration
- `LANGCHAIN_TRACING_V2` - Enable LangSmith tracing (true/false)
- `LANGCHAIN_PROJECT` - LangSmith project name
- `ENABLE_REQUEST_LOGGING` - Enable request logging (default: true)
- `ENABLE_TOKEN_TRACKING` - Enable token tracking (default: true)

## Samples Included

### 1. react-agent-docintelligence
- **Purpose**: ReAct agent with Azure AI Document Intelligence tool
- **Stack**: LangGraph + Azure AI Agent Service + Document Intelligence
- **Key Files**: `src/react_agent/graph.py`
- **Deployment**: LangGraph CLI ready

### 2. multi-agent-travel-planner
- **Purpose**: Multi-agent system with nested agents
- **Pattern**: Supervisor pattern with specialized sub-agents
- **Features**: OpenTelemetry tracing, complex workflows

### 3. rag-storage-document-loaders
- **Purpose**: RAG pipeline with Azure Blob Storage
- **Stack**: Azure AI Search + Azure Storage + Embeddings
- **Pattern**: Traditional RAG (embed + query)

## Known Issues and Considerations

1. **Agent Service is in Preview**: `azure-ai-agents==1.2.0b5` is beta
2. **Breaking Changes in v1.0**: Migration from 0.1.x requires parameter changes
3. **Tracing Setup**: Requires OpenTelemetry extras for full functionality
4. **Model Availability**: Ensure models are deployed in your Azure region

## Related Azure Services Required

To use this repository effectively, you need:
1. **Azure AI Foundry Hub and Project** (required)
2. **Azure OpenAI Service** (for LLM models)
3. **Azure AI Document Intelligence** (optional, for document tools)
4. **Azure AI Search** (optional, for vector stores)
5. **Azure Cosmos DB** (optional, for vector stores/chat history)
6. **Azure Storage Account** (optional, for document loaders)
7. **Azure Application Insights** (optional, for monitoring)

---

## Updates Log

### 2026-01-26 - Phase 3: Copilot Studio, Teams Bot & Azure Functions

**Connectors Module** (`langchain_azure_ai.connectors`)

1. **Copilot Studio Connector** (`copilot_studio.py`)
   - `CopilotStudioConnector` - Export LangChain agents to Microsoft 365 Copilot
   - `AgentManifest` - Plugin manifest with OpenAPI spec generation
   - `CopilotAction`, `CopilotTopic` - Define agent capabilities
   - `export_agent()` - Create M365 Copilot plugin from agent
   - `create_m365_copilot_plugin()` - Full plugin package with Teams manifest
   - `publish_to_copilot_studio()` - Direct deployment to Copilot Studio

2. **Teams Bot Connector** (`teams_bot.py`)
   - `TeamsBotConnector` - Microsoft Teams bot integration
   - `TeamsActivity` - Message parsing and response building
   - `TeamsAdaptiveCard` - Rich card builder (add_text, add_header, add_fact_set, etc.)
   - `create_fastapi_routes()` - FastAPI router generation
   - `generate_manifest()` - Teams app manifest creation
   - Proactive messaging support

3. **Azure Functions Deployer** (`azure_functions.py`)
   - `AzureFunctionsDeployer` - Serverless deployment to Azure Functions
   - `FunctionAppConfig` - Function app configuration
   - `ScalingConfig` - Auto-scaling settings (min/max instances, cooldowns)
   - `generate_scaffold()` - Full function app project generation
   - Bicep template generation for Azure infrastructure
   - Deploy scripts (bash and PowerShell)
   - GitHub Actions CI/CD workflow generation

**Usage Examples:**

```python
# Export agent to Copilot Studio
from langchain_azure_ai.connectors import CopilotStudioConnector

connector = CopilotStudioConnector()
manifest = connector.export_agent(
    agent=my_agent,
    name="IT Helpdesk",
    description="IT support agent"
)
manifest.save("output/")

# Create Teams bot
from langchain_azure_ai.connectors import TeamsBotConnector

bot = TeamsBotConnector()
bot.register_agent("helpdesk", helpdesk_agent)
router = bot.create_fastapi_routes()

# Deploy to Azure Functions
from langchain_azure_ai.connectors import AzureFunctionsDeployer

deployer = AzureFunctionsDeployer()
deployer.generate_scaffold(
    agents={"helpdesk": helpdesk_agent},
    output_dir="functions_app"
)
```

### 2026-01-24 - Phase 2: Observability & Testing
- **Azure Monitor OpenTelemetry Integration**
  - Added `langchain_azure_ai.observability` module with `TelemetryConfig`, `AgentTelemetry`, `ExecutionMetrics`
  - `setup_azure_monitor()` function for Azure Monitor initialization
  - `trace_agent` decorator for automatic method tracing
  - Middleware: `RequestLoggingMiddleware`, `TracingMiddleware`, `MetricsMiddleware`
  - Metrics: duration histogram, token counter, request counter, error counter
  - Graceful fallback when `APPLICATIONINSIGHTS_CONNECTION_STRING` not set

- **Comprehensive Unit Tests**
  - `tests/unit_tests/test_wrappers.py` with tests for all wrapper types
  - Tests for WrapperConfig, FoundryAgentWrapper, IT/Enterprise/Deep agent wrappers
  - Observability module and middleware tests
  - Server endpoint tests with mock agents

- **Integration Tests**
  - `tests/integration_tests/test_agents.py` for Azure OpenAI and Foundry integration
  - Multi-turn conversation tests
  - Streaming response tests
  - `tests/conftest.py` with shared fixtures and pytest configuration

- **API Documentation (OpenAPI/Swagger)**
  - Enhanced FastAPI with comprehensive OpenAPI metadata
  - Tagged endpoints: health, agents, it-agents, enterprise-agents, deep-agents, chat
  - Swagger UI at `/docs`, ReDoc at `/redoc`
  - OpenAPI JSON at `/openapi.json`

### 2026-01-24 - Initial Knowledge Base Creation
- Analyzed repository structure and capabilities
- Documented integration points with LangChain/LangGraph/LangSmith
- Identified deployment patterns for Azure AI Foundry
- Created baseline understanding for agent development

---

## Observability Setup

### Environment Variables for Observability
```bash
# Required for Azure Monitor
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=xxx

# Optional configuration
ENABLE_AZURE_MONITOR=true
OTEL_SERVICE_NAME=my-agent-service
ENABLE_TRACING=true
```

### Using Telemetry in Code
```python
from langchain_azure_ai.observability import AgentTelemetry, setup_azure_monitor

# Initialize Azure Monitor (call once at startup)
setup_azure_monitor()

# Create telemetry for an agent
telemetry = AgentTelemetry("my-agent", "enterprise")

# Track execution
with telemetry.track_execution("invoke") as metrics:
    # Your agent code
    metrics.prompt_tokens = 100
    metrics.completion_tokens = 50

# Access metrics after execution
print(f"Duration: {metrics.duration_ms}ms, Total tokens: {metrics.total_tokens}")
```

### Using the trace_agent Decorator
```python
from langchain_azure_ai.observability import trace_agent

@trace_agent("my_operation", {"custom": "attribute"})
def my_agent_method(self, input_data):
    # This will be automatically traced
    return self.agent.invoke(input_data)
```

---

**Last Updated**: 2026-01-29  
**Repository Version**: langchain-azure-ai v2.0.0  
**LangChain Version**: 1.0.2  
**LangGraph CLI**: 0.4.4+
