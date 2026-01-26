"""FastAPI server for Azure AI Foundry wrapped agents.

This module provides a FastAPI server that exposes wrapped LangChain agents
through REST endpoints and a chat UI, mirroring the langchain-agents patterns.

Endpoints:
- /chat - Interactive chat UI
- /chatui - Alternative chat UI endpoint
- /api/conversation/ - IT agent endpoints
- /api/enterprise/{agent_type}/ - Enterprise agent endpoints
- /api/deepagent/{agent_type}/ - DeepAgent endpoints
- /health - Health check endpoint
- /agents - List available agents
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat endpoints."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    user_id: Optional[str] = Field(None, description="User ID")
    stream: bool = Field(False, description="Whether to stream the response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""

    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    agent_type: str = Field(..., description="Type of agent")
    timestamp: str = Field(..., description="Response timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AgentInfo(BaseModel):
    """Information about an available agent."""

    name: str
    type: str
    subtype: str
    description: str
    endpoints: List[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    agents_loaded: int
    azure_foundry_enabled: bool


# Agent Registry
class AgentRegistry:
    """Registry for managing wrapped agents."""

    def __init__(self):
        self.it_agents: Dict[str, Any] = {}
        self.enterprise_agents: Dict[str, Any] = {}
        self.deep_agents: Dict[str, Any] = {}
        self._initialized = False

    def register_it_agent(self, name: str, agent: Any) -> None:
        """Register an IT agent."""
        self.it_agents[name] = agent
        logger.info(f"Registered IT agent: {name}")

    def register_enterprise_agent(self, name: str, agent: Any) -> None:
        """Register an enterprise agent."""
        self.enterprise_agents[name] = agent
        logger.info(f"Registered Enterprise agent: {name}")

    def register_deep_agent(self, name: str, agent: Any) -> None:
        """Register a DeepAgent."""
        self.deep_agents[name] = agent
        logger.info(f"Registered DeepAgent: {name}")

    def get_it_agent(self, name: str) -> Optional[Any]:
        """Get an IT agent by name."""
        return self.it_agents.get(name)

    def get_enterprise_agent(self, name: str) -> Optional[Any]:
        """Get an enterprise agent by name."""
        return self.enterprise_agents.get(name)

    def get_deep_agent(self, name: str) -> Optional[Any]:
        """Get a DeepAgent by name."""
        return self.deep_agents.get(name)

    def list_agents(self) -> List[AgentInfo]:
        """List all registered agents."""
        agents = []

        for name, agent in self.it_agents.items():
            agents.append(
                AgentInfo(
                    name=name,
                    type="IT",
                    subtype=getattr(agent, "agent_subtype", "unknown"),
                    description=getattr(agent, "description", ""),
                    endpoints=[
                        f"/api/conversation/{name}",
                        f"/api/it/{name}/chat",
                    ],
                )
            )

        for name, agent in self.enterprise_agents.items():
            agents.append(
                AgentInfo(
                    name=name,
                    type="Enterprise",
                    subtype=getattr(agent, "agent_subtype", "unknown"),
                    description=getattr(agent, "description", ""),
                    endpoints=[
                        f"/api/enterprise/{name}/chat",
                        f"/api/enterprise/{name}/analyze",
                    ],
                )
            )

        for name, agent in self.deep_agents.items():
            agents.append(
                AgentInfo(
                    name=name,
                    type="DeepAgent",
                    subtype=getattr(agent, "agent_subtype", "unknown"),
                    description=getattr(agent, "description", ""),
                    endpoints=[
                        f"/api/deepagent/{name}/execute",
                        f"/api/deepagent/{name}/chat",
                    ],
                )
            )

        return agents

    @property
    def total_agents(self) -> int:
        """Total number of registered agents."""
        return len(self.it_agents) + len(self.enterprise_agents) + len(self.deep_agents)


# Global registry
registry = AgentRegistry()


def load_agents():
    """Load and register all wrapped agents.

    This function should be customized to load your specific agents.
    It reads configuration from environment variables.
    """
    from langchain_azure_ai.wrappers import (
        ITHelpdeskWrapper,
        ServiceNowWrapper,
        HITLSupportWrapper,
        ResearchAgentWrapper,
        ContentAgentWrapper,
        DataAnalystWrapper,
        DocumentAgentWrapper,
        CodeAssistantWrapper,
        RAGAgentWrapper,
        DocumentIntelligenceWrapper,
        ITOperationsWrapper,
        SalesIntelligenceWrapper,
        RecruitmentWrapper,
    )

    # Check if Azure AI Foundry is enabled
    use_foundry = os.getenv("USE_AZURE_FOUNDRY", "true").lower() == "true"
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

    logger.info(f"Loading agents with Azure AI Foundry: {use_foundry}")
    logger.info(f"Using model: {model}")

    # Load IT Agents
    try:
        registry.register_it_agent(
            "helpdesk",
            ITHelpdeskWrapper(name="it-helpdesk", model=model),
        )
        registry.register_it_agent(
            "servicenow",
            ServiceNowWrapper(name="servicenow-agent", model=model),
        )
        registry.register_it_agent(
            "hitl_support",
            HITLSupportWrapper(name="hitl-support", model=model),
        )
        logger.info("IT agents loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load IT agents: {e}")

    # Load Enterprise Agents
    try:
        registry.register_enterprise_agent(
            "research",
            ResearchAgentWrapper(name="research-agent", model=model),
        )
        registry.register_enterprise_agent(
            "content",
            ContentAgentWrapper(name="content-agent", model=model),
        )
        registry.register_enterprise_agent(
            "data_analyst",
            DataAnalystWrapper(name="data-analyst-agent", model=model),
        )
        registry.register_enterprise_agent(
            "document",
            DocumentAgentWrapper(name="document-agent", model=model),
        )
        registry.register_enterprise_agent(
            "code_assistant",
            CodeAssistantWrapper(name="code-assistant-agent", model=model),
        )
        registry.register_enterprise_agent(
            "rag",
            RAGAgentWrapper(name="rag-agent", model=model),
        )
        registry.register_enterprise_agent(
            "document_intelligence",
            DocumentIntelligenceWrapper(name="document-intelligence-agent", model=model),
        )
        logger.info("Enterprise agents loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Enterprise agents: {e}")

    # Load DeepAgents
    try:
        registry.register_deep_agent(
            "it_operations",
            ITOperationsWrapper(name="it-operations", model=model),
        )
        registry.register_deep_agent(
            "sales_intelligence",
            SalesIntelligenceWrapper(name="sales-intelligence", model=model),
        )
        registry.register_deep_agent(
            "recruitment",
            RecruitmentWrapper(name="recruitment", model=model),
        )
        logger.info("DeepAgents loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load DeepAgents: {e}")

    registry._initialized = True
    logger.info(f"Total agents loaded: {registry.total_agents}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting Azure AI Foundry Agent Server...")
    load_agents()
    yield
    # Shutdown
    logger.info("Shutting down Azure AI Foundry Agent Server...")


# Create FastAPI app
app = FastAPI(
    title="Azure AI Foundry Agent Server",
    description="LangChain agents wrapped with Azure AI Foundry integration",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        agents_loaded=registry.total_agents,
        azure_foundry_enabled=os.getenv("USE_AZURE_FOUNDRY", "true").lower() == "true",
    )


# List agents endpoint
@app.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    """List all available agents."""
    return registry.list_agents()


# Chat UI endpoint
@app.get("/chat", response_class=HTMLResponse)
@app.get("/chatui", response_class=HTMLResponse)
async def chat_ui():
    """Serve the chat UI."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Azure AI Foundry Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ddd; border-radius: 8px; padding: 20px; height: 500px; overflow-y: auto; }
            .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
            .user { background-color: #e3f2fd; text-align: right; }
            .assistant { background-color: #f5f5f5; }
            .input-container { display: flex; margin-top: 20px; }
            #message-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            #send-btn { padding: 10px 20px; background-color: #0078d4; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px; }
            .agent-selector { margin-bottom: 20px; }
            select { padding: 10px; border-radius: 4px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Azure AI Foundry Chat</h1>
        <div class="agent-selector">
            <label>Select Agent: </label>
            <select id="agent-select">
                <optgroup label="IT Agents">
                    <option value="helpdesk">IT Helpdesk</option>
                    <option value="servicenow">ServiceNow</option>
                    <option value="hitl_support">HITL Support</option>
                </optgroup>
                <optgroup label="Enterprise Agents">
                    <option value="research">Research</option>
                    <option value="content">Content</option>
                    <option value="data_analyst">Data Analyst</option>
                    <option value="document">Document</option>
                    <option value="code_assistant">Code Assistant</option>
                    <option value="rag">RAG</option>
                    <option value="document_intelligence">Document Intelligence</option>
                </optgroup>
                <optgroup label="DeepAgents">
                    <option value="it_operations">IT Operations</option>
                    <option value="sales_intelligence">Sales Intelligence</option>
                    <option value="recruitment">Recruitment</option>
                </optgroup>
            </select>
        </div>
        <div class="chat-container" id="chat-container"></div>
        <div class="input-container">
            <input type="text" id="message-input" placeholder="Type your message..." onkeypress="if(event.key==='Enter')sendMessage()">
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
        <script>
            let sessionId = crypto.randomUUID();
            
            async function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                if (!message) return;
                
                const agent = document.getElementById('agent-select').value;
                const container = document.getElementById('chat-container');
                
                // Add user message
                container.innerHTML += `<div class="message user">${message}</div>`;
                input.value = '';
                
                // Determine endpoint based on agent type
                let endpoint;
                const agentSelect = document.getElementById('agent-select');
                const selectedOption = agentSelect.options[agentSelect.selectedIndex];
                const optgroup = selectedOption.parentElement.label;
                
                if (optgroup === 'IT Agents') {
                    endpoint = `/api/it/${agent}/chat`;
                } else if (optgroup === 'Enterprise Agents') {
                    endpoint = `/api/enterprise/${agent}/chat`;
                } else {
                    endpoint = `/api/deepagent/${agent}/chat`;
                }
                
                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message, session_id: sessionId })
                    });
                    const data = await response.json();
                    container.innerHTML += `<div class="message assistant">${data.response}</div>`;
                } catch (error) {
                    container.innerHTML += `<div class="message assistant" style="color: red;">Error: ${error.message}</div>`;
                }
                
                container.scrollTop = container.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# IT Agent endpoints
@app.post("/api/conversation/{agent_name}", response_model=ChatResponse)
@app.post("/api/it/{agent_name}/chat", response_model=ChatResponse)
async def it_agent_chat(agent_name: str, request: ChatRequest):
    """Chat with an IT agent."""
    agent = registry.get_it_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"IT agent '{agent_name}' not found")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        response = agent.chat(request.message, thread_id=session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            agent_type=f"IT/{getattr(agent, 'agent_subtype', 'unknown')}",
            timestamp=datetime.utcnow().isoformat(),
            metadata=request.metadata,
        )
    except Exception as e:
        logger.error(f"Error in IT agent chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Enterprise Agent endpoints
@app.post("/api/enterprise/{agent_name}/chat", response_model=ChatResponse)
async def enterprise_agent_chat(agent_name: str, request: ChatRequest):
    """Chat with an enterprise agent."""
    agent = registry.get_enterprise_agent(agent_name)
    if not agent:
        raise HTTPException(
            status_code=404, detail=f"Enterprise agent '{agent_name}' not found"
        )

    session_id = request.session_id or str(uuid.uuid4())

    try:
        response = agent.chat(request.message, thread_id=session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            agent_type=f"Enterprise/{getattr(agent, 'agent_subtype', 'unknown')}",
            timestamp=datetime.utcnow().isoformat(),
            metadata=request.metadata,
        )
    except Exception as e:
        logger.error(f"Error in Enterprise agent chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoints."""

    query: str = Field(..., description="Analysis query")
    context: Optional[str] = Field(None, description="Context for analysis")
    output_format: str = Field("text", description="Output format")
    session_id: Optional[str] = Field(None, description="Session ID")


@app.post("/api/enterprise/{agent_name}/analyze")
async def enterprise_agent_analyze(agent_name: str, request: AnalyzeRequest):
    """Perform analysis with an enterprise agent."""
    agent = registry.get_enterprise_agent(agent_name)
    if not agent:
        raise HTTPException(
            status_code=404, detail=f"Enterprise agent '{agent_name}' not found"
        )

    try:
        result = agent.analyze(
            query=request.query,
            context=request.context,
            output_format=request.output_format,
            thread_id=request.session_id,
        )
        return result
    except Exception as e:
        logger.error(f"Error in Enterprise agent analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# DeepAgent endpoints
@app.post("/api/deepagent/{agent_name}/chat", response_model=ChatResponse)
async def deep_agent_chat(agent_name: str, request: ChatRequest):
    """Chat with a DeepAgent."""
    agent = registry.get_deep_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"DeepAgent '{agent_name}' not found")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        response = agent.chat(request.message, thread_id=session_id)
        return ChatResponse(
            response=response,
            session_id=session_id,
            agent_type=f"DeepAgent/{getattr(agent, 'agent_subtype', 'unknown')}",
            timestamp=datetime.utcnow().isoformat(),
            metadata=request.metadata,
        )
    except Exception as e:
        logger.error(f"Error in DeepAgent chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""

    task: str = Field(..., description="Task to execute")
    thread_id: Optional[str] = Field(None, description="Thread ID")
    max_iterations: int = Field(10, description="Maximum iterations")


@app.post("/api/deepagent/{agent_name}/execute")
async def deep_agent_execute(agent_name: str, request: WorkflowRequest):
    """Execute a workflow with a DeepAgent."""
    agent = registry.get_deep_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"DeepAgent '{agent_name}' not found")

    try:
        result = agent.execute_workflow(
            task=request.task,
            thread_id=request.thread_id,
            max_iterations=request.max_iterations,
        )
        return result
    except Exception as e:
        logger.error(f"Error in DeepAgent workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    return app


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(app, host=host, port=port)
