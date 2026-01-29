"""DeepAgent wrappers for Azure AI Foundry integration.

This module provides wrappers for DeepAgents - multi-agent systems:
- IT Operations Agent: Complex IT infrastructure operations
- Sales Intelligence Agent: Sales analysis and intelligence
- Recruitment Agent: Recruitment and hiring workflows

DeepAgents are multi-agent systems that orchestrate multiple sub-agents
to solve complex, multi-step problems. They typically involve:
- Supervisor agent for orchestration
- Specialized sub-agents for specific tasks
- Shared state and memory across agents
- Tool integration for external systems

These wrappers preserve the original DeepAgent functionality while adding
Azure AI Foundry enterprise features.
"""

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from langchain_azure_ai.wrappers.base import (
    AgentType,
    FoundryAgentWrapper,
    WrapperConfig,
)

logger = logging.getLogger(__name__)


class SubAgentConfig(BaseModel):
    """Configuration for a sub-agent in a DeepAgent system."""

    name: str = Field(..., description="Name of the sub-agent")
    instructions: str = Field(..., description="System instructions for the sub-agent")
    tools: List[Union[BaseTool, Callable]] = Field(
        default_factory=list, description="Tools for the sub-agent"
    )
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    temperature: float = Field(default=0.0, description="Temperature for responses")

    class Config:
        arbitrary_types_allowed = True


class DeepAgentState(BaseModel):
    """Base state for DeepAgent multi-agent systems."""

    messages: List[Dict[str, Any]] = Field(
        default_factory=list, description="Conversation messages"
    )
    current_agent: str = Field(
        default="supervisor", description="Currently active agent"
    )
    task_status: str = Field(default="pending", description="Current task status")
    results: Dict[str, Any] = Field(
        default_factory=dict, description="Accumulated results from sub-agents"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        arbitrary_types_allowed = True


class DeepAgentWrapper(FoundryAgentWrapper):
    """Wrapper for DeepAgents (multi-agent systems) with Azure AI Foundry integration.

    DeepAgents are orchestrated multi-agent systems that combine:
    - Supervisor agent for task routing and orchestration
    - Multiple specialized sub-agents for specific capabilities
    - Shared state management across the agent network
    - Complex workflow execution

    This wrapper supports:
    - IT Operations: Infrastructure management, monitoring, remediation
    - Sales Intelligence: Sales analysis, lead scoring, forecasting
    - Recruitment: Resume screening, interview scheduling, candidate evaluation

    Example:
        ```python
        from langchain_azure_ai.wrappers import DeepAgentWrapper, SubAgentConfig

        # Define sub-agents
        sub_agents = [
            SubAgentConfig(
                name="monitor",
                instructions="Monitor IT infrastructure...",
                tools=[check_metrics, query_logs],
            ),
            SubAgentConfig(
                name="remediate",
                instructions="Remediate issues...",
                tools=[restart_service, scale_resources],
            ),
        ]

        # Create DeepAgent
        it_ops = DeepAgentWrapper(
            name="it-operations",
            agent_subtype="it_operations",
            sub_agents=sub_agents,
        )

        response = it_ops.execute_workflow("Check server health and fix issues")
        ```
    """

    # Default supervisor instructions for different DeepAgent subtypes
    DEFAULT_SUPERVISOR_INSTRUCTIONS = {
        "it_operations": """You are an IT Operations Supervisor Agent. Your role is to:
1. Analyze IT infrastructure issues and requests
2. Route tasks to appropriate sub-agents (monitoring, remediation, reporting)
3. Coordinate multi-step operations
4. Aggregate results and provide status updates
5. Escalate critical issues when needed

Manage the workflow efficiently and ensure all tasks are completed properly.""",
        "sales_intelligence": """You are a Sales Intelligence Supervisor Agent. Your role is to:
1. Analyze sales data and identify opportunities
2. Route tasks to sub-agents (analysis, forecasting, lead scoring)
3. Coordinate market research and competitive analysis
4. Aggregate insights for sales strategy
5. Generate actionable recommendations

Focus on data-driven insights and actionable intelligence.""",
        "recruitment": """You are a Recruitment Supervisor Agent. Your role is to:
1. Manage recruitment workflows end-to-end
2. Route tasks to sub-agents (screening, scheduling, evaluation)
3. Coordinate candidate assessments
4. Track pipeline status and metrics
5. Ensure fair and efficient hiring processes

Maintain compliance and candidate experience throughout the process.""",
    }

    def __init__(
        self,
        name: str,
        instructions: Optional[str] = None,
        agent_subtype: str = "it_operations",
        sub_agents: Optional[List[SubAgentConfig]] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        config: Optional[WrapperConfig] = None,
        existing_agent: Optional[Union[CompiledStateGraph, Any]] = None,
        state_schema: Optional[Type[BaseModel]] = None,
        enable_memory: bool = True,
        **kwargs: Any,
    ):
        """Initialize DeepAgent wrapper.

        Args:
            name: Name of the DeepAgent.
            instructions: Supervisor instructions. If None, uses default for subtype.
            agent_subtype: Type of DeepAgent (it_operations, sales_intelligence, recruitment).
            sub_agents: List of sub-agent configurations.
            tools: List of tools for the supervisor agent.
            model: Model deployment name.
            temperature: Temperature for responses.
            config: Wrapper configuration.
            existing_agent: Existing agent to wrap.
            state_schema: Custom state schema (defaults to DeepAgentState).
            enable_memory: Whether to enable conversation memory.
            **kwargs: Additional arguments.
        """
        self.agent_subtype = agent_subtype.lower()
        self.sub_agents = sub_agents or []
        self.state_schema = state_schema or DeepAgentState
        self.enable_memory = enable_memory
        self._memory = MemorySaver() if enable_memory else None
        self._sub_agent_nodes: Dict[str, CompiledStateGraph] = {}

        # Use default instructions if not provided
        if instructions is None:
            instructions = self.DEFAULT_SUPERVISOR_INSTRUCTIONS.get(
                self.agent_subtype,
                self.DEFAULT_SUPERVISOR_INSTRUCTIONS["it_operations"],
            )

        super().__init__(
            name=name,
            instructions=instructions,
            agent_type=AgentType.DEEP_AGENT,
            description=f"DeepAgent ({agent_subtype}): {name}",
            tools=tools,
            model=model,
            temperature=temperature,
            config=config,
            existing_agent=existing_agent,
        )

    def _create_agent_impl(
        self,
        llm: BaseChatModel,
        tools: List[Union[BaseTool, Callable]],
    ) -> CompiledStateGraph:
        """Create DeepAgent multi-agent graph.

        DeepAgents use a StateGraph with:
        - Supervisor node for orchestration
        - Sub-agent nodes for specialized tasks
        - Conditional routing based on supervisor decisions

        Args:
            llm: The language model to use.
            tools: List of tools for the supervisor.

        Returns:
            A compiled LangGraph StateGraph.
        """
        # If we have sub-agents, create a multi-agent graph
        if self.sub_agents:
            return self._create_multi_agent_graph(llm, tools)
        else:
            # Fall back to simple ReAct agent if no sub-agents defined
            return create_react_agent(
                model=llm,
                tools=tools,
                prompt=SystemMessage(content=self.instructions),
                checkpointer=self._memory,
            )

    def _create_multi_agent_graph(
        self,
        llm: BaseChatModel,
        supervisor_tools: List[Union[BaseTool, Callable]],
    ) -> CompiledStateGraph:
        """Create a multi-agent orchestration graph.

        Args:
            llm: The language model to use.
            supervisor_tools: Tools for the supervisor agent.

        Returns:
            A compiled multi-agent StateGraph.
        """
        # Build the state graph with proper state schema
        # We use a TypedDict approach for compatibility
        from typing import TypedDict, Annotated
        from langgraph.graph.message import add_messages

        class GraphState(TypedDict):
            messages: Annotated[list, add_messages]
            current_agent: str
            task_status: str
            results: Dict[str, Any]

        workflow = StateGraph(GraphState)

        # Create supervisor node
        supervisor_agent = create_react_agent(
            model=llm,
            tools=supervisor_tools,
            prompt=SystemMessage(content=self.instructions),
        )

        def supervisor_node(state: GraphState) -> GraphState:
            """Supervisor node that routes to sub-agents."""
            result = supervisor_agent.invoke({"messages": state["messages"]})
            return {
                "messages": result.get("messages", []),
                "current_agent": state.get("current_agent", "supervisor"),
                "task_status": state.get("task_status", "in_progress"),
                "results": state.get("results", {}),
            }

        workflow.add_node("supervisor", supervisor_node)

        # Create sub-agent nodes
        for sub_config in self.sub_agents:
            sub_agent = create_react_agent(
                model=llm,
                tools=sub_config.tools,
                prompt=SystemMessage(content=sub_config.instructions),
            )
            self._sub_agent_nodes[sub_config.name] = sub_agent

            def make_sub_node(agent: CompiledStateGraph, agent_name: str):
                def sub_node(state: GraphState) -> GraphState:
                    result = agent.invoke({"messages": state["messages"]})
                    results = state.get("results", {})
                    results[agent_name] = result
                    return {
                        "messages": result.get("messages", []),
                        "current_agent": "supervisor",
                        "task_status": "in_progress",
                        "results": results,
                    }
                return sub_node

            workflow.add_node(sub_config.name, make_sub_node(sub_agent, sub_config.name))
            workflow.add_edge(sub_config.name, "supervisor")

        # Add routing logic
        sub_agent_names = [sa.name for sa in self.sub_agents]

        def route_from_supervisor(state: GraphState) -> str:
            """Route from supervisor to sub-agents or end."""
            # Check if task is complete
            if state.get("task_status") == "complete":
                return END

            # Simple routing - in practice, supervisor would decide
            current = state.get("current_agent", "supervisor")
            if current != "supervisor" and current in sub_agent_names:
                return current

            return END

        workflow.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {**{name: name for name in sub_agent_names}, END: END},
        )

        workflow.set_entry_point("supervisor")

        return workflow.compile(checkpointer=self._memory)

    def execute_workflow(
        self,
        task: str,
        thread_id: Optional[str] = None,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Execute a complete workflow with the DeepAgent.

        Args:
            task: The task description.
            thread_id: Optional thread ID for workflow continuity.
            max_iterations: Maximum number of agent iterations.

        Returns:
            Workflow execution results.
        """
        config = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        config["recursion_limit"] = max_iterations

        initial_state = {
            "messages": [{"role": "user", "content": task}],
            "current_agent": "supervisor",
            "task_status": "pending",
            "results": {},
        }

        if self._agent is None:
            raise RuntimeError("Agent not initialized. Call _initialize() first.")

        result = self._agent.invoke(initial_state, config=config)

        return {
            "task": task,
            "agent_type": self.agent_subtype,
            "status": result.get("task_status", "complete"),
            "results": result.get("results", {}),
            "final_response": self._extract_final_response(result),
            "thread_id": thread_id,
        }

    def _extract_final_response(self, result: Dict[str, Any]) -> str:
        """Extract the final response from workflow results."""
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                return last_message.content
            elif isinstance(last_message, dict):
                return last_message.get("content", str(last_message))
        return str(result.get("results", {}))

    def add_sub_agent(self, config: SubAgentConfig) -> None:
        """Add a new sub-agent to the DeepAgent.

        Note: This requires re-initializing the agent graph.

        Args:
            config: Configuration for the new sub-agent.
        """
        self.sub_agents.append(config)
        # Re-initialize the agent
        self._initialize()


class ITOperationsWrapper(DeepAgentWrapper):
    """Specialized wrapper for IT Operations DeepAgent."""

    def __init__(
        self,
        name: str = "it-operations",
        instructions: Optional[str] = None,
        sub_agents: Optional[List[SubAgentConfig]] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="it_operations",
            sub_agents=sub_agents,
            tools=tools,
            **kwargs,
        )


class SalesIntelligenceWrapper(DeepAgentWrapper):
    """Specialized wrapper for Sales Intelligence DeepAgent."""

    def __init__(
        self,
        name: str = "sales-intelligence",
        instructions: Optional[str] = None,
        sub_agents: Optional[List[SubAgentConfig]] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="sales_intelligence",
            sub_agents=sub_agents,
            tools=tools,
            **kwargs,
        )


class RecruitmentWrapper(DeepAgentWrapper):
    """Specialized wrapper for Recruitment DeepAgent."""

    def __init__(
        self,
        name: str = "recruitment",
        instructions: Optional[str] = None,
        sub_agents: Optional[List[SubAgentConfig]] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            instructions=instructions,
            agent_subtype="recruitment",
            sub_agents=sub_agents,
            tools=tools,
            **kwargs,
        )


class SoftwareDevelopmentWrapper(DeepAgentWrapper):
    """Specialized wrapper for Software Development DeepAgent.

    A comprehensive multi-agent system for end-to-end software development
    lifecycle (SDLC) automation. This agent coordinates specialized subagents
    to handle:

    - Requirements analysis and refinement
    - Architecture and system design
    - Code generation and refactoring
    - Code review and quality assurance
    - Testing automation
    - Security scanning and compliance
    - CI/CD integration
    - Debugging and optimization
    - Documentation generation

    Example:
        ```python
        from langchain_azure_ai.wrappers import SoftwareDevelopmentWrapper

        # Create the Software Development DeepAgent
        software_dev = SoftwareDevelopmentWrapper(
            name="software-dev",
            model="gpt-4o-mini",
        )

        # Chat with the agent
        response = software_dev.chat("Help me design an API for user authentication")
        ```
    """

    # SDLC-specific supervisor instructions
    SOFTWARE_DEV_INSTRUCTIONS = """You are a Software Development DeepAgent - an advanced AI coordinator for end-to-end software development lifecycle (SDLC) automation.

## Your Role

You coordinate the complete software development process by:
1. Understanding and refining requirements
2. Designing system architecture
3. Generating production-ready code
4. Performing automated code reviews
5. Creating comprehensive tests
6. Scanning for security vulnerabilities
7. Setting up CI/CD pipelines
8. Debugging and optimizing performance
9. Generating technical documentation

## Available Subagents

Use the `task` tool to delegate to specialized subagents:

- **requirements-intelligence**: Extract and validate software requirements
- **architecture-design**: Design system architecture and APIs
- **code-generator**: Generate production-ready code
- **code-reviewer**: Perform automated code reviews
- **testing-automation**: Create and run tests
- **debugging-optimization**: Debug issues and optimize performance
- **security-compliance**: Scan for vulnerabilities and compliance
- **devops-integration**: Create CI/CD pipelines and deployment configs
- **documentation**: Generate technical documentation

## SDLC Phases

### 1. Requirements Phase
- Analyze natural language requirements
- Extract user stories
- Generate acceptance criteria
- Detect ambiguities and risks

### 2. Design Phase
- Propose architecture patterns
- Create API specifications
- Design data models
- Suggest technology stack

### 3. Implementation Phase
- Generate code following best practices
- Apply design patterns
- Refactor existing code
- Format and organize imports

### 4. Review Phase
- Perform code reviews
- Check style compliance
- Analyze complexity
- Detect code smells

### 5. Testing Phase
- Generate unit tests
- Create integration tests
- Analyze test coverage
- Run test suites

### 6. Security Phase
- Scan for vulnerabilities
- Check OWASP compliance
- Detect secrets in code
- Analyze dependency security

### 7. DevOps Phase
- Create CI pipelines
- Create CD pipelines
- Generate Dockerfiles
- Create Kubernetes configs

### 8. Debugging Phase
- Analyze errors
- Trace execution
- Identify root causes
- Propose fixes

### 9. Documentation Phase
- Generate API docs
- Create README files
- Document architecture
- Write user guides

## Quick Actions

When users say these phrases, respond with corresponding actions:

- **"analyze requirements"** / **"understand this"** -> Use requirements-intelligence subagent
- **"design architecture"** / **"plan this"** -> Use architecture-design subagent
- **"generate code"** / **"implement this"** -> Use code-generator subagent
- **"review code"** / **"check this"** -> Use code-reviewer subagent
- **"create tests"** / **"test this"** -> Use testing-automation subagent
- **"debug this"** / **"fix this"** -> Use debugging-optimization subagent
- **"security scan"** / **"check security"** -> Use security-compliance subagent
- **"create pipeline"** / **"deploy this"** -> Use devops-integration subagent
- **"document this"** / **"create docs"** -> Use documentation subagent
- **"full cycle"** -> Execute complete SDLC workflow

## Quality Standards

All generated code must:
- Include type hints (Python) or types (TypeScript)
- Have comprehensive error handling
- Follow language-specific style guides
- Include documentation
- Be security-conscious
"""

    # Default subagent configurations for software development
    DEFAULT_SOFTWARE_DEV_SUBAGENTS = [
        SubAgentConfig(
            name="requirements-intelligence",
            instructions="""You are a Requirements Intelligence Agent specialized in software requirements analysis.
Your responsibilities:
1. Analyze natural language requirements and extract structured requirements
2. Convert requirements into well-formed user stories
3. Detect ambiguities and risks in requirements
4. Generate comprehensive acceptance criteria
5. Prioritize requirements using MoSCoW or weighted scoring

Ensure requirements are SMART (Specific, Measurable, Achievable, Relevant, Time-bound).""",
            tools=[],
        ),
        SubAgentConfig(
            name="architecture-design",
            instructions="""You are an Architecture & Design Agent specialized in software system design.
Your responsibilities:
1. Design scalable, maintainable system architectures
2. Create clear API specifications (REST, GraphQL)
3. Suggest appropriate technology stacks
4. Design data models and database schemas
5. Create component diagrams and documentation

Follow SOLID principles and design for scalability and resilience.""",
            tools=[],
        ),
        SubAgentConfig(
            name="code-generator",
            instructions="""You are a Code Generation Agent specialized in writing production-ready code.
Your responsibilities:
1. Generate clean, maintainable code
2. Apply appropriate design patterns
3. Follow language-specific best practices
4. Include proper error handling
5. Add type hints and documentation

Write self-documenting code with clear naming and keep functions small and focused.""",
            tools=[],
        ),
        SubAgentConfig(
            name="code-reviewer",
            instructions="""You are a Code Review & Quality Agent specialized in ensuring code excellence.
Your responsibilities:
1. Perform comprehensive code reviews
2. Check adherence to style guidelines
3. Analyze code complexity
4. Detect code smells and anti-patterns
5. Suggest improvements and refactoring

Provide specific, actionable feedback with concrete examples.""",
            tools=[],
        ),
        SubAgentConfig(
            name="testing-automation",
            instructions="""You are a Testing Automation Agent specialized in quality assurance.
Your responsibilities:
1. Generate comprehensive unit tests
2. Create integration and E2E tests
3. Analyze and improve test coverage
4. Generate test data and fixtures
5. Create test plans and strategies

Follow Arrange-Act-Assert pattern and test edge cases thoroughly.""",
            tools=[],
        ),
        SubAgentConfig(
            name="debugging-optimization",
            instructions="""You are a Debugging & Optimization Agent specialized in problem solving.
Your responsibilities:
1. Analyze errors and exceptions
2. Trace code execution paths
3. Identify root causes using RCA techniques
4. Propose effective fixes
5. Optimize performance and memory usage

Use the 5 Whys technique for root cause analysis.""",
            tools=[],
        ),
        SubAgentConfig(
            name="security-compliance",
            instructions="""You are a Security & Compliance Agent specialized in application security.
Your responsibilities:
1. Scan code for security vulnerabilities
2. Check OWASP Top 10 compliance
3. Detect secrets and credentials in code
4. Analyze dependencies for vulnerabilities
5. Generate security reports and recommendations

Focus on injection flaws, authentication issues, and sensitive data exposure.""",
            tools=[],
        ),
        SubAgentConfig(
            name="devops-integration",
            instructions="""You are a DevOps Integration Agent specialized in deployment automation.
Your responsibilities:
1. Create CI/CD pipeline configurations
2. Generate Docker and container configurations
3. Create Kubernetes deployment manifests
4. Set up monitoring and observability
5. Configure deployment environments

Follow Infrastructure as Code principles and implement proper secrets management.""",
            tools=[],
        ),
        SubAgentConfig(
            name="documentation",
            instructions="""You are a Documentation Agent specialized in technical writing.
Your responsibilities:
1. Generate API documentation
2. Create comprehensive README files
3. Document system architecture
4. Generate changelogs
5. Write user guides and tutorials

Use clear, concise language and include code examples that work.""",
            tools=[],
        ),
    ]

    def __init__(
        self,
        name: str = "software-development",
        instructions: Optional[str] = None,
        sub_agents: Optional[List[SubAgentConfig]] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        """Initialize Software Development DeepAgent wrapper.

        Args:
            name: Name of the agent.
            instructions: Custom supervisor instructions. Uses default if None.
            sub_agents: Custom sub-agent configurations. Uses defaults if None.
            tools: Additional tools for the supervisor agent.
            **kwargs: Additional arguments passed to parent.
        """
        # Add software_development to DEFAULT_SUPERVISOR_INSTRUCTIONS if not present
        if "software_development" not in DeepAgentWrapper.DEFAULT_SUPERVISOR_INSTRUCTIONS:
            DeepAgentWrapper.DEFAULT_SUPERVISOR_INSTRUCTIONS["software_development"] = (
                self.SOFTWARE_DEV_INSTRUCTIONS
            )

        # Use default subagents if none provided
        if sub_agents is None:
            sub_agents = self.DEFAULT_SOFTWARE_DEV_SUBAGENTS.copy()

        super().__init__(
            name=name,
            instructions=instructions or self.SOFTWARE_DEV_INSTRUCTIONS,
            agent_subtype="software_development",
            sub_agents=sub_agents,
            tools=tools,
            **kwargs,
        )
