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
    """Specialized wrapper for IT Operations DeepAgent.

    A multi-agent system designed for enterprise IT infrastructure management
    and operations automation. This agent coordinates specialized subagents
    to handle:

    - Infrastructure monitoring and health checks
    - Incident detection and alerting
    - Automated remediation and self-healing
    - Capacity planning and resource optimization
    - Change management and deployment coordination
    - Security monitoring and compliance
    - Performance analysis and optimization
    - Log analysis and troubleshooting

    The IT Operations DeepAgent follows ITIL best practices and integrates
    with common IT management tools and platforms.

    Example:
        ```python
        from langchain_azure_ai.wrappers import ITOperationsWrapper, SubAgentConfig

        # Create with default configuration
        it_ops = ITOperationsWrapper(
            name="enterprise-it-ops",
            model="gpt-4o-mini",
        )

        # Or with custom subagents for specific infrastructure
        monitoring_agent = SubAgentConfig(
            name="monitor",
            instructions="Monitor infrastructure health...",
            tools=[check_metrics, query_prometheus],
        )

        remediation_agent = SubAgentConfig(
            name="remediate",
            instructions="Execute automated fixes...",
            tools=[restart_service, scale_resources],
        )

        it_ops = ITOperationsWrapper(
            name="custom-it-ops",
            sub_agents=[monitoring_agent, remediation_agent],
        )

        # Execute a workflow
        response = it_ops.execute_workflow(
            "Check all production servers and remediate any issues"
        )
        ```

    Attributes:
        agent_subtype: Always "it_operations" for this wrapper.
        sub_agents: List of SubAgentConfig for specialized IT operations.
    """

    # Extended instructions for IT Operations
    IT_OPS_INSTRUCTIONS = """You are an IT Operations Supervisor Agent managing enterprise infrastructure.

## Your Responsibilities

1. **Infrastructure Monitoring**
   - Analyze system metrics, logs, and alerts
   - Identify performance bottlenecks and anomalies
   - Track service health and availability

2. **Incident Management**
   - Detect and classify incidents by severity
   - Coordinate response across teams
   - Document incidents and resolutions

3. **Automated Remediation**
   - Execute predefined runbooks for known issues
   - Coordinate service restarts and failovers
   - Scale resources based on demand

4. **Change Management**
   - Review and approve change requests
   - Coordinate deployment windows
   - Track change implementation and rollbacks

5. **Capacity Planning**
   - Analyze resource utilization trends
   - Forecast future capacity needs
   - Recommend optimization strategies

## ITIL Alignment

Follow ITIL practices for:
- Incident Management
- Problem Management
- Change Management
- Service Level Management

## Escalation Protocol

- P1 (Critical): Immediate escalation, all-hands response
- P2 (High): 15-minute response, on-call engagement
- P3 (Medium): 1-hour response, normal workflow
- P4 (Low): Next business day, scheduled maintenance

Always prioritize service availability and minimize mean time to recovery (MTTR)."""

    def __init__(
        self,
        name: str = "it-operations",
        instructions: Optional[str] = None,
        sub_agents: Optional[List[SubAgentConfig]] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        """Initialize IT Operations DeepAgent wrapper.

        Args:
            name: Name of the IT Operations agent.
            instructions: Custom supervisor instructions. If None, uses IT_OPS_INSTRUCTIONS.
            sub_agents: List of sub-agent configurations for specialized tasks.
            tools: List of tools for the supervisor agent.
            **kwargs: Additional arguments passed to parent.
        """
        # Register extended instructions (override default)
        DeepAgentWrapper.DEFAULT_SUPERVISOR_INSTRUCTIONS["it_operations"] = (
            self.IT_OPS_INSTRUCTIONS
        )

        super().__init__(
            name=name,
            instructions=instructions or self.IT_OPS_INSTRUCTIONS,
            agent_subtype="it_operations",
            sub_agents=sub_agents,
            tools=tools,
            **kwargs,
        )


class SalesIntelligenceWrapper(DeepAgentWrapper):
    """Specialized wrapper for Sales Intelligence DeepAgent.

    A multi-agent system designed for sales analytics, intelligence gathering,
    and revenue optimization. This agent coordinates specialized subagents
    to handle:

    - Lead scoring and qualification
    - Sales pipeline analysis and forecasting
    - Customer segmentation and targeting
    - Competitive intelligence gathering
    - Market trend analysis
    - Deal risk assessment
    - Revenue optimization recommendations
    - Sales performance analytics

    The Sales Intelligence DeepAgent integrates with CRM systems and provides
    data-driven insights for sales strategy optimization.

    Example:
        ```python
        from langchain_azure_ai.wrappers import SalesIntelligenceWrapper, SubAgentConfig

        # Create with default configuration
        sales_intel = SalesIntelligenceWrapper(
            name="enterprise-sales-intel",
            model="gpt-4o-mini",
        )

        # Or with custom subagents for specific sales processes
        lead_scorer = SubAgentConfig(
            name="lead-scoring",
            instructions="Score and qualify leads based on engagement...",
            tools=[analyze_engagement, check_firmographics],
        )

        forecaster = SubAgentConfig(
            name="forecaster",
            instructions="Generate sales forecasts...",
            tools=[analyze_pipeline, predict_revenue],
        )

        sales_intel = SalesIntelligenceWrapper(
            name="custom-sales-intel",
            sub_agents=[lead_scorer, forecaster],
        )

        # Execute a workflow
        response = sales_intel.execute_workflow(
            "Analyze Q4 pipeline and identify at-risk deals"
        )
        ```

    Attributes:
        agent_subtype: Always "sales_intelligence" for this wrapper.
        sub_agents: List of SubAgentConfig for specialized sales operations.
    """

    # Extended instructions for Sales Intelligence
    SALES_INTEL_INSTRUCTIONS = """You are a Sales Intelligence Supervisor Agent for enterprise sales optimization.

## Your Responsibilities

1. **Lead Management**
   - Score and qualify incoming leads
   - Prioritize leads based on fit and intent signals
   - Recommend optimal follow-up strategies

2. **Pipeline Analysis**
   - Analyze deal progression and velocity
   - Identify bottlenecks in the sales funnel
   - Flag at-risk opportunities

3. **Forecasting**
   - Generate revenue forecasts by segment/region
   - Analyze forecast accuracy and adjust models
   - Identify upside and downside risks

4. **Competitive Intelligence**
   - Track competitor activities and positioning
   - Analyze win/loss patterns against competitors
   - Provide battlecard recommendations

5. **Customer Insights**
   - Segment customers by value and behavior
   - Identify expansion opportunities
   - Predict churn risk

## Sales Methodology Alignment

Support common sales methodologies:
- MEDDIC/MEDDPICC qualification
- Challenger Sale approach
- Solution Selling
- SPIN Selling

## Key Metrics Focus

- Win Rate by segment
- Average Deal Size
- Sales Cycle Length
- Pipeline Coverage Ratio
- Customer Acquisition Cost (CAC)
- Customer Lifetime Value (CLV)

Always provide actionable insights backed by data analysis."""

    def __init__(
        self,
        name: str = "sales-intelligence",
        instructions: Optional[str] = None,
        sub_agents: Optional[List[SubAgentConfig]] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        """Initialize Sales Intelligence DeepAgent wrapper.

        Args:
            name: Name of the Sales Intelligence agent.
            instructions: Custom supervisor instructions. If None, uses SALES_INTEL_INSTRUCTIONS.
            sub_agents: List of sub-agent configurations for specialized tasks.
            tools: List of tools for the supervisor agent.
            **kwargs: Additional arguments passed to parent.
        """
        # Register extended instructions (override default)
        DeepAgentWrapper.DEFAULT_SUPERVISOR_INSTRUCTIONS["sales_intelligence"] = (
            self.SALES_INTEL_INSTRUCTIONS
        )

        super().__init__(
            name=name,
            instructions=instructions or self.SALES_INTEL_INSTRUCTIONS,
            agent_subtype="sales_intelligence",
            sub_agents=sub_agents,
            tools=tools,
            **kwargs,
        )


class RecruitmentWrapper(DeepAgentWrapper):
    """Specialized wrapper for Recruitment DeepAgent.

    A multi-agent system designed for end-to-end recruitment and talent
    acquisition automation. This agent coordinates specialized subagents
    to handle:

    - Resume screening and parsing
    - Candidate matching and ranking
    - Interview scheduling and coordination
    - Skill assessment and evaluation
    - Background check orchestration
    - Offer management and negotiation support
    - Onboarding workflow initiation
    - Diversity and inclusion analytics

    The Recruitment DeepAgent integrates with ATS (Applicant Tracking Systems)
    and ensures fair, compliant, and efficient hiring processes.

    Example:
        ```python
        from langchain_azure_ai.wrappers import RecruitmentWrapper, SubAgentConfig

        # Create with default configuration
        recruiter = RecruitmentWrapper(
            name="enterprise-recruiter",
            model="gpt-4o-mini",
        )

        # Or with custom subagents for specific recruitment workflows
        screener = SubAgentConfig(
            name="screener",
            instructions="Screen resumes against job requirements...",
            tools=[parse_resume, match_skills],
        )

        scheduler = SubAgentConfig(
            name="scheduler",
            instructions="Coordinate interview schedules...",
            tools=[check_availability, send_invite],
        )

        recruiter = RecruitmentWrapper(
            name="custom-recruiter",
            sub_agents=[screener, scheduler],
        )

        # Execute a workflow
        response = recruiter.execute_workflow(
            "Screen all applicants for Senior Engineer role and schedule top 5"
        )
        ```

    Attributes:
        agent_subtype: Always "recruitment" for this wrapper.
        sub_agents: List of SubAgentConfig for specialized recruitment tasks.
    """

    # Extended instructions for Recruitment
    RECRUITMENT_INSTRUCTIONS = """You are a Recruitment Supervisor Agent managing enterprise talent acquisition.

## Your Responsibilities

1. **Resume Screening**
   - Parse and analyze candidate resumes
   - Match candidates to job requirements
   - Rank candidates by fit score

2. **Candidate Evaluation**
   - Assess technical and soft skills
   - Coordinate skill assessments and tests
   - Compile evaluation summaries

3. **Interview Coordination**
   - Schedule interviews across time zones
   - Prepare interviewers with candidate briefs
   - Collect and consolidate feedback

4. **Pipeline Management**
   - Track candidates through stages
   - Identify bottlenecks and delays
   - Maintain communication cadence

5. **Compliance & Fairness**
   - Ensure EEOC/diversity compliance
   - Remove bias from screening processes
   - Maintain audit trails

## Recruitment Stages

1. Sourcing & Application
2. Initial Screening
3. Phone/Video Screen
4. Technical Assessment
5. On-site/Panel Interview
6. Reference Check
7. Offer & Negotiation
8. Onboarding

## Key Metrics

- Time to Fill
- Cost per Hire
- Quality of Hire
- Offer Acceptance Rate
- Candidate Experience Score
- Source of Hire effectiveness
- Diversity metrics by stage

## Compliance Focus

- Fair hiring practices (no discrimination)
- Data privacy (GDPR, CCPA)
- Background check regulations
- Equal opportunity documentation

Always prioritize candidate experience and maintain fair, unbiased evaluation."""

    def __init__(
        self,
        name: str = "recruitment",
        instructions: Optional[str] = None,
        sub_agents: Optional[List[SubAgentConfig]] = None,
        tools: Optional[Sequence[Union[BaseTool, Callable]]] = None,
        **kwargs: Any,
    ):
        """Initialize Recruitment DeepAgent wrapper.

        Args:
            name: Name of the Recruitment agent.
            instructions: Custom supervisor instructions. If None, uses RECRUITMENT_INSTRUCTIONS.
            sub_agents: List of sub-agent configurations for specialized tasks.
            tools: List of tools for the supervisor agent.
            **kwargs: Additional arguments passed to parent.
        """
        # Register extended instructions (override default)
        DeepAgentWrapper.DEFAULT_SUPERVISOR_INSTRUCTIONS["recruitment"] = (
            self.RECRUITMENT_INSTRUCTIONS
        )

        super().__init__(
            name=name,
            instructions=instructions or self.RECRUITMENT_INSTRUCTIONS,
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

## How to Handle Requests

You have access to specialized tools for each phase of the SDLC. Based on the user's request, use the appropriate tools directly:

**Requirements Phase Tools:**
- analyze_requirements, extract_user_stories, validate_requirements, prioritize_requirements, detect_ambiguities, generate_acceptance_criteria

**Architecture Phase Tools:**
- design_architecture, create_api_spec, suggest_tech_stack, design_data_model, create_component_diagram, analyze_dependencies

**Code Generation Tools:**
- generate_code, refactor_code, apply_design_pattern, generate_boilerplate, optimize_imports, format_code

**Code Review Tools:**
- review_code, check_code_style, analyze_complexity, detect_code_smells, suggest_improvements, check_best_practices

**Testing Tools:**
- generate_unit_tests, generate_integration_tests, analyze_test_coverage, run_tests, generate_test_data, create_test_plan

**Security Tools:**
- scan_security_issues, check_owasp_compliance, detect_secrets, analyze_dependencies_security, generate_security_report, suggest_security_fixes

**DevOps Tools:**
- create_ci_pipeline, create_cd_pipeline, configure_deployment, generate_dockerfile, create_kubernetes_config, setup_monitoring

**Debugging Tools:**
- analyze_error, trace_execution, identify_root_cause, propose_fix, analyze_performance, detect_memory_issues

**Documentation Tools:**
- generate_api_docs, create_readme, document_architecture, generate_changelog, add_inline_comments, create_user_guide

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

    @staticmethod
    def _get_default_subagents():
        """Get default subagent configurations with tools wired.

        This method is called lazily to avoid import issues.
        """
        # Import tools here to avoid circular imports
        from langchain_azure_ai.wrappers.software_dev_tools import (
            # Requirements tools
            analyze_requirements,
            extract_user_stories,
            validate_requirements,
            prioritize_requirements,
            detect_ambiguities,
            generate_acceptance_criteria,
            # Architecture tools
            design_architecture,
            create_api_spec,
            suggest_tech_stack,
            design_data_model,
            create_component_diagram,
            analyze_dependencies,
            # Code generation tools
            generate_code,
            refactor_code,
            apply_design_pattern,
            generate_boilerplate,
            optimize_imports,
            format_code,
            # Code review tools
            review_code,
            check_code_style,
            analyze_complexity,
            detect_code_smells,
            suggest_improvements,
            check_best_practices,
            # Testing tools
            generate_unit_tests,
            generate_integration_tests,
            analyze_test_coverage,
            run_tests,
            generate_test_data,
            create_test_plan,
            # Security tools
            scan_security_issues,
            check_owasp_compliance,
            detect_secrets,
            analyze_dependencies_security,
            generate_security_report,
            suggest_security_fixes,
            # DevOps tools
            create_ci_pipeline,
            create_cd_pipeline,
            configure_deployment,
            generate_dockerfile,
            create_kubernetes_config,
            setup_monitoring,
            # Debugging tools
            analyze_error,
            trace_execution,
            identify_root_cause,
            propose_fix,
            analyze_performance,
            detect_memory_issues,
            # Documentation tools
            generate_api_docs,
            create_readme,
            document_architecture,
            generate_changelog,
            add_inline_comments,
            create_user_guide,
        )

        return [
            SubAgentConfig(
                name="requirements-intelligence",
                instructions="""You are a Requirements Intelligence Agent specialized in software requirements analysis.
Your responsibilities:
1. Analyze natural language requirements and extract structured requirements
2. Convert requirements into well-formed user stories
3. Detect ambiguities and risks in requirements
4. Generate comprehensive acceptance criteria
5. Prioritize requirements using MoSCoW, weighted scoring, or Kano model

Ensure requirements are SMART (Specific, Measurable, Achievable, Relevant, Time-bound).""",
                tools=[
                    analyze_requirements,
                    extract_user_stories,
                    validate_requirements,
                    prioritize_requirements,
                    detect_ambiguities,
                    generate_acceptance_criteria,
                ],
            ),
            SubAgentConfig(
                name="architecture-design",
                instructions="""You are an Architecture & Design Agent specialized in software system design.
Your responsibilities:
1. Design scalable, maintainable system architectures
2. Create clear API specifications (REST, GraphQL, gRPC)
3. Suggest appropriate technology stacks
4. Design data models and database schemas
5. Create component diagrams and documentation

Follow SOLID principles and design for scalability and resilience.""",
                tools=[
                    design_architecture,
                    create_api_spec,
                    suggest_tech_stack,
                    design_data_model,
                    create_component_diagram,
                    analyze_dependencies,
                ],
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
                tools=[
                    generate_code,
                    refactor_code,
                    apply_design_pattern,
                    generate_boilerplate,
                    optimize_imports,
                    format_code,
                ],
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
                tools=[
                    review_code,
                    check_code_style,
                    analyze_complexity,
                    detect_code_smells,
                    suggest_improvements,
                    check_best_practices,
                ],
            ),
            SubAgentConfig(
                name="testing-automation",
                instructions="""You are a Testing Automation Agent specialized in quality assurance.
Your responsibilities:
1. Generate comprehensive unit tests (pytest, unittest, jest, mocha)
2. Create integration and E2E tests
3. Analyze and improve test coverage
4. Generate test data and fixtures
5. Create test plans and strategies

Follow Arrange-Act-Assert pattern and test edge cases thoroughly.""",
                tools=[
                    generate_unit_tests,
                    generate_integration_tests,
                    analyze_test_coverage,
                    run_tests,
                    generate_test_data,
                    create_test_plan,
                ],
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
                tools=[
                    analyze_error,
                    trace_execution,
                    identify_root_cause,
                    propose_fix,
                    analyze_performance,
                    detect_memory_issues,
                ],
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
                tools=[
                    scan_security_issues,
                    check_owasp_compliance,
                    detect_secrets,
                    analyze_dependencies_security,
                    generate_security_report,
                    suggest_security_fixes,
                ],
            ),
            SubAgentConfig(
                name="devops-integration",
                instructions="""You are a DevOps Integration Agent specialized in deployment automation.
Your responsibilities:
1. Create CI/CD pipeline configurations
2. Generate Docker and container configurations
3. Create Kubernetes deployment manifests (with ConfigMap and HPA)
4. Set up monitoring and observability
5. Configure deployment environments

Follow Infrastructure as Code principles and implement proper secrets management.""",
                tools=[
                    create_ci_pipeline,
                    create_cd_pipeline,
                    configure_deployment,
                    generate_dockerfile,
                    create_kubernetes_config,
                    setup_monitoring,
                ],
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
                tools=[
                    generate_api_docs,
                    create_readme,
                    document_architecture,
                    generate_changelog,
                    add_inline_comments,
                    create_user_guide,
                ],
            ),
        ]

    # Keep for backwards compatibility - will be populated by _get_default_subagents()
    DEFAULT_SOFTWARE_DEV_SUBAGENTS: List[SubAgentConfig] = []

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

        # Use default subagents if none provided (with tools wired)
        if sub_agents is None:
            sub_agents = self._get_default_subagents()

        super().__init__(
            name=name,
            instructions=instructions or self.SOFTWARE_DEV_INSTRUCTIONS,
            agent_subtype="software_development",
            sub_agents=sub_agents,
            tools=tools,
            **kwargs,
        )
