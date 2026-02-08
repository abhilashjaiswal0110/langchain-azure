# New Agent Skill - Execution Instructions

You are executing the `new-agent` skill for the LangChain Azure repository.

## Purpose

This skill creates a new LangChain agent with Azure AI integration, following the repository's best practices and standards defined in `.claude/CLAUDE.md`.

## Parameters

You will receive the following parameters:

- **agent_name** (required): Agent name in kebab-case (e.g., "customer-support")
- **type** (optional, default: "react"): Agent type
  - `react`: ReAct agent with tool calling
  - `openai`: OpenAI functions agent
  - `conversational`: Conversational agent with memory
  - `custom`: Custom agent structure
- **model** (optional, default: "gpt-4o"): Azure AI model name
- **with_memory** (optional, default: true): Include Cosmos DB memory
- **with_tools** (optional, default: true): Include sample tools

## Pre-Execution Validation

Before starting, verify:

1. **Current directory** is the repository root:
   ```bash
   pwd  # Should be .../langchain-azure
   ls libs/azure-ai  # Should exist
   ```

2. **Agent name is valid**:
   - Lowercase letters, numbers, hyphens only
   - Starts with a letter
   - Example: "customer-support" ✓, "CustomerSupport" ✗

3. **Agent doesn't already exist**:
   ```bash
   ls libs/azure-ai/langchain_azure_ai/agents/{agent_name}.py
   # Should not exist
   ```

If any validation fails, **STOP** and inform the user.

## Execution Steps

### Step 1: Convert Agent Name

Convert kebab-case input to various formats:
- **Input format**: "customer-support" (kebab-case from user)
- **snake_case**: "customer_support" (for Python filenames and identifiers)
- **PascalCase**: "CustomerSupport" (for class names)
- **Title Case**: "Customer Support" (for documentation)

Example:
- Input: "it-helpdesk-agent"
- snake_case: "it_helpdesk_agent" (used for filename)
- PascalCase: "ItHelpdeskAgent"
- Title: "IT Helpdesk Agent"

**IMPORTANT**: Python module filenames must use snake_case, not kebab-case. Convert all hyphens to underscores for filenames.

### Step 2: Create Agent File

Create: `libs/azure-ai/langchain_azure_ai/agents/{snake_case}.py`

**Note**: Use snake_case for the filename, not kebab-case.

Use the template based on the agent type:

#### Template for ReAct Agent (type=react)

```python
"""{{Title}} Agent.

This module implements a {{Title}} agent using Azure AI and LangGraph.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import create_react_agent

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel


def create_{{snake_case}}_agent(
    model: Optional[BaseChatModel] = None,
    tools: Optional[list[BaseTool]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    **kwargs: Any,
) -> Any:
    """Create a {{Title}} agent.

    This agent is designed to [DESCRIBE PURPOSE AND CAPABILITIES].

    Args:
        model: Language model to use. Defaults to {{model}}.
        tools: List of tools available to the agent.
        checkpointer: Checkpointer for conversation memory.
        **kwargs: Additional arguments passed to create_react_agent.

    Returns:
        Compiled LangGraph agent ready for invocation.

    Example:
        >>> agent = create_{{snake_case}}_agent()
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "Hello"}]
        ... })
    """
    # Default model
    if model is None:
        model = AzureAIChatCompletionsModel(
            model="{{model}}",
            temperature=0.7,
        )

    # Default tools
    if tools is None:
        tools = _get_default_tools()

    # System message
    system_message = """You are a {{Title}} agent.

Your role is to [DESCRIBE ROLE AND RESPONSIBILITIES].

Available tools:
- [TOOL 1]: [Description]
- [TOOL 2]: [Description]

Always be helpful, professional, and accurate in your responses.
"""

    # Create agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=system_message,
        checkpointer=checkpointer,
        **kwargs,
    )

    return agent


def _get_default_tools() -> list[BaseTool]:
    """Get default tools for the {{Title}} agent.

    Returns:
        List of default tools.
    """
    from langchain_core.tools import tool

    @tool
    def sample_tool(query: str) -> str:
        """Sample tool for {{Title}} agent.

        Args:
            query: The query to process.

        Returns:
            The processed result.
        """
        return f"Processed: {query}"

    return [sample_tool]


# Legacy class-based interface (for compatibility)
class {{PascalCase}}Agent:
    """{{Title}} Agent (legacy interface).

    This class provides a legacy interface for the {{Title}} agent.
    For new code, use create_{{snake_case}}_agent() directly.
    """

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        tools: Optional[list[BaseTool]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> None:
        """Initialize the {{Title}} agent.

        Args:
            model: Language model to use.
            tools: List of tools available to the agent.
            checkpointer: Checkpointer for conversation memory.
        """
        self.agent = create_{{snake_case}}_agent(
            model=model,
            tools=tools,
            checkpointer=checkpointer,
        )

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> dict[str, Any]:
        """Invoke the agent with messages.

        Args:
            messages: List of messages to process.
            **kwargs: Additional arguments.

        Returns:
            Agent response.
        """
        return self.agent.invoke({"messages": messages}, **kwargs)


__all__ = [
    "create_{{snake_case}}_agent",
    "{{PascalCase}}Agent",
]
```

**IMPORTANT**: Replace all `{{placeholders}}` with actual values:
- `{{Title}}`: Title case name (e.g., "Customer Support")
- `{{snake_case}}`: Snake case name (e.g., "customer_support")
- `{{PascalCase}}`: Pascal case name (e.g., "CustomerSupport")
- `{{model}}`: Model name (e.g., "gpt-4o")

#### Template Modifications

**If `with_memory=false`**:
- Remove `checkpointer` parameter
- Remove memory-related comments

**If `with_tools=false`**:
- Remove `_get_default_tools()` function
- Set `tools=[]` as default
- Remove tool descriptions from system message

**If `type=openai`**:
- Use `create_openai_functions_agent` instead of `create_react_agent`
- Import from `langchain.agents`

**If `type=conversational`**:
- Add conversation history management
- Include chat history in system message

### Step 3: Create Unit Tests

Create: `tests/unit_tests/agents/test_{agent_name}.py`

```python
"""Unit tests for {{Title}} Agent."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from unittest.mock import MagicMock, patch

from langchain_azure_ai.agents.{{agent_name}} import (
    create_{{snake_case}}_agent,
    {{PascalCase}}Agent,
)


class TestCreate{{PascalCase}}Agent:
    """Test suite for create_{{snake_case}}_agent function."""

    def test_agent_creation_with_defaults(self) -> None:
        """Test agent creation with default parameters."""
        with patch("langchain_azure_ai.chat_models.AzureAIChatCompletionsModel"):
            agent = create_{{snake_case}}_agent()
            assert agent is not None

    def test_agent_creation_with_custom_model(self) -> None:
        """Test agent creation with custom model."""
        mock_model = MagicMock()
        agent = create_{{snake_case}}_agent(model=mock_model)
        assert agent is not None

    def test_agent_creation_with_custom_tools(self) -> None:
        """Test agent creation with custom tools."""
        mock_tool = MagicMock()
        with patch("langchain_azure_ai.chat_models.AzureAIChatCompletionsModel"):
            agent = create_{{snake_case}}_agent(tools=[mock_tool])
            assert agent is not None

    def test_agent_invocation(self) -> None:
        """Test agent can be invoked with messages."""
        with patch("langchain_azure_ai.chat_models.AzureAIChatCompletionsModel"):
            agent = create_{{snake_case}}_agent()

            # Mock the invoke method to avoid actual API calls
            agent.invoke = MagicMock(return_value={
                "messages": [{"role": "assistant", "content": "Hello!"}]
            })

            result = agent.invoke({"messages": [HumanMessage(content="Hi")]})
            assert result is not None
            assert "messages" in result


class Test{{PascalCase}}Agent:
    """Test suite for {{PascalCase}}Agent class."""

    def test_agent_initialization(self) -> None:
        """Test agent class initialization."""
        with patch("langchain_azure_ai.chat_models.AzureAIChatCompletionsModel"):
            agent = {{PascalCase}}Agent()
            assert agent is not None
            assert agent.agent is not None

    def test_agent_invoke_method(self) -> None:
        """Test agent invoke method."""
        with patch("langchain_azure_ai.chat_models.AzureAIChatCompletionsModel"):
            agent = {{PascalCase}}Agent()

            # Mock the internal agent invoke
            agent.agent.invoke = MagicMock(return_value={
                "messages": [{"role": "assistant", "content": "Response"}]
            })

            messages = [HumanMessage(content="Test")]
            result = agent.invoke(messages)

            assert result is not None
            assert "messages" in result


@pytest.mark.integration
class TestIntegration{{PascalCase}}Agent:
    """Integration tests for {{Title}} Agent (requires Azure credentials)."""

    def test_agent_with_real_model(self) -> None:
        """Test agent with real Azure AI model."""
        agent = create_{{snake_case}}_agent()
        result = agent.invoke({
            "messages": [HumanMessage(content="Hello, how can you help me?")]
        })

        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 0
```

**IMPORTANT**: Replace all `{{placeholders}}` with actual values.

### Step 4: Update Exports

Update: `libs/azure-ai/langchain_azure_ai/agents/__init__.py`

Add the following imports (maintain alphabetical order):

```python
from langchain_azure_ai.agents.{{agent_name}} import (
    create_{{snake_case}}_agent,
    {{PascalCase}}Agent,
)
```

Add to `__all__` list:

```python
__all__ = [
    # ... existing exports ...
    "create_{{snake_case}}_agent",
    "{{PascalCase}}Agent",
    # ... more exports ...
]
```

### Step 5: Create Example Usage

Create: `libs/azure-ai/examples/{{agent_name}}_example.py`

```python
"""Example usage of {{Title}} Agent."""

from langchain_core.messages import HumanMessage

from langchain_azure_ai.agents import create_{{snake_case}}_agent


def main() -> None:
    """Run {{Title}} agent example."""
    # Create agent
    agent = create_{{snake_case}}_agent()

    # Example conversation
    messages = [
        HumanMessage(content="Hello! Can you help me?")
    ]

    # Invoke agent
    result = agent.invoke({"messages": messages})

    # Print response
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()
```

### Step 6: Update Documentation

If `docs/agents.md` or similar exists, add a section:

```markdown
## {{Title}} Agent

The {{Title}} agent is designed to [DESCRIBE PURPOSE].

### Features

- [Feature 1]
- [Feature 2]
- [Feature 3]

### Usage

```python
from langchain_azure_ai.agents import create_{{snake_case}}_agent

agent = create_{{snake_case}}_agent()
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello"}]
})
```

### Configuration

- **Model**: {{model}} (customizable)
- **Tools**: [List default tools]
- **Memory**: [Enabled/Disabled based on with_memory]

### API Reference

See [`langchain_azure_ai.agents.{{agent_name}}`](#) for full API documentation.
```

### Step 7: Run Tests

```bash
cd libs/azure-ai
poetry run pytest tests/unit_tests/agents/test_{{agent_name}}.py -v
```

### Step 8: Format and Lint

```bash
cd libs/azure-ai
poetry run ruff format langchain_azure_ai/agents/{{agent_name}}.py
poetry run ruff check langchain_azure_ai/agents/{{agent_name}}.py
poetry run mypy langchain_azure_ai/agents/{{agent_name}}.py
```

## Post-Execution Summary

After successful execution, provide this summary to the user:

```
✅ Successfully created {{Title}} Agent!

📁 Files Created:
- libs/azure-ai/langchain_azure_ai/agents/{{agent_name}}.py
- tests/unit_tests/agents/test_{{agent_name}}.py
- libs/azure-ai/examples/{{agent_name}}_example.py

📝 Files Updated:
- libs/azure-ai/langchain_azure_ai/agents/__init__.py

📊 Configuration:
- Agent Type: {{type}}
- Model: {{model}}
- Memory: {{with_memory}}
- Tools: {{with_tools}}

🧪 Tests Status: [PASS/FAIL based on test run]

📋 Next Steps:
1. Review the generated agent code
2. Customize the system message and tools
3. Run integration tests: pytest --integration libs/azure-ai/tests/unit_tests/agents/test_{{agent_name}}.py
4. Update documentation with specific use cases
5. Set up Azure credentials in .env file

📚 Usage Example:
```python
from langchain_azure_ai.agents import create_{{snake_case}}_agent

agent = create_{{snake_case}}_agent()
result = agent.invoke({
    "messages": [{"role": "user", "content": "Your message here"}]
})
print(result["messages"][-1].content)
```

Need help? Check .claude/SKILLS_GUIDE.md or open an issue.
```

## Error Handling

### If agent already exists:
```
❌ Error: Agent '{{agent_name}}' already exists.

File exists: libs/azure-ai/langchain_azure_ai/agents/{{agent_name}}.py

Options:
1. Choose a different name
2. Delete existing agent: rm libs/azure-ai/langchain_azure_ai/agents/{{agent_name}}.py
3. Update existing agent instead

Would you like to:
- Use a different agent name?
- Overwrite the existing agent? [y/N]
```

### If directory structure is wrong:
```
❌ Error: Invalid repository structure.

Expected: libs/azure-ai/langchain_azure_ai/agents/
Current directory: [CURRENT_DIR]

Please run this skill from the repository root:
cd /path/to/langchain-azure
/new-agent {{agent_name}}
```

### If tests fail:
```
⚠️ Warning: Some tests failed.

Failed tests:
- test_agent_creation_with_defaults
- test_agent_invocation

Agent files created successfully, but tests need attention.

Review test failures:
pytest tests/unit_tests/agents/test_{{agent_name}}.py -v

Common fixes:
1. Check import statements
2. Verify mock configurations
3. Ensure Azure credentials in tests are mocked
```

## Best Practices

1. **Follow CLAUDE.md**: Adhere to all guidelines in `.claude/CLAUDE.md`
2. **Type Hints Everywhere**: All functions must have type hints
3. **Google Docstrings**: Use Google-style docstrings for all public functions
4. **Test Coverage**: Aim for >80% test coverage
5. **No Hardcoded Values**: Use environment variables or parameters
6. **Error Messages**: Provide clear, actionable error messages
7. **Atomic Commits**: Create single commit for all related changes

## Validation Checklist

Before completing, verify:

- [ ] Agent file created with correct template
- [ ] All {{placeholders}} replaced with actual values
- [ ] Tests file created and passing
- [ ] __init__.py updated with exports
- [ ] Example file created
- [ ] Code formatted with ruff
- [ ] No linting errors
- [ ] Type checking passes
- [ ] Documentation added/updated
- [ ] Git status clean (all files staged)

## Success Criteria

Skill succeeds when:
1. All files created without errors
2. Tests pass (unit tests minimum)
3. Code passes linting and formatting
4. Type checking succeeds
5. Exports properly configured
6. User can import and use the agent immediately

---

**Remember**: Quality over speed. Take time to generate clean, well-documented code that follows repository standards.
