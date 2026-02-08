# Claude Agent Skills for LangChain Azure

> **Purpose**: Guide for creating and using Claude Agent SDK skills for LangChain Azure development workflows

---

## Table of Contents

1. [Overview](#overview)
2. [Skills Architecture](#skills-architecture)
3. [Available Skills](#available-skills)
4. [Creating New Skills](#creating-new-skills)
5. [Local vs Global Skills](#local-vs-global-skills)
6. [Usage Examples](#usage-examples)

---

## Overview

Claude Agent Skills are specialized tools that automate common development tasks in this LangChain Azure repository. They provide:

- **Consistency**: Standardized workflows across the team
- **Efficiency**: One-command execution of complex tasks
- **Best Practices**: Built-in adherence to repository guidelines
- **Context Awareness**: Skills understand repository structure

---

## Skills Architecture

```
langchain-azure/
├── .claude/
│   ├── skills/                    # Local skills (repo-specific)
│   │   ├── new-agent/            # Create new LangChain agent
│   │   ├── new-chain/            # Create new chain
│   │   ├── new-rag-pipeline/     # Setup RAG pipeline
│   │   ├── add-azure-integration/ # Add Azure service integration
│   │   ├── run-full-test/        # Run complete test suite
│   │   └── deploy-package/       # Deploy package to PyPI
│   ├── CLAUDE.md                 # Development guidelines
│   └── SKILLS_GUIDE.md           # This file
```

Each skill directory contains:
- `skill.json` - Skill manifest (metadata, parameters)
- `prompt.md` - Execution instructions for Claude
- `examples/` - Usage examples (optional)

---

## Available Skills

### 1. `/new-agent` - Create New LangChain Agent

**Purpose**: Create a new Azure AI agent with proper structure, tools, and tests.

**Usage**:
```bash
/new-agent <agent-name> --type [react|openai|conversational]
```

**What it does**:
- Creates agent file in `libs/azure-ai/langchain_azure_ai/agents/`
- Sets up agent with AzureAIChatCompletionsModel
- Creates unit tests
- Updates `__init__.py` exports
- Updates documentation

**Example**:
```bash
/new-agent customer-support --type react
```

---

### 2. `/new-chain` - Create New LangChain Chain

**Purpose**: Create a new chain with Azure AI models.

**Usage**:
```bash
/new-chain <chain-name> --model [gpt-4o|claude|deepseek-r1]
```

**What it does**:
- Creates chain file with proper imports
- Sets up prompt templates
- Adds output parsers if needed
- Creates tests
- Updates exports

---

### 3. `/new-rag-pipeline` - Setup RAG Pipeline

**Purpose**: Create complete RAG pipeline with Azure services.

**Usage**:
```bash
/new-rag-pipeline <name> --storage [cosmos|search|postgresql] --embeddings [openai|azure]
```

**What it does**:
- Sets up vectorstore (Azure AI Search, Cosmos DB, or PostgreSQL)
- Configures embeddings
- Creates document loaders
- Sets up retriever
- Creates sample queries
- Adds integration tests

---

### 4. `/add-azure-integration` - Add Azure Service Integration

**Purpose**: Add new Azure service integration to langchain-azure-ai.

**Usage**:
```bash
/add-azure-integration <service-name> --type [vectorstore|chat|embeddings|callback]
```

**What it does**:
- Creates integration class with Azure SDK
- Follows LangChain integration patterns
- Adds environment variable handling
- Creates comprehensive tests
- Updates documentation

---

### 5. `/run-full-test` - Run Complete Test Suite

**Purpose**: Run all tests, linting, formatting, and type checking.

**Usage**:
```bash
/run-full-test [--package azure-ai|all]
```

**What it does**:
- Runs unit tests with coverage
- Runs integration tests (if credentials available)
- Executes ruff format check
- Runs ruff linting
- Runs mypy type checking
- Generates coverage report
- Provides summary of all results

---

### 6. `/deploy-package` - Deploy Package to PyPI

**Purpose**: Build and deploy a package to PyPI (Test PyPI optional).

**Usage**:
```bash
/deploy-package <package-name> --test-pypi [true|false]
```

**What it does**:
- Runs full test suite
- Builds package with Poetry
- Validates package metadata
- Publishes to TestPyPI or PyPI
- Creates git tag
- Updates CHANGELOG.md

---

### 7. `/setup-dev-env` - Setup Development Environment

**Purpose**: Setup complete development environment for contributing.

**Usage**:
```bash
/setup-dev-env [--package azure-ai|all]
```

**What it does**:
- Installs Poetry if not present
- Installs dependencies with all extras
- Sets up pre-commit hooks
- Validates environment
- Creates .env from .env.example
- Runs smoke tests

---

### 8. `/add-connector` - Add Enterprise Connector

**Purpose**: Add new enterprise connector (Teams, Functions, etc.).

**Usage**:
```bash
/add-connector <connector-name> --type [teams|functions|copilot|custom]
```

**What it does**:
- Creates connector class
- Implements configuration management
- Adds manifest generation
- Creates deployment templates
- Adds comprehensive tests
- Updates connector documentation

---

## Creating New Skills

### Step 1: Create Skill Directory

```bash
mkdir -p .claude/skills/my-skill
cd .claude/skills/my-skill
```

### Step 2: Create `skill.json` Manifest

```json
{
  "name": "my-skill",
  "version": "1.0.0",
  "description": "Brief description of what this skill does",
  "author": "Your Name",
  "category": "langchain-azure",
  "parameters": [
    {
      "name": "param1",
      "type": "string",
      "required": true,
      "description": "First parameter description"
    },
    {
      "name": "param2",
      "type": "string",
      "required": false,
      "default": "default-value",
      "description": "Optional parameter"
    }
  ],
  "examples": [
    "/my-skill value1",
    "/my-skill value1 --param2 value2"
  ]
}
```

### Step 3: Create `prompt.md` Instructions

````markdown
# My Skill - Execution Instructions

You are executing the `my-skill` skill for the LangChain Azure repository.

## Context

This skill is designed to [describe purpose and use case].

## Parameters

- `param1`: [Description of what this parameter does]
- `param2`: [Description of optional parameter]

## Execution Steps

1. **Validation**
   - Verify parameters are valid
   - Check repository structure
   - Confirm prerequisites

2. **Main Execution**
   - [Step by step instructions]
   - [Be very specific about file paths]
   - [Include code templates]

3. **Testing**
   - Run unit tests
   - Validate functionality

4. **Documentation**
   - Update relevant docs
   - Add usage examples

## File Templates

### Template 1: [Name]

```python
# Template code here
```

## Best Practices

- Follow CLAUDE.md guidelines
- Use type hints
- Add Google-style docstrings
- Include tests
- Update __init__.py exports

## Success Criteria

- [ ] Files created in correct locations
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Follows repository standards
````

### Step 4: Test the Skill

Use the skill in Claude Code:
```bash
/my-skill param1 --param2 value2
```

---

## Local vs Global Skills

### Local Skills (Repository-Specific)

**Location**: `/.claude/skills/`

**Purpose**: Skills specific to this LangChain Azure repository

**Characteristics**:
- Deep integration with repository structure
- Use repository-specific patterns
- Access to local files and configs
- Shared with team via git

**Examples**:
- `/new-agent` - Creates agents in libs/azure-ai structure
- `/add-azure-integration` - Follows Azure-specific patterns
- `/deploy-package` - Uses Poetry and PyPI deployment

### Global Skills (Reusable Across Projects)

**Location**:
- Windows: `%APPDATA%\Claude\skills\`
- macOS/Linux: `~/.config/claude/skills/`

**Purpose**: Generic skills usable in any project

**Characteristics**:
- Project-agnostic
- No assumptions about structure
- Highly parameterized
- Personal skill library

**Examples**:
- `/create-python-package` - Generic Python package scaffolding
- `/setup-ci-cd` - Generic CI/CD setup
- `/write-readme` - Generic README generation

---

## Installation

### Installing Local Skills

Local skills are already in the repository:

```bash
# Clone the repo (if you haven't)
git clone <repo-url>
cd langchain-azure

# Skills are in .claude/skills/ and ready to use
/new-agent --help
```

### Installing Global Skills

1. Create global skills directory:

```bash
# Windows
mkdir %APPDATA%\Claude\skills

# macOS/Linux
mkdir -p ~/.config/claude/skills
```

2. Copy a local skill to global:

```bash
# Windows
xcopy .claude\skills\new-agent %APPDATA%\Claude\skills\new-agent /E /I

# macOS/Linux
cp -r .claude/skills/new-agent ~/.config/claude/skills/
```

3. Or create symlinks for always-up-to-date global skills:

```bash
# macOS/Linux
ln -s $(pwd)/.claude/skills/new-agent ~/.config/claude/skills/new-agent
```

---

## Usage Examples

### Example 1: Create New React Agent

```bash
# Start the skill
/new-agent helpdesk-agent --type react

# Follow prompts:
# - Agent description: "Enterprise IT helpdesk agent for Azure environments"
# - Tools needed: ["search_kb", "create_ticket", "check_status"]
# - Model: "gpt-4o"
```

**Result**:
- `libs/azure-ai/langchain_azure_ai/agents/helpdesk_agent.py` created
- `tests/unit_tests/agents/test_helpdesk_agent.py` created
- `libs/azure-ai/langchain_azure_ai/agents/__init__.py` updated
- Documentation updated

### Example 2: Setup RAG Pipeline

```bash
/new-rag-pipeline docs-qa --storage search --embeddings azure

# Result:
# - Azure AI Search vectorstore configured
# - Azure OpenAI embeddings setup
# - Document loaders for PDF/DOCX
# - Sample query interface
```

### Example 3: Run Full Test Suite

```bash
/run-full-test --package azure-ai

# Runs:
# - Unit tests: pytest tests/unit_tests
# - Integration tests: pytest tests/integration_tests
# - Linting: ruff check
# - Formatting: ruff format --check
# - Type checking: mypy
# - Coverage report
```

---

## Best Practices

### Skill Design

1. **Single Responsibility**: Each skill does one thing well
2. **Idempotent**: Can run multiple times safely
3. **Validated Input**: Check all parameters before execution
4. **Clear Output**: Report what was done and what's next
5. **Rollback Capable**: Provide instructions to undo changes

### Skill Documentation

1. **Clear Description**: Explain purpose and use case
2. **Parameter Docs**: Document all parameters with examples
3. **Prerequisites**: List any requirements
4. **Examples**: Provide real-world usage examples
5. **Troubleshooting**: Common issues and solutions

### Repository Integration

1. **Follow CLAUDE.md**: Adhere to repository guidelines
2. **Consistent Structure**: Use established patterns
3. **Update Docs**: Keep documentation in sync
4. **Add Tests**: Every skill-generated code needs tests
5. **Git Friendly**: Stage and commit atomically

---

## Troubleshooting

### Skill Not Found

**Problem**: `/my-skill` not recognized

**Solution**:
```bash
# Verify skill exists
ls .claude/skills/my-skill/

# Check skill.json is valid
cat .claude/skills/my-skill/skill.json | python -m json.tool

# Restart Claude Code
```

### Skill Execution Fails

**Problem**: Skill starts but fails during execution

**Solution**:
1. Check prerequisites (dependencies, permissions)
2. Verify repository structure matches expectations
3. Review error messages carefully
4. Check logs in `.claude/logs/`
5. Run with --debug flag if available

### Permission Issues

**Problem**: Cannot write files

**Solution**:
```bash
# Check directory permissions
ls -la .claude/skills/

# Fix permissions
chmod -R u+w .claude/skills/
```

---

## Contributing New Skills

### Contribution Checklist

- [ ] Skill solves real repository workflow need
- [ ] Skill is tested on fresh environment
- [ ] skill.json is complete and valid
- [ ] prompt.md has clear instructions
- [ ] Examples are provided
- [ ] Documentation is updated
- [ ] Follows CLAUDE.md guidelines

### Contribution Process

1. Create skill in `.claude/skills/new-skill/`
2. Test thoroughly
3. Update this SKILLS_GUIDE.md
4. Create pull request
5. Tag maintainers for review

---

## Advanced Topics

### Skill Composition

Skills can call other skills:

```markdown
## Execution Steps

1. First, run the `/setup-dev-env` skill
2. Then create the agent with specific parameters
3. Finally, run `/run-full-test --package azure-ai`
```

### Conditional Logic

Skills can adapt based on repository state:

```markdown
## Execution Steps

1. Check if agent already exists
   - If exists: Ask user to rename or update
   - If not exists: Proceed with creation

2. Detect Azure credentials
   - If found: Use for integration tests
   - If not found: Skip integration tests, warn user
```

### Parameterized Templates

Use parameters to customize templates:

```python
# Template with parameters: {{agent_name}}, {{model}}
class {{agent_name}}Agent:
    def __init__(self):
        self.model = "{{model}}"
```

---

## Future Skills (Roadmap)

### Planned Skills

1. `/migrate-to-langgraph` - Migrate legacy agents to LangGraph
2. `/add-streaming` - Add streaming support to agent/chain
3. `/setup-monitoring` - Add Azure Monitor integration
4. `/create-evaluation` - Create LangSmith evaluation suite
5. `/add-tool` - Add new tool to agent
6. `/benchmark-agent` - Run performance benchmarks
7. `/create-example` - Create new sample application
8. `/validate-schema` - Validate Azure integration schemas

### Contributing Skill Ideas

Have an idea for a new skill?

1. Open an issue with title: `[Skill] Name of Skill`
2. Describe the workflow it automates
3. Provide example usage
4. Tag with `enhancement` and `skills` labels

---

## Resources

- **Claude Agent SDK Docs**: [Link to Claude docs when available]
- **LangChain Docs**: https://python.langchain.com/
- **Azure AI Foundry**: https://aka.ms/azureai/langchain
- **Repository Guidelines**: [.claude/CLAUDE.md](.claude/CLAUDE.md)

---

## Support

**Questions or Issues?**

1. Check this guide first
2. Review [CLAUDE.md](.claude/CLAUDE.md) for guidelines
3. Open an issue with `skills` label
4. Contact: abhilashjaiswal0110@gmail.com

---

*Last Updated: 2026-02-08*
*Version: 1.0.0*
