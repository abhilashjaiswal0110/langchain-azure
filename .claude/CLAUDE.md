# LangChain Agents Platform - Development Guidelines

> **Document Version**: 3.0
> **Last Updated**: 2025-12-19
> **Purpose**: Authoritative development guidelines for AI coding agents working on this repository

---

## Table of Contents

1. [Project Context](#project-context)
2. [Core Principles](#core-principles)
3. [Role-Based Development Guidelines](#role-based-development-guidelines)
4. [Project Structure](#project-structure)
5. [LangChain Development Standards](#langchain-development-standards)
6. [Documentation Requirements](#documentation-requirements)
7. [Git Workflow Standards](#git-workflow-standards)
8. [Security & Compliance](#security--compliance)
9. [Quality Gates](#quality-gates)
10. [Quick Reference](#quick-reference)

---

## Project Context

### What is this project?

A **production-ready deployment platform** that serves LangChain chains and LangGraph agents as REST APIs. This project provides:

- FastAPI server with LangServe integration
- Multiple AI endpoints (chat, RAG, agents)
- IT Support Agents (IT Helpdesk, ServiceNow) with conversation memory
- Web UI and CLI for demos and testing
- External Integration Webhooks for Copilot Studio, Azure AI, AWS AI
- Document RAG with PDF/Word/TXT support
- LangSmith tracing for observability
- Docker containerization for deployment

### Repository Structure

```
langchain-agents/
├── deployment/              # Main application
│   ├── app/                 # Source code
│   │   ├── chains/          # LangChain chains
│   │   ├── agents/          # IT Support agents
│   │   └── static/          # Web UI
│   ├── tests/               # Test suite
│   ├── cli_chat.py          # CLI interface
│   ├── Dockerfile           # Container build
│   ├── docker-compose.yml   # Local deployment
│   └── KNOWLEDGE.md         # AI agent knowledge base
├── docs/                    # Documentation
├── .claude/                 # Claude AI configuration
│   └── CLAUDE.md            # This file
├── README.md                # Project overview
└── .gitignore               # Git exclusions
```

### Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Web Framework | FastAPI | >=0.115.0 |
| LLM Framework | LangChain | >=0.3.0 |
| Agent Framework | LangGraph | >=0.2.0 |
| API Serving | LangServe | >=0.3.0 |
| Tracing | LangSmith | >=0.1.0 |
| Primary LLM | OpenAI GPT-4o-mini | - |
| Alternative LLM | Anthropic Claude | - |
| Python | Python | >=3.10 |

### Key Design Decisions

1. **LangGraph over legacy agents**: Uses `langgraph.prebuilt.create_react_agent` instead of deprecated `langchain.agents.create_tool_calling_agent`
2. **Lazy loading**: Chains load only when API keys are available
3. **Provider agnostic**: Supports both OpenAI and Anthropic
4. **Tracing first**: LangSmith tracing enabled by default when configured
5. **Session-based conversations**: IT Support agents use MemorySaver for conversation continuity
6. **Webhook-based integration**: External platforms integrate via standardized webhook API

### Essential References

- **Knowledge Base**: [deployment/KNOWLEDGE.md](../deployment/KNOWLEDGE.md) - Detailed architecture and implementation guide
- **README**: [README.md](../README.md) - Project overview and quick start
- **LangChain Docs**: https://docs.langchain.com/

---

## Core Principles

**Technology-Agnostic Excellence**: These guidelines apply to ALL projects regardless of language, framework, or platform.

### Fundamental Rules

1. **Security First**: Never commit secrets, always validate input
2. **Documentation Always**: Code without docs is incomplete
3. **Test Everything**: Untested code is broken code
4. **Commit Atomically**: One logical change per commit
5. **Review Before Push**: Always verify your changes
6. **Automate Quality**: Use CI/CD to enforce standards

---

## Role-Based Development Guidelines

When working on any project, assume the following roles and apply their respective best practices:

### 1. Software Architect
- Design modular, scalable system architecture
- Create clear separation of concerns
- Document architectural decisions and patterns (ADRs)
- Plan for extensibility and maintainability
- Choose appropriate design patterns for the problem domain

### 2. Security Architect
- Implement security-first design principles
- Ensure no sensitive data exposure
- Follow OWASP and industry security standards
- Document security considerations and threat models
- Apply principle of least privilege

### 3. Data Architect
- Design efficient data structures and flows
- Plan data persistence and caching strategies
- Document data models and relationships
- Ensure data integrity and validation
- Consider scalability and performance

### 4. Software Engineer
- Write clean, maintainable, well-documented code
- Follow language-specific best practices and idioms
- Implement comprehensive error handling
- Ensure code quality through testing
- Practice defensive programming

---

## Project Structure

### Universal Directory Structure

Adapt this structure to your project type while maintaining the core principles:

```
project-root/
├── src/                    # Source code (or lib/, app/, pkg/)
│   ├── core/              # Core business logic
│   ├── services/          # Service layer
│   ├── models/            # Data models/entities
│   ├── utils/             # Utility functions
│   ├── config/            # Configuration management
│   └── [domain-specific]  # API/, handlers/, controllers/, etc.
├── tests/                 # All test files
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── scripts/              # Automation scripts
│   ├── build/            # Build scripts
│   ├── deployment/       # Deployment scripts
│   └── utils/            # Utility scripts
├── docs/                 # Documentation
│   ├── architecture/     # Architecture diagrams & ADRs
│   ├── api/              # API documentation
│   ├── deployment/       # Deployment guides
│   └── operations/       # Operations runbooks
├── config/               # Environment configurations
│   ├── dev/             # Development configs
│   ├── staging/         # Staging configs
│   └── prod/            # Production configs (templates only, no secrets)
├── .github/              # GitHub specific files
│   └── workflows/       # CI/CD workflows
├── .gitignore           # Git exclusions
├── README.md            # Project overview
└── [build-config]       # package.json, Cargo.toml, go.mod, pom.xml, etc.
```

### Organization Principles

**DO:**
- Group by functionality/domain, not file type
- Keep root directory clean - only essential config files
- Separate concerns - each module has single responsibility
- Follow consistent naming conventions (language-specific)
- Use meaningful, descriptive names
- Maintain minimal code complexity

**DON'T:**
- Mix test files with source code
- Put temporary/scratch files in tracked directories
- Create deeply nested structures unnecessarily
- Use generic names like `utils.py` or `helper.js` without context

---

## LangChain Development Standards

### Code Quality Standards

All Python code MUST include type hints and return types.

```python
def filter_unknown_users(users: list[str], known_users: set[str]) -> list[str]:
    """Single line description of the function.

    Any additional context about the function can go here.

    Args:
        users: List of user identifiers to filter.
        known_users: Set of known/valid user identifiers.

    Returns:
        List of users that are not in the known_users set.
    """
```

### Testing Requirements

Every new feature or bugfix MUST be covered by unit tests.

- Unit tests: `tests/unit/` (no network calls allowed)
- Integration tests: `tests/integration/` (network calls permitted)
- Use `pytest` as the testing framework
- Testing file structure should mirror the source code structure

**Checklist:**
- [ ] Tests fail when your new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases and error conditions are tested
- [ ] Use fixtures/mocks for external dependencies
- [ ] Tests are deterministic (no flaky tests)

### Security and Risk Assessment

- No `eval()`, `exec()`, or `pickle` on user-controlled input
- Proper exception handling (no bare `except:`) and use a `msg` variable for error messages
- Remove unreachable/commented code before committing
- Race conditions or resource leaks (file handles, sockets, threads)
- Ensure proper resource cleanup (file handles, connections)

### Documentation Standards

Use Google-style docstrings with Args section for all public functions.

```python
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """Send an email to a recipient with specified priority.

    Any additional context about the function can go here.

    Args:
        to: The email address of the recipient.
        msg: The message body to send.
        priority: Email priority level.

    Returns:
        `True` if email was sent successfully, `False` otherwise.

    Raises:
        InvalidEmailError: If the email address format is invalid.
        SMTPConnectionError: If unable to connect to email server.
    """
```

- Types go in function signatures, NOT in docstrings
- Focus on "why" rather than "what" in descriptions
- Document all parameters, return values, and exceptions
- Keep descriptions concise but clear
- Ensure American English spelling (e.g., "behavior", not "behaviour")

### Adding New Components

**Adding a New Chain:**
1. Create file in `deployment/app/chains/`
2. Implement chain using LangChain patterns
3. Export in `deployment/app/chains/__init__.py`
4. Add route in `deployment/app/server.py`
5. Update `deployment/KNOWLEDGE.md`

**Adding a New Tool:**
```python
@tool
def your_tool(param: str) -> str:
    """Tool description for LLM.

    Args:
        param: Parameter description.

    Returns:
        Result description.
    """
    return result

# Add to tools list
tools = [..., your_tool]
```

**Adding a New IT Support Agent:**
1. Create agent file in `deployment/app/agents/`
2. Register in `conversation_manager.py`
3. Export in `deployment/app/agents/__init__.py`
4. Update `deployment/KNOWLEDGE.md`

---

## Documentation Requirements

### Mandatory Documentation Files

#### 1. README.md
```markdown
# Project Title

## Overview
Brief description and purpose

## Key Features
- Feature 1
- Feature 2

## Quick Start
Basic installation and usage

## Technology Stack
List key technologies and versions

## Documentation
Links to detailed documentation
```

#### 2. KNOWLEDGE.md (for AI agents)
```markdown
# Project Knowledge Base

## Project Overview
What is this project and its purpose

## Architecture
System design and component interactions

## Key Components
Detailed component documentation

## API Endpoints
Complete API reference

## Common Tasks
How to perform common operations

## Troubleshooting
Common issues and solutions
```

### Process Flow Documentation

- Create visual diagrams using Mermaid, PlantUML, or images
- Document functional flows for each major feature
- Include sequence diagrams for complex interactions
- Add state diagrams where applicable

---

## Git Workflow Standards

### Commit Message Format (Conventional Commits)

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no functional change)
- `test`: Adding/updating tests
- `chore`: Maintenance tasks (dependencies, configs)
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `build`: Build system changes
- `revert`: Revert previous commit

#### Examples:

```
feat(agents): Add ServiceNow ITSM agent

Implement ServiceNow integration:
- Add incident search and management
- Implement CMDB queries
- Add change request tracking

Closes #123
```

```
fix(api): Handle null session in conversation manager

Add null check before accessing session data to prevent
KeyError in edge cases where session has expired.

Fixes #456
```

### Staging & Committing Strategy

#### 1. Group Related Changes

Create **atomic commits** - each commit should represent one logical change:

```bash
# Stage by feature/component
git add deployment/app/agents/
git commit -m "feat(agents): Implement IT helpdesk agent"

git add deployment/tests/
git commit -m "test(agents): Add IT helpdesk agent tests"

git add deployment/KNOWLEDGE.md
git commit -m "docs: Update knowledge base with agent documentation"
```

#### 2. Logical Commit Sequence

Follow this order: **Config -> Core -> Tests -> Docs**

#### 3. Verification Before Commit

**ALWAYS verify before committing:**

```bash
# Check what's staged
git status

# Review changes in detail
git diff --cached

# Check recent history
git log --oneline -5

# Verify no secrets
git diff --cached | grep -iE '(password|secret|key|token|api_key)'
```

### What NOT to Commit

**Never commit:**

- **Secrets & Credentials**: API keys, tokens, passwords, private keys
- **Execution Results**: Test results, benchmarks, coverage reports (HTML)
- **Build Artifacts**: Compiled binaries, packaged distributions
- **Temporary Files**: Scratch files, debug outputs, backups
- **IDE-Specific Settings**: Personal IDE configurations
- **Large Files**: Binary files, large datasets (use Git LFS if needed)
- **Environment Files**: `.env` files with real credentials

**Action**: Update `.gitignore` immediately if these appear in `git status`

### .gitignore Template

```gitignore
# Environment & Secrets - CRITICAL
.env
.env.*
!.env.example
!.env.template
*.pem
*.key
credentials.json
secrets.json

# Python
__pycache__/
*.py[cod]
venv/
env/
*.egg-info/
dist/
build/
.mypy_cache/
.ruff_cache/

# Testing & Coverage
.coverage
.pytest_cache/
htmlcov/
coverage/

# IDE & Editors
.vscode/
.idea/
*.swp

# OS Generated
.DS_Store
Thumbs.db
```

---

## Security & Compliance

### Security Checklist (Before Every Commit)

- [ ] No hardcoded secrets or credentials
- [ ] No API keys, tokens, or passwords in code
- [ ] No sensitive data in logs or comments
- [ ] Input validation implemented for user inputs
- [ ] Error messages don't expose system internals
- [ ] Dependencies are up-to-date (no critical vulnerabilities)
- [ ] No SQL injection vulnerabilities
- [ ] Authentication/authorization properly implemented

### Secure Coding Practices

#### 1. Secrets Management

**CORRECT:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not configured")
```

**WRONG:**
```python
api_key = "sk-1234567890abcdef"  # Never hardcode!
```

#### 2. Input Validation

**CORRECT:**
```python
def process_user_input(data: str) -> str:
    if not isinstance(data, str):
        raise ValueError("Input must be string")
    if len(data) > 1000:
        raise ValueError("Input too long")
    return data.strip()
```

**WRONG:**
```python
def process_user_input(data):
    return data  # No validation!
```

#### 3. Error Handling

**CORRECT:**
```python
try:
    result = risky_operation()
except Exception as e:
    logger.error("Operation failed", exc_info=True)
    return {"error": "Operation failed. Please try again."}
```

**WRONG:**
```python
try:
    result = risky_operation()
except Exception as e:
    return {"error": str(e)}  # Exposes internals!
```

---

## Quality Gates

### Before Pushing to Remote

**Mandatory Checks:**

- [ ] All tests pass
- [ ] Code is formatted (ruff format)
- [ ] Linter passes (ruff check)
- [ ] Type checking passes (mypy)
- [ ] No secrets in code
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No temporary/test files staged
- [ ] `.gitignore` is comprehensive

### Running Quality Checks

```bash
# Python
pytest --cov=app --cov-report=html
ruff format deployment/
ruff check deployment/
mypy deployment/app/
```

---

## Quick Reference

### Development Commands

```bash
# Local development
cd deployment
cp .env.example .env
# Edit .env with your API keys
make run-reload

# Docker deployment
docker-compose up -d

# Run tests
make test
pytest tests/test_server.py -v

# Lint and format
ruff check deployment/
ruff format deployment/
```

### Git Workflow

```bash
# Standard workflow
git status
git add <files>
git diff --cached
git commit -m "type(scope): message"
git log --oneline -5
git push origin main
```

### Security Checks

```bash
# Check for potential secrets before commit
git diff --cached | grep -iE '(password|secret|key|token|api_key|credential)'
```

### API Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat/invoke` | POST | Chat completion |
| `/rag/invoke` | POST | RAG query |
| `/agent/invoke` | POST | Agent execution |
| `/api/conversation/start` | POST | Start IT support session |
| `/api/conversation/chat` | POST | Chat in session |
| `/api/webhook/chat` | POST | External integration webhook |
| `/chat` | GET | Web UI |

---

## Guidelines for AI Agents

### When Making Changes

1. **Read KNOWLEDGE.md first** before making any changes to deployment code
2. **Follow existing patterns** in the codebase
3. **Update KNOWLEDGE.md** when adding new features
4. **Run tests** before committing
5. **Use conventional commits** format

### Code Style

- Python 3.10+ features allowed
- Type hints required on all functions
- Google-style docstrings
- Ruff for linting and formatting
- No hardcoded secrets

### Commit Guidelines

- Use conventional commit format
- One logical change per commit
- Include scope when applicable
- Reference issues when fixing bugs

---

*This document is the authoritative source for development standards. When in doubt, refer to this file and the deployment/KNOWLEDGE.md for project-specific details.*
