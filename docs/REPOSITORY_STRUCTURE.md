# Repository Structure - LangChain Azure AI

## Overview

This document describes the clean, production-ready structure of the langchain-azure repository following GitHub enterprise best practices.

**Last Updated**: 2026-01-27
**Status**: Production Ready

---

## Root Directory Structure

```
langchain-azure/
├── .devcontainer/              # Development container configuration
├── .github/                    # GitHub Actions workflows and templates
├── .venv/                      # Python virtual environment (gitignored)
├── docs/                       # All documentation
├── libs/                       # Source code packages
├── tests/                      # Top-level test suite
├── test_results/               # Test artifacts and reports (gitignored)
├── .env                        # Environment configuration (gitignored)
├── .env.example                # Example environment configuration
├── .gitignore                  # Git ignore rules
├── Agents.md                   # Agent architecture documentation
├── CHANGELOG.md                # Version history
├── CODE_OF_CONDUCT.md          # Community guidelines
├── CONTRIBUTING.md             # Contribution guidelines
├── INTEGRATION_PLAN.md         # Integration roadmap
├── Knowledge.md                # Knowledge base documentation
├── LICENSE                     # MIT License
├── README.md                   # Main repository documentation
├── SECURITY.md                 # Security policy
└── pyproject.toml              # Python project configuration
```

---

## Documentation Directory (`docs/`)

All project documentation organized by topic:

```
docs/
├── COPILOT_STUDIO_INTEGRATION.md         # Microsoft Copilot Studio integration guide
├── DEEPAGENTS_EVALUATION_RESULTS.md      # DeepAgents test results
├── EVALUATION_FRAMEWORK_COMPLETE.md      # Evaluation framework guide
├── EVALUATION_IMPLEMENTATION.md          # Implementation details
├── README_EVALUATION.md                  # Evaluation quick start
└── REPOSITORY_STRUCTURE.md               # This file
```

**Purpose**: Centralized location for all technical documentation, guides, and results.

---

## Source Code Directory (`libs/`)

Monorepo structure with multiple packages:

```
libs/
├── azure-ai/                              # Main Azure AI integration
│   ├── langchain_azure_ai/
│   │   ├── agents/                        # Agent implementations
│   │   │   ├── deepagents/                # DeepAgents (IT Ops, Sales, Recruitment)
│   │   │   ├── enterprise/                # Enterprise agents
│   │   │   └── it_agents/                 # IT support agents
│   │   ├── evaluation/                    # Evaluation framework
│   │   │   ├── __init__.py
│   │   │   ├── base_evaluators.py         # Core evaluators (5)
│   │   │   ├── multi_turn_evaluators.py   # Conversation evaluators (4)
│   │   │   ├── langsmith_evaluator.py     # LangSmith integration
│   │   │   ├── azure_foundry_evaluator.py # Azure AI Foundry integration
│   │   │   ├── agent_metrics.py           # Performance tracking
│   │   │   └── datasets.py                # Test datasets (8 agents)
│   │   ├── server/                        # FastAPI server with evaluation endpoints
│   │   ├── tools/                         # Agent tools
│   │   ├── utils/                         # Utilities
│   │   └── __init__.py
│   ├── tests/                             # Library-specific tests
│   │   ├── integration_tests/
│   │   ├── unit_tests/
│   │   └── conftest.py
│   └── pyproject.toml
│
├── azure-dynamic-sessions/                # Dynamic sessions package
├── azure-postgresql/                      # PostgreSQL integration
├── azure-storage/                         # Azure Storage integration
└── sqlserver/                             # SQL Server integration
```

**Key Locations**:
- **Agent Implementations**: `libs/azure-ai/langchain_azure_ai/agents/`
- **Evaluation Framework**: `libs/azure-ai/langchain_azure_ai/evaluation/`
- **API Server**: `libs/azure-ai/langchain_azure_ai/server/__init__.py`
- **Copilot Studio Integration**: `libs/azure-ai/langchain_azure_ai/server/copilot_studio.py`
- **Azure Deployment**: `infra/` (Bicep templates, deployment scripts)

---

## Test Directory (`tests/`)

Top-level integration tests and evaluation tests:

```
tests/
├── __init__.py
└── evaluation/
    ├── __init__.py
    └── test_deep_agents_evaluation.py     # DeepAgents evaluation suite (14 tests)
```

**Purpose**: High-level integration tests and agent evaluation tests.

**Note**: Each library package (`libs/*/`) has its own `tests/` directory for unit tests specific to that package. The root `tests/` directory is for integration tests that span multiple packages.

---

## Test Results Directory (`test_results/`)

**Status**: Gitignored (artifacts only, not committed)

```
test_results/
├── deepagents_evaluation_summary.json     # Test summary JSON
└── test_results_deepagents.log            # Detailed test logs
```

**Purpose**: Local test artifacts, reports, and logs. Generated during test execution.

---

## Evaluation Framework Architecture

### Framework Components

1. **Base Evaluators** (`base_evaluators.py`)
   - ResponseQualityEvaluator
   - TaskCompletionEvaluator
   - FactualAccuracyEvaluator
   - CoherenceEvaluator
   - SafetyEvaluator

2. **Multi-Turn Evaluators** (`multi_turn_evaluators.py`)
   - IntentCompletionEvaluator
   - ContextCoherenceEvaluator
   - ToolSequenceEvaluator
   - ConversationFlowEvaluator

3. **Integration Modules**
   - LangSmith: Dataset sync, offline/online evaluation, tracing
   - Azure AI Foundry: Groundedness, relevance, coherence metrics
   - Performance Metrics: Response time, tokens, cost, satisfaction

4. **Test Datasets** (`datasets.py`)
   - 8 agents with 11 test cases
   - IT Agents: it_operations, it_helpdesk, servicenow, hitl_support
   - Enterprise: research, code_assistant, content, data_analyst
   - DeepAgents: IT Operations, Sales Intelligence, Recruitment

### API Endpoints

All evaluation endpoints in `libs/azure-ai/langchain_azure_ai/server/__init__.py`:

1. `POST /api/evaluate/{agent_name}` - Evaluate single response
2. `GET /api/metrics/{agent_name}` - Get performance metrics
3. `GET /api/evaluation/datasets` - List all datasets
4. `POST /api/evaluation/run-offline` - Run LangSmith evaluation
5. `POST /api/evaluation/run-foundry` - Run Azure AI Foundry evaluation
6. `GET /api/evaluation/langsmith/status` - Check LangSmith status

---

## Configuration Files

### Environment Configuration (`.env`)

**Status**: Gitignored (contains secrets)

Contains:
- Azure OpenAI credentials (endpoint, API key, deployment)
- LangSmith API key and project configuration
- Application Insights connection string
- Azure AI Foundry configuration (optional)

### Example Configuration (`.env.example`)

Template for setting up environment variables. Safe to commit.

### Python Configuration (`pyproject.toml`)

Project metadata, dependencies, and build configuration for the monorepo.

---

## Git Ignore Rules (`.gitignore`)

Key exclusions:
- Virtual environments (`.venv*`, `venv*/`)
- Environment files (`.env`, `.env.local`)
- Test artifacts (`test_results/`, `*.log`)
- Python bytecode (`__pycache__/`, `*.pyc`)
- IDE files (`.vscode/settings.json`, `.idea/`)
- Azure credentials (`*credentials*.json`, `gha-creds-*.json`)
- Build artifacts (`dist/`, `build/`, `*.egg-info/`)
- Test caches (`.pytest_cache/`, `.mypy_cache/`)

---

## Development Workflow

### Setup

```bash
# Clone repository
git clone https://github.com/abhilashjaiswal0110/langchain-azure.git
cd langchain-azure

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e libs/azure-ai

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Running Evaluations

```bash
# Run all DeepAgents evaluation tests
pytest tests/evaluation/test_deep_agents_evaluation.py -v -s

# Run specific agent test
pytest tests/evaluation/test_deep_agents_evaluation.py::TestDeepAgentsEvaluation::test_single_response_evaluation[it_operations] -v

# Generate evaluation summary
pytest tests/evaluation/test_deep_agents_evaluation.py::test_generate_evaluation_summary -v
```

### Running the Server

```bash
cd libs/azure-ai
python -m langchain_azure_ai.server
```

Server runs on `http://localhost:8000` with evaluation endpoints available.

---

## Integration Points

### Observability Stack

1. **LangSmith**
   - Project: `azure-foundry-agents`
   - Dashboard: https://smith.langchain.com
   - Features: Tracing, dataset management, experiments

2. **Azure Monitor / Application Insights**
   - OpenTelemetry integration
   - Custom spans for evaluation metrics
   - Performance dashboards

3. **Local Metrics**
   - Performance tracking via `agent_metrics.py`
   - Cost estimation and token usage
   - Success rates and response times

### Governance

1. **Azure AI Foundry**
   - Built-in metrics (groundedness, relevance, coherence)
   - Evaluation jobs and reports
   - Requires Azure Entra configuration

2. **LangSmith Evaluations**
   - Offline evaluations against datasets
   - Custom evaluator support
   - Experiment comparison

---

## Best Practices Followed

### GitHub Enterprise Standards

✅ **Clean Root Directory**
- Only essential configuration and documentation files
- No temporary files or artifacts
- Clear separation of concerns

✅ **Logical Organization**
- Documentation in `docs/`
- Source code in `libs/`
- Tests in `tests/` and `libs/*/tests/`
- Artifacts in `test_results/` (gitignored)

✅ **Proper .gitignore**
- Excludes all sensitive data
- Excludes all generated artifacts
- Includes example configurations

✅ **Monorepo Structure**
- Multiple packages under `libs/`
- Each package self-contained with own tests
- Shared utilities and common patterns

✅ **Documentation**
- Comprehensive README.md
- Architecture documentation (Agents.md, Knowledge.md)
- Implementation guides (EVALUATION_*.md)
- This structure document

### Python Best Practices

✅ **Package Structure**
- Proper `__init__.py` files with exports
- Clear module organization
- Type hints where applicable

✅ **Testing**
- pytest framework with fixtures
- Parametrized tests for agent coverage
- Integration and unit tests separated

✅ **Configuration Management**
- Environment variables for secrets
- Pydantic for settings validation
- Example configurations provided

---

## Maintenance

### Adding New Agents

1. Create agent implementation in `libs/azure-ai/langchain_azure_ai/agents/`
2. Add test dataset in `libs/azure-ai/langchain_azure_ai/evaluation/datasets.py`
3. Create tests in `tests/evaluation/`
4. Update documentation

### Adding New Evaluators

1. Implement evaluator in `base_evaluators.py` or `multi_turn_evaluators.py`
2. Export in `evaluation/__init__.py`
3. Add tests in `tests/evaluation/`
4. Update documentation

### Running CI/CD

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Evaluation Tests
  run: |
    pytest tests/evaluation/ -v --junitxml=test-results.xml

- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test-results.xml
```

---

## Security Considerations

### Secrets Management

- Never commit `.env` file
- Use Azure Key Vault for production secrets
- Rotate API keys regularly
- Follow principle of least privilege

### Data Privacy

- No PII in test datasets
- SafetyEvaluator detects sensitive content
- All evaluations run in your Azure subscription
- LangSmith traces stay within your project

### Audit Trail

- All executions traced in LangSmith
- Performance metrics logged
- Cost tracking for governance
- Test results timestamped

---

## Support and Resources

### Documentation

- **Quick Start**: [docs/README_EVALUATION.md](README_EVALUATION.md)
- **Complete Guide**: [docs/EVALUATION_FRAMEWORK_COMPLETE.md](EVALUATION_FRAMEWORK_COMPLETE.md)
- **Test Results**: [docs/DEEPAGENTS_EVALUATION_RESULTS.md](DEEPAGENTS_EVALUATION_RESULTS.md)
- **Implementation**: [docs/EVALUATION_IMPLEMENTATION.md](EVALUATION_IMPLEMENTATION.md)

### Dashboards

- **LangSmith**: https://smith.langchain.com (Project: azure-foundry-agents)
- **Azure Portal**: Application Insights (if configured)
- **Local**: test_results/ directory

### External Resources

- LangChain Documentation: https://python.langchain.com
- Azure AI Foundry: https://azure.microsoft.com/products/ai-foundry
- LangSmith: https://docs.smith.langchain.com

---

## Version History

- **2026-01-27**: Initial structure documentation
- Repository reorganized following GitHub enterprise best practices
- Evaluation framework fully implemented and tested
- All redundant files and folders removed

---

**Maintained By**: Azure AI Foundry Team
**Repository**: https://github.com/abhilashjaiswal0110/langchain-azure
**Status**: Production Ready ✅
