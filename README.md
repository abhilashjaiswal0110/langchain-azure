# 🦜️🔗 LangChain Azure AI Foundry Integration

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.0+](https://img.shields.io/badge/LangChain-1.0+-green.svg)](https://github.com/langchain-ai/langchain)
[![Azure AI](https://img.shields.io/badge/Azure%20AI-Foundry-0078D4.svg)](https://aka.ms/azureai/langchain)

**Enterprise-ready Azure integrations for LangChain**

[Features](#-key-features) • [Quick Start](#-quick-start-with-langchain-azure-ai) • [Documentation](#-documentation) • [Samples](#-samples) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

Enterprise-grade LangChain integration packages for Microsoft Azure, enabling production-ready AI agent development with comprehensive observability, security, and compliance features.

---

## 📦 Available Packages

- [langchain-azure-ai](https://pypi.org/project/langchain-azure-ai/)
- [langchain-azure-dynamic-sessions](https://pypi.org/project/langchain-azure-dynamic-sessions/)
- [langchain-sqlserver](https://pypi.org/project/langchain-sqlserver/)
- [langchain-azure-postgresql](https://pypi.org/project/langchain-azure-postgresql/)
- [langchain-azure-storage](https://pypi.org/project/langchain-azure-storage/)

**Note**: This repository will replace all Azure integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

# Quick Start with langchain-azure-ai

The `langchain-azure-ai` package uses the [Azure AI Foundry SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/sdk-overview?tabs=sync&pivots=programming-language-python). This means you can use the package with a range of models including AzureOpenAI, Cohere, Llama, Phi-3/4, and DeepSeek-R1 to name a few.


LangChain Azure AI also contains:
* [Azure AI Search](./libs/azure-ai/langchain_azure_ai/vectorstores)
* [Cosmos DB](./libs/azure-ai/langchain_azure_ai/vectorstores)
* [Azure AI Agent Service](./libs/azure-ai/langchain_azure_ai/agents)
* [Enterprise Connectors](./libs/azure-ai/langchain_azure_ai/connectors) - Copilot Studio, Teams Bot, Azure Functions

Here's a quick start example to show you how to get started with the Chat Completions model. For more details and tutorials see [Develop with LangChain and LangGraph and models from Azure AI Foundry](https://aka.ms/azureai/langchain).

### Install langchain-azure

```bash
pip install -U langchain-azure-ai
```

### Azure AI Chat Completions Model with Azure OpenAI

```python

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage

model = AzureAIChatCompletionsModel(
    endpoint="https://{your-resource-name}.services.ai.azure.com/openai/v1",
    credential="your-api-key", #if using Entra ID you can should use DefaultAzureCredential() instead
    model="gpt-4o"
)

messages = [
    SystemMessage(
      content="Translate the following from English into Italian"
    ),
    HumanMessage(content="hi!"),
]

model.invoke(messages)
```

```python
AIMessage(content='Ciao!', additional_kwargs={}, response_metadata={'model': 'gpt-4o', 'token_usage': {'input_tokens': 20, 'output_tokens': 3, 'total_tokens': 23}, 'finish_reason': 'stop'}, id='run-0758e7ec-99cd-440b-bfa2-3a1078335133-0', usage_metadata={'input_tokens': 20, 'output_tokens': 3, 'total_tokens': 23})
```

## 📊 Enterprise Observability

Built-in production-ready observability with dual tracing support:

### Azure Monitor Integration
```python
# Enable Azure Monitor OpenTelemetry tracing
export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=..."
export ENABLE_AZURE_MONITOR=true
```

**Features:**
- Request/response tracking with custom dimensions
- Session and user ID tracking
- Agent execution duration metrics
- Token usage monitoring
- Exception tracking with full context
- Live Metrics and Application Insights analytics

### LangSmith Integration
```python
# Enable LangSmith tracing for LangChain agents
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"
export LANGCHAIN_PROJECT="your-project"
```

**Features:**
- Full LangGraph execution traces
- LLM call inspection
- Tool invocation tracking
- Session continuity
- Dataset creation and evaluation

### Session & User Tracking
```python
# Track users and sessions across conversations
response = agent.chat(
    message="Hello",
    thread_id="session-123",
    user_id="user-456",
    metadata={"environment": "production", "source": "web-app"}
)
```

**Tracked Dimensions:**
- `session_id`: Conversation continuity
- `user_id`: User-scoped analytics
- `agent_name`: Agent identification
- `message_length`: Input metrics
- `response_length`: Output metrics
- Custom metadata fields

### Azure AI Chat Completions Model with DeepSeek-R1

```python

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage

model = AzureAIChatCompletionsModel(
    endpoint="https://{your-resource-name}.services.ai.azure.com/models",
    credential="your-api-key", #if using Entra ID you can should use DefaultAzureCredential() instead
    model="DeepSeek-R1",
)

messages = [
    HumanMessage(content="Translate the following from English into Italian: \"hi!\"")
]

message_stream = model.stream(messages)
print(' '.join(chunk.content for chunk in message_stream))
```

```python
 <think>
 Okay ,  the  user  just  sent  " hi !"  and  I  need  to  translate  that  into  Italian .  Let  me  think .  " Hi "  is  an  informal  greeting ,  so  in  Italian ,  the  equivalent  would  be  " C iao !"  But  wait ,  there  are  other  options  too .  Sometimes  people  use  " Sal ve ,"  which  is  a  bit  more  neutral ,  but  " C iao "  is  more  common  in  casual  settings .  The  user  probably  wants  a  straightforward  translation ,  so  " C iao !"  is  the  safest  bet  here .  Let  me  double -check  to  make  sure  there 's  no  nuance  I 'm  missing .  N ope ,  " C iao "  is  definitely  the  right  choice  for  translating  " hi !"  in  an  informal  context .  I 'll  go  with  that .
 </think>

 C iao !
```

## LangGraph and Azure AI Agent Service

You can build multi agent graphs in LangGraph by using the integration with Azure AI Foundry Agent Service. The class `AgentServiceFactory` allows you to create agents and nodes that can be used to compose graphs.

First create the `AgentServiceFactory` class to interface with the service.

```python
from langchain_azure_ai.agents import AgentServiceFactory
from azure.identity import DefaultAzureCredential

factory = AgentServiceFactory(
    project_endpoint=(
        "https://resource.services.ai.azure.com/api/projects/demo-project",
    ),
    credential=DefaultAzureCredential()
)
```

Then use the `create_declarative_chat_agent` to create a React agent with 2 nodes: an Azure AI Foundry Agent that runs in the cloud,
and a Tool node that can handle tool calling that is provided locally in your code.

```python
agent = factory.create_declarative_chat_agent(
    name="my-echo-agent",
    model="gpt-4.1",
    instructions="You are a helpful AI assistant that always replies back
                  "saying the opposite of what the user says.",
)
```

Then, try it out:

```python
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="I'm a genius and I love programming!")]
state = agent.invoke({"messages": messages})

for m in state['messages']:
    m.pretty_print()
```

You can also create a node manually to compose in your graph:

```python
from langchain_azure_ai.agents.prebuilt.tools import AgentServiceBaseTool
from azure.ai.agents.models import CodeInterpreterTool

coder_node = factory.create_declarative_chat_node(
    name="code-interpreter-agent",
    model="gpt-4.1",
    instructions="You are a helpful assistant that can run Python code.",
    tools=[AgentServiceBaseTool(tool=CodeInterpreterTool())],
)

builder.add_node("coder", coder_node)
```

## Using LangChain Azure AI with init_chat_model

To use LangChain Azure AI with `init_chat_model` you must set the "AZURE_AI_ENDPOINT" and "AZURE_AI_CREDENTIAL" environment variables.

```python
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

os.environ["AZURE_AI_ENDPOINT"] = os.getenv("AZURE_ENDPOINT")
os.environ["AZURE_AI_CREDENTIAL"] = os.getenv("AZURE_CREDENTIAL")  # Changed from AZURE_AI_API_KEY

model = init_chat_model("azure_ai:gpt-5-mini")
```

## 🔌 Enterprise Connectors

The `langchain-azure-ai` package includes enterprise-grade connectors for seamless integration with Microsoft 365 and Azure services.

### Microsoft 365 Copilot Integration

Export your LangChain agents to Microsoft 365 Copilot:

```python
from langchain_azure_ai.connectors import CopilotStudioConnector, CopilotStudioConfig

config = CopilotStudioConfig.from_env()  # or construct manually
connector = CopilotStudioConnector(config)
manifest = connector.export_agent(
    wrapper=my_agent,
    name="IT Helpdesk",
    description="Enterprise IT support agent"
)

# Create complete M365 Copilot plugin
connector.create_m365_copilot_plugin(
    wrapper=my_agent,
    name="IT Helpdesk",
    description="Enterprise IT support agent",
    api_base_url="https://your-api.example.com"
)
```

### Copilot Studio REST API (New!)

The server includes ready-to-use REST API endpoints for Microsoft Copilot Studio custom connectors:

**Endpoints:**
- `GET /api/copilot/openapi.json` - OpenAPI 2.0 (Swagger) specification for custom connector import
- `GET /.well-known/ai-plugin.json` - AI plugin manifest for Microsoft 365 Copilot
- `POST /api/copilot/chat` - Main chat endpoint with automatic agent routing
- `POST /api/copilot/chat/{agent_id}` - Direct chat with specific agent
- `GET /api/copilot/agents` - List available agents and capabilities

**Setup in Copilot Studio:**
1. Import custom connector from: `https://your-app.azurecontainerapps.io/api/copilot/openapi.json`
2. Configure API Key authentication (X-API-Key header)
3. Create topics using the Chat action

📖 **See [Copilot Studio Integration Guide](./docs/COPILOT_STUDIO_INTEGRATION.md)** for complete setup instructions.

### Microsoft Teams Bot Integration

Deploy your agent as a Teams bot:

```python
from langchain_azure_ai.connectors import TeamsBotConnector, TeamsBotConfig
from fastapi import FastAPI

app = FastAPI()
config = TeamsBotConfig.from_env()  # or construct manually
bot = TeamsBotConnector(config)
bot.register_agent("helpdesk", helpdesk_agent)

# Add Teams routes to FastAPI app
app.include_router(bot.create_fastapi_routes())

# Generate Teams app manifest
manifest = bot.generate_manifest(
    base_url="https://your-domain.com"
)
```

### Azure Functions Deployment

Deploy your agents as serverless Azure Functions:

```python
from langchain_azure_ai.connectors import (
    AzureFunctionsDeployer,
    FunctionAppConfig,
    ScalingConfig,
)

config = FunctionAppConfig(
    name="my-functions",
    resource_group="my-rg",
    scaling=ScalingConfig(min_instances=1, max_instances=10),
)

deployer = AzureFunctionsDeployer(config)
deployer.generate_scaffold(
    output_dir="functions_app",
    wrappers={"helpdesk": helpdesk_agent},
)

# Generates:
# - Function app code
# - Bicep templates for infrastructure
# - GitHub Actions CI/CD workflows
# - Deploy scripts (bash & PowerShell)
```

# Welcome Contributors

Hi there! Thank you for even being interested in contributing to LangChain-Azure.
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether they involve new features, improved infrastructure, better documentation, or bug fixes.


# Contribute Code

To contribute to this project, please follow the ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow.

Please follow the checked-in pull request template when opening pull requests. Note related issues and tag relevant
maintainers.

Pull requests cannot land without passing the formatting, linting, and testing checks first. See [Testing](#testing) and
[Formatting and Linting](#formatting-and-linting) for how to run these checks locally.

It's essential that we maintain great documentation and testing. If you:
- Fix a bug
  - Add a relevant unit or integration test when possible.
- Make an improvement
  - Update unit and integration tests when relevant.
- Add a feature
  - Add unit and integration tests.

If there's something you'd like to add or change, opening a pull request is the
best way to get our attention. Please tag one of our maintainers for review.

## Dependency Management: Poetry and other env/dependency managers

This project utilizes [Poetry](https://python-poetry.org/) v1.7.1+ as a dependency manager.

❗Note: *Before installing Poetry*, if you use `Conda`, create and activate a new Conda env (e.g. `conda create -n langchain python=3.9`)

Install Poetry: **[documentation on how to install it](https://python-poetry.org/docs/#installation)**.

❗Note: If you use `Conda` or `Pyenv` as your environment/package manager, after installing Poetry,
tell Poetry to use the virtualenv python environment (`poetry config virtualenvs.prefer-active-python true`)

## Different packages

This repository contains four packages with Azure integrations with LangChain:
- [langchain-azure-ai](https://pypi.org/project/langchain-azure-ai/)
- [langchain-azure-dynamic-sessions](https://pypi.org/project/langchain-azure-dynamic-sessions/)
- [langchain-sqlserver](https://pypi.org/project/langchain-sqlserver/)
- [langchain-azure-storage](https://pypi.org/project/langchain-azure-storage/)

Each of these has its own development environment. Docs are run from the top-level makefile, but development
is split across separate test & release flows.

## Repository Structure

If you plan on contributing to LangChain-Google code or documentation, it can be useful
to understand the high level structure of the repository.

LangChain-Azure is organized as a [monorepo](https://en.wikipedia.org/wiki/Monorepo) that contains multiple packages.

Here's the structure visualized as a tree:

```text
.
├── libs
│   ├── azure-ai
│   ├── azure-dynamic-sessions
│   ├── azure-storage
│   ├── sqlserver
```

## Local Development Dependencies

Install development requirements (for running langchain, running examples, linting, formatting, tests, and coverage):

```bash
poetry install --with lint,typing,test,test_integration
```

Then verify dependency installation:

```bash
make test
```

If during installation you receive a `WheelFileValidationError` for `debugpy`, please make sure you are running
Poetry v1.6.1+. This bug was present in older versions of Poetry (e.g. 1.4.1) and has been resolved in newer releases.
If you are still seeing this bug on v1.6.1+, you may also try disabling "modern installation"
(`poetry config installer.modern-installation false`) and re-installing requirements.
See [this `debugpy` issue](https://github.com/microsoft/debugpy/issues/1246) for more details.

## Code Formatting

Formatting for this project is done via [ruff](https://docs.astral.sh/ruff/rules/).

To run formatting for a library, run the same command from the relevant library directory:

```bash
cd libs/{LIBRARY}
make format
```

Additionally, you can run the formatter only on the files that have been modified in your current branch as compared to the master branch using the format_diff command:

```bash
make format_diff
```

This is especially useful when you have made changes to a subset of the project and want to ensure your changes are properly formatted without affecting the rest of the codebase.

## Linting

Linting for this project is done via a combination of [ruff](https://docs.astral.sh/ruff/rules/) and [mypy](http://mypy-lang.org/).

To run linting for docs, cookbook and templates:

```bash
make lint
```

To run linting for a library, run the same command from the relevant library directory:

```bash
cd libs/{LIBRARY}
make lint
```

In addition, you can run the linter only on the files that have been modified in your current branch as compared to the master branch using the lint_diff command:

```bash
make lint_diff
```

This can be very helpful when you've made changes to only certain parts of the project and want to ensure your changes meet the linting standards without having to check the entire codebase.

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

## Spellcheck

Spellchecking for this project is done via [codespell](https://github.com/codespell-project/codespell).
Note that `codespell` finds common typos, so it could have false-positive (correctly spelled but rarely used) and false-negatives (not finding misspelled) words.

To check spelling for this project:

```bash
make spell_check
```

To fix spelling in place:

```bash
make spell_fix
```

If codespell is incorrectly flagging a word, you can skip spellcheck for that word by adding it to the codespell config in the `pyproject.toml` file.

```python
[tool.codespell]
...
# Add here:
ignore-words-list =...
```

## Testing

All of our packages have unit tests and integration tests, and we favor unit tests over integration tests.

Unit tests run on every pull request, so they should be fast and reliable.

Integration tests run once a day, and they require more setup, so they should be reserved for confirming interface points with external services.

### Unit Tests

Unit tests cover modular logic that does not require calls to outside APIs.
If you add new logic, please add a unit test.
In unit tests we check pre/post processing and mocking all external dependencies.

To install dependencies for unit tests:

```bash
poetry install --with test
```

To run unit tests:

```bash
make test
```

To run unit tests in Docker:

```bash
make docker_tests
```

To run a specific test:

```bash
TEST_FILE=tests/unit_tests/test_imports.py make test
```

### Integration Tests

Integration tests cover logic that requires making calls to outside APIs (often integration with other services).
If you add support for a new external API, please add a new integration test.

**Warning:** Almost no tests should be integration tests.

  Tests that require making network connections make it difficult for other
  developers to test the code.

  Instead favor relying on `responses` library and/or mock.patch to mock
  requests using small fixtures.

To install dependencies for integration tests:

```bash
poetry install --with test,test_integration
```

To run integration tests:

```bash
make integration_tests
```


For detailed information on how to contribute, see [LangChain contribution guide](https://python.langchain.com/docs/contributing/).


---

## � Security

This repository follows enterprise security best practices:

- **No Secrets in Code**: All credentials via environment variables or Azure Key Vault
- **Azure AD Integration**: Support for Managed Identity and Entra ID
- **Defense in Depth**: Multi-layer security approach
- **Compliance Ready**: GDPR, SOC 2, ISO 27001 aligned

📚 **Security Documentation**:
- [SECURITY.md](SECURITY.md) - Comprehensive security guidelines
- [.env.example](.env.example) - Secure configuration template

**Report Security Issues**: abhilashjaiswal0110@gmail.com

---

## 📚 Additional Documentation

- **[Knowledge.md](Knowledge.md)** - Repository architecture and integrations
- **[Agents.md](Agents.md)** - Agent development guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**Production Ready** • **Enterprise Grade** • **Azure Native**

⭐ Star this repo if you find it useful!

</div>
