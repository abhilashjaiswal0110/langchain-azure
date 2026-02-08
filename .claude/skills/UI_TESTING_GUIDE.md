# Enterprise Hub UI Testing Guide

> **Purpose**: Step-by-step guide for testing the Enterprise Multi-Agent Hub through various UIs

---

## Table of Contents

1. [Testing Prerequisites](#testing-prerequisites)
2. [Local Testing (Command Line)](#local-testing-command-line)
3. [Web UI Testing (Gradio/Streamlit)](#web-ui-testing-gradiostreamlit)
4. [Microsoft Teams Testing](#microsoft-teams-testing)
5. [M365 Copilot Testing](#m365-copilot-testing)
6. [Azure Functions Testing](#azure-functions-testing)
7. [Test Scenarios by Use Case](#test-scenarios-by-use-case)
8. [Monitoring & Observability](#monitoring--observability)
9. [Troubleshooting](#troubleshooting)

---

## Testing Prerequisites

### 1. Complete Setup

```bash
cd samples/enterprise-{hub_name}

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Azure credentials
```

### 2. Required Credentials

```bash
# Essential (all deployments)
AZURE_AI_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key

# Teams Bot (optional)
TEAMS_APP_ID=your-teams-app-id
TEAMS_APP_PASSWORD=your-teams-app-password

# Monitoring (optional)
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
```

### 3. Verify Setup

```bash
# Test Python imports
python -c "from orchestrator import create_hub; print('✅ Setup OK')"
```

---

## Local Testing (Command Line)

### Test 1: Basic Hub Creation

```bash
python3 << 'EOF'
from orchestrator import create_hub

print('Creating hub...')
hub = create_hub()
print('✅ Hub created successfully!')
print(f'Agents: {[k for k in hub.__dict__.keys() if not k.startswith("_")]}')
EOF
```

**Expected Output:**
```
Creating hub...
✅ Hub created successfully!
Agents: ['factory', 'helpdesk', 'knowledge_base', 'servicenow', 'graph']
```

### Test 2: Single Request - IT Support Use Case

```bash
python3 << 'EOF'
from orchestrator import create_hub

hub = create_hub()

# Test request
response = hub.invoke("Hello, I need help resetting my laptop password")

print("\n=== Response ===")
for msg in response["messages"]:
    print(f"\n{msg.type.upper()}:")
    print(msg.content)
print(f"\n=== Routed to: {response.get('current_agent', 'N/A')} ===")
EOF
```

**Expected Output:**
```
=== Response ===