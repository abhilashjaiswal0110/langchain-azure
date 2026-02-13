# Microsoft Copilot Studio Integration Guide

This guide provides comprehensive documentation for integrating Azure AI Foundry Agents with Microsoft Copilot Studio. The integration enables your organization to leverage powerful AI agents directly within Microsoft 365 Copilot and Power Platform environments.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Copilot Studio Setup](#copilot-studio-setup)
- [Security Best Practices](#security-best-practices)
- [Observability & Monitoring](#observability--monitoring)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Overview

### Integration Capabilities

The Copilot Studio integration exposes Azure AI Foundry Agents as:

1. **Custom Connectors** - REST APIs consumable by Power Platform and Copilot Studio
2. **AI Plugins** - Microsoft 365 Copilot plugins via ai-plugin.json manifest
3. **OpenAPI Actions** - Swagger 2.0 compliant endpoints for Copilot Studio actions

### Supported Agent Types

| Agent Category | Agents | Capabilities |
|---------------|--------|--------------|
| **IT Support** | Helpdesk, ServiceNow, HITL Support | Password resets, incident management, IT operations |
| **Enterprise** | Research, Content, Data Analyst, Code Assistant, RAG, Document Intelligence | Business analysis, content generation, code review |
| **Deep Agents** | IT Operations, Sales Intelligence, Recruitment, Software Development | Complex multi-step workflows with subagent coordination |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Microsoft Copilot Studio                          │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │   Topics    │ ── │   Actions    │ ── │  Custom Connector   │    │
│  └─────────────┘    └──────────────┘    └─────────────────────┘    │
│                                                   │                  │
└───────────────────────────────────────────────────┼──────────────────┘
                                                    │ HTTPS
                                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Azure Container Apps                              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Server                             │  │
│  │                                                               │  │
│  │  ┌────────────────────┐    ┌────────────────────────────┐   │  │
│  │  │  Copilot Studio    │    │    Standard Agent APIs     │   │  │
│  │  │  Routes            │    │                            │   │  │
│  │  │  /api/copilot/*    │    │  /api/it/*                 │   │  │
│  │  │                    │    │  /api/enterprise/*         │   │  │
│  │  │  • Plugin Manifest │    │  /api/deepagent/*          │   │  │
│  │  │  • OpenAPI Spec    │    │                            │   │  │
│  │  │  • Chat Endpoint   │    │                            │   │  │
│  │  │  • Agent Routing   │    │                            │   │  │
│  │  └────────────────────┘    └────────────────────────────┘   │  │
│  │                    │                                         │  │
│  │                    ▼                                         │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │              Agent Registry                            │ │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │ │  │
│  │  │  │IT Agents │  │Enterprise│  │   Deep Agents    │    │ │  │
│  │  │  └──────────┘  └──────────┘  └──────────────────┘    │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐    │
│  │ Azure OpenAI   │  │ Azure Monitor  │  │    Key Vault       │    │
│  └────────────────┘  └────────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Communication Flow

1. **User Interaction**: User interacts with Copilot Studio bot or Microsoft 365 Copilot
2. **Topic Routing**: Copilot routes request to appropriate action/connector
3. **API Call**: Custom connector calls Azure Container Apps endpoint
4. **Agent Selection**: Copilot routes determine best agent based on message content
5. **Processing**: Selected agent processes request using Azure OpenAI
6. **Response**: Agent returns structured response with suggestions
7. **Display**: Copilot displays response to user with adaptive cards

---

## Prerequisites

### Azure Resources

| Resource | Purpose | Required |
|----------|---------|----------|
| Azure Subscription | Hosting infrastructure | ✅ |
| Azure OpenAI Service | LLM inference | ✅ |
| Azure Container Apps | Application hosting | ✅ |
| Azure Container Registry | Docker image storage | ✅ |
| Application Insights | Monitoring & telemetry | Recommended |
| Azure Key Vault | Secrets management | Recommended |

### Access Requirements

- **Azure Portal**: Contributor access to target resource group
- **Azure AD**: Application registration permissions (for OAuth)
- **Copilot Studio**: System administrator or maker permissions
- **Power Platform**: Environment admin (for custom connectors)

### Tools

- Azure CLI (`az`) version 2.50+
- Docker Desktop (for local builds)
- PowerShell 7+ (Windows) or Bash (Linux/macOS)

---

## Configuration

### Environment Variables

Configure these environment variables in your `.env` file or Azure Key Vault:

#### Required Settings

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Copilot Studio Authentication
COPILOT_API_KEY_ENABLED=true
COPILOT_API_KEY=<generate-secure-key>  # openssl rand -hex 32
```

#### Optional Settings

```bash
# OAuth Configuration (alternative to API Key)
COPILOT_OAUTH_ENABLED=false
COPILOT_CLIENT_ID=your-azure-ad-client-id
AZURE_TENANT_ID=your-tenant-id

# Rate Limiting
COPILOT_RATE_LIMIT_RPM=60

# Plugin Metadata
COPILOT_CONTACT_EMAIL=admin@example.com
COPILOT_LEGAL_URL=https://example.com/privacy
COPILOT_LOGO_URL=https://example.com/logo.png

# CORS (for development)
COPILOT_ALLOWED_ORIGINS=https://your-tenant.api.powerplatform.com
```

### Generating Secure API Key

```bash
# Linux/macOS
openssl rand -hex 32

# PowerShell
[System.Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
```

---

## Deployment

### Option 1: Azure Container Apps (Recommended)

Use the provided Bicep templates for production deployment:

```bash
# Navigate to infrastructure directory
cd infra

# Deploy using PowerShell (Windows)
.\deploy.ps1 -ResourceGroup "rg-copilot-agents" `
             -Location "eastus" `
             -Environment "prod"

# Or using Bash (Linux/macOS)
./deploy.sh --resource-group "rg-copilot-agents" \
            --location "eastus" \
            --environment "prod"
```

The deployment creates:
- Container Apps Environment with auto-scaling
- Container App with health probes
- Azure Container Registry
- Application Insights
- Log Analytics Workspace
- Azure Key Vault (optional)

### Option 2: Manual Azure Deployment

```bash
# 1. Create resource group
az group create --name rg-copilot-agents --location eastus

# 2. Deploy infrastructure
az deployment group create \
  --resource-group rg-copilot-agents \
  --template-file infra/main.bicep \
  --parameters infra/main.bicepparam

# 3. Build and push Docker image
az acr build \
  --registry <your-acr-name> \
  --image agents:latest \
  --file libs/azure-ai/Dockerfile \
  libs/azure-ai/

# 4. Update container app
az containerapp update \
  --name agents-app \
  --resource-group rg-copilot-agents \
  --image <your-acr-name>.azurecr.io/agents:latest
```

### Option 3: Local Development

```bash
# 1. Install dependencies
cd libs/azure-ai
pip install -e ".[all]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start server
python start_server.py
```

---

## Copilot Studio Setup

### Step 1: Create Custom Connector

1. Navigate to **Power Platform Admin Center**
2. Go to **Custom Connectors** → **+ New custom connector**
3. Select **Import from URL**
4. Enter: `https://your-app.azurecontainerapps.io/api/copilot/openapi.json`
5. Configure authentication:
   - Type: **API Key**
   - Parameter name: `X-API-Key`
   - Parameter location: **Header**
6. Click **Create connector**

### Step 2: Configure in Copilot Studio

1. Open **Copilot Studio** at https://copilotstudio.microsoft.com
2. Select your bot or create new
3. Go to **Actions** → **+ Add action**
4. Search for your custom connector
5. Select the **Chat** action
6. Configure input/output mappings:
   - Input: `message` → User input
   - Output: `response` → Bot response

### Step 3: Create Topics

Example topic configuration:

```yaml
Topic: IT Support
Trigger phrases:
  - help with computer
  - IT issue
  - password reset
  - software request

Action: Azure AI Agents - Chat
Input:
  message: ${recognizedText}
  conversationId: ${System.ConversationId}

Response: ${action.response}
Suggestions: ${action.suggestions}
```

### Step 4: Test Integration

1. Open **Test chat** in Copilot Studio
2. Send a test message: "I need help resetting my password"
3. Verify agent responds with appropriate IT helpdesk content
4. Check conversation context is maintained

---

## Security Best Practices

### Authentication

| Method | Use Case | Security Level |
|--------|----------|----------------|
| API Key | Simple integrations | Medium |
| OAuth 2.0 | Enterprise deployments | High |
| Managed Identity | Azure-to-Azure | Highest |

### Network Security

```bicep
// Recommended Container Apps configuration
resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  properties: {
    configuration: {
      ingress: {
        external: true
        transport: 'http2'
        corsPolicy: {
          allowedOrigins: [
            'https://*.powerplatform.com'
            'https://*.dynamics.com'
          ]
          allowedMethods: ['GET', 'POST', 'OPTIONS']
          allowedHeaders: ['*']
        }
      }
    }
  }
}
```

### Secret Management

```bash
# Store secrets in Key Vault
az keyvault secret set \
  --vault-name <vault-name> \
  --name COPILOT-API-KEY \
  --value $(openssl rand -hex 32)

# Reference in Container Apps
az containerapp update \
  --name agents-app \
  --resource-group rg-copilot-agents \
  --set-env-vars "COPILOT_API_KEY=secretref:copilot-api-key"
```

### Rate Limiting

Configure rate limiting to protect against abuse:

```python
# In .env
COPILOT_RATE_LIMIT_RPM=60  # 60 requests per minute per IP
```

---

## Observability & Monitoring

### Azure Monitor Integration

The Copilot Studio routes include built-in telemetry:

```python
# Automatic tracking includes:
# - Request duration (copilot_chat_duration_ms)
# - Agent selection (copilot.agent_id, copilot.agent_type)
# - Conversation context (copilot.conversation_id)
# - Error tracking with stack traces
```

### Application Insights Queries

**Chat Volume by Agent:**
```kusto
customMetrics
| where name == "copilot_chat_duration_ms"
| extend agent_id = tostring(customDimensions.agent_id)
| summarize count() by agent_id, bin(timestamp, 1h)
| render timechart
```

**Error Analysis:**
```kusto
exceptions
| where customDimensions.source == "copilot_studio"
| summarize count() by type, bin(timestamp, 1h)
| render timechart
```

**Response Time Percentiles:**
```kusto
customMetrics
| where name == "copilot_chat_duration_ms"
| summarize
    p50 = percentile(value, 50),
    p95 = percentile(value, 95),
    p99 = percentile(value, 99)
  by bin(timestamp, 5m)
| render timechart
```

### Dashboard Setup

Create Azure Dashboard with:
1. Request rate (requests/minute)
2. Average response time
3. Error rate percentage
4. Agent distribution pie chart
5. Geographic distribution map

---

## Troubleshooting

### Common Issues

#### 401 Unauthorized

**Cause**: Invalid or missing API key

**Solution**:
```bash
# Verify API key is set
echo $COPILOT_API_KEY

# Test endpoint directly
curl -H "X-API-Key: your-key" \
     https://your-app.azurecontainerapps.io/api/copilot/agents
```

#### 503 Service Unavailable

**Cause**: Agent registry not initialized

**Solution**:
1. Check server logs for initialization errors
2. Verify Azure OpenAI credentials
3. Ensure all required environment variables are set

#### CORS Errors

**Cause**: Missing or incorrect CORS configuration

**Solution**:
```bash
# Add Power Platform origins
COPILOT_ALLOWED_ORIGINS=https://your-tenant.api.powerplatform.com,https://copilotstudio.microsoft.com
```

#### Timeout Errors

**Cause**: Agent processing exceeds Copilot timeout (30s default)

**Solution**:
- Increase timeout in Copilot Studio action settings
- Implement async processing for long-running tasks
- Use streaming responses where supported

### Debug Mode

Enable debug logging:

```bash
# In .env
LOG_LEVEL=DEBUG
ENABLE_REQUEST_LOGGING=true
```

### Health Check

Verify system health:

```bash
# Check overall health
curl https://your-app.azurecontainerapps.io/health

# Check Copilot endpoints
curl https://your-app.azurecontainerapps.io/api/copilot/agents

# View plugin manifest
curl https://your-app.azurecontainerapps.io/.well-known/ai-plugin.json
```

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/copilot/chat` | POST | Main chat endpoint with auto-routing |
| `/api/copilot/chat/{agent_id}` | POST | Direct chat with specific agent |
| `/api/copilot/agents` | GET | List available agents |
| `/api/copilot/openapi.json` | GET | OpenAPI 2.0 specification |
| `/api/copilot/plugin-manifest` | GET | AI plugin manifest |
| `/.well-known/ai-plugin.json` | GET | Standard plugin discovery endpoint |

### Chat Request

```json
POST /api/copilot/chat
Content-Type: application/json
X-API-Key: your-api-key

{
  "message": "I need help resetting my password",
  "conversationId": "optional-conversation-id",
  "userId": "user@example.com",
  "channelId": "copilot-studio",
  "locale": "en-US"
}
```

### Chat Response

```json
{
  "response": "I can help you reset your password. Please provide your employee ID...",
  "conversationId": "conv-123",
  "agentId": "helpdesk",
  "agentType": "IT",
  "timestamp": "2024-01-15T10:30:00Z",
  "suggestions": [
    "Reset password",
    "Check ticket status",
    "Request software"
  ],
  "metadata": {
    "duration_ms": 1234.56,
    "channel": "copilot-studio",
    "locale": "en-US"
  }
}
```

### Agent List Response

```json
GET /api/copilot/agents

{
  "agents": [
    {
      "id": "helpdesk",
      "name": "IT Helpdesk Agent",
      "type": "IT",
      "description": "General IT support and troubleshooting",
      "capabilities": ["password_reset", "software_request", "ticket_status"]
    },
    {
      "id": "software_development",
      "name": "Software Development Agent",
      "type": "DeepAgent",
      "description": "Complete SDLC support with subagent coordination",
      "capabilities": ["requirements", "design", "coding", "testing", "deployment"]
    }
  ],
  "total": 12
}
```

---

## Support

For issues and questions:

1. **GitHub Issues**: https://github.com/your-org/langchain-azure/issues
2. **Documentation**: /docs folder in repository
3. **Microsoft Learn**: https://learn.microsoft.com/copilot-studio

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01 | Initial Copilot Studio integration |
| 1.1.0 | 2024-02 | Added Deep Agents support |
| 1.2.0 | 2024-02 | Enhanced observability & security |

---

*Last updated: February 2024*