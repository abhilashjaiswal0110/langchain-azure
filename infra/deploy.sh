#!/bin/bash
# ============================================================================
# Azure Deployment Script for LangChain Agents with Copilot Studio Integration
# ============================================================================
# This script deploys the Azure infrastructure for running LangChain agents
# with Microsoft Copilot Studio integration.
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Bicep CLI installed (az bicep install)
#
# Usage:
#   ./deploy.sh -e prod -l eastus
#   ./deploy.sh --environment dev --skip-build
# ============================================================================

set -e

# Default values
ENVIRONMENT="prod"
LOCATION="eastus"
BASE_NAME="langchain-agents"
SKIP_BUILD=false
SKIP_DEPLOY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -l|--location)
            LOCATION="$2"
            shift 2
            ;;
        -r|--resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set resource group if not provided
if [ -z "$RESOURCE_GROUP" ]; then
    RESOURCE_GROUP="rg-${BASE_NAME}-${ENVIRONMENT}"
fi

echo "============================================"
echo "  Azure LangChain Agents Deployment"
echo "============================================"
echo ""
echo "Environment:    $ENVIRONMENT"
echo "Location:       $LOCATION"
echo "Resource Group: $RESOURCE_GROUP"
echo ""

# ============================================================================
# Prerequisites Check
# ============================================================================

echo "[1/6] Checking prerequisites..."

# Check Azure CLI
if ! command -v az &> /dev/null; then
    echo "  ✗ Azure CLI not found. Install from https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi
echo "  ✓ Azure CLI available"

# Check login status
if ! az account show &> /dev/null; then
    echo "  ✗ Not logged in. Run 'az login' first."
    exit 1
fi
ACCOUNT_NAME=$(az account show --query "name" -o tsv)
echo "  ✓ Subscription: $ACCOUNT_NAME"

# Check/Install Bicep
if ! az bicep version &> /dev/null; then
    echo "  Installing Bicep CLI..."
    az bicep install
fi
echo "  ✓ Bicep CLI available"

# ============================================================================
# Generate API Key if not set
# ============================================================================

echo ""
echo "[2/6] Setting up Copilot API key..."

if [ -z "$COPILOT_API_KEY" ]; then
    # Try openssl first, then fall back to Python
    if command -v openssl >/dev/null 2>&1; then
        COPILOT_API_KEY=$(openssl rand -hex 32)
        echo "  ✓ Generated secure API key using openssl"
    elif command -v python3 >/dev/null 2>&1; then
        COPILOT_API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        echo "  ✓ Generated secure API key using python3"
    elif command -v python >/dev/null 2>&1; then
        COPILOT_API_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
        echo "  ✓ Generated secure API key using python"
    else
        echo "  ✗ Unable to generate COPILOT_API_KEY: neither 'openssl' nor 'python' is available."
        echo "    Please install 'openssl' or 'python', or set COPILOT_API_KEY manually and re-run this script."
        exit 1
    fi

    echo ""
    echo "  IMPORTANT: Save this API key securely!"
    echo "  COPILOT_API_KEY=$COPILOT_API_KEY"
    echo ""
fi

# ============================================================================
# Create Resource Group
# ============================================================================

echo "[3/6] Creating resource group..."

if ! az group show --name "$RESOURCE_GROUP" &> /dev/null; then
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none
    echo "  ✓ Created resource group: $RESOURCE_GROUP"
else
    echo "  ✓ Resource group exists: $RESOURCE_GROUP"
fi

# ============================================================================
# Deploy Infrastructure
# ============================================================================

if [ "$SKIP_DEPLOY" = false ]; then
    echo ""
    echo "[4/6] Deploying Azure infrastructure..."
    echo "  This may take 5-10 minutes..."

    DEPLOY_PARAMS="environment=$ENVIRONMENT location=$LOCATION copilotApiKey=$COPILOT_API_KEY"

    if [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
        DEPLOY_PARAMS="$DEPLOY_PARAMS azureOpenAIEndpoint=$AZURE_OPENAI_ENDPOINT"
    fi

    if [ -n "$AZURE_OPENAI_API_KEY" ]; then
        DEPLOY_PARAMS="$DEPLOY_PARAMS azureOpenAIKey=$AZURE_OPENAI_API_KEY"
    fi

    # LangSmith dual observability (optional)
    if [ -n "$LANGCHAIN_API_KEY" ]; then
        DEPLOY_PARAMS="$DEPLOY_PARAMS langsmithApiKey=$LANGCHAIN_API_KEY"
        echo "  ✓ LangSmith dual tracing enabled"
    fi

    if [ -n "$LANGCHAIN_PROJECT" ]; then
        DEPLOY_PARAMS="$DEPLOY_PARAMS langsmithProject=$LANGCHAIN_PROJECT"
    fi

    DEPLOYMENT=$(az deployment group create \
        --resource-group "$RESOURCE_GROUP" \
        --template-file main.bicep \
        --parameters $DEPLOY_PARAMS \
        --output json)

    if [ $? -ne 0 ]; then
        echo "  ✗ Deployment failed"
        exit 1
    fi

    echo "  ✓ Infrastructure deployed successfully"

    # Get outputs using az query (no jq dependency)
    CONTAINER_APP_URL=$(az deployment group show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$(az deployment group list --resource-group "$RESOURCE_GROUP" --query '[0].name' -o tsv)" \
        --query 'properties.outputs.containerAppUrl.value' -o tsv 2>/dev/null || echo "")

    PLUGIN_MANIFEST_URL=$(az deployment group show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$(az deployment group list --resource-group "$RESOURCE_GROUP" --query '[0].name' -o tsv)" \
        --query 'properties.outputs.copilotPluginManifestUrl.value' -o tsv 2>/dev/null || echo "")

    OPENAPI_URL=$(az deployment group show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$(az deployment group list --resource-group "$RESOURCE_GROUP" --query '[0].name' -o tsv)" \
        --query 'properties.outputs.copilotOpenApiUrl.value' -o tsv 2>/dev/null || echo "")

    ACR_LOGIN_SERVER=$(az deployment group show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$(az deployment group list --resource-group "$RESOURCE_GROUP" --query '[0].name' -o tsv)" \
        --query 'properties.outputs.acrLoginServer.value' -o tsv 2>/dev/null || echo "")
fi

# ============================================================================
# Build and Push Container
# ============================================================================

if [ "$SKIP_BUILD" = false ] && [ -n "$ACR_LOGIN_SERVER" ]; then
    echo ""
    echo "[5/6] Building and pushing container image..."

    ACR_NAME="${ACR_LOGIN_SERVER%.azurecr.io}"
    az acr login --name "$ACR_NAME"

    IMAGE_TAG="$ACR_LOGIN_SERVER/langchain-agents:latest"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"

    if [ -f "$REPO_ROOT/libs/azure-ai/Dockerfile" ]; then
        docker build -t "$IMAGE_TAG" -f "$REPO_ROOT/libs/azure-ai/Dockerfile" "$REPO_ROOT"
        docker push "$IMAGE_TAG"
        echo "  ✓ Container image pushed: $IMAGE_TAG"
    else
        echo "  ! Dockerfile not found. Skipping build."
    fi
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "[6/6] Deployment Complete!"
echo "============================================"
echo ""
echo "Copilot Studio Integration URLs:"
echo "  App URL:         $CONTAINER_APP_URL"
echo "  Plugin Manifest: $PLUGIN_MANIFEST_URL"
echo "  OpenAPI Spec:    $OPENAPI_URL"
echo ""
echo "Next Steps:"
echo "  1. Go to https://copilotstudio.microsoft.com"
echo "  2. Create or edit a Custom Copilot"
echo "  3. Add Action > Custom Connector > OpenAPI"
echo "  4. Import from URL: $OPENAPI_URL"
echo "  5. Configure API Key authentication"
echo ""
echo "Environment Variables for .env:"
echo "  COPILOT_API_KEY=$COPILOT_API_KEY"
echo "  AZURE_CONTAINER_APP_URL=$CONTAINER_APP_URL"
echo ""
echo "Observability Configuration:"
echo "  Azure App Insights: Enabled (auto-configured)"
if [ -n "$LANGCHAIN_API_KEY" ]; then
    echo "  LangSmith Tracing: Enabled"
    echo "  LangSmith Project: ${LANGCHAIN_PROJECT:-langchain-agents}"
else
    echo "  LangSmith Tracing: Disabled (set LANGCHAIN_API_KEY to enable)"
fi
echo ""
echo "Document Processing Endpoint:"
echo "  POST $CONTAINER_APP_URL/api/copilot/document"
echo "  - Supports: summarize, extract_text, extract_tables, analyze"
echo "  - Accepts: base64 content or URL"
echo ""
