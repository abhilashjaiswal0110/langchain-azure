# ============================================================================
# Azure Deployment Script for LangChain Agents with Copilot Studio Integration
# ============================================================================
# This script deploys the Azure infrastructure for running LangChain agents
# with Microsoft Copilot Studio integration.
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - PowerShell 7+ recommended
#   - Bicep CLI installed (az bicep install)
#
# Usage:
#   .\deploy.ps1 -Environment prod -Location eastus
#   .\deploy.ps1 -Environment dev -SkipBuild
# ============================================================================

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet('dev', 'staging', 'prod')]
    [string]$Environment = 'prod',

    [Parameter()]
    [string]$Location = 'eastus',

    [Parameter()]
    [string]$ResourceGroup = '',

    [Parameter()]
    [switch]$SkipBuild,

    [Parameter()]
    [switch]$SkipDeploy,

    [Parameter()]
    [string]$AzureOpenAIEndpoint = $env:AZURE_OPENAI_ENDPOINT,

    [Parameter()]
    [string]$AzureOpenAIKey = $env:AZURE_OPENAI_API_KEY,

    [Parameter()]
    [string]$CopilotApiKey = $env:COPILOT_API_KEY
)

$ErrorActionPreference = 'Stop'

# ============================================================================
# Configuration
# ============================================================================

$BaseName = 'langchain-agents'
if ([string]::IsNullOrEmpty($ResourceGroup)) {
    $ResourceGroup = "rg-$BaseName-$Environment"
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Azure LangChain Agents Deployment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Environment:    $Environment" -ForegroundColor Yellow
Write-Host "Location:       $Location" -ForegroundColor Yellow
Write-Host "Resource Group: $ResourceGroup" -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# Prerequisites Check
# ============================================================================

Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Blue

# Check Azure CLI
try {
    $azVersion = az version --output json | ConvertFrom-Json
    Write-Host "  ✓ Azure CLI version: $($azVersion.'azure-cli')" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Azure CLI not found. Install from https://docs.microsoft.com/cli/azure/install-azure-cli" -ForegroundColor Red
    exit 1
}

# Check login status
try {
    $account = az account show --output json | ConvertFrom-Json
    Write-Host "  ✓ Logged in as: $($account.user.name)" -ForegroundColor Green
    Write-Host "  ✓ Subscription: $($account.name)" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Not logged in. Run 'az login' first." -ForegroundColor Red
    exit 1
}

# Check Bicep
try {
    az bicep version | Out-Null
    Write-Host "  ✓ Bicep CLI available" -ForegroundColor Green
} catch {
    Write-Host "  Installing Bicep CLI..." -ForegroundColor Yellow
    az bicep install
}

# ============================================================================
# Generate API Key if not provided
# ============================================================================

if ([string]::IsNullOrEmpty($CopilotApiKey)) {
    Write-Host ""
    Write-Host "[2/6] Generating Copilot API key..." -ForegroundColor Blue
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $CopilotApiKey = [System.BitConverter]::ToString($bytes) -replace '-', '' | ForEach-Object { $_.ToLower() }
    Write-Host "  ✓ Generated secure API key" -ForegroundColor Green
    Write-Host ""
    Write-Host "  IMPORTANT: Save this API key securely!" -ForegroundColor Yellow
    Write-Host "  COPILOT_API_KEY=$CopilotApiKey" -ForegroundColor Cyan
    Write-Host ""
}

# ============================================================================
# Create Resource Group
# ============================================================================

Write-Host "[3/6] Creating resource group..." -ForegroundColor Blue

$rgExists = az group exists --name $ResourceGroup
if ($rgExists -eq 'false') {
    az group create --name $ResourceGroup --location $Location | Out-Null
    Write-Host "  ✓ Created resource group: $ResourceGroup" -ForegroundColor Green
} else {
    Write-Host "  ✓ Resource group exists: $ResourceGroup" -ForegroundColor Green
}

# ============================================================================
# Deploy Infrastructure
# ============================================================================

if (-not $SkipDeploy) {
    Write-Host ""
    Write-Host "[4/6] Deploying Azure infrastructure..." -ForegroundColor Blue
    Write-Host "  This may take 5-10 minutes..." -ForegroundColor Yellow

    $deployParams = @(
        '--resource-group', $ResourceGroup,
        '--template-file', 'main.bicep',
        '--parameters', "environment=$Environment",
        '--parameters', "location=$Location",
        '--parameters', "copilotApiKey=$CopilotApiKey"
    )

    if (-not [string]::IsNullOrEmpty($AzureOpenAIEndpoint)) {
        $deployParams += '--parameters', "azureOpenAIEndpoint=$AzureOpenAIEndpoint"
    }

    if (-not [string]::IsNullOrEmpty($AzureOpenAIKey)) {
        $deployParams += '--parameters', "azureOpenAIKey=$AzureOpenAIKey"
    }

    $deployment = az deployment group create @deployParams --output json | ConvertFrom-Json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Deployment failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "  ✓ Infrastructure deployed successfully" -ForegroundColor Green

    # Get outputs
    $outputs = $deployment.properties.outputs
    $containerAppUrl = $outputs.containerAppUrl.value
    $pluginManifestUrl = $outputs.copilotPluginManifestUrl.value
    $openApiUrl = $outputs.copilotOpenApiUrl.value
    $acrLoginServer = $outputs.acrLoginServer.value
}

# ============================================================================
# Build and Push Container (if not skipped)
# ============================================================================

if (-not $SkipBuild -and -not [string]::IsNullOrEmpty($acrLoginServer)) {
    Write-Host ""
    Write-Host "[5/6] Building and pushing container image..." -ForegroundColor Blue

    # Login to ACR
    az acr login --name ($acrLoginServer -replace '\.azurecr\.io$', '')

    # Build and push
    $imageTag = "$acrLoginServer/langchain-agents:latest"

    Push-Location (Join-Path $PSScriptRoot '..')

    # Check if Dockerfile exists in libs/azure-ai
    $dockerfilePath = "libs/azure-ai/Dockerfile"
    if (Test-Path $dockerfilePath) {
        docker build -t $imageTag -f $dockerfilePath .
        docker push $imageTag
        Write-Host "  ✓ Container image pushed: $imageTag" -ForegroundColor Green
    } else {
        Write-Host "  ! Dockerfile not found. Skipping build." -ForegroundColor Yellow
    }

    Pop-Location
}

# ============================================================================
# Summary
# ============================================================================

Write-Host ""
Write-Host "[6/6] Deployment Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Copilot Studio Integration URLs:" -ForegroundColor Yellow
Write-Host "  App URL:         $containerAppUrl" -ForegroundColor White
Write-Host "  Plugin Manifest: $pluginManifestUrl" -ForegroundColor White
Write-Host "  OpenAPI Spec:    $openApiUrl" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Go to https://copilotstudio.microsoft.com" -ForegroundColor White
Write-Host "  2. Create or edit a Custom Copilot" -ForegroundColor White
Write-Host "  3. Add Action > Custom Connector > OpenAPI" -ForegroundColor White
Write-Host "  4. Import from URL: $openApiUrl" -ForegroundColor Cyan
Write-Host "  5. Configure API Key authentication: $CopilotApiKey" -ForegroundColor Cyan
Write-Host ""
Write-Host "Environment Variables for .env:" -ForegroundColor Yellow
Write-Host "  COPILOT_API_KEY=$CopilotApiKey" -ForegroundColor Cyan
Write-Host "  AZURE_CONTAINER_APP_URL=$containerAppUrl" -ForegroundColor Cyan
Write-Host ""
