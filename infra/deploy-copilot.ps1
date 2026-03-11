# ============================================================================
# deploy-copilot.ps1 — Copilot Studio Integration Deployment
# ============================================================================
# Deploys the LangChain Agents FastAPI server as an Azure Container App,
# reusing all existing shared infrastructure in rg-gitba-nonprd-mcp-gewc.
#
# Run from the repository root:
#   .\infra\deploy-copilot.ps1
#
# Flags:
#   -SkipBuild    Skip Docker build/push (reuse existing image in ACR)
#   -SkipDeploy   Skip Bicep deployment (RBAC + secrets grant only)
#   -ImageTag     Docker image tag to build/deploy (default: 'latest')
#
# Prerequisites:
#   1. Edit infra/deploy.config.json — set azureOpenAI.endpoint
#   2. Set AZURE_OPENAI_API_KEY environment variable (or be prompted)
#   3. 'az login' with Contributor access to rg-gitba-nonprd-mcp-gewc
#   4. Docker Desktop running (unless -SkipBuild)
# ============================================================================

[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$SkipDeploy,
    [string]$ImageTag = 'latest'
)

$ErrorActionPreference = 'Stop'
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptDir

# ============================================================================
# Step 1 — Read and Validate Configuration
# ============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  LangChain Agents — Copilot Studio Deployment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[1/7] Reading deploy.config.json..." -ForegroundColor Blue

$ConfigPath = Join-Path $ScriptDir "deploy.config.json"
if (-not (Test-Path $ConfigPath)) {
    Write-Host "  ✗ infra/deploy.config.json not found. Cannot continue." -ForegroundColor Red
    exit 1
}

$Config = Get-Content $ConfigPath -Raw | ConvertFrom-Json

$openAIEndpoint = $Config.azureOpenAI.endpoint
if ($openAIEndpoint -like "*FILL_IN*" -or [string]::IsNullOrWhiteSpace($openAIEndpoint)) {
    Write-Host ""
    Write-Host "  ✗ azureOpenAI.endpoint not configured in deploy.config.json" -ForegroundColor Red
    Write-Host "    Open infra/deploy.config.json and replace the FILL_IN placeholder" -ForegroundColor Yellow
    Write-Host "    with your Azure OpenAI endpoint, e.g.:" -ForegroundColor Yellow
    Write-Host "      https://your-resource.openai.azure.com/" -ForegroundColor Cyan
    exit 1
}

Write-Host "  ✓ Config loaded" -ForegroundColor Green
Write-Host "    Subscription   : $($Config.subscriptionId)" -ForegroundColor White
Write-Host "    Resource Group : $($Config.resourceGroup)" -ForegroundColor White
Write-Host "    Location       : $($Config.location)" -ForegroundColor White
Write-Host "    Container App  : $($Config.newResources.containerAppName)" -ForegroundColor White
Write-Host "    OpenAI Endpoint: $openAIEndpoint" -ForegroundColor White

# ============================================================================
# Step 2 — Collect Secrets
# ============================================================================

Write-Host ""
Write-Host "[2/7] Collecting secrets..." -ForegroundColor Blue

# Azure OpenAI API Key — from env var or interactive prompt
$AzureOpenAIKey = $env:AZURE_OPENAI_API_KEY
if ([string]::IsNullOrWhiteSpace($AzureOpenAIKey)) {
    Write-Host "  AZURE_OPENAI_API_KEY not set in environment." -ForegroundColor Yellow
    $secKey = Read-Host "  Enter Azure OpenAI API Key" -AsSecureString
    $AzureOpenAIKey = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secKey)
    )
}
Write-Host "  ✓ Azure OpenAI API key ready" -ForegroundColor Green

# COPILOT_API_KEY — generate if not already set
$CopilotApiKey = $env:COPILOT_API_KEY
if ([string]::IsNullOrWhiteSpace($CopilotApiKey)) {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $CopilotApiKey = ([System.BitConverter]::ToString($bytes) -replace '-', '').ToLower()
    Write-Host "  ✓ Generated Copilot API key (save this — needed for Copilot Studio setup):" -ForegroundColor Green
    Write-Host "      COPILOT_API_KEY=$CopilotApiKey" -ForegroundColor Cyan
} else {
    Write-Host "  ✓ Using COPILOT_API_KEY from environment" -ForegroundColor Green
}

# Docker push authentication uses 'az acr login' (token-based, no admin credentials needed).

# ============================================================================
# Step 3 — Verify Azure CLI
# ============================================================================

Write-Host ""
Write-Host "[3/7] Verifying Azure CLI..." -ForegroundColor Blue

try {
    $account = az account show -o json | ConvertFrom-Json
    Write-Host "  ✓ Logged in as : $($account.user.name)" -ForegroundColor Green
    Write-Host "  ✓ Subscription : $($account.name) ($($account.id))" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Not logged in. Run 'az login' and retry." -ForegroundColor Red
    exit 1
}

az account set --subscription $Config.subscriptionId 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Could not set subscription $($Config.subscriptionId)" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Active subscription set" -ForegroundColor Green

# ============================================================================
# Step 4 — Grant RBAC Roles + Store Secrets
# ============================================================================

Write-Host ""
Write-Host "[4/7] Granting RBAC roles and storing secrets..." -ForegroundColor Blue

$rg    = $Config.resourceGroup
$mi    = $Config.existingResources.managedIdentity
$acrN  = $Config.existingResources.containerRegistry
$kvN   = $Config.existingResources.keyVault

$miPrincipalId = az identity show --name $mi --resource-group $rg --query principalId -o tsv
$acrId = az acr show --name $acrN --resource-group $rg --query id -o tsv
$kvId  = az keyvault show --name $kvN --resource-group $rg --query id -o tsv

# Try to grant roles. If this account lacks Microsoft.Authorization/roleAssignments/write
# (common in shared enterprise subscriptions) we warn and continue — the existing
# servicenow app already runs with the same identity against the same ACR/KV, so
# the required roles likely already exist at the resource group level.
$rolesGranted = $false

# AcrPull — lets managed identity pull images from the registry
$acrPullExists = az role assignment list `
    --assignee $miPrincipalId --scope $acrId `
    --query "[?roleDefinitionName=='AcrPull'] | length(@)" -o tsv 2>$null

if ($acrPullExists -gt 0) {
    Write-Host "  ✓ AcrPull already exists on $acrN" -ForegroundColor Green
} else {
    Write-Host "  Assigning AcrPull on $acrN..." -ForegroundColor Yellow
    $rbacOut = az role assignment create `
        --assignee $miPrincipalId --role "AcrPull" --scope $acrId --output none 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ AcrPull assigned" -ForegroundColor Green
        $rolesGranted = $true
    } else {
        Write-Host "  ! AcrPull assignment failed (insufficient RBAC permissions)." -ForegroundColor DarkYellow
        Write-Host "    The servicenow app uses the same identity+ACR so the role may" -ForegroundColor DarkYellow
        Write-Host "    exist at RG level — continuing deployment." -ForegroundColor DarkYellow
    }
}

# Key Vault Secrets User — lets managed identity read secrets
$kvRoleExists = az role assignment list `
    --assignee $miPrincipalId --scope $kvId `
    --query "[?roleDefinitionName=='Key Vault Secrets User'] | length(@)" -o tsv 2>$null

if ($kvRoleExists -gt 0) {
    Write-Host "  ✓ Key Vault Secrets User already exists on $kvN" -ForegroundColor Green
} else {
    Write-Host "  Assigning Key Vault Secrets User on $kvN..." -ForegroundColor Yellow
    $rbacOut = az role assignment create `
        --assignee $miPrincipalId --role "Key Vault Secrets User" --scope $kvId --output none 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Key Vault Secrets User assigned" -ForegroundColor Green
        $rolesGranted = $true
    } else {
        Write-Host "  ! Key Vault Secrets User assignment failed (insufficient RBAC permissions)." -ForegroundColor DarkYellow
        Write-Host "    Continuing — secrets are passed directly to Container App config." -ForegroundColor DarkYellow
    }
}

# Store secrets in Key Vault as backup (best-effort — requires Secrets Officer role)
Write-Host "  Backing up secrets to Key Vault ($kvN)..." -ForegroundColor Yellow
az keyvault secret set `
    --vault-name $kvN --name "langchain-azure-openai-key" --value $AzureOpenAIKey `
    --output none 2>&1 | Out-Null
az keyvault secret set `
    --vault-name $kvN --name "langchain-copilot-api-key" --value $CopilotApiKey `
    --output none 2>&1 | Out-Null
Write-Host "  ✓ Secret backup attempted (silent if KV write permission is absent)" -ForegroundColor Green

# ACR admin password for image pull in Container App
$AcrPassword = az acr credential show `
    --name $acrN `
    --resource-group $rg `
    --query 'passwords[0].value' -o tsv 2>&1
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($AcrPassword)) {
    Write-Host "  ✗ Failed to retrieve ACR admin password. Ensure admin is enabled on $acrN" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ ACR admin credentials retrieved" -ForegroundColor Green

# Only wait for RBAC propagation if we actually created new assignments
if ($rolesGranted) {
    Write-Host "  Waiting 35s for new RBAC assignments to propagate..." -ForegroundColor Yellow
    Start-Sleep -Seconds 35
}

# ============================================================================
# Step 5 — Build and Push Docker Image
# ============================================================================

$acrLoginServer = az acr show --name $acrN --resource-group $rg --query loginServer -o tsv
$imageFullTag   = "${acrLoginServer}/langchain-agents:${ImageTag}"

if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "[5/7] Building and pushing Docker image..." -ForegroundColor Blue

    az acr login --name $acrN
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ ACR login failed" -ForegroundColor Red; exit 1
    }

    $dockerContext  = Join-Path $RepoRoot "libs\azure-ai"
    $dockerfilePath = Join-Path $dockerContext "Dockerfile"

    if (-not (Test-Path $dockerfilePath)) {
        Write-Host "  ✗ Dockerfile not found: $dockerfilePath" -ForegroundColor Red; exit 1
    }

    Write-Host "  Building: $imageFullTag" -ForegroundColor Yellow
    docker build -t $imageFullTag -f $dockerfilePath $dockerContext
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ docker build failed" -ForegroundColor Red; exit 1
    }

    Write-Host "  Pushing to ACR..." -ForegroundColor Yellow
    docker push $imageFullTag
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ docker push failed" -ForegroundColor Red; exit 1
    }

    Write-Host "  ✓ Image pushed: $imageFullTag" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[5/7] Build skipped (-SkipBuild). Using existing image: $imageFullTag" -ForegroundColor Yellow
}

# ============================================================================
# Step 6 — Deploy Container App via Bicep
# ============================================================================

if (-not $SkipDeploy) {
    Write-Host ""
    Write-Host "[6/7] Deploying Container App..." -ForegroundColor Blue

    $appInsN = $Config.existingResources.appInsights
    $appInsConnStr = az monitor app-insights component show `
        --app $appInsN --resource-group $rg `
        --query connectionString -o tsv

    $bicepFile      = Join-Path $ScriptDir "main-copilot.bicep"
    $deploymentName = "copilot-agents-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

    # Write a temporary parameters file so secrets never appear on the command line
    $tempParams = [System.IO.Path]::GetTempFileName().Replace('.tmp', '.json')
    $paramsObj = @{
        '$schema'      = 'https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#'
        contentVersion = '1.0.0.0'
        parameters     = @{
            location                   = @{ value = $Config.location }
            environment                = @{ value = $Config.environment }
            containerAppName           = @{ value = $Config.newResources.containerAppName }
            imageTag                   = @{ value = $ImageTag }
            containerEnvName           = @{ value = $Config.existingResources.containerAppsEnvironment }
            acrName                    = @{ value = $Config.existingResources.containerRegistry }
            managedIdentityName        = @{ value = $Config.existingResources.managedIdentity }
            azureOpenAIEndpoint        = @{ value = $openAIEndpoint }
            azureOpenAIDeployment      = @{ value = $Config.azureOpenAI.deploymentName }
            azureOpenAIApiVersion      = @{ value = $Config.azureOpenAI.apiVersion }
            appInsightsConnectionString = @{ value = $appInsConnStr }
            azureOpenAIKey              = @{ value = $AzureOpenAIKey }
            copilotApiKey               = @{ value = $CopilotApiKey }
            acrPassword                 = @{ value = $AcrPassword }
            minReplicas                = @{ value = $Config.scaling.minReplicas }
            maxReplicas                = @{ value = $Config.scaling.maxReplicas }
        }
    } | ConvertTo-Json -Depth 10
    Set-Content -Path $tempParams -Value $paramsObj

    try {
        Write-Host "  Running Bicep deployment..." -ForegroundColor Yellow
        $deployment = az deployment group create `
            --resource-group $rg `
            --name $deploymentName `
            --template-file $bicepFile `
            --parameters "@$tempParams" `
            --output json | ConvertFrom-Json

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ✗ Bicep deployment failed" -ForegroundColor Red; exit 1
        }
    } finally {
        # Always remove the temp file — it contains secrets
        Remove-Item -Path $tempParams -Force -ErrorAction SilentlyContinue
    }

    $outputs              = $deployment.properties.outputs
    $script:containerAppUrl = $outputs.containerAppUrl.value
    $script:openApiUrl      = $outputs.copilotOpenApiUrl.value
    $script:pluginUrl       = $outputs.copilotPluginManifestUrl.value
    $script:healthUrl       = $outputs.healthCheckUrl.value

    Write-Host "  ✓ Container App deployed" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[6/7] Deploy skipped (-SkipDeploy)" -ForegroundColor Yellow
}

# ============================================================================
# Step 7 — Summary and Copilot Studio Setup Instructions
# ============================================================================

Write-Host ""
Write-Host "[7/7] Done!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
if ($script:containerAppUrl) {
    Write-Host "  App URL       : $($script:containerAppUrl)" -ForegroundColor White
    Write-Host "  Health Check  : $($script:healthUrl)" -ForegroundColor White
    Write-Host "  OpenAPI Spec  : $($script:openApiUrl)" -ForegroundColor White
    Write-Host "  Plugin Manifest: $($script:pluginUrl)" -ForegroundColor White
    Write-Host ""
}
Write-Host "  COPILOT_API_KEY : $CopilotApiKey" -ForegroundColor Cyan
Write-Host "  (save this — required for Copilot Studio connector setup)" -ForegroundColor DarkYellow
Write-Host ""
Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
Write-Host " NEXT STEPS — Copilot Studio Setup" -ForegroundColor Cyan
Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
Write-Host ""
Write-Host " 1. Verify health:" -ForegroundColor White
Write-Host "    curl $($script:healthUrl)" -ForegroundColor DarkCyan
Write-Host ""
Write-Host " 2. Open https://copilotstudio.microsoft.com" -ForegroundColor White
Write-Host " 3. Open (or create) your Custom Copilot" -ForegroundColor White
Write-Host " 4. Go to: Actions > Add Action > Custom Connector" -ForegroundColor White
Write-Host " 5. Choose 'Import from OpenAPI URL' and paste:" -ForegroundColor White
Write-Host "    $($script:openApiUrl)" -ForegroundColor Cyan
Write-Host " 6. Set authentication:" -ForegroundColor White
Write-Host "    Type           : API Key" -ForegroundColor White
Write-Host "    Parameter Name : X-API-Key" -ForegroundColor White
Write-Host "    Value          : $CopilotApiKey" -ForegroundColor Cyan
Write-Host " 7. Test with message: 'Help me reset my password'" -ForegroundColor White
Write-Host ""
