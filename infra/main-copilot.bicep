// ============================================================================
// Copilot Studio Integration — Container App Deployment
// ============================================================================
// Deploys ONLY the langchain-agents Container App into existing shared
// resources. All other resources (ACR, Key Vault, App Insights, Container Apps
// Environment, Managed Identity) already exist and are referenced here.
//
// Invoked by: infra/deploy-copilot.ps1
// Do NOT run this directly — the deploy script handles RBAC role assignments
// and secret provisioning before calling this template.
// ============================================================================

targetScope = 'resourceGroup'

// ============================================================================
// Parameters
// ============================================================================

@description('Azure region — must match existing resources')
param location string = 'germanywestcentral'

@description('Environment tag')
param environment string = 'prod'

@description('Name for the new Container App')
param containerAppName string = 'langchain-agents-prod'

@description('Docker image tag to deploy')
param imageTag string = 'latest'

// ---- Existing Resource Names (all pre-provisioned) ----

@description('Existing Container Apps Environment name')
param containerEnvName string = 'servicenowmcp-env-fybcq3hpz3evo'

@description('Existing Azure Container Registry name')
param acrName string = 'servicenowmcpacr'

@description('Existing Managed Identity name')
param managedIdentityName string = 'id-gwc-mcp'

// ---- Runtime Configuration ----

@description('Azure OpenAI endpoint URL')
param azureOpenAIEndpoint string

@description('Azure OpenAI deployment/model name')
param azureOpenAIDeployment string = 'gpt-4o-mini'

@description('Azure OpenAI API version')
param azureOpenAIApiVersion string = '2024-12-01-preview'

// ---- Secure Parameters (not logged in deployment history) ----

@secure()
@description('Application Insights connection string')
param appInsightsConnectionString string

@description('Existing Key Vault name (used for Key Vault-backed Container App secrets)')
param keyVaultName string = 'kv-mcp-nonprd-eiozqzsw'

// ---- Scaling ----

@minValue(0)
@maxValue(10)
param minReplicas int = 1

@minValue(1)
@maxValue(10)
param maxReplicas int = 5

// ============================================================================
// References to Existing Resources (no new resource creation)
// ============================================================================

resource containerEnv 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: containerEnvName
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: managedIdentityName
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

// ============================================================================
// Container App
// ============================================================================

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
        corsPolicy: {
          allowedOrigins: [
            'https://copilotstudio.microsoft.com'
            'https://web.powerva.microsoft.com'
            'https://*.powerplatform.com'
          ]
          allowedMethods: ['GET', 'POST', 'OPTIONS']
          allowedHeaders: ['Authorization', 'Content-Type', 'X-API-Key']
          allowCredentials: true
          maxAge: 3600
        }
      }
      registries: [
        {
          // Managed-identity pull — no ACR admin credentials required.
          server: acr.properties.loginServer
          identity: managedIdentity.id
        }
      ]
      // Secrets backed by Key Vault: values are never embedded in the deployment.
      // App Insights connection string is a non-secret observable value stored inline.
      secrets: [
        {
          name: 'azure-openai-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/langchain-azure-openai-key'
          identity: managedIdentity.id
        }
        {
          name: 'copilot-api-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/langchain-copilot-api-key'
          identity: managedIdentity.id
        }
        {
          name: 'appinsights-connection'
          value: appInsightsConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'langchain-agents'
          image: '${acr.properties.loginServer}/langchain-agents:${imageTag}'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'AZURE_OPENAI_ENDPOINT',                 value: azureOpenAIEndpoint }
            { name: 'AZURE_OPENAI_API_KEY',                  secretRef: 'azure-openai-key' }
            { name: 'AZURE_OPENAI_DEPLOYMENT_NAME',          value: azureOpenAIDeployment }
            { name: 'OPENAI_API_VERSION',                    value: azureOpenAIApiVersion }
            { name: 'COPILOT_API_KEY_ENABLED',               value: 'true' }
            { name: 'COPILOT_OAUTH_ENABLED',                 value: 'false' }
            { name: 'COPILOT_API_KEY',                       secretRef: 'copilot-api-key' }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appinsights-connection' }
            { name: 'ENABLE_AZURE_MONITOR',                  value: 'true' }
            { name: 'USE_AZURE_FOUNDRY',                     value: 'false' }
            { name: 'ENABLE_IT_AGENTS',                      value: 'true' }
            { name: 'ENABLE_ENTERPRISE_AGENTS',              value: 'true' }
            { name: 'ENABLE_DEEP_AGENTS',                    value: 'true' }
            { name: 'PORT',                                  value: '8000' }
            { name: 'LOG_LEVEL',                             value: 'INFO' }
            { name: 'CORS_ORIGINS',                          value: 'https://copilotstudio.microsoft.com,https://web.powerva.microsoft.com' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 30
              periodSeconds: 30
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 15
              periodSeconds: 15
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scale'
            http: {
              metadata: { concurrentRequests: '50' }
            }
          }
        ]
      }
    }
  }
  tags: {
    environment: environment
    application: 'langchain-agents'
    'copilot-integration': 'enabled'
    'managed-by': 'deploy-copilot.ps1'
  }
}

// ============================================================================
// Outputs
// ============================================================================

@description('Base URL of the deployed Container App')
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'

@description('OpenAPI spec URL — import this in Copilot Studio custom connector')
output copilotOpenApiUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}/api/copilot/openapi.json'

@description('Plugin manifest URL for Microsoft 365 Copilot discovery')
output copilotPluginManifestUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}/.well-known/ai-plugin.json'

@description('Health check URL')
output healthCheckUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}/health'
