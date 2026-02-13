// ============================================================================
// Azure AI Foundry LangChain Agents - Infrastructure Deployment
// ============================================================================
// This Bicep template deploys the infrastructure required for running
// LangChain agents on Azure with Copilot Studio integration.
//
// Resources Deployed:
// - Azure Container Apps Environment
// - Azure Container App (with auto-scaling)
// - Azure Container Registry
// - Application Insights (observability)
// - Log Analytics Workspace
// - Key Vault (secrets management)
// - Managed Identity (secure access)
//
// Usage:
//   az deployment group create \
//     --resource-group rg-langchain-agents \
//     --template-file main.bicep \
//     --parameters environment=prod
// ============================================================================

targetScope = 'resourceGroup'

// ============================================================================
// Parameters
// ============================================================================

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

@description('Location for all resources')
param location string = resourceGroup().location

@description('Base name for resources')
param baseName string = 'langchain-agents'

@description('Container image to deploy (leave empty to use ACR)')
param containerImage string = ''

@description('Azure OpenAI endpoint')
@secure()
param azureOpenAIEndpoint string = ''

@description('Azure OpenAI API key')
@secure()
param azureOpenAIKey string = ''

@description('Azure OpenAI deployment name')
param azureOpenAIDeployment string = 'gpt-4o-mini'

@description('Azure AI Project endpoint (for Foundry)')
@secure()
param azureAIProjectEndpoint string = ''

@description('Copilot Studio API key for authentication')
@secure()
param copilotApiKey string = ''

@description('Enable public access (set false for private networking)')
param enablePublicAccess bool = true

@description('Minimum number of replicas')
@minValue(0)
@maxValue(10)
param minReplicas int = 1

@description('Maximum number of replicas')
@minValue(1)
@maxValue(30)
param maxReplicas int = 10

// ============================================================================
// Variables
// ============================================================================

var uniqueSuffix = uniqueString(resourceGroup().id)
var containerAppName = '${baseName}-${environment}'
var containerEnvName = 'cae-${baseName}-${environment}-${uniqueSuffix}'
var acrName = replace('acr${baseName}${uniqueSuffix}', '-', '')
var keyVaultName = 'kv-${take(baseName, 10)}-${uniqueSuffix}'
var appInsightsName = 'appi-${baseName}-${environment}'
var logAnalyticsName = 'log-${baseName}-${environment}'
var managedIdentityName = 'id-${baseName}-${environment}'

// Default container image if not provided
var defaultImage = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
var deployImage = empty(containerImage) ? defaultImage : containerImage

// ============================================================================
// Log Analytics Workspace (Required for Container Apps)
// ============================================================================

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
  tags: {
    environment: environment
    application: baseName
  }
}

// ============================================================================
// Application Insights (Observability)
// ============================================================================

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
  tags: {
    environment: environment
    application: baseName
  }
}

// ============================================================================
// Key Vault (Secrets Management)
// ============================================================================

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: environment == 'prod'
    networkAcls: {
      defaultAction: enablePublicAccess ? 'Allow' : 'Deny'
      bypass: 'AzureServices'
    }
  }
  tags: {
    environment: environment
    application: baseName
  }
}

// Store secrets in Key Vault
resource secretOpenAIKey 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = if (!empty(azureOpenAIKey)) {
  parent: keyVault
  name: 'azure-openai-key'
  properties: {
    value: azureOpenAIKey
  }
}

resource secretCopilotApiKey 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = if (!empty(copilotApiKey)) {
  parent: keyVault
  name: 'copilot-api-key'
  properties: {
    value: copilotApiKey
  }
}

// ============================================================================
// Managed Identity (Secure Access)
// ============================================================================

resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: managedIdentityName
  location: location
  tags: {
    environment: environment
    application: baseName
  }
}

// Grant Key Vault access to Managed Identity
resource keyVaultRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, managedIdentity.id, 'Key Vault Secrets User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6') // Key Vault Secrets User
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ============================================================================
// Azure Container Registry
// ============================================================================

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: environment == 'prod' ? 'Standard' : 'Basic'
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: enablePublicAccess ? 'Enabled' : 'Disabled'
  }
  tags: {
    environment: environment
    application: baseName
  }
}

// Grant ACR pull access to Managed Identity
resource acrRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name: guid(acr.id, managedIdentity.id, 'AcrPull')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d') // AcrPull
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ============================================================================
// Container Apps Environment
// ============================================================================

resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    daprAIInstrumentationKey: appInsights.properties.InstrumentationKey
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
  tags: {
    environment: environment
    application: baseName
  }
}

// ============================================================================
// Container App (LangChain Agents API)
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
    managedEnvironmentId: containerAppEnvironment.id
    configuration: {
      ingress: {
        external: enablePublicAccess
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
        corsPolicy: {
          allowedOrigins: [
            'https://copilotstudio.microsoft.com'
            'https://*.microsoft.com'
            'https://*.azure.com'
          ]
          allowedMethods: ['GET', 'POST', 'OPTIONS']
          allowedHeaders: ['*']
          allowCredentials: true
          maxAge: 3600
        }
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: managedIdentity.id
        }
      ]
      secrets: [
        {
          name: 'azure-openai-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/azure-openai-key'
          identity: managedIdentity.id
        }
        {
          name: 'copilot-api-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/copilot-api-key'
          identity: managedIdentity.id
        }
        {
          name: 'appinsights-connection'
          value: appInsights.properties.ConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'langchain-agents'
          image: deployImage
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            // Azure OpenAI Configuration
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAIEndpoint
            }
            {
              name: 'AZURE_OPENAI_API_KEY'
              secretRef: 'azure-openai-key'
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT_NAME'
              value: azureOpenAIDeployment
            }
            {
              name: 'OPENAI_API_VERSION'
              value: '2024-12-01-preview'
            }
            // Azure AI Foundry
            {
              name: 'AZURE_AI_PROJECT_ENDPOINT'
              value: azureAIProjectEndpoint
            }
            {
              name: 'USE_AZURE_FOUNDRY'
              value: 'true'
            }
            // Copilot Studio Configuration
            {
              name: 'COPILOT_API_KEY_ENABLED'
              value: 'true'
            }
            {
              name: 'COPILOT_API_KEY'
              secretRef: 'copilot-api-key'
            }
            // Observability
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              secretRef: 'appinsights-connection'
            }
            {
              name: 'ENABLE_AZURE_MONITOR'
              value: 'true'
            }
            // Server Configuration
            {
              name: 'PORT'
              value: '8000'
            }
            {
              name: 'CORS_ORIGINS'
              value: 'https://copilotstudio.microsoft.com,https://*.microsoft.com'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 30
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 10
              periodSeconds: 10
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
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
  tags: {
    environment: environment
    application: baseName
    'copilot-integration': 'enabled'
  }
}

// ============================================================================
// Outputs
// ============================================================================

@description('The URL of the deployed Container App')
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'

@description('Copilot Studio Plugin Manifest URL')
output copilotPluginManifestUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}/.well-known/ai-plugin.json'

@description('Copilot Studio OpenAPI Spec URL')
output copilotOpenApiUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}/api/copilot/openapi.json'

@description('Application Insights Connection String')
output appInsightsConnectionString string = appInsights.properties.ConnectionString

@description('Key Vault URI for secrets')
output keyVaultUri string = keyVault.properties.vaultUri

@description('Azure Container Registry login server')
output acrLoginServer string = acr.properties.loginServer

@description('Managed Identity Client ID')
output managedIdentityClientId string = managedIdentity.properties.clientId

@description('Container App Resource ID')
output containerAppResourceId string = containerApp.id
