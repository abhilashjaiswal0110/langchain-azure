// ============================================================================
// Bicep Parameters File
// ============================================================================
// Copy this file to main.bicepparam.local and fill in your values.
// DO NOT commit files with actual secrets to version control.
// ============================================================================

using 'main.bicep'

// Environment Configuration
param environment = 'prod'
param location = 'eastus'
param baseName = 'langchain-agents'

// Container Configuration
// Leave empty to use default image, or specify your ACR image
param containerImage = ''

// Azure OpenAI Configuration
// Get these from Azure Portal > Azure OpenAI > Keys and Endpoint
param azureOpenAIEndpoint = 'https://your-resource.openai.azure.com/'
param azureOpenAIKey = '' // Provide at deployment time with --parameters
param azureOpenAIDeployment = 'gpt-4o-mini'

// Azure AI Foundry Configuration (optional)
param azureAIProjectEndpoint = ''

// Copilot Studio Configuration
// Generate a secure key: openssl rand -hex 32
param copilotApiKey = '' // Provide at deployment time with --parameters

// Networking
param enablePublicAccess = true

// Scaling Configuration
param minReplicas = 1
param maxReplicas = 10
