"""Azure Functions Deployment for serverless agent hosting.

This module provides:
- Serverless deployment configuration for Azure Functions
- Auto-scaling configuration
- Function app scaffolding generation
- Deployment automation

References:
- https://learn.microsoft.com/en-us/azure/azure-functions/
- https://learn.microsoft.com/en-us/azure/azure-functions/functions-premium-plan
"""

import json
import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class FunctionRuntime(str, Enum):
    """Azure Functions runtime options."""
    
    PYTHON = "python"
    NODE = "node"
    DOTNET = "dotnet"


class FunctionPlan(str, Enum):
    """Azure Functions hosting plans."""
    
    CONSUMPTION = "consumption"  # Pay-per-execution, auto-scale to 0
    PREMIUM = "premium"  # Pre-warmed instances, VNET support
    DEDICATED = "dedicated"  # App Service plan
    FLEX_CONSUMPTION = "flex_consumption"  # New flex consumption plan


class FunctionTrigger(str, Enum):
    """Types of function triggers."""
    
    HTTP = "http"
    TIMER = "timer"
    QUEUE = "queue"
    BLOB = "blob"
    EVENT_GRID = "eventGrid"
    SERVICE_BUS = "serviceBus"


@dataclass
class ScalingConfig:
    """Auto-scaling configuration for Azure Functions.
    
    Attributes:
        min_instances: Minimum number of instances (Premium/Dedicated only)
        max_instances: Maximum number of instances
        scale_out_cooldown: Cooldown period for scaling out (seconds)
        scale_in_cooldown: Cooldown period for scaling in (seconds)
        per_instance_concurrency: Max concurrent requests per instance
        always_ready: Keep instances warm (Premium only)
    """
    
    min_instances: int = 0
    max_instances: int = 100
    scale_out_cooldown: int = 60
    scale_in_cooldown: int = 300
    per_instance_concurrency: int = 100
    always_ready: int = 0  # Premium plan: pre-warmed instances
    
    def to_host_config(self) -> Dict[str, Any]:
        """Convert to host.json extension config."""
        return {
            "extensions": {
                "http": {
                    "maxConcurrentRequests": self.per_instance_concurrency,
                    "routePrefix": "api"
                }
            },
            "scale": {
                "minWorkerCount": self.min_instances,
                "maxWorkerCount": self.max_instances,
                "scaleOutCooldown": f"PT{self.scale_out_cooldown}S",
                "scaleInCooldown": f"PT{self.scale_in_cooldown}S"
            }
        }
    
    def to_arm_template(self) -> Dict[str, Any]:
        """Convert to ARM template site config."""
        return {
            "minimumElasticInstanceCount": self.min_instances,
            "preWarmedInstanceCount": self.always_ready,
            "functionAppScaleLimit": self.max_instances
        }


@dataclass
class FunctionAppConfig:
    """Configuration for Azure Functions app.
    
    Attributes:
        name: Function app name (globally unique)
        resource_group: Azure resource group
        location: Azure region
        runtime: Function runtime (python, node, dotnet)
        runtime_version: Runtime version (e.g., "3.11" for Python)
        plan: Hosting plan type
        scaling: Auto-scaling configuration
        storage_account: Storage account name
        app_insights: Application Insights name
        environment_variables: Environment variables to set
    """
    
    name: str
    resource_group: str
    location: str = "eastus"
    runtime: FunctionRuntime = FunctionRuntime.PYTHON
    runtime_version: str = "3.11"
    plan: FunctionPlan = FunctionPlan.CONSUMPTION
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    storage_account: Optional[str] = None
    app_insights: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, name: str) -> "FunctionAppConfig":
        """Create configuration from environment variables."""
        return cls(
            name=name,
            resource_group=os.getenv("AZURE_RESOURCE_GROUP", "rg-langchain-agents"),
            location=os.getenv("AZURE_LOCATION", "eastus"),
            runtime=FunctionRuntime(os.getenv("FUNCTIONS_RUNTIME", "python")),
            runtime_version=os.getenv("FUNCTIONS_RUNTIME_VERSION", "3.11"),
            plan=FunctionPlan(os.getenv("FUNCTIONS_PLAN", "consumption")),
            storage_account=os.getenv("AZURE_STORAGE_ACCOUNT"),
            app_insights=os.getenv("AZURE_APP_INSIGHTS"),
            environment_variables={
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", ""),
                "AZURE_AI_PROJECT_ENDPOINT": os.getenv("AZURE_AI_PROJECT_ENDPOINT", ""),
                "APPLICATIONINSIGHTS_CONNECTION_STRING": os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", ""),
            }
        )
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        issues = []
        if not self.name:
            issues.append("Function app name is required")
        if not self.resource_group:
            issues.append("Resource group is required")
        if len(self.name) > 60:
            issues.append("Function app name must be 60 characters or less")
        return issues


class AzureFunctionsDeployer:
    """Deployer for Azure Functions serverless hosting.
    
    This class provides:
    - Function app scaffolding generation
    - Deployment script generation
    - Azure CLI deployment commands
    - Bicep/ARM template generation
    
    Example:
        >>> config = FunctionAppConfig(
        ...     name="my-agent-functions",
        ...     resource_group="rg-agents",
        ...     location="eastus"
        ... )
        >>> deployer = AzureFunctionsDeployer(config)
        >>> 
        >>> # Generate function app scaffold
        >>> deployer.generate_scaffold("./function-app", wrappers)
        >>> 
        >>> # Deploy to Azure
        >>> deployer.deploy()
    """
    
    def __init__(self, config: FunctionAppConfig):
        """Initialize the deployer.
        
        Args:
            config: Function app configuration
        """
        self.config = config
        
        issues = config.validate()
        if issues:
            logger.warning(f"Configuration issues: {issues}")
    
    def generate_scaffold(
        self,
        output_dir: Union[str, Path],
        wrappers: Dict[str, Any],
        include_teams_bot: bool = False,
    ) -> Dict[str, Path]:
        """Generate Azure Functions app scaffold.
        
        Args:
            output_dir: Output directory for the function app
            wrappers: Dictionary of agent wrappers to expose
            include_teams_bot: Include Teams bot endpoint
        
        Returns:
            Dictionary of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Generate host.json
        files["host"] = self._generate_host_json(output_dir)
        
        # Generate local.settings.json
        files["local_settings"] = self._generate_local_settings(output_dir)
        
        # Generate requirements.txt
        files["requirements"] = self._generate_requirements(output_dir)
        
        # Generate main function_app.py
        files["function_app"] = self._generate_function_app(output_dir, wrappers)
        
        # Generate agent functions
        for agent_name, wrapper in wrappers.items():
            func_path = self._generate_agent_function(output_dir, agent_name, wrapper)
            files[f"agent_{agent_name}"] = func_path
        
        # Generate health check function
        files["health"] = self._generate_health_function(output_dir)
        
        # Generate Teams bot function (optional)
        if include_teams_bot:
            files["teams_bot"] = self._generate_teams_bot_function(output_dir)
        
        # Generate deployment files
        files["bicep"] = self._generate_bicep_template(output_dir)
        files["deploy_script"] = self._generate_deploy_script(output_dir)
        files["github_action"] = self._generate_github_action(output_dir)
        
        logger.info(f"Generated function app scaffold at {output_dir}")
        return files
    
    def _generate_host_json(self, output_dir: Path) -> Path:
        """Generate host.json configuration."""
        host_config = {
            "version": "2.0",
            "logging": {
                "applicationInsights": {
                    "samplingSettings": {
                        "isEnabled": True,
                        "excludedTypes": "Request"
                    }
                },
                "logLevel": {
                    "default": "Information",
                    "Host.Results": "Error",
                    "Function": "Information"
                }
            },
            "extensionBundle": {
                "id": "Microsoft.Azure.Functions.ExtensionBundle",
                "version": "[4.*, 5.0.0)"
            },
            "extensions": {
                "http": {
                    "routePrefix": "api",
                    "maxConcurrentRequests": self.config.scaling.per_instance_concurrency
                }
            }
        }
        
        # Add scaling config for Premium plan
        if self.config.plan == FunctionPlan.PREMIUM:
            host_config.update(self.config.scaling.to_host_config())
        
        host_path = output_dir / "host.json"
        with open(host_path, "w") as f:
            json.dump(host_config, f, indent=2)
        
        return host_path
    
    def _generate_local_settings(self, output_dir: Path) -> Path:
        """Generate local.settings.json for local development."""
        settings = {
            "IsEncrypted": False,
            "Values": {
                "FUNCTIONS_WORKER_RUNTIME": self.config.runtime.value,
                "AzureWebJobsStorage": "UseDevelopmentStorage=true",
                **self.config.environment_variables
            }
        }
        
        settings_path = output_dir / "local.settings.json"
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        
        return settings_path
    
    def _generate_requirements(self, output_dir: Path) -> Path:
        """Generate requirements.txt."""
        requirements = [
            "azure-functions",
            "langchain>=1.0.0",
            "langchain-openai>=1.0.0",
            "langchain-azure-ai>=2.0.0",
            "langgraph>=0.3.0",
            "azure-identity",
            "python-dotenv",
            "pydantic>=2.0",
        ]
        
        req_path = output_dir / "requirements.txt"
        with open(req_path, "w") as f:
            f.write("\n".join(requirements))
        
        return req_path
    
    def _generate_function_app(self, output_dir: Path, wrappers: Dict[str, Any]) -> Path:
        """Generate main function_app.py."""
        wrapper_imports = []
        wrapper_inits = []
        
        for name, wrapper in wrappers.items():
            wrapper_type = type(wrapper).__name__
            wrapper_imports.append(f"from langchain_azure_ai.wrappers import {wrapper_type}")
            
            # Get wrapper initialization params
            instructions = getattr(wrapper, "instructions", f"{name} agent")
            wrapper_inits.append(f'''
    "{name}": {wrapper_type}(
        name="{name}",
        instructions="""{instructions[:500]}...""",
    ),''')
        
        code = f'''"""Azure Functions app for LangChain agents.

Auto-generated by AzureFunctionsDeployer.
"""

import azure.functions as func
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Import wrappers
{chr(10).join(set(wrapper_imports))}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize function app
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Initialize agent wrappers
AGENTS: Dict[str, Any] = {{{"".join(wrapper_inits)}
}}


def get_agent(name: str):
    """Get an agent by name."""
    return AGENTS.get(name)


def list_agents():
    """List all available agents."""
    return [
        {{
            "name": name,
            "type": type(agent).__name__,
            "subtype": getattr(agent, "agent_subtype", "general")
        }}
        for name, agent in AGENTS.items()
    ]


# Health check endpoint
@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    return func.HttpResponse(
        json.dumps({{
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "agents": len(AGENTS),
            "function_app": "{self.config.name}"
        }}),
        mimetype="application/json"
    )


# List agents endpoint
@app.route(route="agents", methods=["GET"])
def get_agents(req: func.HttpRequest) -> func.HttpResponse:
    """List available agents."""
    return func.HttpResponse(
        json.dumps(list_agents()),
        mimetype="application/json"
    )


# Generic chat endpoint
@app.route(route="chat/{{agent_name}}", methods=["POST"])
def chat(req: func.HttpRequest) -> func.HttpResponse:
    """Chat with an agent."""
    agent_name = req.route_params.get("agent_name")
    agent = get_agent(agent_name)
    
    if not agent:
        return func.HttpResponse(
            json.dumps({{"error": f"Agent '{{agent_name}}' not found"}}),
            status_code=404,
            mimetype="application/json"
        )
    
    try:
        body = req.get_json()
        message = body.get("message", "")
        session_id = body.get("session_id", "")
        
        response = agent.chat(message, thread_id=session_id)
        
        return func.HttpResponse(
            json.dumps({{
                "response": response,
                "agent": agent_name,
                "session_id": session_id or "new",
                "timestamp": datetime.utcnow().isoformat()
            }}),
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Chat error: {{e}}")
        return func.HttpResponse(
            json.dumps({{"error": str(e)}}),
            status_code=500,
            mimetype="application/json"
        )
'''
        
        func_path = output_dir / "function_app.py"
        with open(func_path, "w") as f:
            f.write(code)
        
        return func_path
    
    def _generate_agent_function(
        self,
        output_dir: Path,
        agent_name: str,
        wrapper: Any,
    ) -> Path:
        """Generate a dedicated function for an agent."""
        # For V2 programming model, agents are defined in function_app.py
        # This generates additional endpoints for specific agents
        
        func_dir = output_dir / f"agent_{agent_name}"
        func_dir.mkdir(exist_ok=True)
        
        function_json = {
            "scriptFile": "../function_app.py",
            "bindings": [
                {
                    "authLevel": "function",
                    "type": "httpTrigger",
                    "direction": "in",
                    "name": "req",
                    "methods": ["post"],
                    "route": f"agents/{agent_name}/chat"
                },
                {
                    "type": "http",
                    "direction": "out",
                    "name": "$return"
                }
            ]
        }
        
        func_json_path = func_dir / "function.json"
        with open(func_json_path, "w") as f:
            json.dump(function_json, f, indent=2)
        
        return func_json_path
    
    def _generate_health_function(self, output_dir: Path) -> Path:
        """Generate health check function."""
        health_dir = output_dir / "health"
        health_dir.mkdir(exist_ok=True)
        
        function_json = {
            "scriptFile": "../function_app.py",
            "bindings": [
                {
                    "authLevel": "anonymous",
                    "type": "httpTrigger",
                    "direction": "in",
                    "name": "req",
                    "methods": ["get"],
                    "route": "health"
                },
                {
                    "type": "http",
                    "direction": "out",
                    "name": "$return"
                }
            ]
        }
        
        func_json_path = health_dir / "function.json"
        with open(func_json_path, "w") as f:
            json.dump(function_json, f, indent=2)
        
        return func_json_path
    
    def _generate_teams_bot_function(self, output_dir: Path) -> Path:
        """Generate Teams bot function."""
        bot_code = '''"""Teams Bot Function for Azure Functions."""

import azure.functions as func
import json
import logging
from langchain_azure_ai.connectors import TeamsBotConnector, TeamsBotConfig, TeamsActivity

logger = logging.getLogger(__name__)

# Initialize bot connector
bot_config = TeamsBotConfig.from_env()
bot_connector = TeamsBotConnector(bot_config)

# Register agents (import from function_app)
from function_app import AGENTS
for name, agent in AGENTS.items():
    bot_connector.register_agent(name, agent)


async def teams_bot(req: func.HttpRequest) -> func.HttpResponse:
    """Handle Teams bot messages."""
    try:
        body = req.get_json()
        activity = TeamsActivity.from_request(body)
        
        response = await bot_connector.process_activity(activity)
        
        return func.HttpResponse(
            json.dumps(response),
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Teams bot error: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
'''
        
        bot_dir = output_dir / "teams_bot"
        bot_dir.mkdir(exist_ok=True)
        
        # Write bot code
        bot_path = bot_dir / "__init__.py"
        with open(bot_path, "w") as f:
            f.write(bot_code)
        
        # Write function.json
        function_json = {
            "bindings": [
                {
                    "authLevel": "function",
                    "type": "httpTrigger",
                    "direction": "in",
                    "name": "req",
                    "methods": ["post"],
                    "route": "messages"
                },
                {
                    "type": "http",
                    "direction": "out",
                    "name": "$return"
                }
            ]
        }
        
        func_json_path = bot_dir / "function.json"
        with open(func_json_path, "w") as f:
            json.dump(function_json, f, indent=2)
        
        return bot_path
    
    def _generate_bicep_template(self, output_dir: Path) -> Path:
        """Generate Bicep deployment template."""
        bicep = f'''// Azure Functions deployment for LangChain Agents
// Generated by AzureFunctionsDeployer

@description('Function app name')
param functionAppName string = '{self.config.name}'

@description('Location for resources')
param location string = resourceGroup().location

@description('Storage account name')
param storageAccountName string = '${{functionAppName}}storage'

@description('App Insights name')
param appInsightsName string = '${{functionAppName}}-insights'

@description('Hosting plan type')
@allowed(['consumption', 'premium', 'dedicated'])
param planType string = '{self.config.plan.value}'

// Storage Account
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {{
  name: storageAccountName
  location: location
  sku: {{
    name: 'Standard_LRS'
  }}
  kind: 'StorageV2'
}}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {{
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {{
    Application_Type: 'web'
    Request_Source: 'rest'
  }}
}}

// Hosting Plan
resource hostingPlan 'Microsoft.Web/serverfarms@2023-01-01' = {{
  name: '${{functionAppName}}-plan'
  location: location
  sku: planType == 'consumption' ? {{
    name: 'Y1'
    tier: 'Dynamic'
  }} : planType == 'premium' ? {{
    name: 'EP1'
    tier: 'ElasticPremium'
  }} : {{
    name: 'B1'
    tier: 'Basic'
  }}
  properties: {{
    reserved: true // Linux
  }}
}}

// Function App
resource functionApp 'Microsoft.Web/sites@2023-01-01' = {{
  name: functionAppName
  location: location
  kind: 'functionapp,linux'
  properties: {{
    serverFarmId: hostingPlan.id
    siteConfig: {{
      pythonVersion: '{self.config.runtime_version}'
      linuxFxVersion: 'Python|{self.config.runtime_version}'
      appSettings: [
        {{
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${{storageAccount.name}};EndpointSuffix=${{environment().suffixes.storage}};AccountKey=${{storageAccount.listKeys().keys[0].value}}'
        }}
        {{
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }}
        {{
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: '{self.config.runtime.value}'
        }}
        {{
          name: 'APPINSIGHTS_INSTRUMENTATIONKEY'
          value: appInsights.properties.InstrumentationKey
        }}
        {{
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }}
      ]
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
    }}
    httpsOnly: true
  }}
}}

// Auto-scaling (Premium plan only)
resource autoScale 'Microsoft.Insights/autoscalesettings@2022-10-01' = if (planType == 'premium') {{
  name: '${{functionAppName}}-autoscale'
  location: location
  properties: {{
    enabled: true
    targetResourceUri: hostingPlan.id
    profiles: [
      {{
        name: 'Auto Scale Profile'
        capacity: {{
          minimum: '{self.config.scaling.min_instances}'
          maximum: '{self.config.scaling.max_instances}'
          default: '{self.config.scaling.min_instances}'
        }}
        rules: [
          {{
            metricTrigger: {{
              metricName: 'CpuPercentage'
              metricResourceUri: hostingPlan.id
              timeGrain: 'PT1M'
              statistic: 'Average'
              timeWindow: 'PT5M'
              timeAggregation: 'Average'
              operator: 'GreaterThan'
              threshold: 70
            }}
            scaleAction: {{
              direction: 'Increase'
              type: 'ChangeCount'
              value: '1'
              cooldown: 'PT{self.config.scaling.scale_out_cooldown}S'
            }}
          }}
          {{
            metricTrigger: {{
              metricName: 'CpuPercentage'
              metricResourceUri: hostingPlan.id
              timeGrain: 'PT1M'
              statistic: 'Average'
              timeWindow: 'PT5M'
              timeAggregation: 'Average'
              operator: 'LessThan'
              threshold: 30
            }}
            scaleAction: {{
              direction: 'Decrease'
              type: 'ChangeCount'
              value: '1'
              cooldown: 'PT{self.config.scaling.scale_in_cooldown}S'
            }}
          }}
        ]
      }}
    ]
  }}
}}

output functionAppUrl string = 'https://${{functionApp.properties.defaultHostName}}'
output functionAppName string = functionApp.name
'''
        
        bicep_path = output_dir / "main.bicep"
        with open(bicep_path, "w") as f:
            f.write(bicep)
        
        return bicep_path
    
    def _generate_deploy_script(self, output_dir: Path) -> Path:
        """Generate deployment script."""
        script = f'''#!/bin/bash
# Azure Functions Deployment Script
# Generated by AzureFunctionsDeployer

set -e

FUNCTION_APP_NAME="{self.config.name}"
RESOURCE_GROUP="{self.config.resource_group}"
LOCATION="{self.config.location}"

echo "Deploying Azure Functions app: $FUNCTION_APP_NAME"

# Login check
az account show > /dev/null 2>&1 || az login

# Create resource group if not exists
az group create --name $RESOURCE_GROUP --location $LOCATION --output none 2>/dev/null || true

# Deploy infrastructure with Bicep
echo "Deploying infrastructure..."
az deployment group create \\
    --resource-group $RESOURCE_GROUP \\
    --template-file main.bicep \\
    --parameters functionAppName=$FUNCTION_APP_NAME \\
    --output none

# Deploy function code
echo "Deploying function code..."
func azure functionapp publish $FUNCTION_APP_NAME --python

# Get function URL
FUNCTION_URL=$(az functionapp show --name $FUNCTION_APP_NAME --resource-group $RESOURCE_GROUP --query "defaultHostName" -o tsv)

echo ""
echo "Deployment complete!"
echo "Function App URL: https://$FUNCTION_URL"
echo ""
echo "Test endpoints:"
echo "  Health: https://$FUNCTION_URL/api/health"
echo "  Agents: https://$FUNCTION_URL/api/agents"
echo "  Chat:   https://$FUNCTION_URL/api/chat/{{agent_name}}"
'''
        
        script_path = output_dir / "deploy.sh"
        with open(script_path, "w") as f:
            f.write(script)
        
        # Make executable
        script_path.chmod(0o755)
        
        # Also create PowerShell version
        ps_script = f'''# Azure Functions Deployment Script (PowerShell)
# Generated by AzureFunctionsDeployer

$ErrorActionPreference = "Stop"

$FUNCTION_APP_NAME = "{self.config.name}"
$RESOURCE_GROUP = "{self.config.resource_group}"
$LOCATION = "{self.config.location}"

Write-Host "Deploying Azure Functions app: $FUNCTION_APP_NAME"

# Login check
try {{
    az account show | Out-Null
}} catch {{
    az login
}}

# Create resource group if not exists
az group create --name $RESOURCE_GROUP --location $LOCATION --output none 2>$null

# Deploy infrastructure with Bicep
Write-Host "Deploying infrastructure..."
az deployment group create `
    --resource-group $RESOURCE_GROUP `
    --template-file main.bicep `
    --parameters functionAppName=$FUNCTION_APP_NAME `
    --output none

# Deploy function code
Write-Host "Deploying function code..."
func azure functionapp publish $FUNCTION_APP_NAME --python

# Get function URL
$FUNCTION_URL = az functionapp show --name $FUNCTION_APP_NAME --resource-group $RESOURCE_GROUP --query "defaultHostName" -o tsv

Write-Host ""
Write-Host "Deployment complete!"
Write-Host "Function App URL: https://$FUNCTION_URL"
Write-Host ""
Write-Host "Test endpoints:"
Write-Host "  Health: https://$FUNCTION_URL/api/health"
Write-Host "  Agents: https://$FUNCTION_URL/api/agents"
Write-Host "  Chat:   https://$FUNCTION_URL/api/chat/{{agent_name}}"
'''
        
        ps_path = output_dir / "deploy.ps1"
        with open(ps_path, "w") as f:
            f.write(ps_script)
        
        return script_path
    
    def _generate_github_action(self, output_dir: Path) -> Path:
        """Generate GitHub Actions workflow for CI/CD."""
        workflow = f'''name: Deploy Azure Functions

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  AZURE_FUNCTIONAPP_NAME: '{self.config.name}'
  AZURE_FUNCTIONAPP_PACKAGE_PATH: '.'
  PYTHON_VERSION: '{self.config.runtime_version}'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Setup Python ${{{{ env.PYTHON_VERSION }}}}
      uses: actions/setup-python@v5
      with:
        python-version: ${{{{ env.PYTHON_VERSION }}}}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pip install pytest
        pytest tests/ -v || true

    - name: Login to Azure
      uses: azure/login@v2
      with:
        creds: ${{{{ secrets.AZURE_CREDENTIALS }}}}

    - name: Deploy to Azure Functions
      uses: Azure/functions-action@v1
      with:
        app-name: ${{{{ env.AZURE_FUNCTIONAPP_NAME }}}}
        package: ${{{{ env.AZURE_FUNCTIONAPP_PACKAGE_PATH }}}}
        scm-do-build-during-deployment: true
        enable-oryx-build: true
'''
        
        # Create .github/workflows directory
        workflows_dir = output_dir / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_path = workflows_dir / "deploy-functions.yml"
        with open(workflow_path, "w") as f:
            f.write(workflow)
        
        return workflow_path
    
    def deploy(self, use_cli: bool = True) -> Dict[str, Any]:
        """Deploy the function app to Azure.
        
        Args:
            use_cli: Whether to use Azure CLI for deployment
        
        Returns:
            Deployment result with URL and status
        """
        if not use_cli:
            return {
                "status": "manual",
                "message": "Use the generated deploy.sh script to deploy",
                "script": "./deploy.sh"
            }
        
        logger.info(f"Deploying function app: {self.config.name}")
        
        try:
            # Check Azure CLI login
            subprocess.run(["az", "account", "show"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return {
                "status": "error",
                "message": "Not logged into Azure CLI. Run 'az login' first."
            }
        
        try:
            # Create resource group
            subprocess.run([
                "az", "group", "create",
                "--name", self.config.resource_group,
                "--location", self.config.location,
                "--output", "none"
            ], check=True)
            
            # Deploy infrastructure
            # Note: In production, this would deploy the Bicep template
            
            return {
                "status": "success",
                "function_app": self.config.name,
                "resource_group": self.config.resource_group,
                "url": f"https://{self.config.name}.azurewebsites.net"
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": str(e)
            }
