# Server Restart Procedure for Software Development DeepAgent

## Issue Summary

The Software Development DeepAgent is not loading in the running server because the server is running an older version of the code. The agent registration code exists and works correctly (verified by testing).

## Root Cause

- Server was started before the latest code was deployed
- Software Development DeepAgent requires 54 specialized tools that need to be loaded at initialization
- Current server shows 13 agents (should be 14)
- Missing: `software_development` DeepAgent

## Verification

✅ **Code is correct**: `SoftwareDevelopmentWrapper` instantiates successfully with proper credentials
✅ **Tools load properly**: All 54 tools (9 subagents × 6 tools each) load without errors
✅ **Registration logic exists**: Server code (lines 378-385 in `server/__init__.py`) includes registration

## Solution: Restart Server

### Step 1: Find and Stop Current Server

```bash
# Find the process
netstat -ano | findstr ":8000"
# Note the PID (e.g., 63232)

# Stop the server (Windows)
taskkill /F /PID <PID_NUMBER>

# OR on Linux/Mac
kill -9 <PID_NUMBER>
```

### Step 2: Activate Virtual Environment

```bash
cd "c:\Users\a833555\OneDrive - ATOS\Gitwork\langchain-azure"

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 3: Ensure Dependencies are Installed

```bash
pip install python-dotenv  # If not already installed
```

### Step 4: Start Server with Latest Code

```bash
cd libs\azure-ai
python start_server.py
```

### Step 5: Verify Software Development DeepAgent Loaded

```bash
# Check health (should show 14 agents)
curl http://localhost:8000/health

# List all agents
curl http://localhost:8000/agents | python -m json.tool | findstr "software_development"

# Test the endpoint
curl -X POST http://localhost:8000/api/deepagent/software_development/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Hello, can you help me with software development?\"}"
```

## Expected Results

After restart, you should see:

```json
{
  "status": "healthy",
  "timestamp": "2026-01-29T...",
  "agents_loaded": 14,  // <-- Should be 14, not 13
  "azure_foundry_enabled": true
}
```

And the `/agents` endpoint should include:

```json
{
  "name": "software_development",
  "type": "DeepAgent",
  "subtype": "software_development",
  "description": "DeepAgent (software_development): software-development",
  "endpoints": [
    "/api/deepagent/software_development/execute",
    "/api/deepagent/software_development/chat"
  ]
}
```

## Troubleshooting

### If Server Still Shows 13 Agents

Check the server console output for warnings:
```
Failed to load Software Development DeepAgent: <error message>
```

Common issues:
1. **Missing credentials**: Ensure `.env` file has `AZURE_OPENAI_API_KEY` set
2. **Wrong Python environment**: Ensure virtual environment is activated
3. **Import errors**: Check that all dependencies are installed

### If Server Won't Start

1. Check port 8000 is not in use:
   ```bash
   netstat -ano | findstr ":8000"
   ```

2. Check `.env` file exists and has proper values:
   ```bash
   cat .env | grep AZURE_OPENAI
   ```

3. Check Python version (requires 3.10+):
   ```bash
   python --version
   ```

## API Endpoints for Software Development DeepAgent

Once loaded, the agent is available at:

- **Chat**: `POST /api/deepagent/software_development/chat`
- **Execute Workflow**: `POST /api/deepagent/software_development/execute`
- **Streaming Chat**: `POST /api/deepagent/software_development/chat/stream`
- **Get Subagents**: `GET /api/deepagent/software_development/subagents`

## UI Access

Once the server is restarted and the agent loads:

1. Navigate to: http://localhost:8000/chat
2. Select "software_development" from the agent dropdown
3. You should see a welcome message from the Software Development Deep Agent

## Subagents Available

The Software Development DeepAgent includes 9 specialized subagents:

1. **Requirements Intelligence** - Requirements analysis (6 tools)
2. **Architecture Design** - System design and APIs (6 tools)
3. **Code Generator** - Code generation and refactoring (6 tools)
4. **Code Reviewer** - Code review and quality (6 tools)
5. **Testing Automation** - Test generation and coverage (6 tools)
6. **Debugging & Optimization** - Error analysis and performance (6 tools)
7. **Security & Compliance** - Security scanning and OWASP (6 tools)
8. **DevOps Integration** - CI/CD and deployment (6 tools)
9. **Documentation** - API docs and user guides (6 tools)

**Total**: 54 specialized tools across the full SDLC
