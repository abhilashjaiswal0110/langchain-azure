# 🧪 Enterprise Hub - Complete UI Testing Guide

> **Quick Start Testing Guide for Enterprise Multi-Agent Hubs**

---

## 📋 Test Prompts by Use Case

### IT Support Hub

#### Test 1: Password Reset (Should route to: helpdesk)
**Prompt:**
```
"I forgot my password and can't log into my laptop. Can you help me reset it?"
```

**Expected Response:**
- Agent identifies password reset request
- Provides password reset instructions
- May offer to create a ticket if needed

#### Test 2: Documentation Search (Should route to: knowledge_base)
**Prompt:**
```
"How do I configure VPN on my work laptop?"
```

**Expected Response:**
- Searches knowledge base for VPN documentation
- Returns relevant articles/instructions
- Provides step-by-step guidance

#### Test 3: ServiceNow Ticket (Should route to: servicenow)
**Prompt:**
```
"I need to create an incident ticket for my broken monitor. It won't turn on."
```

**Expected Response:**
- Creates incident in ServiceNow
- Returns ticket number
- Provides estimated resolution time

#### Test 4: Complex Multi-Agent Flow
**Prompt:**
```
"My laptop is running very slow. First, can you check if there's any documentation on this? If not, please create a ticket for IT support."
```

**Expected Response:**
- First routes to knowledge_base for documentation
- If no solution, escalates to servicenow for ticket creation
- Provides comprehensive response with both steps### Customer Service Hub

#### Test 1: Order Status (Should route to: support)
**Prompt:**
```
"I placed an order last week (order #12345) and haven't received it yet. Can you check the status?"
```

**Expected Response:**
- Checks order status
- Provides shipping information
- Offers tracking details

#### Test 2: FAQ Query (Should route to: faq)
**Prompt:**
```
"What's your return policy for items purchased within the last 30 days?"
```

**Expected Response:**
- Searches FAQ knowledge base
- Returns return policy details
- Provides instructions for initiating return

#### Test 3: Escalation (Should route to: ticketing)
**Prompt:**
```
"I received a damaged product and need to speak with a supervisor. This is urgent!"
```

**Expected Response:**
- Recognizes escalation need
- Creates priority ticket
- Provides ticket number and expected response time

---

## 🖥️ Command Line Testing

### Interactive Testing Script

Create `test_interactive.py`:

```python
"""Interactive testing for Enterprise Hub."""

from orchestrator import create_hub
import uuid

def main():
    print("🚀 Starting Enterprise Hub Interactive Test\n")

    # Create hub
    print("Creating hub...")
    hub = create_hub()
    print("✅ Hub created!\n")

    # Create session
    session_id = str(uuid.uuid4())
    print(f"📱 Session ID: {session_id}\n")

    # Test scenarios
    scenarios = [
        "I need help resetting my password",
        "How do I configure VPN?",
        "Create an incident ticket for my broken monitor"
    ]

    for i, prompt in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {prompt}")
        print('='*60)

        try:
            response = hub.invoke(prompt, session_id=session_id)

            print(f"\n✅ Response received!")
            print(f"🎯 Routed to: {response.get('current_agent', 'Unknown')}")

            # Print last message
            if response.get("messages"):
                last_msg = response["messages"][-1]
                print(f"\n💬 Response:\n{last_msg.content}\n")

        except Exception as e:
            print(f"\n❌ Error: {e}\n")

    print(f"\n{'='*60}")
    print("✅ All tests completed!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
python test_interactive.py
```

---

## 🌐 Web UI Testing (Gradio)

### Create Simple Gradio UI

Create `web_ui.py`:

```python
"""Web UI for Enterprise Hub using Gradio."""

import gradio as gr
from orchestrator import create_hub
import uuid

# Create hub globally
hub = create_hub()
sessions = {}

def chat(message, session_id, history):
    """Handle chat interaction."""
    if not session_id:
        session_id = str(uuid.uuid4())
        sessions[session_id] = []

    try:
        # Invoke hub
        response = hub.invoke(message, session_id=session_id)

        # Extract response
        if response.get("messages"):
            reply = response["messages"][-1].content
            agent = response.get("current_agent", "unknown")
            reply_with_agent = f"🤖 **Agent: {agent}**\n\n{reply}"
        else:
            reply_with_agent = "No response received"

        # Update history
        history.append((message, reply_with_agent))

        return history, session_id, ""

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append((message, error_msg))
        return history, session_id, ""

def create_ui():
    """Create Gradio interface."""

    with gr.Blocks(title="Enterprise Hub") as demo:
        gr.Markdown("# 🏢 Enterprise Multi-Agent Hub")
        gr.Markdown("Test the enterprise hub with realistic scenarios")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500
                )

                with gr.Row():
                    message = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        scale=4
                    )
                    submit = gr.Button("Send", scale=1)

                session_id_box = gr.Textbox(
                    label="Session ID (auto-generated)",
                    interactive=False
                )

                clear = gr.Button("Clear Conversation")

            with gr.Column(scale=1):
                gr.Markdown("### 📝 Test Scenarios")

                gr.Markdown("#### IT Support")
                test_btn1 = gr.Button("🔑 Password Reset", size="sm")
                test_btn2 = gr.Button("📚 VPN Documentation", size="sm")
                test_btn3 = gr.Button("🎫 Create Ticket", size="sm")

                gr.Markdown("#### Customer Service")
                test_btn4 = gr.Button("📦 Order Status", size="sm")
                test_btn5 = gr.Button("❓ Return Policy", size="sm")
                test_btn6 = gr.Button("⚠️ Escalation", size="sm")

        # Test scenarios
        scenarios = {
            test_btn1: "I forgot my password and can't log into my laptop",
            test_btn2: "How do I configure VPN on my work laptop?",
            test_btn3: "Create an incident for my broken monitor",
            test_btn4: "Check status of order #12345",
            test_btn5: "What's your return policy?",
            test_btn6: "I need to escalate this issue to a supervisor",
        }

        # Event handlers
        def submit_message(msg, sid, hist):
            return chat(msg, sid, hist)

        submit.click(
            submit_message,
            inputs=[message, session_id_box, chatbot],
            outputs=[chatbot, session_id_box, message]
        )

        message.submit(
            submit_message,
            inputs=[message, session_id_box, chatbot],
            outputs=[chatbot, session_id_box, message]
        )

        # Test buttons
        for btn, scenario in scenarios.items():
            btn.click(
                lambda s=scenario: s,
                outputs=message
            )

        clear.click(
            lambda: ([], None),
            outputs=[chatbot, session_id_box]
        )

    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

**Run:**
```bash
pip install gradio
python web_ui.py
```

**Access:** http://localhost:7860

**Test Flow:**
1. Open browser to http://localhost:7860
2. Click test scenario buttons or type custom messages
3. Observe which agent handles each request
4. Verify responses are contextually appropriate
5. Test conversation continuity with follow-up questions

---

## 👥 Microsoft Teams Testing

### Prerequisites

1. **Teams App Registration**: You need a Bot registration in Azure
2. **App ID & Password**: From Azure Bot Service
3. **ngrok or Azure Web App**: For public endpoint

### Setup

```bash
cd deploy/teams

# Configure credentials
export TEAMS_APP_ID=your-app-id
export TEAMS_APP_PASSWORD=your-password
export TEAMS_BOT_ENDPOINT=https://your-ngrok-url.ngrok.io

# Deploy manifest
python deploy_teams.py

# Start server
uvicorn deploy_teams:app --host 0.0.0.0 --port 8000
```

### Using ngrok for Testing

```bash
# In new terminal
ngrok http 8000

# Copy HTTPS URL and update TEAMS_BOT_ENDPOINT
# Example: https://abc123.ngrok.io
```

### Upload to Teams

1. Open Microsoft Teams
2. Go to **Apps** > **Manage your apps** > **Upload an app**
3. Upload the `teams_manifest.zip` generated
4. Add the app to a team or chat

### Test in Teams

**Test 1: Direct Message**
```
@YourBot I need help with my password
```

**Test 2: Channel Mention**
```
@YourBot How do I configure VPN?
```

**Test 3: Adaptive Card Response**
```
@YourBot Create a ticket for broken hardware
```

**Expected:** Bot responds with interactive adaptive card

---

## 🤝 M365 Copilot Testing

### Prerequisites

1. **M365 Copilot License**: Required for testing
2. **Plugin Deployed**: Must complete Copilot Studio deployment

### Setup

```bash
cd deploy/copilot

# Generate plugin
python deploy_copilot.py

# Follow instructions in copilot_plugin/README.md to import
```

### Import to Copilot Studio

1. Open **Copilot Studio**: https://copilotstudio.microsoft.com
2. Navigate to **Plugins** > **Add Plugin**
3. Select **Import from API specification**
4. Upload the generated OpenAPI spec from `copilot_plugin/`
5. Configure authentication (API key or OAuth)
6. Test in Copilot Studio Test Canvas

### Test in M365 Copilot

**Test 1: Invoke from Copilot Chat**
```
Use IT Support Hub to help me reset my password
```

**Test 2: Natural Integration**
```
I'm having issues with my laptop password
```
(Copilot should automatically detect and invoke your plugin)

**Test 3: Specific Plugin Call**
```
@ITSupportHub create a ticket for my monitor
```

---

## ☁️ Azure Functions Testing

### Local Testing (Before Deployment)

```bash
cd deploy/functions

# Generate function app
python deploy_functions.py

cd function_app

# Install Azure Functions Core Tools
# https://learn.microsoft.com/azure/azure-functions/functions-run-local

# Start local function
func start
```

**Test with curl:**
```bash
curl -X POST http://localhost:7071/api/hub \
  -H "Content-Type: application/json" \
  -d '{"message": "I need help with my password", "session_id": "test-123"}'
```

### Deploy to Azure

```bash
cd function_app

# Login to Azure
az login

# Deploy
func azure functionapp publish your-function-app-name
```

### Test Deployed Function

```bash
# Get function URL from Azure Portal
FUNCTION_URL="https://your-app.azurewebsites.net/api/hub?code=your-key"

curl -X POST $FUNCTION_URL \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a ticket for broken monitor"}'
```

---

## 📊 Monitoring & Observability

### Azure Monitor

**View in Azure Portal:**
1. Go to **Azure Portal** > **Application Insights**
2. Navigate to **Transaction Search**
3. Filter by **Custom Events** > Look for agent invocations
4. Check **Performance** > See agent response times

**Query with Kusto:**
```kql
traces
| where customDimensions.agent_name != ""
| project timestamp, message, agent=customDimensions.agent_name, session=customDimensions.session_id
| order by timestamp desc
| take 100
```

### LangSmith

**View Traces:**
1. Open **LangSmith**: https://smith.langchain.com
2. Navigate to your project
3. Filter by **Session ID** to see conversation flows
4. Click traces to see:
   - Which agents were called
   - Token usage per agent
   - Latency breakdown
   - Tool invocations

**Key Metrics to Monitor:**
- **Agent Selection Accuracy**: Is routing correct?
- **Response Time**: <2s per agent call ideal
- **Token Usage**: Monitor costs
- **Error Rate**: Should be <1%

---

## ❗ Troubleshooting

### Issue: "Cannot import orchestrator"

**Solution:**
```bash
# Ensure you're in the correct directory
cd samples/enterprise-{hub_name}

# Check if orchestrator.py exists
ls orchestrator.py

# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Issue: "Agent creation failed"

**Solution:**
```bash
# Check Azure credentials
echo $AZURE_AI_PROJECT_ENDPOINT

# Test connection
az account show

# Verify API keys in .env
cat .env | grep AZURE
```

### Issue: "Router returns wrong agent"

**Diagnosis:**
- Check routing logic in orchestrator.py
- Verify agent instructions are clear
- Test with more explicit prompts

**Fix:**
```python
# Add debug logging in orchestrator.py
print(f"Routing decision: {chosen_agent}")
print(f"User message: {user_message}")
```

### Issue: "Teams bot not responding"

**Checklist:**
- [ ] ngrok or public endpoint running?
- [ ] Bot endpoint configured correctly?
- [ ] App ID and password correct?
- [ ] Manifest uploaded to Teams?
- [ ] Bot added to conversation?

**Test:**
```bash
# Test webhook endpoint
curl -X POST https://your-bot-url/api/messages \
  -H "Content-Type: application/json" \
  -d '{"type":"message","text":"test"}'
```

---

## ✅ Validation Checklist

Before considering testing complete:

### Functionality
- [ ] All agents can be invoked individually
- [ ] Router correctly identifies agent for each test scenario
- [ ] Multi-turn conversations maintain context
- [ ] RAG agents return relevant information
- [ ] Error handling works gracefully

### Performance
- [ ] Response time < 3 seconds average
- [ ] No timeout errors under load
- [ ] Memory usage stable over time

### Deployments
- [ ] Local command-line testing works
- [ ] Web UI (if deployed) is responsive
- [ ] Teams bot (if deployed) responds correctly
- [ ] Copilot plugin (if deployed) is invokable
- [ ] Azure Functions (if deployed) handle requests

### Monitoring
- [ ] Azure Monitor receiving traces
- [ ] LangSmith showing execution graphs
- [ ] Custom metrics logging correctly

---

## 📚 Additional Resources

- **Orchestrator Code**: `samples/enterprise-{hub_name}/orchestrator.py`
- **Agent Implementations**: `samples/enterprise-{hub_name}/agents/`
- **Deployment Guides**: `samples/enterprise-{hub_name}/deploy/*/README.md`
- **Azure AI Foundry**: https://aka.ms/azureai/langchain
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Teams Bot Docs**: https://learn.microsoft.com/microsoftteams/platform/bots/

---

## 🎯 Sample Test Report Template

```markdown
# Test Report: {Hub Name}

**Date**: {date}
**Tester**: {your name}
**Environment**: {local/staging/production}

## Test Results

### IT Support Scenarios
- ✅ Password Reset: PASS - Routed to helpdesk
- ✅ VPN Documentation: PASS - Routed to knowledge_base
- ✅ Ticket Creation: PASS - Routed to servicenow

### Customer Service Scenarios
- ✅ Order Status: PASS - Routed to support
- ✅ Return Policy: PASS - Routed to faq
- ✅ Escalation: PASS - Routed to ticketing

### Performance
- Average Response Time: 1.8s
- Error Rate: 0%
- Token Usage: ~500 tokens/request

### Issues Found
None

### Recommendations
- Consider adding more training examples for routing
- Monitor token usage in production

## Sign-off
Approved for production deployment: ✅
```

---

**Happy Testing! 🎉**

For issues or questions, refer to the main README.md or open an issue.
