# Skills Quick Start

## Test Immediately (No Installation Required)

Skills are ready to use in this repository:

### 1. Create a Test Agent
\`\`\`
/new-agent test-agent --type react
\`\`\`

### 2. Create an Enterprise Hub
\`\`\`
/create-enterprise-hub demo-hub --use_case it-support
\`\`\`

### 3. Test the Hub
\`\`\`bash
cd samples/enterprise-demo-hub
pip install -r requirements.txt
cp .env.example .env
# Edit .env with Azure credentials
python orchestrator.py
\`\`\`

## Install Globally (Optional)

### Unix/macOS/Linux
\`\`\`bash
./.claude/skills/install.sh
\`\`\`

### Windows
\`\`\`powershell
.\.claude\skills\install.ps1
\`\`\`

## Next Steps

- Read: \`.claude/ENTERPRISE_SKILLS_SUMMARY.md\`
- UI Testing: \`docs/ENTERPRISE_UI_TESTING_GUIDE.md\`
- Full Guide: \`.claude/SKILLS_GUIDE.md\`
