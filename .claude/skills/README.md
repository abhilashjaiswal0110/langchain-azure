# Claude Agent Skills for LangChain Azure

> **Quick Reference**: Ready-to-use skills for common LangChain Azure development tasks

---

## Available Skills

### 📦 Agent & Chain Creation

#### `/new-agent` - Create New LangChain Agent
Create a new Azure AI agent with proper structure, tools, and tests.

```bash
/new-agent customer-support --type react --model gpt-4o
```

**Creates:**
- Agent implementation file
- Unit tests
- Example usage
- Updated exports

[Full Documentation](new-agent/prompt.md)

---

#### `/new-chain` - Create New LangChain Chain
Create a new chain with Azure AI models.

```bash
/new-chain summarization --model gpt-4o
```

[Coming Soon]

---

### 🔍 RAG & Retrieval

#### `/new-rag-pipeline` - Setup RAG Pipeline
Create complete RAG pipeline with Azure services.

```bash
/new-rag-pipeline docs-qa --storage search --embeddings azure
```

**Creates:**
- Pipeline module
- Document embedding script
- Query interface
- Configuration templates

[Full Documentation](new-rag-pipeline/prompt.md)

---

### 🧪 Testing & Quality

#### `/run-full-test` - Run Complete Test Suite
Run all tests, linting, formatting, and type checking.

```bash
/run-full-test --package azure-ai --coverage true
```

**Runs:**
- Unit tests
- Integration tests (optional)
- Linting (ruff)
- Formatting check
- Type checking (mypy)
- Coverage report

[Full Documentation](run-full-test/prompt.md)

---

### 🏢 Enterprise Multi-Agent Systems

#### `/create-enterprise-hub` - Create Enterprise Multi-Agent Hub
**NEW!** Create end-to-end enterprise system with multiple agents, routing, and deployment options.

```bash
/create-enterprise-hub it-support-hub --use_case it-support
```

**Creates:**
- Multi-agent orchestration system
- Intelligent routing between agents
- RAG knowledge base integration
- Teams + Copilot Studio + Azure Functions deployment
- Monitoring and observability

**Use Cases:**
- **IT Support Hub**: Helpdesk + Knowledge Base + ServiceNow
- **Customer Service**: Support + FAQ + Ticketing
- **Doc Intelligence**: Processing + Analysis + Q&A

[Full Documentation](create-enterprise-hub/prompt.md)

---

### 🚀 Deployment

#### `/deploy-package` - Deploy Package to PyPI
Build and deploy a package to PyPI.

```bash
/deploy-package azure-ai --test-pypi true
```

[Coming Soon]

---

### 🔌 Enterprise Integrations

#### `/add-connector` - Add Enterprise Connector
Add new enterprise connector (Teams, Functions, Copilot Studio).

```bash
/add-connector teams-bot --type teams
```

[Coming Soon]

---

## Quick Start

### Using Skills Locally (This Repository)

Skills are already available in this repository:

```bash
# Just use them!
/new-agent my-agent
/new-rag-pipeline my-pipeline
/run-full-test --package azure-ai
```

### Installing Skills Globally

Use skills across all your projects:

**macOS/Linux:**
```bash
# Symlink all skills
for skill in .claude/skills/*/; do
    ln -s "$(pwd)/$skill" ~/.config/claude/skills/"$(basename $skill)"
done
```

**Windows:**
```powershell
# Copy all skills
Get-ChildItem .claude\skills -Directory | ForEach-Object {
    Copy-Item -Recurse $_.FullName "$env:APPDATA\Claude\skills\$($_.Name)"
}
```

See [GLOBAL_SKILLS_SETUP.md](../GLOBAL_SKILLS_SETUP.md) for detailed instructions.

---

## Skill Structure

Each skill contains:

```
skill-name/
├── skill.json          # Metadata and parameters
├── prompt.md           # Execution instructions
├── README.md           # User documentation (optional)
└── examples/           # Usage examples (optional)
```

### skill.json
Defines skill metadata, parameters, and examples.

### prompt.md
Detailed instructions for Claude on how to execute the skill.

---

## Creating New Skills

Want to create your own skill?

1. **Create directory**: `.claude/skills/my-skill/`
2. **Add skill.json**: Define metadata and parameters
3. **Add prompt.md**: Write execution instructions
4. **Test**: Try the skill in Claude Code
5. **Document**: Add to this README

See [SKILLS_GUIDE.md](../SKILLS_GUIDE.md) for complete guide.

---

## Skill Development Workflow

### 1. Design Phase
- Identify repetitive workflow
- Define parameters
- Plan file structure

### 2. Implementation Phase
```bash
# Create skill structure
mkdir .claude/skills/my-skill
cd .claude/skills/my-skill

# Create files
touch skill.json prompt.md README.md
```

### 3. Testing Phase
```bash
# Test in Claude Code
/my-skill param1 --param2 value
```

### 4. Documentation Phase
- Update this README
- Add usage examples
- Document prerequisites

---

## Best Practices

### Skill Design
- ✅ Single responsibility per skill
- ✅ Clear parameter names
- ✅ Comprehensive error handling
- ✅ Idempotent operations
- ✅ Detailed documentation

### Prompt Engineering
- ✅ Step-by-step instructions
- ✅ Include file templates
- ✅ Specify validation checks
- ✅ Handle edge cases
- ✅ Provide clear output format

### Repository Integration
- ✅ Follow .claude/CLAUDE.md guidelines
- ✅ Use existing patterns
- ✅ Update documentation
- ✅ Include tests
- ✅ Atomic commits

---

## Troubleshooting

### Skill Not Found
```bash
# Verify skill exists
ls .claude/skills/skill-name/

# Check skill.json is valid
cat .claude/skills/skill-name/skill.json | python -m json.tool
```

### Skill Execution Fails
1. Review prerequisites in skill.json
2. Check current directory
3. Verify repository structure
4. Review error messages
5. Check skill documentation

### Need Help?
- 📖 [SKILLS_GUIDE.md](../SKILLS_GUIDE.md) - Complete guide
- 📖 [GLOBAL_SKILLS_SETUP.md](../GLOBAL_SKILLS_SETUP.md) - Global installation
- 📖 [CLAUDE.md](../CLAUDE.md) - Repository guidelines
- 🐛 Open an issue with `skills` label

---

## Roadmap

### Planned Skills

- [ ] `/new-chain` - Create new LangChain chain
- [ ] `/add-tool` - Add tool to agent
- [ ] `/setup-monitoring` - Add Azure Monitor integration
- [ ] `/create-evaluation` - Create LangSmith eval suite
- [ ] `/add-azure-integration` - Add Azure service integration
- [ ] `/migrate-to-langgraph` - Migrate legacy agents
- [ ] `/add-streaming` - Add streaming support
- [ ] `/benchmark-agent` - Run performance benchmarks
- [ ] `/create-example` - Create sample application
- [ ] `/deploy-package` - Deploy to PyPI

### Ideas Welcome!
Have a skill idea? Open an issue with `[Skill]` prefix.

---

## Contributing

We welcome contributions!

### Adding a Skill
1. Create skill in `.claude/skills/new-skill/`
2. Test thoroughly
3. Update this README
4. Create pull request
5. Tag maintainers

### Improving a Skill
1. Test your improvements
2. Update documentation
3. Create pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for full guidelines.

---

## Resources

### Documentation
- [SKILLS_GUIDE.md](../SKILLS_GUIDE.md) - Complete skill creation guide
- [GLOBAL_SKILLS_SETUP.md](../GLOBAL_SKILLS_SETUP.md) - Global installation
- [CLAUDE.md](../CLAUDE.md) - Repository development guidelines

### External Resources
- [LangChain Documentation](https://python.langchain.com/)
- [Azure AI Foundry](https://aka.ms/azureai/langchain)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

### Support
- 📧 Email: abhilashjaiswal0110@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/abhilashjaiswal0110/langchain-azure/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/abhilashjaiswal0110/langchain-azure/discussions)

---

## License

Skills inherit the repository license (MIT).

---

<div align="center">

**Automate • Accelerate • Create**

Made with ❤️ for the LangChain Azure community

[Report Bug](https://github.com/abhilashjaiswal0110/langchain-azure/issues) • [Request Skill](https://github.com/abhilashjaiswal0110/langchain-azure/issues/new?labels=enhancement,skills)

</div>
