# LangChain Azure AI - Documentation

> **Enterprise-Grade AI Agents Platform Documentation**

This directory contains comprehensive documentation for the LangChain Azure AI Agents platform, including architecture guides, API references, tutorials, and operational runbooks.

---

## Documentation Structure

```
docs/
├── README.md (this file)
├── architecture/           # System architecture and design
│   └── ENTERPRISE_ARCHITECTURE.md
├── guides/                # User guides and tutorials
│   ├── DEEPAGENTS_EVALUATION_RESULTS.md
│   ├── EVALUATION_FRAMEWORK_COMPLETE.md
│   ├── EVALUATION_IMPLEMENTATION.md
│   ├── README_EVALUATION.md
│   └── SOFTWARE_DEV_SAMPLE_QUERIES.md
├── troubleshooting/       # Troubleshooting and debugging
│   └── RESTART_SERVER.md
├── api/                   # API documentation (future)
├── drafts/                # Work-in-progress docs (gitignored)
└── wip/                   # Draft documentation (gitignored)
```

---

## Quick Navigation

### Architecture & Design

- [**Enterprise Architecture**](architecture/ENTERPRISE_ARCHITECTURE.md) - Complete system architecture, component interactions, deployment patterns, and security design
- [Repository Structure](REPOSITORY_STRUCTURE.md) - Codebase organization and module structure

### User Guides

- [**Software Development Sample Queries**](guides/SOFTWARE_DEV_SAMPLE_QUERIES.md) - 40+ test queries for Software Development DeepAgent across all SDLC phases
- [DeepAgents Evaluation Results](guides/DEEPAGENTS_EVALUATION_RESULTS.md) - Performance benchmarks and evaluation metrics
- [Evaluation Framework](guides/EVALUATION_FRAMEWORK_COMPLETE.md) - Comprehensive testing and evaluation methodology
- [Evaluation Implementation](guides/EVALUATION_IMPLEMENTATION.md) - Technical implementation of evaluation framework

### Troubleshooting

- [**Server Restart Procedure**](troubleshooting/RESTART_SERVER.md) - Step-by-step server restart guide for DeepAgent loading issues

---

## Documentation by Use Case

### For Developers

**Getting Started:**
1. Read [Repository Structure](REPOSITORY_STRUCTURE.md) to understand the codebase
2. Review [Enterprise Architecture](architecture/ENTERPRISE_ARCHITECTURE.md) for system design
3. Check [Software Development Sample Queries](guides/SOFTWARE_DEV_SAMPLE_QUERIES.md) for usage examples

**Development:**
- Architecture patterns → [Enterprise Architecture](architecture/ENTERPRISE_ARCHITECTURE.md)
- Code examples → [Software Development Sample Queries](guides/SOFTWARE_DEV_SAMPLE_QUERIES.md)
- Testing methodology → [Evaluation Framework](guides/EVALUATION_FRAMEWORK_COMPLETE.md)

**Troubleshooting:**
- Server issues → [Server Restart Procedure](troubleshooting/RESTART_SERVER.md)
- Performance issues → [Evaluation Results](guides/DEEPAGENTS_EVALUATION_RESULTS.md)

### For DevOps/SRE

**Deployment:**
- Architecture overview → [Enterprise Architecture](architecture/ENTERPRISE_ARCHITECTURE.md)
- Deployment patterns → [Enterprise Architecture - Deployment](architecture/ENTERPRISE_ARCHITECTURE.md#deployment-architecture)
- Monitoring setup → [Enterprise Architecture - Observability](architecture/ENTERPRISE_ARCHITECTURE.md#observability--monitoring)

**Operations:**
- Server management → [Server Restart Procedure](troubleshooting/RESTART_SERVER.md)
- Performance monitoring → [Evaluation Results](guides/DEEPAGENTS_EVALUATION_RESULTS.md)

### For Product Managers

**Platform Overview:**
- System capabilities → [Enterprise Architecture](architecture/ENTERPRISE_ARCHITECTURE.md)
- Agent types and use cases → [Enterprise Architecture - Agent Types](architecture/ENTERPRISE_ARCHITECTURE.md#agent-types-hierarchy)
- Evaluation metrics → [Evaluation Results](guides/DEEPAGENTS_EVALUATION_RESULTS.md)

**Feature Documentation:**
- Software Development Agent → [Sample Queries](guides/SOFTWARE_DEV_SAMPLE_QUERIES.md)
- Multi-agent workflows → [Enterprise Architecture - DeepAgent Patterns](architecture/ENTERPRISE_ARCHITECTURE.md#deepagent-multi-agent-patterns)

---

## Key Documentation

### 1. Enterprise Architecture

[**architecture/ENTERPRISE_ARCHITECTURE.md**](architecture/ENTERPRISE_ARCHITECTURE.md)

Comprehensive system architecture documentation covering:
- System architecture overview (multi-agent orchestration, session management, observability)
- Agent types hierarchy (Foundry, Enterprise, DeepAgents, IT Agents)
- DeepAgent multi-agent patterns (supervisor-worker, tool routing, state management)
- Software Development DeepAgent (9 SDLC phases, 54 specialized tools)
- Security architecture (authentication, data protection, compliance)
- Deployment patterns (containerization, orchestration, scaling)
- Observability and monitoring (tracing, metrics, logging)
- API endpoints and integration patterns
- Performance characteristics and benchmarks

**Use when:** Understanding system design, planning deployment, architectural decisions

### 2. Software Development Sample Queries

[**guides/SOFTWARE_DEV_SAMPLE_QUERIES.md**](guides/SOFTWARE_DEV_SAMPLE_QUERIES.md)

40+ comprehensive test queries for Software Development DeepAgent:
- Requirements Intelligence (4 queries) - Requirements analysis, user stories, prioritization
- Architecture Design (4 queries) - System design, API specs, database schemas
- Code Generation (4 queries) - API endpoints, data models, CRUD operations
- Code Review (4 queries) - Quality analysis, security scanning, complexity analysis
- Testing Automation (5 queries) - Unit tests, integration tests, coverage analysis
- Debugging & Optimization (4 queries) - Error analysis, performance profiling
- Security & Compliance (4 queries) - OWASP scanning, vulnerability assessment
- DevOps Integration (4 queries) - Docker, CI/CD, Kubernetes
- Documentation (4 queries) - API docs, architecture diagrams
- Multi-Phase Workflows (4 queries) - End-to-end feature development

**Use when:** Testing Software Development DeepAgent, learning agent capabilities, demo preparation

### 3. Server Restart Procedure

[**troubleshooting/RESTART_SERVER.md**](troubleshooting/RESTART_SERVER.md)

Step-by-step troubleshooting guide for:
- Identifying server process issues
- Stopping and restarting the server
- Verifying agent loading (especially Software Development DeepAgent)
- Common troubleshooting scenarios
- Verification commands and expected outputs

**Use when:** Agents not loading, server showing wrong agent count, after code updates

---

## Documentation Standards

### File Organization

**DO:**
- ✅ Place architectural docs in [architecture/](architecture/)
- ✅ Place guides and tutorials in [guides/](guides/)
- ✅ Place troubleshooting docs in [troubleshooting/](troubleshooting/)
- ✅ Use descriptive, consistent naming (UPPERCASE for major docs)
- ✅ Include table of contents for docs > 200 lines
- ✅ Add cross-references to related documentation

**DON'T:**
- ❌ Put temporary/draft docs in root folder (use docs/drafts/ instead)
- ❌ Commit WIP documentation (use docs/wip/ which is gitignored)
- ❌ Duplicate information across multiple docs (link instead)
- ❌ Use generic names like "NOTES.md" or "TODO.md" in tracked folders

### Writing Style

- **Clear and concise**: Use active voice, short sentences
- **Code examples**: Include working code snippets with proper syntax highlighting
- **Visual aids**: Add diagrams (Mermaid, PlantUML) for complex concepts
- **Maintenance**: Keep docs updated with code changes
- **Audience-aware**: Write for the intended reader (developer, ops, PM)

### Document Templates

Each major document should include:
1. **Title and description** - What this doc covers
2. **Table of contents** - For easy navigation (docs > 200 lines)
3. **Quick start** - Get readers productive quickly
4. **Main content** - Detailed information with examples
5. **Troubleshooting** - Common issues and solutions
6. **Related resources** - Links to related docs

---

## Contributing to Documentation

### Adding New Documentation

1. **Choose the right location:**
   - Architecture docs → `architecture/`
   - User guides → `guides/`
   - Troubleshooting → `troubleshooting/`
   - API reference → `api/`

2. **Follow naming conventions:**
   - Use UPPERCASE for major documentation (e.g., `ARCHITECTURE.md`)
   - Use descriptive names (e.g., `SOFTWARE_DEV_SAMPLE_QUERIES.md` not `QUERIES.md`)

3. **Update this README:**
   - Add entry to "Quick Navigation"
   - Update "Documentation by Use Case" if applicable

### Updating Existing Documentation

1. Ensure changes are accurate and tested
2. Update modification date at top of document
3. Maintain backward compatibility (don't break links)
4. Update cross-references if structure changes

### Draft Documentation

For work-in-progress documentation:
- Use `docs/drafts/` or `docs/wip/` (gitignored)
- Move to proper location when ready for review
- Delete old drafts to avoid clutter

---

## Documentation Roadmap

### Planned Documentation

- [ ] **API Reference** - Complete REST API documentation with OpenAPI specs
- [ ] **Development Guide** - Step-by-step guide for adding new agents and tools
- [ ] **Deployment Guide** - Production deployment checklist and best practices
- [ ] **Security Guide** - Security best practices and compliance requirements
- [ ] **Performance Tuning Guide** - Optimization strategies and benchmarks
- [ ] **Integration Guide** - Integrating with external systems (Copilot Studio, Azure AI)

### Documentation Improvements

- [ ] Add Mermaid diagrams to architecture documentation
- [ ] Create video tutorials for common workflows
- [ ] Add API usage examples in multiple languages (Python, JavaScript, cURL)
- [ ] Create runbooks for common operational tasks
- [ ] Add architecture decision records (ADRs)

---

## Getting Help

- **Issues**: Report documentation issues on GitHub Issues
- **Questions**: Use GitHub Discussions for questions
- **Contributions**: Submit PRs for documentation improvements

---

## Documentation Maintenance

**Last Updated**: 2026-01-30
**Maintained By**: Development Team
**Review Frequency**: Monthly or with major releases

**Change Log:**
- 2026-01-30: Initial structured documentation with reorganized folders
- 2026-01-29: Added Software Development Sample Queries
- 2026-01-29: Added Enterprise Architecture documentation
- 2026-01-27: Added Evaluation Framework documentation
