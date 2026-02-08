# Global Skills Installation Guide

> **Purpose**: Install Claude Agent SDK skills globally for use across all projects

---

## Table of Contents

1. [Overview](#overview)
2. [Local vs Global Skills](#local-vs-global-skills)
3. [Installation Methods](#installation-methods)
4. [Managing Global Skills](#managing-global-skills)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Overview

Claude Agent SDK skills can be installed:
- **Locally**: In `.claude/skills/` (repository-specific, shared via git)
- **Globally**: In your user config directory (available for all projects)

This guide focuses on **global installation** for skills you want to use across multiple projects.

---

## Local vs Global Skills

### Local Skills (.claude/skills/)

**Advantages:**
- Shared with team via git
- Project-specific customization
- Version controlled
- Automatically available when cloning repo

**Use cases:**
- Project-specific workflows
- Team standardization
- Repository conventions

**Example:** `/new-agent` skill for LangChain Azure

### Global Skills

**Advantages:**
- Available in any project
- Personal skill library
- No need to copy between projects
- Can be project-agnostic

**Use cases:**
- Generic development tasks
- Personal workflows
- Cross-project utilities

**Example:** `/create-python-package`, `/setup-git-hooks`

---

## Installation Methods

### Method 1: Copy Skills to Global Directory

#### Windows

```powershell
# Navigate to the global skills directory
cd $env:APPDATA\Claude\skills

# If directory doesn't exist, create it
if (!(Test-Path $env:APPDATA\Claude\skills)) {
    New-Item -ItemType Directory -Path $env:APPDATA\Claude\skills
}

# Copy a skill from your repository
# (from your repo root)
cd C:\path\to\langchain-azure
Copy-Item -Recurse .claude\skills\new-agent $env:APPDATA\Claude\skills\

# Verify installation
Get-ChildItem $env:APPDATA\Claude\skills\new-agent
```

#### macOS / Linux

```bash
# Navigate to the global skills directory
cd ~/.config/claude/skills

# If directory doesn't exist, create it
mkdir -p ~/.config/claude/skills

# Copy a skill from your repository
# (from your repo root)
cd /path/to/langchain-azure
cp -r .claude/skills/new-agent ~/.config/claude/skills/

# Verify installation
ls -la ~/.config/claude/skills/new-agent
```

### Method 2: Symbolic Links (Recommended for Development)

Symbolic links keep your global skills in sync with your local repository skills.

#### Windows (requires admin privileges or Developer Mode)

```powershell
# Enable Developer Mode or run as Administrator
cd $env:APPDATA\Claude\skills

# Create symbolic link
New-Item -ItemType SymbolicLink `
    -Path new-agent `
    -Target C:\path\to\langchain-azure\.claude\skills\new-agent

# Verify
Get-Item new-agent | Select-Object LinkType, Target
```

#### macOS / Linux

```bash
cd ~/.config/claude/skills

# Create symbolic link
ln -s /path/to/langchain-azure/.claude/skills/new-agent new-agent

# Verify
ls -la new-agent
```

**Benefits of Symlinks:**
- Always up-to-date with repository changes
- Edit once, works everywhere
- No manual syncing needed

### Method 3: Environment Variable (Advanced)

Set a custom skills directory:

#### Windows

```powershell
# Set permanently
[System.Environment]::SetEnvironmentVariable(
    'CLAUDE_SKILLS_PATH',
    'C:\path\to\langchain-azure\.claude\skills',
    'User'
)

# Verify
$env:CLAUDE_SKILLS_PATH
```

#### macOS / Linux

```bash
# Add to ~/.bashrc or ~/.zshrc
export CLAUDE_SKILLS_PATH="/path/to/langchain-azure/.claude/skills"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc

# Verify
echo $CLAUDE_SKILLS_PATH
```

---

## Managing Global Skills

### List Installed Skills

#### Windows

```powershell
Get-ChildItem -Directory $env:APPDATA\Claude\skills | Select-Object Name

# Or with details
Get-ChildItem -Recurse $env:APPDATA\Claude\skills\*\skill.json | ForEach-Object {
    Get-Content $_  | ConvertFrom-Json | Select-Object name, version, description
}
```

#### macOS / Linux

```bash
ls -1 ~/.config/claude/skills/

# Or with details
for skill in ~/.config/claude/skills/*/skill.json; do
    cat "$skill" | jq '{name, version, description}'
done
```

### Update Skills

#### If Copied (Method 1)

Re-copy the skill:

```bash
# macOS/Linux
cp -r /path/to/langchain-azure/.claude/skills/new-agent ~/.config/claude/skills/

# Windows
Copy-Item -Recurse -Force C:\path\to\langchain-azure\.claude\skills\new-agent $env:APPDATA\Claude\skills\
```

#### If Symlinked (Method 2)

Update automatically by pulling the repository:

```bash
cd /path/to/langchain-azure
git pull origin main
# Global skill automatically updated via symlink!
```

### Remove Skills

#### Windows

```powershell
Remove-Item -Recurse $env:APPDATA\Claude\skills\new-agent
```

#### macOS / Linux

```bash
rm -rf ~/.config/claude/skills/new-agent
```

---

## Installing All LangChain Azure Skills Globally

### Option A: Copy All Skills

#### Windows

```powershell
cd C:\path\to\langchain-azure

# Copy all skills
Get-ChildItem .claude\skills -Directory | ForEach-Object {
    Copy-Item -Recurse $_.FullName "$env:APPDATA\Claude\skills\$($_.Name)"
}

# Verify
Get-ChildItem $env:APPDATA\Claude\skills
```

#### macOS / Linux

```bash
cd /path/to/langchain-azure

# Copy all skills
cp -r .claude/skills/* ~/.config/claude/skills/

# Verify
ls ~/.config/claude/skills/
```

### Option B: Symlink All Skills

#### Windows

```powershell
cd C:\path\to\langchain-azure

Get-ChildItem .claude\skills -Directory | ForEach-Object {
    New-Item -ItemType SymbolicLink `
        -Path "$env:APPDATA\Claude\skills\$($_.Name)" `
        -Target $_.FullName
}
```

#### macOS / Linux

```bash
cd /path/to/langchain-azure

for skill in .claude/skills/*/; do
    skill_name=$(basename "$skill")
    ln -s "$(pwd)/$skill" ~/.config/claude/skills/"$skill_name"
done
```

### Option C: Use Custom Skills Path

```bash
# Add to shell config (~/.bashrc or ~/.zshrc)
export CLAUDE_SKILLS_PATH="/path/to/langchain-azure/.claude/skills"

# Now ALL skills from the repo are available globally!
```

---

## Best Practices

### 1. Choose Your Strategy

**Use Local Skills when:**
- Skills are project-specific
- Need to share with team
- Part of project workflow

**Use Global Skills when:**
- Skills work across projects
- Personal productivity
- Generic utilities

**Recommended Approach:**
- Keep project-specific skills local (`.claude/skills/`)
- Symlink useful skills globally for personal use
- Share generic skills via separate "skills-library" repo

### 2. Skill Organization

Create a personal skills library structure:

```
~/.config/claude/skills/
├── langchain/           # LangChain-specific
│   ├── new-agent/
│   ├── new-chain/
│   └── new-rag-pipeline/
├── python/              # Python generic
│   ├── create-package/
│   ├── add-tests/
│   └── setup-venv/
├── git/                 # Git utilities
│   ├── conventional-commit/
│   └── squash-commits/
└── docker/              # Docker skills
    ├── create-dockerfile/
    └── docker-compose-setup/
```

### 3. Version Control Your Global Skills

Create a personal skills repository:

```bash
# Create a git repo for your global skills
cd ~/.config/claude/skills
git init
git remote add origin git@github.com:yourusername/claude-skills.git

# Add skills
git add .
git commit -m "Initial commit: My Claude skills library"
git push -u origin main
```

**Benefits:**
- Backup your skills
- Share across machines
- Version history
- Easy to restore

### 4. Document Your Skills

Create a global README:

```bash
# ~/.config/claude/skills/README.md
```

Content:
```markdown
# My Claude Skills Library

Personal collection of Claude Agent SDK skills.

## Available Skills

- `/new-agent` - Create LangChain agent (from langchain-azure)
- `/new-chain` - Create LangChain chain
- `/run-full-test` - Complete test suite
- ... (list all your skills)

## Installation

# Clone this repo
git clone git@github.com:yourusername/claude-skills.git ~/.config/claude/skills

## Usage

Skills are automatically available in any Claude Code session.

See individual skill READMEs for usage.
```

---

## Troubleshooting

### Skill Not Found

**Problem:** `/my-skill` not recognized

**Solutions:**

1. **Verify skill location:**
   ```bash
   # macOS/Linux
   ls ~/.config/claude/skills/my-skill/

   # Windows
   Get-ChildItem $env:APPDATA\Claude\skills\my-skill\
   ```

2. **Check skill.json exists:**
   ```bash
   # macOS/Linux
   cat ~/.config/claude/skills/my-skill/skill.json

   # Windows
   Get-Content $env:APPDATA\Claude\skills\my-skill\skill.json
   ```

3. **Validate skill.json:**
   ```bash
   # macOS/Linux
   cat ~/.config/claude/skills/my-skill/skill.json | python -m json.tool

   # Windows
   Get-Content $env:APPDATA\Claude\skills\my-skill\skill.json | ConvertFrom-Json
   ```

4. **Restart Claude Code**

### Symbolic Link Issues

**Problem:** Symlink not working

**Windows:**
- Requires Developer Mode or Administrator privileges
- Enable Developer Mode: Settings > Update & Security > For Developers > Developer Mode

**macOS/Linux:**
- Check link with: `ls -la ~/.config/claude/skills/`
- Re-create if broken: `rm link && ln -s target link`

### Conflicting Skill Names

**Problem:** Same skill name exists locally and globally

**Resolution:**
- Local skills take precedence
- Rename global skill
- Or use skill namespaces (if supported)

### Permission Errors

**Problem:** Cannot write to global directory

**Solutions:**

```bash
# macOS/Linux - Fix permissions
chmod -R u+w ~/.config/claude/skills/

# Windows - Run as administrator or check folder permissions
```

---

## Advanced: Creating a Skills Distribution

If you want to share your skills as a package:

### 1. Create Distribution Structure

```
claude-langchain-skills/
├── README.md
├── LICENSE
├── install.sh          # Unix install script
├── install.ps1         # Windows install script
└── skills/
    ├── new-agent/
    ├── new-chain/
    └── new-rag-pipeline/
```

### 2. Create Install Scripts

**install.sh** (macOS/Linux):
```bash
#!/bin/bash
set -e

SKILLS_DIR="$HOME/.config/claude/skills"
mkdir -p "$SKILLS_DIR"

echo "Installing LangChain Azure skills..."
for skill in skills/*/; do
    skill_name=$(basename "$skill")
    cp -r "$skill" "$SKILLS_DIR/$skill_name"
    echo "✓ Installed: $skill_name"
done

echo "✅ All skills installed successfully!"
echo "Available skills: /new-agent, /new-chain, /new-rag-pipeline"
```

**install.ps1** (Windows):
```powershell
$SkillsDir = "$env:APPDATA\Claude\skills"
New-Item -ItemType Directory -Force -Path $SkillsDir | Out-Null

Write-Host "Installing LangChain Azure skills..."
Get-ChildItem skills -Directory | ForEach-Object {
    Copy-Item -Recurse -Force $_.FullName "$SkillsDir\$($_.Name)"
    Write-Host "✓ Installed: $($_.Name)"
}

Write-Host "✅ All skills installed successfully!"
Write-Host "Available skills: /new-agent, /new-chain, /new-rag-pipeline"
```

### 3. Distribute

- Publish to GitHub
- Create releases
- Users run install script:
  ```bash
  git clone https://github.com/user/claude-langchain-skills.git
  cd claude-langchain-skills
  ./install.sh  # or install.ps1 on Windows
  ```

---

## Resources

- **Repository Skills**: [.claude/SKILLS_GUIDE.md](SKILLS_GUIDE.md)
- **Claude Agent SDK**: [Documentation link]
- **Example Skills**: [.claude/skills/](.claude/skills/)

---

## Support

**Questions?**
1. Check [SKILLS_GUIDE.md](SKILLS_GUIDE.md)
2. Review skill README files
3. Open an issue in the repository

---

*Last Updated: 2026-02-08*
