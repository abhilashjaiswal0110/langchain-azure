#!/bin/bash
#
# Install LangChain Azure Claude Skills Globally
#
# This script installs all skills to your global Claude skills directory
# making them available across all your projects.
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  LangChain Azure Skills Installer${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Determine skills directory
SKILLS_DIR="$HOME/.config/claude/skills"

# Check if we're in the repository root
if [ ! -d ".claude/skills" ]; then
    echo -e "${YELLOW}Error: Must run from repository root${NC}"
    echo "Current directory: $(pwd)"
    echo "Expected: .claude/skills/ directory"
    exit 1
fi

# Create global skills directory if it doesn't exist
echo -e "${BLUE}📁 Setting up global skills directory...${NC}"
mkdir -p "$SKILLS_DIR"
echo -e "${GREEN}✓${NC} Directory: $SKILLS_DIR"
echo

# Ask user for installation method
echo -e "${BLUE}Choose installation method:${NC}"
echo "  1) Copy skills (static, won't auto-update)"
echo "  2) Symlink skills (dynamic, auto-updates with repo)"
echo
read -p "Select [1-2]: " choice
echo

case $choice in
    1)
        # Copy method
        echo -e "${BLUE}📦 Installing skills (copy method)...${NC}"
        for skill in .claude/skills/*/; do
            if [ -d "$skill" ]; then
                skill_name=$(basename "$skill")
                echo -e "${BLUE}  Installing: $skill_name${NC}"

                # Remove existing if present
                if [ -e "$SKILLS_DIR/$skill_name" ]; then
                    rm -rf "$SKILLS_DIR/$skill_name"
                fi

                # Copy skill
                cp -r "$skill" "$SKILLS_DIR/$skill_name"
                echo -e "${GREEN}  ✓ Installed: $skill_name${NC}"
            fi
        done

        echo
        echo -e "${YELLOW}📝 Note: To update skills, re-run this installer${NC}"
        ;;

    2)
        # Symlink method
        echo -e "${BLUE}🔗 Installing skills (symlink method)...${NC}"
        REPO_DIR="$(pwd)"

        for skill in .claude/skills/*/; do
            if [ -d "$skill" ]; then
                skill_name=$(basename "$skill")
                echo -e "${BLUE}  Symlinking: $skill_name${NC}"

                # Remove existing if present
                if [ -e "$SKILLS_DIR/$skill_name" ]; then
                    rm -rf "$SKILLS_DIR/$skill_name"
                fi

                # Create symlink
                ln -s "$REPO_DIR/$skill" "$SKILLS_DIR/$skill_name"
                echo -e "${GREEN}  ✓ Symlinked: $skill_name${NC}"
            fi
        done

        echo
        echo -e "${YELLOW}📝 Note: Skills auto-update when you git pull this repo${NC}"
        ;;

    *)
        echo -e "${YELLOW}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# List installed skills
echo -e "${BLUE}📦 Installed Skills:${NC}"
for skill in "$SKILLS_DIR"/*/skill.json; do
    if [ -f "$skill" ]; then
        skill_dir=$(dirname "$skill")
        skill_name=$(basename "$skill_dir")

        # Extract description from skill.json
        if command -v jq &> /dev/null; then
            description=$(jq -r '.description' "$skill" 2>/dev/null || echo "No description")
            echo -e "  ${GREEN}•${NC} /$skill_name - $description"
        else
            echo -e "  ${GREEN}•${NC} /$skill_name"
        fi
    fi
done

echo
echo -e "${BLUE}📚 Documentation:${NC}"
echo -e "  • Skills Guide: .claude/SKILLS_GUIDE.md"
echo -e "  • Global Setup: .claude/GLOBAL_SKILLS_SETUP.md"
echo -e "  • Skills README: .claude/skills/README.md"
echo

echo -e "${BLUE}🚀 Quick Start:${NC}"
echo -e "  # Open any project with Claude Code"
echo -e "  # Skills are now available:"
echo -e "  /new-agent my-agent"
echo -e "  /new-rag-pipeline my-pipeline"
echo -e "  /run-full-test"
echo

echo -e "${GREEN}Happy coding! 🎉${NC}"
