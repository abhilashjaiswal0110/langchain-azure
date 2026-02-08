# Install LangChain Azure Claude Skills Globally
#
# This script installs all skills to your global Claude skills directory
# making them available across all your projects.
#

$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-Host ""
Write-ColorOutput Blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-ColorOutput Blue "  LangChain Azure Skills Installer"
Write-ColorOutput Blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

# Determine skills directory
$SkillsDir = "$env:APPDATA\Claude\skills"

# Check if we're in the repository root
if (!(Test-Path ".claude\skills")) {
    Write-ColorOutput Yellow "Error: Must run from repository root"
    Write-Host "Current directory: $(Get-Location)"
    Write-Host "Expected: .claude\skills\ directory"
    exit 1
}

# Create global skills directory if it doesn't exist
Write-ColorOutput Blue "📁 Setting up global skills directory..."
New-Item -ItemType Directory -Force -Path $SkillsDir | Out-Null
Write-ColorOutput Green "✓ Directory: $SkillsDir"
Write-Host ""

# Ask user for installation method
Write-ColorOutput Blue "Choose installation method:"
Write-Host "  1) Copy skills (static, won't auto-update)"
Write-Host "  2) Symlink skills (dynamic, auto-updates with repo)"
Write-Host "     Note: Symlinks require Developer Mode or Admin privileges"
Write-Host ""
$choice = Read-Host "Select [1-2]"
Write-Host ""

switch ($choice) {
    1 {
        # Copy method
        Write-ColorOutput Blue "📦 Installing skills (copy method)..."

        Get-ChildItem ".claude\skills" -Directory | ForEach-Object {
            $skillName = $_.Name
            Write-ColorOutput Blue "  Installing: $skillName"

            $targetPath = "$SkillsDir\$skillName"

            # Remove existing if present
            if (Test-Path $targetPath) {
                Remove-Item -Recurse -Force $targetPath
            }

            # Copy skill
            Copy-Item -Recurse $_.FullName $targetPath
            Write-ColorOutput Green "  ✓ Installed: $skillName"
        }

        Write-Host ""
        Write-ColorOutput Yellow "📝 Note: To update skills, re-run this installer"
    }

    2 {
        # Symlink method
        Write-ColorOutput Blue "🔗 Installing skills (symlink method)..."
        $repoDir = Get-Location

        Get-ChildItem ".claude\skills" -Directory | ForEach-Object {
            $skillName = $_.Name
            Write-ColorOutput Blue "  Symlinking: $skillName"

            $targetPath = "$SkillsDir\$skillName"

            # Remove existing if present
            if (Test-Path $targetPath) {
                Remove-Item -Recurse -Force $targetPath
            }

            try {
                # Create symlink
                New-Item -ItemType SymbolicLink `
                    -Path $targetPath `
                    -Target $_.FullName `
                    -ErrorAction Stop | Out-Null

                Write-ColorOutput Green "  ✓ Symlinked: $skillName"
            }
            catch {
                Write-ColorOutput Yellow "  ⚠ Failed to create symlink for $skillName"
                Write-ColorOutput Yellow "    Enable Developer Mode or run as Administrator"
                Write-ColorOutput Yellow "    Falling back to copy method..."

                Copy-Item -Recurse $_.FullName $targetPath
                Write-ColorOutput Green "  ✓ Copied: $skillName"
            }
        }

        Write-Host ""
        Write-ColorOutput Yellow "📝 Note: Skills auto-update when you git pull this repo"
    }

    default {
        Write-ColorOutput Yellow "Invalid choice. Exiting."
        exit 1
    }
}

Write-Host ""
Write-ColorOutput Blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-ColorOutput Green "✅ Installation Complete!"
Write-ColorOutput Blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

# List installed skills
Write-ColorOutput Blue "📦 Installed Skills:"
Get-ChildItem "$SkillsDir\*\skill.json" | ForEach-Object {
    $skillDir = Split-Path $_.FullName
    $skillName = Split-Path $skillDir -Leaf

    # Extract description from skill.json
    try {
        $skillData = Get-Content $_.FullName | ConvertFrom-Json
        $description = $skillData.description
        Write-Host "  " -NoNewline
        Write-ColorOutput Green "• /$skillName - $description"
    }
    catch {
        Write-Host "  " -NoNewline
        Write-ColorOutput Green "• /$skillName"
    }
}

Write-Host ""
Write-ColorOutput Blue "📚 Documentation:"
Write-Host "  • Skills Guide: .claude\SKILLS_GUIDE.md"
Write-Host "  • Global Setup: .claude\GLOBAL_SKILLS_SETUP.md"
Write-Host "  • Skills README: .claude\skills\README.md"
Write-Host ""

Write-ColorOutput Blue "🚀 Quick Start:"
Write-Host "  # Open any project with Claude Code"
Write-Host "  # Skills are now available:"
Write-Host "  /new-agent my-agent"
Write-Host "  /new-rag-pipeline my-pipeline"
Write-Host "  /run-full-test"
Write-Host ""

Write-ColorOutput Green "Happy coding! 🎉"
Write-Host ""
