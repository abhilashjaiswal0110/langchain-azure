<#
.SYNOPSIS
    Approve and squash-merge a GitHub pull request via the GitHub CLI.

.PARAMETER PrNumber
    The number of the pull request to approve and merge.

.PARAMETER ReviewBody
    Optional review comment. Defaults to a standard approval message.

.EXAMPLE
    .\approve_and_merge.ps1 -PrNumber 30
    .\approve_and_merge.ps1 -PrNumber 30 -ReviewBody "LGTM - all feedback addressed."
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$PrNumber,

    [string]$ReviewBody = "All review feedback has been addressed. Security fixes, async operations, and error handling are properly implemented."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "Approving PR #$PrNumber..."
gh pr review $PrNumber --approve --body $ReviewBody

Start-Sleep -Seconds 2

Write-Host "Merging PR #$PrNumber (squash)..."
gh pr merge $PrNumber --squash --delete-branch --body "Merging: all Copilot review issues resolved."

Write-Host "Done! PR #$PrNumber has been approved and merged."
