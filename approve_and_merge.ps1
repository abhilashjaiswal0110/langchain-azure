# Navigate to repository
Set-Location "C:\users\a833555\OneDrive - ATOS\Gitwork\langchain-azure"

# Approve the PR
Write-Host "Approving PR #29..."
gh pr review 29 --approve --body "All critical issues have been addressed. Security fixes, async operations, and error handling are properly implemented."

# Wait a moment
Start-Sleep -Seconds 2

# Merge the PR
Write-Host "Merging PR #29..."
gh pr merge 29 --squash --delete-branch --body "Merging: All Copilot review issues resolved"

Write-Host "Done!"
