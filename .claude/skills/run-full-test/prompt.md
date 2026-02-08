# Run Full Test Skill - Execution Instructions

You are executing the `run-full-test` skill for the LangChain Azure repository.

## Purpose

Run comprehensive quality checks including unit tests, integration tests (optional), linting, formatting, and type checking.

## Parameters

- **package** (optional, default: "all"): Which package to test
- **coverage** (optional, default: true): Generate HTML coverage report
- **integration** (optional, default: false): Run integration tests (needs Azure creds)

## Execution Steps

### Step 1: Determine Package Path

Based on `package` parameter:
- `all`: Test all packages in `libs/`
- `azure-ai`: Test `libs/azure-ai/`
- `azure-dynamic-sessions`: Test `libs/azure-dynamic-sessions/`
- `sqlserver`: Test `libs/sqlserver/`
- `azure-storage`: Test `libs/azure-storage/`
- `azure-postgresql`: Test `libs/azure-postgresql/`

### Step 2: Run Tests

For each package:

```bash
cd libs/{package_name}

# Unit tests
echo "🧪 Running unit tests..."
poetry run pytest tests/unit_tests/ -v --tb=short

# Integration tests (if integration=true)
if [[ "$integration" == "true" ]]; then
    echo "🔗 Running integration tests..."
    poetry run pytest tests/integration_tests/ -v --tb=short --integration
fi

# Coverage (if coverage=true)
if [[ "$coverage" == "true" ]]; then
    echo "📊 Generating coverage report..."
    # Convert package name hyphens to underscores for Python module name
    module_name=$(echo "langchain_${package_name}" | tr '-' '_')
    poetry run pytest tests/ --cov="$module_name" --cov-report=html --cov-report=term
fi
```

### Step 3: Run Linting

```bash
echo "🔍 Running linter..."
poetry run ruff check .
```

### Step 4: Run Formatting Check

```bash
echo "💅 Checking code formatting..."
poetry run ruff format --check .
```

### Step 5: Run Type Checking

```bash
echo "🔬 Running type checker..."
# Convert package name hyphens to underscores for Python module name
module_name=$(echo "langchain_${package_name}" | tr '-' '_')
poetry run mypy "$module_name"/
```

## Output Summary

Provide a formatted summary:

```
🎯 Test Results Summary for {package}

Unit Tests: ✅ PASSED (45 tests, 0.8s)
Integration Tests: ✅ PASSED (12 tests, 3.2s) [or ⏭️ SKIPPED]
Coverage: 87% (target: >80%)
Linting: ✅ PASSED (0 errors, 0 warnings)
Formatting: ✅ PASSED
Type Checking: ✅ PASSED (0 errors)

Overall: ✅ ALL CHECKS PASSED

📊 Coverage Report: htmlcov/index.html

Next steps:
- Review any warnings
- Improve coverage if below 80%
- Commit changes if all passed
```

## Error Handling

If any check fails, provide:
1. Which check failed
2. First 20 lines of error output
3. Suggested fixes
4. Link to relevant documentation

Example:
```
❌ Unit Tests FAILED

Failed: tests/unit_tests/test_agents.py::test_agent_creation

Error:
AssertionError: Expected agent to be created with default model

Fix suggestions:
1. Check if AzureAIChatCompletionsModel is properly mocked
2. Verify test imports are correct
3. Review test_agents.py lines 45-52

Docs: .claude/CLAUDE.md#testing-requirements
```

## Success Criteria

- All enabled tests pass
- No linting errors
- Formatting check passes
- Type checking passes
- Coverage above 80% (if enabled)
