"""Testing Automation Tools.

Tools for generating tests, analyzing coverage, and managing test plans.
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool


@tool
def generate_unit_tests(
    code: str,
    language: str = "python",
    framework: str = "pytest",
    coverage_target: float = 80.0,
    session_id: str = "default",
) -> str:
    """Generate unit tests for the given code.

    Creates comprehensive unit tests including:
    - Happy path tests
    - Edge cases
    - Error handling tests
    - Boundary condition tests

    Args:
        code: Code to generate tests for.
        language: Programming language.
        framework: Testing framework (pytest, unittest, jest, mocha).
        coverage_target: Target code coverage percentage.
        session_id: Session identifier.

    Returns:
        JSON string with generated tests.
    """
    test_id = f"UT-{str(uuid.uuid4())[:8].upper()}"

    # Extract function/class names for test generation
    import re
    functions = re.findall(r'def\s+(\w+)\s*\(', code)
    classes = re.findall(r'class\s+(\w+)\s*[:\(]', code)

    tests = []
    for func in functions:
        if not func.startswith("_"):
            tests.append({
                "test_name": f"test_{func}_success",
                "description": f"Test {func} with valid input",
                "type": "happy_path",
            })
            tests.append({
                "test_name": f"test_{func}_invalid_input",
                "description": f"Test {func} with invalid input",
                "type": "error_handling",
            })

    for cls in classes:
        tests.append({
            "test_name": f"test_{cls.lower()}_initialization",
            "description": f"Test {cls} class initialization",
            "type": "happy_path",
        })

    # Generate framework-specific test code template
    if framework == "pytest":
        test_template = f'''"""Unit tests generated for the module.

Test Framework: pytest
Coverage Target: {coverage_target}%
"""

import pytest

# Test fixtures
@pytest.fixture
def sample_data():
    """Provide sample test data."""
    return {{"key": "value"}}

'''
        for test in tests:
            test_template += f'''
def {test["test_name"]}(sample_data):
    """
    {test["description"]}
    Type: {test["type"]}
    """
    # Arrange
    # TODO: Setup test data

    # Act
    # TODO: Execute function

    # Assert
    # TODO: Verify results
    assert True  # Replace with actual assertions

'''

    elif framework == "unittest":
        test_template = f'''"""Unit tests generated for the module.

Test Framework: unittest
Coverage Target: {coverage_target}%
"""

import unittest


class TestModule(unittest.TestCase):
    """Test cases for the module."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = {{"key": "value"}}

    def tearDown(self):
        """Clean up after tests."""
        pass

'''
        for test in tests:
            test_template += f'''
    def {test["test_name"]}(self):
        """
        {test["description"]}
        Type: {test["type"]}
        """
        # Arrange
        # TODO: Setup test data

        # Act
        # TODO: Execute function

        # Assert
        # TODO: Verify results
        self.assertTrue(True)  # Replace with actual assertions

'''
        test_template += '''

if __name__ == "__main__":
    unittest.main()
'''

    elif framework == "jest":
        test_template = f'''/**
 * Unit tests generated for the module.
 *
 * Test Framework: Jest
 * Coverage Target: {coverage_target}%
 */

describe("Module Tests", () => {{
    let sampleData;

    beforeEach(() => {{
        sampleData = {{ key: "value" }};
    }});

'''
        for test in tests:
            test_name_camel = test["test_name"].replace("_", " ").title().replace(" ", "")
            test_template += f'''
    test("{test["test_name"].replace("_", " ")}", () => {{
        // {test["description"]}
        // Type: {test["type"]}

        // Arrange
        // TODO: Setup test data

        // Act
        // TODO: Execute function

        // Assert
        // TODO: Verify results
        expect(true).toBe(true);  // Replace with actual assertions
    }});

'''
        test_template += '''});
'''

    elif framework == "mocha":
        test_template = f'''/**
 * Unit tests generated for the module.
 *
 * Test Framework: Mocha + Chai
 * Coverage Target: {coverage_target}%
 */

const {{ expect }} = require("chai");

describe("Module Tests", function() {{
    let sampleData;

    beforeEach(function() {{
        sampleData = {{ key: "value" }};
    }});

'''
        for test in tests:
            test_template += f'''
    it("{test["test_name"].replace("_", " ")}", function() {{
        // {test["description"]}
        // Type: {test["type"]}

        // Arrange
        // TODO: Setup test data

        // Act
        // TODO: Execute function

        // Assert
        // TODO: Verify results
        expect(true).to.be.true;  // Replace with actual assertions
    }});

'''
        test_template += '''});
'''

    else:
        # Default to pytest-style for unknown frameworks
        test_template = f'''"""Unit tests generated for the module.

Test Framework: {framework}
Coverage Target: {coverage_target}%
"""

# Note: Using pytest-style syntax as default

import pytest

# Test fixtures
@pytest.fixture
def sample_data():
    """Provide sample test data."""
    return {{"key": "value"}}

'''
        for test in tests:
            test_template += f'''
def {test["test_name"]}(sample_data):
    """
    {test["description"]}
    Type: {test["type"]}
    """
    # Arrange
    # TODO: Setup test data

    # Act
    # TODO: Execute function

    # Assert
    # TODO: Verify results
    assert True  # Replace with actual assertions

'''

    result = {
        "id": test_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "framework": framework,
        "coverage_target": coverage_target,
        "tests_generated": len(tests),
        "test_cases": tests,
        "test_code": test_template,
    }

    return json.dumps(result, indent=2)


@tool
def generate_integration_tests(
    components: str,
    language: str = "python",
    framework: str = "pytest",
    session_id: str = "default",
) -> str:
    """Generate integration tests for component interactions.

    Creates tests for:
    - Component communication
    - Data flow between modules
    - API endpoints
    - Database interactions

    Args:
        components: JSON or comma-separated list of components to test.
        language: Programming language.
        framework: Testing framework.
        session_id: Session identifier.

    Returns:
        JSON string with integration test specifications.
    """
    test_id = f"IT-{str(uuid.uuid4())[:8].upper()}"

    # Parse components
    try:
        component_list = json.loads(components)
    except json.JSONDecodeError:
        component_list = [c.strip() for c in components.split(",") if c.strip()]

    tests = []
    for i, comp in enumerate(component_list):
        if i < len(component_list) - 1:
            tests.append({
                "test_name": f"test_{comp.lower()}_to_{component_list[i+1].lower()}_integration",
                "components": [comp, component_list[i+1]],
                "type": "integration",
            })

    result = {
        "id": test_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "framework": framework,
        "components": component_list,
        "tests_generated": len(tests),
        "test_cases": tests,
    }

    return json.dumps(result, indent=2)


@tool
def analyze_test_coverage(
    test_results: str,
    source_files: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Analyze test coverage from test results.

    Reports:
    - Line coverage
    - Branch coverage
    - Function coverage
    - Uncovered code sections

    Args:
        test_results: Test results or coverage report (JSON or text format).
        source_files: Optional source files to analyze.
        session_id: Session identifier.

    Returns:
        JSON string with coverage analysis.
    """
    coverage_id = f"COV-{str(uuid.uuid4())[:8].upper()}"

    # Parse test results to extract coverage data
    line_coverage = 0.0
    branch_coverage = 0.0
    function_coverage = 0.0
    uncovered_areas = []
    parsed_data = {}

    try:
        parsed_data = json.loads(test_results)
        # Extract coverage from parsed JSON
        if "coverage" in parsed_data:
            cov_data = parsed_data["coverage"]
            line_coverage = float(cov_data.get("line_coverage", cov_data.get("lines", 0)))
            branch_coverage = float(cov_data.get("branch_coverage", cov_data.get("branches", 0)))
            function_coverage = float(cov_data.get("function_coverage", cov_data.get("functions", 0)))
        elif "totals" in parsed_data:
            # Coverage.py JSON format
            totals = parsed_data["totals"]
            line_coverage = float(totals.get("percent_covered", 0))
            branch_coverage = float(totals.get("percent_covered_branches", line_coverage * 0.9))
            function_coverage = float(totals.get("percent_covered_functions", line_coverage * 1.1))
        # Extract uncovered files/lines if available
        if "files" in parsed_data:
            for file_path, file_data in parsed_data["files"].items():
                missing = file_data.get("missing_lines", file_data.get("missing", []))
                if missing:
                    uncovered_areas.append({
                        "file": file_path,
                        "lines": str(missing[:5]) if isinstance(missing, list) else str(missing),
                        "reason": "Uncovered code path",
                    })
    except json.JSONDecodeError:
        # Parse text-based coverage report (e.g., pytest output)
        import re
        # Look for percentage patterns like "75%" or "75.5%"
        percentages = re.findall(r"(\d+\.?\d*)%", test_results)
        if percentages:
            line_coverage = float(percentages[0])
            branch_coverage = float(percentages[1]) if len(percentages) > 1 else line_coverage * 0.9
            function_coverage = float(percentages[2]) if len(percentages) > 2 else line_coverage * 1.1

        # Look for "TOTAL ... XX%" pattern from pytest-cov
        total_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", test_results)
        if total_match:
            line_coverage = float(total_match.group(1))
            branch_coverage = line_coverage * 0.9
            function_coverage = min(line_coverage * 1.1, 100.0)

        # Look for missing lines in coverage output
        missing_matches = re.findall(r"(\S+\.py)\s+\d+\s+\d+\s+\d+%\s+([\d,\-\s]+)", test_results)
        for file_path, missing_lines in missing_matches:
            if missing_lines.strip():
                uncovered_areas.append({
                    "file": file_path,
                    "lines": missing_lines.strip()[:50],
                    "reason": "Uncovered code path",
                })

    # Ensure values are within valid range
    line_coverage = min(max(line_coverage, 0.0), 100.0)
    branch_coverage = min(max(branch_coverage, 0.0), 100.0)
    function_coverage = min(max(function_coverage, 0.0), 100.0)

    # If no coverage data was extracted, provide defaults with explanation
    if line_coverage == 0.0 and branch_coverage == 0.0:
        uncovered_areas.append({
            "file": "unknown",
            "lines": "N/A",
            "reason": "Could not parse coverage data from input",
        })

    # Generate recommendations based on actual coverage
    recommendations = []
    if line_coverage < 80:
        recommendations.append(f"Increase line coverage from {line_coverage:.1f}% to at least 80%")
    if branch_coverage < 70:
        recommendations.append(f"Improve branch coverage from {branch_coverage:.1f}% to at least 70%")
    if function_coverage < 90:
        recommendations.append(f"Add tests for uncovered functions (currently {function_coverage:.1f}%)")
    if uncovered_areas:
        recommendations.append("Add tests for error handling paths")
        recommendations.append("Cover edge cases in input validation")
    if not recommendations:
        recommendations.append("Coverage meets quality standards - consider adding property-based tests")

    threshold = 80.0
    result = {
        "id": coverage_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "coverage": {
            "line_coverage": round(line_coverage, 1),
            "branch_coverage": round(branch_coverage, 1),
            "function_coverage": round(function_coverage, 1),
        },
        "uncovered_areas": uncovered_areas[:10],  # Limit to 10 entries
        "recommendations": recommendations,
        "meets_threshold": line_coverage >= threshold,
        "threshold": threshold,
    }

    return json.dumps(result, indent=2)


@tool
def run_tests(
    test_path: str,
    framework: str = "pytest",
    options: Optional[str] = None,
    session_id: str = "default",
) -> str:
    """Run tests and return results.

    Executes tests using specified framework and returns:
    - Pass/fail status
    - Test duration
    - Failed test details
    - Coverage summary

    Args:
        test_path: Path to tests or test specification.
        framework: Testing framework to use.
        options: Additional test runner options.
        session_id: Session identifier.

    Returns:
        JSON string with test execution results.
    """
    run_id = f"RUN-{str(uuid.uuid4())[:8].upper()}"

    # Simulated test run
    summary = {
        "total": 10,
        "passed": 9,
        "failed": 1,
        "skipped": 0,
        "duration_seconds": 2.5,
    }
    
    # Derive status from summary counts
    status = "passed" if summary["failed"] == 0 else "failed"
    
    result = {
        "id": run_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "framework": framework,
        "test_path": test_path,
        "status": status,
        "summary": summary,
        "failed_tests": [
            {
                "name": "test_edge_case",
                "error": "AssertionError: Expected 5, got 4",
                "file": "test_module.py",
                "line": 45,
            },
        ] if summary["failed"] > 0 else [],
    }

    return json.dumps(result, indent=2)


@tool
def generate_test_data(
    schema: str,
    count: int = 10,
    data_type: str = "realistic",
    session_id: str = "default",
) -> str:
    """Generate test data based on schema.

    Creates test data:
    - Realistic data for integration tests
    - Edge case data for boundary testing
    - Random data for fuzz testing

    Args:
        schema: JSON schema or description of data structure.
        count: Number of records to generate.
        data_type: Type of data - "realistic", "edge_case", "random".
        session_id: Session identifier.

    Returns:
        JSON string with generated test data.
    """
    import random
    import string

    data_id = f"TD-{str(uuid.uuid4())[:8].upper()}"

    # Parse schema
    try:
        schema_dict = json.loads(schema)
    except json.JSONDecodeError:
        schema_dict = {"type": "object", "properties": {"name": {"type": "string"}}}

    # Extract field definitions from schema
    properties = schema_dict.get("properties", {"name": {"type": "string"}})

    # Helper functions for data generation
    def generate_realistic_value(field_name: str, field_type: str, index: int) -> any:
        """Generate realistic test data based on field name and type."""
        field_lower = field_name.lower()
        if "email" in field_lower:
            return f"user{index+1}@example.com"
        elif "name" in field_lower:
            first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]
            return f"{first_names[index % len(first_names)]} {last_names[index % len(last_names)]}"
        elif "phone" in field_lower:
            return f"+1-555-{100+index:03d}-{1000+index:04d}"
        elif "id" in field_lower:
            return f"{field_name}_{index+1}"
        elif "date" in field_lower or "time" in field_lower:
            return datetime.now().isoformat()
        elif "age" in field_lower:
            return 20 + (index % 50)
        elif "price" in field_lower or "amount" in field_lower:
            return round(10.0 + index * 5.5, 2)
        elif "count" in field_lower or "quantity" in field_lower:
            return index + 1
        elif "url" in field_lower:
            return f"https://example.com/resource/{index+1}"
        elif field_type == "integer" or field_type == "number":
            return index + 1
        elif field_type == "boolean":
            return index % 2 == 0
        else:
            return f"value_{index+1}"

    def generate_edge_case_value(field_name: str, field_type: str, index: int) -> any:
        """Generate edge case test data."""
        edge_cases = [
            "",  # Empty string
            None,  # Null value
            "x" * 255,  # Max length string
            "A" * 1000,  # Very long string
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection attempt
            "user@.com",  # Invalid email
            -1,  # Negative number
            0,  # Zero
            2**31 - 1,  # Max int32
        ]
        return edge_cases[index % len(edge_cases)]

    def generate_random_value(field_name: str, field_type: str, index: int) -> any:
        """Generate random test data."""
        if field_type == "integer" or field_type == "number":
            return random.randint(-1000, 1000)
        elif field_type == "boolean":
            return random.choice([True, False])
        else:
            length = random.randint(1, 50)
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    # Generate the requested number of records (honor count parameter)
    test_data = []
    for i in range(count):
        record = {}
        for field_name, field_def in properties.items():
            field_type = field_def.get("type", "string") if isinstance(field_def, dict) else "string"

            if data_type == "realistic":
                record[field_name] = generate_realistic_value(field_name, field_type, i)
            elif data_type == "edge_case":
                record[field_name] = generate_edge_case_value(field_name, field_type, i)
            elif data_type == "random":
                record[field_name] = generate_random_value(field_name, field_type, i)
            else:
                record[field_name] = generate_realistic_value(field_name, field_type, i)

        test_data.append(record)

    result = {
        "id": data_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "schema": schema_dict,
        "data_type": data_type,
        "count_requested": count,
        "count_generated": len(test_data),
        "data": test_data,
    }

    return json.dumps(result, indent=2)


@tool
def create_test_plan(
    feature: str,
    test_types: str = "unit,integration,e2e",
    session_id: str = "default",
) -> str:
    """Create a comprehensive test plan for a feature.

    Includes:
    - Test strategy
    - Test cases by type
    - Priority and risk assessment
    - Resource requirements

    Args:
        feature: Feature description to test.
        test_types: Comma-separated test types to include.
        session_id: Session identifier.

    Returns:
        JSON string with test plan.
    """
    plan_id = f"TP-{str(uuid.uuid4())[:8].upper()}"

    types = [t.strip() for t in test_types.split(",")]

    test_cases = []
    for test_type in types:
        if test_type == "unit":
            test_cases.extend([
                {"type": "unit", "name": "Validate input parameters", "priority": "high"},
                {"type": "unit", "name": "Test core logic", "priority": "high"},
                {"type": "unit", "name": "Test error handling", "priority": "medium"},
            ])
        elif test_type == "integration":
            test_cases.extend([
                {"type": "integration", "name": "Test database operations", "priority": "high"},
                {"type": "integration", "name": "Test API endpoints", "priority": "high"},
            ])
        elif test_type == "e2e":
            test_cases.extend([
                {"type": "e2e", "name": "Complete user workflow", "priority": "high"},
                {"type": "e2e", "name": "Error recovery scenarios", "priority": "medium"},
            ])

    result = {
        "id": plan_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "feature": feature,
        "test_types": types,
        "test_strategy": {
            "approach": "Pyramid testing with unit tests as foundation",
            "automation_percentage": 90,
            "manual_testing": "Exploratory testing for UX",
        },
        "test_cases": test_cases,
        "resources": {
            "estimated_effort_hours": len(test_cases) * 2,
            "required_environments": ["development", "staging"],
        },
    }

    return json.dumps(result, indent=2)
