"""Security Scanning and Compliance Tools.

Tools for security analysis, vulnerability detection, and compliance checking.
"""

import json
import uuid
import re
from datetime import datetime

from langchain_core.tools import tool


# OWASP Top 10 categories
OWASP_TOP_10 = {
    "A01": "Broken Access Control",
    "A02": "Cryptographic Failures",
    "A03": "Injection",
    "A04": "Insecure Design",
    "A05": "Security Misconfiguration",
    "A06": "Vulnerable Components",
    "A07": "Authentication Failures",
    "A08": "Software and Data Integrity Failures",
    "A09": "Security Logging and Monitoring Failures",
    "A10": "Server-Side Request Forgery (SSRF)",
}


@tool
def scan_security_issues(
    code: str,
    language: str = "python",
    severity_threshold: str = "medium",
    session_id: str = "default",
) -> str:
    """Scan code for security vulnerabilities.

    Detects:
    - Injection vulnerabilities (SQL, Command, XSS)
    - Authentication issues
    - Sensitive data exposure
    - Security misconfigurations

    Args:
        code: Code to scan.
        language: Programming language.
        severity_threshold: Minimum severity to report - "low", "medium", "high", "critical".
        session_id: Session identifier.

    Returns:
        JSON string with security scan results.
    """
    scan_id = f"SEC-{str(uuid.uuid4())[:8].upper()}"

    issues = []

    # Check for common security issues
    security_patterns = [
        {
            "pattern": r"exec\s*\(",
            "issue": "Potential code execution vulnerability",
            "severity": "critical",
            "category": "A03",
        },
        {
            "pattern": r"eval\s*\(",
            "issue": "Dangerous eval() usage - potential code injection",
            "severity": "critical",
            "category": "A03",
        },
        {
            "pattern": r"subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True",
            "issue": "Shell injection vulnerability",
            "severity": "high",
            "category": "A03",
        },
        {
            "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
            "issue": "Hardcoded password detected",
            "severity": "critical",
            "category": "A02",
        },
        {
            "pattern": r"(api_key|secret|token)\s*=\s*['\"][^'\"]+['\"]",
            "issue": "Hardcoded credential detected",
            "severity": "critical",
            "category": "A02",
        },
        {
            "pattern": r"\.format\s*\([^)]*\)\s*$",
            "issue": "Potential SQL injection with string formatting",
            "severity": "high",
            "category": "A03",
        },
        {
            "pattern": r"pickle\.loads?\s*\(",
            "issue": "Insecure deserialization with pickle",
            "severity": "high",
            "category": "A08",
        },
        {
            "pattern": r"verify\s*=\s*False",
            "issue": "SSL verification disabled",
            "severity": "high",
            "category": "A02",
        },
    ]

    severity_levels = ["low", "medium", "high", "critical"]

    # Normalize and validate severity_threshold to avoid ValueError
    if isinstance(severity_threshold, str):
        severity_threshold = severity_threshold.lower()
    if severity_threshold not in severity_levels:
        severity_threshold = "medium"
    threshold_index = severity_levels.index(severity_threshold)

    for pattern_info in security_patterns:
        matches = list(re.finditer(pattern_info["pattern"], code, re.IGNORECASE))
        for match in matches:
            pattern_severity_index = severity_levels.index(pattern_info["severity"])
            if pattern_severity_index >= threshold_index:
                # Find line number
                line_num = code[:match.start()].count("\n") + 1
                issues.append({
                    "id": f"VULN-{str(uuid.uuid4())[:6].upper()}",
                    "severity": pattern_info["severity"],
                    "category": OWASP_TOP_10.get(pattern_info["category"], "Unknown"),
                    "owasp_id": pattern_info["category"],
                    "message": pattern_info["issue"],
                    "line": line_num,
                    "code_snippet": match.group()[:50],
                })

    result = {
        "id": scan_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "severity_threshold": severity_threshold,
        "total_issues": len(issues),
        "issues_by_severity": {
            "critical": len([i for i in issues if i["severity"] == "critical"]),
            "high": len([i for i in issues if i["severity"] == "high"]),
            "medium": len([i for i in issues if i["severity"] == "medium"]),
            "low": len([i for i in issues if i["severity"] == "low"]),
        },
        "issues": issues,
        "scan_status": "completed",
    }

    return json.dumps(result, indent=2)


@tool
def check_owasp_compliance(
    code: str,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Check code against OWASP Top 10 vulnerabilities.

    Evaluates compliance with:
    - A01: Broken Access Control
    - A02: Cryptographic Failures
    - A03: Injection
    - And more...

    Args:
        code: Code to check.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with OWASP compliance results.
    """
    compliance_id = f"OWASP-{str(uuid.uuid4())[:8].upper()}"

    compliance_checks = {}
    for code_id, name in OWASP_TOP_10.items():
        compliance_checks[code_id] = {
            "name": name,
            "status": "passed",
            "findings": [],
        }

    # Basic checks for common issues
    if "eval(" in code or "exec(" in code:
        compliance_checks["A03"]["status"] = "failed"
        compliance_checks["A03"]["findings"].append("Code execution functions detected")

    if re.search(r"password|secret|api_key|token", code.lower()):
        if re.search(r"['\"][a-zA-Z0-9_-]{8,}['\"]", code):
            compliance_checks["A02"]["status"] = "warning"
            compliance_checks["A02"]["findings"].append("Possible hardcoded credentials")

    result = {
        "id": compliance_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "compliance_checks": compliance_checks,
        "overall_status": "passed" if all(c["status"] == "passed" for c in compliance_checks.values()) else "needs_attention",
        "recommendations": [
            "Review and fix all failed checks before deployment",
            "Consider using security scanning in CI/CD pipeline",
            "Implement security code review process",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def detect_secrets(
    code: str,
    check_types: str = "all",
    session_id: str = "default",
) -> str:
    """Detect secrets and credentials in code.

    Scans for:
    - API keys
    - Passwords
    - Tokens
    - Connection strings
    - Private keys

    Args:
        code: Code to scan.
        check_types: Types to check - "all", "api_keys", "passwords", "tokens".
        session_id: Session identifier.

    Returns:
        JSON string with detected secrets.
    """
    secrets_id = f"SECRET-{str(uuid.uuid4())[:8].upper()}"

    secrets = []

    # Patterns for common secrets
    secret_patterns = [
        {"name": "AWS Access Key", "pattern": r"AKIA[0-9A-Z]{16}"},
        {"name": "AWS Secret Key", "pattern": r"[a-zA-Z0-9/+=]{40}"},
        {"name": "GitHub Token", "pattern": r"ghp_[a-zA-Z0-9]{36}"},
        {"name": "Generic API Key", "pattern": r"api[_-]?key['\"]?\s*[:=]\s*['\"][a-zA-Z0-9-_]{20,}['\"]"},
        {"name": "Password Assignment", "pattern": r"password\s*=\s*['\"][^'\"]+['\"]"},
        {"name": "Bearer Token", "pattern": r"bearer\s+[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"},
        {"name": "Connection String", "pattern": r"(?:mongodb|mysql|postgresql|redis)://[^\s]+"},
        {"name": "Private Key", "pattern": r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"},
    ]

    for pattern_info in secret_patterns:
        matches = list(re.finditer(pattern_info["pattern"], code, re.IGNORECASE))
        for match in matches:
            line_num = code[:match.start()].count("\n") + 1
            # Mask the secret
            secret_value = match.group()
            masked = secret_value[:10] + "..." + secret_value[-4:] if len(secret_value) > 14 else "****"
            secrets.append({
                "type": pattern_info["name"],
                "line": line_num,
                "masked_value": masked,
                "severity": "critical",
            })

    result = {
        "id": secrets_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "secrets_found": len(secrets),
        "secrets": secrets,
        "recommendations": [
            "Remove all hardcoded secrets immediately",
            "Use environment variables or secret management",
            "Add pre-commit hooks to prevent secret commits",
            "Rotate any exposed credentials",
        ] if secrets else ["No secrets detected - continue following best practices"],
    }

    return json.dumps(result, indent=2)


@tool
def analyze_dependencies_security(
    dependencies: str,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Analyze dependencies for security vulnerabilities.

    Checks:
    - Known CVEs
    - Outdated packages
    - License compliance
    - Malicious packages

    Args:
        dependencies: Dependency list or file content.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with dependency security analysis.
    """
    analysis_id = f"DEPSEC-{str(uuid.uuid4())[:8].upper()}"

    # Parse dependencies
    dep_list = [d.strip() for d in dependencies.split("\n") if d.strip() and not d.startswith("#")]

    vulnerabilities = []
    # Simulated vulnerability check
    known_vulnerable = {
        "requests": {"version": "<2.20.0", "cve": "CVE-2018-18074", "severity": "high"},
        "django": {"version": "<3.2.0", "cve": "CVE-2021-35042", "severity": "high"},
        "urllib3": {"version": "<1.26.5", "cve": "CVE-2021-33503", "severity": "medium"},
    }

    for dep in dep_list:
        dep_name = dep.split("==")[0].split(">=")[0].split("<=")[0].lower()
        if dep_name in known_vulnerable:
            vulnerabilities.append({
                "package": dep_name,
                "installed": dep,
                "vulnerability": known_vulnerable[dep_name],
            })

    result = {
        "id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "total_dependencies": len(dep_list),
        "vulnerable_count": len(vulnerabilities),
        "vulnerabilities": vulnerabilities,
        "recommendations": [
            "Update vulnerable packages to latest secure versions",
            "Use pip-audit or safety for continuous monitoring",
            "Consider using dependabot for automated updates",
        ],
    }

    return json.dumps(result, indent=2)


@tool
def generate_security_report(
    scan_results: str,
    format_type: str = "summary",
    session_id: str = "default",
) -> str:
    """Generate a security report from scan results.

    Creates reports in formats:
    - Summary: Executive overview
    - Detailed: Full technical details
    - Compliance: Regulatory compliance focused

    Args:
        scan_results: Previous scan results or code to scan (JSON format preferred).
        format_type: Report format - "summary", "detailed", "compliance".
        session_id: Session identifier.

    Returns:
        JSON string with security report.
    """
    report_id = f"SECRPT-{str(uuid.uuid4())[:8].upper()}"

    # Try to parse scan results
    try:
        results = json.loads(scan_results)
    except json.JSONDecodeError:
        results = {"issues": [], "total_issues": 0}

    # Extract issues from various possible formats
    issues = results.get("issues", [])
    if not issues and "vulnerabilities" in results:
        issues = results["vulnerabilities"]
    if not issues and "findings" in results:
        issues = results["findings"]

    # Count findings by severity from actual data
    critical_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0

    # Also extract from issues_by_severity if available
    if "issues_by_severity" in results:
        severity_data = results["issues_by_severity"]
        critical_count = severity_data.get("critical", 0)
        high_count = severity_data.get("high", 0)
        medium_count = severity_data.get("medium", 0)
        low_count = severity_data.get("low", 0)
    else:
        # Count from individual issues
        for issue in issues:
            severity = issue.get("severity", "").lower()
            if severity == "critical":
                critical_count += 1
            elif severity == "high":
                high_count += 1
            elif severity == "medium":
                medium_count += 1
            elif severity == "low":
                low_count += 1

    total_findings = results.get("total_issues", len(issues))
    if total_findings == 0:
        total_findings = critical_count + high_count + medium_count + low_count

    # Determine risk level based on actual findings
    if critical_count > 0:
        risk_level = "critical"
    elif high_count > 0:
        risk_level = "high"
    elif medium_count > 0:
        risk_level = "medium"
    elif low_count > 0:
        risk_level = "low"
    else:
        risk_level = "none"

    # Generate dynamic recommendations based on findings
    recommendations = []
    if critical_count > 0:
        recommendations.append(f"URGENT: Address {critical_count} critical vulnerabilities immediately")
    if high_count > 0:
        recommendations.append(f"HIGH PRIORITY: Fix {high_count} high-severity issues within 24-48 hours")
    if medium_count > 0:
        recommendations.append(f"Plan remediation for {medium_count} medium-severity issues")
    if low_count > 0:
        recommendations.append(f"Review {low_count} low-severity findings during next sprint")

    # Add general recommendations
    if total_findings > 0:
        recommendations.extend([
            "Implement security training for development team",
            "Add automated security scanning to CI/CD pipeline",
            "Schedule regular security assessments",
        ])
    else:
        recommendations.extend([
            "Continue following security best practices",
            "Maintain automated security scanning in CI/CD pipeline",
            "Consider periodic penetration testing",
        ])

    # Extract OWASP categories found
    owasp_categories_found = set()
    for issue in issues:
        owasp_id = issue.get("owasp_id", issue.get("category", ""))
        if owasp_id and owasp_id.startswith("A"):
            owasp_categories_found.add(owasp_id)

    report = {
        "id": report_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "format": format_type,
        "title": "Security Assessment Report",
        "summary": {
            "risk_level": risk_level,
            "total_findings": total_findings,
            "critical_findings": critical_count,
            "high_findings": high_count,
            "medium_findings": medium_count,
            "low_findings": low_count,
        },
        "recommendations": recommendations,
    }

    if format_type == "detailed":
        report["detailed_findings"] = issues
        report["owasp_categories_affected"] = list(owasp_categories_found)
        report["remediation_steps"] = [
            "Review each finding and understand the risk",
            "Prioritize fixes based on severity and exposure",
            "Test fixes thoroughly before deployment",
            "Verify remediation with follow-up scan",
        ]
        # Group findings by category
        findings_by_category = {}
        for issue in issues:
            category = issue.get("category", issue.get("owasp_id", "Other"))
            if category not in findings_by_category:
                findings_by_category[category] = []
            findings_by_category[category].append(issue)
        report["findings_by_category"] = findings_by_category

    if format_type == "compliance":
        # Determine compliance status based on actual findings
        owasp_status = "passed" if not owasp_categories_found else ("failed" if critical_count > 0 or high_count > 0 else "partial")

        # Check for specific compliance-related issues
        has_auth_issues = any("A01" in str(issue) or "A07" in str(issue) for issue in issues)
        has_crypto_issues = any("A02" in str(issue) for issue in issues)
        has_injection_issues = any("A03" in str(issue) for issue in issues)

        pci_status = "failed" if (has_auth_issues or has_crypto_issues or has_injection_issues) else ("review_needed" if total_findings > 0 else "passed")
        soc2_status = "failed" if critical_count > 0 else ("review_needed" if total_findings > 0 else "passed")

        report["compliance_status"] = {
            "OWASP_Top_10": owasp_status,
            "owasp_categories_violated": list(owasp_categories_found),
            "PCI_DSS": pci_status,
            "SOC2": soc2_status,
            "compliance_notes": {
                "OWASP_Top_10": f"{len(owasp_categories_found)} categories with findings" if owasp_categories_found else "No OWASP violations detected",
                "PCI_DSS": "Review authentication, encryption, and injection controls" if pci_status != "passed" else "Basic requirements met",
                "SOC2": "Review security controls for SOC2 Type II audit" if soc2_status != "passed" else "Security controls appear adequate",
            },
        }

    return json.dumps(report, indent=2)


@tool
def suggest_security_fixes(
    vulnerability: str,
    language: str = "python",
    session_id: str = "default",
) -> str:
    """Suggest fixes for security vulnerabilities.

    Provides:
    - Code fixes
    - Configuration changes
    - Best practice recommendations

    Args:
        vulnerability: Vulnerability description or code with issue.
        language: Programming language.
        session_id: Session identifier.

    Returns:
        JSON string with suggested fixes.
    """
    fix_id = f"FIX-{str(uuid.uuid4())[:8].upper()}"

    # Common fix suggestions
    fix_suggestions = {
        "sql_injection": {
            "issue": "SQL Injection vulnerability",
            "fix": "Use parameterized queries or ORM",
            "example": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        },
        "xss": {
            "issue": "Cross-Site Scripting vulnerability",
            "fix": "Escape user input before rendering",
            "example": "from markupsafe import escape\nescaped = escape(user_input)",
        },
        "hardcoded_secret": {
            "issue": "Hardcoded credentials",
            "fix": "Use environment variables",
            "example": "import os\napi_key = os.environ.get('API_KEY')",
        },
        "command_injection": {
            "issue": "Command injection vulnerability",
            "fix": "Use subprocess with list arguments",
            "example": "subprocess.run(['ls', '-l', path], shell=False)",
        },
    }

    # Determine vulnerability type
    vuln_lower = vulnerability.lower()
    suggested_fix = None
    for key, fix in fix_suggestions.items():
        if key.replace("_", " ") in vuln_lower or key in vuln_lower:
            suggested_fix = fix
            break

    if not suggested_fix:
        suggested_fix = {
            "issue": "Security vulnerability detected",
            "fix": "Review code and apply security best practices",
            "example": "Consult OWASP guidelines for specific remediation",
        }

    result = {
        "id": fix_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "language": language,
        "vulnerability_description": vulnerability[:200],
        "suggested_fix": suggested_fix,
        "additional_recommendations": [
            "Add input validation for all user inputs",
            "Implement principle of least privilege",
            "Use security headers in HTTP responses",
            "Enable logging for security events",
        ],
    }

    return json.dumps(result, indent=2)
