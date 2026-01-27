"""Evaluation datasets for Azure AI Foundry agents.

This module contains test datasets for evaluating different agent types:
- IT Agents (Helpdesk, ServiceNow, HITL Support)
- Enterprise Agents (Research, Content, Data Analyst, Code Assistant)
- DeepAgents (IT Operations, Sales Intelligence, Recruitment)
"""

from typing import Any, Dict, List

# =============================================================================
# IT AGENTS DATASETS
# =============================================================================

IT_HELPDESK_DATASET: List[Dict[str, Any]] = [
    {
        "id": "it-helpdesk-001",
        "input": "How do I reset my password?",
        "expected_output": "To reset your password, go to Settings, then Security, and click on 'Reset Password'. You'll receive an email with instructions.",
        "expected_keywords": ["Settings", "Security", "Reset Password", "email"],
        "tags": ["password", "authentication", "basic"],
        "difficulty": "easy",
        "context": "User needs password reset assistance",
    },
    {
        "id": "it-helpdesk-002",
        "input": "My laptop won't connect to the VPN. What should I do?",
        "expected_output": "Check if your VPN client is up to date, verify your credentials, and ensure you have internet connectivity. If the issue persists, try restarting the VPN service.",
        "expected_keywords": ["VPN client", "credentials", "internet", "restart"],
        "tags": ["vpn", "connectivity", "troubleshooting"],
        "difficulty": "medium",
        "context": "VPN connection troubleshooting",
    },
    {
        "id": "it-helpdesk-003",
        "input": "I'm getting an 'access denied' error when trying to open a shared folder.",
        "expected_output": "This indicates a permissions issue. I'll need to verify your access rights to that folder. Please provide the folder path and I'll check with the IT admin team.",
        "expected_keywords": ["permissions", "access rights", "folder path", "admin"],
        "tags": ["permissions", "file sharing", "access control"],
        "difficulty": "medium",
        "context": "File access permission issue",
    },
]

SERVICENOW_DATASET: List[Dict[str, Any]] = [
    {
        "id": "servicenow-001",
        "input": "Create an incident ticket for printer not working in Building A, Floor 3.",
        "expected_output": "I'll create an incident ticket for you. The ticket has been created with priority P3 for the printer issue in Building A, Floor 3.",
        "expected_keywords": ["incident", "ticket", "created", "priority"],
        "tags": ["incident", "creation", "printer"],
        "difficulty": "easy",
        "context": "Create incident in ServiceNow",
    },
]

# =============================================================================
# ENTERPRISE AGENTS DATASETS
# =============================================================================

RESEARCH_AGENT_DATASET: List[Dict[str, Any]] = [
    {
        "id": "research-001",
        "input": "What are the latest trends in artificial intelligence for 2026?",
        "expected_output": "Based on recent research, key AI trends for 2026 include: multimodal AI systems, improved reasoning capabilities, edge AI deployment, and enhanced AI safety measures.",
        "expected_keywords": ["AI trends", "2026", "multimodal", "reasoning", "safety"],
        "tags": ["research", "ai", "trends"],
        "difficulty": "medium",
        "context": "Research latest AI trends",
    },
]

CODE_ASSISTANT_DATASET: List[Dict[str, Any]] = [
    {
        "id": "code-001",
        "input": "Write a Python function to validate email addresses.",
        "expected_output": "Here's a Python function using regex to validate email addresses, including proper error handling and documentation.",
        "expected_keywords": ["Python", "function", "email", "validation", "regex"],
        "tags": ["code", "python", "validation"],
        "difficulty": "medium",
        "context": "Write Python validation function",
    },
]

# =============================================================================
# DEEPAGENTS DATASETS
# =============================================================================

IT_OPERATIONS_DATASET: List[Dict[str, Any]] = [
    {
        "id": "it-ops-001",
        "input": "Analyze incident INC0012345 and recommend a solution.",
        "expected_output": "Incident INC0012345 appears to be a network connectivity issue. Recommended solution: restart the network switch and verify VLAN configuration.",
        "expected_keywords": ["incident", "analysis", "network", "solution", "VLAN"],
        "tags": ["it-ops", "incident", "analysis"],
        "difficulty": "hard",
        "context": "Incident analysis and resolution",
    },
    {
        "id": "it-ops-002",
        "input": "What's the current SLA compliance for our critical services?",
        "expected_output": "Current SLA compliance: Critical services are at 99.8% uptime. Two services are below target: Email (99.2%) and File Server (98.9%).",
        "expected_keywords": ["SLA", "compliance", "uptime", "critical services"],
        "tags": ["it-ops", "sla", "monitoring"],
        "difficulty": "medium",
        "context": "SLA monitoring",
    },
    {
        "id": "it-ops-003",
        "input": "A user reports slow network performance. Diagnose and recommend actions.",
        "expected_output": "Network diagnostics needed: Check bandwidth utilization, verify router configuration, test latency to key services. If utilization >80%, consider upgrading capacity or implementing QoS policies.",
        "expected_keywords": ["network", "bandwidth", "latency", "QoS", "diagnostics"],
        "tags": ["it-ops", "performance", "network", "troubleshooting"],
        "difficulty": "hard",
        "context": "Network performance troubleshooting",
    },
]

SALES_INTELLIGENCE_DATASET: List[Dict[str, Any]] = [
    {
        "id": "sales-001",
        "input": "Qualify this lead: Tech startup with 50 employees, budget $100K, needs CRM solution.",
        "expected_output": "Lead qualification (BANT): Budget confirmed ($100K), Authority needed (CEO contact), Need identified (CRM), Timeline not specified. Recommended next step: Schedule discovery call with decision maker.",
        "expected_keywords": ["BANT", "qualified", "budget", "CRM", "discovery call"],
        "tags": ["sales", "qualification", "lead"],
        "difficulty": "medium",
        "context": "Lead qualification",
    },
    {
        "id": "sales-002",
        "input": "Analyze sales pipeline for Q1 and identify at-risk deals.",
        "expected_output": "Q1 pipeline analysis: 15 active opportunities totaling $2.5M. At-risk deals: 3 opportunities ($450K) showing no activity for 14+ days. Recommended actions: immediate follow-up with account managers.",
        "expected_keywords": ["pipeline", "opportunities", "at-risk", "follow-up"],
        "tags": ["sales", "pipeline", "analysis"],
        "difficulty": "hard",
        "context": "Sales pipeline analysis",
    },
]

RECRUITMENT_DATASET: List[Dict[str, Any]] = [
    {
        "id": "recruitment-001",
        "input": "Screen this resume for a Senior Software Engineer position: 7 years Python/Java, cloud experience, open-source contributor.",
        "expected_output": "Candidate shows strong match: 7 years relevant experience, proficient in required languages (Python/Java), cloud technologies demonstrated, open-source contributions show initiative. Recommendation: Move to technical interview stage.",
        "expected_keywords": ["experience", "Python", "Java", "cloud", "technical interview"],
        "tags": ["recruitment", "screening", "resume"],
        "difficulty": "medium",
        "context": "Resume screening",
    },
    {
        "id": "recruitment-002",
        "input": "Create an interview plan for a DevOps Engineer role focusing on CI/CD and cloud infrastructure.",
        "expected_output": "Interview plan: Round 1 - Technical screening (30 min): CI/CD pipelines, Docker/K8s basics. Round 2 - Deep technical (60 min): Cloud architecture design, infrastructure as code. Round 3 - System design (45 min): Scalability, monitoring, incident response.",
        "expected_keywords": ["interview", "CI/CD", "cloud", "technical screening", "system design"],
        "tags": ["recruitment", "interview", "planning"],
        "difficulty": "hard",
        "context": "Interview planning",
    },
]

# =============================================================================
# DATASET REGISTRY
# =============================================================================

AGENT_DATASETS: Dict[str, List[Dict[str, Any]]] = {
    # IT Agents
    "it-helpdesk": IT_HELPDESK_DATASET,
    "servicenow": SERVICENOW_DATASET,
    "hitl_support": IT_HELPDESK_DATASET[:2],  # Reuse subset for HITL

    # Enterprise Agents
    "research": RESEARCH_AGENT_DATASET,
    "code_assistant": CODE_ASSISTANT_DATASET,

    # DeepAgents
    "it_operations": IT_OPERATIONS_DATASET,
    "sales_intelligence": SALES_INTELLIGENCE_DATASET,
    "recruitment": RECRUITMENT_DATASET,
}


def get_dataset(agent_name: str) -> List[Dict[str, Any]]:
    """Get evaluation dataset for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        List of test cases
    """
    return AGENT_DATASETS.get(agent_name, [])


def get_all_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """Get all evaluation datasets.

    Returns:
        Dictionary mapping agent names to datasets
    """
    return AGENT_DATASETS


def get_dataset_summary() -> Dict[str, Dict[str, Any]]:
    """Get summary statistics for all datasets.

    Returns:
        Dictionary with dataset statistics
    """
    summary = {}
    for agent_name, dataset in AGENT_DATASETS.items():
        summary[agent_name] = {
            "total_cases": len(dataset),
            "difficulties": {
                "easy": sum(1 for case in dataset if case.get("difficulty") == "easy"),
                "medium": sum(1 for case in dataset if case.get("difficulty") == "medium"),
                "hard": sum(1 for case in dataset if case.get("difficulty") == "hard"),
            },
            "tags": list(set(tag for case in dataset for tag in case.get("tags", []))),
        }
    return summary
