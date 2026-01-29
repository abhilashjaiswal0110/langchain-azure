"""Software Development DeepAgent Tools.

This module exports all tools for the Software Development DeepAgent,
organized by SDLC phase:

- Requirements: analyze_requirements, extract_user_stories, validate_requirements
- Architecture: design_architecture, create_api_spec, suggest_tech_stack
- Code Generation: generate_code, refactor_code, apply_design_pattern
- Code Review: review_code, check_code_style, analyze_complexity
- Testing: generate_unit_tests, generate_integration_tests, analyze_test_coverage
- Security: scan_security_issues, check_owasp_compliance, detect_secrets
- DevOps: create_ci_pipeline, create_cd_pipeline, generate_dockerfile
- Debugging: analyze_error, trace_execution, identify_root_cause
- Documentation: generate_api_docs, create_readme, document_architecture
"""

from langchain_azure_ai.wrappers.software_dev_tools.requirements_tools import (
    analyze_requirements,
    extract_user_stories,
    validate_requirements,
    prioritize_requirements,
    detect_ambiguities,
    generate_acceptance_criteria,
)

from langchain_azure_ai.wrappers.software_dev_tools.architecture_tools import (
    design_architecture,
    create_api_spec,
    suggest_tech_stack,
    design_data_model,
    create_component_diagram,
    analyze_dependencies,
)

from langchain_azure_ai.wrappers.software_dev_tools.codegen_tools import (
    generate_code,
    refactor_code,
    apply_design_pattern,
    generate_boilerplate,
    optimize_imports,
    format_code,
)

from langchain_azure_ai.wrappers.software_dev_tools.review_tools import (
    review_code,
    check_code_style,
    analyze_complexity,
    detect_code_smells,
    suggest_improvements,
    check_best_practices,
)

from langchain_azure_ai.wrappers.software_dev_tools.testing_tools import (
    generate_unit_tests,
    generate_integration_tests,
    analyze_test_coverage,
    run_tests,
    generate_test_data,
    create_test_plan,
)

from langchain_azure_ai.wrappers.software_dev_tools.security_tools import (
    scan_security_issues,
    check_owasp_compliance,
    detect_secrets,
    analyze_dependencies_security,
    generate_security_report,
    suggest_security_fixes,
)

from langchain_azure_ai.wrappers.software_dev_tools.devops_tools import (
    create_ci_pipeline,
    create_cd_pipeline,
    configure_deployment,
    generate_dockerfile,
    create_kubernetes_config,
    setup_monitoring,
)

from langchain_azure_ai.wrappers.software_dev_tools.debugging_tools import (
    analyze_error,
    trace_execution,
    identify_root_cause,
    propose_fix,
    analyze_performance,
    detect_memory_issues,
)

from langchain_azure_ai.wrappers.software_dev_tools.documentation_tools import (
    generate_api_docs,
    create_readme,
    document_architecture,
    generate_changelog,
    add_inline_comments,
    create_user_guide,
)


def get_all_software_dev_tools():
    """Get all available software development tools."""
    return [
        # Requirements
        analyze_requirements,
        extract_user_stories,
        validate_requirements,
        prioritize_requirements,
        detect_ambiguities,
        generate_acceptance_criteria,
        # Architecture
        design_architecture,
        create_api_spec,
        suggest_tech_stack,
        design_data_model,
        create_component_diagram,
        analyze_dependencies,
        # Code Generation
        generate_code,
        refactor_code,
        apply_design_pattern,
        generate_boilerplate,
        optimize_imports,
        format_code,
        # Code Review
        review_code,
        check_code_style,
        analyze_complexity,
        detect_code_smells,
        suggest_improvements,
        check_best_practices,
        # Testing
        generate_unit_tests,
        generate_integration_tests,
        analyze_test_coverage,
        run_tests,
        generate_test_data,
        create_test_plan,
        # Security
        scan_security_issues,
        check_owasp_compliance,
        detect_secrets,
        analyze_dependencies_security,
        generate_security_report,
        suggest_security_fixes,
        # DevOps
        create_ci_pipeline,
        create_cd_pipeline,
        configure_deployment,
        generate_dockerfile,
        create_kubernetes_config,
        setup_monitoring,
        # Debugging
        analyze_error,
        trace_execution,
        identify_root_cause,
        propose_fix,
        analyze_performance,
        detect_memory_issues,
        # Documentation
        generate_api_docs,
        create_readme,
        document_architecture,
        generate_changelog,
        add_inline_comments,
        create_user_guide,
    ]


# Export all tools
__all__ = [
    # Requirements
    "analyze_requirements",
    "extract_user_stories",
    "validate_requirements",
    "prioritize_requirements",
    "detect_ambiguities",
    "generate_acceptance_criteria",
    # Architecture
    "design_architecture",
    "create_api_spec",
    "suggest_tech_stack",
    "design_data_model",
    "create_component_diagram",
    "analyze_dependencies",
    # Code Generation
    "generate_code",
    "refactor_code",
    "apply_design_pattern",
    "generate_boilerplate",
    "optimize_imports",
    "format_code",
    # Code Review
    "review_code",
    "check_code_style",
    "analyze_complexity",
    "detect_code_smells",
    "suggest_improvements",
    "check_best_practices",
    # Testing
    "generate_unit_tests",
    "generate_integration_tests",
    "analyze_test_coverage",
    "run_tests",
    "generate_test_data",
    "create_test_plan",
    # Security
    "scan_security_issues",
    "check_owasp_compliance",
    "detect_secrets",
    "analyze_dependencies_security",
    "generate_security_report",
    "suggest_security_fixes",
    # DevOps
    "create_ci_pipeline",
    "create_cd_pipeline",
    "configure_deployment",
    "generate_dockerfile",
    "create_kubernetes_config",
    "setup_monitoring",
    # Debugging
    "analyze_error",
    "trace_execution",
    "identify_root_cause",
    "propose_fix",
    "analyze_performance",
    "detect_memory_issues",
    # Documentation
    "generate_api_docs",
    "create_readme",
    "document_architecture",
    "generate_changelog",
    "add_inline_comments",
    "create_user_guide",
    # Helper
    "get_all_software_dev_tools",
]
