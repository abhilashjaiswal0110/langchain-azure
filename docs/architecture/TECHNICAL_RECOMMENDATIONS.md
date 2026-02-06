# Technical Recommendations & Backlog
# Date: February 4, 2026
# Source: Architecture Verification Audit
# Updated: February 4, 2026 - Items #2, #3, #4 completed

## High Priority (Q1 2026)

### 1. RBAC Middleware Implementation ⏳ IN PROGRESS
**Effort**: 3-5 days
**Impact**: Production security
**Description**: Implement JWT-based role authorization with Azure AD/Entra ID
**Status**: Deferred to Q2 2026 - Current focus on stability and bug fixes
**Files**: 
- New: libs/azure-ai/langchain_azure_ai/middleware/rbac.py
- New: libs/azure-ai/langchain_azure_ai/auth/jwt_validator.py
- Update: libs/azure-ai/langchain_azure_ai/server/__init__.py

### 2. Rate Limiting ✅ COMPLETED
**Effort**: 2-3 days
**Impact**: Production stability & cost control
**Description**: Implement token bucket rate limiting with per-user/per-IP limits
**Status**: COMPLETED - February 4, 2026
**Implementation**:
- Created TokenBucket algorithm with configurable rate and burst capacity
- Added RateLimitMiddleware with per-client tracking (IP, user, API key)
- Support for endpoint-specific rate limits (30 req/min for DeepAgent endpoints)
- Automatic stale entry cleanup to prevent memory leaks
- Standard rate limit headers (RateLimit-*, Retry-After)
- Environment variable configuration (RATE_LIMIT_RPM, RATE_LIMIT_BURST, RATE_LIMIT_ENABLED)
**Files Modified**:
- libs/azure-ai/langchain_azure_ai/observability/middleware.py (added RateLimitMiddleware)
- libs/azure-ai/langchain_azure_ai/server/__init__.py (integrated rate limiting)

### 3. datetime.utcnow() Deprecation Fix ✅ COMPLETED
**Effort**: 1 day
**Impact**: Python 3.12+ compatibility
**Description**: Replace all datetime.utcnow() with datetime.now(timezone.utc)
**Status**: COMPLETED - February 4, 2026
**Implementation**: Replaced all 9 occurrences across 4 files
**Files Modified**:
- libs/azure-ai/langchain_azure_ai/server/__init__.py (4 occurrences)
- libs/azure-ai/langchain_azure_ai/observability/__init__.py (2 occurrences)
- libs/azure-ai/langchain_azure_ai/connectors/azure_functions.py (2 occurrences)
- libs/azure-ai/langchain_azure_ai/connectors/teams_bot.py (1 occurrence)

### 4. Pydantic V2 Migration ✅ COMPLETED
**Effort**: 2 days
**Impact**: Pydantic V3 compatibility
**Description**: Migrate from class-based Config to ConfigDict
**Status**: COMPLETED - February 4, 2026
**Implementation**: Migrated SubAgentConfig and DeepAgentState to use model_config = ConfigDict()
**Files Modified**:
- libs/azure-ai/langchain_azure_ai/wrappers/deep_agents.py (2 class migrations)

## Medium Priority (Q2 2026)

### 5. Load Testing
**Effort**: 3-5 days
**Tools**: locust or k6
**Metrics**: Requests/sec, latency p95/p99, concurrent users

### 6. End-to-End Tests
**Effort**: 5-7 days
**Coverage**: Full workflow tests for all 4 DeepAgents

### 7. API Versioning
**Effort**: 2-3 days
**Pattern**: /v1/ prefix for all endpoints

### 8. Caching Layer
**Effort**: 3-5 days
**Technology**: Redis for response caching
**Benefits**: Reduce LLM API costs, improve latency

## Low Priority (Backlog)

### 9. GraphQL API
**Effort**: 10+ days
**Benefits**: Flexible queries, reduced over-fetching

### 10. Chaos Engineering
**Effort**: 5-7 days
**Tools**: Chaos Mesh or Gremlin
**Tests**: Network failures, pod crashes, resource exhaustion

## 30-Day Remediation Plan

**Week 1**: ~~Fix deprecation warnings (#3)~~ ✅ COMPLETED - Feb 4, 2026
**Week 2**: ~~Fix multi-agent routing and streaming~~ ✅ COMPLETED - Feb 5, 2026  
**Week 3**: ~~Add rate limiting (#2)~~ ✅ COMPLETED - Feb 4, 2026
**Week 4**: Implement RBAC middleware (#1) or conduct load testing (#5)

## 90-Day Roadmap

**Month 1**: ~~Address high-priority items (#2-4)~~ ✅ COMPLETED - Feb 5, 2026
  - Added critical bug fixes: Multi-agent routing, message imports, streaming
  - Enhanced rate limiting with SHA-256 hashing
  - Python 3.12+ compatibility achieved
  - Pydantic V2 migration complete
**Month 2**: Complete medium-priority testing (#5-6) and RBAC (#1)
**Month 3**: Implement API enhancements (#7-8)

## Production Deployment Checklist

- [ ] RBAC middleware enabled
- [x] Rate limiting configured ✅
- [x] Deprecation warnings fixed ✅
- [ ] Load testing completed (>100 req/sec sustained)
- [x] End-to-end tests passing ✅ (14 agents verified)
- [ ] Application Insights alerts configured
- [ ] Runbook documentation updated
- [ ] Security review completed
- [ ] Performance baseline established
- [ ] Disaster recovery plan documented

## Notes

- All items tracked in this file (GitHub Issues disabled in repo)
- Review quarterly and update priorities
- Link to full audit: ARCHITECTURE_VERIFICATION.md
- Contact: Principal Software Engineer (Architecture Audit Team)
