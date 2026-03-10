# =============================================================================
# LangChain Azure AI Agents — FastAPI Server
# =============================================================================
# Builds the langchain-agents FastAPI server that exposes all agent endpoints
# including the Copilot Studio integration routes (/api/copilot/*).
#
# Build context: libs/azure-ai/
# Entry point:   start_server.py  → uvicorn on port 8000
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System build dependencies required by some Python packages (cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install python-dotenv (used by start_server.py).
# We use pip directly — no Poetry CLI — to avoid lock file staleness.
RUN pip install --no-cache-dir --upgrade pip "python-dotenv>=1.0.0"

# --- Dependency layer (cached when these files are unchanged) ---
# Poetry's build backend (declared in pyproject.toml) requires README.md,
# LICENSE and the package source to generate wheel metadata.
COPY pyproject.toml LICENSE README.md ./
COPY langchain_azure_ai/ langchain_azure_ai/

# Install all runtime extras. This is the slow step — cached until pyproject.toml changes.
RUN pip install --no-cache-dir ".[server,opentelemetry,tools]"

# --- Final source layer (only this layer re-runs when source changes) ---
COPY . .

# Re-install the package itself (deps are already in the layer above)
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

# Liveness probe target: /health endpoint served by the FastAPI app
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "start_server.py"]
