@echo off
call .venv\Scripts\activate.bat
echo ========================================
echo Azure AI Foundry Agent Server
echo ========================================
echo.
echo Starting server on http://localhost:8000
echo Loading 14 agents (may take 30-60 seconds)
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.
python -m uvicorn langchain_azure_ai.server:app --host 0.0.0.0 --port 8000 --reload
