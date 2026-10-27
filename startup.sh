#!/bin/bash
# Azure App Service Linux Startup Script
echo "Starting Banking Chatbot Service on Azure Web App..."

# Set default port to 8000 or the port assigned by Azure App Service ($PORT)
SERVER_PORT=${PORT:-8000}

# Launch using uvicorn ASGI server
exec python -m uvicorn main:app --host 0.0.0.0 --port "$SERVER_PORT" --workers 2 --access-log
