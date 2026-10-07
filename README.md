# Enterprise Banking Chatbot - Request Handler Agent

A production-ready Banking Chatbot backend built with the **Microsoft Agent Framework** (`agent-framework`), engineered for seamless deployment to **Azure Web App (Azure App Service)**.

---

## 🏛️ System Architecture

This project is designed as the frontline gateway of a **4-Agent Enterprise Banking Ecosystem**:

```
                              ┌───────────────────────────────────────────────┐
                              │                 User Request                  │
                              └──────────────────────┬────────────────────────┘
                                                     │
                                                     ▼
                              ┌───────────────────────────────────────────────┐
                              │         AGENT 1: REQUEST HANDLER AGENT        │
                              │          (Microsoft Agent Framework)          │
                              └──────┬─────────────────────────────────┬──────┘
                                     │                                 │
                   [Non-Banking Query]                                 │ [Banking Query]
                                     │                                 │
                                     ▼                                 ▼
                     ┌──────────────────────────────┐  ┌───────────────────────────────────┐
                     │   Polite Refusal & Guidance  │  │  Accept, Enrich & Extract Entities│
                     │  (Scope boundaries & FAQs)   │  │   (Classify Intent, Downstream)   │
                     └──────────────────────────────┘  └─────────────────┬─────────────────┘
                                                                         │
                                       ┌─────────────────────────────────┼─────────────────────────────────┐
                                       │                                 │                                 │
                                       ▼                                 ▼                                 ▼
                        ┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐
                        │      AGENT 2 (PLANNED)       │  │      AGENT 3 (PLANNED)       │  │      AGENT 4 (PLANNED)       │
                        │    Account & Transaction     │  │       Cards & Security       │  │     Advisory & Support       │
                        │  (Balances, transfers, etc.) │  │   (Freeze, PIN, disputes)    │  │   (Loans, mortgages, FAQs)   │
                        └──────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────────┘
```

### Agent 1: Request Handler Agent Responsibilities
1. **Domain Validation**: Evaluates incoming customer inquiries to determine if they belong to banking/financial operations (`is_banking_related: true/false`).
2. **Polite Out-of-Domain Refusal**: If a non-banking query is detected (e.g. cooking recipes, weather, programming, sports), politely declines, clearly outlines supported services, and guides the user.
3. **Structured Entity Extraction**: Parses amounts, currency, accounts (checking, savings, loans), recipient identifiers, card digits, dates, and urgency levels.
4. **Query Enrichment**: Synthesizes unstructured or conversational customer phrasing into standardized, unambiguous instructions for specialist agents.
5. **Missing Information Detection**: Identifies any missing mandatory parameters required to execute banking actions (e.g. transfer amount or recipient).
6. **Downstream Routing Planning**: Designates the ideal candidate agent among the 3 upcoming specialized agents.

---

## 🛠️ Technology Stack

- **Agent Engine**: [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (`agent-framework-core`, `agent-framework-openai`)
- **Language**: Python 3.11+
- **API Framework**: FastAPI & Uvicorn (ASGI)
- **Data Validation**: Pydantic V2 & Pydantic Settings
- **Cloud Platform**: Azure App Service / Azure Web App (Linux Python or Container)
- **Authentication**: Azure Managed Identity (`DefaultAzureCredential`) or Azure OpenAI API Key

---

## 📂 Repository Structure

```
banking_chatbot_backend/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Application settings & environment loader
│   ├── main.py                   # FastAPI app, Azure health probe & endpoints
│   ├── agents/
│   │   ├── __init__.py
│   │   └── request_handler.py    # Request Handler Agent (Microsoft Agent Framework)
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic schemas (Request, Response, Entities)
│   └── services/
│       ├── __init__.py
│       └── client_factory.py     # Microsoft Agent Framework Chat Client Factory
├── tests/
│   ├── __init__.py
│   ├── test_agent.py             # Agent domain validation & enrichment tests
│   └── test_api.py               # FastAPI endpoint integration tests
├── .env.example                  # Environment variable reference
├── Dockerfile                    # Containerization for Azure App Service
├── main.py                       # Root ASGI entrypoint for Azure App Service
├── requirements.txt              # Pinned Python dependencies
├── run_local.py                  # Interactive CLI tool for testing
├── startup.sh                    # Linux startup script for Azure App Service
└── README.md                     # Documentation
```

---

## 🚀 Getting Started Locally

### 1. Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- Git

### 2. Setup Virtual Environment
```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

*Note: If no Azure OpenAI keys are provided, the system automatically runs with an intelligent rule-based mock engine (`MockBankingChatClient`). This allows you to immediately test banking vs. non-banking classification without incurring API costs.*

### 5. Run Interactive CLI
Test the Request Handler Agent directly in your terminal:
```bash
# Interactive mode:
python run_local.py

# Or pass a single query:
python run_local.py "What is my checking account balance?"
python run_local.py "Can you give me a recipe for chocolate cake?"
python run_local.py "I lost my debit card ending in 4455, please freeze it!"
```

### 6. Run the FastAPI Web Server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Probe: [http://localhost:8000/health](http://localhost:8000/health)

### 7. Run Test Suite
```bash
pytest tests/ -v
```

---

## ☁️ Deploying to Azure Web App (Azure App Service)

This application is built specifically for **Azure App Service (Linux)**.

### Method 1: Azure CLI Code Deployment (Recommended)

#### Step 1: Login and Create Resource Group & App Service Plan
```bash
# Log in to Azure
az login

# Set variables
RESOURCE_GROUP="rg-banking-chatbot"
LOCATION="eastus"
PLAN_NAME="plan-banking-chatbot"
APP_NAME="app-banking-chatbot-backend"

# Create Resource Group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Linux App Service Plan (B1 Basic or higher recommended)
az appservice plan create \
    --name $PLAN_NAME \
    --resource-group $RESOURCE_GROUP \
    --sku B1 \
    --is-linux
```

#### Step 2: Create Azure Web App
```bash
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $PLAN_NAME \
    --name $APP_NAME \
    --runtime "PYTHON:3.11" \
    --startup-file "startup.sh"
```

#### Step 3: Configure Azure App Service Application Settings
In Azure Portal or via Azure CLI:
```bash
# Option A: Using Azure OpenAI with API Key
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --settings \
        AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
        AZURE_OPENAI_API_KEY="your-api-key" \
        AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-mini" \
        AZURE_OPENAI_API_VERSION="2024-10-21" \
        ALLOWED_ORIGINS="*"

# Option B: Using Azure Managed Identity (Zero-Key Security)
az webapp identity assign --resource-group $RESOURCE_GROUP --name $APP_NAME
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --settings \
        AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
        AZURE_OPENAI_USE_MANAGED_IDENTITY="true" \
        AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-mini"
```
*(Make sure to grant the Web App's Managed Identity the `Cognitive Services OpenAI User` role on your Azure OpenAI resource).*

#### Step 4: Deploy the Code
```bash
# Deploy using Zip Deploy
az webapp up \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --runtime "PYTHON:3.11"
```

#### Step 5: Verify Health Check & Stream Logs
```bash
# Check Azure Health Probe
curl https://$APP_NAME.azurewebsites.net/health

# Stream live application logs
az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME
```

---

### Method 2: Azure Web App for Containers (Docker)

```bash
# Build and tag image
docker build -t yourregistry.azurecr.io/banking-chatbot-backend:v1 .

# Push to Azure Container Registry
docker push yourregistry.azurecr.io/banking-chatbot-backend:v1

# Deploy container to Azure Web App
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $PLAN_NAME \
    --name $APP_NAME \
    --deployment-container-image-name yourregistry.azurecr.io/banking-chatbot-backend:v1
```

---

## 📡 API Reference

### 1. `POST /api/v1/chat`
Processes an incoming conversation turn through the Request Handler Agent.

**Request Body:**
```json
{
  "message": "Send $250 to Sarah from checking",
  "session_id": "session-123",
  "user_id": "cust-8871"
}
```

**Response (Banking Query Accepted):**
```json
{
  "session_id": "session-123",
  "status": "accepted",
  "data": {
    "is_banking_related": true,
    "confidence": 0.96,
    "category": "funds_transfer",
    "detected_intent": "transfer_funds",
    "entities": {
      "account_type": "checking",
      "amount": 250.0,
      "currency": "USD",
      "recipient": "Sarah",
      "urgency": "normal"
    },
    "enriched_query": "Action: Funds Transfer | Amount: 250.0 USD | Recipient: Sarah | Source Account: checking | Original Request: 'User Request: Send $250 to Sarah from checking'",
    "missing_information": [],
    "recommended_agent": "account_transaction_agent",
    "response_message": "I will assist you in transferring 250.0 USD to Sarah. Ready to route to Account & Transaction Specialist."
  },
  "processing_time_ms": 14.2,
  "timestamp": "2026-10-04T05:35:00Z"
}
```

**Response (Non-Banking Query Rejected):**
```json
{
  "session_id": "session-456",
  "status": "rejected",
  "data": {
    "is_banking_related": false,
    "confidence": 0.98,
    "category": "non_banking",
    "detected_intent": null,
    "entities": {},
    "enriched_query": null,
    "missing_information": [],
    "recommended_agent": null,
    "response_message": "I am the Virtual Banking Assistant. I specialize in banking and financial services including account balance checks, funds transfers, debit/credit cards, statements, and loan services. I cannot assist with non-banking inquiries. How can I help you with your banking needs today?"
  },
  "processing_time_ms": 8.5,
  "timestamp": "2026-10-04T05:35:00Z"
}
```

---

### 2. `GET /health`
Liveness/Readiness probe endpoint for Azure App Service.

**Response:**
```json
{
  "status": "healthy",
  "service": "Enterprise Banking Chatbot API",
  "version": "1.0.0",
  "agent_framework": "Microsoft Agent Framework (agent-framework)",
  "active_agent": "RequestHandlerAgent (Agent 1/4)",
  "llm_provider": "Azure OpenAI (gpt-4o-mini)",
  "mock_mode": false
}
```

---

### 3. `GET /api/v1/agents`
Displays the full topology and roadmap for the 4-agent ecosystem.
