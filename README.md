# 🏛️ Enterprise Multi-Agent Banking Chatbot

A production-ready, hierarchical Multi-Agent Banking Chatbot backend built natively with the **Microsoft Agent Framework** (`agent-framework`), featuring **Azure AI Search** retrieval, **Cross-Encoder** neural reranking, and grounded answer synthesis. Engineered for high-throughput deployment to **Azure Web App (Azure App Service)**.

---

## 🏛️ System Architecture & Workflow Pipeline

The chatbot is orchestrated as a directed acyclic graph (DAG) using Microsoft Agent Framework's `WorkflowBuilder`, `Executor` nodes, typed messages, and conditional early-exit routing:

```
                            ┌───────────────────────────────────────────────┐
                            │            User Banking Question              │
                            └──────────────────────┬────────────────────────┘
                                                   │
                                                   ▼
                     ┌─────────────────────────────────────────────────────────────┐
                     │          AGENT 1: REQUEST HANDLER EXECUTOR                  │
                     │  - Domain Triage & Guardrails (Banking vs Non-Banking)      │
                     │  - Structured Entity Extraction (Accounts, Amounts, Cards)  │
                     │  - Query Enrichment & Normalization                         │
                     └─────────────────────────────┬───────────────────────────────┘
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        │                                                     │
             [Non-Banking Query]                                       [Banking Query]
                        │                                                     │
                        ▼                                                     ▼
         ┌──────────────────────────────┐              ┌─────────────────────────────────────────────┐
         │     ctx.yield_output()       │              │           ctx.send_message()                │
         │   (Early Exit / Guardrail)   │              │   (Dispatches AnalyzerWorkflowInput)        │
         └──────────────────────────────┘              └──────────────────────┬──────────────────────┘
                                                                              │
                                                                              ▼
                                                       ┌─────────────────────────────────────────────┐
                                                       │           AGENT 2: ANALYZER EXECUTOR        │
                                                       │  - Azure AI Search Candidate Retrieval      │
                                                       │  - Cross-Encoder Semantic Reranker          │
                                                       │  - Top-N Chunk Selection & Filtering        │
                                                       └──────────────────────┬──────────────────────┘
                                                                              │
                                                                              ▼ ctx.send_message()
                                                       ┌─────────────────────────────────────────────┐
                                                       │     AGENT 3: CONTENT GENERATOR EXECUTOR     │
                                                       │  - Grounded Answer Synthesis                │
                                                       │  - Strict Citation & Policy Compliance      │
                                                       │  - Final Output Packaging                   │
                                                       └──────────────────────┬──────────────────────┘
                                                                              │
                                                                              ▼ ctx.yield_output()
                                                       ┌─────────────────────────────────────────────┐
                                                       │             ChatWorkflowOutput              │
                                                       │           (HTTP 200 JSON Response)          │
                                                       └─────────────────────────────────────────────┘
```

### Agent Roles & Responsibilities

1. **Agent 1: Request Handler Agent ([RequestHandlerExecutor](file:///c:/Users/User/Documents/Bank%20repo/banking_chatbot_backend/app/workflow.py#L109))**
   - **Domain Validation**: Evaluates incoming inquiries to distinguish legitimate banking queries from non-banking prompts.
   - **Guardrail Enforcement**: Non-banking requests halt early via `ctx.yield_output()`, saving compute and API costs.
   - **Entity Extraction**: Extracts parameters such as account type, amount, currency, recipient, card last four digits, date, and urgency.
   - **Query Enrichment**: Generates standardized, unambiguous queries for downstream agents.

2. **Agent 2: Analyzer Agent ([AnalyzerExecutor](file:///c:/Users/User/Documents/Bank%20repo/banking_chatbot_backend/app/workflow.py#L216))**
   - **Candidate Retrieval**: Queries Azure AI Search indices for top candidate knowledge chunks.
   - **Cross-Encoder Reranking**: Applies a deep neural Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score relevance against the user's enriched prompt.
   - **Audit Metadata**: Attaches retrieval scores, chunk IDs, and source document titles.

3. **Agent 3: Content Generation Agent ([ContentGeneratorExecutor](file:///c:/Users/User/Documents/Bank%20repo/banking_chatbot_backend/app/workflow.py#L300))**
   - **Grounded Answer Synthesis**: Synthesizes natural, helpful responses based strictly on reranked knowledge chunks.
   - **Hallucination Prevention**: Restricts claims to verified AegisBank policies, APY rates, and procedures.

4. **Agent 4: Action Execution Agent (Planned)**
   - Fulfillment specialist for financial transactions (transfers, card locks, disputes).

---

## 🛠️ Technology Stack

- **Orchestration**: [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (`agent-framework-core`, `agent-framework-openai`)
- **Language**: Python 3.11 - 3.14
- **API Framework**: FastAPI & Uvicorn (ASGI)
- **Data Validation**: Pydantic V2 & Pydantic Settings
- **Search & RAG**: Azure AI Search (`azure-search-documents`)
- **Semantic Reranking**: Sentence-Transformers Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Cloud Platform**: Azure App Service / Azure Web App (Linux Python or Container)
- **Authentication**: Azure Managed Identity (`DefaultAzureCredential`) or Azure OpenAI API Key

---

## 📂 Repository Structure

```
banking_chatbot_backend/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings & environment configuration
│   ├── main.py                   # FastAPI REST API, endpoints & health checks
│   ├── workflow.py               # Microsoft Agent Framework Workflow DAG & Executors
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── request_handler.py    # Agent 1: Request Handler Agent
│   │   ├── analyzer.py           # Agent 2: Analyzer Agent (Search & Reranker)
│   │   └── content_generator.py  # Agent 3: Content Generation Agent
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic V2 data models & typed workflow messages
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── azure_search.py       # Azure AI Search integration
│   │   ├── cross_encoder.py      # Cross-Encoder neural reranker
│   │   └── knowledge_base.py     # Local knowledge base fallback
│   └── services/
│       ├── __init__.py
│       └── client_factory.py     # Microsoft Agent Framework Chat Client Factory
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Test isolation fixtures & offline mocks
│   ├── test_agent.py             # Agent 1 classification & extraction tests
│   ├── test_analyzer.py          # Agent 2 retrieval & cross-encoder tests
│   ├── test_api.py               # FastAPI integration & endpoint tests
│   └── test_workflow.py          # Full 3-agent workflow DAG integration tests
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git ignore file (excludes secrets & bytecode)
├── Dockerfile                    # Container definition for Azure Web App
├── main.py                       # Root entry point
├── requirements.txt              # Pinned Python dependencies
├── run_local.py                  # Interactive CLI runner
└── startup.sh                    # Linux startup script for Azure App Service
```

---

## 🚀 Getting Started Locally

### 1. Prerequisites
- Python 3.11+
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
Fill in your Azure OpenAI and Azure AI Search credentials in `.env`.

### 5. Run the FastAPI Server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Probe: [http://localhost:8000/health](http://localhost:8000/health)
- Agent Topology: [http://localhost:8000/api/v1/agents](http://localhost:8000/api/v1/agents)

### 6. Run Test Suite
All 20 unit and workflow tests execute safely offline using test isolation fixtures:
```bash
pytest tests/ -v
```

---

## 📡 API Reference

### 1. `POST /api/v1/chat`
Processes an incoming conversation turn through the Microsoft Agent Framework multi-agent workflow.

**Request:**
```json
{
  "message": "What is the APY rate on high-yield savings accounts?",
  "session_id": "session-123",
  "user_id": "cust-8871"
}
```

**Response (Banking Query Accepted & Grounded):**
```json
{
  "session_id": "session-123",
  "status": "accepted",
  "data": {
    "is_banking_related": true,
    "confidence": 0.98,
    "category": "account_services",
    "detected_intent": "interest_rate_inquiry",
    "entities": {
      "account_type": "savings",
      "urgency": "normal"
    },
    "enriched_query": "Action: interest_rate_inquiry | Topic: high-yield savings accounts APY",
    "recommended_agent": "content_generator_agent",
    "response_message": "AegisBank High-Yield Savings accounts earn an APY of 4.75% with interest compounded daily.",
    "retrieved_documents": ["Checking & Savings Account Overview"],
    "reranked_documents": ["Checking & Savings Account Overview"]
  },
  "processing_time_ms": 128.4,
  "pipeline": {
    "agent_1_request_handler": { "is_banking_related": true },
    "agent_2_analyzer": { "retrieved_count": 3, "reranked_count": 3 },
    "agent_3_content_generator": { "confidence": 0.98 }
  }
}
```

### 2. `POST /api/v1/chat` (Non-Banking Early Exit)
**Request:**
```json
{
  "message": "How do I bake a chocolate cake?",
  "session_id": "session-456"
}
```

**Response:**
```json
{
  "session_id": "session-456",
  "status": "rejected",
  "data": {
    "is_banking_related": false,
    "confidence": 0.98,
    "category": "non_banking",
    "response_message": "I am the AegisBank Virtual Assistant. I specialize in banking and financial inquiries. How can I assist you with your banking needs today?"
  },
  "processing_time_ms": 35.1,
  "pipeline": {
    "agent_1_request_handler": { "is_banking_related": false }
  }
}
```

### 3. `GET /health`
Liveness/Readiness probe endpoint for Azure App Service.

### 4. `GET /api/v1/agents`
Displays the full topology and operational status of all agents in the banking ecosystem.

---

## ☁️ Azure App Service Deployment

### Method 1: Azure CLI Code Deployment
```bash
az webapp up \
    --resource-group rg-banking-chatbot \
    --plan plan-banking-chatbot \
    --name app-banking-chatbot-backend \
    --runtime "PYTHON:3.11"
```

### Method 2: Docker Container Deployment
```bash
docker build -t yourregistry.azurecr.io/banking-chatbot-backend:v1 .
docker push yourregistry.azurecr.io/banking-chatbot-backend:v1
```

---

## 🔒 Security Best Practices
- **No Secrets in Source Control**: `.env` and local secrets are excluded via `.gitignore`.
- **Azure Managed Identity Support**: Zero-key authentication using `DefaultAzureCredential` for Azure OpenAI and Azure AI Search.
- **Guardrail Enforcement**: Prompt injection and non-banking requests are quarantined and rejected at Agent 1.
