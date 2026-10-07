"""
FastAPI Web Application for Enterprise Banking Chatbot.
Connects FastAPI to Microsoft Agent Framework (Request Handler Agent).

Endpoints:
- GET  /                  : Root endpoint with service info
- GET  /health            : Health probe for Azure App Service & frontend
- GET  /api/v1/agents     : 4-Agent architecture topology
- POST /api/v1/chat       : Full chat triage, domain check, and query enrichment
- POST /api/v1/enrich     : Direct query enrichment and entity extraction endpoint
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.agents.request_handler import RequestHandlerAgent
from app.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    EnrichRequest,
    HealthResponse,
    RequestHandlerOutput,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("banking_chatbot")

# Global singleton for Agent 1 (Request Handler Agent)
request_handler_agent: RequestHandlerAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initializes the Microsoft Agent Framework agent upon startup."""
    global request_handler_agent
    logger.info("Initializing Microsoft Agent Framework Request Handler Agent...")
    try:
        request_handler_agent = RequestHandlerAgent()
        logger.info(
            f"Agent ready using provider: {request_handler_agent.provider_name} (Mock Mode: {request_handler_agent.is_mock})"
        )
    except Exception as e:
        logger.warning(f"Live LLM not yet configured in .env: {e}")
        request_handler_agent = None
    yield
    logger.info("Shutting down banking chatbot...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FastAPI + Microsoft Agent Framework Enterprise Banking Chatbot",
    lifespan=lifespan,
)

# Enable CORS for frontend and other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root Status")
async def root() -> Dict[str, Any]:
    """Root landing endpoint providing service identification."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "framework": "Microsoft Agent Framework",
        "agents": [
            "request_handler_agent (Active)",
            "account_transaction_agent (Planned)",
            "card_security_agent (Planned)",
            "advisory_support_agent (Planned)",
        ],
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check Probe")
async def health_check() -> HealthResponse:
    """Liveness and readiness probe for Azure App Service and local frontend."""
    return HealthResponse(
        status="healthy" if request_handler_agent else "awaiting_llm_keys",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        agent_framework="Microsoft Agent Framework (agent-framework)",
        active_agent="RequestHandlerAgent (Agent 1/4)",
        llm_provider=request_handler_agent.provider_name if request_handler_agent else "No LLM configured (Please add keys to .env)",
        mock_mode=request_handler_agent.is_mock if request_handler_agent else False,
    )


@app.get("/api/v1/agents", summary="Multi-Agent Topology")
async def get_agent_topology() -> Dict[str, Any]:
    """Returns the multi-agent topology and agent statuses."""
    return {
        "architecture": "Hierarchical Multi-Agent Banking Chatbot",
        "framework": "Microsoft Agent Framework",
        "total_agents": 4,
        "agents": [
            {
                "id": "request_handler_agent",
                "role": "Agent 1: Triage, Domain Validation, Entity Extraction & Query Enrichment",
                "status": "ACTIVE",
                "implementation": "agent_framework.Agent",
                "provider": request_handler_agent.provider_name if request_handler_agent else "Not Configured",
            },
            {
                "id": "account_transaction_agent",
                "role": "Agent 2: Balance checks, domestic/wire transfers, account statements",
                "status": "PLANNED",
                "target_framework": "Microsoft Agent Framework",
            },
            {
                "id": "card_security_agent",
                "role": "Agent 3: Debit/Credit card freeze, PIN reset, fraud reporting, limits",
                "status": "PLANNED",
                "target_framework": "Microsoft Agent Framework",
            },
            {
                "id": "advisory_support_agent",
                "role": "Agent 4: Loan applications, interest rates, mortgage advice, FAQs, human handoff",
                "status": "PLANNED",
                "target_framework": "Microsoft Agent Framework",
            },
        ],
    }


@app.post("/api/v1/chat", response_model=ChatResponse, summary="Chat Endpoint")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat gateway endpoint:
    1. Direct LLM execution via Microsoft Agent Framework.
    2. Strict LLM answers with zero hardcoded fallbacks.
    """
    global request_handler_agent
    if request_handler_agent is None:
        try:
            request_handler_agent = RequestHandlerAgent()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"LLM Not Configured: {str(e)}\n\n"
                    "All hardcoded responses have been removed. "
                    "Please paste your Azure AI Foundry or OpenAI credentials into .env to receive answers directly from the LLM."
                ),
            )

    start_time = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Received query: '{request.message}' for session: '{session_id}'")

    try:
        agent_output = await request_handler_agent.process(
            query=request.message,
            session_id=session_id,
            context=request.context,
        )
    except Exception as e:
        logger.exception(f"Error calling live LLM: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Live LLM call failed: {str(e)}. No hardcoded fallback was used.",
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response_status = "accepted" if agent_output.is_banking_related else "rejected"

    logger.info(
        f"Completed in {elapsed_ms}ms | Status: {response_status} | Category: {agent_output.category.value} | "
        f"Route: {agent_output.recommended_agent.value if agent_output.recommended_agent else 'None'}"
    )

    return ChatResponse(
        session_id=session_id,
        status=response_status,
        data=agent_output,
        processing_time_ms=elapsed_ms,
    )


@app.post("/api/v1/enrich", response_model=RequestHandlerOutput, summary="Direct Query Enrichment")
async def enrich_endpoint(request: EnrichRequest) -> RequestHandlerOutput:
    """Direct query enrichment via live LLM."""
    global request_handler_agent
    if request_handler_agent is None:
        try:
            request_handler_agent = RequestHandlerAgent()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM Not Configured: {str(e)}",
            )
    return await request_handler_agent.process(query=request.query)
