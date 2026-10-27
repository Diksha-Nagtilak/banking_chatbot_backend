"""
FastAPI Web Application for Enterprise Banking Chatbot.
Connects FastAPI to Microsoft Agent Framework Multi-Agent Pipeline:
- Agent 1: Request Handler Agent (Triage, Domain Classification, Entity Extraction, Enrichment)
- Agent 2: Analyzer Agent (RAG: Azure AI Search + Cross-Encoder Reranker)
- Agent 3: Content Generation Agent (Grounded Answer Synthesis)

Endpoints:
- GET  /                  : Root endpoint with service info
- GET  /health            : Health probe for Azure App Service & frontend
- GET  /api/v1/agents     : Multi-Agent architecture topology
- POST /api/v1/chat       : Full end-to-end multi-agent chat pipeline
- POST /api/v1/enrich     : Direct query enrichment endpoint (Agent 1)
- POST /api/v1/analyze    : Direct RAG retrieval & Cross-Encoder reranking endpoint (Agent 2)
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.agents.request_handler import RequestHandlerAgent
from app.agents.analyzer import AnalyzerAgent
from app.agents.content_generator import ContentGeneratorAgent
from app.workflow import create_banking_workflow, run_banking_workflow
from agent_framework import Workflow
from app.config import settings
from app.models.schemas import (
    AnalyzerOutput,
    ChatRequest,
    ChatResponse,
    EnrichRequest,
    HealthResponse,
    RequestHandlerOutput,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("banking_chatbot")

# Global instances for Multi-Agent Ecosystem
request_handler_agent: RequestHandlerAgent | None = None
analyzer_agent: AnalyzerAgent | None = None
content_generator_agent: ContentGeneratorAgent | None = None
banking_workflow: Workflow | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initializes the multi-agent framework pipeline upon startup."""
    global request_handler_agent, analyzer_agent, content_generator_agent
    logger.info("Initializing Enterprise Banking Chatbot Multi-Agent Pipeline...")

    # Initialize Agent 1: Request Handler Agent
    try:
        request_handler_agent = RequestHandlerAgent()
        logger.info(
            f"Agent 1 (Request Handler) ready using live LLM provider: {request_handler_agent.provider_name}"
        )
    except Exception as e:
        logger.warning(f"Agent 1 initialization warning: {e}")
        request_handler_agent = None

    # Initialize Agent 2: Analyzer Agent (RAG + Cross-Encoder)
    try:
        analyzer_agent = AnalyzerAgent()
        logger.info("Agent 2 (Analyzer) ready for Azure AI Search & Cross-Encoder Reranking.")
    except Exception as e:
        logger.warning(f"Agent 2 initialization warning: {e}")
        analyzer_agent = None

    # Initialize Agent 3: Content Generation Agent
    try:
        content_generator_agent = ContentGeneratorAgent()
        logger.info(
            f"Agent 3 (Content Generator) ready using live LLM provider: {content_generator_agent.provider_name}"
        )
    except Exception as e:
        logger.warning(f"Agent 3 initialization warning: {e}")
        content_generator_agent = None

    # Initialize Microsoft Agent Framework Workflow Orchestrator
    try:
        global banking_workflow
        banking_workflow = create_banking_workflow(
            request_handler=request_handler_agent,
            analyzer=analyzer_agent,
            content_generator=content_generator_agent,
        )
        logger.info("Microsoft Agent Framework Banking Workflow Orchestrator initialized.")
    except Exception as e:
        logger.exception(f"Workflow initialization error: {e}")
        banking_workflow = None

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
        "pipeline": [
            "Agent 1: Request Handler Agent (Active)",
            "Agent 2: Analyzer Agent - Azure AI Search + Cross-Encoder Reranker (Active)",
            "Agent 3: Content Generation Agent (Active)",
            "Agent 4: Transaction & Security Action Specialists (Planned)",
        ],
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check Probe")
async def health_check() -> HealthResponse:
    """Liveness and readiness probe for Azure App Service and local frontend."""
    is_ready = request_handler_agent is not None
    return HealthResponse(
        status="healthy" if is_ready else "awaiting_llm_keys",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        agent_framework="Microsoft Agent Framework (agent-framework)",
        active_agent="RequestHandlerAgent (Agent 1), AnalyzerAgent (Agent 2), ContentGeneratorAgent (Agent 3)",
        llm_provider=request_handler_agent.provider_name if request_handler_agent else "No LLM configured (Please add keys to .env)",
        mock_mode=False,
    )


@app.get("/api/v1/agents", summary="Multi-Agent Topology")
async def get_agent_topology() -> Dict[str, Any]:
    """Returns the multi-agent topology and agent statuses."""
    search_info = (
        f"Azure AI Search ({settings.AZURE_SEARCH_INDEX_NAME})"
        if settings.AZURE_SEARCH_INDEX_NAME
        else "Local Knowledge Base Fallback"
    )
    reranker_info = (
        f"Cross-Encoder ({settings.CROSS_ENCODER_MODEL_NAME})"
        if settings.ENABLE_CROSS_ENCODER
        else "Standard BM25/Coverage Scoring"
    )

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
                "id": "analyzer_agent",
                "role": "Agent 2: RAG Retrieval (Azure AI Search) & Cross-Encoder Reranking",
                "status": "ACTIVE",
                "implementation": "app.agents.analyzer.AnalyzerAgent",
                "retriever": search_info,
                "reranker": reranker_info,
            },
            {
                "id": "content_generation_agent",
                "role": "Agent 3: Grounded Answer Synthesis using Reranked Context",
                "status": "ACTIVE",
                "implementation": "agent_framework.Agent",
                "provider": content_generator_agent.provider_name if content_generator_agent else "Not Configured",
            },
            {
                "id": "action_execution_agent",
                "role": "Agent 4: Transaction & Security Action Fulfillment (Transfers, Card Lock)",
                "status": "PLANNED",
                "target_framework": "Microsoft Agent Framework",
            },
        ],
    }


@app.post("/api/v1/chat", response_model=ChatResponse, summary="Chat Endpoint")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    logger.info("=" * 80)
    logger.info("[FLOW: STEP 1 - FASTAPI GATEWAY] Received HTTP POST /api/v1/chat from frontend")
    logger.info(f"   [1.1] Variable 'request.message'    : '{request.message}'")
    logger.info(f"   [1.2] Variable 'request.session_id' : '{request.session_id}'")
    logger.info(f"   [1.3] Variable 'request.user_id'    : '{request.user_id}'")
    logger.info(f"   [1.4] Variable 'request.context'    : {request.context}")

    global request_handler_agent, analyzer_agent, content_generator_agent, banking_workflow

    logger.info("   [1.5] Checking Agent 1 (RequestHandlerAgent) status.............")
    if request_handler_agent is None:
        logger.info("   [1.5.1] RequestHandlerAgent not yet initialized. Initializing now...")
        try:
            request_handler_agent = RequestHandlerAgent()
            logger.info(f"   [1.5.2] RequestHandlerAgent initialized with: {request_handler_agent.provider_name}")
        except Exception as e:
            logger.error(f"   [1.5.ERROR] Failed to initialize RequestHandlerAgent: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Agent 1 Not Configured: {str(e)}",
            )
    else:
        logger.info(f"   [1.5.1] Agent 1 is ready (Provider: {request_handler_agent.provider_name})")

    logger.info("   [1.6] Checking Agent 2 (AnalyzerAgent) status...")
    if analyzer_agent is None:
        logger.info("   [1.6.1] Initializing AnalyzerAgent...")
        analyzer_agent = AnalyzerAgent()
    logger.info("   [1.6.2] Agent 2 is ready (Azure AI Search + Cross-Encoder)")

    logger.info("   [1.7] Checking Agent 3 (ContentGeneratorAgent) status...")
    if content_generator_agent is None:
        logger.info("   [1.7.1] Initializing ContentGeneratorAgent...")
        try:
            content_generator_agent = ContentGeneratorAgent()
            logger.info(f"   [1.7.2] Agent 3 is ready (Provider: {content_generator_agent.provider_name})")
        except Exception as e:
            logger.warning(f"   [1.7.WARN] ContentGeneratorAgent initialization warning: {e}")
            content_generator_agent = None

    if banking_workflow is None:
        logger.info("   [1.8] Constructing Microsoft Agent Framework BankingMultiAgentWorkflow DAG...")
        banking_workflow = create_banking_workflow(
            request_handler=request_handler_agent,
            analyzer=analyzer_agent,
            content_generator=content_generator_agent,
        )
        logger.info("   [1.8.1] BankingMultiAgentWorkflow DAG constructed successfully.")

    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"   [1.9] Variable 'session_id' determined: '{session_id}'")
    logger.info("   [1.10] Calling run_banking_workflow() to trigger Microsoft Agent Framework execution...")

    try:
        workflow_result = await run_banking_workflow(
            workflow=banking_workflow,
            message=request.message,
            session_id=session_id,
            context=request.context,
        )

        logger.info("[FLOW: STEP 9 - FASTAPI GATEWAY RETURN] Workflow execution finished.....")
        logger.info(f"   [9.1] Workflow Result 'status'             : '{workflow_result.status}'")
        logger.info(f"   [9.2] Workflow Result 'processing_time_ms': {workflow_result.processing_time_ms} ms")
        snippet = workflow_result.data.response_message.replace(chr(10), " ")[:120] if workflow_result.data and workflow_result.data.response_message else "No message"
        logger.info(f"   [9.3] Response message snippet            : '{snippet}...'")
        logger.info(f"   [9.4] Packaging into ChatResponse and returning HTTP 200 JSON to frontend.")
        logger.info("=" * 80)

        return ChatResponse(
            session_id=workflow_result.session_id,
            status=workflow_result.status,
            data=workflow_result.data,
            processing_time_ms=workflow_result.processing_time_ms,
            pipeline=workflow_result.pipeline,
        )
    except Exception as wf_err:
        logger.exception(f"Workflow execution exception: {wf_err}. Gracefully falling back to Request Handler Agent direct response.")
        fallback_data = await request_handler_agent.process(
            query=request.message,
            session_id=session_id,
            context=request.context,
        )
        return ChatResponse(
            session_id=session_id,
            status="accepted" if fallback_data.is_banking_related else "rejected",
            data=fallback_data,
            processing_time_ms=0.0,
            pipeline={
                "agent_1_request_handler": fallback_data.model_dump() if hasattr(fallback_data, "model_dump") else {},
                "fallback_notice": f"Handled directly by Agent 1 (LLM: {request_handler_agent.provider_name})",
            },
        )


@app.post("/api/v1/enrich", response_model=RequestHandlerOutput, summary="Direct Query Enrichment")
async def enrich_endpoint(request: EnrichRequest) -> RequestHandlerOutput:
    """Direct query enrichment via live LLM (Agent 1)."""
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


@app.post("/api/v1/analyze", response_model=AnalyzerOutput, summary="Direct RAG & Reranking")
async def analyze_endpoint(request: EnrichRequest) -> AnalyzerOutput:
    """Direct RAG retrieval and Cross-Encoder reranking via Agent 2."""
    global analyzer_agent
    if analyzer_agent is None:
        analyzer_agent = AnalyzerAgent()
    return await analyzer_agent.process(query=request.query)