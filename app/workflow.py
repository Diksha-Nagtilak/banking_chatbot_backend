"""
Microsoft Agent Framework Orchestration Workflow.
Orchestrates the 3-Agent Banking Chatbot Pipeline:
  1. Agent 1: Request Handler Agent (Triage, Domain Classification, Entity Extraction & Query Enrichment)
  2. Agent 2: Analyzer Agent (Azure AI Search index retrieval + Cross-Encoder reranking)
  3. Agent 3: Content Generation Agent (Grounded answer synthesis from reranked context)

Orchestration is powered natively by Microsoft Agent Framework's WorkflowBuilder,
Executors, typed message passing, and conditional early-exit routing.
"""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent_framework import (
    Executor,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from app.agents.analyzer import AnalyzerAgent
from app.agents.content_generator import ContentGeneratorAgent
from app.agents.request_handler import RequestHandlerAgent
from app.models.schemas import (
    AnalyzerOutput,
    ContentGeneratorOutput,
    RequestHandlerOutput,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Workflow Typed Messages
# ==============================================================================

class ChatWorkflowInput(BaseModel):
    """Initial input sent into the Microsoft Agent Framework Workflow."""
    message: str = Field(..., description="User's raw banking question or prompt")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AnalyzerWorkflowInput(BaseModel):
    """
    Message passed from Agent 1 (Request Handler) to Agent 2 (Analyzer).
    Carries approved and enriched prompt data.
    """
    query: str = Field(..., description="User's original query")
    enriched_query: Optional[str] = Field(None, description="Enriched query from Agent 1")
    category: Optional[str] = Field(None, description="Banking domain category")
    session_id: str
    context: Optional[Dict[str, Any]] = None
    request_output: RequestHandlerOutput


class ContentGeneratorWorkflowInput(BaseModel):
    """
    Message passed from Agent 2 (Analyzer) to Agent 3 (Content Generator).
    Carries the original question and top reranked chunks.
    """
    query: str = Field(..., description="User's original question")
    reranked_chunks: List[RetrievedChunk] = Field(default_factory=list)
    session_id: str
    context: Optional[Dict[str, Any]] = None
    request_output: RequestHandlerOutput
    analyzer_output: AnalyzerOutput


class ChatWorkflowOutput(BaseModel):
    """Final output emitted by the Microsoft Agent Framework Workflow."""
    session_id: str
    status: str = Field("accepted", description="'accepted' or 'rejected'")
    data: RequestHandlerOutput
    processing_time_ms: float = 0.0
    pipeline: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Educational Terminal Printing Helpers (Pure ASCII for Windows compatibility)
# ==============================================================================

def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _print_agent_header(step_num: int, total_steps: int, title: str) -> None:
    border = "=" * 80
    _safe_print(f"\n{border}")
    _safe_print(f"[*] [STEP {step_num}/{total_steps}] {title}")
    _safe_print(border)


def _print_agent_footer() -> None:
    _safe_print(f"{'=' * 80}\n")


# ==============================================================================
# Microsoft Agent Framework Executors
# ==============================================================================

class RequestHandlerExecutor(Executor):
    """
    Agent 1 Executor in the Microsoft Agent Framework Workflow.
    Responsibilities:
    - Domain classification: checks if input is banking-related.
    - If non-banking: yields early output (Workflow stops here!).
    - If banking-related: extracts entities, enriches query, and routes to Agent 2.
    """

    def __init__(
        self,
        request_handler: Optional[RequestHandlerAgent] = None,
        id: str = "banking_request_handler_executor",
    ) -> None:
        super().__init__(id=id)
        self.request_handler = request_handler or RequestHandlerAgent()

    @handler
    async def process(
        self,
        input_data: ChatWorkflowInput,
        ctx: WorkflowContext[AnalyzerWorkflowInput, ChatWorkflowOutput],
    ) -> None:
        logger.info("[FLOW: STEP 3 - AGENT 1 EXECUTOR] RequestHandlerExecutor.process handler triggered.")
        logger.info(f"   [3.1] Variable 'input_data.message'   : '{input_data.message}'")
        logger.info(f"   [3.2] Variable 'input_data.session_id': '{input_data.session_id}'")
        logger.info(f"   [3.3] Variable 'input_data.context'   : {input_data.context}")
        logger.info("   [3.4] Calling Agent 1: RequestHandlerAgent.process()...")

        agent_output: RequestHandlerOutput = await self.request_handler.process(
            query=input_data.message,
            session_id=input_data.session_id,
            context=input_data.context,
        )

        cat_str = (
            agent_output.category.value
            if hasattr(agent_output.category, "value")
            else str(agent_output.category)
        )

        logger.info("[FLOW: STEP 3.5 - AGENT 1 OUTPUT RECEIVED]")
        logger.info(f"   - Variable 'agent_output.is_banking_related': {agent_output.is_banking_related}")
        logger.info(f"   - Variable 'agent_output.category'          : {cat_str}")
        logger.info(f"   - Variable 'agent_output.confidence'        : {agent_output.confidence}")
        logger.info(f"   - Variable 'agent_output.detected_intent'   : '{agent_output.detected_intent}'")
        logger.info(f"   - Variable 'agent_output.enriched_query'    : '{agent_output.enriched_query}'")
        if hasattr(agent_output.entities, "model_dump"):
            logger.info(f"   - Variable 'agent_output.entities'          : {agent_output.entities.model_dump(exclude_none=True)}")

        # Educational Terminal Print (Pure ASCII)
        _print_agent_header(1, 3, "AGENT 1: REQUEST HANDLER AGENT (Triage, Validation & Query Enrichment)")
        _safe_print(f"Customer Prompt        : \"{input_data.message}\"")
        _safe_print(f"Client / Model Engine  : {self.request_handler.provider_name} (Live LLM)")
        _safe_print(f"Domain Evaluation      : {'[APPROVED] (Banking)' if agent_output.is_banking_related else '[REJECTED] (Non-Banking)'} (Confidence: {agent_output.confidence * 100:.1f}%)")
        _safe_print(f"Category & Intent      : Category: {cat_str} | Intent: {agent_output.detected_intent or 'None'}")
        if hasattr(agent_output.entities, "model_dump"):
            extracted = agent_output.entities.model_dump(exclude_none=True)
            if extracted:
                _safe_print(f"Entities Extracted     : {extracted}")
        if agent_output.enriched_query:
            _safe_print(f"Enriched Query Created : \"{agent_output.enriched_query}\"")
        if not agent_output.is_banking_related:
            _safe_print(f"Routing Decision       : Non-banking request -> Halting workflow early (RAG & Generation skipped).")
        else:
            _safe_print(f"Routing Decision       : Query approved -> Dispatching enriched prompt to Agent 2 (Analyzer)...")
        _print_agent_footer()

        pipeline_info: Dict[str, Any] = {
            "agent_1_request_handler": {
                "is_banking_related": agent_output.is_banking_related,
                "category": cat_str,
                "detected_intent": agent_output.detected_intent,
                "entities": agent_output.entities.model_dump() if hasattr(agent_output.entities, "model_dump") else {},
                "enriched_query": agent_output.enriched_query,
            }
        }

        # Early exit for non-banking requests (guardrail enforcement)
        if not agent_output.is_banking_related:
            logger.info("[FLOW: STEP 3A - EARLY EXIT] Non-banking query rejected. Halting workflow early via ctx.yield_output().")
            early_output = ChatWorkflowOutput(
                session_id=input_data.session_id,
                status="rejected",
                data=agent_output,
                processing_time_ms=0.0,
                pipeline=pipeline_info,
            )
            await ctx.yield_output(early_output)
            return

        # Pass approved and enriched request to Agent 2 (Analyzer Agent)
        logger.info("[FLOW: STEP 3B - FORWARD TO AGENT 2] Preparing AnalyzerWorkflowInput...")
        analyzer_msg = AnalyzerWorkflowInput(
            query=input_data.message,
            enriched_query=agent_output.enriched_query,
            category=cat_str,
            session_id=input_data.session_id,
            context=input_data.context,
            request_output=agent_output,
        )
        logger.info(
            f"   [3B.1] Calling ctx.send_message() to dispatch enriched query to Agent 2: '{agent_output.enriched_query}'"
        )
        await ctx.send_message(analyzer_msg)


class AnalyzerExecutor(Executor):
    """
    Agent 2 Executor in the Microsoft Agent Framework Workflow.
    Responsibilities:
    - Receives approved and enriched query from Agent 1.
    - Queries Azure AI Search index for candidate chunks.
    - Applies Cross-Encoder model to rerank chunks for semantic relevance.
    - Dispatches top reranked chunks + user question to Agent 3.
    """

    def __init__(
        self,
        analyzer: Optional[AnalyzerAgent] = None,
        id: str = "banking_analyzer_executor",
    ) -> None:
        super().__init__(id=id)
        self.analyzer = analyzer or AnalyzerAgent()

    @handler
    async def process(
        self,
        input_data: AnalyzerWorkflowInput,
        ctx: WorkflowContext[ContentGeneratorWorkflowInput, Any],
    ) -> None:
        logger.info("[FLOW: STEP 5 - AGENT 2 EXECUTOR] AnalyzerExecutor.process handler received message from Agent 1.")
        logger.info(f"   [5.1] Variable 'input_data.query'         : '{input_data.query}'")
        logger.info(f"   [5.2] Variable 'input_data.enriched_query': '{input_data.enriched_query}'")
        logger.info(f"   [5.3] Variable 'input_data.category'      : '{input_data.category}'")
        logger.info(f"   [5.4] Variable 'input_data.session_id'    : '{input_data.session_id}'")
        logger.info("   [5.5] Calling Agent 2: AnalyzerAgent.process() for Azure AI Search & Cross-Encoder reranking...")

        analyzer_output: AnalyzerOutput = await self.analyzer.process(
            query=input_data.query,
            enriched_query=input_data.enriched_query,
            category=input_data.category,
        )

        logger.info("[FLOW: STEP 5.6 - AGENT 2 OUTPUT RECEIVED]")
        logger.info(f"   - Variable 'analyzer_output.retrieved_count' : {analyzer_output.retrieved_count}")
        logger.info(f"   - Variable 'analyzer_output.search_provider' : '{analyzer_output.search_provider}'")
        logger.info(f"   - Variable 'analyzer_output.reranker_provider': '{analyzer_output.reranker_provider}'")
        logger.info(f"   - Variable 'analyzer_output.reranked_chunks' : {len(analyzer_output.reranked_chunks)} chunks retained")

        # Attach retrieval audit metadata to the cumulative request output
        input_data.request_output.retrieved_documents = [
            c.title or c.id for c in analyzer_output.reranked_chunks
        ]
        input_data.request_output.reranked_documents = [
            c.title or c.id for c in analyzer_output.reranked_chunks
        ]
        input_data.request_output.reranked_chunks = analyzer_output.reranked_chunks

        # Educational Terminal Print (Pure ASCII)
        _print_agent_header(2, 3, "AGENT 2: ANALYZER AGENT (Azure AI Search Index & Cross-Encoder Reranker)")
        _safe_print(f"Search Query Dispatched : \"{input_data.enriched_query or input_data.query}\"")
        _safe_print(f"Search Provider Source  : {analyzer_output.search_provider}")
        _safe_print(f"Candidate Chunks Found  : {analyzer_output.retrieved_count} chunks retrieved")
        _safe_print(f"Cross-Encoder Reranker  : {analyzer_output.reranker_provider}")
        _safe_print("Top Reranked Chunks Forwarded to Agent 3:")
        for idx, c in enumerate(analyzer_output.reranked_chunks, start=1):
            score_str = f"Rerank Score: {c.rerank_score:.2f}" if c.rerank_score is not None else f"Initial Score: {c.retrieval_score:.2f}"
            chunk_idx_str = f" [Chunk #{c.chunk_index}]" if c.chunk_index is not None else ""
            _safe_print(f"   * [Rank {c.rank or idx}] {score_str} | \"{c.title}\"{chunk_idx_str} (ID: {c.id})")
            snippet = c.content.replace('\n', ' ')[:110]
            _safe_print(f"     Excerpt: {snippet}...")
        _safe_print(f"Routing Decision        : Top {len(analyzer_output.reranked_chunks)} chunks ready -> Forwarding to Agent 3 (Content Generator)...")
        _print_agent_footer()

        # Forward reranked chunks + question to Agent 3 (Content Generation Agent)
        logger.info("[FLOW: STEP 5.7 - FORWARD TO AGENT 3] Packaging ContentGeneratorWorkflowInput...")
        content_gen_msg = ContentGeneratorWorkflowInput(
            query=input_data.query,
            reranked_chunks=analyzer_output.reranked_chunks,
            session_id=input_data.session_id,
            context=input_data.context,
            request_output=input_data.request_output,
            analyzer_output=analyzer_output,
        )
        logger.info(
            f"   [5.8] Calling ctx.send_message() to forward {len(analyzer_output.reranked_chunks)} chunks to Agent 3."
        )
        await ctx.send_message(content_gen_msg)


class ContentGeneratorExecutor(Executor):
    """
    Agent 3 Executor in the Microsoft Agent Framework Workflow.
    Responsibilities:
    - Receives user question and reranked chunks from Agent 2.
    - Synthesizes grounded, conversational, and policy-compliant banking answer.
    - Emits final workflow output.
    """

    def __init__(
        self,
        content_generator: Optional[ContentGeneratorAgent] = None,
        id: str = "banking_content_generator_executor",
    ) -> None:
        super().__init__(id=id)
        self.content_generator = content_generator or ContentGeneratorAgent()

    @handler
    async def process(
        self,
        input_data: ContentGeneratorWorkflowInput,
        ctx: WorkflowContext[Any, ChatWorkflowOutput],
    ) -> None:
        logger.info("[FLOW: STEP 7 - AGENT 3 EXECUTOR] ContentGeneratorExecutor.process handler received message from Agent 2.")
        logger.info(f"   [7.1] Variable 'input_data.query'          : '{input_data.query}'")
        logger.info(f"   [7.2] Variable 'input_data.reranked_chunks': {len(input_data.reranked_chunks)} chunks provided")
        logger.info(f"   [7.3] Variable 'input_data.session_id'     : '{input_data.session_id}'")
        logger.info("   [7.4] Calling Agent 3: ContentGeneratorAgent.process() for grounded answer synthesis...")

        gen_output: ContentGeneratorOutput = await self.content_generator.process(
            query=input_data.query,
            reranked_chunks=input_data.reranked_chunks,
            context=input_data.context,
        )

        logger.info("[FLOW: STEP 7.5 - AGENT 3 OUTPUT RECEIVED]")
        logger.info(f"   - Variable 'gen_output.confidence'      : {gen_output.confidence}")
        logger.info(f"   - Variable 'gen_output.grounded_sources': {gen_output.grounded_sources}")
        snippet = gen_output.answer.replace(chr(10), " ")[:120]
        logger.info(f"   - Variable 'gen_output.answer' snippet  : '{snippet}...'")

        # Update final response message and active agent in cumulative output
        req_out = input_data.request_output
        req_out.response_message = gen_output.answer
        req_out.recommended_agent = "content_generator_agent"

        # Educational Terminal Print (Pure ASCII)
        _print_agent_header(3, 3, "AGENT 3: CONTENT GENERATION AGENT (Grounded Synthesis)")
        _safe_print(f"Customer Question       : \"{input_data.query}\"")
        _safe_print(f"Client / Model Engine   : {self.content_generator.provider_name} (Live LLM)")
        _safe_print(f"Reranked Context Given  : {len(input_data.reranked_chunks)} verified chunks from Agent 2")
        _safe_print(f"Grounded Source Titles  : {gen_output.grounded_sources}")
        _safe_print("Final Grounded Answer   :")
        for line in gen_output.answer.split('\n'):
            _safe_print(f"   {line}")
        _safe_print("Pipeline Status         : [SUCCESS] Completed successfully via Microsoft Agent Framework.")
        _print_agent_footer()

        pipeline_diagnostics: Dict[str, Any] = {
            "agent_1_request_handler": {
                "is_banking_related": req_out.is_banking_related,
                "category": req_out.category.value if hasattr(req_out.category, "value") else str(req_out.category),
                "detected_intent": req_out.detected_intent,
                "entities": req_out.entities.model_dump() if hasattr(req_out.entities, "model_dump") else {},
                "enriched_query": req_out.enriched_query,
            },
            "agent_2_analyzer": {
                "retrieved_count": input_data.analyzer_output.retrieved_count,
                "reranked_count": len(input_data.analyzer_output.reranked_chunks),
                "search_provider": input_data.analyzer_output.search_provider,
                "reranker_provider": input_data.analyzer_output.reranker_provider,
                "chunks": [c.model_dump() for c in input_data.analyzer_output.reranked_chunks],
            },
            "agent_3_content_generator": {
                "grounded_sources": gen_output.grounded_sources,
                "confidence": gen_output.confidence,
            },
        }

        final_output = ChatWorkflowOutput(
            session_id=input_data.session_id,
            status="accepted",
            data=req_out,
            processing_time_ms=0.0,
            pipeline=pipeline_diagnostics,
        )
        logger.info("[FLOW: STEP 8 - WORKFLOW YIELD] Final output packaged into ChatWorkflowOutput. Calling ctx.yield_output()...")
        await ctx.yield_output(final_output)


# ==============================================================================
# Microsoft Agent Framework Workflow Builder
# ==============================================================================

def create_banking_workflow(
    request_handler: Optional[RequestHandlerAgent] = None,
    analyzer: Optional[AnalyzerAgent] = None,
    content_generator: Optional[ContentGeneratorAgent] = None,
) -> Workflow:
    """
    Builds the end-to-end 3-Agent Workflow using Microsoft Agent Framework's WorkflowBuilder:
    [Agent 1: Request Handler] -> [Agent 2: Analyzer] -> [Agent 3: Content Generator]
    """
    req_exec = RequestHandlerExecutor(request_handler=request_handler)
    ana_exec = AnalyzerExecutor(analyzer=analyzer)
    gen_exec = ContentGeneratorExecutor(content_generator=content_generator)

    workflow = (
        WorkflowBuilder(start_executor=req_exec, name="BankingMultiAgentWorkflow")
        .add_edge(req_exec, ana_exec)
        .add_edge(ana_exec, gen_exec)
        .build()
    )
    logger.info("Successfully constructed Microsoft Agent Framework BankingMultiAgentWorkflow.")
    return workflow


async def run_banking_workflow(
    workflow: Workflow,
    message: str,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ChatWorkflowOutput:
    """
    Executes the Microsoft Agent Framework Workflow and returns the structured output.
    """
    start_time = time.perf_counter()
    effective_session = session_id or str(uuid.uuid4())

    logger.info("[FLOW: STEP 2 - WORKFLOW ENGINE START] run_banking_workflow() entered.")
    logger.info(f"   [2.1] Variable 'message'          : '{message}'")
    logger.info(f"   [2.2] Variable 'effective_session': '{effective_session}'")
    logger.info(f"   [2.3] Variable 'context'          : {context}")

    workflow_input = ChatWorkflowInput(
        message=message,
        session_id=effective_session,
        context=context or {},
    )
    logger.info("   [2.4] Packaged into ChatWorkflowInput. Triggering await workflow.run(workflow_input)...")

    events = await workflow.run(workflow_input)
    outputs = events.get_outputs()

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(f"   [2.5] workflow.run() completed in {elapsed_ms} ms. Total output events yielded: {len(outputs)}")

    if outputs and isinstance(outputs[0], ChatWorkflowOutput):
        result = outputs[0]
        result.processing_time_ms = elapsed_ms
        logger.info(f"   [2.6] Successfully extracted ChatWorkflowOutput (status='{result.status}'). Returning to caller.")
        return result

    # Fallback if no outputs yielded
    logger.warning("Workflow yielded no outputs. Returning empty ChatWorkflowOutput.")
    from app.models.schemas import BankingCategory, BankingEntities
    return ChatWorkflowOutput(
        session_id=effective_session,
        status="error",
        data=RequestHandlerOutput(
            is_banking_related=False,
            confidence=0.0,
            category=BankingCategory.NON_BANKING,
            response_message="An unexpected workflow execution error occurred.",
            entities=BankingEntities(),
        ),
        processing_time_ms=elapsed_ms,
        pipeline={},
    )
