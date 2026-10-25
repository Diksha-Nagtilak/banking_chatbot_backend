"""
Agent 2: Analyzer Agent (Microsoft Agent Framework RAG & Cross-Encoder Specialist).
Takes the approved and enriched user input from Agent 1 (Request Handler Agent),
queries the Azure AI Search index for relevant candidate chunks,
reranks them using the Cross-Encoder model, and packages high-precision
context for Agent 3 (Content Generation Agent).
"""
import logging
from typing import Any, Dict, List, Optional
from agent_framework import Agent, BaseChatClient, tool
from app.config import settings
from app.models.schemas import AnalyzerOutput, RetrievedChunk
from app.rag.azure_search import search_azure_ai_search
from app.rag.cross_encoder import rerank_chunks

logger = logging.getLogger(__name__)


# ==============================================================================
# Microsoft Agent Framework Native Tools
# ==============================================================================

@tool
def search_azure_search_index(
    query: str,
    category: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    Query the Azure AI Search index for relevant knowledge base chunks.
    Args:
        query: Search prompt or enriched question.
        category: Optional banking category filter.
        top_k: Number of candidate chunks to fetch.
    Returns:
        JSON string representing candidate retrieved chunks.
    """
    chunks, provider = search_azure_ai_search(
        query=query,
        top_k=top_k or settings.AZURE_SEARCH_TOP_K,
        category=category,
    )
    import json
    return json.dumps([c.model_dump() for c in chunks])


@tool
def rerank_candidate_chunks(
    query: str,
    chunks_json: str,
    top_n: Optional[int] = None,
) -> str:
    """
    Rerank candidate chunks using the Cross-Encoder model for deep semantic alignment.
    Args:
        query: User's natural language question.
        chunks_json: JSON string of candidate chunks from Azure AI Search.
        top_n: Number of top reranked chunks to return.
    Returns:
        JSON string representing the top reranked chunks.
    """
    import json
    raw = json.loads(chunks_json)
    chunk_objs = [RetrievedChunk.model_validate(c) for c in raw]
    reranked, provider = rerank_chunks(
        query=query,
        chunks=chunk_objs,
        top_n=top_n or settings.RERANK_TOP_N,
    )
    return json.dumps([c.model_dump() for c in reranked])


ANALYZER_SYSTEM_PROMPT = """
You are the **Banking Analyzer Agent (Agent 2 of 4)** for AegisBank.
Your role:
1. Ingest approved and enriched customer queries from the Request Handler Agent.
2. Retrieve relevant banking policy, fee, account, and rate chunks from the Azure AI Search index.
3. Rerank retrieved candidate chunks using the Cross-Encoder model for maximal semantic relevance.
4. Prepare structured context for the downstream Content Generation Agent.
"""


class AnalyzerAgent:
    """
    Microsoft Agent Framework Implementation of Agent 2: Banking Analyzer Agent.
    Specialist in:
    1. Ingestion of approved and enriched customer queries from Agent 1.
    2. Retrieval of candidate chunks from Azure AI Search index.
    3. Cross-Encoder reranking using query-passage cross-attention.
    4. Packaging high-quality reranked context for Agent 3 (Content Generation Agent).
    """

    def __init__(self, client: Optional[BaseChatClient] = None) -> None:
        self.name = "Banking Analyzer Agent"
        self.role = "Azure AI Search & Cross-Encoder Reranking Specialist"
        self.tools = [search_azure_search_index, rerank_candidate_chunks]
        self.client = client

        # Initialize Microsoft Agent Framework Agent
        if self.client is not None:
            self.agent = Agent(
                id="banking_analyzer_agent",
                name="Banking Analyzer Agent",
                description="Agent 2: RAG Retrieval (Azure AI Search) & Cross-Encoder Reranking",
                client=self.client,
                instructions=ANALYZER_SYSTEM_PROMPT,
                tools=self.tools,
            )
        else:
            self.agent = None

        logger.info("Initialized AnalyzerAgent (Agent 2/4) with Microsoft Agent Framework tools.")

    async def process(
        self,
        query: str,
        enriched_query: Optional[str] = None,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> AnalyzerOutput:
        """
        Executes the RAG and Cross-Encoder pipeline for an approved user query.

        Args:
            query: User's original natural question
            enriched_query: Standardized/enriched query from Request Handler Agent
            category: Domain category (e.g. account_services, funds_transfer, cards_and_security)
            top_k: Number of candidate chunks to fetch from Azure AI Search index
            top_n: Number of top reranked chunks to retain

        Returns:
            AnalyzerOutput with candidate count and top reranked chunks ready for Agent 3.
        """
        # Step 1: Formulate search query (prioritize enriched query for index retrieval)
        search_query = (
            enriched_query
            if enriched_query and len(enriched_query.strip()) > 5
            else query
        )
        effective_top_k = top_k or settings.AZURE_SEARCH_TOP_K or 5
        effective_top_n = top_n or settings.RERANK_TOP_N or 3

        logger.info("[FLOW: STEP 6 - AGENT 2 ANALYZER RAG PIPELINE] AnalyzerAgent.process() started.")
        logger.info(f"   [6.1] Variable 'query'          : '{query}'")
        logger.info(f"   [6.2] Variable 'enriched_query' : '{enriched_query}'")
        logger.info(f"   [6.3] Variable 'category'       : '{category}'")
        logger.info(f"   [6.4] Selected 'search_query'   : '{search_query}' (top_k={effective_top_k}, top_n={effective_top_n})")
        logger.info("   [6.5] Calling search_azure_ai_search() to query Azure AI Search index...")

        # Step 2: Retrieve candidate chunks from Azure AI Search
        candidate_chunks, search_provider = search_azure_ai_search(
            query=search_query,
            top_k=effective_top_k,
            category=category,
        )

        logger.info(
            f"   [6.6] Successfully retrieved {len(candidate_chunks)} candidate chunk(s) via {search_provider}"
        )
        for c_idx, chk in enumerate(candidate_chunks, start=1):
            logger.info(f"         * Candidate {c_idx}: ID='{chk.id}', Title='{chk.title}', Score={chk.retrieval_score}")

        # Step 3: Rerank candidate chunks using Cross-Encoder model
        logger.info("   [6.7] Calling rerank_chunks() with Cross-Encoder for semantic query-passage alignment...")
        reranked_chunks, reranker_provider = rerank_chunks(
            query=query,
            chunks=candidate_chunks,
            top_n=effective_top_n,
        )

        logger.info(
            f"   [6.8] Reranker completed. Retained top {len(reranked_chunks)} chunk(s) via {reranker_provider}"
        )
        for r_idx, r_chk in enumerate(reranked_chunks, start=1):
            logger.info(f"         * Rank {r_idx}: Score={r_chk.rerank_score}, Title='{r_chk.title}' (ID: {r_chk.id})")

        # Step 4: Build context summary
        summary_titles = [c.title for c in reranked_chunks if c.title]
        summary_str = f"Retrieved {len(candidate_chunks)} candidates from {search_provider}, reranked to top {len(reranked_chunks)} via {reranker_provider}"
        if summary_titles:
            summary_str += f" ({', '.join(summary_titles)})"

        output = AnalyzerOutput(
            query=query,
            enriched_query=enriched_query,
            retrieved_count=len(candidate_chunks),
            reranked_chunks=reranked_chunks,
            search_provider=search_provider,
            reranker_provider=reranker_provider,
            summary=summary_str,
        )
        logger.info("[FLOW: STEP 6.9 - AGENT 2 ANALYZER FINISHED] Returning AnalyzerOutput.")
        return output