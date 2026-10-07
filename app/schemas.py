"""
Pydantic Schemas for FastAPI, Microsoft Agent Framework, RAG, and Reranking.
Clean and easy to understand for step-by-step learning.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1. RAG & Reranking Schemas
# ------------------------------------------------------------------------------
class KnowledgeDocument(BaseModel):
    """Represents a document stored in the knowledge base."""
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Text content used for RAG context")
    category: str = Field(..., description="Topic category (e.g. balances, transfers, cards, loans)")


class ScoredDocument(BaseModel):
    """A document with its computed relevance score from Retrieval or Reranking."""
    document: KnowledgeDocument
    score: float = Field(..., description="Relevance score (higher is more relevant)")
    reason: Optional[str] = Field(None, description="Explanation of why this document matched")


# ------------------------------------------------------------------------------
# 2. Agent Output Schema
# ------------------------------------------------------------------------------
class RequestHandlerOutput(BaseModel):
    """
    Structured output from Agent 1 (Request Handler Agent).
    Demonstrates:
    - Domain validation (is it banking?)
    - Intent & Category classification
    - RAG Context utilized
    - Downstream routing (which specialist agent should handle next)
    """
    is_banking_related: bool = Field(
        ...,
        description="True if query is related to banking/finance, False otherwise"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence score between 0.0 and 1.0"
    )
    category: str = Field(
        ...,
        description="Category: account_services, funds_transfer, cards_and_security, loans_and_credit, or non_banking"
    )
    detected_intent: Optional[str] = Field(
        None,
        description="Specific intent (e.g. check_balance, transfer_funds, freeze_card, loan_inquiry)"
    )
    recommended_agent: Optional[str] = Field(
        None,
        description="Downstream agent recommendation (e.g. account_transaction_agent, card_security_agent)"
    )
    enriched_query: Optional[str] = Field(
        None,
        description="Query enriched with domain terminology and intent for high-accuracy RAG retrieval"
    )
    relevance_reason: Optional[str] = Field(
        None,
        description="Explanation of the banking relevance check"
    )
    response_message: str = Field(
        ...,
        description="The customer-facing conversational answer produced by the Agent"
    )
    retrieved_documents: List[str] = Field(
        default_factory=list,
        description="Titles of candidate documents found in Step 1 (RAG Retrieval)"
    )
    reranked_documents: List[str] = Field(
        default_factory=list,
        description="Titles of top-scoring documents selected in Step 2 (Reranking)"
    )


# ------------------------------------------------------------------------------
# 3. FastAPI Request and Response Schemas
# ------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Incoming user request from frontend to FastAPI."""
    message: str = Field(..., min_length=1, description="User question or query")
    session_id: Optional[str] = Field(None, description="Unique conversation session ID")
    user_id: Optional[str] = Field(None, description="User or customer ID")
    channel: Optional[str] = Field("web", description="Channel: web, mobile, etc.")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session context")


class ChatResponse(BaseModel):
    """Standardized API response returned by FastAPI."""
    session_id: str
    status: str = Field("accepted", description="'accepted' or 'rejected'")
    data: RequestHandlerOutput
    processing_time_ms: float
