from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BankingCategory(str, Enum):
    ACCOUNT_SERVICES = "account_services"
    FUNDS_TRANSFER = "funds_transfer"
    CARDS_AND_SECURITY = "cards_and_security"
    LOANS_AND_CREDIT = "loans_and_credit"
    GENERAL_BANKING_FAQ = "general_banking_faq"
    NON_BANKING = "non_banking"


class DownstreamAgent(str, Enum):
    ACCOUNT_TRANSACTION_AGENT = "account_transaction_agent"
    CARD_SECURITY_AGENT = "card_security_agent"
    ADVISORY_SUPPORT_AGENT = "advisory_support_agent"
    REQUEST_HANDLER_AGENT = "request_handler_agent"


class UrgencyLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class BankingEntities(BaseModel):
    """Entities extracted from user banking query."""
    account_type: Optional[str] = Field(None, description="Account type, e.g. checking, savings, credit, loan")
    amount: Optional[float] = Field(None, description="Numeric money amount involved")
    currency: Optional[str] = Field(None, description="Currency ISO code or symbol, e.g. USD, EUR, INR")
    recipient: Optional[str] = Field(None, description="Recipient person name, business name, or account ID")
    card_last_four: Optional[str] = Field(None, description="Last 4 digits of referenced debit/credit card")
    date_or_period: Optional[str] = Field(None, description="Timeframe or date range, e.g. 'last month', 'today'")
    transaction_id: Optional[str] = Field(None, description="Reference code or transaction identifier")
    urgency: UrgencyLevel = Field(UrgencyLevel.NORMAL, description="Urgency classification")


class RequestHandlerOutput(BaseModel):
    """
    Standard output produced by the Request Handler Agent.
    Evaluates if query is banking related, enriches it, and plans routing for 4-agent ecosystem.
    """
    is_banking_related: bool = Field(
        ...,
        description="True if query belongs to banking/financial operations, False otherwise"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0.0 and 1.0"
    )
    category: BankingCategory = Field(
        ...,
        description="High level category classification"
    )
    detected_intent: Optional[str] = Field(
        None,
        description="Specific intent (e.g. check_balance, transfer_money, block_card, loan_inquiry, view_statement)"
    )
    entities: BankingEntities = Field(
        default_factory=BankingEntities,
        description="Extracted structured banking parameters"
    )
    enriched_query: Optional[str] = Field(
        None,
        description="Synthesized, unambiguous, and enriched query for downstream specialized agents"
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Information needed to fulfill the action that was not provided in the prompt"
    )
    recommended_agent: Optional[DownstreamAgent] = Field(
        None,
        description="Which of the 4 agents in the planned banking architecture should process next"
    )
    response_message: str = Field(
        ...,
        description="Conversational customer-facing response message (refusal or acknowledgment)"
    )


class ChatRequest(BaseModel):
    """Incoming user chat request to the API."""
    message: str = Field(..., min_length=1, description="The user prompt or question")
    session_id: Optional[str] = Field(None, description="Unique conversation session identifier")
    user_id: Optional[str] = Field(None, description="Authenticated customer identifier if available")
    channel: Optional[str] = Field("web", description="Originating channel: web, mobile, ivr")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Pre-existing session context")


class ChatResponse(BaseModel):
    """Standardized API response."""
    session_id: str = Field(..., description="Active session ID")
    status: str = Field(..., description="'accepted', 'rejected', or 'error'")
    data: RequestHandlerOutput = Field(..., description="Processed agent output")
    processing_time_ms: float = Field(..., description="Execution duration in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of response")


class HealthResponse(BaseModel):
    """Health probe response for Azure App Service."""
    status: str
    service: str
    version: str
    agent_framework: str
    active_agent: str
    llm_provider: str
    mock_mode: bool


class EnrichRequest(BaseModel):
    """Direct query enrichment request."""
    query: str = Field(..., min_length=1, description="Raw customer query to triage and enrich")
