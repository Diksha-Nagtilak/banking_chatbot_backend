import json
import re
import pytest
from agent_framework import Agent
from app.config import settings
from app.models.schemas import (
    BankingCategory,
    BankingEntities,
    ContentGeneratorOutput,
    DownstreamAgent,
    RequestHandlerOutput,
    RetrievedChunk,
    UrgencyLevel,
)


class MockAgentResponse:
    """Mock response imitating agent_framework's AgentResponse."""
    def __init__(self, value):
        self.value = value
        self.text = json.dumps(value.model_dump() if hasattr(value, "model_dump") else str(value))


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """
    Ensure unit tests run safely in complete isolation without requiring
    live network connections to Azure AI Search or Azure OpenAI.
    """

    # 1. Mock Azure AI Search retrieval
    def mock_azure_search(query: str, top_k=None, category=None):
        return [
            RetrievedChunk(
                id="TEST_001",
                title="Checking & Savings Account Overview",
                content="AegisBank High-Yield Savings accounts earn an APY of 4.75% with interest compounded daily.",
                retrieval_score=5.0,
            ),
            RetrievedChunk(
                id="TEST_002",
                title="Mortgage, Personal Loan & APR Interest Rates",
                content="30-Year Fixed Mortgage: 6.45% APR. Personal loans starting from 7.99% APR.",
                retrieval_score=4.0,
            ),
            RetrievedChunk(
                id="TEST_003",
                title="Funds Transfer Limits & Fees",
                content="Domestic ACH wire transfer daily limits: $50,000 with a $20 fee.",
                retrieval_score=3.5,
            ),
        ], "Mock Azure AI Search (Test Isolation)"

    monkeypatch.setattr("app.rag.azure_search.search_azure_ai_search", mock_azure_search)
    monkeypatch.setattr("app.agents.analyzer.search_azure_ai_search", mock_azure_search)

    # 2. Mock Agent.run for offline deterministic unit testing
    async def mock_agent_run(self, messages=None, **kwargs):
        prompt = ""
        if messages and len(messages) > 0:
            prompt = str(messages[0].contents[0])
        prompt_lower = prompt.lower()

        # Agent 3: Content Generator
        if self.id == "banking_content_generator":
            ans = (
                "AegisBank High-Yield Savings accounts earn an APY of 4.75% with interest compounded daily. "
                "Current 30-year fixed home mortgage interest rate is 6.45% APR."
            )
            return MockAgentResponse(
                ContentGeneratorOutput(
                    answer=ans,
                    grounded_sources=["Checking & Savings Account Overview", "Mortgage, Personal Loan & APR Interest Rates"],
                    confidence=0.98,
                )
            )

        # Agent 1: Request Handler
        # Non-banking checks
        non_banking_triggers = ["recipe", "cake", "bake", "python", "programming", "weather", "capital", "france", "soccer"]
        if any(w in prompt_lower for w in non_banking_triggers):
            output = RequestHandlerOutput(
                is_banking_related=False,
                confidence=0.98,
                category=BankingCategory.NON_BANKING,
                detected_intent=None,
                recommended_agent=None,
                entities=BankingEntities(),
                enriched_query=None,
                response_message="I am the AegisBank Virtual Assistant. I can only assist with banking inquiries.",
            )
            return MockAgentResponse(output)

        # Checking / Balance
        if "balance" in prompt_lower or "checking account" in prompt_lower:
            output = RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.99,
                category=BankingCategory.ACCOUNT_SERVICES,
                detected_intent="check_balance",
                recommended_agent=DownstreamAgent.ACCOUNT_TRANSACTION_AGENT,
                entities=BankingEntities(account_type="checking"),
                enriched_query="Action: check_balance | Account: checking | Subject: balance inquiry",
                response_message="Your checking account balance inquiry has been routed to Account Services.",
            )
            return MockAgentResponse(output)

        # Funds Transfer
        if "transfer" in prompt_lower or "send" in prompt_lower:
            amount_match = re.search(r"(\d+(?:\.\d+)?)", prompt)
            amt = float(amount_match.group(1)) if amount_match else 100.0
            recipient = "Emily" if "emily" in prompt_lower else ("Robert" if "robert" in prompt_lower else "Recipient")
            output = RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.99,
                category=BankingCategory.FUNDS_TRANSFER,
                detected_intent="transfer_funds",
                recommended_agent=DownstreamAgent.ACCOUNT_TRANSACTION_AGENT,
                entities=BankingEntities(amount=amt, recipient=recipient),
                enriched_query=f"Action: transfer_funds | Amount: {amt} | Recipient: {recipient}",
                response_message=f"Initiating funds transfer of ${amt} to {recipient}.",
            )
            return MockAgentResponse(output)

        # Card Security
        if "stolen" in prompt_lower or "freeze" in prompt_lower or "lost" in prompt_lower or "card" in prompt_lower:
            card_match = re.search(r"(\d{4})", prompt)
            last_four = card_match.group(1) if card_match else "4455"
            output = RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.99,
                category=BankingCategory.CARDS_AND_SECURITY,
                detected_intent="freeze_card",
                recommended_agent=DownstreamAgent.CARD_SECURITY_AGENT,
                entities=BankingEntities(urgency=UrgencyLevel.CRITICAL, card_last_four=last_four),
                enriched_query=f"Action: freeze_card | Card: {last_four} | Urgency: CRITICAL",
                response_message=f"Your card ending in {last_four} has been secured and frozen immediately.",
            )
            return MockAgentResponse(output)

        # Loans & Mortgages
        if "mortgage" in prompt_lower or "loan" in prompt_lower:
            output = RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.95,
                category=BankingCategory.LOANS_AND_CREDIT,
                detected_intent="loan_inquiry",
                recommended_agent=DownstreamAgent.ADVISORY_SUPPORT_AGENT,
                entities=BankingEntities(),
                enriched_query="Action: loan_inquiry | Topic: home mortgage and personal loan rates",
                response_message="Here are the current rates for 30-year fixed home mortgage loans.",
            )
            return MockAgentResponse(output)

        # Default Banking
        output = RequestHandlerOutput(
            is_banking_related=True,
            confidence=0.95,
            category=BankingCategory.GENERAL_BANKING_FAQ,
            detected_intent="general_faq",
            recommended_agent=DownstreamAgent.ADVISORY_SUPPORT_AGENT,
            entities=BankingEntities(),
            enriched_query=f"Action: general_faq | Query: {prompt}",
            response_message="Thank you for contacting AegisBank customer service.",
        )
        return MockAgentResponse(output)

    monkeypatch.setattr(Agent, "run", mock_agent_run)
