import pytest
from app.agents.request_handler import RequestHandlerAgent
from app.models.schemas import (
    BankingCategory,
    DownstreamAgent,
    UrgencyLevel,
)


@pytest.fixture
def agent():
    return RequestHandlerAgent()


@pytest.mark.asyncio
async def test_balance_inquiry_accepted(agent):
    query = "What is the available balance in my checking account?"
    result = await agent.process(query)

    assert result.is_banking_related is True
    assert result.category == BankingCategory.ACCOUNT_SERVICES
    assert result.detected_intent == "check_balance"
    assert result.recommended_agent == DownstreamAgent.ACCOUNT_TRANSACTION_AGENT
    assert result.entities.account_type == "checking"
    assert "checking" in result.enriched_query.lower()


@pytest.mark.asyncio
async def test_funds_transfer_enrichment(agent):
    query = "Transfer 150 dollars to Robert"
    result = await agent.process(query)

    assert result.is_banking_related is True
    assert result.category == BankingCategory.FUNDS_TRANSFER
    assert result.detected_intent == "transfer_funds"
    assert result.recommended_agent == DownstreamAgent.ACCOUNT_TRANSACTION_AGENT
    assert result.entities.amount == 150.0
    assert result.entities.recipient == "Robert"
    assert result.enriched_query is not None
    assert "150.0 USD" in result.enriched_query


@pytest.mark.asyncio
async def test_lost_card_critical_urgency(agent):
    query = "My credit card was stolen! Card ending in 4455, please freeze it!"
    result = await agent.process(query)

    assert result.is_banking_related is True
    assert result.category == BankingCategory.CARDS_AND_SECURITY
    assert result.entities.urgency == UrgencyLevel.CRITICAL
    assert result.entities.card_last_four == "4455"
    assert result.recommended_agent == DownstreamAgent.CARD_SECURITY_AGENT
    assert "CRITICAL" in result.enriched_query


@pytest.mark.asyncio
async def test_loan_inquiry(agent):
    query = "What are the interest rates for a 30-year fixed home mortgage loan?"
    result = await agent.process(query)

    assert result.is_banking_related is True
    assert result.category == BankingCategory.LOANS_AND_CREDIT
    assert result.detected_intent == "loan_inquiry"
    assert result.recommended_agent == DownstreamAgent.ADVISORY_SUPPORT_AGENT


@pytest.mark.asyncio
async def test_non_banking_recipe_rejection(agent):
    query = "How do I bake a strawberry cheesecake?"
    result = await agent.process(query)

    assert result.is_banking_related is False
    assert result.category == BankingCategory.NON_BANKING
    assert result.detected_intent is None
    assert result.recommended_agent is None
    assert result.enriched_query is None
    assert "Virtual Banking Assistant" in result.response_message


@pytest.mark.asyncio
async def test_non_banking_programming_rejection(agent):
    query = "Write a python function to sort an array."
    result = await agent.process(query)

    assert result.is_banking_related is False
    assert result.category == BankingCategory.NON_BANKING
    assert result.recommended_agent is None
    assert result.enriched_query is None


@pytest.mark.asyncio
async def test_non_banking_weather_rejection(agent):
    query = "What is the weather forecast for New York today?"
    result = await agent.process(query)

    assert result.is_banking_related is False
    assert result.category == BankingCategory.NON_BANKING
    assert result.recommended_agent is None
