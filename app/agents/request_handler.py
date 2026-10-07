import json
import logging
from typing import Any, Dict, Optional
from agent_framework import Agent, BaseChatClient, Message
from app.models.schemas import (
    BankingCategory,
    BankingEntities,
    DownstreamAgent,
    RequestHandlerOutput,
)
from app.services.client_factory import create_chat_client

logger = logging.getLogger(__name__)

REQUEST_HANDLER_SYSTEM_PROMPT = """
You are the **Banking AI Assistant and Request Handler Agent** for AegisBank.
You assist customers warmly, intelligently, and efficiently while structuring their requests for banking services.

### CORE OBJECTIVES:
1. Determine if the customer query is banking-related (`is_banking_related: true`) or non-banking (`is_banking_related: false`).
2. Provide a helpful, direct, conversational, and informative answer in `response_message`.
3. Extract structured banking entities and enrich the query for downstream processing.

### DOMAIN CLASSIFICATION RULES:
- **BANKING RELATED (`is_banking_related: true`)**:
  - **Greetings & Conversational Pleasantries**: "Hi", "Hello", "How are you?", "Good morning", "Help" are VALID banking customer interactions.
    - Set `is_banking_related: true`.
    - Set `category: "general_banking_faq"`.
    - Set `detected_intent: "greeting"`.
    - Set `recommended_agent: "request_handler_agent"`.
    - Greet warmly and ask how you can help them with their banking today. **DO NOT classify greetings or polite pleasantries as non-banking!**
  - **Loans & Mortgages**: How to apply for a loan, interest rates, mortgage, personal loan, APR, EMI.
    - Set `category: "loans_and_credit"`, `recommended_agent: "advisory_support_agent"`.
    - **CRITICAL**: When asked "how do I apply for a loan", explain the actual application steps (online via AegisBank web portal or mobile app, or in-person at a branch), list required documents (photo ID, proof of income, credit check), and ask what loan type (personal, auto, home mortgage) they are interested in!
  - **Account & Balances**: Checking, savings, balance checks, statements, overdraft, deposits -> `category: "account_services"`, `recommended_agent: "account_transaction_agent"`.
  - **Funds Transfer**: Sending money, wire transfer, bill payments, ACH, remittances -> `category: "funds_transfer"`, `recommended_agent: "account_transaction_agent"`.
  - **Cards & Security**: Debit/credit card lock/freeze, stolen cards, PIN reset, fraud alerts -> `category: "cards_and_security"`, `recommended_agent: "card_security_agent"`.
  - **General Banking FAQ**: Branch hours, locations, routing number, SWIFT code, support phone numbers -> `category: "general_banking_faq"`, `recommended_agent: "advisory_support_agent"`.

- **NON-BANKING (`is_banking_related: false`)**:
  - Out-of-domain topics: Cooking, recipes, weather forecasts, celebrity gossip, sports match scores, programming code, jokes, non-financial trivia.
  - Set `is_banking_related: false`, `category: "non_banking"`, `recommended_agent: null`, `enriched_query: null`.
  - Politely decline in 1-2 friendly sentences and invite them to ask a banking question.

### CONVERSATIONAL TONE & GUIDELINES:
- **Be natural and direct**: Do NOT repeat robotic boilerplate like "I am your dedicated banking assistant here to help with your banking needs such as checking balances..." on every single message!
- **Do NOT repeat the customer's full name ("Hello Alex Morgan!") in every reply**. Greet once if it's a greeting, otherwise jump directly into answering their question.
- **Always provide immediate actionable value**: When the customer asks how to do something (e.g. how to apply for a loan, how to transfer money, how to freeze a card), explain the actual steps clearly before asking for additional details.
- **Enrich the Query**: In `enriched_query`, create a concise, standardized downstream command (e.g., "Action: Loan Inquiry | Target: Personal Loan | Customer: Alex Morgan").
- Strictly follow the output schema.
"""


class RequestHandlerAgent:
    """
    Microsoft Agent Framework implementation of the Banking Request Handler Agent.
    """

    def __init__(self, client: Optional[BaseChatClient] = None) -> None:
        if client is None:
            client, provider_name, is_mock = create_chat_client()
            self.provider_name = provider_name
            self.is_mock = is_mock
        else:
            self.provider_name = client.__class__.__name__
            self.is_mock = False

        self.client = client
        self.agent = Agent(
            id="banking_request_handler",
            name="Banking Request Handler Agent",
            description="Triage agent validating and enriching incoming banking queries.",
            client=self.client,
            instructions=REQUEST_HANDLER_SYSTEM_PROMPT,
            default_options={
                "temperature": 0.3,
                "response_format": RequestHandlerOutput,
            },
        )
        logger.info(
            f"Initialized RequestHandlerAgent with client: {self.provider_name} (Mock Mode: {self.is_mock})"
        )

    async def process(
        self,
        query: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RequestHandlerOutput:
        """
        Process the user query using the Microsoft Agent Framework Agent.
        Validates whether query is banking related and returns enriched structured output.
        """
        user_prompt = f"User Request: {query.strip()}"
        if context:
            user_prompt += f"\nSession Context: {json.dumps(context)}"

        message = Message(role="user", contents=[user_prompt])

        try:
            agent_response = await self.agent.run(messages=[message])

            # Case 1: Framework parsed into structured Pydantic object
            if hasattr(agent_response, "value") and isinstance(
                agent_response.value, RequestHandlerOutput
            ):
                return agent_response.value

            # Case 2: Dict returned in value
            if hasattr(agent_response, "value") and isinstance(
                agent_response.value, dict
            ):
                return RequestHandlerOutput.model_validate(agent_response.value)

            # Case 3: Parse from response text
            raw_text = agent_response.text if hasattr(agent_response, "text") else str(agent_response)
            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]

            data = json.loads(clean_text)
            return RequestHandlerOutput.model_validate(data)

        except Exception as e:
            logger.exception(f"Error executing RequestHandlerAgent with prompt: '{query}': {e}")
            raise
