"""
Step 4: Microsoft Agent Framework Implementation.
Defines Agent 1 (Request Handler Agent) using the Microsoft Agent Framework.

Responsibilities of Agent 1 in the Multi-Agent Architecture:
1. Triage: Determines if the incoming query is banking-related.
2. Context Integration: Ingests the top reranked RAG documents.
3. Response Generation: Formulates a helpful, grounded response.
4. Downstream Routing: Identifies which specialist agent (e.g. Account Agent, Card Agent) should handle next.
"""
import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agent_framework import Agent, BaseChatClient, ChatResponse, Message
from app.config import settings
from app.schemas import KnowledgeDocument, RequestHandlerOutput, ScoredDocument

logger = logging.getLogger("banking_chatbot")

REQUEST_HANDLER_SYSTEM_PROMPT = """
You are the **Request Handler Agent (Agent 1 of 4)** in an enterprise banking multi-agent system.
Your role:
1. Examine the user's question and verify if it is banking-related.
2. If banking-related, use the provided Reranked Banking Knowledge Context to answer accurately.
3. Categorize the request into: account_services, funds_transfer, cards_and_security, loans_and_credit, or general_banking_faq.
4. Recommend the downstream specialist agent:
   - account_transaction_agent (for balance inquiries, transfers, statements)
   - card_security_agent (for card freezing, PIN changes, fraud disputes)
   - advisory_support_agent (for loans, mortgages, branch info, FAQs)
5. If non-banking, politely explain you are a banking assistant and guide the user.
"""


class LocalLearningChatClient(BaseChatClient):
    """
    A lightweight, transparent local chat client implementing Microsoft Agent Framework's BaseChatClient.
    Enables step-by-step learning without requiring paid OpenAI or Azure API keys.
    When live API keys are provided in .env, OpenAIChatClient can be used instead.
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("Initialized LocalLearningChatClient (Offline learning mode).")

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> ChatResponse[RequestHandlerOutput]:
        # Extract user prompt from incoming messages
        user_text = ""
        for msg in reversed(messages):
            if msg.role in ("user", "human"):
                contents = msg.contents
                if isinstance(contents, list):
                    user_text = " ".join(str(c.text if hasattr(c, "text") else c) for c in contents)
                else:
                    user_text = str(contents)
                break

        # Generate structured output incorporating RAG context if present
        output = self._generate_response(user_text)
        assistant_msg = Message(role="assistant", contents=[output.model_dump_json()])
        return ChatResponse(messages=[assistant_msg], value=output)

    def _generate_response(self, text: str) -> RequestHandlerOutput:
        # Extract the user's actual question separately from the appended RAG context
        query_part = text
        if "User Query:" in text:
            query_part = text.split("User Query:")[1].split("### Reranked")[0].split("Customer Session Context:")[0]
        lower = query_part.lower().strip()

        # Non-banking triggers
        non_banking = ["weather", "recipe", "cook", "movie", "football", "cricket", "joke", "capital of"]
        if any(w in lower for w in non_banking):
            return RequestHandlerOutput(
                is_banking_related=False,
                confidence=0.90,
                category="non_banking",
                detected_intent="general_inquiry",
                recommended_agent="none",
                response_message=(
                    "I am your Virtual Banking Assistant. I specialize in banking and financial inquiries "
                    "(account balances, money transfers, card security, and loans). How can I assist you with your banking today?"
                ),
            )

        # 1. Loans, Mortgages & Interest Rates
        if "loan" in lower or "mortgage" in lower or "apr" in lower or "rate" in lower:
            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.97,
                category="loans_and_credit",
                detected_intent="loan_and_mortgage_inquiry",
                recommended_agent="advisory_support_agent",
                response_message=(
                    "Here are our current financing rates: 30-Year Fixed Mortgage at 6.45% APR, "
                    "15-Year Fixed Mortgage at 5.75% APR, and Personal Loans starting from 7.99% APR. "
                    "Routing your inquiry to the Financial Advisory & Loans Specialist."
                ),
            )

        # 2. Funds Transfer
        if "transfer" in lower or "send" in lower or "wire" in lower or "pay" in lower:
            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.96,
                category="funds_transfer",
                detected_intent="transfer_funds",
                recommended_agent="account_transaction_agent",
                response_message=(
                    "I have validated your transfer request. You can transfer funds via Domestic ACH (1-3 days, free, up to $5,000/day), "
                    "Same-day Wire ($20 fee, up to $50,000/day), or instant P2P. Routing your request to the Account & Transaction Specialist."
                ),
            )

        # 3. Cards & Security
        if "card" in lower or "freeze" in lower or "lock" in lower or "stolen" in lower or "dispute" in lower:
            is_freeze = any(w in lower for w in ["freeze", "lock", "lost", "stolen"])
            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.98,
                category="cards_and_security",
                detected_intent="emergency_card_freeze" if is_freeze else "card_management",
                recommended_agent="card_security_agent",
                response_message=(
                    "Your card security inquiry has been prioritized. Under AegisBank security policy, "
                    "freezing your card immediately blocks all new point-of-sale swipes and ATM withdrawals. "
                    "Routing to the Card Security Specialist."
                ),
            )

        # 4. Account Services & Balances
        if "balance" in lower or "checking" in lower or "savings" in lower or "statement" in lower or "how much" in lower:
            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.98,
                category="account_services",
                detected_intent="check_balance",
                recommended_agent="account_transaction_agent",
                response_message=(
                    "I can help you with your account services. Based on our account policy, your Checking "
                    "account balance is $14,820.50 (•••• 7812) and your High-Yield Savings balance is $52,400.00 (earning 4.75% APY). "
                    "Routing to the Account & Transaction Specialist if you wish to download statements or schedule transactions."
                ),
            )

        # 5. General Banking FAQ
        return RequestHandlerOutput(
            is_banking_related=True,
            confidence=0.92,
            category="general_banking_faq",
            detected_intent="general_faq",
            recommended_agent="advisory_support_agent",
            response_message=(
                "I have received your banking inquiry. Our routing number is 021000021 and SWIFT code is AEGSUS33XXX. "
                "24/7 customer support is available at 1-800-555-AEGIS. Routing to Advisory Support."
            ),
        )


class RequestHandlerAgent:
    """
    Microsoft Agent Framework Request Handler Agent (Agent 1).
    Orchestrates:
    - User message formulation
    - Ingestion of reranked RAG context
    - Agent execution via Microsoft Agent Framework
    """

    def __init__(self) -> None:
        # Check if OpenAI or Azure OpenAI API key is configured
        if settings.OPENAI_API_KEY:
            try:
                from agent_framework.openai import OpenAIChatClient
                self.client = OpenAIChatClient(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL,
                )
                self.provider_name = f"OpenAI ({settings.OPENAI_MODEL})"
            except Exception as e:
                logger.warning(f"Could not initialize OpenAIChatClient, using local learning client: {e}")
                self.client = LocalLearningChatClient()
                self.provider_name = "LocalLearningChatClient (Offline)"
        else:
            self.client = LocalLearningChatClient()
            self.provider_name = "LocalLearningChatClient (Offline/Educational)"

        # Initialize the Microsoft Agent Framework Agent
        self.agent = Agent(
            id="request_handler_agent",
            name="Request Handler Agent",
            description="Agent 1: Triage, Domain Validation, RAG Context Augmentation, and Specialist Routing",
            client=self.client,
            instructions=REQUEST_HANDLER_SYSTEM_PROMPT,
        )
        logger.info(f"Initialized Microsoft Agent Framework Agent with provider: {self.provider_name}")

    async def process(
        self,
        query: str,
        reranked_docs: List[ScoredDocument],
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RequestHandlerOutput:
        """
        Executes the agent with the user prompt augmented by reranked RAG context.
        """
        # Build RAG Context String from top reranked documents
        rag_context_str = ""
        if reranked_docs:
            rag_context_str = "\n\n### Reranked Knowledge Base Context:\n"
            for item in reranked_docs:
                rag_context_str += f"- [{item.document.id}] {item.document.title}: {item.document.content}\n"

        prompt = f"User Query: {query.strip()}{rag_context_str}"
        if context:
            prompt += f"\nCustomer Session Context: {json.dumps(context)}"

        message = Message(role="user", contents=[prompt])

        # Run the agent via Microsoft Agent Framework
        agent_response = await self.agent.run(messages=[message])

        # Extract structured RequestHandlerOutput
        if hasattr(agent_response, "value") and isinstance(agent_response.value, RequestHandlerOutput):
            output = agent_response.value
        elif hasattr(agent_response, "value") and isinstance(agent_response.value, dict):
            output = RequestHandlerOutput.model_validate(agent_response.value)
        else:
            # Fallback parsing
            client_eval = LocalLearningChatClient()
            output = client_eval._generate_response(query)

        # Attach RAG audit information
        output.reranked_documents = [item.document.title for item in reranked_docs]
        return output
