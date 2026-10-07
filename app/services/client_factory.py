import logging
import re
from typing import Any, Mapping, Optional, Sequence, Tuple
from agent_framework import (
    BaseChatClient,
    ChatResponse,
    Message,
)
from agent_framework.openai import OpenAIChatClient
from app.config import settings
from app.models.schemas import (
    BankingCategory,
    BankingEntities,
    DownstreamAgent,
    RequestHandlerOutput,
    UrgencyLevel,
)

logger = logging.getLogger(__name__)


class MockBankingChatClient(BaseChatClient):
    """
    Intelligent heuristic fallback client implementing Microsoft Agent Framework's BaseChatClient.
    Enables local testing, automated CI/CD tests, and seamless Azure Web App startup
    before live Azure OpenAI keys are populated.
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("Initialized MockBankingChatClient for local demo and offline mode.")

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> ChatResponse[RequestHandlerOutput]:
        # Extract user text from messages
        user_text = ""
        for msg in reversed(messages):
            if msg.role in ("user", "human"):
                if isinstance(msg.contents, list):
                    user_text = " ".join(
                        str(c.text if hasattr(c, "text") else c) for c in msg.contents
                    )
                else:
                    user_text = str(msg.contents)
                break

        output = self._evaluate_query(user_text)
        assistant_message = Message(
            role="assistant",
            contents=[output.model_dump_json()]
        )
        return ChatResponse(messages=[assistant_message], value=output)

    def _evaluate_query(self, text: str) -> RequestHandlerOutput:
        # Separate the actual user query from Session Context metadata
        query_text = text
        customer_name = "there"
        if "User Request:" in text:
            parts = text.split("User Request:")[1].split("\nSession Context:")
            query_text = parts[0].strip()
            if len(parts) > 1:
                try:
                    import json
                    ctx = json.loads(parts[1].strip())
                    customer_name = ctx.get("customerName", "there")
                except Exception:
                    pass

        lower = query_text.lower().strip()

        # Handle greetings & introductory messages
        greeting_words = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy"}
        if lower in greeting_words or any(lower.startswith(w + " ") for w in greeting_words):
            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.99,
                category=BankingCategory.GENERAL_BANKING_FAQ,
                detected_intent="greeting_and_guidance",
                entities=BankingEntities(),
                enriched_query=f"Action: customer_greeting | User: {customer_name}",
                missing_information=[],
                recommended_agent=DownstreamAgent.REQUEST_HANDLER_AGENT,
                response_message=(
                    f"Hello {customer_name}! I am your Virtual Banking Assistant. "
                    "I can help you check your account balances, transfer funds, manage or freeze your debit/credit cards, "
                    "or answer loan and mortgage questions. How can I assist you with your banking today?"
                ),
            )

        # Non-banking indicators
        non_banking_triggers = [
            "recipe", "cook", "weather", "temperature", "forecast", "movie",
            "film", "sports", "football", "cricket", "code", "python",
            "javascript", "poem", "story", "joke", "capital of", "who is",
            "planet", "biology", "fitness", "gym"
        ]
        if any(trigger in lower for trigger in non_banking_triggers):
            return RequestHandlerOutput(
                is_banking_related=False,
                confidence=0.98,
                category=BankingCategory.NON_BANKING,
                detected_intent=None,
                entities=BankingEntities(),
                enriched_query=None,
                missing_information=[],
                recommended_agent=None,
                response_message=(
                    "I am the Virtual Banking Assistant. I specialize in banking and financial "
                    "services including account balance checks, funds transfers, debit/credit cards, "
                    "statements, and loan services. I cannot assist with non-banking inquiries. "
                    "How can I help you with your banking needs today?"
                ),
            )

        # Entity extraction helpers
        entities = BankingEntities()

        # Account type extraction
        if "checking" in lower:
            entities.account_type = "checking"
        elif "savings" in lower:
            entities.account_type = "savings"
        elif "credit card" in lower:
            entities.account_type = "credit_card"
        elif "loan" in lower or "mortgage" in lower:
            entities.account_type = "loan"

        # Amount and Currency extraction
        amount_match = re.search(r'(\$|usd|eur|gbp|inr|rs\.?|dollars?|bucks)?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(\$|usd|eur|gbp|inr|dollars?|bucks)?', lower)
        if amount_match:
            try:
                num_str = amount_match.group(2).replace(",", "")
                val = float(num_str)
                # Filter out obvious 4-digit card numbers or years unless prefixed with currency
                prefix = amount_match.group(1) or ""
                suffix = amount_match.group(3) or ""
                if prefix or suffix or ("transfer" in lower or "send" in lower or "pay" in lower):
                    entities.amount = val
                    curr_token = f"{prefix} {suffix}".lower()
                    if any(c in curr_token for c in ["eur", "euro", "€"]):
                        entities.currency = "EUR"
                    elif any(c in curr_token for c in ["gbp", "pound", "£"]):
                        entities.currency = "GBP"
                    elif "inr" in curr_token or "rupee" in curr_token or "₹" in curr_token or re.search(r'\brs\.?\b', curr_token):
                        entities.currency = "INR"
                    elif any(c in curr_token for c in ["dollar", "usd", "$", "buck"]):
                        entities.currency = "USD"
                    else:
                        entities.currency = "USD"
            except ValueError:
                pass

        # Card digits extraction
        card_match = re.search(r'(?:card\s*(?:ending\s*(?:in|with)?|number)?\s*#?\s*)(\d{4})\b', lower)
        if card_match:
            entities.card_last_four = card_match.group(1)

        # Recipient extraction (e.g., "to John", "to mom", "to Alice Smith")
        recipient_match = re.search(r'\b(?:to|pay)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b', text)
        if recipient_match:
            raw_rec = recipient_match.group(1).strip()
            stop_preps = {"for", "from", "via", "on", "at", "with", "using", "my", "the", "a", "an", "checking", "savings", "account"}
            cleaned_words = [w for w in raw_rec.split() if w.lower() not in stop_preps]
            if cleaned_words:
                entities.recipient = " ".join(cleaned_words)

        # Check Funds Transfer
        transfer_keywords = ["send", "transfer", "wire", "remit", "pay", "deposit money"]
        if any(kw in lower for kw in transfer_keywords) and (entities.amount or entities.recipient or "money" in lower or "funds" in lower or "$" in lower):
            missing = []
            if not entities.amount:
                missing.append("transfer_amount")
            if not entities.recipient:
                missing.append("recipient_name_or_account")
            if not entities.account_type:
                missing.append("source_account_type")

            urgency = UrgencyLevel.HIGH if "urgent" in lower or "immediately" in lower else UrgencyLevel.NORMAL
            entities.urgency = urgency

            enriched = (
                f"Action: Funds Transfer | "
                f"Amount: {entities.amount or 'UNSPECIFIED'} {entities.currency or 'USD'} | "
                f"Recipient: {entities.recipient or 'UNSPECIFIED'} | "
                f"Source Account: {entities.account_type or 'DEFAULT_PRIMARY'} | "
                f"Original Request: '{query_text}'"
            )

            resp_msg = (
                f"I will assist you in transferring {entities.amount or ''} {entities.currency or ''} "
                f"{f'to {entities.recipient}' if entities.recipient else ''}. "
                f"{'Please provide the ' + ', '.join(missing) if missing else 'Ready to route to Account & Transaction Specialist.'}"
            ).strip()

            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.96,
                category=BankingCategory.FUNDS_TRANSFER,
                detected_intent="transfer_funds",
                entities=entities,
                enriched_query=enriched,
                missing_information=missing,
                recommended_agent=DownstreamAgent.ACCOUNT_TRANSACTION_AGENT,
                response_message=resp_msg,
            )

        # Check Cards and Security
        card_keywords = ["card", "stolen", "lost", "freeze", "block", "pin", "cvv", "unauthorized", "fraud", "skimmed"]
        if any(kw in lower for kw in card_keywords):
            is_critical = any(kw in lower for kw in ["stolen", "lost", "fraud", "unauthorized", "block", "freeze"])
            entities.urgency = UrgencyLevel.CRITICAL if is_critical else UrgencyLevel.HIGH

            missing = []
            if not entities.card_last_four:
                missing.append("card_last_four_digits")

            intent = "block_lost_stolen_card" if is_critical else "card_management_inquiry"
            enriched = (
                f"Action: {intent} | Urgency: {entities.urgency.value.upper()} | "
                f"Card Identifier: {entities.card_last_four or 'UNSPECIFIED'} | "
                f"Issue Description: '{query_text}'"
            )

            resp_msg = (
                "Your card security request has been prioritized with critical urgency. "
                "I am preparing the card security agent to secure your account immediately. "
                + ("Please confirm the last 4 digits of the card if available." if "card_last_four_digits" in missing else "")
            ).strip()

            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.98,
                category=BankingCategory.CARDS_AND_SECURITY,
                detected_intent=intent,
                entities=entities,
                enriched_query=enriched,
                missing_information=missing,
                recommended_agent=DownstreamAgent.CARD_SECURITY_AGENT,
                response_message=resp_msg,
            )

        # Check Balance & Account Services
        account_keywords = ["balance", "statement", "account", "checking", "savings", "overdraft", "deposit", "transaction history"]
        if any(kw in lower for kw in account_keywords):
            intent = "view_statement" if "statement" in lower or "history" in lower else "check_balance"
            enriched = (
                f"Action: {intent} | "
                f"Target Account: {entities.account_type or 'all_accounts'} | "
                f"Timeframe: {entities.date_or_period or 'current'} | "
                f"Original: '{query_text}'"
            )
            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.97,
                category=BankingCategory.ACCOUNT_SERVICES,
                detected_intent=intent,
                entities=entities,
                enriched_query=enriched,
                missing_information=[],
                recommended_agent=DownstreamAgent.ACCOUNT_TRANSACTION_AGENT,
                response_message=f"I can help you check your {entities.account_type or 'account'} {intent.replace('_', ' ')}. Routing your request to the Account Specialist.",
            )

        # Check Loans & Credit
        loan_keywords = ["loan", "mortgage", "borrow", "emi", "interest rate", "apr", "car loan", "home loan", "credit limit"]
        if any(kw in lower for kw in loan_keywords):
            enriched = f"Action: loan_and_credit_inquiry | Topic: '{text}' | Amount Requested: {entities.amount or 'N/A'}"
            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.95,
                category=BankingCategory.LOANS_AND_CREDIT,
                detected_intent="loan_inquiry",
                entities=entities,
                enriched_query=enriched,
                missing_information=[],
                recommended_agent=DownstreamAgent.ADVISORY_SUPPORT_AGENT,
                response_message="I have received your loan and credit inquiry. Connecting you to our Financial Advisory and Loans Specialist.",
            )

        # General Banking FAQ / Branch / Policy
        faq_keywords = ["branch", "atm", "hours", "swift", "routing number", "bank open", "wire instructions", "fee", "customer service"]
        if any(kw in lower for kw in faq_keywords):
            return RequestHandlerOutput(
                is_banking_related=True,
                confidence=0.92,
                category=BankingCategory.GENERAL_BANKING_FAQ,
                detected_intent="general_faq_inquiry",
                entities=entities,
                enriched_query=f"Action: bank_information_inquiry | Query: '{text}'",
                missing_information=[],
                recommended_agent=DownstreamAgent.ADVISORY_SUPPORT_AGENT,
                response_message="I would be happy to help answer your banking information inquiry. Let me retrieve those details for you.",
            )

        # Default fallback: If ambiguous and does not match banking, treat as non-banking
        return RequestHandlerOutput(
            is_banking_related=False,
            confidence=0.85,
            category=BankingCategory.NON_BANKING,
            detected_intent=None,
            entities=BankingEntities(),
            enriched_query=None,
            missing_information=[],
            recommended_agent=None,
            response_message=(
                "I am your Virtual Banking Assistant. I didn't recognize that as a banking-related request. "
                "I can assist you with your account balances, money transfers, card controls, and loan services. "
                "Could you please rephrase or let me know what banking service you need?"
            ),
        )


def create_chat_client() -> Tuple[BaseChatClient, str, bool]:
    """
    Factory function returning the configured Microsoft Agent Framework chat client,
    the provider name, and whether mock mode is active.
    Supports:
    1. Azure OpenAI with Managed Identity (DefaultAzureCredential)
    2. Azure OpenAI with API Key
    3. OpenAI API
    4. MockBankingChatClient for local demo and test safety
    """
    # 1. Check Azure AI Foundry / Azure OpenAI Configuration
    if settings.AZURE_OPENAI_ENDPOINT and not settings.MOCK_LLM_MODE:
        raw_endpoint = settings.AZURE_OPENAI_ENDPOINT.strip().rstrip("/")

        if settings.AZURE_OPENAI_USE_MANAGED_IDENTITY:
            try:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
                clean_endpoint = re.sub(r"/openai(?:/v1)?/?$", "", raw_endpoint)
                logger.info("Initializing Azure OpenAI client via Managed Identity (DefaultAzureCredential).")
                client = OpenAIChatClient(
                    azure_endpoint=clean_endpoint,
                    credential=credential,
                    model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                )
                return client, f"Azure OpenAI ({settings.AZURE_OPENAI_DEPLOYMENT_NAME} via Managed Identity)", False
            except Exception as e:
                logger.error(f"Failed to initialize Azure Managed Identity client: {e}")

        elif settings.AZURE_OPENAI_API_KEY:
            logger.info("Initializing Azure AI Foundry / OpenAI client via API Key.")
            v1_url = raw_endpoint
            if not v1_url.endswith("/openai/v1"):
                base_clean = re.sub(r"/openai(?:/v1)?/?$", "", raw_endpoint)
                v1_url = f"{base_clean}/openai/v1"

            client = OpenAIChatClient(
                base_url=v1_url,
                api_key=settings.AZURE_OPENAI_API_KEY,
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            )
            return client, f"Azure AI Foundry ({settings.AZURE_OPENAI_DEPLOYMENT_NAME})", False

    # 2. Check OpenAI / OpenAI-compatible Configuration (OpenAI, Groq, Ollama, OpenRouter, etc.)
    if (settings.OPENAI_API_KEY or settings.OPENAI_BASE_URL) and not settings.MOCK_LLM_MODE:
        logger.info(f"Initializing OpenAI client (Model: {settings.OPENAI_MODEL}).")
        client_kwargs: Dict[str, Any] = {
            "model": settings.OPENAI_MODEL,
        }
        if settings.OPENAI_API_KEY:
            client_kwargs["api_key"] = settings.OPENAI_API_KEY
        if settings.OPENAI_BASE_URL:
            client_kwargs["base_url"] = settings.OPENAI_BASE_URL

        client = OpenAIChatClient(**client_kwargs)
        provider_name = f"OpenAI ({settings.OPENAI_MODEL})"
        if settings.OPENAI_BASE_URL:
            provider_name += f" @ {settings.OPENAI_BASE_URL}"
        return client, provider_name, False

    # 3. Only use mock client if MOCK_LLM_MODE is explicitly set to True
    if settings.MOCK_LLM_MODE:
        logger.info("Using built-in MockBankingChatClient (Mock Mode explicitly enabled).")
        return MockBankingChatClient(), "Mock Banking Engine (Offline Mode)", True

    # Otherwise, strictly require real LLM credentials to prevent hardcoded responses
    raise RuntimeError(
        "No LLM credentials configured! All answers must come from a real LLM.\n"
        "Please provide your Azure OpenAI credentials in '.env':\n"
        "  AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/\n"
        "  AZURE_OPENAI_API_KEY=<your-azure-api-key>\n"
        "  AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini\n"
        "(or set OPENAI_API_KEY in .env)"
    )
