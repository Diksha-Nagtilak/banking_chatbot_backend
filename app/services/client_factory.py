"""
LLM Client Factory for Microsoft Agent Framework.
Strictly initializes real AI model clients (Azure OpenAI / OpenAI).
No mock heuristics or hardcoded responses.
"""
import logging
import re
from typing import Any, Dict, Optional, Tuple
from agent_framework import BaseChatClient
from agent_framework.openai import OpenAIChatClient
from app.config import settings

logger = logging.getLogger(__name__)


def create_chat_client() -> Tuple[BaseChatClient, str, bool]:
    """
    Factory function returning the configured Microsoft Agent Framework chat client,
    the provider name, and is_mock=False.

    Strictly connects to:
    1. Azure OpenAI via Managed Identity (DefaultAzureCredential)
    2. Azure OpenAI via API Key
    3. OpenAI API

    Raises RuntimeError if credentials are not configured.
    """
    # 1. Check Azure AI Foundry / Azure OpenAI Configuration
    if settings.AZURE_OPENAI_ENDPOINT:
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
                raise RuntimeError(f"Azure Managed Identity initialization failed: {e}") from e

        elif settings.AZURE_OPENAI_API_KEY:
            logger.info("Initializing Azure AI Foundry / Azure OpenAI client via API Key.")
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

    # 2. Check OpenAI / OpenAI-compatible Configuration
    if settings.OPENAI_API_KEY or settings.OPENAI_BASE_URL:
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

    # 3. Strictly require real LLM credentials - NO MOCK FALLBACK
    raise RuntimeError(
        "No LLM credentials configured! All answers must come from a real LLM.\n"
        "Please provide your Azure OpenAI credentials in '.env':\n"
        "  AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/\n"
        "  AZURE_OPENAI_API_KEY=<your-azure-api-key>\n"
        "  AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini\n"
        "(or set OPENAI_API_KEY in .env)"
    )
