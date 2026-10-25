from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration supporting local .env files and Azure Web App Application Settings.
    """
    # Application Info
    APP_NAME: str = "Enterprise Banking Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4o-mini"
    AZURE_OPENAI_API_VERSION: str = "2024-10-21"
    AZURE_OPENAI_USE_MANAGED_IDENTITY: bool = False

    # Standard OpenAI Fallback Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: Optional[str] = None

    # Azure AI Search Configuration (Agent 2: Analyzer Agent)
    AZURE_SEARCH_ENDPOINT: Optional[str] = None
    AZURE_SEARCH_API_KEY: Optional[str] = None
    AZURE_SEARCH_INDEX_NAME: Optional[str] = None
    AZURE_SEARCH_CONTENT_FIELD: str = "content"
    AZURE_SEARCH_TITLE_FIELD: str = "title"
    AZURE_SEARCH_CATEGORY_FIELD: str = "category"
    AZURE_SEARCH_TOP_K: int = 5
    AZURE_SEARCH_USE_MANAGED_IDENTITY: bool = False

    # Cross-Encoder Reranker Configuration
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_N: int = 3
    ENABLE_CROSS_ENCODER: bool = True


    # CORS Configuration (comma-separated origins)
    ALLOWED_ORIGINS: str = "*"

    model_config = SettingsConfigDict(
        env_file=(str(Path(__file__).resolve().parent.parent / ".env") if (Path(__file__).resolve().parent.parent / ".env").exists() else ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


settings = Settings()