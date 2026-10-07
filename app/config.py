from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


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

    # Standard OpenAI Fallback Configuration (Supports OpenAI, Groq, Ollama, OpenRouter, etc.)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: Optional[str] = None

    # Development & Demo Fallback
    # If True or if no API keys are supplied, the app uses the built-in Intelligent Mock Engine
    MOCK_LLM_MODE: bool = False

    # CORS Configuration (comma-separated origins)
    ALLOWED_ORIGINS: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


settings = Settings()
