import pytest
from app.config import settings


@pytest.fixture(autouse=True)
def configure_test_environment():
    """Ensure tests run safely using Mock mode so CI/CD doesn't require live API keys."""
    previous_mode = settings.MOCK_LLM_MODE
    settings.MOCK_LLM_MODE = True
    yield
    settings.MOCK_LLM_MODE = previous_mode
