from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["framework"] == "Microsoft Agent Framework"
    assert "agents" in data


def test_health_check_endpoint():
    # Lifespan will trigger via TestClient context manager or directly
    with TestClient(app) as test_cli:
        response = test_cli.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "Microsoft Agent Framework" in data["agent_framework"]
        assert data["active_agent"] == "RequestHandlerAgent (Agent 1/4)"


def test_chat_banking_accepted():
    with TestClient(app) as test_cli:
        payload = {
            "message": "Transfer 100 dollars to Emily for groceries",
            "session_id": "test-session-123"
        }
        response = test_cli.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["status"] == "accepted"
        assert data["data"]["is_banking_related"] is True
        assert data["data"]["category"] == "funds_transfer"
        assert data["data"]["entities"]["amount"] == 100.0
        assert data["data"]["entities"]["recipient"] == "Emily"
        assert data["data"]["recommended_agent"] == "account_transaction_agent"
        assert data["processing_time_ms"] > 0


def test_chat_non_banking_rejected():
    with TestClient(app) as test_cli:
        payload = {
            "message": "What is the capital of France and who won the soccer match?",
            "session_id": "test-session-456"
        }
        response = test_cli.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["data"]["is_banking_related"] is False
        assert data["data"]["category"] == "non_banking"
        assert data["data"]["recommended_agent"] is None
        assert "Virtual Banking Assistant" in data["data"]["response_message"]


def test_direct_enrich_endpoint():
    with TestClient(app) as test_cli:
        payload = {"query": "I lost my credit card ending with 1234"}
        response = test_cli.post("/api/v1/enrich", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_banking_related"] is True
        assert data["category"] == "cards_and_security"
        assert data["entities"]["card_last_four"] == "1234"
        assert data["recommended_agent"] == "card_security_agent"


def test_agents_topology_endpoint():
    response = client.get("/api/v1/agents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_agents"] == 4
    assert len(data["agents"]) == 4
    assert data["agents"][0]["id"] == "request_handler_agent"
    assert data["agents"][0]["status"] == "ACTIVE"
