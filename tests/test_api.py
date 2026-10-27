from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["framework"] == "Microsoft Agent Framework"
    assert "pipeline" in data


def test_health_check_endpoint():
    with TestClient(app) as test_cli:
        response = test_cli.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "Microsoft Agent Framework" in data["agent_framework"]
        assert "RequestHandlerAgent" in data["active_agent"]
        assert "AnalyzerAgent" in data["active_agent"]


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
        assert data["processing_time_ms"] > 0
        assert "pipeline" in data
        assert "agent_1_request_handler" in data["pipeline"]
        assert "agent_2_analyzer" in data["pipeline"]


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
        assert len(data["data"]["response_message"]) > 10


def test_direct_enrich_endpoint():
    with TestClient(app) as test_cli:
        payload = {"query": "I lost my credit card ending with 1234"}
        response = test_cli.post("/api/v1/enrich", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_banking_related"] is True
        assert data["category"] == "cards_and_security"
        assert data["entities"]["card_last_four"] == "1234"


def test_direct_analyze_endpoint():
    with TestClient(app) as test_cli:
        payload = {"query": "What are the wire transfer fees and daily limits?"}
        response = test_cli.post("/api/v1/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What are the wire transfer fees and daily limits?"
        assert data["retrieved_count"] > 0
        assert len(data["reranked_chunks"]) > 0
        top = data["reranked_chunks"][0]
        assert top["rank"] == 1
        assert "transfer" in top["content"].lower() or "limit" in top["content"].lower() or "wire" in top["content"].lower()


def test_agents_topology_endpoint():
    response = client.get("/api/v1/agents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_agents"] == 4
    assert len(data["agents"]) == 4
    assert data["agents"][0]["id"] == "request_handler_agent"
    assert data["agents"][0]["status"] == "ACTIVE"
    assert data["agents"][1]["id"] == "analyzer_agent"
    assert data["agents"][1]["status"] == "ACTIVE"
    assert data["agents"][2]["id"] == "content_generation_agent"
    assert data["agents"][2]["status"] == "ACTIVE"