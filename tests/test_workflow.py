import pytest
from app.agents.request_handler import RequestHandlerAgent
from app.agents.analyzer import AnalyzerAgent
from app.agents.content_generator import ContentGeneratorAgent
from app.workflow import create_banking_workflow, run_banking_workflow


@pytest.fixture
def workflow():
    req_agent = RequestHandlerAgent()
    analyzer = AnalyzerAgent()
    content_gen = ContentGeneratorAgent()
    return create_banking_workflow(
        request_handler=req_agent,
        analyzer=analyzer,
        content_generator=content_gen,
    )


@pytest.mark.asyncio
async def test_workflow_banking_query_end_to_end(workflow):
    """
    Tests full 3-agent pipeline via Microsoft Agent Framework Workflow:
    Agent 1 (Triage & Enrich) -> Agent 2 (Azure AI Search + Cross-Encoder) -> Agent 3 (Content Generator)
    """
    query = "What is the APY interest rate on savings accounts?"
    result = await workflow_runner(workflow, query)

    assert result.status == "accepted"
    assert result.data.is_banking_related is True
    assert result.data.enriched_query is not None

    # Verify Agent 2 retrieved and reranked chunks
    assert len(result.data.reranked_chunks) > 0
    top_chunk = result.data.reranked_chunks[0]
    assert top_chunk.content is not None
    assert top_chunk.rerank_score is not None

    # Verify Agent 3 generated grounded response
    assert result.data.response_message is not None
    assert len(result.data.response_message) > 20
    assert "4.75%" in result.data.response_message or "savings" in result.data.response_message.lower()

    # Verify Microsoft Agent Framework pipeline diagnostics
    pipeline = result.pipeline
    assert "agent_1_request_handler" in pipeline
    assert "agent_2_analyzer" in pipeline
    assert "agent_3_content_generator" in pipeline
    assert pipeline["agent_2_analyzer"]["reranked_count"] > 0
    assert len(pipeline["agent_3_content_generator"]["grounded_sources"]) > 0


@pytest.mark.asyncio
async def test_workflow_non_banking_early_exit(workflow):
    """
    Tests that non-banking queries halt early at Agent 1, avoiding downstream RAG execution.
    """
    query = "How do I bake a chocolate cake?"
    result = await workflow_runner(workflow, query)

    assert result.status == "rejected"
    assert result.data.is_banking_related is False
    assert result.data.enriched_query is None

    # Pipeline diagnostics should only contain Agent 1
    pipeline = result.pipeline
    assert "agent_1_request_handler" in pipeline
    assert "agent_2_analyzer" not in pipeline
    assert "agent_3_content_generator" not in pipeline


@pytest.mark.asyncio
async def test_workflow_loan_rates_inquiry(workflow):
    """
    Tests loan inquiries through the Microsoft Agent Framework workflow.
    """
    query = "What are the current interest rates for a 30-year fixed mortgage?"
    result = await workflow_runner(workflow, query)

    assert result.status == "accepted"
    assert result.data.is_banking_related is True
    assert result.data.category.value == "loans_and_credit"
    assert len(result.data.reranked_chunks) > 0
    assert "mortgage" in result.data.response_message.lower() or "rate" in result.data.response_message.lower()


async def workflow_runner(wf, query: str):
    return await run_banking_workflow(workflow=wf, message=query)
