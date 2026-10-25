import pytest
from app.agents.analyzer import AnalyzerAgent
from app.agents.content_generator import ContentGeneratorAgent
from app.models.schemas import RetrievedChunk


@pytest.fixture
def analyzer():
    return AnalyzerAgent()


@pytest.fixture
def content_generator():
    return ContentGeneratorAgent()


@pytest.mark.asyncio
async def test_analyzer_retrieves_and_reranks_chunks(analyzer):
    query = "What is the APY interest rate on savings accounts and checking fees?"
    output = await analyzer.process(
        query=query,
        enriched_query="Action: account_services | Topic: savings and checking interest rates",
        category="account_services",
    )

    assert output.query == query
    assert output.retrieved_count > 0
    assert len(output.reranked_chunks) > 0
    assert len(output.reranked_chunks) <= 3  # default top_n = 3

    top_chunk = output.reranked_chunks[0]
    assert top_chunk.rank == 1
    assert top_chunk.content is not None
    assert top_chunk.rerank_score is not None
    # Check that checking/savings account details were matched
    assert "savings" in top_chunk.content.lower() or "checking" in top_chunk.content.lower()


@pytest.mark.asyncio
async def test_analyzer_loan_rates_query(analyzer):
    query = "What are the 30-year fixed mortgage rates and personal loan APR?"
    output = await analyzer.process(
        query=query,
        category="loans_and_credit",
    )

    assert output.retrieved_count > 0
    assert len(output.reranked_chunks) > 0
    # Mortgage doc should be top or near top
    titles = [c.title.lower() for c in output.reranked_chunks if c.title]
    assert any("mortgage" in t or "loan" in t for t in titles)


@pytest.mark.asyncio
async def test_content_generator_grounded_answer(content_generator):
    sample_chunks = [
        RetrievedChunk(
            id="DOC_001",
            title="Checking & Savings Account Overview",
            content="AegisBank High-Yield Savings accounts earn an APY of 4.75% with interest compounded daily.",
            retrieval_score=5.0,
            rerank_score=35.0,
            rank=1,
        )
    ]
    query = "What is the APY for savings accounts?"
    result = await content_generator.process(query=query, reranked_chunks=sample_chunks)

    assert result.answer is not None
    assert len(result.answer) > 10
    assert "Checking & Savings Account Overview" in result.grounded_sources
    assert "4.75%" in result.answer or "savings" in result.answer.lower()