"""
RAG Step 2: Document Reranker.
Re-scores and re-orders candidate documents retrieved in Step 1.

Why Reranking matters:
Standard retrieval (e.g. keyword or initial vector search) is fast and casts a wide net,
but can retrieve documents that merely mention a keyword without answering the question.
A Reranker evaluates fine-grained semantic relevance, exact phrase overlap, and query
coverage to ensure that only the highest quality context reaches the LLM Agent.
"""
from typing import List
from app.schemas import ScoredDocument


def rerank_documents(query: str, candidates: List[ScoredDocument], top_n: int = 2) -> List[ScoredDocument]:
    """
    RAG Step 2 - Reranking:
    Re-scores candidate documents based on:
    1. Exact phrase / n-gram match bonus (+30 points)
    2. Query term coverage percentage (how many unique query words appear in doc)
    3. Token density (relevance concentration)

    Args:
        query: Original user question
        candidates: Candidate documents retrieved in Step 1
        top_n: Number of final top-ranked documents to return to the Agent (default: 2)

    Returns:
        List of ScoredDocument objects sorted from highest to lowest rerank score
    """
    if not candidates:
        return []

    lower_query = query.lower().strip()
    query_words = [w for w in lower_query.split() if len(w) > 2]
    total_query_words = len(query_words) or 1

    reranked: List[ScoredDocument] = []

    for scored_item in candidates:
        doc = scored_item.document
        doc_text = f"{doc.title} {doc.content}".lower()

        # 1. Exact phrase match bonus: does the exact question (or 3+ word phrase) appear?
        exact_phrase_bonus = 0.0
        if len(lower_query) > 5 and lower_query in doc_text:
            exact_phrase_bonus = 40.0
        else:
            # Check 2-word consecutive phrase matches
            words = lower_query.split()
            for i in range(len(words) - 1):
                bi_gram = f"{words[i]} {words[i+1]}"
                if bi_gram in doc_text:
                    exact_phrase_bonus += 10.0

        # 2. Query Term Coverage: What fraction of query words are present?
        matched_words = sum(1 for w in query_words if w in doc_text)
        coverage_ratio = matched_words / total_query_words
        coverage_score = coverage_ratio * 30.0

        # 3. Title Alignment: Does the document title directly match query terms?
        title_bonus = sum(10.0 for w in query_words if w in doc.title.lower())

        # 4. Composite Rerank Score
        initial_score = scored_item.score
        final_rerank_score = initial_score + exact_phrase_bonus + coverage_score + title_bonus

        reranked.append(
            ScoredDocument(
                document=doc,
                score=round(final_rerank_score, 2),
                reason=(
                    f"Reranked: {coverage_ratio*100:.0f}% query coverage, "
                    f"phrase bonus +{exact_phrase_bonus:.0f}, title bonus +{title_bonus:.0f}"
                ),
            )
        )

    # Sort descending by final rerank score
    reranked.sort(key=lambda x: x.score, reverse=True)

    # Return top-N highest quality documents
    return reranked[:top_n]
