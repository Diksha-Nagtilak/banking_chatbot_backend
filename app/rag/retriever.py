"""
RAG Step 1: Document Retriever.
Searches the knowledge base and retrieves candidate documents matching the user query.
"""
import re
from typing import List
from app.rag.knowledge_base import BANKING_KNOWLEDGE_BASE
from app.schemas import KnowledgeDocument, ScoredDocument

# Common English stopwords to ignore during initial keyword extraction
STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "is", "are",
    "was", "were", "i", "my", "you", "your", "can", "could", "would", "should",
    "what", "how", "when", "where", "why", "do", "does", "did", "please", "me"
}


def _tokenize(text: str) -> List[str]:
    """Extract clean words from text."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if len(w) > 1 and w not in STOPWORDS]


def retrieve_candidates(query: str, top_k: int = 4) -> List[ScoredDocument]:
    """
    RAG Step 1 - Retrieval:
    Scans the knowledge base and scores documents based on query term frequency.

    Args:
        query: The user's prompt or question
        top_k: Number of candidate documents to retrieve (default: 4)

    Returns:
        List of ScoredDocument objects sorted by relevance score
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        # Fallback if query only contained stopwords (e.g. "what is it")
        query_tokens = [w for w in query.lower().split() if len(w) > 0]

    scored_candidates: List[ScoredDocument] = []

    for doc in BANKING_KNOWLEDGE_BASE:
        title_tokens = _tokenize(doc.title)
        content_tokens = _tokenize(doc.content)

        # Title match score (weighted higher, 3x)
        title_matches = sum(1 for t in query_tokens if t in title_tokens)

        # Content match score
        content_matches = sum(1 for t in query_tokens if t in content_tokens)

        # Category match bonus
        category_match = 1.0 if any(t in doc.category.lower() for t in query_tokens) else 0.0

        # Total initial lexical retrieval score
        retrieval_score = (title_matches * 3.0) + (content_matches * 1.0) + (category_match * 2.0)

        if retrieval_score > 0:
            scored_candidates.append(
                ScoredDocument(
                    document=doc,
                    score=round(retrieval_score, 2),
                    reason=f"Matched {title_matches} title terms, {content_matches} content terms",
                )
            )

    # Sort descending by retrieval score
    scored_candidates.sort(key=lambda x: x.score, reverse=True)

    # Return top-K candidates
    return scored_candidates[:top_k]
