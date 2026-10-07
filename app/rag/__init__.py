"""
RAG (Retrieval-Augmented Generation) & Reranking Module.
Step-by-step educational implementation:
1. Knowledge Base: Clean banking reference documents
2. Retriever: Candidate document search
3. Reranker: Re-scores and re-orders candidate documents for maximum relevance
"""
from app.rag.knowledge_base import BANKING_KNOWLEDGE_BASE
from app.rag.retriever import retrieve_candidates
from app.rag.reranker import rerank_documents

__all__ = [
    "BANKING_KNOWLEDGE_BASE",
    "retrieve_candidates",
    "rerank_documents",
]
