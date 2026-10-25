"""
Local Knowledge Base (Removed).
All knowledge base retrieval is strictly fetched live from your Azure AI Search index.
"""
from typing import List
from app.schemas import KnowledgeDocument

# Empty knowledge base - all retrieval is routed directly to Azure AI Search
BANKING_KNOWLEDGE_BASE: List[KnowledgeDocument] = []
