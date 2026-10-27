"""
Azure AI Search Client & Document Retriever.
Queries the user's Azure AI Search index for relevant knowledge chunks.
Strictly queries Azure AI Search with no hardcoded fallback.
"""
import logging
from typing import Any, List, Optional
from app.config import settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

# Search candidate field priority heuristics (case-insensitive)
CONTENT_FIELDS = ["content", "chunk", "text", "chunk_content", "body", "page_content", "description"]
TITLE_FIELDS = ["source_document", "title", "document_title", "name", "source", "filepath", "document_name", "id"]


def _extract_field(doc: dict, target_field: Optional[str], fallback_candidates: list[str]) -> Optional[str]:
    """Helper to extract field value using explicit name or candidate heuristics (case-insensitive and handles list values)."""
    # 1. Exact match
    if target_field and target_field in doc and doc[target_field] is not None:
        val = doc[target_field]
        return " ".join(str(item) for item in val if item) if isinstance(val, list) else str(val).strip()

    # Build lower-cased lookup map
    lower_map = {k.lower(): v for k, v in doc.items()}

    # 2. Case-insensitive match for target_field
    if target_field and target_field.lower() in lower_map and lower_map[target_field.lower()] is not None:
        val = lower_map[target_field.lower()]
        return " ".join(str(item) for item in val if item) if isinstance(val, list) else str(val).strip()

    # 3. Candidate heuristics (exact and lowercase)
    for candidate in fallback_candidates:
        if candidate in doc and doc[candidate] is not None:
            val = doc[candidate]
            return " ".join(str(item) for item in val if item) if isinstance(val, list) else str(val).strip()
        cand_lower = candidate.lower()
        if cand_lower in lower_map and lower_map[cand_lower] is not None:
            val = lower_map[cand_lower]
            return " ".join(str(item) for item in val if item) if isinstance(val, list) else str(val).strip()

    # 4. Substring match fallback for chunk/content/text
    for k_lower, v in lower_map.items():
        if any(c in k_lower for c in fallback_candidates) and v is not None:
            return " ".join(str(item) for item in v if item) if isinstance(v, list) else str(v).strip()

    return None


def search_azure_ai_search(
    query: str,
    top_k: Optional[int] = None,
    category: Optional[str] = None,
) -> tuple[List[RetrievedChunk], str]:
    """
    Retrieves candidate chunks strictly from the live Azure AI Search index.

    Returns:
        tuple of (chunks: List[RetrievedChunk], provider_description: str)
    Raises:
        RuntimeError: if Azure AI Search credentials/index name are missing or query fails.
    """
    limit = top_k or settings.AZURE_SEARCH_TOP_K or 5

    # Gracefully check that Azure AI Search endpoint and index name are configured
    if not (settings.AZURE_SEARCH_ENDPOINT and settings.AZURE_SEARCH_INDEX_NAME):
        logger.warning(
            "Azure AI Search endpoint or index name not configured. Continuing without index retrieval."
        )
        return [], "Azure AI Search (Unconfigured)"

    if not (settings.AZURE_SEARCH_API_KEY or settings.AZURE_SEARCH_USE_MANAGED_IDENTITY):
        logger.warning(
            "Neither AZURE_SEARCH_API_KEY nor AZURE_SEARCH_USE_MANAGED_IDENTITY is provided. Continuing without index retrieval."
        )
        return [], "Azure AI Search (Awaiting API Key)"

    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        endpoint = settings.AZURE_SEARCH_ENDPOINT.strip().rstrip("/")
        index_name = settings.AZURE_SEARCH_INDEX_NAME.strip()

        if settings.AZURE_SEARCH_USE_MANAGED_IDENTITY:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
        elif settings.AZURE_SEARCH_API_KEY:
            credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.strip())
        else:
            return [], "Azure AI Search (No credentials)"

        logger.info(f"[FLOW: STEP 6A - AZURE AI SEARCH CLIENT] Connecting to '{endpoint}', index: '{index_name}'...")
        client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
        logger.info(f"   [6A.1] Executing client.search(search_text='{query}', top={limit})...")

        results = client.search(
            search_text=query,
            top=limit,
            include_total_count=True,
        )

        chunks: List[RetrievedChunk] = []
        for doc in results:
            doc_dict = dict(doc)
            content = _extract_field(doc_dict, settings.AZURE_SEARCH_CONTENT_FIELD, CONTENT_FIELDS)
            title = _extract_field(doc_dict, settings.AZURE_SEARCH_TITLE_FIELD, TITLE_FIELDS) or "Azure AI Search Document"
            doc_id = str(doc_dict.get("id") or doc_dict.get("chunk_id") or f"doc_{len(chunks)+1}")
            doc_cat = str(doc_dict.get(settings.AZURE_SEARCH_CATEGORY_FIELD) or doc_dict.get("category") or "banking")
            chunk_idx_val = doc_dict.get("chunk_index")
            try:
                chunk_idx = int(chunk_idx_val) if chunk_idx_val is not None else None
            except (ValueError, TypeError):
                chunk_idx = None
            score = float(doc_dict.get("@search.score", 1.0))

            if content:
                source_val = str(doc_dict.get("source_document") or doc_dict.get("source") or doc_dict.get("filepath") or title)
                logger.info(f"   [6A.2] Match found: ID='{doc_id}', Source/Title='{title}', Chunk #{chunk_idx}, Score={score:.4f}")
                chunks.append(
                    RetrievedChunk(
                        id=doc_id,
                        content=content,
                        title=title,
                        source=source_val,
                        category=doc_cat,
                        chunk_index=chunk_idx,
                        retrieval_score=round(score, 4),
                    )
                )

        logger.info(f"   [6A.3] Azure AI Search retrieval complete: {len(chunks)} chunks loaded from '{index_name}'.")
        return chunks, f"Azure AI Search (Index: {index_name})"

    except Exception as e:
        logger.warning(f"Error querying Azure AI Search index: {e}. Gracefully continuing without search chunks.")
        return [], f"Azure AI Search (Query Error: {type(e).__name__})"