"""
Cross-Encoder Reranker Service.
Scores and re-ranks candidate chunks fetched from Azure AI Search using a Cross-Encoder model.
Provides deep query-passage semantic cross-attention with graceful, resilient fallback.
"""
import logging
from typing import List, Optional
from app.config import settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

# Global cached instance of the Cross-Encoder model
_CROSS_ENCODER_MODEL = None
_MODEL_LOAD_FAILED = False


def _get_cross_encoder_model():
    """Lazily load the Sentence-Transformers Cross-Encoder model."""
    global _CROSS_ENCODER_MODEL, _MODEL_LOAD_FAILED
    if _CROSS_ENCODER_MODEL is not None:
        return _CROSS_ENCODER_MODEL
    if _MODEL_LOAD_FAILED:
        return None

    try:
        from sentence_transformers import CrossEncoder
        model_name = settings.CROSS_ENCODER_MODEL_NAME or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading Cross-Encoder model: '{model_name}'...")
        _CROSS_ENCODER_MODEL = CrossEncoder(model_name)
        logger.info(f"Successfully loaded Cross-Encoder model: '{model_name}'")
        return _CROSS_ENCODER_MODEL
    except Exception as e:
        logger.warning(
            f"Could not load Sentence-Transformers Cross-Encoder ({e}). "
            "Using resilient algorithmic cross-scorer fallback."
        )
        _MODEL_LOAD_FAILED = True
        return None


def rerank_chunks(
    query: str,
    chunks: List[RetrievedChunk],
    top_n: Optional[int] = None,
) -> tuple[List[RetrievedChunk], str]:
    """
    Reranks candidate chunks based on Cross-Encoder scoring.

    Args:
        query: User's banking question or enriched prompt
        chunks: List of RetrievedChunk objects retrieved from Azure AI Search
        top_n: Number of top chunks to return

    Returns:
        tuple of (reranked_chunks: List[RetrievedChunk], provider_name: str)
    """
    if not chunks:
        return [], "No chunks to rerank"

    limit = top_n or settings.RERANK_TOP_N or 3

    logger.info("[FLOW: STEP 6B - CROSS-ENCODER RERANKER] rerank_chunks() invoked.")
    logger.info(f"   [6B.1] Query to align against : '{query}'")
    logger.info(f"   [6B.2] Total candidate chunks : {len(chunks)}")
    logger.info(f"   [6B.3] Target top N chunks    : {limit}")

    # Attempt live Cross-Encoder prediction if enabled
    if settings.ENABLE_CROSS_ENCODER:
        model = _get_cross_encoder_model()
        if model is not None:
            try:
                pairs = [
                    [query, f"{c.title}: {c.content}" if c.title else c.content]
                    for c in chunks
                ]
                logger.info(f"   [6B.4] Running neural cross-attention model for {len(pairs)} query-passage pairs...")
                raw_scores = model.predict(pairs)

                scored_pairs: List[tuple[float, RetrievedChunk]] = []
                for score, chunk in zip(raw_scores, chunks):
                    score_val = float(score)
                    updated_chunk = chunk.model_copy(update={"rerank_score": round(score_val, 4)})
                    scored_pairs.append((score_val, updated_chunk))

                scored_pairs.sort(key=lambda x: x[0], reverse=True)

                final_chunks: List[RetrievedChunk] = []
                for rank_idx, (sc, chk) in enumerate(scored_pairs[:limit], start=1):
                    final_chunks.append(chk.model_copy(update={"rank": rank_idx}))
                    logger.info(f"   [6B.5] Rank #{rank_idx} (Rerank Score: {sc:.4f}): '{chk.title}' (ID: {chk.id})")

                provider = f"Cross-Encoder ({settings.CROSS_ENCODER_MODEL_NAME})"
                logger.info(
                    f"   [6B.6] Cross-Encoder reranking complete. Retained top {len(final_chunks)} chunks via {provider}"
                )
                return final_chunks, provider

            except Exception as ex:
                logger.warning(f"Cross-Encoder prediction error: {ex}. Proceeding to fallback reranker.")

    # Resilient Cross-Scoring Fallback
    lower_query = query.lower().strip()
    query_tokens = [w for w in lower_query.split() if len(w) > 2]
    total_tokens = len(query_tokens) or 1

    scored_fallback: List[tuple[float, RetrievedChunk]] = []
    for chunk in chunks:
        passage = f"{chunk.title or ''} {chunk.content}".lower()

        # 1. Exact query match
        exact_bonus = 25.0 if lower_query in passage and len(lower_query) > 5 else 0.0

        # 2. Bigram matches
        words = lower_query.split()
        bigram_bonus = sum(
            8.0 for i in range(len(words) - 1) if f"{words[i]} {words[i+1]}" in passage
        )

        # 3. Token coverage ratio
        matched_tokens = sum(1 for w in query_tokens if w in passage)
        coverage_score = (matched_tokens / total_tokens) * 20.0

        # 4. Title term alignment
        title_bonus = sum(10.0 for w in query_tokens if chunk.title and w in chunk.title.lower())

        composite_score = (
            chunk.retrieval_score
            + exact_bonus
            + bigram_bonus
            + coverage_score
            + title_bonus
        )
        scored_fallback.append(
            (
                composite_score,
                chunk.model_copy(update={"rerank_score": round(composite_score, 2)}),
            )
        )

    scored_fallback.sort(key=lambda x: x[0], reverse=True)
    ranked_chunks: List[RetrievedChunk] = []
    for rank_idx, (_, chk) in enumerate(scored_fallback[:limit], start=1):
        ranked_chunks.append(chk.model_copy(update={"rank": rank_idx}))

    provider = "Semantic Cross-Scorer (Resilient Fallback)"
    logger.info(
        f"Reranked {len(chunks)} chunks to top {len(ranked_chunks)} using {provider}"
    )
    return ranked_chunks, provider