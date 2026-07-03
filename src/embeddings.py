"""Local embeddings via the BGE model (bge-small-en-v1.5).

Free, runs on CPU, no API cost. The model is loaded lazily and cached for the
process. Embeddings are L2-normalized so cosine similarity == dot product.
"""
from __future__ import annotations

from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so importing this module stays cheap.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages (no query prefix)."""
    vectors = _model().encode(
        texts, normalize_embeddings=True, show_progress_bar=len(texts) > 32
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a search query (BGE recommends a retrieval prefix)."""
    vector = _model().encode(
        config.BGE_QUERY_PREFIX + text, normalize_embeddings=True
    )
    return vector.tolist()
