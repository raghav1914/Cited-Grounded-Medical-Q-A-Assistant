"""Hybrid retrieval: dense (BGE + Chroma) + keyword (BM25), fused with RRF.

Reciprocal Rank Fusion combines the two ranked lists without needing to
normalize their (very different) score scales. We also carry the raw dense
cosine similarity through on each result — that's the signal the Phase 4
abstention rule will threshold on to decide "I don't have a reliable source".

Pure code, no LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import config
from src import vectorstore
from src.keyword_index import KeywordIndex


@lru_cache(maxsize=1)
def _keyword_index() -> KeywordIndex:
    """Load the BM25 index once per process (rebuild the app to pick up changes)."""
    return KeywordIndex.load()


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    rrf_score: float        # fused rank score (for ordering)
    dense_score: float      # cosine similarity in [0, 1] (for abstention)

    @property
    def citation(self) -> str:
        m = self.metadata
        return f"{m.get('source_name', '?')} — {m.get('source_title', self.id)}"


def _rrf(rank: int, k: int = config.RRF_K) -> float:
    return 1.0 / (k + rank + 1)  # rank is 0-based


def retrieve(query_text: str, top_k: int = config.TOP_K,
             pool: int = 20) -> list[RetrievedChunk]:
    """Return the top_k chunks by fused rank.

    `pool` is how many candidates each retriever contributes before fusion.
    """
    dense_hits = vectorstore.query(query_text, pool)
    keyword_hits = _keyword_index().query(query_text, pool)

    fused: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        fused.setdefault(hit["id"], {
            "text": hit["text"], "metadata": hit["metadata"],
            "rrf": 0.0, "dense": 0.0,
        })
        fused[hit["id"]]["rrf"] += _rrf(rank)
        fused[hit["id"]]["dense"] = hit["score"]

    for rank, hit in enumerate(keyword_hits):
        fused.setdefault(hit["id"], {
            "text": hit["text"], "metadata": hit["metadata"],
            "rrf": 0.0, "dense": 0.0,
        })
        fused[hit["id"]]["rrf"] += _rrf(rank)

    ordered = sorted(fused.items(), key=lambda kv: kv[1]["rrf"], reverse=True)
    return [
        RetrievedChunk(
            id=cid,
            text=v["text"],
            metadata=v["metadata"],
            rrf_score=v["rrf"],
            dense_score=v["dense"],
        )
        for cid, v in ordered[:top_k]
    ]
