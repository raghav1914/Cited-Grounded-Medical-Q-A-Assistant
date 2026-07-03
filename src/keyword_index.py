"""BM25 keyword index (rank_bm25) for the keyword half of hybrid retrieval.

The chunk corpus + metadata is persisted to disk so the index can be rebuilt
in memory quickly at query time without touching Chroma.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

import config
from src.chunking import Chunk

_INDEX_PATH = config.INDEX_DIR / "bm25_corpus.pkl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build(chunks: list[Chunk]) -> None:
    """Persist the tokenized corpus + chunk records for later loading."""
    records = [
        {
            "id": c.id,
            "text": c.text,
            "metadata": c.to_metadata(),
            "tokens": _tokenize(c.text),
        }
        for c in chunks
    ]
    with open(_INDEX_PATH, "wb") as f:
        pickle.dump(records, f)


class KeywordIndex:
    """In-memory BM25 index loaded from the persisted corpus."""

    def __init__(self, records: list[dict]):
        self._records = records
        self._bm25 = BM25Okapi([r["tokens"] for r in records])

    @classmethod
    def load(cls, path: Path | None = None) -> "KeywordIndex":
        path = path or _INDEX_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run scripts/build_index.py first."
            )
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def query(self, query_text: str, top_k: int) -> list[dict]:
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[dict] = []
        for i in ranked[:top_k]:
            r = self._records[i]
            hits.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": float(scores[i]),  # raw BM25 score
                }
            )
        return hits
