"""Chroma vector store wrapper (local, file-based, zero setup).

We compute embeddings ourselves with BGE and hand them to Chroma directly, so
Chroma never downloads or runs its own embedding model.
"""
from __future__ import annotations

import chromadb

import config
from src import embeddings
from src.chunking import Chunk


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def reset_collection() -> chromadb.Collection:
    """Drop and recreate the collection (used on a full rebuild)."""
    client = _client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass  # collection didn't exist yet
    return client.create_collection(
        config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def get_collection() -> chromadb.Collection:
    return _client().get_or_create_collection(
        config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(collection: chromadb.Collection, chunks: list[Chunk]) -> None:
    """Embed and store chunks in batches."""
    batch = 128
    for start in range(0, len(chunks), batch):
        window = chunks[start:start + batch]
        collection.add(
            ids=[c.id for c in window],
            documents=[c.text for c in window],
            embeddings=embeddings.embed_documents([c.text for c in window]),
            metadatas=[c.to_metadata() for c in window],
        )


def query(query_text: str, top_k: int) -> list[dict]:
    """Dense search. Returns hits with a cosine-similarity score in [0, 1].

    Chroma returns cosine *distance* (0 = identical); we convert to a
    similarity so a higher number always means "more relevant". That score is
    what the Phase 4 abstention threshold will read.
    """
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embeddings.embed_query(query_text)],
        n_results=top_k,
    )
    hits: list[dict] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for cid, text, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            {
                "id": cid,
                "text": text,
                "metadata": meta,
                "score": 1.0 - float(dist),  # cosine similarity
            }
        )
    return hits
