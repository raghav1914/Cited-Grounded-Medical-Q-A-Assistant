"""Central configuration for the medical Q&A bot.

Everything path- or model-related lives here so the rest of the code never
hardcodes a value. Loaded once at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (this file's directory).
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- Claude (the only place an LLM is used) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Answer generation + faithfulness check. Default to the most capable model;
# you can point FAITHFULNESS_MODEL at a cheaper model to save cost since the
# faithfulness pass is a simpler verification task.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-opus-4-8")
FAITHFULNESS_MODEL = os.getenv("FAITHFULNESS_MODEL", "claude-opus-4-8")

# --- Retrieval ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # free, local, no API cost
# BGE recommends this prefix on the QUERY side only (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking (character-based, paragraph-aware).
CHUNK_SIZE = 900        # target characters per chunk
CHUNK_OVERLAP = 150     # characters of overlap between adjacent chunks

# Hybrid retrieval.
TOP_K = 5               # chunks returned to the answer step
RRF_K = 60              # reciprocal-rank-fusion constant

# Rule-based abstention (Phase 4): if the best retrieved chunk's dense cosine
# similarity is below this, we refuse WITHOUT calling Claude. Calibrated on the
# starter KB: out-of-domain queries top out ~0.52, in-domain start ~0.79.
ABSTAIN_MIN_DENSE = 0.60

# --- Paths ---
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"        # raw knowledge base (.md / .txt / .pdf)
CHROMA_DIR = DATA_DIR / "chroma"    # Chroma persistence
INDEX_DIR = DATA_DIR / "index"      # BM25 corpus + chunk metadata
CHROMA_COLLECTION = "medical_kb"

# Make sure the writable dirs exist.
for _d in (DATA_DIR, DOCS_DIR, CHROMA_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require_api_key() -> str:
    """Return the API key or raise a clear error telling the user what to do."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your key, or set the environment variable."
        )
    return ANTHROPIC_API_KEY
