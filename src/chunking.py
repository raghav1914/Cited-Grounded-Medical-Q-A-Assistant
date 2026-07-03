"""Document loading + chunking.

Loads .md / .txt / .pdf files from the knowledge-base directory and splits each
into overlapping, paragraph-aware chunks. Each chunk keeps a stable ID and the
source metadata needed for citations later (Phase 3).

No LLM here — pure rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class Chunk:
    id: str                       # e.g. "hypertension::3"
    text: str
    doc_id: str                   # source file stem, e.g. "hypertension"
    source_title: str             # human-readable title
    source_name: str              # e.g. "MedlinePlus", "WHO"
    source_url: str               # where the fact came from
    chunk_index: int
    meta: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        """Flat metadata dict for storage in Chroma (values must be scalars)."""
        return {
            "doc_id": self.doc_id,
            "source_title": self.source_title,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
        }


# --- front-matter parsing -------------------------------------------------

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Parse an optional simple `key: value` YAML-ish front-matter block.

    Returns (metadata dict, body). Kept deliberately tiny — no YAML dep.
    """
    m = _FM_RE.match(raw)
    if not m:
        return {}, raw
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, raw[m.end():]


# --- chunking -------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Greedy paragraph packing with character-based overlap.

    Paragraphs are accumulated until adding the next would exceed `size`, then
    a new chunk starts carrying `overlap` characters of tail context. A single
    paragraph longer than `size` is hard-split.
    """
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) > size:
            flush()
            for i in range(0, len(para), size - overlap):
                chunks.append(para[i:i + size].strip())
            continue

        if not current:
            current = para
        elif len(current) + 2 + len(para) <= size:
            current += "\n\n" + para
        else:
            flush()
            tail = chunks[-1][-overlap:] if chunks else ""
            current = (tail + "\n\n" + para).strip() if tail else para

    flush()
    return chunks


def load_and_chunk(docs_dir: Path | None = None) -> list[Chunk]:
    """Load every supported file in `docs_dir` and return all chunks."""
    docs_dir = docs_dir or config.DOCS_DIR
    chunks: list[Chunk] = []

    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue

        if path.suffix.lower() == ".pdf":
            raw = _read_pdf(path)
            meta: dict = {}
        else:
            raw = path.read_text(encoding="utf-8")
            meta, raw = _parse_front_matter(raw)

        doc_id = path.stem
        source_title = meta.get("title", doc_id.replace("_", " ").title())
        source_name = meta.get("source", "Unknown")
        source_url = meta.get("url", "")

        for i, piece in enumerate(_chunk_text(raw, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append(
                Chunk(
                    id=f"{doc_id}::{i}",
                    text=piece,
                    doc_id=doc_id,
                    source_title=source_title,
                    source_name=source_name,
                    source_url=source_url,
                    chunk_index=i,
                )
            )

    return chunks


def _read_pdf(path: Path) -> str:
    """Extract text from a text-based PDF (scanned PDFs need OCR — Phase 5)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)
