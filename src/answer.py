"""Phase 2: grounded answer generation.

Retrieve the top chunks, hand them to Claude as the ONLY allowed source
material, and get back a plain-language answer. Claude is instructed to use
nothing but the provided sources — no outside knowledge, no diagnosing, no
prescribing.

This is the first of Claude's two jobs. The prompt already labels each source
[S1], [S2], ... so Phase 3 (per-sentence citations) and Phase 4 (faithfulness
check) can build directly on this structure.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

import config
from src import llm, retrieval
from src.retrieval import RetrievedChunk


SYSTEM_PROMPT = """You are a careful medical information assistant. You explain \
health information in plain, simple language a non-expert can understand.

Strict rules you must always follow:
1. Answer using ONLY the numbered sources provided in the user message. Do not \
use any outside knowledge or facts that are not in those sources.
2. If the sources do not contain enough information to answer, say clearly that \
you don't have a reliable source for that question. Do not guess or fill gaps.
3. Do not diagnose the user or tell them what condition they have, and do not \
prescribe or recommend specific medicines or doses. You may explain what the \
sources say in general terms.
4. Keep the answer concise and easy to read. Prefer short sentences.
5. Base every statement on the sources. Do not add reassurance, opinions, or \
information the sources do not support."""


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk]  # the chunks given to Claude, in [S1..] order


# --- Phase 3: structured per-sentence citations --------------------------

# Strict system prompt for the STRUCTURED path. Same grounding rules as Phase 2
# but the output is a list of claims, each tagged with the source numbers that
# support it — which is what makes Phase 4's per-claim faithfulness check work.
CITED_SYSTEM_PROMPT = """You are a careful medical information assistant. You \
explain health information in plain, simple language a non-expert can \
understand, and you attribute every statement to its source.

You will be given numbered sources ([S1], [S2], ...) and a question. Produce a \
structured answer as a list of claims.

Rules:
1. Each claim is ONE short, self-contained sentence in plain language.
2. Set is_meta to true ONLY for a conversational or framing sentence that makes \
no factual medical assertion — for example a disclaimer, a refusal such as "I \
can't diagnose conditions", or a suggestion to see a health professional. A \
meta sentence needs no citation, so leave its source_ids empty. Set is_meta to \
false for any sentence that states a health fact, cause, symptom, treatment, \
number, or anything about the user's own data.
3. For each non-meta claim, list in source_ids the source number(s) whose text \
directly supports it. Only cite a source if it genuinely states that claim. \
Never cite a source that does not support the claim, and never leave a non-meta \
factual claim uncited.
4. Use ONLY the provided sources. Do not add outside knowledge, opinions, or \
reassurance the sources do not support.
5. Do not diagnose the user or recommend specific medicines or doses.
6. If the sources do not contain enough information to answer, set answerable to \
false, return an empty claims list, and explain in note that you don't have a \
reliable source. Otherwise put any brief closing guidance (e.g. "see a health \
professional") in note."""

# Reading-level modifiers appended to the system prompt (Phase 8). "standard" is
# the default plain-language voice; "simple" targets a lower reading level.
READING_LEVELS = {
    "standard": "",
    "simple": (
        "\n\nWrite for a reader at about a 6th-grade reading level: use short "
        "sentences and everyday words, and briefly explain any medical term you "
        "have to use. Do not change any facts or citations to do this."
    ),
}


class Claim(BaseModel):
    text: str                # one plain-language sentence
    source_ids: list[int]    # 1-based [S#] numbers that support this claim
    is_meta: bool = False    # conversational/disclaimer sentence (no citation needed)


class GroundedAnswer(BaseModel):
    answerable: bool
    claims: list[Claim]
    note: str


@dataclass
class CitedAnswer:
    question: str
    answerable: bool
    claims: list[Claim]
    note: str
    sources: list[RetrievedChunk]

    def source_for(self, n: int) -> RetrievedChunk | None:
        """Resolve a 1-based [S#] id to its chunk (None if out of range)."""
        return self.sources[n - 1] if 1 <= n <= len(self.sources) else None


def _format_sources(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c.metadata
        header = f"[S{i}] ({m.get('source_name', '?')} — {m.get('source_title', c.id)})"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        "Sources:\n"
        f"{_format_sources(chunks)}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer the question using only the sources above. If the sources do "
        "not contain the answer, say you don't have a reliable source for it."
    )


def answer_question(question: str, top_k: int = config.TOP_K) -> Answer:
    chunks = retrieval.retrieve(question, top_k=top_k)
    if not chunks:
        return Answer(
            question=question,
            text="I don't have any indexed sources to answer from. "
                 "Build the knowledge base first (scripts/build_index.py).",
            sources=[],
        )
    text = llm.complete(
        build_prompt(question, chunks),
        system=SYSTEM_PROMPT,
        model=config.ANSWER_MODEL,
        max_tokens=1024,
    )
    return Answer(question=question, text=text.strip(), sources=chunks)


def answer_question_cited(
    question: str,
    top_k: int = config.TOP_K,
    chunks: list[RetrievedChunk] | None = None,
    reading_level: str = "standard",
) -> CitedAnswer:
    """Phase 3: return an answer whose every sentence carries its source id(s).

    If `chunks` is provided, use them as the sources verbatim (this is how the
    Phase 5 pipeline injects report sources alongside KB sources). Otherwise
    retrieve from the knowledge base. `reading_level` (Phase 8) tunes only the
    wording — it never changes which facts or citations are allowed.
    """
    if chunks is None:
        chunks = retrieval.retrieve(question, top_k=top_k)
    if not chunks:
        return CitedAnswer(
            question=question, answerable=False, claims=[],
            note="No indexed sources to answer from. Run scripts/build_index.py.",
            sources=[],
        )

    system = CITED_SYSTEM_PROMPT + READING_LEVELS.get(reading_level, "")
    result = llm.parse(
        build_prompt(question, chunks),
        GroundedAnswer,
        system=system,
        model=config.ANSWER_MODEL,
        max_tokens=1500,
    )

    # Drop any citation ids the model invented outside the [S#] range. Meta
    # sentences carry no citation by design.
    cleaned = [
        Claim(
            text=c.text,
            source_ids=[n for n in c.source_ids if 1 <= n <= len(chunks)],
            is_meta=c.is_meta,
        )
        for c in result.claims
    ]
    return CitedAnswer(
        question=question,
        answerable=result.answerable,
        claims=cleaned,
        note=result.note,
        sources=chunks,
    )
