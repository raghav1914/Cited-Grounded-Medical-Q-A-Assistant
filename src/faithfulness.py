"""Phase 4b: faithfulness check (Claude's second job).

A separate Claude pass verifies each claim against ONLY the source text that was
cited for it. Claims the cited source does not actually support are dropped, so
an unsupported statement never survives into the final answer even if the first
pass produced it.

Design choices that keep this honest:
- Each claim is checked against just its own cited chunk(s), not the whole
  corpus, so the verifier can't "find support elsewhere".
- A claim with no citation fails by rule (no API call needed) — an uncited
  factual statement is unsupported by definition.
- The verifier is told to use no outside knowledge and to be strict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

import config
from src import llm
from src.answer import Claim, CitedAnswer


FAITHFULNESS_SYSTEM_PROMPT = """You are a strict fact-checker. For each numbered \
claim you are given the exact source text that was cited to support it. Decide \
whether that source text directly states or clearly entails the claim.

Rules:
- Use ONLY the provided source text. Do not use any outside knowledge.
- Be strict. If the source does not clearly support the claim, mark it not \
supported, even if the claim seems plausible or generally true.
- A paraphrase is fine as long as the meaning is genuinely supported by the \
source text.
- Return one verdict per claim, using the same claim number you were given."""


class Verdict(BaseModel):
    index: int          # 1-based claim number as presented in the prompt
    supported: bool
    reason: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


@dataclass
class DroppedClaim:
    claim: Claim
    reason: str


@dataclass
class VerifiedAnswer:
    question: str
    answerable: bool                       # final answerability after verification
    kept_claims: list[Claim]
    dropped_claims: list[DroppedClaim] = field(default_factory=list)
    note: str = ""
    sources: list = field(default_factory=list)
    emergency: bool = False                # Phase 6: red-flag bypass triggered
    disclaimer: str = ""                   # Phase 6: context-aware closing disclaimer
    confidence_level: str = ""             # Phase 8: "high" | "medium" | "low" (rule-based)
    confidence_reasons: list = field(default_factory=list)  # Phase 8: why that level


def _cited_source_text(claim: Claim, answer: CitedAnswer) -> str:
    parts = []
    for n in claim.source_ids:
        chunk = answer.source_for(n)
        if chunk is not None:
            parts.append(f"[S{n}] {chunk.text}")
    return "\n".join(parts)


def _build_prompt(cited: list[Claim], answer: CitedAnswer) -> str:
    blocks = []
    for i, claim in enumerate(cited, 1):
        blocks.append(
            f"Claim {i}: \"{claim.text}\"\n"
            f"Cited source(s) for claim {i}:\n{_cited_source_text(claim, answer)}"
        )
    return (
        "Verify each claim against its cited source(s).\n\n"
        + "\n\n".join(blocks)
    )


def verify(answer: CitedAnswer) -> VerifiedAnswer:
    """Run the faithfulness pass and return the answer with unsupported claims removed."""
    if not answer.answerable:
        return VerifiedAnswer(
            question=answer.question, answerable=False, kept_claims=[],
            note=answer.note, sources=answer.sources,
        )

    # Only cited, non-meta factual claims are fact-checked. Meta sentences
    # (disclaimers/refusals/"see a professional") make no factual assertion, so
    # they bypass the citation requirement and are always kept. An uncited
    # factual claim still fails by rule.
    cited = [c for c in answer.claims if not c.is_meta and c.source_ids]

    verdict_for: dict[int, Verdict] = {}
    if cited:
        result = llm.parse(
            _build_prompt(cited, answer),
            Verdicts,
            system=FAITHFULNESS_SYSTEM_PROMPT,
            model=config.FAITHFULNESS_MODEL,
            max_tokens=1500,
        )
        verdict_by_index = {v.index: v for v in result.verdicts}
        for i, claim in enumerate(cited, 1):
            verdict_for[id(claim)] = verdict_by_index.get(i)

    # Walk the claims in their original order so kept sentences read naturally.
    kept: list[Claim] = []
    dropped: list[DroppedClaim] = []
    for claim in answer.claims:
        if claim.is_meta:
            kept.append(claim)                      # bypasses the citation check
        elif not claim.source_ids:
            dropped.append(DroppedClaim(claim=claim, reason="No source cited."))
        else:
            v = verdict_for.get(id(claim))
            if v is not None and v.supported:
                kept.append(claim)
            else:
                reason = v.reason if v is not None else "No verdict returned."
                dropped.append(DroppedClaim(claim=claim, reason=reason))

    # Answerable requires at least one grounded factual claim to survive; a reply
    # of only meta sentences is not a real answer.
    answerable = any(not c.is_meta for c in kept)
    note = answer.note
    if not answerable:
        note = ("After checking each statement against its cited source, none "
                "could be reliably supported, so I don't have a reliable answer.")

    return VerifiedAnswer(
        question=answer.question,
        answerable=answerable,
        kept_claims=kept,
        dropped_claims=dropped,
        note=note,
        sources=answer.sources,
    )
