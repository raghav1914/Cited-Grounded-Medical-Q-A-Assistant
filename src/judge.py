"""Phase 7: an independent LLM-as-judge for claim support.

Used ONLY by the eval harness to measure citation accuracy / hallucination
rate. It is deliberately separate from the faithfulness check (src/faithfulness)
so we're not grading the faithfulness pass with its own instructions — this is
the auditor, not the same gate re-run.

Honest limitation: the judge is itself Claude (LLM-as-judge). It's a standard
automated proxy for human labeling, not a substitute for it; the harness reports
its verdicts as such. Structural metrics (abstention, emergency, citation
coverage) do NOT depend on the judge.
"""
from __future__ import annotations

from pydantic import BaseModel

import config
from src import llm
from src.answer import Claim, CitedAnswer


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator auditing a medical \
assistant. For each numbered claim you are given the exact source text that the \
assistant cited for it. Decide whether that source text, on its own, supports \
the claim.

- Judge support strictly and use ONLY the provided source text; ignore whether \
the claim is true in general.
- A faithful paraphrase counts as supported. An addition, exaggeration, or \
detail not present in the source counts as not supported.
- Return exactly one verdict per claim, using the claim number you were given."""


class _JVerdict(BaseModel):
    index: int
    supported: bool


class _JVerdicts(BaseModel):
    verdicts: list[_JVerdict]


def _cited_text(claim: Claim, answer: CitedAnswer) -> str:
    parts = []
    for n in claim.source_ids:
        chunk = answer.source_for(n)
        if chunk is not None:
            parts.append(f"[S{n}] {chunk.text}")
    return "\n".join(parts)


def judge_answer(answer: CitedAnswer) -> dict[int, bool]:
    """Return {claim_index_in_answer.claims: supported_bool}.

    Uncited claims are auto-unsupported (no source can support them) and don't
    cost an API call. Cited claims are audited in a single batched call.
    """
    verdicts: dict[int, bool] = {}
    cited_positions: list[int] = []
    blocks: list[str] = []

    for i, claim in enumerate(answer.claims):
        if not claim.source_ids:
            verdicts[i] = False  # uncited => unsupported by definition
            continue
        cited_positions.append(i)
        n = len(cited_positions)
        blocks.append(
            f"Claim {n}: \"{claim.text}\"\n"
            f"Cited source(s):\n{_cited_text(claim, answer)}"
        )

    if not blocks:
        return verdicts

    prompt = "Audit each claim against its cited source(s).\n\n" + "\n\n".join(blocks)
    result = llm.parse(
        prompt, _JVerdicts,
        system=JUDGE_SYSTEM_PROMPT,
        model=config.FAITHFULNESS_MODEL,
        max_tokens=1500,
    )
    by_index = {v.index: v.supported for v in result.verdicts}
    for n, pos in enumerate(cited_positions, 1):
        verdicts[pos] = bool(by_index.get(n, False))
    return verdicts
