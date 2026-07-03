"""Phase 4a: rule-based abstention.

Pure code — no LLM. Before we ever call Claude, decide whether retrieval was
strong enough to bother. If the best chunk's dense cosine similarity is below
the configured threshold, we refuse ("I don't have a reliable source") without
spending an API call and without giving the model a chance to hallucinate over
weak context.

Keeping this deterministic is deliberate: whether to say "I don't know" is a
trust-critical decision, so it must not be an LLM judgment call.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from src.retrieval import RetrievedChunk


@dataclass
class AbstentionDecision:
    abstain: bool
    best_dense: float
    reason: str


def check(chunks: list[RetrievedChunk]) -> AbstentionDecision:
    """Decide whether to abstain based on retrieval strength alone."""
    if not chunks:
        return AbstentionDecision(
            abstain=True, best_dense=0.0,
            reason="No chunks retrieved (is the index built?).",
        )

    best = max(c.dense_score for c in chunks)
    if best < config.ABSTAIN_MIN_DENSE:
        return AbstentionDecision(
            abstain=True, best_dense=best,
            reason=(
                f"Best source match {best:.2f} is below the reliability "
                f"threshold {config.ABSTAIN_MIN_DENSE:.2f}."
            ),
        )
    return AbstentionDecision(abstain=False, best_dense=best, reason="")
