"""Phase 8: a rule-based confidence indicator.

Confidence is derived from deterministic signals the pipeline already produces —
NOT from another Claude call (that would be a third LLM job and, worse, would let
the model grade its own homework). The signals:

- retrieval strength: the dense cosine similarity of the best source actually
  cited by a surviving claim (report values count as a perfect 1.0);
- faithfulness survival: what fraction of factual claims were dropped by the
  verification pass;
- grounding breadth: how many distinct sources the answer leans on.

These map to a coarse high / medium / low label with a plain-language reason for
each, so the UI can be honest about how sure the system is.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from src.faithfulness import VerifiedAnswer

# Retrieval-strength cut points, above config.ABSTAIN_MIN_DENSE (0.60). In-domain
# KB matches start ~0.79; a strong, on-topic hit sits ~0.85+.
_STRONG_DENSE = 0.82
_OK_DENSE = 0.70


@dataclass
class Confidence:
    level: str            # "high" | "medium" | "low"
    reasons: list[str]


def _best_cited_dense(verified: VerifiedAnswer) -> float:
    """Highest dense score among sources actually cited by a kept factual claim."""
    best = 0.0
    for claim in verified.kept_claims:
        if claim.is_meta:
            continue
        for sid in claim.source_ids:
            if 1 <= sid <= len(verified.sources):
                best = max(best, verified.sources[sid - 1].dense_score)
    return best


def assess(verified: VerifiedAnswer, kb_best_dense: float = 0.0) -> Confidence:
    """Grade a verified answer. `kb_best_dense` is the top KB chunk's similarity,
    used as a floor when a claim's own citations don't pin one down."""
    factual_kept = [c for c in verified.kept_claims if not c.is_meta]
    dropped = len(verified.dropped_claims)          # dropped claims are always factual
    total_factual = len(factual_kept) + dropped
    drop_ratio = dropped / total_factual if total_factual else 0.0

    best_dense = max(_best_cited_dense(verified), kb_best_dense)
    n_sources = len({sid for c in factual_kept for sid in c.source_ids})

    reasons: list[str] = []

    # Retrieval strength.
    if best_dense >= _STRONG_DENSE:
        reasons.append(f"Strong source match (similarity {best_dense:.2f}).")
        retrieval_tier = 2
    elif best_dense >= _OK_DENSE:
        reasons.append(f"Moderate source match (similarity {best_dense:.2f}).")
        retrieval_tier = 1
    else:
        reasons.append(f"Weak source match (similarity {best_dense:.2f}).")
        retrieval_tier = 0

    # Faithfulness survival.
    if dropped == 0:
        reasons.append("Every statement passed the faithfulness check.")
    else:
        reasons.append(
            f"{dropped} of {total_factual} statements were dropped as unsupported."
        )

    # Grounding breadth.
    if n_sources >= 2:
        reasons.append(f"Backed by {n_sources} independent sources.")

    # Combine into a coarse label. Any dropped claim or a weak match caps it.
    if retrieval_tier == 2 and drop_ratio == 0.0:
        level = "high"
    elif retrieval_tier >= 1 and drop_ratio <= 0.34:
        level = "medium"
    else:
        level = "low"

    return Confidence(level=level, reasons=reasons)
