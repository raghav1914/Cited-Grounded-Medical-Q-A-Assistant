"""Prove the faithfulness check catches unsupported claims.

We hand-build a cited answer containing three claims — one genuinely supported
by its cited chunk, one that cites a chunk which does NOT support it, and one
with no citation at all — then run the faithfulness pass and show what survives.

Run:  python scripts/demo_faithfulness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import faithfulness, retrieval  # noqa: E402
from src.answer import Claim, CitedAnswer  # noqa: E402


def main() -> None:
    # Real chunks from the KB so citations resolve to actual source text.
    chunks = retrieval.retrieve("high blood pressure", top_k=5)

    # Find the chunk that actually mentions heart attack / stroke so the
    # "supported" claim genuinely cites the source that supports it.
    supported_idx = next(
        (i for i, c in enumerate(chunks, 1) if "heart attack" in c.text.lower()),
        1,
    )
    # And a chunk that does NOT mention coffee (any will do) for the false claim.
    wrong_idx = 1

    adversarial = CitedAnswer(
        question="tell me about high blood pressure",
        answerable=True,
        claims=[
            # Supported: cites the chunk that really says this.
            Claim(text="Uncontrolled high blood pressure can lead to heart "
                       "attack and stroke.", source_ids=[supported_idx]),
            # Unsupported: cites a real chunk, but the source says no such thing.
            Claim(text="Drinking coffee every morning cures high blood pressure.",
                  source_ids=[wrong_idx]),
            # Uncited: a factual claim with no source — fails by rule.
            Claim(text="Blood pressure should be checked once every ten years.",
                  source_ids=[]),
        ],
        note="",
        sources=chunks,
    )

    print("Claims BEFORE faithfulness check:")
    for c in adversarial.claims:
        tag = "".join(f"[S{n}]" for n in c.source_ids) or "[no source]"
        print(f"  - {c.text} {tag}")

    verified = faithfulness.verify(adversarial)

    print("\nKEPT (supported):")
    for c in verified.kept_claims:
        print(f"  ok  {c.text}")

    print("\nDROPPED (self-corrected):")
    for d in verified.dropped_claims:
        print(f"  x   {d.claim.text}\n      reason: {d.reason}")


if __name__ == "__main__":
    main()
