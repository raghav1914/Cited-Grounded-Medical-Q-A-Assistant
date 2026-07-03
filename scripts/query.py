"""Phase 1 acceptance test: a question returns the right source chunks.

No answer generation yet (that's Phase 2) — this just shows retrieval works.

Run:  python scripts/query.py "what is a normal creatinine level?"
      python scripts/query.py            # interactive mode
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import retrieval  # noqa: E402


def show(query: str) -> None:
    hits = retrieval.retrieve(query)
    if not hits:
        print("No chunks retrieved (is the index built?).")
        return
    print(f"\nQuery: {query}")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] {h.id}  (dense={h.dense_score:.3f}  rrf={h.rrf_score:.4f})")
        print(f"    source: {h.citation}")
        preview = h.text.replace("\n", " ")
        print(f"    {preview[:200]}{'...' if len(preview) > 200 else ''}")


def main() -> None:
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return
    print("Interactive retrieval. Blank line to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(q)


if __name__ == "__main__":
    main()
