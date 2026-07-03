"""Acceptance test for the answer pipeline.

Default (Phase 4): retrieve -> rule-based abstention -> cited answer ->
faithfulness check. Refuses unanswerable questions and drops unsupported claims.

Run:  python scripts/ask.py "is high blood pressure dangerous?"
      python scripts/ask.py --no-verify "..."   # Phase 3: cited, no faithfulness pass
      python scripts/ask.py --plain "..."        # Phase 2: free-form prose
      python scripts/ask.py                       # interactive mode
"""
import sys
from pathlib import Path

# Render UTF-8 (em-dashes in source titles, etc.) even on a cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import answer, pipeline  # noqa: E402


def _cite_tag(source_ids: list[int]) -> str:
    if not source_ids:
        return " [no source]"
    return " " + "".join(f"[S{n}]" for n in source_ids)


def _print_sources(sources) -> None:
    if not sources:
        return
    print("\n--- Sources ---")
    for i, c in enumerate(sources, 1):
        url = c.metadata.get("source_url", "")
        print(f"  [S{i}] {c.citation}" + (f"  {url}" if url else ""))


def ask_verified(question: str) -> None:
    result = pipeline.answer_question(question)
    print(f"\nQ: {question}\n")

    if result.emergency:
        print(result.note)
        return

    if not result.answerable:
        print(result.note or "I don't have a reliable source to answer that.")
    else:
        for claim in result.kept_claims:
            print(f"{claim.text}{_cite_tag(claim.source_ids)}")
        if result.note:
            print(f"\n{result.note}")

    if result.dropped_claims:
        print("\n--- Dropped by faithfulness check (unsupported) ---")
        for d in result.dropped_claims:
            print(f"  x {d.claim.text}")
            print(f"    reason: {d.reason}")

    _print_sources(result.sources)

    if result.disclaimer:
        print(f"\nNote: {result.disclaimer}")


def ask_cited(question: str) -> None:
    result = answer.answer_question_cited(question)
    print(f"\nQ: {question}\n")
    if not result.answerable:
        print(result.note or "I don't have a reliable source to answer that.")
    else:
        for claim in result.claims:
            print(f"{claim.text}{_cite_tag(claim.source_ids)}")
        if result.note:
            print(f"\n{result.note}")
    _print_sources(result.sources)


def ask_plain(question: str) -> None:
    result = answer.answer_question(question)
    print(f"\nQ: {question}\n")
    print(result.text)


def main() -> None:
    args = sys.argv[1:]
    ask = ask_verified
    if args and args[0] == "--plain":
        ask, args = ask_plain, args[1:]
    elif args and args[0] == "--no-verify":
        ask, args = ask_cited, args[1:]

    if args:
        ask(" ".join(args))
        return

    print("Ask a health question. Blank line to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ask(q)


if __name__ == "__main__":
    main()
