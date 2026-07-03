"""Phase 5 acceptance test: upload a report, ask about your own results.

    python scripts/ask_report.py <report_file> "what's my creatinine and is it high?"
    python scripts/ask_report.py <report_file>        # interactive, report stays attached

Done when: you upload a lab report and get a cited answer about your values.
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ocr, pipeline, report as report_mod  # noqa: E402


def _cite_tag(source_ids: list[int]) -> str:
    return " " + "".join(f"[S{n}]" for n in source_ids) if source_ids else " [no source]"


def ask(question: str, report) -> None:
    result = pipeline.answer_question(question, report=report)
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
        print("\n--- Dropped by faithfulness check ---")
        for d in result.dropped_claims:
            print(f"  x {d.claim.text}  (reason: {d.reason})")
    if result.sources:
        print("\n--- Sources ---")
        for i, c in enumerate(result.sources, 1):
            url = c.metadata.get("source_url", "")
            print(f"  [S{i}] {c.citation}" + (f"  {url}" if url else ""))
    if result.disclaimer:
        print(f"\nNote: {result.disclaimer}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ask_report.py <report_file> [question]")
        return

    report_path = sys.argv[1]
    try:
        report = report_mod.load_report(report_path)
    except ocr.OCRNotAvailable as e:
        print(f"OCR unavailable: {e}")
        return

    print(f"Loaded report: {report.filename}")
    print(f"Extracted {len(report.lab_values)} lab value(s):")
    for lv in report.lab_values:
        print(f"  - {lv.describe()}")

    if len(sys.argv) > 2:
        ask(" ".join(sys.argv[2:]), report)
        return

    print("\nAsk about your report. Blank line to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ask(q, report)


if __name__ == "__main__":
    main()
