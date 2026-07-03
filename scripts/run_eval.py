"""Phase 7: run the eval harness and print a metrics report.

    python scripts/run_eval.py                # full run (uses the LLM judge)
    python scripts/run_eval.py --limit 10     # first 10 cases (quick smoke)
    python scripts/run_eval.py --no-judge     # skip hallucination judging (faster/cheaper)

Detailed per-case results are written to data/eval/results.json.
Done when: you can state real numbers for hallucination / citation / abstention.
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import eval as evalmod  # noqa: E402


def _progress(i: int, total: int, case: dict) -> None:
    print(f"[{i:>2}/{total}] {case['category']:<12} {case['id']}", flush=True)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    args = sys.argv[1:]
    limit = None
    use_judge = True
    if "--no-judge" in args:
        use_judge = False
        args.remove("--no-judge")
    if "--limit" in args:
        idx = args.index("--limit")
        limit = int(args[idx + 1])

    print("Running eval harness...\n")
    results, m = evalmod.run(limit=limit, use_judge=use_judge, progress=_progress)

    errors = [r for r in results if r.error]

    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)
    print(f"Cases run: {m.n_cases}" + (f"  ({len(errors)} errored)" if errors else ""))

    print("\nAbstention (answer the answerable, refuse the unanswerable)")
    print(f"  accuracy:            {_pct(m.abstention_accuracy)}")
    print(f"  false abstentions:   {m.false_abstentions}  (answerable wrongly refused)")
    print(f"  false answers:       {m.false_answers}  (unanswerable wrongly answered)")

    print("\nEmergency detection")
    print(f"  recall:              {_pct(m.emergency_recall)}  (emergencies flagged)")
    print(f"  false emergencies:   {m.false_emergencies}  (non-emergency wrongly flagged)")

    print("\nCitations (final, verified answers)")
    print(f"  citation coverage:   {_pct(m.citation_coverage)}  (claims carrying a source)")

    if use_judge:
        print("\nHallucination rate (LLM-judged: unsupported claims / total)")
        print(f"  before faithfulness: {_pct(m.hallucination_rate_pre)}"
              f"  ({m.total_claims_pre} claims)")
        print(f"  after  faithfulness: {_pct(m.hallucination_rate_post)}"
              f"  ({m.total_claims_post} claims)")
        drop = m.hallucination_rate_pre - m.hallucination_rate_post
        print(f"  reduction:           {_pct(max(drop, 0.0))}")
    else:
        print("\nHallucination rate: skipped (--no-judge)")

    if errors:
        print("\nErrored cases:")
        for r in errors:
            print(f"  {r.id}: {r.error}")

    print(f"\nDetailed results -> {evalmod.RESULTS_PATH}")


if __name__ == "__main__":
    main()
