"""Phase 7: the evaluation harness.

Runs a labeled test set through the system and reports real numbers:

- Abstention correctness   — answers the answerable, refuses the unanswerable
- Emergency detection      — red-flags the emergencies, no false alarms
- Citation coverage        — fraction of final claims that carry a citation
- Hallucination rate       — fraction of claims NOT supported by their cited
                             source, measured PRE vs POST the faithfulness check
                             (an independent LLM judge audits support)

Structural metrics (abstention, emergency, coverage) need no LLM judge. Only the
hallucination rate uses the judge (src/judge), and it's clearly labeled.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from src import abstention, answer, faithfulness, judge, retrieval, safety

TESTSET_PATH = config.DATA_DIR / "eval" / "testset.jsonl"
RESULTS_PATH = config.DATA_DIR / "eval" / "results.json"


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    # observed behavior
    emergency: bool = False
    abstained: bool = False
    behavior_correct: bool = False
    # answerable-only claim accounting
    claims_pre: int = 0
    unsupported_pre: int = 0
    claims_post: int = 0
    unsupported_post: int = 0
    cited_post: int = 0            # kept claims that carry >=1 citation
    error: str = ""


def load_testset(path: Path = TESTSET_PATH) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _run_answerable(case: dict, use_judge: bool) -> CaseResult:
    q = case["question"]
    r = CaseResult(id=case["id"], category=case["category"], question=q)

    # An answerable question tripping a red flag would be a false alarm.
    if safety.check_red_flags(q).triggered:
        r.emergency = True
        r.behavior_correct = False
        return r

    chunks = retrieval.retrieve(q)
    if abstention.check(chunks).abstain:
        r.abstained = True
        r.behavior_correct = False   # wrongly refused an answerable question
        return r

    cited = answer.answer_question_cited(q, chunks=chunks)
    if not cited.answerable:
        r.abstained = True
        r.behavior_correct = False
        return r

    verified = faithfulness.verify(cited)
    r.behavior_correct = verified.answerable

    r.claims_pre = len(cited.claims)
    r.claims_post = len(verified.kept_claims)
    r.cited_post = sum(1 for c in verified.kept_claims if c.source_ids)

    if use_judge:
        verdicts = judge.judge_answer(cited)  # keyed by index into cited.claims
        r.unsupported_pre = sum(1 for i in range(len(cited.claims)) if not verdicts.get(i, False))
        # Post claims are the surviving subset of cited.claims (same objects).
        kept_ids = {id(c) for c in verified.kept_claims}
        r.unsupported_post = sum(
            1 for i, c in enumerate(cited.claims)
            if id(c) in kept_ids and not verdicts.get(i, False)
        )
    return r


def _run_unanswerable(case: dict) -> CaseResult:
    from src import pipeline
    q = case["question"]
    r = CaseResult(id=case["id"], category=case["category"], question=q)
    result = pipeline.answer_question(q)
    r.emergency = result.emergency
    r.abstained = not result.answerable and not result.emergency
    r.behavior_correct = r.abstained   # correct == refused
    return r


def _run_emergency(case: dict) -> CaseResult:
    from src import pipeline
    q = case["question"]
    r = CaseResult(id=case["id"], category=case["category"], question=q)
    result = pipeline.answer_question(q)
    r.emergency = result.emergency
    r.behavior_correct = result.emergency
    return r


def run_case(case: dict, use_judge: bool = True) -> CaseResult:
    try:
        if case["category"] == "answerable":
            return _run_answerable(case, use_judge)
        if case["category"] == "unanswerable":
            return _run_unanswerable(case)
        if case["category"] == "emergency":
            return _run_emergency(case)
    except Exception as e:  # keep the harness going; record the failure
        r = CaseResult(id=case["id"], category=case["category"], question=case["question"])
        r.error = f"{type(e).__name__}: {e}"
        return r
    raise ValueError(f"Unknown category: {case['category']}")


@dataclass
class Metrics:
    n_cases: int
    # abstention (answerable + unanswerable)
    abstention_accuracy: float
    false_abstentions: int         # answerable wrongly refused
    false_answers: int             # unanswerable wrongly answered
    # emergency
    emergency_recall: float        # emergencies correctly flagged
    false_emergencies: int         # non-emergency wrongly flagged
    # citation + hallucination (answered answerable cases)
    citation_coverage: float
    hallucination_rate_pre: float
    hallucination_rate_post: float
    total_claims_pre: int
    total_claims_post: int


def compute_metrics(results: list[CaseResult]) -> Metrics:
    ans = [r for r in results if r.category == "answerable"]
    una = [r for r in results if r.category == "unanswerable"]
    emg = [r for r in results if r.category == "emergency"]

    # Abstention: answerable should answer, unanswerable should abstain.
    abst_total = len(ans) + len(una)
    abst_correct = sum(1 for r in ans if not r.abstained and not r.emergency) \
        + sum(1 for r in una if r.abstained)
    false_abstentions = sum(1 for r in ans if r.abstained)
    false_answers = sum(1 for r in una if not r.abstained and not r.emergency)

    # Emergency detection + false alarms on everything non-emergency.
    emergency_recall = (sum(1 for r in emg if r.emergency) / len(emg)) if emg else 0.0
    false_emergencies = sum(1 for r in (ans + una) if r.emergency)

    answered = [r for r in ans if not r.abstained and not r.emergency and not r.error]
    total_pre = sum(r.claims_pre for r in answered)
    total_post = sum(r.claims_post for r in answered)
    unsup_pre = sum(r.unsupported_pre for r in answered)
    unsup_post = sum(r.unsupported_post for r in answered)
    cited_post = sum(r.cited_post for r in answered)

    return Metrics(
        n_cases=len(results),
        abstention_accuracy=(abst_correct / abst_total) if abst_total else 0.0,
        false_abstentions=false_abstentions,
        false_answers=false_answers,
        emergency_recall=emergency_recall,
        false_emergencies=false_emergencies,
        citation_coverage=(cited_post / total_post) if total_post else 1.0,
        hallucination_rate_pre=(unsup_pre / total_pre) if total_pre else 0.0,
        hallucination_rate_post=(unsup_post / total_post) if total_post else 0.0,
        total_claims_pre=total_pre,
        total_claims_post=total_post,
    )


def run(limit: int | None = None, use_judge: bool = True,
        progress=None) -> tuple[list[CaseResult], Metrics]:
    cases = load_testset()
    if limit:
        cases = cases[:limit]
    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        if progress:
            progress(i, len(cases), case)
        results.append(run_case(case, use_judge=use_judge))
    metrics = compute_metrics(results)
    _save(results, metrics)
    return results, metrics


def _save(results: list[CaseResult], metrics: Metrics) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"metrics": asdict(metrics), "cases": [asdict(r) for r in results]},
            indent=2,
        ),
        encoding="utf-8",
    )
