"""The full pipeline (Phases 4 + 5 + 6).

    red-flag check  ->  retrieve KB  ->  [+ report sources]  ->  rule-based abstention
                    ->  cited answer (Claude #1)  ->  faithfulness (Claude #2)
                    ->  context-aware disclaimer

Order matters:
- The red-flag/emergency check runs FIRST and, if triggered, returns a
  "seek immediate care" message immediately — no retrieval, no Claude call.
- Rule-based abstention runs before any Claude call, thresholding on KB
  retrieval strength (skipped when a report gives us a real source).
- The faithfulness pass runs LAST and can turn a seemingly-answerable result
  into an abstention if no claim survives verification.
- Report sources are listed FIRST, so [S1], [S2], ... are "your report" values.
"""
from __future__ import annotations

import config
from src import abstention, answer, confidence, faithfulness, retrieval, safety
from src.faithfulness import VerifiedAnswer
from src.report import Report


def answer_question(
    question: str,
    report: Report | None = None,
    top_k: int = config.TOP_K,
    reading_level: str = "standard",
    retrieval_query: str | None = None,
) -> VerifiedAnswer:
    """Run the full pipeline for one question.

    `reading_level` ("standard" | "simple") tunes only the answer's wording.
    `retrieval_query` lets the caller search with extra context (e.g. a
    multi-turn UI that prepends the previous question so a follow-up like "is
    that high?" still retrieves the right topic); the emergency check and the
    answer itself always use the user's real `question`.
    """
    # 0) Emergency red-flag check — bypasses the whole pipeline if triggered.
    flag = safety.check_red_flags(question)
    if flag.triggered:
        return VerifiedAnswer(
            question=question, answerable=False, kept_claims=[],
            note=flag.message, emergency=True,
        )

    kb_chunks = retrieval.retrieve(retrieval_query or question, top_k=top_k)
    report_chunks = report.to_source_chunks() if report else []

    # 1) Rule-based abstention on KB strength — skipped only when a report gives
    #    us a real source to answer from.
    has_out_of_range = bool(
        report and any(lv.status() in ("high", "low") for lv in report.lab_values)
    )
    disclaimer = safety.disclaimer_for(
        has_report=report is not None, has_out_of_range=has_out_of_range
    )

    if not report_chunks:
        decision = abstention.check(kb_chunks)
        if decision.abstain:
            return VerifiedAnswer(
                question=question, answerable=False, kept_claims=[],
                note="I don't have a reliable source to answer that. "
                     f"({decision.reason})",
                sources=kb_chunks,
                disclaimer=disclaimer,
            )

    # 2) Grounded, cited answer (Claude #1). Report sources first.
    sources = report_chunks + kb_chunks
    cited = answer.answer_question_cited(
        question, chunks=sources, reading_level=reading_level
    )

    # 3) Faithfulness check (Claude #2): drop unsupported claims.
    verified = faithfulness.verify(cited)
    verified.disclaimer = disclaimer

    # 4) Rule-based confidence indicator (Phase 8, no LLM).
    if verified.answerable:
        kb_best = max((c.dense_score for c in kb_chunks), default=0.0)
        conf = confidence.assess(verified, kb_best_dense=kb_best)
        verified.confidence_level = conf.level
        verified.confidence_reasons = conf.reasons
    return verified
