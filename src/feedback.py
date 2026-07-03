"""Phase 8: capture user feedback (the thumbs) to grow the eval set.

Every thumbs up/down is appended as one JSON line to data/eval/feedback.jsonl.
The shape mirrors data/eval/testset.jsonl (a `question` plus a `label`) so real
usage can be triaged and folded into the Phase 7 eval harness over time. Pure
file I/O — no LLM.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import config

FEEDBACK_PATH: Path = config.DATA_DIR / "eval" / "feedback.jsonl"


def record(
    question: str,
    rating: str,                     # "up" | "down"
    *,
    answerable: bool,
    answer_text: str = "",
    confidence: str = "",
    note: str = "",
) -> None:
    """Append one feedback event. Best-effort: never raise into the UI."""
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "rating": rating,
        "answerable": answerable,
        "confidence": confidence,
        "answer": answer_text,
        "note": note,
    }
    try:
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass  # feedback is nice-to-have; don't break the app if the write fails
