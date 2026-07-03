"""Phase 8: the Streamlit UI for the Cited Medical Q&A Bot.

A thin presentation layer over `src.pipeline.answer_question`. It adds the
usability polish for the final phase:

- a chat interface with multi-turn history (follow-ups reuse earlier context on
  the retrieval side only — the answer itself is still grounded solely in the
  sources Claude is shown);
- highlighted, clickable citations (inline [S#] badges + expandable sources);
- a reading-level toggle (Simple / Standard);
- a rule-based confidence indicator with its reasons;
- thumbs-up/down feedback that is logged to grow the eval set;
- lab-report upload, shown alongside the knowledge base as "Your report".

Run it with:  .venv\\Scripts\\streamlit.exe run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src import feedback, pipeline, report as report_mod
from src.ocr import OCRNotAvailable

# --- page setup ----------------------------------------------------------
st.set_page_config(page_title="Cited Medical Q&A", page_icon="🩺", layout="centered")

_CONF_COLOR = {"high": "green", "medium": "orange", "low": "red"}
_STATUS_COLOR = {"high": "red", "low": "orange", "normal": "green", "unknown": "gray"}
_REPORT_TYPES = ["txt", "md", "pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"]

st.session_state.setdefault("history", [])       # list of {question, answer}
st.session_state.setdefault("recorded", set())   # feedback keys already logged


# --- sidebar: settings, report upload, about -----------------------------
def sidebar() -> tuple[str, object]:
    with st.sidebar:
        st.header("Settings")
        level_label = st.radio(
            "Reading level",
            ["Standard", "Simple"],
            help="Simple rewrites answers at about a 6th-grade level. "
                 "Facts and citations never change.",
        )
        reading_level = "simple" if level_label == "Simple" else "standard"

        st.divider()
        st.subheader("Your lab report (optional)")
        uploaded = st.file_uploader(
            "Upload a report to ask about your own values",
            type=_REPORT_TYPES,
            help="Plain text (.txt) and text PDFs work out of the box. "
                 "Photos/scanned PDFs need Tesseract/Poppler installed.",
        )
        report = _handle_upload(uploaded)

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state.recorded = set()
            st.rerun()

        st.divider()
        st.caption(
            "This tool answers only from a small set of trusted fact sheets and "
            "your uploaded report, always with citations. If it has no reliable "
            "source, it says so instead of guessing. It does not diagnose or "
            "prescribe. Not a substitute for professional care."
        )
    return reading_level, report


def _handle_upload(uploaded) -> object:
    if uploaded is None:
        st.session_state.pop("report", None)
        st.session_state.pop("report_name", None)
        return None

    if st.session_state.get("report_name") != uploaded.name:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        try:
            rep = report_mod.load_report(tmp_path)
            rep.filename = uploaded.name  # show the real name, not the temp file
            st.session_state.report = rep
            st.session_state.report_name = uploaded.name
        except OCRNotAvailable as e:
            st.error(f"Couldn't read that file: {e}")
            st.session_state.pop("report", None)
            st.session_state.pop("report_name", None)
        except Exception as e:  # noqa: BLE001 - surface any read error to the user
            st.error(f"Couldn't process the report: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    rep = st.session_state.get("report")
    if rep is not None:
        _show_report_values(rep)
    return rep


def _show_report_values(rep) -> None:
    if not rep.lab_values:
        st.caption(f"Loaded **{rep.filename}** — no structured lab values found; "
                   "the raw text is still usable as a source.")
        return
    st.caption(f"Loaded **{rep.filename}** — {len(rep.lab_values)} value(s):")
    for lv in rep.lab_values:
        status = lv.status()
        color = _STATUS_COLOR.get(status, "gray")
        st.markdown(f"- {lv.name}: **{lv._fmt(lv.value)}** "
                    f"{lv.unit or ''} :{color}-background[{status}]")


# --- answer rendering ----------------------------------------------------
def render_answer(verified, turn_index: int) -> None:
    if verified.emergency:
        st.error(verified.note)
        return

    if not verified.answerable:
        st.warning(verified.note)
        if verified.disclaimer:
            st.caption(verified.disclaimer)
        _feedback(verified, turn_index)
        return

    # The answer, sentence by sentence, with inline highlighted citations.
    parts: list[str] = []
    for claim in verified.kept_claims:
        if claim.is_meta:
            parts.append(f"_{claim.text}_")
        else:
            badges = " ".join(f":blue-background[S{n}]" for n in claim.source_ids)
            parts.append(f"{claim.text} {badges}".strip())
    st.markdown(" ".join(parts))

    if verified.note:
        st.caption(verified.note)

    _confidence(verified)
    _sources(verified)
    _dropped(verified)

    if verified.disclaimer:
        st.caption(f"ℹ️ {verified.disclaimer}")

    _feedback(verified, turn_index)


def _confidence(verified) -> None:
    level = verified.confidence_level or "low"
    color = _CONF_COLOR.get(level, "gray")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f":{color}-background[**Confidence: {level.title()}**]")
    with col2:
        with st.popover("Why?", use_container_width=False):
            for r in verified.confidence_reasons:
                st.markdown(f"- {r}")


def _sources(verified) -> None:
    used = sorted({n for c in verified.kept_claims
                   if not c.is_meta for n in c.source_ids})
    if not used:
        return
    st.markdown("**Sources**")
    for n in used:
        chunk = verified.sources[n - 1]
        with st.expander(f"S{n} · {chunk.citation}"):
            st.write(chunk.text)
            url = chunk.metadata.get("source_url")
            if url:
                st.markdown(f"[Open source ↗]({url})")


def _dropped(verified) -> None:
    if not verified.dropped_claims:
        return
    with st.expander(f"Filtered out ({len(verified.dropped_claims)}) — "
                     "statements that failed the source check"):
        for d in verified.dropped_claims:
            st.markdown(f"- ~~{d.claim.text}~~ — _{d.reason}_")


def _feedback(verified, turn_index: int) -> None:
    key = f"fb_{turn_index}"
    choice = st.feedback("thumbs", key=key)
    if choice is not None and key not in st.session_state.recorded:
        answer_text = " ".join(
            c.text for c in verified.kept_claims if not c.is_meta
        ) or verified.note
        feedback.record(
            verified.question,
            "up" if choice == 1 else "down",
            answerable=verified.answerable,
            answer_text=answer_text,
            confidence=verified.confidence_level,
            note=verified.note,
        )
        st.session_state.recorded.add(key)
        st.toast("Thanks — your feedback was saved.")


# --- main ----------------------------------------------------------------
def main() -> None:
    st.title("🩺 Cited Medical Q&A")
    st.caption("Every fact backed by a trusted source, or it says \"I don't know.\"")

    reading_level, report = sidebar()

    # Replay the conversation so far.
    for i, turn in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_answer(turn["answer"], i)

    prompt = st.chat_input("Ask a health question…")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    # Multi-turn: give retrieval the last couple of questions for context, but
    # the emergency check and the answer still use the user's real words.
    prior = [t["question"] for t in st.session_state.history][-2:]
    retrieval_query = " ".join(prior + [prompt]) if prior else None

    with st.chat_message("assistant"):
        with st.spinner("Checking trusted sources…"):
            verified = pipeline.answer_question(
                prompt,
                report=report,
                reading_level=reading_level,
                retrieval_query=retrieval_query,
            )
        render_answer(verified, len(st.session_state.history))

    st.session_state.history.append({"question": prompt, "answer": verified})


main()
