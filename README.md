# Cited Medical Q&A Bot

A medical Q&A bot that never makes things up. Ask a health question — or upload
your own lab report — and get a plain-language answer where **every fact is
backed by a trusted source**. If no reliable source is found, it says
*"I don't know"* instead of guessing.

**Where Claude is used:** answer generation + faithfulness check only.
**Where it is NOT:** OCR, extraction, retrieval, red-flag detection, and
out-of-range checks are all classical, deterministic code.

## Tech stack

- **Backend:** Python
- **LLM:** Claude (Anthropic API) — answer generation + faithfulness check
- **Vector DB:** Chroma (local, file-based)
- **Embeddings:** BGE `bge-small-en-v1.5` (free, local)
- **Keyword search:** `rank_bm25` (hybrid retrieval)
- **OCR (Phase 5):** Tesseract via `pytesseract`
- **Frontend (Phase 8):** Streamlit

## Setup (Phase 0)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env        # then edit .env and add your ANTHROPIC_API_KEY
python scripts/test_claude.py   # confirms Claude is reachable
```

## Build the knowledge base + retrieval (Phase 1)

Drop trusted fact sheets (`.md`, `.txt`, `.pdf`) into `data/docs/`. A few
starter documents from MedlinePlus / WHO are already included — replace or add
to them with official downloads.

```bash
python scripts/build_index.py                       # chunk -> embed -> Chroma + BM25
python scripts/query.py "what is a normal creatinine level?"
```

`query.py` returns the top source chunks with their citation and a retrieval
score. No answer is generated yet — that's Phase 2.

### Adding your own documents

Markdown/text files may start with an optional front-matter block so citations
show a clean title and link:

```
---
title: High Blood Pressure (Hypertension)
source: WHO
url: https://www.who.int/news-room/fact-sheets/detail/hypertension
---
```

Rerun `scripts/build_index.py` after adding or changing documents.

## Ask about your own lab report (Phase 5)

```bash
python scripts/ask_report.py data/samples/sample_lab_report.txt "what is my creatinine and is it high?"
```

The report is OCR'd (or read directly for `.txt`), lab values are extracted by
rules, and each becomes a citable **"Your report"** source alongside the
knowledge base — so the answer cites both your value and the general medical
context, and is faithfulness-checked the same way. The report is per-request
and is **not** added to the persistent index.

**Image / scanned-PDF OCR needs Tesseract installed** (the `pytesseract`
package is only a wrapper). Windows: install from
<https://github.com/UB-Mannheim/tesseract/wiki> and ensure `tesseract.exe` is on
PATH. Scanned PDFs additionally need Poppler. Plain-text (`.txt`) and
text-layer PDFs work with no extra install.

## Project layout

```
config.py               # all paths + model choices
src/
  llm.py                # Claude client (the only LLM entry point)
  chunking.py           # load + chunk documents (rules)
  embeddings.py         # BGE embeddings (local)
  vectorstore.py        # Chroma wrapper
  keyword_index.py      # BM25 index
  retrieval.py          # hybrid retrieval (dense + BM25, RRF fusion)
  answer.py             # Phase 2/3: grounded answer + structured citations (Claude)
  abstention.py         # Phase 4a: rule-based abstention (no LLM)
  faithfulness.py       # Phase 4b: per-claim faithfulness check (Claude)
  pipeline.py           # Phase 4/5: retrieve -> abstain -> answer -> verify
  ocr.py                # Phase 5: OCR / text extraction (Tesseract, rules)
  extraction.py         # Phase 5: rule-based lab-value extraction (regex) + range flag
  report.py             # Phase 5: uploaded report -> citable "Your report" source
  safety.py             # Phase 6: red-flag detection + disclaimers (rules)
  judge.py              # Phase 7: independent LLM-as-judge for claim support
  eval.py               # Phase 7: eval harness + metrics
scripts/
  test_claude.py        # Phase 0 acceptance test
  build_index.py        # Phase 1 build
  query.py              # Phase 1 retrieval demo
  ask.py                # ask a question (full pipeline; --no-verify / --plain)
  ask_report.py         # Phase 5: upload a report, ask about your values
  demo_faithfulness.py  # Phase 4: proves unsupported claims get dropped
  run_eval.py           # Phase 7: run the eval harness, print metrics
data/samples/           # sample lab report (.txt and .png) for testing
data/eval/              # testset.jsonl (labeled Q&A) + results.json
data/
  docs/                 # knowledge base source files
  chroma/               # Chroma persistence (gitignored)
  index/                # BM25 corpus (gitignored)
```

## Roadmap

- **Phase 0** — Setup + Claude test call ✅
- **Phase 1** — Knowledge base + hybrid retrieval ✅
- **Phase 2** — Answer generation grounded only in retrieved chunks ✅
- **Phase 3** — Per-sentence citations ✅
- **Phase 4** — Abstention (rule-based) + faithfulness check (Claude) ✅
- **Phase 5** — Document upload + OCR ✅
- **Phase 6** — Safety layer (red flags, scope guardrails, range checks) ✅
- **Phase 7** — Eval harness (hallucination / citation / abstention metrics) ✅
- **Phase 8** — Usability polish (Streamlit UI) ✅

## Run the app (Phase 8)

A Streamlit UI ties the whole pipeline together: chat with multi-turn context,
highlighted/clickable citations, a Simple/Standard reading-level toggle, a
rule-based confidence indicator, lab-report upload, and 👍/👎 feedback that is
logged to `data/eval/feedback.jsonl` to grow the eval set.

```bash
# Windows
.venv\Scripts\streamlit.exe run app.py
# macOS/Linux
streamlit run app.py
```

It opens in your browser. Upload a report from the sidebar to ask about your own
values. For a plain-language tour of what every feature does, see `GUIDE.md`.
