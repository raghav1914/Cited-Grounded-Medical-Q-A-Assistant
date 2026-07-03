# Manual setup — things to install/configure by hand

Most of the project runs with just `pip install -r requirements.txt`. A few
things need manual installation and are **optional until you need that feature**.

## ✅ Already done
- Python venv + `pip install -r requirements.txt`
- `ANTHROPIC_API_KEY` set in `.env` (Phase 0 verified — Claude reachable)
- BGE embedding model (auto-downloads on first `build_index.py` run)
- **Tesseract OCR** installed + verified — full image-upload path works end to
  end (sample `.png` → 6 lab values extracted → cited answer).
- **Poppler** installed (`pdftoppm -h` works) — enables the scanned-PDF path.
- **Streamlit UI (Phase 8)** built + verified: `.venv\Scripts\streamlit.exe run app.py`.

> Note: `src/ocr.py` auto-detects Tesseract/Poppler at their default install
> locations if they aren't on the current process's PATH, so OCR works even
> right after install without reopening every terminal.

## Reinstalling on a new machine — Tesseract OCR (image / scanned-PDF reports)
Not needed for `.txt` reports or PDFs that already have a text layer.

- **Windows:** install the UB-Mannheim build:
  <https://github.com/UB-Mannheim/tesseract/wiki>
  (default folder `C:\Program Files\Tesseract-OCR`).
- **Add it to PATH** (or rely on the auto-detect fallback in `src/ocr.py`).
- **Verify:** `tesseract --version`, then
  `python scripts/ask_report.py data/samples/sample_lab_report.png "what is my creatinine?"`

## Reinstalling on a new machine — Poppler (scanned PDFs only)
`pdf2image` uses Poppler to turn scanned PDF pages into images before OCR.
Text-layer PDFs and image files don't need it.

- **Windows:** download Poppler binaries
  (<https://github.com/oschwartz10612/poppler-windows/releases>), unzip to
  e.g. `C:\poppler`, and add the `bin` folder to PATH (or rely on auto-detect).
- **Verify:** `pdftoppm -h` runs without error.

---

**Rule of thumb:** if you only ever upload plain-text reports and ask health
questions, you need nothing beyond the ✅ section. Tesseract + Poppler are only
for scanned/photographed reports.
