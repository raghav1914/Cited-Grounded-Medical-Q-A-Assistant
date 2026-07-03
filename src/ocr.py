"""Phase 5: OCR / text extraction for uploaded reports.

Turns an uploaded file into raw text. No LLM.

- .txt / .md         -> read directly
- .pdf               -> pypdf text layer; if the PDF is scanned (little/no text),
                        fall back to rendering pages and running Tesseract
- .png/.jpg/.tiff... -> Tesseract OCR

Tesseract is an external binary (the `pytesseract` package is only a wrapper).
If it isn't installed we raise a clear, actionable error instead of a cryptic
one. Same for Poppler, which `pdf2image` needs to rasterize scanned PDFs.
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
# If a PDF's text layer has fewer than this many characters, treat it as scanned.
_SCANNED_PDF_THRESHOLD = 50

# Tesseract and Poppler are external binaries. If they're installed but the
# running process didn't inherit the updated PATH (common on Windows right after
# installing), fall back to their default install locations so OCR still works.
_TESSERACT_FALLBACKS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]
_POPPLER_GLOBS = [
    r"C:\poppler*\**\pdftoppm.exe",
    r"C:\Program Files\poppler*\**\pdftoppm.exe",
]


class OCRNotAvailable(RuntimeError):
    """Raised when Tesseract (or Poppler) is needed but not installed."""


def _ensure_tesseract_cmd() -> None:
    """Point pytesseract at the tesseract binary if it isn't on PATH."""
    import pytesseract

    if shutil.which("tesseract"):
        return  # already resolvable via PATH
    for cand in _TESSERACT_FALLBACKS:
        if cand.exists():
            pytesseract.pytesseract.tesseract_cmd = str(cand)
            return


def _find_poppler_path() -> str | None:
    """Return Poppler's bin dir if it isn't on PATH, else None (use PATH)."""
    if shutil.which("pdftoppm"):
        return None
    for pattern in _POPPLER_GLOBS:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return str(Path(hits[0]).parent)
    return None


def extract_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return _ocr_image(path)
    raise ValueError(f"Unsupported file type for a report: {suffix}")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if len(text.strip()) >= _SCANNED_PDF_THRESHOLD:
        return text  # has a usable text layer

    # Scanned PDF: rasterize each page, then OCR.
    try:
        from pdf2image import convert_from_path
    except ImportError as e:  # pragma: no cover
        raise OCRNotAvailable("pdf2image is not installed.") from e

    try:
        images = convert_from_path(str(path), poppler_path=_find_poppler_path())
    except Exception as e:
        raise OCRNotAvailable(
            "This looks like a scanned PDF, which needs Poppler to rasterize. "
            "Install Poppler and ensure it's on PATH, or upload the report as an "
            "image (.png/.jpg) or plain text instead."
        ) from e
    return "\n\n".join(_ocr_pil(img) for img in images)


def _ocr_image(path: Path) -> str:
    from PIL import Image

    return _ocr_pil(Image.open(path))


def _ocr_pil(image) -> str:
    try:
        import pytesseract
    except ImportError as e:  # pragma: no cover
        raise OCRNotAvailable("pytesseract is not installed.") from e
    _ensure_tesseract_cmd()
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as e:
        raise OCRNotAvailable(
            "Tesseract OCR is not installed or not on PATH. Install it "
            "(Windows: https://github.com/UB-Mannheim/tesseract/wiki), then "
            "retry. You can also upload the report as plain text (.txt) to skip "
            "OCR entirely."
        ) from e
