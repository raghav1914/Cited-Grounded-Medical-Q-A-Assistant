"""Phase 5: an uploaded report as a labeled, citable grounding source.

Pipeline: file -> OCR/text (src.ocr) -> rule-based extraction (src.extraction)
-> Report -> source chunks that look exactly like KB chunks, but labeled
"Your report" so answers can say "according to your report" and cite them the
same way (Phase 3) and have them faithfulness-checked the same way (Phase 4).

The report is per-request and ephemeral — it is NOT added to the persistent
Chroma index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src import extraction, ocr
from src.extraction import LabValue
from src.retrieval import RetrievedChunk

# Report sources are the user's own data — definitionally relevant — so they get
# a max dense score and never trip the rule-based abstention.
_REPORT_DENSE = 1.0


@dataclass
class Report:
    filename: str
    raw_text: str
    lab_values: list[LabValue] = field(default_factory=list)

    def to_source_chunks(self) -> list[RetrievedChunk]:
        """One citable chunk per extracted lab value.

        If nothing structured was extracted, fall back to a single chunk holding
        the raw report text so the report is still usable as a source.
        """
        meta = {
            "source_name": "Your report",
            "source_title": self.filename,
            "source_url": "",
            "doc_id": "user_report",
        }
        chunks: list[RetrievedChunk] = []

        if self.lab_values:
            for i, lv in enumerate(self.lab_values):
                chunks.append(
                    RetrievedChunk(
                        id=f"report::{i}",
                        text=f"According to your uploaded report: {lv.describe()}",
                        metadata=dict(meta, chunk_index=i),
                        rrf_score=0.0,
                        dense_score=_REPORT_DENSE,
                    )
                )
        else:
            chunks.append(
                RetrievedChunk(
                    id="report::0",
                    text="According to your uploaded report:\n" + self.raw_text.strip(),
                    metadata=dict(meta, chunk_index=0),
                    rrf_score=0.0,
                    dense_score=_REPORT_DENSE,
                )
            )
        return chunks


def load_report(path: str | Path) -> Report:
    path = Path(path)
    raw = ocr.extract_text(path)
    lab_values = extraction.extract_lab_values(raw)
    return Report(filename=path.name, raw_text=raw, lab_values=lab_values)
