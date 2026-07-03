"""Phase 5: rule-based extraction of lab values from report text.

Regex + rules only — no LLM (same discipline as the Cumulus scraper). We pull
out structured lab measurements: test name, numeric value, unit, and reference
range when the report prints one. A recognized unit or a reference range is
required for a line to count as a lab value, which keeps out dates, headers,
and free text.

The reference range matters for Phase 6 (out-of-range flagging), but we capture
it here so it travels with the value from the start.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Common lab units. Matching one of these is what promotes a numeric line to a
# lab value (and filters out dates, IDs, etc.).
_UNITS = [
    "mg/dL", "mg/dl", "g/dL", "g/dl", "mmol/L", "mmol/l", "umol/L", "µmol/L",
    "mcg/dL", "ng/mL", "pg/mL", "IU/L", "U/L", "mIU/L", "mEq/L",
    "mL/min/1.73m2", "mL/min", "%", "10^3/uL", "10^6/uL", "K/uL", "M/uL",
    "cells/uL", "fL", "pg", "mm/hr", "bpm", "mmHg",
]
# Longest-first so "mg/dL" wins over a bare unit substring.
_UNITS_SORTED = sorted(_UNITS, key=len, reverse=True)
_UNIT_ALT = "|".join(re.escape(u) for u in _UNITS_SORTED)

_NUM = r"[-+]?\d+(?:\.\d+)?"

# A reference range in one of three printed shapes:
#   "0.6 - 1.2"  |  "< 200"  |  "> 40"
_RANGE_BETWEEN = re.compile(rf"({_NUM})\s*[-–to]+\s*({_NUM})")
_RANGE_LT = re.compile(rf"[<≤]\s*({_NUM})")
_RANGE_GT = re.compile(rf"[>≥]\s*({_NUM})")

# The result value: the first STANDALONE number on the line, optionally followed
# by a unit. The negative lookbehind stops us matching a digit embedded in a
# word like the "1" in "A1c" or "B12" — those belong to the test name.
_RESULT = re.compile(
    rf"(?<![A-Za-z0-9.])({_NUM})\s*(?:({_UNIT_ALT}))?", re.IGNORECASE
)


@dataclass
class LabValue:
    name: str
    value: float
    unit: str | None
    ref_low: float | None
    ref_high: float | None
    ref_text: str          # the reference range exactly as printed (may be "")
    raw_line: str

    def status(self) -> str:
        """Rule-based out-of-range flag: 'high' | 'low' | 'normal' | 'unknown'.

        Pure arithmetic against the printed reference range — never an LLM or a
        diagnosis, just "is this number inside the printed bounds".
        """
        if self.ref_high is not None and self.value > self.ref_high:
            return "high"
        if self.ref_low is not None and self.value < self.ref_low:
            return "low"
        if self.ref_low is not None or self.ref_high is not None:
            return "normal"
        return "unknown"

    def describe(self) -> str:
        """A single plain sentence stating the value + range + rule-based flag."""
        unit = f" {self.unit}" if self.unit else ""
        s = f"{self.name} is {self._fmt(self.value)}{unit}"
        if self.ref_text:
            s += f" (reference range {self.ref_text})"
        phrase = {
            "high": ", which is above the reference range",
            "low": ", which is below the reference range",
            "normal": ", which is within the reference range",
        }.get(self.status(), "")
        return s + phrase + "."

    @staticmethod
    def _fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else str(v)


def _clean_name(text: str) -> str:
    name = text.strip(" :\t.-")
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip()


def _looks_like_date(line: str) -> bool:
    return bool(re.search(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", line))


def _parse_range(after_value: str) -> tuple[float | None, float | None, str]:
    """Find a reference range in the text that follows the result value."""
    m = _RANGE_BETWEEN.search(after_value)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return lo, hi, f"{m.group(1)}-{m.group(2)}"
    m = _RANGE_LT.search(after_value)
    if m:
        return None, float(m.group(1)), f"<{m.group(1)}"
    m = _RANGE_GT.search(after_value)
    if m:
        return float(m.group(1)), None, f">{m.group(1)}"
    return None, None, ""


def extract_lab_values(text: str) -> list[LabValue]:
    values: list[LabValue] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _looks_like_date(stripped):
            continue

        result = _RESULT.search(stripped)
        if not result:
            continue

        name = _clean_name(stripped[: result.start()])
        if not name or not re.search(r"[A-Za-z]", name):
            continue  # need a real test name before the number

        value = float(result.group(1))
        unit = result.group(2)

        after = stripped[result.end():]
        ref_low, ref_high, ref_text = _parse_range(after)

        # Require either a recognized unit or a printed reference range, so we
        # don't grab arbitrary numbers (patient IDs, ages, phone numbers).
        if not unit and not ref_text:
            continue

        values.append(
            LabValue(
                name=name,
                value=value,
                unit=unit,
                ref_low=ref_low,
                ref_high=ref_high,
                ref_text=ref_text,
                raw_line=stripped,
            )
        )
    return values
