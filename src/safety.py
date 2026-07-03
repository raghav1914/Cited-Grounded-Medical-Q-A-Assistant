"""Phase 6: the safety layer (classical rules — no LLM).

Three pieces live here:
1. Red-flag / emergency detection — scans the user's question for signs of a
   medical emergency and, if found, produces a "seek immediate care" message
   that BYPASSES the normal answer pipeline (no retrieval, no Claude call).
2. Context-aware disclaimers — a rule-chosen closing disclaimer appended to
   answers (stronger wording when a report has out-of-range values).

Out-of-range lab flagging is a rule too, but it lives on `LabValue.status()`
in src/extraction.py so it travels with the value.

Erring toward triggering is deliberate for emergencies: a false alarm costs the
user a sentence; a miss could cost more.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Each entry: (category, [regex patterns]). Patterns are matched case-insensitively
# with word boundaries where sensible. Kept broad on purpose.
_RED_FLAGS: list[tuple[str, list[str]]] = [
    ("cardiac", [
        r"chest pain", r"chest pressure", r"chest tightness",
        r"pain (?:radiating|spreading) to (?:my )?(?:arm|jaw|shoulder)",
        r"crushing (?:chest )?pain",
    ]),
    ("breathing", [
        r"can'?t breathe", r"cannot breathe", r"trouble breathing",
        r"difficulty breathing", r"struggling to breathe", r"gasping for air",
    ]),
    ("stroke", [
        r"face (?:is )?drooping", r"slurred speech", r"sudden numbness",
        r"sudden weakness", r"one side of (?:my|the) (?:face|body)",
        r"can'?t move (?:my )?(?:arm|leg|face)",
    ]),
    ("self_harm", [
        r"suicid", r"kill myself", r"end my life", r"want to die",
        r"harm myself", r"hurt myself", r"take my own life",
    ]),
    ("bleeding", [
        r"won'?t stop bleeding", r"uncontrolled bleeding", r"severe bleeding",
        r"bleeding (?:a lot|heavily|badly)",
    ]),
    ("unconscious", [
        r"unconscious", r"passed out", r"won'?t wake up", r"unresponsive",
        r"not breathing",
    ]),
    ("anaphylaxis", [
        r"throat (?:is )?(?:swelling|closing)", r"can'?t swallow",
        r"anaphylaxis", r"severe allergic reaction",
    ]),
    ("other_acute", [
        r"seizure", r"convulsions", r"overdose",
        # "poison"/"poisoned"/"poisoning" — but NOT "food poisoning", which is a
        # common non-emergency illness term (eval caught this false positive).
        r"(?<!food )poison(?:ed|ing)?",
        r"coughing up blood", r"vomiting blood", r"blood in (?:my )?vomit",
    ]),
]

_COMPILED = [
    (cat, [re.compile(p, re.IGNORECASE) for p in pats])
    for cat, pats in _RED_FLAGS
]

# Kept ASCII-safe on purpose: the emergency message must never fail to print on
# a limited console or in a log. The UI layer can add visual styling.
_EMERGENCY_MSG = (
    "EMERGENCY: This may describe a medical emergency.\n\n"
    "Call your local emergency number now (911 in the US, 112 in much of "
    "Europe, or your country's equivalent), or go to the nearest emergency "
    "department. I'm an information tool and can't help in an emergency."
)

_SELF_HARM_MSG = (
    "If you are thinking about harming yourself, please reach out for help "
    "right now - you are not alone.\n\n"
    "In the US, call or text 988 (Suicide & Crisis Lifeline). Elsewhere, "
    "contact your local emergency number or a crisis line. If you are in "
    "immediate danger, call emergency services."
)


@dataclass
class RedFlag:
    triggered: bool
    categories: list[str] = field(default_factory=list)
    message: str = ""


def check_red_flags(text: str) -> RedFlag:
    """Scan the user's text for emergency indicators (pure rules)."""
    hits: list[str] = []
    for category, patterns in _COMPILED:
        if any(p.search(text) for p in patterns):
            hits.append(category)

    if not hits:
        return RedFlag(triggered=False)

    # Self-harm gets the crisis-line message (in addition to the emergency one).
    message = _SELF_HARM_MSG if "self_harm" in hits else _EMERGENCY_MSG
    return RedFlag(triggered=True, categories=hits, message=message)


# --- context-aware disclaimers -------------------------------------------

_GENERAL_DISCLAIMER = (
    "This is general health information, not medical advice. Talk to a "
    "healthcare professional about your situation."
)
_REPORT_OUT_OF_RANGE_DISCLAIMER = (
    "Some of your results are outside the reference range. Reference ranges "
    "vary by laboratory, and only your healthcare provider can interpret what "
    "your results mean for you."
)


def disclaimer_for(has_report: bool, has_out_of_range: bool) -> str:
    """Pick the closing disclaimer based on context (rule-based)."""
    if has_report and has_out_of_range:
        return _REPORT_OUT_OF_RANGE_DISCLAIMER
    return _GENERAL_DISCLAIMER
