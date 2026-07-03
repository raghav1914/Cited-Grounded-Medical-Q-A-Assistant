"""Thin wrapper around the Anthropic Claude client.

Claude is used for exactly two jobs in this project: answer generation and the
faithfulness check. Both go through this module so the client is created once
and model choice stays centralized. OCR, extraction, retrieval, red-flag
detection, and range checks are all classical code and never touch this file.
"""
from __future__ import annotations

from typing import TypeVar

import anthropic
from pydantic import BaseModel

import config

_client: anthropic.Anthropic | None = None

T = TypeVar("T", bound=BaseModel)


def get_client() -> anthropic.Anthropic:
    """Return a lazily-created, process-wide Anthropic client."""
    global _client
    if _client is None:
        config.require_api_key()
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Single-shot text completion. Returns the concatenated text blocks.

    Used for the simple call path (Phase 0 test, small prompts). Answer
    generation and faithfulness checking build on top of this.
    """
    client = get_client()
    response = client.messages.create(
        model=model or config.ANSWER_MODEL,
        max_tokens=max_tokens,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def parse(
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
) -> T:
    """Structured completion validated against a Pydantic schema.

    Used where we need machine-parseable output (Phase 3 cited answers,
    Phase 4 faithfulness verdicts) rather than free-form prose. Returns the
    parsed model instance.
    """
    client = get_client()
    response = client.messages.parse(
        model=model or config.ANSWER_MODEL,
        max_tokens=max_tokens,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
    )
    return response.parsed_output
