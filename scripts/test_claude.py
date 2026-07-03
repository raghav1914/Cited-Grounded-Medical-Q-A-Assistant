"""Phase 0 acceptance test: prove we can call Claude from Python.

Run:  python scripts/test_claude.py
Done when: this prints a response from Claude.
"""
import sys
from pathlib import Path

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import llm  # noqa: E402
import config  # noqa: E402


def main() -> None:
    print(f"Model: {config.ANSWER_MODEL}")
    try:
        config.require_api_key()
    except RuntimeError as e:
        print(f"\nSetup needed: {e}")
        return
    reply = llm.complete(
        "Reply with exactly one short sentence confirming you are reachable.",
        max_tokens=64,
    )
    print("Claude says:", reply.strip())


if __name__ == "__main__":
    main()
