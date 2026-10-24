"""Seeded tests for src/shared_util.py (must pass offline)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared_util import normalize_token  # noqa: E402


def test_normalize_strips_whitespace():
    assert normalize_token("  HeLLo  ") == "hello"


def test_normalize_is_lowercase():
    assert normalize_token("ABC") == "abc"


def test_normalize_rejects_non_string():
    try:
        normalize_token(None)  # type: ignore[arg-type]
    except TypeError:
        assert True
    else:
        raise AssertionError("expected TypeError for non-string input")
