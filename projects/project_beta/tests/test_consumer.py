"""Seeded tests for src/consumer.py (must pass offline)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consumer import consume  # noqa: E402


def test_consume_normalizes_token():
    assert consume("  MiXeD Case  ") == "mixed case"


def test_consume_handles_empty_string():
    assert consume("") == ""
