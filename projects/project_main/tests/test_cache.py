"""Tests for the JSON disk cache (src/data/cache.py)."""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.data.cache",)
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from pathlib import Path

from src.data.cache import Cache


def _make(tmp_path: Path, ttl: float = 300.0) -> Cache:
    return Cache(root=tmp_path / "cache", ttl_seconds=ttl)


def test_set_then_get_roundtrip(tmp_path):
    cache = _make(tmp_path)
    cache.set("ohlcv:AAPL:6mo", {"rows": [1, 2, 3]})
    assert cache.get("ohlcv:AAPL:6mo") == {"rows": [1, 2, 3]}


def test_get_missing_returns_none(tmp_path):
    cache = _make(tmp_path)
    assert cache.get("never:set") is None


def test_expired_entry_returns_none(tmp_path):
    import time
    cache = _make(tmp_path, ttl=0.05)
    cache.set("k", {"v": 1})
    time.sleep(0.08)
    assert cache.get("k") is None


def test_get_corrupt_json_returns_none(tmp_path):
    cache = _make(tmp_path)
    cache.set("k", {"v": 1})
    # Corrupt the stored file directly.
    cache_file = cache.path_for("k")
    cache_file.write_text("{not json", encoding="utf-8")
    assert cache.get("k") is None


def test_keys_are_safe_filenames(tmp_path):
    cache = _make(tmp_path)
    cache.set("a/b:c*d", 1)
    stored = cache.path_for("a/b:c*d")
    assert stored.parent == cache.root
    name = stored.name
    assert "/" not in name and ":" not in name and "*" not in name
    assert name.endswith(".json")


def test_set_creates_root_and_overwrites(tmp_path):
    cache = _make(tmp_path)
    cache.set("k", 1)
    cache.set("k", 2)
    assert cache.get("k") == 2
