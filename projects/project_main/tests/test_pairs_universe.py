"""Tests for the pairs universe module (src/pairs/universe.py).

Network-free: the Wikipedia fetch (``_fetch_html``) and the table
parser (``pd.read_html``) are monkeypatched; the cache and fallback
file use ``tmp_path``.
"""

import pandas as pd
import pytest

import src.pairs.universe as universe_mod
from src.data.cache import Cache
from src.pairs.universe import (
    SP400_URL,
    UNIVERSE_CACHE_KEY,
    get_sp400_universe,
    load_fallback_universe,
    refresh_universe,
    screen_universe,
)


def _wiki_table() -> list[pd.DataFrame]:
    return [
        pd.DataFrame({
            "Symbol": ["AAA", "BBB", "CCC", None, "AAA"],
            "Security": ["Alpha", "Beta", "Gamma", "Junk", "Dup"],
            "GICS Sector": ["Technology", "Health Care", "Financials", "X", "Other"],
        }),
        pd.DataFrame({"Unrelated": [1, 2, 3]}),
    ]


def _stub_fetch(monkeypatch, html: str = "<html>t</html>") -> None:
    monkeypatch.setattr(
        universe_mod, "_fetch_html", lambda *a, **k: html
    )


# ---------------------------------------------------------------------------
# get_sp400_universe
# ---------------------------------------------------------------------------


def test_scrape_parses_symbol_and_sector(monkeypatch):
    _stub_fetch(monkeypatch)
    monkeypatch.setattr(pd, "read_html", lambda *a, **k: _wiki_table())
    universe = get_sp400_universe()
    assert universe == {
        "AAA": "Technology",
        "BBB": "Health Care",
        "CCC": "Financials",
    }  # None row dropped, duplicate AAA keeps first sector


def test_scrape_failure_returns_empty(monkeypatch):
    def _boom(*a, **k):
        raise OSError("network down")

    _stub_fetch(monkeypatch)
    monkeypatch.setattr(pd, "read_html", _boom)
    assert get_sp400_universe() == {}


def test_scrape_fetch_failure_returns_empty(monkeypatch):
    def _boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(universe_mod, "_fetch_html", _boom)
    assert get_sp400_universe() == {}


def test_scrape_without_symbol_table_returns_empty(monkeypatch):
    _stub_fetch(monkeypatch)
    monkeypatch.setattr(
        pd, "read_html", lambda *a, **k: [pd.DataFrame({"A": [1]})]
    )
    assert get_sp400_universe() == {}


def test_url_is_the_sp400_list():
    assert "S%26P_400" in SP400_URL


# ---------------------------------------------------------------------------
# screen_universe
# ---------------------------------------------------------------------------


def test_screen_filters_by_adv_and_mcap():
    universe = {"AAA": "Tech", "BBB": "Tech", "CCC": "Fin", "DDD": None}
    result = screen_universe(
        universe,
        adv={"AAA": 5e6, "BBB": 5e5, "CCC": 5e6},
        mcap={"AAA": 5e9, "BBB": 5e9, "CCC": 5e9, "DDD": 5e9},
        min_adv=1e6,
        min_mcap=2e9,
        max_mcap=1e10,
    )
    # BBB fails the ADV floor; AAA/CCC pass; DDD has no adv data -> passes.
    assert set(result) == {"AAA", "CCC", "DDD"}
    assert result["AAA"] == "Tech"
    assert result["DDD"] is None


def test_screen_mcap_ceiling_excludes_large_caps():
    result = screen_universe(
        {"AAA": "Tech", "BBB": "Tech"},
        mcap={"AAA": 5e9, "BBB": 5e10},
        min_mcap=2e9,
        max_mcap=1e10,
    )
    assert set(result) == {"AAA"}


def test_screen_without_data_passes_everything_through():
    universe = {"AAA": "Tech", "": "junk", "BBB": None}
    assert set(screen_universe(universe)) == {"AAA", "BBB"}


def test_screen_junk_bounds_fail_open():
    universe = {"AAA": "Tech"}
    assert screen_universe(
        universe, adv={"AAA": 1.0}, min_adv="junk"
    ) == {"AAA": "Tech"}
    assert screen_universe(None) == {}


# ---------------------------------------------------------------------------
# fallback file
# ---------------------------------------------------------------------------


def test_fallback_file_parses_ticker_sector_lines(tmp_path):
    path = tmp_path / "fallback.txt"
    path.write_text(
        "# comment\nAAA|Technology\n\nBBB\nCCC| \n", encoding="utf-8"
    )
    universe = load_fallback_universe(path)
    assert universe == {"AAA": "Technology", "BBB": None, "CCC": None}


def test_fallback_missing_file_is_empty(tmp_path):
    assert load_fallback_universe(tmp_path / "nope.txt") == {}


# ---------------------------------------------------------------------------
# refresh chain
# ---------------------------------------------------------------------------


def test_refresh_uses_cache_hit_without_scrape(monkeypatch, tmp_path):
    cache = Cache(root=tmp_path / "cache", ttl_seconds=3600)
    cache.set(UNIVERSE_CACHE_KEY, {"AAA": "Tech"})
    calls = []

    def _spy(*a, **k):
        calls.append(1)
        return {}

    _stub_fetch(monkeypatch)
    monkeypatch.setattr(pd, "read_html", _spy)
    assert refresh_universe(cache=cache,
                            fallback_path=tmp_path / "none.txt") == {"AAA": "Tech"}
    assert not calls  # scrape never ran


def test_refresh_scrapes_and_populates_cache(monkeypatch, tmp_path):
    cache = Cache(root=tmp_path / "cache", ttl_seconds=3600)
    _stub_fetch(monkeypatch)
    monkeypatch.setattr(pd, "read_html", lambda *a, **k: _wiki_table())
    universe = refresh_universe(cache=cache,
                                fallback_path=tmp_path / "none.txt")
    assert universe["AAA"] == "Technology"
    assert cache.get(UNIVERSE_CACHE_KEY) == universe


def test_refresh_falls_back_to_file(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback.txt"
    fallback.write_text("AAA|Tech\n", encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("down")

    _stub_fetch(monkeypatch)
    monkeypatch.setattr(pd, "read_html", _boom)
    universe = refresh_universe(cache=None, fallback_path=fallback)
    assert universe == {"AAA": "Tech"}


def test_refresh_total_failure_is_empty(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise OSError("down")

    monkeypatch.setattr(universe_mod, "_fetch_html", _boom)
    assert refresh_universe(
        cache=None, fallback_path=tmp_path / "missing.txt"
    ) == {}
