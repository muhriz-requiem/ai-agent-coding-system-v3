"""Tests for the persisted pair book (src/pairs/book.py).

Covers the JSON round-trip, atomic save, the formation-due schedule,
and the fail-safe degradation paths (missing/corrupt/junk state).
"""

import json
from datetime import date, timedelta

import pytest

from src.pairs.book import BookEntry, PairBook, parse_iso_date


def _entry(pair_id: str = "AAA/BBB", **overrides) -> BookEntry:
    payload = dict(
        pair_id=pair_id,
        ticker_a=pair_id.split("/")[0],
        ticker_b=pair_id.split("/")[1],
        hedge_ratio=1.2,
        alpha=0.5,
        spread_mean=0.01,
        spread_std=0.03,
        half_life=12.0,
        formed_on="2026-09-01",
        spread_daily_vol=0.004,
        formation_corr=0.85,
        formation_adv={"AAA": 5_000_000.0, "BBB": None},
        sector_a="Tech",
        sector_b="Tech",
        adf_pvalue=0.01,
    )
    payload.update(overrides)
    return BookEntry(**payload)


# ---------------------------------------------------------------------------
# parse_iso_date
# ---------------------------------------------------------------------------


def test_parse_iso_date_variants():
    assert parse_iso_date("2026-09-01") == date(2026, 9, 1)
    assert parse_iso_date(date(2026, 9, 1)) == date(2026, 9, 1)
    assert parse_iso_date(" 2026-09-01 ") == date(2026, 9, 1)
    assert parse_iso_date("") is None
    assert parse_iso_date("junk") is None
    assert parse_iso_date(None) is None
    assert parse_iso_date(20260901) is None


# ---------------------------------------------------------------------------
# load / save round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "book.json"
    book = PairBook(path)
    book.replace([_entry(), _entry("CCC/DDD", hedge_ratio=0.8)],
                 as_of="2026-09-01")
    loaded = PairBook.load(path)
    assert len(loaded) == 2
    assert loaded.formed_on == "2026-09-01"
    entry = loaded.get("AAA/BBB")
    assert entry is not None
    assert entry.hedge_ratio == pytest.approx(1.2)
    assert entry.alpha == pytest.approx(0.5)
    assert entry.spread_daily_vol == pytest.approx(0.004)
    assert entry.formation_corr == pytest.approx(0.85)
    assert entry.formation_adv == {"AAA": 5_000_000.0, "BBB": None}
    assert entry.sector_a == "Tech"
    assert loaded.tickers() == ["AAA", "BBB", "CCC", "DDD"]


def test_load_missing_file_is_empty(tmp_path):
    book = PairBook.load(tmp_path / "nope.json")
    assert len(book) == 0
    assert book.formed_on is None
    assert book.formation_due("2026-09-01")  # empty book is always due


def test_load_corrupt_file_is_empty(tmp_path):
    path = tmp_path / "book.json"
    path.write_text("{not json", encoding="utf-8")
    assert len(PairBook.load(path)) == 0


def test_load_junk_entries_are_dropped(tmp_path):
    path = tmp_path / "book.json"
    path.write_text(json.dumps({
        "formed_on": "2026-09-01",
        "entries": {
            "GOOD/PAIR": {
                "ticker_a": "GOOD", "ticker_b": "PAIR",
                "hedge_ratio": 1.0, "half_life": 10.0,
            },
            "BROKEN/PAIR": {"hedge_ratio": "junk"},
            "NO/TICK": {"hedge_ratio": 1.0, "half_life": 5.0},
        },
    }), encoding="utf-8")
    book = PairBook.load(path)
    assert list(book.entries) == ["GOOD/PAIR"]


def test_save_is_atomic_no_tmp_leftovers(tmp_path):
    path = tmp_path / "book.json"
    book = PairBook(path)
    book.replace([_entry()], as_of="2026-09-01")
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert path.exists()


def test_save_failure_raises_but_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "book.json"
    book = PairBook(path)
    book.replace([_entry()], as_of="2026-09-01")
    original = path.read_text(encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", _boom)
    with pytest.raises(OSError):
        book.replace([_entry("XXX/YYY")], as_of="2026-09-02")
    assert path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# formation_due
# ---------------------------------------------------------------------------


def test_formation_due_schedule(tmp_path):
    book = PairBook(tmp_path / "book.json")
    book.replace([_entry()], as_of="2026-09-01")
    assert book.formation_due("2026-09-01", rebalance_days=21) is False
    assert book.formation_due("2026-09-15", rebalance_days=21) is False
    assert book.formation_due("2026-09-22", rebalance_days=21) is True
    assert book.formation_due("2026-09-02", rebalance_days=0) is True


def test_formation_due_clock_skew_not_due(tmp_path):
    book = PairBook(tmp_path / "book.json")
    book.replace([_entry()], as_of="2026-09-01")
    assert book.formation_due("2026-08-15", rebalance_days=21) is False


def test_formation_due_corrupt_state_is_due(tmp_path):
    book = PairBook(tmp_path / "book.json")
    book.replace([_entry()], as_of="2026-09-01")
    book.formed_on = "junk"
    assert book.formation_due("2026-09-02", rebalance_days=21) is True
    book.formed_on = None
    assert book.formation_due("2026-09-02", rebalance_days=21) is True


def test_formation_due_empty_book(tmp_path):
    assert PairBook(tmp_path / "book.json").formation_due("2026-09-01") is True


def test_formation_due_junk_interval_defaults(tmp_path):
    book = PairBook(tmp_path / "book.json")
    book.replace([_entry()], as_of="2026-09-01")
    assert book.formation_due("2026-09-02", rebalance_days="junk") is False
    assert book.formation_due("2026-09-22", rebalance_days="junk") is True


# ---------------------------------------------------------------------------
# replace / get / tickers
# ---------------------------------------------------------------------------


def test_replace_swaps_the_whole_book(tmp_path):
    book = PairBook(tmp_path / "book.json")
    book.replace([_entry(), _entry("CCC/DDD")], as_of="2026-09-01")
    book.replace([_entry("EEE/FFF")], as_of="2026-09-22")
    assert list(book.entries) == ["EEE/FFF"]
    assert book.formed_on == "2026-09-22"


def test_replace_drops_junk_entries(tmp_path):
    book = PairBook(tmp_path / "book.json")
    book.replace([_entry(), None, "junk"], as_of="2026-09-01")
    assert list(book.entries) == ["AAA/BBB"]


def test_get_returns_none_for_unknown(tmp_path):
    book = PairBook(tmp_path / "book.json")
    book.replace([_entry()], as_of="2026-09-01")
    assert book.get("AAA/BBB") is not None
    assert book.get("NOPE/NOPE") is None


def test_tickers_sorted_unique(tmp_path):
    book = PairBook(tmp_path / "book.json")
    book.replace([_entry("BBB/AAA"), _entry("AAA/CCC")], as_of="2026-09-01")
    assert book.tickers() == ["AAA", "BBB", "CCC"]
