"""Tests for the dark-pool (FINRA ATS/TRF) fetch module (src/data/dark_pool.py).

The FINRA HTTP seam is monkeypatched -- no network, no API key needed.
"""


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.data.dark_pool") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import pytest

import src.data.dark_pool as dp_mod
from src.data.cache import Cache

# Latest week (08-24) TRF 4.2M / total 10M -> 42%; baseline weeks
# 08-17/08-10/08-03 average 3.8M shares and 38% of total.
ROWS = [
    {"weekStartDate": "2026-08-24", "market": "TRF", "totalWeeklyShareQuantity": 4_200_000},
    {"weekStartDate": "2026-08-24", "market": "ATS", "totalWeeklyShareQuantity": 5_800_000},
    {"weekStartDate": "2026-08-17", "market": "TRF", "totalWeeklyShareQuantity": 4_000_000},
    {"weekStartDate": "2026-08-17", "market": "ATS", "totalWeeklyShareQuantity": 6_000_000},
    {"weekStartDate": "2026-08-10", "market": "TRF", "totalWeeklyShareQuantity": 3_800_000},
    {"weekStartDate": "2026-08-10", "market": "ATS", "totalWeeklyShareQuantity": 6_200_000},
    {"weekStartDate": "2026-08-03", "market": "TRF", "totalWeeklyShareQuantity": 3_600_000},
    {"weekStartDate": "2026-08-03", "market": "ATS", "totalWeeklyShareQuantity": 6_400_000},
]


def test_computes_pct_average_and_zscore(monkeypatch):
    monkeypatch.setenv("FINRA_API_KEY", "test-key")
    monkeypatch.setattr(dp_mod, "_fetch_weekly", lambda ticker: ROWS)
    env = dp_mod.fetch_dark_pool_proxy("AAPL")
    assert env["status"] == "ok"
    assert env["dp_volume_latest"] == pytest.approx(4_200_000)
    assert env["dp_volume_4wk_avg"] == pytest.approx(3_800_000)
    assert env["dp_pct_latest"] == pytest.approx(0.42)
    assert env["dp_pct_4wk_avg"] == pytest.approx(0.38)
    baseline_std = ((0.2e6) ** 2 / 3 + 0 + (0.2e6) ** 2 / 3) ** 0.5
    assert env["dp_volume_zscore"] == pytest.approx(0.4e6 / baseline_std)
    assert env["citations"] == [
        "FINRA ATS: TRF volume 42% of total (4wk avg 38%)"
    ]


def test_missing_key_is_error_envelope(monkeypatch):
    monkeypatch.delenv("FINRA_API_KEY", raising=False)
    env = dp_mod.fetch_dark_pool_proxy("AAPL")
    assert env["status"] == "error"
    assert "FINRA_API_KEY" in env["reason"]


def test_api_failure_is_error_envelope(monkeypatch):
    monkeypatch.setenv("FINRA_API_KEY", "test-key")

    def boom(ticker):
        raise RuntimeError("http 403")

    monkeypatch.setattr(dp_mod, "_fetch_weekly", boom)
    env = dp_mod.fetch_dark_pool_proxy("AAPL")
    assert env["status"] == "error"
    assert "http 403" in env["reason"]


def test_no_trf_rows_is_no_data(monkeypatch):
    monkeypatch.setenv("FINRA_API_KEY", "test-key")
    monkeypatch.setattr(dp_mod, "_fetch_weekly", lambda ticker: [
        {"weekStartDate": "2026-08-24", "market": "ATS",
         "totalWeeklyShareQuantity": 10_000_000},
    ])
    env = dp_mod.fetch_dark_pool_proxy("AAPL")
    assert env["status"] == "no_data"


def test_unparseable_rows_degrade_to_no_data(monkeypatch):
    monkeypatch.setenv("FINRA_API_KEY", "test-key")
    monkeypatch.setattr(dp_mod, "_fetch_weekly", lambda ticker: ["junk", 42])
    env = dp_mod.fetch_dark_pool_proxy("AAPL")
    assert env["status"] == "no_data"


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FINRA_API_KEY", "test-key")
    calls = []

    def seam(ticker):
        calls.append(ticker)
        return ROWS

    monkeypatch.setattr(dp_mod, "_fetch_weekly", seam)
    cache = Cache(root=tmp_path, ttl_seconds=60)

    first = dp_mod.fetch_dark_pool_proxy("AAPL", cache=cache)
    second = dp_mod.fetch_dark_pool_proxy("AAPL", cache=cache)

    assert calls == ["AAPL"]
    assert first == second
