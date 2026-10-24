"""Tests for the deterministic flow-synthesis analyst
(src/agents/flow_analyst.py)."""


def _impl_ready() -> bool:
    import importlib.util

    modules = ("src.agents.flow_analyst", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.agents.flow_analyst import build_flow_report
from src.agents.models import AnalystReport


def _insider_ok(**overrides) -> dict:
    envelope = {
        "status": "ok",
        "cluster_buys": 3,
        "cluster_sells": 0,
        "net_insider_flow": 1_200_000.0,
        "citations": ["Form 4: CFO bought 10,000 shares"],
    }
    envelope.update(overrides)
    return envelope


def _institutional_ok(**overrides) -> dict:
    envelope = {
        "status": "ok",
        "increased_positions": 12,
        "decreased_positions": 4,
        "whale_alerts": [{"manager": "Berkshire", "action": "added"}],
        "citations": ["13F: Berkshire added 1,000,000 shares"],
    }
    envelope.update(overrides)
    return envelope


def _options_ok(**overrides) -> dict:
    envelope = {
        "status": "ok",
        "call_put_premium_ratio": 2.1,
        "total_unusual_premium": 5_000_000.0,
        "citations": ["UOA: call 450 2026-10-16 vol 5,000/oi 1,000 (5.0x)"],
    }
    envelope.update(overrides)
    return envelope


# ---------------------------------------------------------------------------
# Dispatch gating
# ---------------------------------------------------------------------------


def test_returns_none_without_usable_data():
    assert build_flow_report("AAPL", None) is None
    assert build_flow_report("AAPL", {}) is None
    assert build_flow_report("AAPL", "junk") is None
    assert build_flow_report("AAPL", {"insider": {"status": "no_data"}}) is None
    assert build_flow_report("AAPL", {"insider": {"status": "error",
                                                  "reason": "down"}}) is None
    assert build_flow_report("AAPL", {"insider": "not-a-dict"}) is None


def test_single_ok_source_produces_report():
    report = build_flow_report("AAPL", {"insider": _insider_ok()})
    assert report is not None
    assert report.agent == "flow"
    assert report.ticker == "AAPL"


def test_mixed_signals_produce_deterministic_stance():
    mixed = {
        "insider": _insider_ok(cluster_buys=0, cluster_sells=3,
                               net_insider_flow=-500_000.0),  # -0.20
        "institutional": _institutional_ok(
            increased_positions=2, decreased_positions=8,
            whale_alerts=[{"manager": "X", "action": "trimmed"}]),  # -0.15
        "options_flow": _options_ok(call_put_premium_ratio=0.8),  # 0
    }
    report = build_flow_report("AAPL", mixed)
    assert report is not None
    # net = -0.35 -> bearish
    assert report.stance == "bearish"
    assert report.confidence == 0.85


# ---------------------------------------------------------------------------
# Rubric arithmetic (deterministic)
# ---------------------------------------------------------------------------


def test_all_bullish_scores():
    report = build_flow_report("AAPL", {
        "insider": _insider_ok(),
        "institutional": _institutional_ok(),
        "options_flow": _options_ok(),
    })
    assert report is not None
    # insider +0.20, institutional +0.15, options +0.10 -> net +0.45
    assert report.stance == "bullish"
    assert report.confidence == 0.95  # 0.5 + 0.45 clamped at MAX


def test_all_bearish_scores():
    report = build_flow_report("AAPL", {
        "insider": _insider_ok(cluster_buys=0, cluster_sells=4,
                               net_insider_flow=-900_000.0),
        "institutional": _institutional_ok(
            increased_positions=1, decreased_positions=9,
            whale_alerts=[{"manager": "X", "action": "trimmed"}]),
        "options_flow": _options_ok(call_put_premium_ratio=0.4),
    })
    assert report is not None
    # insider -0.20, institutional -0.15, options -0.15 -> net -0.50
    assert report.stance == "bearish"
    assert report.confidence == 0.95


def test_mild_signals_stay_neutral():
    report = build_flow_report("AAPL", {
        "insider": _insider_ok(cluster_buys=0, cluster_sells=0,
                               net_insider_flow=10.0),  # +0.05
    })
    assert report is not None
    assert report.stance == "neutral"
    assert report.confidence == 0.55


def test_thesis_states_the_arithmetic():
    report = build_flow_report("AAPL", {"insider": _insider_ok()})
    assert "deterministic rubric" in report.thesis
    assert "net=+0.20" in report.thesis


# ---------------------------------------------------------------------------
# Citation contract (min_length=1, evidence-grounded)
# ---------------------------------------------------------------------------


def test_citations_quote_envelope_evidence():
    report = build_flow_report("AAPL", {
        "insider": _insider_ok(),
        "institutional": _institutional_ok(),
        "options_flow": _options_ok(),
    })
    assert len(report.citations) >= 3
    assert "Form 4: CFO bought 10,000 shares" in report.citations
    assert "13F: Berkshire added 1,000,000 shares" in report.citations
    # Factual rubric lines are always appended as evidence too.
    assert any(c.startswith("insider: cluster_buys=") for c in report.citations)


def test_citations_fallback_when_envelope_has_none():
    report = build_flow_report("AAPL", {
        "insider": _insider_ok(citations=[]),
    })
    assert report is not None
    assert len(report.citations) >= 1
    assert all(isinstance(c, str) and c.strip() for c in report.citations)


def test_report_validates_against_schema():
    report = build_flow_report("AAPL", {"insider": _insider_ok()})
    assert isinstance(report, AnalystReport)


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------


def test_never_raises_on_junk_envelopes():
    junk = {
        "insider": {"status": "ok"},  # missing every scored field
        "institutional": {"status": "ok", "whale_alerts": "not-a-list"},
        "options_flow": {"status": "ok", "call_put_premium_ratio": "junk"},
    }
    report = build_flow_report("AAPL", junk)
    assert report is not None  # ok sources still synthesize (all zeros)
    assert report.stance == "neutral"
    assert report.confidence == 0.5
