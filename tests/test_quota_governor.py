"""Quota governor: buckets, reserve, cooldowns, role caps, resets."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import config
import providers
from quota_governor import Governor


def _groq(**overrides) -> providers.ProviderDef:
    return replace(providers.provider_by_key("groq"), **overrides)


def test_daily_cap_blocks_and_reports(tmp_path):
    governor = Governor(tmp_path)
    pdef = _groq(rpd=2)
    assert governor.can_call(pdef)[0]
    governor.record(pdef, tokens=10)
    governor.record(pdef, tokens=10)
    allowed, reason = governor.can_call(pdef)
    assert not allowed and reason == "rpd"
    # Cooldown is independent of counters.
    governor.set_cooldown(pdef.key, 60, reason="test")
    assert governor.cooldown_active(pdef.key)


def test_reserve_holds_capacity_back(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QUOTA_RESERVE_PCT", 0.5)
    governor = Governor(tmp_path)
    pdef = _groq(rpd=10)
    for _ in range(5):  # unreserved budget = 5
        assert governor.can_call(pdef)[0]
        governor.record(pdef)
    allowed, reason = governor.can_call(pdef)
    assert not allowed and reason == "rpd"
    # Retries/probes may dip into the reserve.
    assert governor.can_call(pdef, reserve=True)[0]


def test_role_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QUOTA_ROLE_CAPS", {"builder": 1})
    governor = Governor(tmp_path)
    pdef = _groq(rpd=100)
    governor.record(pdef, role="builder")
    allowed, reason = governor.can_call(pdef, role="builder")
    assert not allowed and reason == "role_cap"
    assert governor.can_call(pdef, role="planner")[0]


def test_token_cap(tmp_path):
    governor = Governor(tmp_path)
    pdef = _groq(tpd=1000)
    governor.record(pdef, tokens=900)
    allowed, reason = governor.can_call(pdef, tokens=200)
    assert not allowed and reason == "tpd"


def test_reset_is_in_the_future(tmp_path):
    governor = Governor(tmp_path)
    reset = governor.next_reset(_groq())
    assert reset > datetime.now(timezone.utc)


def test_snapshot_shape(tmp_path, clean_env):
    clean_env.setenv("GROQ_API_KEY", "test")
    governor = Governor(tmp_path)
    snapshot = governor.snapshot()
    keys = [row["provider"] for row in snapshot["providers"]]
    assert "groq" in keys
    assert "total_available_requests" in snapshot


def test_live_openrouter_override_local_rpd(tmp_path, monkeypatch):
    governor = Governor(tmp_path)
    pdef = replace(providers.provider_by_key("openrouter"), rpd=50)
    # Simulate a fresh live reading revealing the higher tier.
    governor._live["openrouter"] = {"used": 10, "limit": 1000,
                                     "remaining": 990, "checked_ts": 1}
    # Local ledger says used=0 but live is authoritative.
    assert governor.available_requests(pdef) == 990
    # can_call uses the live remaining (not the local rpd).
    allowed, reason = governor.can_call(pdef, reserve=True)
    assert allowed
    governor._live["openrouter"]["remaining"] = 0
    allowed, reason = governor.can_call(pdef)
    assert not allowed and reason == "rpd_live"
