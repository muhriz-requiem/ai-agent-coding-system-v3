"""v2.9 Phase H: model intelligence, and Phase B: learned rate limits.

The v2.9 thesis, in one place: **stop trusting numbers someone typed.**

Three of the defects these tests pin down are worth stating, because each is a
way the harness was confidently wrong rather than merely unhelpful:

* The ranking weighted a hand-typed ``coding_index`` (0.35) above measured
  success (0.30).  ``groq/qwen3.8-27b`` scored 0.857 on a ``60.0`` typed in
  months earlier while measuring 0.67 across 202 calls with 66 hard errors.
* ``probe_models`` took its targets from ``rank_candidates``, so it could only
  ever confirm the ranking's own favourites.  13 of 17 unannotated seeds were
  locked out of evidence permanently and no probe would ever have noticed.
* ``GROQ_RPM=30`` was simply wrong.  It cost 217 genuine ``http_429``s in one
  reference build -- 281 of 325 cooldowns.  Backoff was never the problem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
import model_router
import providers


# --------------------------------------------------------------- fixtures
@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A workspace whose ``.system/`` registries are writable and isolated."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def routable(monkeypatch):
    """Make a couple of providers routable without any real credentials.

    ``tests/conftest.py`` blanks every provider key for the whole suite, which
    is correct -- a test run must not spend quota or touch a live key -- and it
    means ``enabled_providers()`` is empty by default.  Anything that exercises
    routing has to opt back in, and doing it through ``enabled_providers``
    rather than ``has_credentials`` keeps the rest of the provider policy
    (priority ordering, the default-disabled set) in play.
    """
    def _only(keys=("groq", "openrouter")):
        monkeypatch.setattr(
            providers, "enabled_providers",
            lambda: [p for p in providers.all_provider_defs()
                     if p.key in keys])
    return _only


def _write_health(root: Path, entries: dict) -> None:
    (root / ".system" / "model_health.json").write_text(
        json.dumps({"entries": entries}), encoding="utf-8")


def _entry(calls=0, ok=0, **kw) -> dict:
    row = {"calls": calls, "ok": ok, "empty": 0, "timeout": 0, "error": 0,
           "rate_limited": 0, "parses": 0, "parse_fail": 0,
           "outcomes_ok": 0, "outcomes_bad": 0, "latency_ewma": 20.0}
    row.update(kw)
    return row


# =============================================== H4: measurement > assertion
def test_health_outweighs_capability():
    """The whole point of the inversion, asserted on the constants."""
    assert model_router.HEALTH_WEIGHT > model_router.CAPABILITY_WEIGHT, (
        "v2.9 inverts v2.8's 0.35 capability / 0.30 health; if this is equal "
        "again the typed-in coding_index is back in charge")


def test_capability_term_spread_shrank():
    """Capability must no longer dominate the blend's usable range.

    v2.8's capability term spanned 0.175..0.341, which is wider than the entire
    realistic spread of a health score.  A single stale number could therefore
    outrank 200 observations.
    """
    observed = [m.coding_index for p in providers.all_provider_defs()
                for m in p.models if m.coding_index is not None]
    assert observed, "fixture problem: no annotated models at all"
    spread = (model_router.CAPABILITY_WEIGHT * min(observed) / 70.0,
              model_router.CAPABILITY_WEIGHT * max(observed) / 70.0)
    assert (spread[1] - spread[0]) < 0.20, (
        f"capability still spans {spread[1]-spread[0]:.3f}, too wide to be a "
        f"tie-breaker prior: {spread}")


def test_evidence_label_tracks_observation_count():
    assert model_router._evidence_label({"calls": 0}) == "asserted"
    assert model_router._evidence_label({"calls": 2}) == "sparse"
    mature = {"calls": model_router.HEALTH_MATURE_CALLS}
    assert model_router._evidence_label(mature) == "measured"


def test_candidate_carries_its_evidence(registry, monkeypatch, routable):
    """A routing decision has to be auditable without re-deriving it."""
    routable()
    _write_health(registry, {
        "openrouter|qwen/qwen3.8-27b:free|builder_initial":
            _entry(calls=40, ok=29, error=11, parses=20, parse_fail=1,
                   outcomes_ok=10, outcomes_bad=2),
    })
    monkeypatch.setattr(config, "FREE_MODEL_ROUTING", True)
    ranked = model_router.rank_candidates("builder_initial", workspace_root=registry)
    assert ranked, "no candidates at all"
    assert any(c.evidence == "measured" for c in ranked)


# ================================ H3: probing cannot confirm its own priors
def test_discovery_order_puts_least_observed_first(registry, monkeypatch, routable):
    """The circularity fix, stated as an ordering property."""
    routable()
    _write_health(registry, {
        "openrouter|qwen/qwen3.8-27b:free|builder_initial":
            _entry(calls=202, ok=136, error=66),
        "groq|qwen/qwen3.8-27b|builder_initial":
            _entry(calls=136, ok=136, error=0),
    })
    order = model_router._discovery_order(registry, cap=40)
    assert order, "discovery pool is empty"
    counts = []
    for cand in order:
        key = f"{cand.pdef.key}|{cand.model.model}"
        total = 0
        for k, v in json.loads(
                (registry / ".system" / "model_health.json").read_text(
                    encoding="utf-8"))["entries"].items():
            if k.split("|")[0] + "|" + k.split("|")[1] == key:
                total += int(v.get("calls", 0))
        counts.append(total)
    assert counts == sorted(counts), "probe order is not ascending by calls"
    # The models we already know best must be probed last, if at all.
    assert counts[-1] >= 136, "a well-measured model was probed early"


def test_discovery_order_aggregates_across_roles(registry, routable):
    """The registry keys per role; 'was this tried?' must not depend on which.

    2 calls under one role key and 2 under another is 4 observations of the same
    model, one short of maturity.  Read as never-tried it would sit at the front
    of the probe queue for no reason; read per-role it would look untried twice
    over.  Summing is what makes the ordering mean anything.
    """
    routable(keys=("groq",))
    _write_health(registry, {
        "groq|qwen/qwen3.8-27b|planner": _entry(calls=2),
        "groq|qwen/qwen3.8-27b|builder_initial": _entry(calls=2),
    })
    order = model_router._discovery_order(registry, cap=40)
    target = next((c for c in order
                   if f"{c.pdef.key}|{c.model.model}" == "groq|qwen/qwen3.8-27b"),
                  None)
    assert target is not None, "groq/qwen3.8-27b vanished from the pool"
    assert target.evidence == "sparse", (
        "4 split calls across two role keys did not aggregate")

    # The aggregation has to reach the *ordering*, not just the label.
    observations = {}
    for key, row in json.loads(
            (registry / ".system" / "model_health.json").read_text(
                encoding="utf-8"))["entries"].items():
        ident = "|".join(key.split("|")[:2])
        observations[ident] = observations.get(ident, 0) + int(row["calls"])
    calls_in_order = [observations.get(f"{c.pdef.key}|{c.model.model}", 0)
                      for c in order]
    assert calls_in_order == sorted(calls_in_order), (
        "ordering does not reflect the aggregated call counts")


def test_discovery_order_skips_quarantined(registry, routable):
    routable(keys=("groq",))
    _write_health(registry, {
        "groq|openai/gpt-oss-120b|builder_initial":
            _entry(calls=55, ok=0, error=55),
    })
    order = model_router._discovery_order(registry, cap=40)
    keys = {f"{c.pdef.key}|{c.model.model}" for c in order}
    assert "groq|openai/gpt-oss-120b" not in keys, (
        "a model verified dead is still being spent probe calls on")


# ==================================== B2: quarantine the verified-dead only
def test_zero_success_is_quarantined_regardless_of_error_mix(registry):
    """A 429 on a model that has never once answered is not congestion.

    This is the real reference-build shape: groq/gpt-oss-120b and gpt-oss-20b
    were each 0/55 with 54 of those rate limits.  A rule that spares any model
    whose failures look transient would keep paying for both forever.
    """
    _write_health(registry, {
        "p|gpt-oss-120b|role": _entry(calls=55, ok=0, rate_limited=54, error=1),
    })
    assert model_router.is_quarantined("p", "gpt-oss-120b", registry)


def test_working_but_throttled_model_is_kept(registry):
    """The opposite case, and the one a naive rate check gets wrong."""
    _write_health(registry, {
        "p|throttled|role": _entry(calls=40, ok=12, rate_limited=28),
    })
    assert not model_router.is_quarantined("p", "throttled", registry)


def test_thin_evidence_never_quarantines(registry):
    """Three failures is noise, not a verdict."""
    _write_health(registry, {"p|new|role": _entry(calls=3, ok=0, error=3)})
    assert not model_router.is_quarantined("p", "new", registry)


def test_healthy_model_is_kept(registry):
    _write_health(registry, {"p|good|role": _entry(calls=202, ok=136, error=66)})
    assert not model_router.is_quarantined("p", "good", registry)


def test_quarantine_reason_is_human_readable(registry):
    _write_health(registry, {"p|dead|role": _entry(calls=55, ok=0, error=55)})
    reason = model_router.quarantined_models(registry)["p|dead"]
    assert "0/55" in reason and "never responded" in reason


# ==================================== B1: the RPM limit is learned, not typed
def test_429_ratchets_the_effective_rpm_down(registry, monkeypatch):
    import quota_governor
    monkeypatch.setattr(config, "RPM_SELF_TUNING", True)
    gov = quota_governor.Governor(registry)
    pdef = next(p for p in providers.all_provider_defs() if p.key == "groq")
    before = gov.effective_rpm(pdef)
    gov.set_cooldown("groq", 30, reason="http_429")
    assert gov.effective_rpm(pdef) < before, (
        "a 429 did not lower the rate we will keep sending requests at")


def test_non_429_cooldown_does_not_ratchet(registry, monkeypatch):
    import quota_governor
    monkeypatch.setattr(config, "RPM_SELF_TUNING", True)
    gov = quota_governor.Governor(registry)
    pdef = next(p for p in providers.all_provider_defs() if p.key == "groq")
    before = gov.effective_rpm(pdef)
    gov.set_cooldown("groq", 30, reason="connection reset")
    assert gov.effective_rpm(pdef) == before, (
        "an unrelated failure throttled the rate")


def test_ratchet_respects_the_floor(registry, monkeypatch):
    import quota_governor
    monkeypatch.setattr(config, "RPM_SELF_TUNING", True)
    gov = quota_governor.Governor(registry)
    pdef = next(p for p in providers.all_provider_defs() if p.key == "groq")
    for _ in range(60):
        gov.set_cooldown("groq", 30, reason="http_429")
        gov.ledger["providers"]["groq"]["cooldowns"]["*"] = 0.0
    assert gov.effective_rpm(pdef) >= config.RPM_LEARNED_FLOOR


def test_recovery_never_exceeds_the_configured_ceiling(registry, monkeypatch):
    """A ratchet that cannot recover is a permanent, silent throttle."""
    import quota_governor
    monkeypatch.setattr(config, "RPM_SELF_TUNING", True)
    gov = quota_governor.Governor(registry)
    pdef = next(p for p in providers.all_provider_defs() if p.key == "groq")
    for _ in range(5):
        gov.set_cooldown("groq", 30, reason="http_429")
        gov.ledger["providers"]["groq"]["cooldowns"]["*"] = 0.0
    throttled = gov.effective_rpm(pdef)
    for _ in range(config.RPM_RECOVER_AFTER_CLEAN * 8):
        gov.record(pdef, tokens=5, outcome="ok")
    assert gov.effective_rpm(pdef) > throttled, "never recovered off the floor"
    assert gov.effective_rpm(pdef) <= pdef.rpm, (
        "recovered above the configured ceiling; the governor would then be "
        "back at the rate that caused the 429s")


def test_recovery_is_slower_than_tightening(registry, monkeypatch):
    """Asymmetry is deliberate: a limit that oscillates never settles."""
    assert config.RPM_RATCHET_UP < (1.0 / config.RPM_RATCHET_DOWN) + 0.5


def test_learned_limit_survives_restart_and_day_rollover(registry, monkeypatch):
    """The daily reset must not erase what the provider taught us."""
    import quota_governor
    monkeypatch.setattr(config, "RPM_SELF_TUNING", True)
    gov = quota_governor.Governor(registry)
    pdef = next(p for p in providers.all_provider_defs() if p.key == "groq")
    for _ in range(4):
        gov.set_cooldown("groq", 30, reason="http_429")
    learned = gov.effective_rpm(pdef)
    assert learned < pdef.rpm

    fresh = quota_governor.Governor(registry)
    assert fresh.effective_rpm(pdef) == learned, (
        "a restart forgot the learned limit and resumed at the too-high rate")
    fresh.ledger["providers"]["groq"]["day"] = "1999-01-01"
    fresh._entry("groq")
    assert fresh.effective_rpm(pdef) == learned, (
        "the daily counter reset discarded the learned limit")


def test_self_tuning_can_be_disabled(registry, monkeypatch):
    import quota_governor
    monkeypatch.setattr(config, "RPM_SELF_TUNING", False)
    gov = quota_governor.Governor(registry)
    pdef = next(p for p in providers.all_provider_defs() if p.key == "groq")
    gov.set_cooldown("groq", 30, reason="http_429")
    assert gov.effective_rpm(pdef) == pdef.rpm


def test_can_call_uses_the_learned_limit(registry, monkeypatch):
    """The ratchet has to reach the actual gate, not just be recorded."""
    import quota_governor
    monkeypatch.setattr(config, "RPM_SELF_TUNING", True)
    monkeypatch.setattr(config, "QUOTA_RESERVE_PCT", 0.0)
    gov = quota_governor.Governor(registry)
    pdef = next(p for p in providers.all_provider_defs() if p.key == "groq")
    for _ in range(5):
        gov.set_cooldown("groq", 30, reason="http_429")
        gov.ledger["providers"]["groq"]["cooldowns"]["*"] = 0.0
    allowed = 0
    for _ in range(200):
        ok, reason = gov.can_call(pdef, tokens=0)
        if ok:
            allowed += 1
            gov.record(pdef, tokens=1, outcome="ok")
        else:
            assert reason == "rpm"
            break
    assert allowed == gov.effective_rpm(pdef), (
        f"gate allowed {allowed} but the learned limit is {gov.effective_rpm(pdef)}")


# ================================= H1/H2: live discovery and the roster diff
def test_models_endpoint_flag_is_now_honoured():
    """v2.8 declared this field, set it once, and read it nowhere.

    That is why only OpenRouter was ever refreshed, and why NVIDIA's retired
    gpt-oss-120b/llama-3.3 line was discovered by a human editing a seed list.
    """
    keys = {pdef.key for pdef in providers.all_provider_defs()
            if model_router._models_url(pdef)}
    assert "openrouter" in keys
    assert {"groq", "nvidia"} <= keys, (
        f"live discovery still skips providers: {sorted(keys)}")


def test_models_url_refuses_unresolvable_base_urls():
    """A literal ``{account_id}`` must never reach the wire."""
    cloudflare = next(p for p in providers.all_provider_defs()
                      if p.key == "cloudflare")
    assert model_router._models_url(cloudflare) is None


def test_generic_list_parsing_marks_coding_index_unknown():
    """A discovered model arrives with no asserted capability.

    That is the point: it then gets carried by the discovery-order probe
    instead of being ranked on a number nobody typed.
    """
    pdef = next(p for p in providers.all_provider_defs() if p.key == "groq")
    entries = [{"id": "vendor/new-model"}, {"id": ""}, "junk",
               {"id": "old/retired", "status": "retired"},
               {"id": "vendor/big", "context_length": 200000}]
    models = model_router._models_from_generic_list(pdef, entries)
    ids = {m.model for m in models}
    assert ids == {"vendor/new-model", "vendor/big"}
    assert all(m.coding_index is None for m in models)
    big = next(m for m in models if m.model == "vendor/big")
    assert big.context_length == 200000


def test_refresh_isolates_provider_failures(monkeypatch):
    """One provider failing must not discard another's good refresh."""
    calls = {"n": 0}

    def fake_fetch(pdef, timeout):
        calls["n"] += 1
        if pdef.key == "groq":
            raise RuntimeError("simulated outage")
        return [{"id": f"{pdef.key}/model-a"}, {"id": f"{pdef.key}/model-b"}]

    monkeypatch.setattr(model_router, "_fetch_models", fake_fetch)
    monkeypatch.setattr(providers, "has_credentials", lambda _p: True)
    monkeypatch.setattr(providers, "all_provider_defs",
                        lambda: [p for p in providers.DEFAULT_PROVIDERS
                                 if p.key in ("groq", "nvidia")])
    result = model_router.refresh_catalog(timeout=1, workspace_root=Path("/tmp/x"))

    assert result["providers"]["groq"]["ok"] is False
    assert result["providers"]["nvidia"]["ok"] is True
    assert result["providers"]["nvidia"]["count"] == 2


def test_roster_diff_reports_additions_and_removals(monkeypatch, tmp_path):
    """The diff is how a retirement gets *noticed* rather than discovered."""
    monkeypatch.setattr(model_router, "_fetch_models",
                        lambda pdef, timeout: [{"id": "vendor/brand-new"}])
    monkeypatch.setattr(providers, "has_credentials", lambda _p: True)
    monkeypatch.setattr(providers, "all_provider_defs",
                        lambda: [p for p in providers.DEFAULT_PROVIDERS
                                 if p.key == "groq"])
    result = model_router.refresh_catalog(timeout=1, workspace_root=tmp_path)

    assert result["added"] == ["groq:vendor/brand-new"]
    # groq's three seeds are all absent from that one-model response.
    assert any(m.startswith("groq:") for m in result["removed"])
    assert result["removed"], "a shrinking roster must be reported"


def test_refresh_failure_leaves_seeds_intact(monkeypatch, tmp_path):
    monkeypatch.setattr(model_router, "_fetch_models",
                        lambda pdef, timeout: (_ for _ in ()).throw(
                            RuntimeError("offline")))
    monkeypatch.setattr(providers, "has_credentials", lambda _p: True)
    monkeypatch.setattr(providers, "all_provider_defs",
                        lambda: [p for p in providers.DEFAULT_PROVIDERS
                                 if p.key == "groq"])
    monkeypatch.setattr(config, "MODEL_REGISTRY_TTL_H", 24)
    model_router.refresh_catalog(timeout=1, workspace_root=tmp_path)

    pdef = next(p for p in providers.all_provider_defs() if p.key == "groq")
    models = model_router._models_for(pdef, tmp_path)
    assert {m.model for m in models} == {m.model for m in providers.GROQ_MODELS}, (
        "a failed refresh destroyed the working seed roster")


def test_version_has_a_single_source_of_truth():
    """v2.9 added this: the version lived only as prose and could silently drift.

    Verification found ``main.py`` and ``README.md`` carrying the version only as
    independent prose -- no constant, nothing to assert against, so they could
    disagree indefinitely without anything failing.
    """
    assert providers.VERSION.count(".") == 2
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8")
    assert providers.VERSION in readme, (
        f"README does not mention {providers.VERSION}")
