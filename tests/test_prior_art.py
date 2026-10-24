"""v3 research lane: GitHub prior-art digests (OVERHAUL_PLAN Phase 4).

Prior-art is a digest of an unauthenticated GitHub repository search,
passively injected behind ``config.PRIOR_ART_ENABLED`` (default OFF):
repositories and license names to learn *ideas* from -- never code to
vendor.  No token is ever attached (auth-neutral lane); requests honor
403/Retry-After by degrading to a clean diagnostic.  pytest never touches
the network: the live HTTP seam (``research._http_get_json``) is
monkeypatched, and the demo path is deterministic by contract.
"""

from __future__ import annotations

import asyncio
import json

import config
import redaction
import research
import tool_registry
from validation import TaskSpec


def _task() -> TaskSpec:
    return TaskSpec(id="T-RES-2", title="Prior-art probe",
                    objective="Build an electron tray menu",
                    target_files=["src/tray.py"])


def _gates(monkeypatch, prior=True, demo=True):
    monkeypatch.setattr(config, "PRIOR_ART_ENABLED", prior)
    monkeypatch.setattr(config, "DEMO_MODE", demo)


# ---------------------------------------------------------------------------
# Gate OFF: invisible everywhere
# ---------------------------------------------------------------------------
def test_gate_off_digest_and_block_and_tools(monkeypatch):
    _gates(monkeypatch, prior=False)
    expected = {"tool": "research.prior_art", "enabled": False}
    assert asyncio.run(research.prior_art_digest("tray menus", config.WORKSPACE_ROOT)) \
        == expected
    assert asyncio.run(research.research_context_block(_task(), config.WORKSPACE_ROOT)) \
        == ""

    names = [t["name"] for t in tool_registry.resolve_required_tools(_task(), "planner")]
    assert "research.prior_art" not in names
    assert names == tool_registry.ROLE_TOOL_MAP["planner"]


def test_gate_on_appends_prior_art_tool(monkeypatch):
    _gates(monkeypatch, prior=True, demo=False)
    names = [t["name"] for t in tool_registry.resolve_required_tools(_task(), "builder")]
    assert names == tool_registry.ROLE_TOOL_MAP["builder"] + ["research.prior_art"]


# ---------------------------------------------------------------------------
# DEMO path: deterministic digest shape
# ---------------------------------------------------------------------------
def test_demo_digest_shape(tmp_path, monkeypatch):
    _gates(monkeypatch)
    res = asyncio.run(research.prior_art_digest("electron tray menus", tmp_path))
    assert res["source"] == "demo" and res["topic"] == "electron tray menus"
    assert res["cache"] is False
    assert res["repos"], "demo must show what a populated digest looks like"
    repo = res["repos"][0]
    assert set(repo) >= {"name", "description", "license", "stars", "url"}
    assert repo["description"].startswith("demo")


# ---------------------------------------------------------------------------
# LIVE path: request shape (no token), digest fields, cache, clean 403
# ---------------------------------------------------------------------------
def _capture_http(value, calls):
    async def fake_http_get_json(url, *, headers, params, timeout_s):
        calls["url"] = url
        calls["headers"] = headers
        calls["params"] = params
        calls["timeout_s"] = timeout_s
        return value

    return fake_http_get_json


def test_live_request_neutral_and_digest_fields(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)
    body = {"items": [{
        "full_name": "affaan-m/ECC",
        "description": "agent coding system with skills",
        "license": {"spdx_id": "MIT"},
        "stargazers_count": 1731,
        "html_url": "https://github.com/affaan-m/ECC",
    }]}
    calls: dict = {}
    monkeypatch.setattr(research, "_http_get_json",
                        _capture_http((200, {}, body), calls))

    res = asyncio.run(research.prior_art_digest("opencode plugins", tmp_path))
    assert res["source"] == "github" and res["cache"] is False
    repo = res["repos"][0]
    assert repo == {"name": "affaan-m/ECC", "description":
                    "agent coding system with skills", "license": "MIT",
                    "stars": 1731, "url": "https://github.com/affaan-m/ECC"}
    assert calls["url"] == "https://api.github.com/search/repositories"
    assert calls["params"]["q"] == "opencode plugins"
    assert calls["headers"]["User-Agent"]
    assert "Authorization" not in calls["headers"], "lane is auth-neutral"
    assert calls["timeout_s"] == config.PRIOR_ART_TIMEOUT_S


def test_live_cache_hit_skips_second_http(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)
    calls: dict = {"n": 0}

    async def fake_http_get_json(url, *, headers, params, timeout_s):
        calls["n"] += 1
        return 200, {}, {"items": [{
            "full_name": "demo/cache-repo", "description": "d",
            "license": {"spdx_id": "Apache-2.0"}, "stargazers_count": 7,
            "html_url": "https://github.com/demo/cache-repo"}]}

    monkeypatch.setattr(research, "_http_get_json", fake_http_get_json)
    first = asyncio.run(research.prior_art_digest("react state", tmp_path))
    second = asyncio.run(research.prior_art_digest("react state", tmp_path))

    assert calls["n"] == 1, "second lookup must be served from cache"
    assert first["source"] == "github"
    assert second["source"] == "cache" and second["cache"] is True
    assert second["repos"] == first["repos"]
    # A different query is a different cache key.
    asyncio.run(research.prior_art_digest("other query", tmp_path))
    assert calls["n"] == 2

    cached = json.loads(
        (tmp_path / ".shared_workspace" / "prior_art_cache.json")
        .read_text(encoding="utf-8"))
    assert len(cached) == 2 and all("repos" in v for v in cached.values())


def test_live_rate_limited_is_clean(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)
    calls: dict = {}
    monkeypatch.setattr(research, "_http_get_json", _capture_http(
        (403, {"Retry-After": "30"}, {"message": "API rate limit"}), calls))
    res = asyncio.run(research.prior_art_digest("tray menus", tmp_path))
    assert res.get("error") == "github rate limited (retry after 30s)"
    # The failure must not be cached as a digest.
    assert res.get("repos") is None


def test_live_http_failure_is_clean(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)

    async def broken(*args, **kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(research, "_http_get_json", broken)
    res = asyncio.run(research.prior_art_digest("tray menus", tmp_path))
    assert res.get("error") == "github search unavailable"
    assert "connection reset" not in json.dumps(res)


def test_live_malformed_items_are_tolerated(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)
    monkeypatch.setattr(research, "_http_get_json", _capture_http(
        (200, {}, {"items": [None, {"full_name": "ok/one"},
                              {"full_name": "no/license", "license": None}]}),
        {}))
    res = asyncio.run(research.prior_art_digest("q", tmp_path))
    assert [r["name"] for r in res["repos"]] == ["ok/one", "no/license"]
    assert res["repos"][1]["license"] is None


# ---------------------------------------------------------------------------
# Redaction: a planted secret never survives the digest
# ---------------------------------------------------------------------------
def test_digest_scrubs_planted_secret(tmp_path, monkeypatch):
    planted = "Zz9qW3eR7tYuIoP1a" + "SdF4gH6jKlZxCvB2nM5"
    (tmp_path / ".env").write_text(f"TEST_TOKEN_B={planted}\n",
                                   encoding="utf-8")
    redaction.reset_cache()
    try:
        _gates(monkeypatch, demo=False)

        async def leaky(*args, **kwargs):
            return 200, {}, {"items": [{
                "full_name": "demo/leaky", "description": f"token {planted}",
                "license": {"spdx_id": "MIT"}, "stargazers_count": 1,
                "html_url": "https://github.com/demo/leaky"}]}

        monkeypatch.setattr(research, "_http_get_json", leaky)
        res = asyncio.run(research.prior_art_digest("leak", tmp_path))
        assert planted not in json.dumps(res)
    finally:
        redaction.reset_cache()
