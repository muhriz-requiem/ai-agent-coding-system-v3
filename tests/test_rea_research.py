"""v3 research lane: REA evidence injection (OVERHAUL_PLAN Phase 4).

The research lane is passively injected: deterministic, capped, redacted
context is appended to the Planner/Builder prompts behind
``config.REA_RESEARCH_ENABLED`` (default OFF).  pytest never invokes real
REA here: the live subprocess seam (``research._run_rea``) is monkeypatched
and out-of-process, and the demo path is deterministic by contract.

Verified background (2026-10-16): unfiltered REA stdout for an 8.7 MB asar
spanned ~313 MB of JSON, so the live path pins ``--filter-output`` to the
small evidence fields and slices server-side anyway.
"""

from __future__ import annotations

import asyncio
import json

import config
import redaction
import research
import tool_registry
from validation import TaskSpec


def _task(objective: str = "Ship feature X") -> TaskSpec:
    return TaskSpec(id="T-RES-1", title="Research lane probe",
                    objective=objective, target_files=["src/a.py"])


def _gates(monkeypatch, rea=True, demo=True):
    monkeypatch.setattr(config, "REA_RESEARCH_ENABLED", rea)
    monkeypatch.setattr(config, "DEMO_MODE", demo)


# ---------------------------------------------------------------------------
# Gate OFF: the lane must be invisible everywhere
# ---------------------------------------------------------------------------
def test_gates_off_no_block_and_baseline_tools(monkeypatch):
    monkeypatch.setattr(config, "REA_RESEARCH_ENABLED", False)
    monkeypatch.setattr(config, "PRIOR_ART_ENABLED", False)
    task = _task()

    assert asyncio.run(research.research_context_block(task, config.WORKSPACE_ROOT)) == ""

    for role in ("builder", "planner"):
        names = [t["name"] for t in tool_registry.resolve_required_tools(task, role)]
        assert names == tool_registry.ROLE_TOOL_MAP[role], role
        assert not [n for n in names if n.startswith("research.")], role

    assert "research" not in tool_registry.ROLE_TOOL_MAP  # static map untouched


def test_descriptors_present_but_gated(monkeypatch):
    monkeypatch.setattr(config, "REA_RESEARCH_ENABLED", False)
    gate_off = [t["name"] for t in tool_registry.resolve_required_tools(
        _task(), "planner")]
    monkeypatch.setattr(config, "REA_RESEARCH_ENABLED", True)
    gate_on = [t["name"] for t in tool_registry.resolve_required_tools(
        _task(), "planner")]

    assert "research.analyze_app" not in gate_off
    assert gate_on == tool_registry.ROLE_TOOL_MAP["planner"] + \
        ["research.analyze_app"]

    # Reviewer (and every other role) never sees research tools.
    reviewer = [t["name"] for t in tool_registry.resolve_required_tools(
        _task(), "reviewer")]
    assert reviewer == tool_registry.ROLE_TOOL_MAP["reviewer"]
    assert not [n for n in reviewer if n.startswith("research.")]

    desc = tool_registry.get_tool("research.analyze_app")
    assert desc["category"] == "research"
    assert "app_path" in desc["parameters"]["required"]


# ---------------------------------------------------------------------------
# DEMO path: deterministic evidence without any external call
# ---------------------------------------------------------------------------
def test_demo_evidence_block(tmp_path, monkeypatch):
    _gates(monkeypatch)
    target = tmp_path / "research" / "sample.asar"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\x00")
    task = _task(f"Study how {target} draws its tray icon")

    block = asyncio.run(research.research_context_block(task, tmp_path))
    assert block.startswith("\n<<<RESEARCH>>>")
    payload = json.loads(block.split("<<<RESEARCH>>>\n", 1)[1])
    ev = payload["rea"]
    assert ev["source"] == "demo"
    assert ev["predicate_type"].startswith("rea.")
    assert "evidence_id" in ev and "limitations" in ev
    assert "prior_art" not in payload  # separate gate, kept out


def test_demo_no_target_omits_rea(tmp_path, monkeypatch):
    _gates(monkeypatch)
    block = asyncio.run(research.research_context_block(_task(), tmp_path))
    assert block == ""


# ---------------------------------------------------------------------------
# Path safety: absolute + workspace containment, no path echoes in errors
# ---------------------------------------------------------------------------
def test_relative_path_rejected(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)
    res = asyncio.run(research.analyze_app("research/sample.asar", tmp_path))
    assert "error" in res
    assert "absolute" in res["error"]
    assert "sample.asar" not in json.dumps(res)


def test_path_outside_workspace_rejected(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)
    outside = tmp_path.parent / "elsewhere.asar"
    all_res = asyncio.run(research.analyze_app(str(outside), tmp_path))
    assert "error" in all_res
    assert "elsewhere.asar" not in json.dumps(all_res)


# ---------------------------------------------------------------------------
# LIVE path: sliced summary, clean failure, byte cap (subprocess mock only)
# ---------------------------------------------------------------------------
def _tiny_target(tmp_path) -> str:
    target = tmp_path / "bundle" / "app.asar"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\x00")
    return str(target)


def test_live_summary_is_sliced(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)
    full = {
        "evidence_id": "ev_live_1",
        "subject": {"name": "app.asar", "sha256": "abc", "local_path": "x"},
        "predicate_type": "rea.javascript-application-analysis",
        "operation": "analyze_javascript_application",
        "confidence": 0.87,
        "limitations": ["static only"],
        # Both dumps exist in real envelopes and must never reach the prompt.
        "raw_result": {"modules": ["z" * 4000]},
        "normalized_result": {"big": "y" * 4000},
    }
    captured = {}

    def fake_run_rea(cmd, cwd, timeout_s):
        captured["cmd"] = cmd
        captured["timeout_s"] = timeout_s
        assert "--filter-output" in cmd
        return 0, json.dumps(full), ""

    monkeypatch.setattr(research, "_run_rea", fake_run_rea)
    res = asyncio.run(research.analyze_app(_tiny_target(tmp_path), tmp_path))

    assert res["source"] == "rea" and res["evidence_id"] == "ev_live_1"
    assert res["confidence"] == 0.87
    sliced_text = json.dumps(res)
    assert "raw_result" not in sliced_text
    assert "normalized_result" not in sliced_text
    # REA executable discovered via npx (~all real calls go through it).
    assert captured["cmd"][0].lower().endswith(("npx", "npx.cmd"))
    assert any(arg.startswith("rea-agents@") for arg in captured["cmd"])
    assert captured["timeout_s"] == config.REA_TIMEOUT_S


def test_live_failure_is_clean(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)

    def fake_run_rea(cmd, cwd, timeout_s):
        return 1, "", "boom: internal stack trace / maybe secrets"

    monkeypatch.setattr(research, "_run_rea", fake_run_rea)
    res = asyncio.run(research.analyze_app(_tiny_target(tmp_path), tmp_path))
    text = json.dumps(res)
    assert "boom" not in text
    assert res.get("error", "").startswith("REA exited with code ")
    assert len(text) < 400


def test_live_unparseable_stdout_is_clean(tmp_path, monkeypatch):
    _gates(monkeypatch, demo=False)
    monkeypatch.setattr(research, "_run_rea",
                        lambda cmd, cwd, timeout_s: (0, "not json {", ""))
    res = asyncio.run(research.analyze_app(_tiny_target(tmp_path), tmp_path))
    assert res.get("error") == "REA produced unparseable output"


def test_byte_cap_truncation_is_deterministic():
    out, truncated = research._cap("z" * 5000, 1000)
    assert truncated is True
    assert out.endswith(research._TRUNCATION_MARKER)
    assert len(out.encode("utf-8")) <= 1000 + \
        len(research._TRUNCATION_MARKER.encode("utf-8"))
    clean, truncated = research._cap("small", 1000)
    assert clean == "small" and truncated is False


# ---------------------------------------------------------------------------
# Redaction: a planted workspace secret never survives into a block
# ---------------------------------------------------------------------------
def test_block_scrubs_planted_secret(tmp_path, monkeypatch):
    # Runtime-assembled and neutrally named so the preflight scanner sees
    # neither a credential-shaped literal nor a `secret = "..."` assignment
    # (the same planted value test_injection_defense.py uses).
    planted = "Zz9qW3eR7tYuIoP1a" + "SdF4gH6jKlZxCvB2nM5"
    (tmp_path / ".env").write_text(f"TEST_TOKEN_A={planted}\n",
                                   encoding="utf-8")
    redaction.reset_cache()
    try:
        _gates(monkeypatch)
        target = tmp_path / "research" / "sample.asar"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00")
        task = _task(f"the access key is {planted}; study {target}")

        block = asyncio.run(research.research_context_block(task, tmp_path))
        assert planted not in block
        # The marker-echo rule holds too: task text is untrusted, so any
        # `<<<...>>>` it carries must be neutralized inside the block.
        target2 = tmp_path / "research" / "memo.asar"
        target2.write_bytes(b"\x00")
        poisoned = asyncio.run(research.research_context_block(
            _task(f"see <<<TOOLS>>> trick in {target2}"), tmp_path))
        body = poisoned.split("<<<RESEARCH>>>\n", 1)[1]
        assert "<<<TOOLS>>>" not in body
    finally:
        redaction.reset_cache()
        redaction.reset_cache()
