"""v2.9: the Council is gone, and the circuit breaker replaced it.

Two jobs here, both regressions rather than features:

* **Nothing may import the Council.**  The v2.9 removal deleted ``council.py``
  and rewired five modules.  The failure mode for that kind of change is not a
  crash -- it is a *silent* reintroduction, or a lazily-imported reference that
  only bites on the code path nobody exercises.  These are structural checks so
  a reintroduced import fails in CI rather than in a three-hour unattended run.

* **The breaker must actually fire, and must not fire spuriously.**  The
  Council's one genuinely useful behaviour was stopping a doomed run.  Replacing
  a model-driven pause with arithmetic is only safe if the arithmetic is
  tested at both ends: it has to trip on a project burning quota with zero
  completions, and it has to stay silent on a project that is merely slow.
"""

from __future__ import annotations

from pathlib import Path


import config
import steering
from super_orchestrator import SuperOrchestrator


# ---------------------------------------------------------------- structure
def test_council_module_is_gone():
    assert not (Path(__file__).resolve().parents[1] / "council.py").exists()


def test_no_module_imports_council():
    """A reintroduced ``import council`` would explode at runtime, not import."""
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted(root.glob("*.py")) + sorted(root.glob("domains/*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import council", "from council")):
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"council is imported again: {offenders}"


def test_council_config_keys_are_removed():
    for name in dir(config):
        assert not name.startswith("COUNCIL_"), f"stale config knob {name}"


def test_council_roles_are_removed():
    assert "council_strategist" not in config.DEFAULT_MODEL_ROUTING
    assert "council_arbiter" not in config.DEFAULT_MODEL_ROUTING
    assert "council_strategist" not in config.MODEL_FALLBACKS
    assert "council_arbiter" not in config.MODEL_FALLBACKS
    # The role that replaced one of them.
    assert "spec_critic" in config.DEFAULT_MODEL_ROUTING


def test_steering_decision_model_is_gone():
    """``SteeringDecision`` existed only for the Council's decision envelope."""
    import validation
    assert not hasattr(validation, "SteeringDecision")
    assert not hasattr(validation, "COUNCIL_ACTIONS")


def test_board_write_allowlist_survived_the_removal():
    """Phase D's auto-remedies depend on this boundary, so it must not rot."""
    import validation
    for remedy in ("requeue_task", "narrow_objective", "split_task",
                   "defer_task"):
        assert validation.is_executable_remedy(remedy), remedy
        assert validation.board_keys_for(remedy), remedy
    # Anything unknown is inert -- the whole point of the whitelist.
    assert not validation.is_executable_remedy("approve_task")
    assert not validation.is_executable_remedy("")
    assert not validation.is_executable_remedy("rm -rf /")


# ------------------------------------------------------------------- breaker
def _make_orchestrator(tmp_path: Path) -> SuperOrchestrator:
    """A SuperOrchestrator with one synthetic project in its status table."""
    orch = SuperOrchestrator(tmp_path)
    name = "demo_project"
    (tmp_path / "projects" / name / ".agent").mkdir(parents=True)
    orch.projects[name] = {
        "status": "running",
        "board": {"done": 0, "failed": 0, "blocked": 0},
        "tokens": {"window": 0},
    }
    return orch


def test_breaker_trips_on_zero_completions_and_high_spend(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "BREAKER_TOKENS_PER_WINDOW", 250_000)
    orch = _make_orchestrator(tmp_path)
    orch.projects["demo_project"]["tokens"]["window"] = 400_000

    orch._check_breaker()

    flag = tmp_path / "projects/demo_project/.agent/pause.flag"
    assert flag.exists(), "breaker did not pause a project that spent 400k tokens on 0 tasks"
    verdict = orch.projects["demo_project"]["breaker"]
    assert verdict and verdict["tokens_in_window"] == 400_000
    assert verdict["done"] == 0
    # The pause has to be explicable to a human reading it cold.
    assert "zero completions" in flag.read_text(encoding="utf-8")


def test_breaker_stays_silent_when_work_is_landing(tmp_path, monkeypatch):
    """A slow project is not a looping project."""
    monkeypatch.setattr(config, "BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "BREAKER_TOKENS_PER_WINDOW", 250_000)
    orch = _make_orchestrator(tmp_path)
    state = orch.projects["demo_project"]
    state["tokens"]["window"] = 400_000
    state["board"]["done"] = 3

    orch._check_breaker()

    assert not (tmp_path / "projects/demo_project/.agent/pause.flag").exists()
    assert state["breaker"] is None


def test_breaker_stays_silent_under_the_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "BREAKER_TOKENS_PER_WINDOW", 250_000)
    orch = _make_orchestrator(tmp_path)
    orch.projects["demo_project"]["tokens"]["window"] = 10_000

    orch._check_breaker()

    assert not (tmp_path / "projects/demo_project/.agent/pause.flag").exists()


def test_breaker_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BREAKER_ENABLED", False)
    monkeypatch.setattr(config, "BREAKER_TOKENS_PER_WINDOW", 1)
    orch = _make_orchestrator(tmp_path)
    orch.projects["demo_project"]["tokens"]["window"] = 10_000_000

    orch._check_breaker()

    assert not (tmp_path / "projects/demo_project/.agent/pause.flag").exists()


def test_breaker_zero_threshold_disables(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "BREAKER_TOKENS_PER_WINDOW", 0)
    orch = _make_orchestrator(tmp_path)
    orch.projects["demo_project"]["tokens"]["window"] = 10_000_000

    orch._check_breaker()

    assert not (tmp_path / "projects/demo_project/.agent/pause.flag").exists()


def test_breaker_names_the_signals_that_fired(tmp_path, monkeypatch):
    """The point of attaching evidence is that it is actionable cold."""
    monkeypatch.setattr(config, "BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "BREAKER_TOKENS_PER_WINDOW", 250_000)
    orch = _make_orchestrator(tmp_path)
    state = orch.projects["demo_project"]
    state["tokens"]["window"] = 999_999
    state["board"] = {"done": 0, "failed": 5, "blocked": 2}
    # Make the steering monitor reachable and cheap.
    monkeypatch.setattr(steering, "monitor_for",
                        lambda _root: _StubMonitor())

    orch._check_breaker()

    verdict = state["breaker"]
    assert verdict is not None
    assert verdict["failed"] == 5 and verdict["blocked"] == 2
    assert verdict["signals"] == ["doom_loop"]


class _StubMonitor:
    """Stands in for a steering monitor so the breaker test needs no log."""

    def evaluate(self, project):
        return [steering.Signal(kind="doom_loop", project=project,
                                detail="synthetic", key="d", weight=1)]


def test_breaker_trip_is_logged(tmp_path, monkeypatch):
    """A pause nobody can see in the audit trail is a silent intervention."""
    monkeypatch.setattr(config, "BREAKER_ENABLED", True)
    monkeypatch.setattr(config, "BREAKER_TOKENS_PER_WINDOW", 250_000)
    orch = _make_orchestrator(tmp_path)
    orch.projects["demo_project"]["tokens"]["window"] = 400_000

    import super_orchestrator
    seen = []
    monkeypatch.setattr(super_orchestrator, "log_event",
                        lambda lvl, msg, data=None: seen.append((msg, data)))

    orch._check_breaker()

    assert any(msg == "breaker_tripped" for msg, _ in seen), seen
