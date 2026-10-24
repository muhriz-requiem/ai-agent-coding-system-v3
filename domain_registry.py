"""Domain pack registry (v2.6).

The core coding company stays domain-agnostic: a **domain pack** is an
optional vertical that plugs into two generic hooks:

1. ``propose_tasks(project_root, board)`` -- called when a project loop has
   no eligible work (starvation time), asking the pack for new tasks
   (the Kaggle pack's campaign director uses this to queue experiments).
2. ``on_task_completed(orchestrator, task, objective_result, files)`` --
   called after an APPROVED task, letting the pack react to results
   (e.g. validate + CV-gate a submission).

A project opts in with ``.agent/domain.json``::

    {"pack": "kaggle", "competition": "titanic", "metric": "accuracy"}

No core module needs to know what a "kaggle" is -- a repo-wide grep for
pack-specific terms matches only ``domains/``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from logger import log_event


class DomainPack:
    """Duck-typed base: packs only need a ``name`` and the hook methods."""

    name = "base"

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        """Receive the parsed ``.agent/domain.json`` payload."""

    async def propose_tasks(self, project_root: Path,
                            board: Dict[str, Any]) -> list:
        return []

    async def tasks_accepted(self, project_root: Path,
                             tasks: list) -> None:
        """Called after proposed tasks were appended to the board.

        Lets a pack commit bookkeeping (e.g. drop queued hypotheses) only
        once the core has actually accepted the tasks.
        """
        return None

    async def on_task_completed(self, orchestrator: Any, task: Any,
                                objective_result: Any,
                                applied_files: Any = None) -> None:
        return None


_PACKS: Dict[str, DomainPack] = {}


def register_pack(pack: DomainPack) -> None:
    _PACKS[pack.name] = pack


def get_pack(name: str) -> Optional[DomainPack]:
    _ensure_builtins()
    return _PACKS.get(name)


def list_packs() -> list:
    _ensure_builtins()
    return sorted(_PACKS)


def _ensure_builtins() -> None:
    if _PACKS:
        return
    try:
        from domains import kaggle_pack
        register_pack(kaggle_pack.PACK)
    except Exception as exc:  # noqa: BLE001 -- a broken pack must not
        log_event("WARNING", "domain_pack_load_failed", {"error": str(exc)})


def project_domain_config(project_root: Path) -> Dict[str, Any]:
    """Parse ``.agent/domain.json`` (``{}`` when absent/invalid)."""
    try:
        payload = json.loads(
            (Path(project_root) / ".agent" / "domain.json")
            .read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def active_pack(project_root: Path) -> Optional[DomainPack]:
    """The pack this project opted into, configured, or ``None``.

    Returns a per-project shallow copy so concurrently supervised
    projects cannot bleed configuration into each other through a shared
    singleton.
    """
    cfg = project_domain_config(project_root)
    name = str(cfg.get("pack") or "").strip()
    if not name:
        return None
    registered = get_pack(name)
    if registered is None:
        log_event("WARNING", "domain_pack_unknown",
                  {"project": str(project_root), "pack": name})
        return None
    try:
        pack = copy.copy(registered)
        pack.configure(cfg)
    except Exception as exc:  # noqa: BLE001
        log_event("WARNING", "domain_pack_configure_failed",
                  {"pack": name, "error": str(exc)})
        return None
    return pack
