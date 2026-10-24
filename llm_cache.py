"""Exact-match LLM response cache (v2.6 capacity layer).

Free inference capacity is the binding constraint of the whole system, so
every byte-identical call to the same provider+model with the same prompt
is served from disk instead of spending quota again.  Deliberately
**exact-match only**: semantic caching can return stale or wrong code for
slightly different prompts, which is unacceptable for a deterministic
build pipeline (the literature's 60-70% hit rates are not worth a wrong
build).

Key = sha256(role | provider | model | system | user | tools-json).
Entries hold the raw response plus timestamps; TTL and a max-entry prune
bound disk growth.  Every operation is failure-tolerant: a cache problem
must never break a build.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import config
import redaction
from logger import log_event


def _cache_dir(workspace_root: Optional[Path] = None) -> Path:
    root = Path(workspace_root) if workspace_root else config.WORKSPACE_ROOT
    return root / ".system" / "llm_cache"


def _digest(role: str, provider: str, model: str,
            system: str, user: str, tools: Optional[Any]) -> str:
    payload = json.dumps(
        {"role": role or "", "provider": provider or "", "model": model or "",
         "system": system or "", "user": user or "",
         "tools": tools if tools is not None else None},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _entry_path(digest: str, workspace_root: Optional[Path] = None) -> Path:
    return _cache_dir(workspace_root) / digest[:2] / f"{digest}.json"


def get(role: str, provider: str, model: str, system: str, user: str,
        tools: Optional[Any] = None,
        workspace_root: Optional[Path] = None) -> Optional[str]:
    """Return a cached response, or ``None`` on miss/expiry/any error."""
    if not config.LLM_CACHE_ENABLED:
        return None
    path = _entry_path(_digest(role, provider, model, system, user, tools),
                       workspace_root)
    try:
        raw = path.read_text(encoding="utf-8")
        entry = json.loads(raw)
        created = float(entry.get("created_ts", 0.0))
        if time.time() - created > config.LLM_CACHE_TTL_H * 3600:
            path.unlink(missing_ok=True)
            return None
        response = entry.get("response")
        if isinstance(response, str) and response.strip():
            # v2.8: INFO, not DEBUG.  At DEBUG this event was filtered out by
            # the default LOG_LEVEL, so cache behaviour was completely
            # unmeasurable -- there was no way to tell "the cache never hits"
            # from "the cache never reports".  Only 3 entries survived on disk
            # across the whole reference run, which is why this matters.
            log_event("INFO", "llm_cache_hit",
                      {"role": role, "provider": provider, "model": model})
            return response
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None
    return None


def put(role: str, provider: str, model: str, system: str, user: str,
        response: str, tools: Optional[Any] = None,
        workspace_root: Optional[Path] = None) -> None:
    """Store a successful response (best-effort; never raises)."""
    if not config.LLM_CACHE_ENABLED or not (response or "").strip():
        return
    path = _entry_path(_digest(role, provider, model, system, user, tools),
                       workspace_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "role": role, "provider": provider, "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_ts": time.time(),
            # Redacted on write: a cached response is a verbatim copy of model
            # output, which may echo a credential the model was shown.  The
            # cache *key* still hashes the unredacted prompt (see _digest), so
            # redacting here never orphans or conflates entries.
            "response": redaction.redact_known_secrets(response),
        }
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        _prune(workspace_root)
    except OSError:
        return


def _prune(workspace_root: Optional[Path] = None) -> None:
    """Drop oldest entries when the cache exceeds the configured cap."""
    try:
        files = sorted(_cache_dir(workspace_root).rglob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[config.LLM_CACHE_MAX_ENTRIES:]:
            stale.unlink(missing_ok=True)
    except OSError:
        return


def stats(workspace_root: Optional[Path] = None) -> Dict[str, Any]:
    """Entry count + total size, for ``--models status``."""
    directory = _cache_dir(workspace_root)
    count = 0
    total = 0
    try:
        for path in directory.rglob("*.json"):
            count += 1
            total += path.stat().st_size
    except OSError:
        pass
    return {"entries": count, "bytes": total,
            "enabled": bool(config.LLM_CACHE_ENABLED),
            "ttl_h": config.LLM_CACHE_TTL_H}


def clear(workspace_root: Optional[Path] = None) -> int:
    """Delete every cached entry; returns the number removed."""
    removed = 0
    try:
        for path in _cache_dir(workspace_root).rglob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
    except OSError:
        pass
    return removed
