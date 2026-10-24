"""Structured JSONL logging for every component of the system.

Every significant event (task start / complete / fail / escalate / restart /
pause / LLM fallback / integration check ...) is appended as a single JSON
line to ``logs/agent_log.jsonl``::

    {"ts": "...", "level": "INFO", "message": "...", "data": {...}}

``log_event`` is thread-safe (a module-level lock serializes appends) and
**never raises**: a logging failure must never take down the orchestration.

Event payloads are passed through ``redaction.redact_obj`` before being
written, so a credential that reaches a log call site -- a provider exception
body, a pytest output tail, a model response head -- is blanked rather than
persisted.  Set ``REDACTION_ENABLED=false`` to disable.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import config

#: Module-level lock: appends from multiple asyncio tasks / threads are
#: serialized so each JSON line stays intact.
_APPEND_LOCK = threading.Lock()

_LEVEL_ORDER = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

#: ANSI colors per level (used only when the console supports them).
_LEVEL_COLOR = {
    "DEBUG": "90",      # bright black / gray
    "INFO": "36",       # cyan
    "WARNING": "33",    # yellow
    "ERROR": "31",      # red
    "CRITICAL": "1;41", # bold on red background
}

_COLOR_ONCE = {"done": False, "supported": False}


def _enable_console_colors() -> bool:
    """Detect (and on Windows enable) ANSI support in the console.

    One-time check, cached.  On win32 this flips the console into virtual
    terminal processing mode via ctypes; on POSIX a simple isatty check.
    Returns False on any problem -- plain text then, no crashes.
    """
    if _COLOR_ONCE["done"]:
        return _COLOR_ONCE["supported"]
    _COLOR_ONCE["done"] = True
    supported = False
    try:
        stream = sys.stderr
        if not stream.isatty():
            supported = False
        elif os.name == "posix":
            supported = True
        else:  # windows: enable ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004)
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                if kernel32.SetConsoleMode(
                        handle, mode.value | 0x0004):
                    supported = True
    except Exception:
        supported = False
    _COLOR_ONCE["supported"] = supported
    return supported


def _echo_line(level: str, message: str, data: Dict[str, Any]) -> None:
    """Write one compact console line to stderr (best effort, never raises)."""
    try:
        stamp = datetime.now().strftime("%H:%M:%S")
        extras = ""
        if data:
            extras = " :: " + " ".join(
                f"{key}={_compact_value(value)}"
                for key, value in list(data.items())[:6])
        line = f"[{stamp} {level}] {message}{extras}"
        if _enable_console_colors():
            color = _LEVEL_COLOR.get(level, "0")
            print(f"\x1b[{color}m{line}\x1b[0m", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr)
    except Exception:
        pass


def _compact_value(value: Any) -> str:
    """Short scalar-ish rendering for console echo (no newlines, <= 60 chars)."""
    text = str(value)
    text = " ".join(text.split())
    return text[:60] + ("…" if len(text) > 60 else "")


def _min_level() -> int:
    """Minimum severity that should be written, from ``config.LOG_LEVEL``."""
    return _LEVEL_ORDER.get(config.LOG_LEVEL, 20)


def _safe_data(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Make *data* JSON-serializable and redact live secret values.

    Serializability is the original contract; redaction is layered on top
    because this is the single funnel every ``log_event`` call site passes
    through, and several of those sites carry provider exception text, pytest
    output tails and model response heads -- any of which can contain a
    credential.  Non-string leaves pass through untouched so the log stays
    JSON-shaped.
    """
    if not data:
        return {}
    try:
        json.dumps(data, ensure_ascii=False, default=str)
        return _redact_structure(data)
    except (TypeError, ValueError):
        return _redact_structure({str(k): str(v) for k, v in dict(data).items()})


def _redact_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact secret values in a mapping, never raising (logging must not fail)."""
    try:
        import redaction
        redacted = redaction.redact_obj(data)
        return redacted if isinstance(redacted, dict) else data
    except Exception:  # noqa: BLE001 -- logging must never raise
        return data


#: Optional log-directory override.  ``tests/conftest.py`` points this at a
#: separate file so a test run cannot interleave synthetic events into the
#: production ``logs/agent_log.jsonl``.  That pollution is not cosmetic: it
#: makes an audit of the run log -- counting real failures, measuring pause
#: spin, spotting a resurrected event -- quietly wrong, because ~3000 test
#: rows are indistinguishable from real ones once written.
_LOG_DIR_OVERRIDE: Optional[Path] = None


def set_log_dir(path: Optional[Path]) -> None:
    """Redirect subsequent writes to *path* (``None`` restores the default)."""
    global _LOG_DIR_OVERRIDE
    _LOG_DIR_OVERRIDE = Path(path) if path is not None else None


def log_event(level: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Append one structured JSON line to ``logs/agent_log.jsonl``.

    Additionally echoes a compact one-line summary to the console when
    ``config.CONSOLE_ECHO`` is enabled (default).  Never raises and never
    interferes with orchestration.
    """
    try:
        level = str(level).upper()
        if _LEVEL_ORDER.get(level, 20) < _min_level():
            return  # filtered out by configured LOG_LEVEL
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            # v2.8: the message is redacted too, not just ``data``.  Every
            # current call site passes a literal string here (verified: zero
            # interpolate into the message), so this is defence in depth -- but
            # ``data`` was covered and ``message`` was not, which is exactly
            # the kind of asymmetry a future f-string turns into a leak.
            "message": _redact_text(str(message)),
            "data": _safe_data(data),
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _APPEND_LOCK:
            log_dir = _LOG_DIR_OVERRIDE or (config.WORKSPACE_ROOT / "logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "agent_log.jsonl", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if config.CONSOLE_ECHO:
            _echo_line(level, entry["message"], entry["data"])
    except Exception:
        # Logging must never raise -- swallow anything (disk full, permissions,
        # workspace not bootstrapped yet, ...).
        pass


def _redact_text(text: str) -> str:
    """Blank live secret values in a log message. Never raises."""
    if not text:
        return text
    try:
        import redaction

        out = redaction.redact_known_secrets(text)
        return out if isinstance(out, str) else text
    except Exception:  # noqa: BLE001 -- logging must never fail
        return text


# Convenience wrappers so call sites stay terse and readable.
def log_debug(message: str, data: Optional[Dict[str, Any]] = None) -> None:
    log_event("DEBUG", message, data)


def log_info(message: str, data: Optional[Dict[str, Any]] = None) -> None:
    log_event("INFO", message, data)


def log_warning(message: str, data: Optional[Dict[str, Any]] = None) -> None:
    log_event("WARNING", message, data)


def log_error(message: str, data: Optional[Dict[str, Any]] = None) -> None:
    log_event("ERROR", message, data)


def log_critical(message: str, data: Optional[Dict[str, Any]] = None) -> None:
    log_event("CRITICAL", message, data)
