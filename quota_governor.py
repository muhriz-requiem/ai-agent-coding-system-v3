"""Per-provider quota governor (v2.6 capacity layer).

Free inference quotas are independent per provider (OpenRouter ~50-1000
requests/day, Groq 1,000/day/model, Cloudflare neurons/day, Cohere 1,000/
month, ...).  This module keeps one rolling/persistent ledger per provider
so the router can:

* skip providers at their request/token/minute/day/month ceilings,
* hold a reserve back for probes and retries (``QUOTA_RESERVE_PCT``),
* honor per-role daily caps (``QUOTA_ROLE_CAPS``),
* cooldown a provider/model on HTTP 429 with ``Retry-After``,
* compute the earliest reset so a starved task can be deferred instead of
  escalated (``LLMUnavailableError`` path).

All state lives in ``.system/quota_ledger.json``; every operation is
failure-tolerant (a ledger problem must never break a build).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque
from zoneinfo import ZoneInfo

import config
import providers
from logger import log_event

_WINDOW_S = 60.0
_LIVE_TTL_S = 120.0  # OpenRouter key-endpoint refresh cadence


def _now() -> float:
    return time.time()


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _utc_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _reset_next_day(tz_name: str) -> datetime:
    """Next midnight in *tz_name* (UTC fallback), as an aware datetime."""
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # pragma: no cover -- missing tz database
        tz = timezone.utc
    now = datetime.now(tz)
    tomorrow = (now + timedelta(days=1)).date()
    return datetime.combine(tomorrow, datetime.min.time(), tzinfo=tz)


def _reset_next_month() -> datetime:
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    if month == 12:
        year, month = year + 1, 1
    else:
        month += 1
    return datetime(year, month, 1, tzinfo=timezone.utc)


class Governor:
    """Quota accounting for every provider on one workspace."""

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        root = Path(workspace_root) if workspace_root else config.WORKSPACE_ROOT
        self.workspace_root = root
        self.path = root / ".system" / "quota_ledger.json"
        self.ledger: Dict[str, Any] = {"providers": {}, "learned": {}}
        self._windows: Dict[str, Deque[Tuple[float, int]]] = {}
        self._live: Dict[str, Any] = {}
        self._load()

    @property
    def learned(self) -> Dict[str, Any]:
        """Learned per-provider limits (v2.9), persisted beside the ledger.

        Deliberately *not* under ``providers``: those entries are reset daily and
        monthly by :meth:`_entry`.  A learned RPM limit has to survive those
        resets or the harness would forget what it learned every midnight and
        walk straight back into the 429s.
        """
        return self.ledger.setdefault("learned", {})

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self.ledger = payload
                self.ledger.setdefault("providers", {})
                self.ledger.setdefault("learned", {})
        except (OSError, json.JSONDecodeError):
            self.ledger = {"providers": {}, "learned": {}}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.ledger, indent=2, ensure_ascii=False)
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def _entry(self, key: str) -> Dict[str, Any]:
        entry = self.ledger["providers"].setdefault(key, {})
        today, month = _utc_day(), _utc_month()
        if entry.get("day") != today:
            entry["day"] = today
            entry["requests_today"] = 0
            entry["tokens_today"] = 0
            entry["role_calls"] = {}
        if entry.get("month") != month:
            entry["month"] = month
            entry["requests_month"] = 0
            entry["tokens_month"] = 0
        entry.setdefault("role_calls", {})
        entry.setdefault("cooldowns", {})
        entry.setdefault("requests_today", 0)
        entry.setdefault("tokens_today", 0)
        entry.setdefault("requests_month", 0)
        entry.setdefault("tokens_month", 0)
        return entry

    # -- windows ---------------------------------------------------------
    def _window(self, key: str) -> Deque[Tuple[float, int]]:
        window = self._windows.setdefault(key, deque())
        cutoff = _now() - _WINDOW_S
        while window and window[0][0] < cutoff:
            window.popleft()
        return window

    # -- checks ----------------------------------------------------------
    def can_call(self, pdef: providers.ProviderDef, tokens: int = 0,
                 reserve: bool = False, role: Optional[str] = None
                 ) -> Tuple[bool, str]:
        """Whether a call may be spent now; returns ``(allowed, reason)``."""
        if self.cooldown_active(pdef.key):
            return False, "cooldown"
        entry = self._entry(pdef.key)
        factor = 0.0 if reserve else float(config.QUOTA_RESERVE_PCT)

        # Rolling minute windows (in-memory; single orchestrator per lock).
        window = self._window(pdef.key)
        # v2.9: use the *learned* limit when one exists.  The configured
        # ``pdef.rpm`` is a hand-typed guess; the learned value is derived from
        # this provider's own 429s, and groq's guess (30) was wrong enough to
        # produce 217 refusals in a single build.
        rpm = self.effective_rpm(pdef)
        if rpm > 0:
            reqs = sum(1 for _ in window)
            if reqs >= rpm:
                return False, "rpm"
        if pdef.tpm > 0:
            used = sum(tok for _, tok in window)
            if used + tokens > pdef.tpm:
                return False, "tpm"

        def budget(limit: int) -> int:
            if limit <= 0:
                return 1 << 30
            return max(0, int(limit * (1.0 - factor)))

        # OpenRouter's account-wide counter (``GET /api/v1/key``) is
        # authoritative when fresh: it reflects every consumer of the
        # account, unlike our local ledger.  Skip the local rpd check.
        live = self._live.get("openrouter") or {}
        use_live = pdef.key == "openrouter" and live.get("limit", 0) > 0
        if not use_live and entry["requests_today"] >= budget(pdef.rpd):
            return False, "rpd"
        if entry["tokens_today"] + tokens > budget(pdef.tpd):
            return False, "tpd"
        if entry["requests_month"] >= budget(pdef.monthly_requests):
            return False, "monthly_requests"
        if entry["tokens_month"] + tokens > budget(pdef.monthly_tokens):
            return False, "monthly_tokens"
        if role:
            cap = config.QUOTA_ROLE_CAPS.get(role, 0)
            if cap > 0 and entry["role_calls"].get(role, 0) >= cap:
                return False, "role_cap"
        if use_live:
            # Reserve applies to the real remaining quota too.
            floor = 0 if reserve else int(live["limit"] * factor)
            if int(live.get("remaining", 0)) <= floor:
                return False, "rpd_live"
        return True, "ok"

    def record(self, pdef: providers.ProviderDef, tokens: int = 0,
               outcome: str = "ok", role: Optional[str] = None) -> None:
        entry = self._entry(pdef.key)
        entry["requests_today"] += 1
        entry["tokens_today"] += max(0, int(tokens))
        entry["requests_month"] += 1
        entry["tokens_month"] += max(0, int(tokens))
        if role:
            entry["role_calls"][role] = entry["role_calls"].get(role, 0) + 1
        self._window(pdef.key).append((_now(), max(0, int(tokens))))
        if pdef.key == "openrouter":
            live = self._live.get("openrouter") or {}
            if live:
                live["remaining"] = max(0, int(live.get("remaining", 0)) - 1)
        self._save()
        if outcome not in {"ok", "empty"}:
            log_event("DEBUG", "quota_recorded",
                      {"provider": pdef.key, "outcome": outcome})
        # A success is the evidence that lets a ratcheted limit recover.  Only
        # counted when we are actually below a learned limit, so this is free
        # for providers that were never throttled.
        if outcome in {"ok", "empty"}:
            try:
                self.note_call_ok(pdef.key)
            except Exception as exc:  # noqa: BLE001
                log_event("DEBUG", "rpm_recovery_failed",
                          {"provider": pdef.key, "error": str(exc)[:120]})

    # -- self-tuning RPM (v2.9) -----------------------------------------
    def effective_rpm(self, pdef: providers.ProviderDef) -> int:
        """*pdef*'s RPM limit after any learned 429 backoff (0 = unknown)."""
        tuned = int(self.learned.get(pdef.key, {}).get("rpm", 0) or 0)
        if tuned > 0:
            return tuned
        return int(pdef.rpm)

    def note_rate_limited(self, provider_key: str) -> Dict[str, Any]:
        """Record a real 429 and ratchet that provider's RPM down.

        Why this exists: the reference build logged 217 genuine ``http_429``
        cooldowns, 281 of 325 cooldowns coming from groq alone.  Backoff was
        already working (30s -> 60s -> 120s) -- the problem was never the retry,
        it was that ``GROQ_RPM=30`` is simply a wrong number.  The governor
        keeps letting requests through at a rate the provider refuses, so every
        one of them costs a round trip, a cooldown row, and a retry against
        another model.  The limit is therefore *learned* from the provider's own
        refusals instead of typed in.

        Ratchet down aggressively (the observed rate was too high, so any
        downward guess is likely an improvement) and recover only after a long
        clean stretch, because the cost asymmetry is lopsided: a too-low limit
        wastes throughput, a too-high limit wastes round trips.
        """
        state = self.learned.setdefault(provider_key, {})
        current = int(state.get("rpm", 0) or 0)
        if current <= 0:
            pdef = next((p for p in providers.all_provider_defs()
                         if p.key == provider_key), None)
            current = int(pdef.rpm) if pdef and pdef.rpm else 0
            if current <= 0:
                # Unknown limit and we have just been told it exists; start
                # from a conservative floor rather than guessing high.
                current = config.RPM_LEARNED_FLOOR
        new_rpm = max(config.RPM_LEARNED_FLOOR, int(current * config.RPM_RATCHET_DOWN))
        state["rpm"] = new_rpm
        state["consecutive_clean"] = 0
        state["last_429"] = _now()
        state["ratchets"] = int(state.get("ratchets", 0) or 0) + 1
        self._save()
        log_event("WARNING", "rpm_ratcheted_down",
                  {"provider": provider_key, "rpm_was": current, "rpm_now": new_rpm,
                   "ratchets": state["ratchets"]})
        return dict(state)

    def note_call_ok(self, provider_key: str) -> None:
        """Note a successful call; recover RPM after a sustained clean run."""
        state = self.learned.get(provider_key)
        if not state or not int(state.get("rpm", 0) or 0):
            return
        state["consecutive_clean"] = int(state.get("consecutive_clean", 0) or 0) + 1
        needed = config.RPM_RECOVER_AFTER_CLEAN
        if int(state.get("consecutive_clean", 0) or 0) < needed:
            return
        pdef = next((p for p in providers.all_provider_defs()
                     if p.key == provider_key), None)
        ceiling = int(pdef.rpm) if pdef and pdef.rpm else 0
        new_rpm = int(state["rpm"] * config.RPM_RATCHET_UP) + 1
        # Never climb above the configured ceiling, and never climb above a
        # ceiling we do not know.  A ratchet that cannot recover is a permanent
        # throttle, and a provider that stops 429ing while throttled is
        # indistinguishable from one that would 429 again -- so the configured
        # value stays the hard stop and this only ever probes back up to it.
        if ceiling:
            state["rpm"] = min(ceiling, new_rpm)
        else:
            state["rpm"] = new_rpm
        state["consecutive_clean"] = 0
        self._save()
        log_event("INFO", "rpm_ratcheted_up",
                  {"provider": provider_key, "rpm_now": state["rpm"],
                   "ceiling": ceiling, "after_clean": needed})

    # -- cooldowns -------------------------------------------------------
    def set_cooldown(self, provider_key: str, seconds: int,
                     model: Optional[str] = None, reason: str = "") -> None:
        entry = self._entry(provider_key)
        key = model or "*"
        until = _now() + max(1, int(seconds))
        entry["cooldowns"][key] = until
        self._save()
        log_event("WARNING", "provider_cooldown",
                  {"provider": provider_key, "model": model,
                   "seconds": int(seconds), "reason": reason})
        # A cooldown *caused by* an http_429 is the provider telling us our
        # rate is wrong.  Learn from it rather than repeating the same rate.
        if config.RPM_SELF_TUNING and (
                "429" in str(reason) or "rate" in str(reason).lower()):
            try:
                self.note_rate_limited(provider_key)
            except Exception as exc:  # noqa: BLE001 -- never block on tuning
                log_event("DEBUG", "rpm_tuning_failed",
                          {"provider": provider_key, "error": str(exc)[:120]})

    def cooldown_active(self, provider_key: str,
                        model: Optional[str] = None) -> bool:
        entry = self.ledger.get("providers", {}).get(provider_key) or {}
        cooldowns = entry.get("cooldowns") or {}
        now = _now()
        for key in ([model] if model else []) + ["*"]:
            until = float(cooldowns.get(key, 0.0) or 0.0)
            if until > now:
                return True
        return False

    # -- reporting -------------------------------------------------------
    def utilization(self, pdef: providers.ProviderDef) -> float:
        """Worst-case utilization across all constraints (0..1, -1 unknown)."""
        entry = self._entry(pdef.key)
        scores: List[float] = []

        def ratio(used: float, limit: float) -> None:
            if limit and limit > 0:
                scores.append(min(1.0, used / limit))

        ratio(len(self._window(pdef.key)), self.effective_rpm(pdef))
        ratio(sum(t for _, t in self._window(pdef.key)), pdef.tpm)
        ratio(entry["requests_today"], pdef.rpd)
        ratio(entry["tokens_today"], pdef.tpd)
        ratio(entry["requests_month"], pdef.monthly_requests)
        ratio(entry["tokens_month"], pdef.monthly_tokens)
        if pdef.key == "openrouter":
            live = self._live.get("openrouter") or {}
            if live.get("limit"):
                scores.append(min(1.0, live.get("used", 0) / max(1, live["limit"])))
        if not scores:
            return -1.0  # completely unknown
        return max(scores)

    def available_requests(self, pdef: providers.ProviderDef) -> int:
        entry = self._entry(pdef.key)
        # The live OpenRouter counter is authoritative when fresh.
        if pdef.key == "openrouter":
            live = self._live.get("openrouter") or {}
            if live.get("limit", 0) > 0:
                return max(0, int(live.get("remaining", 0)))
        caps: List[int] = []
        if pdef.rpd > 0:
            caps.append(max(0, pdef.rpd - entry["requests_today"]))
        if pdef.monthly_requests > 0:
            caps.append(max(0, pdef.monthly_requests - entry["requests_month"]))
        return min(caps) if caps else 1000  # unknown -> optimistic

    def total_available_requests(self) -> int:
        total = 0
        for pdef in providers.enabled_providers():
            if not self.cooldown_active(pdef.key):
                total += self.available_requests(pdef)
        return total

    def next_reset(self, pdef: providers.ProviderDef) -> datetime:
        if pdef.quota_style == "monthly" and pdef.monthly_requests > 0 \
                and pdef.rpd <= 0:
            return _reset_next_month()
        if pdef.quota_style == "local":
            return datetime.now(timezone.utc)
        return _reset_next_day(pdef.reset_tz)

    def next_reset_iso(self) -> str:
        """Earliest reset across enabled providers (ISO-8601, UTC)."""
        candidates = [self.next_reset(p).astimezone(timezone.utc)
                      for p in providers.enabled_providers()]
        if not candidates:
            return (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        return min(candidates).isoformat()

    def snapshot(self) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for pdef in providers.all_provider_defs():
            if not any(p.key == pdef.key for p in providers.enabled_providers()):
                continue
            entry = self._entry(pdef.key)
            rows.append({
                "provider": pdef.key,
                "requests_today": entry["requests_today"],
                "rpd": pdef.rpd or None,
                "tokens_today": entry["tokens_today"],
                "tpd": pdef.tpd or None,
                "requests_month": entry["requests_month"],
                "monthly_requests": pdef.monthly_requests or None,
                "available": self.available_requests(pdef),
                "utilization": round(self.utilization(pdef), 3),
                "cooldown": self.cooldown_active(pdef.key),
                "next_reset": self.next_reset(pdef).isoformat(),
            })
        return {
            "providers": rows,
            "total_available_requests": self.total_available_requests(),
            "reserve_pct": config.QUOTA_RESERVE_PCT,
            "role_caps": config.QUOTA_ROLE_CAPS,
            "openrouter_live": self._live.get("openrouter"),
        }

    # -- live OpenRouter reading -----------------------------------------
    async def refresh_live(self, force: bool = False) -> None:
        """Read the account's free-model counter from ``GET /api/v1/key``."""
        live = self._live.get("openrouter") or {}
        if not force and live and _now() - live.get("checked_ts", 0) < _LIVE_TTL_S:
            return
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            return
        try:
            import httpx  # dependency of openai; optional here
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/key",
                    headers={"Authorization": f"Bearer {key}"})
            payload = resp.json().get("data", {})
            quota = payload.get("free_model_daily_requests") or {}
            self._live["openrouter"] = {
                "used": int(quota.get("used", 0) or 0),
                "limit": int(quota.get("limit", 0) or 0),
                "remaining": int(quota.get("remaining", 0) or 0),
                "checked_ts": _now(),
            }
            log_event("INFO", "openrouter_quota_refreshed",
                      self._live["openrouter"])
        except Exception as exc:  # noqa: BLE001 -- optional telemetry
            log_event("DEBUG", "openrouter_quota_refresh_failed",
                      {"error": str(exc)})


_GOVERNORS: Dict[str, Governor] = {}


def get_governor(workspace_root: Optional[Path] = None) -> Governor:
    """Return the shared governor for a workspace (created once)."""
    root = Path(workspace_root) if workspace_root else config.WORKSPACE_ROOT
    key = str(root.resolve())
    if key not in _GOVERNORS:
        _GOVERNORS[key] = Governor(root)
    return _GOVERNORS[key]
