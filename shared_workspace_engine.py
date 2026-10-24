"""SharedWorkspaceEngine: cross-project pattern / anti-pattern memory.

Owns three pieces of shared state under ``.shared_workspace/``:

* ``pattern_memory.json``  -- promoted solution patterns (``P-0001``, ...),
* ``anti_patterns.json``   -- known-bad approaches,
* ``draft_logs/``          -- transient draft records written before a
  pattern is promoted (deleted after promotion).

Also hosts :func:`apply_cache_headers`, the provider-specific prompt-caching
annotation helper applied to *static* prompt parts (system prompt, global
rules, pattern excerpts).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from logger import log_event
from validation import (AntiPatternRecord, PatternRecord, ReviewVerdict,
                        TaskSpec, validate_or_error)

_PATTERN_ID_RE = re.compile(r"^P-(\d{4,})$")
_ANTI_ID_RE = re.compile(r"^A-(\d{4,})$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Severity ordering used when merging anti-pattern records.
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

#: Prune patterns whose ``last_used`` timestamp is older than this.
UNUSED_AFTER_DAYS = 30

#: How many patterns / anti-patterns a query returns at most.
QUERY_LIMIT = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens (``ERR_JWT_EXPIRED`` -> ``err jwt expired``)."""
    return _TOKEN_RE.findall((text or "").lower())


def _parse_iso(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


class SharedWorkspaceEngine:
    """Read/write access to the shared cross-project memory."""

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = Path(workspace_root) if workspace_root \
            else config.WORKSPACE_ROOT
        self.shared_dir = self.workspace_root / ".shared_workspace"
        self.patterns_path = self.shared_dir / "pattern_memory.json"
        self.anti_patterns_path = self.shared_dir / "anti_patterns.json"
        self.draft_dir = self.shared_dir / "draft_logs"
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.draft_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Low-level JSON helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log_event("ERROR", "shared_state_unreadable",
                      {"path": str(path), "error": "invalid or unreadable JSON"})
        return default

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        """Persist shared-workspace state atomically, with secrets redacted.

        The single write funnel for ``pattern_memory.json`` and
        ``anti_patterns.json``.  Redaction here matters more than it looks:
        an escalation's anti-pattern ``description`` is built from the raw
        provider exception text, and an OpenAI-compatible SDK error body
        embeds the account's ``user_id``.  That string is then re-read by
        ``query_anti_patterns`` and spliced into every subsequent Builder,
        Reviewer and CampaignDirector prompt -- so an account identifier that
        landed here reached third-party providers on every run.
        """
        import redaction
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(redaction.redact_obj(payload), indent=2,
                                  ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)  # atomic-ish swap so readers never see a partial file

    # ------------------------------------------------------------------
    # Pattern memory
    # ------------------------------------------------------------------
    def _load_patterns(self) -> List[Dict[str, Any]]:
        payload = self._load_json(self.patterns_path, {"patterns": []})
        return list(payload.get("patterns", [])) if isinstance(payload, dict) else []

    def _save_patterns(self, patterns: List[Dict[str, Any]]) -> None:
        """Persist patterns after pruning (unused > 30 days or conf < 0)."""
        now = datetime.now(timezone.utc)
        kept: List[Dict[str, Any]] = []
        for pattern in patterns:
            confidence = float(pattern.get("confidence", 0.0))
            if confidence < 0:
                continue  # pruned by decrement
            last_used = _parse_iso(pattern.get("last_used", ""))
            if last_used is not None:
                age_days = (now - last_used).total_seconds() / 86400.0
                if age_days > UNUSED_AFTER_DAYS:
                    log_event("INFO", "pattern_pruned_unused",
                              {"pattern_id": pattern.get("id"),
                               "age_days": round(age_days, 1)})
                    continue
            kept.append(pattern)
        self._write_json(self.patterns_path, {"patterns": kept})

    def query_patterns(self, error_signature_or_keywords: str) -> List[Dict[str, Any]]:
        """Return up to 3 relevant patterns (confidence > 0, best first).

        Scoring: token overlap between the query and the pattern's
        ``signature`` + ``fix_summary``; exact ``error_signature`` matches
        rank above fuzzy keyword matches.  Hits refresh ``last_used`` (that
        is what feeds the 30-day prune).

        v2.8: the ``last_used`` refresh no longer rewrites the store.  It used
        to call ``_save_patterns`` here, which runs the 30-day TTL prune -- so a
        pure *read* could delete records.  That is how 11 of 23 promoted
        patterns disappeared without any write intent.  Refreshing an access
        timestamp is not worth silently destroying history; the TTL now only
        applies on an explicit save.
        """
        tokens = set(_tokenize(error_signature_or_keywords))
        if not tokens:
            return []
        scored: List[Dict[str, Any]] = []
        for pattern in self._load_patterns():
            try:
                record, err = validate_or_error(PatternRecord, pattern)
                if err:
                    continue
            except Exception:  # malformed entry must not break queries
                continue
            if record.confidence <= 0:
                continue
            haystack = _tokenize(f"{record.signature} {record.fix_summary}")
            overlap = len(tokens.intersection(haystack))
            exact = 1 if _tokenize(record.signature) and \
                set(_tokenize(record.signature)) & tokens else 0
            if overlap <= 0 and not exact:
                continue
            entry = dict(pattern)
            entry["_score"] = (exact, overlap, record.confidence)
            scored.append(entry)

        scored.sort(key=lambda item: item["_score"], reverse=True)
        top = scored[:QUERY_LIMIT]
        results = []
        for entry in top:
            entry.pop("_score", None)
            results.append(entry)
        return results

    def propose_and_verify_pattern(self, task: TaskSpec,
                                   review_result: ReviewVerdict,
                                   diff: str) -> Optional[Dict[str, Any]]:
        """Promote a solved task into pattern memory.

        Promotion rule (strict): ``verdict == APPROVED`` **and**
        ``review_result.automated_tests.failed == 0``.  A draft record is
        written to ``draft_logs/`` first and deleted once promoted.
        """
        if review_result.verdict != "APPROVED" or \
                review_result.automated_tests.failed != 0:
            return None  # never promote a red or unreviewed change

        signature = task.error_signature or f"TASK:{task.id}:{task.title}"
        fix_summary = task.objective or task.title
        diff_hash = hashlib.sha256((diff or "").encode("utf-8")).hexdigest()

        draft: Dict[str, Any] = {
            "task_id": task.id,
            "signature": signature,
            "fix_summary": fix_summary,
            "diff_hash": diff_hash,
            "proposed_at": _now_iso(),
        }
        draft_path = self.draft_dir / \
            f"draft_{task.id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
        try:
            import redaction
            draft_path.write_text(
                json.dumps(redaction.redact_obj(draft), indent=2), encoding="utf-8")
        except OSError as exc:
            log_event("ERROR", "pattern_draft_write_failed",
                      {"task_id": task.id, "error": str(exc)})

        patterns = self._load_patterns()
        next_number = 1
        for pattern in patterns:
            match = _PATTERN_ID_RE.match(str(pattern.get("id", "")))
            if match:
                next_number = max(next_number, int(match.group(1)) + 1)
        record = {
            "id": f"P-{next_number:04d}",
            "signature": signature,
            "fix_summary": fix_summary,
            "diff_hash": diff_hash,
            "confidence": 1.0,
            "created_at": _now_iso(),
            "last_used": _now_iso(),
        }
        validated, err = validate_or_error(PatternRecord, record)
        if err:
            log_event("ERROR", "pattern_record_invalid",
                      {"task_id": task.id, "error": err["error"]})
            return None

        patterns.append(record)
        self._save_patterns(patterns)  # prunes as a side effect of every write
        try:
            draft_path.unlink(missing_ok=True)  # draft consumed on promotion
        except OSError:
            pass
        log_event("INFO", "pattern_promoted",
                  {"pattern_id": record["id"], "task_id": task.id,
                   "signature": signature, "diff_hash": diff_hash})
        return record

    def decrement_pattern_confidence(self, pattern_id: str,
                                     amount: float = 0.25) -> None:
        """Lower a pattern's confidence by 0.25; prune it below zero."""
        patterns = self._load_patterns()
        updated: List[Dict[str, Any]] = []
        found = False
        for pattern in patterns:
            if pattern.get("id") == pattern_id and not found:
                found = True
                pattern = dict(pattern)
                pattern["confidence"] = round(
                    float(pattern.get("confidence", 0.0)) - amount, 4)
                if pattern["confidence"] < 0:
                    log_event("INFO", "pattern_pruned_low_confidence",
                              {"pattern_id": pattern_id})
                    continue  # drop the pattern entirely
            updated.append(pattern)
        if found:
            self._save_patterns(updated)
            log_event("INFO", "pattern_confidence_decremented",
                      {"pattern_id": pattern_id, "amount": amount})

    # ------------------------------------------------------------------
    # Anti-pattern memory
    # ------------------------------------------------------------------
    def _load_anti_patterns(self) -> List[Dict[str, Any]]:
        payload = self._load_json(self.anti_patterns_path, {"anti_patterns": []})
        return list(payload.get("anti_patterns", [])) \
            if isinstance(payload, dict) else []

    def query_anti_patterns(self, error_signature_or_keywords: str) -> List[Dict[str, Any]]:
        """Return up to 3 anti-patterns related to the query tokens."""
        tokens = set(_tokenize(error_signature_or_keywords))
        if not tokens:
            return []
        scored: List[Dict[str, Any]] = []
        for anti in self._load_anti_patterns():
            record, err = validate_or_error(AntiPatternRecord, anti)
            if err:
                continue
            haystack = _tokenize(
                f"{record.signature} {record.description} {record.remediation}")
            overlap = len(tokens.intersection(haystack))
            if overlap <= 0:
                continue
            entry = dict(anti)
            entry["_score"] = overlap
            scored.append(entry)
        scored.sort(key=lambda item: item["_score"], reverse=True)
        return scored[:QUERY_LIMIT]

    @staticmethod
    def _merge_anti_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge same-signature anti-pattern records (pure; no disk access).

        Historical stores contain repeated signatures (one ``A-xxxx`` was
        appended per escalation).  The merge keeps the *lowest* id, sums
        ``occurrences``, and lets the later (newest) description win while
        keeping the strongest severity -- so the freshest failure
        information is preserved without losing history volume.
        """
        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for record in records:
            signature = str(record.get("signature", ""))
            if not signature:
                continue  # unusable entry: dropped on save
            if signature in merged:
                kept = merged[signature]
                kept["occurrences"] = int(kept.get("occurrences", 1)) + \
                    int(record.get("occurrences", 1))
                if record.get("description"):
                    kept["description"] = record["description"]
                if record.get("remediation"):
                    kept["remediation"] = record["remediation"]
                if _SEVERITY_RANK.get(str(record.get("severity", "")).lower(), 1) > \
                        _SEVERITY_RANK.get(str(kept.get("severity", "")).lower(), 1):
                    kept["severity"] = record["severity"]
            else:
                entry = dict(record)
                entry.setdefault("occurrences", 1)
                merged[signature] = entry
                order.append(signature)
        return [merged[s] for s in order]

    def _save_anti_patterns(self, records: List[Dict[str, Any]]) -> None:
        """Persist anti-patterns after merging same-signature duplicates."""
        self._write_json(self.anti_patterns_path,
                         {"anti_patterns": self._merge_anti_records(records)})

    def add_anti_pattern(self, signature: str, description: str,
                         remediation: str = "", severity: str = "medium") -> Optional[Dict[str, Any]]:
        """Record a new anti-pattern (used when cycles escalate unresolved).

        Same-signature records are merged instead of duplicated: the
        existing record's ``occurrences`` is incremented and the newest
        description / remediation / strongest severity are kept, with a
        stable id.  New ids derive from ``max(existing) + 1`` so merges and
        prunes can never cause an id collision.
        """
        anti_patterns = self._merge_anti_records(self._load_anti_patterns())
        existing = next((a for a in anti_patterns
                         if str(a.get("signature", "")) == str(signature)),
                        None)
        if existing is not None:
            existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
            if description:
                existing["description"] = description
            if remediation:
                existing["remediation"] = remediation
            if _SEVERITY_RANK.get(str(severity).lower(), 1) > \
                    _SEVERITY_RANK.get(
                        str(existing.get("severity", "medium")).lower(), 1):
                existing["severity"] = severity
            record = dict(existing)
            record_id = str(existing.get("id", ""))
            # Re-append as the newest entry and drop any same-signature
            # duplicates so the merge-on-save keeps THIS description.
            anti_patterns = [a for a in anti_patterns
                             if str(a.get("signature", "")) != str(signature)]
            anti_patterns.append(record)
        else:
            next_number = 1
            for anti in anti_patterns:
                match = _ANTI_ID_RE.match(str(anti.get("id", "")))
                if match:
                    next_number = max(next_number, int(match.group(1)) + 1)
            record = {
                "id": f"A-{next_number:04d}",
                "signature": signature,
                "description": description,
                "remediation": remediation,
                "severity": severity,
                "occurrences": 1,
            }
            validated, err = validate_or_error(AntiPatternRecord, record)
            if err:
                log_event("ERROR", "anti_pattern_record_invalid",
                          {"error": err["error"]})
                return None
            anti_patterns.append(record)
            record_id = record["id"]
        self._save_anti_patterns(anti_patterns)  # merge is a save side effect
        log_event("INFO", "anti_pattern_recorded",
                  {"id": record_id, "signature": signature,
                   "occurrences": int(record.get("occurrences", 1))})
        return record

    # ------------------------------------------------------------------
    # Provider-specific prompt caching (static prompt parts only)
    # ------------------------------------------------------------------
