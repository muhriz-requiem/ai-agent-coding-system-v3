"""Secret redaction and egress scanning.

Two deliberately separate primitives, because conflating them is what makes
naive scanners useless:

``redact_known_secrets(text)``
    Replaces the **literal values of secrets this machine already holds**
    (credential-shaped environment variables, read from ``os.environ`` and from
    the workspace/project ``.env`` files) with ``[REDACTED]``.  Because it can
    only match values the process legitimately has, it has **zero false
    positives**: the identifier ``ALPACA_API_KEY`` and the expression
    ``os.getenv("ALPACA_API_KEY")`` pass through untouched, so reviewing code
    that legitimately handles credentials keeps working.  Safe to apply
    unconditionally to logs and to outbound prompts.

``scan_secrets(text)``
    Pure *detection* of credential-shaped patterns (provider key formats,
    private-key blocks, provider **account identifiers**).  Never mutates the
    text it inspects.  Used to **gate egress** -- a submission file bound for
    Kaggle, or the tree before a commit -- where a false positive costs one
    blocked submission and a false negative ships a live key.

``scrub_secrets(text)``
    The applying counterpart of :func:`scan_secrets`, for one-shot remediation
    of files that already contain leaked values.

Findings carry a ``hint`` (``sha256(value)[:8]``) rather than the value or any
prefix of it, so a report lets an operator confirm that four hits are the same
identifier without rendering any of it readable.

Every public function is total: a bug here must never escape into the logging
or orchestration paths that call it.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import config

__all__ = [
    "Finding",
    "REDACTION_MARK",
    "known_secret_values",
    "redact_known_secrets",
    "redact_obj",
    "reset_cache",
    "scan_secrets",
    "scan_path",
    "scan_paths",
    "scan_tree",
    "tracked_files",
    "scrub_secrets",
    "scrub_json_file",
    "apply_scrub_json_file",
    "write_json_atomic",
]

#: Replacement token written in place of a secret value.
REDACTION_MARK = "[REDACTED]"

#: Values that are documentation rather than credentials.  Matched by
#: **normalized equality**, never by substring: short words like "here" or
#: "none" occur by chance inside real base64/hex secrets, and a substring test
#: silently fails to redact them -- the one failure direction that must not
#: happen.
_PLACEHOLDER_EXACT = frozenset({
    "", "-", "none", "null", "nil", "na", "n/a", "true", "false", "unset",
    "empty", "changeme", "todo", "tbd", "xxx", "xxxx", "test", "dummy",
    "example", "placeholder", "redacted", "your_key_here",
})

#: A value that is *entirely* a placeholder phrase (``YOUR_API_KEY_HERE``,
#: ``changeme-please``, ``insert-token``).  Anchored full-match so a real key
#: can never match it.
_PLACEHOLDER_SHAPE_RE = re.compile(
    r"(?:your|yours|my|our|insert|enter|replace|changeme|change[_\- ]?me"
    r"|placeholder|example|sample|dummy|fake|test|todo|tbd|xxx+|abc|foo|bar"
    r"|baz|secret|password|passwd|api[_\- ]?key|access[_\- ]?key|token"
    r"|key|here|redacted|hidden)"
    r"(?:[_\-. ]*(?:your|yours|insert|enter|replace|here|changeme|value"
    r"|secret|token|key|id|api|password|me|now|below|above))*"
    r"[_\-. ]*",
    re.IGNORECASE)

#: Eight or more identical characters in a row (``xxxxxxxx``, ``00000000``).
_REPEATS_RE = re.compile(r"(.)\1{7,}")

#: Environment variable names treated as credential carriers.  Matched
#: case-insensitively against the whole name as a word-ish token so that
#: ``MONKEY_APPLE`` (no underscore boundary before "KEY") is not matched while
#: ``ALPACA_API_KEY`` and ``X-Api-Key`` are.
_SECRET_NAME_RE = re.compile(
    r"(?:^|[_\-.])(?:API_?KEY|KEYS?|SECRET|SECRETS?|TOKEN|TOKENS?|PASSWORD|PASSWD"
    r"|PASS(?:PHRASE)?|CREDENTIALS?|AUTH)(?:$|[_\-.0-9])",
    re.IGNORECASE)

#: Names that *look* credential-shaped but hold a list of paths or names rather
#: than a secret.  ``SECRET_SCRUB_TARGETS`` is the live case: its value is
#: ``.shared_workspace/anti_patterns.json``, a path.  Harvesting it as a secret
#: caused two real problems -- the same path string is documented in
#: ``.env.example``, so ``--preflight scan --committed`` reported a leak in a
#: tracked file, and redaction would blank the list of files the operator is
#: supposed to scrub, hiding the very diagnostic that matters.
#:
#: This is a precision fix for a well-understood class rather than a loosening
#: of :func:`_usable_secret`, which stays biased toward including.
_NON_SECRET_NAME_RE = re.compile(
    r"(?:_TARGETS|_PATHS|_FILES|_DIRS|_WHITELIST|_ALLOWLIST|_DENYLIST"
    r"|_PREFIXES|_SUFFIXES|_NAMES|_KEYS_LIST)$",
    re.IGNORECASE)


def _is_secret_bearing_name(name: str) -> bool:
    """True when *name* denotes an actual secret rather than a list of them."""
    return bool(_SECRET_NAME_RE.search(name or "")) and not _NON_SECRET_NAME_RE.search(
        name or "")


#: Directories never walked by :func:`scan_tree`.
_SKIP_DIRS = frozenset({
    ".git", ".pytest_cache", "__pycache__", ".ruff_cache", ".mypy_cache",
    "node_modules", ".venv", "venv", ".idea", ".vscode",
})


class Finding(Dict[str, Any]):
    """One detected credential.

    A plain ``dict`` (so it serializes straight into ``log_event``) with keys
    ``rule``, ``line``, ``hint`` and ``path`` (when file-scanned).  Never
    contains the matched value, nor any substring of it.
    """

    __slots__ = ()


# ---------------------------------------------------------------------------
# Pattern rules (detection only)
# ---------------------------------------------------------------------------
#: ``(rule name, compiled pattern)``.  Ordered most-specific first so a
#: provider-specific rule claims its match before the generic assignment rule
#: can.  Every pattern is anchored to a credential-shaped literal; none of them
#: match prose, identifiers or ordinary code.
#:
#: Note on trailing anchors: rules below deliberately do **not** end in ``\b``.
#: A word boundary after a counted class only holds when the next character
#: happens to be a non-word character, so ``\bAIza[...]{35}\b`` silently fails
#: on any key with more than the nominal number of trailing characters -- and a
#: secret-detection rule that quietly stops matching is the one failure mode
#: this module exists to prevent.
_SECRET_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("private_key_block",
     re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}")),
    ("openrouter_key", re.compile(r"\bsk-or-v[0-9]+-[A-Za-z0-9_\-]{16,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}")),
    ("google_api_key", re.compile(r"\bAIza[A-Za-z0-9_\-]{35,}")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[A-Z0-9]{16}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    # Provider *account* identifiers.  These are not API keys and never
    # authenticate anything, but they are stable account metadata and they do
    # leak: an OpenRouter SDK error body embeds ``user_id``, which reached this
    # workspace's anti-pattern memory via a failed escalation and was then
    # replayed into every subsequent model prompt.
    ("openrouter_account_id", re.compile(r"\buser_[A-Za-z0-9]{20,}")),
    ("openrouter_org_id", re.compile(r"\borg_[A-Za-z0-9]{20,}")),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_\-]{33}\b")),
    # Generic assignment: ``api_key = "..."`` / ``SECRET: '...'``.  Requires a
    # quoted or bare value of at least 16 credential-ish characters, so
    # ``api_key = os.getenv("X")`` and ``password = ""`` do not match.
    ("generic_assignment",
     re.compile(
         r"(?i)\b(?:api[_\-]?key|secret|token|password|passwd|credential"
         r"|private[_\-]?key|access[_\-]?key)\b\s*[:=]\s*"
         r"[\"']?([A-Za-z0-9_\-\+/=]{16,})[\"']?")),
)


def _hint(value: str) -> str:
    """Non-reversible correlation id for *value* (never the value itself)."""
    try:
        return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:8]
    except Exception:  # noqa: BLE001 -- hint must never raise
        return "?"


def _iter_text_nodes(node: Any) -> Iterable[str]:
    """Yield every string in a nested JSON-like structure (keys included).

    Keys are included deliberately: a leaked identifier has been observed in a
    serialized ``description`` value, and a credential pasted into a JSON *key*
    is just as much a leak.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            yield current
        elif isinstance(current, dict):
            for key, value in current.items():
                if isinstance(key, str):
                    yield key
                stack.append(value)
        elif isinstance(current, (list, tuple, set)):
            stack.extend(current)


# ---------------------------------------------------------------------------
# Known-secret discovery
# ---------------------------------------------------------------------------
_VALUE_CACHE: Dict[str, Any] = {"key": None, "values": (), "pattern": None}


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Minimal ``.env`` reader (stdlib only; no python-dotenv dependency).

    Reads ``KEY=VALUE`` lines, tolerating ``export`` prefixes, ``#`` comments
    and quoted values.  Never raises.
    """
    out: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        if not sep:
            continue
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name:
            out[name] = value
    return out


def _env_secret_values(workspace_root: Optional[Path]) -> List[str]:
    """Every credential-shaped value visible to this process.

    Values are collected from **all** sources rather than first-wins-per-name:
    if ``os.environ`` carries a placeholder for a key while the ``.env`` file
    carries the live one, a name-keyed map would keep only the placeholder and
    silently fail to redact the real credential.  Redaction needs the value
    *set*, not the name mapping, so duplicates across sources are harmless.
    """
    values: List[str] = []
    try:
        for name, value in os.environ.items():
            if _is_secret_bearing_name(name or ""):
                values.append(value)
    except Exception:  # noqa: BLE001 -- never raise out of a redaction call
        pass

    root = Path(workspace_root or config.WORKSPACE_ROOT)
    env_paths = [root / ".env"]
    try:
        projects = root / "projects"
        if projects.is_dir():
            for child in sorted(projects.iterdir()):
                if child.is_dir():
                    env_paths.append(child / ".env")
    except OSError:
        pass

    for env_path in env_paths:
        for name, value in _parse_env_file(env_path).items():
            if _is_secret_bearing_name(name):
                values.append(value)
    return values


def _env_candidates(workspace_root: Optional[Path]) -> Dict[str, str]:
    """Diagnostic view of discovered ``name -> value`` pairs (first wins).

    Not used for redaction -- see :func:`_env_secret_values` -- but handy for
    inspecting which credential-shaped variables a deployment exposes.
    """
    found: Dict[str, str] = {}
    try:
        for name, value in os.environ.items():
            if _is_secret_bearing_name(name or ""):
                found.setdefault(name, value)
    except Exception:  # noqa: BLE001
        pass
    root = Path(workspace_root or config.WORKSPACE_ROOT)
    env_paths = [root / ".env"]
    try:
        projects = root / "projects"
        if projects.is_dir():
            for child in sorted(projects.iterdir()):
                if child.is_dir():
                    env_paths.append(child / ".env")
    except OSError:
        pass
    for env_path in env_paths:
        for name, value in _parse_env_file(env_path).items():
            if _is_secret_bearing_name(name):
                found.setdefault(name, value)
    return found


def _usable_secret(value: Any, min_len: int) -> bool:
    """True when *value* is plausibly a live secret worth redacting.

    Deliberately biased toward **including** a value: a false positive blanks
    an innocuous string that merely looks key-ish, while a false negative lets
    a real credential reach a log, a prompt or a submission.  Placeholder
    detection is therefore exact/shape-based rather than substring-based.
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) < min_len:
        return False
    normalized = stripped.lower()
    if normalized in _PLACEHOLDER_EXACT:
        return False
    # Only treat it as a placeholder when the WHOLE value is placeholder
    # prose -- a partial shape match would skip real keys again.
    if _PLACEHOLDER_SHAPE_RE.fullmatch(normalized):
        return False
    if _REPEATS_RE.search(stripped):
        return False
    if len(set(stripped)) < 8:
        return False
    return True


def known_secret_values(workspace_root: Optional[Path] = None,
                        refresh: bool = False) -> Tuple[str, ...]:
    """Live secret values this process should never emit, longest first.

    Cached per ``workspace_root``: this is called on every log event and every
    outbound prompt, so re-reading ``.env`` each time would be wasteful.  Pass
    ``refresh=True`` after mutating the environment (tests do).
    """
    try:
        root = Path(workspace_root or config.WORKSPACE_ROOT)
    except Exception:  # noqa: BLE001
        return ()
    cache_key = str(root)
    if not refresh and _VALUE_CACHE["key"] == cache_key:
        return _VALUE_CACHE["values"]

    min_len = getattr(config, "SECRET_SCAN_MIN_LEN", 8)
    try:
        candidates = _env_secret_values(root)
    except Exception:  # noqa: BLE001
        candidates = ()
    values = tuple(sorted({v for v in candidates if _usable_secret(v, min_len)},
                          key=len, reverse=True))
    pattern = None
    if values:
        try:
            pattern = re.compile("|".join(re.escape(v) for v in values))
        except Exception:  # noqa: BLE001 -- pathological input: fall back to none
            pattern = None
    _VALUE_CACHE.update({"key": cache_key, "values": values, "pattern": pattern})
    return values


def _known_pattern(workspace_root: Optional[Path] = None) -> Optional["re.Pattern[str]"]:
    """Compiled alternation of known secret values, or None when disabled."""
    if not getattr(config, "REDACTION_ENABLED", True):
        return None
    known_secret_values(workspace_root)
    return _VALUE_CACHE["pattern"]


def reset_cache() -> None:
    """Drop the discovered-secret cache (tests; after mutating the env)."""
    _VALUE_CACHE.update({"key": None, "values": (), "pattern": None})


def write_json_atomic(path: Any, payload: Any) -> None:
    """Serialize *payload* to *path* via tmp-file + ``replace``.

    The shared write discipline for state that other processes poll: readers
    never observe a torn or half-written file.  Secret values are redacted on
    the way out, so no caller needs its own redaction step to keep a
    persisted artifact safe.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(redact_obj(payload), indent=2,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)


# ---------------------------------------------------------------------------
# Primitive 1: redact known secrets (the always-on one)
# ---------------------------------------------------------------------------
#: Identifier patterns applied at SINKS, not just at scan time.
#:
#: Sink redaction is normally exact-value-only on purpose: blanking every
#: credential *shape* mid-prompt would blind a Reviewer that must judge code
#: handling credentials.  Account/org identifiers are the deliberate exception.
#: They carry no diagnostic value -- "the provider returned 402" is the signal,
#: not *whose* account it was -- and they are the one shape that demonstrably
#: leaked: an OpenAI-compatible error body embeds ``user_id``, that string was
#: replayed into shared memory and then into every later prompt.
#:
#: Found by the ``logtriage`` project this harness built to diagnose itself:
#: 48 raw ``user_*`` and 11 raw ``org_*`` identifiers were still sitting in
#: ``logs/agent_log.jsonl`` inside provider exception bodies, because the scan
#: rules existed but only ever ran at the egress gates.
_SINK_IDENTIFIER_PATTERNS = (
    re.compile(r"\buser_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\borg_[A-Za-z0-9]{16,}\b"),
)


def redact_identifiers(text: Any) -> str:
    """Blank provider account/org identifiers. Never raises."""
    if not isinstance(text, str) or not text:
        return text
    if "user_" not in text and "org_" not in text:
        return text          # fast path: the common clean case
    try:
        for pattern in _SINK_IDENTIFIER_PATTERNS:
            text = pattern.sub("[REDACTED_ACCOUNT_ID]", text)
        return text
    except Exception:  # noqa: BLE001
        return text


def redact_known_secrets(text: Any, workspace_root: Optional[Path] = None) -> str:
    """Replace known live secret values in *text* with ``[REDACTED]``.

    Returns *text* unchanged (same object) when redaction is disabled, when
    the input is not a string, or when no known value occurs in it -- the fast
    path for every log line and prompt that happens to be clean.
    """
    if not isinstance(text, str) or not text:
        return text
    text = redact_identifiers(text)
    pattern = _known_pattern(workspace_root)
    if pattern is None:
        return text
    try:
        if not pattern.search(text):
            return text
        return pattern.sub(REDACTION_MARK, text)
    except Exception:  # noqa: BLE001 -- redaction must never break its caller
        return text


def redact_obj(node: Any, workspace_root: Optional[Path] = None) -> Any:
    """Recursively redact every string in a nested structure.

    Non-string leaves are returned unchanged, so the result stays JSON-shaped
    and callers can persist it directly.
    """
    pattern = _known_pattern(workspace_root)
    if pattern is None:
        return node

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            try:
                value = redact_identifiers(value)
                return pattern.sub(REDACTION_MARK, value) if pattern.search(value) else value
            except Exception:  # noqa: BLE001
                return value
        if isinstance(value, dict):
            return {(_walk(k) if isinstance(k, str) else k): _walk(v)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_walk(v) for v in value)
        return value

    try:
        return _walk(node)
    except Exception:  # noqa: BLE001
        return node


# ---------------------------------------------------------------------------
# Primitive 2: scan (detection only)
# ---------------------------------------------------------------------------
def _line_of(text: str, index: int) -> int:
    """1-based line number of *index* within *text* (capped work)."""
    return text.count("\n", 0, index) + 1


def scan_secrets(text: Any) -> List[Finding]:
    """Detect credential-shaped patterns in *text*.  Never mutates.

    Also reports any **known live secret value** found, under the rule
    ``known_value`` -- that check is exact-match and therefore has no false
    positives, which is why the egress gates can act on the result.
    """
    findings: List[Finding] = []
    if not isinstance(text, str) or not text:
        return findings

    try:
        known = known_secret_values()
        for value in known:
            index = text.find(value)
            if index >= 0:
                findings.append(Finding(rule="known_value", line=_line_of(text, index),
                                        hint=_hint(value)))
    except Exception:  # noqa: BLE001
        pass

    try:
        for rule, pattern in _SECRET_RULES:
            for match in pattern.finditer(text):
                # Prefer the capture group (generic_assignment) when present so
                # the hint identifies the value, not the whole assignment.
                matched = match.group(1) if match.groups() else match.group(0)
                findings.append(Finding(rule=rule, line=_line_of(text, match.start()),
                                        hint=_hint(matched)))
    except Exception:  # noqa: BLE001
        pass
    return findings


def scan_path(path: Any, max_bytes: int = 4 * 1024 * 1024) -> List[Finding]:
    """Scan one file for secrets.  Unreadable/oversized files yield no findings."""
    try:
        target = Path(path)
        if not target.is_file():
            return []
        if target.stat().st_size > max_bytes:
            return []
        text = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError, ValueError):
        return []
    findings = scan_secrets(text)
    for finding in findings:
        finding["path"] = str(target)
    return findings


def scan_paths(paths: Sequence[Any], max_bytes: int = 4 * 1024 * 1024
               ) -> List[Finding]:
    """Scan an explicit list of files, in order, skipping unreadable ones.

    Separate from :func:`scan_tree` on purpose.  ``scan_tree`` answers "is there
    credential-shaped content anywhere in this tree", which is the right
    question for a *working directory* and the wrong one for a *repository*:
    it is always red, because a live ``.env`` and the secret-scanner test
    fixtures both match, and a check that can never pass is a check nobody
    reads.  This takes the caller's list, so a caller can scope the question to
    what could actually be committed.
    """
    findings: List[Finding] = []
    for raw in paths or ():
        try:
            candidate = Path(raw)
        except TypeError:
            continue
        if not candidate.is_file():
            continue
        findings.extend(scan_path(candidate, max_bytes=max_bytes))
    return findings


def scan_allowlist(root: Any) -> List[str]:
    """Patterns in ``<root>/.secret-scan-allow`` -- files allowed to hold secrets.

    A secret scanner needs a way to say "this file is *supposed* to contain
    credential-shaped text".  Without one, the check is permanently red and
    therefore permanently ignored: this repo's own scanner fixtures are 38
    synthetic matches across five files, all of them necessary, so
    ``--preflight scan`` could never pass and taught the operator to skim past
    it.  A red check is not a safe check.

    Deliberately narrow, and only honoured by the ``--committed`` scope:

    * it lists **paths**, never values, so the allowlist cannot itself become a
      place secrets are stored;
    * it is itself tracked, so every exemption is reviewable in a diff;
    * it is ignored by the whole-tree scan, which still reports everything --
      suppressing a finding in a committed file must not silence the wider
      sweep.

    Patterns are matched with :func:`fnmatch.fnmatch` against the
    repo-relative POSIX path, so ``tests/test_redaction.py`` and
    ``tests/test_*_gates.py`` both work.
    """
    path = Path(root) / ".secret-scan-allow"
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: List[str] = []
    for line in raw.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.replace("\\", "/"))
    return out


def is_scan_allowed(rel_path: Any, patterns: Sequence[str]) -> bool:
    """True when *rel_path* matches an allowlist pattern."""
    rel = Path(rel_path).as_posix()
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns or ())


def tracked_files(root: Any) -> Tuple[List[Path], str]:
    """``(paths, note)`` for the files git currently tracks under *root*.

    ``git ls-files -z`` is the only authoritative answer to "what could a commit
    pick up", which is strictly better than re-implementing ``.gitignore``
    parsing (and cannot drift from git's own semantics).  When git is missing or
    the directory is not a repository the caller must be told, because a scan
    that silently degrades to "everything" would report a green result while
    having checked nothing -- the exact failure mode this mode exists to remove.
    """
    base = Path(root)
    try:
        proc = subprocess.run(
            ["git", "-C", str(base), "ls-files", "-z"],
            capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"git unavailable ({exc}); cannot determine tracked files"
    if proc.returncode != 0:
        return [], (f"not a git repository (git ls-files exited "
                    f"{proc.returncode}); cannot determine tracked files")
    out: List[Path] = []
    for entry in proc.stdout.split(b"\0"):
        if not entry:
            continue
        rel = entry.decode("utf-8", errors="replace")
        candidate = base / rel
        if candidate.is_file():
            out.append(candidate)
    return out, (f"{len(out)} tracked file(s)" if out
                 else "repository tracks no files")


def scan_tree(root: Any, include: Optional[Sequence[str]] = None,
              max_bytes: int = 4 * 1024 * 1024) -> List[Finding]:
    """Scan every file under *root*, skipping caches and VCS metadata.

    *include*, when given, restricts the walk to those path suffixes/prefixes
    (e.g. ``[".env"]``).  Used by ``--preflight scan`` to bound the work.
    """
    findings: List[Finding] = []
    try:
        base = Path(root)
        if not base.is_dir():
            return findings
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for name in sorted(filenames):
                candidate = Path(dirpath) / name
                if include and not any(
                        name == pat or name.endswith(pat) for pat in include):
                    continue
                findings.extend(scan_path(candidate, max_bytes=max_bytes))
    except (OSError, ValueError):
        return findings
    return findings


# ---------------------------------------------------------------------------
# Primitive 3: scrub (applying counterpart -- one-shot remediation)
# ---------------------------------------------------------------------------
def scrub_secrets(text: Any) -> Tuple[str, List[Finding]]:
    """Return *text* with detected credential patterns replaced.

    Used by ``--preflight scrub`` to remediate files that already hold leaked
    values.  **Not** applied to prompts or logs: pattern rules are tuned for
    egress blocking, and blanking them mid-prompt would blind the Reviewer on
    exactly the credential-handling code it is supposed to check.
    """
    if not isinstance(text, str) or not text:
        return text, []

    out = text
    findings: List[Finding] = []
    try:
        for rule, pattern in _SECRET_RULES:
            if not pattern.search(out):
                continue
            replacement = _make_replacer(rule, out, findings)
            out = pattern.sub(replacement, out)
    except Exception:  # noqa: BLE001
        return out, findings

    # Exact known values are handled separately: the rules above are static and
    # cannot know what this particular deployment has configured.
    try:
        for value in known_secret_values():
            while value in out:
                index = out.find(value)
                findings.append(Finding(rule="known_value",
                                        line=_line_of(out, index),
                                        hint=_hint(value)))
                out = out[:index] + REDACTION_MARK + out[index + len(value):]
    except Exception:  # noqa: BLE001
        pass
    return out, findings


def _make_replacer(rule: str, subject: str,
                   findings: List[Finding]) -> Any:
    """Build a :func:`re.sub` callback returning **only** the replacement span.

    A sub callback must return the text for the matched region, not a rebuilt
    copy of the whole subject -- returning the subject would make N matches
    produce N concatenated copies.
    """
    def _replace(match: "re.Match[str]") -> str:
        if match.groups():
            # Keep the assignment prefix (``api_key = "``) so the scrubbed file
            # still reads as valid code/config; blank only the value.
            start, end = match.span(1)
            whole = match.group(0)
            head = whole[:start - match.start()]
            tail = whole[end - match.start():]
            matched = match.group(1)
        else:
            head = tail = ""
            matched = match.group(0)
        findings.append(Finding(rule=rule, line=_line_of(subject, match.start()),
                                hint=_hint(matched)))
        return f"{head}{REDACTION_MARK}{tail}"
    return _replace


def _walk_and_scrub(payload: Any, target: str) -> Tuple[Any, int, List[Finding]]:
    """Recursively scrub every string in a parsed JSON structure."""
    total = 0
    findings: List[Finding] = []

    def _walk(node: Any) -> Any:
        nonlocal total
        if isinstance(node, str):
            new, found = scrub_secrets(node)
            if found:
                total += len(found)
                for item in found:
                    entry = Finding(item)
                    entry["path"] = target
                    findings.append(entry)
            return new
        if isinstance(node, dict):
            return {(_walk(k) if isinstance(k, str) else k): _walk(v)
                    for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, tuple):
            return tuple(_walk(v) for v in node)
        return node

    cleaned = _walk(payload)
    return cleaned, total, findings


def scrub_json_file(path: Any) -> Tuple[int, List[Finding]]:
    """Dry-run *path*: report how many strings would change, changing nothing.

    Returns ``(replacements, findings)``.  Only counts, rule names and hints
    come back out -- never the matched text.
    """
    try:
        target = Path(path)
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return 0, []
    try:
        _, total, findings = _walk_and_scrub(payload, str(target))
    except Exception:  # noqa: BLE001
        return 0, []
    return total, findings


def apply_scrub_json_file(path: Any) -> Tuple[int, Optional[Path], List[Finding]]:
    """Rewrite *path* with detected credentials blanked, after taking a backup.

    Returns ``(replacements, backup_path, findings)``.  A timestamped sibling
    backup is written **before** the replacement, and the rewrite itself is
    tmp-file + ``replace`` so a reader never observes a partial file.  Returns
    ``(0, None, [])`` without writing anything when there is nothing to change
    or the file cannot be parsed.
    """
    from datetime import datetime, timezone

    try:
        target = Path(path)
        original = target.read_text(encoding="utf-8")
        payload = json.loads(original)
    except (OSError, UnicodeError, ValueError):
        return 0, None, []

    try:
        cleaned, total, findings = _walk_and_scrub(payload, str(target))
    except Exception:  # noqa: BLE001
        return 0, None, []
    if not total:
        return 0, None, []

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    backup = target.with_name(f"{target.name}.bak-{stamp}")
    try:
        backup.write_text(original, encoding="utf-8")
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(target)
    except OSError:
        return 0, None, findings
    return total, backup, findings