"""T-004: manifest resolution and filtered downloads for modinstall.

Contract (pinned by tests/test_resolver.py, offline):

* manifest schema shared with T-006:
  {"packages": [{"name", "url", "sha256"?}, ...]}
* ``resolve_and_download(manifest, target_dir) -> list[str]`` returns the
  installed file paths.
* non-HTTPS URLs are rejected with ValueError during validation, BEFORE any
  transport is touched (the pinned tests ban real sockets to prove this).
* the transport is late-bound: ``urllib.request.urlopen`` is resolved at call
  time so tests can monkeypatch the module attribute to swap it out.
* entries carrying a ``sha256`` must match the bytes that arrived; a
  mismatching artifact is excluded and never installed. Quarantine and run
  rollback semantics belong to T-002/T-006, not to the resolver.
"""

from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def validate_url(url: str) -> str:
    """Validate ``url`` and return its normalized hostname.

    Raises:
        ValueError: when the URL is not HTTPS or is missing a hostname.
    """
    if not isinstance(url, str):
        raise ValueError("manifest url must be a string")
    parsed = urllib.parse.urlparse(url)
    if (parsed.scheme or "").lower() != "https":
        raise ValueError(f"only HTTPS URLs are allowed: {url!r}")
    if not parsed.hostname:
        raise ValueError(f"URL is missing a hostname: {url!r}")
    return parsed.hostname


def fetch_bytes(url: str) -> bytes:
    """Read a single artifact body over the transport of record.

    Late-bound on purpose: ``urllib.request.urlopen`` is read at call time,
    so monkeypatching the module attribute (as the pinned tests do) swaps
    the transport without touching this module.
    """
    response = urllib.request.urlopen(url)
    try:
        return response.read()
    finally:
        close = getattr(response, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass


def parse_manifest(manifest: Any) -> list:
    """Normalize a manifest into a list of package entry dicts.

    Accepts the project schema ``{"packages": [...]}`` (T-006 shares it) and,
    defensively, a bare list of entries. Anything else normalizes to an
    empty manifest.
    """
    if isinstance(manifest, dict):
        entries = manifest.get("packages", [])
    elif isinstance(manifest, list):
        entries = manifest
    else:
        return []
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def artifact_filename(entry: dict, url: str) -> str:
    """Derive a safe on-disk file name for one package entry.

    Prefers the manifest ``name`` (stripped to the bare file name) and falls
    back to the URL path basename, then to a constant.
    """
    candidate = entry.get("name")
    if isinstance(candidate, str) and candidate.strip():
        candidate = candidate.strip().replace("\\", "/").split("/")[-1]
        if candidate not in ("", ".", ".."):
            return candidate
    try:
        base = Path(urllib.parse.urlparse(url).path).name if \
            isinstance(url, str) else ""
    except Exception:
        base = ""
    return base or "download.bin"


def resolve_and_download(manifest: dict, target_dir: str) -> list:
    """Resolve manifest entries and install their artifacts locally.

    Args:
        manifest: project manifest, ``{"packages": [...]}``.
        target_dir: directory to place downloaded artifacts in.

    Returns:
        List of installed absolute file path strings.

    Raises:
        ValueError: for non-HTTPS URLs (raised before any network activity).
    """
    target = Path(target_dir)
    placed: list = []
    for entry in parse_manifest(manifest):
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        url = url.strip()
        # Validation precedes transport: a rejected scheme must never fetch.
        validate_url(url)
        payload = fetch_bytes(url)
        expected = str(entry.get("sha256") or "").strip().lower()
        if expected and hashlib.sha256(payload).hexdigest() != expected:
            # Digest mismatch: the artifact is excluded, never installed.
            continue
        name = artifact_filename(entry, url)
        target.mkdir(parents=True, exist_ok=True)
        destination = target / name
        destination.write_bytes(payload)
        placed.append(str(destination.resolve()))
    return placed
