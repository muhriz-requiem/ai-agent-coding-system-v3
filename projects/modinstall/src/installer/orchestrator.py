"""T-006: install orchestrator composing the modinstall chain.

Contract (pinned by tests/test_orchestrator.py, offline):

* ``run_install(manifest, target_dir, allowed_domains) -> bool``
* every package URL is gated with T-001's ``check_url_allowed`` BEFORE any
  network activity; a disallowed domain short-circuits to False and leaves
  target_dir exactly as it was (the pinned tests ban real sockets to prove
  nothing ever gets fetched or written).
* a digest mismatch (T-002 semantics) is quarantined OUTSIDE target_dir, the
  run is abandoned, and the tampered bytes never surface under target_dir.
* one failing dependency rolls the whole run back: no artifact is made
  visible unless every package verified.
* placements finalize through T-003's ``InstallTransaction``: bytes are
  verified first, staged outside target_dir, and committed atomically only
  when every package verified.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import uuid
from pathlib import Path

import urllib.request

from src.installer.allowlist import check_url_allowed
from src.installer.resolver import artifact_filename, parse_manifest
from src.installer.transaction import InstallTransaction


def _quarantine_dir() -> Path:
    """A fresh quarantine directory guaranteed OUTSIDE target_dir.

    Uses the OS temp dir rather than target_dir's parent so the pinned
    assert (no tampered bytes under target_dir) holds even when target_dir
    is itself the tmp_path root in tests.
    """
    root = Path(tempfile.gettempdir()) / \
        f"modinstall-quarantine-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def quarantine_artifact(payload: bytes, name: str) -> str:
    """Park a rejected artifact's bytes outside any target directory.

    Quarantine mutates nothing in the install tree and only ever writes into
    the OS temp dir. Best-effort by design: quarantine failure must not
    raise out of the rollback path.
    """
    try:
        dest_dir = _quarantine_dir() / uuid.uuid4().hex[:8]
        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / name
        destination.write_bytes(payload)
        return str(destination.resolve())
    except Exception:
        return ""


def run_install(manifest: dict, target_dir: str, allowed_domains: list) -> bool:
    """Install all packages of ``manifest`` under ``target_dir``.

    Args:
        manifest: project manifest, ``{"packages": [...]}``.
        target_dir: directory every verified artifact is placed in.
        allowed_domains: domains permitted by the T-001 allowlist.

    Returns:
        True when every package downloaded, verified, and installed;
        False for an empty workflow artifact, a disallowed or malformed
        domain, a digest mismatch, or any transport/commit failure.
    """
    entries = parse_manifest(manifest)

    # R5: allowlist gate for every package BEFORE any transport activity.
    packages: list = []
    for entry in entries:
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            return False
        url = url.strip()
        try:
            allowed = check_url_allowed(url, list(allowed_domains or []))
        except ValueError:
            return False
        if not allowed:
            return False
        packages.append((url, entry))
    if not packages:
        # Nothing to install is a successful, untouched run.
        return True
    return _run_staged(packages, target_dir)


def _run_staged(packages: list, target_dir: str) -> bool:
    """Fetch, digest-verify, stage-outside-target, commit atomically."""
    stage_dir = Path(tempfile.mkdtemp(prefix="modinstall-stage-"))
    transaction = InstallTransaction()
    committed = False
    try:
        # Verified payload for each package KEEPING manifest order.
        verified: list = []
        for url, entry in packages:
            name = artifact_filename(entry, url)
            payload = urllib.request.urlopen(url).read()
            expected = str(entry.get("sha256") or "").strip().lower()
            if expected:
                actual = hashlib.sha256(payload).hexdigest()
                if actual != expected:
                    # R3: rejected bytes never touch target_dir.
                    quarantine_artifact(payload, name)
                    return False
            staged = stage_dir / name
            staged.write_bytes(payload)
            verified.append((name, staged))
        if not verified:
            return False
        # R4: nothing is visible under target_dir until EVERY package
        # verified; one commit finalizes all placements at once.
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        for name, src in verified:
            transaction.stage_copy(str(src), str(Path(target_dir) / name))
        transaction.commit()
        committed = True
        return True
    except Exception:
        return False
    finally:
        if not committed:
            try:
                transaction.rollback()
            except Exception:
                pass
        shutil.rmtree(stage_dir, ignore_errors=True)
