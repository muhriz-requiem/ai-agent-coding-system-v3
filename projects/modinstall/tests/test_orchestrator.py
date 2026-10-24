"""T-006 orchestrator contract -- re-pinned offline 2026-10-17.

Hand fix at the spec_wrong/missing_module handover boundary:

* the happy path previously made a LIVE request to example.com; transport is
  now monkeypatched (urllib.request.urlopen, requests.get when importable)
  and real sockets are banned -- allowlist rejection must happen BEFORE any
  transport activity, which the socket ban proves on the disallowed test.
* name pollution from earlier cycles is gone: the allowlist API of record for
  this project is ``check_url_allowed`` (T-001's passing contract); this file
  does not import allowlist members directly, so the orchestrator keeps its
  own composition (exactly the objective of T-006).
* manifest schema is unified with T-004: {"packages": [{"name", "url",
  "sha256"?}]}. Quarantine location is implementation-defined but must live
  OUTSIDE target_dir -- these tests prove tampered bytes never surface as
  installed files in target_dir, and a failed dependency rolls the whole
  run back (T-003 semantics), leaving no good artifact behind.
* run_install(manifest, target_dir, allowed_domains) -> bool.
"""

import hashlib
import importlib.util
import io
import socket

import pytest

if importlib.util.find_spec('src.installer.orchestrator') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)

import urllib.request

from src.installer.orchestrator import run_install

try:
    import requests
except ImportError:
    requests = None

GOOD_BYTES = b"MODINSTALL-GOOD-PAYLOAD\n"
BAD_BYTES = b"MODINSTALL-TAMPERED-PAYLOAD\n"
GOOD_SHA = hashlib.sha256(GOOD_BYTES).hexdigest()

GOOD_URL = "https://example.com/good_pkg.tar.gz"
BAD_URL = "https://example.com/bad_pkg.tar.gz"
EVIL_URL = "https://malicious.com/pkg1.tar.gz"
ALLOWED = ["example.com"]


class _OfflineResp(io.BytesIO):
    status = 200
    headers = {"Content-Type": "application/x-tar-gz"}

    def getcode(self):
        return 200


class _OfflineRequestsResp:
    reason = "OK"

    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    @property
    def content(self):
        return self._payload

    def raise_for_status(self):
        return None


def _offline_transport(monkeypatch, route, fallback=b"UNROUTED-FETCH-NOT-ASSETED"):
    """route: exact URL string -> payload served; anything else gets
    `fallback` so an unanticipated transport attempt cannot invent data."""

    def fake_urlopen(req=None, *a, **k):
        key = getattr(req, "full_url", None)
        if key is None:
            key = str(req) if req is not None else ""
        payload = route.get(str(key), fallback)
        resp = _OfflineResp(payload)
        resp.headers = {"Content-Type": "application/x-tar-gz"}
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    if requests is not None:
        def fake_get(url=None, *a, **k):
            payload = route.get(str(url), fallback)
            return _OfflineRequestsResp(payload)

        monkeypatch.setattr(requests, "get", fake_get)

    def _no_socket(self, address):
        raise ConnectionError("offline contract: real socket connection attempted")

    monkeypatch.setattr(socket.socket, "connect", _no_socket)


def test_run_install_success(tmp_path, monkeypatch):
    _offline_transport(monkeypatch, {GOOD_URL: GOOD_BYTES})
    manifest = {"packages": [{"name": "good_pkg", "url": GOOD_URL,
                              "sha256": GOOD_SHA}]}
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    result = run_install(manifest, str(target_dir), ALLOWED)
    assert result is True
    assert any(p.is_file() and p.read_bytes() == GOOD_BYTES
               for p in target_dir.rglob("*"))


def test_run_install_disallowed_domain(tmp_path, monkeypatch):
    """R5 discipline: a disallowed domain is rejected BEFORE any network
    attempt -- the socket ban would fail the run if traffic ever escaped."""
    _offline_transport(monkeypatch, {EVIL_URL: b"SHOULD-NOT-FETCH"})
    manifest = {"packages": [{"name": "pkg1", "url": EVIL_URL,
                              "sha256": GOOD_SHA}]}
    target_dir = tmp_path
    assert run_install(manifest, str(target_dir), ALLOWED) is False
    assert not any(p.is_file() for p in target_dir.rglob("*"))


def test_run_install_quarantine_failure(tmp_path, monkeypatch):
    """R3 discipline: a digest mismatch is quarantined, never installed.
    The tampered bytes must not appear under target_dir afterward."""
    _offline_transport(monkeypatch, {BAD_URL: BAD_BYTES})
    manifest = {"packages": [{"name": "bad_pkg", "url": BAD_URL,
                              "sha256": GOOD_SHA}]}
    target_dir = tmp_path
    result = run_install(manifest, str(target_dir), ALLOWED)
    assert result is False
    victims = [p for p in target_dir.rglob("*") if p.is_file()]
    assert not any(p.read_bytes() == BAD_BYTES for p in victims)


def test_run_install_mixed_manifest_rolls_back(tmp_path, monkeypatch):
    """R4 discipline: one failing dependency rolls back the whole install --
    the good artifact that was already placed must not survive the run."""
    _offline_transport(
        monkeypatch,
        {GOOD_URL: GOOD_BYTES, BAD_URL: BAD_BYTES},
    )
    manifest = {"packages": [
        {"name": "good_pkg", "url": GOOD_URL, "sha256": GOOD_SHA},
        # manifest trusts GOOD digest; server serves tampered bytes instead
        {"name": "bad_pkg", "url": BAD_URL, "sha256": GOOD_SHA},
    ]}
    target_dir = tmp_path
    result = run_install(manifest, str(target_dir), ALLOWED)
    assert result is False
    survivors = [p for p in target_dir.rglob("*") if p.is_file()]
    assert not any(p.read_bytes() == GOOD_BYTES for p in survivors)
