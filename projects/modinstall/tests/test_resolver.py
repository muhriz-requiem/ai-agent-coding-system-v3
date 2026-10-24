"""T-004 resolver contract -- re-pinned offline 2026-10-17.

Hand fix at the spec_wrong handover boundary: the originally authored happy
path made a LIVE request to https://example.com/pkg1.tar.gz (which is a 404),
so the acceptance was untestable and burned three Builder attempts before
escalating with DIAGNOSIS spec_wrong (0.98).

This file pins an offline contract instead:

* every transport is monkeypatched -- urllib.request.urlopen and requests.get
  (when importable) -- and real sockets are banned (any outbound connect
  raises). An implementation must drive the patched transports.
* the manifest schema is unified across the project with T-006:
  {"packages": [{"name", "url", "sha256"?}, ...]}.
* resolve_and_download returns a list of installed file path strings
  (matches the task objective: -> list[str]).
* shelf semantics across other tasks: T-002 owns the quarantine module;
  digest-mismatch handling is tested at the ORCHESTRATOR level (T-006),
  not here.
"""

import hashlib
import importlib.util
import io
import socket
from pathlib import Path

import pytest

if importlib.util.find_spec('src.installer.resolver') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)

import urllib.request

from src.installer.resolver import resolve_and_download

try:
    import requests
except ImportError:
    requests = None

PAYLOAD = b"MODINSTALL-GOOD-PAYLOAD\n"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _OfflineResp(io.BytesIO):
    """Round shape enough for urllib flows (read(), getcode(), headers)."""

    status = 200
    headers = {"Content-Type": "application/x-tar-gz"}

    def getcode(self):
        return 200

    def geturl(self):
        return "https://example.com/payload"


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


def _ban_real_sockets(monkeypatch):
    def _no_connect(self, address):
        raise ConnectionError("offline contract: real socket connection attempted")

    monkeypatch.setattr(socket.socket, "connect", _no_connect)


def _patch_transport(monkeypatch, served=PAYLOAD):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req=None, *a, **k: _OfflineResp(served))
    if requests is not None:
        monkeypatch.setattr(
            requests, "get", lambda url=None, *a, **k: _OfflineRequestsResp(served)
        )
    _ban_real_sockets(monkeypatch)


def test_resolve_and_download_happy_path(tmp_path, monkeypatch):
    """Matching sha256: download through the (patched) transport, place the
    exact bytes under target_dir, and return the installed path."""
    _patch_transport(monkeypatch)
    manifest = {
        "packages": [
            {"name": "pkg1", "url": "https://example.com/pkg1.tar.gz",
             "sha256": PAYLOAD_SHA}
        ]
    }
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    result = resolve_and_download(manifest, str(target_dir))
    assert isinstance(result, list) and len(result) == 1
    assert all(isinstance(p, str) for p in result)
    placed = Path(result[0]).resolve()
    assert placed.is_file()
    assert placed.read_bytes() == PAYLOAD
    assert placed.is_relative_to(target_dir.resolve())


def test_resolve_and_download_empty_manifest(tmp_path, monkeypatch):
    _patch_transport(monkeypatch, served=b"NEVER-FETCHED")
    result = resolve_and_download({"packages": []}, str(tmp_path))
    assert result == []


def test_resolve_and_download_invalid_scheme(tmp_path, monkeypatch):
    """Non-HTTPS is rejected in validation, before any transport activity
    (the socket ban proves zero outbound traffic on this path)."""
    _patch_transport(monkeypatch, served=b"NEVER-FETCHED")
    manifest = {
        "packages": [
            {"name": "pkg1", "url": "http://example.com/pkg1.tar.gz"}
        ]
    }
    with pytest.raises(ValueError):
        resolve_and_download(manifest, str(tmp_path))
