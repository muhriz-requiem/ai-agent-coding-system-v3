"""
Quarantine verification utilities.
"""

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

def verify_artifact(file_path: str, expected_sha256: str) -> bool:
    """
    Verify that the file's SHA-256 digest matches the expected value.

    Args:
        file_path: Path to the file to verify.
        expected_sha256: Expected SHA-256 digest (hex string, case‑insensitive).

    Returns:
        True if the digests match, False otherwise or on any read error.
        Errors are logged but not raised.
    """
    if not expected_sha256:
        logger.error("Empty expected SHA‑256 provided for verification.")
        return False

    expected = expected_sha256.strip().lower()
    # Basic sanity check for hex length (SHA‑256 is 64 hex chars)
    if len(expected) != 64 or not all(c in "0123456789abcdef" for c in expected):
        logger.warning(f"Expected SHA‑256 appears malformed: {expected_sha256}")

    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return False
    except PermissionError:
        logger.error(f"Permission denied while reading: {file_path}")
        return False
    except OSError as e:
        logger.error(f"OS error reading {file_path}: {e}")
        return False

    computed = sha256.hexdigest().lower()
    if computed == expected:
        return True
    else:
        logger.error(f"Hash mismatch for {file_path}: expected {expected_sha256}, got {computed}")
        return False
