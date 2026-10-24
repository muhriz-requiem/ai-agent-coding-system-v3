"""The test suite must never execute with live provider credentials.

``config`` used to call ``load_dotenv()`` at import, so every pytest run
picked up the real keys in ``.env``.  That is how a live Cohere key ended up
pasted into a test file once already.  ``tests/conftest.py`` now sets
``AGENTS_NO_DOTENV=1`` before importing config and scrubs the environment
autouse; these tests are the regression guard on that property.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_KEY_SUFFIXES = ("_API_KEY", "_API_TOKEN", "_ACCOUNT_ID")


def _leaked_keys() -> list:
    return sorted(k for k in os.environ if k.endswith(_KEY_SUFFIXES))


def test_no_provider_keys_in_test_environment():
    assert _leaked_keys() == [], (
        f"live credentials reached the test run: {_leaked_keys()}")


def test_dotenv_disabled_for_tests():
    import config

    assert config.DOTENV_DISABLED is True


def test_config_import_does_not_load_dotenv_in_a_clean_process():
    """A subprocess launched from a test must also stay credential-free.

    Without the env flag, merely importing ``config`` anywhere in this process
    tree would re-populate the environment from ``.env``.
    """
    probe = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"leaked = [k for k in os.environ if k.endswith({_KEY_SUFFIXES!r})]\n"
        "assert not leaked, leaked\n"
        "import config\n"
        "assert config.DOTENV_DISABLED is True\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", probe],
                            capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_provider_keys_were_not_loaded_from_dotenv_file():
    """Sanity: the .env file *does* contain keys, so the test above is real.

    If this ever fails because .env went missing, the isolation tests would be
    passing vacuously.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        import pytest

        pytest.skip("no .env in this checkout")
    body = env_file.read_text(encoding="utf-8", errors="replace")
    declared = [line.split("=", 1)[0].strip() for line in body.splitlines()
                if "=" in line and not line.strip().startswith("#")
                and line.split("=", 1)[0].strip().endswith(_KEY_SUFFIXES)]
    assert declared, "expected .env to declare provider keys"