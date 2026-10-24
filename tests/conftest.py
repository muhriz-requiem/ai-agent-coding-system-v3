"""Shared pytest setup: isolated env + repo root on sys.path.

Two jobs, both about not letting a test touch production credentials.

``AGENTS_NO_DOTENV`` is set **before** ``config`` is imported, so importing the
harness never runs ``load_dotenv()``.  Without this, every pytest run executes
with the live provider keys from ``.env`` in ``os.environ`` -- which is how a
real Cohere key ended up pasted into a test file once already.

The autouse fixture below is the second line of defence and also runs for tests
that never opt into ``clean_env``.  It scrubs provider keys and routing toggles
from ``os.environ`` and resets the config module attributes that were derived
from them, so one test's provider selection cannot leak into the next.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Must precede any `import config` in this process.
os.environ["AGENTS_NO_DOTENV"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep synthetic test events out of the production run log.  Without this a
# test run interleaves thousands of rows into logs/agent_log.jsonl alongside
# real ones, which silently corrupts any audit of that file.
import logger  # noqa: E402

_TEST_LOG_DIR = ROOT / "logs" / "test"
logger.set_log_dir(_TEST_LOG_DIR)

PROVIDER_KEY_ENVS = (
    "OPENROUTER_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
    "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "ZAI_API_KEY",
    "NVIDIA_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
    "DASHSCOPE_API_KEY", "CEREBRAS_API_KEY", "LOCAL_LLM_API_KEY",
)

PROVIDER_TOGGLE_ENVS = (
    "PROVIDERS_ENABLED", "PROVIDERS_DISABLED", "LOCAL_LLM_ENABLED",
    "MODEL_OVERRIDES",
)

#: Config attributes that ``config`` derives from the environment at import.
#: Reset them so a test that flips e.g. ``DEMO_MODE`` cannot affect its peers.
_CONFIG_ATTRS = (
    "DEMO_MODE", "REAL_EXECUTION", "ENABLE_INTEGRATION_AGENT",
    "REDACTION_ENABLED", "SECRET_SCAN_ENABLED", "KAGGLE_ALLOW_ARBITRARY_RUN_COMMAND",
)


@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch):
    """Blank the credential environment for every test, opt-in or not."""
    for name in PROVIDER_KEY_ENVS + PROVIDER_TOGGLE_ENVS:
        monkeypatch.delenv(name, raising=False)
    import config

    saved = {attr: getattr(config, attr) for attr in _CONFIG_ATTRS
             if hasattr(config, attr)}
    try:
        yield
    finally:
        for attr, value in saved.items():
            setattr(config, attr, value)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every provider key/toggle so tests start from a blank slate.

    Retained for readability at call sites; ``_isolate_credentials`` now does
    this automatically for the whole suite.
    """
    for name in PROVIDER_KEY_ENVS + PROVIDER_TOGGLE_ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def restore_test_log_dir():
    """Hand a test the suite-wide test log dir and restore it on teardown.

    Tests that assert on what they log redirect the logger into their own
    ``tmp_path``.  They must restore it afterwards or every later test in the
    process inherits the redirection.

    This is a fixture rather than a plain ``from conftest import ...`` on
    purpose.  Four directories in this repo ship a module named ``conftest``,
    so a direct import binds to whichever one landed in ``sys.modules`` first
    -- which silently pointed the core suite's redaction tests at
    ``projects/project_main/tests/conftest.py`` and made the whole core suite
    uncollectable.  Fixtures resolve through pytest's own registry, not
    ``sys.modules``, so they cannot collide.
    """
    yield _TEST_LOG_DIR
    logger.set_log_dir(_TEST_LOG_DIR)