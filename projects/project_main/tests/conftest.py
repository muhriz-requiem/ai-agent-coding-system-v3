from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so tests can import src modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
"""Pytest configuration shared across the suite.

Puts the project root at the front of ``sys.path`` so tests can import the
``src`` package regardless of the invocation directory.

Tests-first workflow: each ``test_*.py`` file for an unimplemented module
skips itself at import time (see the guard at the top of every test file).
The moment the module is implemented its tests start running for real --
no manual bookkeeping anywhere.
"""


import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The whole suite runs offline: the global liquidity & fear regime fetch
# (14 yfinance series) is disabled unless a test explicitly re-enables it
# via monkeypatch.setenv("REGIME_ENABLED", "1").
os.environ.setdefault("REGIME_ENABLED", "0")
