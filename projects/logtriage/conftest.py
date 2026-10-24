"""Make the project root importable regardless of pytest's invocation cwd.

The tests guard themselves tests-first: each module does a
``find_spec`` probe and calls ``pytest.skip(allow_module_level=True)`` when its
implementation is absent.  That guard only ever fires when the project root is
on ``sys.path``.

``python -m pytest`` from this directory happens to work because Python puts
the CWD on ``sys.path``.  A bare ``pytest``, or an invocation from the workspace
root, does not -- and then *every* module skips and the suite reports
``6 skipped`` with exit code 0, which reads exactly like a healthy run while
verifying nothing.  ``executors.run_tests`` happens to catch it (pytest exits 5,
"no tests collected"), but a human running the suite by hand would not.

Two lines here remove the dependency on how pytest was invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
