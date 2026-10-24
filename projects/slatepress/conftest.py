"""Put the project root on sys.path.

The tests guard themselves tests-first with a find_spec probe and a
module-level skip, which only fires correctly when the project root is
importable. ``python -m pytest`` happens to work because Python puts the CWD on
sys.path; a bare ``pytest``, or any invocation from elsewhere, does not -- and then every
module skips and the suite reports "N skipped" with exit code 0, which
reads exactly like a healthy run while verifying nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
