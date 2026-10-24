# Environmental fix for the failed T-002 (DIAGNOSIS: missing_module ->
# fix_env): both import styles must resolve — the tests use the
# `src.qreturns.*` spelling while the machine plan also names the bare
# `qreturns` package.  Adding src/ to sys.path makes both work without
# touching any contracted file.  Mirrors the boardscan/modinstall
# conftest role.
import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
