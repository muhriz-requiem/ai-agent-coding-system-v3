"""Consumer module demonstrating cross-project reuse of shared_util.

project_beta imports ``normalize_token`` from project_alpha's
``shared_util`` module.  When project_alpha changes that module, the
IntegrationAgent detects the dependency via its import-scan heuristic.
"""

from __future__ import annotations

from src.shared_util import normalize_token


def consume(raw: str) -> str:
    """Normalize *raw* via ``shared_util.normalize_token``."""
    return normalize_token(raw)
