"""Shared token-normalization utility exported by project_alpha.

Other projects (e.g. project_beta) import ``normalize_token`` from this
module, which is what the IntegrationAgent's cross-project conflict
heuristic watches.
"""

from __future__ import annotations


def normalize_token(raw: str) -> str:
    """Return *raw* stripped of surrounding whitespace and lower-cased.

    Raises ``TypeError`` when *raw* is not a string so callers fail fast
    instead of silently normalizing numbers or ``None``.
    """
    if not isinstance(raw, str):
        raise TypeError("raw token must be a string")
    return raw.strip().lower()
