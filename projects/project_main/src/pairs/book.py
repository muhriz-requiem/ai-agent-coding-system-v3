"""Persisted pair book: formation stats that survive restarts.

The single source of truth for live pair-trading formation state.  A
formation run replaces the whole book atomically; between formations
the book is reused as-is so signals stay stable across days and the
process can crash/restart without losing the spread statistics the
z-scores depend on.

Degradation contract (fail-safe for pairs, invisible to the
single-leg pipeline):
  * missing / corrupt / stale book -> no pair signals -> no pair trades
  * a book file is JSON; writes are atomic (tmp file + os.replace)
  * ``formation_due`` treats any unparsable state as due (re-form)

Pure module: the only I/O is the book file itself.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from src.pairs.cointegration import PairCandidate, spread_residuals

__all__ = ["BookEntry", "PairBook", "entry_from_candidate", "parse_iso_date",
           "pairs_book_path"]


def parse_iso_date(value: Any) -> Optional[date]:
    """Parse an ISO date string; ``date``/``datetime`` pass through."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except (ValueError, TypeError):
        return None


@dataclass
class BookEntry:
    """One active pair with its formation-window statistics.

    ``hedge_ratio``/``alpha`` parameterize the log-price spread
    ``log(P_A) - alpha - beta * log(P_B)``; ``spread_mean``/``spread_std``
    are the formation residual mean/std (z-score normalization);
    ``spread_daily_vol`` is the formation daily std of spread changes
    (pairs-side volatility scaling input, persisted so signal generation
    never re-fits); ``formation_corr`` anchors the correlation-breakdown
    veto; ``formation_adv`` is the per-leg dollar ADV at formation
    (absent leg -> ``None`` -> the liquidity gate fails open);
    ``half_life`` anchors the half-life-drift veto.
    """

    pair_id: str
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    alpha: float
    spread_mean: float
    spread_std: float
    half_life: float
    formed_on: str
    spread_daily_vol: Optional[float] = None
    formation_corr: Optional[float] = None
    formation_adv: dict[str, Optional[float]] = field(default_factory=dict)
    sector_a: Optional[str] = None
    sector_b: Optional[str] = None
    adf_pvalue: Optional[float] = None


class PairBook:
    """Load / save / query the active pair book (JSON file).

    Attributes:
        path: Book file path (``PAIR_BOOK_PATH``).
        entries: Mapping ``pair_id -> BookEntry``.
        formed_on: ISO date of the last formation (``None`` when empty).
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, BookEntry] = {}
        self.formed_on: Optional[str] = None

    # -- loading / saving ---------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "PairBook":
        """Load the book from disk; missing/corrupt file -> empty book."""
        book = cls(path)
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - missing/corrupt -> empty book
            return book
        if not isinstance(raw, dict):
            return book
        formed = raw.get("formed_on")
        book.formed_on = formed if isinstance(formed, str) and formed else None
        entries = raw.get("entries")
        if isinstance(entries, dict):
            for pair_id, payload in entries.items():
                entry = _entry_from_dict(pair_id, payload)
                if entry is not None:
                    book.entries[entry.pair_id] = entry
        return book

    def save(self) -> None:
        """Atomically persist the book (tmp file + ``os.replace``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "formed_on": self.formed_on,
            "entries": {pid: asdict(entry) for pid, entry in self.entries.items()},
        }
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp_name, self.path)
        except Exception:  # noqa: BLE001 - never leave a tmp file behind
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # -- formation schedule ---------------------------------------------------

    def formation_due(self, as_of: Any, rebalance_days: int = 21) -> bool:
        """Whether a (re-)formation is due on ``as_of``.

        Due when the book is empty, the last formation date is missing
        or unparsable (corrupt state -> re-form), or ``rebalance_days``
        or more trading-calendar days have elapsed since ``formed_on``
        (calendar-day arithmetic; monthly cadence tolerance).  A future
        ``formed_on`` (clock skew) is NOT due.
        """
        try:
            interval = int(rebalance_days)
        except (TypeError, ValueError):
            interval = 21
        if interval < 0:
            interval = 0
        if not self.entries:
            return True
        formed = parse_iso_date(self.formed_on)
        anchor = _as_date(as_of)
        if formed is None or anchor is None:
            return True
        return anchor >= formed + timedelta(days=interval)

    # -- book contents --------------------------------------------------------

    def replace(
        self,
        entries: list[BookEntry],
        as_of: Any,
    ) -> None:
        """Swap in a freshly formed book (atomic in-memory, then save)."""
        rebuilt: dict[str, BookEntry] = {}
        for entry in entries or []:
            if entry is None or not getattr(entry, "pair_id", ""):
                continue
            rebuilt[entry.pair_id] = entry
        anchor = _as_date(as_of)
        self.entries = rebuilt
        self.formed_on = anchor.isoformat() if anchor is not None else None
        self.save()

    def get(self, pair_id: str) -> Optional[BookEntry]:
        return self.entries.get(pair_id)

    def tickers(self) -> list[str]:
        """Sorted unique tickers across all active pairs."""
        names: set[str] = set()
        for entry in self.entries.values():
            if entry.ticker_a:
                names.add(entry.ticker_a)
            if entry.ticker_b:
                names.add(entry.ticker_b)
        return sorted(names)

    def __len__(self) -> int:
        return len(self.entries)


def _as_date(value: Any) -> Optional[date]:
    """Accept ISO strings / date / datetime and reduce to ``date``."""
    parsed = parse_iso_date(value)
    if parsed is not None:
        return parsed
    try:
        # Date-like objects (e.g. pandas Timestamps) reduce to date.
        if hasattr(value, "date") and callable(getattr(value, "date")):
            return value.date()
    except Exception:  # noqa: BLE001 - junk -> None
        return None
    return None


def _entry_from_dict(pair_id: Any, payload: Any) -> Optional[BookEntry]:
    """Rebuild one BookEntry from its JSON dict; ``None`` when junk."""
    if not isinstance(payload, dict):
        return None
    try:
        hedge_ratio = float(payload.get("hedge_ratio"))
        half_life = float(payload.get("half_life"))
        alpha = float(payload.get("alpha", 0.0))
        spread_mean = float(payload.get("spread_mean", 0.0))
        spread_std = float(payload.get("spread_std", 0.0))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (hedge_ratio, half_life, alpha,
                                          spread_mean, spread_std)):
        return None
    ticker_a = str(payload.get("ticker_a") or "").strip().upper()
    ticker_b = str(payload.get("ticker_b") or "").strip().upper()
    if not ticker_a or not ticker_b:
        return None
    formed = payload.get("formed_on")
    spread_vol = payload.get("spread_daily_vol")
    try:
        spread_vol = float(spread_vol) if spread_vol is not None else None
    except (TypeError, ValueError):
        spread_vol = None
    corr = payload.get("formation_corr")
    try:
        corr = float(corr) if corr is not None else None
    except (TypeError, ValueError):
        corr = None
    adv_raw = payload.get("formation_adv")
    adv: dict[str, Optional[float]] = {}
    if isinstance(adv_raw, dict):
        for leg, value in adv_raw.items():
            try:
                adv[str(leg)] = float(value) if value is not None else None
            except (TypeError, ValueError):
                adv[str(leg)] = None
    adf = payload.get("adf_pvalue")
    try:
        adf = float(adf) if adf is not None else None
    except (TypeError, ValueError):
        adf = None
    return BookEntry(
        pair_id=str(pair_id or payload.get("pair_id") or f"{ticker_a}/{ticker_b}"),
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        hedge_ratio=hedge_ratio,
        alpha=alpha,
        spread_mean=spread_mean,
        spread_std=spread_std,
        half_life=half_life,
        formed_on=str(formed) if formed else "",
        spread_daily_vol=spread_vol if (spread_vol is not None and math.isfinite(spread_vol) and spread_vol > 0) else None,
        formation_corr=corr if (corr is not None and math.isfinite(corr)) else None,
        formation_adv=adv,
        sector_a=payload.get("sector_a") if isinstance(payload.get("sector_a"), str) else None,
        sector_b=payload.get("sector_b") if isinstance(payload.get("sector_b"), str) else None,
        adf_pvalue=adf,
    )


def entry_from_candidate(
    candidate: PairCandidate,
    closes: Optional[dict[str, Any]] = None,
    advs: Optional[dict[str, float]] = None,
) -> BookEntry:
    """Freeze a discovery candidate into a persistent book entry.

    ``spread_daily_vol`` (the pairs-side volatility-scaling input) is
    computed from the formation residual first differences when the
    formation closes are supplied; ``formation_adv`` rides along when a
    dollar-ADV mapping is supplied (absent values -> the liquidity gate
    fails open).  The returned entry is pure data -- the caller assigns
    ``formed_on`` via :meth:`PairBook.replace`.
    """
    closes = closes or {}
    spread_vol: Optional[float] = None
    try:
        residuals = spread_residuals(
            candidate,
            closes.get(candidate.ticker_a),
            closes.get(candidate.ticker_b),
        )
        if residuals is not None and len(residuals) >= 2:
            diffs = residuals.diff().dropna()
            if len(diffs):
                value = float(diffs.std())
                if math.isfinite(value) and value > 0.0:
                    spread_vol = value
    except Exception:  # noqa: BLE001 - advisory stats only
        spread_vol = None
    advs = advs or {}

    def _adv(ticker: str) -> Optional[float]:
        value = advs.get(ticker)
        try:
            adv = float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
        return adv if adv is not None and math.isfinite(adv) and adv > 0.0 else None

    return BookEntry(
        pair_id=candidate.pair_id,
        ticker_a=candidate.ticker_a,
        ticker_b=candidate.ticker_b,
        hedge_ratio=candidate.hedge_ratio,
        alpha=candidate.alpha,
        spread_mean=candidate.spread_mean,
        spread_std=candidate.spread_std,
        half_life=candidate.half_life,
        formed_on="",
        spread_daily_vol=spread_vol,
        formation_corr=candidate.corr,
        formation_adv={
            candidate.ticker_a: _adv(candidate.ticker_a),
            candidate.ticker_b: _adv(candidate.ticker_b),
        },
        sector_a=candidate.sector_a,
        sector_b=candidate.sector_b,
        adf_pvalue=candidate.adf_pvalue,
    )


def pairs_book_path(default: str = "data/pairs_active.json") -> Path:
    """Resolve the book path from ``PAIR_BOOK_PATH`` (env) or default."""
    return Path(os.environ.get("PAIR_BOOK_PATH", default))
