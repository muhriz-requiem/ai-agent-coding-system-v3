"""Session-scoped action log for continuous (intraday) operation.

The one-shot pipeline is safe because every invocation is independent;
a ``session`` loop is not: without cross-cycle state, each cycle would
re-submit the same orders and independently consume the full equity
budget.  :class:`SessionStore` is that cross-cycle memory: an
append-only SQLite table of submitted actions per trade date, used by
the session loop to enforce

* per-ticker-per-day action budgets (default: 1 BUY + 1 SELL),
* a session-wide cumulative BUY notional cap (equity minus what this
  session already submitted today), and
* crash-safe resume: ``actions_for_date`` rebuilds today's state after
  a restart, so budgets are never reset by a new process.

Same database file as the decision audit log (``DECISIONS_DB``) but a
separate table; both are append-only.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS session_actions ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "trade_date TEXT NOT NULL, "
    "ts TEXT NOT NULL, "
    "ticker TEXT NOT NULL, "
    "side TEXT NOT NULL, "
    "notional REAL NOT NULL DEFAULT 0.0, "
    "kind TEXT NOT NULL DEFAULT 'cycle', "
    "order_id TEXT"
    ")"
)


class SessionStore:
    """Append-only per-day record of submitted session actions."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL + busy timeout (see decisions.py).

        The session and overnight loops share this database file; WAL
        keeps their append-only writes from blocking each other.  Pragma
        failures degrade to the default journal mode (fail-open).
        """
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
        except sqlite3.Error:
            pass
        return conn

    def log_action(
        self,
        trade_date: str,
        ticker: str,
        side: str,
        notional: float = 0.0,
        ts: str = "",
        kind: str = "cycle",
        order_id: Optional[str] = None,
    ) -> int:
        """Append one action; returns the row id."""
        if not ts:
            from datetime import datetime, timezone

            ts = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO session_actions "
                    "(trade_date, ts, ticker, side, notional, kind, order_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(trade_date),
                        str(ts),
                        str(ticker).upper(),
                        str(side).lower(),
                        float(notional),
                        str(kind),
                        order_id,
                    ),
                )
                return int(cursor.lastrowid)

    def actions_for_date(self, trade_date: str) -> List[Dict[str, Any]]:
        """All actions logged for ``trade_date`` (ISO date), oldest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, trade_date, ts, ticker, side, notional, kind, order_id "
                "FROM session_actions WHERE trade_date = ? ORDER BY id",
                (str(trade_date),),
            ).fetchall()
        return [
            {
                "id": row[0],
                "trade_date": row[1],
                "ts": row[2],
                "ticker": row[3],
                "side": row[4],
                "notional": float(row[5]),
                "kind": row[6],
                "order_id": row[7],
            }
            for row in rows
        ]

    def buy_notional_for_date(self, trade_date: str) -> float:
        """Total BUY notional submitted today (the session equity budget)."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(notional), 0.0) FROM session_actions "
                "WHERE trade_date = ? AND side = 'buy'",
                (str(trade_date),),
            ).fetchone()
        return float(row[0] or 0.0)

    def action_counts_for_date(self, trade_date: str) -> Dict[str, Dict[str, int]]:
        """Per-ticker per-side action counts: ``{TICKER: {"buy": n, "sell": m}}``."""
        counts: Dict[str, Dict[str, int]] = {}
        for action in self.actions_for_date(trade_date):
            ticker = action["ticker"]
            side = action["side"]
            bucket = counts.setdefault(ticker, {"buy": 0, "sell": 0})
            if side in bucket:
                bucket[side] += 1
        return counts


__all__ = ["SessionStore"]
