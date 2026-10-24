"""Append-only SQLite store for pipeline decisions.

Each decision dict is serialized to JSON and inserted as a single row in
the ``decisions`` table. Rows are never updated or deleted, making the
store strictly append-only.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS decisions ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "payload TEXT NOT NULL"
    ")"
)


class DecisionStore:
    """Persist decision dicts as JSON payloads in a SQLite database.

    The store is append-only: ``log_decision`` inserts new rows with
    autoincrement ids, and no update or delete operations are exposed.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        # Ensure the parent directory exists so relative or nested paths work.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent schema creation: repeated instantiation on the same
        # path must not fail or duplicate the table.
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection so no handles leak between calls.

        WAL journaling + a busy timeout let a session loop and an
        overnight loop share the same database file without tripping
        over each other's write locks (both stores are append-only, so
        WAL's single-writer model is enough).  Pragma failures (e.g.
        WAL on a network filesystem) degrade to the default journal
        mode.
        """
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
        except sqlite3.Error:
            pass
        return conn

    def log_decision(self, decision: dict) -> None:
        """Append ``decision`` as a new row, storing the whole dict as JSON."""
        payload = json.dumps(decision)
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO decisions (payload) VALUES (?)",
                    (payload,),
                )

    def log_pair_decision(self, decision: dict) -> None:
        """Append a pairs decision, injecting the ``kind: "pair"`` marker.

        The ``decisions`` table has no kind column -- the discriminator
        lives inside the JSON payload (the schema stays unchanged) so
        audit consumers can filter single-leg vs pair decisions without
        a migration.
        """
        payload = dict(decision) if isinstance(decision, dict) else {}
        payload.setdefault("kind", "pair")
        self.log_decision(payload)

    def tail(self, n: int) -> list[dict]:
        """Return the last ``n`` decisions oldest-first, JSON-decoded.

        An empty database yields ``[]``; if ``n`` exceeds the number of
        rows, all rows are returned in insertion order.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT payload FROM decisions ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        # Fetched newest-first for efficiency; reverse to oldest-first.
        return [json.loads(row[0]) for row in reversed(rows)]

    def count(self) -> int:
        """Return the total number of stored decisions."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return int(row[0])
