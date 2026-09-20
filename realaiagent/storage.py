"""Persistent storage + the universal usage ledger.

One SQLite database (stdlib ``sqlite3``) holds all durable state:
API keys, conversations, memory, goals, skills, permissions, pending
permission requests, devices, the Q-learning table, the event log, and
the usage ledger.

"Everything is counted": every meaningful thing the agent does calls
:meth:`Storage.count`, which (a) increments an in-memory counter and
(b) appends a durable row to the ``usage`` table. Counters are rebuilt
from the ledger on startup, so counts survive restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional


def _now() -> float:
    return time.time()


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    scopes TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    last_used REAL,
    revoked INTEGER NOT NULL DEFAULT 0,
    request_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    key_id TEXT,
    category TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_cat ON usage(category, event);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    sender TEXT,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    intent TEXT,
    confidence REAL,
    plan TEXT,
    actions TEXT,
    success INTEGER,
    duration_ms REAL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(kind, key)
);
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    source TEXT NOT NULL DEFAULT 'owner',
    steps TEXT,
    steps_done INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL
);
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    description TEXT,
    steps TEXT NOT NULL,
    examples TEXT,
    success_count INTEGER NOT NULL DEFAULT 0,
    fail_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS permissions (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    pattern TEXT,
    policy TEXT NOT NULL,
    note TEXT,
    created_by TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    category TEXT NOT NULL,
    action TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    requested_by TEXT,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    decided_by TEXT,
    decided_at REAL
);
CREATE TABLE IF NOT EXISTS devices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'generic',
    capabilities TEXT NOT NULL DEFAULT '[]',
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    last_seen REAL
);
CREATE TABLE IF NOT EXISTS qtable (
    state TEXT NOT NULL,
    action TEXT NOT NULL,
    q REAL NOT NULL DEFAULT 0.0,
    visits INTEGER NOT NULL DEFAULT 0,
    last_update REAL NOT NULL,
    PRIMARY KEY (state, action)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Storage:
    """Thread-safe SQLite store with the universal counting ledger."""

    def __init__(self, db_path: Any) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()
        # In-memory counters: {"category": {"event": n}} and a grand total.
        self._counters: Dict[str, Dict[str, int]] = {}
        self._total = 0
        self._rebuild_counters()

    # ------------------------------------------------------------------ core

    def execute(self, sql: str, params: tuple = ()) -> "sqlite3.Cursor":
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----------------------------------------------------------------- state

    def state_get(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM state WHERE key=?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (ValueError, TypeError):
            return row["value"]

    def state_set(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    # --------------------------------------------------------------- counting

    def count(self, category: str, event: str, key_id: Optional[str] = None,
              detail: Optional[Dict[str, Any]] = None) -> None:
        """Count one thing. Durable + in-memory. Never raises."""
        try:
            self.execute(
                "INSERT INTO usage(ts, key_id, category, event, detail) "
                "VALUES(?,?,?,?,?)",
                (_now(), key_id, category, event,
                 json.dumps(detail) if detail else None),
            )
        except sqlite3.Error:
            return
        with self._lock:
            self._total += 1
            cat = self._counters.setdefault(category, {})
            cat[event] = cat.get(event, 0) + 1

    def get_counters(self) -> Dict[str, Any]:
        """Full counter snapshot (nested by category, plus totals)."""
        with self._lock:
            out: Dict[str, Any] = {
                cat: dict(events) for cat, events in self._counters.items()
            }
            total = self._total
        cat_totals = {}
        for cat, events in out.items():
            cat_totals[cat] = sum(events.values())
        out["totals"] = dict(cat_totals)
        out["grand_total"] = total
        return out

    def rebuild_counters(self) -> None:
        self._rebuild_counters()

    def _rebuild_counters(self) -> None:
        rows = self.query("SELECT category, event, COUNT(*) n FROM usage "
                          "GROUP BY category, event")
        counters: Dict[str, Dict[str, int]] = {}
        total = 0
        for r in rows:
            counters.setdefault(r["category"], {})[r["event"]] = r["n"]
            total += r["n"]
        with self._lock:
            self._counters = counters
            self._total = total

    # ---------------------------------------------------------------- events

    def log_event(self, etype: str, payload: Optional[Dict[str, Any]] = None,
                  key_id: Optional[str] = None) -> None:
        self.execute(
            "INSERT INTO events(ts, type, payload) VALUES(?,?,?)",
            (_now(), etype, json.dumps(payload or {})),
        )
        self.count("events", etype, key_id=key_id)

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.query(
            "SELECT ts, type, payload FROM events ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        )
        out = []
        for r in rows:
            try:
                payload = json.loads(r["payload"])
            except (ValueError, TypeError):
                payload = {"raw": r["payload"]}
            out.append({"ts": r["ts"], "type": r["type"], "payload": payload})
        return out

    # ------------------------------------------------------------------ keys

    def key_insert(self, key_id: str, name: str, key_hash: str, scopes: List[str]) -> None:
        self.execute(
            "INSERT INTO keys(id, name, key_hash, scopes, created_at) "
            "VALUES(?,?,?,?,?)",
            (key_id, name, key_hash, json.dumps(scopes), _now()),
        )

    def key_get_by_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        row = self.query_one(
            "SELECT * FROM keys WHERE key_hash=? AND revoked=0", (key_hash,))
        if row:
            row["scopes"] = json.loads(row.get("scopes") or "[]")
        return row

    def key_list(self) -> List[Dict[str, Any]]:
        rows = self.query("SELECT * FROM keys ORDER BY created_at DESC")
        for row in rows:
            row["scopes"] = json.loads(row.get("scopes") or "[]")
            row.pop("key_hash", None)
        return rows

    def key_revoke(self, key_id: str) -> bool:
        cur = self.execute("UPDATE keys SET revoked=1 WHERE id=?", (key_id,))
        return cur.rowcount > 0

    def key_touch(self, key_id: str) -> None:
        self.execute(
            "UPDATE keys SET last_used=?, request_count=request_count+1 "
            "WHERE id=?", (_now(), key_id),
        )

    # ------------------------------------------------------------ usage query

    def usage_breakdown(self, by: str = "category") -> List[Dict[str, Any]]:
        if by == "key":
            rows = self.query(
                "SELECT COALESCE(k.name, 'unknown') name, k.id key_id, "
                "COUNT(*) n FROM usage u LEFT JOIN keys k ON k.id=u.key_id "
                "GROUP BY u.key_id ORDER BY n DESC")
        elif by == "day":
            rows = self.query(
                "SELECT CAST(ts/86400 AS INTEGER) day, COUNT(*) n FROM usage "
                "GROUP BY day ORDER BY day DESC LIMIT 14")
        else:
            rows = self.query(
                "SELECT category, event, COUNT(*) n FROM usage "
                "GROUP BY category, event ORDER BY n DESC")
        return rows
