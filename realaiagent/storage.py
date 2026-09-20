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
from typing import Any, Callable, Dict, List, Optional


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
    request_count INTEGER NOT NULL DEFAULT 0,
    user TEXT,
    request_limit INTEGER
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    is_vip INTEGER NOT NULL DEFAULT 0,
    balance REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    last_active REAL
);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
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
        # Live observers (e.g. the 0.3.0 SSE hub). Each hook is called as
        # hook(source, name, data) with source in {"count", "event"};
        # name is the category (count) or event type (log_event).
        # Hooks must be cheap and never block; errors are swallowed.
        self.hooks: List[Callable[[str, str, Dict[str, Any]], None]] = []
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns to tables created by older versions (idempotent)."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(keys)")}
        if "user" not in cols:
            self._conn.execute("ALTER TABLE keys ADD COLUMN user TEXT")
        if "request_limit" not in cols:
            self._conn.execute("ALTER TABLE keys ADD COLUMN request_limit INTEGER")
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
        self._fire_hooks("count", category, {"category": category,
                                             "event": event,
                                             "detail": detail})

    def _fire_hooks(self, source: str, name: str,
                    data: Dict[str, Any]) -> None:
        if not self.hooks:
            return
        for hook in list(self.hooks):
            try:
                hook(source, name, data)
            except Exception:  # noqa: BLE001 - observers must not break count
                pass

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
        self._fire_hooks("event", etype, payload or {})

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

    def key_insert(self, key_id: str, name: str, key_hash: str, scopes: List[str],
                   user: Optional[str] = None,
                   request_limit: Optional[int] = None) -> None:
        self.execute(
            "INSERT INTO keys(id, name, key_hash, scopes, created_at, user, "
            "request_limit) VALUES(?,?,?,?,?,?,?)",
            (key_id, name, key_hash, json.dumps(scopes), _now(), user,
             request_limit),
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
            "UPDATE keys SET last_used=? WHERE id=?", (_now(), key_id),
        )

    def key_bump_usage(self, key_id: str) -> int:
        """Increment request_count; return the new value."""
        self.execute(
            "UPDATE keys SET request_count=request_count+1, last_used=? "
            "WHERE id=?", (_now(), key_id))
        row = self.query_one("SELECT request_count FROM keys WHERE id=?",
                             (key_id,))
        return int(row["request_count"]) if row else 0

    def key_set_limit(self, key_id: str, limit: Optional[int]) -> None:
        self.execute("UPDATE keys SET request_limit=? WHERE id=?",
                     (limit, key_id))

    # ----------------------------------------------------------------- users

    def user_request(self, username: str) -> Dict[str, Any]:
        """Create (or re-request) a user account as PENDING."""
        username = username.strip()
        existing = self.query_one("SELECT * FROM users WHERE username=?",
                                  (username,))
        if existing:
            self.execute(
                "UPDATE users SET status='PENDING' WHERE username=?",
                (username,))
            self.count("users", "re_requested")
            return self.user_get(username) or {}
        self.execute(
            "INSERT INTO users(username, status, created_at) VALUES(?,?,?)",
            (username, "PENDING", _now()))
        self.count("users", "requested")
        self.log_event("user_requested", {"username": username})
        return self.user_get(username) or {}

    def user_get(self, username: str) -> Optional[Dict[str, Any]]:
        row = self.query_one("SELECT * FROM users WHERE username=?",
                             (username,))
        if row:
            row["is_vip"] = bool(row["is_vip"])
        return row

    def user_list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            rows = self.query(
                "SELECT * FROM users WHERE status=? ORDER BY created_at DESC",
                (status,))
        else:
            rows = self.query("SELECT * FROM users ORDER BY created_at DESC")
        for r in rows:
            r["is_vip"] = bool(r["is_vip"])
        return rows

    def user_set_status(self, username: str, status: str) -> Optional[Dict[str, Any]]:
        cur = self.execute(
            "UPDATE users SET status=? WHERE username=?", (status, username))
        if cur.rowcount:
            self.count("users", f"status_{status.lower()}")
            self.log_event("user_status", {"username": username,
                                           "status": status})
        return self.user_get(username)

    def user_set_vip(self, username: str, vip: bool) -> Optional[Dict[str, Any]]:
        self.execute("UPDATE users SET is_vip=? WHERE username=?",
                     (1 if vip else 0, username))
        self.count("users", "vip_granted" if vip else "vip_revoked")
        return self.user_get(username)

    def user_add_balance(self, username: str, amount: float) -> Optional[Dict[str, Any]]:
        self.execute(
            "UPDATE users SET balance=ROUND(balance+?, 6) WHERE username=?",
            (amount, username))
        self.count("billing", "topup")
        self.log_event("billing_topup", {"username": username,
                                         "amount": amount})
        return self.user_get(username)

    def user_charge(self, username: str, amount: float) -> Optional[float]:
        """Atomically deduct `amount`; return new balance or None."""
        self.execute(
            "UPDATE users SET balance=ROUND(balance-?, 6) WHERE username=?",
            (amount, username))
        row = self.query_one("SELECT balance FROM users WHERE username=?",
                             (username,))
        return float(row["balance"]) if row else None

    def user_touch(self, username: str) -> None:
        self.execute("UPDATE users SET last_active=? WHERE username=?",
                     (_now(), username))

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
