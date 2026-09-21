"""Turso / libSQL **SQL-over-HTTP** driver — stdlib only.

Turso databases are SQLite (libSQL) served over HTTPS. Their
``POST /v2/pipeline`` endpoint (the *Hrana over HTTP* protocol) takes a
list of statements with typed positional args and returns typed rows,
``affected_row_count`` and ``last_insert_rowid``. That is everything the
rest of RealAI already asks of ``sqlite3`` — so this module exposes a
tiny connection object with the same surface the :class:`Storage`
class uses (``execute`` → cursor with ``.lastrowid`` / ``.rowcount`` /
``.fetchall()``), and Storage picks it instead of ``sqlite3`` when
``REALAI_DATABASE_URL`` is set.

Why this exists: on Vercel the filesystem is ephemeral, so a SQLite file
in ``/tmp`` is lost on every cold start — owner key, users, memories,
the trained intent model, everything. Turso keeps that state durable
across cold starts and across regions, with a free tier, and this
driver needs **no package** (``libsql-client`` etc. are not required),
keeping the project's zero-dependency rule.

Protocol reference: https://docs.turso.tech/sdk/http/reference
"""

from __future__ import annotations

import base64
import json
import re
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class LibsqlError(Exception):
    """Raised when the remote database rejects a statement."""


def normalize_url(url: str) -> str:
    """``libsql://`` / ``turso://`` → ``https://`` (what the HTTP API wants)."""
    url = url.strip()
    for scheme in ("libsql://", "turso://", "wss://", "ws://"):
        if url.startswith(scheme):
            url = "https://" + url[len(scheme):]
    return url.rstrip("/")


def _encode_arg(v: Any) -> Dict[str, Any]:
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    if isinstance(v, (bytes, bytearray, memoryview)):
        return {"type": "blob", "base64": base64.b64encode(bytes(v)).decode()}
    return {"type": "text", "value": str(v)}


def _decode_cell(cell: Dict[str, Any]) -> Any:
    t = cell.get("type")
    if t == "null":
        return None
    if t == "integer":
        return int(cell["value"])
    if t == "float":
        return float(cell["value"])
    if t == "blob":
        return base64.b64decode(cell.get("base64", ""))
    return cell.get("value")


class Row(dict):
    """A dict row that also supports positional access like ``sqlite3.Row``."""

    def __init__(self, cols: Sequence[str], values: Sequence[Any]) -> None:
        super().__init__(zip(cols, values))
        self._values = list(values)

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)

    def keys(self):  # noqa: D102 - dict API
        return super().keys()


class Cursor:
    """The subset of ``sqlite3.Cursor`` that Storage and its callers use."""

    def __init__(self, cols: List[str], rows: List[Row], rowcount: int,
                 lastrowid: Optional[int]) -> None:
        self._cols = cols
        self._rows = rows
        self.rowcount = rowcount
        self.lastrowid = lastrowid
        self.description = [(c, None, None, None, None, None, None) for c in cols]

    def fetchall(self) -> List[Row]:
        return list(self._rows)

    def fetchone(self) -> Optional[Row]:
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


_SPLIT_RE = re.compile(r";\s*(?:\n|$)")


class Connection:
    """One logical connection to a Turso database over HTTP.

    Every :meth:`execute` is a single stateless pipeline request (statement
    + ``close``), so a serverless function never leaks a server-side
    connection. Multi-statement :meth:`executescript` runs the statements
    in one pipeline (one round trip).
    """

    def __init__(self, url: str, auth_token: str = "", timeout: float = 15.0) -> None:
        self.url = normalize_url(url)
        self.token = auth_token.strip()
        self.timeout = timeout
        self.row_factory: Any = None     # accepted for sqlite3 parity, ignored
        self._lock = threading.Lock()

    # ---------------------------------------------------------- transport
    def _pipeline(self, statements: Iterable[Tuple[str, Sequence[Any]]]) -> List[Dict[str, Any]]:
        requests = [{"type": "execute",
                     "stmt": {"sql": sql, "args": [_encode_arg(a) for a in args]}}
                    for sql, args in statements]
        requests.append({"type": "close"})
        body = json.dumps({"requests": requests}).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(f"{self.url}/v2/pipeline", data=body,
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise LibsqlError(f"database HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LibsqlError(f"database unreachable: {exc.reason}") from exc
        results = data.get("results") or []
        out: List[Dict[str, Any]] = []
        for res in results[:-1]:          # drop the trailing close
            if res.get("type") == "error":
                err = res.get("error") or {}
                raise LibsqlError(err.get("message") or str(err))
            out.append((res.get("response") or {}).get("result") or {})
        return out

    @staticmethod
    def _cursor(result: Dict[str, Any]) -> Cursor:
        cols = [c.get("name") or f"col{i}" for i, c in enumerate(result.get("cols") or [])]
        rows = [Row(cols, [_decode_cell(c) for c in r]) for r in result.get("rows") or []]
        last = result.get("last_insert_rowid")
        return Cursor(cols, rows, int(result.get("affected_row_count") or 0),
                      int(last) if last is not None else None)

    # -------------------------------------------------------- sqlite3 API
    def execute(self, sql: str, params: Sequence[Any] = ()) -> Cursor:
        sql = sql.strip()
        if sql.upper().startswith("PRAGMA"):
            return self._pragma(sql)
        with self._lock:
            result = self._pipeline([(sql, tuple(params))])[0]
        return self._cursor(result)

    def executescript(self, script: str) -> None:
        stmts = [s.strip() for s in _SPLIT_RE.split(script) if s.strip()]
        stmts = [s for s in stmts if not s.upper().startswith("PRAGMA")]
        if not stmts:
            return
        with self._lock:
            # chunk so one huge schema never exceeds request limits
            for i in range(0, len(stmts), 25):
                self._pipeline([(s, ()) for s in stmts[i:i + 25]])

    def _pragma(self, sql: str) -> Cursor:
        """The only PRAGMA Storage relies on is ``table_info`` (migrations)."""
        m = re.match(r"PRAGMA\s+table_info\((\w+)\)", sql, re.IGNORECASE)
        if m:
            with self._lock:
                result = self._pipeline([(
                    "SELECT cid, name, type, \"notnull\", dflt_value, pk "
                    f"FROM pragma_table_info('{m.group(1)}')", ())])[0]
            return self._cursor(result)
        return Cursor([], [], 0, None)      # journal_mode etc.: no-ops remotely

    def commit(self) -> None:      # autocommit over HTTP
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass

    def ping(self) -> bool:
        try:
            self.execute("SELECT 1")
            return True
        except LibsqlError:
            return False


def connect(url: str, auth_token: str = "", timeout: float = 15.0) -> Connection:
    return Connection(url, auth_token, timeout)
