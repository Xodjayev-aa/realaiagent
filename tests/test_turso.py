"""The Turso / libSQL-over-HTTP path.

A local HTTP server implements the Hrana ``/v2/pipeline`` contract on top
of a real SQLite file (typed args, typed cells, affected_row_count,
last_insert_rowid, error results, Bearer auth). The whole agent — keys,
users, memory, learning, chat, web layer — is then driven through the
remote driver, so every SQL statement in the codebase is proven to work
over HTTP, not just the driver in isolation.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from realaiagent.config import Config
from realaiagent.engine import Agent
from realaiagent.libsql_http import LibsqlError, connect, normalize_url
from realaiagent.storage import Storage

TOKEN = "test-token"


def _typed(v):
    if v is None:
        return {"type": "null"}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    if isinstance(v, bytes):
        return {"type": "blob", "base64": base64.b64encode(v).decode()}
    return {"type": "text", "value": str(v)}


def _untyped(a):
    t = a["type"]
    if t == "null":
        return None
    if t == "integer":
        return int(a["value"])
    if t == "float":
        return float(a["value"])
    if t == "blob":
        return base64.b64decode(a["base64"])
    return a["value"]


class FakeTurso(BaseHTTPRequestHandler):
    db_path = ""
    lock = threading.Lock()
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            return self._send(401, {"error": "unauthorized"})
        if self.path != "/v2/pipeline":
            return self._send(404, {"error": "no"})
        req = json.loads(raw)
        results = []
        with FakeTurso.lock:
            FakeTurso.calls += 1
            conn = sqlite3.connect(FakeTurso.db_path)
            try:
                for r in req["requests"]:
                    if r["type"] == "close":
                        results.append({"type": "ok", "response": {"type": "close"}})
                        continue
                    stmt = r["stmt"]
                    args = [_untyped(a) for a in stmt.get("args", [])]
                    try:
                        cur = conn.execute(stmt["sql"], args)
                        rows = cur.fetchall()
                        cols = [{"name": d[0], "decltype": None}
                                for d in (cur.description or [])]
                        conn.commit()
                        results.append({"type": "ok", "response": {
                            "type": "execute", "result": {
                                "cols": cols,
                                "rows": [[_typed(c) for c in row] for row in rows],
                                "affected_row_count": max(cur.rowcount, 0),
                                "last_insert_rowid": (str(cur.lastrowid)
                                                      if cur.lastrowid else None),
                            }}})
                    except sqlite3.Error as exc:
                        results.append({"type": "error", "error": {
                            "message": str(exc), "code": "SQLITE_ERROR"}})
            finally:
                conn.close()
        self._send(200, {"baton": None, "base_url": None, "results": results})

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Server:
    def __enter__(self):
        import warnings
        warnings.simplefilter("ignore", ResourceWarning)
        FakeTurso.db_path = str(Path(tempfile.mkdtemp()) / "remote.db")
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeTurso)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __exit__(self, *a):
        self.httpd.shutdown()


class DriverTests(unittest.TestCase):
    def test_normalize_url(self):
        self.assertEqual(normalize_url("libsql://db-org.turso.io/"),
                         "https://db-org.turso.io")
        self.assertEqual(normalize_url("https://x"), "https://x")

    def test_roundtrip_types_and_cursor(self):
        with _Server() as url:
            c = connect(url, TOKEN)
            c.executescript("CREATE TABLE t(id INTEGER PRIMARY KEY, a TEXT, "
                            "b REAL, c BLOB, d INTEGER);\nPRAGMA foo;\n")
            cur = c.execute("INSERT INTO t(a,b,c,d) VALUES(?,?,?,?)",
                            ("x", 1.5, b"\x00\xff", None))
            self.assertEqual(cur.lastrowid, 1)
            self.assertEqual(cur.rowcount, 1)
            rows = c.execute("SELECT * FROM t").fetchall()
            self.assertEqual(dict(rows[0]), {"id": 1, "a": "x", "b": 1.5,
                                             "c": b"\x00\xff", "d": None})
            self.assertEqual(rows[0][1], "x")          # positional too
            info = c.execute("PRAGMA table_info(t)").fetchall()
            self.assertEqual([r[1] for r in info], ["id", "a", "b", "c", "d"])
            self.assertEqual(c.execute("DELETE FROM t WHERE id=99").rowcount, 0)
            with self.assertRaises(LibsqlError):
                c.execute("SELECT * FROM nope")
            self.assertTrue(c.ping())

    def test_auth_and_unreachable(self):
        with _Server() as url:
            with self.assertRaises(LibsqlError) as cm:
                connect(url, "wrong").execute("SELECT 1")
            self.assertIn("401", str(cm.exception))
        with self.assertRaises(LibsqlError):
            connect("http://127.0.0.1:9", TOKEN, timeout=2).execute("SELECT 1")


class AgentOnTursoTests(unittest.TestCase):
    """The full agent, storage pointed at the remote database."""

    def setUp(self):
        self._srv = _Server()
        self.url = self._srv.__enter__()
        self.tmp = Path(tempfile.mkdtemp(prefix="realai-turso-"))

    def tearDown(self):
        self._srv.__exit__(None, None, None)

    def _agent(self, **kw) -> Agent:
        cfg = Config(data_dir=self.tmp, tick_seconds=3600,
                     database_url=self.url, database_token=TOKEN, **kw)
        return Agent(cfg)

    def test_storage_backend_flag(self):
        st = Storage(self.tmp / "unused.db", database_url=self.url, auth_token=TOKEN)
        self.assertEqual(st.backend, "turso")
        st.state_set("k", {"a": 1})
        self.assertEqual(st.state_get("k"), {"a": 1})
        st.count("x", "y")
        self.assertEqual(st.get_counters()["x"]["y"], 1)

    def test_state_survives_cold_start(self):
        """A second Agent (new process on Vercel) sees the first one's state,
        including the owner key and the hashing pepper."""
        a1 = self._agent()
        kid, key, created = a1.keys.ensure_owner_key()
        self.assertTrue(created)
        a1.handle_message("remember that the wifi password is hunter2",
                          sender="boss", scopes=["owner"])
        a1.mind.set_identity(agent_name="NOVA", owner_name="Ali")
        # wipe the "disk": the data dir is gone, only the database remains
        import shutil
        shutil.rmtree(self.tmp)
        self.tmp = Path(tempfile.mkdtemp(prefix="realai-turso2-"))
        a2 = self._agent()
        kid2, key2, created2 = a2.keys.ensure_owner_key()
        self.assertFalse(created2)
        self.assertEqual((kid2, key2), (kid, key))
        self.assertIsNotNone(a2.keys.verify(key))          # pepper matched
        self.assertEqual(a2.mind.agent_name, "NOVA")
        rows = a2.storage.search_memories(["wifi", "password"], limit=3)
        self.assertTrue(rows and "hunter2" in json.dumps(rows[0]["value"]))

    def test_full_chat_pipeline_remote(self):
        agent = self._agent()
        r = agent.reply("hello")
        self.assertTrue(r["response"])
        r = agent.handle_message("status", sender="boss", scopes=["owner"])
        self.assertEqual(r["meta"]["intent"], "status")
        r = agent.handle_message('teach: "flip the switch" means command',
                                 sender="boss", scopes=["owner"])
        self.assertEqual(r["meta"]["intent"], "learn")
        agent.users.request("customer1")
        agent.users.approve("customer1")
        self.assertEqual(agent.storage.user_get("customer1")["status"], "APPROVED")
        self.assertGreater(FakeTurso.calls, 20)

    def test_web_layer_and_cron_remote(self):
        from realaiagent.web import WebApp
        agent = self._agent(is_vercel=True, web_token="wt",
                            serverless_tick_seconds=0)
        web = WebApp(agent)
        st, _, body, _ = web.handle("GET", "/healthz", {}, {})
        self.assertEqual(json.loads(body)["storage"], "turso")
        st, _, body, _ = web.handle("GET", "/cron/tick", {}, {})
        self.assertEqual(st, 401)
        st, _, body, _ = web.handle("GET", "/cron/tick", {"token": "wt"}, {})
        self.assertEqual(st, 200, body)
        self.assertTrue(json.loads(body)["ticked"])
        before = agent.mind.snapshot()["thoughts"]
        st, _, body, _ = web.handle("POST", "/public/chat", {},
                                    {"content-type": "application/json"},
                                    b'{"message":"hi"}')
        self.assertEqual(st, 200)
        self.assertGreater(agent.mind.snapshot()["thoughts"], before)

    def test_generated_deck_survives_disk_loss(self):
        from realaiagent.web import WebApp
        agent = self._agent()
        r = agent.reply("make a presentation about bees")
        url = r["meta"]["attachments"][0]["url"]
        import shutil
        shutil.rmtree(self.tmp)
        self.tmp = Path(tempfile.mkdtemp(prefix="realai-turso3-"))
        agent2 = self._agent()
        st, ct, body, hdr = WebApp(agent2).handle("GET", url, {}, {})
        self.assertEqual(st, 200)
        self.assertTrue(body.startswith(b"PK"))
        self.assertIn("attachment", hdr.get("Content-Disposition", ""))

    def test_tick_if_due_is_rate_limited_through_db(self):
        agent = self._agent(serverless_tick_seconds=3600)
        self.assertTrue(agent.tick_if_due())
        self.assertFalse(agent.tick_if_due())
        self.assertTrue(agent.tick_if_due(force=True))


if __name__ == "__main__":
    unittest.main()
