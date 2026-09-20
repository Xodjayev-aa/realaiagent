"""Vercel serverless handler (api/index.py) - 0.3.0 free-tier entrypoint.

Two layers are covered:

* :class:`TestHandlerShapes` - the dict-in/dict-out ``handle_event`` seam
  (Lambda/Netlify shape, also used by the routing/normalisation tests).
* :class:`TestWsgiAdapter` - the PEP 3333 WSGI ``app`` that
  ``@vercel/python`` *actually* invokes. This is the path that was
  returning HTTP 500 in production.
"""

import base64
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from realaiagent.config import Config
import realaiagent.vercel as vcore

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "realai_vercel_index", ROOT / "api" / "index.py")
vercel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vercel)

_ENV_KEYS = ["VERCEL", "REALAI_DATA_DIR", "REALAI_TELEGRAM_BOT_TOKEN",
             "REALAI_TELEGRAM_CHAT_ID", "REALAI_TELEGRAM_POLL",
             "REALAI_TELEGRAM_WEBHOOK_SECRET", "REALAI_WEB_TOKEN",
             "REALAI_SSE_WINDOW", "REALAI_FILE_ROOTS"]


class VercelBase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ["REALAI_DATA_DIR"] = tempfile.mkdtemp(
            prefix="realai-vercel-")
        vcore.reset_state()

    def tearDown(self):
        vcore.reset_state()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def call(self, event):
        return vcore.handle_event(event)


class TestHandlerShapes(VercelBase):
    def test_healthz_vercel_shape(self):
        out = self.call({"requestMethod": "GET", "rawUrl": "/healthz",
                         "requestHeaders": {}})
        self.assertEqual(out["statusCode"], 200)
        self.assertFalse(out["isBase64Encoded"])
        self.assertTrue(out["body"])  # str body
        self.assertIn("application/json", out["headers"]["Content-Type"])
        self.assertEqual(out["headers"]["Access-Control-Allow-Origin"], "*")
        self.assertIn("X-Request-Id", out["headers"])
        self.assertTrue(json.loads(out["body"])["ok"])

    def test_healthz_aws_shape(self):
        out = self.call({"httpMethod": "GET", "path": "/healthz",
                         "headers": {}, "queryString": {}})
        self.assertEqual(out["statusCode"], 200)
        self.assertTrue(json.loads(out["body"])["ok"])

    def test_api_prefix_stripped(self):
        out = self.call({"requestMethod": "GET", "rawUrl": "/api/healthz",
                         "requestHeaders": {}})
        self.assertEqual(out["statusCode"], 200)
        # bare /api mount -> the agent index page (HTML)
        out = self.call({"requestMethod": "GET", "rawUrl": "/api",
                         "requestHeaders": {}})
        self.assertEqual(out["statusCode"], 200)
        self.assertIn("text/html", out["headers"]["Content-Type"])

    def test_chat_with_base64_body(self):
        _kid, plain, _created = vcore.get_app().agent.keys.ensure_owner_key()
        body = base64.b64encode(
            json.dumps({"message": "status"}).encode()).decode()
        out = self.call({
            "requestMethod": "POST", "rawUrl": "/api/v1/chat",
            "requestHeaders": {
                "Authorization": f"Bearer {plain}",
                "Content-Type": "application/json"},
            "body": body, "isBase64Encoded": True,
        })
        self.assertEqual(out["statusCode"], 200)
        self.assertIn("response", json.loads(out["body"]))

    def test_webhook_disabled_503(self):
        out = self.call({"requestMethod": "POST", "rawUrl": "/webhook",
                         "requestHeaders": {}, "body": "{}"})
        self.assertEqual(out["statusCode"], 503)
        self.assertFalse(json.loads(out["body"])["ok"])

    def test_sse_response_shape(self):
        out = self.call({"requestMethod": "GET",
                         "rawUrl": "/stream/all?window=0",
                         "requestHeaders": {}})
        self.assertEqual(out["statusCode"], 200)
        self.assertTrue(
            out["headers"]["Content-Type"].startswith("text/event-stream"))
        self.assertIn("event: open", out["body"])
        self.assertIn("event: window", out["body"])

    def test_dashboard_route(self):
        out = self.call({"requestMethod": "GET", "rawUrl": "/dashboard",
                         "requestHeaders": {}})
        self.assertEqual(out["statusCode"], 200)
        self.assertIn("text/html", out["headers"]["Content-Type"])
        self.assertIn("RealAI", out["body"])

    def test_404(self):
        out = self.call({"requestMethod": "GET", "rawUrl": "/nope",
                         "requestHeaders": {}})
        self.assertEqual(out["statusCode"], 404)
        self.assertEqual(
            json.loads(out["body"])["error"]["code"], "not_found")


class TestVercelConfig(VercelBase):
    def test_vercel_env_config(self):
        os.environ["VERCEL"] = "1"
        os.environ.pop("REALAI_DATA_DIR", None)
        cfg = Config.from_env()
        self.assertTrue(cfg.is_vercel)
        # /tmp is the only writable disk on the free tier
        self.assertEqual(cfg.data_dir, Path("/tmp/realai"))
        self.assertFalse(cfg.telegram_poll)  # webhook mode, not long-poll
        self.assertEqual(cfg.sse_window, 20.0)

    def test_no_vercel_defaults(self):
        os.environ.pop("REALAI_DATA_DIR", None)
        cfg = Config.from_env()
        self.assertFalse(cfg.is_vercel)
        self.assertEqual(cfg.data_dir, Path("data").resolve())
        self.assertTrue(cfg.telegram_poll)

    def test_sse_window_clamped(self):
        os.environ["REALAI_SSE_WINDOW"] = "999"
        self.assertEqual(Config.from_env().sse_window, 25.0)
        os.environ["REALAI_SSE_WINDOW"] = "0.001"
        self.assertEqual(Config.from_env().sse_window, 1.0)


# --------------------------------------------------------------- WSGI layer
def make_environ(method="GET", path="/", query="", body=b"", headers=None):
    """A minimal but faithful WSGI environ, as the Vercel bridge builds it."""
    env = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": query,
        "SERVER_NAME": "realai.vercel.app",
        "SERVER_PORT": "443",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.url_scheme": "https",
        "wsgi.input": io.BytesIO(body),
        "wsgi.errors": io.StringIO(),
    }
    if body:
        env["CONTENT_LENGTH"] = str(len(body))
    for name, value in (headers or {}).items():
        key = name.upper().replace("-", "_")
        env[key if key in ("CONTENT_TYPE", "CONTENT_LENGTH")
            else "HTTP_" + key] = value
    return env


class TestWsgiAdapter(VercelBase):
    """The path @vercel/python really takes: app(environ, start_response)."""

    def setUp(self):
        super().setUp()
        self.started = []

    def start_response(self, status, headers, exc_info=None):
        self.started.append({"status": status, "headers": dict(headers)})

    def wsgi(self, **kw):
        rv = vcore.app(make_environ(**kw), self.start_response)
        return rv

    def test_returns_iterable_of_bytes(self):
        """The original 500: a dict response iterates to *str* keys."""
        rv = self.wsgi(path="/healthz")
        chunks = list(rv)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertIsInstance(chunk, bytes)
        self.assertTrue(b"".join(chunks))

    def test_start_response_always_called_with_status_line(self):
        self.wsgi(path="/healthz")
        self.assertEqual(len(self.started), 1)
        self.assertEqual(self.started[0]["status"], "200 OK")

    def test_healthz(self):
        rv = self.wsgi(path="/healthz")
        payload = json.loads(b"".join(rv))
        self.assertTrue(payload["ok"])
        self.assertIn("application/json",
                      self.started[0]["headers"]["Content-Type"])
        self.assertEqual(
            self.started[0]["headers"]["Access-Control-Allow-Origin"], "*")

    def test_root_serves_agent_index(self):
        rv = self.wsgi(path="/")
        self.assertIn("text/html", self.started[0]["headers"]["Content-Type"])
        self.assertIn(b"talk to it", b"".join(rv))

    def test_api_prefix_stripped(self):
        self.wsgi(path="/api/healthz")
        self.assertEqual(self.started[0]["status"], "200 OK")
        self.wsgi(path="/api")
        self.assertIn("text/html", self.started[1]["headers"]["Content-Type"])

    def test_query_string_parsed(self):
        rv = self.wsgi(path="/stream/all", query="window=0")
        ctype = self.started[0]["headers"]["Content-Type"]
        self.assertTrue(ctype.startswith("text/event-stream"))
        self.assertIn(b"event: open", b"".join(rv))

    def test_post_body_and_headers_read_from_environ(self):
        _kid, plain, _created = vcore.get_app().agent.keys.ensure_owner_key()
        body = json.dumps({"message": "status"}).encode()
        rv = self.wsgi(
            method="POST", path="/api/v1/chat", body=body,
            headers={"Authorization": f"Bearer {plain}",
                     "Content-Type": "application/json"})
        self.assertEqual(self.started[0]["status"], "200 OK")
        self.assertIn("response", json.loads(b"".join(rv)))

    def test_unauthorized_without_key(self):
        self.wsgi(method="POST", path="/v1/chat",
                  body=b'{"message":"hi"}',
                  headers={"Content-Type": "application/json"})
        self.assertEqual(self.started[0]["status"], "401 Unauthorized")

    def test_dashboard(self):
        rv = self.wsgi(path="/dashboard")
        self.assertIn(b"RealAI", b"".join(rv))

    def test_options_preflight_204_has_no_content_length(self):
        self.wsgi(method="OPTIONS", path="/v1/chat")
        self.assertEqual(self.started[0]["status"], "204 No Content")
        self.assertNotIn("Content-Length", self.started[0]["headers"])
        self.assertIn("Access-Control-Allow-Methods",
                      self.started[0]["headers"])

    def test_404(self):
        rv = self.wsgi(path="/nope")
        self.assertEqual(self.started[0]["status"], "404 Not Found")
        self.assertEqual(json.loads(b"".join(rv))["error"]["code"],
                         "not_found")

    def test_content_length_matches_body(self):
        rv = self.wsgi(path="/healthz")
        body = b"".join(rv)
        self.assertEqual(int(self.started[0]["headers"]["Content-Length"]),
                         len(body))

    def test_environ_to_request_header_mapping(self):
        req = vcore.environ_to_request(make_environ(
            method="POST", path="/v1/chat", query="a=1&b=",
            body=b'{"x":1}',
            headers={"Content-Type": "application/json",
                     "X-Web-Token": "t0k"}))
        self.assertEqual(req["method"], "POST")
        self.assertEqual(req["path"], "/v1/chat")
        self.assertEqual(req["query"], {"a": "1", "b": ""})
        self.assertEqual(req["body"], b'{"x":1}')
        self.assertEqual(req["headers"]["content-type"], "application/json")
        self.assertEqual(req["headers"]["x-web-token"], "t0k")
        self.assertNotIn("http_content_type", req["headers"])

    def test_unhandled_error_returns_json_500_not_an_exception(self):
        real = vcore.get_app
        vcore.get_app = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            rv = self.wsgi(path="/healthz")
            body = b"".join(rv)
        finally:
            vcore.get_app = real
        self.assertEqual(self.started[0]["status"], "500 Internal Server Error")
        self.assertEqual(json.loads(body)["error"]["code"], "internal")

    def test_status_line_phrases(self):
        self.assertEqual(vcore._status_line(204), "204 No Content")
        self.assertEqual(vcore._status_line(429), "429 Too Many Requests")
        self.assertEqual(vcore._status_line(418), "418 Unknown")


class TestEntrypointContract(VercelBase):
    """Guards the two mistakes that produced the production 500."""

    def test_entrypoints_export_a_wsgi_app(self):
        for name in ("app", "application"):
            self.assertIs(getattr(vercel, name), vcore.wsgi_app)
        self.assertFalse(
            __import__("inspect").iscoroutinefunction(vercel.app))

    def test_no_plain_function_handler_exported(self):
        """`handler` must be a BaseHTTPRequestHandler *class*, if present."""
        from http.server import BaseHTTPRequestHandler
        for mod in (vercel, vcore):
            h = getattr(mod, "handler", None)
            self.assertTrue(
                h is None or (isinstance(h, type)
                              and issubclass(h, BaseHTTPRequestHandler)),
                f"{mod.__name__} exports a non-BaseHTTPRequestHandler "
                f"'handler'; @vercel/python would reject it")

    def test_catchall_loads_with_absolute_import(self):
        """A relative `from .index import ...` raises ImportError here."""
        spec = importlib.util.spec_from_file_location(
            "realai_vercel_catchall", ROOT / "api" / "[...path].py")
        catchall = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(catchall)
        self.assertIs(catchall.app, vcore.wsgi_app)
        self.assertIs(catchall.application, vcore.wsgi_app)

    def test_functions_glob_matches_both_entrypoints(self):
        """`api/[...path].py` as a glob key is a character class and
        matches nothing - it must be `api/**/*.py`."""
        import glob
        cfg = json.loads((ROOT / "vercel.json").read_text())
        for pattern, fn_cfg in cfg["functions"].items():
            matched = {Path(p).name
                       for p in glob.glob(str(ROOT / pattern), recursive=True)}
            self.assertIn("index.py", matched)
            self.assertIn("[...path].py", matched)
            self.assertEqual(fn_cfg["maxDuration"], 30)


if __name__ == "__main__":
    unittest.main()
