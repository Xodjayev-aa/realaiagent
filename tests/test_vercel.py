"""Vercel serverless handler (api/index.py) - 0.3.0 free-tier entrypoint."""

import base64
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from realaiagent.config import Config

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
        vercel.reset_state()

    def tearDown(self):
        vercel.reset_state()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def call(self, event):
        return vercel.handler(event)


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
        _kid, plain, _created = vercel.get_app().agent.keys.ensure_owner_key()
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


if __name__ == "__main__":
    unittest.main()
