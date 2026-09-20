"""Web layer (0.3.0): dashboard, 15 SSE topics, webhook, shared dispatch."""

import json
import unittest

from realaiagent.web import TOPICS, TOPIC_NAMES, WebApp

from .helpers import make_agent


def call(app, method, path, query=None, headers=None, body=b""):
    status, ctype, body_bytes, extra = app.handle(
        method, path, query or {}, headers or {}, body)
    return status, ctype, body_bytes, extra


def jbody(body_bytes):
    return json.loads(body_bytes.decode("utf-8"))


class TestTopicsAndIndex(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.app = WebApp(self.agent)

    def test_topics_list_15_unique(self):
        self.assertGreaterEqual(len(TOPICS), 13)  # "13+ SSE"
        self.assertEqual(len(TOPICS), 15)
        self.assertEqual(len(set(TOPIC_NAMES)), 15)
        self.assertIn("all", TOPIC_NAMES)
        self.assertIn("security", TOPIC_NAMES)
        self.assertIn("health", TOPIC_NAMES)

    def test_stream_index_lists_topics(self):
        status, ctype, body, _ = call(self.app, "GET", "/stream")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "application/json")
        out = jbody(body)
        self.assertEqual(out["count"], 15)
        names = [t["topic"] for t in out["topics"]]
        self.assertEqual(names, list(TOPIC_NAMES))
        self.assertIn("description", out["topics"][0])

    def test_stream_unknown_topic_404(self):
        status, _, body, _ = call(self.app, "GET", "/stream/nope")
        self.assertEqual(status, 404)
        self.assertEqual(jbody(body)["error"]["code"], "not_found")


class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.app = WebApp(self.agent)

    def test_dashboard_renders(self):
        status, ctype, body, _ = call(self.app, "GET", "/dashboard")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "text/html; charset=utf-8")
        html = body.decode("utf-8")
        self.assertIn("REAL", html)
        self.assertIn("EventSource", html)
        self.assertIn("/stream/", html)

    def test_dashboard_token_enforced(self):
        self.agent.cfg.web_token = "tok-123"
        status, _, body, _ = call(self.app, "GET", "/dashboard")
        self.assertEqual(status, 401)
        status, _, body, _ = call(self.app, "GET", "/dashboard",
                                  {"token": "tok-123"})
        self.assertEqual(status, 200)
        self.assertIn(b"REAL", body)


class TestSse(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.app = WebApp(self.agent)

    def sse(self, topic="chat", **query):
        query.setdefault("window", "0")
        status, ctype, body, headers = call(
            self.app, "GET", f"/stream/{topic}", query)
        return status, ctype, body.decode("utf-8"), headers

    def test_stream_headers(self):
        status, ctype, body, headers = self.sse()
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("text/event-stream"))
        self.assertIn("no-cache", headers.get("Cache-Control", ""))
        self.assertEqual(headers.get("X-Accel-Buffering"), "no")

    def test_sse_open_frame(self):
        _s, _c, body, _h = self.sse(topic="actions")
        self.assertTrue(body.startswith("retry: 2000\n"))
        self.assertIn("event: open", body)
        open_data = json.loads(body.split("event: open", 1)[1]
                               .split("data: ", 1)[1].split("\n", 1)[0])
        self.assertEqual(open_data["topic"], "actions")
        self.assertIn("since", open_data)
        self.assertIn("seq", open_data)

    def test_sse_count_event_streams(self):
        self.agent.storage.count("actions", "success",
                                 detail={"action": "device.control"})
        _s, _c, body, _h = self.sse(topic="actions")
        self.assertIn("event: actions", body)
        self.assertIn('"event": "success"', body)

    def test_sse_log_event_streams(self):
        self.agent.storage.log_event("goal_created",
                                     {"id": "g1", "description": "x"})
        _s, _c, body, _h = self.sse(topic="goals")
        self.assertIn("event: goals", body)
        self.assertIn('"type": "goal_created"', body)
        self.assertIn('"id": "g1"', body)

    def test_sse_topic_isolation(self):
        self.agent.storage.count("telegram", "sent")
        _s, _c, chat_body, _h = self.sse(topic="chat")
        _s, _c, tg_body, _h = self.sse(topic="telegram")
        self.assertNotIn("event: telegram", chat_body)
        self.assertIn("event: telegram", tg_body)

    def test_sse_all_topic_includes_everything(self):
        self.agent.storage.count("chat", "message")
        self.agent.storage.log_event("device_registered", {"id": "d1"})
        _s, _c, body, _h = self.sse(topic="all")
        self.assertIn("event: chat", body)
        self.assertIn("event: devices", body)
        self.assertIn("device_registered", body)

    def test_sse_since_cursor(self):
        self.agent.storage.count("chat", "message")
        seq = self.app.hub.last_seq()
        self.agent.storage.count("chat", "reply")
        _s, _c, body, _h = self.sse(topic="chat", window="0",
                                    since=str(seq))
        self.assertIn('"event": "reply"', body)
        self.assertNotIn('"event": "message"', body)

    def test_sse_last_event_id_header(self):
        self.agent.storage.count("chat", "message")
        seq = self.app.hub.last_seq()
        self.agent.storage.count("chat", "reply")
        _s, _c, body, _h = call(self.app, "GET", "/stream/chat",
                                {"window": "0"},
                                {"last-event-id": str(seq)})
        text = body.decode("utf-8")
        self.assertIn('"event": "reply"', text)
        self.assertNotIn('"event": "message"', text)

    def test_sse_replay_buffer(self):
        for i in range(3):
            self.agent.storage.count("chat", f"msg{i}")
        _s, _c, body, _h = self.sse(topic="chat")
        self.assertIn('"event": "msg0"', body)
        self.assertIn('"event": "msg1"', body)
        self.assertIn('"event": "msg2"', body)
        self.assertGreaterEqual(body.count("id: "), 3)

    def test_sse_window_close_frame(self):
        _s, _c, body, _h = self.sse()
        self.assertIn("event: window", body)
        tail = body.split("event: window", 1)[1]
        close = json.loads(tail.split("data: ", 1)[1].split("\n", 1)[0])
        self.assertEqual(close["close"], "window-ended")

    def test_sse_heartbeat_in_health(self):
        _s, _c, body, _h = self.sse(topic="health", window="1.5",
                                    beat="0.1")
        self.assertIn("event: health", body)
        hb = json.loads(body.split("event: health", 1)[1]
                        .split("data: ", 1)[1].split("\n", 1)[0])
        self.assertIn("uptime_s", hb)
        self.assertIn("grand_total", hb)

    def test_sse_status_snapshot_on_open(self):
        _s, _c, body, _h = self.sse(topic="status")
        self.assertIn("event: status", body)
        snap = json.loads(body.split("event: status", 1)[1]
                          .split("data: ", 1)[1].split("\n", 1)[0])
        self.assertIn("grand_total", snap)
        self.assertIn("drives", snap)
        # "all" gets the snapshot too
        _s, _c, all_body, _h = self.sse(topic="all")
        self.assertIn("event: status", all_body)

    def test_chat_turn_publishes_plan(self):
        _kid, plain, _created = self.agent.keys.ensure_owner_key()
        status, _c, _b, _x = call(
            self.app, "POST", "/v1/chat", {},
            {"authorization": f"Bearer {plain}",
             "content-type": "application/json"},
            json.dumps({"message": "status"}).encode())
        self.assertEqual(status, 200)
        _s, _c, body, _h = self.sse(topic="plans")
        self.assertIn("event: plans", body)
        plan = json.loads(body.split("event: plans", 1)[1]
                          .split("data: ", 1)[1].split("\n", 1)[0])
        self.assertIn("intent", plan)
        self.assertIn("plan", plan)

    def test_stream_token_enforced(self):
        self.agent.cfg.web_token = "sekrit"
        status, _, body, _ = call(self.app, "GET", "/stream/chat",
                                  {"window": "0"})
        self.assertEqual(status, 401)
        status, _, _b, _h = call(self.app, "GET", "/stream/chat",
                                 {"window": "0", "token": "sekrit"})
        self.assertEqual(status, 200)


class TestTransport(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.app = WebApp(self.agent)

    def test_options_cors(self):
        status, ctype, body, extra = call(self.app, "OPTIONS",
                                          "/stream/chat")
        self.assertEqual(status, 204)
        self.assertIsNone(ctype)
        self.assertEqual(body, b"")
        self.assertIn("OPTIONS",
                      extra.get("Access-Control-Allow-Methods", ""))

    def test_api_dispatch_through_webapp(self):
        status, _c, body, _x = call(self.app, "GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(jbody(body)["ok"])

        status, _c, body, _x = call(self.app, "GET", "/v1/status")
        self.assertEqual(status, 401)  # key required

        _kid, plain, _created = self.agent.keys.ensure_owner_key()
        status, _c, body, _x = call(
            self.app, "GET", "/v1/status", {},
            {"authorization": f"Bearer {plain}"})
        self.assertEqual(status, 200)
        self.assertIn("drives", jbody(body))

    def test_404_unknown_route(self):
        status, _c, body, _x = call(self.app, "GET", "/definitely/not/here")
        self.assertEqual(status, 404)
        self.assertEqual(jbody(body)["error"]["code"], "not_found")


class TestWebhook(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.app = WebApp(self.agent)
        # enable the (fake-token) telegram gateway for webhook tests
        self.agent.telegram.token = "123:ABC"
        self.agent.telegram.chat_id = "99"
        self.agent.telegram.enabled = True

    def post(self, body, headers=None, query=None):
        return call(self.app, "POST", "/webhook", query or {},
                    headers or {}, body)

    def test_webhook_rejects_bad_secret(self):
        self.agent.cfg.telegram_webhook_secret = "s3cret"
        status, _c, body, _x = self.post(
            b'{"update_id": 1}',
            {"x-telegram-bot-api-secret-token": "wrong"})
        self.assertEqual(status, 403)
        self.assertFalse(jbody(body)["ok"])
        c = self.agent.storage.get_counters()
        self.assertEqual(c.get("telegram", {}).get("webhook_rejected"), 1)

    def test_webhook_bad_json(self):
        status, _c, body, _x = self.post(b"not-json{")
        self.assertEqual(status, 400)
        self.assertEqual(jbody(body)["error"]["code"], "bad_json")

    def test_webhook_ok_processes_update(self):
        status, _c, body, _x = self.post(b'{"update_id": 42}')
        self.assertEqual(status, 200)
        self.assertTrue(jbody(body)["ok"])
        self.assertIsNotNone(self.app.master)
        c = self.agent.storage.get_counters()
        self.assertEqual(c.get("telegram", {}).get("webhook_received"), 1)

    def test_webhook_disabled_telegram_503(self):
        a2 = make_agent()
        app2 = WebApp(a2)  # telegram not configured
        status, _c, body, _x = call(app2, "POST", "/webhook", {}, {}, b"{}")
        self.assertEqual(status, 503)
        self.assertFalse(jbody(body)["ok"])


if __name__ == "__main__":
    unittest.main()
