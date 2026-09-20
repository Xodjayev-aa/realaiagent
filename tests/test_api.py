import json
import unittest

from .helpers import ApiClient, make_agent, register_device, start_server


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = make_agent()
        cls.agent.cfg.rate_limit_per_min = 100_000
        cls.agent.cfg.rate_burst = 10_000
        cls.server = start_server(cls.agent)
        _, cls.owner_key, _ = cls.agent.keys.ensure_owner_key()
        cls.base = f"http://127.0.0.1:{cls.server.bound_port}"
        cls.owner = ApiClient(cls.base, cls.owner_key)
        cls.anon = ApiClient(cls.base, None)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.agent.stop()

    # -------------------------------------------------------------- basics

    def test_01_healthz_public(self):
        code, body = self.anon.req("GET", "/healthz")
        self.assertEqual(code, 200)
        self.assertTrue(body["ok"])
        self.assertIn("mood", body)

    def test_02_auth_required(self):
        code, _ = self.anon.req("GET", "/v1/status")
        self.assertEqual(code, 401)
        code, _ = ApiClient(self.base, "rxa_wrong").req("GET", "/v1/status")
        self.assertEqual(code, 401)

    def test_03_unknown_route(self):
        code, _ = self.owner.req("GET", "/v1/nope")
        self.assertEqual(code, 404)

    def test_04_chat_roundtrip(self):
        code, body = self.owner.req("POST", "/v1/chat",
                                    {"message": "hello there"})
        self.assertEqual(code, 200)
        self.assertIsInstance(body["response"], str)
        self.assertEqual(body["meta"]["intent"], "greet")

    def test_05_chat_requires_message(self):
        code, body = self.owner.req("POST", "/v1/chat", {})
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["code"], "bad_request")

    # ------------------------------------------------------------ full flow

    def test_10_device_permission_flow(self):
        register_device(self.agent, did="lamp-api", name="API Lamp")
        code, body = self.owner.req("POST", "/v1/chat",
                                    {"message": "turn on the API Lamp"})
        self.assertEqual(code, 200)
        act = body["meta"]["actions"][0]
        self.assertTrue(act["awaiting"])
        pid = act["pending_id"]

        code, pend = self.owner.req("GET", "/v1/permissions/pending")
        self.assertEqual(code, 200)
        self.assertEqual(pend["pending"][0]["id"], pid)

        code, body = self.owner.req(
            "POST", f"/v1/permissions/pending/{pid}/approve")
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "approved")

        code, body = self.owner.req("POST", "/v1/chat",
                                    {"message": "turn on the API Lamp"})
        act = body["meta"]["actions"][0]
        self.assertTrue(act["ok"])
        self.assertIn("on", act["output"])

    def test_11_direct_device_control(self):
        register_device(self.agent, did="direct-1", name="Direct Lamp")
        self.agent.permissions.set_policy("device.control", "allow",
                                          pattern="direct-1:on")
        code, body = self.owner.req(
            "POST", "/v1/devices/direct-1/control", {"action": "on"})
        self.assertEqual(code, 200)
        self.assertTrue(body["ok"])

    # ---------------------------------------------------------------- keys

    def test_20_key_management_and_scopes(self):
        code, body = self.owner.req("POST", "/v1/keys",
                                    {"name": "ui-client",
                                     "scopes": ["chat", "devices"]})
        self.assertEqual(code, 200)
        self.assertTrue(body["key"].startswith("rxa_"))
        kid = body["id"]

        client = ApiClient(self.base, body["key"])
        code, _ = client.req("GET", "/v1/status")
        self.assertEqual(code, 200)
        code, _ = client.req("GET", "/v1/keys")  # owner-only
        self.assertEqual(code, 403)

        code, _ = self.owner.req("POST", f"/v1/keys/{kid}/revoke")
        self.assertEqual(code, 200)
        code, _ = client.req("GET", "/v1/status")
        self.assertEqual(code, 401)

    def test_21_invalid_scopes_rejected(self):
        code, body = self.owner.req("POST", "/v1/keys",
                                    {"name": "x", "scopes": ["root"]})
        self.assertEqual(code, 400)

    # -------------------------------------------------------------- learn

    def test_30_learn_endpoint(self):
        code, body = self.owner.req(
            "POST", "/v1/learn",
            {"examples": [{"text": "make the toast golden",
                           "intent": "toast"}]})
        self.assertEqual(code, 200)
        self.assertEqual(body["trained"], 1)
        intent, _, _ = self.agent.model.predict("make the toast golden")
        self.assertEqual(intent, "toast")

    def test_31_learn_requires_examples(self):
        code, _ = self.owner.req("POST", "/v1/learn", {})
        self.assertEqual(code, 400)

    # ------------------------------------------------------------- owner

    def test_40_owner_controls(self):
        code, body = self.owner.req("POST", "/v1/owner/autonomy", {"on": False})
        self.assertEqual(code, 200)
        self.assertFalse(body["autonomy"])
        code, body = self.owner.req("POST", "/v1/owner/autonomy", {"on": True})
        self.assertTrue(body["autonomy"])

        code, body = self.owner.req(
            "POST", "/v1/owner/values",
            {"drives": {"curiosity": 0.88, "duty": 0.77}})
        self.assertEqual(code, 200)
        self.assertAlmostEqual(body["drives"]["curiosity"], 0.88, places=2)

        code, body = self.owner.req("GET", "/v1/owner/state")
        self.assertEqual(code, 200)
        self.assertIn("identity", body)

    # ------------------------------------------------------------ counters

    def test_90_everything_counted(self):
        code, body = self.owner.req("GET", "/v1/counters")
        self.assertEqual(code, 200)
        self.assertGreater(body["grand_total"], 20)
        self.assertIn("requests", body["totals"])
        self.assertIn("chat", body["totals"])
        self.assertIn("actions", body["totals"])

        code, body = self.owner.req("GET", "/v1/usage?by=key")
        self.assertEqual(code, 200)
        self.assertGreaterEqual(len(body["rows"]), 1)

    def test_91_status_shape(self):
        code, body = self.owner.req("GET", "/v1/status")
        self.assertEqual(code, 200)
        for k in ("agent", "owner", "mood", "drives", "counters"):
            self.assertIn(k, body)


if __name__ == "__main__":
    unittest.main()
