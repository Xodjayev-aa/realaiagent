import unittest

from .helpers import ApiClient, make_agent, start_server


class TestUserEcosystem(unittest.TestCase):
    """The null-49.private layer: users, keys, billing, security."""

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

    # ------------------------------------------------------------- lifecycle

    def test_01_request_then_list(self):
        code, body = self.anon.req("POST", "/v1/users/request",
                                   {"username": "student1"})
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "PENDING")
        code, body = self.owner.req("GET", "/v1/users?status=PENDING")
        self.assertEqual(code, 200)
        self.assertTrue(any(u["username"] == "student1"
                            for u in body["users"]))

    def test_02_approve_mints_key(self):
        self.anon.req("POST", "/v1/users/request", {"username": "coder1"})
        code, body = self.owner.req(
            "POST", "/v1/users/coder1/approve", {"request_limit": 100})
        self.assertEqual(code, 200)
        self.assertTrue(body["api_key"].startswith("rxa_"))
        self.assertEqual(body["user"]["status"], "APPROVED")
        self.assertEqual(body["key_scopes"], ["chat", "devices", "learn"])
        self.assertIs(body["user"]["is_vip"], False)
        self.coder_key = body["api_key"]

    def test_03_denied_user_banned_and_key_revoked(self):
        self.anon.req("POST", "/v1/users/request", {"username": "bad1"})
        code, ap = self.owner.req("POST", "/v1/users/bad1/approve")
        bad_key = ap["api_key"]
        code, body = self.owner.req("POST", "/v1/users/bad1/deny")
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "BANNED")
        # their key is now revoked -> 401
        code, _ = ApiClient(self.base, bad_key).req("GET", "/v1/status")
        self.assertEqual(code, 401)

    # --------------------------------------------------------------- billing

    def test_10_billing_charges_and_exhausts(self):
        self.anon.req("POST", "/v1/users/request", {"username": "paid1"})
        code, ap = self.owner.req(
            "POST", "/v1/users/paid1/approve", {"request_limit": 100})
        key = ap["api_key"]
        self.owner.req("POST", "/v1/users/paid1/topup", {"amount": 0.15})
        # 0.15 / 0.05 = exactly 3 requests allowed
        client = ApiClient(self.base, key)
        for _ in range(3):
            code, _ = client.req("GET", "/v1/status")
            self.assertEqual(code, 200)
        code, body = client.req("GET", "/v1/status")
        self.assertEqual(code, 402)
        self.assertEqual(body["error"]["code"], "insufficient_funds")
        u = self.agent.storage.user_get("paid1")
        self.assertAlmostEqual(u["balance"], 0.0, places=6)

    def test_11_vip_is_free(self):
        self.anon.req("POST", "/v1/users/request", {"username": "vip1"})
        code, ap = self.owner.req(
            "POST", "/v1/users/vip1/approve", {"request_limit": 10})
        self.owner.req("POST", "/v1/users/vip1/vip", {"vip": True})
        client = ApiClient(self.base, ap["api_key"])
        for _ in range(10):
            code, _ = client.req("GET", "/v1/status")
            self.assertEqual(code, 200)  # never charges
        u = self.agent.storage.user_get("vip1")
        self.assertEqual(u["balance"], 0.0)  # nothing deducted

    def test_12_owner_key_exempt(self):
        for _ in range(12):
            code, _ = self.owner.req("GET", "/v1/status")
            self.assertEqual(code, 200)  # owner never billed

    def test_13_request_limit_enforced(self):
        self.anon.req("POST", "/v1/users/request", {"username": "limit1"})
        code, ap = self.owner.req(
            "POST", "/v1/users/limit1/approve", {"request_limit": 2})
        self.owner.req("POST", "/v1/users/limit1/topup", {"amount": 5.0})
        client = ApiClient(self.base, ap["api_key"])
        code, _ = client.req("GET", "/v1/status")
        self.assertEqual(code, 200)
        code, _ = client.req("GET", "/v1/status")
        self.assertEqual(code, 200)
        code, body = client.req("GET", "/v1/status")
        self.assertEqual(code, 429)
        self.assertEqual(body["error"]["code"], "limit_exceeded")

    def test_14_unapproved_is_blocked(self):
        self.anon.req("POST", "/v1/users/request", {"username": "wait1"})
        # no approve yet: owner can't even hand them a key; simulate via
        # a raw key for them
        _kid, plain, _meta = self.agent.keys.create_key(
            "user:wait1", ["chat"], user="wait1", request_limit=10)
        code, body = ApiClient(self.base, plain).req("GET", "/v1/status")
        self.assertEqual(code, 403)
        self.assertEqual(body["error"]["code"], "not_approved")

    # ------------------------------------------------------------- security

    def test_20_invalid_key_counts_intrusion(self):
        before = (self.agent.storage.get_counters()
                  .get("security", {}).get("intrusion", 0))
        code, _ = ApiClient(self.base, "rxa_totally_fake_key_123") \
            .req("GET", "/v1/status")
        self.assertEqual(code, 401)
        after = (self.agent.storage.get_counters()
                 .get("security", {}).get("intrusion", 0))
        self.assertGreater(after, before)

    def test_21_banned_user_counts_intrusion(self):
        self.anon.req("POST", "/v1/users/request", {"username": "banme1"})
        code, ap = self.owner.req(
            "POST", "/v1/users/banme1/approve", {"request_limit": 10})
        key = ap["api_key"]
        self.owner.req("POST", "/v1/users/banme1/topup", {"amount": 5.0})
        before = (self.agent.storage.get_counters()
                  .get("security", {}).get("intrusion", 0))
        code, _ = ApiClient(self.base, key).req("GET", "/v1/status")
        self.assertEqual(code, 200)  # fine while approved
        self.owner.req("POST", "/v1/users/banme1/deny")
        code, _ = ApiClient(self.base, key).req("GET", "/v1/status")
        self.assertEqual(code, 401)  # key revoked after ban
        after = (self.agent.storage.get_counters()
                 .get("security", {}).get("intrusion", 0))
        self.assertGreater(after, before)

    # ------------------------------------------------------------- business

    def test_30_usage_view(self):
        code, body = self.owner.req("GET", "/v1/users/paid1/usage")
        self.assertEqual(code, 200)
        self.assertEqual(body["user"]["username"], "paid1")
        self.assertTrue(len(body["keys"]) >= 1)
        k = body["keys"][0]
        self.assertGreaterEqual(k["request_count"], 3)
        self.assertEqual(k["request_limit"], 100)

    def test_31_everything_counted(self):
        c = self.agent.storage.get_counters()
        for cat in ("users", "billing", "keys", "security", "requests"):
            self.assertIn(cat, c["totals"])
        self.assertGreaterEqual(c["users"].get("requested", 0), 5)
        self.assertGreaterEqual(c["users"].get("status_approved", 0), 4)
        self.assertGreaterEqual(c["billing"].get("charged", 0), 3)
        self.assertGreaterEqual(c["billing"].get("vip_free", 0), 10)
        self.assertGreaterEqual(c["security"].get("intrusion", 0), 2)


if __name__ == "__main__":
    unittest.main()
