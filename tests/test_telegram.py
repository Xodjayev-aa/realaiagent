import unittest

from realaiagent.telegram import Telegram, TelegramMaster

from .helpers import make_agent


class FakeTelegram:
    """Stands in for the real Telegram gateway (no network)."""

    def __init__(self) -> None:
        self.enabled = True
        self.chat_id = "owner-chat"
        self.sent = []

    def send(self, text, chat_id=None, dedup_key="",
             min_interval=60.0) -> bool:
        self.sent.append({"text": text, "chat_id": chat_id or self.chat_id,
                          "dedup_key": dedup_key})
        return True

    def get(self, method, params, timeout=30.0):
        return {"result": []}


class TestTelegramDispatch(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.tg = FakeTelegram()
        self.master = TelegramMaster(self.tg, self.agent, self.agent.users)
        self.agent.users.request("alice")

    def dispatch(self, text):
        return self.master._dispatch(text)

    def test_help(self):
        self.assertIn("/approve", self.dispatch("/help"))

    def test_status_and_report(self):
        self.assertIn("status", self.dispatch("/status"))
        self.assertIn("GRAND TOTAL", self.dispatch("/report"))

    def test_approve_mints_key_and_replies(self):
        reply = self.dispatch("/approve alice")
        self.assertIn("approved", reply)
        self.assertIn("rxa_", reply)
        u = self.agent.storage.user_get("alice")
        self.assertEqual(u["status"], "APPROVED")

    def test_vip_and_topup(self):
        self.agent.users.request("bob")
        self.dispatch("/approve bob")
        self.assertIn("VIP", self.dispatch("/vip bob"))
        self.assertTrue(self.agent.storage.user_get("bob")["is_vip"])
        self.assertIn("topped up", self.dispatch("/topup bob 5"))
        self.assertAlmostEqual(
            self.agent.storage.user_get("bob")["balance"], 5.0)

    def test_deny_bans(self):
        self.agent.users.request("carl")
        self.dispatch("/approve carl")
        self.dispatch("/deny carl")
        self.assertEqual(self.agent.storage.user_get("carl")["status"],
                         "BANNED")

    def test_users_listing(self):
        out = self.dispatch("/users")
        self.assertIn("alice", out)

    def test_kill_switch(self):
        self.assertTrue(self.agent.mind.autonomy)
        self.dispatch("/kill")
        self.assertFalse(self.agent.mind.autonomy)
        self.dispatch("/on")
        self.assertTrue(self.agent.mind.autonomy)

    def test_chat_forwards_as_owner(self):
        reply = self.dispatch("/chat hello there")
        self.assertIn("Hello", reply)  # the agent greets its owner

    def test_unknown_command(self):
        self.assertIn("Unknown", self.dispatch("/frobnicate"))

    def test_unauthorized_chat_rejected(self):
        update = {"update_id": 1, "message": {
            "text": "/status", "chat": {"id": "stranger-chat"}}}
        self.master._handle(update)
        unauthorized = [s for s in self.tg.sent
                        if "unauthorized" in s["text"].lower()]
        self.assertTrue(unauthorized)
        # and nothing was sent to the stranger except the rejection
        to_stranger = [s for s in self.tg.sent
                       if s["chat_id"] == "stranger-chat"]
        self.assertEqual(len(to_stranger), 1)

    def test_owner_chat_accepted(self):
        update = {"update_id": 1, "message": {
            "text": "/status", "chat": {"id": "owner-chat"}}}
        self.master._handle(update)
        status_replies = [s for s in self.tg.sent
                          if "status" in s["text"].lower()
                          and s["chat_id"] == "owner-chat"]
        self.assertTrue(status_replies)


class TestTelegramGateway(unittest.TestCase):
    def test_disabled_noop(self):
        a = make_agent()
        self.assertFalse(a.telegram.enabled)
        self.assertFalse(a.telegram.send("hi"))

    def test_enabled_flag(self):
        from realaiagent.telegram import Telegram
        from realaiagent.storage import Storage
        import tempfile
        from pathlib import Path
        s = Storage(Path(tempfile.mkdtemp()) / "t.db")
        tg = Telegram("123:ABC", "42", s)
        self.assertTrue(tg.enabled)
        self.assertFalse(Telegram("", "42", s).enabled)
        s.close()


if __name__ == "__main__":
    unittest.main()
