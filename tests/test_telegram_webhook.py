"""Telegram webhook mode (0.3.0): synchronous replies + setWebhook."""

import unittest

from realaiagent.telegram import Telegram, TelegramMaster

from .helpers import make_agent


class FakeTg:
    """Stand-in gateway with the webhook-mode sync send (no network)."""

    def __init__(self, enabled=True, chat_id="owner-chat") -> None:
        self.enabled = enabled
        self.chat_id = chat_id
        self.sent_sync = []
        self.posts = []

    def send_sync(self, text, chat_id=None) -> bool:
        self.sent_sync.append(
            {"text": text, "chat_id": chat_id or self.chat_id})
        return self.enabled

    def send(self, text, chat_id=None, dedup_key="",
             min_interval=60.0) -> bool:
        return False

    def get(self, method, params, timeout=30.0):
        return {"result": []}

    def _post(self, method, payload, timeout=10.0):
        self.posts.append({"method": method, "payload": payload})
        return {"ok": True}


class TestHandleUpdate(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.tg = FakeTg()
        self.master = TelegramMaster(self.tg, self.agent, self.agent.users)

    def update(self, text, chat_id="owner-chat"):
        return {"update_id": 1, "message": {"text": text,
                                            "chat": {"id": chat_id}}}

    def test_owner_command_replies_sync(self):
        reply = self.master.handle_update(self.update("/status"))
        self.assertIsNotNone(reply)
        self.assertIn("status", reply)
        last = self.tg.sent_sync[-1]
        self.assertEqual(last["chat_id"], "owner-chat")
        self.assertEqual(last["text"], reply)

    def test_wrong_chat_rejected(self):
        reply = self.master.handle_update(self.update("/status", "stranger"))
        self.assertIsNone(reply)
        last = self.tg.sent_sync[-1]
        self.assertEqual(last["chat_id"], "stranger")
        self.assertIn("unauthorized", last["text"].lower())

    def test_no_text_ignored(self):
        upd = {"update_id": 2, "message": {
            "chat": {"id": "owner-chat"}, "photo": ["f1"]}}
        self.assertIsNone(self.master.handle_update(upd))
        self.assertEqual(self.tg.sent_sync, [])

    def test_unknown_command_answered(self):
        reply = self.master.handle_update(self.update("/frobnicate"))
        self.assertIn("Unknown", reply)
        self.assertEqual(len(self.tg.sent_sync), 1)


class TestWebhookGateway(unittest.TestCase):
    def test_send_sync_disabled(self):
        agent = make_agent()  # no bot token -> disabled
        self.assertFalse(agent.telegram.enabled)
        self.assertFalse(agent.telegram.send_sync("hi"))

    def test_set_webhook_params(self):
        agent = make_agent()
        tg = Telegram("123:ABC", "42", agent.storage)
        cap = FakeTg()
        tg._post = cap._post  # capture, no network
        out = tg.set_webhook("https://app.vercel.app/webhook", "s3cret")
        self.assertTrue(out["ok"])
        self.assertEqual(cap.posts, [{
            "method": "setWebhook",
            "payload": {"url": "https://app.vercel.app/webhook",
                        "drop_pending_updates": True,
                        "secret_token": "s3cret"}}])
        tg.delete_webhook()
        self.assertEqual(cap.posts[1]["method"], "deleteWebhook")


if __name__ == "__main__":
    unittest.main()
