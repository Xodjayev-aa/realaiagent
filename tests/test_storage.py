import tempfile
import unittest
from pathlib import Path

from realaiagent.storage import Storage


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s = Storage(self.tmp / "t.db")

    def tearDown(self):
        self.s.close()

    def test_count_and_snapshot(self):
        self.s.count("requests", "200_chat")
        self.s.count("requests", "200_chat")
        self.s.count("actions", "success")
        c = self.s.get_counters()
        self.assertEqual(c["requests"]["200_chat"], 2)
        self.assertEqual(c["totals"]["requests"], 2)
        self.assertEqual(c["totals"]["actions"], 1)
        self.assertEqual(c["grand_total"], 3)

    def test_counters_survive_restart(self):
        self.s.count("requests", "200_chat")
        self.s.count("chat", "message")
        self.s.close()
        s2 = Storage(self.tmp / "t.db")
        c = s2.get_counters()
        self.assertEqual(c["grand_total"], 2)
        self.assertEqual(c["chat"]["message"], 1)
        s2.close()

    def test_events(self):
        self.s.log_event("thought", {"line": "hi"})
        self.s.log_event("action_success", {"x": 1})
        evs = self.s.recent_events(10)
        self.assertEqual(len(evs), 2)
        self.assertEqual(evs[0]["type"], "action_success")
        self.assertEqual(evs[1]["payload"], {"line": "hi"})

    def test_state_json(self):
        self.s.state_set("x", {"a": [1, 2]})
        self.assertEqual(self.s.state_get("x"), {"a": [1, 2]})
        self.assertEqual(self.s.state_get("missing", "dflt"), "dflt")

    def test_usage_breakdowns(self):
        self.s.count("a", "b", key_id="k1")
        self.s.count("a", "c", key_id="k2")
        self.s.count("d", "e", key_id="k1")
        self.assertEqual(len(self.s.usage_breakdown("category")), 3)
        self.assertEqual(len(self.s.usage_breakdown("key")), 2)
        self.assertEqual(len(self.s.usage_breakdown("day")), 1)

    def test_keys_crud(self):
        self.s.key_insert("k1", "owner", "hash1", ["owner"])
        self.s.key_insert("k2", "ui", "hash2", ["chat"])
        self.s.key_touch("k1")
        row = self.s.key_get_by_hash("hash1")
        self.assertEqual(row["scopes"], ["owner"])
        self.assertEqual(row["request_count"], 1)
        self.assertEqual(len(self.s.key_list()), 2)
        self.assertTrue(self.s.key_revoke("k1"))
        self.assertIsNone(self.s.key_get_by_hash("hash1"))


if __name__ == "__main__":
    unittest.main()
