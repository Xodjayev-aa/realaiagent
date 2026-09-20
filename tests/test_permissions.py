import unittest

from realaiagent.actions.base import ActionRequest
from realaiagent.actions.permissions import PermissionManager

from .helpers import make_agent


class TestPermissionGate(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.pm = self.agent.permissions

    def test_everything_asks_by_default(self):
        d = self.pm.check(ActionRequest("system.command",
                                        "system.command:echo hi",
                                        {"command": "echo hi"}))
        self.assertFalse(d.granted)
        self.assertTrue(d.awaiting)
        self.assertIsNotNone(d.pending_id)

    def test_default_harmless_allows(self):
        d = self.pm.check(ActionRequest("notify", "notify:x",
                                        {"message": "hi"}))
        self.assertTrue(d.granted)

    def test_approve_creates_exact_allow(self):
        d = self.pm.check(ActionRequest("device.control", "lamp-1:on",
                                        {"action": "on"}))
        row = self.pm.decide(d.pending_id, approve=True)
        self.assertEqual(row["status"], "approved")
        # same action now allowed
        d2 = self.pm.check(ActionRequest("device.control", "lamp-1:on",
                                         {"action": "on"}))
        self.assertTrue(d2.granted)
        # different action still asks
        d3 = self.pm.check(ActionRequest("device.control", "lamp-1:off",
                                         {"action": "off"}))
        self.assertTrue(d3.awaiting)

    def test_deny_stays_pending_then_denied(self):
        d = self.pm.check(ActionRequest("files.write", "files.write:x",
                                        {"path": "x"}))
        row = self.pm.decide(d.pending_id, approve=False)
        self.assertEqual(row["status"], "denied")
        # no allow policy created -> still asks
        d2 = self.pm.check(ActionRequest("files.write", "files.write:x",
                                         {"path": "x"}))
        self.assertTrue(d2.awaiting)

    def test_explicit_deny_policy_blocks(self):
        self.pm.set_policy("system.command", "deny",
                           pattern="system.command:rm -rf /")
        d = self.pm.check(ActionRequest("system.command",
                                        "system.command:rm -rf /",
                                        {"command": "rm -rf /"}))
        self.assertFalse(d.granted)
        self.assertFalse(d.awaiting)

    def test_category_wide_allow(self):
        self.pm.set_policy("device.control", "allow")
        d = self.pm.check(ActionRequest("device.control", "fan-1:on",
                                        {"action": "on"}))
        self.assertTrue(d.granted)

    def test_deny_overrides_allow_for_same_pattern(self):
        self.pm.set_policy("device.control", "allow", pattern="lamp-1:on")
        self.pm.set_policy("device.control", "deny", pattern="lamp-1:on")
        d = self.pm.check(ActionRequest("device.control", "lamp-1:on",
                                        {"action": "on"}))
        self.assertFalse(d.granted)

    def test_pending_list_and_latest(self):
        d = self.pm.check(ActionRequest("device.control", "a:1", {}))
        pend = self.pm.pending_list()
        self.assertEqual(len(pend), 1)
        self.assertEqual(self.pm.latest_pending()["id"], d.pending_id)

    def test_decide_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.pm.decide(9999, True)

    def test_everything_counted(self):
        self.pm.check(ActionRequest("system.command", "x", {}))
        self.pm.decide(1, True)
        c = self.agent.storage.get_counters()
        self.assertGreaterEqual(c["totals"].get("permissions", 0), 2)
        self.assertGreaterEqual(c["permissions"].get("requested", 0), 1)
        self.assertGreaterEqual(c["permissions"].get("approved", 0), 1)


if __name__ == "__main__":
    unittest.main()
