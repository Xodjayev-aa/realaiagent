import json
import unittest

from realaiagent.actions.base import ActionRequest

from .helpers import make_agent, register_device


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.ex = self.agent.executor
        # widen file scope to the agent data dir (default already does)
        self.data = self.agent.cfg.data_dir

    def exec_ok(self, req):
        out = self.ex.execute(req)
        self.assertTrue(out.ok, out.error)
        return out

    def test_system_command_allowed_runs(self):
        self.agent.permissions.set_policy("system.command", "allow")
        out = self.exec_ok(ActionRequest("system.command",
                                         "system.command:echo hello",
                                         {"command": "echo hello"}))
        self.assertIn("hello", out.output)

    def test_system_command_denied_by_default(self):
        out = self.ex.execute(ActionRequest("system.command", "x",
                                            {"command": "echo hi"}))
        self.assertTrue(out.awaiting)

    def test_files_write_and_read_scoped(self):
        self.agent.permissions.set_policy("files.write", "allow")
        self.agent.permissions.set_policy("files.read", "allow")
        self.exec_ok(ActionRequest("files.write", "files.write:notes.txt",
                                   {"path": "notes.txt",
                                    "content": "remember this"}))
        out = self.exec_ok(ActionRequest("files.read",
                                         "files.read:notes.txt",
                                         {"path": "notes.txt"}))
        self.assertIn("remember this", out.output)
        self.assertTrue((self.data / "notes.txt").exists())

    def test_files_write_outside_scope_rejected(self):
        self.agent.permissions.set_policy("files.write", "allow")
        out = self.ex.execute(ActionRequest(
            "files.write", "files.write:/etc/evil",
            {"path": "/etc/evil", "content": "x"}))
        self.assertFalse(out.ok)
        self.assertFalse(out.awaiting)  # scope error, not a permission ask
        self.assertIn("scope", out.error)

    def test_device_control_http_capability(self):
        # http device pointed at a fake url -> controlled failure, counted
        self.agent.permissions.set_policy("device.control", "allow")
        self.agent.storage.execute(
            "INSERT INTO devices(id,name,kind,capabilities,meta,created_at,"
            " last_seen) VALUES(?,?,?,?,?,?,?)",
            ("thermo-1", "Thermostat", "climate",
             json.dumps([{"action": "set", "executor": "http",
                          "endpoint": "http://127.0.0.1:1/hvac/{level}",
                          "method": "POST"}]), "{}", 1, 1))
        out = self.ex.execute(ActionRequest(
            "device.control", "thermo-1:set",
            {"action": "set", "level": 21}, device_id="thermo-1"))
        self.assertFalse(out.ok)  # connection refused (nothing on port 1)
        self.assertIn("http", out.error)

    def test_device_control_command_capability(self):
        register_device(self.agent)
        self.agent.permissions.set_policy("device.control", "allow")
        out = self.exec_ok(ActionRequest(
            "device.control", "lamp-1:on", {"action": "on"},
            device_id="lamp-1"))
        self.assertIn("on", out.output)

    def test_unknown_device(self):
        self.agent.permissions.set_policy("device.control", "allow")
        out = self.ex.execute(ActionRequest(
            "device.control", "nope:on", {"action": "on"}, device_id="nope"))
        self.assertFalse(out.ok)

    def test_every_execution_counted(self):
        self.agent.permissions.set_policy("system.command", "allow")
        self.ex.execute(ActionRequest("system.command", "a",
                                      {"command": "echo 1"}))
        self.ex.execute(ActionRequest("system.command", "b",
                                      {"command": "false"}))
        c = self.agent.storage.get_counters()["actions"]
        self.assertEqual(c.get("success"), 1)
        self.assertEqual(c.get("failed"), 1)

    def test_q_table_updated(self):
        self.agent.permissions.set_policy("system.command", "allow")
        self.ex.execute(ActionRequest("system.command", "a",
                                      {"command": "echo 1"}))
        rows = self.agent.storage.query(
            "SELECT * FROM qtable WHERE state='act|system.command'")
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["q"], 0)


if __name__ == "__main__":
    unittest.main()
