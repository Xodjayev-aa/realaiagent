"""Tests for the stdlib ReAct loop (no model server needed)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from realaiagent.config import Config
from realaiagent.react import (
    ReActAgent, ToolRegistry, arena_agent_handler, parse_action,
    parse_thought, safe_calculate,
)


def _resp(thought: str, name: str, **args) -> str:
    return (f"THOUGHT: {thought}\nACTION:\n```json\n"
            f"{json.dumps({'name': name, 'arguments': args})}\n```")


class ScriptedBackend:
    """Replays canned responses and records what the loop sent it."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def __call__(self, messages):
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0) if self.responses else \
            _resp("out of script", "final_answer", output="?")


class ReactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="realai-react-"))
        self.cfg = Config(data_dir=self.tmp, tick_seconds=3600)
        self.tools = ToolRegistry(self.cfg)

    # --- calculator -----------------------------------------------------
    def test_safe_calculate(self):
        self.assertEqual(safe_calculate("(142 * 3) / 2"), 213)
        self.assertEqual(safe_calculate("-3 + 2 ** 3 % 5"), 0)
        for bad in ("__import__('os')", "a + 1", "1 if 1 else 2", "(1).real",
                    "2 ** 99999", ""):
            with self.assertRaises((ValueError, SyntaxError)):
                safe_calculate(bad)

    def test_calculator_tool_messages(self):
        self.assertEqual(self.tools.execute("calculate_expression",
                                            {"expression": "(44 * 2) + 12"}),
                         "Success: Result is 100")
        self.assertIn("division by zero",
                      self.tools.execute("calculate_expression", {"expression": "1/0"}))
        self.assertIn("Execution Error",
                      self.tools.execute("calculate_expression",
                                         {"expression": "__import__('os')"}))

    # --- file tool ------------------------------------------------------
    def test_read_file_inside_root(self):
        (self.tmp / "config.json").write_text('{"configuration_v2": "Active"}')
        out = self.tools.execute("read_local_file", {"file_path": "config.json"})
        self.assertIn("configuration_v2", out)

    def test_read_file_outside_root_blocked(self):
        out = self.tools.execute("read_local_file", {"file_path": "/etc/passwd"})
        self.assertIn("outside allowed roots", out)

    def test_unknown_tool(self):
        self.assertIn("not registered", self.tools.execute("nope", {}))

    # --- parser ---------------------------------------------------------
    def test_parse_action_fenced_and_inline(self):
        fenced = _resp("t", "calculate_expression", expression="1+1")
        self.assertEqual(parse_action(fenced)["arguments"]["expression"], "1+1")
        self.assertEqual(parse_thought(fenced), "t")
        inline = 'THOUGHT: x\nACTION: {"name": "final_answer", "arguments": {"output": "ok"}} trailing'
        self.assertEqual(parse_action(inline)["name"], "final_answer")
        self.assertIsNone(parse_action("THOUGHT: no action here"))
        self.assertIsNone(parse_action("ACTION: ```json\n{not json}\n```"))

    # --- loop -----------------------------------------------------------
    def test_full_loop_think_act_observe(self):
        (self.tmp / "config.json").write_text("configuration_v2=Active")
        backend = ScriptedBackend([
            _resp("read first", "read_local_file", file_path="config.json"),
            _resp("now math", "calculate_expression", expression="(142 * 3) / 2"),
            _resp("done", "final_answer", output="config active; result 213"),
        ])
        agent = ReActAgent(backend, self.tools, max_iterations=5)
        res = agent.run("Read config.json and calculate (142 * 3) / 2")
        self.assertTrue(res.completed)
        self.assertEqual(res.iterations, 3)
        self.assertEqual(res.output, "config active; result 213")
        # observations were fed back into memory
        self.assertIn("configuration_v2=Active", backend.seen[1][-1]["content"])
        self.assertIn("Result is 213", backend.seen[2][-1]["content"])
        self.assertEqual(backend.seen[0][0]["role"], "system")

    def test_malformed_action_gets_correction_and_continues(self):
        backend = ScriptedBackend([
            "THOUGHT: I forgot the action",
            _resp("ok", "final_answer", output="fixed"),
        ])
        res = ReActAgent(backend, self.tools, max_iterations=3).run("x")
        self.assertTrue(res.completed)
        self.assertIsNone(res.steps[0].action)
        self.assertIn("ACTION section was missing", backend.seen[1][-1]["content"])

    def test_iteration_budget(self):
        backend = ScriptedBackend(
            [_resp("loop", "calculate_expression", expression="1")] * 10)
        res = ReActAgent(backend, self.tools, max_iterations=2).run("x")
        self.assertFalse(res.completed)
        self.assertEqual(res.iterations, 2)
        self.assertIn("maximum iterations", res.output)

    # --- handler --------------------------------------------------------
    def test_arena_handler(self):
        backend = ScriptedBackend([_resp("d", "final_answer", output="42")])
        out = arena_agent_handler({"prompt": "answer", "max_steps": 3},
                                  backend=backend, cfg=self.cfg)
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["output"], "42")
        self.assertEqual(out["iterations"], 1)
        self.assertEqual(len(out["trace"]), 1)

    def test_backend_error_aborts_cleanly(self):
        from realaiagent.react import BackendError

        def dead(_messages):
            raise BackendError("connection refused")

        out = arena_agent_handler({"prompt": "x"}, backend=dead, cfg=self.cfg)
        self.assertEqual(out["status"], "incomplete")
        self.assertIn("connection refused", out["output"])
        self.assertEqual(out["iterations"], 0)

    def test_arena_handler_requires_prompt(self):
        out = arena_agent_handler({}, backend=ScriptedBackend([]), cfg=self.cfg)
        self.assertEqual(out["status"], "error")


if __name__ == "__main__":
    unittest.main()
