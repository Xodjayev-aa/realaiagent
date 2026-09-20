import unittest

from realaiagent.nlp import parse_message
from realaiagent.planner import Planner

from .helpers import make_agent, register_device


def _parse(agent, text, skills=None):
    return parse_message(agent.model, text, agent._devices(),
                         skills or agent.learner.skill_names())


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.planner = Planner()

    def test_single_device_step(self):
        register_device(self.agent)
        plan = self.planner.plan(_parse(self.agent, "turn on the lamp"),
                                 self.agent._devices())
        self.assertEqual(len(plan.steps), 1)
        s = plan.steps[0]
        self.assertEqual(s.category, "device.control")
        self.assertEqual(s.device_id, "lamp-1")
        self.assertEqual(s.params["action"], "on")
        self.assertEqual(plan.strategy, "single")

    def test_set_with_value(self):
        self.agent.storage.execute(
            "INSERT INTO devices(id,name,kind,capabilities,meta,created_at,"
            " last_seen) VALUES(?,?,?,?,?,?,?)",
            ("thermo-1", "Thermostat", "climate",
             '[{"action":"set","executor":"command","template":"echo {level}"}]',
             "{}", 1, 1))
        plan = self.planner.plan(_parse(self.agent, "set the thermostat to 21"),
                                 self.agent._devices())
        self.assertEqual(plan.steps[0].params["action"], "set")
        self.assertEqual(plan.steps[0].params["level"], 21)

    def test_chain_of_two(self):
        register_device(self.agent)
        self.agent.storage.execute(
            "INSERT INTO devices(id,name,kind,capabilities,meta,created_at,"
            " last_seen) VALUES(?,?,?,?,?,?,?)",
            ("fan-1", "Ceiling Fan", "fan",
             '[{"action":"off","executor":"command","template":"echo off"}]',
             "{}", 1, 1))
        plan = self.planner.plan(
            _parse(self.agent, "turn on the lamp then turn off the fan"),
            self.agent._devices())
        self.assertEqual(plan.strategy, "chain")
        self.assertEqual(len(plan.steps), 2)

    def test_run_command(self):
        plan = self.planner.plan(_parse(self.agent, "run: git status"),
                                 [])
        self.assertEqual(plan.steps[0].category, "system.command")
        self.assertEqual(plan.steps[0].params["command"], "git status")

    def test_file_read(self):
        plan = self.planner.plan(_parse(self.agent, "open /home/user/notes.md"),
                                 [])
        self.assertEqual(plan.steps[0].category, "files.read")

    def test_no_device_no_verb_clarifies(self):
        plan = self.planner.plan(_parse(self.agent, "make it so"),
                                 [])
        self.assertTrue(plan.empty)

    def test_skill_plan(self):
        self.agent.learner.learn_skill(
            "morning routine",
            [{"category": "device.control", "action": "lamp-1:on",
              "params": {"action": "on"}, "device_id": "lamp-1"},
             {"category": "notify", "action": "notify:good morning",
              "params": {"message": "good morning"}}])
        parsed = _parse(self.agent, "do the morning routine")
        self.assertEqual(parsed["slots"]["skill"], "morning-routine")
        # full engine path resolves the skill's steps
        plan = self.agent._make_plan(parsed)
        self.assertEqual(plan.strategy, "skill")
        self.assertEqual(len(plan.steps), 2)


if __name__ == "__main__":
    unittest.main()
