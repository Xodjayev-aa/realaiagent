import unittest

from .helpers import make_agent


class TestMind(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.mind = self.agent.mind

    def test_snapshot_fields(self):
        s = self.mind.snapshot()
        for k in ("agent", "owner", "autonomy", "mood", "drives",
                  "active_goals", "thoughts"):
            self.assertIn(k, s)

    def test_identity_and_values_persist(self):
        self.mind.set_identity(agent_name="MIRA", owner_name="Alex")
        self.mind.set_values({"curiosity": 0.9, "duty": 0.1})
        # reload from the same storage
        from realaiagent.mind import Mind
        m2 = Mind(self.agent.storage)
        self.assertEqual(m2.agent_name, "MIRA")
        self.assertEqual(m2.owner_name, "Alex")
        self.assertAlmostEqual(m2.drives["curiosity"], 0.9, places=2)

    def test_mood_moves_with_outcomes(self):
        v0 = self.mind.valence
        self.mind.on_outcome(True, "x")
        self.assertGreater(self.mind.valence, v0)
        v1 = self.mind.valence
        self.mind.on_outcome(False, "x")
        self.assertLess(self.mind.valence, v1)

    def test_goals_lifecycle(self):
        g = self.mind.add_goal("test goal", priority=0.7)
        self.assertEqual(self.mind.goals(status="active")[0]["id"], g["id"])
        self.assertTrue(self.mind.complete_goal(g["id"]))
        self.assertEqual(self.mind.goals(status="active"), [])
        self.assertFalse(self.mind.complete_goal(g["id"]))

    def test_autonomy_kill_switch(self):
        self.mind.set_autonomy(False)
        self.assertEqual(self.mind.tick(), [])
        self.mind.set_autonomy(True)
        self.assertIsInstance(self.mind.tick(), list)

    def test_tick_counts_thoughts(self):
        self.mind.tick()
        self.mind.tick()
        c = self.agent.storage.get_counters()
        self.assertGreaterEqual(c["totals"].get("thoughts", 0), 2)

    def test_goal_advancement(self):
        g = self.mind.add_goal("do the thing", priority=0.9,
                               steps=[{"category": "notify",
                                       "action": "notify:done",
                                       "params": {"message": "done"}}])
        thoughts = self.mind.tick(
            advance_goal_fn=self.agent._advance_top_goal)
        self.assertTrue(any("step 1/1" in t for t in thoughts))
        # second tick completes the goal
        self.mind.tick(advance_goal_fn=self.agent._advance_top_goal)
        self.assertEqual(self.mind.goals(status="active"), [])
        goals = self.mind.goals()
        self.assertEqual(goals[0]["status"], "completed")

    def test_memory_consolidation(self):
        for i in range(3):
            self.agent.learner.remember("episodic", f"m{i}", {"i": i},
                                        strength=0.5)
        # make them frequently accessed and a week old
        self.agent.storage.execute(
            "UPDATE memories SET access_count=5, "
            "last_accessed=last_accessed-8*86400 WHERE kind='episodic'")
        self.mind.drives["curiosity"] = 0.9
        line = self.mind._consolidate_memory()
        self.assertIn("consolidated", line)
        c = self.agent.storage.get_counters()
        self.assertGreaterEqual(c["totals"].get("memory", 0), 1)


if __name__ == "__main__":
    unittest.main()
