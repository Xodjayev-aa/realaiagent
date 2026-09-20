import unittest

from .helpers import make_agent, owner_chat


class TestLearning(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.learner = self.agent.learner

    def test_intent_training_changes_prediction(self):
        for _ in range(8):
            self.learner.train_intents(
                [{"text": "brew a fresh pot", "intent": "coffee"}])
        intent, _conf, _top = self.agent.model.predict("brew a fresh pot")
        self.assertEqual(intent, "coffee")

    def test_intent_training_persisted(self):
        self.learner.train_intents(
            [{"text": "special persistence phrase", "intent": "persist"}])
        from realaiagent.nlp import IntentModel
        m2 = IntentModel.from_json(
            self.agent.storage.state_get("nlp.model"))
        self.assertEqual(m2.predict("special persistence phrase")[0],
                         "persist")

    def test_q_learning_converges(self):
        l = self.learner
        for _ in range(30):
            l.q_update("s", "good", +1.0)
            l.q_update("s", "bad", -1.0)
        self.assertEqual(l.best_action("s"), "good")

    def test_skills_crud(self):
        s = self.learner.learn_skill(
            "My Skill",
            [{"category": "notify", "action": "notify:a",
              "params": {"message": "a"}}],
            description="test")
        self.assertEqual(s["name"], "my-skill")
        self.assertIn("my-skill", self.learner.skill_names())
        self.assertEqual(len(self.learner.skill_steps("my-skill")), 1)
        self.assertTrue(self.learner.remove_skill("my-skill"))
        self.assertNotIn("my-skill", self.learner.skill_names())

    def test_auto_skill_from_success_episode(self):
        plan = {"strategy": "chain", "steps": [
            {"category": "notify", "action": "notify:a",
             "params": {"message": "a"}},
            {"category": "notify", "action": "notify:b",
             "params": {"message": "b"}}]}
        self.learner.record_episode(
            "water the garden then check the greenhouse", "done",
            "command", True, plan=plan,
            actions=[{"ok": True}, {"ok": True}])
        names = self.learner.skill_names()
        self.assertTrue(any("water" in n for n in names))

    def test_remember_and_recall(self):
        self.learner.remember("semantic", "lamp:room", {"room": "living"})
        self.assertEqual(self.learner.recall("semantic", "lamp:room"),
                         {"room": "living"})
        self.assertIsNone(self.learner.recall("semantic", "nope"))

    def test_chat_teach_syntax(self):
        res = owner_chat(self.agent,
                         'teach: "power up the lamp" means command')
        self.assertIn("Learned", res["response"])
        intent, _, _ = self.agent.model.predict("power up the lamp")
        self.assertEqual(intent, "command")

    def test_chat_remember_freeform(self):
        res = owner_chat(self.agent,
                         "remember that the garage code is 4321")
        self.assertIn("semantic memory", res["response"])
        notes = self.agent.storage.query(
            "SELECT * FROM memories WHERE kind='semantic' AND key LIKE "
            "'note:%'")
        self.assertGreaterEqual(len(notes), 1)


if __name__ == "__main__":
    unittest.main()
