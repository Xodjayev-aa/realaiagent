import unittest

from realaiagent.nlp import (
    IntentModel, best_fuzzy_match, extract_number, extract_path,
    extract_verb, parse_message,
)


class TestVerbAndSlots(unittest.TestCase):
    def test_verb_on(self):
        self.assertEqual(extract_verb("please turn on the lamp"), "on")

    def test_verb_off_phrase(self):
        self.assertEqual(extract_verb("switch off the fan"), "off")

    def test_verb_set(self):
        self.assertEqual(extract_verb("set the temperature to 21"), "set")

    def test_verb_run(self):
        self.assertEqual(extract_verb("run the backup script"), "run")

    def test_no_verb(self):
        self.assertIsNone(extract_verb("hello there"))

    def test_number(self):
        self.assertEqual(extract_number("set it to 21.5 now"), 21.5)

    def test_path(self):
        self.assertEqual(extract_path("open /home/user/notes.md please"),
                         "/home/user/notes.md")
        self.assertEqual(extract_path("show me report.txt"), "report.txt")

    def test_fuzzy_device(self):
        cands = [("lamp-1", "Living Room Lamp light"),
                 ("fan-1", "Ceiling Fan fan")]
        self.assertEqual(best_fuzzy_match("turn on the lamp", cands),
                         "lamp-1")
        self.assertEqual(best_fuzzy_match("stop the ceiling fan", cands),
                         "fan-1")
        self.assertIsNone(best_fuzzy_match("hello", cands))


class TestIntentModel(unittest.TestCase):
    def test_seed_predicts_core_intents(self):
        m = IntentModel()
        m.seed()
        self.assertEqual(m.predict("hello")[0], "greet")
        self.assertEqual(m.predict("turn on the lamp")[0], "command")
        self.assertEqual(m.predict("status")[0], "status")

    def test_learning_shifts_confidence(self):
        m = IntentModel()
        m.seed()
        for i in range(6):
            m.learn("make the coffee strong", "coffee", lr=0.8)
        intent, conf, _ = m.predict("make the coffee strong")
        self.assertEqual(intent, "coffee")
        self.assertGreater(conf, 0.3)

    def test_persistence_roundtrip(self):
        m = IntentModel()
        m.seed()
        m.learn("special phrase alpha", "alpha")
        m2 = IntentModel.from_json(m.to_json())
        self.assertEqual(m2.predict("special phrase alpha")[0], "alpha")

    def test_parse_message_slots(self):
        m = IntentModel()
        m.seed()
        devices = [{"id": "lamp-1", "name": "Living Room Lamp", "kind": "light"}]
        p = parse_message(m, "turn on the lamp to 50% now", devices, [])
        self.assertEqual(p["intent"], "command")
        self.assertEqual(p["slots"]["verb"], "on")
        self.assertEqual(p["slots"]["device_id"], "lamp-1")
        self.assertEqual(p["slots"]["number"], 50.0)

    def test_parse_teach(self):
        m = IntentModel()
        m.seed()
        p = parse_message(m, 'teach: "water the plants" means garden',
                          [], [])
        self.assertEqual(p["intent"], "learn")
        self.assertEqual(p["slots"]["learn"]["intent"], "garden")

    def test_parse_approve_id(self):
        m = IntentModel()
        m.seed()
        p = parse_message(m, "approve 42", [], [])
        self.assertEqual(p["intent"], "approve")
        self.assertEqual(p["slots"]["approve_id"], 42)

    def test_parse_goal(self):
        m = IntentModel()
        m.seed()
        p = parse_message(m, "goal: water the garden every morning", [], [])
        self.assertEqual(p["intent"], "goal")
        self.assertIn("water the garden", p["slots"]["goal"])


if __name__ == "__main__":
    unittest.main()
