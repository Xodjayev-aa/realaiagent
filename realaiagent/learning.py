"""The agent's built-in learning.

Four mechanisms, all local and persistent:

1. **Intent learning** - SGD updates to the local intent model
   (``/v1/learn`` or the chat ``teach:`` syntax).
2. **Q-learning** - a state-action value table for action selection,
   updated on every executed action (success +1 / failure -1).
3. **Skill memory** - named procedures. Owner-defined, or auto-extracted
   from successful multi-step episodes.
4. **Episodic/semantic memory** - key-value memories with strength,
   consolidated by the mind.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .nlp import IntentModel
from .storage import Storage


class Learner:
    def __init__(self, storage: Storage, model: IntentModel) -> None:
        self.storage = storage
        self.model = model
        self.gamma = 0.8
        self.lr_q = 0.15

    # ------------------------------------------------------------- intents

    def train_intents(self, examples: List[Dict[str, Any]],
                      key_id: Optional[str] = None) -> Dict[str, Any]:
        trained = 0
        intents: List[str] = []
        for ex in examples:
            text = str(ex.get("text", "")).strip()
            intent = str(ex.get("intent", "")).strip().lower()
            if not text or not intent:
                continue
            self.model.learn(text, intent)
            # remember the association semantically
            self.remember("learning", f"intent:{intent}",
                          {"example": text}, key_id=key_id)
            intents.append(intent)
            trained += 1
        if trained:
            self.storage.state_set("nlp.model", self.model.to_json())
            self.storage.count("learning", "intent_examples",
                               key_id=key_id, detail={"n": trained})
            self.storage.log_event("learn_intents", {
                "n": trained, "intents": sorted(set(intents))},
                key_id=key_id)
        return {"trained": trained, "intents": sorted(set(intents))}

    # ------------------------------------------------------------ q-learning

    def q_update(self, state: str, action: str, reward: float) -> None:
        row = self.storage.query_one(
            "SELECT q, visits FROM qtable WHERE state=? AND action=?",
            (state, action))
        if row is None:
            q, visits = 0.0, 0
        else:
            q, visits = row["q"], row["visits"]
        q = (1 - self.lr_q) * q + self.lr_q * reward
        visits += 1
        self.storage.execute(
            "INSERT INTO qtable(state, action, q, visits, last_update) "
            "VALUES(?,?,?,?,?) ON CONFLICT(state, action) DO UPDATE SET "
            "q=excluded.q, visits=excluded.visits, last_update=excluded."
            "last_update",
            (state, action, q, visits, time.time()),
        )
        self.storage.count("learning", "q_update")

    def best_action(self, state: str) -> Optional[str]:
        row = self.storage.query_one(
            "SELECT action, q FROM qtable WHERE state=? ORDER BY q DESC "
            "LIMIT 1", (state,))
        return row["action"] if row else None

    # ---------------------------------------------------------------- skills

    def skills(self) -> List[Dict[str, Any]]:
        rows = self.storage.query("SELECT * FROM skills ORDER BY updated_at DESC")
        for r in rows:
            r["steps"] = _loads(r.get("steps"))
            r["examples"] = _loads(r.get("examples"))
        return rows

    def learn_skill(self, name: str, steps: List[Dict[str, Any]],
                    description: str = "",
                    examples: Optional[List[str]] = None,
                    key_id: Optional[str] = None) -> Dict[str, Any]:
        name = _slug(name)
        if not name or not steps:
            raise ValueError("skill needs a name and at least one step")
        now = time.time()
        existing = self.storage.query_one("SELECT * FROM skills WHERE name=?",
                                          (name,))
        if existing:
            self.storage.execute(
                "UPDATE skills SET steps=?, description=?, examples=?, "
                "updated_at=? WHERE name=?",
                (json.dumps(steps), description,
                 json.dumps(examples or existing["examples"]), now, name))
            self.storage.count("learning", "skill_updated", key_id=key_id)
        else:
            self.storage.execute(
                "INSERT INTO skills(name, description, steps, examples, "
                "created_at, updated_at) VALUES(?,?,?,?,?,?)",
                (name, description, json.dumps(steps),
                 json.dumps(examples or []), now, now))
            self.storage.count("learning", "skill_created", key_id=key_id)
        self.storage.log_event("learn_skill", {
            "name": name, "steps": len(steps)}, key_id=key_id)
        return self.storage.query_one("SELECT * FROM skills WHERE name=?",
                                      (name,)) or {}

    def remove_skill(self, name: str) -> bool:
        cur = self.storage.execute("DELETE FROM skills WHERE name=?",
                                   (_slug(name),))
        if cur.rowcount:
            self.storage.count("learning", "skill_removed")
        return cur.rowcount > 0

    def skill_steps(self, name: str) -> List[Dict[str, Any]]:
        row = self.storage.query_one("SELECT steps FROM skills WHERE name=?",
                                     (_slug(name),))
        return _loads(row["steps"]) if row else []

    def skill_names(self) -> List[str]:
        rows = self.storage.query("SELECT name FROM skills")
        return [r["name"] for r in rows]

    def record_skill_outcome(self, name: str, ok: bool) -> None:
        col = "success_count" if ok else "fail_count"
        self.storage.execute(
            f"UPDATE skills SET {col}={col}+1, updated_at=? WHERE name=?",
            (time.time(), _slug(name)))

    # --------------------------------------------------- episodic memory

    def remember(self, kind: str, key: str, value: Any,
                 strength: float = 1.0, key_id: Optional[str] = None) -> None:
        now = time.time()
        row = self.storage.query_one(
            "SELECT id FROM memories WHERE kind=? AND key=?", (kind, key))
        if row:
            self.storage.execute(
                "UPDATE memories SET value=?, strength=MIN(1.0, strength+0.1), "
                "last_accessed=?, access_count=access_count+1 WHERE id=?",
                (json.dumps(value), now, row["id"]))
        else:
            self.storage.execute(
                "INSERT INTO memories(kind, key, value, strength, created_at, "
                "last_accessed, access_count) VALUES(?,?,?,?,?,?,0)",
                (kind, key, json.dumps(value), strength, now, now))
            self.storage.count("memory", "stored", key_id=key_id)

    def recall(self, kind: str, key: str) -> Optional[Any]:
        row = self.storage.query_one(
            "SELECT value FROM memories WHERE kind=? AND key=?", (kind, key))
        if row is None:
            return None
        self.storage.execute(
            "UPDATE memories SET last_accessed=?, access_count=access_count+1 "
            "WHERE kind=? AND key=?", (time.time(), kind, key))
        return _loads(row["value"])

    # ------------------------------------------------------------- episodes

    def record_episode(self, message: str, response: str,
                       intent: str, success: bool,
                       plan: Optional[Dict[str, Any]] = None,
                       actions: Optional[List[Dict[str, Any]]] = None,
                       duration_ms: float = 0.0,
                       sender: str = "user") -> str:
        eid = f"ep-{int(time.time()*1000)}-{len(message) % 1000}"
        self.storage.execute(
            "INSERT INTO conversations(id, ts, sender, message, response, "
            "intent, confidence, plan, actions, success, duration_ms) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (eid, time.time(), sender, message, response, intent, 0.0,
             json.dumps(plan or {}), json.dumps(actions or []),
             1 if success else 0, duration_ms),
        )
        # episodic memory of the interaction (for the mind to consolidate)
        self.remember("episodic", eid, {
            "message": message[:200], "intent": intent, "success": success,
        }, strength=0.6)
        # auto-skill extraction: successful multi-step episodes become
        # reusable procedures
        if success and plan and plan.get("steps") and len(plan["steps"]) > 1:
            name = _slug(message[:40]) or "procedure"
            if not self.storage.query_one("SELECT name FROM skills "
                                          "WHERE name=?", (name,)):
                try:
                    self.learn_skill(name, plan["steps"],
                                     description=f"learned from: {message[:80]}",
                                     examples=[message])
                    self.storage.count("learning", "skill_auto")
                except ValueError:
                    pass
        return eid

    def reflect_on_failure(self, plan: Dict[str, Any],
                           actions: List[Dict[str, Any]]) -> Optional[str]:
        """After a failure: store a lesson so the policy avoids it."""
        failed = [a for a in actions if not a.get("ok") and a.get("denied")]
        if not failed:
            return None
        a = failed[0]
        lesson = (f"avoid repeating: {a.get('category')} "
                  f"{a.get('action')} failed (denied)")
        self.remember("semantic", f"lesson:{a.get('category')}:"
                                  f"{a.get('action', '')[:32]}", lesson)
        self.storage.count("learning", "lesson")
        return lesson


def _loads(raw: Optional[str]) -> Any:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return []


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:48]
