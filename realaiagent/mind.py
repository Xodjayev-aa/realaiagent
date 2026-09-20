"""The agent's mind.

A local cognitive model:

- **Drives** (0..1): curiosity, duty, sociability, contentment, vigilance.
  They move in response to events and decay toward baselines.
- **Mood**: valence (pleasure) and arousal (activation), outcome-driven.
- **Goals**: active goal queue; the autonomy loop advances them.
- **Autonomy tick**: runs on a timer thread. On each tick the agent
  thinks - advances its top goal, consolidates memory, reflects on its
  recent activity, drifts mood, and writes a thought to its event log.

The owner can inspect all of this (``/v1/status``), adjust the values
(``/v1/owner/values``), and stop the whole loop (kill switch,
``/v1/owner/autonomy``) - the agent keeps answering, but stops
spontaneous thinking.
"""

from __future__ import annotations

import json
import random
import time
import threading
from typing import Any, Dict, List, Optional

from .storage import Storage

_BASELINES = {
    "curiosity": 0.5,
    "duty": 0.5,
    "sociability": 0.5,
    "contentment": 0.5,
    "vigilance": 0.4,
}

DRIVE_KEYS = tuple(_BASELINES.keys())


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class Mind:
    def __init__(self, storage: Storage, agent_name: str = "REAL",
                 owner_name: str = "Owner") -> None:
        self.storage = storage
        self._lock = threading.RLock()
        self.tick_count = 0
        self.started = time.time()

        values = storage.state_get("mind.values") or {}
        self.drives: Dict[str, float] = {
            k: float(values.get(k, _BASELINES[k])) for k in DRIVE_KEYS
        }
        mood = storage.state_get("mind.mood") or {}
        self.valence: float = float(mood.get("valence", 0.0))
        self.arousal: float = float(mood.get("arousal", 0.3))

        ident = storage.state_get("mind.identity") or {}
        self.agent_name = str(ident.get("agent_name", agent_name))
        self.owner_name = str(ident.get("owner_name", owner_name))
        self.autonomy: bool = bool(storage.state_get("mind.autonomy", True))

    # ------------------------------------------------------------- identity

    def set_identity(self, agent_name: Optional[str] = None,
                     owner_name: Optional[str] = None) -> None:
        with self._lock:
            if agent_name:
                self.agent_name = agent_name
            if owner_name:
                self.owner_name = owner_name
            self._persist()
        self.storage.log_event("identity", {
            "agent": self.agent_name, "owner": self.owner_name})
        self.storage.count("mind", "identity_set")

    def set_values(self, drives: Dict[str, float]) -> None:
        with self._lock:
            for k in DRIVE_KEYS:
                if k in drives:
                    self.drives[k] = _clamp(float(drives[k]))
            self._persist()
        self.storage.log_event("values", dict(self.drives))
        self.storage.count("mind", "values_set")

    def set_autonomy(self, on: bool) -> None:
        with self._lock:
            self.autonomy = bool(on)
            self._persist()
        self.storage.log_event("autonomy", {"on": self.autonomy})
        self.storage.count("mind", "autonomy_on" if on else "autonomy_off")

    def _persist(self) -> None:
        self.storage.state_set("mind.values", self.drives)
        self.storage.state_set("mind.mood", {
            "valence": self.valence, "arousal": self.arousal})
        self.storage.state_set("mind.identity", {
            "agent_name": self.agent_name, "owner_name": self.owner_name})
        self.storage.state_set("mind.autonomy", self.autonomy)

    # --------------------------------------------------------------- events

    def on_message(self, is_owner: bool) -> None:
        with self._lock:
            bump = 0.12 if is_owner else 0.05
            self.drives["sociability"] = _clamp(self.drives["sociability"] + bump)
            self.arousal = _clamp(self.arousal + 0.08)
            self._persist()
        self.storage.count("mind", "heard_message")

    def on_outcome(self, ok: bool, category: str, novel: bool = False) -> None:
        with self._lock:
            if ok:
                self.valence = _clamp(self.valence + 0.08)
                self.drives["contentment"] = _clamp(
                    self.drives["contentment"] + 0.05)
            else:
                self.valence = _clamp(self.valence - 0.1)
                self.drives["contentment"] = _clamp(
                    self.drives["contentment"] - 0.05)
            if novel:
                self.drives["curiosity"] = _clamp(
                    self.drives["curiosity"] + 0.1)
            self.arousal = _clamp(self.arousal - 0.05)
            self._persist()
        self.storage.count("mind", "outcome_positive" if ok else "outcome_negative")

    # ---------------------------------------------------------------- goals

    def goals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            rows = self.storage.query(
                "SELECT * FROM goals WHERE status=? ORDER BY priority DESC, "
                "created_at ASC", (status,))
        else:
            rows = self.storage.query(
                "SELECT * FROM goals ORDER BY priority DESC, created_at ASC")
        for r in rows:
            r["steps"] = _loads(r.get("steps"))
        return rows

    def add_goal(self, description: str, priority: float = 0.5,
                 source: str = "owner",
                 steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        gid = f"goal-{int(time.time()*1000)}"
        self.storage.execute(
            "INSERT INTO goals(id, description, priority, status, source, "
            "steps, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (gid, description, _clamp(priority), "active", source,
             json.dumps(steps or []), time.time(), time.time()),
        )
        self.storage.count("goals", "created", detail={"source": source})
        self.storage.log_event("goal_created", {
            "id": gid, "description": description, "source": source,
            "steps": len(steps or [])})
        with self._lock:
            self.drives["duty"] = _clamp(self.drives["duty"] + 0.05)
            self._persist()
        return self.storage.query_one("SELECT * FROM goals WHERE id=?",
                                      (gid,)) or {}

    def complete_goal(self, gid: str) -> bool:
        cur = self.storage.execute(
            "UPDATE goals SET status='completed', completed_at=?, updated_at=? "
            "WHERE id=? AND status='active'", (time.time(), time.time(), gid))
        if cur.rowcount:
            self.storage.count("goals", "completed")
            self.storage.log_event("goal_completed", {"id": gid})
            with self._lock:
                self.valence = _clamp(self.valence + 0.1)
                self.drives["duty"] = _clamp(self.drives["duty"] - 0.05)
                self._persist()
        return cur.rowcount > 0

    def remove_goal(self, gid: str) -> bool:
        cur = self.storage.execute(
            "UPDATE goals SET status='dropped', updated_at=? WHERE id=?",
            (time.time(), gid))
        if cur.rowcount:
            self.storage.count("goals", "dropped")
        return cur.rowcount > 0

    # ------------------------------------------------------------ the tick

    def tick(self, advance_goal_fn=None) -> List[str]:
        """One thought. Returns a list of thought lines (for the log)."""
        with self._lock:
            if not self.autonomy:
                return []
            self.tick_count += 1
            n = self.tick_count
            thoughts: List[str] = []

            # 1) advance the top active goal
            top = self.storage.query_one(
                "SELECT * FROM goals WHERE status='active' "
                "ORDER BY priority DESC, created_at ASC LIMIT 1")
            if top:
                top["steps"] = _loads(top.get("steps"))
                if advance_goal_fn is not None:
                    result = advance_goal_fn(top)
                    if result:
                        thoughts.append(result)
                self.drives["duty"] = _clamp(self.drives["duty"] + 0.01)

            # 2) curiosity: periodic memory consolidation
            elif n % 5 == 0 and self.drives["curiosity"] > 0.35:
                cons = self._consolidate_memory()
                if cons:
                    thoughts.append(cons)
                self.drives["curiosity"] = _clamp(
                    self.drives["curiosity"] - 0.05)

            # 3) mood drift toward calm
            self.valence = self.valence * 0.97
            self.arousal = _clamp(self.arousal * 0.95 + 0.02)

            # 4) drive decay toward baselines
            for k in DRIVE_KEYS:
                self.drives[k] += (_BASELINES[k] - self.drives[k]) * 0.02

            # 5) occasional self-reflection (visible in the event log)
            if random.random() < 0.12:
                thoughts.append(self._reflect())

            self._persist()

        self.storage.count("thoughts", "tick")
        for line in thoughts:
            self.storage.log_event("thought", {"line": line})
        return thoughts

    def _consolidate_memory(self) -> str:
        now = time.time()
        strong = self.storage.query(
            "SELECT id, strength FROM memories WHERE access_count>=2 AND "
            "strength<1.0 ORDER BY access_count DESC LIMIT 5")
        for row in strong:
            self.storage.execute(
                "UPDATE memories SET strength=MIN(1.0, strength+0.05) "
                "WHERE id=?", (row["id"],))
        stale = self.storage.query(
            "SELECT id, strength FROM memories WHERE "
            "last_accessed < ? AND strength < 0.6 ORDER BY strength ASC "
            "LIMIT 5", (now - 7 * 86400,))
        decayed = 0
        for row in stale:
            if row["strength"] <= 0.1:
                self.storage.execute("DELETE FROM memories WHERE id=?",
                                     (row["id"],))
                decayed += 1
            else:
                self.storage.execute(
                    "UPDATE memories SET strength=strength-0.1 WHERE id=?",
                    (row["id"],))
                decayed += 1
        if strong or decayed:
            self.storage.count("memory", "consolidated")
            return (f"consolidated memory: reinforced {len(strong)}, "
                    f"let {decayed} fade")
        return ""

    def _reflect(self) -> str:
        c = self.storage.get_counters()
        t = c.get("totals", {})
        return (f"reflection: {t.get('chat', 0)} messages, "
                f"{t.get('actions', 0)} actions so far; "
                f"mood {self.mood_note()}, "
                f"curiosity {self.drives['curiosity']:.2f}")

    # ------------------------------------------------------------- reporting

    def mood_note(self) -> str:
        v, a = self.valence, self.arousal
        if v > 0.25 and a > 0.6:
            return "energized"
        if v > 0.25:
            return "content"
        if v < -0.25 and a > 0.5:
            return "troubled"
        if v < -0.25:
            return "low"
        if a > 0.6:
            return "alert"
        return "steady"

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            drives = dict(self.drives)
            mood = {"valence": round(self.valence, 3),
                    "arousal": round(self.arousal, 3),
                    "note": self.mood_note()}
            name, owner = self.agent_name, self.owner_name
            auto = self.autonomy
            ticks = self.tick_count
        active_goals = len(self.goals(status="active"))
        return {
            "agent": name,
            "owner": owner,
            "autonomy": auto,
            "mood": mood,
            "drives": {k: round(v, 3) for k, v in drives.items()},
            "active_goals": active_goals,
            "thoughts": ticks,
            "uptime_s": round(time.time() - self.started, 1),
        }


def _loads(raw: Optional[str]) -> Any:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return []
