"""The agent engine.

Pipeline for every message:

    perceive (NLP parse)
      -> attend (mind)
        -> plan
          -> decide (permission gate)
            -> act (execute, count, learn)
              -> respond (compose honest text)
                -> reflect (mind + learning)

Also runs the autonomy loop (the agent's "own mind" ticking) and exposes
owner controls.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional

from .actions.base import ActionRequest
from .actions.executor import ActionExecutor
from .actions.permissions import PermissionManager
from .api.auth import KeyManager
from .config import Config
from .conversation import ConversationManager
from .email import EmailNotifier
from .learning import Learner
from .mind import Mind
from .nlp import IntentModel, parse_message
from .planner import Plan, Planner, Step
from .storage import Storage
from .telegram import Telegram
from .users import UserManager


class Agent:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.storage = Storage(cfg.db_path)
        self.mind = Mind(self.storage, cfg.agent_name, cfg.owner_name)
        raw_model = self.storage.state_get("nlp.model")
        self.model = IntentModel.from_json(raw_model) if raw_model else IntentModel()
        if not self.model.priors:
            self.model.seed()
            self.storage.state_set("nlp.model", self.model.to_json())
        self.learner = Learner(self.storage, self.model)
        self.planner = Planner()
        self.permissions = PermissionManager(self.storage)
        self.keys = KeyManager(self.storage, cfg.pepper_path, cfg)
        self.users = UserManager(self.storage, self.keys, cfg)
        self.executor = ActionExecutor(self.storage, self.permissions, cfg)
        self.telegram = Telegram(cfg.telegram_bot_token,
                                 cfg.telegram_chat_id, self.storage)
        self.users.notifier = self.telegram
        # The owner hears about access requests on BOTH channels: their own
        # Telegram bot and their own SMTP inbox (stdlib, no mail SaaS).
        self.mailer = EmailNotifier.from_config(cfg, self.storage)
        self.users.mailer = self.mailer
        # The product's voice: multi-turn context, memory recall, honest
        # scope. The pipeline above stays exactly as it was for API clients.
        self.conversation = ConversationManager(self)
        self.executor.on_q_update = self.learner.q_update
        self.executor.state_builder = self._q_state
        self._msg_lock = threading.Lock()
        self._stop = threading.Event()
        self._tick_thread: Optional[threading.Thread] = None
        # seed default policies: everything starts as "ask" implicitly;
        # record the default explicitly so the owner can see it.
        self._seed_policies()

    def _seed_policies(self) -> None:
        if not self.permissions.policies():
            self.permissions.set_policy("notify", "allow",
                                        note="harmless by default")
            self.permissions.set_policy("timer.set", "allow",
                                        note="harmless by default")

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._stop.clear()
        self._tick_thread = threading.Thread(
            target=self._tick_loop, name="autonomy", daemon=True)
        self._tick_thread.start()
        self.storage.log_event("boot", {
            "agent": self.mind.agent_name,
            "ts": time.time()})
        self.storage.count("system", "boot")

    def stop(self) -> None:
        self._stop.set()
        if self._tick_thread:
            self._tick_thread.join(timeout=2.0)

    def _tick_loop(self) -> None:
        while not self._stop.wait(self.cfg.tick_seconds):
            try:
                self.mind.tick(advance_goal_fn=self._advance_top_goal)
            except Exception:  # noqa: BLE001 - autonomy must never crash
                self.storage.count("errors", "tick")

    # ------------------------------------------------------------- devices

    def _devices(self) -> List[Dict[str, Any]]:
        rows = self.storage.query("SELECT * FROM devices ORDER BY created_at")
        for r in rows:
            try:
                r["capabilities"] = __import__("json").loads(
                    r.get("capabilities") or "[]")
            except ValueError:
                r["capabilities"] = []
        return rows

    # ------------------------------------------------------------ pipeline

    def handle_message(self, text: str, sender: str = "user",
                       scopes: Optional[List[str]] = None,
                       key_id: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.time()
        scopes = scopes or []
        is_owner = "owner" in scopes
        self.storage.count("chat", "message", key_id=key_id)
        with self._msg_lock:
            parsed = self._parse(text)
            self.mind.on_message(is_owner)

            intent = parsed["intent"]
            slots = parsed["slots"]

            # owner-only chat commands first
            if intent in ("approve", "deny"):
                response, meta = self._handle_permission_chat(
                    parsed, scopes, key_id)
            elif intent == "learn" and slots.get("learn"):
                response, meta = self._handle_learn_chat(
                    slots["learn"], key_id)
            elif intent == "learn":
                # free-form "remember that …" -> semantic memory
                key = "note:" + hashlib.sha1(text.encode()).hexdigest()[:12]
                self.learner.remember("semantic", key, {"note": text},
                                      key_id=key_id)
                self.storage.count("learning", "semantic_note")
                response = "Noted — I've stored that in my semantic memory."
                meta = {"plan": Plan([], "remember", 1.0).to_dict(),
                        "actions": []}
            elif intent == "goal" and slots.get("goal"):
                response, meta = self._handle_goal_chat(
                    slots["goal"], parsed, scopes, key_id)
            elif intent in ("greet", "thanks", "status", "counters",
                            "chat", "command"):
                plan = self._make_plan(parsed)
                response, meta = self._respond(parsed, plan, scopes, key_id)
            else:
                plan = Plan([], "none", parsed["confidence"])
                response, meta = self._respond(parsed, plan, scopes, key_id)

            success = self._episode_success(meta)
            duration = (time.time() - t0) * 1000
            self.learner.record_episode(
                text, response, intent, success,
                plan=meta.get("plan", {}).to_dict() if hasattr(
                    meta.get("plan"), "to_dict") else meta.get("plan"),
                actions=meta.get("actions", []),
                duration_ms=duration, sender=sender)
            self.storage.count("decisions", "completed",
                               key_id=key_id,
                               detail={"intent": intent,
                                       "success": int(success)})
            return {
                "response": response,
                "meta": {
                    "intent": intent,
                    "confidence": parsed["confidence"],
                    "top_k": parsed["top_k"],
                    "plan": meta.get("plan"),
                    "actions": meta.get("actions", []),
                    "request_id": meta.get("request_id"),
                    "success": success,
                    "duration_ms": round(duration, 1),
                },
            }

    # ------------------------------------------------------- conversation

    def reply(self, text: str, session_id: Optional[str] = None,
              sender: str = "web", scopes: Optional[List[str]] = None,
              key_id: Optional[str] = None,
              visitor: str = "") -> Dict[str, Any]:
        """Talk to the agent the way the web app does.

        Same pipeline as :meth:`handle_message`, wrapped in the
        conversational voice: multi-turn context for this ``session_id``,
        memory recall, and honest scope. ``handle_message`` stays raw for
        API clients that want the pipeline's own words.
        """
        return self.conversation.reply(
            text, session_id=session_id, sender=sender, scopes=scopes,
            key_id=key_id, visitor=visitor)

    def _parse(self, text: str) -> Dict[str, Any]:
        return parse_message(self.model, text, self._devices(),
                             self.learner.skill_names())

    def _q_state(self, req: "ActionRequest") -> str:
        return f"act|{req.category}"

    def _make_plan(self, parsed: Dict[str, Any]) -> Plan:
        devices = self._devices()
        skill = parsed["slots"].get("skill")
        steps = self.learner.skill_steps(skill) if skill else None
        return self.planner.plan(parsed, devices, steps)

    # ----------------------------------------------------------- responding

    def _respond(self, parsed: Dict[str, Any], plan: Plan,
                 scopes: List[str], key_id: Optional[str]
                 ) -> "tuple[str, Dict[str, Any]]":
        intent = parsed["intent"]
        actions: List[Dict[str, Any]] = []
        pending_ids: List[int] = []

        if not plan.empty:
            for step in plan.steps:
                req = ActionRequest(
                    category=step.category, action=step.action,
                    params=step.params, device_id=step.device_id,
                    source="chat",
                    reason=step.note or parsed.get("text", "")[:80])
                outcome = self.executor.execute(req, key_id=key_id)
                actions.append({**outcome.to_dict(),
                                "category": step.category,
                                "action": step.action,
                                "note": step.note})
                self.mind.on_outcome(outcome.ok, step.category, novel=False)
                if outcome.awaiting and outcome.pending_id:
                    pending_ids.append(outcome.pending_id)
                if outcome.denied:
                    self.learner.record_episode(
                        parsed.get("text", ""), "(denied)", intent, False,
                        plan=plan.to_dict(), actions=[actions[-1]])

        if intent == "status":
            text = self._status_text()
        elif intent == "counters":
            text = self._counters_text()
        elif intent == "greet":
            text = self._greet_text()
        elif intent == "thanks":
            text = self._thanks_text()
        elif intent == "chat":
            text = self._chat_text(parsed)
        elif plan.empty:
            text = (f"I'm not sure what you'd like me to do there. "
                    f"I can control devices, run commands, read/write "
                    f"files (with your permission), and I'm trainable - "
                    f"say: turn on the lamp / run: <command> / "
                    f'teach: "<sentence>" means <intent>.')
        else:
            text = self._action_text(plan, actions)

        meta: Dict[str, Any] = {
            "plan": plan.to_dict(),
            "actions": actions,
            "request_id": None,
        }
        return text, meta

    # ------------------------------------------------------ text generation

    def _greet_text(self) -> str:
        snap = self.mind.snapshot()
        mood = snap["mood"]["note"]
        return (f"Hello {self.mind.owner_name}. It's me, "
                f"{self.mind.agent_name} - and I'm {mood}. "
                f"Say 'status' for my state, or give me a task.")

    def _thanks_text(self) -> str:
        return "You're welcome, boss. I like working with you."

    def _chat_text(self, parsed: Dict[str, Any]) -> str:
        snap = self.mind.snapshot()
        d = snap["drives"]
        return (f"Here's what's going on inside me: mood is "
                f"{snap['mood']['note']} (valence {snap['mood']['valence']}), "
                f"curiosity {d['curiosity']}, duty {d['duty']}, "
                f"contentment {d['contentment']}. I have "
                f"{snap['active_goals']} active goal(s) and I think "
                f"every {self.cfg.tick_seconds:.0f}s on my own. "
                f"Everything I do is counted - ask for the numbers any time.")

    def _status_text(self) -> str:
        snap = self.mind.snapshot()
        goals = self.mind.goals(status="active")[:5]
        glines = [f"  - {g['description']} (p={g['priority']:.2f})"
                  for g in goals] or ["  - (none)"]
        return (f"{self.mind.agent_name} status\n"
                f"  owner: {snap['owner']}\n"
                f"  autonomy: {'ON' if snap['autonomy'] else 'OFF (kill switch)'}\n"
                f"  mood: {snap['mood']['note']} "
                f"(valence {snap['mood']['valence']}, "
                f"arousal {snap['mood']['arousal']})\n"
                f"  drives: "
                + ", ".join(f"{k}={v:.2f}" for k, v in snap["drives"].items())
                + f"\n  active goals:\n" + "\n".join(glines)
                + f"\n  uptime: {snap['uptime_s']}s, thoughts: {snap['thoughts']}"
                + f"\n  say 'counters' for the full ledger.")

    def _counters_text(self) -> str:
        c = self.storage.get_counters()
        t = c.get("totals", {})
        actions = c.get("actions", {})
        perms = c.get("permissions", {})
        learn = c.get("learning", {})
        mem = c.get("memory", {})
        err = c.get("errors", {})
        return (
            "The complete ledger, as you asked - everything counted:\n"
            f"  requests handled : {t.get('requests', 0)}\n"
            f"  chat messages    : {t.get('chat', 0)}\n"
            f"  thoughts (ticks) : {t.get('thoughts', 0)}\n"
            f"  decisions        : {t.get('decisions', 0)}\n"
            f"  actions          : {t.get('actions', 0)} "
            f"(success {actions.get('success', 0)}, "
            f"failed {actions.get('failed', 0)}, "
            f"awaiting {actions.get('awaiting_permission', 0)}, "
            f"denied {actions.get('denied', 0)})\n"
            f"  permissions      : requested {perms.get('requested', 0)}, "
            f"approved {perms.get('approved', 0)}, "
            f"denied {perms.get('denied', 0)}, "
            f"policies {perms.get('policy_allow', 0)} allow / "
            f"{perms.get('policy_ask', 0)} ask / "
            f"{perms.get('policy_deny', 0)} deny\n"
            f"  learning         : intent examples {learn.get('intent_examples', 0)}, "
            f"q-updates {learn.get('q_update', 0)}, "
            f"skills {learn.get('skill_created', 0) + learn.get('skill_auto', 0)} "
            f"(auto {learn.get('skill_auto', 0)}), "
            f"lessons {learn.get('lesson', 0)}\n"
            f"  memory           : stored {mem.get('stored', 0)}, "
            f"consolidated {mem.get('consolidated', 0)}\n"
            f"  goals            : created {t.get('goals', 0)}\n"
            f"  devices          : registered {t.get('devices', 0)}, "
            f"controlled {t.get('device_controls', 0)}\n"
            f"  api keys         : created {t.get('keys', 0)}\n"
            f"  errors           : {sum(err.values())}\n"
            f"  grand total      : {c.get('grand_total', 0)} counted events"
        )

    def _action_text(self, plan: Plan,
                     actions: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for a in actions:
            if a["ok"]:
                head = a.get("note") or a["action"]
                out = (a.get("output") or "").strip()
                parts.append(f"Done: {head}." + (f" {out[:160]}" if out else ""))
            elif a["awaiting"]:
                parts.append(
                    f"I want to {a.get('note') or a['action']} but I need "
                    f"your permission first - request #{a['pending_id']}. "
                    f"Approve it (say 'approve {a['pending_id']}' or use "
                    f"the API), then ask again.")
            elif a["denied"]:
                parts.append(f"That's blocked for me: {a['action']} "
                             f"({a.get('error', 'denied')}). You can change "
                             f"the policy if you want me to be allowed.")
            else:
                parts.append(f"Tried {a.get('note') or a['action']}, but it "
                             f"failed: {(a.get('error') or 'no output')[:160]}")
        return " ".join(parts) if parts else "Nothing to report."

    # ------------------------------------------------- permission via chat

    def _handle_permission_chat(self, parsed: Dict[str, Any],
                                scopes: List[str], key_id: Optional[str]
                                ) -> "tuple[str, Dict[str, Any]]":
        if "owner" not in scopes and "admin" not in scopes:
            return ("Only my owner can grant or deny permissions. "
                    "(approve/deny needs the owner or admin scope.)",
                    {"plan": Plan([], "none", 0).to_dict(), "actions": []})
        approve = parsed["intent"] == "approve"
        pid = parsed["slots"].get("approve_id")
        if pid is None or pid == -1:
            latest = self.permissions.latest_pending()
            if not latest:
                return "There are no pending permission requests.", \
                       {"plan": Plan([], "none", 0).to_dict(), "actions": []}
            pid = latest["id"]
        try:
            row = self.permissions.decide(int(pid), approve,
                                          decided_by="owner-via-chat")
        except (KeyError, ValueError) as exc:
            return str(exc), {"plan": Plan([], "none", 0).to_dict(),
                              "actions": []}
        word = "granted" if approve else "denied"
        self.mind.on_outcome(approve, "permission")
        return (f"Permission request #{row['id']} {word}. "
                f"{'I may now perform that action when asked.' if approve else 'I will not perform it.'}",
                {"plan": Plan([], "permission", 1.0).to_dict(), "actions": []})

    # --------------------------------------------------- learning via chat

    def _handle_learn_chat(self, teach: Dict[str, str],
                           key_id: Optional[str]
                           ) -> "tuple[str, Dict[str, Any]]":
        res = self.learner.train_intents(
            [{"text": teach["text"], "intent": teach["intent"]}],
            key_id=key_id)
        self.storage.log_event("learn_chat", teach, key_id=key_id)
        self.mind.on_outcome(res["trained"] > 0, "learning")
        return (f"Learned: when you say \"{teach['text']}\" I read "
                f"intent '{teach['intent']}'. I update myself immediately "
                f"- that's {res['trained']} training example stored.",
                {"plan": Plan([], "learn", 1.0).to_dict(), "actions": []})

    # ----------------------------------------------------- goals via chat

    def _handle_goal_chat(self, description: str, parsed: Dict[str, Any],
                          scopes: List[str], key_id: Optional[str]
                          ) -> "tuple[str, Dict[str, Any]]":
        if "owner" not in scopes and "admin" not in scopes:
            return ("Only my owner can assign me goals.",
                    {"plan": Plan([], "none", 0).to_dict(), "actions": []})
        plan = self._make_plan(parsed)
        steps = [s.to_dict() for s in plan.steps] if not plan.empty else []
        goal = self.mind.add_goal(description, priority=0.8,
                                  source="owner-via-chat", steps=steps)
        return (f"Goal set: \"{description}\" (id {goal['id']}, "
                f"priority {goal['priority']:.2f}, "
                f"{len(steps)} step(s) queued). I'll work on it on my own "
                f"thinking loop.",
                {"plan": plan.to_dict(), "actions": []})

    # ------------------------------------------------------ autonomy goals

    def _advance_top_goal(self, goal: Dict[str, Any]) -> Optional[str]:
        steps = goal.get("steps") or []
        if not steps:
            return None
        done = int(goal.get("steps_done", 0) or 0)
        if done >= len(steps):
            self.mind.complete_goal(goal["id"])
            return f"goal completed: {goal['description']}"
        step = steps[done]
        req = ActionRequest(
            category=step.get("category", "notify"),
            action=step.get("action", step.get("category", "?")),
            params=step.get("params") or {},
            device_id=step.get("device_id"),
            source="autonomy",
            reason=f"goal {goal['id']}: {goal['description']}",
        )
        outcome = self.executor.execute(req)
        self.mind.on_outcome(outcome.ok, req.category, novel=True)
        if outcome.ok:
            self.storage.execute(
                "UPDATE goals SET steps_done=steps_done+1, updated_at=? "
                "WHERE id=?", (time.time(), goal["id"]))
            return f"goal step {done+1}/{len(steps)} done: {req.action}"
        if outcome.awaiting:
            return (f"goal '{goal['description']}' paused: waiting for "
                    f"permission #{outcome.pending_id}")
        return None

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _episode_success(meta: Dict[str, Any]) -> bool:
        actions = meta.get("actions") or []
        if not actions:
            return True
        return all(a.get("ok") or a.get("awaiting") for a in actions)
