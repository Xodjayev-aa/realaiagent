"""Task planning.

Turns a parsed message into a plan: an ordered list of action steps, or a
clarification (no steps). Strategies:

- single action  ("turn on the lamp" -> one device.control step)
- chaining       ("turn on the lamp then set the temp to 21")
- skill          (a named multi-step procedure from skill memory)
- clarify        (not enough to act on - ask the owner)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import nlp

_CHAIN_SPLIT = re.compile(r"\s+then\s+|\s+after that\s+|\s+and then\s+|(?<!\d) ; (?!\d)")


@dataclass
class Step:
    category: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    device_id: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "action": self.action,
            "params": self.params,
            "device_id": self.device_id,
            "note": self.note,
        }


@dataclass
class Plan:
    steps: List[Step] = field(default_factory=list)
    strategy: str = "none"
    confidence: float = 0.0

    @property
    def empty(self) -> bool:
        return not self.steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "confidence": self.confidence,
            "steps": [s.to_dict() for s in self.steps],
        }


class Planner:
    def __init__(self) -> None:
        pass

    def plan(self, parsed: Dict[str, Any],
             devices: List[Dict[str, Any]],
             skill_steps: Optional[List[Dict[str, Any]]] = None
             ) -> Plan:
        intent = parsed.get("intent")
        slots = parsed.get("slots", {})
        conf = parsed.get("confidence", 0.0)

        # skill invocation wins (explicit named procedure)
        if slots.get("skill") and skill_steps:
            steps = [self._step_from_dict(s) for s in skill_steps]
            if steps:
                return Plan(steps, "skill", max(conf, 0.9))

        if intent == "command" or (slots.get("verb") and intent != "chat"):
            parts = _CHAIN_SPLIT.split(parsed.get("text", ""))
            steps: List[Step] = []
            for part in parts:
                sub = self._command_step(part, slots, devices)
                if sub is not None:
                    steps.append(sub)
            if steps:
                strategy = "chain" if len(steps) > 1 else "single"
                return Plan(steps, strategy, max(conf, 0.7))
            return Plan([], "clarify", conf * 0.5)

        if intent == "goal" and slots.get("verb"):
            sub = self._command_step(parsed.get("text", ""), slots, devices)
            if sub is not None:
                return Plan([sub], "goal_step", max(conf, 0.7))
        return Plan([], "none", conf)

    # ------------------------------------------------------------ commands

    def _command_step(self, text: str, slots: Dict[str, Any],
                      devices: List[Dict[str, Any]]
                      ) -> Optional[Step]:
        verb = slots.get("verb") or nlp.extract_verb(text)
        if not verb:
            return None
        device = self._find_device(text, slots, devices)

        # device control
        if device is not None:
            params: Dict[str, Any] = {"action": verb}
            if verb == "set" and slots.get("number") is not None:
                params["action"] = "set"
                params["level"] = slots["number"]
            if verb in ("on", "off"):
                params["action"] = verb
                human = f"turn {verb} the {device['name']}"
            elif verb == "set" and slots.get("number") is not None:
                human = f"set the {device['name']} to {slots['number']:g}"
            else:
                human = f"{verb} the {device['name']}"
            return Step("device.control",
                        f"{device['id']}:{params['action']}",
                        params, device["id"],
                        note=human)

        # file operations
        path = slots.get("path")
        if verb in ("read", "open") and path:
            return Step("files.read", f"files.read:{path}",
                        {"path": path}, note=f"read {path}")
        if verb in ("write",) and path:
            content = slots.get("quoted") or ""
            return Step("files.write", f"files.write:{path}",
                        {"path": path, "content": content},
                        note=f"write {path}")
        if verb == "delete" and path:
            return Step("system.command", f"system.command:rm {path}",
                        {"command": f"rm -f {path!r}"}, note=f"delete {path}")

        # explicit command
        if verb == "run":
            cmd = _extract_command_after(text)
            if cmd:
                return Step("system.command",
                            f"system.command:{cmd[:48]}",
                            {"command": cmd}, note=cmd)

        if path:
            if verb in ("read", "open"):
                return Step("files.read", f"files.read:{path}",
                            {"path": path}, note=f"read {path}")
        return None

    @staticmethod
    def _find_device(text: str, slots: Dict[str, Any],
                     devices: List[Dict[str, Any]]
                     ) -> Optional[Dict[str, Any]]:
        if not devices:
            return None
        did = slots.get("device_id") or nlp.best_fuzzy_match(
            text, [(d["id"], f"{d['name']} {d.get('kind','')}") for d in devices])
        if not did:
            return None
        for d in devices:
            if d["id"] == did:
                return d
        return None

    @staticmethod
    def _step_from_dict(d: Dict[str, Any]) -> Step:
        return Step(
            category=d.get("category", "notify"),
            action=d.get("action", d.get("category", "?")),
            params=d.get("params") or {},
            device_id=d.get("device_id"),
            note=d.get("note", ""),
        )


def _extract_command_after(text: str) -> Optional[str]:
    t = text.strip()
    m = re.match(r"^(?:run|execute|do)\s*[:\-]?\s+(.+)$", t, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(?:run|execute)\s+(.+)$", t, re.I)
    if m:
        return m.group(1).strip()
    return None
