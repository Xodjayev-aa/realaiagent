"""Offline demo: a scripted conversation proving the full pipeline works
without any network or external AI.

    python -m realaiagent demo

It registers a toy device, talks to the agent, requests permission,
approves it, trains the intent model, and prints the full ledger.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

from .config import Config
from .engine import Agent

OWNER_SCOPES = ["owner"]


def _say(agent: Agent, text: str) -> None:
    res = agent.handle_message(text, sender="owner-demo", scopes=OWNER_SCOPES)
    print(f"\n🧑 you  : {text}")
    print(f"🤖 {agent.mind.agent_name}: {res['response']}")
    actions = res["meta"].get("actions") or []
    for a in actions:
        print(f"     ↳ action {a['category']}:{a['action']} "
              f"ok={a['ok']} awaiting={a['awaiting']}")


def run_demo(data_dir: str = "data/demo") -> int:
    path = Path(data_dir)
    if path.exists():
        shutil.rmtree(path)
    cfg = Config(data_dir=path, tick_seconds=3600)  # no autonomous tick in demo
    agent = Agent(cfg)
    agent.mind.set_identity(agent_name="REAL", owner_name="Boss")

    print("=" * 62)
    print("  REAL — RealAI Agent · offline demo")
    print("  (no network, no external AI — pure local code)")
    print("=" * 62)

    # 1. identity + greeting
    _say(agent, "hello!")
    _say(agent, "status")

    # 2. register a device the agent can control (permission-gated)
    agent.storage.execute(
        "INSERT INTO devices(id, name, kind, capabilities, meta, created_at,"
        " last_seen) VALUES(?,?,?,?,?,?,?)",
        ("lamp-1", "Living Room Lamp", "light",
         json.dumps([
             {"action": "on", "executor": "command",
              "template": "echo lamp-on"},
             {"action": "off", "executor": "command",
              "template": "echo lamp-off"},
         ]),
         json.dumps({"room": "living room"}),
         __import__("time").time(), __import__("time").time()))
    agent.storage.count("devices", "registered")
    print("\n🔌 registered device: lamp-1 (Living Room Lamp)")

    # 3. ask it to act -> permission gate asks first
    _say(agent, "turn on the lamp")
    pending = agent.permissions.latest_pending()
    if pending:
        print(f"   permission request #{pending['id']} is awaiting you…")

    # 4. owner approves
    _say(agent, f"approve {pending['id'] if pending else 0}")

    # 5. now the action goes through (exact action is trusted)
    _say(agent, "turn on the lamp")

    # 6. train the intent model with a new phrase
    res = agent.learner.train_intents(
        [{"text": "power up the lamp", "intent": "command"}])
    print(f"\n🎓 trained {res['trained']} new intent example")

    # 7. a goal the agent will advance on its own
    agent.mind.add_goal(
        "keep the living room light on in the evening", priority=0.9,
        source="owner-demo",
        steps=[{"category": "device.control",
                "action": "lamp-1:on",
                "params": {"action": "on"}, "device_id": "lamp-1"}])
    _say(agent, "what are your goals")

    # 8. show its mind + the ledger
    _say(agent, "counters")
    snap = agent.mind.snapshot()
    print(f"\n🧠 mind snapshot: mood={snap['mood']['note']} "
          f"drives={ {k: v for k, v in snap['drives'].items()} }")
    print("\n✅ demo complete — every step above is in the usage ledger.")
    agent.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
