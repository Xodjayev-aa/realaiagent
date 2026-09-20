"""Shared test helpers."""

from __future__ import annotations

import json
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from realaiagent.config import Config
from realaiagent.engine import Agent

OWNER = ["owner"]


def make_agent(tick: float = 3600.0, tmp: Optional[Path] = None) -> Agent:
    d = tmp if tmp else Path(tempfile.mkdtemp(prefix="realai-test-"))
    cfg = Config(data_dir=d, tick_seconds=tick)
    agent = Agent(cfg)
    agent.mind.set_identity(agent_name="REAL", owner_name="Boss")
    return agent


def owner_chat(agent: Agent, text: str) -> Dict[str, Any]:
    return agent.handle_message(text, sender="boss", scopes=OWNER)


def register_device(agent: Agent, did: str = "lamp-1",
                    name: str = "Living Room Lamp") -> None:
    import time as _t
    agent.storage.execute(
        "INSERT INTO devices(id, name, kind, capabilities, meta, created_at,"
        " last_seen) VALUES(?,?,?,?,?,?,?)",
        (did, name, "light", json.dumps([
            {"action": "on", "executor": "command", "template": "echo on"},
            {"action": "off", "executor": "command", "template": "echo off"},
        ]), "{}", _t.time(), _t.time()))


class ApiClient:
    """Tiny stdlib HTTP client for the test server."""

    def __init__(self, base: str, key: Optional[str] = None) -> None:
        self.base = base
        self.key = key

    def req(self, method: str, path: str,
            body: Optional[Dict[str, Any]] = None,
            key: Optional[str] = "__inherit__"
            ) -> "tuple[int, Dict[str, Any]]":
        token = self.key if key == "__inherit__" else key
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method)
        r.add_header("Content-Type", "application/json")
        if token:
            r.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode())
            except (ValueError, json.JSONDecodeError):
                return e.code, {}


def start_server(agent: Agent):
    """Start the real API server on an ephemeral port (daemon thread)."""
    import threading
    from realaiagent.api.server import ApiServer
    server = ApiServer(agent, "127.0.0.1", 0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
