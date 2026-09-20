"""Action execution: permission gate -> dispatch -> count -> learn.

Every execution is counted (allowed / awaiting / denied / failed /
success), logged as an event, and fed back to the Q-learning policy.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from ..storage import Storage
from . import builtins
from .base import ActionOutcome, ActionRequest
from .permissions import PermissionManager


class ActionExecutor:
    def __init__(self, storage: Storage, permissions: PermissionManager,
                 cfg: Config) -> None:
        self.storage = storage
        self.permissions = permissions
        self.cfg = cfg
        self.on_q_update: Optional[Callable[[str, str, float], None]] = None
        self.state_builder: Optional[Callable[[ActionRequest], str]] = None

    # ------------------------------------------------------------------ api

    def execute(self, req: ActionRequest, key_id: Optional[str] = None) -> ActionOutcome:
        t0 = time.time()
        decision = self.permissions.check(req)
        if not decision.granted:
            outcome = ActionOutcome(
                ok=False,
                awaiting=decision.awaiting,
                denied=not decision.awaiting,
                pending_id=decision.pending_id,
                error=decision.reason,
                ms=(time.time() - t0) * 1000,
            )
            self.storage.count("actions", "awaiting_permission"
                                 if decision.awaiting else "denied",
                               key_id=key_id,
                               detail={"category": req.category,
                                       "action": req.action})
            self.storage.log_event("action_blocked", {
                **req.to_dict(), "reason": decision.reason,
                "pending_id": decision.pending_id,
            }, key_id=key_id)
            return outcome

        try:
            ok, output, error = self._dispatch(req)
        except builtins.ScopeError as exc:
            ok, output, error = False, "", f"scope: {exc}"
        except Exception as exc:  # noqa: BLE001
            ok, output, error = False, "", f"error: {exc}"

        outcome = ActionOutcome(
            ok=ok, output=_clip_str(output), error=error,
            ms=(time.time() - t0) * 1000,
        )
        event = "success" if ok else "failed"
        self.storage.count("actions", event, key_id=key_id,
                           detail={"category": req.category})
        self.storage.log_event(f"action_{event}", {
            **req.to_dict(), "output_head": _clip_str(output)[:200],
            "error": error, "ms": outcome.ms,
        }, key_id=key_id)
        if req.device_id:
            self.storage.execute(
                "UPDATE devices SET last_seen=? WHERE id=?",
                (time.time(), req.device_id))
        self._q_update(req, 1.0 if ok else -1.0)
        return outcome

    # ------------------------------------------------------------- dispatch

    def _dispatch(self, req: ActionRequest):
        if req.category == "device.control":
            return self._device_control(req)
        if req.category == "files.read":
            return builtins.files_read(self.cfg, req.params)
        if req.category == "files.write":
            return builtins.files_write(self.cfg, req.params)
        if req.category == "system.command":
            return builtins.system_command(self.cfg, req.params)
        if req.category == "http.request":
            return builtins.http_request(self.cfg, req.params)
        if req.category == "notify":
            return builtins.notify(self.cfg, req.params)
        if req.category == "timer.set":
            return builtins.timer_set(self.cfg, req.params)
        return False, "", f"unknown action category: {req.category}"

    def _device_control(self, req: ActionRequest):
        device = self.storage.query_one(
            "SELECT * FROM devices WHERE id=?", (req.device_id or "",))
        if device is None:
            return False, "", f"unknown device: {req.device_id}"
        caps = _load(device.get("capabilities") or "[]")
        cap = self._match_capability(caps, req.params)
        if cap is None:
            return False, "", (f"device '{device['name']}' has no such "
                               f"capability (has: "
                               f"{', '.join(c.get('action','?') for c in caps)})")
        executor_kind = cap.get("executor", "http")
        params = {**req.params, "device": device["id"], "name": device["name"]}
        if executor_kind == "command":
            cmd = builtins.render_template(str(cap.get("template", "")), params)
            if not cmd:
                return False, "", "device capability has no command template"
            return builtins.system_command(self.cfg, {"command": cmd})
        if executor_kind == "http":
            url = builtins.render_template(str(cap.get("endpoint", "")), params)
            if not url:
                return False, "", "device capability has no endpoint"
            body = cap.get("body")
            if body:
                body = builtins.render_template(str(body), params)
                if body.strip().startswith(("{", "[")):
                    try:
                        body = json.loads(body)
                    except ValueError:
                        pass
            return builtins.http_request(self.cfg, {
                "url": url,
                "method": cap.get("method", "POST"),
                "headers": cap.get("headers") or {},
                "body": body,
            })
        return False, "", f"unknown device executor: {executor_kind}"

    @staticmethod
    def _match_capability(caps: List[Dict[str, Any]],
                          params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wanted = str(params.get("action", ""))
        if not caps:
            return None
        for cap in caps:
            if cap.get("action") == wanted:
                return cap
        for cap in caps:  # prefix / containment match
            if wanted and (wanted.startswith(str(cap.get("action", "")))
                           or str(cap.get("action", "")).startswith(wanted)):
                return cap
        if len(caps) == 1:
            return caps[0]
        return None

    # --------------------------------------------------------------- q-learn

    def _q_update(self, req: ActionRequest, reward: float) -> None:
        if self.on_q_update is None:
            return
        state = self.state_builder(req) if self.state_builder else req.category
        try:
            self.on_q_update(state, req.action, reward)
        except Exception:  # noqa: BLE001
            pass


def _load(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return []


def _clip_str(text: str) -> str:
    return text if len(text) <= 4000 else text[:4000] + " …[truncated]"
