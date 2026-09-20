"""API routes.

Every route returns (http_status, payload). ``ctx`` carries the verified
key row (None for public routes), the request body, and the request id.

Scopes required per route are declared in ``ROUTES`` and enforced by the
server. Owner scope implies everything.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..engine import Agent
from ..nlp import tokenize  # noqa: F401 (helpers share normalization)

Handler = Callable[["Agent", Dict[str, Any], Dict[str, Any]],
                   Tuple[int, Any]]


def _ok(payload: Any = None) -> Tuple[int, Any]:
    return 200, payload if payload is not None else {"ok": True}


def _err(status: int, code: str, message: str) -> Tuple[int, Any]:
    return status, {"error": {"code": code, "message": message}}


# ------------------------------------------------------------------ public

def healthz(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    snap = agent.mind.snapshot()
    return _ok({
        "ok": True,
        "agent": snap["agent"],
        "version": __import__("realaiagent").__version__,
        "uptime_s": snap["uptime_s"],
        "autonomy": snap["autonomy"],
        "mood": snap["mood"]["note"],
    })


# ------------------------------------------------------------------- status

def status(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    snap = agent.mind.snapshot()
    return _ok({
        **snap,
        "counters": agent.storage.get_counters(),
    })


def counters(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    return _ok(agent.storage.get_counters())


def usage(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    by = str(body.get("by", ctx.get("query", {}).get("by", "category")))
    if by not in ("category", "key", "day"):
        return _err(400, "bad_request", "by must be category|key|day")
    return _ok({"by": by, "rows": agent.storage.usage_breakdown(by),
                "counters": agent.storage.get_counters()})


def events(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    try:
        limit = int(ctx.get("query", {}).get("limit", 50))
    except ValueError:
        limit = 50
    return _ok({"events": agent.storage.recent_events(limit)})


# -------------------------------------------------------------------- chat

def chat(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    text = str(body.get("message", "")).strip()
    if not text:
        return _err(400, "bad_request", "message is required")
    if len(text) > 4000:
        return _err(400, "bad_request", "message too long (max 4000)")
    sender = str(body.get("sender", "api-client"))
    key = ctx.get("key") or {}
    res = agent.handle_message(
        text, sender=sender, scopes=key.get("scopes", []),
        key_id=key.get("id"))
    return _ok({**res, "request_id": ctx.get("request_id")})


# ------------------------------------------------------------------- goals

def goals_list(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    st = ctx.get("query", {}).get("status")
    return _ok({"goals": agent.mind.goals(status=st)})


def goals_create(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    description = str(body.get("description", "")).strip()
    if not description:
        return _err(400, "bad_request", "description is required")
    try:
        priority = float(body.get("priority", 0.5))
    except ValueError:
        return _err(400, "bad_request", "priority must be a number")
    steps = body.get("steps") or []
    if not isinstance(steps, list):
        return _err(400, "bad_request", "steps must be a list")
    goal = agent.mind.add_goal(description, priority,
                               source="api", steps=steps)
    return _ok(goal)


def goals_complete(agent: Agent, body: Dict[str, Any],
                   ctx: Dict[str, Any]):
    gid = ctx["params"]["id"]
    ok = agent.mind.complete_goal(gid)
    return _ok({"completed": ok}) if ok else \
        _err(404, "not_found", f"active goal {gid} not found")


def goals_remove(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    gid = ctx["params"]["id"]
    ok = agent.mind.remove_goal(gid)
    return _ok({"removed": ok}) if ok else \
        _err(404, "not_found", f"goal {gid} not found")


# ------------------------------------------------------------------- memory

def memory_list(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    q = ctx.get("query", {})
    kind = q.get("kind")
    needle = q.get("q")
    try:
        limit = int(q.get("limit", 100))
    except ValueError:
        limit = 100
    limit = max(1, min(limit, 1000))
    rows = agent.storage.query(
        "SELECT * FROM memories ORDER BY last_accessed DESC LIMIT 1000")
    out = []
    for r in rows:
        if kind and r["kind"] != kind:
            continue
        if needle and needle.lower() not in (
                r["key"] + json.dumps(_j(r["value"]))).lower():
            continue
        r["value"] = _j(r["value"])
        out.append(r)
        if len(out) >= limit:
            break
    return _ok({"count": len(out), "memories": out})


def memory_put(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    kind = str(body.get("kind", "semantic")).strip()
    key = str(body.get("key", "")).strip()
    if not kind or not key:
        return _err(400, "bad_request", "kind and key are required")
    value = body.get("value")
    agent.learner.remember(kind, key, value,
                           strength=float(body.get("strength", 1.0)),
                           key_id=(ctx.get("key") or {}).get("id"))
    return _ok({"stored": True, "kind": kind, "key": key})


def memory_delete(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    mid = ctx["params"]["id"]
    row = agent.storage.query_one(
        "SELECT kind, key FROM memories WHERE id=?", (mid,))
    if row is None:
        return _err(404, "not_found", f"memory {mid} not found")
    agent.storage.execute("DELETE FROM memories WHERE id=?", (mid,))
    agent.storage.count("memory", "deleted")
    return _ok({"deleted": mid})


# -------------------------------------------------------------------- learn

def learn(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    examples = body.get("examples")
    if not isinstance(examples, list) or not examples:
        return _err(400, "bad_request",
                    "examples: [{text, intent, slots?}] is required")
    try:
        res = agent.learner.train_intents(
            examples[:50], key_id=(ctx.get("key") or {}).get("id"))
    except Exception as exc:  # noqa: BLE001
        return _err(400, "bad_request", str(exc))
    return _ok(res)


# ------------------------------------------------------------------- skills

def skills_list(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    return _ok({"skills": agent.learner.skills()})


def skills_create(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    name = str(body.get("name", "")).strip()
    steps = body.get("steps")
    if not name or not isinstance(steps, list) or not steps:
        return _err(400, "bad_request", "name and steps[] are required")
    try:
        skill = agent.learner.learn_skill(
            name, steps,
            description=str(body.get("description", "")),
            examples=body.get("examples"),
            key_id=(ctx.get("key") or {}).get("id"))
    except ValueError as exc:
        return _err(400, "bad_request", str(exc))
    return _ok(skill)


def skills_delete(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    ok = agent.learner.remove_skill(ctx["params"]["name"])
    return _ok({"removed": ok}) if ok else \
        _err(404, "not_found", "skill not found")


# ------------------------------------------------------------------ devices

def devices_list(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    return _ok({"devices": agent._devices()})


def devices_create(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    did = str(body.get("id", "")).strip()
    name = str(body.get("name", "")).strip()
    if not did or not name:
        return _err(400, "bad_request", "id and name are required")
    caps = body.get("capabilities") or []
    if not isinstance(caps, list):
        return _err(400, "bad_request", "capabilities must be a list")
    for cap in caps:
        if not isinstance(cap, dict) or not cap.get("action"):
            return _err(400, "bad_request",
                        "each capability needs an 'action' name")
    if agent.storage.query_one("SELECT id FROM devices WHERE id=?",
                               (did,)):
        return _err(409, "conflict", f"device {did} already exists")
    agent.storage.execute(
        "INSERT INTO devices(id, name, kind, capabilities, meta, created_at, "
        "last_seen) VALUES(?,?,?,?,?,?,?)",
        (did, name, str(body.get("kind", "generic")),
         json.dumps(caps), json.dumps(body.get("meta") or {}),
         __import__("time").time(), __import__("time").time()))
    agent.storage.count("devices", "registered")
    agent.storage.log_event("device_registered", {"id": did, "name": name,
                                                  "kind": body.get("kind")})
    agent.mind.on_outcome(True, "device")
    return _ok({"registered": True, "id": did})


def devices_delete(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    ok = agent.storage.execute(
        "DELETE FROM devices WHERE id=?", (ctx["params"]["id"],)).rowcount > 0
    if ok:
        agent.storage.count("devices", "removed")
    return _ok({"removed": ok}) if ok else \
        _err(404, "not_found", "device not found")


def devices_control(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    from ..actions.base import ActionRequest
    did = ctx["params"]["id"]
    if not agent.storage.query_one("SELECT id FROM devices WHERE id=?",
                                   (did,)):
        return _err(404, "not_found", f"device {did} not found")
    action = str(body.get("action", "")).strip()
    if not action:
        return _err(400, "bad_request", "action is required")
    req = ActionRequest(
        category="device.control", action=f"{did}:{action}",
        params={**body.get("params", {}), "action": action},
        device_id=did, source="direct",
        reason=f"direct control of {did}")
    outcome = agent.executor.execute(req,
                                     key_id=(ctx.get("key") or {}).get("id"))
    agent.storage.count("device_controls", "requested",
                        detail={"device": did})
    return _ok(outcome.to_dict())


# ----------------------------------------------------------------permissions

def permissions_list(agent: Agent, body: Dict[str, Any],
                     ctx: Dict[str, Any]):
    return _ok({"policies": agent.permissions.policies(),
                "pending": agent.permissions.pending_list()})


def permissions_set(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    category = str(body.get("category", "")).strip()
    policy = str(body.get("policy", "")).strip()
    if not category or not policy:
        return _err(400, "bad_request", "category and policy are required")
    try:
        row = agent.permissions.set_policy(
            category, policy, pattern=body.get("pattern"),
            note=str(body.get("note", "")))
    except ValueError as exc:
        return _err(400, "bad_request", str(exc))
    return _ok(row)


def permissions_pending(agent: Agent, body: Dict[str, Any],
                        ctx: Dict[str, Any]):
    return _ok({"pending": agent.permissions.pending_list()})


def permissions_decide(agent: Agent, body: Dict[str, Any],
                       ctx: Dict[str, Any]):
    try:
        pid = int(ctx["params"]["id"])
    except ValueError:
        return _err(400, "bad_request", "bad pending id")
    if ctx["params"]["decision"] not in ("approve", "deny"):
        return _err(400, "bad_request",
                    "decision must be 'approve' or 'deny'")
    approve = ctx["params"]["decision"] == "approve"
    try:
        row = agent.permissions.decide(pid, approve, decided_by="owner-api")
    except (KeyError, ValueError) as exc:
        return _err(400, "bad_request", str(exc))
    return _ok(row)


# --------------------------------------------------------------------- keys

def keys_list(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    return _ok({"keys": agent.keys.list_keys()})


def keys_create(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    name = str(body.get("name", "")).strip()
    scopes = body.get("scopes") or ["chat"]
    if not name:
        return _err(400, "bad_request", "name is required")
    if not isinstance(scopes, list):
        return _err(400, "bad_request", "scopes must be a list")
    try:
        key_id, plaintext, meta = agent.keys.create_key(
            name, scopes, created_by=(ctx.get("key") or {}).get("name", "api"))
    except ValueError as exc:
        return _err(400, "bad_request", str(exc))
    return _ok({"key": plaintext, **meta,
                "note": "store this now - it is not shown again"})


def keys_revoke(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    ok = agent.keys.revoke(ctx["params"]["id"])
    return _ok({"revoked": ok}) if ok else \
        _err(404, "not_found", "key not found")


# -------------------------------------------------------------------- owner

def owner_state(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    return _ok({
        "identity": {"agent": agent.mind.agent_name,
                     "owner": agent.mind.owner_name},
        "values": agent.mind.drives,
        "mood": {"valence": round(agent.mind.valence, 3),
                 "arousal": round(agent.mind.arousal, 3)},
        "autonomy": agent.mind.autonomy,
    })


def owner_identity(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    agent.mind.set_identity(
        agent_name=body.get("agent_name"),
        owner_name=body.get("owner_name"))
    return _ok(agent.mind.snapshot())


def owner_values(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    drives = body.get("drives")
    if not isinstance(drives, dict):
        return _err(400, "bad_request", "drives must be an object")
    try:
        agent.mind.set_values({k: float(v) for k, v in drives.items()})
    except (TypeError, ValueError) as exc:
        return _err(400, "bad_request", f"bad drive value: {exc}")
    return _ok({"drives": agent.mind.drives})


def owner_autonomy(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    on = bool(body.get("on", True))
    agent.mind.set_autonomy(on)
    return _ok({"autonomy": on})


# ------------------------------------------------------------------ routing

ROUTES: List[Tuple[str, str, List[str], Handler]] = [
    # (method, path, scopes (empty = public), handler)
    ("GET", "/healthz", [], healthz),

    ("GET", "/v1/status", ["chat"], status),
    ("GET", "/v1/counters", ["chat"], counters),
    ("GET", "/v1/usage", ["admin"], usage),
    ("GET", "/v1/events", ["chat"], events),

    ("POST", "/v1/chat", ["chat"], chat),

    ("GET", "/v1/goals", ["chat"], goals_list),
    ("POST", "/v1/goals", ["admin"], goals_create),
    ("POST", "/v1/goals/{id}/complete", ["owner"], goals_complete),
    ("DELETE", "/v1/goals/{id}", ["owner"], goals_remove),

    ("GET", "/v1/memory", ["admin"], memory_list),
    ("POST", "/v1/memory", ["admin"], memory_put),
    ("DELETE", "/v1/memory/{id}", ["owner"], memory_delete),

    ("POST", "/v1/learn", ["learn"], learn),

    ("GET", "/v1/skills", ["chat"], skills_list),
    ("POST", "/v1/skills", ["admin"], skills_create),
    ("DELETE", "/v1/skills/{name}", ["owner"], skills_delete),

    ("GET", "/v1/devices", ["devices"], devices_list),
    ("POST", "/v1/devices", ["admin"], devices_create),
    ("DELETE", "/v1/devices/{id}", ["owner"], devices_delete),
    ("POST", "/v1/devices/{id}/control", ["devices"], devices_control),

    ("GET", "/v1/permissions", ["admin"], permissions_list),
    ("POST", "/v1/permissions", ["owner"], permissions_set),
    ("GET", "/v1/permissions/pending", ["chat"], permissions_pending),
    ("POST", "/v1/permissions/pending/{id}/{decision}", ["owner"],
     permissions_decide),

    ("GET", "/v1/keys", ["owner"], keys_list),
    ("POST", "/v1/keys", ["owner"], keys_create),
    ("POST", "/v1/keys/{id}/revoke", ["owner"], keys_revoke),

    ("GET", "/v1/owner/state", ["owner"], owner_state),
    ("POST", "/v1/owner/identity", ["owner"], owner_identity),
    ("POST", "/v1/owner/values", ["owner"], owner_values),
    ("POST", "/v1/owner/autonomy", ["owner"], owner_autonomy),
]


def _j(raw: Optional[str]) -> Any:
    try:
        return json.loads(raw) if raw else None
    except (ValueError, TypeError):
        return raw
