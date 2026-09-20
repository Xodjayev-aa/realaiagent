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


def index_page(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Human-friendly root page (the API is the product; this is its face)."""
    snap = agent.mind.snapshot()
    c = agent.storage.get_counters()
    t = c.get("totals", {})
    endpoints = [
        ("POST", "/v1/chat", "talk to the agent  {\"message\": \"...\"}"),
        ("GET", "/v1/status", "mind state: drives, mood, goals, counters"),
        ("GET", "/v1/counters", "everything counted - the full ledger"),
        ("GET", "/v1/usage?by=category|key|day", "usage breakdowns (admin)"),
        ("GET", "/v1/events", "the agent's thoughts & actions log"),
        ("GET", "/v1/goals", "goal queue"),
        ("GET", "/v1/skills", "learned skills"),
        ("GET", "/v1/devices", "registered devices"),
        ("GET", "/v1/permissions", "policies + pending requests"),
        ("GET", "/v1/users", "clients: PENDING/APPROVED/BANNED (owner)"),
        ("POST", "/v1/learn", "train the language model (learn scope)"),
        ("GET", "/healthz", "liveness (public)"),
    ]
    rows = "\n".join(
        f"<tr><td class='m'>{m}</td><td class='p'>{p}</td><td>{d}</td></tr>"
        for m, p, d in endpoints)
    return 200, f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{snap['agent']} — RealAI Agent</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ background:#0b0e14; color:#d7dce5; font:15px/1.55
        ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;
        margin:0; padding:40px 20px; }}
 main {{ max-width: 860px; margin: 0 auto; }}
 h1 {{ font-size: 26px; margin: 0 0 4px; }}
 .sub {{ color:#8b94a7; margin-bottom: 24px; }}
 .card {{ background:#12161f; border:1px solid #1f2633; border-radius:12px;
         padding:18px 20px; margin-bottom:16px; }}
 .k {{ color:#8b94a7; font-size:12px; text-transform:uppercase;
       letter-spacing:.08em; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fit,
         minmax(150px,1fr)); gap:12px; }}
 .v {{ font-size:18px; font-weight:600; margin-top:2px; }}
 .ok {{ color:#3fd68f; }} .warn {{ color:#f5c542; }}
 table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
 td {{ padding:7px 10px; border-top:1px solid #1f2633;
       vertical-align:top; }}
 .m {{ color:#7aa2f7; font-family:ui-monospace,monospace; white-space:nowrap; }}
 .p {{ color:#d7dce5; font-family:ui-monospace,monospace; }}
 code {{ background:#1a2030; padding:1px 6px; border-radius:6px;
         font-size:13px; }}
 footer {{ color:#5c6577; font-size:12.5px; margin-top:20px; }}
</style></head><body><main>
 <h1>🤖 {snap['agent']} — RealAI Agent</h1>
 <div class="sub">A fully local, owner-controlled cognitive AI agent —
  pure Python, no external AI, no external keys. API-only (your UI talks
  to <code>/v1/chat</code>).</div>

 <div class="card"><div class="k">live status</div>
  <div class="grid">
   <div><div class="k">mood</div><div class="v ok">{snap['mood']['note']}</div></div>
   <div><div class="k">autonomy</div>
     <div class="v {'ok' if snap['autonomy'] else 'warn'}">{'ON' if snap['autonomy'] else 'OFF'}</div></div>
   <div><div class="k">active goals</div><div class="v">{snap['active_goals']}</div></div>
   <div><div class="k">thoughts</div><div class="v">{snap['thoughts']}</div></div>
   <div><div class="k">uptime</div><div class="v">{snap['uptime_s']}s</div></div>
   <div><div class="k">counted events</div><div class="v">{c.get('grand_total', 0)}</div></div>
  </div>
  <div style="margin-top:12px;color:#8b94a7;font-size:13.5px">
   drives: {' · '.join(f'{k} {v:.2f}' for k, v in snap['drives'].items())}
   &nbsp;|&nbsp; requests {t.get('requests',0)} · chat {t.get('chat',0)} ·
   actions {t.get('actions',0)} · users {t.get('users',0)} ·
   intrusions {c.get('security',{}).get('intrusion',0)}
  </div>
 </div>

 <div class="card"><div class="k">endpoints (auth: <code>Authorization: Bearer &lt;key&gt;</code>)</div>
  <table>{rows}
   <tr><td class="m">full list</td><td class="p">/api.json</td>
       <td>machine-readable index of every route</td></tr>
  </table>
 </div>

 <div class="card"><div class="k">quick start</div>
  <code>curl -X POST /v1/chat -H 'Authorization: Bearer &lt;key&gt;'
  -H 'Content-Type: application/json' -d '{{"message":"status"}}'</code>
 </div>

 <footer>owner: {snap['owner']} · everything is counted ·
  MIT — 100% your code</footer>
</main></body></html>"""


def api_index(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Machine-readable index of every route and its required scopes."""
    routes = []
    for method, path, scopes, handler in ROUTES:
        routes.append({
            "method": method,
            "path": path,
            "scopes": scopes or ["public"],
        })
    return _ok({"agent": agent.mind.snapshot()["agent"],
                "routes": routes})


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
    request_limit = body.get("request_limit")
    if request_limit is not None:
        try:
            request_limit = int(request_limit)
        except (TypeError, ValueError):
            return _err(400, "bad_request", "request_limit must be a number")
    try:
        key_id, plaintext, meta = agent.keys.create_key(
            name, scopes,
            created_by=(ctx.get("key") or {}).get("name", "api"),
            user=body.get("user"), request_limit=request_limit)
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


# ------------------------------------------------------------- users / biz

def users_request(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    username = str(body.get("username", "")).strip().lower()
    if not username or len(username) > 48:
        return _err(400, "bad_request", "username (1-48 chars) required")
    scopes = body.get("scopes")
    if scopes is not None and not isinstance(scopes, list):
        return _err(400, "bad_request", "scopes must be a list")
    user = agent.users.request(username, scopes=scopes,
                               note=str(body.get("note", "")))
    return _ok({
        "status": "PENDING",
        "username": user["username"],
        "message": "Request received. The owner will approve or deny it "
                   "(they are notified via their control channel).",
    })


def users_list(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    st = ctx.get("query", {}).get("status")
    return _ok({"users": agent.users.list(st)})


def users_usage(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    u = ctx["params"]["username"]
    if not agent.storage.user_get(u):
        return _err(404, "not_found", f"user {u} not found")
    return _ok(agent.users.usage_for_user(u))


def users_approve(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    u = ctx["params"]["username"]
    try:
        out = agent.users.approve(
            u, scopes=body.get("scopes"),
            request_limit=body.get("request_limit"))
    except KeyError:
        return _err(404, "not_found", f"user {u} not found")
    user = {k: v for k, v in out.items()
            if k not in ("api_key", "key_id", "key_scopes")}
    return _ok({
        "user": user,
        "api_key": out["api_key"],
        "key_id": out["key_id"],
        "key_scopes": out["key_scopes"],
        "note": "store this key now - it is not shown again",
    })


def users_deny(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    u = ctx["params"]["username"]
    try:
        out = agent.users.deny(u)
    except KeyError:
        return _err(404, "not_found", f"user {u} not found")
    return _ok(out)


def users_unban(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    u = ctx["params"]["username"]
    try:
        out = agent.users.unban(u)
    except KeyError:
        return _err(404, "not_found", f"user {u} not found")
    return _ok(out)


def users_vip(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    u = ctx["params"]["username"]
    try:
        out = agent.users.set_vip(u, bool(body.get("vip", True)))
    except KeyError:
        return _err(404, "not_found", f"user {u} not found")
    return _ok(out)


def users_topup(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    u = ctx["params"]["username"]
    try:
        amount = float(body.get("amount", 0))
    except (TypeError, ValueError):
        return _err(400, "bad_request", "amount must be a number")
    if amount <= 0:
        return _err(400, "bad_request", "amount must be positive")
    try:
        out = agent.users.topup(u, amount)
    except KeyError:
        return _err(404, "not_found", f"user {u} not found")
    return _ok(out)


# ------------------------------------------------------------------ routing

ROUTES: List[Tuple[str, str, List[str], Handler]] = [
    # (method, path, scopes (empty = public), handler)
    ("GET", "/", [], index_page),
    ("GET", "/api.json", [], api_index),
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

    ("POST", "/v1/users/request", [], users_request),
    ("GET", "/v1/users", ["owner"], users_list),
    ("GET", "/v1/users/{username}/usage", ["owner"], users_usage),
    ("POST", "/v1/users/{username}/approve", ["owner"], users_approve),
    ("POST", "/v1/users/{username}/deny", ["owner"], users_deny),
    ("POST", "/v1/users/{username}/unban", ["owner"], users_unban),
    ("POST", "/v1/users/{username}/vip", ["owner"], users_vip),
    ("POST", "/v1/users/{username}/topup", ["owner"], users_topup),

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
