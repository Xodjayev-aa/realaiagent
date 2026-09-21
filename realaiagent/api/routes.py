"""API routes.

Every route returns (http_status, payload). ``ctx`` carries the verified
key row (None for public routes), the request body, and the request id.

Scopes required per route are declared in ``ROUTES`` and enforced by the
server. Owner scope implies everything.
"""

from __future__ import annotations

import json
import threading
import time
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
        "storage": agent.storage.backend,
    })


def index_page(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """The product's front door: a branded chat app, open to everyone.

    Anyone can talk to the agent right here — keyless, per-visitor
    rate-limited, no scopes, so nothing privileged is reachable from a
    browser. The raw ``/v1/*`` API stays behind owner-issued keys, which is
    the whole point of the split: the website is public, the machine is the
    owner's.

    Rendering lives in :mod:`realaiagent.web_pages` (logo, favicon, session
    sidebar, markdown, streaming reveal, typing state, mobile layout).
    """
    from .. import web_pages
    return 200, web_pages.chat_page(agent)


def developers(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """The developer portal: a live endpoint table + key request form.

    The table is generated from the same route registry ``/api.json``
    serves, so the documentation cannot drift from the code.
    """
    from .. import web_pages
    host = web_pages.request_host(ctx.get("headers") or {})
    return 200, web_pages.developers_page(agent, api_routes(agent), host=host)


def api_routes(agent: Agent) -> List[Dict[str, Any]]:
    """Every route as data: method, path and the scopes that guard it."""
    return [{"method": method, "path": path, "scopes": scopes or ["public"]}
            for method, path, scopes, _handler in ROUTES]


# --------------------------------------------------- public demo chat
#
# The website is open to everyone, so this chat runs keyless and with no
# scopes: the engine's permission gate still applies, so nothing privileged
# can be triggered from a browser. Real clients use /v1/chat with an
# owner-issued key, where each key gets its own bucket in the server.
#
# The demo bucket is PER VISITOR, not one shared pool: with a single shared
# bucket, one script hammering the page would lock the demo out for every
# honest visitor (and the 429s would look like the product was broken).

#: How many visitor buckets we keep before pruning the stalest ones.
_MAX_VISITORS = 4096


class _PublicLimiter:
    """Token buckets for the keyless demo chat — one per visitor.

    ``allow()`` with no argument uses a single shared bucket, which is the
    behaviour the rest of the code (and the test-suite) has always had;
    ``allow(visitor)`` gives that visitor their own bucket.
    """

    SHARED = "shared"

    def __init__(self, rate_per_min: int = 12, burst: int = 6,
                 max_visitors: int = _MAX_VISITORS) -> None:
        self.rate = rate_per_min / 60.0
        self.burst = float(burst)
        self.max_visitors = max(16, int(max_visitors))
        self._lock = threading.Lock()
        # visitor id -> [tokens, last_refill]
        self._buckets: Dict[str, List[float]] = {}

    # ---------------------------------------------------------------- api

    def configure(self, rate_per_min: Optional[float] = None,
                  burst: Optional[float] = None) -> None:
        """Apply the owner's configured demo budget (REALAI_DEMO_*)."""
        if rate_per_min is not None:
            self.rate = max(0.0, float(rate_per_min) / 60.0)
        if burst is not None:
            self.burst = max(0.0, float(burst))

    def allow(self, visitor: Optional[str] = None) -> bool:
        """Take one token from this visitor's bucket (True = allowed)."""
        key = str(visitor or self.SHARED)
        now = time.time()
        with self._lock:
            if len(self._buckets) >= self.max_visitors and \
                    key not in self._buckets:
                self._prune(now)
            bucket = self._buckets.setdefault(key, [self.burst, now])
            bucket[0] = min(self.burst,
                            bucket[0] + (now - bucket[1]) * self.rate)
            bucket[1] = now
            if bucket[0] < 1.0:
                return False
            bucket[0] -= 1.0
            return True

    def reset(self, visitor: Optional[str] = None) -> None:
        with self._lock:
            if visitor is None:
                self._buckets.clear()
            else:
                self._buckets.pop(str(visitor), None)

    def visitors(self) -> int:
        with self._lock:
            return len(self._buckets)

    # ------------------------------------------------------------- internal

    def _prune(self, now: float) -> None:
        """Drop the stalest buckets so a long-running server cannot grow
        without bound. Called with ``self._lock`` held."""
        stalest = sorted(self._buckets.items(), key=lambda kv: kv[1][1])
        for key, _bucket in stalest[:max(1, len(stalest) // 2)]:
            self._buckets.pop(key, None)


#: The demo limiter. Tests (and the owner, via REALAI_DEMO_*) may swap or
#: reconfigure this; :func:`_demo_limiter` keeps both working.
_DEFAULT_DEMO_LIMITER = _PublicLimiter()
_public_demo_limiter = _DEFAULT_DEMO_LIMITER


def _demo_limiter(cfg: Any = None) -> "_PublicLimiter":
    """The limiter to use for this request.

    If somebody replaced the module-level limiter we use theirs untouched
    (that is how the suite simulates an exhausted bucket). Otherwise the
    owner's configured demo budget is applied to the default instance.
    """
    limiter = _public_demo_limiter
    if limiter is _DEFAULT_DEMO_LIMITER and cfg is not None:
        limiter.configure(getattr(cfg, "demo_rate_per_min", None),
                          getattr(cfg, "demo_burst", None))
    return limiter


def visitor_id(ctx: Dict[str, Any]) -> str:
    """A stable-enough id for the visitor behind a keyless request.

    Serverless and proxied deployments put the real client in
    ``X-Forwarded-For``; the threaded server injects the peer address. This
    is a rate-limit hint, not an identity — it is never used for auth and
    never stored anywhere.
    """
    headers = ctx.get("headers") or {}
    for name in ("x-forwarded-for", "x-real-ip", "cf-connecting-ip"):
        raw = str(headers.get(name) or "").strip()
        if raw:
            first = raw.split(",")[0].strip()
            if first:
                return first[:64]
    return str(ctx.get("client_ip") or "anonymous")


def public_chat(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Keyless demo chat behind the public website.

    No API key, no scopes, a 4000-char cap and a small rate limit *per
    visitor*. This is the product's conversational surface, so it goes
    through :meth:`Agent.reply` — multi-turn context for the browser's
    ``session_id``, memory recall, and honest scope. Programmatic access
    stays on the gated ``/v1/chat``, which keeps the raw pipeline voice.
    """
    text = str(body.get("message", "")).strip()
    if not text:
        return _err(400, "bad_request", "message is required")
    if len(text) > 4000:
        return _err(400, "bad_request", "message too long (max 4000)")

    visitor = visitor_id(ctx)
    if not _demo_limiter(agent.cfg).allow(visitor):
        agent.storage.count("chat", "demo_rate_limited",
                            detail={"visitor": visitor})
        return _err(429, "rate_limited",
                    "public demo is rate-limited - try again shortly, or "
                    "use /v1/chat with an owner-issued key")

    session_id = body.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        return _err(400, "bad_request", "session_id must be a string")

    res = agent.reply(text, session_id=session_id, sender="website-demo",
                      scopes=[], visitor=visitor)
    return _ok({**res, "request_id": ctx.get("request_id")})


def api_index(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Machine-readable index of every route and its required scopes."""
    return _ok({"agent": agent.mind.snapshot()["agent"],
                "routes": api_routes(agent),
                "pages": ["/", "/developers", "/approve", "/dashboard",
                          "/logo.svg", "/favicon.ico"],
                "note": "the browser pages are rendered by realaiagent."
                        "web_pages; /v1/* needs an owner-issued key"})


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


# -------------------------------------------------------------- generative

def gen_capabilities(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Which generative abilities are switched on (all local, all optional)."""
    return _ok({"capabilities": agent.generative.capabilities()})


def gen_talk(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Free conversation with the local language model (ChatGPT-style).

    ``messages`` = OpenAI-style ``[{"role","content"}]`` or a single
    ``message``. Nothing is sent off the machine: the model is Ollama on
    the owner's host.
    """
    msgs = body.get("messages")
    if not msgs:
        text = str(body.get("message", "")).strip()
        if not text:
            return _err(400, "bad_request", "message or messages required")
        msgs = [{"role": "user", "content": text}]
    if not isinstance(msgs, list) or len(msgs) > 40 or not all(
            isinstance(m, dict) and m.get("role") in ("user", "assistant", "system")
            and isinstance(m.get("content"), str) for m in msgs):
        return _err(400, "bad_request", "messages must be a list of "
                                        "{role, content} (max 40)")
    res = agent.generative.chat(
        [{"role": m["role"], "content": m["content"][:8000]} for m in msgs])
    if not res.ok:
        return _err(503, "unavailable", res.error)
    return _ok({"response": res.text, **res.to_dict(),
                "request_id": ctx.get("request_id")})


def gen_image(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        return _err(400, "bad_request", "prompt is required")
    res = agent.generative.image(
        prompt, width=int(body.get("width", 768)),
        height=int(body.get("height", 768)), steps=int(body.get("steps", 25)),
        negative=str(body.get("negative_prompt", "")))
    if not res.ok:
        return _err(503, "unavailable", res.error)
    return _ok({**res.to_dict(), "request_id": ctx.get("request_id")})


def gen_speak(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    text = str(body.get("text", "")).strip()
    if not text:
        return _err(400, "bad_request", "text is required")
    res = agent.generative.speak(text)
    if not res.ok:
        return _err(503, "unavailable", res.error)
    return _ok({**res.to_dict(), "request_id": ctx.get("request_id")})


def gen_transcribe(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Speech → text. ``audio`` is base64 (WAV/WebM/OGG), ``mime`` optional."""
    import base64 as _b64
    raw = body.get("audio")
    if not isinstance(raw, str) or not raw:
        return _err(400, "bad_request", "audio (base64) is required")
    try:
        audio = _b64.b64decode(raw.split(",", 1)[-1], validate=False)
    except Exception:  # noqa: BLE001
        return _err(400, "bad_request", "audio is not valid base64")
    mime = str(body.get("mime", "audio/wav"))
    ext = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mpeg": "mp3",
           "audio/mp4": "m4a"}.get(mime.split(";")[0], "wav")
    res = agent.generative.transcribe(audio, filename=f"audio.{ext}", mime=mime)
    if not res.ok:
        return _err(503, "unavailable", res.error)
    return _ok({**res.to_dict(), "request_id": ctx.get("request_id")})


def gen_slides(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Build a .pptx. ``topic`` (LLM/template outline) or explicit ``slides``."""
    topic = str(body.get("topic", "")).strip()
    slides = body.get("slides")
    if slides is not None and not isinstance(slides, list):
        return _err(400, "bad_request", "slides must be a list")
    if not topic and not slides:
        return _err(400, "bad_request", "topic or slides required")
    res = agent.generative.presentation(topic, slides=slides,
                                        count=int(body.get("count", 6)))
    if not res.ok:
        return _err(422, "failed", res.error)
    return _ok({**res.to_dict(), "request_id": ctx.get("request_id")})


def public_transcribe(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Keyless mic → text for the website chat (per-visitor rate limit)."""
    visitor = visitor_id(ctx)
    if not _demo_limiter(agent.cfg).allow(visitor):
        return _err(429, "rate_limited", "public demo is rate-limited")
    return gen_transcribe(agent, body, ctx)


def public_speak(agent: Agent, body: Dict[str, Any], ctx: Dict[str, Any]):
    """Keyless read-aloud for the website chat (per-visitor rate limit)."""
    visitor = visitor_id(ctx)
    if not _demo_limiter(agent.cfg).allow(visitor):
        return _err(429, "rate_limited", "public demo is rate-limited")
    body = {**body, "text": str(body.get("text", ""))[:1200]}
    return gen_speak(agent, body, ctx)


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
    ("GET", "/developers", [], developers),
    ("POST", "/public/chat", [], public_chat),
    ("GET", "/api.json", [], api_index),
    ("GET", "/healthz", [], healthz),

    ("GET", "/v1/status", ["chat"], status),
    ("GET", "/v1/counters", ["chat"], counters),
    ("GET", "/v1/usage", ["admin"], usage),
    ("GET", "/v1/events", ["chat"], events),

    ("POST", "/v1/chat", ["chat"], chat),

    ("GET", "/v1/generate/capabilities", [], gen_capabilities),
    ("POST", "/v1/generate/talk", ["chat"], gen_talk),
    ("POST", "/v1/generate/image", ["chat"], gen_image),
    ("POST", "/v1/generate/speak", ["chat"], gen_speak),
    ("POST", "/v1/generate/transcribe", ["chat"], gen_transcribe),
    ("POST", "/v1/generate/slides", ["chat"], gen_slides),
    ("POST", "/public/transcribe", [], public_transcribe),
    ("POST", "/public/speak", [], public_speak),

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
