"""Web layer (RealAI 0.3.0): dashboard, SSE streams, Telegram webhook.

Everything runs on the pure-stdlib core - no frameworks:

- ``GET /dashboard``   the owner's live dashboard (HTML + EventSource)
- ``GET /stream/<t>``  15 Server-Sent-Event topics (see :data:`TOPICS`)
- ``POST /webhook``    Telegram webhook (the owner's own bot, same repo)
- plus the full RealAI API (same ``dispatch`` pipeline as the threaded
  server: auth, scopes, rate limit, billing gate, ledger).

Built for the Vercel serverless handler (``api/index.py``): each SSE
response is a bounded window (``REALAI_SSE_WINDOW`` seconds, hard cap 25;
``maxDuration`` 30 in ``vercel.json``). The browser's EventSource
auto-reconnects and resends ``Last-Event-ID``, so streams stay gapless
across serverless freezes - no sticky sessions, no external AI.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from string import Template
from typing import Any, Deque, Dict, List, Optional, Tuple

from .engine import Agent
from .api.server import _RateLimiter, client_ip, dispatch, _serialize
from .telegram import TelegramMaster

#: The 15 SSE topics the web streams (each is ``GET /stream/<topic>``).
TOPICS: Tuple[Tuple[str, str], ...] = (
    ("all", "everything, everything"),
    ("thoughts", "the autonomous thinking loop"),
    ("actions", "executor outcomes (success / denied / awaiting)"),
    ("plans", "intent -> plan -> actions (per chat turn)"),
    ("chat", "incoming messages + replies"),
    ("learning", "intent training, Q-learning, skills"),
    ("memory", "episodic / semantic memory ops"),
    ("goals", "goal queue changes"),
    ("security", "intrusions, denials, permission asks"),
    ("billing", "users, balances, keys (business layer)"),
    ("devices", "device registration & control"),
    ("telegram", "telegram commands & outbound messages"),
    ("health", "heartbeat: uptime + ledger total"),
    ("status", "full mind snapshot (drives, mood, goals)"),
    ("errors", "counted errors"),
)

TOPIC_NAMES: Tuple[str, ...] = tuple(name for name, _ in TOPICS)

# storage count(category, ...) -> topic  (None = not streamed)
_COUNT_TOPICS = {
    "thoughts": "thoughts", "mind": "thoughts",
    "actions": "actions", "decisions": "actions",
    "chat": "chat",
    "learning": "learning",
    "memory": "memory",
    "goals": "goals",
    "security": "security", "permissions": "security",
    "billing": "billing", "users": "billing", "keys": "billing",
    "devices": "devices", "device_controls": "devices",
    "telegram": "telegram",
    "errors": "errors",
}
# (deliberately NOT streamed: "requests" - too noisy, "events" - covered
# by the log_event hook below, "system" - boot bookkeeping)

# storage log_event(etype, ...) -> topic
_EVENT_TOPICS = {
    "thought": "thoughts",
    "boot": "health",
    "autonomy": "status", "identity": "status", "values": "status",
    "intrusion": "security",
    "permission_request": "security", "permission_decided": "security",
    "permission": "security", "action_blocked": "security",
    "learn_skill": "learning", "learn_intents": "learning",
    "learn_chat": "learning",
    "user_requested": "billing", "user_status": "billing",
    "user_denied": "billing", "user_approved": "billing",
    "billing_topup": "billing", "key_created": "billing",
    "key_revoked": "billing",
    "goal_created": "goals", "goal_completed": "goals",
    "device_registered": "devices",
}


class SseHub:
    """In-process pub/sub bus fed by the storage ledger.

    ``Storage`` fires ``hook(source, name, data)`` on every ``count()``
    and ``log_event()``; this class maps those to topics and keeps a
    bounded ring buffer so late joiners (SSE reconnects) get a replay.
    Heartbeats and status snapshots are published by the web layer.
    """

    MAXLEN = 512

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self._lock = threading.Lock()
        self._seq = 0
        # (seq, ts, [topics], data)
        self._buf: Deque[Tuple[int, float, List[str], Dict[str, Any]]] = \
            deque(maxlen=self.MAXLEN)
        agent.storage.hooks.append(self._on_hook)

    # ------------------------------------------------------------- feed

    def _on_hook(self, source: str, name: str,
                 data: Dict[str, Any]) -> None:
        if source == "count":
            topic = _COUNT_TOPICS.get(name)
            if topic is None:
                return
            payload = {"category": name, "event": data.get("event"),
                       "detail": data.get("detail")}
        elif source == "event":
            topic = _EVENT_TOPICS.get(name)
            if topic is None:
                return
            payload = {"type": name, **data}
        else:
            return
        self.publish(topic, payload)

    def publish(self, topic: str, data: Dict[str, Any]) -> int:
        with self._lock:
            self._seq += 1
            self._buf.append((self._seq, time.time(), [topic, "all"], data))
            return self._seq

    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    # ------------------------------------------------------------- poll

    def read(self, topic: str, since: int = 0,
             wait: float = 0.0) -> Tuple[int, List[Dict[str, Any]]]:
        """Poll the buffer. Returns ``(last_seq, items)`` where each item
        is ``{"id", "topic", "ts", "data"}`` with ``id > since``. Blocks
        up to ``wait`` seconds when nothing new has arrived."""
        deadline = time.time() + max(0.0, wait)
        while True:
            with self._lock:
                last = self._seq
                # each item's PRIMARY topic (topics[0]) is used as the SSE
                # event name; the queried topic is in the item's topic list
                items = [
                    {"id": seq, "topic": topics[0], "ts": ts, "data": data}
                    for seq, ts, topics, data in self._buf
                    if seq > since and topic in topics
                ]
            if items or time.time() >= deadline:
                return last, items
            time.sleep(min(0.2, max(0.005, deadline - time.time())))


class TokenGuard:
    """Lockout for the web-token gated surfaces (``/approve``, ``/dashboard``,
    ``/stream/*``).

    A shared secret in a URL is only as strong as the number of guesses an
    attacker gets, so wrong tokens are counted **per visitor** and the
    visitor is locked out after a handful of failures. In-memory on purpose:
    it is a speed bump for a browser-facing gate, and the durable record of
    the attempt lives in the ledger (``security/web_token_denied``) plus an
    intrusion event and a Telegram alert when a lockout trips.
    """

    def __init__(self, max_failures: int = 10, window_s: float = 600.0,
                 lockout_s: float = 300.0) -> None:
        self.max_failures = max(1, int(max_failures))
        self.window_s = float(window_s)
        self.lockout_s = float(lockout_s)
        self._lock = threading.Lock()
        # visitor -> [failure_count, first_failure_ts, locked_until]
        self._state: Dict[str, List[float]] = {}

    def blocked_for(self, visitor: str) -> float:
        """Seconds left on a lockout, or 0 when the visitor may try."""
        now = time.time()
        with self._lock:
            row = self._state.get(visitor)
            if not row:
                return 0.0
            locked_until = row[2]
            if locked_until > now:
                return locked_until - now
            if now - row[1] > self.window_s:
                # the window rolled over: forget the old failures
                self._state.pop(visitor, None)
            return 0.0

    def fail(self, visitor: str) -> bool:
        """Record a wrong token. Returns True when this tripped a lockout."""
        now = time.time()
        with self._lock:
            row = self._state.setdefault(visitor, [0.0, now, 0.0])
            if now - row[1] > self.window_s:
                row[0], row[1] = 0.0, now
            row[0] += 1
            if row[0] >= self.max_failures and row[2] <= now:
                row[2] = now + self.lockout_s
                row[0] = 0.0
                return True
            return False

    def succeed(self, visitor: str) -> None:
        with self._lock:
            self._state.pop(visitor, None)

    def visitors(self) -> int:
        with self._lock:
            return len(self._state)


class WebApp:
    """One transport-agnostic app: API + dashboard + SSE + webhook."""

    def __init__(self, agent: Agent, hub: Optional[SseHub] = None,
                 master: Optional[TelegramMaster] = None) -> None:
        self.agent = agent
        self.hub = hub or SseHub(agent)
        self.master = master
        self.limiter = _RateLimiter(agent.cfg.rate_limit_per_min,
                                    agent.cfg.rate_burst)
        self.token_guard = TokenGuard()

    # ------------------------------------------------------------ handling

    def handle(self, method: str, path: str, query: Dict[str, str],
               headers: Dict[str, str],
               raw_body: bytes = b"") -> Tuple[int, str, bytes,
                                               Dict[str, str]]:
        """Serve one request. ``headers``: lowercased name -> value.
        Returns ``(status, content_type, body_bytes, extra_headers)``;
        ``content_type`` may be ``None`` for 204."""
        method = (method or "GET").upper()
        path = (path or "/").split("?", 1)[0]
        if len(path) > 1:
            path = path.rstrip("/")

        if method == "OPTIONS":
            return 204, None, b"", {
                "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                "Access-Control-Allow-Headers":
                    "Authorization, Content-Type",
            }
        if path == "/webhook" and method == "POST":
            return self._webhook(headers, raw_body)
        if path in ("/logo.svg", "/favicon.ico"):
            return self._asset(path)
        if path.startswith("/media/") and method == "GET":
            return self._media(path[len("/media/"):])
        if path == "/approve":
            return self._approve(method, query, headers, raw_body)
        if path == "/stream" or path.startswith("/stream/"):
            return self._stream(path, query, headers)
        if path == "/dashboard":
            return self._dashboard(query, headers)
        if path == "/cron/tick":
            return self._cron_tick(query, headers)

        status, ctype, body, extra = dispatch(
            self.agent, self.limiter, method, path, query, headers, raw_body)

        # Serverless autonomy: no background thread exists on Vercel, so
        # the agent thinks (rate-limited through the database) on the back
        # of ordinary traffic. Self-hosted keeps its own tick thread.
        if self.agent.cfg.is_vercel and method == "POST" \
                and path in ("/v1/chat", "/public/chat"):
            try:
                self.agent.tick_if_due()
            except Exception:  # noqa: BLE001
                pass

        # feed the "plans" topic with every completed chat turn
        if method == "POST" and path in ("/v1/chat", "/public/chat") \
                and status == 200:
            try:
                meta = (json.loads(body).get("meta") or {})
                self.hub.publish("plans", {
                    "intent": meta.get("intent"),
                    "confidence": meta.get("confidence"),
                    "plan": meta.get("plan"),
                    "actions": meta.get("actions"),
                    "success": meta.get("success"),
                })
            except (ValueError, AttributeError):
                pass
        return status, ctype, body, extra

    def _cron_tick(self, query: Dict[str, str],
                   headers: Dict[str, str]) -> Tuple[int, str, bytes, Dict[str, str]]:
        """``GET /cron/tick`` — one autonomous thought, for Vercel Cron.

        Guarded like the dashboard (``REALAI_WEB_TOKEN``); Vercel Cron also
        sends ``Authorization: Bearer $CRON_SECRET`` which is accepted when
        that env var is set.
        """
        import os as _os
        secret = _os.environ.get("CRON_SECRET", "")
        auth = headers.get("authorization", "")
        if secret and auth == f"Bearer {secret}":
            allowed = True
        else:
            allowed, _reason, _retry = self._web_gate(query, headers, "cron")
        if not allowed:
            return self._json(401, {"error": {"code": "unauthorized",
                                              "message": "cron token required"}})
        thought = self.agent.tick_if_due(force=True)
        snap = self.agent.mind.snapshot()
        return self._json(200, {"ok": True, "ticked": thought,
                                "thoughts": snap["thoughts"],
                                "mood": snap["mood"]["note"],
                                "storage": self.agent.storage.backend})

    # ------------------------------------------------------------- helpers

    def _json(self, status: int, payload: Any) -> Tuple[int, str, bytes,
                                                        Dict[str, str]]:
        body, ctype = _serialize(payload)
        return status, ctype, body, {}

    def _web_gate(self, query: Dict[str, str], headers: Dict[str, str],
                  surface: str) -> Tuple[bool, str, int]:
        """Decide whether a web-token gated surface may be served.

        Returns ``(allowed, reason, retry_after_seconds)`` where ``reason`` is
        ``""`` (allowed), ``"denied"`` (wrong/missing token) or ``"locked"``
        (too many wrong tokens from this visitor).

        The token is accepted **only** as ``?token=`` or the ``X-Web-Token``
        header — never from a request body. A body-borne secret would let a
        cross-site form forge a state-changing POST; a custom header cannot
        be set by a plain HTML form at all.
        """
        visitor = client_ip(headers) or "local"
        retry = self.token_guard.blocked_for(visitor)
        if retry > 0:
            self.agent.storage.count("security", "web_token_lockout",
                                     detail={"surface": surface})
            return False, "locked", int(retry) + 1

        expected = (self.agent.cfg.web_token or "").strip()
        given = str(query.get("token") or headers.get("x-web-token") or "")
        if not expected or given == expected:
            self.token_guard.succeed(visitor)
            return True, "", 0

        tripped = self.token_guard.fail(visitor)
        self.agent.storage.count("security", "web_token_denied",
                                 detail={"surface": surface})
        if tripped:
            self.agent.storage.log_event("intrusion", {
                "reason": "web_token_bruteforce", "surface": surface,
                "visitor": visitor})
            self.agent.telegram.send(
                f"🚨 [Security] repeated wrong web tokens on {surface} from "
                f"{visitor} - that visitor is locked out for "
                f"{int(self.token_guard.lockout_s)}s",
                dedup_key="intrusion-web-token", min_interval=300)
        return False, "denied", 0

    def _refusal(self, reason: str, retry: int,
                 surface: str) -> Tuple[int, str, bytes, Dict[str, str]]:
        """The JSON refusal for a gated surface (401, or 429 when locked)."""
        if reason == "locked":
            status, ctype, body, extra = self._json(429, {"error": {
                "code": "too_many_attempts",
                "message": f"too many wrong web tokens for {surface} - "
                           f"try again in {retry}s"}})
            extra = dict(extra, **{"Retry-After": str(max(1, retry))})
            return status, ctype, body, extra
        return self._json(401, {"error": {
            "code": "unauthorized",
            "message": "web token required (set REALAI_WEB_TOKEN)"}})

    def _status_snapshot(self) -> Dict[str, Any]:
        snap = self.agent.mind.snapshot()
        c = self.agent.storage.get_counters()
        return {
            "agent": snap["agent"], "autonomy": snap["autonomy"],
            "mood": snap["mood"], "drives": snap["drives"],
            "active_goals": snap["active_goals"],
            "thoughts": snap["thoughts"], "uptime_s": snap["uptime_s"],
            "grand_total": c.get("grand_total", 0),
            "totals": c.get("totals", {}),
        }

    def _heartbeat(self) -> Dict[str, Any]:
        snap = self.agent.mind.snapshot()
        return {
            "ok": True, "agent": snap["agent"],
            "uptime_s": snap["uptime_s"], "autonomy": snap["autonomy"],
            "grand_total": self.agent.storage.get_counters()
            .get("grand_total", 0),
        }

    # -------------------------------------------------------------- stream

    def _stream(self, path: str, query: Dict[str, str],
                headers: Dict[str, str]) -> Tuple[int, str, bytes,
                                                  Dict[str, str]]:
        allowed, reason, retry = self._web_gate(query, headers, "stream")
        if not allowed:
            return self._refusal(reason, retry, "/stream")

        topic = "" if path == "/stream" else path[len("/stream/"):]
        if not topic:
            snap = self.agent.mind.snapshot()
            return self._json(200, {
                "agent": snap["agent"],
                "topics": [
                    {"topic": n, "description": d} for n, d in TOPICS
                ],
                "count": len(TOPICS),
                "hint": "GET /stream/<topic> — SSE; window bounded by "
                        "REALAI_SSE_WINDOW (max 25s); EventSource "
                        "auto-reconnects with Last-Event-ID",
            })
        if topic not in TOPIC_NAMES:
            return self._json(404, {"error": {
                "code": "not_found",
                "message": f"unknown topic '{topic}' "
                           f"(see GET /stream)"}})

        cfg = self.agent.cfg
        try:
            window = min(float(query.get("window", cfg.sse_window)),
                         cfg.sse_window)
        except (TypeError, ValueError):
            window = cfg.sse_window
        window = max(0.0, min(window, 25.0))
        try:
            beat = float(query.get("beat", 5.0))
        except (TypeError, ValueError):
            beat = 5.0
        beat = max(0.1, beat)

        since = _int_or(query.get("since") or headers.get("last-event-id"),
                        0)

        lines = ["retry: 2000",
                 "event: open",
                 "data: " + json.dumps({
                     "topic": topic, "since": since,
                     "seq": self.hub.last_seq(),
                     "window_s": round(window, 2)}),
                 ""]
        if topic in ("status", "all"):
            self.hub.publish("status", self._status_snapshot())

        def _emit(items) -> None:
            nonlocal since
            for it in items:
                since = it["id"]
                lines.extend([f"id: {it['id']}", f"event: {it['topic']}",
                              "data: " + json.dumps(it["data"],
                                                    default=str),
                              ""])

        t0 = time.time()
        last_heartbeat = t0
        last_ping = t0
        while True:
            remaining = t0 + window - time.time()
            if remaining <= 0:
                break
            _last, items = self.hub.read(topic, since,
                                         wait=min(0.4, remaining))
            _emit(items)
            if time.time() - last_heartbeat >= beat:
                last_heartbeat = time.time()
                self.hub.publish("health", self._heartbeat())
            if not items and time.time() - last_ping >= beat:
                last_ping = time.time()
                lines.append(": keepalive")

        # final drain: covers window=0 (immediate replay) and any items
        # that landed during the last wait
        _last, items = self.hub.read(topic, since, wait=0)
        _emit(items)

        lines += ["event: window",
                  "data: " + json.dumps({"close": "window-ended",
                                         "seq": since}),
                  ""]
        sse_headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        }
        body = "\n".join(lines).encode("utf-8")
        return 200, "text/event-stream; charset=utf-8", body, sse_headers

    # ------------------------------------------------------------- webhook

    def _webhook(self, headers: Dict[str, str],
                 raw_body: bytes) -> Tuple[int, str, bytes, Dict[str, str]]:
        agent = self.agent
        if not agent.telegram.enabled:
            agent.storage.count("telegram", "webhook_rejected")
            return self._json(503, {"ok": False, "error": {
                "code": "telegram_not_configured",
                "message": "set REALAI_TELEGRAM_BOT_TOKEN + "
                           "REALAI_TELEGRAM_CHAT_ID"}})

        secret = agent.cfg.telegram_webhook_secret
        sent = headers.get("x-telegram-bot-api-secret-token", "")
        if secret and sent != secret:
            agent.storage.count("telegram", "webhook_rejected")
            return self._json(403, {"ok": False, "error": {
                "code": "bad_secret",
                "message": "X-Telegram-Bot-Api-Secret-Token mismatch"}})

        try:
            update = json.loads(raw_body.decode("utf-8"))
            if not isinstance(update, dict):
                raise ValueError("JSON object required")
        except (ValueError, UnicodeDecodeError):
            return self._json(400, {"ok": False, "error": {
                "code": "bad_json", "message": "invalid JSON update"}})

        if self.master is None:
            self.master = TelegramMaster(agent.telegram, agent,
                                         agent.users)
        # synchronous reply delivery (serverless: no worker threads)
        self.master.handle_update(update)
        agent.storage.count("telegram", "webhook_received")
        return self._json(200, {"ok": True})

    # ----------------------------------------------------------- dashboard

    def _dashboard(self, query: Dict[str, str],
                   headers: Optional[Dict[str, str]] = None
                   ) -> Tuple[int, str, bytes, Dict[str, str]]:
        allowed, reason, retry = self._web_gate(query, headers or {},
                                                "dashboard")
        if not allowed:
            return self._refusal(reason, retry, "/dashboard")
        snap = self.agent.mind.snapshot()
        topics_js = json.dumps(
            [{"name": n, "description": d} for n, d in TOPICS])
        html = _DASHBOARD_HTML.substitute(
            agent=snap["agent"], topics=topics_js,
            token=self.agent.cfg.web_token)
        body = html.encode("utf-8")
        return 200, "text/html; charset=utf-8", body, {}


    # ---------------------------------------------------------- brand assets

    def _asset(self, path: str) -> Tuple[int, str, bytes, Dict[str, str]]:
        """``/logo.svg`` and ``/favicon.ico``: our own generated SVG mark.

        No external asset is ever fetched, embedded or linked — the browser
        gets bytes this repo produced.
        """
        from . import web_pages
        body = web_pages.favicon_bytes(self.agent.mind.agent_name)
        self.agent.storage.count("requests", "asset",
                                 detail={"path": path})
        return 200, "image/svg+xml", body, {
            "Cache-Control": "public, max-age=3600",
        }

    def _media(self, name: str) -> Tuple[int, str, bytes, Dict[str, str]]:
        """``/media/<file>`` — images, audio and decks the agent generated.

        Only files the generative layer itself wrote (strict name pattern,
        inside ``data/media``) are served; they expire after
        ``media_ttl_hours``.
        """
        gen = getattr(self.agent, "generative", None)
        found = gen.fetch_media(name) if gen is not None else None
        if found is None:
            return 404, "application/json", json.dumps({"error": {
                "code": "not_found", "message": "no such media"}}).encode(), {}
        headers = {"Cache-Control": "private, max-age=3600"}
        if name.endswith(".pptx"):
            headers["Content-Disposition"] = \
                f'attachment; filename="{self.agent.mind.agent_name}-presentation.pptx"'
        self.agent.storage.count("requests", "media", detail={"name": name})
        return 200, found["mime"], found["data"], headers

    # --------------------------------------------------------- owner inbox

    def _approve(self, method: str, query: Dict[str, str],
                 headers: Dict[str, str],
                 raw_body: bytes) -> Tuple[int, str, bytes, Dict[str, str]]:
        """``/approve`` — the owner's approval inbox, web-token gated.

        Same gate as ``/dashboard`` and ``/stream/*``, enforced by
        :meth:`_web_gate`: with ``REALAI_WEB_TOKEN`` set the token must
        arrive as ``?token=`` or the ``X-Web-Token`` header — never in the
        body — and repeated wrong tokens lock the visitor out with a 429.
        Without it configured the inbox is open, and says so loudly at the
        top of the page.
        """
        from . import web_pages
        agent = self.agent
        method = (method or "GET").upper()

        # Gate before anything else is parsed: a locked-out visitor should
        # not even get to probe the JSON decoder.
        allowed, reason, retry = self._web_gate(query, headers, "approve")
        if not allowed:
            agent.storage.count("security", "approve_unauthorized")
            if reason == "locked":
                if method == "GET" and "text/html" in str(
                        headers.get("accept", "")):
                    return self._locked_page(retry)
                return self._refusal(reason, retry, "/approve")
            if method == "GET" and "text/html" in str(
                    headers.get("accept", "")):
                return self._token_gate()
            return self._json(401, {"error": {
                "code": "unauthorized",
                "message": "web token required (set REALAI_WEB_TOKEN, then "
                           "open /approve?token=<token>)"}})

        token = str(query.get("token") or headers.get("x-web-token") or "")
        payload: Dict[str, Any] = {}
        if raw_body and method == "POST":
            try:
                parsed = json.loads(raw_body.decode("utf-8"))
                if isinstance(parsed, dict):
                    payload = parsed
            except (ValueError, UnicodeDecodeError):
                return self._json(400, {"error": {
                    "code": "bad_json", "message": "invalid JSON body"}})

        if method == "POST":
            return self._approve_action(payload)
        if method != "GET":
            return self._json(405, {"error": {
                "code": "method_not_allowed",
                "message": "use GET (the inbox) or POST (a decision)"}})

        page = web_pages.approve_page(agent, token=token)
        return 200, "text/html; charset=utf-8", page.encode("utf-8"), {
            "Cache-Control": "no-store",
        }

    def _locked_page(self, retry: int) -> Tuple[int, str, bytes,
                                                 Dict[str, str]]:
        """HTML 429 — shown when a locked-out visitor reloads /approve."""
        from . import web_pages
        page = web_pages.locked_page(retry)
        return (429, "text/html; charset=utf-8", page.encode("utf-8"),
                {"Cache-Control": "no-store", "Retry-After": str(retry)})

    def _approve_action(self, payload: Dict[str, Any]
                        ) -> Tuple[int, str, bytes, Dict[str, str]]:
        """One tap in the inbox: approve / deny / unban / vip."""
        agent = self.agent
        action = str(payload.get("action", "")).strip().lower()
        username = str(payload.get("username", "")).strip().lower()
        if action not in ("approve", "deny", "unban", "vip"):
            return self._json(400, {"error": {
                "code": "bad_request",
                "message": "action must be approve|deny|unban|vip"}})
        if not username or len(username) > 48:
            return self._json(400, {"error": {
                "code": "bad_request",
                "message": "username (1-48 chars) is required"}})

        users = agent.users
        try:
            if action == "approve":
                out = users.approve(username, created_by="owner-web")
                # The key is returned once, to the gated page, and is never
                # written to the ledger or mailed anywhere.
                return self._json(200, {
                    "ok": True, "action": action, "username": username,
                    "status": out.get("status"),
                    "api_key": out["api_key"], "key_id": out["key_id"],
                    "key_scopes": out["key_scopes"],
                    "message": "approved - copy the key now, it is shown "
                               "exactly once",
                })
            if action == "deny":
                row = users.deny(username, created_by="owner-web")
                message = "denied and banned; their keys were revoked"
            elif action == "unban":
                row = users.unban(username, created_by="owner-web")
                message = "unbanned - they are PENDING again and still need " \
                          "an approval to get a key"
            else:
                vip = bool(payload.get("vip", True))
                row = users.set_vip(username, vip)
                message = ("VIP: requests are free" if vip
                           else "VIP removed: per-request billing resumes")
        except KeyError:
            return self._json(404, {"error": {
                "code": "not_found", "message": f"user {username} not found"}})

        agent.storage.count("users", f"web_{action}")
        agent.storage.log_event("user_status", {
            "username": username, "status": (row or {}).get("status"),
            "by": "owner-web", "action": action})
        return self._json(200, {
            "ok": True, "action": action, "username": username,
            "status": (row or {}).get("status"),
            "is_vip": bool((row or {}).get("is_vip")),
            "message": message,
        })

    def _token_gate(self) -> Tuple[int, str, bytes, Dict[str, str]]:
        """A browser-friendly 401: enter the web token, then see the inbox."""
        html = (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,"
            "initial-scale=1\"><title>Approvals — token required</title>"
            "<style>body{background:#0b0e14;color:#d7dce5;margin:0;"
            "font:15px/1.6 ui-sans-serif,system-ui,sans-serif;"
            "display:flex;align-items:center;justify-content:center;"
            "min-height:100vh}form{background:#12161f;border:1px solid "
            "#1f2633;border-radius:14px;padding:22px;max-width:420px;"
            "width:calc(100% - 40px)}h1{font-size:19px;margin:0 0 6px}"
            "p{color:#8b94a7;font-size:13.5px;margin:0 0 14px}"
            "input{width:100%;background:#0e1219;color:#d7dce5;border:1px "
            "solid #1f2633;border-radius:10px;padding:10px 12px;font:inherit;"
            "box-sizing:border-box}button{margin-top:12px;width:100%;"
            "background:#1d2a44;color:#fff;border:1px solid #7aa2f7;"
            "border-radius:10px;padding:10px;font:inherit;cursor:pointer}"
            "a{color:#7aa2f7}</style></head><body>"
            "<form method=\"get\" action=\"/approve\">"
            "<h1>Owner approval inbox</h1>"
            "<p>This page is gated by <code>REALAI_WEB_TOKEN</code>. Enter it "
            "to continue — the token stays in the URL of your own browser "
            "and is never stored by the agent.</p>"
            "<input name=\"token\" type=\"password\" autofocus "
            "placeholder=\"web token\" autocomplete=\"off\">"
            "<button type=\"submit\">Open the inbox</button>"
            "<p style=\"margin-top:14px\"><a href=\"/\">← back to the chat "
            "app</a></p></form></body></html>")
        return 401, "text/html; charset=utf-8", html.encode("utf-8"), {
            "Cache-Control": "no-store"}


def _int_or(raw: Any, default: int) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


_DASHBOARD_HTML = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>$agent — RealAI live</title>
<style>
 :root { color-scheme: dark; }
 body { background:#0b0e14; color:#d7dce5; font:14px/1.5
        ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;
        margin:0; padding:24px 16px; }
 main { max-width: 980px; margin: 0 auto; }
 h1 { font-size: 22px; margin: 0 0 2px; }
 .sub { color:#8b94a7; margin-bottom: 14px; font-size: 13px; }
 .bar { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
 .stat { color:#3fd68f; font-weight:600; }
 .stat.err { color:#f5c542; }
 .cnt { color:#8b94a7; font-size:12.5px; }
 .topics { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }
 button.t { background:#12161f; color:#d7dce5; border:1px solid #1f2633;
            border-radius:8px; padding:5px 10px; font-size:12.5px;
            cursor:pointer; }
 button.t:hover { border-color:#33405a; }
 button.t.on { background:#1d2a44; border-color:#7aa2f7; color:#fff; }
 #feed { background:#0e1219; border:1px solid #1f2633; border-radius:12px;
         padding:12px 14px; height: 62vh; overflow-y:auto;
         font:12.5px/1.7 ui-monospace,SFMono-Regular,monospace; }
 .row { border-bottom:1px solid #151b26; padding:2px 0; word-break:break-all; }
 .seq { color:#5c6577; } .tp { color:#7aa2f7; font-weight:600; }
 code { color:#a8b2c5; }
 footer { color:#5c6577; font-size:12px; margin-top:12px; }
</style></head><body><main>
 <h1>🤖 $agent — RealAI · live</h1>
 <div class="sub">fully local, owner-controlled · no external AI ·
  pure Python stdlib · Vercel free tier (SSE windows auto-reconnect)</div>
 <div class="bar">
   <span class="stat" id="stat">starting…</span>
   <span class="cnt" id="cnt"></span>
 </div>
 <div class="topics" id="topics"></div>
 <div id="feed"></div>
 <footer>15 SSE topics · bounded windows · everything is counted</footer>
</main>
<script>
const TOPICS = $topics;
const TOKEN = "$token";
const params = TOKEN ? ("?token=" + TOKEN) : "";
const feed = document.getElementById("feed");
const stat = document.getElementById("stat");
const cnt = document.getElementById("cnt");
const box = document.getElementById("topics");
let es = null;

function line(html) {
  const div = document.createElement("div");
  div.className = "row";
  div.innerHTML = html;
  feed.appendChild(div);
  while (feed.children.length > 300) feed.removeChild(feed.firstChild);
  feed.scrollTop = feed.scrollHeight;
}
function esc(s) {
  return String(s).replace(/[&<>"]/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}
function open(topic) {
  if (es) { es.close(); es = null; }
  document.querySelectorAll("button.t")
    .forEach(b => b.classList.toggle("on", b.dataset.t === topic));
  stat.textContent = "connecting…";
  stat.classList.remove("err");
  es = new EventSource("/stream/" + topic + params);
  const names = ["open", "window"].concat(TOPICS.map(t => t.name));
  names.forEach(ev => es.addEventListener(ev, e => {
    let d = {};
    try { d = JSON.parse(e.data); } catch (_) {}
    if (ev === "open") {
      stat.textContent = "live — " + topic;
      return;
    }
    if (ev === "window") {
      stat.textContent = "window closed — reconnecting…";
      return;
    }
    if (ev === "health" || ev === "status") {
      cnt.textContent = "grand_total " + (d.grand_total == null ? "?"
        : d.grand_total) + " · uptime " + Math.round(d.uptime_s || 0)
        + "s · autonomy " + (d.autonomy ? "ON" : "off");
    }
    line("<span class='seq'>" + (e.lastEventId || "") + "</span> "
      + "<span class='tp'>" + esc(ev) + "</span> "
      + "<code>" + esc(JSON.stringify(d)) + "</code>");
  }));
  es.onerror = () => {
    stat.textContent = "reconnecting…";
    stat.classList.add("err");
  };
}
TOPICS.forEach(t => {
  const b = document.createElement("button");
  b.className = "t";
  b.dataset.t = t.name;
  b.title = t.description;
  b.textContent = t.name;
  b.addEventListener("click", () => open(t.name));
  box.appendChild(b);
});
open("all");
</script>
</body></html>""")
