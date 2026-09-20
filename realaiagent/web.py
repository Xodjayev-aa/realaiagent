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
from .api.server import _RateLimiter, dispatch, _serialize
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


class WebApp:
    """One transport-agnostic app: API + dashboard + SSE + webhook."""

    def __init__(self, agent: Agent, hub: Optional[SseHub] = None,
                 master: Optional[TelegramMaster] = None) -> None:
        self.agent = agent
        self.hub = hub or SseHub(agent)
        self.master = master
        self.limiter = _RateLimiter(agent.cfg.rate_limit_per_min,
                                    agent.cfg.rate_burst)

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
        if path == "/stream" or path.startswith("/stream/"):
            return self._stream(path, query, headers)
        if path == "/dashboard":
            return self._dashboard(query)

        status, ctype, body, extra = dispatch(
            self.agent, self.limiter, method, path, query, headers, raw_body)

        # feed the "plans" topic with every completed chat turn
        if method == "POST" and path == "/v1/chat" and status == 200:
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

    # ------------------------------------------------------------- helpers

    def _json(self, status: int, payload: Any) -> Tuple[int, str, bytes,
                                                        Dict[str, str]]:
        body, ctype = _serialize(payload)
        return status, ctype, body, {}

    def _check_web_token(self, query: Dict[str, str],
                         headers: Dict[str, str]) -> bool:
        tok = self.agent.cfg.web_token
        if not tok:
            return True
        return (query.get("token") == tok
                or headers.get("x-web-token") == tok)

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
        if not self._check_web_token(query, headers):
            return self._json(401, {"error": {
                "code": "unauthorized",
                "message": "web token required (set REALAI_WEB_TOKEN)"}})

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

    def _dashboard(self, query: Dict[str, str]) -> Tuple[int, str, bytes,
                                                         Dict[str, str]]:
        if not self._check_web_token(query, {}):
            return self._json(401, {"error": {
                "code": "unauthorized",
                "message": "web token required (set REALAI_WEB_TOKEN)"}})
        snap = self.agent.mind.snapshot()
        topics_js = json.dumps(
            [{"name": n, "description": d} for n, d in TOPICS])
        html = _DASHBOARD_HTML.substitute(
            agent=snap["agent"], topics=topics_js,
            token=self.agent.cfg.web_token)
        body = html.encode("utf-8")
        return 200, "text/html; charset=utf-8", body, {}


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
