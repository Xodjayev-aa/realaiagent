"""Vercel serverless entrypoint for RealAI Agent (free tier).

One pure-stdlib ``handler(event, context)`` serves the whole product:

    /            the agent index page (API face)
    /api.json    machine-readable route index
    /healthz     liveness (public)
    /v1/...      the full RealAI API (Bearer auth, scopes, billing, ledger)
    /dashboard   live owner dashboard (HTML + EventSource)
    /stream/<t>  15 SSE topics (bounded window + auto-reconnect)
    /webhook     Telegram webhook (the owner's own bot, same repo)

Vercel specifics (free tier):

* **Storage lives in /tmp** - the only writable disk.
  ``Config.from_env()`` detects ``VERCEL=1`` and switches the data dir to
  ``/tmp/realai``; the sqlite ledger, pepper and keys all go there.
* **Telegram runs in webhook mode.** No long-poll thread on serverless -
  the bot points at ``/webhook`` (``setWebhook`` with the shared secret),
  and replies are sent synchronously before the invocation ends.
* **SSE streams are bounded windows** (``REALAI_SSE_WINDOW`` seconds,
  default 20, hard cap 25; ``maxDuration`` 30 in ``vercel.json``).
  EventSource auto-reconnects and resends ``Last-Event-ID`` - the hub
  replays from there, so the stream stays gapless across freezes.
* **No external AI.** The handler is stdlib Python; the only outbound
  calls the agent ever makes are to the owner's own Telegram bot.

The handler also accepts the AWS-Lambda-style event shape, so it can be
run on any serverless platform (or unit-tested with a plain dict).
"""

from __future__ import annotations

import base64
from typing import Any, Dict
from urllib.parse import parse_qs

from realaiagent.telegram import TelegramMaster
from realaiagent.web import WebApp

_STATE: Dict[str, Any] = {}


def get_app() -> WebApp:
    """The process-wide app (warm across invocations in one container).

    The agent is built once per cold start from ``Config.from_env()`` -
    on Vercel that means ``VERCEL=1`` -> data dir ``/tmp/realai`` and
    telegram webhook mode.
    """
    app = _STATE.get("app")
    if app is not None:
        return app

    from realaiagent.config import Config
    from realaiagent.engine import Agent

    cfg = Config.from_env()
    agent = Agent(cfg)
    try:
        agent.keys.ensure_owner_key()
    except Exception:  # noqa: BLE001 - /tmp should always work; ignore
        pass

    master = None
    if agent.telegram.enabled:
        master = TelegramMaster(agent.telegram, agent, agent.users)
        if not cfg.is_vercel:
            master.start()  # long-poll mode (local / self-hosted box)
        # on Vercel: webhook mode - handle_update() runs per /webhook call

    app = WebApp(agent, master=master)
    _STATE["app"] = app
    return app


def reset_state() -> None:
    """Drop the singleton (tests, local re-runs)."""
    _STATE.clear()


def normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Accept the Vercel event shape or the AWS-Lambda shape.

    Vercel:  {requestMethod, rawUrl, requestHeaders, body, isBase64Encoded}
    Lambda:  {httpMethod, path, queryString, headers, body, isBase64Encoded}
    """
    event = event or {}
    method = str(event.get("requestMethod") or event.get("httpMethod")
                 or "GET").upper()

    raw_url = str(event.get("rawUrl") or event.get("path") or "/")
    if "//" in raw_url:  # absolute URL https://host/path?qs
        raw_url = "/" + raw_url.split("//", 1)[1].split("/", 1)[1] \
            if "/" in raw_url.split("//", 1)[1] else "/"
    path, _, qs = raw_url.partition("?")
    query = {k: v[0] for k, v in parse_qs(qs).items()} if qs else {}
    qs_map = event.get("queryString") or event.get("query") or {}
    if isinstance(qs_map, dict) and qs_map:
        query.update({k: str(v) for k, v in qs_map.items()})

    headers = {str(k).lower(): str(v)
               for k, v in (event.get("requestHeaders")
                            or event.get("headers") or {}).items()}

    body = event.get("body")
    if event.get("isBase64Encoded") and isinstance(body, str):
        raw = base64.b64decode(body or "")
    elif isinstance(body, str):
        raw = body.encode("utf-8")
    else:
        raw = b""
    return {"method": method, "path": path or "/", "query": query,
            "headers": headers, "body": raw}


def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """The Vercel (or Lambda-style) serverless entrypoint."""
    req = normalize_event(event)
    path = req["path"]
    # requests arrive as /api/<route> (Vercel function mount) - normalize
    if path == "/api" or path.startswith("/api/"):
        path = path[4:] or "/"

    status, ctype, body, extra = get_app().handle(
        req["method"], path, req["query"], req["headers"], req["body"])

    resp_headers: Dict[str, str] = {"Access-Control-Allow-Origin": "*",
                                    **extra}
    if ctype:
        resp_headers["Content-Type"] = ctype
    if status != 204:
        resp_headers["Content-Length"] = str(len(body))
    return {
        "statusCode": status,
        "headers": resp_headers,
        "body": body.decode("utf-8", "replace"),
        "isBase64Encoded": False,
    }
