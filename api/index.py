"""Vercel serverless entrypoint for RealAI Agent (free tier)."""
from __future__ import annotations

import base64
import os
from typing import Any, Dict
from urllib.parse import parse_qs

from realaiagent.web import WebApp
from realaiagent.telegram import TelegramMaster

_STATE: Dict[str, Any] = {}

def get_app() -> WebApp:
    app = _STATE.get("app")
    if app is not None:
        return app
    from realaiagent.config import Config
    from realaiagent.engine import Agent
    cfg = Config.from_env()
    agent = Agent(cfg)
    try:
        agent.keys.ensure_owner_key()
    except Exception:
        pass
    master = None
    if agent.telegram.enabled:
        master = TelegramMaster(agent.telegram, agent, agent.users)
        if not os.getenv("VERCEL"):
            master.start()
    app = WebApp(agent)
    try:
        app._tg_master = master
    except Exception:
        pass
    _STATE["app"] = app
    return app

def reset_state() -> None:
    _STATE.clear()

def normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    event = event or {}
    method = str(event.get("requestMethod") or event.get("httpMethod") or "GET").upper()
    raw_url = str(event.get("rawUrl") or event.get("path") or "/")
    if "//" in raw_url:
        try:
            raw_url = "/" + raw_url.split("//",1)[1].split("/",1)[1] if "/" in raw_url.split("//",1)[1] else "/"
        except Exception:
            raw_url = "/"
    path, _, qs = raw_url.partition("?")
    query = {k: v[0] for k,v in parse_qs(qs).items()} if qs else {}
    qs_map = event.get("queryString") or event.get("query") or {}
    if isinstance(qs_map, dict) and qs_map:
        query.update({k: str(v) for k,v in qs_map.items()})
    headers = {str(k).lower(): str(v) for k,v in (event.get("requestHeaders") or event.get("headers") or {}).items()}
    body = event.get("body")
    if event.get("isBase64Encoded") and isinstance(body, str):
        import base64 as _b64
        raw = _b64.b64decode(body or "")
    elif isinstance(body, str):
        raw = body.encode("utf-8")
    elif isinstance(body, bytes):
        raw = body
    else:
        raw = b""
    return {"method": method, "path": path or "/", "query": query, "headers": headers, "body": raw}

def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    req = normalize_event(event)
    path = req["path"]
    if path == "/api" or path.startswith("/api/"):
        path = path[4:] or "/"
    status, ctype, body, extra = get_app().handle(req["method"], path, req["query"], req["headers"], req["body"])
    resp_headers: Dict[str, str] = {"Access-Control-Allow-Origin": "*", **extra}
    if ctype:
        resp_headers["Content-Type"] = ctype
    if status != 204:
        resp_headers["Content-Length"] = str(len(body))
    return {"statusCode": status, "headers": resp_headers, "body": body.decode("utf-8", "replace"), "isBase64Encoded": False}

app = handler
application = handler
