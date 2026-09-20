"""Vercel serverless adapter (free tier) - the real entrypoint contract.

Vercel's Python runtime does **not** support the AWS-Lambda style
``handler(event, context) -> dict`` shape. Per the runtime reference, each
``.py`` file under ``/api`` must define one of::

    app          an ASGI or WSGI application
    application  a WSGI application
    handler      a class inheriting from ``BaseHTTPRequestHandler``

The bridge loads ``app`` first and, because it is not a coroutine
function, drives it as **WSGI**: ``app(environ, start_response)``,
expecting an iterable of *bytes* back. Exporting a plain function that
returns a ``dict`` makes the bridge iterate the dict's *string keys*,
which raises ``TypeError: sequence item 0: expected a bytes-like object,
str found`` and surfaces to the browser as **HTTP 500**.

So this module exposes a genuine PEP 3333 WSGI callable (:data:`app`)
built on the stdlib only - no Flask, no dependencies - that adapts the
WSGI ``environ`` onto :meth:`WebApp.handle`. The dict-in/dict-out
:func:`handle_event` is kept for Lambda/Netlify style callers and tests.

Both ``api/index.py`` and ``api/[...path].py`` import from here with an
*absolute* import (``from realaiagent.vercel import app``): a relative
``from .index import ...`` fails with ``ImportError: attempted relative
import with no known parent package`` because Vercel loads each function
file as a standalone module.
"""

from __future__ import annotations

import base64
import os
import sys
import traceback
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Tuple
from urllib.parse import parse_qs

from .web import WebApp
from .telegram import TelegramMaster

_STATE: Dict[str, Any] = {}

#: WSGI reason phrases, so we emit a correct ``"200 OK"`` status line.
_REASONS: Dict[int, str] = {
    200: "OK", 201: "Created", 202: "Accepted", 204: "No Content",
    206: "Partial Content", 301: "Moved Permanently", 302: "Found",
    304: "Not Modified", 400: "Bad Request", 401: "Unauthorized",
    403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
    408: "Request Timeout", 413: "Payload Too Large", 422:
    "Unprocessable Entity", 429: "Too Many Requests",
    500: "Internal Server Error", 501: "Not Implemented",
    503: "Service Unavailable", 504: "Gateway Timeout",
}


# --------------------------------------------------------------------- app
def get_app() -> WebApp:
    """Build (and memoise per warm instance) the :class:`WebApp`."""
    app = _STATE.get("app")
    if app is not None:
        return app
    from .config import Config
    from .engine import Agent
    cfg = Config.from_env()
    agent = Agent(cfg)
    try:
        agent.keys.ensure_owner_key()
    except Exception:
        pass
    master = None
    if agent.telegram.enabled:
        master = TelegramMaster(agent.telegram, agent, agent.users)
        # No long-running threads on the free tier: webhook mode instead.
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


# ------------------------------------------------------- dict-style events
def normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a Lambda/Netlify shaped ``event`` into one request dict."""
    event = event or {}
    method = str(event.get("requestMethod") or event.get("httpMethod")
                 or "GET").upper()
    raw_url = str(event.get("rawUrl") or event.get("path") or "/")
    if "//" in raw_url:
        try:
            tail = raw_url.split("//", 1)[1]
            raw_url = ("/" + tail.split("/", 1)[1]) if "/" in tail else "/"
        except Exception:
            raw_url = "/"
    path, _, qs = raw_url.partition("?")
    query = {k: v[0] for k, v in parse_qs(qs, keep_blank_values=True).items()
             } if qs else {}
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
    elif isinstance(body, bytes):
        raw = body
    else:
        raw = b""
    return {"method": method, "path": path or "/", "query": query,
            "headers": headers, "body": raw}


def _strip_api_prefix(path: str) -> str:
    """/api is where Vercel mounts us; the web layer is rooted at /."""
    if path == "/api" or path.startswith("/api/"):
        return path[4:] or "/"
    return path


def handle_event(event: Dict[str, Any],
                 context: Any = None) -> Dict[str, Any]:
    """Serve one dict-shaped event -> dict-shaped response.

    Kept for Lambda/Netlify style callers and the test-suite. **Vercel
    does not use this** - it uses :data:`app` below.
    """
    req = normalize_event(event)
    path = _strip_api_prefix(req["path"])
    status, ctype, body, extra = get_app().handle(
        req["method"], path, req["query"], req["headers"], req["body"])
    return _response(status, ctype, body, extra)


def _response(status: int, ctype: str, body: bytes,
              extra: Dict[str, str]) -> Dict[str, Any]:
    headers: Dict[str, str] = {"Access-Control-Allow-Origin": "*", **extra}
    if ctype:
        headers["Content-Type"] = ctype
    if status != 204:
        headers["Content-Length"] = str(len(body))
    return {"statusCode": status, "headers": headers,
            "body": body.decode("utf-8", "replace"), "isBase64Encoded": False}


# ------------------------------------------------------------------- WSGI
def environ_to_request(environ: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a WSGI ``environ`` into the same shape as :func:`normalize_event`.

    WSGI carries the method in ``REQUEST_METHOD``, the path in
    ``PATH_INFO``, the query string in ``QUERY_STRING``, headers as
    ``HTTP_*`` keys (except ``CONTENT_TYPE`` / ``CONTENT_LENGTH``) and the
    body as a stream in ``wsgi.input``.
    """
    method = str(environ.get("REQUEST_METHOD") or "GET").upper()
    path = environ.get("PATH_INFO") or "/"
    if not path.startswith("/"):
        path = "/" + path
    qs = environ.get("QUERY_STRING") or ""
    query = {k: v[0] for k, v in parse_qs(qs, keep_blank_values=True).items()}

    headers: Dict[str, str] = {}
    for key, value in environ.items():
        if key.startswith("HTTP_"):
            name = key[5:].replace("_", "-").lower()
            headers[name] = str(value)
        elif key in ("CONTENT_TYPE", "CONTENT_LENGTH"):
            headers[key.replace("_", "-").lower()] = str(value)

    raw = b""
    stream = environ.get("wsgi.input")
    if stream is not None:
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except (TypeError, ValueError):
            length = 0
        try:
            raw = stream.read(length) if length > 0 else b""
        except Exception:
            raw = b""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    return {"method": method, "path": path, "query": query,
            "headers": headers, "body": raw or b""}


def _status_line(status: int) -> str:
    return "%d %s" % (status, _REASONS.get(status, "Unknown"))


def wsgi_app(environ: Dict[str, Any],
             start_response: Callable[..., Any]) -> Iterable[bytes]:
    """The PEP 3333 WSGI application Vercel's Python runtime invokes.

    Exported below as both ``app`` and ``application``. Errors are turned
    into a JSON 500 *and* written to stderr, so Vercel's function log
    shows the traceback instead of a blank 500 page.
    """
    try:
        req = environ_to_request(environ)
        path = _strip_api_prefix(req["path"])
        status, ctype, body, extra = get_app().handle(
            req["method"], path, req["query"], req["headers"], req["body"])
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        payload = (b'{"ok":false,"error":{"code":"internal",'
                   b'"message":"unhandled error in the RealAI adapter"}}')
        start_response("500 Internal Server Error", [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(payload))),
            ("Access-Control-Allow-Origin", "*"),
        ])
        return [payload]

    headers: List[Tuple[str, str]] = [
        ("Access-Control-Allow-Origin", "*")]
    headers.extend((str(k), str(v)) for k, v in (extra or {}).items())
    if ctype:
        headers.append(("Content-Type", ctype))
    if status != 204:
        headers.append(("Content-Length", str(len(body))))
    start_response(_status_line(status), headers)
    return [body if isinstance(body, bytes) else bytes(body)]


#: What ``@vercel/python`` actually loads: a WSGI callable, not a dict
#: handler. ``app`` is checked first, ``application`` is the WSGI alias.
app = wsgi_app
application = wsgi_app
