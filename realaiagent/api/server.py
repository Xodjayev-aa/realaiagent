"""The HTTP API server (stdlib http.server, threaded).

Middleware order: CORS -> route match -> auth (Bearer key) -> scope check
-> rate limit -> body parse -> handler -> JSON response. Every request is
counted in the ledger (including 4xx/5xx, with latency).

The whole pipeline is a pure function, :func:`dispatch`, so any transport
can run it: the threaded HTTP server (``ApiServer``), the Vercel
serverless handler (``api/index.py``), or tests directly.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from ..engine import Agent
from . import routes as R


class _RateLimiter:
    """Per-key token bucket."""

    def __init__(self, rate_per_min: int, burst: int) -> None:
        self.rate = rate_per_min / 60.0
        self.burst = float(burst)
        self._lock = threading.Lock()
        self._buckets: Dict[str, List[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            tokens = self._buckets.setdefault(key, [self.burst, now])
            tokens[0] = min(self.burst, tokens[0] + (now - tokens[1]) * self.rate)
            tokens[1] = now
            if tokens[0] < 1.0:
                return False
            tokens[0] -= 1.0
            return True


def client_ip(headers: Dict[str, str]) -> str:
    """Best-effort client address, for per-visitor demo rate limiting only.

    Proxied deployments (Vercel, nginx) put the real client in
    ``X-Forwarded-For``; the threaded server injects the socket peer as
    ``X-Real-Ip``. This is a rate-limit hint, never an identity: it grants
    nothing and is not stored.
    """
    for name in ("x-forwarded-for", "x-real-ip", "cf-connecting-ip"):
        raw = str(headers.get(name) or "").strip()
        if raw:
            first = raw.split(",")[0].strip()
            if first:
                return first[:64]
    return ""


def _compile_routes() -> List[Tuple[str, "re.Pattern[str]", List[str], Any]]:
    out = []
    for method, path, scopes, handler in R.ROUTES:
        parts = path.split("/")
        regex = []
        param_names: List[str] = []
        for part in parts:
            if part.startswith("{") and part.endswith("}"):
                param_names.append(part[1:-1])
                regex.append(r"(?P<" + part[1:-1] + r">[^/]+)")
            else:
                regex.append(re.escape(part))
        out.append((method, re.compile("^" + "/".join(regex) + "/?$"),
                    scopes, handler))
    return out


_ROUTES = _compile_routes()


def _scope_ok(key_row: Dict[str, Any], scopes: List[str]) -> bool:
    key_scopes = set(key_row.get("scopes", []))
    if "owner" in key_scopes:
        return True
    return bool(key_scopes & set(scopes))


def _serialize(payload: Any) -> Tuple[bytes, str]:
    if isinstance(payload, str):
        return payload.encode("utf-8"), "text/html; charset=utf-8"
    return (json.dumps(payload, default=str).encode("utf-8"),
            "application/json")


def dispatch(agent: Agent, limiter: _RateLimiter, method: str, path: str,
             query: Dict[str, str], headers: Dict[str, str],
             raw_body: bytes = b"") -> Tuple[int, str, bytes, Dict[str, str]]:
    """Run the full middleware pipeline for one request.

    ``headers`` maps lowercased header name -> value. ``raw_body`` is the
    request body (may be empty). Returns ``(status, content_type,
    body_bytes, extra_headers)`` where ``extra_headers`` always carries
    the ``X-Request-Id``.
    """
    request_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    storage = agent.storage

    def _count_request(key_id: Optional[str], route: str, code: int) -> None:
        parts = route.split()
        name = parts[1] if len(parts) > 1 else route
        storage.count("requests", f"{code}_{name}", key_id=key_id,
                      detail={"route": route,
                              "latency_ms": round((time.time() - t0) * 1000,
                                                  2)})

    def _resp(status: int, payload: Any) -> Tuple[int, str, bytes,
                                                  Dict[str, str]]:
        body, ctype = _serialize(payload)
        return status, ctype, body, {"X-Request-Id": request_id}

    # ---- route match
    matched = None
    for m, pattern, scopes, handler in _ROUTES:
        if m != method:
            continue
        mm = pattern.match(path)
        if mm:
            matched = (scopes, handler, mm.groupdict())
            break
    if matched is None:
        _count_request(None, method + " " + path, 404)
        return _resp(404, {"error": {
            "code": "not_found",
            "message": f"no route {method} {path}"}})

    scopes, handler, params = matched
    key_row: Optional[Dict[str, Any]] = None

    # ---- auth
    if scopes:
        auth = headers.get("authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else auth
        key_row = agent.keys.verify(token) if token else None
        if key_row is None:
            if token:
                # someone presented a key that does not exist
                storage.count("security", "intrusion")
                storage.log_event("intrusion", {
                    "reason": "invalid_key",
                    "path": f"{method} {path}"})
                agent.telegram.send(
                    "🚨 [Security] an invalid API key was "
                    f"presented for {method} {path}",
                    dedup_key="intrusion-invalid", min_interval=60)
            _count_request(None, method + " " + path, 401)
            return _resp(401, {"error": {
                "code": "unauthorized",
                "message": "missing or invalid API key"}})
        if not _scope_ok(key_row, scopes):
            _count_request(key_row["id"], method + " " + path, 403)
            return _resp(403, {"error": {
                "code": "forbidden",
                "message": f"key lacks scope for {path} "
                           f"(needs one of {scopes})"}})
        if not limiter.allow(key_row["id"]):
            _count_request(key_row["id"], method + " " + path, 429)
            return _resp(429, {"error": {
                "code": "rate_limited",
                "message": "too many requests"}})

        # ---- user/billing gate (customer keys)
        bill = agent.users.authorize_request(key_row)
        if not bill.ok:
            if bill.intrusion:
                storage.count("security", "intrusion")
                storage.log_event("intrusion", {
                    "reason": bill.code,
                    "user": key_row.get("user"),
                    "path": f"{method} {path}"})
                agent.telegram.send(
                    "🚨 [Security] " + bill.message,
                    dedup_key=f"intrusion-{bill.code}", min_interval=60)
            _count_request(key_row["id"], method + " " + path, bill.status)
            return _resp(bill.status, {"error": {
                "code": bill.code, "message": bill.message}})

        storage.key_bump_usage(key_row["id"])

    # ---- body
    max_body = agent.cfg.max_body_bytes
    if len(raw_body) > max_body:
        _count_request((key_row or {}).get("id"), method + " " + path, 413)
        return _resp(413, {"error": {
            "code": "body_too_large", "message": f"body > {max_body} bytes"}})
    body: Dict[str, Any] = {}
    if raw_body:
        try:
            body = json.loads(raw_body.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
        except (ValueError, UnicodeDecodeError):
            _count_request((key_row or {}).get("id"), method + " " + path,
                           400)
            return _resp(400, {"error": {
                "code": "bad_json", "message": "invalid JSON body"}})

    ctx = {"key": key_row, "query": query,
           "request_id": request_id, "params": params,
           "headers": headers, "client_ip": client_ip(headers)}
    try:
        status, payload = handler(agent, body, ctx)
        code = 500 if status >= 500 else status
        out = _resp(status, payload)
        _count_request((key_row or {}).get("id"), method + " " + path, code)
        return out
    except Exception as exc:  # noqa: BLE001
        storage.count("errors", "http_500")
        _count_request((key_row or {}).get("id"), method + " " + path, 500)
        return _resp(500, {"error": {
            "code": "internal", "message": str(exc)}})


class ApiServer:
    """The threaded HTTP server.

    It runs the *same* :class:`~realaiagent.web.WebApp` object as the
    serverless handler, so both serve exactly the same surface: the product
    pages (``/``, ``/developers``), the owner inbox (``/approve``), the
    assets (``/logo.svg``, ``/favicon.ico``), the dashboard, the 15 SSE
    streams, the Telegram webhook and the whole ``/v1/*`` API. It used to
    call :func:`dispatch` directly, which silently served only the API half
    — a browser pointed at the threaded server got 404s for everything the
    web layer adds.
    """

    def __init__(self, agent: Agent, host: str, port: int,
                 web: Any = None) -> None:
        self.agent = agent
        self.host = host
        self.port = port
        if web is None:
            from ..web import WebApp   # late import: web imports this module
            web = WebApp(agent)
        self.web = web
        self.limiter = web.limiter
        handler = self._make_handler()
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True

    def _make_handler(self):
        web = self.web
        max_body = self.agent.cfg.max_body_bytes

        class Handler(BaseHTTPRequestHandler):
            server_version = "RealAI/0.3"
            protocol_version = "HTTP/1.1"

            # ---------------------------------------------------- plumbing

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
                pass  # ledger counts instead of stderr spam

            def _send_raw(self, status: int, body: bytes,
                          content_type: str,
                          extra_headers: Dict[str, str]) -> None:
                self.send_response(status)
                if content_type:
                    self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Authorization, Content-Type")
                for k, v in extra_headers.items():
                    self.send_header(k, v)
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Authorization, Content-Type")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _dispatch(self) -> None:
                url = urlparse(self.path)
                path = url.path.rstrip("/") or "/"
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                # cap the read at the body limit + 1 byte so oversized
                # bodies are rejected (413) without full allocation
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except (TypeError, ValueError):
                    length = 0
                read = min(max(length, 0), max_body + 1)
                try:
                    raw = self.rfile.read(read) if read else b""
                except OSError:
                    raw = b""
                headers = {k.lower(): v for k, v in self.headers.items()}
                # the peer address: used only as a per-visitor demo hint
                try:
                    headers.setdefault("x-real-ip", self.client_address[0])
                except (AttributeError, IndexError, TypeError):
                    pass
                status, ctype, body, extra = web.handle(
                    self.command, path, query, headers, raw)
                self._send_raw(status, body, ctype, extra)

            # ------------------------------------------------------ methods

            def do_GET(self) -> None:  # noqa: N802
                self._dispatch()

            def do_POST(self) -> None:  # noqa: N802
                self._dispatch()

            def do_DELETE(self) -> None:  # noqa: N802
                self._dispatch()

        return Handler

    # ----------------------------------------------------------------- api

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        """Stop serving. **Must not** be called from the serving thread:
        ``BaseServer.shutdown`` waits for ``serve_forever`` to return, so
        calling it from there deadlocks. Signal handlers use
        :meth:`server_close` instead."""
        self.httpd.shutdown()
        self.httpd.server_close()

    def server_close(self) -> None:
        """Release the listening socket without joining the serve loop.

        This is the only safe way to tear down from a signal handler, which
        runs on the same thread as ``serve_forever``.
        """
        try:
            self.httpd.server_close()
        except OSError:
            pass

    @property
    def bound_port(self) -> int:
        return self.httpd.server_address[1]
