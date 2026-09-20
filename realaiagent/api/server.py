"""The HTTP API server (stdlib http.server, threaded).

Middleware order: CORS -> route match -> auth (Bearer key) -> scope check
-> rate limit -> body parse -> handler -> JSON response. Every request is
counted in the ledger (including 4xx/5xx, with latency).
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


class ApiServer:
    def __init__(self, agent: Agent, host: str, port: int) -> None:
        self.agent = agent
        self.host = host
        self.port = port
        self.limiter = _RateLimiter(agent.cfg.rate_limit_per_min,
                                    agent.cfg.rate_burst)
        handler = self._make_handler()
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True

    def _make_handler(self):
        agent = self.agent
        limiter = self.limiter
        max_body = agent.cfg.max_body_bytes
        storage = agent.storage

        class Handler(BaseHTTPRequestHandler):
            server_version = "RealAI/0.1"

            # ---------------------------------------------------- plumbing

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
                pass  # ledger counts instead of stderr spam

            def _send(self, status: int, payload: Any,
                      extra_headers: Optional[Dict[str, str]] = None,
                      request_id: str = "") -> None:
                if isinstance(payload, str):
                    body = payload.encode("utf-8")
                    content_type = "text/html; charset=utf-8"
                else:
                    body = json.dumps(payload, default=str).encode("utf-8")
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Authorization, Content-Type")
                self.send_header("X-Request-Id", request_id)
                for k, v in (extra_headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
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
                request_id = uuid.uuid4().hex[:12]
                t0 = time.time()
                url = urlparse(self.path)
                path = url.path.rstrip("/") or "/"
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                method = self.command

                matched = None
                for m, pattern, scopes, handler in _ROUTES:
                    if m != method:
                        continue
                    mm = pattern.match(path)
                    if mm:
                        matched = (m, pattern, scopes, handler,
                                   mm.groupdict())
                        break
                if matched is None:
                    self._count_request(None, method + " " + path, 404, t0)
                    return self._send(404, {"error": {
                        "code": "not_found", "message": f"no route {method} {path}"}},
                        request_id=request_id)

                _m, _p, scopes, handler, params = matched

                # ---- auth
                key_row: Optional[Dict[str, Any]] = None
                if scopes:
                    auth = self.headers.get("Authorization", "")
                    token = auth[7:] if auth.startswith("Bearer ") else auth
                    key_row = agent.keys.verify(token) if token else None
                    if key_row is None:
                        if token:
                            # someone presented a key that does not exist
                            agent.storage.count("security", "intrusion")
                            agent.storage.log_event("intrusion", {
                                "reason": "invalid_key",
                                "path": f"{method} {path}"})
                            agent.telegram.send(
                                "🚨 [Security] an invalid API key was "
                                f"presented for {method} {path}",
                                dedup_key="intrusion-invalid",
                                min_interval=60)
                        self._count_request(None, method + " " + path,
                                            401, t0)
                        return self._send(401, {"error": {
                            "code": "unauthorized",
                            "message": "missing or invalid API key"}},
                            request_id=request_id)
                    if not self._scope_ok(key_row, scopes):
                        self._count_request(key_row["id"], method + " " + path,
                                            403, t0)
                        return self._send(403, {"error": {
                            "code": "forbidden",
                            "message": f"key lacks scope for {path} "
                                       f"(needs one of {scopes})"}},
                            request_id=request_id)
                    if not limiter.allow(key_row["id"]):
                        self._count_request(key_row["id"],
                                            method + " " + path, 429, t0)
                        return self._send(429, {"error": {
                            "code": "rate_limited",
                            "message": "too many requests"}},
                            request_id=request_id)

                    # ---- user/billing gate (customer keys)
                    bill = agent.users.authorize_request(key_row)
                    if not bill.ok:
                        if bill.intrusion:
                            agent.storage.count("security", "intrusion")
                            agent.storage.log_event("intrusion", {
                                "reason": bill.code,
                                "user": key_row.get("user"),
                                "path": f"{method} {path}"})
                            agent.telegram.send(
                                "🚨 [Security] " + bill.message,
                                dedup_key=f"intrusion-{bill.code}",
                                min_interval=60)
                        self._count_request(key_row["id"], method + " " + path,
                                            bill.status, t0)
                        return self._send(bill.status, {"error": {
                            "code": bill.code, "message": bill.message}},
                            request_id=request_id)

                    agent.storage.key_bump_usage(key_row["id"])

                # ---- body
                body: Dict[str, Any] = {}
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    if length > max_body:
                        self._count_request(
                            (key_row or {}).get("id"),
                            method + " " + path, 413, t0)
                        return self._send(413, {"error": {
                            "code": "body_too_large",
                            "message": f"body > {max_body} bytes"}},
                            request_id=request_id)
                    raw = self.rfile.read(length)
                    try:
                        body = json.loads(raw.decode("utf-8"))
                        if not isinstance(body, dict):
                            raise ValueError("JSON object required")
                    except (ValueError, UnicodeDecodeError):
                        self._count_request(
                            (key_row or {}).get("id"),
                            method + " " + path, 400, t0)
                        return self._send(400, {"error": {
                            "code": "bad_json", "message": "invalid JSON body"}},
                            request_id=request_id)

                ctx = {"key": key_row, "query": query,
                       "request_id": request_id, "params": params}
                try:
                    status, payload = handler(agent, body, ctx)
                    code = 500 if status >= 500 else status
                    self._send(status, payload, request_id=request_id)
                    self._count_request((key_row or {}).get("id"),
                                        method + " " + path, code, t0)
                except Exception as exc:  # noqa: BLE001
                    storage.count("errors", "http_500")
                    self._count_request((key_row or {}).get("id"),
                                        method + " " + path, 500, t0)
                    self._send(500, {"error": {
                        "code": "internal", "message": str(exc)}},
                        request_id=request_id)

            def _scope_ok(self, key_row: Dict[str, Any],
                          scopes: List[str]) -> bool:
                key_scopes = set(key_row.get("scopes", []))
                if "owner" in key_scopes:
                    return True
                return bool(key_scopes & set(scopes))

            def _count_request(self, key_id: Optional[str], route: str,
                               code: int, t0: float) -> None:
                storage.count(
                    "requests",
                    f"{code}_{route.split()[1] if len(route.split()) > 1 else route}",
                    key_id=key_id,
                    detail={"route": route, "latency_ms":
                            round((time.time() - t0) * 1000, 2)})

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
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def bound_port(self) -> int:
        return self.httpd.server_address[1]
