#!/usr/bin/env python3
"""Example client for the RealAI agent API (stdlib only).

Usage:
    python examples/client.py --base http://127.0.0.1:8100 --key rxa_...

Or point it at the stored owner key file:
    python examples/client.py --key-file data/owner_key.txt

It walks through the main features: chat, status, the full counters
ledger, creating a scoped key for your UI, teaching the agent, and the
permission approve flow.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


class Client:
    def __init__(self, base: str, key: str) -> None:
        self.base = base.rstrip("/")
        self.key = key

    def req(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method)
        r.add_header("Content-Type", "application/json")
        r.add_header("Authorization", f"Bearer {self.key}")
        try:
            with urllib.request.urlopen(r, timeout=15) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode())
            except ValueError:
                return e.code, {}

    def chat(self, message: str):
        code, body = self.req("POST", "/v1/chat", {"message": message})
        if code != 200:
            print(f"  [error {code}] {body}")
            return None
        print(f"  {body['response']}")
        for a in body["meta"].get("actions", []):
            print(f"    ↳ {a['category']}:{a['action']} "
                  f"ok={a['ok']} awaiting={a['awaiting']} "
                  f"pending={a['pending_id']}")
        return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--key", default=None, help="API key (rxa_...)")
    ap.add_argument("--key-file", default=None,
                    help="path to a file containing the key")
    args = ap.parse_args()

    key = args.key
    if key is None and args.key_file:
        key = Path(args.key_file).read_text().strip()
    if key is None:
        print("provide --key or --key-file", file=sys.stderr)
        return 2
    c = Client(args.base, key)

    print("== health ==")
    print(" ", c.req("GET", "/healthz")[1])

    print("\n== chat ==")
    c.chat("hello!")
    c.chat("status")

    print("\n== counters (everything counted) ==")
    code, body = c.req("GET", "/v1/counters")
    print(" ", json.dumps(body.get("totals", body), indent=2))

    print("\n== create a scoped key for your UI ==")
    code, body = c.req("POST", "/v1/keys",
                       {"name": "my-ui", "scopes": ["chat", "devices"]})
    print(" ", json.dumps(body, indent=2))

    print("\n== teach the agent a new phrase ==")
    code, body = c.req("POST", "/v1/learn",
                       {"examples": [{"text": "power up the lights",
                                      "intent": "command"}]})
    print(" ", body)

    print("\n== pending permissions ==")
    code, body = c.req("GET", "/v1/permissions/pending")
    pending = body.get("pending", [])
    print(f"  {len(pending)} pending")
    if pending:
        pid = pending[0]["id"]
        code, body = c.req("POST", f"/v1/permissions/pending/{pid}/approve")
        print("  approved #", pid, "->", body.get("status"))

    # Business layer: a client requests access, we approve + bill them.
    print("\n== business: onboard a client ==")
    code, body = c.req("POST", "/v1/users/request",
                       {"username": "demo-client"})
    print(" ", body.get("status"), body.get("message", "")[:40])
    code, body = c.req("POST", "/v1/users/demo-client/approve",
                       {"request_limit": 100})
    if code == 200:
        print("  approved; their key:", body["api_key"][:18], "…")
        c.req("POST", "/v1/users/demo-client/topup", {"amount": 0.10})
        client = Client(args.base, body["api_key"])
        print("  client makes 2 paid requests:")
        for i in range(2):
            ccode, _ = client.req("GET", "/v1/status")
            print(f"    request {i+1} -> {ccode}")
    code, body = c.req("GET", "/v1/users?status=APPROVED")
    for u in body.get("users", []):
        print(f"  user {u['username']}: {u['status']} "
              f"vip={u['is_vip']} balance={u['balance']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
