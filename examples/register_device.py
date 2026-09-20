#!/usr/bin/env python3
"""Register a device with the RealAI agent (stdlib only).

A device is a JSON description of what it can do. Each capability either
runs a local *command template* or calls an HTTP *endpoint* (with
{placeholder} params rendered from the control request).

Examples:

  # A lamp that a smart-plug API controls (Home Assistant-style)
  python examples/register_device.py --key rxa_... --id lamp-1 \
      --name "Living Room Lamp" --kind light \
      --cap on=http://192.168.1.50:8123/api/lights/lamp-1/turn_on \
      --cap off=http://192.168.1.50:8123/api/lights/lamp-1/turn_off

  # A fan controlled by a local script (command executor)
  python examples/register_device.py --key rxa_... --id fan-1 \
      --name "Ceiling Fan" --kind fan \
      --cap on="fanctl.sh on" --cap off="fanctl.sh off" --cap "set=echo {level}"

Then talk to the agent:
  POST /v1/chat {"message": "turn on the Living Room Lamp"}
  (first time it asks permission; approve once and it's trusted.)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--key", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", default="generic")
    ap.add_argument("--cap", action="append", default=[],
                    help="action=command-or-url  (repeatable)")
    args = ap.parse_args()

    caps = []
    for item in args.cap:
        if "=" not in item:
            print(f"bad --cap (want action=value): {item}", file=sys.stderr)
            return 2
        action, value = item.split("=", 1)
        if value.startswith("http://") or value.startswith("https://"):
            caps.append({"action": action.strip(), "executor": "http",
                         "endpoint": value.strip(), "method": "POST"})
        else:
            caps.append({"action": action.strip(), "executor": "command",
                         "template": value.strip()})

    body = {"id": args.id, "name": args.name, "kind": args.kind,
            "capabilities": caps}
    data = json.dumps(body).encode()
    r = urllib.request.Request(
        args.base.rstrip("/") + "/v1/devices", data=data, method="POST")
    r.add_header("Content-Type", "application/json")
    r.add_header("Authorization", f"Bearer {args.key}")
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            print(resp.status, json.loads(resp.read().decode()))
    except urllib.error.HTTPError as e:
        print(e.code, e.read().decode(), file=sys.stderr)
        return 1
    print(f"device {args.id} registered. Ask the agent: "
          f'"turn on the {args.name}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
