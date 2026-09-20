"""CLI entry point: ``python -m realaiagent serve|demo|owner-key``."""

from __future__ import annotations

import argparse
import signal
import sys


def _cmd_serve(args: argparse.Namespace) -> int:
    from .config import Config
    from .engine import Agent
    from .api.server import ApiServer

    cfg = Config(
        data_dir=args.data,
        host=args.host,
        port=args.port,
        tick_seconds=args.tick,
        agent_name=args.name,
    )
    agent = Agent(cfg)
    _, owner_key, created = agent.keys.ensure_owner_key()

    print("=" * 62)
    print(f"  {agent.mind.agent_name} — RealAI Agent (fully local core)")
    print("=" * 62)
    if created:
        print(f"  OWNER API KEY (shown once, also saved to")
        print(f"  {cfg.owner_key_path}):")
        print(f"  {owner_key}")
    else:
        print(f"  owner key: {cfg.owner_key_path}")
    print(f"  data dir : {cfg.data_dir}")
    print(f"  api      : http://{args.host}:{args.port}")
    print(f"  autonomy : on (tick every {args.tick}s, owner kill switch at")
    print(f"             POST /v1/owner/autonomy)")
    print("  try     : curl -H 'Authorization: Bearer <key>' "
          "http://127.0.0.1:%d/v1/chat -d '{\"message\":\"status\"}' -X POST"
          % args.port)
    print("=" * 62, flush=True)

    agent.start()
    server = ApiServer(agent, cfg.host, cfg.port)

    def _sig(_signum, _frame):
        print("\nshutting down…", flush=True)
        agent.stop()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        server.serve_forever()
    finally:
        agent.stop()
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo
    return run_demo(data_dir=args.data)


def _cmd_owner_key(args: argparse.Namespace) -> int:
    from pathlib import Path
    p = Path(args.data) / "owner_key.txt"
    if not p.exists():
        print(f"no owner key yet at {p} - start the server once to create it")
        return 1
    print(p.read_text().strip())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="realai",
        description="RealAI Agent — a fully local, owner-controlled "
                    "cognitive AI agent. Pure Python stdlib, no external AI.")
    sub = parser.add_subparsers(dest="cmd")

    p_serve = sub.add_parser("serve", help="run the API server (default)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8100)
    p_serve.add_argument("--data", default="data")
    p_serve.add_argument("--tick", type=float, default=10.0,
                         help="autonomy thinking interval in seconds")
    p_serve.add_argument("--name", default="REAL")
    p_serve.set_defaults(func=_cmd_serve)

    p_demo = sub.add_parser("demo",
                            help="run an offline scripted conversation")
    p_demo.add_argument("--data", default="data/demo")
    p_demo.set_defaults(func=_cmd_demo)

    p_key = sub.add_parser("owner-key", help="print the stored owner key")
    p_key.add_argument("--data", default="data")
    p_key.set_defaults(func=_cmd_owner_key)

    args = parser.parse_args(argv)
    if getattr(args, "cmd", None) is None:
        args = parser.parse_args(["serve"] + (argv or []))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
