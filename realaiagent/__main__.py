"""CLI entry point: ``python -m realaiagent serve|demo|owner-key``."""

from __future__ import annotations

import argparse
import signal


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the whole product: API + web app + owner inbox + Telegram.

    Configuration comes from ``REALAI_*`` environment variables first —
    that is what the deployment (Vercel, systemd, docker) sets — and the
    command line only overrides what was *explicitly* typed. Before this,
    ``serve`` rebuilt a default ``Config`` from its argparse defaults and
    quietly ignored every ``REALAI_*`` variable, so a deployment that
    configured itself through the environment ran with the wrong data dir,
    the wrong identity and no web token.
    """
    from .config import Config
    from .engine import Agent
    from .api.server import ApiServer

    cfg = Config.from_env()
    # explicit CLI flags win over the environment
    if args.data is not None:
        from pathlib import Path as _P
        cfg.data_dir = _P(args.data).expanduser().resolve()
        cfg.file_roots = [cfg.data_dir]
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = int(args.port)
    if args.tick is not None:
        cfg.tick_seconds = float(args.tick)
    if args.name is not None:
        cfg.agent_name = args.name

    agent = Agent(cfg)
    _, owner_key, created = agent.keys.ensure_owner_key()
    if args.name is not None:
        agent.mind.set_identity(agent_name=args.name)

    shown_host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::", "") else cfg.host
    base = f"http://{shown_host}:{cfg.port}"
    name = agent.mind.agent_name

    print("=" * 66)
    print(f"  {name} — RealAI (fully local core, no external AI)")
    print("=" * 66)
    if created:
        print("  OWNER API KEY (shown once, also saved to")
        print(f"  {cfg.owner_key_path}):")
        print(f"  {owner_key}")
    else:
        print(f"  owner key : {cfg.owner_key_path}")
    print(f"  data dir  : {cfg.data_dir}")
    print("-" * 66)
    print(f"  chat app  : {base}/            (public, keyless demo)")
    print(f"  developers: {base}/developers   (live endpoint table + key request)")
    print(f"  approvals : {base}/approve      (owner inbox"
          + (f", token-gated)" if cfg.web_token else ", NOT token-gated —"
             " set REALAI_WEB_TOKEN)"))
    print(f"  telemetry : {base}/dashboard    (15 live SSE topics"
          + (", token-gated)" if cfg.web_token else ")"))
    print(f"  api index : {base}/api.json     ·  liveness: {base}/healthz")
    print("-" * 66)
    print(f"  autonomy  : on (tick every {cfg.tick_seconds:g}s, kill switch"
          f" at POST /v1/owner/autonomy)")
    print(f"  api call  : curl -H 'Authorization: Bearer <key>' {base}/v1/chat"
          f" -X POST -d '{{\"message\":\"status\"}}'")
    print(f"  billing   : {'on' if cfg.billing_enabled else 'off'} "
          f"(cost {cfg.request_cost:.2f}/req, "
          f"limit {cfg.default_request_limit}/key, VIP free)")
    print(f"  demo chat : {cfg.demo_rate_per_min}/min per visitor, "
          f"burst {cfg.demo_burst} · context {cfg.conversation_turns} turns")
    if agent.telegram.enabled:
        print("  telegram  : master control + autonomous reports ON")
    else:
        print("  telegram  : off (set REALAI_TELEGRAM_BOT_TOKEN + "
              "REALAI_TELEGRAM_CHAT_ID)")
    if agent.mailer.enabled:
        print(f"  email     : owner mail ON via {agent.mailer.host}"
              f":{agent.mailer.port} ({agent.mailer.mode}) → "
              f"{', '.join(agent.mailer.recipients)}")
    else:
        missing = agent.mailer.status()["missing"]
        print("  email     : off (" + (", ".join(missing) if missing
                                        else "no REALAI_SMTP_HOST")
              + ") — access requests go to Telegram only")
    print("=" * 66, flush=True)

    agent.start()
    server = ApiServer(agent, cfg.host, cfg.port)

    from .telegram import TelegramMaster
    master = TelegramMaster(agent.telegram, agent, agent.users)
    master.start()

    def _sig(_signum, _frame):
        # Never call server.shutdown() in here: a signal handler runs on the
        # serving thread, and shutdown() waits for serve_forever() to return
        # — the process would hang until something sent SIGKILL. Unwind with
        # SystemExit instead and let the finally block release the port.
        print("\nshutting down…", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        server.serve_forever()
    finally:
        master.stop()
        agent.stop()
        server.server_close()
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


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (its own function so the defaults are testable).

    Every ``serve`` flag defaults to ``None`` on purpose: ``None`` means
    "not typed", so the value from ``Config.from_env()`` survives. A flag
    with a hard default would silently overwrite the deployment's
    environment.
    """
    parser = argparse.ArgumentParser(
        prog="realai",
        description="RealAI Agent — a fully local, owner-controlled "
                    "cognitive AI agent. Pure Python stdlib, no external AI.")
    sub = parser.add_subparsers(dest="cmd")

    p_serve = sub.add_parser(
        "serve",
        help="run the API + web app (default). Reads REALAI_* environment "
             "variables; the flags below only override what you type.")
    # default=None everywhere: an untyped flag must NOT clobber the
    # environment-derived Config (that was the serve/env bug).
    p_serve.add_argument("--host", default=None,
                         help="bind address (env REALAI_HOST, default 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=None,
                         help="bind port (env REALAI_PORT, default 8100)")
    p_serve.add_argument("--data", default=None,
                         help="data dir (env REALAI_DATA_DIR, default ./data)")
    p_serve.add_argument("--tick", type=float, default=None,
                         help="autonomy thinking interval in seconds "
                              "(env REALAI_TICK_SECONDS)")
    p_serve.add_argument("--name", default=None,
                         help="agent name (env REALAI_AGENT_NAME)")
    p_serve.set_defaults(func=_cmd_serve)

    p_demo = sub.add_parser("demo",
                            help="run an offline scripted conversation")
    p_demo.add_argument("--data", default="data/demo")
    p_demo.set_defaults(func=_cmd_demo)

    p_key = sub.add_parser("owner-key", help="print the stored owner key")
    p_key.add_argument("--data", default="data")
    p_key.set_defaults(func=_cmd_owner_key)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "cmd", None) is None:
        args = parser.parse_args(["serve"] + (argv or []))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
