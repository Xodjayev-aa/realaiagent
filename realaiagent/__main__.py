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
    """Print the owner key. With REALAI_DATABASE_URL set (Turso), it is
    read from - or created in - the shared database, which is how you get
    the key for a Vercel deploy: run this once on your laptop with the
    same env vars."""
    from pathlib import Path
    from .config import Config
    cfg = Config.from_env()
    if args.data is not None:
        cfg.data_dir = Path(args.data).expanduser().resolve()
        cfg.file_roots = [cfg.data_dir]
    if cfg.database_url:
        from .engine import Agent
        agent = Agent(cfg)
        _, key, created = agent.keys.ensure_owner_key()
        print(key)
        if created:
            print("(created now in the shared database)", file=__import__("sys").stderr)
        return 0
    p = cfg.owner_key_path
    if not p.exists():
        print(f"no owner key yet at {p} - start the server once to create it")
        return 1
    print(p.read_text().strip())
    return 0


def _cmd_db_check(args: argparse.Namespace) -> int:
    """Verify the Turso/libSQL connection and create the schema."""
    import time as _t
    from .config import Config
    cfg = Config.from_env()
    if not cfg.database_url:
        print("REALAI_DATABASE_URL (or TURSO_DATABASE_URL) is not set - "
              "local SQLite would be used.")
        return 1
    from .libsql_http import LibsqlError, connect
    t0 = _t.time()
    try:
        conn = connect(cfg.database_url, cfg.database_token)
        conn.execute("SELECT 1")
    except LibsqlError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"connected to {conn.url} in {(_t.time() - t0) * 1000:.0f} ms")
    from .storage import Storage
    st = Storage(cfg.db_path, database_url=cfg.database_url,
                 auth_token=cfg.database_token)
    tables = st.query("SELECT name FROM sqlite_master WHERE type='table' "
                      "ORDER BY name")
    print("schema ok, tables:", ", ".join(t["name"] for t in tables))
    print("usage rows:", st.query_one("SELECT COUNT(*) AS n FROM usage")["n"])
    return 0


def _cmd_react(args: argparse.Namespace) -> int:
    """Run one ReAct (Think -> Act -> Observe) task against a local Ollama.

    Pure stdlib; the model is the only thing outside this process and it
    runs on your machine. Prints each step, then the Arena-style result.
    """
    import json
    from pathlib import Path as _P
    from .config import Config
    from .react import arena_agent_handler

    cfg = Config.from_env()
    if args.data is not None:
        cfg.data_dir = _P(args.data).expanduser().resolve()
        cfg.file_roots = [cfg.data_dir]

    def show(step) -> None:
        print(f"\n--- step {step.iteration} ---")
        if step.thought:
            print(f"THOUGHT: {step.thought}")
        print(f"ACTION : {json.dumps(step.action) if step.action else '(invalid)'}")
        print(f"OBSERVE: {step.observation}")

    payload = {"prompt": args.prompt, "max_steps": args.max_steps,
               "model": args.model, "ollama_url": args.ollama_url}
    result = arena_agent_handler(payload, cfg=cfg, on_step=show)
    result.pop("trace", None)
    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


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

    p_react = sub.add_parser(
        "react",
        help="run one Think->Act->Observe task with a local Ollama model")
    p_react.add_argument("prompt", help="the task to solve")
    p_react.add_argument("--model", default="qwen2.5-coder:7b",
                         help="Ollama model name (default qwen2.5-coder:7b)")
    p_react.add_argument("--ollama-url", default="http://localhost:11434",
                         help="local Ollama base URL")
    p_react.add_argument("--max-steps", type=int, default=6)
    p_react.add_argument("--data", default=None,
                         help="data dir the read_local_file tool is confined to")
    p_react.set_defaults(func=_cmd_react)

    p_key = sub.add_parser("owner-key",
                           help="print the owner key (from the shared "
                                "database when REALAI_DATABASE_URL is set)")
    p_key.add_argument("--data", default=None)
    p_key.set_defaults(func=_cmd_owner_key)

    p_db = sub.add_parser("db-check",
                          help="test the Turso/libSQL connection and create "
                               "the schema")
    p_db.set_defaults(func=_cmd_db_check)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "cmd", None) is None:
        args = parser.parse_args(["serve"] + (argv or []))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
