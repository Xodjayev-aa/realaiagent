"""Configuration for the RealAI agent.

Everything is local. The only secrets ever created are the agent's own
API keys (for its own API) and a local hashing pepper - both live in the
owner's data directory, owned by the owner.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # Where all persistent state lives (sqlite db, pepper, owner key file).
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # HTTP server
    host: str = "0.0.0.0"
    port: int = 8100

    # Autonomous thinking loop cadence (seconds).
    tick_seconds: float = 10.0

    # Safety / limits
    max_body_bytes: int = 1_000_000        # JSON request body cap
    max_output_bytes: int = 16_384         # truncation for command/file output
    command_timeout: float = 10.0          # subprocess timeout (s)
    http_timeout: float = 8.0              # outbound http timeout (s)
    rate_limit_per_min: int = 60           # token bucket refill per key
    rate_burst: int = 20                   # token bucket capacity
    max_file_read: int = 65_536            # files.read cap

    # Path roots that file actions may touch unless the owner widens them.
    file_roots: List[Path] = field(default_factory=list)

    # Business / billing (the "null-49.private" ecosystem)
    billing_enabled: bool = True
    request_cost: float = 0.05          # charged per request for non-VIP users
    default_request_limit: int = 2500   # per customer key (None = unlimited)

    # Telegram master control + autonomous reports (optional).
    # Use YOUR OWN bot (token from @BotFather). Off unless a token is set.
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""          # the owner's chat id
    telegram_poll: bool = True          # run the master-control poller
    autonomous_report_minutes: float = 60.0   # free-will report cadence

    # --- Vercel serverless (free tier) -----------------------------------
    # Vercel sets VERCEL=1 inside its functions. When detected, from_env()
    # switches the data dir to the ephemeral /tmp disk and turns the
    # long-poll telegram worker OFF (webhook mode instead - see
    # api/index.py, which serves POST /webhook).
    is_vercel: bool = False

    # Shared secret Telegram sends in X-Telegram-Bot-Api-Secret-Token when
    # the bot is in webhook mode (set via setWebhook). Empty = open
    # webhook (fine locally, do not leave open in production).
    telegram_webhook_secret: str = ""

    # Optional shared secret guarding /dashboard and /stream/* (passed as
    # ?token=... or the X-Web-Token header). Empty = public telemetry.
    web_token: str = ""

    # Max seconds a single SSE stream window stays open. Bounded because
    # serverless functions freeze; the browser's EventSource reconnects
    # (with Last-Event-ID) and the stream continues gapless.
    sse_window: float = 20.0

    # Identity defaults
    agent_name: str = "REAL"
    owner_name: str = "Owner"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.file_roots:
            self.file_roots = [self.data_dir]
        self.file_roots = [Path(p).expanduser().resolve() for p in self.file_roots]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.db"

    @property
    def pepper_path(self) -> Path:
        return self.data_dir / ".pepper"

    @property
    def owner_key_path(self) -> Path:
        return self.data_dir / "owner_key.txt"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from REALAI_* environment variables (all optional)."""
        roots = os.environ.get("REALAI_FILE_ROOTS", "")
        root_list = [Path(p) for p in roots.split(os.pathsep) if p]

        def _bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        # Vercel sets VERCEL=1 in its serverless functions; the only
        # writable disk there is /tmp, and no long-running threads
        # (so telegram switches from long-poll to webhook mode).
        vercel = os.environ.get("VERCEL", "").strip().lower() in (
            "1", "true", "yes", "on")

        return cls(
            data_dir=Path(os.environ.get(
                "REALAI_DATA_DIR", "/tmp/realai" if vercel else "data")),
            host=os.environ.get("REALAI_HOST", "0.0.0.0"),
            port=int(os.environ.get("REALAI_PORT", "8100")),
            tick_seconds=float(os.environ.get("REALAI_TICK_SECONDS", "10")),
            rate_limit_per_min=int(os.environ.get("REALAI_RATE_PER_MIN", "60")),
            file_roots=root_list,
            billing_enabled=_bool("REALAI_BILLING", True),
            request_cost=float(os.environ.get("REALAI_REQUEST_COST", "0.05")),
            default_request_limit=int(os.environ.get(
                "REALAI_DEFAULT_REQUEST_LIMIT", "2500")),
            telegram_bot_token=os.environ.get("REALAI_TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.environ.get("REALAI_TELEGRAM_CHAT_ID", ""),
            telegram_poll=_bool("REALAI_TELEGRAM_POLL", not vercel),
            autonomous_report_minutes=float(os.environ.get(
                "REALAI_AUTONOMOUS_REPORT_MIN", "60")),
            agent_name=os.environ.get("REALAI_AGENT_NAME", "REAL"),
            owner_name=os.environ.get("REALAI_OWNER_NAME", "Owner"),
            is_vercel=vercel,
            telegram_webhook_secret=os.environ.get(
                "REALAI_TELEGRAM_WEBHOOK_SECRET", ""),
            web_token=os.environ.get("REALAI_WEB_TOKEN", ""),
            sse_window=min(25.0, max(1.0, float(os.environ.get(
                "REALAI_SSE_WINDOW", "20")))),
        )
