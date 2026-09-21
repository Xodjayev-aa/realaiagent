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
    # Archive every generated file (images, decks, audio) to a Telegram chat.
    # Defaults to the owner chat; a forum supergroup + topic ids gives
    # "folders": REALAI_TG_ARCHIVE_TOPICS="image:12,slides:13,audio:14".
    tg_archive_chat_id: str = ""
    tg_archive_topics: str = ""

    # Optional shared secret guarding /dashboard, /stream/* and the owner's
    # /approve inbox (passed as ?token=... or the X-Web-Token header).
    # Empty = public telemetry (fine locally, set it in production).
    web_token: str = ""

    # --- the product web app (branded chat, developers portal) -----------
    # Tagline shown under the wordmark on every page.
    brand_tagline: str = ("A cognitive AI built from scratch — pure Python, "
                          "no external AI, no external keys.")

    # Keyless demo chat: a token bucket PER VISITOR (browser fingerprint /
    # client IP), so one rude guest cannot exhaust the demo for everyone.
    demo_rate_per_min: int = 12
    demo_burst: int = 6

    # How many previous turns the conversational voice keeps in context.
    conversation_turns: int = 12

    # --- owner email (stdlib SMTP only, e.g. a Gmail app password) -------
    # Access requests reach the owner on Telegram AND email, so approvals
    # never wait on one channel. Empty host = email disabled (no-ops).
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""       # Gmail *app* password, never the real one
    smtp_from: str = ""           # defaults to smtp_user
    smtp_to: str = ""             # comma separated owner inboxes
    smtp_tls: str = "starttls"    # starttls | ssl | none
    smtp_timeout: float = 10.0

    # Max seconds a single SSE stream window stays open. Bounded because
    # serverless functions freeze; the browser's EventSource reconnects
    # (with Last-Event-ID) and the stream continues gapless.
    sse_window: float = 20.0

    # --- durable database (Turso / libSQL over HTTPS) ---------------------
    # Empty = local SQLite file in data_dir (self-hosted). Set both to run
    # on Vercel/serverless with state that survives cold starts. Talks the
    # SQL-over-HTTP protocol with urllib - no driver package needed.
    database_url: str = ""          # libsql://<db>-<org>.turso.io  (or https://)
    database_token: str = ""        # turso db tokens create <db>

    # Serverless autonomy: with no background thread, the agent "thinks"
    # at most once per this many seconds, piggybacking on a request or a
    # Vercel cron hitting GET /cron/tick.
    serverless_tick_seconds: float = 60.0

    # --- generative provider ---------------------------------------------
    # "local"  : only the local backends below (Ollama, SD, whisper, Piper)
    # "hosted" : a keyless public inference API (Pollinations.AI, no signup,
    #            no key) for talk / images / voice, so the agent has ChatGPT-
    #            style abilities on Vercel with no hardware. Local backends
    #            still win when their URL is set. Anonymous tier is
    #            rate-limited (~1 request / 15 s per IP); an optional token
    #            from auth.pollinations.ai raises it - never required.
    provider: str = "local"
    hosted_text_url: str = "https://text.pollinations.ai"
    hosted_image_url: str = "https://image.pollinations.ai"
    hosted_text_model: str = "openai"
    hosted_image_model: str = "flux"
    hosted_voice: str = "nova"
    hosted_token: str = ""
    hosted_timeout: float = 60.0

    # --- optional LOCAL generative backends (all off unless a URL is set) --
    # These are servers running on YOUR machine: Ollama for fluent talk,
    # Stable Diffusion WebUI (--api) for images, whisper.cpp server for
    # speech-to-text, Piper for speech. No keys, nothing leaves the host.
    llm_url: str = ""               # e.g. http://127.0.0.1:11434 (Ollama)
    llm_model: str = "llama3.1:8b"
    llm_timeout: float = 90.0
    image_url: str = ""             # e.g. http://127.0.0.1:7860 (SD WebUI)
    image_timeout: float = 180.0
    stt_url: str = ""               # e.g. http://127.0.0.1:8178 (whisper.cpp)
    tts_url: str = ""               # e.g. http://127.0.0.1:5000 (piper http)
    tts_command: str = ""           # e.g. "piper --model en_US-lessac-medium.onnx"
    media_ttl_hours: float = 24.0   # generated files older than this are pruned

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
    def media_dir(self) -> Path:
        """Where generated images / audio / slide decks are written."""
        return self.data_dir / "media"

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
            provider=os.environ.get("REALAI_PROVIDER", "local").strip().lower(),
            hosted_text_model=os.environ.get("REALAI_HOSTED_TEXT_MODEL", "openai"),
            hosted_image_model=os.environ.get("REALAI_HOSTED_IMAGE_MODEL", "flux"),
            hosted_voice=os.environ.get("REALAI_HOSTED_VOICE", "nova"),
            hosted_token=os.environ.get("REALAI_HOSTED_TOKEN", "").strip(),
            hosted_timeout=float(os.environ.get("REALAI_HOSTED_TIMEOUT", "60")),
            llm_url=os.environ.get("REALAI_LLM_URL", "").strip(),
            llm_model=os.environ.get("REALAI_LLM_MODEL", "llama3.1:8b"),
            llm_timeout=float(os.environ.get("REALAI_LLM_TIMEOUT", "90")),
            image_url=os.environ.get("REALAI_IMAGE_URL", "").strip(),
            stt_url=os.environ.get("REALAI_STT_URL", "").strip(),
            tts_url=os.environ.get("REALAI_TTS_URL", "").strip(),
            tts_command=os.environ.get("REALAI_TTS_COMMAND", "").strip(),
            media_ttl_hours=float(os.environ.get("REALAI_MEDIA_TTL_HOURS", "24")),
            owner_name=os.environ.get("REALAI_OWNER_NAME", "Owner"),
            is_vercel=vercel,
            database_url=(os.environ.get("REALAI_DATABASE_URL")
                          or os.environ.get("TURSO_DATABASE_URL", "")).strip(),
            database_token=(os.environ.get("REALAI_DATABASE_TOKEN")
                            or os.environ.get("TURSO_AUTH_TOKEN", "")).strip(),
            serverless_tick_seconds=float(os.environ.get(
                "REALAI_SERVERLESS_TICK_SECONDS", "60")),
            tg_archive_chat_id=os.environ.get("REALAI_TG_ARCHIVE_CHAT_ID", ""),
            tg_archive_topics=os.environ.get("REALAI_TG_ARCHIVE_TOPICS", ""),
            telegram_webhook_secret=os.environ.get(
                "REALAI_TELEGRAM_WEBHOOK_SECRET", ""),
            web_token=os.environ.get("REALAI_WEB_TOKEN", ""),
            sse_window=min(25.0, max(1.0, float(os.environ.get(
                "REALAI_SSE_WINDOW", "20")))),
            brand_tagline=os.environ.get("REALAI_BRAND_TAGLINE")
            or cls.brand_tagline,
            demo_rate_per_min=int(os.environ.get(
                "REALAI_DEMO_RATE_PER_MIN", "12")),
            demo_burst=int(os.environ.get("REALAI_DEMO_BURST", "6")),
            conversation_turns=int(os.environ.get(
                "REALAI_CONVERSATION_TURNS", "12")),
            smtp_host=os.environ.get("REALAI_SMTP_HOST", "").strip(),
            smtp_port=int(os.environ.get("REALAI_SMTP_PORT", "587")),
            smtp_user=os.environ.get("REALAI_SMTP_USER", "").strip(),
            smtp_password=os.environ.get("REALAI_SMTP_PASSWORD", ""),
            smtp_from=os.environ.get("REALAI_SMTP_FROM", "").strip(),
            smtp_to=os.environ.get("REALAI_SMTP_TO", "").strip(),
            smtp_tls=(os.environ.get("REALAI_SMTP_TLS", "starttls")
                      .strip().lower() or "starttls"),
            smtp_timeout=float(os.environ.get("REALAI_SMTP_TIMEOUT", "10")),
        )
