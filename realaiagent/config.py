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
        return cls(
            data_dir=Path(os.environ.get("REALAI_DATA_DIR", "data")),
            host=os.environ.get("REALAI_HOST", "0.0.0.0"),
            port=int(os.environ.get("REALAI_PORT", "8100")),
            tick_seconds=float(os.environ.get("REALAI_TICK_SECONDS", "10")),
            rate_limit_per_min=int(os.environ.get("REALAI_RATE_PER_MIN", "60")),
            file_roots=root_list,
            agent_name=os.environ.get("REALAI_AGENT_NAME", "REAL"),
            owner_name=os.environ.get("REALAI_OWNER_NAME", "Owner"),
        )
