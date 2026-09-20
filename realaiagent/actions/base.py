"""Action data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ActionRequest:
    """One concrete thing the agent wants to do.

    category: stable action category, e.g. ``device.control``,
              ``system.command``, ``files.read``, ``files.write``,
              ``http.request``, ``notify``.
    action:   stable key for the exact action - used for permission
              patterns and Q-learning. e.g. ``lamp:on`` or the command.
    """

    category: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    device_id: Optional[str] = None
    source: str = "chat"          # chat | autonomy | direct
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "action": self.action,
            "params": self.params,
            "device_id": self.device_id,
            "source": self.source,
        }


@dataclass
class ActionOutcome:
    ok: bool = False
    awaiting: bool = False        # needs owner permission
    denied: bool = False
    pending_id: Optional[int] = None
    output: str = ""
    error: str = ""
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "awaiting": self.awaiting,
            "denied": self.denied,
            "pending_id": self.pending_id,
            "output": self.output,
            "error": self.error,
            "ms": round(self.ms, 2),
        }
