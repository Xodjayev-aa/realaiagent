"""Owner email — stdlib ``smtplib`` only, no third-party mail service.

The product needs one thing from email: when somebody asks for an API key,
the *owner* must hear about it immediately, on a channel they actually
watch. Telegram already does that (:mod:`realaiagent.telegram`); this is
the second channel, so an approval never waits on one app being open.

Design rules, same as everywhere else in this repo:

- **stdlib only.** ``smtplib`` + ``email.message``. No SDK, no HTTP API,
  no external AI, no third-party account.
- **The owner's own credentials.** Typically a Gmail *app password*
  (Google Account → Security → 2-Step Verification → App passwords), which
  is a revocable 16-char secret — never the real account password.
- **Never raises.** Mail is a nicety; a refused SMTP connection must not
  break an access request. Failures are *counted* (``email/send_failed``)
  so the ledger tells the truth.
- **Off unless configured.** No ``REALAI_SMTP_HOST`` → every method is a
  cheap no-op returning ``False``.

Three TLS modes, because free SMTP hosts differ:

=========== =========================================================
``starttls`` port 587 — connect plain, then upgrade (Gmail, most hosts)
``ssl``      port 465 — implicit TLS from the first byte
``none``     port 25/1025 — plaintext, for a local relay (dev only)
=========== =========================================================
"""

from __future__ import annotations

import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from typing import Any, Dict, List, Optional

from .storage import Storage

#: TLS modes understood by :class:`EmailNotifier`.
TLS_MODES = ("starttls", "ssl", "none")


def parse_recipients(raw: Any) -> List[str]:
    """Split ``"a@x.com, b@y.com;c@z.com d@w.com"`` into a clean list."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple, set)):
        items = [str(x) for x in raw]
    else:
        text = str(raw)
        for sep in (",", ";", " ", "\n", "\t"):
            text = text.replace(sep, ",")
        items = text.split(",")
    out: List[str] = []
    for item in items:
        addr = item.strip().strip("<>").lower()
        if addr and "@" in addr and addr not in out:
            out.append(addr)
    return out


class EmailNotifier:
    """Fire-and-forget owner mail, with the same dedup cooldown as Telegram.

    ``send()`` is synchronous (serverless functions cannot keep a worker
    thread alive), bounded by ``timeout``, and always returns a bool.
    """

    def __init__(self, host: str = "", port: int = 587, user: str = "",
                 password: str = "", sender: str = "", recipients: Any = "",
                 mode: str = "starttls", timeout: float = 10.0,
                 storage: Optional[Storage] = None) -> None:
        self.host = (host or "").strip()
        self.port = int(port or 0)
        self.user = (user or "").strip()
        self.password = password or ""
        self.sender = (sender or "").strip() or self.user
        self.recipients = parse_recipients(recipients)
        self.mode = (mode or "starttls").strip().lower()
        if self.mode not in TLS_MODES:
            self.mode = "starttls"
        self.timeout = float(timeout or 10.0)
        self.storage = storage
        self._cooldown: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

    # ----------------------------------------------------------- factories

    @classmethod
    def from_config(cls, cfg: Any,
                    storage: Optional[Storage] = None) -> "EmailNotifier":
        """Build a notifier from a :class:`~realaiagent.config.Config`."""
        return cls(host=getattr(cfg, "smtp_host", ""),
                   port=getattr(cfg, "smtp_port", 587),
                   user=getattr(cfg, "smtp_user", ""),
                   password=getattr(cfg, "smtp_password", ""),
                   sender=getattr(cfg, "smtp_from", ""),
                   recipients=getattr(cfg, "smtp_to", ""),
                   mode=getattr(cfg, "smtp_tls", "starttls"),
                   timeout=getattr(cfg, "smtp_timeout", 10.0),
                   storage=storage)

    # --------------------------------------------------------------- state

    @property
    def enabled(self) -> bool:
        """Configured enough to actually deliver mail."""
        return bool(self.host and self.recipients and self.sender)

    def status(self) -> Dict[str, Any]:
        """Small diagnostic dict (never leaks the password)."""
        return {
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "mode": self.mode,
            "sender": self.sender,
            "recipients": list(self.recipients),
            "configured": bool(self.host),
            "missing": [name for name, ok in (
                ("REALAI_SMTP_HOST", bool(self.host)),
                ("REALAI_SMTP_TO", bool(self.recipients)),
                ("REALAI_SMTP_FROM/USER", bool(self.sender)),
            ) if not ok],
        }

    # ---------------------------------------------------------------- send

    def _count(self, event: str, detail: Optional[Dict[str, Any]] = None
               ) -> None:
        if self.storage is not None:
            try:
                self.storage.count("email", event, detail=detail)
            except Exception:  # noqa: BLE001 - counting must never break mail
                pass

    def _cooled_down(self, dedup_key: str, min_interval: float) -> bool:
        if not dedup_key or min_interval <= 0:
            return False
        now = time.time()
        with self._cooldown_lock:
            last = self._cooldown.get(dedup_key, 0.0)
            if now - last < min_interval:
                return True
            self._cooldown[dedup_key] = now
            return False

    def send(self, subject: str, body: str,
             recipients: Optional[List[str]] = None,
             dedup_key: str = "", min_interval: float = 0.0) -> bool:
        """Deliver one plain-text mail. Returns True only on real delivery."""
        if not self.enabled:
            self._count("skipped_disabled")
            return False
        if self._cooled_down(dedup_key, min_interval):
            self._count("skipped_cooldown", detail={"key": dedup_key})
            return False
        to = parse_recipients(recipients) if recipients else self.recipients
        if not to:
            self._count("skipped_no_recipient")
            return False

        msg = EmailMessage()
        msg["Subject"] = str(subject)[:200]
        msg["From"] = self.sender
        msg["To"] = ", ".join(to)
        msg["Auto-Submitted"] = "auto-generated"
        msg.set_content(str(body)[:8000])

        try:
            self._deliver(msg, to)
        except Exception as exc:  # noqa: BLE001 - mail must never propagate
            self._count("send_failed",
                        detail={"error": type(exc).__name__,
                                "host": self.host})
            return False
        self._count("sent", detail={"to": len(to)})
        return True

    def _deliver(self, msg: EmailMessage, to: List[str]) -> None:
        """Open the connection for ``self.mode`` and hand the message over."""
        context = ssl.create_default_context()
        if self.mode == "ssl":
            client: smtplib.SMTP = smtplib.SMTP_SSL(
                self.host, self.port, timeout=self.timeout, context=context)
        else:
            client = smtplib.SMTP(self.host, self.port,
                                  timeout=self.timeout)
        try:
            if self.mode == "starttls":
                client.ehlo()
                client.starttls(context=context)
                client.ehlo()
            if self.user and self.password:
                client.login(self.user, self.password)
            client.send_message(msg, from_addr=self.sender, to_addrs=to)
        finally:
            try:
                client.quit()
            except Exception:  # noqa: BLE001 - a dead socket is not an error
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------- product mails

    def notify_access_request(self, username: str, note: str = "",
                              approve_url: str = "",
                              agent_name: str = "REAL") -> bool:
        """Somebody asked for an API key: tell the owner, with a one-tap link."""
        lines = [
            f"{agent_name} got a new access request.",
            "",
            f"  username : {username}",
            f"  status   : PENDING (no key until you approve)",
        ]
        if note:
            lines.append(f"  their note: {note[:300]}")
        lines += [
            "",
            "Approve or deny it in the owner inbox:",
            approve_url or "  /approve  (set REALAI_WEB_TOKEN to gate it)",
            "",
            "Or from Telegram:  /approve " + username + "   ·   /deny "
            + username,
            "",
            "-- sent by your own agent, over your own SMTP. No third party "
            "saw this.",
        ]
        return self.send(f"[{agent_name}] access request: {username}",
                         "\n".join(lines),
                         dedup_key=f"request:{username}", min_interval=30.0)

    def notify_owner(self, subject: str, body: str, dedup_key: str = "",
                     min_interval: float = 0.0) -> bool:
        """Generic owner mail (security alerts, approvals, daily ledger)."""
        return self.send(subject, body, dedup_key=dedup_key,
                         min_interval=min_interval)
