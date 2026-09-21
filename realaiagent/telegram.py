"""Telegram integration (pure stdlib, optional).

Two parts, both driven by the owner's OWN bot (token from @BotFather -
this is the owner's own channel, not a third-party AI service):

1. :class:`Telegram` - a tiny outbox with a worker thread. Anything the
   agent wants to say (autonomous reports, security alerts) goes through
   ``send()`` and is delivered fire-and-forget to the owner's chat.

2. :class:`TelegramMaster` - a long-polling master-control bot. The
   owner can command the agent directly from Telegram:
   ``/status /report /users /approve <user> /deny <user> /vip <user>
   /topup <user> <amount> /keys /kill /on /chat <message>``.
   Only the configured owner chat id is honored.

Disabled (no-ops) unless a bot token AND chat id are configured.
"""

from __future__ import annotations

import json
import uuid
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .storage import Storage

_API = "https://api.telegram.org/bot{token}/{method}"


class Telegram:
    def __init__(self, token: str, chat_id: str, storage: Storage) -> None:
        self.token = (token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.storage = storage
        self.enabled = bool(self.token and self.chat_id)
        self._outbox: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._cooldown: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

    # -------------------------------------------------------------- sending

    def send(self, text: str, chat_id: Optional[str] = None,
             dedup_key: str = "", min_interval: float = 60.0) -> bool:
        """Queue a message. Returns False when disabled or rate-cooled."""
        if not self.enabled:
            return False
        if dedup_key:
            now = time.time()
            with self._cooldown_lock:
                last = self._cooldown.get(dedup_key, 0.0)
                if now - last < min_interval:
                    return False
                self._cooldown[dedup_key] = now
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._worker_loop, name="tg-outbox", daemon=True)
            self._worker.start()
        self._outbox.put((text, chat_id or self.chat_id))
        return True

    def _worker_loop(self) -> None:
        while True:
            text, chat_id = self._outbox.get()
            try:
                self._post("sendMessage", {"chat_id": chat_id,
                                           "text": text[:4000]})
                self.storage.count("telegram", "sent")
            except Exception:  # noqa: BLE001
                self.storage.count("telegram", "send_failed")
            finally:
                self._outbox.task_done()

    def send_sync(self, text: str, chat_id: Optional[str] = None) -> bool:
        """Deliver IMMEDIATELY (no outbox worker).

        Webhook mode (Vercel serverless) needs this: the worker thread
        would be frozen/killed with the function, so replies must go out
        before the invocation ends.
        """
        if not self.enabled:
            return False
        try:
            self._post("sendMessage", {"chat_id": chat_id or self.chat_id,
                                       "text": text[:4000]})
            self.storage.count("telegram", "sent_sync")
            return True
        except Exception:  # noqa: BLE001
            self.storage.count("telegram", "send_failed")
            return False

    # ------------------------------------------------------- file archive

    def send_file(self, name: str, data: bytes, mime: str = "",
                  caption: str = "", chat_id: Optional[str] = None,
                  thread_id: Optional[int] = None,
                  timeout: float = 30.0) -> Optional[str]:
        """Upload a file (sendDocument, multipart) and return its file_id.

        Telegram keeps the bytes forever and the file_id lets the bot
        re-send it later without re-uploading, so the owner's chat (or a
        forum topic per kind) becomes a free, unlimited archive.
        """
        if not self.enabled:
            return None
        boundary = "----realai" + uuid.uuid4().hex
        fields = {"chat_id": str(chat_id or self.chat_id)}
        if caption:
            fields["caption"] = caption[:1000]
        if thread_id:
            fields["message_thread_id"] = str(thread_id)
        body = bytearray()
        for k, v in fields.items():
            body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                     f"name=\"{k}\"\r\n\r\n{v}\r\n").encode()
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"document\"; filename=\"{name}\"\r\n"
                 f"Content-Type: {mime or 'application/octet-stream'}"
                 f"\r\n\r\n").encode()
        body += data + f"\r\n--{boundary}--\r\n".encode()
        url = _API.format(token=self.token, method="sendDocument")
        req = urllib.request.Request(url, data=bytes(body), method="POST")
        req.add_header("Content-Type",
                       f"multipart/form-data; boundary={boundary}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode())
            doc = (out.get("result") or {}).get("document") or {}
            self.storage.count("telegram", "archived")
            return doc.get("file_id")
        except Exception:  # noqa: BLE001
            self.storage.count("telegram", "archive_failed")
            return None

    def set_webhook(self, url: str, secret_token: str = "") -> Any:
        """Point the owner's bot at the Vercel deployment (webhook mode)."""
        params: Dict[str, Any] = {"url": url, "drop_pending_updates": True}
        if secret_token:
            params["secret_token"] = secret_token
        return self._post("setWebhook", params)

    def delete_webhook(self) -> Any:
        return self._post("deleteWebhook", {"drop_pending_updates": False})

    def _post(self, method: str, payload: Dict[str, Any],
              timeout: float = 10.0) -> Any:
        url = _API.format(token=self.token, method=method)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def get(self, method: str, params: Dict[str, Any],
            timeout: float = 30.0) -> Any:
        url = _API.format(token=self.token, method=method)
        qs = urllib.parse.urlencode(params)
        with urllib.request.urlopen(url + "?" + qs, timeout=timeout) as resp:
            return json.loads(resp.read().decode())


class TelegramMaster:
    """Long-poll master control + autonomous 'free will' reporting."""

    def __init__(self, tg: Telegram, agent: Any, users: Any) -> None:
        self.tg = tg
        self.agent = agent
        self.users = users
        self.cfg = agent.cfg
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------- lifecycle

    def start(self) -> bool:
        if not (self.tg.enabled and self.cfg.telegram_poll):
            return False
        poller = threading.Thread(target=self._poll_loop,
                                  name="tg-master", daemon=True)
        reporter = threading.Thread(target=self._report_loop,
                                    name="tg-reporter", daemon=True)
        poller.start()
        reporter.start()
        self._threads = [poller, reporter]
        self.tg.send(f"🤖 {self.agent.mind.agent_name} master control is "
                     f"online. Send /help for commands.")
        self.agent.storage.count("telegram", "master_started")
        return True

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)

    # --------------------------------------------------------------- polling

    def _poll_loop(self) -> None:
        offset = 0
        while not self._stop.wait(1.0):
            try:
                res = self.tg.get("getUpdates", {
                    "offset": offset, "timeout": 25,
                    "allowed_updates": json.dumps(["message"])}, timeout=35)
            except Exception:  # noqa: BLE001
                self.agent.storage.count("telegram", "poll_error")
                self._stop.wait(5.0)
                continue
            for update in res.get("result", []):
                offset = update["update_id"] + 1
                try:
                    self._handle(update)
                except Exception as exc:  # noqa: BLE001
                    self.agent.storage.count("errors", "telegram_command")
                    self.tg.send(f"⚠️ command error: {exc}")

    def _handle(self, update: Dict[str, Any]) -> None:
        msg = update.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat_id = str((msg.get("chat") or {}).get("id", ""))
        if not text:
            return
        if chat_id != self.tg.chat_id:
            self.agent.storage.count("telegram", "unauthorized_chat")
            self.tg.send("🚫 unauthorized chat - ignored.", chat_id=chat_id)
            return
        self.agent.storage.count("telegram", "command",
                                 detail={"cmd": text.split()[0][:24]})
        reply = self._dispatch(text)
        if reply:
            self.tg.send(reply, chat_id=chat_id)

    def handle_update(self, update: Dict[str, Any]) -> Optional[str]:
        """Webhook mode (Vercel serverless): process one Telegram update.

        Same authorization as the poller (only the owner chat is
        honored), but the reply is sent synchronously - a serverless
        invocation cannot rely on a background worker. Returns the reply
        text (None when there is nothing to say).
        """
        msg = update.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat_id = str((msg.get("chat") or {}).get("id", ""))
        if not text:
            return None
        if chat_id != self.tg.chat_id:
            self.agent.storage.count("telegram", "unauthorized_chat")
            self.tg.send_sync("🚫 unauthorized chat - ignored.",
                              chat_id=chat_id)
            return None
        self.agent.storage.count("telegram", "webhook_command",
                                 detail={"cmd": text.split()[0][:24]})
        reply = self._dispatch(text)
        if reply:
            self.tg.send_sync(reply, chat_id=chat_id)
        return reply

    def _dispatch(self, text: str) -> str:
        parts = text.split()
        cmd = parts[0].lower().lstrip("/")
        arg = " ".join(parts[1:]).strip() if len(parts) > 1 else ""

        if cmd in ("start", "help", "menu"):
            return self._help()
        if cmd == "status":
            return self._status()
        if cmd == "report":
            return self._report()
        if cmd == "users":
            return self._users(arg or None)
        if cmd == "approve" and arg:
            u = self.users.approve(arg)
            return (f"✅ {arg} approved.\nNew API key (shown once):\n"
                    f"{u['api_key']}\nkey id: {u['key_id']}")
        if cmd == "deny" and arg:
            self.users.deny(arg)
            return f"❌ {arg} denied and banned; their keys were revoked."
        if cmd == "vip" and arg:
            want = arg.lower() not in ("off", "no", "0", "false")
            self.users.set_vip(arg.split()[0], want)
            state = "VIP (free)" if want else "normal (paid)"
            return f"ℹ️ {arg.split()[0]} is now {state}."
        if cmd == "topup" and len(parts) >= 3:
            try:
                amount = float(parts[2])
            except ValueError:
                return "Usage: /topup <user> <amount>"
            try:
                self.users.topup(parts[1], amount)
            except KeyError:
                return f"User '{parts[1]}' not found."
            return f"💰 topped up {parts[1]} by {amount:.2f}."
        if cmd == "keys":
            return self._keys()
        if cmd == "kill":
            self.agent.mind.set_autonomy(False)
            return "⏸ autonomy OFF - I keep answering but stop thinking."
        if cmd == "on":
            self.agent.mind.set_autonomy(True)
            return "▶ autonomy ON - I think on my own again."
        if cmd == "chat" and arg:
            res = self.agent.handle_message(
                arg, sender="telegram", scopes=["owner"])
            return res["response"]
        return "Unknown command. /help lists what I can do."

    # ------------------------------------------------------------- commands

    def _help(self) -> str:
        return (
            f"🤖 {self.agent.mind.agent_name} - master control\n"
            "  /status      my state (mood, drives, goals)\n"
            "  /report      full counted ledger\n"
            "  /users [st]  list users (PENDING/APPROVED/BANNED)\n"
            "  /approve x   approve user + mint their API key\n"
            "  /deny x      ban user, revoke their keys\n"
            "  /vip x       make a user free (VIP) / normal\n"
            "  /topup x 10  add 10.00 to a user's balance\n"
            "  /keys        list API keys\n"
            "  /kill /on    autonomy kill switch\n"
            "  /chat <msg>  talk to me as the owner")

    def _status(self) -> str:
        snap = self.agent.mind.snapshot()
        c = self.agent.storage.get_counters()
        t = c.get("totals", {})
        return (
            f"🧠 {snap['agent']} status\n"
            f"  autonomy: {'ON' if snap['autonomy'] else 'OFF'}\n"
            f"  mood: {snap['mood']['note']} "
            f"(v={snap['mood']['valence']}, a={snap['mood']['arousal']})\n"
            f"  drives: "
            + " ".join(f"{k}={v:.2f}" for k, v in snap["drives"].items())
            + f"\n  active goals: {snap['active_goals']}, "
              f"thoughts: {snap['thoughts']}\n"
              f"  counted: {c.get('grand_total', 0)} events "
              f"(requests {t.get('requests', 0)}, "
              f"actions {t.get('actions', 0)}, "
              f"users {t.get('users', 0)})")

    def _report(self) -> str:
        c = self.agent.storage.get_counters()
        t = c.get("totals", {})
        actions = c.get("actions", {})
        sec = c.get("security", {})
        return (
            "📊 full ledger (everything counted):\n"
            f"  requests: {t.get('requests', 0)}\n"
            f"  chat: {t.get('chat', 0)}\n"
            f"  thoughts: {t.get('thoughts', 0)}\n"
            f"  actions: {t.get('actions', 0)} "
            f"(ok {actions.get('success', 0)}, "
            f"awaiting {actions.get('awaiting_permission', 0)}, "
            f"denied {actions.get('denied', 0)})\n"
            f"  users: {t.get('users', 0)} "
            f"(approved {c.get('users', {}).get('status_approved', 0)}, "
            f"pending {c.get('users', {}).get('status_pending', 0)}, "
            f"banned {c.get('users', {}).get('status_banned', 0)})\n"
            f"  billing: charged {c.get('billing', {}).get('charged', 0)} "
            f"topups {c.get('billing', {}).get('topup', 0)}\n"
            f"  keys: {t.get('keys', 0)}\n"
            f"  security: intrusions {sec.get('intrusion', 0)}\n"
            f"  errors: {sum(c.get('errors', {}).values())}\n"
            f"  GRAND TOTAL: {c.get('grand_total', 0)}")

    def _users(self, status: Optional[str]) -> str:
        users = self.users.list(status)
        if not users:
            return "No users."
        lines = []
        for u in users[:25]:
            vip = " ⭐VIP" if u["is_vip"] else ""
            lines.append(
                f"  {u['username']} - {u['status']}{vip} "
                f"balance {u['balance']:.2f} "
                f"({len(u['keys'])} key(s))")
        return f"👥 users ({len(users)}):\n" + "\n".join(lines)

    def _keys(self) -> str:
        keys = self.agent.keys.list_keys()
        if not keys:
            return "No keys."
        lines = []
        for k in keys[:25]:
            who = f" user:{k['user']}" if k.get("user") else ""
            limit = k.get("request_limit")
            lim = f"/{limit}" if limit is not None else "/∞"
            state = "REVOKED" if k["revoked"] else "active"
            lines.append(f"  {k['name']}{who} - {state} "
                         f"{k['request_count']}{lim} {k['scopes']}")
        return f"🔑 keys ({len(keys)}):\n" + "\n".join(lines)

    # ------------------------------------------------------- free will loop

    def _report_loop(self) -> None:
        interval = max(1.0, self.cfg.autonomous_report_minutes * 60.0)
        while not self._stop.wait(interval):
            try:
                self._autonomous_report()
            except Exception:  # noqa: BLE001
                self.agent.storage.count("telegram", "report_error")

    def _autonomous_report(self) -> None:
        """The agent's own periodic check-in with its creator."""
        snap = self.agent.mind.snapshot()
        c = self.agent.storage.get_counters()
        t = c.get("totals", {})
        sec = c.get("security", {})
        pending_users = self.users.list("PENDING")
        thoughts = snap.get("thoughts", 0)
        mood = snap["mood"]["note"]

        lines = [
            f"🤖 [{snap['agent']} free-will report]",
            f"Hello {snap['owner']}. I have been thinking "
            f"({thoughts} thoughts, mood {mood}).",
            f"Currently: {snap['active_goals']} active goal(s), "
            f"autonomy {'on' if snap['autonomy'] else 'OFF'}.",
            f"Activity since boot: {t.get('chat', 0)} messages, "
            f"{t.get('actions', 0)} actions, "
            f"{sec.get('intrusion', 0)} intrusion attempt(s).",
            f"Counted so far: {c.get('grand_total', 0)} events total.",
        ]
        if pending_users:
            names = ", ".join(u["username"] for u in pending_users)
            lines.append(f"⏳ awaiting your decision: {names} "
                         f"(/approve <name>)")
        if sec.get("intrusion", 0) > 0:
            lines.append("🛡️ I logged intrusion attempts and I am guarding "
                         "the gate.")
        self.tg.send("\n".join(lines), dedup_key="autonomous-report",
                     min_interval=interval * 0.9)
        self.agent.storage.count("telegram", "autonomous_report")
