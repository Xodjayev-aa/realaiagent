"""The business layer: users, access approval, and billing.

Models the "null-49.private" ecosystem:

- A *client* requests access  -> account created as ``PENDING``.
- The *owner* approves        -> account becomes ``APPROVED`` and a scoped
  API key is minted for that user automatically (the key remembers its
  owner user for billing).
- The owner can deny/ban, grant VIP (free), or top up a user's balance.
- Every request made with a *user key* is checked: status must be
  APPROVED, the per-key request limit must not be exhausted, and - unless
  the user is VIP - the per-request cost is deducted from their balance.
  Ran-out of balance returns ``402``; a banned/unknown key raises a
  security intrusion event.

The owner's own key is exempt from billing and limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import Config
from .storage import Storage
from .api.auth import KeyManager, SCOPE_ALL

PENDING, APPROVED, BANNED = "PENDING", "APPROVED", "BANNED"
_USER_SCOPES = ["chat", "devices", "learn"]


@dataclass
class AuthResult:
    ok: bool
    status: int = 200
    code: str = ""
    message: str = ""
    intrusion: bool = False


class UserManager:
    def __init__(self, storage: Storage, keys: KeyManager,
                 cfg: Config) -> None:
        self.storage = storage
        self.keys = keys
        self.cfg = cfg
        self.notifier = None  # set to a Telegram() by the Agent

    def _notify(self, text: str, dedup_key: str = "",
                min_interval: float = 30.0) -> None:
        if self.notifier is not None:
            self.notifier.send(text, dedup_key=dedup_key,
                               min_interval=min_interval)

    # ------------------------------------------------------------ lifecycle

    def request(self, username: str, scopes: Optional[List[str]] = None,
                note: str = "") -> Dict[str, Any]:
        user = self.storage.user_request(username)
        user["requested_scopes"] = scopes or _USER_SCOPES
        if note:
            user["note"] = note
        self._notify(
            f"🔔 New access request from '{username}'. "
            f"Approve: /approve {username}  ·  Deny: /deny {username}")
        return user

    def approve(self, username: str, scopes: Optional[List[str]] = None,
                request_limit: Optional[int] = None,
                created_by: str = "owner") -> Dict[str, Any]:
        """Approve a user and mint them an API key. Returns the user plus
        (one time) the plaintext key."""
        user = self.storage.user_get(username)
        if user is None:
            raise KeyError(f"user {username} not found")
        scopes = scopes or _USER_SCOPES
        # owner approval implies the user may talk + use devices + learn
        key_id, plaintext, meta = self.keys.create_key(
            name=f"user:{username}", scopes=scopes,
            created_by=created_by, user=username,
            request_limit=request_limit or self.cfg.default_request_limit)
        self.storage.user_set_status(username, APPROVED)
        self.storage.log_event("user_approved", {
            "username": username, "key_id": key_id, "by": created_by},
            key_id=key_id)
        out = self.storage.user_get(username) or {}
        out["api_key"] = plaintext
        out["key_id"] = key_id
        out["key_scopes"] = meta["scopes"]
        return out

    def deny(self, username: str, created_by: str = "owner") -> Dict[str, Any]:
        user = self.storage.user_set_status(username, BANNED)
        if user is None:
            raise KeyError(f"user {username} not found")
        # revoke any of their keys
        for k in self.storage.query(
                "SELECT id FROM keys WHERE user=? AND revoked=0",
                (username,)):
            self.keys.revoke(k["id"])
        self.storage.log_event("user_denied", {"username": username,
                                               "by": created_by})
        return user

    def ban(self, username: str, created_by: str = "owner") -> Dict[str, Any]:
        return self.deny(username, created_by)

    def unban(self, username: str,
              created_by: str = "owner") -> Dict[str, Any]:
        user = self.storage.user_set_status(username, PENDING)
        if user is None:
            raise KeyError(f"user {username} not found")
        return user

    def set_vip(self, username: str, vip: bool) -> Dict[str, Any]:
        user = self.storage.user_set_vip(username, vip)
        if user is None:
            raise KeyError(f"user {username} not found")
        return user

    def topup(self, username: str, amount: float) -> Dict[str, Any]:
        user = self.storage.user_add_balance(username, amount)
        if user is None:
            raise KeyError(f"user {username} not found")
        return user

    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        users = self.storage.user_list(status)
        for u in users:
            u["keys"] = self.storage.query(
                "SELECT id, name, scopes, request_limit, request_count, "
                "revoked, last_used FROM keys WHERE user=?",
                (u["username"],))
        return users

    # -------------------------------------------------------------- billing

    def authorize_request(self, key_row: Dict[str, Any]) -> AuthResult:
        """Gate for every incoming request that carries a user key.

        ``key_row`` is the verified key row (has ``user`` set for customer
        keys; owner keys have no user). Returns an :class:`AuthResult`.
        """
        username = key_row.get("user")
        if not username:
            # owner / service key - no billing, no limit wall
            return AuthResult(True)
        if not self.cfg.billing_enabled:
            return AuthResult(True)

        user = self.storage.user_get(username)
        if user is None:
            return AuthResult(False, 403, "intrusion",
                              f"key for unknown user '{username}'",
                              intrusion=True)
        if user["status"] == BANNED:
            return AuthResult(False, 403, "banned",
                              f"user '{username}' is banned by the owner",
                              intrusion=True)
        if user["status"] != APPROVED:
            return AuthResult(False, 403, "not_approved",
                              f"user '{username}' is {user['status']}; "
                              "awaiting owner approval")

        # per-key request limit
        limit = key_row.get("request_limit")
        if limit is not None and int(key_row.get("request_count", 0)) >= int(limit):
            return AuthResult(False, 429, "limit_exceeded",
                              f"request limit ({limit}) reached for this key")

        # billing: VIP is free, otherwise charge from balance
        if not user["is_vip"]:
            cost = self.cfg.request_cost
            # epsilon guards against float drift (0.15-0.05-0.05)
            if float(user["balance"]) + 1e-9 < cost:
                return AuthResult(False, 402, "insufficient_funds",
                                  f"balance {user['balance']:.2f} < cost "
                                  f"{cost:.2f}; ask the owner to top up or "
                                  "grant VIP")
            self.storage.user_charge(username, cost)
            self.storage.count("billing", "charged")
        else:
            self.storage.count("billing", "vip_free")

        self.storage.user_touch(username)
        return AuthResult(True)

    def usage_for_user(self, username: str) -> Dict[str, Any]:
        user = self.storage.user_get(username) or {}
        keys = self.storage.query(
            "SELECT id, name, request_limit, request_count, revoked, last_used "
            "FROM keys WHERE user=?", (username,))
        return {"user": user, "keys": keys}
