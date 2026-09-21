"""API key management - the agent issues its own keys for its own API.

Keys are ``rxa_...`` tokens, stored only as SHA-256 hashes of
``pepper + key`` (the pepper is a random local secret generated on first
boot and kept in the owner's data dir). The owner key is printed once at
first boot and also written to ``data/owner_key.txt`` (chmod 600).

Scopes:
- ``owner``  -> everything (including key management, kill switch, values)
- ``admin``  -> manage devices/skills/memory/goals/policies (except keys/identity)
- ``learn``  -> POST /v1/learn
- ``devices``-> device listing + direct control
- ``chat``   -> talk to the agent, read status/counters/events

Any owner can create scoped keys for their own clients (e.g. their UI).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from typing import List, Optional, Tuple

from ..storage import Storage

SCOPE_ALL = ["owner", "admin", "learn", "devices", "chat"]
VALID_SCOPES = ["owner", "admin", "learn", "devices", "chat"]


class KeyManager:
    def __init__(self, storage: Storage, pepper_path: Optional[object] = None,
                 cfg=None) -> None:
        self.storage = storage
        self.cfg = cfg
        self._pepper = self._load_pepper(pepper_path)

    # -------------------------------------------------------------- pepper

    def _load_pepper(self, path) -> bytes:
        """The pepper lives in the database (durable on serverless) and is
        mirrored to the data dir when that is writable. Order: DB, file,
        fresh - and whichever is found first is written to the other."""
        if path is None and self.cfg is not None:
            path = self.cfg.pepper_path
        stored = None
        try:
            stored = self.storage.secret_get("pepper")
        except Exception:  # noqa: BLE001
            stored = None
        if stored:
            pepper = bytes.fromhex(stored)
            self._mirror_file(path, pepper)
            return pepper
        pepper = None
        if path is not None:
            try:
                with open(str(path), "rb") as fh:
                    pepper = fh.read() or None
            except OSError:
                pepper = None
        if pepper is None:
            pepper = secrets.token_bytes(32)
        try:
            durable = self.storage.secret_set_if_absent("pepper", pepper.hex())
            pepper = bytes.fromhex(durable)
        except Exception:  # noqa: BLE001
            pass
        self._mirror_file(path, pepper)
        return pepper

    @staticmethod
    def _mirror_file(path, data: bytes) -> None:
        if path is None:
            return
        try:
            with open(str(path), "wb") as fh:
                fh.write(data)
            os.chmod(str(path), 0o600)
        except OSError:
            pass

    def _hash(self, plaintext: str) -> str:
        return hashlib.sha256(self._pepper + plaintext.encode()).hexdigest()

    # ------------------------------------------------------------- keys api

    def create_key(self, name: str, scopes: List[str],
                   created_by: str = "owner", user: Optional[str] = None,
                   request_limit: Optional[int] = None) -> Tuple[str, str, dict]:
        bad = [s for s in scopes if s not in VALID_SCOPES]
        if bad:
            raise ValueError(f"invalid scopes: {bad} "
                             f"(valid: {VALID_SCOPES})")
        if "owner" in scopes:
            scopes = SCOPE_ALL[:]
        else:
            scopes = list(dict.fromkeys(scopes)) or ["chat"]
        key_id = f"key-{secrets.token_hex(8)}"
        plaintext = "rxa_" + secrets.token_urlsafe(28)
        self.storage.key_insert(key_id, name, self._hash(plaintext), scopes,
                                user=user, request_limit=request_limit)
        self.storage.count("keys", "created", detail={"name": name})
        self.storage.log_event("key_created", {
            "id": key_id, "name": name, "scopes": scopes, "user": user,
            "by": created_by})
        return key_id, plaintext, {
            "id": key_id, "name": name, "scopes": scopes, "user": user,
            "request_limit": request_limit, "created_at": time.time(),
        }

    def ensure_owner_key(self) -> Tuple[str, str, bool]:
        """Return (key_id, plaintext, created_now).

        The plaintext is recoverable for the owner key only (it is written
        to the data dir on first boot); other keys are shown once.
        """
        from pathlib import Path
        owner_key_file: Optional[Path] = None
        if self.cfg is not None:
            owner_key_file = self.cfg.owner_key_path
            try:
                owner_key_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                owner_key_file = None
        # 1. the database (survives serverless cold starts)
        candidates = []
        try:
            stored = self.storage.secret_get("owner_key")
            if stored:
                candidates.append(stored)
        except Exception:  # noqa: BLE001
            pass
        # 2. the data dir file (self-hosted installs from before 0.5)
        if owner_key_file is not None:
            try:
                candidates.append(owner_key_file.read_text().strip())
            except OSError:
                pass
        for raw in candidates:
            if raw.startswith("rxa_"):
                row = self.storage.key_get_by_hash(self._hash(raw))
                if row and "owner" in row["scopes"]:
                    self._persist_owner_key(owner_key_file, raw)
                    return row["id"], raw, False
        key_id, plaintext, meta = self.create_key(
            "owner", SCOPE_ALL, created_by="first-boot")
        self._persist_owner_key(owner_key_file, plaintext)
        return key_id, plaintext, True

    def _persist_owner_key(self, owner_key_file, plaintext: str) -> None:
        try:
            self.storage.execute(
                "INSERT INTO state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("secret.owner_key", plaintext))
        except Exception:  # noqa: BLE001
            pass
        if owner_key_file is not None:
            try:
                with open(str(owner_key_file), "w", encoding="utf-8") as fh:
                    fh.write(plaintext + "\n")
                os.chmod(str(owner_key_file), 0o600)
            except OSError:
                pass

    def verify(self, plaintext: str) -> Optional[dict]:
        if not plaintext or not plaintext.startswith("rxa_"):
            return None
        row = self.storage.key_get_by_hash(self._hash(plaintext))
        if row:
            self.storage.key_touch(row["id"])
        return row

    def list_keys(self) -> List[dict]:
        return self.storage.key_list()

    def revoke(self, key_id: str) -> bool:
        ok = self.storage.key_revoke(key_id)
        if ok:
            self.storage.count("keys", "revoked")
            self.storage.log_event("key_revoked", {"id": key_id})
        return ok
