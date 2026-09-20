"""The permission gate.

Nothing executes without passing here. Policies are per action category
(``device.control``, ``system.command``, ``files.read``, ...) and can be
narrowed with an exact action pattern.

- ``allow`` -> execute immediately
- ``deny``  -> refuse
- ``ask``   (the default for everything) -> the request is queued as a
  pending permission request with an id. The owner approves or denies it
  (chat: "approve 7" / API). Approving stores an exact allow pattern, so
  that one action is then trusted; anything new asks again.

This is what makes "control any device, anything" safe: the agent can do
anything the owner lets it, and the owner always sees and controls what
that is.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..storage import Storage
from .base import ActionRequest

ALLOW, ASK, DENY = "allow", "ask", "deny"
_POLICIES = (ALLOW, ASK, DENY)


class Decision:
    def __init__(self, granted: bool, awaiting: bool = False,
                 pending_id: Optional[int] = None, reason: str = "") -> None:
        self.granted = granted
        self.awaiting = awaiting
        self.pending_id = pending_id
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "granted": self.granted,
            "awaiting": self.awaiting,
            "pending_id": self.pending_id,
            "reason": self.reason,
        }


class PermissionManager:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    # ------------------------------------------------------------- policies

    def policies(self) -> List[Dict[str, Any]]:
        return self.storage.query(
            "SELECT * FROM permissions ORDER BY created_at DESC")

    def set_policy(self, category: str, policy: str,
                   pattern: Optional[str] = None, note: str = "",
                   created_by: str = "owner") -> Dict[str, Any]:
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {list(_POLICIES)}")
        # replace any existing identical (category, pattern) policy
        self.storage.execute("DELETE FROM permissions WHERE category=? AND "
                             "pattern IS ?", (category, pattern))
        self.storage.execute(
            "INSERT INTO permissions(id, category, pattern, policy, note, "
            "created_by, created_at) VALUES(?,?,?,?,?,?,?)",
            (f"pol-{int(time.time()*1000)}", category, pattern, policy,
             note, created_by, time.time()),
        )
        self.storage.count("permissions", f"policy_{policy}",
                           detail={"category": category, "pattern": pattern})
        self.storage.log_event("permission", {
            "policy": policy, "category": category, "pattern": pattern,
        })
        return self.storage.query_one(
            "SELECT * FROM permissions WHERE category=? AND pattern IS ?",
            (category, pattern)) or {}

    # --------------------------------------------------------------- check

    def check(self, req: ActionRequest) -> Decision:
        rows = self.storage.query(
            "SELECT * FROM permissions WHERE category=? "
            "ORDER BY (pattern IS NULL) DESC", (req.category,))
        for row in rows:
            pattern = row["pattern"]
            if pattern is not None and pattern != req.action:
                continue
            if row["policy"] == DENY:
                self.storage.count("actions", "denied",
                                   detail={"category": req.category,
                                           "reason": "policy_deny"})
                return Decision(False, reason="denied by policy")
            if row["policy"] == ALLOW:
                return Decision(True, reason="allowed by policy")
        # default: ask
        pending_id = self.storage.execute(
            "INSERT INTO pending(ts, category, action, params, "
            "requested_by, reason) VALUES(?,?,?,?,?,?)",
            (time.time(), req.category, req.action,
             _dump(req.params), req.source, req.reason),
        ).lastrowid
        self.storage.count("permissions", "requested",
                           detail={"category": req.category})
        self.storage.log_event("permission_request", {
            "pending_id": pending_id, "category": req.category,
            "action": req.action,
        })
        return Decision(False, awaiting=True, pending_id=pending_id,
                        reason="awaiting owner approval")

    # ----------------------------------------------------------- decisions

    def pending_list(self, include_decided: bool = False,
                     limit: int = 50) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM pending"
        if not include_decided:
            sql += " WHERE status='pending'"
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self.storage.query(sql, (limit,))
        for r in rows:
            r["params"] = _load(r.get("params") or "{}")
        return rows

    def decide(self, pending_id: int, approve: bool,
               decided_by: str = "owner") -> Dict[str, Any]:
        row = self.storage.query_one("SELECT * FROM pending WHERE id=?",
                                     (pending_id,))
        if row is None:
            raise KeyError(f"pending request {pending_id} not found")
        if row["status"] != "pending":
            raise ValueError(f"pending request {pending_id} already "
                             f"{row['status']}")
        status = "approved" if approve else "denied"
        self.storage.execute(
            "UPDATE pending SET status=?, decided_by=?, decided_at=? "
            "WHERE id=?", (status, decided_by, time.time(), pending_id))
        if approve:
            # exact one-action allow (re-asks for anything else)
            self.set_policy(row["category"], ALLOW, pattern=row["action"],
                            note=f"approved #{pending_id}",
                            created_by=decided_by)
        self.storage.count("permissions", status,
                           detail={"pending_id": pending_id})
        self.storage.log_event("permission_decided", {
            "pending_id": pending_id, "status": status,
            "category": row["category"], "action": row["action"],
            "by": decided_by,
        })
        return self.storage.query_one("SELECT * FROM pending WHERE id=?",
                                      (pending_id,)) or {}

    def latest_pending(self) -> Optional[Dict[str, Any]]:
        row = self.storage.query_one(
            "SELECT * FROM pending WHERE status='pending' ORDER BY id DESC "
            "LIMIT 1")
        if row:
            row["params"] = _load(row.get("params") or "{}")
        return row


def _dump(obj: Any) -> str:
    import json
    return json.dumps(obj)


def _load(raw: str) -> Any:
    import json
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {"raw": raw}
