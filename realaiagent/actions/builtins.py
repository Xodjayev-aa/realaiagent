"""Built-in action handlers (pure stdlib).

Every handler returns (ok, output, error). Safety:

- ``files.*`` are confined to the owner-configured file roots.
- ``system.command`` is always permission-gated (default ``ask``) and
  run with a hard timeout.
- ``http.request`` uses stdlib urllib with a hard timeout.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config


class ScopeError(Exception):
    """Path outside the allowed roots."""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n…[truncated {len(text)-limit} chars]"


def files_read(cfg: Config, params: Dict[str, Any]) -> Tuple[bool, str, str]:
    path = _resolve_scoped(cfg, str(params.get("path", "")))
    if not path.exists():
        return False, "", f"not found: {path}"
    if not path.is_file():
        return False, "", f"not a file: {path}"
    data = path.read_bytes()
    if len(data) > cfg.max_file_read:
        data = data[: cfg.max_file_read]
    try:
        return True, _clip(data.decode("utf-8", "replace"), cfg.max_output_bytes), ""
    except Exception:
        return False, "", "unreadable file"


def files_write(cfg: Config, params: Dict[str, Any]) -> Tuple[bool, str, str]:
    path = _resolve_scoped(cfg, str(params.get("path", "")))
    content = str(params.get("content", ""))
    if params.get("append"):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(content)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    return True, f"wrote {len(content)} chars to {path.name}", ""


def _resolve_scoped(cfg: Config, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = cfg.data_dir / p
    p = p.resolve()
    for root in cfg.file_roots:
        if p == root or root in p.parents:
            return p
    raise ScopeError(f"path outside allowed roots: {p}")


def system_command(cfg: Config, params: Dict[str, Any]) -> Tuple[bool, str, str]:
    command = str(params.get("command", "")).strip()
    if not command:
        return False, "", "empty command"
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=cfg.command_timeout, cwd=str(cfg.data_dir),
        )
    except subprocess.TimeoutExpired:
        return False, "", f"timed out after {cfg.command_timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, "", f"spawn error: {exc}"
    out = _clip((proc.stdout or "") + (("\n[stderr]\n" + proc.stderr)
                                       if proc.stderr else ""),
                cfg.max_output_bytes)
    if proc.returncode != 0:
        return False, out, f"exit code {proc.returncode}"
    return True, out, ""


def http_request(cfg: Config, params: Dict[str, Any]) -> Tuple[bool, str, str]:
    url = str(params.get("url", ""))
    if not url.lower().startswith(("http://", "https://")):
        return False, "", "invalid url"
    method = str(params.get("method", "GET")).upper()
    headers = dict(params.get("headers") or {})
    body = params.get("body")
    data = None
    if body is not None:
        data = body.encode() if isinstance(body, str) \
            else json.dumps(body).encode()
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=cfg.http_timeout) as resp:
            raw = resp.read(cfg.max_output_bytes)
            return True, _clip(raw.decode("utf-8", "replace"),
                               cfg.max_output_bytes), ""
    except urllib.error.HTTPError as exc:
        return False, "", f"http {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, "", f"http error: {exc}"


def notify(_cfg: Config, params: Dict[str, Any]) -> Tuple[bool, str, str]:
    return True, f"notified: {params.get('message', '')}", ""


def timer_set(cfg: Config, params: Dict[str, Any]) -> Tuple[bool, str, str]:
    """Record a timer (executed by the autonomy loop when due)."""
    delay = float(params.get("delay_seconds", 0))
    return True, f"timer set for {delay}s", ""


def render_template(template: str, params: Dict[str, Any]) -> str:
    """Render ``{placeholder}`` tokens with params; unknown left intact."""

    class _Safe(dict):
        def __missing__(self, key: str) -> str:  # noqa: D105
            return "{" + key + "}"

    try:
        return template.format_map(_Safe(**{k: str(v) for k, v in params.items()}))
    except (ValueError, IndexError):
        return template
