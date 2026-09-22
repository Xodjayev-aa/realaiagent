"""The product's own pages: the chat app, developer portal, owner inbox.

Server-rendered HTML from the standard library — ``string.Template`` for
substitution, no template engine, no build step, no CDN, no analytics, no
third-party font or script. Everything the browser gets comes from this
repo, which is the same rule the brain obeys: nothing external.

0.7.0 — a senior-level interface, ChatGPT/Gemini-grade:

=============== =========================================================
``/``            the chat app — a count-ring mark, a quiet sidebar with
                 searchable conversations, a centered empty state with
                 suggestion chips, a single pill composer, clean bubbles,
                 light and dark themes (system default, toggleable)
``/developers``  the portal — live endpoint table from the real route
                 registry (the same data ``/api.json`` serves) plus the
                 key-request form
``/approve``     the owner's approval inbox — web-token gated, one-tap
                 approve (key shown exactly once), deny, unban, VIP
=============== =========================================================

The chat app talks to ``POST /public/chat``: keyless, rate-limited per
visitor, no scopes — so nothing privileged can be reached from a browser.
``/v1/*`` stays behind owner-issued keys.

The mark: a "count ring" — a ring with one opening and one dot, a thought
being counted. Hand-written SVG in this module; the same bytes become the
favicon. No external asset is ever fetched, embedded or linked.

Deliberate omissions: no cookies (sessions live in ``localStorage`` and in
the database), no external requests of any kind, and no secret is ever
rendered into a page that is not gated.
"""

from __future__ import annotations

import html
import json
import re
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Human descriptions for the developer portal's endpoint table. Anything
#: not listed here still renders — with an em dash — because the table is
#: driven by the live route registry, not by this dict.
ENDPOINT_DOCS: Dict[str, str] = {
    "/": "this app — the chat page (public)",
    "/public/chat": "keyless demo chat: {\"message\": \"...\", "
                    "\"session_id\": \"...\"}",
    "/developers": "the developer portal (public)",
    "/logo.svg": "the mark (SVG), also used as the favicon",
    "/favicon.ico": "favicon (SVG payload, served on the classic path too)",
    "/api.json": "machine-readable index of every route and its scopes",
    "/healthz": "liveness: agent name, version, uptime, mood",
    "/dashboard": "the owner's live telemetry (15 SSE topics, web token)",
    "/stream": "list the SSE topics; /stream/<topic> streams one",
    "/webhook": "Telegram webhook (owner's own bot, secret-token checked)",
    "/approve": "owner approval inbox (web token): approve/deny/unban/VIP",
    "/v1/chat": "talk to the agent  {\"message\": \"...\"}",
    "/v1/status": "mind state: drives, mood, goals, counters",
    "/v1/counters": "everything counted — the full ledger",
    "/v1/usage": "usage breakdowns  ?by=category|key|day (admin)",
    "/v1/events": "the agent's thoughts & actions log",
    "/v1/goals": "goal queue (list / create / complete / remove)",
    "/v1/memory": "semantic + episodic memory (list / store / delete)",
    "/v1/learn": "train the language model  {\"examples\": [...]} (learn)",
    "/v1/skills": "learned procedures (list / create / delete)",
    "/v1/devices": "registered devices (list / register / control)",
    "/v1/permissions": "policies + pending requests (owner decides)",
    "/v1/keys": "API keys: list / mint / revoke (owner)",
    "/v1/users": "clients: PENDING / APPROVED / BANNED (owner)",
    "/v1/users/request": "ask the owner for a key (public, no key needed)",
    "/v1/owner/state": "identity, values, mood, autonomy (owner)",
    "/v1/owner/identity": "rename the agent or its owner (owner)",
    "/v1/owner/values": "set the drives (owner)",
    "/v1/owner/autonomy": "the kill switch (owner)",
}

#: Error codes the API can return, for the portal's reference table.
ERROR_CODES: Tuple[Tuple[str, str], ...] = (
    ("400 bad_request", "missing/invalid field, or a body that is not a "
                        "JSON object"),
    ("401 unauthorized", "no key, an unknown key, or a revoked one"),
    ("402 insufficient_funds", "a billed client ran out of balance"),
    ("403 forbidden", "the key exists but lacks the scope (or the user is "
                      "banned / not approved)"),
    ("404 not_found", "no such route, goal, key, device or user"),
    ("409 conflict", "that id already exists (e.g. a device)"),
    ("413 body_too_large", "request body over REALAI_MAX_BODY bytes"),
    ("429 rate_limited", "token bucket empty — per key, or per demo visitor"),
    ("429 too_many_attempts", "too many wrong web tokens from your address — "
                              "the owner surfaces are locked out briefly"),
    ("500 internal", "the agent raised; it is counted in the ledger"),
)


# ------------------------------------------------------------------ helpers

def esc(value: Any) -> str:
    """Escape anything that is about to be interpolated into HTML."""
    return html.escape("" if value is None else str(value), quote=True)


# ---------------------------------------------------------------------- mark
# The "count ring": a ring with one opening, one dot in it — a thought being
# counted. Geometry: center (32,32), radius 20, stroke 7, round caps; the
# gap is 72 degrees centred at -55 degrees (upper right) and the dot sits in
# the middle of the gap. viewBox 0 0 64 64 so it scales from favicon to hero.

#: Ring + dot as a bare <g> — colours via attributes (endpoint SVGs) or via
#: classes (in-page SVGs, where the dot picks up the theme accent).
_MARK_PATH = ("M50.91 25.47A20 20 0 1 1 31.65 12")

_MARK_INNER_ATTR = (
    '<path d="' + _MARK_PATH + '" fill="none" stroke-width="7" '
    'stroke-linecap="round" stroke="@@RING@@"/>'
    '<circle cx="43.47" cy="15.62" r="5" fill="@@DOT@@"/>'
)


def _logo_mark(agent_name: str = "REAL") -> str:
    """The product's own mark, adaptive to the page theme.

    Ring strokes ``currentColor``; the dot takes ``--accent`` from the CSS
    (see ``.mark .md``). The agent's name appears in the wordmark next to
    the mark, the way a name sits next to a company glyph.
    """
    return (
        '<svg viewBox="0 0 64 64" class="mark" role="img" '
        'aria-label="' + esc(agent_name) + ' mark" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<path class="mr" d="' + _MARK_PATH + '" fill="none" '
        'stroke="currentColor" stroke-width="7" stroke-linecap="round"/>'
        '<circle class="md" cx="43.47" cy="15.62" r="5"/></svg>'
    )


#: The mark as a self-contained app icon: dark tile, light ring, accent
#: dot. This is what ``/logo.svg`` and ``/favicon.ico`` serve, so it reads
#: on any browser tab or OS context.
LOGO_SVG = (
    '<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="0" y="0" width="64" height="64" rx="14.5" fill="#0f1115"/>'
    + _MARK_INNER_ATTR.replace("@@RING@@", "#f4f6f8")
                      .replace("@@DOT@@", "#6c9bff")
    + "</svg>"
)

#: Decorative (aria-hidden) mark for JS-built avatars — no theme text.
MARK_DECORATIVE = (
    '<svg viewBox="0 0 64 64" class="mark" aria-hidden="true" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<path class="mr" d="' + _MARK_PATH + '" fill="none" '
    'stroke="currentColor" stroke-width="7" stroke-linecap="round"/>'
    '<circle class="md" cx="43.47" cy="15.62" r="5"/></svg>'
)


def favicon_bytes(agent_name: str = "REAL") -> bytes:
    """The favicon payload — an SVG we generate ourselves."""
    return LOGO_SVG.encode("utf-8")


# ------------------------------------------------------------------ markdown

_FENCE_RE = re.compile(r"^```[ \t]*(\w*)[ \t]*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ULIST_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_OLIST_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def _inline(text: str) -> str:
    """Inline markdown on already-escaped text (so it stays XSS-safe)."""
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)

    def _link(match: "re.Match[str]") -> str:
        label, href = match.group(1), match.group(2)
        # Only http(s) and same-origin relative links; everything else is
        # rendered as text so a javascript: URL can never become clickable.
        if re.match(r"^(https?://|/|#)", href):
            rel = ' rel="noopener noreferrer"' if href.startswith("http") \
                else ""
            return f'<a href="{href}"{rel}>{label}</a>'
        return f"{label} ({href})"

    return _LINK_RE.sub(_link, text)


def markdown_to_html(text: str) -> str:
    """A small, safe markdown renderer (server side).

    Escapes first, then formats — the only way to keep user/agent text from
    ever becoming markup. Handles fenced code, headings, bullet and numbered
    lists, inline code, bold, italic and links. The browser has its own
    equivalent for streamed text; both produce the same shapes.
    """
    lines = (text or "").replace("\r\n", "\n").split("\n")
    out: List[str] = []
    list_kind: Optional[str] = None
    in_code = False
    para: List[str] = []

    def _flush_para() -> None:
        nonlocal para
        if para:
            out.append("<p>" + "<br>".join(_inline(html.escape(p))
                                          for p in para) + "</p>")
            para = []

    def _flush_list() -> None:
        nonlocal list_kind
        if list_kind:
            out.append(f"</{list_kind}>")
            list_kind = None

    for line in lines:
        fence = _FENCE_RE.match(line)
        if fence:
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                _flush_para()
                _flush_list()
                lang = fence.group(1)
                cls = f' class="lang-{esc(lang)}"' if lang else ""
                out.append(f"<pre><code{cls}>")
                in_code = True
            continue
        if in_code:
            out.append(html.escape(line) + "\n")
            continue

        heading = _HEADING_RE.match(line)
        bullet = _ULIST_RE.match(line)
        numbered = _OLIST_RE.match(line)

        if not line.strip():
            _flush_para()
            _flush_list()
            continue
        if heading:
            _flush_para()
            _flush_list()
            level = min(len(heading.group(1)) + 2, 6)
            out.append(f"<h{level}>" + _inline(html.escape(heading.group(2)))
                       + f"</h{level}>")
            continue
        if bullet:
            _flush_para()
            if list_kind != "ul":
                _flush_list()
                out.append("<ul>")
                list_kind = "ul"
            out.append("<li>" + _inline(html.escape(bullet.group(1)))
                       + "</li>")
            continue
        if numbered:
            _flush_para()
            if list_kind != "ol":
                _flush_list()
                out.append("<ol>")
                list_kind = "ol"
            out.append("<li>" + _inline(html.escape(numbered.group(2)))
                       + "</li>")
            continue
        _flush_list()
        para.append(line)

    if in_code:
        out.append("</code></pre>")
    _flush_para()
    _flush_list()
    return "\n".join(out)


# --------------------------------------------------------------------- icons
# Hand-drawn 24x24 line icons: 1.7-2px strokes, round caps, currentColor.

def _ic(inner: str, size: int = 16, width: str = "1.7") -> str:
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" '
            f'fill="none" stroke="currentColor" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{inner}</svg>')


IC_PLUS = _ic('<path d="M12 5v14M5 12h14"/>', 17)
IC_SEARCH = _ic('<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.2-3.2"/>', 15)
IC_TRASH = _ic('<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12h10l1-12M9 7V5h6v2"/>',
               14)
IC_SEND = _ic('<path d="M12 19V5M5 12l7-7 7 7"/>', 18, "2")
IC_MIC = _ic('<path d="M12 3a3 3 0 0 1 3 3v5a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z"/>'
             '<path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3"/>', 18)
IC_SUN = _ic('<circle cx="12" cy="12" r="4"/>'
             '<path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5 5l1.4 1.4'
             'M17.6 17.6L19 19M19 5l-1.4 1.4M6.4 17.6L5 19"/>', 17)
IC_MOON = _ic('<path d="M20.5 13.2A8.5 8.5 0 1 1 10.8 3.5a7 7 0 0 0 9.7 9.7z"/>'
              , 17)
IC_CHEV_L = _ic('<path d="M15 6l-6 6 6 6"/>', 17)
IC_BURGER = _ic('<path d="M4 7h16M4 12h16M4 17h16"/>', 19, "1.8")
IC_COPY = _ic('<rect x="9" y="9" width="11" height="11" rx="2"/>'
              '<path d="M5 15V6a2 2 0 0 1 2-2h9"/>', 15)
IC_CHECK = _ic('<path d="M5 13l4 4L19 7"/>', 15, "2")
IC_SPEAK = _ic('<path d="M4 9.5v5h3.5L13 19V5L7.5 9.5H4z"/>'
               '<path d="M16.5 9a4.5 4.5 0 0 1 0 6M19 6.5a8 8 0 0 1 0 11"/>',
               15)
IC_DL = _ic('<path d="M12 4v11M7 11l5 5 5-5M5 20h14"/>', 15)
IC_SPARK = _ic('<path d="M12 3l2.1 5.9L20 11l-5.9 2.1L12 19l-2.1-5.9L4 11'
               'l5.9-2.1z"/>', 16)
IC_BUBBLE = _ic('<path d="M21 14a2 2 0 0 1-2 2H8l-4 4V5a2 2 0 0 1 2-2h13'
                'a2 2 0 0 1 2 2z"/>', 16)
IC_IMG = _ic('<rect x="3" y="5" width="18" height="14" rx="2.5"/>'
             '<circle cx="8.7" cy="10" r="1.6"/>'
             '<path d="M21 15.5L16 10.5 7 19"/>', 16)
IC_SLIDES = _ic('<rect x="3" y="4" width="18" height="12" rx="2"/>'
                '<path d="M12 16v4M8.5 20h7"/>', 16)
IC_CODE = _ic('<path d="M8.5 8.5L5 12l3.5 3.5M15.5 8.5L19 12l-3.5 3.5"/>',
              16)
IC_PULSE = _ic('<path d="M3 12h4l3 7 4-14 3 7h4"/>', 16)
IC_USER = _ic('<circle cx="9.5" cy="7.5" r="3.5"/>'
              '<path d="M3.5 20v-.5a5.5 5.5 0 0 1 5.5-5.5h1a5.5 5.5 0 0 1 5.5 '
              '5.5v.5M16 4.6a3.5 3.5 0 0 1 0 5.8M18.5 14.6a5.5 5.5 0 0 1 2 '
              '4.4v1"/>', 16)
IC_API = _ic('<path d="M6 16l-3-4 3-4M18 8l3 4-3 4M13.5 6l-3 12"/>', 16)


# ----------------------------------------------------------------------- css
#: The whole design system. Light theme is the default; ``data-theme="dark"``
#: on <html> flips it. The chat app (body.chat) is a 100dvh two-column shell
#: (quiet sidebar + composer docked under the thread); the document pages
#: (body.page) share the same tokens under a sticky header.

_CSS = """
:root {
  color-scheme: light;
  --bg:#ffffff;          /* chat main / cards   */
  --bg2:#f7f7f8;         /* sidebar, page body  */
  --card:#ffffff;        /* cards on pages      */
  --ink:#0d0d0d; --ink2:#56565d; --ink3:#8e8e96;
  --line:#e8e8eb; --line2:#d6d6dc;
  --bub:#f3f3f5; --bub2:#e9e9ec;
  --code-bg:#f6f6f7;
  --accent:#2f6bff; --accent-soft:rgba(47,107,255,.12);
  --ok:#187a48; --warn:#9a5b00; --bad:#d13438;
  --send-bg:#0d0d0d; --send-ink:#ffffff;
  --headbg:rgba(247,247,248,.88);
  --shadow:0 1px 2px rgba(16,16,20,.04),0 8px 28px rgba(16,16,20,.07);
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,Roboto,
    "Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,
    "Liberation Mono",monospace;
}
[data-theme="dark"] {
  color-scheme: dark;
  --bg:#212121; --bg2:#171717; --card:#262628;
  --ink:#ededed; --ink2:#b2b2b8; --ink3:#7d7d84;
  --line:rgba(255,255,255,.11); --line2:rgba(255,255,255,.22);
  --bub:#2f2f31; --bub2:#3a3a3d;
  --code-bg:#1b1b1c;
  --accent:#7aa2f7; --accent-soft:rgba(122,162,247,.16);
  --ok:#4cc38a; --warn:#e8a33d; --bad:#ff7b72;
  --send-bg:#ededed; --send-ink:#141414;
  --headbg:rgba(23,23,23,.88);
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 32px rgba(0,0,0,.45);
}
* { box-sizing:border-box; }
html, body { height:100%; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 var(--font); letter-spacing:-.004em;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
  -webkit-text-size-adjust:100%; }
body.page { background:var(--bg2); }
button { font:inherit; color:inherit; }
a { color:var(--accent); text-decoration:none; }
::selection { background:var(--accent-soft); }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
* { scrollbar-width:thin; scrollbar-color:var(--line2) transparent; }
*::-webkit-scrollbar { width:10px; height:10px; }
*::-webkit-scrollbar-thumb { background:var(--line2); border-radius:99px;
  border:3px solid transparent; background-clip:content-box; }
*::-webkit-scrollbar-track { background:transparent; }
code, pre { font-family:var(--mono); }
code { font-size:.88em; background:var(--bub); padding:1.5px 6px;
  border-radius:6px; }
pre { background:var(--code-bg); border:1px solid var(--line);
  border-radius:12px; padding:13px 15px; overflow-x:auto; font-size:12.8px;
  line-height:1.65; margin:12px 0; }
pre code { background:none; padding:0; font-size:inherit; }
/* ---------- the mark ---------- */
.mark { display:block; }
.mark .mr { stroke:currentColor; }
.mark .md { fill:var(--accent,#2f6bff); }
/* ---------- shared bits ---------- */
.iconbtn { width:34px; height:34px; flex:0 0 auto; border-radius:9px;
  border:0; background:transparent; color:var(--ink3); display:grid;
  place-items:center; cursor:pointer; }
.iconbtn:hover { background:var(--bub); color:var(--ink); }
.live { width:7px; height:7px; border-radius:50%; background:var(--ok);
  display:inline-block; flex:0 0 auto; }
.live.off { background:var(--ink3); }
.pill { display:inline-block; padding:1px 8px; border-radius:999px;
  font-size:11px; border:1px solid var(--line2); color:var(--ink2);
  background:var(--bg2); white-space:nowrap; }
.pill.pub { color:var(--ok); border-color:var(--ok); background:transparent; }
.pill.own { color:var(--warn); border-color:var(--warn);
  background:transparent; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .bad { color:var(--bad); }
.dim { color:var(--ink3); }
/* ---------- document pages (developers / approve) ---------- */
header.top { position:sticky; top:0; z-index:30; background:var(--headbg);
  backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  border-bottom:1px solid var(--line); }
.top-in { max-width:1060px; margin:0 auto; padding:9px 20px;
  display:flex; align-items:center; gap:14px; }
.brand { display:flex; align-items:center; gap:9px; color:var(--ink);
  font-weight:650; font-size:15.5px; letter-spacing:-.015em; }
.brand:hover { text-decoration:none; }
.brand .mark { width:23px; height:23px; }
nav.links { margin-left:auto; display:flex; gap:2px; align-items:center; }
nav.links a, nav.links .iconbtn { font-size:13.5px; color:var(--ink2);
  padding:7px 11px; border-radius:9px; font-weight:500; }
nav.links a:hover { background:var(--bub); color:var(--ink); }
nav.links a.here { color:var(--ink); font-weight:600; background:var(--bub); }
main.page-main { max-width:1060px; margin:0 auto; padding:26px 20px 12px; }
h1 { font-size:24px; font-weight:660; letter-spacing:-.02em; margin:2px 0 8px; }
.tag { color:var(--ink2); font-size:14.5px; line-height:1.6; max-width:80ch;
  margin:0 0 22px; }
.card { background:var(--card); border:1px solid var(--line);
  border-radius:16px; padding:20px 22px; margin-bottom:16px; }
.k { color:var(--ink3); font-size:11.5px; text-transform:uppercase;
  letter-spacing:.09em; margin-bottom:12px; font-weight:650; }
.grid { display:grid; gap:12px;
  grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); }
.tile { background:var(--bg2); border:1px solid var(--line);
  border-radius:12px; padding:13px 15px; }
.tile h3 { margin:0 0 5px; font-size:14px; letter-spacing:-.01em; }
.tile p { margin:0; color:var(--ink2); font-size:13px; line-height:1.55; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th { text-align:left; color:var(--ink3); font-size:11px; font-weight:650;
  text-transform:uppercase; letter-spacing:.07em; padding:8px 10px;
  border-bottom:1px solid var(--line); }
td { padding:9px 10px; vertical-align:top; }
tbody tr + tr td { border-top:1px solid var(--line); }
tbody tr:hover td { background:var(--bg2); }
.m { color:var(--ink2); font-family:var(--mono); white-space:nowrap;
  font-size:12px; }
.p { color:var(--ink); font-family:var(--mono); font-size:12.3px;
  word-break:break-all; }
input, textarea, select { background:var(--card); color:var(--ink);
  border:1px solid var(--line2); border-radius:10px; padding:10px 12px;
  font:inherit; font-size:14px; width:100%; }
input::placeholder, textarea::placeholder { color:var(--ink3); }
input:focus, textarea:focus { outline:none; border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-soft); }
label { display:block; color:var(--ink2); font-size:12.5px; font-weight:550;
  margin:0 0 6px; }
.field { margin-bottom:14px; }
.btn { display:inline-flex; align-items:center; justify-content:center;
  gap:7px; border-radius:10px; padding:9px 16px; font-size:13.5px;
  font-weight:600; cursor:pointer; border:1px solid transparent;
  letter-spacing:-.01em; }
.btn:disabled { opacity:.55; cursor:progress; }
.btn-p { background:var(--ink); color:var(--bg); }
.btn-p:hover { opacity:.88; }
.btn-g { background:var(--card); border-color:var(--line2); color:var(--ink); }
.btn-g:hover { background:var(--bub); }
.btn-d { background:none; border-color:transparent; color:var(--bad); }
.btn-d:hover { background:rgba(209,52,56,.1); }
.rowbtns { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.hint { color:var(--ink3); font-size:12.5px; margin-top:8px; line-height:1.5; }
.note { border:1px solid var(--line); border-left:3px solid var(--accent);
  background:var(--bg2); padding:11px 14px; border-radius:12px;
  color:var(--ink2); font-size:13.5px; margin:12px 0; }
.note.warn { border-left-color:var(--warn); }
.note.good { border-left-color:var(--ok); }
footer.page-foot { max-width:1060px; margin:0 auto; padding:14px 20px 30px;
  color:var(--ink3); font-size:12.5px; border-top:1px solid var(--line); }
/* key reveal */
#keybox { display:none; border-color:var(--ok); }
#keybox.on { display:block; }
#keybox .kv { font-family:var(--mono); font-size:14px;
  background:var(--code-bg); border:1px solid var(--line2); color:var(--ink);
  padding:12px 14px; border-radius:10px; word-break:break-all; }
/* ---------- the chat shell ---------- */
body.chat { overflow:hidden; }
.shell { display:grid; grid-template-columns:268px minmax(0,1fr);
  height:100vh; height:100dvh; overflow:hidden; }
body.side-off .shell { grid-template-columns:0 minmax(0,1fr); }
body.side-off .side { border-right:0; }
aside.side { background:var(--bg2); border-right:1px solid var(--line);
  display:flex; flex-direction:column; min-height:0; overflow:hidden; }
.side-top { display:flex; align-items:center; justify-content:space-between;
  padding:14px 12px 8px; }
.side-top .brand { padding:4px 6px; border-radius:9px; }
.side-top .brand:hover { background:var(--bub); }
.side-actions { display:flex; gap:2px; }
.newchat { display:flex; align-items:center; gap:9px; width:100%;
  padding:9px 12px; border-radius:10px; border:0; background:transparent;
  color:var(--ink); font-size:14px; font-weight:550; cursor:pointer;
  text-align:left; }
.newchat:hover { background:var(--bub); }
.newchat svg { color:var(--ink2); }
.side-new { padding:2px 12px 8px; }
.side-search { display:flex; align-items:center; gap:8px; margin:0 12px 6px;
  padding:7px 11px; border-radius:10px; background:var(--card);
  border:1px solid var(--line); color:var(--ink3); }
.side-search input { border:0; outline:0; box-shadow:none; background:none;
  padding:0; font-size:13.5px; }
.side-sess { flex:1; overflow-y:auto; padding:2px 8px 8px; min-height:0; }
.slabel { font-size:11px; font-weight:650; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink3); padding:10px 10px 6px; }
.sess { display:flex; align-items:center; gap:8px; padding:8px 10px;
  border-radius:9px; cursor:pointer; color:var(--ink2); font-size:13.5px;
  border:1px solid transparent; }
.sess:hover { background:var(--bub); color:var(--ink); }
.sess.on { background:var(--bub); color:var(--ink); font-weight:550; }
.sess .t { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }
.sess .x { display:none; width:24px; height:24px; border:0; background:none;
  color:var(--ink3); border-radius:6px; cursor:pointer; place-items:center;
  flex:0 0 auto; }
.sess:hover .x { display:grid; }
.sess .x:hover { color:var(--bad); background:var(--bub2); }
.sess-hint { padding:8px 10px; color:var(--ink3); font-size:12.8px; }
.side-foot { border-top:1px solid var(--line); padding:10px 10px 12px; }
.side-user { display:flex; align-items:center; gap:10px; padding:6px 8px 10px;
  min-width:0; }
.su-av { width:30px; height:30px; flex:0 0 30px; border-radius:50%;
  background:var(--bub); display:grid; place-items:center; color:var(--ink); }
.su-av .mark { width:17px; height:17px; }
.su-name { font-size:13.5px; font-weight:600; display:flex; align-items:center;
  gap:7px; min-width:0; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }
.su-sub { font-size:11.5px; color:var(--ink3); margin-top:1px;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.side-links { display:flex; flex-direction:column; gap:1px; }
.side-links a, .side-links button { display:flex; align-items:center;
  gap:10px; padding:8px 10px; border-radius:9px; border:0; background:none;
  color:var(--ink2); font-size:13.5px; cursor:pointer; text-align:left;
  width:100%; }
.side-links a:hover, .side-links button:hover { background:var(--bub);
  color:var(--ink); }
.side-links svg { color:var(--ink3); flex:0 0 auto; }
.side-links .tail { margin-left:auto; color:var(--ink3); }
main.main { display:flex; flex-direction:column; min-width:0; min-height:0;
  background:var(--bg); }
.topbar { display:flex; align-items:center; gap:6px; padding:9px 14px;
  flex:0 0 auto; position:relative; }
.top-links { margin-left:auto; display:flex; align-items:center; gap:2px; }
.top-links a { font-size:13px; color:var(--ink2); padding:7px 11px;
  border-radius:8px; font-weight:500; }
.top-links a:hover { color:var(--ink); background:var(--bub); }
.top-links a.here { color:var(--ink); font-weight:600; }
.chat-scroll { flex:1; min-height:0; overflow-y:auto; overflow-x:hidden;
  scroll-behavior:smooth; }
.chat-body { display:flex; flex-direction:column; min-height:100%; }
/* empty state */
.hero { flex:1; display:flex; flex-direction:column; align-items:center;
  justify-content:center; text-align:center; padding:24px 20px 44px; }
.hero .mark { width:46px; height:46px; color:var(--ink); margin-bottom:14px; }
.hero h1 { font-size:27px; font-weight:640; letter-spacing:-.022em; margin:0;
  line-height:1.25; }
.hero .sub { color:var(--ink3); font-size:14px; margin:8px 0 0; max-width:52ch;
  line-height:1.55; }
.hero.hide { display:none; }
.chips { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px;
  width:100%; max-width:560px; margin-top:22px; }
.chip { display:flex; align-items:center; gap:11px; text-align:left;
  padding:11px 13px; border:1px solid var(--line); border-radius:14px;
  background:var(--card); cursor:pointer; transition:border-color .12s,
  box-shadow .12s; }
.chip:hover { border-color:var(--line2); box-shadow:0 2px 10px
  rgba(16,16,20,.06); }
.chip .ci { width:32px; height:32px; flex:0 0 32px; border-radius:10px;
  background:var(--bub); color:var(--ink2); display:grid; place-items:center; }
.chip b { display:block; font-size:13.3px; font-weight:600; color:var(--ink);
  letter-spacing:-.01em; }
.chip span.d { display:block; font-size:11.8px; color:var(--ink3);
  margin-top:1px; }
/* thread */
.thread { width:100%; max-width:780px; margin:0 auto; padding:14px 20px 8px; }
.msg { display:flex; gap:12px; padding:9px 0; align-items:flex-start; }
.msg .av { width:30px; height:30px; flex:0 0 30px; border-radius:50%;
  background:var(--bub); display:grid; place-items:center; color:var(--ink);
  margin-top:1px; }
.msg .av .mark { width:17px; height:17px; }
.msg .flow { min-width:0; flex:1; }
.msg.me { justify-content:flex-end; }
.msg.me .flow { display:flex; justify-content:flex-end; }
.bub-u { display:block; background:var(--bub); border-radius:20px;
  padding:10px 16px; font-size:14.8px; line-height:1.55;
  word-break:break-word; max-width:min(78%,640px); }
.bub-a { font-size:15.2px; line-height:1.66; letter-spacing:-.008em;
  max-width:720px; word-break:break-word; }
.bub-a p { margin:0 0 10px; } .bub-a p:last-child { margin:0; }
.bub-a ul, .bub-a ol { margin:8px 0 10px; padding-left:22px; }
.bub-a li { margin:3px 0; }
.bub-a h3, .bub-a h4, .bub-a h5, .bub-a h6 { margin:14px 0 6px;
  font-size:15.5px; letter-spacing:-.01em; }
.bub-a code { background:var(--bub); }
.bub-a pre { margin:10px 0; }
.bub-a a { text-decoration:underline; text-underline-offset:2px; }
.bub-a img.inl { max-width:100%; max-height:420px; border-radius:12px;
  border:1px solid var(--line); margin:8px 0; display:block; }
.acts { display:flex; gap:2px; margin-top:4px; opacity:0;
  transition:opacity .15s; }
.msg:hover .acts { opacity:1; }
@media (hover:none) { .acts { opacity:.65; } }
.actb { width:28px; height:28px; border-radius:7px; border:0; background:none;
  color:var(--ink3); display:grid; place-items:center; cursor:pointer; }
.actb:hover { background:var(--bub); color:var(--ink); }
.att { margin-top:10px; display:flex; flex-direction:column; gap:8px;
  max-width:480px; }
.att img { max-width:100%; max-height:420px; border-radius:12px;
  border:1px solid var(--line); display:block; }
.att audio { width:100%; max-width:420px; }
.filedl { display:inline-flex; align-items:center; gap:9px; padding:9px 13px;
  border:1px solid var(--line); border-radius:12px; background:var(--card);
  color:var(--ink); font-size:13.5px; max-width:100%; }
.filedl:hover { background:var(--bub); }
.filedl .fn { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
/* typing */
.typing { display:none; }
.typing.on { display:flex; }
.tdots { display:flex; gap:5px; padding:13px 15px; background:var(--bub);
  border-radius:18px; width:fit-content; }
.tdots span { width:7px; height:7px; border-radius:50%; background:var(--ink3);
  animation:bob 1.2s infinite ease-in-out; }
.tdots span:nth-child(2) { animation-delay:.15s; }
.tdots span:nth-child(3) { animation-delay:.3s; }
@keyframes bob { 0%,60%,100% { transform:translateY(0); opacity:.5; }
  30% { transform:translateY(-4px); opacity:1; } }
.cursor { display:inline-block; width:7px; height:15px; background:var(--ink);
  opacity:.65; animation:blink 1s steps(2) infinite; vertical-align:-2px;
  border-radius:2px; margin-left:2px; }
@keyframes blink { 50% { opacity:0; } }
/* composer dock */
.dock { flex:0 0 auto; padding:4px 16px 8px; background:var(--bg); }
.dock-in { max-width:780px; margin:0 auto; }
.composer { display:flex; align-items:flex-end; gap:6px;
  background:var(--card); border:1px solid var(--line2); border-radius:26px;
  padding:8px; box-shadow:var(--shadow); transition:box-shadow .15s; }
.composer:focus-within { box-shadow:var(--shadow),0 0 0 3px
  var(--accent-soft); }
.composer textarea { flex:1; min-width:0; border:0; outline:0; resize:none;
  background:none; font:inherit; font-size:15px; line-height:1.5;
  max-height:200px; padding:9px 6px 9px 9px; min-height:24px; }
.composer textarea::placeholder { color:var(--ink3); }
.sbtn { width:36px; height:36px; flex:0 0 36px; border-radius:50%;
  display:grid; place-items:center; border:1px solid var(--line);
  background:none; color:var(--ink2); cursor:pointer; }
.sbtn:hover { background:var(--bub); color:var(--ink); }
.sbtn.rec { background:var(--bad); border-color:var(--bad); color:#fff;
  animation:pulse 1.2s infinite; }
@keyframes pulse { 50% { opacity:.55; } }
.send { border:0; background:var(--send-bg); color:var(--send-ink);
  cursor:pointer; transition:transform .06s; }
.send:hover { background:var(--send-bg); opacity:.88; }
.send:active { transform:scale(.93); }
.send:disabled { background:var(--bub); color:var(--ink3); cursor:default;
  opacity:1; }
.fine { text-align:center; color:var(--ink3); font-size:12px; padding:9px 0 2px;
  letter-spacing:0; }
/* scrim (mobile drawer) */
#scrim { display:none; position:fixed; inset:0; background:rgba(0,0,0,.42);
  z-index:40; }
#scrim.on { display:block; }
/* ---------- dashboard (telemetry) ---------- */
.dstat { display:flex; align-items:center; gap:8px; font-size:13px;
  color:var(--ink2); margin-left:6px; }
.statrow { display:grid; gap:12px; margin-bottom:16px;
  grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); }
.big { font-size:21px; font-weight:650; letter-spacing:-.02em;
  margin-top:2px; }
.dtopics { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }
.tchip { background:var(--bg2); color:var(--ink2); border:1px solid var(--line);
  border-radius:999px; padding:5px 12px; font-size:12.5px; cursor:pointer; }
.tchip:hover { border-color:var(--line2); color:var(--ink); }
.tchip.on { background:var(--accent-soft); border-color:var(--accent);
  color:var(--accent); font-weight:600; }
.dfeed { background:var(--code-bg); border:1px solid var(--line);
  border-radius:12px; padding:10px 13px; height:56vh; min-height:300px;
  overflow-y:auto; font:12.3px/1.7 var(--mono); }
.row { display:flex; gap:10px; padding:3px 0; border-bottom:1px solid
  var(--line); word-break:break-all; }
.seq { color:var(--ink3); flex:0 0 auto; }
.tp { color:var(--accent); font-weight:600; flex:0 0 auto; }
.row code { background:none; padding:0; color:var(--ink2); }
/* ---------- token gate / locked ---------- */
.gate-wrap { min-height:100vh; min-height:100dvh; display:grid;
  place-items:center; padding:24px; background:var(--bg2); }
.gate-card { background:var(--card); border:1px solid var(--line);
  border-radius:18px; padding:28px 28px 24px; max-width:400px; width:100%;
  box-shadow:var(--shadow); }
.gate-card .mark { width:34px; height:34px; margin-bottom:14px; }
.gate-card h1 { font-size:19px; margin:0 0 8px; letter-spacing:-.015em; }
.gate-card p { color:var(--ink2); font-size:13.5px; margin:0 0 16px;
  line-height:1.55; }
.gate-card form input { margin-bottom:12px; }
.gate-back { margin-top:16px; }
.gate-back a { font-size:13px; color:var(--ink2); }
.gate-back a:hover { color:var(--ink); }
/* ---------- mobile ---------- */
@media (max-width: 860px) {
  .shell { grid-template-columns:1fr; }
  body.side-off .shell { grid-template-columns:1fr; }
  aside.side { position:fixed; top:0; left:0; bottom:0; width:min(320px,86vw);
    z-index:50; transform:translateX(-103%); transition:transform .2s ease;
    box-shadow:none; }
  aside.side.open { transform:none; box-shadow:var(--shadow); }
  body.side-off .side { border-right:1px solid var(--line); }
  .top-links { display:none; position:absolute; top:52px; right:10px;
    background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:6px; flex-direction:column; align-items:stretch;
    box-shadow:var(--shadow); margin:0; z-index:60; }
  .top-links.open { display:flex; }
  nav.links { display:none; position:fixed; top:52px; right:10px;
    background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:6px; flex-direction:column; align-items:stretch;
    box-shadow:var(--shadow); margin:0; z-index:60; }
  nav.links.open { display:flex; }
  main.page-main { padding:18px 14px 8px; }
  .thread { padding:10px 14px 6px; }
  .bub-u { max-width:88%; }
  .dock { padding:4px 10px 6px; }
  .hero h1 { font-size:23px; }
  .chips { grid-template-columns:1fr; max-width:440px; }
  h1 { font-size:21px; }
  .card { padding:16px; }
  header.top .iconbtn.menu { display:grid; }
}
@media (min-width: 861px) {
  header.top .iconbtn.menu { display:none; }
}
@media (prefers-reduced-motion: reduce) {
  .cursor, .tdots span, .sbtn.rec { animation:none; }
  .chat-scroll { scroll-behavior:auto; }
  aside.side { transition:none; }
}
"""


#: Tiny boot script, inlined in <head> before first paint so the theme never
#: flashes: stored choice wins, otherwise follow the OS.
THEME_BOOT = """(function(){
try{
  var t=null;
  try{t=localStorage.getItem("realai.theme");}catch(e){}
  if(t!=="dark"&&t!=="light"){
    t=(window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)")
      .matches)?"dark":"light";
  }
  document.documentElement.setAttribute("data-theme",t);
}catch(e){document.documentElement.setAttribute("data-theme","light");}
})();"""


# ------------------------------------------------------------- page scaffolds

def _head(name: str, title: str, here: str = "",
          description: str = "") -> str:
    """Shared ``<head>`` + sticky header for the document pages
    (``/developers``, ``/approve``). ``here`` marks the active link."""
    def link(href: str, label: str, key: str) -> str:
        cls = ' class="here"' if key == here else ""
        return f'<a href="{href}"{cls}>{esc(label)}</a>'

    meta = (f'<meta name="description" content="{esc(description)}">'
            if description else "")
    return f"""<!doctype html>
<html lang="en" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
{meta}
<link rel="icon" type="image/svg+xml" href="/logo.svg">
<link rel="alternate icon" href="/favicon.ico">
<meta name="theme-color" content="#ffffff">
<script>{THEME_BOOT}</script>
<style>{_CSS}</style></head><body class="page">
<header class="top"><div class="top-in">
 <a class="brand" href="/" aria-label="{esc(name)} home">
   {_logo_mark(name)}<span>{esc(name)}</span>
 </a>
 <button class="iconbtn menu" id="burger" aria-label="menu"
   aria-expanded="false">{IC_BURGER}</button>
 <nav class="links" id="links">
   {link("/", "Chat", "chat")}
   {link("/developers", "Developers", "developers")}
   {link("/dashboard", "Telemetry", "dashboard")}
   {link("/approve", "Approvals", "approve")}
   {link("/api.json", "API", "api")}
   <button class="iconbtn" data-themebtn type="button"
     title="toggle theme">{IC_MOON}</button>
 </nav>
</div></header>"""


def _chat_head(name: str, title: str, description: str) -> str:
    """Head + shell start for the chat app (the sidebar lives in the shell,
    not in a site header — the way the real apps do it)."""
    return f"""<!doctype html>
<html lang="en" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="icon" type="image/svg+xml" href="/logo.svg">
<link rel="alternate icon" href="/favicon.ico">
<meta name="theme-color" content="#ffffff">
<script>{THEME_BOOT}</script>
<style>{_CSS}</style></head><body class="chat">
<div class="shell" id="shell">
<aside class="side" id="sidebar">
 <div class="side-top">
  <a class="brand" href="/" aria-label="{esc(name)} home">
    {_logo_mark(name)}<span>{esc(name)}</span>
  </a>
  <div class="side-actions">
   <button class="iconbtn" data-themebtn type="button"
     title="toggle theme">{IC_MOON}</button>
   <button class="iconbtn" id="sidetoggle" type="button"
     title="hide sidebar">{IC_CHEV_L}</button>
  </div>
 </div>
 <div class="side-new">
  <button class="newchat" id="newchat" type="button">{IC_PLUS}New chat</button>
 </div>
 <div class="side-search">{IC_SEARCH}<input id="sessfilter" type="text"
   placeholder="Search chats" autocomplete="off"
   aria-label="search chats"></div>
 <div class="side-sess" id="sessions"></div>
 <div class="side-foot">
  <div class="side-user">
   <span class="su-av">{MARK_DECORATIVE}</span>
   <span style="min-width:0">
    <span class="su-name">{esc(name)}
     <span class="live off" id="livedot"></span></span>
    <span class="su-sub" id="agentstatus">checking…</span>
   </span>
  </div>
  <nav class="side-links">
   <a href="/developers">{IC_CODE}Developers</a>
   <a href="/dashboard">{IC_PULSE}Live telemetry</a>
   <a href="/approve">{IC_USER}Approvals</a>
   <a href="/api.json">{IC_API}API index</a>
  </nav>
 </div>
</aside>
<main class="main">
 <header class="topbar">
  <button class="iconbtn" id="burger" type="button" aria-label="menu"
    aria-expanded="false">{IC_BURGER}</button>
  <nav class="top-links" id="links">
   <a href="/" class="here">Chat</a>
   <a href="/developers">Developers</a>
   <a href="/dashboard">Telemetry</a>
   <a href="/approve">Approvals</a>
   <a href="/api.json">API</a>
  </nav>
 </header>
 <div class="chat-scroll" id="scroller">
  <div class="chat-body">"""


_FOOTER = Template("""
<footer class="page-foot">$agent · owner $owner · mood $mood ·
$grand events counted · pure Python stdlib · no external AI, no external
keys · MIT</footer>
""")

#: Shared chrome script: theme boot/toggle, burger (drawer on mobile,
#: sidebar collapse on desktop), scrim, copy-to-clipboard. Plain string —
#: the icons are spliced in with ``@@...@@`` so no f-string braces.
_CHROME_JS = """
<script>
(function(){
  var root=document.documentElement;
  var meta=document.querySelector('meta[name="theme-color"]');
  var cur=root.getAttribute("data-theme")||"light";
  var ICON_SUN="@@SUN@@", ICON_MOON="@@MOON@@", ICON_CHECK="@@CHECK@@";
  function paint(){
    if(meta) meta.setAttribute("content",cur==="dark"?"#171717":"#ffffff");
  }
  function syncIcons(){
    var btns=document.querySelectorAll("[data-themebtn]");
    Array.prototype.forEach.call(btns,function(b){
      b.innerHTML=cur==="dark"?ICON_SUN:ICON_MOON;
      b.title=cur==="dark"?"switch to light theme":"switch to dark theme";
    });
  }
  window.realaiTheme={
    get:function(){return cur;},
    set:function(t){
      cur=(t==="dark")?"dark":"light";
      root.setAttribute("data-theme",cur);
      try{localStorage.setItem("realai.theme",cur);}catch(e){}
      paint();syncIcons();
    }
  };
  document.addEventListener("click",function(e){
    var t=e.target;
    var b=t&&t.closest?t.closest("[data-themebtn]"):null;
    if(b) window.realaiTheme.set(cur==="dark"?"light":"dark");
    var s=t&&t.closest?t.closest("#scrim"):null;
    if(s) closeAll();
  });
  paint();syncIcons();
  /* ---- chrome: burger / drawer / sidebar collapse ---- */
  var burger=document.getElementById("burger");
  var links=document.getElementById("links");
  var side=document.getElementById("sidebar");
  var scrim=document.getElementById("scrim");
  function isMobile(){
    return window.matchMedia("(max-width:860px)").matches;
  }
  function closeAll(){
    if(links) links.classList.remove("open");
    if(side) side.classList.remove("open");
    if(scrim) scrim.classList.remove("on");
    if(burger) burger.setAttribute("aria-expanded","false");
  }
  if(burger) burger.addEventListener("click",function(){
    if(isMobile()){
      var anyOpen=(links&&links.classList.contains("open"))||
                  (side&&side.classList.contains("open"));
      closeAll();
      if(!anyOpen){
        if(side){
          side.classList.add("open");
          if(scrim) scrim.classList.add("on");
        } else if(links){
          links.classList.add("open");
        }
        burger.setAttribute("aria-expanded","true");
      }
      return;
    }
    if(side){
      var off=document.body.classList.toggle("side-off");
      try{localStorage.setItem("realai.side",off?"off":"on");}catch(e){}
      burger.setAttribute("aria-expanded",off?"false":"true");
    } else if(links){
      links.classList.toggle("open");
    }
  });
  var st=document.getElementById("sidetoggle");
  if(st) st.addEventListener("click",function(){
    document.body.classList.add("side-off");
    try{localStorage.setItem("realai.side","off");}catch(e){}
  });
  document.addEventListener("keydown",function(e){
    if(e.key==="Escape") closeAll();
  });
  window.addEventListener("resize",function(){
    if(!isMobile()&&side) side.classList.remove("open");
    if(!isMobile()&&scrim) scrim.classList.remove("on");
  });
  /* restore desktop sidebar preference (no flash: it is a layout choice) */
  try{
    if(!isMobile()&&side&&localStorage.getItem("realai.side")==="off"){
      document.body.classList.add("side-off");
    }
  }catch(e){}
  /* ---- copy helper (works for text and icon buttons) ---- */
  window.realaiCopy=function(text,btn){
    function done(){
      if(!btn||btn.dataset.copied==="1") return;
      btn.dataset.copied="1";
      var orig=btn.innerHTML;
      if(btn.classList.contains("actb")||btn.classList.contains("iconbtn")){
        btn.innerHTML=ICON_CHECK;
        btn.style.color="var(--ok)";
      }else{
        btn.innerHTML="copied";
      }
      setTimeout(function(){
        btn.innerHTML=orig;
        btn.style.color="";
        delete btn.dataset.copied;
      },1300);
    }
    if(navigator.clipboard&&navigator.clipboard.writeText){
      navigator.clipboard.writeText(text).then(done,done);
    }else{
      var ta=document.createElement("textarea");
      ta.value=text;document.body.appendChild(ta);ta.select();
      try{document.execCommand("copy");}catch(e){}
      document.body.removeChild(ta);done();
    }
  };
})();
</script>"""


def _js_str(s: str) -> str:
    """Make a string safe inside a JS double-quoted literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _chrome_js() -> str:
    return (_CHROME_JS.replace("@@SUN@@", _js_str(IC_SUN))
                     .replace("@@MOON@@", _js_str(IC_MOON))
                     .replace("@@CHECK@@", _js_str(IC_CHECK)))


def _footer(agent: Any) -> str:
    snap = agent.mind.snapshot()
    grand = agent.storage.get_counters().get("grand_total", 0)
    return _FOOTER.substitute(
        agent=esc(snap["agent"]), owner=esc(snap["owner"]),
        mood=esc(snap["mood"]["note"]), grand=grand)


# --------------------------------------------------------------- the chat app

#: ``string.Template``: only ``$agent`` and ``$tagline`` are substituted.
#: Icons and the mark are spliced into the template string at import time.
_CHAT_BODY = Template((
"""
<div class="hero" id="hero">
 <svg viewBox="0 0 64 64" class="mark" width="46" height="46"
   aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
  <path class="mr" d="M50.91 25.47A20 20 0 1 1 31.65 12" fill="none"
   stroke="currentColor" stroke-width="7" stroke-linecap="round"/>
  <circle class="md" cx="43.47" cy="15.62" r="5"/></svg>
 <h1>What can I help with?</h1>
 <p class="sub">$tagline</p>
 <div class="chips">
  <button type="button" class="chip" data-msg="what can you do?">
   <span class="ci">__IC_BUBBLE__</span>
   <span class="ct"><b>What can you do?</b><span class="d">the honest version</span></span>
  </button>
  <button type="button" class="chip" data-msg="status">
   <span class="ci">__IC_PULSE__</span>
   <span class="ct"><b>Show your status</b><span class="d">mood, drives, goals, memory</span></span>
  </button>
  <button type="button" class="chip" data-msg="draw me a lighthouse at dawn">
   <span class="ci">__IC_IMG__</span>
   <span class="ct"><b>Draw me a lighthouse at dawn</b><span class="d">I'll generate the image</span></span>
  </button>
  <button type="button" class="chip" data-msg="make a presentation about the history of AI">
   <span class="ci">__IC_SLIDES__</span>
   <span class="ct"><b>Make a presentation</b><span class="d">a real .pptx, with cover slides</span></span>
  </button>
 </div>
</div>
<div class="thread" id="log"></div>
<div class="msg typing" id="typing">
 <div class="av">__MARK__</div>
 <div class="flow"><div class="tdots"><span></span><span></span>
  <span></span></div></div>
</div>
</div>
</div>
<div class="dock"><div class="dock-in">
<form class="composer" id="composer">
 <button type="button" class="sbtn" id="mic" style="display:none"
   title="speak — I'll transcribe it" aria-label="record a voice
 message">__IC_MIC__</button>
 <textarea id="input" rows="1" autocomplete="off"
   placeholder="talk to it — no key needed · Enter sends, Shift+Enter for a
 new line" aria-label="message to $agent"></textarea>
 <button type="submit" class="sbtn send" id="send" title="send"
   aria-label="send">__IC_SEND__</button>
</form>
<div class="fine">No key, no signup · every request is counted · $agent can
 make mistakes — check important answers</div>
</div></div>
</main>
<div id="scrim"></div>
</div>
"""
              .replace("__MARK__", MARK_DECORATIVE)
              .replace("__IC_BUBBLE__", IC_BUBBLE)
              .replace("__IC_PULSE__", IC_PULSE)
              .replace("__IC_IMG__", IC_IMG)
              .replace("__IC_SLIDES__", IC_SLIDES)
              .replace("__IC_MIC__", IC_MIC)
              .replace("__IC_SEND__", IC_SEND)))


#: Plain (non-Template) so JS regexes and ``$1`` replacements stay literal.
#: ``@@AGENT@@`` / ``@@MARK@@`` / ``@@COPY@@`` / ``@@SPEAK@@`` / ``@@DL@@``
#: are substituted by :func:`chat_page`.
_CHAT_JS = """
<script>
var AGENT = "@@AGENT@@";
var MARK = "@@MARK@@";
var STORE = "realai.sessions.v1";
var log = document.getElementById("log");
var hero = document.getElementById("hero");
var input = document.getElementById("input");
var send = document.getElementById("send");
var typing = document.getElementById("typing");
var sessionsBox = document.getElementById("sessions");
var sidebar = document.getElementById("sidebar");
var scrim = document.getElementById("scrim");
var filterInput = document.getElementById("sessfilter");
var current = null;
var isBusy = false;
var CAPS = { talk:false, images:false, listen:false, speak:false };

function esc(s){
  return String(s).replace(/[&<>"']/g, function(ch){
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch];
  });
}
/* --- a tiny markdown renderer: escape first, then format --- */
function md(src){
  var lines = String(src == null ? "" : src).split("\\n");
  var out = [], para = [], list = null, code = false;
  function inline(t){
    t = t.replace(/!\\[([^\\]]*)\\]\\((\\/media\\/[\\w.-]+)\\)/g,
                  '<img class="inl" src="$2" alt="$1">');
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>")
         .replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>")
         .replace(/(^|[^*])\\*([^*\\n]+)\\*/g, "$1<em>$2</em>");
    return t.replace(/\\[([^\\]]+)\\]\\(([^)\\s]+)\\)/g, function(m, a, b){
      return (/^(https?:\\/\\/|\\/|#)/.test(b))
        ? ('<a href="' + b + '">' + a + '</a>') : (a + " (" + b + ")");
    });
  }
  function flushPara(){
    if (para.length){
      out.push("<p>" + para.map(function(p){ return inline(esc(p)); })
        .join("<br>") + "</p>");
      para = [];
    }
  }
  function flushList(){ if (list){ out.push("</" + list + ">"); list = null; } }
  for (var i = 0; i < lines.length; i++){
    var ln = lines[i];
    if (/^```/.test(ln)){
      if (code){ out.push("</code></pre>"); code = false; }
      else { flushPara(); flushList(); out.push("<pre><code>"); code = true; }
      continue;
    }
    if (code){ out.push(esc(ln) + "\\n"); continue; }
    var h = /^(#{1,6})\\s+(.*)$/.exec(ln);
    var ul = /^\\s*[-*+]\\s+(.*)$/.exec(ln);
    var ol = /^\\s*\\d+[.)]\\s+(.*)$/.exec(ln);
    if (!ln.trim()){ flushPara(); flushList(); continue; }
    if (h){
      flushPara(); flushList();
      var lv = Math.min(h[1].length + 2, 6);
      out.push("<h" + lv + ">" + inline(esc(h[2])) + "</h" + lv + ">");
      continue;
    }
    if (ul){
      flushPara();
      if (list !== "ul"){ flushList(); out.push("<ul>"); list = "ul"; }
      out.push("<li>" + inline(esc(ul[1])) + "</li>");
      continue;
    }
    if (ol){
      flushPara();
      if (list !== "ol"){ flushList(); out.push("<ol>"); list = "ol"; }
      out.push("<li>" + inline(esc(ol[1])) + "</li>");
      continue;
    }
    flushList(); para.push(ln);
  }
  if (code) out.push("</code></pre>");
  flushPara(); flushList();
  return out.join("");
}
/* --- sessions in localStorage --- */
function load(){
  try { return JSON.parse(localStorage.getItem(STORE) || "[]") || []; }
  catch (e) { return []; }
}
function save(list){
  try { localStorage.setItem(STORE, JSON.stringify(list.slice(0, 40))); }
  catch (e) {}
}
function title(s){
  for (var i = 0; i < (s.messages || []).length; i++){
    if (s.messages[i].role === "user")
      return (s.messages[i].text || "").slice(0, 34) || "chat";
  }
  return "new chat";
}
function find(id){
  var list = load();
  for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
  return null;
}
function upsert(s){
  var list = load(), hit = false;
  for (var i = 0; i < list.length; i++)
    if (list[i].id === s.id){ list[i] = s; hit = true; }
  if (!hit) list.unshift(s);
  list.sort(function(a, b){ return b.ts - a.ts; });
  save(list);
}
function refreshHero(){
  var has = current && current.messages && current.messages.length > 0;
  hero.style.display = has ? "none" : "";
}
/* --- rendering --- */
function attach(where, atts){
  if (!atts || !atts.length) return;
  var box = where.querySelector(".att");
  if (!box){
    box = document.createElement("div");
    box.className = "att";
    where.appendChild(box);
  }
  atts.forEach(function(a){
    if (!a || !a.url || !/^\\/media\\//.test(a.url)) return;
    if (a.kind === "image"){
      var img = document.createElement("img");
      img.src = a.url; img.alt = "generated image"; img.loading = "lazy";
      var lnk = document.createElement("a"); lnk.href = a.url;
      lnk.target = "_blank";
      lnk.appendChild(img); box.appendChild(lnk);
    } else if (a.kind === "audio"){
      var au = document.createElement("audio");
      au.controls = true; au.src = a.url; au.autoplay = !!a.autoplay;
      box.appendChild(au);
    } else {
      var dl = document.createElement("a");
      dl.href = a.url; dl.className = "filedl";
      dl.setAttribute("download", a.name || "");
      dl.innerHTML = "@@DL@@" + '<span class="fn">'
        + esc(a.name || "download") + "</span>";
      box.appendChild(dl);
    }
  });
}
function copyBtn(target, text){
  var b = document.createElement("button");
  b.type = "button"; b.className = "actb"; b.title = "copy";
  b.innerHTML = "@@COPY@@";
  b.onclick = function(){ window.realaiCopy(text, b); };
  return b;
}
function speakBtn(target, text){
  var b = document.createElement("button");
  b.type = "button"; b.className = "actb"; b.title = "read aloud";
  b.innerHTML = "@@SPEAK@@";
  b.onclick = function(){
    b.disabled = true; b.style.opacity = ".5";
    fetch("/public/speak", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }) })
    .then(function(r){ return r.json(); })
    .then(function(j){
      b.disabled = false; b.style.opacity = "";
      if (j.url) attach(target, [{ kind: "audio", url: j.url,
                                   autoplay: true }]);
    }).catch(function(){ b.disabled = false; b.style.opacity = ""; });
  };
  return b;
}
function bubble(role, text, animate, atts){
  if (role === "user"){
    var wrap = document.createElement("div");
    wrap.className = "msg me";
    var flow = document.createElement("div");
    flow.className = "flow";
    var bub = document.createElement("div");
    bub.className = "bub-u";
    bub.innerHTML = md(text);
    flow.appendChild(bub); wrap.appendChild(flow);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return bub;
  }
  var w = document.createElement("div");
  w.className = "msg ai";
  var av = document.createElement("div");
  av.className = "av"; av.innerHTML = MARK;
  var f = document.createElement("div");
  f.className = "flow";
  var b = document.createElement("div");
  b.className = "bub-a";
  var acts = document.createElement("div");
  acts.className = "acts";
  acts.appendChild(copyBtn(f, text));
  f.appendChild(b); f.appendChild(acts);
  w.appendChild(av); w.appendChild(f);
  log.appendChild(w);
  function after(){
    attach(f, atts);
    if (CAPS.speak) acts.appendChild(speakBtn(f, text));
    log.scrollTop = log.scrollHeight;
  }
  if (!animate){ b.innerHTML = md(text); after(); }
  else reveal(b, text, after);
  log.scrollTop = log.scrollHeight;
  return b;
}
/* streaming reveal: words land progressively, click to skip */
function reveal(el, text, onDone){
  var parts = String(text).split(/(\\s+)/);
  var i = 0, done = false, timer = null;
  function finish(){
    if (done) return;
    done = true;
    if (timer) clearTimeout(timer);
    el.innerHTML = md(text);
    log.scrollTop = log.scrollHeight;
    if (onDone) onDone();
  }
  el.addEventListener("click", finish);
  (function step(){
    if (done) return;
    i = Math.min(parts.length, i + Math.max(2, Math.round(parts.length / 90)));
    el.innerHTML = md(parts.slice(0, i).join("")) + '<span class="cursor">'
      + "&nbsp;</span>";
    log.scrollTop = log.scrollHeight;
    if (i >= parts.length){ finish(); return; }
    timer = setTimeout(step, 18);
  })();
}
function renderSessions(){
  var list = load();
  var q = (filterInput.value || "").trim().toLowerCase();
  var shown = 0;
  sessionsBox.innerHTML = "";
  var label = document.createElement("div");
  label.className = "slabel";
  label.textContent = "Chats";
  sessionsBox.appendChild(label);
  list.forEach(function(s){
    var t = title(s);
    if (q && t.toLowerCase().indexOf(q) === -1) return;
    shown++;
    var row = document.createElement("div");
    row.className = "sess" + (current && current.id === s.id ? " on" : "");
    var el = document.createElement("div");
    el.className = "t";
    el.textContent = t;
    el.title = ((s.messages || []).length) + " messages";
    var x = document.createElement("button");
    x.className = "x";
    x.type = "button";
    x.innerHTML = "\\u00d7";
    x.setAttribute("aria-label", "delete conversation");
    x.addEventListener("click", function(ev){
      ev.stopPropagation();
      save(load().filter(function(o){ return o.id !== s.id; }));
      if (current && current.id === s.id){ current = null; newChat(); }
      else renderSessions();
    });
    row.appendChild(el); row.appendChild(x);
    row.addEventListener("click", function(){ openSession(s.id); });
    sessionsBox.appendChild(row);
  });
  if (!shown){
    var e = document.createElement("div");
    e.className = "sess-hint";
    e.textContent = q ? "no chats match \\u201C" + q + "\\u201D"
                      : "no conversations yet";
    sessionsBox.appendChild(e);
  }
}
function paint(s){
  log.innerHTML = "";
  (s.messages || []).forEach(function(m){
    bubble(m.role, m.text, false, m.atts);
  });
  refreshHero();
  log.scrollTop = log.scrollHeight;
}
function newChat(){
  current = { id: "local-" + Date.now().toString(36) + "-"
    + Math.random().toString(36).slice(2, 8),
    session_id: null, ts: Date.now(), messages: [] };
  upsert(current); paint(current); renderSessions();
  refreshHero(); input.focus();
}
function openSession(id){
  var s = find(id);
  if (!s){ newChat(); return; }
  current = s; paint(s); renderSessions();
  if (sidebar) sidebar.classList.remove("open");
  if (scrim) scrim.classList.remove("on");
}
/* --- sending --- */
function autosize(){
  input.style.height = "auto";
  input.style.height = Math.min(200, input.scrollHeight) + "px";
}
function refreshSend(){
  send.disabled = isBusy || !(input.value || "").trim();
}
function setBusy(on){
  isBusy = on;
  refreshSend();
  typing.classList.toggle("on", on);
}
function sendMsg(){
  var text = (input.value || "").trim();
  if (!text || isBusy) return;
  if (!current) newChat();
  current.messages.push({ role: "user", text: text, ts: Date.now() });
  current.ts = Date.now();
  upsert(current);
  bubble("user", text, false);
  renderSessions();
  refreshHero();
  input.value = ""; autosize();
  setBusy(true);
  var mine = current;
  /* POST /public/chat — keyless, rate-limited per visitor, no scopes */
  fetch("/public/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text, session_id: mine.session_id })
  }).then(function(r){
    return r.json().then(function(j){ return [r.status, j]; });
  }).then(function(res){
    var status = res[0], j = res[1] || {};
    setBusy(false);
    var reply = j.response
      || (j.error && j.error.message)
      || "I could not answer that — the request came back empty.";
    if (status === 429) reply = "Slow down a little: the keyless demo is "
      + "rate-limited per visitor. Try again in a few seconds.";
    if (j.session_id) mine.session_id = j.session_id;
    var atts = (j.meta && j.meta.attachments) || [];
    mine.messages.push({ role: "ai", text: reply, ts: Date.now(),
                         atts: atts });
    mine.ts = Date.now();
    upsert(mine);
    bubble("ai", reply, true, atts);
    renderSessions();
    input.focus();
  }).catch(function(){
    setBusy(false);
    var err = "network error — nothing reached me. Try again.";
    mine.messages.push({ role: "ai", text: err, ts: Date.now() });
    upsert(mine);
    bubble("ai", err, true);
  });
}
document.getElementById("composer").addEventListener("submit", function(e){
  e.preventDefault(); sendMsg();
});
document.getElementById("newchat").addEventListener("click", newChat);
input.addEventListener("keydown", function(e){
  if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); sendMsg(); }
});
input.addEventListener("input", function(){
  autosize(); refreshSend();
});
if (filterInput) filterInput.addEventListener("input", renderSessions);
/* suggestion chips fill the composer (and focus it) */
Array.prototype.forEach.call(document.querySelectorAll(".chip"),
function(ch){
  ch.addEventListener("click", function(){
    input.value = ch.getAttribute("data-msg") || "";
    autosize(); refreshSend(); input.focus();
  });
});
/* --- live status line (mood + autonomy), polled, cheap --- */
function statusTick(){
  fetch("/healthz").then(function(r){ return r.json(); })
    .then(function(j){
      var d = document.getElementById("agentstatus");
      var dot = document.getElementById("livedot");
      if (d) d.textContent = (j.mood || "online") + " · "
        + (j.autonomy ? "thinks on its own" : "autonomy paused");
      if (dot) dot.className = "live" + (j.autonomy ? "" : " off");
    }).catch(function(){});
}
statusTick();
setInterval(statusTick, 45000);
/* --- generative abilities: discovered at load, all optional --- */
var micBtn = document.getElementById("mic");
fetch("/v1/generate/capabilities").then(function(r){ return r.json(); })
.then(function(j){
  var c = (j && j.capabilities) || {};
  Object.keys(CAPS).forEach(function(k){ CAPS[k] = !!(c[k] && c[k].enabled); });
  if (CAPS.listen && navigator.mediaDevices && window.MediaRecorder)
    micBtn.style.display = "grid";
  var hints = [];
  if (CAPS.talk) hints.push("talk about anything");
  if (CAPS.images) hints.push("\\u201Cdraw me a …\\u201D");
  hints.push("\\u201Cmake a presentation about …\\u201D");
  if (CAPS.speak) hints.push("\\u{1F50A} read aloud");
  input.placeholder = hints.join(" · ") + " · Enter sends";
}).catch(function(){});
var rec = null, chunks = [];
function stopRec(){
  if (rec && rec.state !== "inactive") rec.stop();
  micBtn.classList.remove("rec");
}
micBtn.addEventListener("click", function(){
  if (rec && rec.state === "recording"){ stopRec(); return; }
  navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream){
    chunks = [];
    rec = new MediaRecorder(stream);
    rec.ondataavailable = function(e){ if (e.data.size) chunks.push(e.data); };
    rec.onstop = function(){
      stream.getTracks().forEach(function(t){ t.stop(); });
      var blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
      var fr = new FileReader();
      fr.onload = function(){
        setBusy(true);
        fetch("/public/transcribe", { method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ audio: fr.result, mime: blob.type }) })
        .then(function(r){ return r.json(); })
        .then(function(j){
          setBusy(false);
          if (j.text){ input.value = j.text; autosize(); refreshSend();
            sendMsg(); }
          else {
            var f = bubble("ai", "I could not hear that"
              + (j.error ? ": " + j.error.message : "."), true);
          }
        }).catch(function(){ setBusy(false); });
      };
      fr.readAsDataURL(blob);
    };
    rec.start();
    micBtn.classList.add("rec");
    setTimeout(stopRec, 30000);
  }).catch(function(){
    bubble("ai", "The browser did not allow microphone access.", true);
  });
});
(function boot(){
  var list = load();
  if (list.length) openSession(list[0].id); else newChat();
  renderSessions();
  refreshSend();
})();
</script>
</body></html>"""


def chat_page(agent: Any, tagline: Optional[str] = None) -> str:
    """The chat app served at ``/``."""
    snap = agent.mind.snapshot()
    cfg = agent.cfg
    name = snap["agent"]
    body = _CHAT_BODY.substitute(
        agent=esc(name),
        tagline=esc(tagline or cfg.brand_tagline))
    return (_chat_head(name, f"{name} — your own AI, live",
                       f"Chat with {name}: talk, draw images, build "
                       f"presentations. No key, no signup.")
            + body + _chrome_js()
            + _CHAT_JS.replace("@@AGENT@@", esc(name))
                      .replace("@@MARK@@", _js_str(MARK_DECORATIVE))
                      .replace("@@COPY@@", _js_str(IC_COPY))
                      .replace("@@SPEAK@@", _js_str(IC_SPEAK))
                      .replace("@@DL@@", _js_str(IC_DL)))


# --------------------------------------------------------- developer portal

_DEV_BODY = Template("""
<main class="page-main">
<h1>Developers</h1>
<p class="tag">Every endpoint below is read live from the agent's own route
 registry — the same data <a href="/api.json">/api.json</a> serves — so this
 table cannot drift from the code. The public pages need nothing at all; the
 <code>/v1/*</code> API needs a key that only the owner can issue.</p>

<div class="card">
 <div class="k">ask the owner for a key</div>
 <form id="reqform">
  <div class="grid">
   <div class="field"><label for="username">your handle (1-48 chars)</label>
    <input id="username" name="username" maxlength="48" required
     placeholder="e.g. mira" autocomplete="username"></div>
   <div class="field"><label for="note">what will you build? (optional)</label>
    <input id="note" name="note" maxlength="200"
     placeholder="a small dashboard for my workshop"></div>
  </div>
  <div class="rowbtns">
   <button class="btn btn-p" type="submit" id="reqsend">Request access</button>
   <span class="hint" style="align-self:center">You start PENDING. The owner
    hears about it on Telegram <b>and</b> by email, then decides from
    <code>/approve</code>. No key exists until they mint one.</span>
  </div>
 </form>
 <div id="reqout"></div>
</div>

<div class="card">
 <div class="k">authenticating</div>
 <p class="dim" style="margin-top:0;font-size:13.5px">Gated routes take an
  owner-issued key as a bearer token. A key passes when it holds at least one
  of the route's scopes; <code>owner</code> implies everything. Requests are
  counted even when they fail.</p>
 <pre><code>curl -s https://$host/v1/chat \\
  -H "Authorization: Bearer rxa_..." \\
  -H "Content-Type: application/json" \\
  -d '{"message": "status"}'</code></pre>
 <div class="rowbtns">
  <button class="btn btn-g" id="copycurl" type="button">copy curl</button>
  <a class="btn btn-g" href="/api.json">raw index (JSON)</a>
  <a class="btn btn-g" href="/dashboard">live telemetry</a>
 </div>
</div>

<div class="card">
 <div class="k">live endpoint table · $count routes</div>
 <div style="overflow-x:auto">
 <table id="endpoints">
  <thead><tr><th>method</th><th>path</th><th>access</th>
   <th>what it does</th></tr></thead>
  <tbody>$rows</tbody>
 </table>
 </div>
 <div class="hint" id="livehint">rendered server-side; checking against
  <code>/api.json</code>…</div>
</div>

<div class="card">
 <div class="k">limits &amp; errors</div>
 <div class="grid">
  <div class="tile"><h3>Rate limits</h3><p>Per key: $rate/min with a burst of
   $burst (token bucket). Keyless demo chat: $drate/min per visitor, burst
   $dburst — one rude guest cannot exhaust the demo for everyone. Bodies over
   $maxbody bytes are rejected with 413.</p></div>
  <div class="tile"><h3>Billing</h3><p>Approved clients are charged $cost per
   request unless the owner marks them VIP (free). A per-key request limit
   returns 429; an empty balance returns 402.</p></div>
  <div class="tile"><h3>Everything counted</h3><p>Every request — including
   4xx and 5xx — lands in a durable ledger with its latency. Read it at
   <code>/v1/counters</code> or watch it live at
   <a href="/dashboard">/dashboard</a>.</p></div>
 </div>
 <h2 style="margin-top:18px;font-size:15px;letter-spacing:-.01em">status
  codes</h2>
 <table><tbody>$errors</tbody></table>
</div>
</main>""")

#: Plain string: ``@@HOST@@`` is substituted by :func:`developers_page`.
_DEV_JS = """
<script>
(function(){
  var form = document.getElementById("reqform");
  var out = document.getElementById("reqout");
  var btn = document.getElementById("reqsend");
  form.addEventListener("submit", function(e){
    e.preventDefault();
    var username = document.getElementById("username").value.trim();
    var note = document.getElementById("note").value.trim();
    if (!username){ out.innerHTML = ""; return; }
    btn.disabled = true;
    fetch("/v1/users/request", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: username, note: note })
    }).then(function(r){ return r.json(); }).then(function(j){
      btn.disabled = false;
      if (j && j.status === "PENDING"){
        out.innerHTML = '<div class="note good"><b>Request received.</b> You '
          + 'are PENDING. The owner just heard about it on Telegram and by '
          + 'email; when they approve, a key is minted for you and shown to '
          + 'them once. Nothing works until then.</div>';
        form.reset();
      } else {
        var msg = (j && j.error && j.error.message) || "request failed";
        out.innerHTML = '<div class="note warn">' + msg + '</div>';
      }
    }).catch(function(){
      btn.disabled = false;
      out.innerHTML = '<div class="note warn">network error — the request '
        + 'did not reach the agent.</div>';
    });
  });
  var curl = document.getElementById("copycurl");
  if (curl) curl.addEventListener("click", function(){
    window.realaiCopy('curl -s https://@@HOST@@/v1/chat '
      + '-H "Authorization: Bearer rxa_..." '
      + '-H "Content-Type: application/json" '
      + '-d \\'{"message": "status"}\\'', curl);
  });
  /* hydrate from the live index so the table can never go stale */
  fetch("/api.json").then(function(r){ return r.json(); }).then(function(j){
    var routes = (j && j.routes) || [];
    document.getElementById("livehint").textContent = routes.length
      + " routes confirmed live from /api.json just now.";
  }).catch(function(){
    document.getElementById("livehint").textContent =
      "could not reach /api.json — showing the server-rendered table.";
  });
})();
</script>
</body></html>"""


def _scope_pill(scopes: Sequence[str]) -> str:
    if not scopes or list(scopes) == ["public"]:
        return '<span class="pill pub">public</span>'
    if "owner" in scopes:
        return '<span class="pill own">owner</span>'
    return '<span class="pill">' + esc(", ".join(scopes)) + "</span>"


def endpoint_rows(routes: Iterable[Dict[str, Any]]) -> str:
    """Build the live endpoint table from ``/api.json``-shaped route dicts."""
    rows: List[str] = []
    for route in routes:
        path = str(route.get("path", ""))
        method = str(route.get("method", "GET"))
        scopes = route.get("scopes") or ["public"]
        doc = ENDPOINT_DOCS.get(path)
        if doc is None:
            # parameterised twins ("/v1/goals/{id}") inherit the family's docs
            doc = ENDPOINT_DOCS.get(re.sub(r"/\{[^}]+\}", "", path), "—")
        rows.append(
            f"<tr><td class='m'>{esc(method)}</td>"
            f"<td class='p'>{esc(path)}</td>"
            f"<td>{_scope_pill(scopes)}</td>"
            f"<td class='dim'>{esc(doc)}</td></tr>")
    return "\n".join(rows) or "<tr><td class='dim'>no routes</td></tr>"


def developers_page(agent: Any, routes: Iterable[Dict[str, Any]],
                    host: str = "your-host.example") -> str:
    """The developer portal served at ``/developers``."""
    snap = agent.mind.snapshot()
    cfg = agent.cfg
    routes = list(routes)
    errors = "\n".join(
        f"<tr><td class='m'>{esc(code)}</td><td class='dim'>{esc(why)}</td>"
        f"</tr>" for code, why in ERROR_CODES)
    body = _DEV_BODY.substitute(
        host=esc(host), count=len(routes), rows=endpoint_rows(routes),
        errors=errors, rate=int(cfg.rate_limit_per_min),
        burst=int(cfg.rate_burst),
        drate=int(getattr(cfg, "demo_rate_per_min", 12)),
        dburst=int(getattr(cfg, "demo_burst", 6)),
        maxbody=f"{int(cfg.max_body_bytes):,}",
        cost=f"{float(cfg.request_cost):.2f}")
    return (_head(snap["agent"], f"{snap['agent']} — developers",
                  "developers",
                  "Live endpoint reference and key requests for the RealAI "
                  "API.")
            + body + _footer(agent) + _chrome_js()
            + _DEV_JS.replace("@@HOST@@", esc(host)))


# -------------------------------------------------------- owner approval box

_APPROVE_BODY = Template("""
<main class="page-main">
<h1>Approvals</h1>
<p class="tag">Your inbox. Every client starts <b>PENDING</b> and gets nothing
 until you decide here — or from Telegram with
 <code>/approve &lt;name&gt;</code>. Approving mints a scoped API key and shows
 it to you <b>exactly once</b>: it is stored only as a hash, so it cannot be
 fetched later, and it is never emailed.</p>

$warning
<div id="keybox" class="card">
 <div class="k ok" style="color:var(--ok)">key issued — copy it now, it will
  not be shown again</div>
 <div class="kv" id="keyval"></div>
 <div class="rowbtns" style="margin-top:12px">
  <button class="btn btn-p" id="keycopy" type="button">copy key</button>
  <span class="hint" id="keymeta" style="align-self:center"></span>
 </div>
</div>
<div id="flash"></div>

<div class="card">
 <div class="k">waiting for you · $pending_n</div>
 $pending_rows
</div>

<div class="card">
 <div class="k">approved clients · $approved_n</div>
 $approved_rows
</div>

<div class="card">
 <div class="k">banned · $banned_n</div>
 $banned_rows
</div>

<div class="card">
 <div class="k">what each button does</div>
 <div class="grid">
  <div class="tile"><h3 class="ok">Approve</h3><p>Status → APPROVED and a key
   is minted with scopes <code>$scopes</code>, limited to $limit requests. The
   plaintext appears once, above, and nowhere else.</p></div>
  <div class="tile"><h3 class="bad">Deny</h3><p>Status → BANNED and every key
   issued to them is revoked. A banned key that retries the API is counted as
   a security intrusion.</p></div>
  <div class="tile"><h3 class="warn">VIP</h3><p>Free requests: billing is
   skipped for this client. Toggle it off and per-request charging
   resumes.</p></div>
  <div class="tile"><h3>Unban</h3><p>Status → PENDING. They are back in the
   queue above, but they still need a fresh approval to get a key.</p></div>
 </div>
</div>
</main>""")

#: Plain string: ``@@TOKEN@@`` is substituted by :func:`approve_page`.
_APPROVE_JS = """
<script>
var TOKEN = "@@TOKEN@@";
function flash(htmlText, kind){
  document.getElementById("flash").innerHTML =
    '<div class="note ' + (kind || "") + '">' + htmlText + "</div>";
  window.scrollTo({ top: 0, behavior: "smooth" });
}
function buttonsFor(username){
  return document.querySelectorAll("button[data-u='" + username + "']");
}
function setDisabled(nodes, on){
  Array.prototype.forEach.call(nodes, function(b){ b.disabled = on; });
}
function act(username, action, vip){
  var body = { action: action, username: username };
  if (vip !== undefined && vip !== null) body.vip = vip;
  var btns = buttonsFor(username);
  setDisabled(btns, true);
  fetch("/approve", {
    method: "POST",
    headers: { "Content-Type": "application/json",
               "X-Web-Token": TOKEN || "" },  // secret travels here, never in the body
    body: JSON.stringify(body)
  }).then(function(r){
    return r.json().then(function(j){ return [r.status, j]; });
  }).then(function(res){
    var status = res[0], j = res[1] || {};
    if (status !== 200){
      var msg = (j && j.error && j.error.message) || ("HTTP " + status);
      flash("<b>Could not " + action + " " + username + ".</b> " + msg, "warn");
      setDisabled(btns, false);
      return;
    }
    if (action === "approve" && j.api_key){
      document.getElementById("keyval").textContent = j.api_key;
      document.getElementById("keymeta").textContent = "key id "
        + (j.key_id || "?") + " · scopes "
        + ((j.key_scopes || []).join(", ") || "-") + " · client " + username;
      document.getElementById("keybox").classList.add("on");
      document.getElementById("keycopy").onclick = function(){
        window.realaiCopy(j.api_key, this);
      };
      flash("<b>" + username + " is approved.</b> Their key is above — copy it "
        + "now, this is the only time it is shown.", "good");
    } else {
      flash("<b>" + username + ":</b> " + (j.message || (action + " done."))
        + (j.status ? (" Status is now " + j.status + ".") : ""), "good");
    }
    setTimeout(function(){ window.location.reload(); }, 1000);
  }).catch(function(){
    flash("<b>network error</b> — nothing was changed.", "warn");
    setDisabled(btns, false);
  });
}
</script>
</body></html>"""


def _age(created: Any) -> str:
    if not created:
        return ""
    try:
        import time as _time
        seconds = max(0, int(_time.time() - float(created)))
    except (TypeError, ValueError):
        return ""
    if seconds < 60:
        return f"<span class='dim'>{seconds}s ago</span>"
    if seconds < 3600:
        return f"<span class='dim'>{seconds // 60} min ago</span>"
    if seconds < 86400:
        return f"<span class='dim'>{seconds // 3600} h ago</span>"
    return f"<span class='dim'>{seconds // 86400} d ago</span>"


def _user_row(user: Dict[str, Any], buttons: str) -> str:
    vip = ' <span class="pill own">VIP</span>' if user.get("is_vip") else ""
    balance = float(user.get("balance") or 0.0)
    keys = user.get("keys")
    key_note = ""
    if keys is not None:
        live = [k for k in keys if not k.get("revoked")]
        key_note = f"<span class='dim'>{len(live)} live key(s)</span>"
    return (f"<tr><td class='p'>{esc(user.get('username'))}{vip}</td>"
            f"<td>{_age(user.get('created_at'))}</td>"
            f"<td class='dim'>balance {balance:.2f}</td>"
            f"<td>{key_note}</td>"
            f"<td><div class='rowbtns'>{buttons}</div></td></tr>")


def _button(username: str, action: str, label: str, css: str = "btn btn-g",
            vip: Optional[bool] = None) -> str:
    extra = "" if vip is None else f", {'true' if vip else 'false'}"
    name = esc(username)
    return (f"<button class='{css}' data-u='{name}' "
            f"onclick=\"act('{name}','{esc(action)}'{extra})\" "
            f"type=\"button\">{esc(label)}</button>")


def _user_table(users: List[Dict[str, Any]], buttons_for: Any) -> str:
    head = ("<thead><tr><th>client</th><th>requested</th><th>billing</th>"
            "<th>keys</th><th>decide</th></tr></thead>")
    rows = "\n".join(_user_row(u, buttons_for(u)) for u in users)
    return f"<table>{head}<tbody>{rows}</tbody></table>"


def approve_page(agent: Any, token: str = "",
                 notice: Optional[str] = None) -> str:
    """The owner's approval inbox served at ``/approve`` (web-token gated)."""
    snap = agent.mind.snapshot()
    cfg = agent.cfg
    users = agent.users.list()
    pending = [u for u in users if u.get("status") == "PENDING"]
    approved = [u for u in users if u.get("status") == "APPROVED"]
    banned = [u for u in users if u.get("status") == "BANNED"]

    warning = ""
    if not (cfg.web_token or "").strip():
        warning = ('<div class="note warn"><b>This inbox is not gated.</b> Set '
                   '<code>REALAI_WEB_TOKEN</code> so only you can reach '
                   '<code>/approve</code> — right now anyone with the URL '
                   'could mint a key.</div>')
    if notice:
        warning += f'<div class="note">{esc(notice)}</div>'

    if pending:
        pending_rows = _user_table(
            pending,
            lambda u: _button(u["username"], "approve", "Approve", "btn btn-p")
            + _button(u["username"], "deny", "Deny", "btn btn-d"))
    else:
        pending_rows = ("<p class='dim'>Nobody is waiting. Requests from "
                        "<a href='/developers'>/developers</a> land here "
                        "instantly.</p>")
    if approved:
        approved_rows = _user_table(
            approved,
            lambda u: _button(u["username"], "vip",
                              "Remove VIP" if u.get("is_vip") else "Make VIP",
                              "btn btn-g", vip=not bool(u.get("is_vip")))
            + _button(u["username"], "deny", "Deny", "btn btn-d"))
    else:
        approved_rows = "<p class='dim'>No approved clients yet.</p>"
    banned_rows = _user_table(
        banned, lambda u: _button(u["username"], "unban", "Unban")) \
        if banned else "<p class='dim'>Nobody is banned.</p>"

    body = _APPROVE_BODY.substitute(
        warning=warning, pending_n=len(pending), approved_n=len(approved),
        banned_n=len(banned), pending_rows=pending_rows,
        approved_rows=approved_rows, banned_rows=banned_rows,
        limit=int(cfg.default_request_limit),
        scopes=esc(", ".join(_USER_SCOPES)))
    return (_head(snap["agent"], f"{snap['agent']} — approvals", "approve",
                  "Owner approval inbox: approve, deny, unban and VIP.")
            + body + _footer(agent) + _chrome_js()
            + _APPROVE_JS.replace("@@TOKEN@@", esc(token or "")))


def locked_page(retry: int) -> str:
    """A human-readable 429 for the gated surfaces.

    Wrong web tokens are counted per visitor (see ``web.TokenGuard``); once a
    visitor trips the lockout this page tells them plainly what happened and
    how long to wait, instead of handing back a bare status code.
    """
    retry = max(1, int(retry))
    return (
        "<!doctype html><html lang=\"en\" data-theme=\"light\"><head>"
        "<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>RealAI — too many attempts</title>"
        "<link rel=\"icon\" type=\"image/svg+xml\" href=\"/logo.svg\">"
        "<meta name=\"theme-color\" content=\"#ffffff\">"
        f"<meta http-equiv=\"refresh\" content=\"{retry}\">"
        f"<script>{THEME_BOOT}</script>"
        f"<style>{_CSS}</style></head><body class=\"page\">"
        "<div class=\"gate-wrap\"><div class=\"gate-card\">"
        f"{_logo_mark()}"
        "<h1>Too many wrong web tokens</h1>"
        "<p>This visitor has been locked out of the owner surfaces for about "
        f"<code>{retry}s</code>. The page reloads itself when the lockout "
        "expires.</p>"
        "<p>The attempt was counted in the agent ledger and the owner was "
        "told — this gate protects key minting, so it is deliberately "
        "unforgiving.</p>"
        "<p class=\"gate-back\"><a href=\"/\">← back to the chat app</a></p>"
        "</div></div></body></html>")


# ------------------------------------------------------------------ plumbing

#: Scopes a newly approved client key gets (mirrors ``users._USER_SCOPES``).
_USER_SCOPES = ("chat", "devices", "learn")


def api_routes_payload(agent: Any) -> List[Dict[str, Any]]:
    """The same route list ``/api.json`` serves, without an HTTP round trip."""
    from .api.routes import ROUTES
    return [{"method": method, "path": path, "scopes": scopes or ["public"]}
            for method, path, scopes, _handler in ROUTES]


def request_host(headers: Optional[Dict[str, str]] = None) -> str:
    """Best guess at the public host, for the portal's curl example."""
    headers = headers or {}
    for name in ("x-forwarded-host", "host", "x-forwarded-server"):
        value = str(headers.get(name) or "").strip()
        if value:
            return value.split(",")[0].strip()
    return "your-host.example"


def json_bytes(payload: Any) -> bytes:
    """Small helper so page handlers can return JSON without another import."""
    return json.dumps(payload, default=str).encode("utf-8")
