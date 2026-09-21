"""The product's own pages: branded chat app, developer portal, owner inbox.

Server-rendered HTML from the standard library — ``string.Template`` for
substitution, no template engine, no build step, no CDN, no analytics, no
third-party font or script. Everything the browser gets comes from this
repo, which is the same rule the brain obeys: nothing external.

Three pages make RealAI a product rather than an API with a demo attached:

=============== =========================================================
``/``            the branded chat app — own logo and favicon, session
                 sidebar, markdown, streaming reveal, typing state,
                 mobile layout
``/developers``  the portal — a live endpoint table built from the real
                 route registry (the same data ``/api.json`` serves) plus
                 the key-request form
``/approve``     the owner's approval inbox — web-token gated, one-tap
                 approve (key shown exactly once), deny, unban, VIP
=============== =========================================================

The chat app talks to ``POST /public/chat``: keyless, rate-limited per
visitor, no scopes — so nothing privileged can be reached from a browser.
``/v1/*`` stays behind owner-issued keys.

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
    "/": "this app — the branded chat page (public)",
    "/public/chat": "keyless demo chat: {\"message\": \"...\", "
                    "\"session_id\": \"...\"}",
    "/developers": "the developer portal (public)",
    "/logo.svg": "the wordmark/logo, also used as the favicon",
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


def _logo_mark(agent_name: str = "REAL") -> str:
    """The product's own mark: a local, hand-written SVG (no external asset).

    A rounded square with the agent's initial and a small 'counted' dot —
    the dot is the ledger, which is the thing this project never stops
    talking about.
    """
    initial = (agent_name or "R").strip()[:1].upper() or "R"
    return (
        '<svg viewBox="0 0 64 64" width="34" height="34" role="img" '
        'aria-label="' + esc(agent_name) + ' logo" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#7aa2f7"/><stop offset="1" '
        'stop-color="#3fd68f"/></linearGradient></defs>'
        '<rect x="3" y="3" width="58" height="58" rx="16" fill="#0e1219" '
        'stroke="url(#g)" stroke-width="3"/>'
        '<text x="32" y="42" text-anchor="middle" font-family="ui-sans-serif,'
        'system-ui,sans-serif" font-size="30" font-weight="700" '
        'fill="url(#g)">' + esc(initial) + '</text>'
        '<circle cx="48" cy="17" r="4.5" fill="#3fd68f"/></svg>')


LOGO_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
            'width="64" height="64"><defs><linearGradient id="g" x1="0" '
            'y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7aa2f7"/>'
            '<stop offset="1" stop-color="#3fd68f"/></linearGradient>'
            '</defs><rect x="3" y="3" width="58" height="58" rx="16" '
            'fill="#0e1219" stroke="url(#g)" stroke-width="3"/><text x="32" '
            'y="42" text-anchor="middle" font-family="ui-sans-serif,'
            'system-ui,sans-serif" font-size="30" font-weight="700" '
            'fill="url(#g)">R</text><circle cx="48" cy="17" r="4.5" '
            'fill="#3fd68f"/></svg>')


def favicon_bytes(agent_name: str = "REAL") -> bytes:
    """The favicon payload — an SVG we generate ourselves."""
    return _logo_mark(agent_name).encode("utf-8")


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


# --------------------------------------------------------------------- shell

_CSS = """
 :root { color-scheme: dark;
   --bg:#0b0e14; --panel:#12161f; --panel2:#0e1219; --line:#1f2633;
   --ink:#d7dce5; --dim:#8b94a7; --faint:#5c6577;
   --blue:#7aa2f7; --green:#3fd68f; --amber:#f5c542; --red:#f2777a; }
 * { box-sizing:border-box; }
 body { background:var(--bg); color:var(--ink); margin:0;
   font:15px/1.6 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;
   -webkit-text-size-adjust:100%; }
 a { color:var(--blue); text-decoration:none; }
 a:hover { text-decoration:underline; }
 code, pre, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
 code { background:#1a2030; padding:1px 6px; border-radius:6px;
   font-size:.9em; color:#a8b2c5; }
 pre { background:var(--panel2); border:1px solid var(--line);
   border-radius:10px; padding:12px; overflow-x:auto; font-size:12.5px; }
 pre code { background:none; padding:0; }
 /* ---------- brand header ---------- */
 header.top { position:sticky; top:0; z-index:20; background:rgba(11,14,20,.92);
   backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }
 .topin { max-width:1120px; margin:0 auto; padding:10px 18px; display:flex;
   align-items:center; gap:12px; }
 .brand { display:flex; align-items:center; gap:10px; font-weight:700;
   font-size:17px; letter-spacing:.2px; }
 .brand svg { display:block; }
 .brand .dot { color:var(--green); }
 nav.links { margin-left:auto; display:flex; gap:6px; align-items:center;
   flex-wrap:wrap; }
 nav.links a { color:var(--dim); padding:6px 10px; border-radius:9px;
   font-size:13.5px; border:1px solid transparent; }
 nav.links a:hover { color:var(--ink); border-color:var(--line);
   text-decoration:none; background:var(--panel); }
 nav.links a.here { color:#fff; background:#1d2a44; border-color:var(--blue); }
 button.burger { display:none; margin-left:auto; background:var(--panel);
   color:var(--ink); border:1px solid var(--line); border-radius:9px;
   padding:7px 10px; font-size:15px; cursor:pointer; }
 main { max-width:1120px; margin:0 auto; padding:22px 18px 40px; }
 h1 { font-size:26px; margin:0 0 6px; }
 h2 { font-size:17px; margin:26px 0 10px; }
 .tag { color:var(--dim); margin:0 0 20px; max-width:70ch; }
 .tag b { color:var(--green); }
 .card { background:var(--panel); border:1px solid var(--line);
   border-radius:14px; padding:16px 18px; margin-bottom:16px; }
 .k { color:var(--dim); font-size:11.5px; text-transform:uppercase;
   letter-spacing:.09em; margin-bottom:10px; font-weight:600; }
 .grid { display:grid; gap:12px;
   grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); }
 .tile { background:var(--panel2); border:1px solid var(--line);
   border-radius:11px; padding:12px 13px; }
 .tile h3 { margin:0 0 5px; font-size:14px; }
 .tile p { margin:0; color:var(--dim); font-size:12.8px; }
 .ok { color:var(--green); } .warn { color:var(--amber); }
 .bad { color:var(--red); } .dim { color:var(--dim); }
 table { width:100%; border-collapse:collapse; font-size:13px; }
 th { text-align:left; color:var(--dim); font-size:11.5px;
   text-transform:uppercase; letter-spacing:.07em; padding:6px 9px;
   border-bottom:1px solid var(--line); }
 td { padding:7px 9px; border-top:1px solid var(--line); vertical-align:top; }
 tr:hover td { background:#141926; }
 .m { color:var(--blue); font-family:ui-monospace,monospace;
   white-space:nowrap; font-size:12.5px; }
 .p { color:var(--ink); font-family:ui-monospace,monospace; font-size:12.5px; }
 .pill { display:inline-block; padding:1px 7px; border-radius:999px;
   font-size:11px; border:1px solid var(--line); color:var(--dim);
   background:var(--panel2); }
 .pill.pub { color:var(--green); border-color:#1f4433; }
 .pill.own { color:var(--amber); border-color:#4a3d16; }
 input, textarea, select { background:var(--panel2); color:var(--ink);
   border:1px solid var(--line); border-radius:10px; padding:10px 12px;
   font:inherit; width:100%; }
 input:focus, textarea:focus { outline:none; border-color:#33405a; }
 label { display:block; color:var(--dim); font-size:12px; margin:0 0 5px; }
 .field { margin-bottom:12px; }
 button.act { background:#1d2a44; color:#fff; border:1px solid var(--blue);
   border-radius:10px; padding:9px 15px; font:inherit; font-size:13.5px;
   cursor:pointer; }
 button.act:hover { background:#24345a; }
 button.act:disabled { opacity:.5; cursor:progress; }
 button.ghost { background:var(--panel2); color:var(--ink);
   border:1px solid var(--line); border-radius:10px; padding:8px 13px;
   font:inherit; font-size:13px; cursor:pointer; }
 button.ghost:hover { border-color:#33405a; }
 button.danger { background:#2a1a1c; border-color:#5d2b2e; color:#f2b8ba; }
 button.good { background:#152e24; border-color:#2c6b4c; color:#b9f0d6; }
 .hint { color:var(--faint); font-size:12.3px; margin-top:8px; }
 .note { border-left:3px solid var(--blue); background:var(--panel2);
   padding:10px 13px; border-radius:0 10px 10px 0; color:var(--dim);
   font-size:13px; margin:12px 0; }
 .note.warn { border-color:var(--amber); }
 .note.good { border-color:var(--green); }
 footer { color:var(--faint); font-size:12.5px; margin-top:22px;
   border-top:1px solid var(--line); padding-top:14px; }
 /* ---------- chat layout ---------- */
 .chatwrap { display:grid; grid-template-columns:250px 1fr; gap:16px;
   align-items:start; }
 aside.side { background:var(--panel); border:1px solid var(--line);
   border-radius:14px; padding:12px; position:sticky; top:64px;
   max-height:calc(100vh - 90px); display:flex; flex-direction:column; }
 aside.side .k { margin-bottom:8px; }
 #sessions { overflow-y:auto; flex:1; margin:8px 0; }
 .sess { display:flex; align-items:center; gap:6px; padding:8px 9px;
   border-radius:9px; cursor:pointer; border:1px solid transparent; }
 .sess:hover { background:var(--panel2); border-color:var(--line); }
 .sess.on { background:#1d2a44; border-color:var(--blue); }
 .sess .t { flex:1; overflow:hidden; text-overflow:ellipsis;
   white-space:nowrap; font-size:13px; }
 .sess .x { color:var(--faint); background:none; border:none; cursor:pointer;
   font-size:14px; padding:0 3px; }
 .sess .x:hover { color:var(--red); }
 section.chat { background:var(--panel); border:1px solid var(--line);
   border-radius:14px; overflow:hidden; display:flex; flex-direction:column;
   height:calc(100vh - 150px); min-height:420px; }
 #log { flex:1; overflow-y:auto; padding:18px; background:var(--panel2);
   scroll-behavior:smooth; }
 .msg { margin-bottom:16px; display:flex; gap:10px; }
 .msg .av { flex:0 0 28px; height:28px; border-radius:8px; display:flex;
   align-items:center; justify-content:center; font-size:12px;
   font-weight:700; border:1px solid var(--line); background:#131a26; }
 .msg.me { flex-direction:row-reverse; }
 .msg.me .av { color:var(--blue); }
 .msg.ai .av { color:var(--green); }
 .bub { max-width:min(72ch,86%); }
 .who { font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
   color:var(--faint); margin-bottom:3px; }
 .msg.me .who { text-align:right; }
 .body { background:#141a26; border:1px solid var(--line);
   border-radius:12px; padding:10px 13px; word-break:break-word; }
 .msg.me .body { background:#182338; border-color:#26365a; }
 .body p { margin:0 0 8px; } .body p:last-child { margin:0; }
 .body ul, .body ol { margin:6px 0 8px; padding-left:20px; }
 .body li { margin:2px 0; }
 .body h3, .body h4, .body h5, .body h6 { margin:10px 0 5px; font-size:14px; }
 .body pre { margin:8px 0; }
 .cursor { display:inline-block; width:7px; background:var(--green);
   animation:blink 1s steps(2) infinite; }
 @keyframes blink { 50% { opacity:0; } }
 #typing { display:none; align-items:center; gap:8px; padding:0 18px 8px;
   color:var(--dim); font-size:12.5px; background:var(--panel2); }
 #typing.on { display:flex; }
 #typing .d { width:6px; height:6px; border-radius:50%; background:var(--green);
   animation:bob 1.1s infinite ease-in-out; }
 #typing .d:nth-child(2) { animation-delay:.15s; }
 #typing .d:nth-child(3) { animation-delay:.3s; }
 @keyframes bob { 0%,60%,100% { transform:translateY(0); opacity:.5; }
   30% { transform:translateY(-4px); opacity:1; } }
 .composer { display:flex; gap:8px; padding:12px; border-top:1px solid
   var(--line); background:var(--panel); align-items:flex-end; }
 .composer textarea { resize:none; min-height:44px; max-height:180px;
   line-height:1.45; }
 /* ---------- key reveal ---------- */
 #keybox { display:none; }
 #keybox.on { display:block; }
 #keybox .kv { font-family:ui-monospace,monospace; font-size:14px;
   background:#0a1a12; border:1px solid #2c6b4c; color:#b9f0d6;
   padding:11px 12px; border-radius:10px; word-break:break-all; }
 .rowbtns { display:flex; gap:6px; flex-wrap:wrap; }
 @media (max-width: 860px) {
   .chatwrap { grid-template-columns:1fr; }
   aside.side { position:fixed; inset:56px 0 0 0; z-index:30; max-height:none;
     border-radius:0; transform:translateX(-102%);
     transition:transform .18s ease; }
   aside.side.open { transform:none; }
   button.burger { display:block; }
   nav.links { display:none; position:absolute; top:56px; right:10px;
     background:var(--panel); border:1px solid var(--line);
     border-radius:12px; padding:8px; flex-direction:column;
     align-items:stretch; margin:0; }
   nav.links.open { display:flex; }
   section.chat { height:calc(100vh - 190px); min-height:340px; }
   .bub { max-width:92%; }
   main { padding:14px 12px 30px; }
 }
 @media (prefers-reduced-motion: reduce) {
   .cursor, #typing .d { animation:none; }
   #log { scroll-behavior:auto; }
 }
"""


def _head(agent_name: str, title: str, here: str = "",
          description: str = "") -> str:
    """Shared ``<head>`` + branded header. ``here`` marks the active link."""
    def link(href: str, label: str, key: str) -> str:
        cls = ' class="here"' if key == here else ""
        return f'<a href="{href}"{cls}>{esc(label)}</a>'

    meta = (f'<meta name="description" content="{esc(description)}">'
            if description else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
{meta}
<link rel="icon" type="image/svg+xml" href="/logo.svg">
<link rel="alternate icon" href="/favicon.ico">
<meta name="theme-color" content="#0b0e14">
<meta name="color-scheme" content="dark">
<style>{_CSS}</style></head><body>
<header class="top"><div class="topin">
 <a class="brand" href="/" aria-label="{esc(agent_name)} home">
   {_logo_mark(agent_name)}<span>{esc(agent_name)}<span class="dot">.</span></span>
 </a>
 <button class="burger" id="burger" aria-label="menu"
   aria-expanded="false">☰</button>
 <nav class="links" id="links">
   {link("/", "Chat", "chat")}
   {link("/developers", "Developers", "developers")}
   {link("/dashboard", "Live telemetry", "dashboard")}
   {link("/approve", "Approvals", "approve")}
   {link("/api.json", "API index", "api")}
 </nav>
</div></header>"""


_FOOTER = Template("""
<footer>$agent · owner $owner · mood $mood · $grand events counted ·
 pure Python stdlib · no external AI, no external keys · MIT</footer>
</main>""")

#: Small shared script: menu toggle + a copy-to-clipboard helper.
_NAV_JS = """
<script>
(function(){
  var burger = document.getElementById("burger");
  var links = document.getElementById("links");
  var side = document.getElementById("sidebar");
  function closeAll(){
    if (links) links.classList.remove("open");
    if (side) side.classList.remove("open");
    if (burger) burger.setAttribute("aria-expanded", "false");
  }
  if (burger) burger.addEventListener("click", function(){
    var open = (links && links.classList.contains("open")) ||
               (side && side.classList.contains("open"));
    closeAll();
    if (!open) {
      if (side) { side.classList.add("open"); }
      else if (links) { links.classList.add("open"); }
      burger.setAttribute("aria-expanded", "true");
    }
  });
  document.addEventListener("keydown", function(e){
    if (e.key === "Escape") closeAll();
  });
  window.realaiCopy = function(text, btn){
    function done(){
      if (!btn) return;
      var old = btn.textContent;
      btn.textContent = "copied";
      setTimeout(function(){ btn.textContent = old; }, 1200);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, done);
    } else {
      var ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(ta); done();
    }
  };
})();
</script>"""


def _footer(agent: Any) -> str:
    snap = agent.mind.snapshot()
    grand = agent.storage.get_counters().get("grand_total", 0)
    return _FOOTER.substitute(
        agent=esc(snap["agent"]), owner=esc(snap["owner"]),
        mood=esc(snap["mood"]["note"]), grand=grand)


# --------------------------------------------------------------- the chat app

_CHAT_BODY = Template("""
<main>
<div class="chatwrap">
 <aside class="side" id="sidebar">
  <div class="k">conversations</div>
  <button class="act" id="newchat" style="width:100%">+ New chat</button>
  <div id="sessions"></div>
  <div class="hint">Sessions live in this browser <b>and</b> in the agent's own
   database, so $agent keeps the last few turns of each one in context.
   Nothing is sent anywhere else — talk to it as much as you like.</div>
 </aside>
 <section class="chat">
  <div id="log"></div>
  <div id="typing"><span class="d"></span><span class="d"></span>
   <span class="d"></span><span id="typingwho">$agent is typing…</span></div>
  <form class="composer" id="composer">
   <textarea id="input" rows="1" autocomplete="off"
     placeholder="talk to it — no key needed · Enter sends, Shift+Enter newline"
     aria-label="message to $agent"></textarea>
   <button class="act" id="send" type="submit">Send</button>
  </form>
 </section>
</div>
<div class="card" style="margin-top:16px">
 <div class="k">what this is</div>
 <div class="grid">
  <div class="tile"><h3 class="ok">A real AI, built from scratch</h3>
   <p>$tagline It has drives, a mood, goals, memory and its own thinking
    loop — and you can read every line of it.</p></div>
  <div class="tile"><h3 class="ok">Honest about its limits</h3>
   <p>No internet, no external model, no live data feed. Ask it for something
    it does not have and it says so plainly, then tells you what it can do
    instead.</p></div>
  <div class="tile"><h3 class="warn">The API belongs to the owner</h3>
   <p>This demo chat is keyless and rate-limited per visitor (about
    $rate/min). Programmatic access needs a key only the owner can issue —
    see <a href="/developers">the developer portal</a>.</p></div>
 </div>
</div>""")

#: Plain (non-Template) so JS regexes and ``$1`` replacements stay literal.
#: ``@@AGENT@@`` is substituted by :func:`chat_page`.
_CHAT_JS = """
<script>
var AGENT = "@@AGENT@@";
var STORE = "realai.sessions.v1";
var log = document.getElementById("log");
var input = document.getElementById("input");
var send = document.getElementById("send");
var typing = document.getElementById("typing");
var sessionsBox = document.getElementById("sessions");
var sidebar = document.getElementById("sidebar");
var current = null;

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
/* --- rendering --- */
function bubble(role, text, animate){
  var wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "me" : "ai");
  var av = document.createElement("div");
  av.className = "av";
  av.textContent = role === "user" ? "you" : AGENT.slice(0, 1);
  var bub = document.createElement("div");
  bub.className = "bub";
  var who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "you" : AGENT;
  var body = document.createElement("div");
  body.className = "body";
  bub.appendChild(who); bub.appendChild(body);
  wrap.appendChild(av); wrap.appendChild(bub);
  log.appendChild(wrap);
  if (role === "user" || !animate) body.innerHTML = md(text);
  else reveal(body, text);
  log.scrollTop = log.scrollHeight;
  return body;
}
/* streaming reveal: words land progressively, click to skip */
function reveal(el, text){
  var parts = String(text).split(/(\\s+)/);
  var i = 0, done = false, timer = null;
  function finish(){
    if (done) return;
    done = true;
    if (timer) clearTimeout(timer);
    el.innerHTML = md(text);
    log.scrollTop = log.scrollHeight;
  }
  el.addEventListener("click", finish);
  (function step(){
    if (done) return;
    i = Math.min(parts.length, i + Math.max(2, Math.round(parts.length / 90)));
    el.innerHTML = md(parts.slice(0, i).join("")) + '<span class="cursor">'
      + '&nbsp;</span>';
    log.scrollTop = log.scrollHeight;
    if (i >= parts.length){ finish(); return; }
    timer = setTimeout(step, 18);
  })();
}
function renderSessions(){
  var list = load();
  sessionsBox.innerHTML = "";
  if (!list.length){
    var empty = document.createElement("div");
    empty.className = "hint";
    empty.textContent = "no conversations yet";
    sessionsBox.appendChild(empty);
    return;
  }
  list.forEach(function(s){
    var row = document.createElement("div");
    row.className = "sess" + (current && current.id === s.id ? " on" : "");
    var t = document.createElement("div");
    t.className = "t";
    t.textContent = title(s);
    t.title = ((s.messages || []).length) + " messages";
    var x = document.createElement("button");
    x.className = "x";
    x.textContent = "\\u00d7";
    x.setAttribute("aria-label", "delete conversation");
    x.addEventListener("click", function(ev){
      ev.stopPropagation();
      save(load().filter(function(o){ return o.id !== s.id; }));
      if (current && current.id === s.id){ current = null; newChat(); }
      else renderSessions();
    });
    row.appendChild(t); row.appendChild(x);
    row.addEventListener("click", function(){ openSession(s.id); });
    sessionsBox.appendChild(row);
  });
}
function paint(s){
  log.innerHTML = "";
  (s.messages || []).forEach(function(m){ bubble(m.role, m.text, false); });
  if (!(s.messages || []).length) greet();
}
function greet(){
  bubble("ai", "Hi — I'm " + AGENT + ", a self-contained AI with my own mind, "
    + "memory and goals. Ask me `status`, tell me something to keep with "
    + "`remember that ...`, or ask me for something I cannot do and I will "
    + "answer honestly.", false);
}
function newChat(){
  current = { id: "local-" + Date.now().toString(36) + "-"
    + Math.random().toString(36).slice(2, 8),
    session_id: null, ts: Date.now(), messages: [] };
  upsert(current); paint(current); renderSessions(); input.focus();
}
function openSession(id){
  var s = find(id);
  if (!s){ newChat(); return; }
  current = s; paint(s); renderSessions();
  if (sidebar) sidebar.classList.remove("open");
}
/* --- sending --- */
function busy(on){
  send.disabled = on; input.disabled = on;
  if (typing.classList) typing.classList.toggle("on", on);
  send.textContent = on ? "\\u2026" : "Send";
}
function sendMsg(){
  var text = (input.value || "").trim();
  if (!text || send.disabled) return;
  if (!current) newChat();
  current.messages.push({ role: "user", text: text, ts: Date.now() });
  current.ts = Date.now();
  upsert(current);
  bubble("user", text, false);
  renderSessions();
  input.value = ""; input.style.height = "auto";
  busy(true);
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
    busy(false);
    var reply = j.response
      || (j.error && j.error.message)
      || "I could not answer that — the request came back empty.";
    if (status === 429) reply = "Slow down a little: the keyless demo is "
      + "rate-limited per visitor. Try again in a few seconds.";
    if (j.session_id) mine.session_id = j.session_id;
    mine.messages.push({ role: "ai", text: reply, ts: Date.now() });
    mine.ts = Date.now();
    upsert(mine);
    bubble("ai", reply, true);
    renderSessions();
    input.focus();
  }).catch(function(){
    busy(false);
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
  input.style.height = "auto";
  input.style.height = Math.min(180, input.scrollHeight) + "px";
});
(function boot(){
  var list = load();
  if (list.length) openSession(list[0].id); else newChat();
  renderSessions();
})();
</script>
</body></html>"""


def chat_page(agent: Any, tagline: Optional[str] = None) -> str:
    """The branded chat app served at ``/``."""
    snap = agent.mind.snapshot()
    cfg = agent.cfg
    name = snap["agent"]
    body = _CHAT_BODY.substitute(
        agent=esc(name),
        tagline=esc(tagline or cfg.brand_tagline),
        rate=int(getattr(cfg, "demo_rate_per_min", 12)))
    return (_head(name, f"{name} — your own AI, live", "chat",
                  f"Chat with {name}, a fully local cognitive AI. No key "
                  f"needed, no external AI involved.")
            + body + _footer(agent) + _NAV_JS
            + _CHAT_JS.replace("@@AGENT@@", esc(name)))


# --------------------------------------------------------- developer portal

_DEV_BODY = Template("""
<main>
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
   <button class="act" type="submit" id="reqsend">Request access</button>
   <span class="hint" style="align-self:center">You start PENDING. The owner
    hears about it on Telegram <b>and</b> by email, then decides from
    <code>/approve</code>. No key exists until they mint one.</span>
  </div>
 </form>
 <div id="reqout"></div>
</div>

<div class="card">
 <div class="k">authenticating</div>
 <p class="dim" style="margin-top:0">Gated routes take an owner-issued key as a
  bearer token. A key passes when it holds at least one of the route's scopes;
  <code>owner</code> implies everything. Requests are counted even when they
  fail.</p>
 <pre><code>curl -s https://$host/v1/chat \\
  -H "Authorization: Bearer rxa_..." \\
  -H "Content-Type: application/json" \\
  -d '{"message": "status"}'</code></pre>
 <div class="rowbtns">
  <button class="ghost" id="copycurl">copy curl</button>
  <a class="ghost" style="padding:8px 13px" href="/api.json">raw index
   (JSON)</a>
  <a class="ghost" style="padding:8px 13px" href="/dashboard">live
   telemetry</a>
 </div>
</div>

<div class="card">
 <div class="k">live endpoint table · $count routes</div>
 <table id="endpoints">
  <thead><tr><th>method</th><th>path</th><th>access</th>
   <th>what it does</th></tr></thead>
  <tbody>$rows</tbody>
 </table>
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
 <h2 style="margin-top:18px">status codes</h2>
 <table><tbody>$errors</tbody></table>
</div>""")

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
            + body + _footer(agent) + _NAV_JS
            + _DEV_JS.replace("@@HOST@@", esc(host)))


# -------------------------------------------------------- owner approval box

_APPROVE_BODY = Template("""
<main>
<h1>Approvals</h1>
<p class="tag">Your inbox. Every client starts <b>PENDING</b> and gets nothing
 until you decide here — or from Telegram with
 <code>/approve &lt;name&gt;</code>. Approving mints a scoped API key and shows
 it to you <b>exactly once</b>: it is stored only as a hash, so it cannot be
 fetched later, and it is never emailed.</p>

$warning
<div id="keybox" class="card" style="border-color:#2c6b4c">
 <div class="k ok">key issued — copy it now, it will not be shown again</div>
 <div class="kv" id="keyval"></div>
 <div class="rowbtns" style="margin-top:10px">
  <button class="ghost" id="keycopy">copy key</button>
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
</div>""")

#: Plain string: ``@@TOKEN@@`` is substituted by :func:`approve_page`.
_APPROVE_JS = """
<script>
var TOKEN = "@@TOKEN@@";
function flash(htmlText, kind){
  document.getElementById("flash").innerHTML =
    '<div class="note ' + (kind || "") + '">' + htmlText + '</div>';
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


def _button(username: str, action: str, label: str, css: str = "ghost",
            vip: Optional[bool] = None) -> str:
    extra = "" if vip is None else f", {'true' if vip else 'false'}"
    name = esc(username)
    return (f"<button class='{css}' data-u='{name}' "
            f"onclick=\"act('{name}','{esc(action)}'{extra})\">"
            f"{esc(label)}</button>")


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
            lambda u: _button(u["username"], "approve", "Approve", "good")
            + _button(u["username"], "deny", "Deny", "danger"))
    else:
        pending_rows = ("<p class='dim'>Nobody is waiting. Requests from "
                        "<a href='/developers'>/developers</a> land here "
                        "instantly.</p>")
    if approved:
        approved_rows = _user_table(
            approved,
            lambda u: _button(u["username"], "vip",
                              "Remove VIP" if u.get("is_vip") else "Make VIP",
                              "ghost", vip=not bool(u.get("is_vip")))
            + _button(u["username"], "deny", "Deny", "danger"))
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
            + body + _footer(agent) + _NAV_JS
            + _APPROVE_JS.replace("@@TOKEN@@", esc(token or "")))


def locked_page(retry: int) -> str:
    """A human-readable 429 for the gated surfaces.

    Wrong web tokens are counted per visitor (see ``web.TokenGuard``); once a
    visitor trips the lockout this page tells them plainly what happened and
    how long to wait, instead of handing back a bare status code.
    """
    retry = max(1, int(retry))
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,"
        "initial-scale=1\"><title>RealAI — too many attempts</title>"
        "<meta http-equiv=\"refresh\" content=\"" + str(retry) + "\">"
        "<style>body{background:#0b0e14;color:#d7dce5;margin:0;"
        "font:15px/1.6 ui-sans-serif,system-ui,sans-serif;display:flex;"
        "align-items:center;justify-content:center;min-height:100vh}"
        "main{background:#12161f;border:1px solid #1f2633;border-radius:14px;"
        "padding:26px;max-width:460px;width:calc(100% - 40px)}h1{font-size:19px;"
        "margin:0 0 8px}p{color:#8b94a7;font-size:13.5px;margin:0 0 10px}"
        "code{background:#0e1219;border:1px solid #1f2633;border-radius:6px;"
        "padding:1px 6px;color:#7aa2f7}a{color:#7aa2f7}</style></head>"
        "<body><main><h1>Too many wrong web tokens</h1>"
        "<p>This visitor has been locked out of the owner surfaces for about "
        "<code>" + str(retry) + "s</code>. The page reloads itself when the "
        "lockout expires.</p>"
        "<p>The attempt was counted in the agent ledger and the owner was "
        "told — this gate protects key minting, so it is deliberately "
        "unforgiving.</p>"
        "<p><a href=\"/\">← back to the chat app</a></p>"
        "</main></body></html>")


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
