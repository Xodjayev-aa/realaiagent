# RealAI Agent — what it is and how it is made

*Version 0.5.0 · pure Python standard library · runs on Vercel + Turso, or on
any machine you own.*

---

## 1. In one paragraph

RealAI is an **owner-controlled AI agent you host yourself**. It has a
public chat website, an API with per-customer keys and billing, a
Telegram remote control, a trainable language-understanding model, a
memory, goals it pursues on its own, and a permission gate in front of
every action. It is written in ~9,000 lines of Python using **only the
standard library** — no framework, no pip packages, no external AI
service, no key to anyone else — and it deploys as one serverless
function on Vercel with all of its state in a Turso database. Optionally,
it plugs into open models running on your own hardware to talk fluently,
draw images, speak, listen, and it builds PowerPoint decks by itself.

---

## 2. The two brains, and why there are two

| | The **core** brain | The **optional** brain |
|---|---|---|
| what | trainable intent classifier + planner + memory + goals + Q-learning | an open large language model (Llama, Qwen, Mistral…) through Ollama |
| where it runs | inside the function, everywhere (including Vercel) | only on a machine you own with enough RAM/GPU |
| what it's for | commands, devices, permissions, teaching, status, memory | free conversation, writing, outlines for presentations |
| cost | zero | electricity |

The core was built first because of a hard rule: *no other AI inside it,
no external keys*. It is narrow but honest — it says "I did not follow
that" instead of guessing. The optional brain exists because you asked
for ChatGPT/Gemini-style talking; their models cannot be copied, but open
models can be run locally, so the agent has a switch for one. When the
switch is off (as on Vercel), the agent explains what is missing and how
the owner can enable it.

---

## 3. What a request goes through

```
browser / curl / Telegram
        │
        ▼
 vercel.py  (WSGI adapter)  ──►  web.py  WebApp.handle()
        │                            │ pages, /media, /stream, /approve, /cron/tick
        │                            ▼
        │                     api/server.py  dispatch()
        │                       route match → API key → scope → rate limit → billing
        │                            ▼
        │                     api/routes.py  handler
        │                            ▼
        │                 conversation.py  ConversationManager.reply()
        │                   1. media request?  (draw / presentation / say)   → generative.py
        │                   2. out of scope?   honest answer
        │                   3. follow-up / pronoun resolution from session
        │                   4. engine.handle_message()
        │                        nlp.py  IntentModel (trainable)  → intent + confidence
        │                        planner.py  → Plan of ActionRequests
        │                        actions/permissions.py  allow / ask / deny
        │                        actions/executor.py  run + Q-learning update
        │                   5. engine unsure & local LLM configured? → generative.chat()
        │                            ▼
        └──────────────  storage.py  ──► sqlite3 (file)  or  libsql_http.py ──► Turso (HTTPS)
```

Every box above is a plain module; every arrow is a Python call. There is
no message bus, no ORM, no dependency injection framework.

---

## 4. The pieces, one by one

### `nlp.py` — the trainable understanding model
A multinomial naive-Bayes classifier over word and character n-grams with
a small seeded vocabulary of intents (`greet`, `status`, `command`,
`learn`, `goal`, `approve`, `deny`, `chat`, …). Its weights are JSON in
the `state` table. `teach: "<sentence>" means <intent>` reweights it
immediately — that is the "training".

### `planner.py` + `actions/` — from intent to permission-gated action
The planner turns a parsed message into a `Plan` of `ActionRequest`s
(`device.control`, `system.command`, `files.read/write`, `http.request`,
`notify`, `timer.set`). The `PermissionManager` matches each against
owner policies: `allow`, `deny`, or `ask` (default) — `ask` parks it in a
`pending` table and notifies the owner (Telegram / email / web inbox at
`/approve`). The executor runs allowed actions with timeouts, records the
outcome, and updates a Q-table so repeated successes are preferred.

### `mind.py` — drives, mood, goals, thinking
Numeric drives (curiosity, duty, contentment) and a mood (valence,
arousal) that move with outcomes. `tick()` is one thought: advance the top
goal, consolidate memory, reflect. Self-hosted, a thread ticks every 10 s.
On Vercel there are no threads, so `Agent.tick_if_due()` runs on the back
of chat requests, rate-limited through a database row, and Vercel Cron
calls `/cron/tick` on a schedule.

### `conversation.py` — the voice
Multi-turn sessions (pronoun and follow-up resolution: "turn it off",
"why?", "more"), memory recall quoted honestly ("you told me…"), scope
answers for things it cannot do, and — new in 0.5 — routing of media
requests and the LLM fallback.

### `generative.py` — talk, draw, speak, listen, present
One class, one method per ability, all returning `GenResult`, never
raising, all **off until configured**:

- `chat()` / `stream_chat()` → Ollama `/api/chat` (persona + last turns + recalled facts)
- `image()` → Stable Diffusion WebUI `/sdapi/v1/txt2img` → PNG
- `transcribe()` → whisper.cpp `/inference` (hand-rolled multipart body) → text
- `speak()` → Piper HTTP or CLI → WAV
- `presentation()` → **built-in `.pptx` writer**: the Office Open XML
  package (content types, rels, presentation, master, layout, theme,
  slides) written with `zipfile` and string templates; the outline comes
  from the LLM when present, otherwise a structured skeleton. Works
  everywhere, including Vercel.

Generated files get an unguessable name, are served at `/media/<name>`
with a strict pattern (no traversal), expire after 24 h, and are also
stored in the database when storage is remote, so a download works from a
different serverless instance than the one that produced it.

### `react.py` — the Think → Act → Observe loop
The classic ReAct agent, hand-written: prompt the model for `THOUGHT` +
one JSON `ACTION`, parse, run the tool, feed `OBSERVATION` back, repeat
until `final_answer`. Tools: an `ast`-based calculator (no `eval`), a
scoped file reader, and the generative tools. `realai react "<task>"`.

### `users.py`, `api/auth.py` — the business layer
Customers `request` access → owner `approve`/`deny`/`ban` → each gets an
`rxa_` key with scopes, a request limit and a balance; every request is
charged (`request_cost`) unless VIP; wrong keys and over-limit calls are
counted as intrusions and pushed to Telegram. Keys are stored as
SHA-256(pepper + key); the pepper and the owner key now live in the
database (mirrored to disk when writable), which is what makes a
serverless deploy keep the same owner key across cold starts.

### `storage.py` + `libsql_http.py` — one schema, two backends
Sixteen tables (keys, users, usage ledger, conversations, memories, goals,
skills, permissions, pending, devices, qtable, events, state,
session_turns, media). `Storage` speaks plain SQLite SQL through a
cursor-like object. With no `REALAI_DATABASE_URL` it is `sqlite3` on a
file. With one, it is `libsql_http.Connection`: every `execute()` becomes
one `POST /v2/pipeline` to Turso with typed arguments, and the typed rows,
`affected_row_count` and `last_insert_rowid` come back — the same
interface, so the other 110 queries in the codebase did not change. This
is why **Turso** and not Supabase: Turso *is* SQLite, hosted.

### `web.py`, `web_pages.py` — the product
Server-rendered pages with inline CSS/JS (no build step): the public chat
(sessions in localStorage, markdown, streamed reveal, image/audio/file
attachments, 🎤 when whisper is on, 🔊 when Piper is on), the developers
portal generated from the route table, the owner approval inbox, and a
dashboard fed by 15 Server-Sent-Events topics.

### `vercel.py`, `api/*.py`, `vercel.json` — the deploy
A genuine PEP 3333 WSGI callable (what `@vercel/python` actually loads),
built on `BytesIO` and `environ`. `vercel.json` rewrites every path to
the function, sets a 30 s max duration and declares the cron.

---

## 5. Security model, briefly

- The website is public but **keyless callers can reach nothing
  privileged**: `/public/chat` has no scopes and a per-visitor token
  bucket.
- `/v1/*` needs an owner-issued key; scopes are checked per route.
- `/dashboard`, `/approve`, `/stream/*`, `/cron/tick` are gated by
  `REALAI_WEB_TOKEN` (header or query — never body, so a cross-site form
  cannot forge them).
- Every action is `ask` by default; the owner widens it.
- `files.*` are confined to `file_roots`; commands have hard timeouts.
- Calculator uses the AST, never `eval`. Media names are pattern-matched.
- Secrets never leave the database/data dir; the only outbound calls are
  the ones you configure (Turso, your Telegram bot, your SMTP, your
  local models).

---

## 6. How it was made (method)

1. **Constraint first.** Stdlib-only was fixed before any feature, which
   forced hand-written pieces that are normally imported: a naive-Bayes
   classifier, an HTTP server, an SSE hub, a WSGI adapter, a multipart
   encoder, a libSQL driver, a PPTX writer.
2. **Every claim testable offline.** The 253 tests use fake local servers
   that implement the real wire protocols of Ollama, Stable Diffusion,
   whisper.cpp, Piper and Turso's Hrana pipeline (the Turso fake is a real
   SQLite behind HTTP), so the suite proves the integrations without any
   model or account.
3. **Cold-start tests.** A test builds an agent, stores memory and an
   owner key, deletes the entire data directory, builds a second agent
   against the same remote database, and asserts the key verifies, the
   memory recalls and the generated deck downloads.
4. **Honesty as a feature.** Whatever is not configured is *said*, not
   faked — the scope answers are part of the product, not an error path.

---

## 7. Limits you should know

- The core brain is not an LLM. Without Ollama it understands trained
  phrasings, not open conversation.
- Local models need hardware: ~8 GB RAM for a 7–8 B chat model, a GPU for
  images. They cannot run on Vercel; point the URLs at a machine you own
  that is reachable from your deployment.
- Vercel Hobby crons are not precise; autonomy also piggybacks on traffic.
- Turso free tier is generous but metered; the ledger writes a row per
  request. Prune `usage` if you get very popular.
- Postgres/Supabase is not a backend (SQLite dialect throughout).
