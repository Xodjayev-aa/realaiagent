# RealAI Agent

**A fully local, owner-controlled AI agent with its own mind — built
entirely from code you own. No external AI. No external APIs. No
third-party packages. No keys to anyone.**

RealAI is a cognitive agent written in pure Python (standard library
only). It perceives what you say, interprets it with a **trainable local
language model**, forms plans, **decides through a hard permission gate**,
acts on devices and the system, counts **everything it does**, and — on its
own timer — keeps **thinking**: advancing your goals, consolidating
memory, and reflecting on its results.

You are the owner. You:

- hold the **owner API key** (created on first boot, yours alone);
- grant or deny **permissions** — the agent can control any device or run
  anything **you let it**, and asks before everything else;
- **train it** — every example you give reweights its local intent model,
  and it additionally learns action policies (Q-learning) and reusable
  skills from what succeeds;
- **inspect and steer its mind** — drives, mood, goals, memory, values;
- **kill-switch its autonomy** at any time;
- run it as a **business**: approve clients (PENDING → APPROVED /
  BANNED), mint each of them an API key, charge per request or grant
  **VIP** (free), watch **security alerts** — all from the API or from
  **Telegram** with your own bot.

The agent also runs a **business layer** (the "null-49.private"
ecosystem from your plan): multi-user access control, per-user API keys,
balance/VIP billing, per-key request limits, intrusion detection, and an
autonomous "free-will" reporting channel to your Telegram.

> **What this is:** a self-contained cognitive engine — a trainable local
> NLP classifier, a planner, an autonomous goal loop, a memory system, a
> Q-learning action policy, and a permission-gated action layer — running
> 100% on your machine.
>
> **What this is not:** a neural large language model. You explicitly asked
> for an AI with *no other AI and no external keys inside it*, which rules
> out calling OpenAI/Gemini/etc. This agent understands and speaks through
> its own local, trainable statistical model. Teach it more and it gets
> better at understanding *you*.

---

## Quickstart

```bash
# 1. run the agent (no install needed, no dependencies)
python3 -m realaiagent serve --port 8100

# first boot prints the OWNER key once (also saved to data/owner_key.txt):
#   OWNER API KEY: rxa_...

# 2. talk to it (from your own UI or curl)
curl -s http://127.0.0.1:8100/v1/chat \
  -H "Authorization: Bearer rxa_..." \
  -H "Content-Type: application/json" \
  -d '{"message": "status"}'
```

```bash
# offline demo — a scripted conversation, no network at all
python3 -m realaiagent demo

# print the stored owner key
python3 -m realaiagent owner-key
```

The agent listens on `0.0.0.0:8100` by default (configurable, see
[Configuration](#configuration)). Your existing UI talks to it over this
REST API — there is no UI in this repo by design.

---

## Ownership & API keys

- On **first boot** the agent creates one **owner key** (`rxa_...`), prints
  it once, and stores it in `data/owner_key.txt` (mode `600`). Keys are
  stored only as `SHA-256(pepper + key)`; the pepper is a random local
  secret in your data dir.
- The owner key has the `owner` scope and can do everything, including
  **issuing scoped keys** for your clients:

```bash
curl -s -X POST http://127.0.0.1:8100/v1/keys \
  -H "Authorization: Bearer rxa_owner..." \
  -d '{"name": "my-ui", "scopes": ["chat", "devices"]}'
```

  Scopes: `owner` (everything) · `admin` (manage devices/skills/memory/
  goals/policies) · `learn` (train it) · `devices` (list + direct control)
  · `chat` (talk, status, counters, events).
- Keys are revocable (`POST /v1/keys/{id}/revoke`) and rate-limited
  (token bucket, 60 req/min by default).

```bash
# create a key
curl -X POST .../v1/keys -d '{"name":"ui","scopes":["chat","devices"]}'
# list keys
curl .../v1/keys
# revoke
curl -X POST .../v1/keys/<id>/revoke
```

---

## API reference

Base: `http://<host>:8100` · Auth: `Authorization: Bearer <key>` ·
All bodies are JSON.

### Talk to the agent

| Method & path | Scope | Description |
|---|---|---|
| `GET /healthz` | public | liveness, agent name, mood, uptime |
| `POST /v1/chat` | `chat` | **the main entry** — `{"message": "..."}` → `{"response": "...", "meta": {intent, confidence, plan, actions, …}}` |
| `GET /v1/status` | `chat` | full mind state: drives, mood, goals, autonomy, counters |
| `GET /v1/counters` | `chat` | **everything counted** — the whole ledger |
| `GET /v1/usage?by=category\|key\|day` | `admin` | usage breakdowns |
| `GET /v1/events?limit=50` | `chat` | the agent's event log — its thoughts, actions, decisions |

Things the agent understands out of the box (and improves as you teach it):

```text
hello!                     → greeting
status                     → full state report
counters                   → the complete counted ledger
turn on the lamp           → device action (asks permission first time)
set the thermostat to 21   → device action with value
open /path/to/file.md      → file read (permission-gated, scoped)
run: git status            → system command (permission-gated)
goal: water the garden     → sets a goal it works on in its own loop
remember that X is Y       → stores it in semantic memory
teach: "do X" means intent → trains its language model
approve 7 / deny 7         → decide a pending permission request
```

### Goals

| Method & path | Scope | Description |
|---|---|---|
| `GET /v1/goals?status=active` | `chat` | list goals |
| `POST /v1/goals` | `admin` | `{"description": "…", "priority": 0.8, "steps": [ {category, action, params, device_id} ]}` |
| `POST /v1/goals/{id}/complete` | `owner` | mark done |
| `DELETE /v1/goals/{id}` | `owner` | drop |

Goals with `steps` are advanced **autonomously** on the agent's thinking
tick — including stopping to ask you for permission when needed.

### Memory

| Method & path | Scope | Description |
|---|---|---|
| `GET /v1/memory?kind=&q=&limit=` | `admin` | search memories (kinds: `episodic`, `semantic`, `learning`) |
| `POST /v1/memory` | `admin` | `{"kind": "semantic", "key": "lamp:room", "value": {"room": "living"}}` |
| `DELETE /v1/memory/{id}` | `owner` | delete |

### Training (the "i am gonna train it" part)

| Method & path | Scope | Description |
|---|---|---|
| `POST /v1/learn` | `learn` | `{"examples": [{"text": "make the toast golden", "intent": "toast"}]}` — each example reweights the local intent model **immediately** and the new model state is persisted |

Plus, automatically and without you asking:

- **Q-learning** — every executed action updates an action-value table;
  successes are reinforced, failures punished.
- **Auto-skills** — any successful multi-step episode is saved as a named
  procedure the agent can repeat.
- **Episodic memory** — every conversation is stored; the mind
  **consolidates** frequently-used memories and lets stale ones fade.
- **Lessons** — denied/failed actions are recorded so the policy avoids
  repeating them.

### Skills

| Method & path | Scope | Description |
|---|---|---|
| `GET /v1/skills` | `chat` | list learned/defined skills |
| `POST /v1/skills` | `admin` | `{"name": "morning routine", "steps": [ … ]}` |
| `DELETE /v1/skills/{name}` | `owner` | remove |

### Devices

| Method & path | Scope | Description |
|---|---|---|
| `GET /v1/devices` | `devices` | list |
| `POST /v1/devices` | `admin` | register a device (see below) |
| `DELETE /v1/devices/{id}` | `owner` | remove |
| `POST /v1/devices/{id}/control` | `devices` | `{"action": "on", "params": {...}}` — direct control, still permission-gated |

A device is a JSON description of its capabilities. Each capability is
either an HTTP call or a local command, with `{placeholder}` params:

```json
{
  "id": "thermo-1",
  "name": "Thermostat",
  "kind": "climate",
  "capabilities": [
    {"action": "set", "executor": "http",
     "endpoint": "http://192.168.1.50:8123/api/climate/thermo/set",
     "method": "POST", "body": "{\"temperature\": {level}}"},
    {"action": "status", "executor": "http",
     "endpoint": "http://192.168.1.50:8123/api/climate/thermo",
     "method": "GET"}
  ]
}
```

Command executor example (the demo uses these):

```json
{"action": "on", "executor": "command", "template": "fanctl.sh on"}
```

`examples/register_device.py` wraps this for you.

### Permissions — "control anything, if you let it"

| Method & path | Scope | Description |
|---|---|---|
| `GET /v1/permissions` | `admin` | all policies + pending requests |
| `POST /v1/permissions` | `owner` | `{"category": "device.control", "policy": "allow", "pattern": "thermo-1:set"}` |
| `GET /v1/permissions/pending` | `chat` | requests awaiting you |
| `POST /v1/permissions/pending/{id}/approve` | `owner` | approve |
| `POST /v1/permissions/pending/{id}/deny` | `owner` | deny |

Policies are `allow` / `ask` / `deny` per **category**
(`device.control`, `system.command`, `files.read`, `files.write`,
`http.request`, `notify`, `timer.set`), optionally narrowed to an exact
action `pattern`.

- Default for everything real is **`ask`**: the agent queues the request
  and tells you `request #N`. **Approving stores an exact allow pattern** —
  that one action is trusted, anything new asks again.
- `notify` and `timer.set` are allowed by default (harmless).
- Chat shortcuts: `approve 7` / `deny 7` / `approve it` (latest).

### Users & billing (the business layer)

The agent runs a client-access business on top of itself. A client
requests access, the owner approves (which mints the client an API key),
and each request the client makes is billed from their balance — unless
they're VIP.

| Method & path | Scope | Description |
|---|---|---|
| `POST /v1/users/request` | **public** | `{"username": "alice"}` → creates a `PENDING` account; the owner is notified |
| `GET /v1/users?status=PENDING` | `owner` | list users (optional filter PENDING/APPROVED/BANNED), with their keys |
| `GET /v1/users/{u}/usage` | `owner` | a user's balance + keys + usage |
| `POST /v1/users/{u}/approve` | `owner` | approve + **mint an API key** (returns the plaintext once). Optional `{"request_limit": 1000, "scopes": [...]}` |
| `POST /v1/users/{u}/deny` | `owner` | ban + revoke all their keys |
| `POST /v1/users/{u}/unban` | `owner` | set back to PENDING |
| `POST /v1/users/{u}/vip` | `owner` | `{"vip": true}` — free forever (no billing) |
| `POST /v1/users/{u}/topup` | `owner` | `{"amount": 10}` — add to their balance |

**Billing rules** (per request, for customer keys only):

- owner key → always free, no limit
- banned / unknown user → `403` + **intrusion logged** (+ Telegram alert)
- not approved → `403`
- per-key request limit reached → `429`
- non-VIP with balance < cost → `402 Payment Required`
- otherwise the cost (default `0.05`) is deducted and the request proceeds

Everything about this is counted: `users.requested / status_* / vip_*`,
`billing.charged / topup / vip_free`, `security.intrusion`, per-key
`request_count` vs `request_limit`.

### Owner controls (the "i control it" part)

| Method & path | Scope | Description |
|---|---|---|
| `GET /v1/owner/state` | `owner` | identity, drives, mood, autonomy |
| `POST /v1/owner/identity` | `owner` | `{"agent_name": "MIRA", "owner_name": "Alex"}` |
| `POST /v1/owner/values` | `owner` | `{"drives": {"curiosity": 0.9, "duty": 0.7}}` |
| `POST /v1/owner/autonomy` | `owner` | `{"on": false}` — **kill switch**: the agent still answers you but stops spontaneous thinking |

### Telegram master control (optional, your own bot)

Set `REALAI_TELEGRAM_BOT_TOKEN` (from @BotFather — **your** bot) and
`REALAI_TELEGRAM_CHAT_ID` (your chat id) and the agent:

- **reports to you on its own "free will"** — a periodic check-in with its
  mood, activity, counts, pending users, and any intrusion attempts;
- **alerts you about security** — invalid/banned keys, in near-real-time;
- **notifies you of new access requests** — with approve/deny commands;
- **obeys you from Telegram** (only your chat id is honored):

```
/status      my state (mood, drives, goals)
/report      full counted ledger
/users [st]  list users (PENDING/APPROVED/BANNED)
/approve x   approve user + mint their API key
/deny x      ban user, revoke their keys
/vip x       make a user free (VIP) / normal
/topup x 10  add 10.00 to a user's balance
/keys        list API keys
/kill /on    autonomy kill switch
/chat <msg>  talk to the agent as the owner
```

Everything runs on the stdlib `urllib` — no third-party Telegram library.
With no token configured, this whole layer is a no-op and the agent still
works 100% offline.

---

## Its own mind

The agent runs an autonomous loop (`tick`, every 10 s by default). On each
thought it:

1. **advances its top-priority goal** (executes the next step; pauses to
   ask for permission when needed; marks goals complete);
2. **consolidates memory** — reinforces what it uses often, fades what it
   doesn't (driven by its *curiosity*);
3. **drifts mood toward calm** (valence/arousal updated by real outcomes —
   successes raise it, failures lower it);
4. **decays drives toward baselines** (curiosity, duty, sociability,
   contentment, vigilance);
5. **occasionally reflects** — writes a self-assessment into its event log
   (visible at `GET /v1/events`).

Inspect all of it live with `GET /v1/status`. The autonomy loop is fully
yours to stop: `POST /v1/owner/autonomy {"on": false}`.

## Everything is counted

Every meaningful event is written to a durable ledger (SQLite) *and* an
in-memory counter — requests (with status code, route, latency, key),
chat messages, thoughts, decisions, actions (success/failed/awaiting/
denied), permission requests/approvals/denies, learning updates (intent
examples, q-updates, skills, lessons), memory stores/consolidations,
goals, device registrations/controls, key creation/revocation, and errors.
Counters survive restarts (rebuilt from the ledger on boot).

- `GET /v1/counters` — the full snapshot
- `GET /v1/usage?by=category|key|day` — breakdowns
- the agent will recite it to you if you just say **`counters`**

## Architecture

```
                 ┌────────────────────────────────────────────────┐
   your UI ────▶ │  HTTP API (stdlib, threaded)                   │
   (API keys)    │   auth → scopes → rate limit → routes          │
                 └──────────────────┬─────────────────────────────┘
                                    ▼
                 ┌────────────────────────────────────────────────┐
                 │  ENGINE  (perceive → attend → plan → decide →  │
                 │  act → respond → reflect)                     │
                 │                                                │
                 │  NLP (trainable local intent model + slots)    │
                 │  MIND  (drives · mood · goals · autonomy tick) │
                 │  PLANNER (single · chain · skill · clarify)    │
                 │  LEARNER (SGD intents · Q-table · skills · mem)│
                 └───────┬──────────────────────┬─────────────────┘
                         ▼                      ▼
                 ┌──────────────┐      ┌────────────────────────┐
                 │ PERMISSIONS  │      │  STORAGE (SQLite)      │
                 │ allow/ask/   │      │  ledger · memory ·     │
                 │ deny +       │      │  goals · devices ·     │
                 │ pending queue│      │  q-table · events ·keys│
                 └──────┬───────┘      └────────────────────────┘
                        ▼ allowed only
                 ┌────────────────────────────────────────┐
                 │  EXECUTOR → device.control (http/cmd)  │
                 │             system.command · files.*   │
                 │             http.request · notify …    │
                 └────────────────────────────────────────┘

   BUSINESS LAYER (null-49.private):
   users (PENDING/APPROVED/BANNED) · per-user API keys · balance/VIP
   billing · per-key limits · intrusion detection
        │                                        │
        ▼                                        ▼
   Telegram master control              autonomous "free-will"
   (/approve /deny /vip /topup …)       reports + security alerts
```

## Deployment (keeping it stable & safe, per your plan)

The agent is a single process + one SQLite file. To run it as a real
service:

```bash
# e.g. systemd: keep it alive, restart on crash, run as a dedicated user
systemctl start realai
```

- **Expose only the API port** (8100) through your firewall — the data
  dir (DB, pepper, owner key) must never be reachable from the network.
  Any other AI engine you might run later (e.g. a local model on 11434)
  should stay loopback-only, behind this gateway.
- **Rate limiting** is built in (per-key token bucket) — tune it with
  `REALAI_RATE_PER_MIN`.
- **All SQL is parameterized** (no injection), all JSON is size-capped,
  all subprocess/HTTP actions have hard timeouts.
- It is **stable by design**: WAL-mode SQLite, counters rebuilt from the
  ledger on boot, autonomy loop wrapped so a bad thought can never crash
  the process.

## Safety

- **Nothing executes without the permission gate.** The default for all
  real actions is `ask`.
- `files.*` are confined to configured roots (`REALAI_FILE_ROOTS`).
- `system.command` runs with a hard timeout; `http.request` with a hard
  timeout and body caps.
- You can **see** every pending request, **approve/deny** individually,
  set blanket `allow`/`deny` policies, and stop the whole autonomy loop.
- The data dir is your data: delete it to erase the agent completely.

## Testing

```bash
python3 -m unittest discover -s . -p "test_*.py" -v   # 103 tests
```

Covers: NLP training/parsing, permission gate, executor (command/file/
device/scope), planner (single/chain/skill), mind (drives/mood/goals/
consolidation/kill switch), learning (intents/Q/skills/memory), storage
ledger, the full HTTP API (auth, scopes, keys, rate-limit, the whole
device→permission→approve→control flow, counters), the **user/billing
ecosystem** (request→approve→key, billing charge/exhaust/402, VIP free,
per-key limits, unapproved 403, ban/401, intrusion counting, usage view),
and **Telegram master control** (command dispatch, owner-only chat,
approve/deny/vip/topup, kill switch) against a fake gateway.

## Configuration

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `REALAI_DATA_DIR` | `data` | all state lives here (db, pepper, owner key) |
| `REALAI_HOST` / `REALAI_PORT` | `0.0.0.0` / `8100` | bind address |
| `REALAI_TICK_SECONDS` | `10` | autonomy thinking interval |
| `REALAI_RATE_PER_MIN` | `60` | per-key rate limit (token bucket) |
| `REALAI_FILE_ROOTS` | data dir | paths `files.*` actions may touch (`:`-separated) |
| `REALAI_BILLING` | `1` | enable per-request billing (0 = off) |
| `REALAI_REQUEST_COST` | `0.05` | charged per request for non-VIP users |
| `REALAI_DEFAULT_REQUEST_LIMIT` | `2500` | per customer key |
| `REALAI_TELEGRAM_BOT_TOKEN` | *(off)* | **your** bot token → enables master control + reports |
| `REALAI_TELEGRAM_CHAT_ID` | *(off)* | your chat id for reports/alerts |
| `REALAI_TELEGRAM_POLL` | `1` | run the Telegram master-control poller |
| `REALAI_AUTONOMOUS_REPORT_MIN` | `60` | free-will report cadence (minutes) |
| `REALAI_AGENT_NAME` / `REALAI_OWNER_NAME` | `REAL` / `Owner` | identity |

## Files you own

```
realaiagent/           the whole agent — pure Python stdlib
  engine.py            the pipeline + autonomy loop
  mind.py              drives, mood, goals, the thinking tick
  nlp.py               the trainable local language model
  planner.py           task planning
  learning.py          SGD intents, Q-table, skills, memory
  storage.py           SQLite state + the universal ledger
  users.py             users, approval, billing (the business layer)
  telegram.py          master control + free-will reports (optional)
  actions/             permission gate, executor, built-in actions
  api/                 keys, HTTP server, routes
examples/              client + device registration helpers
tests/                 103 tests
data/                  created at runtime — all of its state
```

**MIT licensed. Yours.**
