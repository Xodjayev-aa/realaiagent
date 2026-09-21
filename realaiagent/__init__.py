"""RealAI Agent.

A fully local, owner-controlled cognitive AI agent.

- Its own mind: drives, mood, goals, and an autonomous thinking loop.
- Built-in learning: trainable intent model, Q-learning action policy,
  procedural skill memory, episodic/semantic memory.
- Device control with a hard permission gate (allow / ask / deny).
- Its own doors: a branded chat app at ``/``, a live developer portal at
  ``/developers``, an owner approval inbox at ``/approve``, and a keyed
  API (``/v1/*``, scoped, revocable) behind them all.
- Everything is counted: a durable usage ledger for requests, thoughts,
  decisions, actions, permissions, learning updates, memory and errors.
- 0.3.0: a live web layer (dashboard + 15 SSE streams), a Vercel
  serverless handler (free tier, /tmp storage, Telegram webhook mode)
  and the same-repo Telegram master control - all still pure stdlib.
- 0.5.0: it became a product a stranger can use. A conversational voice
  with multi-turn context and durable per-session memory, a keyless demo
  chat with per-visitor rate limits, an owner approval inbox that mints
  keys one tap at a time (web-token gated, brute-force locked, alerted on
  Telegram), stdlib SMTP so requests reach the owner by email as well, and
  a threaded server that serves exactly the serverless surface.
- 0.6.0: ``REALAI_PROVIDER=hosted`` - fluent chat, images, presentations
  with AI cover slides and voice on the free Vercel tier through a
  keyless public inference API (no account, no key); every generated
  file is archived to the owner's Telegram (forum topics as folders).

Pure Python standard library, no third-party packages. The agent - its
mind, memory, ledger and controls - is 100% the owner's code; generative
abilities are optional tools it calls (local engines or the hosted
provider).
"""

from __future__ import annotations

__version__ = "0.6.0"

__all__ = ["__version__"]
