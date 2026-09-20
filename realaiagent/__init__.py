"""RealAI Agent.

A fully local, owner-controlled cognitive AI agent.

- Its own mind: drives, mood, goals, and an autonomous thinking loop.
- Built-in learning: trainable intent model, Q-learning action policy,
  procedural skill memory, episodic/semantic memory.
- Device control with a hard permission gate (allow / ask / deny).
- API-only surface with its own API-key management (scoped, revocable).
- Everything is counted: a durable usage ledger for requests, thoughts,
  decisions, actions, permissions, learning updates, memory and errors.

Pure Python standard library. No external AI, no external APIs, no
third-party packages. The agent is 100% the owner's code.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
