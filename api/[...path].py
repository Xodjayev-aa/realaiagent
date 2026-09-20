"""Catch-all: route every /api/* path to the same serverless handler.

The RealAI web layer (dashboard, /stream/* SSE, /webhook) is mounted at
these paths by vercel.json rewrites, so the whole product is one Python
serverless function on the free tier.
"""

from .index import handler  # noqa: F401
