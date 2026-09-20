"""Catch-all: route every /api/* path to the same serverless handler.

The RealAI web layer (dashboard, /stream/* SSE, /webhook) is mounted at
these paths by vercel.json rewrites, so the whole product is one Python
serverless function on the free tier.

The import is **absolute** (``from realaiagent.vercel import ...``).
A relative ``from .index import ...`` raises ``ImportError: attempted
relative import with no known parent package`` - Vercel loads each
function file as a standalone module, and ``[...path]`` is not even a
valid Python identifier, so this file can never be part of a package.
"""

from realaiagent.vercel import (  # noqa: F401
    app,
    application,
    environ_to_request,
    get_app,
    handle_event,
    normalize_event,
    reset_state,
    wsgi_app,
)
