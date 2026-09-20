"""Vercel serverless entrypoint for RealAI Agent (free tier).

Serves ``/api`` (and, via vercel.json rewrites, ``/``).

Contract: Vercel's Python runtime loads this module and looks for a
top-level ``app`` (ASGI or WSGI) or ``application`` (WSGI). It does NOT
support a Lambda-style ``handler(event, context) -> dict`` function, and
``handler`` must be a ``BaseHTTPRequestHandler`` *class* - not a plain
function. The WSGI application lives in :mod:`realaiagent.vercel` so both
this file and ``api/[...path].py`` share one implementation.
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
