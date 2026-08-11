"""A ComfyUI API route the web extension uses to poll a llama-server.

The model list lives on the remote server, so it cannot be baked into
``INPUT_TYPES`` at load time — the node does not even know the URL until the
graph is wired up.  The web extension asks this route instead, which performs
exactly the same ``/v1/models`` request the nodes make when they run.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from .server import LlamaServer, LlamaServerError

ROUTE = "/comfyllama/models"

# Polling happens while somebody is looking at the node, so it must give up
# long before a generation request would.
MAX_POLL_TIMEOUT = 30.0


def probe_models(payload: Dict[str, Any]) -> Dict[str, Any]:
    """List the models one endpoint serves.

    Always answers with a dict rather than raising: the caller is a button in
    the UI, and a wrong URL should show a message, not blow up a request
    handler.
    """
    try:
        timeout = min(float(payload.get("timeout") or 15.0), MAX_POLL_TIMEOUT)
    except (TypeError, ValueError):
        timeout = 15.0

    try:
        connection = LlamaServer(
            payload.get("base_url") or "",
            api_key=payload.get("api_key") or "",
            username=payload.get("username") or "",
            password=payload.get("password") or "",
            auth=payload.get("auth") or "auto",
            timeout=timeout,
        )
        return {"models": connection.models(), "base_url": connection.base_url}
    except (LlamaServerError, ValueError) as exc:
        return {"models": [], "error": str(exc)}


def register_routes() -> bool:
    """Attach the route to ComfyUI's aiohttp app. No-op outside ComfyUI."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False

    instance = getattr(PromptServer, "instance", None)
    routes = getattr(instance, "routes", None)
    if routes is None:
        return False

    @routes.post(ROUTE)
    async def list_models(request):  # pragma: no cover - needs a running server
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        # The probe blocks on a socket, so keep it off the event loop.
        result = await asyncio.to_thread(probe_models, payload)
        return web.json_response(result)

    return True
