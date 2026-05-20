"""
HTTP health endpoint for the HIP-4 observer.

Usage:
    def get_stats() -> dict:
        return {
            "ws_connected": ws.connected,
            "n_subs_active": ws.n_subscriptions,
            "msgs_per_sec": rate_counter.rate(),
            "buffer_size": store.buffer_size(),
        }

    server = HealthServer(stats_fn=get_stats)
    await server.start()     # binds 127.0.0.1:8765
    ...
    await server.stop()

GET /health returns:
    {
        "ws_connected": true,
        "n_subs_active": 8,
        "msgs_per_sec": 12.5,
        "buffer_size": 0,
        "uptime_s": 42.31,
        "ts_utc": "2026-05-19T12:34:56.789Z"
    }

`stats_fn` is called on every request and may return any extra keys —
they are passed through verbatim. `uptime_s` and `ts_utc` are always
added by the server and will override any identically-named key in stats.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from aiohttp import web


log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class HealthServer:
    """
    Minimal aiohttp HTTP server exposing a /health endpoint.

    Decoupled from ws_client / store via `stats_fn` injection.
    """

    def __init__(
        self,
        stats_fn: Callable[[], dict],
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._stats_fn = stats_fn
        self.host = host
        self.port = port

        self._started_at: float | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bind and start the HTTP server."""
        if self._runner is not None:
            return

        app = web.Application()
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        self._started_at = asyncio.get_event_loop().time()
        log.info(
            f"health server listening on "
            f"http://{self.host}:{self.effective_port}/health"
        )

    async def stop(self) -> None:
        """Stop the HTTP server and release the port."""
        if self._runner is None:
            return
        await self._runner.cleanup()
        self._runner = None
        self._site = None
        log.info("health server stopped")

    # ─── Context manager ──────────────────────────────────────────────────

    async def __aenter__(self) -> "HealthServer":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()

    # ─── Properties ───────────────────────────────────────────────────────

    @property
    def effective_port(self) -> int:
        """Actual bound port — useful when `port=0` (OS-assigned)."""
        if self._site is not None:
            try:
                server = self._site._server
                if server and server.sockets:
                    return server.sockets[0].getsockname()[1]
            except Exception:
                pass
        return self.port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.effective_port}"

    # ─── Handler ─────────────────────────────────────────────────────────

    async def _handle_health(self, request: web.Request) -> web.Response:
        stats = self._stats_fn()
        now = asyncio.get_event_loop().time()
        uptime = round(now - self._started_at, 2) if self._started_at is not None else 0.0
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        payload = {**stats, "uptime_s": uptime, "ts_utc": ts}
        return web.json_response(payload)
