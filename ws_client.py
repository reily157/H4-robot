"""
Async WebSocket client for Hyperliquid with multiplexed subscriptions
and exponential-backoff reconnect.

Usage:
    async with WsClient(queue=my_queue) as ws:
        await ws.subscribe("l2Book", "#670")
        await ws.subscribe("activeAssetCtx", "BTC")
        # decoded dicts arrive in my_queue

Reconnect behaviour:
    On disconnect / error, pending subs are replayed after a backoff delay.
    Backoff: base=1s, multiplier=2, cap=60s, ±20% jitter.
    The caller's queue is unaffected between reconnects.

Hyperliquid subscribe frame format:
    {"method": "subscribe", "subscription": {"type": <channel>, "coin": <coin>}}
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

import aiohttp


log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────

WS_URL = "wss://api.hyperliquid.xyz/ws"

_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 60.0
_BACKOFF_JITTER = 0.20          # ±20 % of the computed backoff


# ─── Client ───────────────────────────────────────────────────────────────────

class WsClient:
    """
    Async WebSocket multiplexer for Hyperliquid.

    All received messages (decoded JSON dicts) are pushed into `queue`.
    Subscriptions survive reconnects — they are replayed automatically.
    """

    def __init__(
        self,
        url: str = WS_URL,
        queue: asyncio.Queue | None = None,
    ) -> None:
        self.url = url
        self.queue: asyncio.Queue = queue if queue is not None else asyncio.Queue()

        # Subscription frames to replay after every reconnect.
        self._subs: list[dict] = []

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._recv_task: asyncio.Task | None = None
        self._stopping = False

        self._connected = False
        self._n_reconnects = 0          # does not count the initial connection
        self._connect_count = 0         # total successful connections

    # ─── Public state ─────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def n_reconnects(self) -> int:
        return self._n_reconnects

    @property
    def n_subscriptions(self) -> int:
        return len(self._subs)

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Open session and launch the reconnect loop."""
        if self._recv_task is not None:
            return
        self._stopping = False
        self._session = aiohttp.ClientSession()
        self._recv_task = asyncio.create_task(self._reconnect_loop())

    async def close(self) -> None:
        """Stop the reconnect loop, close the WS and the session."""
        self._stopping = True
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._connected = False
        log.info("ws_client closed")

    # ─── Context manager ──────────────────────────────────────────────────

    async def __aenter__(self) -> "WsClient":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ─── Subscriptions ────────────────────────────────────────────────────

    async def subscribe(self, channel: str, coin: str) -> None:
        """
        Subscribe to `channel` / `coin`.

        Appended to the replay list so the subscription survives reconnects.
        If already connected, the frame is sent immediately.
        """
        frame = {
            "method": "subscribe",
            "subscription": {"type": channel, "coin": coin},
        }
        if frame not in self._subs:
            self._subs.append(frame)
        if self._ws is not None and not self._ws.closed:
            await self._send_json(frame)

    # ─── Internal helpers ─────────────────────────────────────────────────

    async def _send_json(self, frame: dict) -> None:
        if self._ws is None or self._ws.closed:
            return
        try:
            await self._ws.send_json(frame)
        except Exception as e:
            log.warning(f"send failed: {e}")

    # ─── Reconnect loop ───────────────────────────────────────────────────

    async def _reconnect_loop(self) -> None:
        attempt = 0
        while not self._stopping:
            try:
                log.info(f"connecting to {self.url}")
                assert self._session is not None
                async with self._session.ws_connect(self.url) as ws:
                    self._ws = ws
                    self._connected = True
                    self._connect_count += 1
                    if self._connect_count > 1:
                        self._n_reconnects += 1
                        log.info(
                            f"reconnected (#{self._n_reconnects}) — "
                            f"replaying {len(self._subs)} subscription(s)"
                        )
                    else:
                        log.info(f"connected to {self.url}")

                    # Replay all subscriptions on (re)connect.
                    for frame in self._subs:
                        await self._send_json(frame)

                    attempt = 0     # reset backoff after successful connect
                    await self._recv_loop(ws)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                self._ws = None
                if self._stopping:
                    break

                attempt += 1
                backoff = min(_BACKOFF_BASE_S * (2 ** (attempt - 1)), _BACKOFF_CAP_S)
                jitter = backoff * _BACKOFF_JITTER * (random.random() * 2 - 1)
                delay = max(0.1, backoff + jitter)
                log.warning(
                    f"ws error (attempt {attempt}): {e!r} — "
                    f"retrying in {delay:.2f}s"
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    break

        self._connected = False
        log.debug("reconnect loop exited")

    # ─── Receive loop ─────────────────────────────────────────────────────

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """
        Drain incoming messages into self.queue until the connection closes.
        Returns normally on any close/error so _reconnect_loop can restart.
        """
        async for msg in ws:
            if self._stopping:
                return
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    await self.queue.put(data)
                except json.JSONDecodeError as e:
                    log.warning(f"invalid JSON from WS: {e}")
            elif msg.type == aiohttp.WSMsgType.ERROR:
                log.warning(f"ws error frame: {ws.exception()}")
                return
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                log.info(f"ws closed by server: {msg.type.name}")
                return
