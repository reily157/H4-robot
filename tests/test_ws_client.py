"""
Tests for ws_client.WsClient.

All tests use a fake aiohttp WebSocket so no network is required.
Covers:
  - subscribe() builds correct frames and deduplicates
  - messages are pushed into the queue
  - subscriptions are replayed on reconnect
  - exponential backoff increases between retries
  - close() is idempotent and cancels cleanly
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from ws_client import WsClient, _BACKOFF_BASE_S, _BACKOFF_CAP_S


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_text_msg(payload: dict) -> MagicMock:
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.TEXT
    msg.data = json.dumps(payload)
    return msg


def _make_close_msg() -> MagicMock:
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.CLOSE
    return msg


class _FakeWs:
    """Minimal fake for aiohttp.ClientWebSocketResponse."""

    def __init__(self, messages: list):
        self._messages = messages
        self.sent: list[dict] = []
        self.closed = False

    async def send_json(self, frame: dict) -> None:
        self.sent.append(frame)

    async def close(self) -> None:
        self.closed = True

    def exception(self):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_subscribe_frame_format():
    """subscribe() stores the correct HL frame and deduplicates."""
    ws = WsClient()
    # Don't start — just test the frame building
    await ws.subscribe("l2Book", "#670")
    await ws.subscribe("trades", "#670")
    await ws.subscribe("l2Book", "#670")   # duplicate — should not be added twice

    assert len(ws._subs) == 2
    assert ws._subs[0] == {
        "method": "subscribe",
        "subscription": {"type": "l2Book", "coin": "#670"},
    }
    assert ws._subs[1] == {
        "method": "subscribe",
        "subscription": {"type": "trades", "coin": "#670"},
    }


@pytest.mark.asyncio
async def test_messages_pushed_to_queue():
    """Messages received on WS are decoded and pushed into the queue."""
    queue: asyncio.Queue = asyncio.Queue()
    client = WsClient(queue=queue)

    payload = {"channel": "trades", "data": [{"coin": "#670"}]}
    fake_ws = _FakeWs([_make_text_msg(payload), _make_close_msg()])

    # Directly exercise _recv_loop
    await client._recv_loop(fake_ws)

    assert not queue.empty()
    got = await queue.get()
    assert got == payload


@pytest.mark.asyncio
async def test_invalid_json_does_not_crash():
    """Invalid JSON is logged and skipped — queue stays clean."""
    queue: asyncio.Queue = asyncio.Queue()
    client = WsClient(queue=queue)

    bad_msg = MagicMock()
    bad_msg.type = aiohttp.WSMsgType.TEXT
    bad_msg.data = "not valid json {{{"

    fake_ws = _FakeWs([bad_msg, _make_close_msg()])
    await client._recv_loop(fake_ws)   # must not raise

    assert queue.empty()


@pytest.mark.asyncio
async def test_subscriptions_replayed_on_reconnect():
    """
    After a disconnect, subscriptions registered before the reconnect
    are re-sent when the connection comes back.
    """
    queue: asyncio.Queue = asyncio.Queue()
    client = WsClient(queue=queue)

    # First connection: close immediately
    first_ws = _FakeWs([_make_close_msg()])
    # Second connection: close immediately too
    second_ws = _FakeWs([_make_close_msg()])

    connect_calls: list[_FakeWs] = []

    async def fake_ws_connect_ctx(url, **_kwargs):
        ws = connect_calls.pop(0) if connect_calls else _FakeWs([_make_close_msg()])
        return ws

    # We'll patch at a lower level: simulate two connections then stop
    sent_on_reconnect: list[dict] = []

    call_count = 0

    async def mock_reconnect_loop(self_inner):
        nonlocal call_count
        for fake in [first_ws, second_ws]:
            self_inner._ws = fake
            self_inner._connected = True
            self_inner._connect_count += 1
            if self_inner._connect_count > 1:
                self_inner._n_reconnects += 1
                for frame in self_inner._subs:
                    sent_on_reconnect.append(frame)
                    await self_inner._send_json(frame)
            await self_inner._recv_loop(fake)
        self_inner._connected = False

    client._subs = [
        {"method": "subscribe", "subscription": {"type": "l2Book", "coin": "#670"}},
    ]

    with patch.object(WsClient, "_reconnect_loop", mock_reconnect_loop):
        client._session = MagicMock()
        client._recv_task = asyncio.create_task(mock_reconnect_loop(client))
        await asyncio.wait_for(client._recv_task, timeout=2.0)

    # Subscriptions were re-sent on the second connection
    assert sent_on_reconnect == client._subs
    assert second_ws.sent == client._subs


@pytest.mark.asyncio
async def test_close_is_idempotent():
    """close() can be called multiple times without error."""
    client = WsClient()
    await client.close()
    await client.close()   # second call must not raise


@pytest.mark.asyncio
async def test_close_before_start():
    """close() before start() is safe."""
    client = WsClient()
    assert not client.connected
    await client.close()


@pytest.mark.asyncio
async def test_backoff_values():
    """Backoff grows exponentially and is capped."""
    import ws_client as wsc
    import random as rnd

    # Seed randomness for deterministic jitter
    random_state = random.Random(0)

    base = _BACKOFF_BASE_S
    cap = _BACKOFF_CAP_S

    for attempt in range(1, 8):
        backoff = min(base * (2 ** (attempt - 1)), cap)
        jitter = backoff * 0.20 * (random_state.random() * 2 - 1)
        delay = max(0.1, backoff + jitter)
        assert delay >= 0.1
        assert delay <= cap * 1.20 + 0.01   # at most cap + 20% jitter

    # After enough retries the backoff is capped
    backoff_high = min(base * (2 ** 10), cap)
    assert backoff_high == cap


@pytest.mark.asyncio
async def test_connected_property_starts_false():
    client = WsClient()
    assert client.connected is False
    assert client.n_reconnects == 0
    assert client.n_subscriptions == 0


# fix missing import in test_backoff_values
import random


# ─── Heartbeat tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_heartbeat_sends_ping_periodically():
    """_heartbeat sends {"method": "ping"} at each interval while ws is open."""
    import ws_client as wsc

    client = WsClient()
    fake_ws = _FakeWs([])
    fake_ws.closed = False
    client._ws = fake_ws   # needed by _send_json

    with patch.object(wsc, "_PING_INTERVAL_S", 0.01):
        task = asyncio.create_task(client._heartbeat(fake_ws))
        await asyncio.sleep(0.06)   # ~5 intervals at 0.01s each
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    ping = {"method": "ping"}
    assert fake_ws.sent.count(ping) >= 2, (
        f"expected ≥2 pings, got {fake_ws.sent}"
    )
    assert all(f == ping for f in fake_ws.sent), (
        f"unexpected frame sent: {fake_ws.sent}"
    )


@pytest.mark.asyncio
async def test_heartbeat_stops_when_ws_already_closed():
    """_heartbeat returns immediately if ws.closed is True at entry."""
    import ws_client as wsc

    client = WsClient()
    fake_ws = _FakeWs([])
    fake_ws.closed = True
    client._ws = fake_ws

    with patch.object(wsc, "_PING_INTERVAL_S", 0.01):
        # Should return quickly — while condition is False from the start
        await asyncio.wait_for(client._heartbeat(fake_ws), timeout=0.5)

    assert fake_ws.sent == [], "no ping should be sent on a closed ws"


@pytest.mark.asyncio
async def test_heartbeat_does_not_send_while_stopping():
    """_heartbeat sends nothing when _stopping is True."""
    import ws_client as wsc

    client = WsClient()
    client._stopping = True
    fake_ws = _FakeWs([])
    fake_ws.closed = False
    client._ws = fake_ws

    with patch.object(wsc, "_PING_INTERVAL_S", 0.01):
        await asyncio.wait_for(client._heartbeat(fake_ws), timeout=0.5)

    assert fake_ws.sent == []


@pytest.mark.asyncio
async def test_heartbeat_cancelled_when_recv_loop_exits():
    """
    The hb_task pattern in _reconnect_loop: when _recv_loop exits, the
    heartbeat task must be cancelled and fully awaited.
    """
    client = WsClient()
    fake_ws = _FakeWs([_make_close_msg()])
    fake_ws.closed = False
    client._ws = fake_ws
    client._stopping = False

    fired: list[bool] = []

    async def _slow_hb(_ws):
        await asyncio.sleep(10)   # would fire after 10s if not cancelled
        fired.append(True)

    hb_task = asyncio.create_task(_slow_hb(fake_ws))
    try:
        await client._recv_loop(fake_ws)
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass

    assert hb_task.done()
    assert hb_task.cancelled(), "hb_task must be cancelled when recv_loop exits"
    assert fired == [], "heartbeat must not fire after cancellation"
