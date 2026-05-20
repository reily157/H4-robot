"""
Tests for health.HealthServer.

All tests bind on port=0 (OS-assigned) to avoid conflicts.
No network beyond localhost.
"""

from __future__ import annotations

import asyncio

import aiohttp
import pytest

from health import HealthServer


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_stats(**extra) -> dict:
    base = {
        "ws_connected": True,
        "n_subs_active": 4,
        "msgs_per_sec": 7.5,
        "buffer_size": 0,
    }
    return {**base, **extra}


async def _get_health(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{url}/health") as resp:
            assert resp.status == 200
            return await resp.json()


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_endpoint_responds_200():
    server = HealthServer(stats_fn=lambda: _make_stats(), port=0)
    async with server:
        data = await _get_health(server.url)
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_response_contains_required_keys():
    server = HealthServer(stats_fn=lambda: _make_stats(), port=0)
    async with server:
        data = await _get_health(server.url)

    for key in ("ws_connected", "n_subs_active", "msgs_per_sec",
                "buffer_size", "uptime_s", "ts_utc"):
        assert key in data, f"missing key: {key}"


@pytest.mark.asyncio
async def test_stats_fn_values_are_reflected():
    stats = {
        "ws_connected": False,
        "n_subs_active": 12,
        "msgs_per_sec": 99.9,
        "buffer_size": 42,
    }
    server = HealthServer(stats_fn=lambda: dict(stats), port=0)
    async with server:
        data = await _get_health(server.url)

    assert data["ws_connected"] is False
    assert data["n_subs_active"] == 12
    assert data["msgs_per_sec"] == 99.9
    assert data["buffer_size"] == 42


@pytest.mark.asyncio
async def test_uptime_is_non_negative():
    server = HealthServer(stats_fn=lambda: _make_stats(), port=0)
    async with server:
        data = await _get_health(server.url)

    assert data["uptime_s"] >= 0.0


@pytest.mark.asyncio
async def test_uptime_increases_over_time():
    server = HealthServer(stats_fn=lambda: _make_stats(), port=0)
    async with server:
        first = (await _get_health(server.url))["uptime_s"]
        await asyncio.sleep(0.05)
        second = (await _get_health(server.url))["uptime_s"]

    assert second > first


@pytest.mark.asyncio
async def test_ts_utc_format():
    server = HealthServer(stats_fn=lambda: _make_stats(), port=0)
    async with server:
        data = await _get_health(server.url)

    ts = data["ts_utc"]
    assert isinstance(ts, str)
    assert ts.endswith("Z"), f"ts_utc must end with 'Z', got {ts!r}"
    # Must be parseable ISO 8601
    from datetime import datetime, timezone
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


@pytest.mark.asyncio
async def test_extra_stats_keys_passed_through():
    """Stats fn may return extra keys — they appear verbatim in the response."""
    server = HealthServer(
        stats_fn=lambda: {**_make_stats(), "custom_metric": 3.14},
        port=0,
    )
    async with server:
        data = await _get_health(server.url)

    assert data["custom_metric"] == 3.14


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    server = HealthServer(stats_fn=lambda: _make_stats(), port=0)
    await server.start()
    await server.stop()
    await server.stop()   # second call must not raise


@pytest.mark.asyncio
async def test_start_is_idempotent():
    server = HealthServer(stats_fn=lambda: _make_stats(), port=0)
    await server.start()
    port_first = server.effective_port
    await server.start()   # second call must be a no-op
    assert server.effective_port == port_first
    await server.stop()


@pytest.mark.asyncio
async def test_effective_port_is_nonzero_after_start():
    server = HealthServer(stats_fn=lambda: _make_stats(), port=0)
    async with server:
        assert server.effective_port > 0
        assert server.url.startswith("http://127.0.0.1:")


@pytest.mark.asyncio
async def test_url_property():
    server = HealthServer(stats_fn=lambda: _make_stats(), host="127.0.0.1", port=0)
    async with server:
        assert server.url.startswith("http://127.0.0.1:")


@pytest.mark.asyncio
async def test_server_unreachable_after_stop():
    server = HealthServer(stats_fn=lambda: _make_stats(), port=0)
    await server.start()
    url = server.url
    await server.stop()

    with pytest.raises(Exception):   # connection refused
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=1)):
                pass
