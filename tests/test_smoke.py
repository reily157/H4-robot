"""
Smoke test — 60s live mainnet capture.

Run with:
    pytest -m slow tests/test_smoke.py -v -s

Excluded from the normal suite by default (addopts = -m "not slow" in pytest.ini).

Requirements:
  - Network access to api.hyperliquid.xyz
  - Active HIP-4 markets (run outside the daily opening auction)

Hard assertions (must pass):
  - discovery finds at least one target market
  - ≥ 1 row in perp_ctx  (BTC activeAssetCtx updates every ~3s)
  - ≥ 1 row in book_levels for coin='BTC'  (l2Book snapshots are periodic)
  - GET /health returns 200 with ws_connected=true

Soft checks (logged, not asserted):
  - bbo, trades, raw_ctx row counts
  - HIP-4-specific row counts

Total wall time: ~65s (60s capture + init/teardown).
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp
import pytest

from recorder import Recorder


log = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_smoke_60s():
    rec = Recorder(db_path=":memory:", health_port=0)

    try:
        # ── Init ────────────────────────────────────────────────────────
        spec = await rec.init()

        assert spec.has_bucket or spec.has_binary, (
            "discovery returned an empty CycleSpec — "
            "no active HIP-4 markets found on mainnet"
        )
        log.info(
            f"cycle: complete={spec.is_complete} "
            f"bucket_q={spec.bucket.question_id if spec.has_bucket else None} "
            f"binary_oid={spec.binary.outcome_id if spec.has_binary else None}"
        )

        # ── Run 60s ─────────────────────────────────────────────────────
        await rec.run(duration_s=60)

        # ── Query before close (store still open) ────────────────────────
        store = rec._store
        assert store is not None

        n_perp_ctx   = store.query("SELECT COUNT(*) FROM perp_ctx")[0][0]
        n_book_btc   = store.query(
            "SELECT COUNT(*) FROM book_levels WHERE coin='BTC'"
        )[0][0]
        n_bbo        = store.query("SELECT COUNT(*) FROM bbo")[0][0]
        n_bbo_btc    = store.query(
            "SELECT COUNT(*) FROM bbo WHERE coin='BTC'"
        )[0][0]
        n_trades     = store.query("SELECT COUNT(*) FROM trades")[0][0]
        n_raw_ctx    = store.query("SELECT COUNT(*) FROM raw_ctx")[0][0]
        n_book_hip4  = store.query(
            "SELECT COUNT(*) FROM book_levels WHERE coin LIKE '#%'"
        )[0][0]

        log.info(
            f"rows captured — perp_ctx={n_perp_ctx} book_btc={n_book_btc} "
            f"bbo={n_bbo} (btc={n_bbo_btc}) trades={n_trades} "
            f"raw_ctx={n_raw_ctx} book_hip4={n_book_hip4}"
        )

        # ── Health check (server still running) ──────────────────────────
        health_url = rec.health_url
        assert health_url is not None, "health server not started"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{health_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                assert resp.status == 200, f"health returned HTTP {resp.status}"
                health_data = await resp.json()

        log.info(f"health: {health_data}")
        assert health_data["ws_connected"] is True, (
            f"ws_connected=False after 60s — "
            f"n_reconnects may be high: {health_data}"
        )
        assert health_data["uptime_s"] >= 60.0, (
            f"uptime_s={health_data['uptime_s']} < 60"
        )

    finally:
        await rec.close()

    # ── Hard assertions ──────────────────────────────────────────────────
    assert n_perp_ctx >= 1, (
        f"no perp_ctx rows after 60s — "
        "BTC activeAssetCtx subscription may not be working"
    )
    assert n_book_btc >= 1, (
        f"no book_levels rows for BTC after 60s — "
        "BTC l2Book subscription may not be working"
    )

    # ── Soft warnings ────────────────────────────────────────────────────
    if n_bbo_btc == 0:
        log.warning("bbo: 0 rows for BTC — market may be quiet or bbo sub not firing")
    if n_book_hip4 == 0:
        log.warning("book_levels: 0 HIP-4 rows — HIP-4 markets may be in opening auction")
    if n_trades == 0:
        log.warning("trades: 0 rows — no fills during the 60s window")
