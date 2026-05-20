"""
Tests for recorder.py — unit tests using in-memory store and synthetic messages.

No network: discovery and WsClient are mocked where needed.
Tests that exercise _dispatch use a real in-memory Store to verify row counts.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from recorder import (
    Recorder,
    _RateCounter,
    _coins_from_cycle,
    _cycle_id,
    _extract_coin,
    _market_kind,
    _write_outcome_map,
)
from discovery import CycleSpec, PriceBinarySpec, PriceBucketSpec
from store import Store


# ─── Fixtures ─────────────────────────────────────────────────────────────────

REF_EXPIRY = datetime(2026, 5, 20, 6, 0, tzinfo=timezone.utc)

REF_BUCKET = PriceBucketSpec(
    question_id=12,
    underlying="BTC",
    expiry=REF_EXPIRY,
    period="1d",
    thresholds=[75348.0, 78423.0],
    named_outcome_ids=[67, 68, 69],
    fallback_outcome_id=66,
)

REF_BINARY = PriceBinarySpec(
    outcome_id=65,
    underlying="BTC",
    target_price=76886.0,
    expiry=REF_EXPIRY,
    period="1d",
)

REF_SPEC = CycleSpec(bucket=REF_BUCKET, binary=REF_BINARY, raw_meta={})


async def _make_store() -> Store:
    store = Store(":memory:")
    await store.init()
    return store


def _make_recorder_with_store(store: Store) -> Recorder:
    rec = Recorder(db_path=":memory:")
    rec._store = store
    return rec


# ─── Helper functions ─────────────────────────────────────────────────────────

class TestMarketKind:
    def test_hip4_prefix(self):
        assert _market_kind("#670") == "HIP4"
        assert _market_kind("#0") == "HIP4"

    def test_perp(self):
        assert _market_kind("BTC") == "PERP"
        assert _market_kind("ETH") == "PERP"


class TestExtractCoin:
    def test_l2book(self):
        msg = {"channel": "l2Book", "data": {"coin": "#670", "time": 1}}
        assert _extract_coin(msg) == "#670"

    def test_bbo(self):
        msg = {"channel": "bbo", "data": {"coin": "#671", "time": 1}}
        assert _extract_coin(msg) == "#671"

    def test_active_asset_ctx(self):
        msg = {"channel": "activeAssetCtx", "data": {"coin": "BTC"}}
        assert _extract_coin(msg) == "BTC"

    def test_trades(self):
        msg = {"channel": "trades", "data": [{"coin": "#670", "side": "B"}]}
        assert _extract_coin(msg) == "#670"

    def test_trades_empty_list(self):
        msg = {"channel": "trades", "data": []}
        assert _extract_coin(msg) is None

    def test_unknown_channel(self):
        msg = {"channel": "subscriptionResponse", "data": {}}
        assert _extract_coin(msg) is None


class TestCycleId:
    def test_uses_bucket_expiry(self):
        cid = _cycle_id(REF_SPEC, "BTC")
        assert cid == "BTC_202605200600"

    def test_binary_only_uses_binary_expiry(self):
        spec = CycleSpec(bucket=None, binary=REF_BINARY, raw_meta={})
        assert _cycle_id(spec, "BTC") == "BTC_202605200600"

    def test_no_expiry_falls_back_to_timestamp(self):
        spec = CycleSpec(bucket=None, binary=None, raw_meta={})
        cid = _cycle_id(spec, "BTC")
        assert cid.startswith("BTC_")
        assert len(cid) > len("BTC_")


class TestCoinsFromCycle:
    def test_reference_spec(self):
        coins = _coins_from_cycle(REF_SPEC)
        # binary 65: enc=650,651 → #650, #651
        assert "#650" in coins
        assert "#651" in coins
        # named 67: #670,#671; 68: #680,#681; 69: #690,#691
        assert "#670" in coins and "#671" in coins
        assert "#680" in coins and "#681" in coins
        assert "#690" in coins and "#691" in coins
        # fallback 66: #660,#661
        assert "#660" in coins and "#661" in coins

    def test_binary_only(self):
        spec = CycleSpec(bucket=None, binary=REF_BINARY, raw_meta={})
        coins = _coins_from_cycle(spec)
        assert coins == ["#650", "#651"]

    def test_bucket_only(self):
        spec = CycleSpec(bucket=REF_BUCKET, binary=None, raw_meta={})
        coins = _coins_from_cycle(spec)
        # named 67,68,69 + fallback 66 → 8 coins
        assert len(coins) == 8

    def test_sorted(self):
        coins = _coins_from_cycle(REF_SPEC)
        assert coins == sorted(coins)


# ─── _RateCounter ─────────────────────────────────────────────────────────────

class TestRateCounter:
    @pytest.mark.asyncio
    async def test_zero_before_ticks(self):
        rc = _RateCounter(window_s=5.0)
        assert rc.rate() == 0.0

    @pytest.mark.asyncio
    async def test_rate_positive_after_ticks(self):
        rc = _RateCounter(window_s=5.0)
        for _ in range(10):
            rc.tick()
        assert rc.rate() > 0.0

    @pytest.mark.asyncio
    async def test_rate_finite_and_positive(self):
        rc = _RateCounter(window_s=1.0)
        for _ in range(100):
            rc.tick()
        r = rc.rate()
        assert r > 0.0
        assert r != float("inf")

    @pytest.mark.asyncio
    async def test_rate_drops_after_window_expires(self):
        rc = _RateCounter(window_s=0.05)
        for _ in range(10):
            rc.tick()
        await asyncio.sleep(0.1)   # let the window expire
        assert rc.rate() == 0.0


# ─── _dispatch — l2Book ───────────────────────────────────────────────────────

class TestDispatchL2Book:
    @pytest.mark.asyncio
    async def test_hip4_book_enqueues_levels(self):
        store = await _make_store()
        rec = _make_recorder_with_store(store)

        msg = {
            "channel": "l2Book",
            "data": {
                "coin": "#670",
                "time": 1700000000000,
                "levels": [
                    [{"px": "0.155", "sz": "100"}, {"px": "0.154", "sz": "200"}],
                    [{"px": "0.160", "sz": "150"}],
                ],
            },
        }
        rec._dispatch(msg)
        await store.flush()

        rows = store.query("SELECT side, level_idx FROM book_levels ORDER BY side, level_idx")
        assert len(rows) == 3  # 2 bids + 1 ask
        sides = {r[0] for r in rows}
        assert sides == {"bid", "ask"}

    @pytest.mark.asyncio
    async def test_perp_book_enqueues(self):
        store = await _make_store()
        rec = _make_recorder_with_store(store)

        msg = {
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "time": 1700000000000,
                "levels": [
                    [{"px": "67000", "sz": "1.0"}],
                    [{"px": "67001", "sz": "0.5"}],
                ],
            },
        }
        rec._dispatch(msg)
        await store.flush()

        rows = store.query("SELECT coin FROM book_levels")
        assert all(r[0] == "BTC" for r in rows)


# ─── _dispatch — trades ───────────────────────────────────────────────────────

class TestDispatchTrades:
    @pytest.mark.asyncio
    async def test_trades_enqueued(self):
        store = await _make_store()
        rec = _make_recorder_with_store(store)

        msg = {
            "channel": "trades",
            "data": [
                {"coin": "#670", "time": 1700000000000, "side": "B",
                 "px": "0.155", "sz": "10", "tid": 1},
                {"coin": "#670", "time": 1700000001000, "side": "A",
                 "px": "0.156", "sz": "5", "tid": 2},
            ],
        }
        rec._dispatch(msg)
        await store.flush()

        rows = store.query("SELECT side FROM trades")
        assert len(rows) == 2
        sides = {r[0] for r in rows}
        assert sides == {"BUY", "SELL"}

    @pytest.mark.asyncio
    async def test_empty_trades_no_error(self):
        store = await _make_store()
        rec = _make_recorder_with_store(store)
        msg = {"channel": "trades", "data": []}
        rec._dispatch(msg)   # must not raise


# ─── _dispatch — bbo ─────────────────────────────────────────────────────────

class TestDispatchBbo:
    @pytest.mark.asyncio
    async def test_bbo_enqueued(self):
        store = await _make_store()
        rec = _make_recorder_with_store(store)

        msg = {
            "channel": "bbo",
            "data": {
                "coin": "#670",
                "time": 1700000000000,
                "bbo": [{"px": "0.155", "sz": "100"}, {"px": "0.160", "sz": "120"}],
            },
        }
        rec._dispatch(msg)
        await store.flush()

        rows = store.query("SELECT bid_px, ask_px FROM bbo")
        assert len(rows) == 1
        assert rows[0][0] == pytest.approx(0.155)
        assert rows[0][1] == pytest.approx(0.160)


# ─── _dispatch — activeAssetCtx ───────────────────────────────────────────────

class TestDispatchActiveAssetCtx:
    @pytest.mark.asyncio
    async def test_perp_ctx_goes_to_perp_ctx_table(self):
        store = await _make_store()
        rec = _make_recorder_with_store(store)

        msg = {
            "channel": "activeAssetCtx",
            "data": {
                "coin": "BTC",
                "time": 1700000000000,
                "ctx": {"markPx": "67000", "oraclePx": "67005", "funding": "0.0001"},
            },
        }
        rec._dispatch(msg)
        await store.flush()

        rows = store.query("SELECT mark_px FROM perp_ctx")
        assert len(rows) == 1
        assert rows[0][0] == pytest.approx(67000.0)
        # raw_ctx must be empty
        assert store.query("SELECT COUNT(*) FROM raw_ctx")[0][0] == 0

    @pytest.mark.asyncio
    async def test_hip4_ctx_goes_to_raw_ctx_table(self):
        store = await _make_store()
        rec = _make_recorder_with_store(store)

        msg = {
            "channel": "activeAssetCtx",
            "data": {
                "coin": "#670",
                "time": 1700000000000,
                "ctx": {"markPx": "0.155"},
            },
        }
        rec._dispatch(msg)
        await store.flush()

        rows = store.query("SELECT sub_type FROM raw_ctx")
        assert len(rows) == 1
        assert rows[0][0] == "activeAssetCtx"
        assert store.query("SELECT COUNT(*) FROM perp_ctx")[0][0] == 0


# ─── _dispatch — ignored messages ────────────────────────────────────────────

class TestDispatchIgnored:
    @pytest.mark.asyncio
    async def test_subscription_response_ignored(self):
        store = await _make_store()
        rec = _make_recorder_with_store(store)
        msg = {
            "channel": "subscriptionResponse",
            "data": {"method": "subscribe"},
        }
        rec._dispatch(msg)   # must not raise, nothing enqueued
        await store.flush()
        assert store.buffer_size() == 0

    @pytest.mark.asyncio
    async def test_malformed_message_does_not_crash(self):
        store = await _make_store()
        rec = _make_recorder_with_store(store)
        rec._dispatch({})
        rec._dispatch({"channel": "l2Book"})   # no data key
        rec._dispatch({"channel": "bbo", "data": None})


# ─── _write_outcome_map ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_outcome_map_all_roles():
    store = await _make_store()
    store.write_cycle(
        cycle_id="BTC_202605200600",
        started_at=datetime.now(timezone.utc),
        bucket_question_id=12,
        bucket_expiry=REF_EXPIRY,
        bucket_thresholds="75348,78423",
        bucket_underlying="BTC",
        binary_outcome_id=65,
        binary_target_price=76886.0,
        binary_expiry=REF_EXPIRY,
        raw_meta={},
    )
    _write_outcome_map(store, "BTC_202605200600", REF_SPEC)

    rows = store.query("SELECT role FROM outcomes_map ORDER BY role")
    roles = {r[0] for r in rows}
    assert "binary" in roles
    assert "fallback" in roles
    assert "named_0" in roles
    assert "named_1" in roles
    assert "named_2" in roles


# ─── run() duration ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_processes_messages_from_queue():
    """run() drains messages from the queue and routes them to _dispatch."""
    store = await _make_store()
    rec = Recorder(db_path=":memory:")
    rec._store = store
    rec._ws = MagicMock()
    rec._ws.connected = True
    rec._ws.n_subscriptions = 4

    # Seed the queue with a known message before run starts
    bbo_msg = {
        "channel": "bbo",
        "data": {
            "coin": "#670",
            "time": 1700000000000,
            "bbo": [{"px": "0.15", "sz": "10"}, {"px": "0.16", "sz": "10"}],
        },
    }
    await rec._queue.put(bbo_msg)

    # Run for 0.2s — enough to drain the single message
    await rec.run(duration_s=0.2)
    await store.flush()

    rows = store.query("SELECT coin FROM bbo")
    assert len(rows) == 1
    assert rows[0][0] == "#670"


@pytest.mark.asyncio
async def test_run_raises_if_not_initialized():
    rec = Recorder()
    with pytest.raises(RuntimeError, match="init"):
        await rec.run(duration_s=0.1)


# ─── get_stats ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_stats_when_not_initialized():
    rec = Recorder()
    stats = rec._get_stats()
    assert stats["ws_connected"] is False
    assert stats["n_subs_active"] == 0
    assert stats["msgs_per_sec"] == 0.0
    assert stats["buffer_size"] == 0
