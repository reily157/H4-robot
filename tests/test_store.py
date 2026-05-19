"""
Tests for store.py — DuckDB store with async batched writes.

All tests use :memory: and pytest-asyncio.
"""

import asyncio
import json
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from store import Store


pytestmark = pytest.mark.asyncio


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def store():
    """Fresh in-memory store for each test."""
    s = Store(":memory:")
    await s.init()
    yield s
    await s.close()


def now():
    """Helper: current UTC datetime."""
    return datetime.now(timezone.utc)


# ─── Initialization ────────────────────────────────────────────────────────────

class TestInit:

    async def test_init_applies_migrations(self, store):
        # After init, _schema_version should be populated
        rows = store.query("SELECT version FROM _schema_version")
        assert rows == [(1,)]

    async def test_init_is_idempotent(self):
        s = Store(":memory:")
        await s.init()
        await s.init()  # second call should be a no-op
        rows = s.query("SELECT COUNT(*) FROM _schema_version")
        assert rows[0][0] == 1
        await s.close()


# ─── Cycle metadata ────────────────────────────────────────────────────────────

class TestCycleMetadata:

    async def test_write_cycle_basic(self, store):
        store.write_cycle(
            cycle_id="20260520",
            started_at=datetime(2026, 5, 20, 5, 30, tzinfo=timezone.utc),
            bucket_question_id=12,
            bucket_expiry=datetime(2026, 5, 20, 6, 0, tzinfo=timezone.utc),
            bucket_thresholds="75348,78423",
            bucket_underlying="BTC",
            binary_outcome_id=65,
            binary_target_price=76886.0,
            binary_expiry=datetime(2026, 5, 20, 6, 0, tzinfo=timezone.utc),
            raw_meta={"outcomes": [], "questions": []},
        )
        rows = store.query("SELECT cycle_id, binary_target_price FROM cycles")
        assert rows == [("20260520", 76886.0)]

    async def test_write_cycle_dedup(self, store):
        """Writing the same cycle_id twice should not insert duplicates."""
        for _ in range(3):
            store.write_cycle(
                cycle_id="20260520",
                started_at=now(),
                bucket_question_id=12,
                bucket_expiry=None, bucket_thresholds=None, bucket_underlying="BTC",
                binary_outcome_id=65, binary_target_price=76886.0,
                binary_expiry=None,
                raw_meta={},
            )
        rows = store.query("SELECT COUNT(*) FROM cycles")
        assert rows[0][0] == 1

    async def test_write_outcome_map(self, store):
        store.write_outcome_map(
            cycle_id="20260520",
            outcome_id=67,
            role="bucket_idx_0",
            yes_coin="#670",
            no_coin="#671",
            description="index:0",
        )
        rows = store.query("SELECT outcome_id, yes_coin FROM outcomes_map")
        assert rows == [(67, "#670")]

    async def test_raw_meta_is_valid_json(self, store):
        meta = {"outcomes": [{"outcome": 65}], "questions": [{"question": 12}]}
        store.write_cycle(
            cycle_id="20260520",
            started_at=now(),
            bucket_question_id=None, bucket_expiry=None,
            bucket_thresholds=None, bucket_underlying=None,
            binary_outcome_id=None, binary_target_price=None, binary_expiry=None,
            raw_meta=meta,
        )
        rows = store.query("SELECT raw_meta FROM cycles WHERE cycle_id = ?", ["20260520"])
        decoded = json.loads(rows[0][0])
        assert decoded == meta


# ─── Enqueue + flush ───────────────────────────────────────────────────────────

class TestEnqueueAndFlush:

    async def test_enqueue_then_flush_book_level(self, store):
        store.enqueue_book_level(now(), None, "#670", "bid", 0, 0.155, 100.0)
        store.enqueue_book_level(now(), None, "#670", "ask", 0, 0.160, 120.0)
        assert store.buffer_size() == 2

        flushed = await store.flush()
        assert flushed == 2
        assert store.buffer_size() == 0

        rows = store.query("SELECT coin, side, px, sz FROM book_levels ORDER BY side")
        assert rows == [("#670", "ask", 0.160, 120.0), ("#670", "bid", 0.155, 100.0)]

    async def test_enqueue_trade(self, store):
        store.enqueue_trade(now(), None, "#670", 0.155, 10.0, "BUY", "tid_1")
        await store.flush()
        rows = store.query("SELECT coin, px, sz, side, tid FROM trades")
        assert rows == [("#670", 0.155, 10.0, "BUY", "tid_1")]

    async def test_enqueue_bbo_with_nulls(self, store):
        store.enqueue_bbo(now(), None, "#670", 0.155, 100.0, None, None)
        await store.flush()
        rows = store.query("SELECT bid_px, bid_sz, ask_px, ask_sz FROM bbo")
        assert rows == [(0.155, 100.0, None, None)]

    async def test_enqueue_perp_ctx(self, store):
        store.enqueue_perp_ctx(now(), "BTC", 67000.0, 67000.5, 67005.0, 0.0001, 1234.5)
        await store.flush()
        rows = store.query("SELECT coin, mark_px, oracle_px FROM perp_ctx")
        assert rows == [("BTC", 67000.0, 67005.0)]

    async def test_enqueue_raw_ctx(self, store):
        store.enqueue_raw_ctx(now(), "#670", "activeAssetCtx", '{"foo":"bar"}')
        await store.flush()
        rows = store.query("SELECT coin, sub_type, payload_json FROM raw_ctx")
        assert rows == [("#670", "activeAssetCtx", '{"foo":"bar"}')]

    async def test_enqueue_health(self, store):
        store.enqueue_health(now(), True, 24, 50.0, 100, None)
        await store.flush()
        rows = store.query("SELECT ws_connected, n_subs_active, msgs_per_sec FROM health_log")
        assert rows == [(True, 24, 50.0)]

    async def test_buffer_size_aggregates_all_queues(self, store):
        store.enqueue_book_level(now(), None, "#670", "bid", 0, 0.1, 1)
        store.enqueue_trade(now(), None, "#670", 0.1, 1)
        store.enqueue_bbo(now(), None, "#670", 0.1, 1, 0.2, 1)
        store.enqueue_perp_ctx(now(), "BTC", 67000, None, None)
        store.enqueue_raw_ctx(now(), "#670", "x", "{}")
        store.enqueue_health(now(), True, 1, 0.0, 0, None)
        assert store.buffer_size() == 6

    async def test_flush_empty_buffer_returns_zero(self, store):
        flushed = await store.flush()
        assert flushed == 0

    async def test_flush_clears_buffer(self, store):
        for i in range(100):
            store.enqueue_trade(now(), None, "#670", 0.155, float(i))
        assert store.buffer_size() == 100
        await store.flush()
        assert store.buffer_size() == 0


# ─── Background flusher ────────────────────────────────────────────────────────

class TestBackgroundFlusher:

    async def test_flusher_auto_drains(self, store):
        # Use a very short interval so the test is fast
        store.flush_interval_s = 0.1
        await store.start_flusher()
        try:
            for i in range(50):
                store.enqueue_trade(now(), None, "#670", 0.155, float(i))

            # Wait for the flusher to drain
            await asyncio.sleep(0.5)
            assert store.buffer_size() == 0

            rows = store.query("SELECT COUNT(*) FROM trades")
            assert rows[0][0] == 50
        finally:
            store._stopping = True

    async def test_close_does_final_flush(self):
        s = Store(":memory:")
        await s.init()
        await s.start_flusher()
        s.enqueue_trade(now(), None, "#670", 0.155, 10.0)
        await s.close()
        # Reopen the same DB (won't work for :memory: — instead re-query the closed conn)
        # so we use a fresh store on a temp file:

    async def test_close_with_pending_data_flushes(self, tmp_path):
        db_path = str(tmp_path / "obs.duckdb")
        s = Store(db_path)
        await s.init()
        await s.start_flusher()
        s.enqueue_trade(now(), None, "#670", 0.155, 10.0)
        await s.close()

        # Reopen and verify the trade landed
        s2 = Store(db_path)
        await s2.init()
        rows = s2.query("SELECT coin, px FROM trades")
        assert rows == [("#670", 0.155)]
        await s2.close()


# ─── Bulk insert performance sanity check ──────────────────────────────────────

class TestBulkInsert:

    async def test_thousand_trades(self, store):
        for i in range(1000):
            store.enqueue_trade(now(), None, "#670", 0.155, float(i))
        n = await store.flush()
        assert n == 1000
        rows = store.query("SELECT COUNT(*) FROM trades")
        assert rows[0][0] == 1000


# ─── Persistence across reopens (file mode only) ───────────────────────────────

class TestFilePersistence:

    async def test_data_persists_between_sessions(self, tmp_path):
        db_path = str(tmp_path / "obs.duckdb")

        # Session 1: write some data
        s1 = Store(db_path)
        await s1.init()
        s1.enqueue_trade(now(), None, "#670", 0.155, 10.0)
        s1.enqueue_trade(now(), None, "#671", 0.160, 20.0)
        await s1.flush()
        await s1.close()

        # Session 2: read it back
        s2 = Store(db_path)
        await s2.init()
        rows = s2.query("SELECT coin, sz FROM trades ORDER BY coin")
        assert rows == [("#670", 10.0), ("#671", 20.0)]
        # Schema version should still be 1, migrations not re-applied
        v = s2.query("SELECT COUNT(*) FROM _schema_version")
        assert v[0][0] == 1
        await s2.close()
