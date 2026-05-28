"""
Tests for the JSONL-based store.

All tests use a temporary directory (via Store(":memory:") which creates
a tmpdir under the hood) and pytest-asyncio.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

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

    async def test_init_creates_base_dir(self, store):
        """After init the base_dir should exist."""
        assert store.base_dir.exists()
        assert store.base_dir.is_dir()

    async def test_init_is_idempotent(self):
        s = Store(":memory:")
        await s.init()
        await s.init()  # second call should be a no-op
        assert s.base_dir.exists()
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
        # Read cycles file
        path = store.base_dir / "cycles.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["cycle_id"] == "20260520"
        assert rec["binary_target_price"] == 76886.0
        assert rec["bucket_question_id"] == 12

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
        path = store.base_dir / "cycles.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    async def test_write_outcome_map(self, store):
        store.write_outcome_map(
            cycle_id="20260520",
            outcome_id=67,
            role="bucket_idx_0",
            yes_coin="#670",
            no_coin="#671",
            description="index:0",
        )
        path = store.base_dir / "outcomes_map.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["outcome_id"] == 67
        assert rec["yes_coin"] == "#670"

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
        path = store.base_dir / "cycles.jsonl"
        lines = path.read_text().strip().split("\n")
        rec = json.loads(lines[0])
        # raw_meta is nested in the record, not stringified
        assert rec["raw_meta"] == meta


# ─── Enqueue + flush ───────────────────────────────────────────────────────────

class TestEnqueueAndFlush:

    async def test_enqueue_then_flush_book_level(self, store):
        store.enqueue_book_level(now(), None, "#670", "bid", 0, 0.155, 100.0)
        store.enqueue_book_level(now(), None, "#670", "ask", 0, 0.160, 120.0)
        assert store.buffer_size() == 2

        flushed = await store.flush()
        assert flushed == 2
        assert store.buffer_size() == 0

        records = store.read_jsonl("book_levels")
        assert len(records) == 2
        # Sort by side for deterministic check
        records.sort(key=lambda r: r["side"])
        assert records[0]["side"] == "ask"
        assert records[0]["px"] == 0.160
        assert records[1]["side"] == "bid"
        assert records[1]["px"] == 0.155

    async def test_enqueue_trade(self, store):
        store.enqueue_trade(now(), None, "#670", 0.155, 10.0, "BUY", "tid_1")
        await store.flush()
        records = store.read_jsonl("trades")
        assert len(records) == 1
        assert records[0]["coin"] == "#670"
        assert records[0]["px"] == 0.155
        assert records[0]["sz"] == 10.0
        assert records[0]["side"] == "BUY"
        assert records[0]["tid"] == "tid_1"

    async def test_enqueue_bbo_with_nulls(self, store):
        store.enqueue_bbo(now(), None, "#670", 0.155, 100.0, None, None)
        await store.flush()
        records = store.read_jsonl("bbo")
        assert len(records) == 1
        assert records[0]["bid_px"] == 0.155
        assert records[0]["bid_sz"] == 100.0
        assert records[0]["ask_px"] is None
        assert records[0]["ask_sz"] is None

    async def test_enqueue_perp_ctx(self, store):
        store.enqueue_perp_ctx(now(), "BTC", 67000.0, 67000.5, 67005.0, 0.0001, 1234.5)
        await store.flush()
        records = store.read_jsonl("perp_ctx")
        assert len(records) == 1
        assert records[0]["coin"] == "BTC"
        assert records[0]["mark_px"] == 67000.0
        assert records[0]["oracle_px"] == 67005.0

    async def test_enqueue_raw_ctx(self, store):
        store.enqueue_raw_ctx(now(), "#670", "activeAssetCtx", '{"foo":"bar"}')
        await store.flush()
        records = store.read_jsonl("raw_ctx")
        assert len(records) == 1
        assert records[0]["coin"] == "#670"
        assert records[0]["sub_type"] == "activeAssetCtx"
        assert records[0]["payload_json"] == '{"foo":"bar"}'

    async def test_enqueue_health(self, store):
        store.enqueue_health(now(), True, 24, 50.0, 100, None)
        await store.flush()
        records = store.read_jsonl("health")
        assert len(records) == 1
        assert records[0]["ws_connected"] is True
        assert records[0]["n_subs_active"] == 24
        assert records[0]["msgs_per_sec"] == 50.0

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

            records = store.read_jsonl("trades")
            assert len(records) == 50
        finally:
            store._stopping = True

    async def test_close_with_pending_data_flushes(self, tmp_path):
        """Final flush on close should persist data to disk."""
        db_dir = str(tmp_path)
        s = Store(db_dir)
        await s.init()
        await s.start_flusher()
        s.enqueue_trade(now(), None, "#670", 0.155, 10.0)
        await s.close()

        # Reopen the directory and verify the trade landed
        s2 = Store(db_dir)
        await s2.init()
        records = s2.read_jsonl("trades")
        assert len(records) == 1
        assert records[0]["coin"] == "#670"
        assert records[0]["px"] == 0.155
        await s2.close()


# ─── Bulk insert performance sanity check ──────────────────────────────────────

class TestBulkInsert:

    async def test_thousand_trades(self, store):
        for i in range(1000):
            store.enqueue_trade(now(), None, "#670", 0.155, float(i))
        n = await store.flush()
        assert n == 1000
        records = store.read_jsonl("trades")
        assert len(records) == 1000


# ─── Persistence across reopens ────────────────────────────────────────────────

class TestFilePersistence:

    async def test_data_persists_between_sessions(self, tmp_path):
        db_dir = str(tmp_path)

        # Session 1: write some data
        s1 = Store(db_dir)
        await s1.init()
        s1.enqueue_trade(now(), None, "#670", 0.155, 10.0)
        s1.enqueue_trade(now(), None, "#671", 0.160, 20.0)
        await s1.flush()
        await s1.close()

        # Session 2: read it back
        s2 = Store(db_dir)
        await s2.init()
        records = s2.read_jsonl("trades")
        records.sort(key=lambda r: r["coin"])
        assert len(records) == 2
        assert records[0]["coin"] == "#670"
        assert records[0]["sz"] == 10.0
        assert records[1]["coin"] == "#671"
        assert records[1]["sz"] == 20.0
        await s2.close()


# ─── JSONL date rollover ───────────────────────────────────────────────────────

class TestDateRollover:
    """Events near midnight UTC should split cleanly into separate daily files."""

    async def test_events_split_by_utc_date(self, store):
        # Two events: one on May 27 23:59, one on May 28 00:01
        ts1 = datetime(2026, 5, 27, 23, 59, 30, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 28, 0, 0, 30, tzinfo=timezone.utc)
        store.enqueue_trade(ts1, None, "#670", 0.155, 10.0)
        store.enqueue_trade(ts2, None, "#670", 0.156, 11.0)
        await store.flush()

        records_27 = store.read_jsonl("trades", "20260527")
        records_28 = store.read_jsonl("trades", "20260528")
        assert len(records_27) == 1
        assert len(records_28) == 1
        assert records_27[0]["sz"] == 10.0
        assert records_28[0]["sz"] == 11.0