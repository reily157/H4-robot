"""
DuckDB store with async batched writes.

Design:
    - Single `Store` class owns one DuckDB connection.
    - Producers (WS callbacks) call typed enqueue methods (no I/O).
    - A single background flusher task drains queues every FLUSH_INTERVAL_S
      or when total queued events exceed FLUSH_BATCH_SIZE.
    - DuckDB writes use executemany() for efficiency.
    - All flushes are wrapped in a try/except with retry on lock contention.

Concurrency model:
    Async-pure (no threads). DuckDB's execute() is synchronous but each flush
    is brief (<10ms for ~1000 events), which is acceptable inside an asyncio
    event loop. If profiling later shows lag, this can be moved to a thread
    pool without changing the public API.

Public API:
    store = Store(db_path=":memory:")
    await store.init()                                        # apply migrations
    store.enqueue_book_level(ts_local, coin, side, level_idx, px, sz, ...)
    store.enqueue_trade(...)
    store.enqueue_bbo(...)
    store.enqueue_perp_ctx(...)
    store.enqueue_raw_ctx(...)
    store.enqueue_health(...)
    await store.start_flusher()                               # background task
    ...
    await store.flush()                                       # force flush now
    await store.close()                                       # stop + final flush
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb

from migrations import apply_pending


log = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_FLUSH_INTERVAL_S = 1.0
DEFAULT_FLUSH_BATCH_SIZE = 1000
DEFAULT_LOCK_RETRY_MAX = 5
DEFAULT_LOCK_RETRY_BACKOFF_S = 0.2


# ─── Store ────────────────────────────────────────────────────────────────────

class Store:
    """
    DuckDB-backed event store with async batched writes.

    Use as:
        store = Store("data/observer.duckdb")
        await store.init()
        store.enqueue_trade(...)
        await store.start_flusher()
        ...
        await store.close()
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
        flush_batch_size: int = DEFAULT_FLUSH_BATCH_SIZE,
    ):
        self.db_path = db_path
        self.flush_interval_s = flush_interval_s
        self.flush_batch_size = flush_batch_size

        self._conn: duckdb.DuckDBPyConnection | None = None
        self._flusher_task: asyncio.Task | None = None
        self._stopping = False

        # In-memory queues — one per table.
        # Each queue holds tuples ready for executemany().
        self._q_book_levels: list[tuple] = []
        self._q_trades: list[tuple] = []
        self._q_bbo: list[tuple] = []
        self._q_perp_ctx: list[tuple] = []
        self._q_raw_ctx: list[tuple] = []
        self._q_health: list[tuple] = []

        # Track when each cycle / outcome_map was last written
        # (used to dedup repeated writes from discovery)
        self._cycles_written: set[str] = set()

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Open connection and apply schema migrations."""
        if self._conn is not None:
            return
        # DuckDB handles parent dir creation lazily, but make sure it exists.
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(self.db_path)
        self._conn.execute("SET memory_limit = '512MB'")
        apply_pending(self._conn)
        log.info(f"store initialized: db_path={self.db_path}")

    async def start_flusher(self) -> None:
        """Start the background flusher task."""
        if self._conn is None:
            raise RuntimeError("call init() before start_flusher()")
        if self._flusher_task is not None:
            return
        self._stopping = False
        self._flusher_task = asyncio.create_task(self._flusher_loop())
        log.info("flusher task started")

    async def close(self) -> None:
        """Stop flusher, do final flush, close connection."""
        self._stopping = True
        if self._flusher_task is not None:
            try:
                await asyncio.wait_for(self._flusher_task, timeout=10)
            except asyncio.TimeoutError:
                log.warning("flusher task did not finish in 10s, cancelling")
                self._flusher_task.cancel()
            self._flusher_task = None
        if self._conn is not None:
            await self.flush()
            self._conn.close()
            self._conn = None
        log.info("store closed")

    # ─── Cycle metadata (immediate writes, not batched) ──────────────────

    def write_cycle(
        self,
        cycle_id: str,
        started_at: datetime,
        bucket_question_id: int | None,
        bucket_expiry: datetime | None,
        bucket_thresholds: str | None,
        bucket_underlying: str | None,
        binary_outcome_id: int | None,
        binary_target_price: float | None,
        binary_expiry: datetime | None,
        raw_meta: dict,
    ) -> None:
        """Write cycle metadata (INSERT OR REPLACE semantics)."""
        if self._conn is None:
            raise RuntimeError("store not initialized")
        if cycle_id in self._cycles_written:
            return
        raw_meta_json = json.dumps(raw_meta, separators=(",", ":"))
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cycles
                (cycle_id, started_at, bucket_question_id, bucket_expiry,
                 bucket_thresholds, bucket_underlying,
                 binary_outcome_id, binary_target_price, binary_expiry, raw_meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                cycle_id, started_at,
                bucket_question_id, bucket_expiry,
                bucket_thresholds, bucket_underlying,
                binary_outcome_id, binary_target_price, binary_expiry,
                raw_meta_json,
            ],
        )
        self._cycles_written.add(cycle_id)

    def write_outcome_map(
        self,
        cycle_id: str,
        outcome_id: int,
        role: str,
        yes_coin: str,
        no_coin: str,
        description: str | None,
    ) -> None:
        """Write a single outcome mapping for a cycle (INSERT OR REPLACE)."""
        if self._conn is None:
            raise RuntimeError("store not initialized")
        self._conn.execute(
            """
            INSERT OR REPLACE INTO outcomes_map
                (cycle_id, outcome_id, role, yes_coin, no_coin, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [cycle_id, outcome_id, role, yes_coin, no_coin, description],
        )

    # ─── Enqueue methods (called from WS callbacks, never block) ─────────

    def enqueue_book_level(
        self,
        ts_local: datetime, ts_remote: datetime | None,
        coin: str, side: str, level_idx: int,
        px: float, sz: float, n_orders: int | None = None,
    ) -> None:
        self._q_book_levels.append((ts_local, ts_remote, coin, side, level_idx, px, sz, n_orders))

    def enqueue_trade(
        self,
        ts_local: datetime, ts_remote: datetime | None,
        coin: str, px: float, sz: float,
        side: str | None = None, tid: str | None = None,
    ) -> None:
        self._q_trades.append((ts_local, ts_remote, coin, px, sz, side, tid))

    def enqueue_bbo(
        self,
        ts_local: datetime, ts_remote: datetime | None,
        coin: str,
        bid_px: float | None, bid_sz: float | None,
        ask_px: float | None, ask_sz: float | None,
    ) -> None:
        self._q_bbo.append((ts_local, ts_remote, coin, bid_px, bid_sz, ask_px, ask_sz))

    def enqueue_perp_ctx(
        self,
        ts_local: datetime, coin: str,
        mark_px: float | None, mid_px: float | None, oracle_px: float | None,
        funding: float | None = None, open_interest: float | None = None,
    ) -> None:
        self._q_perp_ctx.append((ts_local, coin, mark_px, mid_px, oracle_px, funding, open_interest))

    def enqueue_raw_ctx(
        self,
        ts_local: datetime, coin: str, sub_type: str, payload_json: str,
    ) -> None:
        self._q_raw_ctx.append((ts_local, coin, sub_type, payload_json))

    def enqueue_health(
        self,
        ts: datetime, ws_connected: bool, n_subs_active: int,
        msgs_per_sec: float, buffer_size: int, last_db_flush: datetime | None,
    ) -> None:
        self._q_health.append((ts, ws_connected, n_subs_active, msgs_per_sec, buffer_size, last_db_flush))

    def buffer_size(self) -> int:
        """Total events queued across all tables."""
        return (
            len(self._q_book_levels)
            + len(self._q_trades)
            + len(self._q_bbo)
            + len(self._q_perp_ctx)
            + len(self._q_raw_ctx)
            + len(self._q_health)
        )

    # ─── Flushing ────────────────────────────────────────────────────────

    async def flush(self) -> int:
        """
        Drain all queues to DuckDB with executemany().
        Returns the number of rows flushed.
        """
        if self._conn is None:
            return 0

        total = 0

        if self._q_book_levels:
            total += await self._flush_batch(
                "INSERT INTO book_levels (ts_local, ts_remote, coin, side, level_idx, px, sz, n_orders) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                self._q_book_levels,
            )
            self._q_book_levels.clear()

        if self._q_trades:
            total += await self._flush_batch(
                "INSERT INTO trades (ts_local, ts_remote, coin, px, sz, side, tid) VALUES (?, ?, ?, ?, ?, ?, ?)",
                self._q_trades,
            )
            self._q_trades.clear()

        if self._q_bbo:
            total += await self._flush_batch(
                "INSERT INTO bbo (ts_local, ts_remote, coin, bid_px, bid_sz, ask_px, ask_sz) VALUES (?, ?, ?, ?, ?, ?, ?)",
                self._q_bbo,
            )
            self._q_bbo.clear()

        if self._q_perp_ctx:
            total += await self._flush_batch(
                "INSERT INTO perp_ctx (ts_local, coin, mark_px, mid_px, oracle_px, funding, open_interest) VALUES (?, ?, ?, ?, ?, ?, ?)",
                self._q_perp_ctx,
            )
            self._q_perp_ctx.clear()

        if self._q_raw_ctx:
            total += await self._flush_batch(
                "INSERT INTO raw_ctx (ts_local, coin, sub_type, payload_json) VALUES (?, ?, ?, ?)",
                self._q_raw_ctx,
            )
            self._q_raw_ctx.clear()

        if self._q_health:
            total += await self._flush_batch(
                "INSERT INTO health_log (ts, ws_connected, n_subs_active, msgs_per_sec, buffer_size, last_db_flush) VALUES (?, ?, ?, ?, ?, ?)",
                self._q_health,
            )
            self._q_health.clear()

        return total

    async def _flush_batch(self, sql: str, rows: list[tuple]) -> int:
        """
        Execute a single executemany() with retry on lock contention.
        """
        if not rows:
            return 0
        attempt = 0
        while True:
            try:
                self._conn.executemany(sql, rows)
                return len(rows)
            except duckdb.Error as e:
                msg = str(e).lower()
                # DuckDB locks are rare in single-connection mode, but handle gracefully
                if "lock" in msg or "busy" in msg:
                    attempt += 1
                    if attempt > DEFAULT_LOCK_RETRY_MAX:
                        log.error(f"flush failed after {attempt} retries: {e}")
                        raise
                    backoff = DEFAULT_LOCK_RETRY_BACKOFF_S * (2 ** (attempt - 1))
                    log.warning(f"db lock, retry {attempt}/{DEFAULT_LOCK_RETRY_MAX} in {backoff}s")
                    await asyncio.sleep(backoff)
                else:
                    raise

    async def _flusher_loop(self) -> None:
        """Background loop: flush every interval or when buffer exceeds batch size."""
        while not self._stopping:
            try:
                await asyncio.sleep(self.flush_interval_s)
                if self.buffer_size() > 0:
                    n = await self.flush()
                    if n > 0:
                        log.debug(f"flushed {n} rows")
                # Force flush if buffer is huge regardless of timer
                while self.buffer_size() >= self.flush_batch_size:
                    n = await self.flush()
                    log.debug(f"forced flush (buffer overflow): {n} rows")
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"flusher error: {e}", exc_info=True)
                # Don't die — keep trying. Backoff briefly.
                await asyncio.sleep(1.0)
        log.info("flusher loop exited")

    # ─── Read helpers (for tests and offline analysis) ───────────────────

    def query(self, sql: str, params: list | None = None) -> list[tuple]:
        """Execute a SELECT and return all rows. Convenience for tests/analysis."""
        if self._conn is None:
            raise RuntimeError("store not initialized")
        cursor = self._conn.execute(sql, params or [])
        return cursor.fetchall()
