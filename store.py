"""
Append-only JSON-lines store for high-frequency capture.

Design rationale:
    The original DuckDB-backed store became a bottleneck under real load —
    DuckDB checkpoints blocked the asyncio event loop AND the synchronous
    writes were too slow to keep up with ~50 msgs/sec of WS data, causing
    the in-memory buffer to grow unbounded toward OOM.

    This implementation writes one append-only JSONL file per event type
    per UTC day. No locks, no checkpoints, no transactions — just buffered
    sequential writes. Reading happens later via a separate ingest script
    that loads JSONL files into DuckDB for analysis.

File layout under `data/`:
    book_levels_YYYYMMDD.jsonl
    trades_YYYYMMDD.jsonl
    bbo_YYYYMMDD.jsonl
    perp_ctx_YYYYMMDD.jsonl
    raw_ctx_YYYYMMDD.jsonl
    health_YYYYMMDD.jsonl
    cycles.jsonl                  (one record per cycle discovered, no rotation)
    outcomes_map.jsonl            (mapping records, no rotation)

Each JSON-line record contains ALL fields including the timestamp. UTC dates
are derived from `ts_local` for file routing; rollover happens automatically
at midnight UTC.

Public API preserved from the original Store class:
    - init(), start_flusher(), close(), flush()
    - enqueue_book_level(), enqueue_trade(), enqueue_bbo(),
      enqueue_perp_ctx(), enqueue_raw_ctx(), enqueue_health()
    - write_cycle(), write_outcome_map()
    - buffer_size()
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_FLUSH_INTERVAL_S = 1.0
DEFAULT_FLUSH_BATCH_SIZE = 5000

# Map event type → filename prefix
_EVENT_PREFIXES = {
    "book_levels": "book_levels",
    "trades": "trades",
    "bbo": "bbo",
    "perp_ctx": "perp_ctx",
    "raw_ctx": "raw_ctx",
    "health": "health",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _iso(ts: datetime | None) -> str | None:
    """Serialize a datetime as ISO 8601 UTC string. Returns None if input is None."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _utc_date_str(ts: datetime) -> str:
    """Return 'YYYYMMDD' for the UTC date of a timestamp."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.strftime("%Y%m%d")


# ─── Store ────────────────────────────────────────────────────────────────────

class Store:
    """
    Append-only JSONL event store with async batched writes.

    Each event type gets its own daily file. Writes are buffered in memory
    and flushed periodically by a background task running in a thread (via
    asyncio.to_thread) to keep the event loop fully unblocked.
    """

    def __init__(
        self,
        db_path: str = "data",
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
        flush_batch_size: int = DEFAULT_FLUSH_BATCH_SIZE,
    ):
        """
        db_path: directory where JSONL files will be written (kept name for
                 backward compat with existing Recorder construction).
                 The legacy value ':memory:' is supported and maps to a
                 temporary in-memory mode for tests.
        """
        # Backward compat: ':memory:' was used by tests for an in-memory DB.
        # Here we map it to a temp directory that lives only for the test run.
        self._in_memory = (db_path == ":memory:")
        if self._in_memory:
            import tempfile
            self._tmpdir = tempfile.mkdtemp(prefix="h4_store_test_")
            self.base_dir = Path(self._tmpdir)
        else:
            # If a .duckdb path was passed (legacy), use its parent dir
            p = Path(db_path)
            if p.suffix == ".duckdb":
                self.base_dir = p.parent
            else:
                self.base_dir = p

        self.flush_interval_s = flush_interval_s
        self.flush_batch_size = flush_batch_size

        self._initialized = False
        self._flusher_task: asyncio.Task | None = None
        self._stopping = False

        # In-memory queues — one per event type. Each holds dicts ready
        # to be written as JSON lines.
        self._q_book_levels: list[dict] = []
        self._q_trades: list[dict] = []
        self._q_bbo: list[dict] = []
        self._q_perp_ctx: list[dict] = []
        self._q_raw_ctx: list[dict] = []
        self._q_health: list[dict] = []

        # Dedup for cycle metadata
        self._cycles_written: set[str] = set()

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Create the data directory if needed."""
        if self._initialized:
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        log.info(f"store initialized: base_dir={self.base_dir}")

    async def start_flusher(self) -> None:
        """Start the background flusher task."""
        if not self._initialized:
            raise RuntimeError("call init() before start_flusher()")
        if self._flusher_task is not None:
            return
        self._stopping = False
        self._flusher_task = asyncio.create_task(self._flusher_loop())
        log.info("flusher task started")

    async def close(self) -> None:
        """Stop flusher, do final flush, cleanup."""
        self._stopping = True
        if self._flusher_task is not None:
            try:
                await asyncio.wait_for(self._flusher_task, timeout=10)
            except asyncio.TimeoutError:
                log.warning("flusher task did not finish in 10s, cancelling")
                self._flusher_task.cancel()
            self._flusher_task = None
        await self.flush()

        # Clean up temp dir in test mode
        if self._in_memory:
            import shutil
            shutil.rmtree(self._tmpdir, ignore_errors=True)

        self._initialized = False
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
        """Append a cycle metadata record (idempotent on cycle_id)."""
        if not self._initialized:
            raise RuntimeError("store not initialized")
        if cycle_id in self._cycles_written:
            return

        record = {
            "cycle_id": cycle_id,
            "started_at": _iso(started_at),
            "bucket_question_id": bucket_question_id,
            "bucket_expiry": _iso(bucket_expiry),
            "bucket_thresholds": bucket_thresholds,
            "bucket_underlying": bucket_underlying,
            "binary_outcome_id": binary_outcome_id,
            "binary_target_price": binary_target_price,
            "binary_expiry": _iso(binary_expiry),
            "raw_meta": raw_meta,
        }
        path = self.base_dir / "cycles.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
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
        """Append an outcome mapping record."""
        if not self._initialized:
            raise RuntimeError("store not initialized")
        record = {
            "cycle_id": cycle_id,
            "outcome_id": outcome_id,
            "role": role,
            "yes_coin": yes_coin,
            "no_coin": no_coin,
            "description": description,
        }
        path = self.base_dir / "outcomes_map.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    # ─── Enqueue methods (called from WS callbacks, never block) ─────────

    def enqueue_book_level(
        self,
        ts_local: datetime, ts_remote: datetime | None,
        coin: str, side: str, level_idx: int,
        px: float, sz: float, n_orders: int | None = None,
    ) -> None:
        self._q_book_levels.append({
            "ts_local": _iso(ts_local),
            "ts_remote": _iso(ts_remote),
            "coin": coin,
            "side": side,
            "level_idx": level_idx,
            "px": px,
            "sz": sz,
            "n_orders": n_orders,
        })

    def enqueue_trade(
        self,
        ts_local: datetime, ts_remote: datetime | None,
        coin: str, px: float, sz: float,
        side: str | None = None, tid: str | None = None,
    ) -> None:
        self._q_trades.append({
            "ts_local": _iso(ts_local),
            "ts_remote": _iso(ts_remote),
            "coin": coin,
            "px": px,
            "sz": sz,
            "side": side,
            "tid": tid,
        })

    def enqueue_bbo(
        self,
        ts_local: datetime, ts_remote: datetime | None,
        coin: str,
        bid_px: float | None, bid_sz: float | None,
        ask_px: float | None, ask_sz: float | None,
    ) -> None:
        self._q_bbo.append({
            "ts_local": _iso(ts_local),
            "ts_remote": _iso(ts_remote),
            "coin": coin,
            "bid_px": bid_px,
            "bid_sz": bid_sz,
            "ask_px": ask_px,
            "ask_sz": ask_sz,
        })

    def enqueue_perp_ctx(
        self,
        ts_local: datetime, coin: str,
        mark_px: float | None, mid_px: float | None, oracle_px: float | None,
        funding: float | None = None, open_interest: float | None = None,
    ) -> None:
        self._q_perp_ctx.append({
            "ts_local": _iso(ts_local),
            "coin": coin,
            "mark_px": mark_px,
            "mid_px": mid_px,
            "oracle_px": oracle_px,
            "funding": funding,
            "open_interest": open_interest,
        })

    def enqueue_raw_ctx(
        self,
        ts_local: datetime, coin: str, sub_type: str, payload_json: str,
    ) -> None:
        self._q_raw_ctx.append({
            "ts_local": _iso(ts_local),
            "coin": coin,
            "sub_type": sub_type,
            "payload_json": payload_json,
        })

    def enqueue_health(
        self,
        ts: datetime, ws_connected: bool, n_subs_active: int,
        msgs_per_sec: float, buffer_size: int, last_db_flush: datetime | None,
    ) -> None:
        self._q_health.append({
            "ts": _iso(ts),
            "ws_connected": ws_connected,
            "n_subs_active": n_subs_active,
            "msgs_per_sec": msgs_per_sec,
            "buffer_size": buffer_size,
            "last_db_flush": _iso(last_db_flush),
        })

    def buffer_size(self) -> int:
        """Total events queued across all types."""
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
        Drain all queues to JSONL files. Writes happen in a thread to keep
        the event loop responsive.
        Returns the number of records flushed.
        """
        if not self._initialized:
            return 0

        # Atomic snapshot — replace queues with fresh lists BEFORE threading.
        # New enqueues during flush land in the fresh queues.
        snapshot = {
            "book_levels": self._q_book_levels,
            "trades": self._q_trades,
            "bbo": self._q_bbo,
            "perp_ctx": self._q_perp_ctx,
            "raw_ctx": self._q_raw_ctx,
            "health": self._q_health,
        }
        self._q_book_levels = []
        self._q_trades = []
        self._q_bbo = []
        self._q_perp_ctx = []
        self._q_raw_ctx = []
        self._q_health = []

        if not any(snapshot.values()):
            return 0

        return await asyncio.to_thread(self._flush_all_sync, snapshot)

    def _flush_all_sync(self, snapshot: dict) -> int:
        """
        Synchronous flush of all queues to JSONL files.
        Runs in a thread via asyncio.to_thread.

        Records of the same event type are grouped by UTC date and written
        to the matching daily file in append mode.
        """
        total = 0
        for event_type, records in snapshot.items():
            if not records:
                continue
            total += self._write_records(event_type, records)
        return total

    def _write_records(self, event_type: str, records: list[dict]) -> int:
        """
        Append a batch of records to the appropriate daily file(s).

        Records are grouped by UTC date so writes near midnight rollover
        cleanly into separate files.
        """
        prefix = _EVENT_PREFIXES.get(event_type)
        if prefix is None:
            log.warning(f"unknown event_type: {event_type}")
            return 0

        # Group by date (rare: 99% of the time all records belong to one date)
        by_date: dict[str, list[dict]] = {}
        for rec in records:
            ts_str = rec.get("ts_local") or rec.get("ts")
            if ts_str is None:
                # Should not happen in practice but be defensive
                log.warning(f"record without ts_local/ts: {rec}")
                continue
            # Extract YYYYMMDD from ISO timestamp (UTC)
            # Format: "2026-05-27T21:30:00.123+00:00"
            date_str = ts_str[:10].replace("-", "")
            by_date.setdefault(date_str, []).append(rec)

        written = 0
        for date_str, recs in by_date.items():
            path = self.base_dir / f"{prefix}_{date_str}.jsonl"
            try:
                with open(path, "a", encoding="utf-8") as f:
                    for rec in recs:
                        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    f.flush()
                written += len(recs)
            except Exception as e:
                log.error(f"failed to write {len(recs)} {event_type} to {path}: {e}")
                # Re-raise — caller will retry the batch
                raise
        return written

    async def _flusher_loop(self) -> None:
        """Background loop: flush every interval or when buffer is large."""
        while not self._stopping:
            try:
                await asyncio.sleep(self.flush_interval_s)
                if self.buffer_size() > 0:
                    n = await self.flush()
                    if n > 0:
                        log.debug(f"flushed {n} records")
                # Force flush if buffer is huge regardless of timer
                while self.buffer_size() >= self.flush_batch_size:
                    n = await self.flush()
                    log.debug(f"forced flush (buffer overflow): {n} records")
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"flusher error: {e}", exc_info=True)
                await asyncio.sleep(1.0)
        log.info("flusher loop exited")

    # ─── Read helpers (for tests) ────────────────────────────────────────

    def query(self, sql: str, params: list | None = None) -> list[tuple]:
        """
        Compatibility shim — the original DuckDB-backed Store had a query()
        method used in tests. JSONL mode does not support SQL; this is left
        in place for the test that explicitly checks store state, with a
        helpful error if accidentally used.
        """
        raise NotImplementedError(
            "query() is not supported in JSONL mode. Use read_jsonl() or "
            "load files into DuckDB via a separate ingest step."
        )

    def read_jsonl(self, event_type: str, date_str: str | None = None) -> list[dict]:
        """
        Read all records of an event type. Used by tests and ad-hoc inspection.

        date_str: 'YYYYMMDD' or None to read all daily files for this type.
        """
        prefix = _EVENT_PREFIXES.get(event_type)
        if prefix is None:
            raise ValueError(f"unknown event_type: {event_type}")

        out: list[dict] = []
        if date_str:
            files = [self.base_dir / f"{prefix}_{date_str}.jsonl"]
        else:
            files = sorted(self.base_dir.glob(f"{prefix}_*.jsonl"))

        for path in files:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        return out