"""
HIP-4 observer recorder — top-level orchestrator.

Lifecycle:
    rec = Recorder(db_path="data/observer.duckdb")
    spec = await rec.init()      # discovery → subscribe → start health
    await rec.run(duration_s=60) # process WS messages until deadline
    await rec.close()            # flush → stop

Pure observer: no signal generation, no order submission, no fair-value output.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

import aiohttp

import codec
from discovery import CycleSpec, discover
from health import HealthServer
from parsers import (
    parse_active_asset_ctx,
    parse_bbo,
    parse_l2_book,
    parse_trades,
)
from store import Store
from ws_client import WsClient, WS_URL


log = logging.getLogger(__name__)

_HIP4_CHANNELS = ("l2Book", "trades", "bbo", "activeAssetCtx")
_PERP_COIN = "BTC"


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _market_kind(coin: str) -> str:
    return "HIP4" if coin.startswith("#") else "PERP"


def _ts_from_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _cycle_id(spec: CycleSpec, underlying: str) -> str:
    if spec.has_bucket:
        expiry = spec.bucket.expiry
    elif spec.has_binary:
        expiry = spec.binary.expiry
    else:
        expiry = None
    if expiry:
        return f"{underlying}_{expiry.strftime('%Y%m%d%H%M')}"
    return f"{underlying}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


def _coins_from_cycle(spec: CycleSpec) -> list[str]:
    """Return sorted WS coin identifiers for all outcomes in the CycleSpec."""
    coins: set[str] = set()
    if spec.has_binary:
        y, n = codec.both_ws_coins(spec.binary.outcome_id)
        coins |= {y, n}
    if spec.has_bucket:
        for oid in spec.bucket.named_outcome_ids:
            y, n = codec.both_ws_coins(oid)
            coins |= {y, n}
        fb = spec.bucket.fallback_outcome_id
        y, n = codec.both_ws_coins(fb)
        coins |= {y, n}
    return sorted(coins)


def _extract_coin(msg: dict) -> str | None:
    """Extract the coin identifier from a raw WS message."""
    channel = msg.get("channel")
    data = msg.get("data")
    if channel in ("l2Book", "bbo", "activeAssetCtx"):
        if isinstance(data, dict):
            return data.get("coin")
    elif channel == "trades":
        if isinstance(data, list) and data:
            first = data[0]
            return first.get("coin") if isinstance(first, dict) else None
    return None


def _write_outcome_map(store: Store, cycle_id: str, spec: CycleSpec) -> None:
    """Write all outcome → coin mappings for the cycle."""
    if spec.has_binary:
        oid = spec.binary.outcome_id
        y, n = codec.both_ws_coins(oid)
        store.write_outcome_map(cycle_id, oid, "binary", y, n, None)

    if spec.has_bucket:
        for i, oid in enumerate(spec.bucket.named_outcome_ids):
            y, n = codec.both_ws_coins(oid)
            store.write_outcome_map(cycle_id, oid, f"named_{i}", y, n, None)
        fb = spec.bucket.fallback_outcome_id
        y, n = codec.both_ws_coins(fb)
        store.write_outcome_map(cycle_id, fb, "fallback", y, n, None)


# ─── Rate counter ─────────────────────────────────────────────────────────────

class _RateCounter:
    """Rolling message-per-second counter over a sliding window."""

    def __init__(self, window_s: float = 5.0) -> None:
        self._window = window_s
        self._times: deque[float] = deque()

    def tick(self) -> None:
        now = asyncio.get_event_loop().time()
        self._times.append(now)
        cutoff = now - self._window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    def rate(self) -> float:
        if not self._times:
            return 0.0
        now = asyncio.get_event_loop().time()
        cutoff = now - self._window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
        if not self._times:
            return 0.0
        elapsed = min(self._window, now - self._times[0])
        return len(self._times) / elapsed if elapsed > 0 else float(len(self._times))


# ─── Recorder ─────────────────────────────────────────────────────────────────

class Recorder:
    """
    Orchestrates discovery → WS subscribe → parse → store for one observer cycle.

    Scope: pure data collection. No signals, no orders, no fair-value output.
    """

    def __init__(
        self,
        db_path: str = "data/observer.duckdb",
        health_port: int = 8765,
        underlying: str = "BTC",
        ws_url: str = WS_URL,
    ) -> None:
        self._db_path = db_path
        self._health_port = health_port
        self._underlying = underlying
        self._ws_url = ws_url

        self._queue: asyncio.Queue = asyncio.Queue()
        self._store: Store | None = None
        self._ws: WsClient | None = None
        self._health: HealthServer | None = None
        self._rate = _RateCounter()
        self._cycle_id: str = ""

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def init(self) -> CycleSpec:
        """
        Open store, run discovery, register cycle + outcomes, subscribe WS.
        Returns the CycleSpec for inspection.
        """
        # 1. Store
        self._store = Store(self._db_path)
        await self._store.init()

        # 2. Discovery
        log.info(f"discovering cycle for underlying={self._underlying!r}")
        async with aiohttp.ClientSession() as session:
            spec = await discover(session, underlying=self._underlying)
        log.info(f"cycle: complete={spec.is_complete} bucket={spec.has_bucket} binary={spec.has_binary}")

        # 3. Persist cycle metadata
        cid = _cycle_id(spec, self._underlying)
        self._cycle_id = cid
        self._store.write_cycle(
            cycle_id=cid,
            started_at=datetime.now(timezone.utc),
            bucket_question_id=spec.bucket.question_id if spec.has_bucket else None,
            bucket_expiry=spec.bucket.expiry if spec.has_bucket else None,
            bucket_thresholds=(
                ",".join(str(t) for t in spec.bucket.thresholds) if spec.has_bucket else None
            ),
            bucket_underlying=spec.bucket.underlying if spec.has_bucket else None,
            binary_outcome_id=spec.binary.outcome_id if spec.has_binary else None,
            binary_target_price=spec.binary.target_price if spec.has_binary else None,
            binary_expiry=spec.binary.expiry if spec.has_binary else None,
            raw_meta=spec.raw_meta,
        )
        _write_outcome_map(self._store, cid, spec)

        # 4. WS subscriptions
        hip4_coins = _coins_from_cycle(spec)
        log.info(f"subscribing to {len(hip4_coins)} HIP-4 coins + {_PERP_COIN} perp")

        self._ws = WsClient(url=self._ws_url, queue=self._queue)
        await self._ws.start()

        for coin in hip4_coins:
            for ch in _HIP4_CHANNELS:
                await self._ws.subscribe(ch, coin)
        for ch in _HIP4_CHANNELS:
            await self._ws.subscribe(ch, _PERP_COIN)

        # 5. Background flusher + health server
        await self._store.start_flusher()
        self._health = HealthServer(
            stats_fn=self._get_stats,
            port=self._health_port,
        )
        await self._health.start()

        return spec

    async def run(self, duration_s: float | None = None) -> None:
        """
        Process WS messages until `duration_s` elapses (or forever if None).
        Exits cleanly on CancelledError (SIGINT from asyncio.run).
        """
        if self._store is None or self._ws is None:
            raise RuntimeError("call init() before run()")

        loop = asyncio.get_event_loop()
        deadline = (loop.time() + duration_s) if duration_s is not None else None
        log.info(
            f"recorder running"
            + (f" for {duration_s:.0f}s" if duration_s else " until cancelled")
        )
        msgs = 0

        while True:
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                timeout = min(remaining, 1.0)
            else:
                timeout = 1.0

            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                log.info("recorder run cancelled")
                raise

            self._dispatch(msg)
            msgs += 1
        log.info(f"recorder run finished — {msgs} messages processed")

    async def close(self) -> None:
        """Stop WsClient → HealthServer → Store (final flush)."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._health is not None:
            await self._health.stop()
            self._health = None
        if self._store is not None:
            await self._store.close()
            self._store = None
        log.info("recorder closed")

    # ─── Properties ──────────────────────────────────────────────────────

    @property
    def health_url(self) -> str | None:
        """URL of the health server once started, else None."""
        return self._health.url if self._health is not None else None

    # ─── Stats (consumed by HealthServer) ────────────────────────────────

    def _get_stats(self) -> dict:
        return {
            "ws_connected": self._ws.connected if self._ws else False,
            "n_subs_active": self._ws.n_subscriptions if self._ws else 0,
            "msgs_per_sec": round(self._rate.rate(), 2),
            "buffer_size": self._store.buffer_size() if self._store else 0,
        }

    # ─── Dispatch ─────────────────────────────────────────────────────────

    def _dispatch(self, msg: dict) -> None:
        """Route one raw WS message to the appropriate store enqueue method."""
        channel = msg.get("channel")
        if channel is None or channel == "subscriptionResponse":
            return

        coin = _extract_coin(msg)
        if coin is None:
            return

        market_kind = _market_kind(coin)
        ts_local = datetime.now(timezone.utc)
        self._rate.tick()

        try:
            if channel == "l2Book":
                self._handle_l2book(msg, market_kind, ts_local)
            elif channel == "trades":
                self._handle_trades(msg, market_kind, ts_local)
            elif channel == "bbo":
                self._handle_bbo(msg, market_kind, ts_local)
            elif channel == "activeAssetCtx":
                self._handle_ctx(msg, market_kind, ts_local)
        except Exception as e:
            log.warning(f"dispatch error {channel}/{coin}: {e}")

    def _handle_l2book(
        self, msg: dict, market_kind: str, ts_local: datetime
    ) -> None:
        assert self._store is not None
        snap = parse_l2_book(msg, market_kind)
        ts_remote = _ts_from_ms(snap.ts_ms)
        for i, lvl in enumerate(snap.bids):
            self._store.enqueue_book_level(
                ts_local, ts_remote, snap.market_id, "bid", i, lvl.price, lvl.size
            )
        for i, lvl in enumerate(snap.asks):
            self._store.enqueue_book_level(
                ts_local, ts_remote, snap.market_id, "ask", i, lvl.price, lvl.size
            )

    def _handle_trades(
        self, msg: dict, market_kind: str, ts_local: datetime
    ) -> None:
        assert self._store is not None
        for t in parse_trades(msg, market_kind):
            self._store.enqueue_trade(
                ts_local, _ts_from_ms(t.ts_ms),
                t.market_id, t.price, t.size, t.side, t.trade_id,
            )

    def _handle_bbo(
        self, msg: dict, market_kind: str, ts_local: datetime
    ) -> None:
        assert self._store is not None
        bbo = parse_bbo(msg, market_kind)
        self._store.enqueue_bbo(
            ts_local, _ts_from_ms(bbo.ts_ms),
            bbo.market_id, bbo.bid_px, bbo.bid_sz, bbo.ask_px, bbo.ask_sz,
        )

    def _handle_ctx(
        self, msg: dict, market_kind: str, ts_local: datetime
    ) -> None:
        assert self._store is not None
        ctx = parse_active_asset_ctx(msg, market_kind)
        if market_kind == "PERP":
            self._store.enqueue_perp_ctx(
                ts_local, ctx.market_id,
                ctx.mark_px, ctx.mid_px, ctx.oracle_px,
                ctx.funding, ctx.open_interest,
            )
        else:
            self._store.enqueue_raw_ctx(
                ts_local, ctx.market_id, "activeAssetCtx", ctx.raw_payload
            )





# ─── Entry point ───────────────────────────────────────────────────────────────



if __name__ == "__main__":

    import asyncio

    import logging

    import os

    from pathlib import Path



    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",

    )



    DB_PATH = os.environ.get("H4_DB_PATH", "data/observer.duckdb")

    HEALTH_PORT = int(os.environ.get("H4_HEALTH_PORT", "8765"))



    Path("data").mkdir(exist_ok=True)

    Path("logs").mkdir(exist_ok=True)



    async def _main():

        rec = Recorder(db_path=DB_PATH, health_port=HEALTH_PORT)

        await rec.init()

        # Run indefinitely (no duration_s param = run until interrupted)

        try:

            await rec.run()

        except KeyboardInterrupt:

            pass

        finally:

            await rec.close()



    asyncio.run(_main())

