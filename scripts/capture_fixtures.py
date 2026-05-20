#!/usr/bin/env python
"""
Capture live Hyperliquid mainnet WS messages and write fixture files.

Usage:
    python scripts/capture_fixtures.py [--duration 30] [--out tests/fixtures]
    python scripts/capture_fixtures.py --duration 10 --coins "#670,#671,BTC"

Output:
    One JSON file per (channel, coin) subscription, each containing an array
    of raw WS message dicts compatible with parsers.parse_*.

    Example files:
        tests/fixtures/l2Book___670.json
        tests/fixtures/trades___670.json
        tests/fixtures/bbo___670.json
        tests/fixtures/activeAssetCtx__BTC.json

These fixtures replace the synthetic messages in tests/test_parsers.py and
are used to validate parsers against real Hyperliquid payloads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import aiohttp

# Allow `python scripts/capture_fixtures.py` from the project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import codec
from discovery import discover, CycleSpec
from ws_client import WsClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("capture_fixtures")

# Channels to capture for every coin.
_HIP4_CHANNELS = ["l2Book", "trades", "bbo", "activeAssetCtx"]
_PERP_CHANNELS = ["l2Book", "trades", "bbo", "activeAssetCtx"]
_PERP_COIN = "BTC"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _coins_from_cycle(cycle: CycleSpec) -> list[str]:
    """Extract WS coin identifiers from a CycleSpec."""
    coins: set[str] = set()
    if cycle.has_binary:
        yes, no = codec.both_ws_coins(cycle.binary.outcome_id)
        coins.add(yes)
        coins.add(no)
    if cycle.has_bucket:
        for oid in cycle.bucket.named_outcome_ids:
            yes, no = codec.both_ws_coins(oid)
            coins.add(yes)
            coins.add(no)
    return sorted(coins)


def _safe_filename(channel: str, coin: str) -> str:
    """Map (channel, coin) to a safe filename stem."""
    safe_coin = coin.replace("#", "_").replace("+", "p")
    return f"{channel}__{safe_coin}.json"


def _extract_coin(msg: dict) -> str | None:
    """Extract the coin identifier from a raw WS message dict."""
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


# ─── Discovery ────────────────────────────────────────────────────────────────

async def _discover_coins(session: aiohttp.ClientSession) -> list[str]:
    """Discover active HIP-4 WS coins via outcomeMeta."""
    log.info("running discovery …")
    try:
        cycle = await discover(session)
        coins = _coins_from_cycle(cycle)
        if coins:
            log.info(f"discovered {len(coins)} HIP-4 coins: {coins}")
            return coins
        log.warning("discovery returned no coins — check mainnet state")
    except Exception as e:
        log.warning(f"discovery failed: {e} — no HIP-4 coins will be captured")
    return []


# ─── Capture ──────────────────────────────────────────────────────────────────

async def capture(
    duration_s: float,
    out_dir: Path,
    explicit_coins: list[str] | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """
    Connect, subscribe, capture `duration_s` seconds, return buckets.

    Returns:
        dict mapping (channel, coin) → list[raw_message_dict]
    """
    async with aiohttp.ClientSession() as session:
        # Resolve coins.
        if explicit_coins:
            hip4_coins = [c for c in explicit_coins if c != _PERP_COIN]
            log.info(f"using explicit HIP-4 coins: {hip4_coins}")
        else:
            hip4_coins = await _discover_coins(session)

        # Build subscription plan: (channel, coin)
        subs: list[tuple[str, str]] = []
        for coin in hip4_coins:
            for ch in _HIP4_CHANNELS:
                subs.append((ch, coin))
        for ch in _PERP_CHANNELS:
            subs.append((ch, _PERP_COIN))

        if not subs:
            log.error("no subscriptions — aborting")
            return {}

        log.info(f"subscribing to {len(subs)} channel/coin pairs")

        # Initialise buckets (preserves order for the summary).
        buckets: dict[tuple[str, str], list[dict]] = {k: [] for k in subs}
        sub_set = set(subs)

        queue: asyncio.Queue = asyncio.Queue()
        async with WsClient(queue=queue) as ws:
            for ch, coin in subs:
                await ws.subscribe(ch, coin)

            log.info(f"capturing for {duration_s:.0f}s …")
            deadline = asyncio.get_event_loop().time() + duration_s
            msg_total = 0

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    msg = await asyncio.wait_for(
                        queue.get(), timeout=min(remaining, 1.0)
                    )
                except asyncio.TimeoutError:
                    continue

                channel = msg.get("channel")
                if channel == "subscriptionResponse":
                    continue

                coin = _extract_coin(msg)
                if coin is None or channel is None:
                    continue

                key = (channel, coin)
                if key in sub_set:
                    buckets[key].append(msg)
                    msg_total += 1

        log.info(f"capture done — {msg_total} messages routed")
        return buckets


# ─── Write fixtures ───────────────────────────────────────────────────────────

def write_fixtures(
    buckets: dict[tuple[str, str], list[dict]],
    out_dir: Path,
) -> None:
    """Write each bucket to a JSON fixture file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    empty: list[tuple[str, str]] = []

    print(f"\nFixtures written to {out_dir}/")
    print(f"{'File':<45}  {'Messages':>8}")
    print("-" * 55)

    for (channel, coin), messages in buckets.items():
        fname = _safe_filename(channel, coin)
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(messages, fh, indent=2)
        n = len(messages)
        print(f"  {fname:<43}  {n:>8}")
        if n == 0:
            empty.append((channel, coin))

    if empty:
        print(
            f"\nWARNING: {len(empty)} subscription(s) captured 0 messages — "
            "re-run during active trading or increase --duration"
        )
        for ch, coin in empty:
            print(f"  {ch} / {coin}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--duration", type=float, default=30.0,
        help="Seconds to capture (default: 30)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("tests/fixtures"),
        help="Output directory (default: tests/fixtures)",
    )
    p.add_argument(
        "--coins",
        help='Comma-separated WS coin list to override discovery (e.g. "#670,#671,BTC")',
    )
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()
    explicit: list[str] | None = None
    if args.coins:
        explicit = [c.strip() for c in args.coins.split(",") if c.strip()]

    buckets = await capture(
        duration_s=args.duration,
        out_dir=args.out,
        explicit_coins=explicit,
    )
    if buckets:
        write_fixtures(buckets, args.out)


if __name__ == "__main__":
    asyncio.run(_main())
