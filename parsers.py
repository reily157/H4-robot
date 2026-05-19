"""
Parsers for Hyperliquid WebSocket messages.

Scope:
    Pure functions that convert raw WS JSON payloads into typed dataclasses.
    No I/O, no async, no side effects. Robust against None / missing keys /
    unexpected types.

Supported subscriptions:
    - l2Book           → L2Snapshot
    - trades           → list[Trade]
    - bbo              → Bbo
    - activeAssetCtx   → ActiveAssetCtx (BTC perp only — schema validated live)

Conventions adopted from zaakirio/hl-hip4-arb reference:
    - market_kind tags messages as "PERP" or "HIP4" for downstream routing.
    - Trade.side is normalized: "B" → "BUY", "A" → "SELL" (aggressor).
    - Timestamps are stored as int milliseconds (ts_ms), matching HL convention.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class L2Level:
    price: float
    size: float


@dataclass(frozen=True)
class L2Snapshot:
    ts_ms: int
    market_kind: str        # "PERP" | "HIP4"
    market_id: str          # "#670" for HIP-4, "BTC" for perp
    bids: list[L2Level]     # sorted high → low
    asks: list[L2Level]     # sorted low  → high


@dataclass(frozen=True)
class Trade:
    ts_ms: int
    market_kind: str
    market_id: str
    side: str               # "BUY" | "SELL" (aggressor)
    price: float
    size: float
    trade_id: str | None


@dataclass(frozen=True)
class Bbo:
    ts_ms: int
    market_kind: str
    market_id: str
    bid_px: float | None
    bid_sz: float | None
    ask_px: float | None
    ask_sz: float | None


@dataclass(frozen=True)
class ActiveAssetCtx:
    """
    Generic active context for an asset.
    Fields populated when present in payload; None otherwise.

    Used for BTC perp (mark/oracle/funding) and best-effort on HIP-4 outcomes.
    """
    ts_ms: int
    market_kind: str
    market_id: str
    mark_px: float | None
    mid_px: float | None
    oracle_px: float | None
    funding: float | None
    open_interest: float | None
    raw_payload: str        # JSON dump of the full data block for offline analysis


# ─── Constants ─────────────────────────────────────────────────────────────────

_SIDE_MAP = {"B": "BUY", "A": "SELL"}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    """Convert val to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _require_channel(msg: dict, expected: str) -> dict:
    """Validate the channel field and return the data block."""
    if not isinstance(msg, dict):
        raise ValueError(f"msg must be dict, got {type(msg).__name__}")
    channel = msg.get("channel")
    if channel != expected:
        raise ValueError(f"expected channel={expected!r}, got {channel!r}")
    if "data" not in msg:
        raise ValueError(f"msg missing 'data' key")
    return msg["data"]


def _levels_from_rows(rows) -> list[L2Level]:
    """Convert a list of {px, sz} dicts to L2Level objects, filtering invalid rows."""
    if not isinstance(rows, list):
        return []
    out: list[L2Level] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        px = _safe_float(r.get("px"))
        sz = _safe_float(r.get("sz"))
        if px is None or sz is None:
            continue
        out.append(L2Level(price=px, size=sz))
    return out


# ─── Parsers ───────────────────────────────────────────────────────────────────

def parse_l2_book(msg: dict, market_kind: str) -> L2Snapshot:
    """
    Parse an l2Book WS message.

    Expected payload shape:
        {"channel": "l2Book",
         "data": {"coin": "#670", "time": 1234567890123,
                  "levels": [[bids], [asks]]}}

    Each level is a {"px": "0.155", "sz": "100"} dict.

    Raises:
        ValueError: if channel is wrong, data is missing, or coin/time absent.
    """
    if market_kind not in ("PERP", "HIP4"):
        raise ValueError(f"market_kind must be 'PERP' or 'HIP4', got {market_kind!r}")

    data = _require_channel(msg, "l2Book")

    coin = data.get("coin")
    if coin is None:
        raise ValueError("l2Book data missing 'coin'")
    time_ms = data.get("time")
    if time_ms is None:
        raise ValueError("l2Book data missing 'time'")

    levels = data.get("levels", [[], []])
    if not isinstance(levels, list) or len(levels) < 2:
        bids_raw, asks_raw = [], []
    else:
        bids_raw = levels[0] if isinstance(levels[0], list) else []
        asks_raw = levels[1] if isinstance(levels[1], list) else []

    return L2Snapshot(
        ts_ms=int(time_ms),
        market_kind=market_kind,
        market_id=str(coin),
        bids=_levels_from_rows(bids_raw),
        asks=_levels_from_rows(asks_raw),
    )


def parse_trades(msg: dict, market_kind: str) -> list[Trade]:
    """
    Parse a trades WS message.

    Expected payload shape:
        {"channel": "trades",
         "data": [{"coin": "#670", "time": 1234, "side": "B" | "A",
                   "px": "0.155", "sz": "10", "tid": 12345}, ...]}

    Returns a list (may be empty). Invalid individual trades are skipped
    rather than crashing the whole batch.
    """
    if market_kind not in ("PERP", "HIP4"):
        raise ValueError(f"market_kind must be 'PERP' or 'HIP4', got {market_kind!r}")

    data = _require_channel(msg, "trades")
    if not isinstance(data, list):
        raise ValueError(f"trades data must be list, got {type(data).__name__}")

    out: list[Trade] = []
    for t in data:
        if not isinstance(t, dict):
            continue
        side_code = t.get("side")
        if side_code not in _SIDE_MAP:
            continue
        coin = t.get("coin")
        time_ms = t.get("time")
        px = _safe_float(t.get("px"))
        sz = _safe_float(t.get("sz"))
        if coin is None or time_ms is None or px is None or sz is None:
            continue
        tid_raw = t.get("tid")
        tid = str(tid_raw) if tid_raw is not None else None
        out.append(Trade(
            ts_ms=int(time_ms),
            market_kind=market_kind,
            market_id=str(coin),
            side=_SIDE_MAP[side_code],
            price=px,
            size=sz,
            trade_id=tid,
        ))
    return out


def parse_bbo(msg: dict, market_kind: str) -> Bbo:
    """
    Parse a bbo (best bid/best offer) WS message.

    Expected payload shape (inferred — to be validated with capture_fixtures.py):
        {"channel": "bbo",
         "data": {"coin": "#670", "time": 1234567890,
                  "bbo": [{"px": "0.15", "sz": "100"},      # bid
                          {"px": "0.16", "sz": "120"}]}}    # ask

    A bbo with only one side present (e.g. empty book on one side) returns
    None for that side.
    """
    if market_kind not in ("PERP", "HIP4"):
        raise ValueError(f"market_kind must be 'PERP' or 'HIP4', got {market_kind!r}")

    data = _require_channel(msg, "bbo")

    coin = data.get("coin")
    if coin is None:
        raise ValueError("bbo data missing 'coin'")
    time_ms = data.get("time")
    if time_ms is None:
        raise ValueError("bbo data missing 'time'")

    bbo = data.get("bbo", [None, None])
    if not isinstance(bbo, list) or len(bbo) < 2:
        bid_raw, ask_raw = None, None
    else:
        bid_raw = bbo[0] if isinstance(bbo[0], dict) else None
        ask_raw = bbo[1] if isinstance(bbo[1], dict) else None

    bid_px = _safe_float(bid_raw.get("px")) if bid_raw else None
    bid_sz = _safe_float(bid_raw.get("sz")) if bid_raw else None
    ask_px = _safe_float(ask_raw.get("px")) if ask_raw else None
    ask_sz = _safe_float(ask_raw.get("sz")) if ask_raw else None

    return Bbo(
        ts_ms=int(time_ms),
        market_kind=market_kind,
        market_id=str(coin),
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
    )


def parse_active_asset_ctx(msg: dict, market_kind: str) -> ActiveAssetCtx:
    """
    Parse an activeAssetCtx WS message.

    Expected payload shape (inferred from HL docs — validate with capture_fixtures.py):
        {"channel": "activeAssetCtx",
         "data": {"coin": "BTC", "ctx": {"markPx": "67000.5",
                                          "oraclePx": "67005",
                                          "midPx": "67000",
                                          "funding": "0.0001",
                                          "openInterest": "1234.5",
                                          ...}}}

    Schema is flexible: fields are populated when present, None otherwise.
    Full raw payload is preserved in raw_payload for offline analysis.
    """
    if market_kind not in ("PERP", "HIP4"):
        raise ValueError(f"market_kind must be 'PERP' or 'HIP4', got {market_kind!r}")

    data = _require_channel(msg, "activeAssetCtx")

    coin = data.get("coin")
    if coin is None:
        raise ValueError("activeAssetCtx data missing 'coin'")

    # time may not always be present in activeAssetCtx — fall back to 0
    time_ms = data.get("time", 0)

    ctx = data.get("ctx", {})
    if not isinstance(ctx, dict):
        ctx = {}

    import json
    raw = json.dumps(data, separators=(",", ":"), sort_keys=True)

    return ActiveAssetCtx(
        ts_ms=int(time_ms) if time_ms is not None else 0,
        market_kind=market_kind,
        market_id=str(coin),
        mark_px=_safe_float(ctx.get("markPx")),
        mid_px=_safe_float(ctx.get("midPx")),
        oracle_px=_safe_float(ctx.get("oraclePx")),
        funding=_safe_float(ctx.get("funding")),
        open_interest=_safe_float(ctx.get("openInterest")),
        raw_payload=raw,
    )
