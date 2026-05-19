"""
Asset encoding for Hyperliquid HIP-4 outcome markets.

HIP-4 uses three coexisting representations of the same logical asset:

    enc        = 10 * outcome_id + side     (side: 0=YES, 1=NO)
    ws_coin    = "#" + str(enc)             (WebSocket subscriptions)
    spot_coin  = "+" + str(enc)             (Spot balance keys)
    asset_int  = 100_000_000 + enc          (Order action asset integer)

This module is the single source of truth for these conversions.
It is pure (no I/O, no async, no side effects) and fully covered by
round-trip tests.

Validation rules:
    - outcome_id must be a non-negative int
    - side must be 0 or 1
    - ws_coin / spot_coin strings must match exact prefix + int format
    - asset_int must be >= 100_000_000

All invalid inputs raise ValueError with a descriptive message.
"""

from __future__ import annotations

# ─── Constants ─────────────────────────────────────────────────────────────────

ASSET_INT_OFFSET = 100_000_000
SIDE_YES = 0
SIDE_NO = 1
WS_PREFIX = "#"
SPOT_PREFIX = "+"


# ─── Forward encoding ──────────────────────────────────────────────────────────

def encode(outcome_id: int, side: int) -> int:
    """
    Encode (outcome_id, side) into the canonical `enc` integer.

    enc = 10 * outcome_id + side

    Examples:
        encode(67, 0) → 670   (outcome 67, YES side)
        encode(67, 1) → 671   (outcome 67, NO side)
        encode(0, 0)  → 0     (edge case: outcome 0 YES)

    Raises:
        ValueError: if outcome_id < 0 or side not in {0, 1}.
    """
    if not isinstance(outcome_id, int) or isinstance(outcome_id, bool):
        raise ValueError(f"outcome_id must be int, got {type(outcome_id).__name__}")
    if outcome_id < 0:
        raise ValueError(f"outcome_id must be >= 0, got {outcome_id}")
    if not isinstance(side, int) or isinstance(side, bool):
        raise ValueError(f"side must be int, got {type(side).__name__}")
    if side not in (SIDE_YES, SIDE_NO):
        raise ValueError(f"side must be 0 (YES) or 1 (NO), got {side}")
    return 10 * outcome_id + side


def ws_coin(enc: int) -> str:
    """
    Format `enc` as a WebSocket coin identifier.

    Examples:
        ws_coin(670) → "#670"
        ws_coin(0)   → "#0"

    Raises:
        ValueError: if enc < 0.
    """
    if not isinstance(enc, int) or isinstance(enc, bool):
        raise ValueError(f"enc must be int, got {type(enc).__name__}")
    if enc < 0:
        raise ValueError(f"enc must be >= 0, got {enc}")
    return f"{WS_PREFIX}{enc}"


def spot_coin(enc: int) -> str:
    """
    Format `enc` as a Spot balance coin identifier.

    Examples:
        spot_coin(670) → "+670"
    """
    if not isinstance(enc, int) or isinstance(enc, bool):
        raise ValueError(f"enc must be int, got {type(enc).__name__}")
    if enc < 0:
        raise ValueError(f"enc must be >= 0, got {enc}")
    return f"{SPOT_PREFIX}{enc}"


def asset_int(enc: int) -> int:
    """
    Convert `enc` to the Order action asset integer.

    asset_int = 100_000_000 + enc

    Examples:
        asset_int(670) → 100_000_670
    """
    if not isinstance(enc, int) or isinstance(enc, bool):
        raise ValueError(f"enc must be int, got {type(enc).__name__}")
    if enc < 0:
        raise ValueError(f"enc must be >= 0, got {enc}")
    return ASSET_INT_OFFSET + enc


# ─── Reverse decoding ──────────────────────────────────────────────────────────

def decode_enc(enc: int) -> tuple[int, int]:
    """
    Decode `enc` back into (outcome_id, side).

    Examples:
        decode_enc(670) → (67, 0)
        decode_enc(671) → (67, 1)
        decode_enc(0)   → (0, 0)

    Raises:
        ValueError: if enc < 0.
    """
    if not isinstance(enc, int) or isinstance(enc, bool):
        raise ValueError(f"enc must be int, got {type(enc).__name__}")
    if enc < 0:
        raise ValueError(f"enc must be >= 0, got {enc}")
    outcome_id, side = divmod(enc, 10)
    return outcome_id, side


def decode_ws_coin(coin: str) -> tuple[int, int]:
    """
    Decode a WebSocket coin string back into (outcome_id, side).

    Examples:
        decode_ws_coin("#670") → (67, 0)
        decode_ws_coin("#0")   → (0, 0)

    Raises:
        ValueError: if string does not start with "#" or suffix is not int.
    """
    if not isinstance(coin, str):
        raise ValueError(f"coin must be str, got {type(coin).__name__}")
    if not coin.startswith(WS_PREFIX):
        raise ValueError(f"ws_coin must start with '{WS_PREFIX}', got {coin!r}")
    suffix = coin[len(WS_PREFIX):]
    if not suffix.isdigit():
        raise ValueError(f"ws_coin suffix must be a non-negative integer, got {coin!r}")
    return decode_enc(int(suffix))


def decode_spot_coin(coin: str) -> tuple[int, int]:
    """
    Decode a Spot balance coin string back into (outcome_id, side).

    Examples:
        decode_spot_coin("+670") → (67, 0)
    """
    if not isinstance(coin, str):
        raise ValueError(f"coin must be str, got {type(coin).__name__}")
    if not coin.startswith(SPOT_PREFIX):
        raise ValueError(f"spot_coin must start with '{SPOT_PREFIX}', got {coin!r}")
    suffix = coin[len(SPOT_PREFIX):]
    if not suffix.isdigit():
        raise ValueError(f"spot_coin suffix must be a non-negative integer, got {coin!r}")
    return decode_enc(int(suffix))


def decode_asset_int(asset: int) -> tuple[int, int]:
    """
    Decode the Order action asset integer back into (outcome_id, side).

    Examples:
        decode_asset_int(100_000_670) → (67, 0)

    Raises:
        ValueError: if asset < ASSET_INT_OFFSET.
    """
    if not isinstance(asset, int) or isinstance(asset, bool):
        raise ValueError(f"asset must be int, got {type(asset).__name__}")
    if asset < ASSET_INT_OFFSET:
        raise ValueError(
            f"asset must be >= {ASSET_INT_OFFSET}, got {asset}"
        )
    enc = asset - ASSET_INT_OFFSET
    return decode_enc(enc)


# ─── Convenience helpers ──────────────────────────────────────────────────────

def both_sides(outcome_id: int) -> tuple[int, int]:
    """
    Return (yes_enc, no_enc) for a given outcome_id.

    Examples:
        both_sides(67) → (670, 671)
    """
    return encode(outcome_id, SIDE_YES), encode(outcome_id, SIDE_NO)


def both_ws_coins(outcome_id: int) -> tuple[str, str]:
    """
    Return (yes_ws_coin, no_ws_coin) for a given outcome_id.

    Examples:
        both_ws_coins(67) → ("#670", "#671")
    """
    y, n = both_sides(outcome_id)
    return ws_coin(y), ws_coin(n)
