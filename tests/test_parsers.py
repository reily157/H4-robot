"""
Tests for parsers.py — WS message parsers.

Fixtures here are synthetic — built from the documented schema and the
zaakirio/hl-hip4-arb reference. They will be replaced with real captured
payloads in phase 3.2 via scripts/capture_fixtures.py.
"""

import json
import pytest

from parsers import (
    L2Level,
    L2Snapshot,
    Trade,
    Bbo,
    ActiveAssetCtx,
    parse_l2_book,
    parse_trades,
    parse_bbo,
    parse_active_asset_ctx,
)


# ─── Synthetic fixtures ────────────────────────────────────────────────────────

def make_l2book_msg(coin="#670", time_ms=1700000000000, n_bids=3, n_asks=3):
    return {
        "channel": "l2Book",
        "data": {
            "coin": coin,
            "time": time_ms,
            "levels": [
                # bids high → low
                [{"px": f"{0.155 - i * 0.001:.5f}", "sz": f"{100 + i * 10}"}
                 for i in range(n_bids)],
                # asks low → high
                [{"px": f"{0.160 + i * 0.001:.5f}", "sz": f"{100 + i * 10}"}
                 for i in range(n_asks)],
            ],
        },
    }


def make_trades_msg(coin="#670", n=3):
    return {
        "channel": "trades",
        "data": [
            {
                "coin": coin,
                "time": 1700000000000 + i * 1000,
                "side": "B" if i % 2 == 0 else "A",
                "px": f"{0.155 + i * 0.001:.5f}",
                "sz": f"{10 + i}",
                "tid": 50000 + i,
            }
            for i in range(n)
        ],
    }


def make_bbo_msg(coin="#670", with_bid=True, with_ask=True):
    bbo = []
    bbo.append({"px": "0.155", "sz": "100"} if with_bid else None)
    bbo.append({"px": "0.160", "sz": "120"} if with_ask else None)
    return {
        "channel": "bbo",
        "data": {
            "coin": coin,
            "time": 1700000000000,
            "bbo": bbo,
        },
    }


def make_active_asset_ctx_msg(coin="BTC"):
    return {
        "channel": "activeAssetCtx",
        "data": {
            "coin": coin,
            "time": 1700000000000,
            "ctx": {
                "markPx": "67000.5",
                "midPx": "67000.0",
                "oraclePx": "67005.0",
                "funding": "0.0001",
                "openInterest": "1234.5",
            },
        },
    }


# ─── parse_l2_book ─────────────────────────────────────────────────────────────

class TestParseL2Book:

    def test_basic(self):
        msg = make_l2book_msg(coin="#670", n_bids=3, n_asks=3)
        snap = parse_l2_book(msg, "HIP4")
        assert isinstance(snap, L2Snapshot)
        assert snap.market_kind == "HIP4"
        assert snap.market_id == "#670"
        assert snap.ts_ms == 1700000000000
        assert len(snap.bids) == 3
        assert len(snap.asks) == 3
        # bids sorted high → low
        assert snap.bids[0].price > snap.bids[1].price > snap.bids[2].price
        # asks sorted low → high
        assert snap.asks[0].price < snap.asks[1].price < snap.asks[2].price

    def test_empty_book(self):
        msg = make_l2book_msg(n_bids=0, n_asks=0)
        snap = parse_l2_book(msg, "HIP4")
        assert snap.bids == []
        assert snap.asks == []

    def test_perp_market_kind(self):
        msg = make_l2book_msg(coin="BTC")
        snap = parse_l2_book(msg, "PERP")
        assert snap.market_kind == "PERP"
        assert snap.market_id == "BTC"

    def test_invalid_market_kind(self):
        msg = make_l2book_msg()
        with pytest.raises(ValueError, match="market_kind"):
            parse_l2_book(msg, "INVALID")

    def test_wrong_channel(self):
        msg = make_l2book_msg()
        msg["channel"] = "trades"
        with pytest.raises(ValueError, match="channel"):
            parse_l2_book(msg, "HIP4")

    def test_missing_data(self):
        msg = {"channel": "l2Book"}
        with pytest.raises(ValueError, match="data"):
            parse_l2_book(msg, "HIP4")

    def test_missing_coin(self):
        msg = make_l2book_msg()
        del msg["data"]["coin"]
        with pytest.raises(ValueError, match="coin"):
            parse_l2_book(msg, "HIP4")

    def test_missing_time(self):
        msg = make_l2book_msg()
        del msg["data"]["time"]
        with pytest.raises(ValueError, match="time"):
            parse_l2_book(msg, "HIP4")

    def test_invalid_levels_returns_empty(self):
        msg = make_l2book_msg()
        msg["data"]["levels"] = "garbage"
        snap = parse_l2_book(msg, "HIP4")
        assert snap.bids == []
        assert snap.asks == []

    def test_malformed_level_row_is_skipped(self):
        msg = make_l2book_msg(n_bids=2)
        msg["data"]["levels"][0].append({"px": "not_a_number", "sz": "10"})
        msg["data"]["levels"][0].append({"px": "0.1"})  # missing sz
        snap = parse_l2_book(msg, "HIP4")
        # Only the 2 valid rows remain
        assert len(snap.bids) == 2

    def test_msg_not_dict_raises(self):
        with pytest.raises(ValueError, match="dict"):
            parse_l2_book("not a dict", "HIP4")


# ─── parse_trades ──────────────────────────────────────────────────────────────

class TestParseTrades:

    def test_basic(self):
        msg = make_trades_msg(n=3)
        trades = parse_trades(msg, "HIP4")
        assert len(trades) == 3
        assert all(isinstance(t, Trade) for t in trades)

    def test_side_mapping(self):
        msg = make_trades_msg(n=2)  # alternates B and A
        trades = parse_trades(msg, "HIP4")
        assert trades[0].side == "BUY"
        assert trades[1].side == "SELL"

    def test_empty_data(self):
        msg = {"channel": "trades", "data": []}
        trades = parse_trades(msg, "HIP4")
        assert trades == []

    def test_unknown_side_skipped(self):
        msg = make_trades_msg(n=2)
        msg["data"][0]["side"] = "X"
        trades = parse_trades(msg, "HIP4")
        # Only the second trade remains
        assert len(trades) == 1

    def test_missing_field_skipped(self):
        msg = make_trades_msg(n=2)
        del msg["data"][0]["px"]
        trades = parse_trades(msg, "HIP4")
        assert len(trades) == 1

    def test_invalid_px_skipped(self):
        msg = make_trades_msg(n=2)
        msg["data"][0]["px"] = "not_a_number"
        trades = parse_trades(msg, "HIP4")
        assert len(trades) == 1

    def test_tid_optional(self):
        msg = make_trades_msg(n=1)
        del msg["data"][0]["tid"]
        trades = parse_trades(msg, "HIP4")
        assert trades[0].trade_id is None

    def test_data_not_list_raises(self):
        msg = {"channel": "trades", "data": "garbage"}
        with pytest.raises(ValueError, match="list"):
            parse_trades(msg, "HIP4")

    def test_wrong_channel(self):
        msg = make_trades_msg()
        msg["channel"] = "l2Book"
        with pytest.raises(ValueError, match="channel"):
            parse_trades(msg, "HIP4")


# ─── parse_bbo ─────────────────────────────────────────────────────────────────

class TestParseBbo:

    def test_basic(self):
        msg = make_bbo_msg()
        bbo = parse_bbo(msg, "HIP4")
        assert isinstance(bbo, Bbo)
        assert bbo.market_id == "#670"
        assert bbo.bid_px == 0.155
        assert bbo.bid_sz == 100.0
        assert bbo.ask_px == 0.160
        assert bbo.ask_sz == 120.0

    def test_missing_bid_side(self):
        msg = make_bbo_msg(with_bid=False)
        bbo = parse_bbo(msg, "HIP4")
        assert bbo.bid_px is None
        assert bbo.bid_sz is None
        assert bbo.ask_px == 0.160

    def test_missing_ask_side(self):
        msg = make_bbo_msg(with_ask=False)
        bbo = parse_bbo(msg, "HIP4")
        assert bbo.ask_px is None
        assert bbo.bid_px == 0.155

    def test_both_sides_missing(self):
        msg = make_bbo_msg(with_bid=False, with_ask=False)
        bbo = parse_bbo(msg, "HIP4")
        assert bbo.bid_px is None
        assert bbo.ask_px is None

    def test_missing_coin(self):
        msg = make_bbo_msg()
        del msg["data"]["coin"]
        with pytest.raises(ValueError, match="coin"):
            parse_bbo(msg, "HIP4")

    def test_wrong_channel(self):
        msg = make_bbo_msg()
        msg["channel"] = "l2Book"
        with pytest.raises(ValueError, match="channel"):
            parse_bbo(msg, "HIP4")


# ─── parse_active_asset_ctx ────────────────────────────────────────────────────

class TestParseActiveAssetCtx:

    def test_basic_btc(self):
        msg = make_active_asset_ctx_msg(coin="BTC")
        ctx = parse_active_asset_ctx(msg, "PERP")
        assert isinstance(ctx, ActiveAssetCtx)
        assert ctx.market_id == "BTC"
        assert ctx.mark_px == 67000.5
        assert ctx.mid_px == 67000.0
        assert ctx.oracle_px == 67005.0
        assert ctx.funding == 0.0001
        assert ctx.open_interest == 1234.5

    def test_raw_payload_preserved(self):
        msg = make_active_asset_ctx_msg()
        ctx = parse_active_asset_ctx(msg, "PERP")
        # raw_payload must be valid JSON containing the original data
        decoded = json.loads(ctx.raw_payload)
        assert decoded["coin"] == "BTC"
        assert decoded["ctx"]["markPx"] == "67000.5"

    def test_partial_ctx_fields(self):
        msg = make_active_asset_ctx_msg()
        msg["data"]["ctx"] = {"markPx": "67000"}
        ctx = parse_active_asset_ctx(msg, "PERP")
        assert ctx.mark_px == 67000.0
        assert ctx.mid_px is None
        assert ctx.oracle_px is None

    def test_missing_ctx_block(self):
        msg = make_active_asset_ctx_msg()
        del msg["data"]["ctx"]
        ctx = parse_active_asset_ctx(msg, "PERP")
        assert ctx.mark_px is None
        assert ctx.mid_px is None

    def test_missing_time_defaults_to_zero(self):
        msg = make_active_asset_ctx_msg()
        del msg["data"]["time"]
        ctx = parse_active_asset_ctx(msg, "PERP")
        assert ctx.ts_ms == 0

    def test_hip4_market_kind(self):
        msg = make_active_asset_ctx_msg(coin="#670")
        ctx = parse_active_asset_ctx(msg, "HIP4")
        assert ctx.market_kind == "HIP4"
        assert ctx.market_id == "#670"

    def test_missing_coin(self):
        msg = make_active_asset_ctx_msg()
        del msg["data"]["coin"]
        with pytest.raises(ValueError, match="coin"):
            parse_active_asset_ctx(msg, "PERP")

    def test_wrong_channel(self):
        msg = make_active_asset_ctx_msg()
        msg["channel"] = "l2Book"
        with pytest.raises(ValueError, match="channel"):
            parse_active_asset_ctx(msg, "PERP")
