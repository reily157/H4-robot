"""
Tests for codec.py — asset encoding for HIP-4.

Coverage strategy:
    - Concrete examples from the brief (outcome 67, 68, 69; binary 65)
    - Edge cases (outcome 0, very large outcome_id)
    - Full round-trip: encode → decode returns original
    - All invalid inputs raise ValueError with the expected message shape
    - Cross-consistency between forward and reverse helpers
"""

import pytest

from codec import (
    ASSET_INT_OFFSET,
    SIDE_YES,
    SIDE_NO,
    encode,
    ws_coin,
    spot_coin,
    asset_int,
    decode_enc,
    decode_ws_coin,
    decode_spot_coin,
    decode_asset_int,
    both_sides,
    both_ws_coins,
)


# ─── Forward encoding ──────────────────────────────────────────────────────────

class TestEncode:
    """encode(outcome_id, side) → enc int"""

    @pytest.mark.parametrize("outcome_id,side,expected", [
        (67, 0, 670),    # brief example: bucket idx:0 YES
        (67, 1, 671),    # brief example: bucket idx:0 NO
        (68, 0, 680),    # bucket idx:1 YES
        (68, 1, 681),    # bucket idx:1 NO
        (69, 0, 690),    # bucket idx:2 YES
        (69, 1, 691),    # bucket idx:2 NO
        (65, 0, 650),    # binary YES
        (65, 1, 651),    # binary NO
        (0, 0, 0),       # edge: outcome 0 YES
        (0, 1, 1),       # edge: outcome 0 NO
        (12345, 0, 123450),  # large outcome_id
    ])
    def test_known_values(self, outcome_id, side, expected):
        assert encode(outcome_id, side) == expected

    def test_invalid_outcome_id_negative(self):
        with pytest.raises(ValueError, match=">= 0"):
            encode(-1, 0)

    def test_invalid_outcome_id_type(self):
        with pytest.raises(ValueError, match="must be int"):
            encode("67", 0)
        with pytest.raises(ValueError, match="must be int"):
            encode(67.0, 0)

    def test_bool_is_rejected_as_int(self):
        # bool is technically int in Python — we explicitly reject it
        with pytest.raises(ValueError, match="must be int"):
            encode(True, 0)

    def test_invalid_side(self):
        with pytest.raises(ValueError, match="must be 0"):
            encode(67, 2)
        with pytest.raises(ValueError, match="must be 0"):
            encode(67, -1)

    def test_invalid_side_type(self):
        with pytest.raises(ValueError, match="must be int"):
            encode(67, "0")


# ─── Coin string formatting ────────────────────────────────────────────────────

class TestWsCoin:
    @pytest.mark.parametrize("enc,expected", [
        (670, "#670"),
        (671, "#671"),
        (0, "#0"),
        (100, "#100"),
    ])
    def test_known(self, enc, expected):
        assert ws_coin(enc) == expected

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            ws_coin(-1)

    def test_type_raises(self):
        with pytest.raises(ValueError):
            ws_coin("670")


class TestSpotCoin:
    @pytest.mark.parametrize("enc,expected", [
        (670, "+670"),
        (0, "+0"),
        (100, "+100"),
    ])
    def test_known(self, enc, expected):
        assert spot_coin(enc) == expected

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            spot_coin(-1)


class TestAssetInt:
    @pytest.mark.parametrize("enc,expected", [
        (670, 100_000_670),
        (0, 100_000_000),
        (1, 100_000_001),
    ])
    def test_known(self, enc, expected):
        assert asset_int(enc) == expected

    def test_offset_matches_constant(self):
        assert asset_int(0) == ASSET_INT_OFFSET

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            asset_int(-1)


# ─── Reverse decoding ──────────────────────────────────────────────────────────

class TestDecodeEnc:
    @pytest.mark.parametrize("enc,expected", [
        (670, (67, 0)),
        (671, (67, 1)),
        (0, (0, 0)),
        (1, (0, 1)),
        (123450, (12345, 0)),
    ])
    def test_known(self, enc, expected):
        assert decode_enc(enc) == expected

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            decode_enc(-1)


class TestDecodeWsCoin:
    @pytest.mark.parametrize("coin,expected", [
        ("#670", (67, 0)),
        ("#671", (67, 1)),
        ("#0", (0, 0)),
    ])
    def test_known(self, coin, expected):
        assert decode_ws_coin(coin) == expected

    def test_missing_prefix(self):
        with pytest.raises(ValueError, match="must start with"):
            decode_ws_coin("670")

    def test_wrong_prefix(self):
        with pytest.raises(ValueError, match="must start with"):
            decode_ws_coin("+670")

    def test_non_numeric_suffix(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            decode_ws_coin("#abc")

    def test_empty_suffix(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            decode_ws_coin("#")

    def test_negative_suffix(self):
        # "-1" is not a digit string, so isdigit() returns False
        with pytest.raises(ValueError, match="non-negative integer"):
            decode_ws_coin("#-1")

    def test_type_raises(self):
        with pytest.raises(ValueError, match="must be str"):
            decode_ws_coin(670)


class TestDecodeSpotCoin:
    @pytest.mark.parametrize("coin,expected", [
        ("+670", (67, 0)),
        ("+0", (0, 0)),
    ])
    def test_known(self, coin, expected):
        assert decode_spot_coin(coin) == expected

    def test_missing_prefix(self):
        with pytest.raises(ValueError, match="must start with"):
            decode_spot_coin("670")


class TestDecodeAssetInt:
    @pytest.mark.parametrize("asset,expected", [
        (100_000_670, (67, 0)),
        (100_000_000, (0, 0)),
        (100_000_001, (0, 1)),
    ])
    def test_known(self, asset, expected):
        assert decode_asset_int(asset) == expected

    def test_below_offset_raises(self):
        with pytest.raises(ValueError, match=">= 100000000"):
            decode_asset_int(99_999_999)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            decode_asset_int(0)


# ─── Round-trip invariants ─────────────────────────────────────────────────────

class TestRoundTrip:
    """Encoding then decoding must return the original inputs, exhaustively."""

    @pytest.mark.parametrize("outcome_id", [0, 1, 65, 67, 68, 69, 100, 999, 12345])
    @pytest.mark.parametrize("side", [SIDE_YES, SIDE_NO])
    def test_encode_decode_enc(self, outcome_id, side):
        enc = encode(outcome_id, side)
        assert decode_enc(enc) == (outcome_id, side)

    @pytest.mark.parametrize("outcome_id", [0, 67, 12345])
    @pytest.mark.parametrize("side", [SIDE_YES, SIDE_NO])
    def test_full_chain_ws(self, outcome_id, side):
        enc = encode(outcome_id, side)
        coin = ws_coin(enc)
        assert decode_ws_coin(coin) == (outcome_id, side)

    @pytest.mark.parametrize("outcome_id", [0, 67, 12345])
    @pytest.mark.parametrize("side", [SIDE_YES, SIDE_NO])
    def test_full_chain_spot(self, outcome_id, side):
        enc = encode(outcome_id, side)
        coin = spot_coin(enc)
        assert decode_spot_coin(coin) == (outcome_id, side)

    @pytest.mark.parametrize("outcome_id", [0, 67, 12345])
    @pytest.mark.parametrize("side", [SIDE_YES, SIDE_NO])
    def test_full_chain_asset_int(self, outcome_id, side):
        enc = encode(outcome_id, side)
        asset = asset_int(enc)
        assert decode_asset_int(asset) == (outcome_id, side)


# ─── Convenience helpers ──────────────────────────────────────────────────────

class TestBothSides:
    def test_brief_examples(self):
        # The 4 markets from the brief
        assert both_sides(67) == (670, 671)
        assert both_sides(68) == (680, 681)
        assert both_sides(69) == (690, 691)
        assert both_sides(65) == (650, 651)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            both_sides(-1)


class TestBothWsCoins:
    def test_brief_examples(self):
        assert both_ws_coins(67) == ("#670", "#671")
        assert both_ws_coins(65) == ("#650", "#651")

    def test_yes_then_no_order(self):
        """First element must always be YES side."""
        for oid in [0, 65, 67, 68, 69, 100]:
            yes, no = both_ws_coins(oid)
            assert decode_ws_coin(yes) == (oid, SIDE_YES)
            assert decode_ws_coin(no) == (oid, SIDE_NO)
