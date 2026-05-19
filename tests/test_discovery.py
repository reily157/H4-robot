"""
Tests for discovery.py — outcomeMeta parsing and target identification.

Uses the exact snapshot from OBSERVER_V3_BRIEF.md as the primary fixture
(outcome 65/67/68/69, question 12, expiry 2026-05-20 06:00 UTC).
"""

from datetime import datetime, timezone
import pytest

from discovery import (
    CycleSpec,
    DiscoveryError,
    PriceBinarySpec,
    PriceBucketSpec,
    identify_targets,
    parse_price_binary,
    parse_price_bucket,
    _parse_expiry,
    _parse_kv_description,
)


# ─── Reference fixture from OBSERVER_V3_BRIEF.md ───────────────────────────────

REFERENCE_META = {
    "outcomes": [
        {"outcome": 65, "name": "Recurring",
         "description": "class:priceBinary|underlying:BTC|expiry:20260520-0600|targetPrice:76886|period:1d"},
        {"outcome": 66, "name": "Recurring Fallback", "description": "other"},
        {"outcome": 67, "name": "Recurring Named Outcome", "description": "index:0"},
        {"outcome": 68, "name": "Recurring Named Outcome", "description": "index:1"},
        {"outcome": 69, "name": "Recurring Named Outcome", "description": "index:2"},
    ],
    "questions": [
        {"question": 12, "name": "Recurring",
         "description": "class:priceBucket|underlying:BTC|expiry:20260520-0600|priceThresholds:75348,78423|period:1d",
         "fallbackOutcome": 66, "namedOutcomes": [67, 68, 69]},
    ],
}

REF_EXPIRY = datetime(2026, 5, 20, 6, 0, 0, tzinfo=timezone.utc)


# ─── Description parsing primitives ────────────────────────────────────────────

class TestParseKvDescription:
    def test_binary(self):
        desc = "class:priceBinary|underlying:BTC|expiry:20260520-0600|targetPrice:76886|period:1d"
        kv = _parse_kv_description(desc)
        assert kv["class"] == "priceBinary"
        assert kv["underlying"] == "BTC"
        assert kv["expiry"] == "20260520-0600"
        assert kv["targetPrice"] == "76886"
        assert kv["period"] == "1d"

    def test_bucket(self):
        desc = "class:priceBucket|underlying:BTC|expiry:20260520-0600|priceThresholds:75348,78423|period:1d"
        kv = _parse_kv_description(desc)
        assert kv["class"] == "priceBucket"
        assert kv["priceThresholds"] == "75348,78423"

    def test_empty(self):
        assert _parse_kv_description("") == {}

    def test_non_string(self):
        assert _parse_kv_description(None) == {}
        assert _parse_kv_description(123) == {}


class TestParseExpiry:
    def test_valid(self):
        dt = _parse_expiry("20260520-0600")
        assert dt == REF_EXPIRY

    def test_midnight(self):
        dt = _parse_expiry("20260101-0000")
        assert dt == datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            _parse_expiry("2026052-0600")

    def test_missing_dash(self):
        with pytest.raises(ValueError):
            _parse_expiry("202605200600x")

    def test_invalid_date(self):
        with pytest.raises(ValueError):
            _parse_expiry("20261320-0600")  # month 13

    def test_non_string(self):
        with pytest.raises(ValueError):
            _parse_expiry(20260520)


# ─── parse_price_binary ────────────────────────────────────────────────────────

class TestParsePriceBinary:
    def test_reference_outcome_65(self):
        out = REFERENCE_META["outcomes"][0]
        spec = parse_price_binary(out)
        assert spec is not None
        assert spec.outcome_id == 65
        assert spec.underlying == "BTC"
        assert spec.target_price == 76886.0
        assert spec.expiry == REF_EXPIRY
        assert spec.period == "1d"

    def test_non_binary_returns_none(self):
        out = REFERENCE_META["outcomes"][2]  # "index:0"
        spec = parse_price_binary(out)
        assert spec is None

    def test_other_description_returns_none(self):
        out = REFERENCE_META["outcomes"][1]  # "other"
        spec = parse_price_binary(out)
        assert spec is None

    def test_missing_outcome_id(self):
        out = {"description": "class:priceBinary|underlying:BTC|expiry:20260520-0600|targetPrice:1|period:1d"}
        assert parse_price_binary(out) is None

    def test_malformed_target_price(self):
        out = {"outcome": 99, "description": "class:priceBinary|underlying:BTC|expiry:20260520-0600|targetPrice:abc|period:1d"}
        assert parse_price_binary(out) is None

    def test_missing_expiry(self):
        out = {"outcome": 99, "description": "class:priceBinary|underlying:BTC|targetPrice:1|period:1d"}
        assert parse_price_binary(out) is None

    def test_not_dict(self):
        assert parse_price_binary("not_a_dict") is None


# ─── parse_price_bucket ────────────────────────────────────────────────────────

class TestParsePriceBucket:
    def test_reference_question_12(self):
        q = REFERENCE_META["questions"][0]
        spec = parse_price_bucket(q)
        assert spec is not None
        assert spec.question_id == 12
        assert spec.underlying == "BTC"
        assert spec.expiry == REF_EXPIRY
        assert spec.thresholds == [75348.0, 78423.0]
        assert spec.named_outcome_ids == [67, 68, 69]
        assert spec.fallback_outcome_id == 66

    def test_missing_named_outcomes_returns_empty_list(self):
        q = {
            "question": 12,
            "description": "class:priceBucket|underlying:BTC|expiry:20260520-0600|priceThresholds:75348,78423|period:1d",
            "fallbackOutcome": 66,
        }
        spec = parse_price_bucket(q)
        assert spec is not None
        assert spec.named_outcome_ids == []

    def test_missing_fallback_returns_none(self):
        q = {
            "question": 12,
            "description": "class:priceBucket|underlying:BTC|expiry:20260520-0600|priceThresholds:75348,78423|period:1d",
            "namedOutcomes": [67, 68, 69],
        }
        assert parse_price_bucket(q) is None

    def test_malformed_thresholds(self):
        q = {
            "question": 12,
            "description": "class:priceBucket|underlying:BTC|expiry:20260520-0600|priceThresholds:abc,xyz|period:1d",
            "fallbackOutcome": 66, "namedOutcomes": [67],
        }
        assert parse_price_bucket(q) is None

    def test_non_priceBucket_returns_none(self):
        q = {
            "question": 1,
            "description": "class:somethingElse|underlying:BTC",
            "fallbackOutcome": 0, "namedOutcomes": [],
        }
        assert parse_price_bucket(q) is None


# ─── identify_targets ──────────────────────────────────────────────────────────

class TestIdentifyTargets:

    def test_reference_meta(self):
        spec = identify_targets(REFERENCE_META)
        assert isinstance(spec, CycleSpec)
        assert spec.has_bucket
        assert spec.has_binary
        assert spec.is_complete
        assert spec.bucket.question_id == 12
        assert spec.binary.outcome_id == 65
        assert spec.bucket.expiry == spec.binary.expiry

    def test_eth_underlying_not_found(self):
        # No ETH in reference meta — should raise
        with pytest.raises(DiscoveryError):
            identify_targets(REFERENCE_META, underlying="ETH")

    def test_picks_latest_when_multiple_buckets(self):
        meta = {
            "outcomes": [],
            "questions": [
                {"question": 1,
                 "description": "class:priceBucket|underlying:BTC|expiry:20260518-0600|priceThresholds:1,2|period:1d",
                 "fallbackOutcome": 0, "namedOutcomes": [1]},
                {"question": 2,
                 "description": "class:priceBucket|underlying:BTC|expiry:20260520-0600|priceThresholds:1,2|period:1d",
                 "fallbackOutcome": 0, "namedOutcomes": [1]},
            ],
        }
        spec = identify_targets(meta)
        assert spec.bucket.question_id == 2  # most recent expiry

    def test_bucket_only_no_binary(self):
        meta = {
            "outcomes": [],
            "questions": REFERENCE_META["questions"],
        }
        spec = identify_targets(meta)
        assert spec.has_bucket
        assert not spec.has_binary
        assert not spec.is_complete

    def test_binary_only_no_bucket(self):
        meta = {
            "outcomes": REFERENCE_META["outcomes"],
            "questions": [],
        }
        spec = identify_targets(meta)
        assert spec.has_binary
        assert not spec.has_bucket
        assert not spec.is_complete

    def test_misaligned_expiries_returns_spec_but_not_complete(self):
        meta = {
            "outcomes": [
                {"outcome": 65,
                 "description": "class:priceBinary|underlying:BTC|expiry:20260519-0600|targetPrice:1|period:1d"},
            ],
            "questions": [
                {"question": 12,
                 "description": "class:priceBucket|underlying:BTC|expiry:20260520-0600|priceThresholds:1,2|period:1d",
                 "fallbackOutcome": 0, "namedOutcomes": [67]},
            ],
        }
        spec = identify_targets(meta)
        assert spec.has_bucket
        assert spec.has_binary
        assert not spec.is_complete  # expiries differ

    def test_both_empty_raises(self):
        with pytest.raises(DiscoveryError):
            identify_targets({"outcomes": [], "questions": []})

    def test_malformed_meta_not_dict_raises(self):
        with pytest.raises(ValueError, match="dict"):
            identify_targets("not a dict")

    def test_malformed_outcomes_field_handled(self):
        meta = {"outcomes": "garbage", "questions": REFERENCE_META["questions"]}
        spec = identify_targets(meta)
        assert spec.has_bucket
        assert not spec.has_binary

    def test_raw_meta_preserved(self):
        spec = identify_targets(REFERENCE_META)
        assert spec.raw_meta is REFERENCE_META
