"""
Tests for H1 — sum-to-1 invariant on categorical bucket outcomes.

All tests use synthetic DuckDB connections — real data is NEVER loaded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from analyzer.hypotheses.h1_sum_to_one import _compute_metrics, run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "hypothesis_id",
    "description",
    "metric_values",
    "p_value",
    "ci_lower_95",
    "ci_upper_95",
    "rejected_null",
    "oos_consistency",
    "interpretation_notes",
    "figures",
}

_CYCLE_A_START = datetime(2026, 6, 2, 6, 0, 0, tzinfo=timezone.utc)
_CYCLE_B_START = datetime(2026, 6, 3, 6, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Synthetic DuckDB builder
# ---------------------------------------------------------------------------


def _make_conn(
    bbo_rows: list[tuple],
    cycles: list[tuple] | None = None,
    outcomes: list[tuple] | None = None,
) -> duckdb.DuckDBPyConnection:
    """
    Build a minimal in-memory DuckDB with the three tables H1 needs.

    bbo_rows: list of (ts, coin, bid_px, ask_px)
    cycles:   list of (cycle_id, started_at)
    outcomes: list of (cycle_id, role, yes_coin)
    """
    conn = duckdb.connect(":memory:")

    conn.execute(
        """
        CREATE TABLE cycles (
            cycle_id VARCHAR, started_at TIMESTAMPTZ,
            bucket_question_id BIGINT, bucket_expiry TIMESTAMPTZ,
            bucket_thresholds VARCHAR, threshold_low DOUBLE, threshold_high DOUBLE,
            bucket_underlying VARCHAR, binary_outcome_id BIGINT,
            binary_target_price DOUBLE, binary_expiry TIMESTAMPTZ, raw_meta VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE outcomes_map (
            cycle_id VARCHAR, outcome_id BIGINT, role VARCHAR,
            yes_coin VARCHAR, no_coin VARCHAR, description VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE bbo (
            ts_local TIMESTAMPTZ, ts_remote VARCHAR, coin VARCHAR,
            bid_px DOUBLE, bid_sz DOUBLE, ask_px DOUBLE, ask_sz DOUBLE
        )
        """
    )

    if cycles is None:
        cycles = [("cycle_A", _CYCLE_A_START)]
    for cycle_id, started_at in cycles:
        conn.execute(
            "INSERT INTO cycles(cycle_id, started_at) VALUES (?, ?)",
            [cycle_id, started_at],
        )

    if outcomes is None:
        outcomes = [
            ("cycle_A", "named_0", "COIN_0"),
            ("cycle_A", "named_1", "COIN_1"),
            ("cycle_A", "named_2", "COIN_2"),
        ]
    for cycle_id, role, yes_coin in outcomes:
        conn.execute(
            "INSERT INTO outcomes_map(cycle_id, role, yes_coin) VALUES (?, ?, ?)",
            [cycle_id, role, yes_coin],
        )

    for ts, coin, bid_px, ask_px in bbo_rows:
        conn.execute(
            "INSERT INTO bbo(ts_local, coin, bid_px, bid_sz, ask_px, ask_sz) "
            "VALUES (?, ?, ?, 1.0, ?, 1.0)",
            [ts, coin, bid_px, ask_px],
        )

    return conn


def _bbo_rows(
    coin: str,
    mid: float,
    start: datetime,
    n: int,
    interval_s: int = 30,
    spread: float = 0.002,
) -> list[tuple]:
    """Generate (n) BBO rows at regular intervals with a fixed mid."""
    half = spread / 2
    return [
        (start + timedelta(seconds=i * interval_s), coin, mid - half, mid + half)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_required_schema():
    """All required keys must be present in the returned dict."""
    rows = (
        _bbo_rows("COIN_0", 0.333, _CYCLE_A_START, 40)
        + _bbo_rows("COIN_1", 0.333, _CYCLE_A_START, 40)
        + _bbo_rows("COIN_2", 0.334, _CYCLE_A_START, 40)
    )
    conn = _make_conn(rows)
    result = run(conn)
    plt.close("all")
    assert REQUIRED_KEYS == set(result.keys())
    assert result["hypothesis_id"] == "H1"
    assert isinstance(result["figures"], list)
    assert isinstance(result["interpretation_notes"], list)
    assert len(result["interpretation_notes"]) > 0


def test_perfect_sum_not_rejected():
    """
    Sum of mids = 1.0 at every timestamp → rejected_null must be False
    and median_abs_dev must be near zero.
    """
    # 40 points × 30s = 20 minutes (covers both opening and continuous)
    rows = (
        _bbo_rows("COIN_0", 0.333, _CYCLE_A_START, 40)
        + _bbo_rows("COIN_1", 0.333, _CYCLE_A_START, 40)
        + _bbo_rows("COIN_2", 0.334, _CYCLE_A_START, 40)
    )
    conn = _make_conn(rows)
    result = run(conn)
    plt.close("all")
    assert result["rejected_null"] is False
    mv = result["metric_values"]
    assert mv["median_abs_dev_all"] is not None
    assert mv["median_abs_dev_all"] < 1e-6  # floating-point tolerance


def test_large_constant_bias_metrics():
    """
    Sum of mids = 1.10 constantly → median_abs_dev > 0.5% and Wilcoxon p tiny.
    Note: rejected_null also requires rho1 > 0.3; iid synthetic data may not
    satisfy that, so we only assert the metric values here.
    """
    # 60 points × 30s = 30 minutes; sum = 0.40 + 0.40 + 0.30 = 1.10
    rows = (
        _bbo_rows("COIN_0", 0.40, _CYCLE_A_START, 60)
        + _bbo_rows("COIN_1", 0.40, _CYCLE_A_START, 60)
        + _bbo_rows("COIN_2", 0.30, _CYCLE_A_START, 60)
    )
    conn = _make_conn(rows)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["median_abs_dev_all"] is not None
    assert mv["median_abs_dev_all"] > 0.005  # exceeds 0.5% threshold
    assert mv["p_wilcoxon_combined"] is not None
    assert mv["p_wilcoxon_combined"] < 0.01  # strong signal


def test_opening_vs_continuous_split():
    """
    Timestamps before started_at + 15min → 'opening'.
    Timestamps from started_at + 15min onward → 'continuous'.
    The two sets must be disjoint and cover all timestamps.
    """
    # 60 points × 30s → first 30 in opening (0s..14:30), next 30 in continuous (15:00..29:30)
    rows = (
        _bbo_rows("COIN_0", 0.333, _CYCLE_A_START, 60)
        + _bbo_rows("COIN_1", 0.333, _CYCLE_A_START, 60)
        + _bbo_rows("COIN_2", 0.334, _CYCLE_A_START, 60)
    )
    conn = _make_conn(rows)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["n_timestamps_opening"] > 0
    assert mv["n_timestamps_continuous"] > 0
    assert (
        mv["n_timestamps_opening"] + mv["n_timestamps_continuous"]
        == mv["n_timestamps_combined"]
    )


def test_empty_cycles_graceful():
    """cycles_to_test=[] returns a valid dict with p_value=None; must not crash."""
    conn = _make_conn([], cycles=[], outcomes=[])
    result = run(conn, cycles_to_test=[])
    plt.close("all")
    assert result["hypothesis_id"] == "H1"
    assert result["p_value"] is None
    assert result["rejected_null"] is False
    assert isinstance(result["figures"], list)
    mv = result["metric_values"]
    assert mv["n_cycles"] == 0


def test_incomplete_cycle_skipped():
    """Cycle missing named_2 in outcomes_map is silently skipped (no exception)."""
    rows = (
        _bbo_rows("COIN_0", 0.50, _CYCLE_A_START, 20)
        + _bbo_rows("COIN_1", 0.50, _CYCLE_A_START, 20)
        # COIN_2 intentionally absent
    )
    outcomes = [
        ("cycle_A", "named_0", "COIN_0"),
        ("cycle_A", "named_1", "COIN_1"),
        # named_2 missing
    ]
    conn = _make_conn(rows, outcomes=outcomes)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["n_cycles"] == 0
    assert mv["n_cycles_skipped"] == 1


def test_zero_bids_filtered():
    """
    BBO rows with bid_px=0 or ask_px=0 must be excluded from mid computation.
    Valid rows have sum=1.0, so median_abs_dev must remain near zero
    even when invalid rows are present.
    """
    valid = (
        _bbo_rows("COIN_0", 0.333, _CYCLE_A_START, 20)
        + _bbo_rows("COIN_1", 0.333, _CYCLE_A_START, 20)
        + _bbo_rows("COIN_2", 0.334, _CYCLE_A_START, 20)
    )
    # Inject invalid BBO rows (bid_px=0) at a timestamp that would contaminate the mid
    bad_ts = _CYCLE_A_START + timedelta(seconds=15)
    invalid = [
        (bad_ts, "COIN_0", 0.0, 0.40),
        (bad_ts, "COIN_1", 0.0, 0.40),
        (bad_ts, "COIN_2", 0.0, 0.30),
    ]
    conn = _make_conn(valid + invalid)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    if mv["median_abs_dev_all"] is not None:
        assert mv["median_abs_dev_all"] < 1e-6


def test_oos_consistency_keys():
    """With ≥2 cycles, oos_consistency must have per-cycle entries and 'aggregate'."""
    cycles = [("cycle_A", _CYCLE_A_START), ("cycle_B", _CYCLE_B_START)]
    outcomes = [
        ("cycle_A", "named_0", "A0"),
        ("cycle_A", "named_1", "A1"),
        ("cycle_A", "named_2", "A2"),
        ("cycle_B", "named_0", "B0"),
        ("cycle_B", "named_1", "B1"),
        ("cycle_B", "named_2", "B2"),
    ]
    rows = (
        _bbo_rows("A0", 0.333, _CYCLE_A_START, 20)
        + _bbo_rows("A1", 0.333, _CYCLE_A_START, 20)
        + _bbo_rows("A2", 0.334, _CYCLE_A_START, 20)
        + _bbo_rows("B0", 0.333, _CYCLE_B_START, 20)
        + _bbo_rows("B1", 0.333, _CYCLE_B_START, 20)
        + _bbo_rows("B2", 0.334, _CYCLE_B_START, 20)
    )
    conn = _make_conn(rows, cycles=cycles, outcomes=outcomes)
    result = run(conn)
    plt.close("all")
    oos = result["oos_consistency"]
    assert oos is not None
    assert "aggregate" in oos
    # Each cycle must appear as a held-out fold
    assert "cycle_A" in oos
    assert "cycle_B" in oos
