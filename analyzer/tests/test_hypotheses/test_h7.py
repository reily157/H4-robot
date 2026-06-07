"""
Tests for H7 — opening auction overnight bias.

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

from analyzer.hypotheses.h7_auction_bias import _compute_metrics, run

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

# All cycles start at 06:00 UTC on consecutive days
_STARTS = [
    datetime(2026, 6, 2, 6, 0, 0, tzinfo=timezone.utc) + timedelta(days=i)
    for i in range(7)
]


# ---------------------------------------------------------------------------
# Synthetic DuckDB builder
# ---------------------------------------------------------------------------


def _make_conn(
    bbo_rows: list[tuple],
    cycles: list[tuple] | None = None,
    outcomes: list[tuple] | None = None,
) -> duckdb.DuckDBPyConnection:
    """
    Minimal in-memory DuckDB for H7 tests.

    bbo_rows:  list of (ts, coin, bid_px, ask_px)
    cycles:    list of (cycle_id, started_at)
    outcomes:  list of (cycle_id, role, yes_coin)
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

    for cycle_id, started_at in (cycles or []):
        conn.execute(
            "INSERT INTO cycles(cycle_id, started_at) VALUES (?, ?)",
            [cycle_id, started_at],
        )
    for cycle_id, role, yes_coin in (outcomes or []):
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


def _snapshot(ts: datetime, coin: str, mid: float, spread: float = 0.002) -> tuple:
    """Single BBO row at an exact timestamp."""
    h = spread / 2
    return (ts, coin, mid - h, mid + h)


def _three_snapshots(started_at: datetime, coin: str, mid_pre: float, mid_end: float, mid_post: float) -> list[tuple]:
    """The three BBO rows needed for one H7 cycle (at ±0s from each target)."""
    t_pre = started_at - timedelta(minutes=1)
    t_end = started_at + timedelta(minutes=15)
    t_post = started_at + timedelta(minutes=20)
    return [
        _snapshot(t_pre, coin, mid_pre),
        _snapshot(t_end, coin, mid_end),
        _snapshot(t_post, coin, mid_post),
    ]


def _build_cycles_and_outcomes(n: int) -> tuple[list[tuple], list[tuple]]:
    """Create n cycle rows and their binary outcomes (one coin per cycle)."""
    cycles = [(_cid(i), _STARTS[i]) for i in range(n)]
    outcomes = [(_cid(i), "binary", _coin(i)) for i in range(n)]
    return cycles, outcomes


def _cid(i: int) -> str:
    return f"cycle_{i}"


def _coin(i: int) -> str:
    return f"BIN_{i}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_required_schema():
    """All required keys must be present in the returned dict."""
    n = 4
    cycles, outcomes = _build_cycles_and_outcomes(n)
    bbo = []
    for i in range(n):
        bbo += _three_snapshots(_STARTS[i], _coin(i), 0.50, 0.52, 0.53)
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    assert REQUIRED_KEYS == set(result.keys())
    assert result["hypothesis_id"] == "H7"
    assert isinstance(result["figures"], list)
    assert isinstance(result["interpretation_notes"], list)
    assert len(result["interpretation_notes"]) > 0


def test_no_signal_not_rejected():
    """
    Δ_postopen is constant (≈0) regardless of Δ_auction → β = 0 by construction.
    All three rejection conditions fail: |β| = 0 < 0.15, R² = NaN, p = NaN.
    Note: alternating patterns must be avoided — they create perfect negative
    correlation (β ≈ -1), which WOULD trigger rejection.
    """
    cycle_params = [
        # (Δ_auction varies, Δ_postopen constant near zero)
        (+0.10, +0.01),
        (-0.10, +0.01),
        (+0.08, +0.01),
    ]
    n = len(cycle_params)
    cycles, outcomes = _build_cycles_and_outcomes(n)
    bbo = []
    for i, (da, dp) in enumerate(cycle_params):
        mid_pre = 0.50
        bbo += _three_snapshots(_STARTS[i], _coin(i), mid_pre, mid_pre + da, mid_pre + da + dp)
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    assert result["rejected_null"] is False
    mv = result["metric_values"]
    if mv["beta_hat"] is not None:
        assert abs(mv["beta_hat"]) < 1e-9  # β = 0 exactly (constant Y → zero slope)


def test_perfect_positive_beta():
    """
    Δ_postopen = 0.5 × Δ_auction (noiseless, varying Δ_auction) → β ≈ 0.5, R² ≈ 1.0.
    Varying Δ_auction is critical: identical values make X singular in OLS.
    """
    cycle_params = [
        # (Δ_auction, Δ_postopen) satisfying Δ_post = 0.5 × Δ_auction
        (+0.10, +0.05),
        (-0.08, -0.04),
        (+0.06, +0.03),
        (-0.04, -0.02),
        (+0.12, +0.06),
    ]
    n = len(cycle_params)
    cycles, outcomes = _build_cycles_and_outcomes(n)
    bbo = []
    for i, (da, dp) in enumerate(cycle_params):
        mid_pre = 0.50
        mid_end = mid_pre + da
        mid_post = mid_end + dp
        bbo += _three_snapshots(_STARTS[i], _coin(i), mid_pre, mid_end, mid_post)
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["beta_hat"] is not None
    assert abs(mv["beta_hat"] - 0.5) < 0.01   # noiseless data → very close to 0.5
    assert mv["r_squared"] is not None
    assert mv["r_squared"] > 0.95


def test_sign_consistency_counted():
    """
    5 cycles with product sign matching β, 2 with opposite sign →
    n_cycles_sign_consistent must equal 5.
    """
    # β will be positive (consistent cycles dominate with large magnitudes)
    cycle_params = [
        # Consistent (positive product): both same sign → product > 0
        (+0.10, +0.08),
        (+0.08, +0.06),
        (-0.10, -0.08),
        (-0.06, -0.05),
        (+0.12, +0.09),
        # Inconsistent (opposite signs → product < 0, but small magnitudes)
        (+0.02, -0.09),
        (-0.02, +0.09),
    ]
    n = len(cycle_params)
    cycles, outcomes = _build_cycles_and_outcomes(n)
    bbo = []
    for i, (da, dp) in enumerate(cycle_params):
        mid_pre = 0.50
        mid_end = mid_pre + da
        mid_post = mid_end + dp
        bbo += _three_snapshots(_STARTS[i], _coin(i), mid_pre, mid_end, mid_post)
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["beta_hat"] is not None
    # β_hat > 0 (positive correlation dominates)
    assert mv["beta_hat"] > 0
    assert mv["n_cycles_sign_consistent"] == 5


def test_rejection_requires_all_three():
    """
    n_cycles_sign_consistent < 5 prevents rejection even when β and R² pass.
    7 cycles: only 4 have consistent sign → condition 3 fails → not rejected.
    """
    cycle_params = [
        # 4 consistent (β > 0):
        (+0.10, +0.12),
        (+0.08, +0.10),
        (-0.06, -0.07),
        (-0.08, -0.09),
        # 3 inconsistent (small Δ_auction to not dominate, large Δ_postopen):
        (+0.04, -0.11),
        (+0.06, -0.10),
        (-0.04, +0.11),
    ]
    n = len(cycle_params)
    cycles, outcomes = _build_cycles_and_outcomes(n)
    bbo = []
    for i, (da, dp) in enumerate(cycle_params):
        mid_pre = 0.50
        mid_end = mid_pre + da
        mid_post = mid_end + dp
        bbo += _three_snapshots(_STARTS[i], _coin(i), mid_pre, mid_end, mid_post)
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["n_cycles_sign_consistent"] < 5
    assert result["rejected_null"] is False


def test_missing_snapshot_cycle_skipped():
    """
    Cycle with no BBO near T_pre (05:59) is skipped without exception.
    """
    # Cycle 0: complete snapshots at all 3 targets
    # Cycle 1: BBO data only after 06:01 → T_pre (05:59) will be unmatched
    cycles, outcomes = _build_cycles_and_outcomes(2)
    bbo = _three_snapshots(_STARTS[0], _coin(0), 0.50, 0.52, 0.53)
    # Cycle 1: only provide snapshots at T_end and T_post, not T_pre
    t_end = _STARTS[1] + timedelta(minutes=15)
    t_post = _STARTS[1] + timedelta(minutes=20)
    bbo += [
        _snapshot(t_end, _coin(1), 0.48),
        _snapshot(t_post, _coin(1), 0.49),
        # T_pre = 05:59 intentionally missing
    ]
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["n_cycles_skipped"] == 1  # cycle_1 was skipped


def test_empty_cycles_graceful():
    """cycles_to_test=[] returns a valid dict with p_value=None; must not crash."""
    conn = _make_conn([], cycles=[], outcomes=[])
    result = run(conn, cycles_to_test=[])
    plt.close("all")
    assert result["hypothesis_id"] == "H7"
    assert result["p_value"] is None
    assert result["rejected_null"] is False
    assert isinstance(result["figures"], list)
    assert result["metric_values"]["n_cycles"] == 0


def test_oos_consistency_keys():
    """With ≥2 valid cycles, oos_consistency must contain per-cycle entries and 'aggregate'."""
    n = 3
    cycles, outcomes = _build_cycles_and_outcomes(n)
    bbo = []
    for i in range(n):
        bbo += _three_snapshots(_STARTS[i], _coin(i), 0.50, 0.52, 0.53)
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    oos = result["oos_consistency"]
    assert oos is not None
    assert "aggregate" in oos
    for i in range(n):
        assert _cid(i) in oos
