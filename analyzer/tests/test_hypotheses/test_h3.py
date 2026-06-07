"""
Tests for H3 — Bucket 'in range' convergence non-monotonicity.

All tests use synthetic DuckDB connections — real data is NEVER loaded.

Analysis window: started_at + 15min → started_at + 24h (continuous phase only).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist

matplotlib.use("Agg")

from analyzer.hypotheses.h3_bucket_convergence import _compute_metrics, run

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

# 7 consecutive days starting 2026-06-02 at 06:00 UTC
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
    Minimal in-memory DuckDB for H3 tests.

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
    # perp_ctx must exist for _append_normalized_figure; leave empty in tests
    conn.execute(
        """
        CREATE TABLE perp_ctx (
            ts_local TIMESTAMPTZ, coin VARCHAR, mark_px DOUBLE,
            mid_px DOUBLE, oracle_px DOUBLE, funding DOUBLE, open_interest DOUBLE
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


def _cid(i: int) -> str:
    return f"cycle_{i}"


def _coin(i: int) -> str:
    return f"MID1_{i}"


def _bbo_row(ts: datetime, coin: str, mid: float, spread: float = 0.002) -> tuple:
    h = spread / 2
    return (ts, coin, mid - h, mid + h)


def _gen_series(
    started_at: datetime,
    coin: str,
    values_func,
    freq_sec: int = 300,
) -> list[tuple]:
    """
    Generate BBO rows in the continuous phase [started_at + 15min, started_at + 24h].
    values_func(t_sec) -> float, where t_sec = seconds since started_at.
    freq_sec: interval between points (default 5min = 300s).
    """
    start = started_at + timedelta(minutes=15)
    end = started_at + timedelta(hours=24)
    rows = []
    t = start
    while t <= end:
        t_sec = (t - started_at).total_seconds()
        mid = float(values_func(t_sec))
        rows.append(_bbo_row(t, coin, mid))
        t += timedelta(seconds=freq_sec)
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_required_schema():
    """All required keys must be present in the returned dict."""
    n = 2
    cycles = [(_cid(i), _STARTS[i]) for i in range(n)]
    outcomes = [(_cid(i), "named_1", _coin(i)) for i in range(n)]
    bbo = []
    for i in range(n):
        period_sec = 4 * 3600
        bbo += _gen_series(
            _STARTS[i], _coin(i),
            lambda t, p=period_sec: 0.5 + 0.3 * np.sin(2 * np.pi * t / p),
        )
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    assert REQUIRED_KEYS == set(result.keys())
    assert result["hypothesis_id"] == "H3"
    assert isinstance(result["figures"], list)
    assert isinstance(result["interpretation_notes"], list)
    assert len(result["interpretation_notes"]) > 0


def test_monotone_decreasing_not_rejected():
    """
    Perfectly linear decreasing series → 0 reversals → rejected_null = False.
    median_reversals must be 0 and n_cycles_skipped must be 0.
    """
    total_sec = 24 * 3600
    cycles = [(_cid(0), _STARTS[0])]
    outcomes = [(_cid(0), "named_1", _coin(0))]
    bbo = _gen_series(
        _STARTS[0], _coin(0),
        lambda t: 0.9 - 0.8 * (t / total_sec),
    )
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    assert result["rejected_null"] is False
    mv = result["metric_values"]
    assert mv["n_cycles_skipped"] == 0
    assert mv["n_cycles"] == 1
    assert mv["median_reversals"] is not None
    assert mv["median_reversals"] == 0  # perfectly monotone → no sign changes


def test_oscillatory_reversals_counted():
    """
    Sine wave with 4h period over 24h → many reversals well above the threshold.
    Amplitude 0.3 >> _MIN_REVERSAL_MAGNITUDE so all oscillations are detected.
    """
    period_sec = 4 * 3600
    n = 2
    cycles = [(_cid(i), _STARTS[i]) for i in range(n)]
    outcomes = [(_cid(i), "named_1", _coin(i)) for i in range(n)]
    bbo = []
    for i in range(n):
        bbo += _gen_series(
            _STARTS[i], _coin(i),
            lambda t, p=period_sec: 0.5 + 0.3 * np.sin(2 * np.pi * t / p),
        )
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["median_reversals"] is not None
    # 24h / 4h period = 6 full cycles → ~12 sign changes; conservatively expect ≥8
    assert mv["median_reversals"] >= 8
    # Each individual cycle should also have many reversals
    for cid, count in mv["reversal_counts"].items():
        assert count >= 8, f"Cycle {cid} had only {count} reversals"


def test_min_magnitude_guard_filters_noise():
    """
    Series with oscillations far below _MIN_REVERSAL_MAGNITUDE (5e-5) → 0 reversals.
    This validates the numerical floor is effective at suppressing floating-point noise.
    """
    period_sec = 3600
    cycles = [(_cid(0), _STARTS[0])]
    outcomes = [(_cid(0), "named_1", _coin(0))]
    # amplitude 5e-5 << 1e-4 = _MIN_REVERSAL_MAGNITUDE
    bbo = _gen_series(
        _STARTS[0], _coin(0),
        lambda t, p=period_sec: 0.5 + 5e-5 * np.sin(2 * np.pi * t / p),
    )
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["median_reversals"] is not None
    assert mv["median_reversals"] == 0  # all oscillations below magnitude floor


def test_empty_cycles_graceful():
    """cycles_to_test=[] returns a valid dict with p_value=None; must not crash."""
    conn = _make_conn([], cycles=[], outcomes=[])
    result = run(conn, cycles_to_test=[])
    plt.close("all")
    assert result["hypothesis_id"] == "H3"
    assert result["p_value"] is None
    assert result["rejected_null"] is False
    assert isinstance(result["figures"], list)
    assert result["metric_values"]["n_cycles"] == 0


def test_insufficient_data_skipped():
    """
    Cycle with only 2 BBO snapshots near end of window → smoothed series is all NaN
    (rolling 1h needs ≥6 non-NaN points in window) → cycle skipped gracefully.
    """
    cycles = [(_cid(0), _STARTS[0])]
    outcomes = [(_cid(0), "named_1", _coin(0))]
    # Place 2 points at the very end of the 24h window
    end_of_cycle = _STARTS[0] + timedelta(hours=24)
    bbo = [
        _bbo_row(end_of_cycle - timedelta(minutes=10), _coin(0), 0.5),
        _bbo_row(end_of_cycle - timedelta(minutes=5), _coin(0), 0.5),
    ]
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert mv["n_cycles_skipped"] == 1
    assert mv["n_cycles"] == 0
    assert result["p_value"] is None


def test_oos_consistency_keys():
    """With ≥2 valid cycles, oos_consistency must have per-cycle entries and 'aggregate'."""
    total_sec = 24 * 3600
    n = 2
    cycles = [(_cid(i), _STARTS[i]) for i in range(n)]
    outcomes = [(_cid(i), "named_1", _coin(i)) for i in range(n)]
    bbo = []
    for i in range(n):
        bbo += _gen_series(
            _STARTS[i], _coin(i),
            lambda t: 0.9 - 0.8 * (t / total_sec),
        )
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    oos = result["oos_consistency"]
    assert oos is not None
    assert "aggregate" in oos
    for i in range(n):
        assert _cid(i) in oos


def test_fisher_p_value_computed():
    """
    Fisher p_combined = 1 - chi2.cdf(-2 * sum(ln(p_i)), df=2k).

    Use oscillatory (non-monotone) series so per-cycle MK p-values are
    positive and in a computable range. A pure sine wave has τ ≈ 0 → p ≈ 1,
    so MK p-values are guaranteed non-zero here.
    Verify p_fisher_combined matches the formula applied to stored per-cycle p-values.
    """
    n = 3
    period_sec = 4 * 3600
    cycles = [(_cid(i), _STARTS[i]) for i in range(n)]
    outcomes = [(_cid(i), "named_1", _coin(i)) for i in range(n)]
    bbo = []
    for i in range(n):
        bbo += _gen_series(
            _STARTS[i], _coin(i),
            lambda t, p=period_sec: 0.5 + 0.3 * np.sin(2 * np.pi * t / p),
        )
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")

    mv = result["metric_values"]
    mk_ps = [v for v in mv["per_cycle_mk_p"].values() if v is not None]
    assert len(mk_ps) == n
    # Oscillatory data → τ ≈ 0 → p > 0 (no divide-by-zero in log)
    assert all(p > 0 for p in mk_ps), "Oscillatory series should give positive MK p-values"

    # Recompute Fisher formula from the stored per-cycle p-values
    clipped = np.maximum(mk_ps, np.finfo(float).tiny)
    chi2_stat = float(-2.0 * np.sum(np.log(clipped)))
    expected_fisher = float(1.0 - chi2_dist.cdf(chi2_stat, df=2 * len(mk_ps)))
    assert mv["p_fisher_combined"] == pytest.approx(expected_fisher, abs=1e-10)


def test_metric_values_contain_floor():
    """metric_values must expose min_reversal_magnitude_floor for traceability."""
    cycles = [(_cid(0), _STARTS[0])]
    outcomes = [(_cid(0), "named_1", _coin(0))]
    total_sec = 24 * 3600
    bbo = _gen_series(_STARTS[0], _coin(0), lambda t: 0.9 - 0.8 * (t / total_sec))
    conn = _make_conn(bbo, cycles, outcomes)
    result = run(conn)
    plt.close("all")
    mv = result["metric_values"]
    assert "min_reversal_magnitude_floor" in mv
    assert mv["min_reversal_magnitude_floor"] == pytest.approx(1e-4)
