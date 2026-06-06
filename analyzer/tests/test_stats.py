"""Tests for analyzer/stats.py."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.stats.multitest as smm

from analyzer.stats import (
    fdr_correction,
    leave_one_cycle_out,
    merge_asof_safe,
    stationary_bootstrap,
)


# ---------------------------------------------------------------------------
# fdr_correction
# ---------------------------------------------------------------------------


def test_fdr_matches_statsmodels_reference():
    p_values = [0.001, 0.01, 0.04, 0.09, 0.20]
    rejected, p_adj = fdr_correction(p_values)
    _, ref_p_adj, _, _ = smm.multipletests(p_values, alpha=0.05, method="fdr_bh")
    assert rejected == [r for r in smm.multipletests(p_values, alpha=0.05, method="fdr_bh")[0]]
    np.testing.assert_allclose(p_adj, ref_p_adj)


def test_fdr_all_significant():
    p_values = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    rejected, _ = fdr_correction(p_values)
    assert all(rejected)


def test_fdr_none_significant():
    p_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    rejected, _ = fdr_correction(p_values)
    assert not any(rejected)


def test_fdr_returns_correct_lengths():
    p_values = [0.01, 0.05, 0.10]
    rejected, p_adj = fdr_correction(p_values)
    assert len(rejected) == 3
    assert len(p_adj) == 3


def test_fdr_adjusted_p_values_bounded():
    p_values = [0.001, 0.01, 0.04, 0.09, 0.20]
    _, p_adj = fdr_correction(p_values)
    assert all(0.0 <= p <= 1.0 for p in p_adj)


# ---------------------------------------------------------------------------
# stationary_bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_iid_mean_ci_contains_zero():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(200)
    result = stationary_bootstrap(data, statistic=np.mean, n_boot=2000)
    assert result["ci_lower_95"] < 0 < result["ci_upper_95"], (
        "Bootstrap CI on mean of iid N(0,1) should contain 0"
    )


def test_bootstrap_iid_std_close_to_analytical():
    rng = np.random.default_rng(1)
    n = 500
    data = rng.standard_normal(n)
    result = stationary_bootstrap(data, statistic=np.mean, n_boot=5000)
    analytical_se = 1.0 / np.sqrt(n)
    # Allow generous tolerance: bootstrap std should be within 50% of analytical SE
    assert abs(result["std"] - analytical_se) / analytical_se < 0.5


def test_bootstrap_ar1_ci_wider_than_iid():
    rng = np.random.default_rng(2)
    n = 300

    # iid series
    iid_data = rng.standard_normal(n)
    iid_result = stationary_bootstrap(iid_data, statistic=np.mean, n_boot=2000)

    # AR(1) with strong autocorrelation ρ=0.8
    ar1_data = np.zeros(n)
    ar1_data[0] = rng.standard_normal()
    for i in range(1, n):
        ar1_data[i] = 0.8 * ar1_data[i - 1] + rng.standard_normal() * 0.6
    ar1_result = stationary_bootstrap(ar1_data, statistic=np.mean, n_boot=2000)

    # AR(1) CI should be wider (longer range) than iid CI
    iid_width = iid_result["ci_upper_95"] - iid_result["ci_lower_95"]
    ar1_width = ar1_result["ci_upper_95"] - ar1_result["ci_lower_95"]
    assert ar1_width > iid_width, (
        f"AR(1) CI width ({ar1_width:.4f}) should exceed iid CI width ({iid_width:.4f})"
    )


def test_bootstrap_returns_expected_keys():
    data = np.arange(50, dtype=float)
    result = stationary_bootstrap(data, statistic=np.mean, n_boot=100)
    for key in ("mean", "std", "ci_lower_95", "ci_upper_95", "all_samples"):
        assert key in result


def test_bootstrap_all_samples_length():
    data = np.ones(50)
    result = stationary_bootstrap(data, statistic=np.mean, n_boot=200)
    assert len(result["all_samples"]) == 200


# ---------------------------------------------------------------------------
# merge_asof_safe
# ---------------------------------------------------------------------------


def _make_ts(n: int, start: str = "2026-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="1s")
    return pd.DataFrame({"ts": ts, "val": np.arange(n, dtype=float)})


def test_merge_asof_safe_backward_join():
    left = _make_ts(5)
    right = _make_ts(3)
    merged = merge_asof_safe(left, right, on="ts")
    assert len(merged) == 5
    # First right row (t=0) should match left rows at t=0..4
    assert merged["val_x"].tolist() == [0, 1, 2, 3, 4]


def test_merge_asof_safe_raises_unsorted():
    left = _make_ts(5)
    left = left.iloc[::-1].reset_index(drop=True)  # reverse order → unsorted
    right = _make_ts(3)
    with pytest.raises(ValueError, match="must be sorted"):
        merge_asof_safe(left, right, on="ts")


def test_merge_asof_safe_raises_if_direction_passed():
    left = _make_ts(5)
    right = _make_ts(3)
    with pytest.raises(TypeError, match="does not accept"):
        merge_asof_safe(left, right, on="ts", direction="nearest")


def test_merge_asof_safe_with_by():
    coins = ["A", "B"]
    frames = []
    for coin in coins:
        ts = pd.date_range("2026-01-01", periods=4, freq="1s")
        frames.append(pd.DataFrame({"ts": ts, "coin": coin, "val": np.arange(4, dtype=float)}))
    left = pd.concat(frames).sort_values("ts").reset_index(drop=True)
    right = left.copy().rename(columns={"val": "val_r"})
    merged = merge_asof_safe(left, right, on="ts", by="coin")
    assert "val_r" in merged.columns


# ---------------------------------------------------------------------------
# leave_one_cycle_out
# ---------------------------------------------------------------------------


def _dummy_test(cycles: list[str]) -> dict:
    """Fake hypothesis returning p_value = len(cycles) / 10."""
    return {"p_value": len(cycles) / 10.0, "rejected_null": False}


def test_loocv_returns_one_result_per_cycle():
    cycles = ["C1", "C2", "C3"]
    results = leave_one_cycle_out(cycles, _dummy_test)
    for cid in cycles:
        assert cid in results


def test_loocv_aggregate_keys():
    cycles = ["C1", "C2", "C3"]
    results = leave_one_cycle_out(cycles, _dummy_test)
    agg = results["aggregate"]
    for key in ("n_cycles", "values", "mean", "std", "min", "max"):
        assert key in agg


def test_loocv_aggregate_n_cycles():
    cycles = ["C1", "C2", "C3"]
    results = leave_one_cycle_out(cycles, _dummy_test)
    assert results["aggregate"]["n_cycles"] == 3


def test_loocv_no_consistency_bool():
    cycles = ["C1", "C2", "C3"]
    results = leave_one_cycle_out(cycles, _dummy_test)
    assert "consistent" not in results["aggregate"], (
        "'consistent' boolean must not be in aggregate — each hypothesis defines its own criterion"
    )


def test_loocv_aggregate_stats_are_descriptive():
    cycles = ["C1", "C2", "C3", "C4"]
    results = leave_one_cycle_out(cycles, _dummy_test)
    agg = results["aggregate"]
    # _dummy_test returns p_value = len(remaining) / 10 = 3/10 for each held-out
    assert agg["mean"] == pytest.approx(0.3)
    assert agg["min"] == pytest.approx(0.3)
    assert agg["max"] == pytest.approx(0.3)
