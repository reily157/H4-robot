"""
H1 — Sum-to-1 invariant on categorical bucket outcomes.

Hypothesis: (mid_named_0 + mid_named_1 + mid_named_2) ≈ 1.0 if the market
is efficient and risk-neutral.

Rejection criterion (BOTH must hold, from hypotheses.md):
  - |median dev| > 0.5% at α=0.05 (before FDR correction)
  - autocorrelation ρ(1) > 0.3 on 10s-resampled dev(t) series
"""

from __future__ import annotations

import logging
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyzer.stats import leave_one_cycle_out, merge_asof_safe, stationary_bootstrap

logger = logging.getLogger(__name__)

_OPENING_MINUTES = 15
_ACF_RESAMPLE_FREQ = "10s"
_MERGE_TOLERANCE = pd.Timedelta("100ms")
_DROP_WARN_THRESHOLD = 0.50
_WILCOXON_MIN_N = 10


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _to_utc(raw) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _get_cycles(conn: duckdb.DuckDBPyConnection, cycles_to_test: list[str] | None) -> list[str]:
    if cycles_to_test is not None:
        return list(cycles_to_test)
    return [r[0] for r in conn.execute("SELECT cycle_id FROM cycles ORDER BY cycle_id").fetchall()]


def _get_named_coins(conn: duckdb.DuckDBPyConnection, cycle_id: str) -> dict[str, str] | None:
    rows = conn.execute(
        "SELECT role, yes_coin FROM outcomes_map "
        "WHERE cycle_id = ? AND role IN ('named_0','named_1','named_2')",
        [cycle_id],
    ).fetchall()
    mapping = {role: yes_coin for role, yes_coin in rows}
    if len(mapping) < 3:
        logger.warning("Cycle %s: missing named outcomes %s, skipping", cycle_id, sorted(mapping))
        return None
    return mapping


def _get_bbo_mid(conn: duckdb.DuckDBPyConnection, coin: str) -> pd.DataFrame:
    """Load BBO for one coin, filter valid quotes, return (ts_local, mid)."""
    df = conn.execute(
        "SELECT ts_local, bid_px, ask_px FROM bbo WHERE coin = ? ORDER BY ts_local",
        [coin],
    ).df()
    df = df[(df["bid_px"] > 0) & (df["ask_px"] > 0)].copy()
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    df = df.drop_duplicates(subset="ts_local", keep="last")
    return df[["ts_local", "mid"]].sort_values("ts_local").reset_index(drop=True)


def _build_dev_for_cycle(
    conn: duckdb.DuckDBPyConnection,
    cycle_id: str,
    coins: dict[str, str],
    started_at: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """
    Compute dev(t) = mid_0 + mid_1 + mid_2 - 1.0 for one cycle.

    Returns (df, n_dropped) where df has columns:
        ts_local, cycle_id, dev, phase ('opening'|'continuous')
    n_dropped is the count of anchor rows lost after dropna.
    """
    bbo_0 = _get_bbo_mid(conn, coins["named_0"])
    bbo_1 = _get_bbo_mid(conn, coins["named_1"])
    bbo_2 = _get_bbo_mid(conn, coins["named_2"])

    # Union of all timestamps as anchor — captures every BBO update
    all_ts = (
        pd.concat([bbo_0["ts_local"], bbo_1["ts_local"], bbo_2["ts_local"]])
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    anchor = pd.DataFrame({"ts_local": all_ts})

    joined = merge_asof_safe(
        anchor, bbo_0.rename(columns={"mid": "mid_0"}), on="ts_local", tolerance=_MERGE_TOLERANCE
    )
    joined = merge_asof_safe(
        joined, bbo_1.rename(columns={"mid": "mid_1"}), on="ts_local", tolerance=_MERGE_TOLERANCE
    )
    joined = merge_asof_safe(
        joined, bbo_2.rename(columns={"mid": "mid_2"}), on="ts_local", tolerance=_MERGE_TOLERANCE
    )

    n_before = len(joined)
    joined = joined.dropna(subset=["mid_0", "mid_1", "mid_2"])
    n_dropped = n_before - len(joined)

    if n_before > 0 and n_dropped / n_before > _DROP_WARN_THRESHOLD:
        logger.warning(
            "Cycle %s: dropped %.0f%% of rows (%d/%d) — many BBO snapshots missing within 100ms tolerance",
            cycle_id, 100.0 * n_dropped / n_before, n_dropped, n_before,
        )

    joined["dev"] = joined["mid_0"] + joined["mid_1"] + joined["mid_2"] - 1.0
    joined["cycle_id"] = cycle_id

    opening_end = started_at + pd.Timedelta(minutes=_OPENING_MINUTES)
    joined["phase"] = "continuous"
    mask = (joined["ts_local"] >= started_at) & (joined["ts_local"] < opening_end)
    joined.loc[mask, "phase"] = "opening"

    return joined[["ts_local", "cycle_id", "dev", "phase"]], n_dropped


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _acf_lag1(series: pd.Series) -> tuple[float, int, int]:
    """
    ACF lag-1 on 10s-resampled series; NaN gaps are NOT forward-filled.
    Returns (rho1, n_valid_pairs, n_nan_pairs).
    """
    resampled = series.resample(_ACF_RESAMPLE_FREQ).last()
    vals = resampled.values
    if len(vals) < 2:
        return float("nan"), 0, 0
    x, y = vals[:-1], vals[1:]
    valid = ~(np.isnan(x) | np.isnan(y))
    n_valid = int(valid.sum())
    n_nan = len(x) - n_valid
    if n_valid < 2:
        return float("nan"), n_valid, n_nan
    rho = float(np.corrcoef(x[valid], y[valid])[0, 1])
    return rho, n_valid, n_nan


def _wilcoxon_safe(data: np.ndarray) -> float:
    """Wilcoxon signed-rank test vs median=0. Returns p-value or NaN."""
    data = data[~np.isnan(data)]
    if len(data) < _WILCOXON_MIN_N:
        logger.warning("Wilcoxon: n=%d < %d minimum, returning NaN", len(data), _WILCOXON_MIN_N)
        return float("nan")
    try:
        _, p = stats.wilcoxon(data, alternative="two-sided")
        return float(p)
    except Exception as exc:
        logger.warning("Wilcoxon failed: %s", exc)
        return float("nan")


def _ks_test(data: np.ndarray) -> float:
    """KS test: data vs N(0, std(data)). Returns p-value or NaN."""
    data = data[~np.isnan(data)]
    if len(data) < 5:
        return float("nan")
    sigma = float(np.std(data, ddof=1))
    if sigma == 0:
        return float("nan")
    _, p = stats.kstest(data, "norm", args=(0, sigma))
    return float(p)


def _maybe_float(x) -> float | None:
    if x is None:
        return None
    try:
        return None if np.isnan(x) else float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Empty result helper
# ---------------------------------------------------------------------------


def _empty_result(notes: list[str] | None = None, skipped: int = 0) -> dict:
    return {
        "hypothesis_id": "H1",
        "description": "Sum-to-1 invariant on categorical bucket outcomes",
        "metric_values": {"n_cycles": 0, "n_cycles_skipped": skipped},
        "p_value": None,
        "ci_lower_95": None,
        "ci_upper_95": None,
        "rejected_null": False,
        "oos_consistency": None,
        "interpretation_notes": notes or ["No cycles to test."],
        "figures": [],
    }


# ---------------------------------------------------------------------------
# Core computation (no OOS — called by run() and leave_one_cycle_out)
# ---------------------------------------------------------------------------


def _compute_metrics(
    conn: duckdb.DuckDBPyConnection,
    cycles: list[str],
) -> dict:
    """
    Core H1 computation. Does NOT perform OOS validation.
    Called by run() for main results and by leave_one_cycle_out for each fold.
    """
    if not cycles:
        return _empty_result()

    cycle_meta = {
        r[0]: _to_utc(r[1])
        for r in conn.execute(
            "SELECT cycle_id, started_at FROM cycles WHERE cycle_id IN ({})".format(
                ", ".join(f"'{c}'" for c in cycles)
            )
        ).fetchall()
    }

    all_devs: list[pd.DataFrame] = []
    n_total_dropped = 0
    skipped: list[str] = []

    for cycle_id in cycles:
        coins = _get_named_coins(conn, cycle_id)
        if coins is None:
            skipped.append(cycle_id)
            continue
        started_at = cycle_meta.get(cycle_id)
        if started_at is None:
            logger.warning("Cycle %s: started_at not found in cycles table, skipping", cycle_id)
            skipped.append(cycle_id)
            continue
        try:
            df, n_dropped = _build_dev_for_cycle(conn, cycle_id, coins, started_at)
            n_total_dropped += n_dropped
            if len(df) > 0:
                all_devs.append(df)
        except Exception as exc:
            logger.warning("Cycle %s: error building dev(t): %s", cycle_id, exc)
            skipped.append(cycle_id)

    if not all_devs:
        return _empty_result(
            notes=["No valid data after loading; check outcomes_map and bbo tables."],
            skipped=len(skipped),
        )

    combined = pd.concat(all_devs, ignore_index=True)
    dev_all = combined["dev"].values
    dev_opening = combined.loc[combined["phase"] == "opening", "dev"].values
    dev_continuous = combined.loc[combined["phase"] == "continuous", "dev"].values

    # Wilcoxon signed-rank (H0: median = 0)
    p_combined = _wilcoxon_safe(dev_all)
    p_opening = _wilcoxon_safe(dev_opening) if len(dev_opening) >= _WILCOXON_MIN_N else float("nan")
    p_continuous = _wilcoxon_safe(dev_continuous) if len(dev_continuous) >= _WILCOXON_MIN_N else float("nan")

    # KS tests vs N(0, σ̂)
    p_ks_combined = _ks_test(dev_all)
    p_ks_opening = _ks_test(dev_opening)
    p_ks_continuous = _ks_test(dev_continuous)

    # Medians
    median_dev_all = float(np.median(dev_all)) if len(dev_all) else float("nan")
    median_abs_all = float(np.median(np.abs(dev_all))) if len(dev_all) else float("nan")
    median_abs_opening = float(np.median(np.abs(dev_opening))) if len(dev_opening) else float("nan")
    median_abs_continuous = float(np.median(np.abs(dev_continuous))) if len(dev_continuous) else float("nan")

    # Bootstrap CI on median |dev| (combined)
    ci_lower, ci_upper = None, None
    if len(dev_all) >= _WILCOXON_MIN_N:
        try:
            bs = stationary_bootstrap(dev_all, lambda x: float(np.median(np.abs(x))))
            ci_lower = bs["ci_lower_95"]
            ci_upper = bs["ci_upper_95"]
        except Exception as exc:
            logger.warning("Bootstrap CI failed: %s", exc)

    # ACF lag-1 on 10s-resampled combined series (NaN gaps skipped, not filled)
    indexed = combined.set_index("ts_local").sort_index()
    if indexed.index.tz is None:
        indexed.index = indexed.index.tz_localize("UTC")

    rho1_all, acf_valid, acf_nan = _acf_lag1(indexed["dev"])
    op_df = indexed[indexed["phase"] == "opening"]
    co_df = indexed[indexed["phase"] == "continuous"]
    rho1_opening = _acf_lag1(op_df["dev"])[0] if len(op_df) >= 2 else float("nan")
    rho1_continuous = _acf_lag1(co_df["dev"])[0] if len(co_df) >= 2 else float("nan")

    # Rejection criterion: BOTH conditions from hypotheses.md
    rejected_null = (
        not np.isnan(median_abs_all)
        and median_abs_all > 0.005
        and not np.isnan(p_combined)
        and p_combined < 0.05
        and not np.isnan(rho1_all)
        and rho1_all > 0.3
    )

    n_out_of_range = int(((dev_all < -1.0) | (dev_all > 1.0)).sum())

    metric_values = {
        "n_cycles": len(cycles) - len(skipped),
        "n_cycles_skipped": len(skipped),
        "n_timestamps_combined": int(len(dev_all)),
        "n_timestamps_opening": int(len(dev_opening)),
        "n_timestamps_continuous": int(len(dev_continuous)),
        "n_dropped_for_missing_data": int(n_total_dropped),
        "n_mid_out_of_range": n_out_of_range,
        "median_dev_all": _maybe_float(median_dev_all),
        "median_abs_dev_all": _maybe_float(median_abs_all),
        "median_abs_dev_opening": _maybe_float(median_abs_opening),
        "median_abs_dev_continuous": _maybe_float(median_abs_continuous),
        "p_wilcoxon_combined": _maybe_float(p_combined),
        "p_wilcoxon_opening": _maybe_float(p_opening),
        "p_wilcoxon_continuous": _maybe_float(p_continuous),
        "p_ks_combined": _maybe_float(p_ks_combined),
        "p_ks_opening": _maybe_float(p_ks_opening),
        "p_ks_continuous": _maybe_float(p_ks_continuous),
        "rho1_all": _maybe_float(rho1_all),
        "rho1_opening": _maybe_float(rho1_opening),
        "rho1_continuous": _maybe_float(rho1_continuous),
        "n_acf_valid_pairs": acf_valid,
        "n_acf_nan_pairs": acf_nan,
    }

    return {
        "hypothesis_id": "H1",
        "description": "Sum-to-1 invariant on categorical bucket outcomes",
        "metric_values": metric_values,
        "p_value": _maybe_float(p_combined),
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "rejected_null": rejected_null,
        "oos_consistency": None,  # filled by run()
        "interpretation_notes": [
            "Rejection requires BOTH: |median dev| > 0.5% AND ρ(1) > 0.3 (at α=0.05, before FDR).",
            "Non-zero deviation may reflect risk-aversion, resolution fees, or smart-contract discount — not necessarily arbitrage.",
            "If rejected, verify 1 – sum(ask_yes_i) exceeds total execution costs before claiming arbitrage.",
        ],
        "figures": _make_figures(combined),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _make_figures(combined: pd.DataFrame) -> list[plt.Figure]:
    figures = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Figure 1: histogram of dev(t) by phase
        fig, ax = plt.subplots(figsize=(8, 4))
        for phase, color in [("opening", "steelblue"), ("continuous", "coral")]:
            subset = combined.loc[combined["phase"] == phase, "dev"]
            if len(subset):
                ax.hist(subset.values, bins=60, alpha=0.6, color=color, label=phase, density=True)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("dev(t) = mid_0 + mid_1 + mid_2 – 1")
        ax.set_ylabel("Density")
        ax.set_title("H1 — Distribution of sum-to-1 deviation")
        ax.legend()
        fig.tight_layout()
        figures.append(fig)

        # Figure 2: time series per cycle
        cycle_ids = sorted(combined["cycle_id"].unique()) if "cycle_id" in combined.columns else []
        if cycle_ids:
            n = len(cycle_ids)
            fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), squeeze=False)
            for i, cid in enumerate(cycle_ids):
                ax = axes[i, 0]
                sub = combined[combined["cycle_id"] == cid].sort_values("ts_local")
                ax.plot(sub["ts_local"], sub["dev"], linewidth=0.5, color="steelblue")
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax.axhline(0.005, color="red", linewidth=0.5, linestyle=":", label="±0.5% threshold")
                ax.axhline(-0.005, color="red", linewidth=0.5, linestyle=":")
                opening_ts = sub.loc[sub["phase"] == "opening", "ts_local"]
                if len(opening_ts):
                    ax.axvline(opening_ts.iloc[-1], color="green", linewidth=0.8, alpha=0.7, label="06:15 UTC")
                ax.set_title(cid, fontsize=9)
                ax.set_ylabel("dev(t)")
            fig.suptitle("H1 — dev(t) per cycle", y=1.01)
            fig.tight_layout()
            figures.append(fig)

        # Figure 3: ACF plot (up to lag 50 × 10s)
        indexed = combined.set_index("ts_local").sort_index()
        if indexed.index.tz is None:
            indexed.index = indexed.index.tz_localize("UTC")
        resampled = indexed["dev"].resample(_ACF_RESAMPLE_FREQ).last()
        vals = resampled.values
        if len(vals) > 2:
            max_lag = min(50, len(vals) - 1)
            acf_vals: list[float] = [1.0]
            for lag in range(1, max_lag + 1):
                x, y = vals[:-lag], vals[lag:]
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() < 2:
                    acf_vals.append(float("nan"))
                else:
                    acf_vals.append(float(np.corrcoef(x[valid], y[valid])[0, 1]))
            acf_arr = np.array(acf_vals, dtype=float)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(range(len(acf_arr)), np.where(np.isnan(acf_arr), 0, acf_arr), width=0.8, color="steelblue")
            ax.axhline(0.3, color="red", linestyle="--", linewidth=0.8, label="ρ=0.3 threshold")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Lag (×10s)")
            ax.set_ylabel("ACF")
            ax.set_title("H1 — Autocorrelation of dev(t) resampled at 10s")
            ax.legend()
            fig.tight_layout()
            figures.append(fig)

    return figures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    conn: duckdb.DuckDBPyConnection,
    cycles_to_test: list[str] | None = None,
) -> dict:
    """
    Execute the H1 sum-to-1 invariant test.

    Args:
        conn: DuckDB connection from ingest.load_to_duckdb().
        cycles_to_test: cycle_ids to include. If None, uses all cycles in DB.

    Returns dict conforming to the hypothesis module interface (ANALYZER_V1_BRIEF §5.3).
    p_value is raw (before FDR correction across H1-H7).
    """
    cycles = _get_cycles(conn, cycles_to_test)
    result = _compute_metrics(conn, cycles)

    if len(cycles) >= 2:
        try:
            result["oos_consistency"] = leave_one_cycle_out(
                cycles,
                lambda remaining: _compute_metrics(conn, remaining),
            )
        except Exception as exc:
            logger.warning("OOS validation failed: %s", exc)
            result["oos_consistency"] = {"error": str(exc)}
    else:
        result["oos_consistency"] = {"note": "Need ≥2 cycles for OOS validation."}

    return result
