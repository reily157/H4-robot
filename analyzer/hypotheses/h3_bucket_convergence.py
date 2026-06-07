"""
H3 — Bucket 'in range' convergence non-monotonicity.

Analysis window (pre-registered decision, 2026-06-07):
  start = started_at + 15min  (end of opening auction, 06:15 UTC)
  end   = started_at + 24h    (next-day resolution, 06:00 UTC)
The opening phase (06:00-06:15) is excluded: it has distinct mechanics
already tested by H7.

Mann-Kendall interpretation (Fisher combined):
  A significant combined p-value indicates directional convergence is
  occurring within cycles. Combined with median_reversals > 4, this
  confirms the convergence is oscillatory rather than monotone.

Rejection criterion (BOTH must hold, from hypotheses.md):
  - Fisher combined MK p-value < 0.05 (at α=0.05, before FDR correction)
  - median reversals per cycle > 4
"""

from __future__ import annotations

import logging
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyzer.stats import leave_one_cycle_out, stationary_bootstrap

logger = logging.getLogger(__name__)

_OPENING_DURATION_MIN = 15  # Skip opening auction (06:00-06:15 UTC)
_CYCLE_DURATION_H = 24      # Analysis ends at started_at + 24h
_RESAMPLE_FREQ = "5min"
_FFILL_LIMIT = 12           # max 12 × 5min = 1h of forward fill
_ROLLING_WINDOW = "1h"
_ROLLING_MIN_PERIODS = 6    # require ≥30min of data in rolling window
_MIN_REVERSAL_MAGNITUDE = 1e-4  # numerical floor — see _count_reversals
_MIN_SERIES_POINTS = 6      # min non-NaN smoothed points for MK test
_REVERSAL_THRESHOLD = 4     # from hypotheses.md: median > 4 required


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _to_utc(raw) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _maybe_float(x) -> float | None:
    if x is None:
        return None
    try:
        return None if np.isnan(x) else float(x)
    except (TypeError, ValueError):
        return None


def _get_cycles(conn: duckdb.DuckDBPyConnection, cycles_to_test: list[str] | None) -> list[str]:
    if cycles_to_test is not None:
        return list(cycles_to_test)
    return [r[0] for r in conn.execute("SELECT cycle_id FROM cycles ORDER BY cycle_id").fetchall()]


def _get_named1_coin(conn: duckdb.DuckDBPyConnection, cycle_id: str) -> str | None:
    row = conn.execute(
        "SELECT yes_coin FROM outcomes_map WHERE cycle_id = ? AND role = 'named_1'",
        [cycle_id],
    ).fetchone()
    if row is None:
        logger.warning("Cycle %s: no named_1 outcome in outcomes_map, skipping", cycle_id)
        return None
    return row[0]


def _get_mid1_series(
    conn: duckdb.DuckDBPyConnection,
    coin: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Series, int]:
    """
    Load BBO for `coin` in [start, end], compute mid, return time-indexed Series.
    Returns (series, n_rows_before_dedup).
    """
    df = conn.execute(
        "SELECT ts_local, bid_px, ask_px FROM bbo "
        "WHERE coin = ? AND ts_local >= ? AND ts_local <= ? "
        "AND bid_px > 0 AND ask_px > 0 ORDER BY ts_local",
        [coin, start, end],
    ).df()
    if df.empty:
        return pd.Series(dtype=float, name="mid"), 0
    n_rows = len(df)
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    df = df.drop_duplicates(subset="ts_local", keep="last").sort_values("ts_local")
    ts_index = pd.DatetimeIndex([_to_utc(t) for t in df["ts_local"]])
    series = pd.Series(df["mid"].values, index=ts_index, name="mid")
    return series, n_rows


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _smooth_1h(raw: pd.Series) -> pd.Series:
    """5-min resample → ffill (max 1h) → 1h rolling mean."""
    resampled = raw.resample(_RESAMPLE_FREQ).last()
    filled = resampled.ffill(limit=_FFILL_LIMIT)
    return filled.rolling(window=_ROLLING_WINDOW, min_periods=_ROLLING_MIN_PERIODS).mean()


def _count_reversals(smoothed: pd.Series) -> tuple[int, pd.DatetimeIndex]:
    """
    Count sign changes in first differences of the smoothed series.

    _MIN_REVERSAL_MAGNITUDE = 1e-4:
        Numerical floor to suppress floating-point artifacts on flat segments.
        NOT a significance threshold. Do not increase without justification.

    Only consecutive significant differences (|diff| > floor) are compared.
    Returns (n_reversals, reversal_timestamps).
    """
    diff = smoothed.diff()
    sig_mask = diff.abs() > _MIN_REVERSAL_MAGNITUDE
    sig_diff = diff[sig_mask]
    if len(sig_diff) < 2:
        return 0, pd.DatetimeIndex([])
    signs = np.sign(sig_diff.values)
    change_mask = signs[1:] != signs[:-1]
    n_reversals = int(change_mask.sum())
    reversal_times = sig_diff.index[1:][change_mask]
    return n_reversals, reversal_times


def _mann_kendall(series: pd.Series) -> tuple[float, float]:
    """
    Mann-Kendall test via Kendall's τ on (time_rank, value).
    H0: τ = 0 (no monotone trend).
    Returns (tau, p_value), or (nan, nan) if insufficient data.
    """
    clean = series.dropna()
    if len(clean) < _MIN_SERIES_POINTS:
        return float("nan"), float("nan")
    tau, p_value = stats.kendalltau(np.arange(len(clean)), clean.values)
    return float(tau), float(p_value)


def _fisher_combine(p_vals: list[float]) -> float:
    """
    Fisher's method for combining independent p-values.
    χ² = -2 Σ ln(p_i), follows χ²(2k) under H0.
    Assumes independence across cycles (cycles separated by 24h).

    p = 0.0 from scipy (numerical underflow on perfectly monotone data) is
    clipped to machine epsilon rather than excluded — a p-value of 0 is a
    valid extreme observation, not missing data.
    """
    valid = [p for p in p_vals if p is not None and not np.isnan(p)]
    if not valid:
        return float("nan")
    clipped = np.maximum(np.array(valid, dtype=float), np.finfo(float).tiny)
    chi2_stat = float(-2.0 * np.sum(np.log(clipped)))
    return float(1.0 - stats.chi2.cdf(chi2_stat, df=2 * len(valid)))


# ---------------------------------------------------------------------------
# Empty result helper
# ---------------------------------------------------------------------------


def _empty_result(notes: list[str] | None = None, skipped: int = 0) -> dict:
    return {
        "hypothesis_id": "H3",
        "description": "Bucket 'in range' convergence non-monotonicity",
        "metric_values": {
            "n_cycles": 0,
            "n_cycles_skipped": skipped,
            "min_reversal_magnitude_floor": _MIN_REVERSAL_MAGNITUDE,
        },
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
    Core H3 computation. Does NOT perform OOS validation.
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

    per_cycle_data: list[dict] = []
    skipped: list[str] = []

    for cycle_id in cycles:
        named1_coin = _get_named1_coin(conn, cycle_id)
        if named1_coin is None:
            skipped.append(cycle_id)
            continue

        started_at = cycle_meta.get(cycle_id)
        if started_at is None:
            logger.warning("Cycle %s: started_at not found in cycles table, skipping", cycle_id)
            skipped.append(cycle_id)
            continue

        start = started_at + pd.Timedelta(minutes=_OPENING_DURATION_MIN)
        end = started_at + pd.Timedelta(hours=_CYCLE_DURATION_H)

        try:
            raw, n_rows = _get_mid1_series(conn, named1_coin, start, end)

            if n_rows == 0:
                logger.warning(
                    "Cycle %s: no BBO data for %s in [+15min, +24h], skipping",
                    cycle_id, named1_coin,
                )
                skipped.append(cycle_id)
                continue

            smoothed = _smooth_1h(raw)
            smoothed_clean = smoothed.dropna()

            if len(smoothed_clean) < _MIN_SERIES_POINTS:
                logger.warning(
                    "Cycle %s: only %d valid smoothed points (need ≥%d), skipping",
                    cycle_id, len(smoothed_clean), _MIN_SERIES_POINTS,
                )
                skipped.append(cycle_id)
                continue

            n_rev, rev_times = _count_reversals(smoothed)
            mk_tau, mk_p = _mann_kendall(smoothed_clean)

            per_cycle_data.append({
                "cycle_id": cycle_id,
                "started_at": started_at,
                "raw_series": raw,
                "smoothed": smoothed,
                "reversal_count": n_rev,
                "reversal_times": rev_times,
                "mk_tau": mk_tau,
                "mk_p": mk_p,
            })

        except Exception as exc:
            logger.warning("Cycle %s: error processing: %s", cycle_id, exc)
            skipped.append(cycle_id)

    n_valid = len(per_cycle_data)

    if n_valid == 0:
        return _empty_result(
            notes=["No valid cycles after processing; check outcomes_map and bbo tables."],
            skipped=len(skipped),
        )

    reversal_counts = np.array([d["reversal_count"] for d in per_cycle_data], dtype=float)
    median_reversals = float(np.median(reversal_counts))

    mk_p_values = [d["mk_p"] for d in per_cycle_data]
    p_fisher = _fisher_combine(mk_p_values)

    # Bootstrap CI on median reversal count across cycles
    ci_lower, ci_upper = None, None
    if n_valid >= 2:
        try:
            bs = stationary_bootstrap(reversal_counts, lambda x: float(np.median(x)))
            ci_lower = bs["ci_lower_95"]
            ci_upper = bs["ci_upper_95"]
        except Exception as exc:
            logger.warning("Bootstrap CI on median reversals failed: %s", exc)

    rejected_null = (
        not np.isnan(p_fisher)
        and p_fisher < 0.05
        and not np.isnan(median_reversals)
        and median_reversals > _REVERSAL_THRESHOLD
    )

    metric_values = {
        "n_cycles": n_valid,
        "n_cycles_skipped": len(skipped),
        "median_reversals": _maybe_float(median_reversals),
        "reversal_counts": {d["cycle_id"]: d["reversal_count"] for d in per_cycle_data},
        "per_cycle_mk_tau": {
            d["cycle_id"]: _maybe_float(d["mk_tau"]) for d in per_cycle_data
        },
        "per_cycle_mk_p": {
            d["cycle_id"]: _maybe_float(d["mk_p"]) for d in per_cycle_data
        },
        "p_fisher_combined": _maybe_float(p_fisher),
        "ci_lower_95_reversals": _maybe_float(ci_lower),
        "ci_upper_95_reversals": _maybe_float(ci_upper),
        "min_reversal_magnitude_floor": _MIN_REVERSAL_MAGNITUDE,
    }

    return {
        "hypothesis_id": "H3",
        "description": "Bucket 'in range' convergence non-monotonicity",
        "metric_values": metric_values,
        "p_value": _maybe_float(p_fisher),
        "ci_lower_95": _maybe_float(ci_lower),
        "ci_upper_95": _maybe_float(ci_upper),
        "rejected_null": rejected_null,
        "oos_consistency": None,  # filled by run()
        "interpretation_notes": [
            "Rejection requires BOTH: Fisher combined MK p < 0.05 AND median reversals > 4 (at α=0.05, before FDR).",
            "Analysis window: 06:15 → next-day 06:00 UTC (continuous phase only; opening auction excluded as tested by H7).",
            "Fisher's method assumes independent tests across cycles. Cycles separated by 24h → independence reasonable but not guaranteed.",
            "Significant MK confirms directional convergence is occurring. High reversal count confirms it is oscillatory rather than monotone.",
            "Non-monotone convergence may suggest exploitable mean-reversion, but requires strict OOS validation before any strategy deployment.",
        ],
        "figures": _make_figures(per_cycle_data, conn),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _make_figures(
    per_cycle_data: list[dict],
    conn: duckdb.DuckDBPyConnection,
) -> list[plt.Figure]:
    figures: list[plt.Figure] = []
    if not per_cycle_data:
        return figures

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Figure 1: per-cycle mid_1 raw + smoothed + reversal markers
        n = len(per_cycle_data)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
        for i, d in enumerate(per_cycle_data):
            ax = axes[i, 0]
            raw = d["raw_series"]
            smoothed = d["smoothed"]
            rev_times = d["reversal_times"]

            if len(raw):
                ax.plot(
                    raw.index, raw.values,
                    linewidth=0.5, alpha=0.5, color="steelblue", label="raw BBO",
                )
            smoothed_clean = smoothed.dropna()
            if len(smoothed_clean):
                ax.plot(
                    smoothed_clean.index, smoothed_clean.values,
                    linewidth=1.5, color="navy", label="1h smooth",
                )
            if len(rev_times):
                rev_vals = smoothed.reindex(rev_times, method="nearest")
                ax.scatter(
                    rev_times, rev_vals, color="red", s=40, zorder=5,
                    label=f"reversals (n={d['reversal_count']})",
                )
            tau_str = f"{d['mk_tau']:.3f}" if not np.isnan(d["mk_tau"]) else "NaN"
            p_str = f"{d['mk_p']:.4f}" if not np.isnan(d["mk_p"]) else "NaN"
            ax.set_title(f"{d['cycle_id']} — τ={tau_str}, p_mk={p_str}", fontsize=9)
            ax.set_ylabel("mid_1")
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle("H3 — Bucket 'in range' mid trajectory (per cycle)", y=1.02)
        fig.tight_layout()
        figures.append(fig)

        # Figure 2: mid_1 + BTC mark_px normalized per cycle (optional)
        try:
            _append_normalized_figure(figures, per_cycle_data, conn)
        except Exception as exc:
            logger.warning("H3 Figure 2 (normalized comparison) failed: %s", exc)

        # Figure 3: reversal count bar chart per cycle
        fig, ax = plt.subplots(figsize=(8, 3))
        cids = [d["cycle_id"] for d in per_cycle_data]
        counts = [d["reversal_count"] for d in per_cycle_data]
        ax.bar(range(len(cids)), counts, color="steelblue", width=0.6)
        ax.axhline(
            _REVERSAL_THRESHOLD, color="red", linestyle="--", linewidth=1,
            label=f"threshold = {_REVERSAL_THRESHOLD}",
        )
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels(cids, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Reversals per cycle")
        ax.set_title("H3 — Reversal count per cycle")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures.append(fig)

    return figures


def _append_normalized_figure(
    figures: list[plt.Figure],
    per_cycle_data: list[dict],
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Append Figure 2: mid_1 vs BTC mark_px normalized, with bucket thresholds."""
    try:
        n_perp = conn.execute("SELECT COUNT(*) FROM perp_ctx").fetchone()[0]
    except Exception:
        return  # perp_ctx table not present

    if n_perp == 0:
        return  # no perp data available

    n = len(per_cycle_data)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)

    for i, d in enumerate(per_cycle_data):
        ax = axes[i, 0]
        cycle_id = d["cycle_id"]
        start = d["started_at"] + pd.Timedelta(minutes=_OPENING_DURATION_MIN)
        end = d["started_at"] + pd.Timedelta(hours=_CYCLE_DURATION_H)
        raw = d["raw_series"]

        if len(raw) > 1 and raw.max() != raw.min():
            norm = (raw - raw.min()) / (raw.max() - raw.min())
            ax.plot(norm.index, norm.values, color="steelblue", linewidth=0.8, label="mid_1 (norm)")

        try:
            perp_df = conn.execute(
                "SELECT ts_local, mark_px FROM perp_ctx "
                "WHERE ts_local >= ? AND ts_local <= ? AND mark_px IS NOT NULL "
                "ORDER BY ts_local",
                [start, end],
            ).df()
            if len(perp_df) > 1:
                ts = pd.DatetimeIndex([_to_utc(t) for t in perp_df["ts_local"]])
                mark = perp_df["mark_px"].values.astype(float)
                mn, mx = mark.min(), mark.max()
                if mx != mn:
                    mark_norm = (mark - mn) / (mx - mn)
                    ax.plot(ts, mark_norm, color="coral", linewidth=0.8, alpha=0.8, label="BTC mark (norm)")
        except Exception:
            pass

        try:
            thresh = conn.execute(
                "SELECT threshold_low, threshold_high FROM cycles WHERE cycle_id = ?",
                [cycle_id],
            ).fetchone()
            if thresh and thresh[0] is not None:
                ax.set_title(
                    f"{cycle_id} — bucket [{thresh[0]:.0f}, {thresh[1]:.0f}]", fontsize=9
                )
            else:
                ax.set_title(cycle_id, fontsize=9)
        except Exception:
            ax.set_title(cycle_id, fontsize=9)

        ax.set_ylabel("Normalized [0,1]")
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("H3 — mid_1 vs BTC mark (normalized, continuous phase)", y=1.02)
    fig.tight_layout()
    figures.append(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    conn: duckdb.DuckDBPyConnection,
    cycles_to_test: list[str] | None = None,
) -> dict:
    """
    Execute the H3 bucket convergence non-monotonicity test.

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
