"""
H7 — Opening auction overnight bias.

Hypothesis: the auction-period price move (05:59→06:15 UTC) predicts the
post-open correction (06:15→06:20 UTC), suggesting the auction under- or
over-reacts to overnight information.

Rejection criterion (ALL THREE must hold, from hypotheses.md):
  - |β| > 0.15 at α=0.05 (before FDR correction)
  - R² > 0.10
  - sign(Δ_auction_i × Δ_postopen_i) == sign(β) for ≥5 of the tested cycles
"""

from __future__ import annotations

import logging
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analyzer.stats import leave_one_cycle_out, stationary_bootstrap

logger = logging.getLogger(__name__)

_PRE_OFFSET_MIN = -1         # T_pre  = started_at - 1min  → 05:59 UTC
_END_OFFSET_MIN = 15         # T_end  = started_at + 15min → 06:15 UTC
_POST_OFFSET_MIN = 20        # T_post = started_at + 20min → 06:20 UTC
_SNAPSHOT_TOLERANCE_S = 5
_OLS_MIN_N = 3
_BETA_THRESHOLD = 0.15
_R2_THRESHOLD = 0.10
_SIGN_CONSISTENCY_MIN = 5


# ---------------------------------------------------------------------------
# Helpers
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


def _get_binary_coin(conn: duckdb.DuckDBPyConnection, cycle_id: str) -> str | None:
    row = conn.execute(
        "SELECT yes_coin FROM outcomes_map WHERE cycle_id = ? AND role = 'binary'",
        [cycle_id],
    ).fetchone()
    if row is None:
        logger.warning("Cycle %s: no binary outcome in outcomes_map, skipping", cycle_id)
        return None
    return row[0]


def _get_bbo_df(conn: duckdb.DuckDBPyConnection, coin: str) -> pd.DataFrame:
    """Load BBO for a coin; filter valid quotes; compute mid."""
    df = conn.execute(
        "SELECT ts_local, bid_px, ask_px FROM bbo WHERE coin = ? ORDER BY ts_local",
        [coin],
    ).df()
    df = df[(df["bid_px"] > 0) & (df["ask_px"] > 0)].copy()
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    df = df.drop_duplicates(subset="ts_local", keep="last")
    return df[["ts_local", "mid"]].sort_values("ts_local").reset_index(drop=True)


def _nearest_snapshot(
    bbo_df: pd.DataFrame,
    target_ts: pd.Timestamp,
    tolerance_s: int = _SNAPSHOT_TOLERANCE_S,
) -> float | None:
    """
    Return the mid of the BBO snapshot closest to target_ts within ±tolerance_s.

    Why not merge_asof_safe: the target timestamps (T_pre = 05:59, T_end = 06:15,
    T_post = 06:20) are fixed by the cycle structure, NOT derived from the data
    series. With fixed targets there is no look-ahead bias regardless of search
    direction. A nearest-neighbour lookup within a fixed window is the correct
    and sufficient tool — merge_asof is designed for rolling series joins, not
    point lookups.

    Tie-breaking: if two snapshots are equidistant, the more recent is used.
    """
    tol = pd.Timedelta(seconds=tolerance_s)
    window = bbo_df[
        (bbo_df["ts_local"] >= target_ts - tol) & (bbo_df["ts_local"] <= target_ts + tol)
    ].copy()
    if window.empty:
        return None
    window["dist"] = (window["ts_local"] - target_ts).abs()
    row = window.sort_values(["dist", "ts_local"], ascending=[True, False]).iloc[0]
    return float(row["mid"])


def _extract_cycle_deltas(
    conn: duckdb.DuckDBPyConnection,
    cycle_id: str,
    binary_coin: str,
    started_at: pd.Timestamp,
) -> dict | None:
    """
    Extract the three snapshot mids and compute Δ_auction / Δ_postopen.
    Returns None if any snapshot is unavailable within ±5s.
    """
    bbo = _get_bbo_df(conn, binary_coin)

    t_pre = started_at + pd.Timedelta(minutes=_PRE_OFFSET_MIN)
    t_end = started_at + pd.Timedelta(minutes=_END_OFFSET_MIN)
    t_post = started_at + pd.Timedelta(minutes=_POST_OFFSET_MIN)

    mid_pre = _nearest_snapshot(bbo, t_pre)
    mid_end = _nearest_snapshot(bbo, t_end)
    mid_post = _nearest_snapshot(bbo, t_post)

    missing = [name for name, val in [("T_pre", mid_pre), ("T_end", mid_end), ("T_post", mid_post)] if val is None]
    if missing:
        logger.warning(
            "Cycle %s: snapshot(s) missing at %s (tolerance ±%ds), skipping",
            cycle_id, missing, _SNAPSHOT_TOLERANCE_S,
        )
        return None

    return {
        "cycle_id": cycle_id,
        "binary_coin": binary_coin,
        "started_at": started_at,
        "mid_pre": mid_pre,
        "mid_end": mid_end,
        "mid_post": mid_post,
        "delta_auction": mid_end - mid_pre,
        "delta_postopen": mid_post - mid_end,
    }


def _fit_ols(
    delta_auctions: list[float],
    delta_postoopens: list[float],
) -> tuple[float, float, float, float]:
    """
    OLS: Δ_postopen = α + β × Δ_auction.
    Returns (β, α, p_value_β, R²) or (nan, nan, nan, nan) on failure.
    """
    X = sm.add_constant(np.array(delta_auctions, dtype=float))
    Y = np.array(delta_postoopens, dtype=float)
    try:
        fit = sm.OLS(Y, X).fit()
        return float(fit.params[1]), float(fit.params[0]), float(fit.pvalues[1]), float(fit.rsquared)
    except Exception as exc:
        logger.warning("OLS failed: %s", exc)
        return float("nan"), float("nan"), float("nan"), float("nan")


def _beta_from_pairs(data: np.ndarray) -> float:
    """Bootstrap statistic: OLS β on a resampled (Δ_auction, Δ_postopen) array."""
    if len(data) < _OLS_MIN_N:
        return float("nan")
    try:
        X = sm.add_constant(data[:, 0])
        return float(sm.OLS(data[:, 1], X).fit().params[1])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Empty result helper
# ---------------------------------------------------------------------------


def _empty_result(notes: list[str] | None = None, skipped: int = 0) -> dict:
    return {
        "hypothesis_id": "H7",
        "description": "Opening auction overnight bias",
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
# Core computation (no OOS)
# ---------------------------------------------------------------------------


def _compute_metrics(
    conn: duckdb.DuckDBPyConnection,
    cycles: list[str],
) -> dict:
    """
    Core H7 computation. Does NOT perform OOS validation.
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
        binary_coin = _get_binary_coin(conn, cycle_id)
        if binary_coin is None:
            skipped.append(cycle_id)
            continue
        started_at = cycle_meta.get(cycle_id)
        if started_at is None:
            logger.warning("Cycle %s: started_at not found, skipping", cycle_id)
            skipped.append(cycle_id)
            continue
        try:
            entry = _extract_cycle_deltas(conn, cycle_id, binary_coin, started_at)
            if entry is None:
                skipped.append(cycle_id)
            else:
                per_cycle_data.append(entry)
        except Exception as exc:
            logger.warning("Cycle %s: %s", cycle_id, exc)
            skipped.append(cycle_id)

    n_valid = len(per_cycle_data)

    if n_valid < _OLS_MIN_N:
        result = _empty_result(
            notes=[f"Only {n_valid} valid cycle(s); need ≥{_OLS_MIN_N} for OLS."],
            skipped=len(skipped),
        )
        result["metric_values"].update({
            "n_cycles": n_valid,
            "per_cycle": [
                {k: v for k, v in d.items() if k not in ("binary_coin", "started_at")}
                for d in per_cycle_data
            ],
            "beta_loo_estimates": None,
        })
        return result

    delta_auctions = [d["delta_auction"] for d in per_cycle_data]
    delta_postoopens = [d["delta_postopen"] for d in per_cycle_data]

    beta_hat, alpha_hat, p_value_beta, r_squared = _fit_ols(delta_auctions, delta_postoopens)

    # Bootstrap CI on β (stationary_bootstrap with pairs; block_size=1 with n=7 → iid resample)
    ci_lower, ci_upper = None, None
    if not np.isnan(beta_hat):
        try:
            pairs = np.column_stack([delta_auctions, delta_postoopens])
            bs = stationary_bootstrap(pairs, _beta_from_pairs)
            ci_lower = bs["ci_lower_95"]
            ci_upper = bs["ci_upper_95"]
        except Exception as exc:
            logger.warning("Bootstrap CI on β failed: %s", exc)

    # Sign consistency — Option C: per-cycle direction check
    # For each cycle i, check sign(Δ_auction_i × Δ_postopen_i) == sign(β_hat).
    # This tests whether individual cycles show the pattern predicted by the model,
    # not whether β_LOO is numerically stable (that is logged separately in run()).
    n_sign_consistent = 0
    if not np.isnan(beta_hat) and beta_hat != 0:
        for d in per_cycle_data:
            if np.sign(d["delta_auction"] * d["delta_postopen"]) == np.sign(beta_hat):
                n_sign_consistent += 1

    rejected_null = (
        not np.isnan(beta_hat)
        and abs(beta_hat) > _BETA_THRESHOLD
        and not np.isnan(p_value_beta)
        and p_value_beta < 0.05
        and not np.isnan(r_squared)
        and r_squared > _R2_THRESHOLD
        and n_sign_consistent >= _SIGN_CONSISTENCY_MIN
    )

    metric_values = {
        "n_cycles": n_valid,
        "n_cycles_skipped": len(skipped),
        "per_cycle": [
            {k: v for k, v in d.items() if k not in ("binary_coin", "started_at")}
            for d in per_cycle_data
        ],
        "beta_hat": _maybe_float(beta_hat),
        "alpha_hat": _maybe_float(alpha_hat),
        "r_squared": _maybe_float(r_squared),
        "p_value_beta": _maybe_float(p_value_beta),
        "n_cycles_sign_consistent": n_sign_consistent,
        "beta_loo_estimates": None,  # populated by run() after OOS
    }

    return {
        "hypothesis_id": "H7",
        "description": "Opening auction overnight bias",
        "metric_values": metric_values,
        "p_value": _maybe_float(p_value_beta),
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "rejected_null": rejected_null,
        "oos_consistency": None,  # filled by run()
        "interpretation_notes": [
            "Rejection requires ALL: |β|>0.15, p<0.05, R²>0.10, sign consistent ≥5 cycles.",
            f"With n={n_valid} cycle(s), statistical power is limited — a positive signal requires "
            "3-6 months of additional data before any capital deployment.",
            "β>0: post-open continues auction direction (under-reaction). "
            "β<0: post-open reverses (over-reaction).",
        ],
        "figures": _make_figures(per_cycle_data, beta_hat, alpha_hat, conn),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _make_figures(
    per_cycle_data: list[dict],
    beta_hat: float,
    alpha_hat: float,
    conn: duckdb.DuckDBPyConnection,
) -> list[plt.Figure]:
    figures: list[plt.Figure] = []
    if not per_cycle_data:
        return figures

    delta_a = np.array([d["delta_auction"] for d in per_cycle_data])
    delta_p = np.array([d["delta_postopen"] for d in per_cycle_data])
    cycle_ids = [d["cycle_id"] for d in per_cycle_data]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Figure 1: scatter Δ_auction vs Δ_postopen with OLS line
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(delta_a, delta_p, color="steelblue", zorder=3)
        for cid, x, y in zip(cycle_ids, delta_a, delta_p):
            ax.annotate(cid, (x, y), fontsize=7, textcoords="offset points", xytext=(4, 4))
        if not np.isnan(beta_hat) and len(delta_a) >= 2:
            x_line = np.linspace(delta_a.min(), delta_a.max(), 50)
            ax.plot(x_line, alpha_hat + beta_hat * x_line, color="red", linewidth=1.2,
                    label=f"OLS β={beta_hat:.3f}")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Δ_auction  (mid at 06:15 − mid at 05:59)")
        ax.set_ylabel("Δ_postopen  (mid at 06:20 − mid at 06:15)")
        ax.set_title("H7 — Auction move vs post-open correction")
        ax.legend()
        fig.tight_layout()
        figures.append(fig)

        # Figure 2: time series of binary mid during opening, one subplot per cycle
        # Window: [started_at − 15min, started_at + 25min] → 05:45–06:25 UTC
        try:
            n = len(per_cycle_data)
            fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), squeeze=False)
            for i, entry in enumerate(per_cycle_data):
                ax = axes[i, 0]
                coin = entry["binary_coin"]
                t0 = entry["started_at"]
                t_win_start = t0 - pd.Timedelta(minutes=15)
                t_win_end = t0 + pd.Timedelta(minutes=25)
                bbo = _get_bbo_df(conn, coin)
                sub = bbo[(bbo["ts_local"] >= t_win_start) & (bbo["ts_local"] <= t_win_end)]
                if len(sub):
                    # Align x-axis to minutes relative to started_at
                    rel_min = (sub["ts_local"] - t0).dt.total_seconds() / 60
                    ax.plot(rel_min, sub["mid"], linewidth=0.8, color="steelblue")
                for label, offset in [("T_pre (−1′)", _PRE_OFFSET_MIN),
                                       ("T_end (+15′)", _END_OFFSET_MIN),
                                       ("T_post (+20′)", _POST_OFFSET_MIN)]:
                    ax.axvline(offset, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
                ax.set_title(entry["cycle_id"], fontsize=9)
                ax.set_xlabel("Minutes relative to 06:00 UTC")
                ax.set_ylabel("Binary mid")
            fig.suptitle("H7 — Binary mid during opening (per cycle)", y=1.01)
            fig.tight_layout()
            figures.append(fig)
        except Exception as exc:
            logger.warning("Figure 2 (time series) generation failed: %s", exc)

    return figures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    conn: duckdb.DuckDBPyConnection,
    cycles_to_test: list[str] | None = None,
) -> dict:
    """
    Execute the H7 opening auction overnight bias test.

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
            oos = leave_one_cycle_out(
                cycles,
                lambda remaining: _compute_metrics(conn, remaining),
            )
            result["oos_consistency"] = oos

            # Extract β_LOO from each fold and store in metric_values (informational only)
            beta_loo = {
                cid: v.get("metric_values", {}).get("beta_hat")
                for cid, v in oos.items()
                if cid != "aggregate" and isinstance(v, dict)
            }
            result["metric_values"]["beta_loo_estimates"] = beta_loo

            # Figure: β_LOO bar chart (appended after scatter)
            _append_beta_loo_figure(result, beta_loo)

        except Exception as exc:
            logger.warning("OOS validation failed: %s", exc)
            result["oos_consistency"] = {"error": str(exc)}
    else:
        result["oos_consistency"] = {"note": "Need ≥2 cycles for OOS validation."}

    return result


def _append_beta_loo_figure(result: dict, beta_loo: dict) -> None:
    """Append a bar chart of β_LOO estimates to result['figures']."""
    loo_items = [(cid, v) for cid, v in beta_loo.items() if v is not None]
    if not loo_items:
        return
    beta_full = result["metric_values"].get("beta_hat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots(figsize=(8, 3))
        cids = [x[0] for x in loo_items]
        betas = [x[1] for x in loo_items]
        ax.bar(range(len(cids)), betas, color="steelblue", width=0.6)
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels(cids, rotation=30, ha="right", fontsize=8)
        if beta_full is not None:
            ax.axhline(beta_full, color="red", linestyle="--", linewidth=1, label=f"β_full={beta_full:.3f}")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(_BETA_THRESHOLD, color="orange", linestyle=":", linewidth=0.8, label=f"threshold={_BETA_THRESHOLD}")
        ax.axhline(-_BETA_THRESHOLD, color="orange", linestyle=":", linewidth=0.8)
        ax.set_ylabel("β (LOO estimate)")
        ax.set_title("H7 — β stability across leave-one-cycle-out folds")
        ax.legend(fontsize=8)
        fig.tight_layout()
        result["figures"].append(fig)
