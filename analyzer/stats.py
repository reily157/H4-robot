"""
Statistical utilities shared across all hypothesis modules.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm
from arch.bootstrap import StationaryBootstrap


def fdr_correction(
    p_values: list[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """
    Benjamini-Hochberg FDR correction.

    Returns (rejected, adjusted_p_values).
    """
    rejected, p_adj, _, _ = smm.multipletests(p_values, alpha=alpha, method="fdr_bh")
    return list(rejected), list(p_adj)


def stationary_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 10_000,
    block_size: int | None = None,
) -> dict:
    """
    Politis-Romano stationary bootstrap for autocorrelated time series.

    block_size: average block length. If None, estimated as max(1, n^(1/3)).
    Returns dict with mean, std, ci_lower_95, ci_upper_95, all_samples.
    """
    data = np.asarray(data)
    n = len(data)
    if block_size is None:
        block_size = max(1, int(round(n ** (1 / 3))))

    bs = StationaryBootstrap(block_size, data, seed=42)
    samples = np.array([statistic(d[0]) for d in bs.bootstrap(n_boot)])

    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "ci_lower_95": float(np.percentile(samples, 2.5)),
        "ci_upper_95": float(np.percentile(samples, 97.5)),
        "all_samples": samples,
    }


def leave_one_cycle_out(
    cycles: list[str],
    hypothesis_test: Callable[[list[str]], dict],
) -> dict:
    """
    OOS validation: for each cycle, exclude it and run hypothesis_test on the rest.

    Returns:
        {
            cycle_id: result_dict,   # one entry per held-out cycle
            'aggregate': {
                'n_cycles': int,
                'values': list[float | None],   # main metric from each result
                'mean': float | None,
                'std': float | None,
                'min': float | None,
                'max': float | None,
            }
        }

    Each hypothesis module is responsible for interpreting 'aggregate' according
    to its own consistency criterion (e.g. H7 checks sign consistency >= 5/7).
    """
    results: dict = {}
    for held_out in cycles:
        remaining = [c for c in cycles if c != held_out]
        if not remaining:
            continue
        results[held_out] = hypothesis_test(remaining)

    # Descriptive aggregate — no boolean 'consistent' flag; each hypothesis defines its own
    p_values = [v.get("p_value") for v in results.values() if isinstance(v, dict)]
    numeric = [p for p in p_values if p is not None]

    results["aggregate"] = {
        "n_cycles": len(cycles),
        "values": p_values,
        "mean": float(np.mean(numeric)) if numeric else None,
        "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else None,
        "min": float(np.min(numeric)) if numeric else None,
        "max": float(np.max(numeric)) if numeric else None,
    }
    return results


def merge_asof_safe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    by: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper around pd.merge_asof that ALWAYS uses direction='backward'
    to prevent look-ahead bias.

    Raises ValueError if the 'on' column in left is not monotonically sorted.
    Raises TypeError if caller tries to pass direction= explicitly.
    """
    if "direction" in kwargs:
        raise TypeError(
            "merge_asof_safe does not accept a 'direction' argument. "
            "Direction is hardcoded to 'backward' to prevent look-ahead bias."
        )
    if not left[on].is_monotonic_increasing:
        raise ValueError(
            f"Column '{on}' in left DataFrame must be sorted (monotonically increasing) "
            "before calling merge_asof_safe."
        )
    merge_kwargs: dict = {"direction": "backward"}
    if by is not None:
        merge_kwargs["by"] = by
    return pd.merge_asof(left, right, on=on, **merge_kwargs, **kwargs)
