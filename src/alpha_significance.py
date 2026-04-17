from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_block_size_selector, bootstrap_paired_difference
from .settings import resolve_settings


def _annualized_return(log_returns: pd.Series) -> float:
    clean = pd.Series(log_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(np.exp(clean.mean() * 252.0) - 1.0)


def _annualized_alpha(fund: pd.Series, benchmark: pd.Series) -> float:
    return _annualized_return(fund) - _annualized_return(benchmark)


def _information_ratio(fund: pd.Series, benchmark: pd.Series) -> float:
    diff = pd.Series(fund, dtype=float) - pd.Series(benchmark, dtype=float)
    return float(np.sqrt(252.0) * diff.mean() / (diff.std(ddof=0) + 1e-9))


def _tracking_error(fund: pd.Series, benchmark: pd.Series) -> float:
    diff = pd.Series(fund, dtype=float) - pd.Series(benchmark, dtype=float)
    return float(diff.std(ddof=0) * np.sqrt(252.0))


def compute_benchmark_alpha_significance(
    returns_fund: pd.Series,
    benchmark_returns: pd.DataFrame,
    settings: dict | None = None,
) -> dict[str, dict]:
    """Compute paired stationary-bootstrap significance diagnostics vs benchmarks."""
    cfg = resolve_settings(settings)
    if benchmark_returns is None or benchmark_returns.empty or not cfg["bootstrap_enabled"]:
        return {}

    out: dict[str, dict] = {}
    for benchmark in benchmark_returns.columns:
        aligned = pd.concat(
            [pd.Series(returns_fund, dtype=float), benchmark_returns[benchmark].astype(float)],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) < 20:
            continue

        block = (
            bootstrap_block_size_selector(aligned.iloc[:, 0] - aligned.iloc[:, 1])
            if cfg["bootstrap_block_size"] == "auto"
            else int(cfg["bootstrap_block_size"])
        )
        kwargs = {
            "block_size": block,
            "n_reps": int(cfg["bootstrap_n_reps"]),
            "confidence": float(cfg["bootstrap_confidence"]),
            "seed": int(cfg["bootstrap_seed"]),
        }
        alpha_stats = bootstrap_paired_difference(aligned.iloc[:, 0], aligned.iloc[:, 1], _annualized_alpha, **kwargs)
        ir_stats = bootstrap_paired_difference(aligned.iloc[:, 0], aligned.iloc[:, 1], _information_ratio, **kwargs)
        te_stats = bootstrap_paired_difference(aligned.iloc[:, 0], aligned.iloc[:, 1], _tracking_error, **kwargs)
        te_stats.pop("p_value", None)
        out[str(benchmark)] = {
            "alpha_annualized": alpha_stats,
            "information_ratio": ir_stats,
            "tracking_error": te_stats,
        }
    return out
