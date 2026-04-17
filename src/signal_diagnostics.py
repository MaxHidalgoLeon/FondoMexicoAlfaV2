from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .bootstrap import bootstrap_metric
from .settings import resolve_settings

SIGNAL_COLUMNS = [
    "momentum_63",
    "momentum_126",
    "volatility_63",
    "value_score",
    "quality_score",
    "liquidity_score",
]


def _sign_p_value(point: float, distribution: np.ndarray) -> float:
    dist = np.asarray(distribution, dtype=float)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0 or abs(point) < 1e-12:
        return 1.0
    if point > 0:
        return float(np.mean(dist <= 0.0))
    return float(np.mean(dist >= 0.0))


def compute_signal_ic_diagnostics(
    feature_df: pd.DataFrame,
    forecast_df: pd.DataFrame | None = None,
    settings: dict | None = None,
) -> dict[str, dict]:
    """Compute monthly Spearman IC diagnostics with stationary-bootstrap CIs."""
    cfg = resolve_settings(settings)
    if feature_df is None or feature_df.empty or not cfg["ic_diagnostics_enabled"]:
        return {}

    df = feature_df.copy().sort_values(["ticker", "date"])
    if "price" not in df.columns:
        return {}
    df["forward_return_21d"] = df.groupby("ticker")["price"].transform(
        lambda s: np.log(s.shift(-21) / (s + 1e-9))
    )

    if forecast_df is not None and not forecast_df.empty and "date" in forecast_df.columns:
        rebalance_dates = pd.DatetimeIndex(sorted(pd.to_datetime(forecast_df["date"]).dropna().unique()))
        df = df[df["date"].isin(rebalance_dates)]
    else:
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
        df = df.groupby(["month", "ticker"], as_index=False).tail(1)

    diagnostics: dict[str, dict] = {}
    for signal in [c for c in SIGNAL_COLUMNS if c in df.columns]:
        ic_values: list[float] = []
        ic_dates: list[pd.Timestamp] = []
        for date, group in df.groupby("date"):
            valid = group[[signal, "forward_return_21d"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 4:
                continue
            corr, _ = spearmanr(valid[signal], valid["forward_return_21d"])
            if np.isfinite(corr):
                ic_values.append(float(corr))
                ic_dates.append(pd.Timestamp(date))

        if len(ic_values) < 4:
            continue

        ic_series = pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates), dtype=float).sort_index()
        stats = bootstrap_metric(
            ic_series,
            lambda s: float(pd.Series(s, dtype=float).mean()),
            block_size=int(cfg["ic_bootstrap_block_size"]),
            n_reps=int(cfg["bootstrap_n_reps"]),
            confidence=float(cfg["bootstrap_confidence"]),
            seed=int(cfg["bootstrap_seed"]),
        )
        point = float(ic_series.mean())
        diagnostics[signal] = {
            "ic_mean": point,
            "ic_t_stat": float(point / (ic_series.std(ddof=1) + 1e-9) * np.sqrt(len(ic_series))),
            "ci_low": float(stats["ci_low"]),
            "ci_high": float(stats["ci_high"]),
            "p_value": _sign_p_value(point, stats["distribution"]),
            "significant": bool(_sign_p_value(point, stats["distribution"]) < 0.05),
            "n_periods": int(len(ic_series)),
            "series": ic_series,
        }
    return diagnostics
