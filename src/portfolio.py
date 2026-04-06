from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


def _portfolio_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
    turnover_penalty: float,
    prev_weights: np.ndarray,
) -> float:
    expected_cost = -weights.dot(expected_returns)
    variance = risk_aversion * weights.dot(cov_matrix).dot(weights)
    turnover = turnover_penalty * np.sum(np.abs(weights - prev_weights))
    return expected_cost + variance + turnover


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    max_position: float = 0.15,
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.1,
) -> pd.Series:
    tickers = expected_returns.index.tolist()
    if prev_weights is None:
        prev_weights = pd.Series(0.0, index=tickers)

    x0 = np.repeat(target_net_exposure / len(tickers), len(tickers))
    bounds = [(min_position, max_position)] * len(tickers)
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - target_net_exposure,
        }
    ]
    result = minimize(
        _portfolio_objective,
        x0,
        args=(expected_returns.values, cov_matrix.values, risk_aversion, turnover_penalty, prev_weights.values),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return pd.Series(result.x, index=tickers)


def apply_fx_overlay(
    expected_returns: pd.Series,
    usd_exposure: pd.Series,
    usd_mxn_level: float,
    hedge_ratio: float = 0.5,
) -> pd.Series:
    fx_sensitivity = usd_exposure * 0.5
    fx_adjustment = -hedge_ratio * fx_sensitivity * np.log(usd_mxn_level / max(usd_mxn_level, 1.0))
    adjusted = expected_returns + fx_adjustment
    return adjusted
