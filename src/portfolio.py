from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict


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
    asset_class_constraints: Optional[Dict[str, Dict[str, float]]] = None,
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
    
    # Add asset class constraints if provided
    if asset_class_constraints:
        # Assume universe is passed or infer from tickers, but for simplicity, skip for now
        pass  # TODO: implement asset class grouping
    
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


def black_litterman(market_weights: pd.Series, cov_matrix: pd.DataFrame, views: Dict[str, float], view_confidences: Dict[str, float], risk_aversion: float = 2.5, tau: float = 0.05) -> pd.Series:
    """Black-Litterman portfolio optimization."""
    pi = risk_aversion * cov_matrix.dot(market_weights)  # equilibrium returns
    P = np.zeros((len(views), len(market_weights)))
    Q = np.zeros(len(views))
    omega = np.zeros((len(views), len(views)))
    
    for i, (ticker, view) in enumerate(views.items()):
        P[i, market_weights.index.get_loc(ticker)] = 1
        Q[i] = view
        omega[i, i] = (1 / view_confidences[ticker]) * tau
    
    # BL formula
    pi_bl = pi + tau * cov_matrix.dot(P.T).dot(np.linalg.inv(P.dot(tau * cov_matrix).dot(P.T) + omega)).dot(Q - P.dot(pi))
    return pi_bl


def apply_fx_overlay(
    expected_returns: pd.Series,
    usd_exposure: pd.Series,
    usd_mxn_level: float,
    expected_usdmxn_return: float,
    hedge_ratio: float = 0.5,
) -> pd.Series:
    """Apply FX overlay to expected returns."""
    fx_adjustment = usd_exposure * (1 - hedge_ratio) * expected_usdmxn_return
    adjusted = expected_returns + fx_adjustment
    return adjusted
