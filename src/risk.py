from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / (excess.std(ddof=0) + 1e-9)


def compute_sortino(returns: pd.Series, required_return: float = 0.0) -> float:
    downside = returns[returns < required_return]
    downside_std = np.sqrt((downside**2).mean()) if not downside.empty else 0.0
    return returns.mean() / (downside_std + 1e-9) * np.sqrt(252)


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def compute_var(returns: pd.Series, alpha: float = 0.95) -> float:
    return np.percentile(returns.dropna(), 100 * (1 - alpha))


def compute_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    var_level = compute_var(returns, alpha)
    tail = returns[returns <= var_level]
    return tail.mean() if not tail.empty else var_level


def stress_test(
    portfolio_returns: pd.Series,
    scenario_shocks: dict[str, float],
    exposures: dict[str, float],
) -> pd.DataFrame:
    scenario_results = []
    base_return = portfolio_returns.mean()
    for label, shock in scenario_shocks.items():
        exposure = exposures.get(label, 0.0)
        adjusted = portfolio_returns + shock * exposure
        scenario_results.append(
            {
                "scenario": label,
                "mean_return": adjusted.mean(),
                "volatility": adjusted.std(ddof=0),
                "sharpe": compute_sharpe(adjusted),
                "max_drawdown": max_drawdown(adjusted),
            }
        )
    return pd.DataFrame(scenario_results)
