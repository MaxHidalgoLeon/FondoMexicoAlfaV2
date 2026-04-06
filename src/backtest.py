from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

from .portfolio import optimize_portfolio
from .risk import compute_cvar, max_drawdown, compute_sortino, compute_sharpe


def get_rebalance_dates(prices: pd.DataFrame, freq: str = "ME") -> pd.DatetimeIndex:
    return prices.resample(freq).first().index


def build_covariance_matrix(returns: pd.DataFrame, date: pd.Timestamp, window: int = 63) -> pd.DataFrame:
    subset = returns.loc[:date].tail(window)
    return subset.cov().fillna(0.0)


def run_backtest(
    prices: pd.DataFrame,
    signal_df: pd.DataFrame,
    universe: pd.DataFrame,
    transaction_cost: float = 0.001,
    rebalance_freq: str = "M",
) -> Dict[str, pd.DataFrame]:
    returns = prices.pct_change().fillna(0.0)
    rebalance_dates = get_rebalance_dates(prices, rebalance_freq)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    portfolio_returns = pd.Series(0.0, index=prices.index)
    prev_weights = pd.Series(0.0, index=prices.columns)
    turnover = pd.Series(0.0, index=prices.index)

    for date in rebalance_dates:
        if date not in prices.index:
            continue
        date_signal = signal_df[signal_df["date"] == date]
        if date_signal.empty:
            continue
        expected_returns = date_signal.set_index("ticker")["expected_return"].reindex(prices.columns).fillna(0.0)
        cov_matrix = build_covariance_matrix(returns, date)
        try:
            target_weights = optimize_portfolio(
                expected_returns,
                cov_matrix,
                prev_weights=prev_weights,
                max_position=0.15,
                min_position=0.0,
                target_net_exposure=0.9,
                risk_aversion=4.0,
                turnover_penalty=0.05,
            )
        except RuntimeError:
            target_weights = prev_weights
        weights.loc[date:, :] = target_weights.values
        turnover.loc[date] = np.sum(np.abs(target_weights - prev_weights))
        prev_weights = target_weights

    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
    transaction_costs = turnover * transaction_cost
    portfolio_returns = portfolio_returns - transaction_costs
    metrics = {
        "sharpe": compute_sharpe(portfolio_returns),
        "sortino": compute_sortino(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_returns),
        "cvar_95": compute_cvar(portfolio_returns, alpha=0.95),
        "annualized_return": ((1 + portfolio_returns).prod() ** (252 / len(portfolio_returns))) - 1,
        "annualized_vol": portfolio_returns.std() * np.sqrt(252),
        "turnover": turnover.mean(),
    }
    return {
        "weights": weights,
        "returns": portfolio_returns,
        "metrics": metrics,
        "turnover": turnover,
    }
