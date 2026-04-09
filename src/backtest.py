from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.covariance import LedoitWolf

from .portfolio import optimize_portfolio, optimize_portfolio_cvar, optimize_portfolio_robust
from .risk import compute_cvar, max_drawdown, compute_sortino, compute_sharpe, detect_macro_regime, regime_asset_class_constraints

logger = logging.getLogger(__name__)


def get_rebalance_dates(prices: pd.DataFrame, freq: str = "ME") -> pd.DatetimeIndex:
    """Return the last available trading date in each rebalancing period."""
    resampled = prices.resample(freq).last().dropna(how="all")
    # Snap to the nearest actual trading day that exists in prices.index
    snapped = []
    for dt in resampled.index:
        # Find the last trading day on or before dt
        candidates = prices.index[prices.index <= dt]
        if len(candidates):
            snapped.append(candidates[-1])
    return pd.DatetimeIndex(sorted(set(snapped)))


def build_covariance_matrix(
    returns: pd.DataFrame, date: pd.Timestamp, window: int = 63
) -> pd.DataFrame:
    """Build a Ledoit-Wolf shrunk covariance matrix using at most `window` days."""
    subset = returns.loc[:date].tail(window).fillna(0.0)
    if subset.shape[0] < 10:
        # Fallback to identity-scaled matrix when data is scarce
        n = len(returns.columns)
        avg_var = returns.var().mean() if not returns.empty else 1e-4
        return pd.DataFrame(
            np.eye(n) * avg_var, index=returns.columns, columns=returns.columns
        )
    try:
        lw = LedoitWolf()
        lw.fit(subset)
        cov_array = lw.covariance_
        return pd.DataFrame(cov_array, index=subset.columns, columns=subset.columns).reindex(
            index=returns.columns, columns=returns.columns
        ).fillna(0.0)
    except Exception:
        return subset.cov().reindex(
            index=returns.columns, columns=returns.columns
        ).fillna(0.0)


def run_backtest(
    prices: pd.DataFrame,
    signal_df: pd.DataFrame,
    universe: pd.DataFrame,
    transaction_cost: float = 0.001,
    rebalance_freq: str = "ME",
    risk_free_rate: float = 0.02,
    asset_class_constraints: Optional[Dict] = None,
    optimizer: str = "mv",
    adtv_scores: Optional[pd.Series] = None,
    macro: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Run backtest.

    optimizer: "mv"      — mean-variance (SLSQP) only
               "cvar"   — mean-CVaR only
               "robust" — Michaud resampled MV only
               "both"   — MV + CVaR comparison; standard keys hold MV
                          results, extra keys (*_cvar) hold CVaR results.

    adtv_scores : per-ticker ADTV liquidity scores (0–1) forwarded to the
                  optimizer as market-impact weights.
    macro       : macro DataFrame used by detect_macro_regime() at each
                  rebalance date to override static asset-class constraints.
    """
    returns = prices.pct_change().fillna(0.0)
    rebalance_dates = get_rebalance_dates(prices, rebalance_freq)

    run_mv     = optimizer in ("mv",     "both")
    run_cvar   = optimizer in ("cvar",   "both")
    run_robust = optimizer == "robust"

    # Prepare macro data for regime detection (snap 'date' column to index once)
    _macro_indexed: Optional[pd.DataFrame] = None
    if macro is not None:
        _macro_indexed = macro.copy()
        if "date" in _macro_indexed.columns:
            _macro_indexed = _macro_indexed.set_index("date")
        _macro_indexed = _macro_indexed.sort_index()

    # MV tracking — only allocate when needed
    if run_mv:
        weights_mv    = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev_w_mv     = pd.Series(0.0, index=prices.columns)
        turnover_mv   = pd.Series(0.0, index=prices.index)

    # CVaR tracking — only allocate when needed
    if run_cvar:
        weights_cvar  = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev_w_cvar   = pd.Series(0.0, index=prices.columns)
        turnover_cvar = pd.Series(0.0, index=prices.index)

    # Robust (Michaud) tracking
    if run_robust:
        weights_rob   = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev_w_rob    = pd.Series(0.0, index=prices.columns)
        turnover_rob  = pd.Series(0.0, index=prices.index)

    for date in rebalance_dates:
        if date not in prices.index:
            continue
        date_signal = signal_df[signal_df["date"] == date]
        if date_signal.empty:
            logger.warning("No signal for rebalance date %s — carrying previous weights.", date)
            continue
        expected_returns = date_signal.set_index("ticker")["expected_return"].reindex(prices.columns).fillna(0.0)
        cov_matrix = build_covariance_matrix(returns, date)

        # --- Regime-based constraint override ---
        effective_constraints = asset_class_constraints
        if _macro_indexed is not None and asset_class_constraints is not None:
            macro_up_to_date = _macro_indexed[_macro_indexed.index <= date]
            # restore date column so detect_macro_regime() accepts both forms
            _tmp = macro_up_to_date.reset_index().rename(columns={"index": "date"})
            regime = detect_macro_regime(_tmp)
            regime_bounds = regime_asset_class_constraints(regime)
            # Merge regime bounds into constraints (preserve __asset_class_map__)
            effective_constraints = dict(asset_class_constraints)
            for ac, bounds in regime_bounds.items():
                if ac in effective_constraints:
                    effective_constraints[ac] = bounds
            logger.debug("Rebalance %s: macro regime = %s", date.date(), regime)

        if run_mv:
            try:
                target_mv = optimize_portfolio(
                    expected_returns, cov_matrix,
                    prev_weights=prev_w_mv,
                    max_position=0.15, min_position=0.0,
                    target_net_exposure=0.9, risk_aversion=4.0,
                    turnover_penalty=0.05,
                    asset_class_constraints=effective_constraints,
                    adtv_scores=adtv_scores,
                )
            except RuntimeError as exc:
                logger.warning("MV optimization failed on %s: %s — carrying previous weights.", date, exc)
                target_mv = prev_w_mv
            weights_mv.loc[date:, :] = target_mv.values
            turnover_mv.loc[date] = np.sum(np.abs(target_mv - prev_w_mv))
            prev_w_mv = target_mv

        if run_cvar:
            scen = returns.loc[:date].tail(63)
            try:
                target_cvar = optimize_portfolio_cvar(
                    expected_returns, scen,
                    prev_weights=prev_w_cvar,
                    max_position=0.15, min_position=0.0,
                    target_net_exposure=0.9, risk_aversion=4.0,
                    turnover_penalty=0.05, alpha=0.95,
                    asset_class_constraints=effective_constraints,
                    adtv_scores=adtv_scores,
                )
            except RuntimeError as exc:
                logger.warning("CVaR optimization failed on %s: %s — carrying previous weights.", date, exc)
                target_cvar = prev_w_cvar
            weights_cvar.loc[date:, :] = target_cvar.values
            turnover_cvar.loc[date] = np.sum(np.abs(target_cvar - prev_w_cvar))
            prev_w_cvar = target_cvar

        if run_robust:
            try:
                target_rob = optimize_portfolio_robust(
                    expected_returns, cov_matrix,
                    prev_weights=prev_w_rob,
                    max_position=0.15, min_position=0.0,
                    target_net_exposure=0.9, risk_aversion=4.0,
                    turnover_penalty=0.05,
                    asset_class_constraints=effective_constraints,
                    adtv_scores=adtv_scores,
                )
            except RuntimeError as exc:
                logger.warning("Robust optimization failed on %s: %s — carrying previous weights.", date, exc)
                target_rob = prev_w_rob
            weights_rob.loc[date:, :] = target_rob.values
            turnover_rob.loc[date] = np.sum(np.abs(target_rob - prev_w_rob))
            prev_w_rob = target_rob

    def _compute_returns_and_metrics(w: pd.DataFrame, tv: pd.Series) -> tuple:
        port_ret = (w.shift(1) * returns).sum(axis=1)
        port_ret = port_ret - tv * transaction_cost
        ann_ret = ((1 + port_ret).prod() ** (252 / max(len(port_ret), 1))) - 1
        mdd = max_drawdown(port_ret)
        m = {
            "sharpe": compute_sharpe(port_ret, risk_free_rate=risk_free_rate),
            "sortino": compute_sortino(port_ret, required_return=risk_free_rate / 252),
            "max_drawdown": mdd,
            "calmar": ann_ret / max(abs(mdd), 1e-9),
            "cvar_95": compute_cvar(port_ret, alpha=0.95),
            "annualized_return": ann_ret,
            "annualized_vol": port_ret.std() * np.sqrt(252),
            "turnover": tv.mean(),
        }
        return port_ret, m

    # Build output — standard keys always hold the "primary" optimizer's results
    if run_mv:
        portfolio_returns, metrics = _compute_returns_and_metrics(weights_mv, turnover_mv)
        out = {
            "weights": weights_mv,
            "returns": portfolio_returns,
            "metrics": metrics,
            "turnover": turnover_mv,
        }
    elif run_robust:
        portfolio_returns, metrics = _compute_returns_and_metrics(weights_rob, turnover_rob)
        out = {
            "weights": weights_rob,
            "returns": portfolio_returns,
            "metrics": metrics,
            "turnover": turnover_rob,
        }
    else:
        portfolio_returns, metrics = _compute_returns_and_metrics(weights_cvar, turnover_cvar)
        out = {
            "weights": weights_cvar,
            "returns": portfolio_returns,
            "metrics": metrics,
            "turnover": turnover_cvar,
        }

    # When "both", attach CVaR results under _cvar keys for comparison
    if optimizer == "both":
        ret_cvar, met_cvar = _compute_returns_and_metrics(weights_cvar, turnover_cvar)
        out["weights_cvar"]  = weights_cvar
        out["returns_cvar"]  = ret_cvar
        out["metrics_cvar"]  = met_cvar
        out["turnover_cvar"] = turnover_cvar

    return out
