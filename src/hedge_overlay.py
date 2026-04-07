"""Hedge overlay module - Layer 2 market-neutral and FX positioning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genextreme
from scipy.special import expit  # sigmoid function

from .risk import compute_cvar, max_drawdown, compute_sharpe, compute_sortino, compute_var, gev_var
from .portfolio import optimize_portfolio


def long_short_portfolio(
    signal_df: pd.DataFrame,
    top_n: int = 5,
    bottom_n: int = 5,
    sector_neutral: bool = True,
) -> pd.DataFrame:
    """Build a market-neutral long/short book within each sector."""
    results = []
    np.random.seed(42)
    
    for date in signal_df["date"].unique():
        date_signals = signal_df[signal_df["date"] == date].copy()
        
        if sector_neutral:
            sectors = date_signals["sector"].unique()
            for sector in sectors:
                sector_signals = date_signals[date_signals["sector"] == sector].copy()
                if len(sector_signals) < top_n + bottom_n:
                    continue
                
                # Sort by expected_return
                sector_signals = sector_signals.sort_values("expected_return", ascending=False)
                
                # Long top_n
                long_tickers = sector_signals.head(top_n)
                long_weights = pd.DataFrame({
                    "date": date,
                    "ticker": long_tickers["ticker"],
                    "side": "long",
                    "raw_weight": 1.0,
                    "sector": sector,
                })
                
                # Short bottom_n
                short_tickers = sector_signals.tail(bottom_n)
                short_weights = pd.DataFrame({
                    "date": date,
                    "ticker": short_tickers["ticker"],
                    "side": "short",
                    "raw_weight": -1.0,
                    "sector": sector,
                })
                
                results.append(pd.concat([long_weights, short_weights], ignore_index=True))
        else:
            # Non-sector-neutral: long/short across all
            date_signals = date_signals.sort_values("expected_return", ascending=False)
            long_tickers = date_signals.head(top_n)
            long_weights = pd.DataFrame({
                "date": date,
                "ticker": long_tickers["ticker"],
                "side": "long",
                "raw_weight": 1.0,
                "sector": long_tickers["sector"],
            })
            
            short_tickers = date_signals.tail(bottom_n)
            short_weights = pd.DataFrame({
                "date": date,
                "ticker": short_tickers["ticker"],
                "side": "short",
                "raw_weight": -1.0,
                "sector": short_tickers["sector"],
            })
            
            results.append(pd.concat([long_weights, short_weights], ignore_index=True))
    
    if not results:
        return pd.DataFrame(columns=["date", "ticker", "side", "raw_weight", "net_weight", "sector"])
    
    portfolio = pd.concat(results, ignore_index=True)
    
    # Normalize to dollar-neutral approximation
    # Gross = 1.0, net in [-0.1, 0.1]
    portfolio_by_date = []
    for date in portfolio["date"].unique():
        date_book = portfolio[portfolio["date"] == date].copy()
        gross_long = date_book[date_book["side"] == "long"]["raw_weight"].sum()
        gross_short = abs(date_book[date_book["side"] == "short"]["raw_weight"].sum())
        
        if gross_long + gross_short == 0:
            continue
        
        # Scale to gross = 1.0 and achieve net in [-0.1, 0.1]
        total_gross = gross_long + gross_short
        long_scale = 0.5 / gross_long if gross_long > 0 else 0
        short_scale = 0.5 / gross_short if gross_short > 0 else 0
        
        date_book["net_weight"] = np.where(
            date_book["side"] == "long",
            date_book["raw_weight"] * long_scale,
            date_book["raw_weight"] * short_scale
        )
        
        portfolio_by_date.append(date_book)
    
    if not portfolio_by_date:
        return pd.DataFrame(columns=["date", "ticker", "side", "raw_weight", "net_weight", "sector"])
    
    return pd.concat(portfolio_by_date, ignore_index=True)


def dynamic_leverage(
    portfolio_returns: pd.Series,
    max_leverage: float = 1.5,
    cvar_limit: float = 0.02,
    alpha: float = 0.95,
    window: int = 63,
) -> pd.Series:
    """Compute a daily leverage scalar in [0.5, max_leverage]."""
    np.random.seed(42)
    leverage = pd.Series(1.0, index=portfolio_returns.index)
    
    for i in range(window, len(portfolio_returns)):
        rolling_returns = portfolio_returns.iloc[i - window:i]
        rolling_cvar = compute_cvar(rolling_returns, alpha=alpha)
        
        if rolling_cvar > cvar_limit:
            # Scale down linearly toward 0.5
            leverage.iloc[i] = 0.5 + (max_leverage - 0.5) * max(0, (cvar_limit - rolling_cvar) / cvar_limit)
        elif rolling_cvar <= cvar_limit * 0.5:
            # Scale up toward max_leverage
            leverage.iloc[i] = 0.5 + (max_leverage - 0.5) * (1 - rolling_cvar / (cvar_limit * 0.5))
        else:
            # Linear interpolation
            leverage.iloc[i] = 0.5 + (max_leverage - 0.5) * (cvar_limit - rolling_cvar) / (cvar_limit * 0.5)
    
    # Smooth with 5-day EMA
    leverage = leverage.ewm(span=5, adjust=False).mean()
    leverage = leverage.clip(lower=0.5, upper=max_leverage)
    
    return leverage


def fx_directional_overlay(
    macro_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    usd_exposure: pd.Series,
    hedge_ratio_base: float = 0.5,
    max_hedge_ratio: float = 0.95,
    min_hedge_ratio: float = 0.10,
) -> pd.DataFrame:
    """Active FX positioning beyond passive hedging."""
    np.random.seed(42)
    
    # Ensure us_fed_rate exists in macro_df
    if "us_fed_rate" not in macro_df.columns:
        macro_df = macro_df.copy()
        macro_df["us_fed_rate"] = 5.25  # default fallback
    
    macro_df = macro_df.copy()
    macro_df = macro_df.set_index("date").reindex(signal_df["date"].unique()).reset_index()
    macro_df = macro_df.ffill()
    
    # Compute rate differential
    macro_df["rate_differential"] = macro_df["banxico_rate"] - macro_df["us_fed_rate"]
    
    # Compute MXN momentum (usd_mxn.pct_change(21)) - negative = MXN strengthening
    macro_df["usd_mxn_pct_change"] = macro_df["usd_mxn"].pct_change(21)
    macro_df["mxn_momentum"] = macro_df["usd_mxn_pct_change"]
    
    # Compute zscores
    rd_zscore = (macro_df["rate_differential"] - macro_df["rate_differential"].mean()) / \
                (macro_df["rate_differential"].std(ddof=0) + 1e-9)
    mom_zscore = (macro_df["mxn_momentum"] - macro_df["mxn_momentum"].mean()) / \
                 (macro_df["mxn_momentum"].std(ddof=0) + 1e-9)
    
    # FX signal score
    macro_df["fx_signal_score"] = -0.6 * rd_zscore + 0.4 * mom_zscore
    
    # Map signal to hedge ratio via sigmoid
    macro_df["hedge_ratio"] = min_hedge_ratio + (max_hedge_ratio - min_hedge_ratio) * \
                              expit(macro_df["fx_signal_score"])
    
    # Compute estimated FX PnL with next-period returns
    macro_df["mxn_return_next"] = macro_df["usd_mxn"].pct_change(-1)  # -1 shifts forward
    mean_usd_exposure = usd_exposure.mean() if len(usd_exposure) > 0 else 0.3
    macro_df["estimated_fx_pnl"] = mean_usd_exposure * \
                                    (1 - macro_df["hedge_ratio"]) * \
                                    macro_df["mxn_return_next"]
    
    return macro_df[["date", "hedge_ratio", "fx_signal_score", "rate_differential", "mxn_momentum", "estimated_fx_pnl"]].reset_index(drop=True)


def tail_risk_hedge(
    portfolio_returns: pd.Series,
    gev_params: tuple[float, float, float],
    protection_level: float = 0.99,
    cost_bps: float = 30.0,
) -> dict:
    """Simulate the cost and benefit of a synthetic tail hedge."""
    shape, loc, scale = gev_params
    
    # Compute unhedged loss at protection_level (left tail VaR)
    unhedged_loss = -genextreme.ppf(1 - protection_level, shape, loc=loc, scale=scale)
    
    # Compute VaR at 95% for put spread
    var_95 = -genextreme.ppf(1 - 0.95, shape, loc=loc, scale=scale)
    
    # Hedge payoff: max(0, loss - VaR_95)
    hedge_payoff = max(0, unhedged_loss - var_95)
    
    # Daily cost drag (annualized cost as daily)
    daily_cost_drag = cost_bps / 10000
    
    # Net benefit per occurrence
    net_benefit = hedge_payoff - daily_cost_drag
    
    return {
        "unhedged_loss_at_99": float(unhedged_loss),
        "hedge_payoff": float(hedge_payoff),
        "daily_cost_drag": float(daily_cost_drag),
        "net_benefit": float(net_benefit),
        "recommended": bool(net_benefit > 0),
    }


def run_hedge_backtest(
    prices: pd.DataFrame,
    signal_df: pd.DataFrame,
    universe: pd.DataFrame,
    macro_df: pd.DataFrame,
    max_leverage: float = 1.5,
    cvar_limit: float = 0.02,
    transaction_cost: float = 0.0015,
) -> dict:
    """Full Layer 2 backtest combining all hedge components."""
    np.random.seed(42)
    
    # Step 1: Build long_short_portfolio weights per rebalance date
    long_short = long_short_portfolio(signal_df, top_n=5, bottom_n=5, sector_neutral=True)
    
    if long_short.empty:
        return {
            "returns": pd.Series(0.0, index=prices.index),
            "leverage_series": pd.Series(1.0, index=prices.index),
            "fx_overlay": fx_directional_overlay(macro_df, signal_df, universe.set_index("ticker")["usd_exposure"]),
            "tail_hedge": {"unhedged_loss_at_99": 0.0, "hedge_payoff": 0.0, "daily_cost_drag": 0.0, "net_benefit": 0.0, "recommended": False},
            "metrics": {},
            "long_book": long_short[long_short["side"] == "long"],
            "short_book": long_short[long_short["side"] == "short"],
        }
    
    # Build daily portfolio returns from long/short weights
    returns = prices.pct_change().fillna(0.0)
    rebalance_dates = long_short["date"].unique()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    portfolio_returns = pd.Series(0.0, index=prices.index)
    prev_weights = pd.Series(0.0, index=prices.columns)
    turnover = pd.Series(0.0, index=prices.index)
    
    # Step 2: Compute base portfolio returns from long/short weights
    for date in rebalance_dates:
        if date not in prices.index:
            continue
        date_book = long_short[long_short["date"] == date]
        if date_book.empty:
            continue
        
        target_weights = pd.Series(0.0, index=prices.columns)
        for _, row in date_book.iterrows():
            if row["ticker"] in target_weights.index:
                target_weights[row["ticker"]] = row["net_weight"]
        
        if date in prices.index:
            weights.loc[date:, :] = target_weights.values
            turnover.loc[date] = np.sum(np.abs(target_weights - prev_weights))
            prev_weights = target_weights
    
    base_returns = (weights.shift(1) * returns).sum(axis=1)
    
    # Step 3: Apply dynamic_leverage() to scale daily returns
    leverage_series = dynamic_leverage(base_returns, max_leverage=max_leverage, cvar_limit=cvar_limit)
    leveraged_returns = base_returns * leverage_series
    
    # Step 4: Apply fx_directional_overlay() estimated_fx_pnl as additive daily return
    fx_overlay = fx_directional_overlay(
        macro_df,
        signal_df,
        universe.set_index("ticker")["usd_exposure"]
    )
    fx_overlay = fx_overlay.set_index("date").reindex(prices.index, method="ffill").reset_index()
    fx_pnl = pd.Series(fx_overlay["estimated_fx_pnl"].values, index=prices.index).fillna(0.0)
    
    # Step 5: Subtract transaction_cost * gross_turnover per rebalance
    gross_turnover = turnover * 2  # long and short
    transaction_costs = gross_turnover * transaction_cost
    
    final_returns = leveraged_returns + fx_pnl - transaction_costs
    
    # Step 6: Compute full risk metrics
    metrics = {
        "sharpe": compute_sharpe(final_returns),
        "sortino": compute_sortino(final_returns),
        "max_drawdown": max_drawdown(final_returns),
        "cvar_95": compute_cvar(final_returns, alpha=0.95),
        "annualized_return": ((1 + final_returns).prod() ** (252 / len(final_returns))) - 1,
        "annualized_vol": final_returns.std() * np.sqrt(252),
        "turnover": turnover.mean(),
    }
    
    # Step 7: Run tail_risk_hedge() on the resulting return series
    try:
        gev_shape, gev_loc, gev_scale = genextreme.fit(-final_returns[final_returns < 0])
        tail_hedge = tail_risk_hedge(
            final_returns,
            (gev_shape, gev_loc, gev_scale),
            protection_level=0.99,
            cost_bps=30.0
        )
    except Exception:
        tail_hedge = {
            "unhedged_loss_at_99": 0.0,
            "hedge_payoff": 0.0,
            "daily_cost_drag": 0.0,
            "net_benefit": 0.0,
            "recommended": False,
        }
    
    # Step 8: Return complete results
    return {
        "returns": final_returns,
        "leverage_series": leverage_series,
        "fx_overlay": fx_overlay,
        "tail_hedge": tail_hedge,
        "metrics": metrics,
        "long_book": long_short[long_short["side"] == "long"],
        "short_book": long_short[long_short["side"] == "short"],
    }
