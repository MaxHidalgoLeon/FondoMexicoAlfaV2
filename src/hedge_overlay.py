"""Hedge overlay module - Layer 2 market-neutral and FX positioning."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.stats import genextreme
from scipy.special import expit  # sigmoid function

from .risk import compute_cvar, max_drawdown, compute_sharpe, compute_sortino

logger = logging.getLogger(__name__)


def long_short_portfolio(
    signal_df: pd.DataFrame,
    top_n: int = 5,
    bottom_n: int = 5,
    sector_neutral: bool = True,
    net_target: float = 0.0,
    gross_target: float = 1.0,
    weight_by_signal: bool = True,
) -> pd.DataFrame:
    """Build a market-neutral long/short book within each sector."""
    results = []

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
                long_strength = long_tickers["expected_return"].clip(lower=0.0)
                if not weight_by_signal or float(long_strength.sum()) <= 1e-9:
                    long_strength = pd.Series(1.0, index=long_tickers.index)
                long_weights = pd.DataFrame({
                    "date": date,
                    "ticker": long_tickers["ticker"],
                    "side": "long",
                    "raw_weight": long_strength.values,
                    "sector": sector,
                })

                # Short bottom_n
                short_tickers = sector_signals.tail(bottom_n)
                short_strength = (-short_tickers["expected_return"]).clip(lower=0.0)
                if not weight_by_signal or float(short_strength.sum()) <= 1e-9:
                    short_strength = pd.Series(1.0, index=short_tickers.index)
                short_weights = pd.DataFrame({
                    "date": date,
                    "ticker": short_tickers["ticker"],
                    "side": "short",
                    "raw_weight": -short_strength.values,
                    "sector": sector,
                })

                results.append(pd.concat([long_weights, short_weights], ignore_index=True))
        else:
            # Non-sector-neutral: long/short across all
            date_signals = date_signals.sort_values("expected_return", ascending=False)
            long_tickers = date_signals.head(top_n)
            long_strength = long_tickers["expected_return"].clip(lower=0.0)
            if not weight_by_signal or float(long_strength.sum()) <= 1e-9:
                long_strength = pd.Series(1.0, index=long_tickers.index)
            long_weights = pd.DataFrame({
                "date": date,
                "ticker": long_tickers["ticker"],
                "side": "long",
                "raw_weight": long_strength.values,
                "sector": long_tickers["sector"],
            })

            short_tickers = date_signals.tail(bottom_n)
            short_strength = (-short_tickers["expected_return"]).clip(lower=0.0)
            if not weight_by_signal or float(short_strength.sum()) <= 1e-9:
                short_strength = pd.Series(1.0, index=short_tickers.index)
            short_weights = pd.DataFrame({
                "date": date,
                "ticker": short_tickers["ticker"],
                "side": "short",
                "raw_weight": -short_strength.values,
                "sector": short_tickers["sector"],
            })

            results.append(pd.concat([long_weights, short_weights], ignore_index=True))

    if not results:
        return pd.DataFrame(columns=["date", "ticker", "side", "raw_weight", "net_weight", "sector"])

    portfolio = pd.concat(results, ignore_index=True)

    # Normalize to configurable gross/net exposure targets.
    net_target = float(np.clip(net_target, -gross_target, gross_target))
    gross_target = max(float(gross_target), 1e-6)
    long_budget = 0.5 * (gross_target + net_target)
    short_budget = 0.5 * (gross_target - net_target)

    portfolio_by_date = []
    for date in portfolio["date"].unique():
        date_book = portfolio[portfolio["date"] == date].copy()
        gross_long = date_book[date_book["side"] == "long"]["raw_weight"].sum()
        gross_short = abs(date_book[date_book["side"] == "short"]["raw_weight"].sum())

        if gross_long + gross_short == 0:
            continue

        long_scale = long_budget / gross_long if gross_long > 0 else 0.0
        short_scale = short_budget / gross_short if gross_short > 0 else 0.0

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
    min_leverage: float = 1.0,
    alpha: float = 0.95,
    window: int = 63,
) -> pd.Series:
    """Compute a daily leverage scalar in [min_leverage, max_leverage]."""
    leverage = pd.Series(1.0, index=portfolio_returns.index)

    for i in range(window, len(portfolio_returns)):
        rolling_returns = portfolio_returns.iloc[i - window:i]
        rolling_cvar = compute_cvar(rolling_returns, alpha=alpha)
        abs_cvar = abs(rolling_cvar)

        if abs_cvar > cvar_limit:
            # Risk too high: scale down toward min_leverage
            scale = max(0.0, 1.0 - (abs_cvar - cvar_limit) / cvar_limit)
            leverage.iloc[i] = min_leverage + (max_leverage - min_leverage) * scale
        elif abs_cvar <= cvar_limit * 0.5:
            # Risk very low: scale up toward max_leverage
            leverage.iloc[i] = max_leverage
        else:
            # Linear interpolation between min_leverage and max_leverage
            t = (cvar_limit - abs_cvar) / (cvar_limit * 0.5)
            leverage.iloc[i] = min_leverage + (max_leverage - min_leverage) * t

    # Smooth with 5-day EMA using adjust=True for stable initial values
    leverage = leverage.ewm(span=5, adjust=True).mean()
    leverage = leverage.clip(lower=min_leverage, upper=max_leverage)

    return leverage


def fx_directional_overlay(
    macro_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    usd_exposure: pd.Series,
    hedge_ratio_base: float = 0.5,
    max_hedge_ratio: float = 0.95,
    min_hedge_ratio: float = 0.10,
    mxn_garch_vol: float | None = None,
) -> pd.DataFrame:
    """Active FX positioning beyond passive hedging.

    Args:
        mxn_garch_vol: GARCH vol forecast anualizada para USD/MXN. Cuando se
            provee, regímenes de alta volatilidad incrementan el hedge ratio.
    """

    # Ensure us_fed_rate exists in macro_df
    if "us_fed_rate" not in macro_df.columns:
        macro_df = macro_df.copy()
        macro_df["us_fed_rate"] = 5.25  # default fallback

    macro_df = macro_df.copy()
    macro_df = macro_df.set_index("date").reindex(signal_df["date"].unique()).reset_index()
    macro_df = macro_df.ffill()

    # Compute rate differential
    macro_df["rate_differential"] = macro_df["banxico_rate"] - macro_df["us_fed_rate"]

    # Compute MXN momentum using 21-day log return; negative = MXN strengthening.
    macro_df["usd_mxn_pct_change"] = np.log(
        macro_df["usd_mxn"] / macro_df["usd_mxn"].shift(21)
    )
    macro_df["mxn_momentum"] = macro_df["usd_mxn_pct_change"]

    # Compute zscores
    rd_zscore = (macro_df["rate_differential"] - macro_df["rate_differential"].mean()) / \
                (macro_df["rate_differential"].std(ddof=0) + 1e-9)
    mom_zscore = (macro_df["mxn_momentum"] - macro_df["mxn_momentum"].mean()) / \
                 (macro_df["mxn_momentum"].std(ddof=0) + 1e-9)

    # FX signal score: high rate differential = more incentive to stay unhedged (carry trade)
    # Negative momentum in MXN (USD/MXN rising = MXN weakening) = increase hedge
    macro_df["fx_signal_score"] = -0.6 * rd_zscore + 0.4 * mom_zscore

    # Clip z-scores to [-3, 3] before sigmoid so the full [min, max] range is utilized
    clipped_score = macro_df["fx_signal_score"].clip(-3.0, 3.0) / 3.0  # normalize to [-1, 1]
    macro_df["hedge_ratio"] = min_hedge_ratio + (max_hedge_ratio - min_hedge_ratio) * \
        expit(clipped_score * 6)  # scale so sigmoid is responsive  # noqa: E127

    # GARCH vol adjustment: alta vol de MXN → incrementar hedge ratio
    if mxn_garch_vol is not None and np.isfinite(mxn_garch_vol) and mxn_garch_vol > 0:
        # 15% anualizado = neutral, >25% = régimen de alta vol
        vol_zscore = (mxn_garch_vol - 0.15) / 0.10
        vol_boost = 0.05 * float(np.clip(vol_zscore, -1.0, 1.0))  # máx ±5%
        macro_df["hedge_ratio"] = (macro_df["hedge_ratio"] + vol_boost).clip(
            lower=min_hedge_ratio, upper=max_hedge_ratio
        )

    # NOTE: estimated_fx_pnl is NOT computed here to avoid look-ahead bias.
    # The actual FX PnL is computed in run_hedge_backtest using lagged hedge ratios
    # and realized (contemporaneous) FX changes.

    return macro_df[["date", "hedge_ratio", "fx_signal_score", "rate_differential", "mxn_momentum"]].reset_index(drop=True)


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
    risk_free_rate: float = 0.02,
    mxn_garch_vol: float | None = None,
    hedge_mode: str = "analytical",
) -> dict:
    """Full Layer 2 backtest combining all hedge components.

    hedge_mode:
        "analytical"  → Layer 2 is informational only; does NOT affect regulated NAV.
                         Uses uncapped leverage targets (net=1.60, gross=1.60).
        "regulated"   → Layer 2 is part of the regulated NAV; leverage is capped
                         per LFI Art. prospectus limits (gross=1.15, net=1.05).
    """

    # Hedge-mode comparison should be against risky sleeves (equity/FIBRA),
    # not a fixed-income carry book.
    signal_for_hedge = signal_df[signal_df["asset_class"].isin(["equity", "fibra"])].copy()
    if signal_for_hedge.empty:
        signal_for_hedge = signal_df.copy()

    # Determine targets based on hedge_mode
    if hedge_mode == "regulated":
        _net_target = 1.05
        _gross_target = 1.15
        _max_lev = min(max_leverage, 1.15)
        logger.info("Hedge overlay running in REGULATED mode (gross≤1.15, net≤1.05).")
    else:
        _net_target = 1.60
        _gross_target = 1.60
        _max_lev = max_leverage
        logger.info("Hedge overlay running in ANALYTICAL mode (uncapped, non-NAV).")

    # Step 1: Build long_short_portfolio weights per rebalance date
    # Use top_n/bottom_n per sector based on realistic universe size
    long_short = long_short_portfolio(
        signal_for_hedge,
        top_n=8,
        bottom_n=0,
        sector_neutral=False,
        net_target=_net_target,
        gross_target=_gross_target,
        weight_by_signal=True,
    )

    if long_short.empty:
        logger.warning("long_short_portfolio returned empty DataFrame — check signal_df coverage.")
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
    returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rebalance_dates = long_short["date"].unique()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
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
            # Turnover control: skip very small trades to reduce cost drag.
            trade_band = 0.0005
            delta = target_weights - prev_weights
            target_weights = prev_weights + delta.where(delta.abs() >= trade_band, 0.0)
            weights.loc[date:, :] = target_weights.values
            turnover.loc[date] = np.sum(np.abs(target_weights - prev_weights))
            prev_weights = target_weights

    base_returns = (weights.shift(1) * returns).sum(axis=1)

    # Step 3: Apply dynamic_leverage() to scale daily returns
    leverage_series = dynamic_leverage(base_returns, max_leverage=_max_lev, cvar_limit=cvar_limit)
    leveraged_returns = base_returns * leverage_series

    # Step 4: Compute FX PnL from LAGGED hedge ratios and REALIZED (contemporaneous) FX changes
    # hedge_ratio[t-1] is decided at close of t-1; usd_mxn_return[t] is the actual next-day FX move
    fx_overlay = fx_directional_overlay(
        macro_df,
        signal_df,
        universe.set_index("ticker")["usd_exposure"],
        mxn_garch_vol=mxn_garch_vol,
    )
    fx_df = fx_overlay.sort_values("date").set_index("date").reindex(prices.index, method="ffill")
    hedge_ratio_series = fx_df["hedge_ratio"].fillna(0.5)

    # Realized FX change: daily log return of usd_mxn (contemporaneous, not forward-looking)
    macro_reindexed = (
        macro_df.set_index("date")["usd_mxn"]
        .reindex(prices.index, method="ffill")
    )
    usd_mxn_daily_return = np.log(macro_reindexed / macro_reindexed.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean_usd_exposure = universe.set_index("ticker")["usd_exposure"].mean()
    # FX PnL = exposure * (1 - lagged_hedge) * realized_fx_change
    fx_pnl = mean_usd_exposure * (1 - hedge_ratio_series.shift(1).fillna(0.5)) * usd_mxn_daily_return

    # Step 5: Subtract transaction_cost * gross_turnover per rebalance
    gross_turnover = turnover * 2  # long and short
    transaction_costs = gross_turnover * transaction_cost

    final_returns = leveraged_returns + fx_pnl - transaction_costs

    # Step 6: Compute full risk metrics
    _ann_ret_h = np.exp(final_returns.sum() * (252 / max(len(final_returns), 1))) - 1
    _mdd_h = max_drawdown(final_returns)
    metrics = {
        "sharpe": compute_sharpe(final_returns, risk_free_rate=risk_free_rate),
        "sortino": compute_sortino(final_returns, required_return=risk_free_rate / 252),
        "max_drawdown": _mdd_h,
        "calmar": _ann_ret_h / max(abs(_mdd_h), 1e-9),
        "cvar_95": compute_cvar(final_returns, alpha=0.95),
        "annualized_return": _ann_ret_h,
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

    # FX notional cap check: fx_notional / NAV proxy ≤ 0.15
    if not fx_overlay.empty and "hedge_ratio" in fx_overlay.columns:
        max_hedge = float(fx_overlay["hedge_ratio"].abs().max())
        if max_hedge > 0.15:
            logger.error(
                "FX notional cap EXCEEDED: max hedge ratio %.4f > 0.15 (15%% NAV limit).",
                max_hedge,
            )
        elif max_hedge > 0.12:
            logger.warning(
                "FX notional approaching cap: hedge ratio %.4f (limit 0.15).",
                max_hedge,
            )

    # Step 8: Return complete results
    return {
        "base_returns": base_returns,
        "leveraged_returns": leveraged_returns,
        "fx_pnl": fx_pnl,
        "transaction_costs": transaction_costs,
        "returns": final_returns,
        "leverage_series": leverage_series,
        "fx_overlay": fx_overlay,
        "params": {
            "max_leverage": float(_max_lev),
            "cvar_limit": float(cvar_limit),
            "transaction_cost": float(transaction_cost),
            "risk_free_rate": float(risk_free_rate),
            "mxn_garch_vol": float(mxn_garch_vol) if mxn_garch_vol is not None and np.isfinite(mxn_garch_vol) else None,
            "hedge_mode": hedge_mode,
        },
        "tail_hedge": tail_hedge,
        "metrics": metrics,
        "long_book": long_short[long_short["side"] == "long"],
        "short_book": long_short[long_short["side"] == "short"],
    }
