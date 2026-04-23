from __future__ import annotations

import pandas as pd
import numpy as np

# Fundamental column names per asset class
_EQUITY_FUND_COLS = [
    "pe_ratio", "pb_ratio", "roe", "profit_margin",
    "net_debt_to_ebitda", "ebitda_growth", "capex_to_sales",
]
_FIBRA_FUND_COLS = [
    "cap_rate", "ffo_yield", "dividend_yield", "ltv", "vacancy_rate",
]


def _pit_merge_fundamentals(
    feature_df: pd.DataFrame,
    fundamentals: pd.DataFrame,
    fund_cols: list,
) -> pd.DataFrame:
    """Point-in-time asof merge: each (date, ticker) row gets the most recent
    fundamental on or before that date, eliminating look-ahead bias.

    For Yahoo (single today-dated snapshot), all historical dates receive NaN,
    which features.py then fills with neutral defaults — correct behavior.
    For Refinitiv (quarterly history), each date gets its proper lagged value.
    """
    if fundamentals.empty:
        return feature_df
    available_cols = [c for c in fund_cols if c in fundamentals.columns]
    if not available_cols:
        return feature_df

    fund_sorted = (
        fundamentals[["date", "ticker"] + available_cols]
        .dropna(subset=["date", "ticker"])
        .copy()
    )
    fund_sorted["date"] = pd.to_datetime(fund_sorted["date"])
    fund_sorted = fund_sorted.sort_values("date")

    feat_dates = feature_df[["date", "ticker"]].copy()
    feat_dates["date"] = pd.to_datetime(feat_dates["date"])

    parts = []
    for ticker, t_fund in fund_sorted.groupby("ticker", sort=False):
        t_feat = feat_dates.loc[feat_dates["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if t_feat.empty:
            continue
        merged = pd.merge_asof(
            t_feat,
            t_fund.drop(columns=["ticker"]).sort_values("date"),
            on="date",
            direction="backward",
        )
        merged["ticker"] = ticker
        parts.append(merged)

    if not parts:
        return feature_df

    pit = pd.concat(parts, ignore_index=True)
    out = feature_df.drop(columns=[c for c in available_cols if c in feature_df.columns], errors="ignore")
    return out.merge(pit[["date", "ticker"] + available_cols], on=["date", "ticker"], how="left")


def calculate_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    ratio = price_df / price_df.shift(1)
    return np.log(ratio).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rolling_momentum(price_df: pd.DataFrame, window: int = 126, skip: int = 1) -> pd.DataFrame:
    """Log-return over `window` trading days, lagged by `skip` days.

    skip=1  — standard for short-term signals (63-day).
    skip=21 — excludes the most recent month, following the academic convention
               for medium/long-term momentum (126-day+) to avoid short-term reversal.
    On date t the signal reflects P_{t-skip} / P_{t-window-skip}.
    """
    ratio = price_df.shift(skip) / price_df.shift(window + skip)
    return np.log(ratio).replace([np.inf, -np.inf], np.nan)


def volatility_signal(return_df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return return_df.rolling(window).std().shift(1)


def build_signal_matrix(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    fibra_fundamentals: pd.DataFrame,
    bonds: pd.DataFrame,
    macro: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Unified signal matrix builder for all asset classes."""
    investable_universe = universe[universe["investable"]]
    equity_tickers = investable_universe.loc[investable_universe["asset_class"] == "equity", "ticker"].tolist()
    fibra_tickers = investable_universe.loc[investable_universe["asset_class"] == "fibra", "ticker"].tolist()

    equity_prices = prices[equity_tickers + fibra_tickers]  # equities and fibras have prices
    fibra_prices = prices[fibra_tickers]
    equity_fund = fundamentals[fundamentals["ticker"].isin(equity_tickers)]

    equity_features = build_equity_features(equity_prices, equity_fund, macro, investable_universe[investable_universe["asset_class"].isin(["equity", "fibra"])])
    fibra_features = build_fibra_features(fibra_prices, fibra_fundamentals, macro, investable_universe[investable_universe["asset_class"] == "fibra"])
    fixed_features = build_fixed_income_features(bonds, macro)

    # Concatenate all (exclude empty frames to avoid FutureWarning on all-NA dtype inference)
    frames = [f for f in [equity_features, fibra_features, fixed_features] if not f.empty]
    all_features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_features


def build_equity_features(prices: pd.DataFrame, fundamentals: pd.DataFrame, macro: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Build features for equity assets."""
    returns = calculate_returns(prices)
    momentum_63 = rolling_momentum(prices, 63, skip=1)
    momentum_126 = rolling_momentum(prices, 126, skip=21)
    volatility_63 = volatility_signal(returns, 63)

    universe_df = universe[["ticker", "sector", "liquidity_score", "market_cap_mxn", "usd_exposure", "asset_class"]]

    daily_macro = (
        macro.set_index("date")
             .reindex(prices.index, method="ffill")
             .rename_axis("date")
             .reset_index()
    )
    price_stack = prices.stack().rename("price").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    momentum_63_stack = momentum_63.stack().rename("momentum_63").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    momentum_126_stack = momentum_126.stack().rename("momentum_126").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    vol_stack = volatility_63.stack().rename("volatility_63").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})

    feature_df = (
        price_stack
        .merge(momentum_63_stack, on=["date", "ticker"], how="left")
        .merge(momentum_126_stack, on=["date", "ticker"], how="left")
        .merge(vol_stack, on=["date", "ticker"], how="left")
        .merge(universe_df, on="ticker", how="left")
        .merge(daily_macro, on="date", how="left")
    )
    # Point-in-time fundamentals: each date gets the latest quarterly value on or before
    # that date. For Yahoo (single today snapshot), historical dates receive NaN and fall
    # through to the default fill below — no future data leaks into the backtest.
    feature_df = _pit_merge_fundamentals(feature_df, fundamentals, _EQUITY_FUND_COLS)
    # Detect whether real historical fundamentals exist (Refinitiv) or only a
    # today-snapshot is available (Yahoo). In Yahoo mode all historical rows are
    # NaN after the PIT merge — filling them with sector-neutral constants gives
    # every ticker the same value, producing zero cross-sectional discrimination
    # and introducing look-ahead bias. We leave them NaN and rely only on
    # price-based signals; ElasticNet / ranking will ignore zero-variance columns.
    fund_cols_present = [c for c in _EQUITY_FUND_COLS if c in feature_df.columns]
    equity_mask = feature_df["asset_class"] == "equity"
    has_hist_fundamentals = (
        feature_df.loc[equity_mask, fund_cols_present].notna().any().any()
        if fund_cols_present else False
    )
    if has_hist_fundamentals:
        # Fill gaps in real quarterly data with sector-neutral defaults.
        # FIBRAs must remain NaN here so the dropna below excludes them
        # (prevents duplicate rows in the signal matrix).
        for col, default in [("pe_ratio", 14.0), ("pb_ratio", 1.8), ("roe", 0.12),
                             ("profit_margin", 0.08), ("net_debt_to_ebitda", 2.5),
                             ("ebitda_growth", 0.05), ("capex_to_sales", 0.05)]:
            feature_df.loc[equity_mask, col] = feature_df.loc[equity_mask, col].fillna(default)
        feature_df = feature_df.dropna(subset=["momentum_63", "volatility_63", "pe_ratio", "pb_ratio"])
    else:
        feature_df = feature_df.dropna(subset=["momentum_63", "volatility_63"])
        feature_df = feature_df[feature_df["asset_class"] == "equity"]

    # Add legacy columns for compatibility
    feature_df["momentum"] = feature_df["momentum_63"]
    feature_df["volatility"] = feature_df["volatility_63"]
    feature_df = feature_df.assign(
        value_score=lambda x: -0.5 * x["pe_ratio"] - 0.5 * x["pb_ratio"],
        quality_score=lambda x: x["roe"] + x["profit_margin"] - 0.25 * x["net_debt_to_ebitda"],
        macro_exposure=lambda x: x["industrial_production_yoy"] + 0.5 * x["exports_yoy"].fillna(0),
    )

    return feature_df


def build_fibra_features(prices: pd.DataFrame, fibra_fundamentals: pd.DataFrame, macro: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Build features for FIBRA assets."""
    returns = calculate_returns(prices)
    momentum_63 = rolling_momentum(prices, 63)
    volatility_63 = volatility_signal(returns, 63)

    universe_df = universe[["ticker", "sector", "liquidity_score", "market_cap_mxn", "usd_exposure", "asset_class"]]

    daily_macro = (
        macro.set_index("date")
             .reindex(prices.index, method="ffill")
             .rename_axis("date")
             .reset_index()
    )
    price_stack = prices.stack().rename("price").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    momentum_stack = momentum_63.stack().rename("momentum_63").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    vol_stack = volatility_63.stack().rename("volatility_63").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})

    feature_df = (
        price_stack
        .merge(momentum_stack, on=["date", "ticker"], how="left")
        .merge(vol_stack, on=["date", "ticker"], how="left")
        .merge(universe_df, on="ticker", how="left")
        .merge(daily_macro, on="date", how="left")
    )
    feature_df = _pit_merge_fundamentals(feature_df, fibra_fundamentals, _FIBRA_FUND_COLS)
    fibra_cols_present = [c for c in _FIBRA_FUND_COLS if c in feature_df.columns]
    has_hist_fibra_fundamentals = (
        feature_df[fibra_cols_present].notna().any().any() if fibra_cols_present else False
    )
    if has_hist_fibra_fundamentals:
        for col, default in [("cap_rate", 0.075), ("ffo_yield", 0.065), ("dividend_yield", 0.055),
                             ("ltv", 0.35), ("vacancy_rate", 0.08)]:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(default)
        feature_df = feature_df.dropna(subset=["momentum_63", "volatility_63", "cap_rate"])
    else:
        feature_df = feature_df.dropna(subset=["momentum_63", "volatility_63"])

    # Macro sensitivity (simplified)
    feature_df["macro_sensitivity_usd"] = 0.5
    feature_df["macro_sensitivity_rate"] = -0.2

    # Add legacy columns
    feature_df["momentum"] = feature_df["momentum_63"]
    feature_df["volatility"] = feature_df["volatility_63"]
    # For fibra: higher cap_rate = better value; higher FFO yield with low leverage = higher quality
    feature_df["value_score"] = feature_df["cap_rate"]  # higher cap_rate = cheaper / more value
    feature_df["quality_score"] = (
        feature_df["ffo_yield"] - feature_df["vacancy_rate"] - feature_df["ltv"] * 0.15
    )
    # FIBRA macro exposure: sensitive to rate moves and FX (via cap rate re-pricing)
    feature_df["macro_exposure"] = (
        feature_df["macro_sensitivity_usd"].fillna(0.5) * feature_df.get("exports_yoy", pd.Series(0.0, index=feature_df.index)).fillna(0.0)
        - feature_df["macro_sensitivity_rate"].abs().fillna(0.2) * feature_df.get("banxico_rate", pd.Series(0.0, index=feature_df.index)).fillna(0.0)  # noqa: W503
    )

    return feature_df


def build_fixed_income_features(bond_df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Build features for fixed income assets."""
    if bond_df.empty:
        return pd.DataFrame()
    feature_df = bond_df.copy()
    for col in ("duration", "price", "ytm", "credit_spread"):
        if col not in feature_df.columns:
            feature_df[col] = np.nan
    feature_df["dv01"] = feature_df["duration"] * feature_df["price"] * 0.0001  # rough dv01
    macro_daily = macro.set_index("date").reindex(feature_df["date"], method="ffill").reset_index()
    feature_df["carry"] = feature_df["ytm"] - macro_daily["banxico_rate"]
    feature_df["banxico_sensitivity"] = feature_df["duration"] * 0.0025  # 25bp shock impact

    # Add legacy columns for fixed income
    feature_df["momentum"] = 0.0  # no price momentum for bonds
    feature_df["volatility"] = feature_df["dv01"].fillna(0.0)  # proxy: price sensitivity to rate
    feature_df["value_score"] = feature_df["carry"].fillna(0.0)  # higher carry = more value
    # Quality: low credit spread + short duration = safer
    feature_df["quality_score"] = -feature_df["credit_spread"] - 0.02 * feature_df["duration"]
    # Macro exposure: interest rate sensitivity via DV01 and duration
    feature_df["macro_exposure"] = -feature_df["duration"] * feature_df["banxico_sensitivity"].fillna(0.0)
    # Bonds are always fully investable — score_cross_section needs this column to avoid NaN composite_score
    feature_df["liquidity_score"] = 1.0

    return feature_df


def build_etf_features(prices: pd.DataFrame, macro: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Price-only signal matrix for the ETF universe (no fundamentals required).

    Uses momentum and volatility signals only. value_score and quality_score are
    set to zero so score_cross_section can still run without crashing.
    """
    tickers = [t for t in universe.loc[universe["investable"], "ticker"] if t in prices.columns]
    if not tickers:
        return pd.DataFrame()

    prices_etf = prices[tickers]
    returns = calculate_returns(prices_etf)
    momentum_63  = rolling_momentum(prices_etf, 63,  skip=1)
    momentum_126 = rolling_momentum(prices_etf, 126, skip=21)
    volatility_63 = volatility_signal(returns, 63)

    universe_df = universe[["ticker", "sector", "liquidity_score", "market_cap_mxn",
                             "usd_exposure", "asset_class"]].copy()

    daily_macro = (
        macro.set_index("date")
             .reindex(prices_etf.index, method="ffill")
             .rename_axis("date")
             .reset_index()
    )

    price_stack       = prices_etf.stack().rename("price").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    mom63_stack       = momentum_63.stack().rename("momentum_63").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    mom126_stack      = momentum_126.stack().rename("momentum_126").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    vol_stack         = volatility_63.stack().rename("volatility_63").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})

    feature_df = (
        price_stack
        .merge(mom63_stack,  on=["date", "ticker"], how="left")
        .merge(mom126_stack, on=["date", "ticker"], how="left")
        .merge(vol_stack,    on=["date", "ticker"], how="left")
        .merge(universe_df,  on="ticker",           how="left")
        .merge(daily_macro,  on="date",              how="left")
    )

    feature_df = feature_df.dropna(subset=["momentum_63", "volatility_63"])
    feature_df["momentum"]       = feature_df["momentum_63"]
    feature_df["volatility"]     = feature_df["volatility_63"]
    feature_df["value_score"]    = 0.0
    feature_df["quality_score"]  = 0.0
    feature_df["macro_exposure"] = (
        feature_df["industrial_production_yoy"].fillna(0.0)
        + 0.5 * feature_df["exports_yoy"].fillna(0.0)
    )
    return feature_df
