from __future__ import annotations

import pandas as pd


def calculate_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().fillna(0.0)


def rolling_momentum(price_df: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    return price_df.pct_change(periods=window).shift(1)


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

    # Concatenate all
    all_features = pd.concat([equity_features, fibra_features, fixed_features], ignore_index=True)
    return all_features


def build_equity_features(prices: pd.DataFrame, fundamentals: pd.DataFrame, macro: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Build features for equity assets."""
    returns = calculate_returns(prices)
    momentum_63 = rolling_momentum(prices, 63)
    momentum_126 = rolling_momentum(prices, 126)
    volatility_63 = volatility_signal(returns, 63)

    latest_fundamentals = (
        fundamentals.sort_values(["ticker", "date"]).groupby("ticker").tail(1).set_index("ticker")
    )
    fundamentals_df = latest_fundamentals.reset_index().drop(columns=["date"])
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
        .merge(fundamentals_df, on="ticker", how="left")
        .merge(daily_macro, on="date", how="left")
    )
    feature_df = feature_df.dropna(subset=["momentum_63", "volatility_63", "pe_ratio", "pb_ratio"])

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

    latest_fibra = (
        fibra_fundamentals.sort_values(["ticker", "date"]).groupby("ticker").tail(1).set_index("ticker")
    )
    fibra_df = latest_fibra.reset_index().drop(columns=["date"])
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
        .merge(fibra_df, on="ticker", how="left")
        .merge(daily_macro, on="date", how="left")
    )
    feature_df = feature_df.dropna(subset=["momentum_63", "volatility_63", "cap_rate"])

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
    feature_df = bond_df.copy()
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
