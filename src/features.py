from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from typing import Dict


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
    fixed_tickers = investable_universe.loc[investable_universe["asset_class"] == "fixed_income", "ticker"].tolist()

    equity_prices = prices[equity_tickers + fibra_tickers]  # equities and fibras have prices
    fibra_prices = prices[fibra_tickers]
    equity_fund = fundamentals[fundamentals["ticker"].isin(equity_tickers)]

    equity_features = build_equity_features(equity_prices, equity_fund, macro, investable_universe[investable_universe["asset_class"].isin(["equity", "fibra"])])
    fibra_features = build_fibra_features(fibra_prices, fibra_fundamentals, macro, investable_universe[investable_universe["asset_class"] == "fibra"])
    fixed_features = build_fixed_income_features(bonds, macro)

    # Concatenate all
    all_features = pd.concat([equity_features, fibra_features, fixed_features], ignore_index=True)
    return all_features


def normalize_features(feature_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = feature_df.copy()
    for col in columns:
        out[f"{col}_z"] = (out[col] - out[col].mean()) / (out[col].std(ddof=0) + 1e-9)
    return out


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
    universe_df = universe[["ticker", "liquidity_score", "market_cap_mxn", "usd_exposure", "asset_class"]]

    daily_macro = macro.set_index("date").reindex(prices.index, method="ffill").reset_index().rename(columns={"index": "date"})
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

    # Elastic Net signal
    feature_cols = ["momentum_63", "momentum_126", "volatility_63", "pe_ratio", "pb_ratio", "roe", "profit_margin", "net_debt_to_ebitda", "industrial_production_yoy", "usd_mxn", "exports_yoy"]
    X = feature_df[feature_cols].fillna(0.0)
    y = feature_df.groupby("date").apply(lambda x: x["price"].pct_change().shift(-21)).reset_index(drop=True).fillna(0.0)  # 21-day forward return
    elastic_net = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], max_iter=2000, random_state=42)
    elastic_net.fit(X, y)
    feature_df["elastic_net_signal"] = elastic_net.predict(X)

    # Fama-French proxies (simplified)
    feature_df["beta_mkt"] = 1.0
    feature_df["beta_smb"] = 0.0
    feature_df["beta_hml"] = 0.0

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
    universe_df = universe[["ticker", "liquidity_score", "market_cap_mxn", "usd_exposure", "asset_class"]]

    daily_macro = macro.set_index("date").reindex(prices.index, method="ffill").reset_index().rename(columns={"index": "date"})
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
    # For fibra, value_score could be based on cap_rate, etc.
    feature_df["value_score"] = -feature_df["cap_rate"]  # lower cap_rate better
    feature_df["quality_score"] = feature_df["ffo_yield"] - feature_df["ltv"] * 0.1
    feature_df["macro_exposure"] = 0.0  # placeholder

    return feature_df


def build_fixed_income_features(bond_df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Build features for fixed income assets."""
    feature_df = bond_df.copy()
    feature_df["dv01"] = feature_df["duration"] * feature_df["price"] * 0.0001  # rough dv01
    macro_daily = macro.set_index("date").reindex(feature_df["date"], method="ffill").reset_index()
    feature_df["carry"] = feature_df["ytm"] - macro_daily["banxico_rate"]
    feature_df["banxico_sensitivity"] = feature_df["duration"] * 0.0025  # 25bp shock impact
    
    # Add legacy columns for fixed income
    feature_df["momentum"] = 0.0  # no momentum for bonds
    feature_df["volatility"] = 0.0
    feature_df["value_score"] = -feature_df["credit_spread"]  # lower spread better
    feature_df["quality_score"] = feature_df["duration"]  # longer duration higher quality?
    feature_df["macro_exposure"] = 0.0
    
    return feature_df
