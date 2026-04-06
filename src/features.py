from __future__ import annotations

import numpy as np
import pandas as pd
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
    macro: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    returns = calculate_returns(prices)
    momentum = rolling_momentum(prices)
    vol = volatility_signal(returns)

    latest_fundamentals = (
        fundamentals.sort_values(["ticker", "date"]).groupby("ticker").tail(1).set_index("ticker")
    )
    fundamentals_df = latest_fundamentals.reset_index().drop(columns=["date"])
    universe_df = universe[["ticker", "liquidity_score", "market_cap_mxn", "usd_exposure"]]

    daily_macro = macro.set_index("date").reindex(prices.index, method="ffill").reset_index().rename(columns={"index": "date"})
    price_stack = prices.stack().rename("price").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    momentum_stack = momentum.stack().rename("momentum").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    vol_stack = vol.stack().rename("volatility").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})

    feature_df = (
        price_stack
        .merge(momentum_stack, on=["date", "ticker"], how="left")
        .merge(vol_stack, on=["date", "ticker"], how="left")
        .merge(universe_df, on="ticker", how="left")
        .merge(fundamentals_df, on="ticker", how="left")
        .merge(daily_macro, on="date", how="left")
    )
    feature_df = feature_df.dropna(subset=["momentum", "volatility", "pe_ratio", "pb_ratio"])
    feature_df = feature_df.assign(
        value_score=lambda x: -0.5 * x["pe_ratio"] - 0.5 * x["pb_ratio"],
        quality_score=lambda x: x["roe"] + x["profit_margin"] - 0.25 * x["net_debt_to_ebitda"],
        macro_exposure=lambda x: x["industrial_production_yoy"] + 0.5 * x["exports_yoy"].fillna(0),
    )
    return feature_df


def normalize_features(feature_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = feature_df.copy()
    for col in columns:
        out[f"{col}_z"] = (out[col] - out[col].mean()) / (out[col].std(ddof=0) + 1e-9)
    return out
