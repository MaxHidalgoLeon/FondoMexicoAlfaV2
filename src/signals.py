from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def score_cross_section(feature_df: pd.DataFrame) -> pd.DataFrame:
    scores = feature_df.copy()
    score_columns = [
        "momentum",
        "value_score",
        "quality_score",
        "liquidity_score",
    ]
    for col in score_columns:
        scores[f"{col}_rank"] = scores.groupby("date")[col].rank(ascending=False, pct=True)
    scores["composite_score"] = scores[[f"{col}_rank" for col in score_columns]].mean(axis=1)
    return scores


def forecast_returns(feature_df: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    forecasts = []
    merged = feature_df.copy()
    merged = merged.merge(
        returns.stack().rename("next_return").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"}),
        on=["date", "ticker"],
        how="left",
    )
    merged["next_return"] = merged.groupby("ticker")["next_return"].shift(-1)
    train_data = merged.dropna(subset=["next_return"])
    feature_cols = ["momentum", "value_score", "quality_score", "liquidity_score"]
    X = train_data[feature_cols].fillna(0.0)
    y = train_data["next_return"].fillna(0.0)
    model = LinearRegression().fit(X, y)
    merged["expected_return"] = model.predict(merged[feature_cols].fillna(0.0))
    merged["expected_return"] = merged.groupby("date")["expected_return"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    return merged.drop(columns=["next_return"])


def build_trade_signal(predictions: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    signal = predictions.copy()
    signal["rank"] = signal.groupby("date")["expected_return"].rank(ascending=False, method="first")
    signal["signal_weight"] = 0.0
    in_universe = signal[signal["rank"] <= top_n].copy()
    signal.loc[in_universe.index, "signal_weight"] = 1.0
    return signal
