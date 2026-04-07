from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV


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
    """Forecast returns using expanding-window Elastic Net per asset class."""
    forecasts = []
    metadata = {}
    asset_classes = feature_df["asset_class"].unique()
    
    for asset_class in asset_classes:
        class_features = feature_df[feature_df["asset_class"] == asset_class].copy()
        if asset_class == "fixed_income":
            # For bonds, use duration and credit_spread as features
            feature_cols = ["duration", "credit_spread", "carry", "banxico_sensitivity"]
        else:
            # For equities and fibras
            feature_cols = ["momentum_63", "volatility_63", "pe_ratio", "pb_ratio", "roe", "profit_margin", "net_debt_to_ebitda", "industrial_production_yoy", "usd_mxn", "exports_yoy"]
            if asset_class == "fibra":
                feature_cols += ["cap_rate", "ffo_yield", "dividend_yield", "ltv", "vacancy_rate"]
        
        rebalance_dates = class_features["date"].unique()
        for i, date in enumerate(rebalance_dates):
            train_end = date
            train_data = class_features[class_features["date"] <= train_end]
            if len(train_data) < 50:  # minimum training size
                continue
            X_train = train_data[feature_cols].fillna(0.0)
            y_train = train_data.groupby("ticker").apply(lambda x: x["price"].pct_change().shift(-21)).reset_index(drop=True).fillna(0.0)
            
            model = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], max_iter=2000, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict on current date
            current_data = class_features[class_features["date"] == date]
            X_pred = current_data[feature_cols].fillna(0.0)
            preds = model.predict(X_pred)
            current_data = current_data.copy()
            current_data["expected_return"] = preds
            
            forecasts.append(current_data)
            metadata[date] = {"alpha": model.alpha_, "l1_ratio": model.l1_ratio_}
    
    if forecasts:
        result = pd.concat(forecasts, ignore_index=True)
        result["expected_return"] = result.groupby("date")["expected_return"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
        result["_metadata"] = result["date"].map(metadata)
        return result
    else:
        return pd.DataFrame()


def build_trade_signal(predictions: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    signal = predictions.copy()
    signal["rank"] = signal.groupby("date")["expected_return"].rank(ascending=False, method="first")
    signal["signal_weight"] = 0.0
    in_universe = signal[signal["rank"] <= top_n].copy()
    signal.loc[in_universe.index, "signal_weight"] = 1.0
    return signal
