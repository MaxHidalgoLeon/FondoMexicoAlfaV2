from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV

logger = logging.getLogger(__name__)

# Feature sets per asset class (externalized so they're easy to update)
_EQUITY_FEATURES = [
    "momentum_63", "momentum_126", "volatility_63",
    "pe_ratio", "pb_ratio", "roe", "profit_margin", "net_debt_to_ebitda",
    "ebitda_growth", "capex_to_sales",
    "industrial_production_yoy", "usd_mxn", "exports_yoy",
]
_FIBRA_FEATURES = [
    "momentum_63", "volatility_63",
    "cap_rate", "ffo_yield", "dividend_yield", "ltv", "vacancy_rate",
    "industrial_production_yoy", "usd_mxn",
]
_FIXED_INCOME_FEATURES = [
    "duration", "credit_spread", "carry", "banxico_sensitivity",
]

_FORWARD_DAYS = 21  # prediction horizon in trading days


def _compute_forward_returns(group_df: pd.DataFrame) -> pd.Series:
    """
    Compute FORWARD_DAYS-ahead return per ticker using only past prices.
    The last FORWARD_DAYS rows per ticker receive NaN (return not yet realized).
    This avoids any look-ahead bias.
    """
    n = len(group_df)
    fwd = np.full(n, np.nan)
    if n > _FORWARD_DAYS:
        prices = group_df["price"].values
        fwd[: n - _FORWARD_DAYS] = (
            prices[_FORWARD_DAYS:] / (prices[: n - _FORWARD_DAYS] + 1e-9) - 1.0
        )
    return pd.Series(fwd, index=group_df.index)


def _end_of_month_dates(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return the last available date in each calendar month."""
    s = pd.Series(dates, index=dates)
    return s.resample("ME").last().dropna().values


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
    """
    Forecast returns using an expanding-window Elastic Net per asset class.

    Training target: forward FORWARD_DAYS-day return computed WITHOUT look-ahead.
    Models are retrained monthly (end-of-month dates only) for efficiency.
    """
    forecasts: list[pd.DataFrame] = []
    asset_classes = feature_df["asset_class"].unique()

    for asset_class in asset_classes:
        class_df = feature_df[feature_df["asset_class"] == asset_class].copy()
        class_df = class_df.sort_values(["ticker", "date"])

        # Select feature columns that actually exist in this slice
        if asset_class == "fixed_income":
            feature_cols = [c for c in _FIXED_INCOME_FEATURES if c in class_df.columns]
        elif asset_class == "fibra":
            feature_cols = [c for c in _FIBRA_FEATURES if c in class_df.columns]
        else:
            feature_cols = [c for c in _EQUITY_FEATURES if c in class_df.columns]

        if not feature_cols:
            logger.warning("No feature columns found for asset_class=%s — skipping.", asset_class)
            continue

        # Precompute forward returns with NO look-ahead (NaN for last FORWARD_DAYS rows per ticker)
        _fwd_parts = []
        for ticker, grp in class_df.groupby("ticker", group_keys=False):
            _fwd_parts.append(_compute_forward_returns(grp))
        class_df["_fwd_return"] = pd.concat(_fwd_parts).reindex(class_df.index)

        # Only retrain at end-of-month dates (efficiency + monthly rebalancing cadence)
        all_dates = pd.DatetimeIndex(sorted(class_df["date"].unique()))
        rebal_dates = _end_of_month_dates(all_dates)

        for date in rebal_dates:
            # Training set: all data up to current date, EXCLUDING rows whose forward
            # return is not yet realized (NaN).  The NaN-drop is the only look-ahead guard needed.
            train_mask = (class_df["date"] <= date) & class_df["_fwd_return"].notna()
            train_data = class_df.loc[train_mask]

            if len(train_data) < 50:
                logger.debug(
                    "Skipping %s on %s: only %d training rows.", asset_class, date, len(train_data)
                )
                continue

            X_train = train_data[feature_cols].fillna(0.0)
            y_train = train_data["_fwd_return"]

            model = ElasticNetCV(
                cv=5, l1_ratio=[0.1, 0.5, 0.9], max_iter=2000,
                random_state=42, n_jobs=-1,
            )
            try:
                model.fit(X_train, y_train)
            except Exception as exc:
                logger.warning("ElasticNetCV fit failed for %s on %s: %s", asset_class, date, exc)
                continue

            current_data = class_df[class_df["date"] == date].copy()
            if current_data.empty:
                continue
            X_pred = current_data[feature_cols].fillna(0.0)
            current_data["expected_return"] = model.predict(X_pred)
            forecasts.append(current_data)

    if not forecasts:
        return pd.DataFrame()

    result = pd.concat(forecasts, ignore_index=True)
    # Cross-sectional z-score per date so signals are comparable across asset classes
    result["expected_return"] = result.groupby("date")["expected_return"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    )
    # Drop internal column
    result = result.drop(columns=["_fwd_return"], errors="ignore")
    return result
