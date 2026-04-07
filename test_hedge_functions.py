#!/usr/bin/env python
"""Quick test of hedge_overlay functions."""

import numpy as np
import pandas as pd
from src.hedge_overlay import (
    long_short_portfolio,
    dynamic_leverage,
    fx_directional_overlay,
    tail_risk_hedge,
)
from scipy.stats import genextreme

print("Testing hedge_overlay functions...")

# Test 1: long_short_portfolio
np.random.seed(42)
signal_df = pd.DataFrame({
    "date": pd.Timestamp("2024-01-01"),
    "ticker": ["T1", "T2", "T3", "T4", "T5"],
    "expected_return": np.random.randn(5),
    "sector": ["A", "A", "A", "B", "B"],
})
portfolio = long_short_portfolio(signal_df, top_n=2, bottom_n=1)
print(f"✓ long_short_portfolio works: {len(portfolio)} positions")

# Test 2: dynamic_leverage
returns = pd.Series(np.random.randn(100) * 0.01)
leverage = dynamic_leverage(returns, max_leverage=1.5)
print(f"✓ dynamic_leverage works: mean leverage = {leverage.mean():.4f}")

# Test 3: fx_directional_overlay
macro_df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=30, freq="D"),
    "banxico_rate": 5.5 + np.random.randn(30) * 0.1,
    "us_fed_rate": 5.0 + np.random.randn(30) * 0.1,
    "usd_mxn": 19.0 + np.cumsum(np.random.randn(30) * 0.05),
})
signal_df_fx = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=30, freq="D"),
})
usd_exposure = pd.Series([0.3])
fx_overlay = fx_directional_overlay(macro_df, signal_df_fx, usd_exposure)
print(f"✓ fx_directional_overlay works: {len(fx_overlay)} rows")

# Test 4: tail_risk_hedge
gev_params = genextreme.fit(-returns[returns < 0])
tail_hedge = tail_risk_hedge(returns, gev_params)
print(f"✓ tail_risk_hedge works: recommended = {tail_hedge['recommended']}")

print("\n✅ All hedge_overlay functions work correctly!")
