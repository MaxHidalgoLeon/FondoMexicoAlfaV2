from __future__ import annotations

import pandas as pd

from .backtest import run_backtest
from .data_loader import load_mock_data
from .features import build_signal_matrix
from .signals import score_cross_section, forecast_returns
from .risk import stress_test


def run_pipeline() -> dict[str, object]:
    data = load_mock_data()
    universe = data["universe"]
    prices = data["prices"]
    fundamentals = data["fundamentals"]
    macro = data["macro"]

    feature_df = build_signal_matrix(prices, fundamentals, macro, universe)
    scored = score_cross_section(feature_df)
    forecast_df = forecast_returns(scored, prices.pct_change().fillna(0.0))
    backtest_results = run_backtest(prices, forecast_df, universe)

    exposures = {
        "banxico_shock": 0.5,
        "peso_depreciation": 0.6,
        "us_slowdown": 0.4,
    }
    scenario_shocks = {
        "banxico_shock": -0.03,
        "peso_depreciation": -0.05,
        "us_slowdown": -0.04,
    }
    stress = stress_test(backtest_results["returns"], scenario_shocks, exposures)

    summary = {
        "universe_size": len(universe),
        "start_date": prices.index.min(),
        "end_date": prices.index.max(),
        "metrics": backtest_results["metrics"],
        "stress": stress,
    }

    return {
        "data": data,
        "feature_df": feature_df,
        "forecast_df": forecast_df,
        "backtest": backtest_results,
        "summary": summary,
    }


def print_summary(results: dict[str, object]) -> None:
    summary = results["summary"]
    print("=== Strategy Pipeline Summary ===")
    print(f"Universe size: {summary['universe_size']}")
    print(f"Backtest period: {summary['start_date'].date()} to {summary['end_date'].date()}")
    for key, value in summary["metrics"].items():
        print(f"{key}: {value:.4f}")
    print("\nStress test results:")
    print(summary["stress"].to_string(index=False))


if __name__ == "__main__":
    results = run_pipeline()
    print_summary(results)
