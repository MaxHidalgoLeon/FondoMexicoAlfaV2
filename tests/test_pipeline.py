import unittest
import numpy as np
import pandas as pd

from src.pipeline import run_pipeline


class PipelineTestCase(unittest.TestCase):
    """Integration tests for the full Layer 1 + Layer 2 pipeline.

    run_pipeline() is called once per class (setUpClass) to avoid redundant
    computation across individual test methods.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = run_pipeline(hedge_mode=False)
        cls.results_hedge = run_pipeline(hedge_mode=True)

    # ------------------------------------------------------------------
    # Layer 1 — basic smoke tests
    # ------------------------------------------------------------------

    def test_pipeline_runs_without_errors(self) -> None:
        self.assertIn("backtest", self.results)
        self.assertGreater(self.results["summary"]["universe_size"], 0)

    def test_annualized_vol_is_positive_and_finite(self) -> None:
        vol = self.results["summary"]["metrics"]["annualized_vol"]
        self.assertGreater(vol, 0.0)
        self.assertTrue(np.isfinite(vol))

    def test_sharpe_is_finite(self) -> None:
        sharpe = self.results["summary"]["metrics"]["sharpe"]
        self.assertTrue(np.isfinite(sharpe))

    def test_max_drawdown_is_non_positive(self) -> None:
        mdd = self.results["summary"]["metrics"]["max_drawdown"]
        self.assertLessEqual(mdd, 0.0)

    def test_cvar_is_negative(self) -> None:
        cvar = self.results["summary"]["metrics"]["cvar_95"]
        # CVaR 95% should be a loss (negative daily return)
        self.assertLess(cvar, 0.0)

    def test_fibra_features_present(self) -> None:
        feature_df = self.results["feature_df"]
        fibra_features = feature_df[feature_df["asset_class"] == "fibra"]
        self.assertFalse(fibra_features.empty, "FIBRA features should not be empty")
        for col in ("cap_rate", "ffo_yield"):
            self.assertIn(col, fibra_features.columns)

    def test_fixed_income_features_present(self) -> None:
        feature_df = self.results["feature_df"]
        fixed_features = feature_df[feature_df["asset_class"] == "fixed_income"]
        self.assertFalse(fixed_features.empty, "Fixed income features should not be empty")
        for col in ("duration", "credit_spread"):
            self.assertIn(col, fixed_features.columns)

    # ------------------------------------------------------------------
    # Risk module unit tests
    # ------------------------------------------------------------------

    def test_garch_forecast_positive_vol(self) -> None:
        from src.risk import fit_garch, garch_forecast_vol
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        fitted = fit_garch(returns, "GJR")
        vol = garch_forecast_vol(fitted)
        self.assertGreater(vol, 0.0)
        self.assertTrue(np.isfinite(vol))

    def test_cvar_less_than_var(self) -> None:
        from src.risk import compute_var, compute_cvar
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0, 0.01, 500))
        var = compute_var(returns, alpha=0.95)
        cvar = compute_cvar(returns, alpha=0.95)
        self.assertLessEqual(cvar, var)  # CVaR should be worse (more negative) than VaR

    def test_sortino_greater_than_zero_for_positive_returns(self) -> None:
        from src.risk import compute_sortino
        positive_returns = pd.Series([0.001] * 252)
        sortino = compute_sortino(positive_returns, required_return=0.0)
        self.assertGreater(sortino, 0.0)

    def test_gev_var_finite(self) -> None:
        from src.risk import gev_var
        np.random.seed(1)
        returns = pd.Series(np.random.normal(0, 0.015, 500))
        var, cvar = gev_var(returns)
        self.assertTrue(np.isfinite(var))
        self.assertTrue(np.isfinite(cvar))
        self.assertLess(var, 0.0)   # VaR must be negative (loss convention)
        self.assertLess(cvar, 0.0)  # CVaR must be worse than VaR

    # ------------------------------------------------------------------
    # Portfolio optimization unit tests
    # ------------------------------------------------------------------

    def test_black_litterman_shape(self) -> None:
        from src.portfolio import black_litterman
        market_weights = pd.Series([0.5, 0.5], index=["A", "B"])
        cov_matrix = pd.DataFrame(np.eye(2) * 0.0004, index=["A", "B"], columns=["A", "B"])
        views = {"A": 0.1}
        view_confidences = {"A": 0.5}
        bl_returns = black_litterman(market_weights, cov_matrix, views, view_confidences)
        self.assertEqual(len(bl_returns), len(market_weights))
        self.assertTrue(all(np.isfinite(bl_returns)))

    def test_optimizer_respects_weight_sum(self) -> None:
        from src.portfolio import optimize_portfolio
        # 6 assets × max_position 0.15 = 0.90 = target_net_exposure (feasible)
        tickers = [f"X{i}" for i in range(6)]
        expected_returns = pd.Series([0.10, 0.05, 0.08, 0.06, 0.09, 0.07], index=tickers)
        cov_matrix = pd.DataFrame(np.eye(6) * 0.0004, index=tickers, columns=tickers)
        weights = optimize_portfolio(expected_returns, cov_matrix, target_net_exposure=0.9)
        self.assertAlmostEqual(weights.sum(), 0.9, places=4)

    def test_optimizer_respects_max_position(self) -> None:
        from src.portfolio import optimize_portfolio
        # Force all weight onto A with high expected return; target sum = max × n_assets so feasible
        tickers = ["A", "B", "C"]
        expected_returns = pd.Series([1.0, 0.01, 0.01], index=tickers)
        cov_matrix = pd.DataFrame(np.eye(3) * 0.0004, index=tickers, columns=tickers)
        # target = 0.45 = 3 × 0.15 (exactly feasible at bounds)
        weights = optimize_portfolio(expected_returns, cov_matrix,
                                     max_position=0.15, target_net_exposure=0.45)
        self.assertLessEqual(float(weights.max()), 0.15 + 1e-5)

    # ------------------------------------------------------------------
    # Layer 2 — hedge overlay tests
    # ------------------------------------------------------------------

    def test_hedge_backtest_runs(self) -> None:
        hedge_layer = self.results_hedge["hedge_layer"]
        self.assertIn("metrics", hedge_layer)
        self.assertIn("sharpe", hedge_layer["metrics"])
        self.assertTrue(np.isfinite(hedge_layer["metrics"]["sharpe"]))

    def test_long_short_approximate_neutrality(self) -> None:
        from src.hedge_overlay import long_short_portfolio

        np.random.seed(42)
        signal_df = pd.DataFrame({
            "date": pd.Timestamp("2024-01-01"),
            "ticker": [f"T{i}" for i in range(12)],
            "expected_return": np.random.randn(12),
            "sector": ["A"] * 6 + ["B"] * 6,
        })
        portfolio = long_short_portfolio(signal_df, top_n=3, bottom_n=3, sector_neutral=True)

        if not portfolio.empty:
            for date in portfolio["date"].unique():
                date_book = portfolio[portfolio["date"] == date]
                net_long = date_book[date_book["side"] == "long"]["net_weight"].sum()
                net_short = date_book[date_book["side"] == "short"]["net_weight"].sum()
                self.assertLess(abs(net_long + net_short), 0.05)

    def test_leverage_within_bounds(self) -> None:
        from src.hedge_overlay import dynamic_leverage
        np.random.seed(42)
        returns = pd.Series(np.random.randn(150) * 0.01)
        leverage = dynamic_leverage(returns, max_leverage=1.5, cvar_limit=0.02, window=30)
        self.assertGreaterEqual(float(leverage.min()), 0.5 - 1e-9)
        self.assertLessEqual(float(leverage.max()), 1.5 + 1e-9)
        self.assertEqual(len(leverage), len(returns))

    def test_fx_overlay_hedge_ratio_bounds(self) -> None:
        from src.hedge_overlay import fx_directional_overlay
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        macro_df = pd.DataFrame({
            "date": dates,
            "banxico_rate": 5.5 + np.random.randn(60) * 0.1,
            "us_fed_rate": 5.0 + np.random.randn(60) * 0.1,
            "usd_mxn": 19.0 + np.cumsum(np.random.randn(60) * 0.1),
        })
        signal_df = pd.DataFrame({"date": dates, "ticker": "TEST", "sector": "X", "expected_return": 0.0})
        usd_exposure = pd.Series([0.3])
        fx_overlay = fx_directional_overlay(macro_df, signal_df, usd_exposure,
                                            min_hedge_ratio=0.10, max_hedge_ratio=0.95)
        self.assertIn("hedge_ratio", fx_overlay.columns)
        self.assertGreaterEqual(float(fx_overlay["hedge_ratio"].min()), 0.10 - 1e-9)
        self.assertLessEqual(float(fx_overlay["hedge_ratio"].max()), 0.95 + 1e-9)

    def test_tail_hedge_returns_all_keys_and_types(self) -> None:
        from src.hedge_overlay import tail_risk_hedge
        from scipy.stats import genextreme
        np.random.seed(42)
        returns = pd.Series(np.random.randn(200) * 0.01)
        gev_params = genextreme.fit(-returns[returns < 0])
        result = tail_risk_hedge(returns, gev_params, protection_level=0.99, cost_bps=30.0)
        expected_keys = {"unhedged_loss_at_99", "hedge_payoff", "daily_cost_drag", "net_benefit", "recommended"}
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertIsInstance(result["recommended"], bool)
        for k in ("unhedged_loss_at_99", "hedge_payoff", "daily_cost_drag", "net_benefit"):
            self.assertTrue(np.isfinite(result[k]), f"{k} should be finite")

    # ------------------------------------------------------------------
    # Ledoit-Wolf covariance
    # ------------------------------------------------------------------

    def test_ledoit_wolf_covariance_is_positive_semidefinite(self) -> None:
        from src.backtest import build_covariance_matrix
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=120, freq="B")
        tickers = ["A", "B", "C", "D"]
        prices = pd.DataFrame(
            np.exp(np.cumsum(np.random.normal(0, 0.01, (120, 4)), axis=0)),
            index=dates, columns=tickers,
        )
        returns = prices.pct_change().fillna(0.0)
        cov = build_covariance_matrix(returns, dates[-1])
        eigvals = np.linalg.eigvalsh(cov.values)
        self.assertTrue(all(eigvals >= -1e-9), "Covariance matrix should be positive semi-definite")


if __name__ == "__main__":
    unittest.main()
