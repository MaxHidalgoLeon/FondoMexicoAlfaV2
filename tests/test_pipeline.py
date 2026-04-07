import unittest
import numpy as np

from src.pipeline import run_pipeline


class PipelineTestCase(unittest.TestCase):
    def test_pipeline_runs_without_errors(self) -> None:
        results = run_pipeline()
        self.assertIn("backtest", results)
        self.assertGreater(results["summary"]["universe_size"], 0)
        self.assertGreaterEqual(results["summary"]["metrics"]["annualized_vol"], 0.0)

    def test_fibra_features_present(self) -> None:
        results = run_pipeline()
        feature_df = results["feature_df"]
        fibra_features = feature_df[feature_df["asset_class"] == "fibra"]
        self.assertTrue("cap_rate" in fibra_features.columns)
        self.assertTrue("ffo_yield" in fibra_features.columns)

    def test_fixed_income_features_present(self) -> None:
        results = run_pipeline()
        feature_df = results["feature_df"]
        fixed_features = feature_df[feature_df["asset_class"] == "fixed_income"]
        self.assertTrue("duration" in fixed_features.columns)
        self.assertTrue("credit_spread" in fixed_features.columns)

    def test_garch_runs(self) -> None:
        from src.risk import fit_garch, garch_forecast_vol
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        fitted = fit_garch(returns, "GJR")
        vol = garch_forecast_vol(fitted)
        self.assertGreater(vol, 0.0)

    def test_black_litterman_shape(self) -> None:
        from src.portfolio import black_litterman
        import pandas as pd
        import numpy as np
        market_weights = pd.Series([0.5, 0.5], index=["A", "B"])
        cov_matrix = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
        views = {"A": 0.1}
        view_confidences = {"A": 0.5}
        bl_returns = black_litterman(market_weights, cov_matrix, views, view_confidences)
        self.assertEqual(len(bl_returns), len(market_weights))

    def test_optimizer_constraints(self) -> None:
        from src.portfolio import optimize_portfolio
        import pandas as pd
        import numpy as np
        expected_returns = pd.Series([0.1, 0.1], index=["A", "B"])
        cov_matrix = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
        weights = optimize_portfolio(expected_returns, cov_matrix, target_net_exposure=0.9)
        self.assertAlmostEqual(weights.sum(), 0.9, places=5)
        self.assertTrue(all(weights <= 0.15))

    def test_hedge_backtest_runs(self) -> None:
        """Test that hedge layer backtest runs and produces expected outputs."""
        results = run_pipeline(hedge_mode=True)
        self.assertIn("hedge_layer", results)
        hedge_layer = results["hedge_layer"]
        self.assertIn("metrics", hedge_layer)
        self.assertIn("sharpe", hedge_layer["metrics"])
        sharpe = hedge_layer["metrics"]["sharpe"]
        self.assertTrue(isinstance(sharpe, (float, int)))
        self.assertTrue(np.isfinite(sharpe))

    def test_long_short_neutral(self) -> None:
        """Test that long/short book maintains approximate neutrality."""
        from src.hedge_overlay import long_short_portfolio
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        # Create mock signal dataframe
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        signal_df = pd.DataFrame({
            "date": dates[0],
            "ticker": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"],
            "expected_return": np.random.randn(10),
            "sector": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        })
        
        portfolio = long_short_portfolio(signal_df, top_n=3, bottom_n=2, sector_neutral=True)
        
        if not portfolio.empty:
            for date in portfolio["date"].unique():
                date_book = portfolio[portfolio["date"] == date]
                net_long = date_book[date_book["side"] == "long"]["net_weight"].sum()
                net_short = date_book[date_book["side"] == "short"]["net_weight"].sum()
                # Net exposure should be close to zero (within 0.05)
                self.assertLess(abs(net_long + net_short), 0.05)

    def test_leverage_within_bounds(self) -> None:
        """Test that dynamic leverage stays within [0.5, max_leverage]."""
        from src.hedge_overlay import dynamic_leverage
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        leverage = dynamic_leverage(returns, max_leverage=1.5, cvar_limit=0.02, window=30)
        
        self.assertTrue(leverage.min() >= 0.5)
        self.assertTrue(leverage.max() <= 1.5)
        self.assertEqual(len(leverage), len(returns))

    def test_fx_overlay_hedge_ratio_bounds(self) -> None:
        """Test that FX overlay hedge ratios stay within bounds."""
        from src.hedge_overlay import fx_directional_overlay
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        # Create mock macro data
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        macro_df = pd.DataFrame({
            "date": dates,
            "banxico_rate": 5.5 + np.random.randn(50) * 0.1,
            "us_fed_rate": 5.0 + np.random.randn(50) * 0.1,
            "usd_mxn": 19.0 + np.cumsum(np.random.randn(50) * 0.1),
        })
        
        signal_df = pd.DataFrame({
            "date": dates,
            "ticker": "TEST",
            "sector": "Test",
            "expected_return": 0.0,
        })
        
        usd_exposure = pd.Series([0.3])
        
        fx_overlay = fx_directional_overlay(
            macro_df,
            signal_df,
            usd_exposure,
            min_hedge_ratio=0.10,
            max_hedge_ratio=0.95
        )
        
        self.assertIn("hedge_ratio", fx_overlay.columns)
        self.assertTrue(fx_overlay["hedge_ratio"].min() >= 0.10)
        self.assertTrue(fx_overlay["hedge_ratio"].max() <= 0.95)

    def test_tail_hedge_returns_dict(self) -> None:
        """Test that tail_risk_hedge returns all expected keys."""
        from src.hedge_overlay import tail_risk_hedge
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        
        # Use GEV fit
        from scipy.stats import genextreme
        gev_params = genextreme.fit(-returns[returns < 0])
        
        result = tail_risk_hedge(returns, gev_params, protection_level=0.99, cost_bps=30.0)
        
        expected_keys = {
            "unhedged_loss_at_99",
            "hedge_payoff",
            "daily_cost_drag",
            "net_benefit",
            "recommended",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertIsInstance(result["recommended"], bool)


if __name__ == "__main__":
    unittest.main()
