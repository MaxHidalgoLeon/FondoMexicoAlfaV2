import unittest
import numpy as np
import pandas as pd

from src.pipeline import run_pipeline

FAST_TEST_SETTINGS = {
    "bootstrap_enabled": False,
    "fan_chart_enabled": False,
    "ic_diagnostics_enabled": False,
    "stress_distributional_enabled": False,
    "enable_method_comparison": False,
    "covariance_method": "rolling_ledoit_wolf",
    "regime_method": "threshold_discrete",
    "adtv_method": "uniform",
}

FAST_BOOTSTRAP_SETTINGS = {
    **FAST_TEST_SETTINGS,
    "covariance_method": "ewma_ledoit_wolf",
    "regime_method": "ewma_composite",
    "adtv_method": "ewma",
    "bootstrap_enabled": True,
    "bootstrap_n_reps": 200,
    "bootstrap_block_size": 20,
    "fan_chart_enabled": False,
    "ic_diagnostics_enabled": True,
    "stress_distributional_enabled": True,
}


class PipelineTestCase(unittest.TestCase):
    """Integration tests for the full Layer 1 + Layer 2 pipeline.

    run_pipeline() is called once per class (setUpClass) to avoid redundant
    computation across individual test methods.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = run_pipeline(hedge_mode=False, settings=FAST_TEST_SETTINGS)
        cls.results_hedge = run_pipeline(hedge_mode=True, settings=FAST_TEST_SETTINGS)
        cls.results_bootstrap = run_pipeline(hedge_mode=False, settings=FAST_BOOTSTRAP_SETTINGS)

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
        # 6 assets × max_position 0.10 = 0.60; target=0.60 (feasible at 0.10 cap)
        tickers = [f"X{i}" for i in range(6)]
        expected_returns = pd.Series([0.10, 0.05, 0.08, 0.06, 0.09, 0.07], index=tickers)
        cov_matrix = pd.DataFrame(np.eye(6) * 0.0004, index=tickers, columns=tickers)
        weights = optimize_portfolio(expected_returns, cov_matrix, target_net_exposure=0.6)
        self.assertAlmostEqual(weights.sum(), 0.6, places=4)

    def test_optimizer_respects_max_position(self) -> None:
        from src.portfolio import optimize_portfolio
        # Force all weight onto A with high expected return; CNBV cap at 0.10
        tickers = ["A", "B", "C"]
        expected_returns = pd.Series([1.0, 0.01, 0.01], index=tickers)
        cov_matrix = pd.DataFrame(np.eye(3) * 0.0004, index=tickers, columns=tickers)
        # target = 0.30 = 3 × 0.10 (exactly feasible at bounds)
        weights = optimize_portfolio(expected_returns, cov_matrix,
                                     max_position=0.10, target_net_exposure=0.30)
        self.assertLessEqual(float(weights.max()), 0.10 + 1e-5)

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

    # ==================================================================
    # CNBV REGULATORY COMPLIANCE TESTS
    # ==================================================================

    def test_no_consumer_staples_in_universe(self) -> None:
        """Consumer staples (FEMSAUBD, KIMBERA, BIMBOA, GRUMAB, BECLE)
        must not appear in the investable universe."""
        from src.data_loader import get_investable_universe
        universe = get_investable_universe()
        banned = {"FEMSAUBD", "KIMBERA", "BIMBOA", "GRUMAB", "BECLE"}
        found = banned & set(universe["ticker"])
        self.assertEqual(found, set(), f"Consumer staples found in universe: {found}")

    def test_universe_has_industrial_fibras(self) -> None:
        """TERRA13 and FMTY14 should be in the investable universe."""
        from src.data_loader import get_investable_universe
        universe = get_investable_universe()
        required = {"TERRA13", "FMTY14"}
        missing = required - set(universe["ticker"])
        self.assertEqual(missing, set(), f"Industrial FIBRAs missing from universe: {missing}")

    def test_universe_has_thematic_purity_and_issuer_id(self) -> None:
        """Universe must have thematic_purity, issuer_id, max_position_override."""
        from src.data_loader import get_investable_universe
        universe = get_investable_universe()
        for col in ("thematic_purity", "issuer_id", "max_position_override"):
            self.assertIn(col, universe.columns, f"Missing column: {col}")

    def test_funo11_has_position_override(self) -> None:
        """FUNO11 must be capped at 4% via max_position_override."""
        from src.data_loader import get_investable_universe
        universe = get_investable_universe()
        funo = universe[universe["ticker"] == "FUNO11"]
        self.assertEqual(len(funo), 1)
        override = funo["max_position_override"].iloc[0]
        self.assertAlmostEqual(override, 0.04, places=4)

    def test_max_position_below_regulatory_cap(self) -> None:
        """No individual position should exceed 10% of NAV (CNBV limit)."""
        weights_df = self.results["backtest"]["weights"]
        if weights_df.empty:
            self.skipTest("No weights available")
        max_weight = weights_df.abs().max().max()
        self.assertLessEqual(max_weight, 0.10 + 5e-3,
                             f"Max position {max_weight:.4f} exceeds CNBV 10% cap")

    def test_fixed_income_not_in_optimizer_weights(self) -> None:
        """Fixed income tickers should NOT appear in optimizer weights
        since they are allocated as a non-optimizable liquidity sleeve."""
        from src.data_loader import get_investable_universe
        universe = get_investable_universe()
        fi_tickers = set(universe.loc[universe["asset_class"] == "fixed_income", "ticker"])
        weights_df = self.results["backtest"]["weights"]
        fi_in_weights = fi_tickers & set(weights_df.columns)
        self.assertEqual(fi_in_weights, set(),
                         f"Fixed income tickers found in optimizer weights: {fi_in_weights}")

    def test_only_liquidity_sleeve_bonds_in_universe(self) -> None:
        """Only CETES28, CETES91, MBONO3Y should be in fixed income universe."""
        from src.data_loader import get_investable_universe
        universe = get_investable_universe()
        fi_tickers = set(universe.loc[universe["asset_class"] == "fixed_income", "ticker"])
        expected = {"CETES28", "CETES91", "MBONO3Y"}
        self.assertEqual(fi_tickers, expected,
                         f"Fixed income tickers {fi_tickers} != expected {expected}")

    def test_issuer_id_no_double_count(self) -> None:
        """CORP1 (CEMEX bond) must not coexist with CEMEXCPO in the universe,
        since they share issuer_id='CEMEX'."""
        from src.data_loader import get_investable_universe
        universe = get_investable_universe()
        tickers = set(universe["ticker"])
        # Both CORP1 and CEMEXCPO should not be present simultaneously
        self.assertNotIn("CORP1", tickers, "CORP1 should have been removed (double-count with CEMEXCPO)")

    def test_regime_detection_returns_valid_state(self) -> None:
        """detect_macro_regime should return one of the three valid states."""
        from src.risk import detect_macro_regime
        valid_regimes = {"expansion", "tightening", "stress"}
        # Test with minimal data → should fallback to "expansion"
        regime = detect_macro_regime(None)
        self.assertIn(regime, valid_regimes)
        # Test with normal data
        dates = pd.date_range("2024-01-01", periods=12, freq="ME")
        macro_df = pd.DataFrame({
            "date": dates,
            "banxico_rate": [0.055] * 12,
            "industrial_production_yoy": [0.03] * 12,
            "usd_mxn": [19.0] * 12,
        })
        regime = detect_macro_regime(macro_df)
        self.assertIn(regime, valid_regimes)

    def test_dynamic_liquidity_threshold_used(self) -> None:
        """Pipeline should use a dynamic threshold (20th pctl) instead of
        the old static min_liquidity_score=0.47."""
        import inspect
        from src.pipeline import run_pipeline as _rp
        sig = inspect.signature(_rp)
        self.assertNotIn("min_liquidity_score", sig.parameters,
                         "run_pipeline still accepts min_liquidity_score — should be dynamic")

    # ------------------------------------------------------------------
    # EWMA / bootstrap extensions
    # ------------------------------------------------------------------

    def test_ewma_covariance_is_psd(self) -> None:
        from src.backtest import build_covariance_matrix

        rng = np.random.default_rng(123)
        dates = pd.date_range("2022-01-01", periods=320, freq="B")
        returns = pd.DataFrame(
            rng.normal(0.0, 0.01, size=(320, 4)),
            index=dates,
            columns=["A", "B", "C", "D"],
        )
        cov = build_covariance_matrix(
            returns,
            dates[-1],
            settings={**FAST_BOOTSTRAP_SETTINGS, "covariance_method": "ewma_ledoit_wolf"},
        )
        eigvals = np.linalg.eigvalsh(cov.values)
        self.assertTrue(np.all(eigvals >= -1e-9))

    def test_ewma_cov_more_sensitive_to_recent_shock(self) -> None:
        from src.backtest import build_covariance_matrix

        rng = np.random.default_rng(321)
        dates = pd.date_range("2022-01-01", periods=300, freq="B")
        base = rng.normal(0.0, 0.005, size=(280, 3))
        shock = rng.normal(0.0, [0.04, 0.035, 0.005], size=(20, 3))
        returns = pd.DataFrame(
            np.vstack([base, shock]),
            index=dates,
            columns=["A", "B", "C"],
        )
        cov_roll = build_covariance_matrix(
            returns,
            dates[-1],
            settings={**FAST_BOOTSTRAP_SETTINGS, "covariance_method": "rolling_ledoit_wolf"},
        )
        cov_ewma = build_covariance_matrix(
            returns,
            dates[-1],
            settings={**FAST_BOOTSTRAP_SETTINGS, "covariance_method": "ewma_ledoit_wolf"},
        )
        self.assertGreater(float(cov_ewma.loc["A", "A"]), float(cov_roll.loc["A", "A"]))
        self.assertGreater(float(cov_ewma.loc["B", "B"]), float(cov_roll.loc["B", "B"]))

    def test_regime_ewma_reduces_switching_frequency(self) -> None:
        from src.data_loader import build_mock_macro_series
        from src.risk import compute_macro_regime_history

        macro = build_mock_macro_series("2017-01-01", "2026-03-31")
        discrete = compute_macro_regime_history(macro, settings={"regime_method": "threshold_discrete"})
        ewma = compute_macro_regime_history(macro, settings={"regime_method": "ewma_composite"})
        switches_discrete = int((discrete["regime"] != discrete["regime"].shift(1)).sum() - 1)
        switches_ewma = int((ewma["regime"] != ewma["regime"].shift(1)).sum() - 1)
        self.assertLessEqual(switches_ewma, switches_discrete)

    def test_adtv_ewma_decay_halflife(self) -> None:
        from src.data_loader import ewma_decay_weights

        weights = ewma_decay_weights(252, 0.97)
        self.assertAlmostEqual(float(weights[0]), 0.5 ** (251 / 23) * (1 - 0.97), delta=5e-4)
        self.assertAlmostEqual(float(weights[-23:].sum()), 0.5, delta=0.08)

    def test_ewma_config_fallback_to_uniform(self) -> None:
        from src.backtest import _rolling_ledoit_wolf_covariance, build_covariance_matrix

        rng = np.random.default_rng(11)
        dates = pd.date_range("2023-01-01", periods=120, freq="B")
        returns = pd.DataFrame(rng.normal(0.0, 0.01, size=(120, 4)), index=dates, columns=list("ABCD"))
        legacy = _rolling_ledoit_wolf_covariance(returns, dates[-1], 63)
        fallback = build_covariance_matrix(
            returns,
            dates[-1],
            settings={"covariance_method": "rolling_ledoit_wolf", "rolling_cov_window": 63},
        )
        self.assertTrue(np.allclose(legacy.values, fallback.values))

    def test_realized_vol_ewma_finite_and_positive(self) -> None:
        from reports.charts import compute_realized_volatility

        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(0.0, 0.01, 80))
        vol = compute_realized_volatility(returns, method="ewma", span=21).dropna()
        self.assertTrue((vol > 0).all())
        self.assertTrue(np.isfinite(vol).all())

    def test_bootstrap_metric_returns_valid_ci(self) -> None:
        from src.bootstrap import bootstrap_metric

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.01, 300))
        stats = bootstrap_metric(returns, lambda s: float(pd.Series(s).mean()), block_size=20, n_reps=300, seed=42)
        self.assertLessEqual(stats["ci_low"], stats["point"])
        self.assertGreaterEqual(stats["ci_high"], stats["point"])
        self.assertTrue(np.isfinite(stats["se"]))

    def test_bootstrap_paired_difference_vs_self_is_zero(self) -> None:
        from src.bootstrap import bootstrap_paired_difference

        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.0, 0.01, 240))
        stats = bootstrap_paired_difference(
            returns,
            returns,
            lambda a, b: float((pd.Series(a) - pd.Series(b)).mean()),
            block_size=20,
            n_reps=300,
            seed=7,
        )
        self.assertAlmostEqual(stats["point"], 0.0, places=10)
        self.assertLessEqual(stats["ci_low"], 0.0)
        self.assertGreaterEqual(stats["ci_high"], 0.0)
        self.assertGreaterEqual(stats["p_value"], 0.9)

    def test_stationary_bootstrap_preserves_autocorrelation(self) -> None:
        from src.bootstrap import bootstrap_paths

        rng = np.random.default_rng(1234)
        shocks = rng.normal(0.0, 1.0, 400)
        x = np.zeros(400)
        for i in range(1, len(x)):
            x[i] = 0.3 * x[i - 1] + shocks[i]
        paths = bootstrap_paths(pd.Series(x), n_paths=40, block_size=20, seed=1234)
        acfs = []
        for path in paths:
            acfs.append(float(pd.Series(path).autocorr(lag=1)))
        mean_acf = float(np.nanmean(acfs))
        self.assertGreaterEqual(mean_acf, 0.1)
        self.assertLessEqual(mean_acf, 0.5)

    def test_stress_distributional_more_dispersed_than_deterministic(self) -> None:
        from src.risk import distributional_stress_test

        rng = np.random.default_rng(17)
        dates = pd.date_range("2024-01-01", periods=220, freq="B")
        asset_returns = pd.DataFrame({"A": rng.normal(0.0, 0.015, len(dates))}, index=dates)
        macro = pd.DataFrame({
            "date": dates,
            "banxico_rate": np.linspace(0.08, 0.10, len(dates)),
            "usd_mxn": np.concatenate([np.full(180, 17.0), np.linspace(17.0, 19.5, 40)]),
            "us_ip_yoy": np.linspace(0.02, -0.03, len(dates)),
        })
        dist = distributional_stress_test(asset_returns, pd.Series({"A": 1.0}), macro, n_reps=300, window_days=21, seed=17)
        self.assertIn("peso_depreciation", dist)
        self.assertGreater(dist["peso_depreciation"]["pnl_distribution"]["std"], 0.0)

    def test_ic_bootstrap_finite_and_ordered(self) -> None:
        signal_diag = self.results_bootstrap["signal_diagnostics"]
        self.assertTrue(signal_diag)
        for stats in signal_diag.values():
            self.assertLess(stats["ci_low"], stats["ic_mean"])
            self.assertGreater(stats["ci_high"], stats["ic_mean"])
            self.assertTrue(np.isfinite(stats["p_value"]))

    def test_fan_chart_median_close_to_realized(self) -> None:
        from reports.charts import build_bootstrap_fan_chart_data

        rng = np.random.default_rng(55)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)
        fan = build_bootstrap_fan_chart_data(returns, n_paths=300, block_size=20, seed=55)
        observed_final = float(np.exp(returns.cumsum()).iloc[-1])
        final_distribution = fan["paths"][:, -1]
        median_final = float(fan["median"].iloc[-1])
        self.assertLess(abs(observed_final - median_final), float(np.std(final_distribution, ddof=1)))

    def test_bootstrap_disabled_fallback(self) -> None:
        self.assertNotIn("metrics_ci", self.results["backtest"])
        self.assertEqual(self.results["benchmarks"].get("alpha_significance"), {})
        self.assertIn("metrics_ci", self.results_bootstrap["backtest"])

    def test_bootstrap_enabled_does_not_change_point_estimates(self) -> None:
        disabled = run_pipeline(
            hedge_mode=False,
            settings={**FAST_BOOTSTRAP_SETTINGS, "bootstrap_enabled": False, "ic_diagnostics_enabled": False, "stress_distributional_enabled": False},
        )
        enabled = run_pipeline(
            hedge_mode=False,
            settings={**FAST_BOOTSTRAP_SETTINGS, "bootstrap_enabled": True, "bootstrap_n_reps": 100},
        )
        disabled_metrics = disabled["summary"]["metrics"]
        enabled_metrics = enabled["summary"]["metrics"]
        for key in ("sharpe", "sortino", "cvar_95", "annualized_return", "annualized_vol", "turnover"):
            self.assertAlmostEqual(disabled_metrics[key], enabled_metrics[key], places=10)

    def test_block_size_auto_returns_sensible_value(self) -> None:
        from src.bootstrap import bootstrap_block_size_selector

        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.0, 0.01, 300))
        block = bootstrap_block_size_selector(returns)
        self.assertIsInstance(block, int)
        self.assertGreaterEqual(block, 5)
        self.assertLessEqual(block, 60)


if __name__ == "__main__":
    unittest.main()
