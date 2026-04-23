from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .alpha_significance import compute_benchmark_alpha_significance
from .backtest import run_backtest
from .signal_diagnostics import compute_signal_ic_diagnostics
from .features import build_signal_matrix
from .signals import score_cross_section, forecast_returns
from .portfolio import black_litterman, apply_fx_overlay
from .risk import (
    compute_macro_regime_history,
    distributional_stress_test,
    stress_test,
    fit_garch,
    garch_forecast_vol,
    rolling_garch_forecast,
    dynamic_var,
    monte_carlo_var,
    gev_var,
)
from .settings import resolve_settings

logger = logging.getLogger(__name__)


def _load_benchmark_returns(
    data_source: str,
    start_date: str,
    end_date: str,
    benchmark_tickers: list[str] | None = None,
    provider_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Load benchmark return series (daily) when available.

    Defaults to IPC (^MXX). Additional tickers can be passed (e.g., GBM funds)
    and will be added automatically once provided.
    """
    source = (data_source or "").lower().strip()
    tickers = benchmark_tickers or (["^MXX"] if source == "yahoo" else [])
    if not tickers:
        return pd.DataFrame()

    try:
        if source == "yahoo":
            import yfinance as yf
            from .data_providers import get_provider

            raw_yahoo_tickers = [t for t in tickers if isinstance(t, str) and (t.startswith("^") or "." in t)]
            canonical_tickers = [t for t in tickers if t not in raw_yahoo_tickers]

            price_parts = []
            if canonical_tickers:
                provider = get_provider("yahoo")
                mapped_prices = provider.get_prices(canonical_tickers, start_date, end_date)
                if mapped_prices is not None and not mapped_prices.empty:
                    price_parts.append(mapped_prices)

            if raw_yahoo_tickers:
                raw = yf.download(
                    raw_yahoo_tickers,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )["Close"]
                if isinstance(raw, pd.Series):
                    raw = raw.to_frame(name=raw_yahoo_tickers[0])
                raw_prices = raw.sort_index().ffill(limit=5)
                price_parts.append(raw_prices)

            if not price_parts:
                return pd.DataFrame()

            prices = pd.concat(price_parts, axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated()].sort_index().ffill(limit=5)
        else:
            from .data_providers import get_provider

            provider = get_provider(source, **(provider_kwargs or {}))
            prices = provider.get_prices(tickers, start_date, end_date)
            if prices is None or prices.empty:
                logger.warning("Benchmark provider load returned empty for source=%s tickers=%s", source, tickers)
                return pd.DataFrame()
            prices = prices.sort_index().ffill(limit=5)

        returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
        return returns
    except Exception as exc:
        logger.warning("Benchmark load failed for source=%s (%s).", source, exc)
        return pd.DataFrame()


def run_pipeline(
    hedge_mode: bool = False,
    data_source: str = "mock",
    start_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    optimizer: str = "mv",
    benchmark_tickers: list[str] | None = None,
    hedge_mode_config: str = "analytical",
    settings: dict | None = None,
    **provider_kwargs,
) -> dict[str, object]:
    from .data_loader import load_data, compute_adtv_liquidity_scores
    from .risk import detect_macro_regime
    cfg = resolve_settings(settings)
    data = load_data(source=data_source, start_date=start_date, end_date=end_date, **provider_kwargs)
    universe = data["universe"]
    prices = data["prices"]
    fundamentals = data["fundamentals"]
    fibra_fundamentals = data["fibra_fundamentals"]
    bonds = data["bonds"]
    macro = data["macro"]

    # ------------------------------------------------------------------
    # ADTV-based liquidity scores — only for real data sources.
    # For mock, the hardcoded scores in get_investable_universe() are used.
    # ------------------------------------------------------------------
    adtv_scores_selected: pd.Series | None = None
    adtv_scores_uniform: pd.Series | None = None
    if data_source != "mock":
        try:
            from .data_providers import get_provider
            provider = get_provider(data_source, **provider_kwargs)
            equity_tickers = universe.loc[
                universe["asset_class"].isin(["equity", "fibra"]), "ticker"
            ].tolist()
            volume = provider.get_volume(equity_tickers, start_date, end_date)
            adtv_scores_uniform = compute_adtv_liquidity_scores(
                prices,
                volume,
                window=int(cfg["adtv_window"]),
                method="uniform",
            )
            adtv_scores_selected = compute_adtv_liquidity_scores(
                prices,
                volume,
                window=int(cfg["adtv_window"]),
                method=str(cfg["adtv_method"]),
                ewma_lambda=float(cfg["adtv_ewma_lambda"]),
                min_periods=int(cfg["adtv_min_periods"]),
            )
            # Update universe with real ADTV-based scores; bonds keep their 1.0
            universe = universe.copy()
            universe.loc[universe["ticker"].isin(adtv_scores_selected.index), "liquidity_score"] = (
                universe.loc[universe["ticker"].isin(adtv_scores_selected.index), "ticker"]
                .map(adtv_scores_selected)
            )
            logger.info(
                "ADTV liquidity scores updated from %s for %d tickers using method=%s.",
                data_source,
                len(adtv_scores_selected),
                cfg["adtv_method"],
            )
        except Exception as exc:
            logger.warning("ADTV score update failed (%s) — using hardcoded scores.", exc)

    # ------------------------------------------------------------------
    # Dynamic liquidity filter — remove bottom 20th percentile of
    # equity/FIBRA liquidity scores. Fixed-income is never filtered.
    # Replaces the old static min_liquidity_score=0.47.
    # ------------------------------------------------------------------
    eq_fibra_scores = universe.loc[
        universe["asset_class"].isin(["equity", "fibra"]), "liquidity_score"
    ]
    if not eq_fibra_scores.empty:
        dynamic_threshold = float(eq_fibra_scores.quantile(0.20))
    else:
        dynamic_threshold = 0.0
    illiquid = universe.loc[
        universe["asset_class"].isin(["equity", "fibra"]) &
        (universe["liquidity_score"] < dynamic_threshold),
        "ticker"
    ].tolist()
    if illiquid:
        logger.info(
            "Dynamic liquidity filter (20th pctl = %.4f) removing %d ticker(s): %s",
            dynamic_threshold, len(illiquid), illiquid,
        )
        universe = universe[
            ~(universe["ticker"].isin(illiquid))
        ].reset_index(drop=True)
        keep = set(universe["ticker"])
        prices = prices[[c for c in prices.columns if c in keep]]
        # Sync data dict so the report sees the post-filter universe
        data["universe"] = universe

    feature_df = build_signal_matrix(prices, fundamentals, fibra_fundamentals, bonds, macro, universe)
    scored = score_cross_section(feature_df)
    forecast_df = forecast_returns(
        scored,
        np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0),
        settings=cfg,
    )

    # ------------------------------------------------------------------
    # Separate equity/FIBRA tickers (optimizable) from fixed_income (sleeve)
    # ------------------------------------------------------------------
    fi_tickers = universe.loc[universe["asset_class"] == "fixed_income", "ticker"].tolist()
    optimizable_tickers = universe.loc[
        universe["asset_class"].isin(["equity", "fibra"]) & universe["investable"], "ticker"
    ].tolist()

    # Filter forecast_df and prices to exclude fixed_income from optimizer
    forecast_df_opt = forecast_df[~forecast_df["ticker"].isin(fi_tickers)].copy() if not forecast_df.empty else forecast_df
    prices_opt = prices[[c for c in prices.columns if c not in fi_tickers]]

    # Black-Litterman
    forecast_tickers = forecast_df_opt["ticker"].unique() if not forecast_df_opt.empty else []
    market_weights = universe.set_index("ticker")["market_cap_mxn"] / (universe["market_cap_mxn"].sum() + 1e-9)
    market_weights = market_weights.reindex(forecast_tickers).fillna(0.0)

    # Warn about any ticker mismatch between forecast and universe
    missing_from_universe = set(forecast_tickers) - set(universe["ticker"])
    if missing_from_universe:
        logger.warning("Tickers in forecast but not in universe (will get 0 market weight): %s", missing_from_universe)

    # Build covariance with Ledoit-Wolf shrinkage
    raw_returns = np.log(prices_opt / prices_opt.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    cov_tickers = [t for t in forecast_tickers if t in raw_returns.columns]
    if cov_tickers:
        lw = LedoitWolf()
        lw.fit(raw_returns[cov_tickers].fillna(0.0))
        cov_matrix = pd.DataFrame(lw.covariance_, index=cov_tickers, columns=cov_tickers)
    else:
        cov_matrix = pd.DataFrame(dtype=float)

    views = forecast_df_opt.groupby("ticker")["expected_return"].mean().to_dict()
    # Data-driven view confidences: scale |expected_return| to [0.30, 0.70]
    abs_views = pd.Series(views).abs()
    view_range = abs_views.max() - abs_views.min()
    if view_range > 1e-9:
        view_confidences = (0.30 + 0.40 * (abs_views - abs_views.min()) / view_range).to_dict()
    else:
        view_confidences = {t: 0.50 for t in views}

    # Align market_weights to the covariance matrix's ticker set before BL
    market_weights = market_weights.reindex(cov_matrix.columns).fillna(0.0)

    bl_returns = black_litterman(
        market_weights, cov_matrix, views, view_confidences,
        risk_aversion=float(cfg["bl_risk_aversion"]),
        tau=float(cfg["bl_tau"]),
    )

    # Risk-free rate: último valor de Banxico disponible en macro
    banxico_series = macro["banxico_rate"].dropna()
    risk_free_rate = float(banxico_series.iloc[-1]) if not banxico_series.empty else 0.02
    # Normalize units: providers may deliver 11.25 (percent) instead of 0.1125 (decimal).
    if risk_free_rate > 1.0:
        risk_free_rate = risk_free_rate / 100.0
    logger.info("Risk-free rate (Banxico): %.4f", risk_free_rate)

    # GARCH sobre retornos de USD/MXN para proyectar vol y drift
    usdmxn_returns = np.log(macro["usd_mxn"] / macro["usd_mxn"].shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    mxn_garch_vol = None
    if len(usdmxn_returns) >= 30:
        try:
            _mxn_garch = fit_garch(usdmxn_returns)
            mxn_garch_vol = garch_forecast_vol(_mxn_garch)
            expected_usdmxn_return = float(usdmxn_returns.mean() + _mxn_garch.resid.mean())
            logger.info("GARCH USD/MXN vol forecast (anualizada): %.4f", mxn_garch_vol)
        except Exception:
            logger.warning("GARCH fit para USD/MXN falló — usando media histórica.")
            expected_usdmxn_return = float(usdmxn_returns.mean())
    else:
        expected_usdmxn_return = float(usdmxn_returns.mean()) if len(usdmxn_returns) > 0 else 0.0

    # FX overlay
    usd_exposure = universe.set_index("ticker")["usd_exposure"]
    current_usdmxn = macro["usd_mxn"].iloc[-1] if not macro.empty and "usd_mxn" in macro.columns and len(macro) > 0 else 20.0
    adjusted_returns = apply_fx_overlay(
        bl_returns,
        usd_exposure,
        current_usdmxn,
        expected_usdmxn_return,
        hedge_ratio=float(cfg["fx_hedge_ratio_default"]),
    )

    # Merge BL-adjusted expected returns back into forecast_df before backtesting
    if not adjusted_returns.empty and not forecast_df_opt.empty:
        bl_lookup = adjusted_returns.rename("bl_expected_return")
        forecast_df_opt = forecast_df_opt.copy()
        forecast_df_opt["expected_return"] = forecast_df_opt["ticker"].map(bl_lookup).fillna(
            forecast_df_opt["expected_return"]
        )

    # ------------------------------------------------------------------
    # Liquidity sleeve — regime-based fixed income allocation (non-optimizable)
    # CETES28 + CETES91 bounds by regime; MBONO3Y optional buffer.
    # ------------------------------------------------------------------
    _sleeve_bounds = {
        "expansion":  {"min": 0.03, "max": 0.05},
        "tightening": {"min": 0.05, "max": 0.08},
        "stress":     {"min": 0.08, "max": 0.15},
    }
    _macro_for_regime = macro.copy()
    regime = detect_macro_regime(_macro_for_regime, settings=cfg)
    sleeve_range = _sleeve_bounds.get(regime, _sleeve_bounds["expansion"])
    cetes_weight = (sleeve_range["min"] + sleeve_range["max"]) / 2.0
    cetes28_weight = cetes_weight / 2.0
    cetes91_weight = cetes_weight / 2.0
    mbono3y_weight = 0.0  # disabled by default (mbono3y_buffer_enabled=false)
    total_sleeve = cetes_weight + mbono3y_weight

    logger.info(
        "Liquidity sleeve: regime=%s, CETES=%.4f (28d=%.4f + 91d=%.4f), MBONO3Y=%.4f",
        regime, cetes_weight, cetes28_weight, cetes91_weight, mbono3y_weight,
    )

    # ------------------------------------------------------------------
    # Asset-class constraints — equity + FIBRA only (fixed_income removed)
    # Target net exposure for optimizer = 1.0 - total_sleeve
    # ------------------------------------------------------------------
    optimizable_universe = universe[universe["asset_class"].isin(["equity", "fibra"])]
    ac_map = optimizable_universe.set_index("ticker")["asset_class"].to_dict()
    equity_target = 1.0 - total_sleeve
    asset_class_constraints = {
        "__asset_class_map__": ac_map,
        "equity": {"min": 0.50 * equity_target, "max": 0.90 * equity_target},
        "fibra":  {"min": 0.05 * equity_target, "max": 0.30 * equity_target},
    }

    # ADTV proxy: use universe liquidity_score as the per-ticker score vector
    adtv_scores = universe.set_index("ticker")["liquidity_score"].astype(float)

    # ------------------------------------------------------------------
    # Build issuer consolidated limits (CNBV 10% per issuer)
    # ------------------------------------------------------------------
    issuer_consolidated_limits = {}
    if "issuer_id" in universe.columns:
        _opt_tickers = [t for t in optimizable_tickers if t in prices_opt.columns]
        for issuer, group in universe[universe["ticker"].isin(_opt_tickers)].groupby("issuer_id"):
            group_tickers = group["ticker"].tolist()
            if len(group_tickers) > 1:
                idx = [_opt_tickers.index(t) for t in group_tickers if t in _opt_tickers]
                if idx:
                    issuer_consolidated_limits[issuer] = idx

    # Build max_position_overrides
    max_position_overrides = {}
    if "max_position_override" in universe.columns:
        for _, row in universe.iterrows():
            if pd.notna(row.get("max_position_override")):
                max_position_overrides[row["ticker"]] = row["max_position_override"]

    backtest_results = run_backtest(
        prices_opt, forecast_df_opt, optimizable_universe,
        risk_free_rate=risk_free_rate,
        asset_class_constraints=asset_class_constraints,
        optimizer=optimizer,
        adtv_scores=adtv_scores,
        macro=macro,
        issuer_consolidated_limits=issuer_consolidated_limits,
        max_position_overrides=max_position_overrides,
        settings=cfg,
    )

    # Store sleeve and regime info for reporting
    data["liquidity_sleeve"] = {
        "regime": regime,
        "cetes28_weight": cetes28_weight,
        "cetes91_weight": cetes91_weight,
        "mbono3y_weight": mbono3y_weight,
        "total_sleeve": total_sleeve,
        "sleeve_bounds": sleeve_range,
    }

    # Dynamic stress exposures derived from the final portfolio weights.
    # Uses the last rebalance snapshot (row where total weight > 0).
    _wt_rows = backtest_results["weights"].loc[
        backtest_results["weights"].abs().sum(axis=1) > 1e-9
    ]
    if not _wt_rows.empty:
        _final_w = _wt_rows.iloc[-1]
        _usd_exp = universe.set_index("ticker")["usd_exposure"]
        _fi_tickers = universe.loc[universe["asset_class"] == "fixed_income", "ticker"].tolist()
        # Peso-depreciation exposure: domestic (non-USD) weight hurts most
        _usd_w = float(_final_w.dot(_usd_exp.reindex(_final_w.index).fillna(0.0)))
        _peso_exp = float(np.clip(1.0 - _usd_w, 0.20, 0.85))
        # Banxico-shock exposure: bond allocation amplified by duration effect
        _fi_w = float(_final_w.reindex(_fi_tickers).fillna(0.0).sum())
        _banxico_exp = float(np.clip(_fi_w * 3.0 + (1.0 - _fi_w) * 0.25, 0.15, 0.75))
        # US-slowdown exposure: USD-linked / export names move most
        _us_exp = float(np.clip(_usd_w * 1.2, 0.15, 0.80))
    else:
        _peso_exp, _banxico_exp, _us_exp = 0.6, 0.5, 0.4

    exposures = {
        "banxico_shock": _banxico_exp,
        "peso_depreciation": _peso_exp,
        "us_slowdown": _us_exp,
    }
    scenario_shocks = {
        "banxico_shock": -0.03,
        "peso_depreciation": -0.05,
        "us_slowdown": -0.04,
    }
    stress_deterministic = stress_test(
        backtest_results["returns"],
        scenario_shocks,
        exposures,
        risk_free_rate=risk_free_rate,
        shock_days=21,
        event_spacing_days=126,
    )

    # Additional risk metrics
    returns = backtest_results["returns"]
    # Per-ticker daily returns aligned to the backtest period (for multivariate MC)
    if not backtest_results["weights"].empty:
        filtered_weights = backtest_results["weights"].loc[
            backtest_results["weights"].abs().sum(axis=1) > 1e-9
        ]
        final_weights = filtered_weights.iloc[-1] if not filtered_weights.empty else None
    else:
        final_weights = None
    raw_daily_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    garch_vol_series = rolling_garch_forecast(
        returns,
        horizon=int(cfg["garch_forecast_horizon"]),
        lookback=int(cfg["garch_lookback"]),
        refit_every=int(cfg["garch_refit_every"]),
    )
    garch_vol = (
        float(garch_vol_series.dropna().iloc[-1])
        if not garch_vol_series.dropna().empty
        else garch_forecast_vol(fit_garch(returns))
    )
    dyn_var = dynamic_var(returns).iloc[-1]
    mc_var = monte_carlo_var(returns, asset_returns=raw_daily_returns, weights=final_weights)
    gev_v, gev_cv = gev_var(returns)
    stress_distributional = {}
    if cfg["stress_distributional_enabled"] and final_weights is not None:
        stress_distributional = distributional_stress_test(
            raw_daily_returns.reindex(columns=final_weights.index).dropna(how="all"),
            final_weights,
            macro,
            n_reps=int(cfg["bootstrap_n_reps"]),
            window_days=int(cfg["stress_window_days"]),
            seed=int(cfg["bootstrap_seed"]),
        )

    regime_history_discrete = compute_macro_regime_history(
        macro,
        settings={**cfg, "regime_method": "threshold_discrete"},
    )
    regime_history_ewma = compute_macro_regime_history(
        macro,
        settings={**cfg, "regime_method": "ewma_composite"},
    )

    def _count_switches(series: pd.Series | None) -> int:
        if series is None or len(series) == 0:
            return 0
        seq = pd.Series(series).dropna().astype(str)
        if seq.empty:
            return 0
        return int((seq != seq.shift(1)).sum() - 1)

    method_comparison = {}
    if cfg["enable_method_comparison"]:
        baseline_settings = {
            **cfg,
            "covariance_method": "rolling_ledoit_wolf",
            "regime_method": "threshold_discrete",
            "adtv_method": "uniform",
            "bootstrap_enabled": False,
        }
        baseline_adtv = adtv_scores
        if adtv_scores_uniform is not None:
            baseline_adtv = adtv_scores_uniform.reindex(adtv_scores.index).fillna(adtv_scores)
        baseline_backtest = run_backtest(
            prices_opt,
            forecast_df_opt,
            optimizable_universe,
            risk_free_rate=risk_free_rate,
            asset_class_constraints=asset_class_constraints,
            optimizer=optimizer,
            adtv_scores=baseline_adtv,
            macro=macro,
            issuer_consolidated_limits=issuer_consolidated_limits,
            max_position_overrides=max_position_overrides,
            settings=baseline_settings,
        )
        method_comparison = {
            "current": backtest_results["metrics"],
            "baseline": baseline_backtest["metrics"],
            "regime_switches_before": _count_switches(regime_history_discrete.get("regime")),
            "regime_switches_after": _count_switches(regime_history_ewma.get("regime")),
            "turnover_before": float(baseline_backtest["metrics"].get("turnover", np.nan)),
            "turnover_after": float(backtest_results["metrics"].get("turnover", np.nan)),
            "transaction_cost_saved_bps_annualized": float(
                (
                    baseline_backtest["metrics"].get("turnover", 0.0)
                    - backtest_results["metrics"].get("turnover", 0.0)
                )
                * 252.0
                * 10000.0
                * 0.001
            ),
        }

    summary = {
        "universe_size": len(universe),
        "start_date": prices.index.min(),
        "end_date": prices.index.max(),
        "optimizer": optimizer,
        "metrics": backtest_results["metrics"],
        "metrics_ci": backtest_results.get("metrics_ci", {}),
        "stress": stress_deterministic,
        "stress_test_deterministic": stress_deterministic,
        "stress_test_distributional": stress_distributional,
        "garch_vol_forecast": garch_vol if np.isfinite(garch_vol) else None,
        "garch_vol_series": garch_vol_series,
        "dynamic_var": float(dyn_var) if np.isfinite(dyn_var) else None,
        "monte_carlo_var": mc_var,
        "gev_var": gev_v,
        "gev_cvar": gev_cv,
        "covariance_diagnostics": backtest_results.get("covariance_diagnostics", {}),
        "regime_diagnostics": backtest_results.get("regime_diagnostics"),
        "regime_history": backtest_results.get("regime_history"),
        "method_comparison": method_comparison,
    }

    benchmark_returns = _load_benchmark_returns(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        benchmark_tickers=benchmark_tickers,
        provider_kwargs=provider_kwargs,
    )
    alpha_significance = compute_benchmark_alpha_significance(
        backtest_results["returns"],
        benchmark_returns,
        settings=cfg,
        risk_free_rate=risk_free_rate,
    )
    backtest_results["benchmarks_alpha_significance"] = alpha_significance
    signal_diagnostics = compute_signal_ic_diagnostics(
        feature_df,
        forecast_df=forecast_df_opt,
        settings=cfg,
    )

    # When both optimizers ran, store CVaR metrics for comparison
    if optimizer == "both":
        summary["metrics_cvar"] = backtest_results["metrics_cvar"]

    results = {
        "settings": cfg,
        "data": data,
        "feature_df": feature_df,
        "forecast_df": forecast_df,
        "backtest": backtest_results,
        "summary": summary,
        "signal_diagnostics": signal_diagnostics,
        "benchmarks": {
            "returns": benchmark_returns,
            "tickers": benchmark_returns.columns.tolist() if not benchmark_returns.empty else [],
            "alpha_significance": alpha_significance,
        },
    }

    # Layer 2 hedge mode
    if hedge_mode:
        from .hedge_overlay import run_hedge_backtest
        hedge_results = run_hedge_backtest(
            prices,
            forecast_df,
            universe,
            macro,
            max_leverage=1.3,
            cvar_limit=0.04,
            transaction_cost=0.0010,
            risk_free_rate=risk_free_rate,
            mxn_garch_vol=mxn_garch_vol,
            hedge_mode=hedge_mode_config,
            borrow_cost_bps=150.0,
            leverage_cost_bps=5.0,
        )

        hedge_ret = hedge_results.get("returns")
        if isinstance(hedge_ret, pd.Series) and not hedge_ret.dropna().empty:
            hedge_ret = hedge_ret.replace([np.inf, -np.inf], np.nan).dropna()
            hedge_garch_series = rolling_garch_forecast(hedge_ret, horizon=21, lookback=252, refit_every=5)
            hedge_garch = (
                float(hedge_garch_series.dropna().iloc[-1])
                if not hedge_garch_series.dropna().empty
                else garch_forecast_vol(fit_garch(hedge_ret))
            )
            try:
                _dv = dynamic_var(hedge_ret).dropna()
                hedge_dyn_var = float(_dv.iloc[-1]) if not _dv.empty else float(np.percentile(hedge_ret, 5))
            except Exception:
                hedge_dyn_var = float(np.percentile(hedge_ret, 5))

            try:
                hedge_mc_var = float(monte_carlo_var(hedge_ret))
                if not np.isfinite(hedge_mc_var):
                    hedge_mc_var = float(np.percentile(hedge_ret, 5))
            except Exception:
                hedge_mc_var = float(np.percentile(hedge_ret, 5))

            try:
                hedge_gev_var, hedge_gev_cvar = gev_var(hedge_ret)
                if not np.isfinite(hedge_gev_var):
                    hedge_gev_var = float(np.percentile(hedge_ret, 5))
                if not np.isfinite(hedge_gev_cvar):
                    _tail = hedge_ret[hedge_ret <= hedge_gev_var]
                    hedge_gev_cvar = float(_tail.mean()) if len(_tail) else float(hedge_gev_var)
            except Exception:
                hedge_gev_var = float(np.percentile(hedge_ret, 5))
                _tail = hedge_ret[hedge_ret <= hedge_gev_var]
                hedge_gev_cvar = float(_tail.mean()) if len(_tail) else float(hedge_gev_var)
            
            # Compute stress testing for hedge returns
            hedge_stress = stress_test(
                hedge_results.get("returns"),
                scenario_shocks,
                exposures,
                risk_free_rate=risk_free_rate,
                shock_days=21,
                event_spacing_days=126,
            )
            
            hedge_results["garch_vol_series"] = hedge_garch_series
            hedge_results["garch_vol_forecast"] = hedge_garch if np.isfinite(hedge_garch) else None
            hedge_results["stress"] = hedge_stress
            hedge_results.setdefault("metrics", {})["garch_vol_series"] = hedge_garch_series
            hedge_results.setdefault("metrics", {})["garch_vol_forecast"] = (
                hedge_garch if np.isfinite(hedge_garch) else None
            )
            hedge_results.setdefault("metrics", {})["dynamic_var"] = (
                float(hedge_dyn_var) if np.isfinite(hedge_dyn_var) else None
            )
            hedge_results.setdefault("metrics", {})["monte_carlo_var"] = (
                float(hedge_mc_var) if np.isfinite(hedge_mc_var) else None
            )
            hedge_results.setdefault("metrics", {})["gev_var"] = (
                float(hedge_gev_var) if np.isfinite(hedge_gev_var) else None
            )
            hedge_results.setdefault("metrics", {})["gev_cvar"] = (
                float(hedge_gev_cvar) if np.isfinite(hedge_gev_cvar) else None
            )

        results["hedge_layer"] = hedge_results

    return results


def print_summary(results: dict[str, object], hedge_mode: bool = False) -> None:
    summary = results["summary"]
    print("=== Strategy Pipeline Summary ===")
    print(f"Universe size: {summary['universe_size']}")
    print(f"Backtest period: {summary['start_date'].date()} to {summary['end_date'].date()}")

    if hedge_mode and "hedge_layer" in results:
        # Side-by-side comparison
        print("\n=== Layer 1 vs Layer 2 (Hedge) Metrics ===")
        print(f"{'Metric':<22} {'Layer 1':<15} {'Layer 2':<15}")
        print("-" * 52)

        layer1_metrics = summary["metrics"]
        layer2_metrics = results["hedge_layer"]["metrics"]

        for key in layer1_metrics.keys():
            l1_val = layer1_metrics.get(key, 0.0)
            l2_val = layer2_metrics.get(key, 0.0)
            l1_str = f"{l1_val:>14.4f}" if isinstance(l1_val, (int, float)) and np.isfinite(l1_val) else f"{'N/A':>14}"
            l2_str = f"{l2_val:>14.4f}" if isinstance(l2_val, (int, float)) and np.isfinite(l2_val) else f"{'N/A':>14}"
            print(f"{key:<22} {l1_str} {l2_str}")

        print("\nTail hedge analysis (Layer 2):")
        tail_hedge = results["hedge_layer"]["tail_hedge"]
        print(f"  Unhedged loss at 99%: {tail_hedge['unhedged_loss_at_99']:.4f}")
        print(f"  Hedge payoff: {tail_hedge['hedge_payoff']:.4f}")
        print(f"  Daily cost drag: {tail_hedge['daily_cost_drag']:.6f}")
        print(f"  Net benefit: {tail_hedge['net_benefit']:.4f}")
        print(f"  Recommended: {tail_hedge['recommended']}")
    else:
        print("\nLayer 1 metrics:")
        for key, value in summary["metrics"].items():
            val_str = f"{value:.4f}" if isinstance(value, (int, float)) and np.isfinite(value) else "N/A"
            print(f"{key}: {val_str}")
        print("\nStress test results:")
        print(summary["stress"].to_string(index=False))


def run_etf_pipeline(
    data_source: str = "yahoo",
    start_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    optimizer: str = "mv",
    benchmark_tickers: list[str] | None = None,
    settings: dict | None = None,
    **provider_kwargs,
) -> dict[str, object]:
    """Pipeline for the ETF version of the strategy.

    Uses the 5-ETF universe (EWW, INDS, IGF, ILF, EMLC) with per-ETF
    allocation bounds. Calls the same optimizer, backtest and report
    infrastructure as the main pipeline but skips fundamental data —
    all signals are price-based. Does NOT modify or call run_pipeline().

    Returns the same structure as run_pipeline() so that build_dashboard_html
    and generate_report work unchanged.
    """
    from .data_loader import load_etf_data
    from .features import build_etf_features
    from .risk import detect_macro_regime
    from .alpha_significance import compute_benchmark_alpha_significance
    from .signal_diagnostics import compute_signal_ic_diagnostics

    cfg = resolve_settings(settings)
    data = load_etf_data(source=data_source, start_date=start_date, end_date=end_date, **provider_kwargs)

    universe = data["universe"]
    prices   = data["prices"]
    macro    = data["macro"]

    if prices.empty:
        raise RuntimeError(f"ETF pipeline: no price data returned for source='{data_source}'.")

    feature_df  = build_etf_features(prices, macro, universe)
    scored      = score_cross_section(feature_df)
    log_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    forecast_df = forecast_returns(scored, log_returns, settings=cfg)

    # Per-ETF allocation constraints (min_weight / max_weight from universe)
    ac_map: dict[str, str] = {}
    ac_constraints: dict[str, dict] = {"__asset_class_map__": ac_map}
    for _, row in universe.iterrows():
        t = row["ticker"]
        sleeve = f"etf_{t.lower()}"
        ac_map[t] = sleeve
        ac_constraints[sleeve] = {
            "min": float(row.get("min_weight", 0.0)),
            "max": float(row.get("max_weight", 1.0)),
        }

    optimizable_tickers = universe.loc[universe["investable"], "ticker"].tolist()
    prices_opt   = prices[[c for c in prices.columns if c in optimizable_tickers]]
    forecast_opt = forecast_df[forecast_df["ticker"].isin(optimizable_tickers)].copy()

    backtest_results = run_backtest(
        prices_opt,
        forecast_opt,
        universe,
        risk_free_rate=float(cfg.get("risk_free_rate", 0.04)),
        asset_class_constraints=ac_constraints,
        optimizer=optimizer,
        macro=macro,
        settings=cfg,
    )

    # ---- Stress test (simplified — ETF-appropriate exposures) ----
    exposures = {"banxico_shock": 0.3, "peso_depreciation": 0.5, "us_slowdown": 0.6}
    scenario_shocks = {"banxico_shock": -0.03, "peso_depreciation": -0.05, "us_slowdown": -0.04}
    stress_deterministic = stress_test(
        backtest_results["returns"],
        scenario_shocks,
        exposures,
        risk_free_rate=float(cfg.get("risk_free_rate", 0.04)),
        shock_days=21,
        event_spacing_days=126,
    )

    # ---- Risk metrics ----
    ret = backtest_results["returns"]
    garch_vol_series = rolling_garch_forecast(
        ret,
        horizon=int(cfg["garch_forecast_horizon"]),
        lookback=int(cfg["garch_lookback"]),
        refit_every=int(cfg["garch_refit_every"]),
    )
    garch_vol = (
        float(garch_vol_series.dropna().iloc[-1])
        if not garch_vol_series.dropna().empty
        else garch_forecast_vol(fit_garch(ret))
    )
    dyn_var = dynamic_var(ret).iloc[-1]
    mc_var  = monte_carlo_var(ret)
    gev_v, gev_cv = gev_var(ret)

    regime_history_discrete = compute_macro_regime_history(
        macro, settings={**cfg, "regime_method": "threshold_discrete"},
    )

    summary = {
        "universe_size": len(universe),
        "start_date": prices.index.min(),
        "end_date":   prices.index.max(),
        "optimizer":  optimizer,
        "metrics":    backtest_results["metrics"],
        "metrics_ci": backtest_results.get("metrics_ci", {}),
        "stress":     stress_deterministic,
        "stress_test_deterministic": stress_deterministic,
        "stress_test_distributional": {},
        "garch_vol_forecast": garch_vol if np.isfinite(garch_vol) else None,
        "garch_vol_series":   garch_vol_series,
        "dynamic_var":  float(dyn_var) if np.isfinite(dyn_var) else None,
        "monte_carlo_var": mc_var,
        "gev_var":  gev_v,
        "gev_cvar": gev_cv,
        "covariance_diagnostics": backtest_results.get("covariance_diagnostics", {}),
        "regime_diagnostics":     backtest_results.get("regime_diagnostics"),
        "regime_history":         regime_history_discrete,
        "method_comparison": {},
    }

    benchmark_returns = _load_benchmark_returns(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        benchmark_tickers=benchmark_tickers or [],
        provider_kwargs=provider_kwargs,
    )
    alpha_significance = compute_benchmark_alpha_significance(
        backtest_results["returns"], benchmark_returns, settings=cfg,
        risk_free_rate=float(cfg.get("risk_free_rate", 0.04)),
    )
    backtest_results["benchmarks_alpha_significance"] = alpha_significance
    signal_diagnostics = compute_signal_ic_diagnostics(
        feature_df, forecast_df=forecast_opt, settings=cfg,
    )

    return {
        "settings":           cfg,
        "data":               data,
        "feature_df":         feature_df,
        "forecast_df":        forecast_df,
        "backtest":           backtest_results,
        "summary":            summary,
        "signal_diagnostics": signal_diagnostics,
        "benchmarks": {
            "returns":            benchmark_returns,
            "tickers":            benchmark_returns.columns.tolist() if not benchmark_returns.empty else [],
            "alpha_significance": alpha_significance,
        },
        "mode": "etf",
    }


if __name__ == "__main__":
    results = run_pipeline()
    print_summary(results)
