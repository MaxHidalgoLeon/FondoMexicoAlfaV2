from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .backtest import run_backtest
from .features import build_signal_matrix
from .signals import score_cross_section, forecast_returns
from .portfolio import black_litterman, apply_fx_overlay
from .risk import stress_test, fit_garch, garch_forecast_vol, dynamic_var, monte_carlo_var, gev_var

logger = logging.getLogger(__name__)


def run_pipeline(
    hedge_mode: bool = False,
    data_source: str = "mock",
    start_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    min_liquidity_score: float = 0.47,
    optimizer: str = "mv",
    **provider_kwargs,
) -> dict[str, object]:
    from .data_loader import load_data, compute_adtv_liquidity_scores
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
    if data_source != "mock":
        try:
            from .data_providers import get_provider
            provider = get_provider(data_source, **provider_kwargs)
            equity_tickers = universe.loc[
                universe["asset_class"].isin(["equity", "fibra"]), "ticker"
            ].tolist()
            volume = provider.get_volume(equity_tickers, start_date, end_date)
            adtv_scores = compute_adtv_liquidity_scores(prices, volume)
            # Update universe with real ADTV-based scores; bonds keep their 1.0
            universe = universe.copy()
            universe.loc[universe["ticker"].isin(adtv_scores.index), "liquidity_score"] = (
                universe.loc[universe["ticker"].isin(adtv_scores.index), "ticker"]
                .map(adtv_scores)
            )
            logger.info("ADTV liquidity scores updated from %s for %d tickers.", data_source, len(adtv_scores))
        except Exception as exc:
            logger.warning("ADTV score update failed (%s) — using hardcoded scores.", exc)

    # ------------------------------------------------------------------
    # Liquidity filter — remove tickers below min_liquidity_score.
    # Fixed-income tickers always have score 1.0 and are never removed.
    # ------------------------------------------------------------------
    illiquid = universe.loc[universe["liquidity_score"] < min_liquidity_score, "ticker"].tolist()
    if illiquid:
        logger.info(
            "Liquidity filter (<%.2f) removing %d ticker(s): %s",
            min_liquidity_score, len(illiquid), illiquid,
        )
        universe = universe[universe["liquidity_score"] >= min_liquidity_score].reset_index(drop=True)
        keep = set(universe["ticker"])
        prices = prices[[c for c in prices.columns if c in keep]]
        # Sync data dict so the report sees the post-filter universe
        data["universe"] = universe

    feature_df = build_signal_matrix(prices, fundamentals, fibra_fundamentals, bonds, macro, universe)
    scored = score_cross_section(feature_df)
    forecast_df = forecast_returns(scored, prices.pct_change().fillna(0.0))

    # Black-Litterman
    forecast_tickers = forecast_df["ticker"].unique() if not forecast_df.empty else []
    market_weights = universe.set_index("ticker")["market_cap_mxn"] / (universe["market_cap_mxn"].sum() + 1e-9)
    market_weights = market_weights.reindex(forecast_tickers).fillna(0.0)

    # Warn about any ticker mismatch between forecast and universe
    missing_from_universe = set(forecast_tickers) - set(universe["ticker"])
    if missing_from_universe:
        logger.warning("Tickers in forecast but not in universe (will get 0 market weight): %s", missing_from_universe)

    # Build covariance with Ledoit-Wolf shrinkage
    raw_returns = prices.pct_change().dropna(how="all")
    cov_tickers = [t for t in forecast_tickers if t in raw_returns.columns]
    if cov_tickers:
        lw = LedoitWolf()
        lw.fit(raw_returns[cov_tickers].fillna(0.0))
        cov_matrix = pd.DataFrame(lw.covariance_, index=cov_tickers, columns=cov_tickers)
    else:
        cov_matrix = pd.DataFrame(dtype=float)

    views = forecast_df.groupby("ticker")["expected_return"].mean().to_dict()
    # Data-driven view confidences: scale |expected_return| to [0.30, 0.70]
    abs_views = pd.Series(views).abs()
    view_range = abs_views.max() - abs_views.min()
    if view_range > 1e-9:
        view_confidences = (0.30 + 0.40 * (abs_views - abs_views.min()) / view_range).to_dict()
    else:
        view_confidences = {t: 0.50 for t in views}

    # Align market_weights to the covariance matrix's ticker set before BL
    market_weights = market_weights.reindex(cov_matrix.columns).fillna(0.0)

    bl_returns = black_litterman(market_weights, cov_matrix, views, view_confidences)

    # Risk-free rate: último valor de Banxico disponible en macro
    banxico_series = macro["banxico_rate"].dropna()
    risk_free_rate = float(banxico_series.iloc[-1]) if not banxico_series.empty else 0.02
    logger.info("Risk-free rate (Banxico): %.4f", risk_free_rate)

    # GARCH sobre retornos de USD/MXN para proyectar vol y drift
    usdmxn_returns = macro["usd_mxn"].pct_change().dropna()
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
    adjusted_returns = apply_fx_overlay(bl_returns, usd_exposure, macro["usd_mxn"].iloc[-1], expected_usdmxn_return)

    # Merge BL-adjusted expected returns back into forecast_df before backtesting
    if not adjusted_returns.empty and not forecast_df.empty:
        bl_lookup = adjusted_returns.rename("bl_expected_return")
        forecast_df = forecast_df.copy()
        forecast_df["expected_return"] = forecast_df["ticker"].map(bl_lookup).fillna(
            forecast_df["expected_return"]
        )
    # Asset-class allocation constraints: equity 50–90 %, FIBRA 5–25 %, bonds 0–15 %.
    # These provide the static baseline; detect_macro_regime() dynamically adjusts
    # the bounds at each rebalance date inside run_backtest().
    ac_map = universe.set_index("ticker")["asset_class"].to_dict()
    asset_class_constraints = {
        "__asset_class_map__": ac_map,
        "equity":       {"min": 0.50, "max": 0.90},
        "fibra":        {"min": 0.05, "max": 0.25},
        "fixed_income": {"min": 0.00, "max": 0.15},
    }

    # ADTV proxy: use universe liquidity_score as the per-ticker score vector
    adtv_scores = universe.set_index("ticker")["liquidity_score"].astype(float)

    backtest_results = run_backtest(
        prices, forecast_df, universe,
        risk_free_rate=risk_free_rate,
        asset_class_constraints=asset_class_constraints,
        optimizer=optimizer,
        adtv_scores=adtv_scores,
        macro=macro,
    )

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
    stress = stress_test(backtest_results["returns"], scenario_shocks, exposures, risk_free_rate=risk_free_rate)

    # Additional risk metrics
    returns = backtest_results["returns"]
    # Per-ticker daily returns aligned to the backtest period (for multivariate MC)
    final_weights = backtest_results["weights"].loc[
        backtest_results["weights"].abs().sum(axis=1) > 1e-9
    ].iloc[-1] if not backtest_results["weights"].empty else None
    raw_daily_returns = prices.pct_change().dropna(how="all")
    garch_vol = garch_forecast_vol(fit_garch(returns))
    dyn_var = dynamic_var(returns).iloc[-1]
    mc_var = monte_carlo_var(returns, asset_returns=raw_daily_returns, weights=final_weights)
    gev_v, gev_cv = gev_var(returns)

    summary = {
        "universe_size": len(universe),
        "start_date": prices.index.min(),
        "end_date": prices.index.max(),
        "optimizer": optimizer,
        "metrics": backtest_results["metrics"],
        "stress": stress,
        "garch_vol_forecast": garch_vol if np.isfinite(garch_vol) else None,
        "dynamic_var": float(dyn_var) if np.isfinite(dyn_var) else None,
        "monte_carlo_var": mc_var,
        "gev_var": gev_v,
        "gev_cvar": gev_cv,
    }

    # When both optimizers ran, store CVaR metrics for comparison
    if optimizer == "both":
        summary["metrics_cvar"] = backtest_results["metrics_cvar"]

    results = {
        "data": data,
        "feature_df": feature_df,
        "forecast_df": forecast_df,
        "backtest": backtest_results,
        "summary": summary,
    }

    # Layer 2 hedge mode
    if hedge_mode:
        from .hedge_overlay import run_hedge_backtest
        hedge_results = run_hedge_backtest(
            prices,
            forecast_df,
            universe,
            macro,
            max_leverage=1.5,
            cvar_limit=0.02,
            transaction_cost=0.0015,
            risk_free_rate=risk_free_rate,
            mxn_garch_vol=mxn_garch_vol,
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


if __name__ == "__main__":
    results = run_pipeline()
    print_summary(results)
