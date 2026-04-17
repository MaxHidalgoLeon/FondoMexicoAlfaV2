from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.covariance import LedoitWolf

from .portfolio import optimize_portfolio, optimize_portfolio_cvar, optimize_portfolio_robust
from .bootstrap import bootstrap_block_size_selector, bootstrap_metric
from .risk import (
    blend_regime_constraints,
    compute_cvar,
    compute_macro_regime_history,
    max_drawdown,
    compute_sortino,
    compute_sharpe,
    regime_asset_class_constraints,
)
from .settings import resolve_settings

logger = logging.getLogger(__name__)


def get_rebalance_dates(prices: pd.DataFrame, freq: str = "ME") -> pd.DatetimeIndex:
    """Return the last available trading date in each rebalancing period."""
    resampled = prices.resample(freq).last().dropna(how="all")
    # Snap to the nearest actual trading day that exists in prices.index
    snapped = []
    for dt in resampled.index:
        # Find the last trading day on or before dt
        candidates = prices.index[prices.index <= dt]
        if len(candidates):
            snapped.append(candidates[-1])
    return pd.DatetimeIndex(sorted(set(snapped)))


def _annualized_return_from_log_returns(returns: pd.Series) -> float:
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return 0.0
    return float(np.exp(clean.mean() * 252.0) - 1.0)


def _identity_scaled_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    n = len(returns.columns)
    avg_var = returns.var().mean() if not returns.empty else 1e-4
    return pd.DataFrame(np.eye(n) * avg_var, index=returns.columns, columns=returns.columns)


def _rolling_ledoit_wolf_covariance(
    returns: pd.DataFrame,
    date: pd.Timestamp,
    window: int,
) -> pd.DataFrame:
    subset = returns.loc[:date].tail(window).fillna(0.0)
    if subset.shape[0] < 10:
        return _identity_scaled_covariance(returns)
    try:
        lw = LedoitWolf()
        lw.fit(subset)
        cov_array = lw.covariance_
        return pd.DataFrame(cov_array, index=subset.columns, columns=subset.columns).reindex(
            index=returns.columns, columns=returns.columns
        ).fillna(0.0)
    except Exception:
        return subset.cov().reindex(
            index=returns.columns, columns=returns.columns
        ).fillna(0.0)


def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    arr = cov.values.astype(float)
    std = np.sqrt(np.clip(np.diag(arr), 0.0, None))
    denom = np.outer(std, std) + 1e-12
    corr = arr / denom
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def _ewma_ledoit_wolf_covariance(
    returns: pd.DataFrame,
    date: pd.Timestamp,
    settings: dict[str, Any],
) -> pd.DataFrame | None:
    subset = returns.loc[:date].tail(int(settings["ewma_window_cov"])).fillna(0.0)
    min_periods = int(settings["ewma_min_periods_cov"])
    if subset.shape[0] < min_periods:
        return None

    alpha = 1.0 - float(settings["ewma_lambda_cov"])
    try:
        ewm_cov = subset.ewm(alpha=alpha, min_periods=min_periods, adjust=False).cov()
        last_ts = ewm_cov.index.get_level_values(0).max()
        ewma_cov = ewm_cov.xs(last_ts, level=0).reindex(
            index=subset.columns,
            columns=subset.columns,
        ).fillna(0.0)

        lw = LedoitWolf().fit(subset.values)
        shrinkage = float(getattr(lw, "shrinkage_", 0.0))
        mu = float(np.trace(ewma_cov.values) / max(len(ewma_cov), 1))
        target = np.eye(len(ewma_cov)) * mu
        shrunk = (1.0 - shrinkage) * ewma_cov.values + shrinkage * target
        return pd.DataFrame(shrunk, index=ewma_cov.index, columns=ewma_cov.columns)
    except Exception:
        return None


def build_covariance_matrix(
    returns: pd.DataFrame,
    date: pd.Timestamp,
    window: int = 63,
    settings: dict | None = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Hybrid EWMA -> Ledoit-Wolf covariance with PSD fallback to rolling LW."""
    cfg = resolve_settings(settings)
    rolling_cov = _rolling_ledoit_wolf_covariance(returns, date, window)
    requested_method = str(cfg["covariance_method"]).lower().strip()
    diagnostics: dict[str, Any] = {
        "date": pd.Timestamp(date),
        "method_requested": requested_method,
        "method_used": "rolling_ledoit_wolf",
        "fallback_to_rolling": False,
        "condition_ratio": np.nan,
        "det_ratio": np.nan,
        "rolling_corr": _cov_to_corr(rolling_cov),
        "ewma_corr": None,
    }

    if requested_method == "rolling_ledoit_wolf":
        if return_diagnostics:
            return rolling_cov, diagnostics
        return rolling_cov

    ewma_cov = _ewma_ledoit_wolf_covariance(returns, date, cfg)
    if ewma_cov is None:
        diagnostics["fallback_to_rolling"] = True
        if return_diagnostics:
            return rolling_cov, diagnostics
        return rolling_cov

    diagnostics["ewma_corr"] = _cov_to_corr(ewma_cov)
    tol = float(cfg["covariance_psd_tolerance"])
    eigvals = np.linalg.eigvalsh(ewma_cov.values)
    if np.any(eigvals < -tol):
        logger.error(
            "EWMA covariance is not PSD on %s; falling back to rolling Ledoit-Wolf.",
            pd.Timestamp(date).date(),
        )
        diagnostics["fallback_to_rolling"] = True
        if return_diagnostics:
            return rolling_cov, diagnostics
        return rolling_cov

    try:
        cond_ewma = float(np.linalg.cond(ewma_cov.values + np.eye(len(ewma_cov)) * 1e-12))
        cond_rolling = float(np.linalg.cond(rolling_cov.values + np.eye(len(rolling_cov)) * 1e-12))
        diagnostics["condition_ratio"] = cond_ewma / (cond_rolling + 1e-12)

        sign_ewma, logdet_ewma = np.linalg.slogdet(ewma_cov.values + np.eye(len(ewma_cov)) * 1e-12)
        sign_roll, logdet_roll = np.linalg.slogdet(rolling_cov.values + np.eye(len(rolling_cov)) * 1e-12)
        if sign_ewma > 0 and sign_roll > 0:
            diagnostics["det_ratio"] = float(np.exp(logdet_ewma - logdet_roll))
        logger.debug(
            "Rebalance %s: cond(EWMA)/cond(rolling)=%.4f",
            pd.Timestamp(date).date(),
            diagnostics["condition_ratio"],
        )
    except Exception:
        pass

    diagnostics["method_used"] = "ewma_ledoit_wolf"
    if return_diagnostics:
        return ewma_cov, diagnostics
    return ewma_cov

def run_backtest(
    prices: pd.DataFrame,
    signal_df: pd.DataFrame,
    universe: pd.DataFrame,
    transaction_cost: float = 0.001,
    rebalance_freq: str = "ME",
    risk_free_rate: float = 0.02,
    asset_class_constraints: Optional[Dict] = None,
    optimizer: str = "mv",
    adtv_scores: Optional[pd.Series] = None,
    macro: Optional[pd.DataFrame] = None,
    issuer_consolidated_limits: Optional[Dict[str, list]] = None,
    max_position_overrides: Optional[Dict[str, float]] = None,
    settings: dict | None = None,
) -> Dict[str, pd.DataFrame]:
    """Run backtest.

    optimizer: "mv"      — mean-variance (SLSQP) only
               "cvar"   — mean-CVaR only
               "robust" — Michaud resampled MV only
               "both"   — MV + CVaR comparison; standard keys hold MV
                          results, extra keys (*_cvar) hold CVaR results.

    adtv_scores : per-ticker ADTV liquidity scores (0–1) forwarded to the
                  optimizer as market-impact weights.
    macro       : macro DataFrame used by detect_macro_regime() at each
                  rebalance date to override static asset-class constraints.
    """
    cfg = resolve_settings(settings)
    price_ratio = prices / prices.shift(1)
    returns = np.log(price_ratio).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rebalance_dates = get_rebalance_dates(prices, rebalance_freq)
    if not signal_df.empty and "date" in signal_df.columns:
        first_signal_date = pd.Timestamp(signal_df["date"].min())
        rebalance_dates = rebalance_dates[rebalance_dates >= first_signal_date]

    run_mv     = optimizer in ("mv",     "both")
    run_cvar   = optimizer in ("cvar",   "both")
    run_robust = optimizer == "robust"

    # Prepare macro data for regime detection (snap 'date' column to index once)
    _macro_indexed: Optional[pd.DataFrame] = None
    if macro is not None:
        _macro_indexed = macro.copy()
        if "date" in _macro_indexed.columns:
            _macro_indexed = _macro_indexed.set_index("date")
        _macro_indexed = _macro_indexed.sort_index()
    regime_history = compute_macro_regime_history(_macro_indexed, settings=cfg) if _macro_indexed is not None else pd.DataFrame()
    regime_records: list[dict[str, Any]] = []
    covariance_records: list[dict[str, Any]] = []
    prev_effective_regime: str | None = None
    transition_state: dict[str, Any] | None = None

    # MV tracking — only allocate when needed
    if run_mv:
        weights_mv    = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev_w_mv     = pd.Series(0.0, index=prices.columns)
        turnover_mv   = pd.Series(0.0, index=prices.index)

    # CVaR tracking — only allocate when needed
    if run_cvar:
        weights_cvar  = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev_w_cvar   = pd.Series(0.0, index=prices.columns)
        turnover_cvar = pd.Series(0.0, index=prices.index)

    # Robust (Michaud) tracking
    if run_robust:
        weights_rob   = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev_w_rob    = pd.Series(0.0, index=prices.columns)
        turnover_rob  = pd.Series(0.0, index=prices.index)

    for date in rebalance_dates:
        if date not in prices.index:
            continue
        date_signal = signal_df[signal_df["date"] == date]
        if date_signal.empty:
            continue
        expected_returns = date_signal.set_index("ticker")["expected_return"].reindex(prices.columns).fillna(0.0)
        cov_matrix, covariance_diag = build_covariance_matrix(
            returns,
            date,
            window=int(cfg["rolling_cov_window"]),
            settings=cfg,
            return_diagnostics=True,
        )
        covariance_records.append(covariance_diag)

        # --- Regime-based constraint override ---
        effective_constraints = asset_class_constraints
        if not regime_history.empty and asset_class_constraints is not None:
            regime_slice = regime_history.loc[regime_history.index <= date]
            if not regime_slice.empty:
                regime_row = regime_slice.iloc[-1]
                regime = str(regime_row["regime"])
                regime_confidence = float(regime_row.get("regime_confidence", np.nan))
                regime_score = float(regime_row.get("regime_score_smoothed", np.nan))
                effective_label = regime
                blend_alpha = np.nan

                if prev_effective_regime is None:
                    prev_effective_regime = regime

                regime_bounds = regime_asset_class_constraints(regime)
                min_conf = float(cfg["regime_min_confidence_for_switch"])
                if (
                    regime != prev_effective_regime
                    and np.isfinite(regime_confidence)
                    and regime_confidence < min_conf
                ):
                    if transition_state is None or transition_state.get("to") != regime:
                        transition_state = {
                            "from": prev_effective_regime,
                            "to": regime,
                            "step": 1,
                            "total_steps": 2,
                        }
                    blend_alpha = float(transition_state["step"] / transition_state["total_steps"])
                    regime_bounds = blend_regime_constraints(
                        regime_asset_class_constraints(transition_state["from"]),
                        regime_asset_class_constraints(transition_state["to"]),
                        blend_alpha,
                    )
                    effective_label = f'{transition_state["from"]}->{transition_state["to"]}'
                    if transition_state["step"] >= transition_state["total_steps"]:
                        prev_effective_regime = regime
                        transition_state = None
                    else:
                        transition_state["step"] += 1
                else:
                    transition_state = None
                    prev_effective_regime = regime

                # Merge regime bounds into constraints (preserve __asset_class_map__)
                effective_constraints = dict(asset_class_constraints)
                for ac, bounds in regime_bounds.items():
                    if ac in effective_constraints:
                        effective_constraints[ac] = bounds
                logger.debug(
                    "Rebalance %s: macro regime=%s confidence=%.3f score=%.3f",
                    date.date(),
                    effective_label,
                    regime_confidence if np.isfinite(regime_confidence) else np.nan,
                    regime_score if np.isfinite(regime_score) else np.nan,
                )
                regime_records.append(
                    {
                        "date": pd.Timestamp(date),
                        "regime": regime,
                        "effective_regime": effective_label,
                        "regime_confidence": regime_confidence,
                        "regime_score_smoothed": regime_score,
                        "transition_alpha": blend_alpha,
                    }
                )

        if run_mv:
            try:
                target_mv = optimize_portfolio(
                    expected_returns, cov_matrix,
                    prev_weights=prev_w_mv,
                    max_position=0.10, min_position=0.0,
                    target_net_exposure=0.9, risk_aversion=4.0,
                    turnover_penalty=0.05,
                    asset_class_constraints=effective_constraints,
                    adtv_scores=adtv_scores,
                    issuer_consolidated_limits=issuer_consolidated_limits,
                    max_position_overrides=max_position_overrides,
                )
            except RuntimeError as exc:
                logger.warning("MV optimization failed on %s: %s — carrying previous weights.", date, exc)
                target_mv = prev_w_mv
            weights_mv.loc[date:, :] = target_mv.values
            turnover_mv.loc[date] = np.sum(np.abs(target_mv - prev_w_mv))
            prev_w_mv = target_mv

        if run_cvar:
            # Use a longer scenario window and stricter tail confidence so CVaR
            # optimizer is meaningfully different from the MV solution.
            scen = returns.loc[:date].tail(252)
            try:
                target_cvar = optimize_portfolio_cvar(
                    expected_returns, scen,
                    prev_weights=prev_w_cvar,
                    max_position=0.10, min_position=0.0,
                    target_net_exposure=0.75, risk_aversion=25.0,
                    turnover_penalty=0.01, alpha=0.99,
                    asset_class_constraints=effective_constraints,
                    adtv_scores=adtv_scores,
                    issuer_consolidated_limits=issuer_consolidated_limits,
                    max_position_overrides=max_position_overrides,
                )
            except RuntimeError as exc:
                logger.warning("CVaR optimization failed on %s: %s — carrying previous weights.", date, exc)
                target_cvar = prev_w_cvar
            weights_cvar.loc[date:, :] = target_cvar.values
            turnover_cvar.loc[date] = np.sum(np.abs(target_cvar - prev_w_cvar))
            prev_w_cvar = target_cvar

        if run_robust:
            try:
                target_rob = optimize_portfolio_robust(
                    expected_returns, cov_matrix,
                    prev_weights=prev_w_rob,
                    max_position=0.10, min_position=0.0,
                    target_net_exposure=0.9, risk_aversion=4.0,
                    turnover_penalty=0.05,
                    asset_class_constraints=effective_constraints,
                    adtv_scores=adtv_scores,
                    issuer_consolidated_limits=issuer_consolidated_limits,
                    max_position_overrides=max_position_overrides,
                )
            except RuntimeError as exc:
                logger.warning("Robust optimization failed on %s: %s — carrying previous weights.", date, exc)
                target_rob = prev_w_rob
            weights_rob.loc[date:, :] = target_rob.values
            turnover_rob.loc[date] = np.sum(np.abs(target_rob - prev_w_rob))
            prev_w_rob = target_rob

    def _compute_returns_and_metrics(w: pd.DataFrame, tv: pd.Series) -> tuple:
        port_ret = (w.shift(1) * returns).sum(axis=1)
        port_ret = port_ret - tv * transaction_cost
        ann_ret = _annualized_return_from_log_returns(port_ret)
        mdd = max_drawdown(port_ret)
        m = {
            "sharpe": compute_sharpe(port_ret, risk_free_rate=risk_free_rate),
            "sortino": compute_sortino(port_ret, required_return=risk_free_rate / 252),
            "max_drawdown": mdd,
            "calmar": ann_ret / max(abs(mdd), 1e-9),
            "cvar_95": compute_cvar(port_ret, alpha=0.95),
            "annualized_return": ann_ret,
            "annualized_vol": port_ret.std() * np.sqrt(252),
            "turnover": tv.mean(),
        }
        metrics_ci: dict[str, dict[str, Any]] = {}
        if cfg["bootstrap_enabled"]:
            block_size = (
                bootstrap_block_size_selector(port_ret)
                if cfg["bootstrap_block_size"] == "auto"
                else int(cfg["bootstrap_block_size"])
            )
            common_kwargs = {
                "block_size": int(block_size),
                "n_reps": int(cfg["bootstrap_n_reps"]),
                "confidence": float(cfg["bootstrap_confidence"]),
                "seed": int(cfg["bootstrap_seed"]),
            }
            metric_fns = {
                "sharpe": lambda s: compute_sharpe(pd.Series(s), risk_free_rate=risk_free_rate),
                "sortino": lambda s: compute_sortino(pd.Series(s), required_return=risk_free_rate / 252),
                "cvar_95": lambda s: compute_cvar(pd.Series(s), alpha=0.95),
                "max_drawdown": lambda s: max_drawdown(pd.Series(s)),
                "cagr": lambda s: _annualized_return_from_log_returns(pd.Series(s)),
            }
            metrics_ci = {
                name: bootstrap_metric(port_ret, fn, **common_kwargs)
                for name, fn in metric_fns.items()
            }
        return port_ret, m, metrics_ci

    # Build output — standard keys always hold the "primary" optimizer's results
    if run_mv:
        portfolio_returns, metrics, metrics_ci = _compute_returns_and_metrics(weights_mv, turnover_mv)
        out = {
            "weights": weights_mv,
            "returns": portfolio_returns,
            "metrics": metrics,
            "turnover": turnover_mv,
        }
    elif run_robust:
        portfolio_returns, metrics, metrics_ci = _compute_returns_and_metrics(weights_rob, turnover_rob)
        out = {
            "weights": weights_rob,
            "returns": portfolio_returns,
            "metrics": metrics,
            "turnover": turnover_rob,
        }
    else:
        portfolio_returns, metrics, metrics_ci = _compute_returns_and_metrics(weights_cvar, turnover_cvar)
        out = {
            "weights": weights_cvar,
            "returns": portfolio_returns,
            "metrics": metrics,
            "turnover": turnover_cvar,
        }
    if metrics_ci:
        out["metrics_ci"] = metrics_ci

    # When "both", attach CVaR results under _cvar keys for comparison
    if optimizer == "both":
        ret_cvar, met_cvar, metrics_ci_cvar = _compute_returns_and_metrics(weights_cvar, turnover_cvar)
        out["weights_cvar"]  = weights_cvar
        out["returns_cvar"]  = ret_cvar
        out["metrics_cvar"]  = met_cvar
        out["turnover_cvar"] = turnover_cvar
        if metrics_ci_cvar:
            out["metrics_ci_cvar"] = metrics_ci_cvar

    cov_diag_df = pd.DataFrame(
        [
            {
                "date": rec["date"],
                "method_used": rec["method_used"],
                "fallback_to_rolling": rec["fallback_to_rolling"],
                "condition_ratio": rec["condition_ratio"],
                "det_ratio": rec["det_ratio"],
            }
            for rec in covariance_records
        ]
    )
    if not cov_diag_df.empty:
        cov_diag_df = cov_diag_df.set_index("date").sort_index()
        latest_cov = covariance_records[-1]
        out["covariance_diagnostics"] = {
            "series": cov_diag_df,
            "condition_ratio": cov_diag_df["condition_ratio"],
            "det_ratio": cov_diag_df["det_ratio"],
            "rolling_corr": latest_cov.get("rolling_corr"),
            "ewma_corr": latest_cov.get("ewma_corr"),
        }

    out["regime_history"] = regime_history
    if regime_records:
        out["regime_diagnostics"] = pd.DataFrame(regime_records).set_index("date").sort_index()

    return out
