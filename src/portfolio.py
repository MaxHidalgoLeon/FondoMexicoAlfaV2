from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def _portfolio_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
    turnover_penalty: float,
    prev_weights: np.ndarray,
    market_impact: np.ndarray | None = None,
) -> float:
    expected_cost = -weights.dot(expected_returns)
    variance = risk_aversion * weights.dot(cov_matrix).dot(weights)
    # Scale turnover penalty by expected return magnitude so it doesn't dominate
    er_scale = max(np.abs(expected_returns).mean(), 1e-4)
    delta = np.abs(weights - prev_weights)
    turnover = turnover_penalty * er_scale * delta.sum()
    # Market-impact cost: η_i * σ_i * (Δw_i / ADTV_i)  — Almgren-Chriss linear term
    # market_impact[i] = η * σ_i / ADTV_i, pre-computed outside the loop
    impact = 0.0
    if market_impact is not None:
        impact = float((market_impact * delta).sum())
    return expected_cost + variance + turnover + impact


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    max_position: float = 0.15,
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.1,
    asset_class_constraints: Optional[Dict[str, Dict[str, float]]] = None,
    adtv_scores: Optional[pd.Series] = None,
    market_impact_eta: float = 0.1,
) -> pd.Series:
    """Mean-variance optimizer (SLSQP).

    adtv_scores: Series indexed by ticker with normalized ADTV scores in [0, 1].
        When provided, a market-impact penalty η·σ_i/ADTV_i·|Δw_i| is added to
        the objective (Almgren-Chriss linear model).  Tickers absent from
        adtv_scores receive ADTV = 0.5 (neutral).
    market_impact_eta: scaling coefficient η for the market-impact term.
    """
    tickers = expected_returns.index.tolist()
    if prev_weights is None:
        prev_weights = pd.Series(0.0, index=tickers)

    # Build per-ticker market-impact vector: η * σ_i / ADTV_i
    vol_diag = np.sqrt(np.diag(cov_matrix.reindex(index=tickers, columns=tickers).fillna(0.0).values))
    if adtv_scores is not None:
        adtv = adtv_scores.reindex(tickers).fillna(0.5).clip(lower=0.05).values
    else:
        adtv = np.full(len(tickers), 0.5)
    market_impact = market_impact_eta * vol_diag / adtv

    x0 = np.repeat(target_net_exposure / len(tickers), len(tickers))
    bounds = [(min_position, max_position)] * len(tickers)
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - target_net_exposure,
        }
    ]

    # Asset-class group constraints
    if asset_class_constraints:
        asset_class_constraints = dict(asset_class_constraints)  # defensive copy
        ac_map: Dict[str, str] = asset_class_constraints.get("__asset_class_map__", {})
        asset_class_constraints = {k: v for k, v in asset_class_constraints.items() if k != "__asset_class_map__"}
        if ac_map:
            for ac, bounds_dict in asset_class_constraints.items():
                ac_tickers = [t for t in tickers if ac_map.get(t) == ac]
                if not ac_tickers:
                    continue
                idx = [tickers.index(t) for t in ac_tickers]
                ac_min = bounds_dict.get("min", 0.0)
                ac_max = bounds_dict.get("max", 1.0)
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mn=ac_min: np.sum(x[i]) - mn}
                )
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mx=ac_max: mx - np.sum(x[i])}
                )

    result = minimize(
        _portfolio_objective,
        x0,
        args=(expected_returns.values, cov_matrix.values, risk_aversion, turnover_penalty,
              prev_weights.values, market_impact),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        logger.warning("Portfolio optimization did not converge: %s", result.message)
        if np.any(np.isnan(result.x)):
            raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return pd.Series(result.x, index=tickers)


def optimize_portfolio_cvar(
    expected_returns: pd.Series,
    scenario_returns: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    max_position: float = 0.15,
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.1,
    alpha: float = 0.95,
    asset_class_constraints: Optional[Dict] = None,
    adtv_scores: Optional[pd.Series] = None,
    market_impact_eta: float = 0.1,
) -> pd.Series:
    """Mean-CVaR portfolio optimization.

    Objective: minimize  -w'μ  +  λ·|CVaR(α)|  +  κ·turnover  +  market_impact
    Uses historical scenario_returns (T × N) for CVaR estimation.
    Falls back to equal-weight if there are not enough scenarios.

    adtv_scores / market_impact_eta: same semantics as optimize_portfolio().
    """
    tickers = expected_returns.index.tolist()
    if prev_weights is None:
        prev_weights = pd.Series(0.0, index=tickers)

    common = [t for t in tickers if t in scenario_returns.columns]
    if len(common) < 2 or len(scenario_returns) < 20:
        w = np.repeat(target_net_exposure / len(tickers), len(tickers))
        return pd.Series(np.clip(w, min_position, max_position), index=tickers)

    scen = scenario_returns[common].fillna(0.0).values  # T × N_common
    common_idx = [tickers.index(t) for t in common]
    er_scale = max(float(np.abs(expected_returns.values).mean()), 1e-4)

    # Per-ticker vol proxy from scenario returns for market-impact term
    scen_vol = np.std(scenario_returns[common].fillna(0.0).values, axis=0, ddof=1)  # N_common
    if adtv_scores is not None:
        adtv_common = adtv_scores.reindex(common).fillna(0.5).clip(lower=0.05).values
    else:
        adtv_common = np.full(len(common), 0.5)
    market_impact_common = market_impact_eta * scen_vol / adtv_common

    def _objective(w: np.ndarray) -> float:
        port_scen = scen @ w[common_idx]
        var_level = np.percentile(port_scen, (1 - alpha) * 100)
        tail = port_scen[port_scen <= var_level]
        cvar = float(tail.mean()) if len(tail) > 0 else float(var_level)
        return_term = -float(w @ expected_returns.values)
        cvar_term = risk_aversion * abs(cvar)
        delta = np.abs(w[common_idx] - prev_weights.values[common_idx])
        turnover_term = turnover_penalty * er_scale * float(np.sum(np.abs(w - prev_weights.values)))
        impact_term = float((market_impact_common * delta).sum())
        return return_term + cvar_term + turnover_term + impact_term

    x0 = np.repeat(target_net_exposure / len(tickers), len(tickers))
    bounds = [(min_position, max_position)] * len(tickers)
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - target_net_exposure}]

    if asset_class_constraints:
        asset_class_constraints = dict(asset_class_constraints)
        ac_map: Dict[str, str] = asset_class_constraints.get("__asset_class_map__", {})
        asset_class_constraints = {k: v for k, v in asset_class_constraints.items() if k != "__asset_class_map__"}
        if ac_map:
            for ac, bounds_dict in asset_class_constraints.items():
                ac_tickers = [t for t in tickers if ac_map.get(t) == ac]
                if not ac_tickers:
                    continue
                idx = [tickers.index(t) for t in ac_tickers]
                ac_min = bounds_dict.get("min", 0.0)
                ac_max = bounds_dict.get("max", 1.0)
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mn=ac_min: np.sum(x[i]) - mn}
                )
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mx=ac_max: mx - np.sum(x[i])}
                )

    result = minimize(
        _objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        logger.warning("CVaR optimization did not converge: %s", result.message)
        if np.any(np.isnan(result.x)):
            raise RuntimeError(f"CVaR optimization failed: {result.message}")
    return pd.Series(result.x, index=tickers)


def optimize_portfolio_robust(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    n_simulations: int = 100,
    max_position: float = 0.15,
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.1,
    asset_class_constraints: Optional[Dict] = None,
    adtv_scores: Optional[pd.Series] = None,
    market_impact_eta: float = 0.1,
    random_seed: int = 42,
) -> pd.Series:
    """Michaud Resampled Efficiency optimizer.

    Runs the MV optimizer ``n_simulations`` times, each time with μ perturbed
    by a draw from its estimation-error distribution:

        μ_k  ~  N(μ̂,  Σ / T_effective)

    where T_effective is a conservative 252 trading days (1 year of data).
    The final portfolio is the average of all simulated optimal weights,
    which yields a smoother, more diversified allocation that is robust to
    errors in the expected return estimates.

    All other parameters are forwarded to optimize_portfolio().
    """
    rng = np.random.default_rng(random_seed)
    tickers = expected_returns.index.tolist()
    n = len(tickers)

    cov_aligned = cov_matrix.reindex(index=tickers, columns=tickers).fillna(0.0)
    cov_arr = cov_aligned.values

    # Estimation-error covariance: Σ / T_effective
    T_effective = 252
    mu_cov = cov_arr / T_effective

    # Cholesky for efficient sampling (fallback to SVD if not PD)
    try:
        L = np.linalg.cholesky(mu_cov + np.eye(n) * 1e-8)
        def _sample_mu() -> np.ndarray:
            return expected_returns.values + L @ rng.standard_normal(n)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(mu_cov)
        eigvals = np.maximum(eigvals, 0.0)
        sqrt_cov = eigvecs * np.sqrt(eigvals)
        def _sample_mu() -> np.ndarray:
            return expected_returns.values + sqrt_cov @ rng.standard_normal(n)

    accumulated = np.zeros(n)
    successes = 0
    for _ in range(n_simulations):
        mu_sim = pd.Series(_sample_mu(), index=tickers)
        try:
            w_sim = optimize_portfolio(
                mu_sim, cov_aligned,
                prev_weights=prev_weights,
                max_position=max_position,
                min_position=min_position,
                target_net_exposure=target_net_exposure,
                risk_aversion=risk_aversion,
                turnover_penalty=turnover_penalty,
                asset_class_constraints=asset_class_constraints,
                adtv_scores=adtv_scores,
                market_impact_eta=market_impact_eta,
            )
            accumulated += w_sim.values
            successes += 1
        except RuntimeError:
            continue

    if successes == 0:
        logger.warning("Robust optimizer: all simulations failed — returning equal weight.")
        w = np.repeat(target_net_exposure / n, n)
        return pd.Series(w, index=tickers)

    avg_weights = accumulated / successes
    # Re-scale to ensure the sum constraint is exactly met after averaging
    w_sum = avg_weights.sum()
    if abs(w_sum) > 1e-9:
        avg_weights = avg_weights * (target_net_exposure / w_sum)
    avg_weights = np.clip(avg_weights, min_position, max_position)
    return pd.Series(avg_weights, index=tickers)


def black_litterman(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    views: Dict[str, float],
    view_confidences: Dict[str, float],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> pd.Series:
    """Black-Litterman posterior expected returns."""
    pi = risk_aversion * cov_matrix.dot(market_weights)  # CAPM equilibrium returns
    P = np.zeros((len(views), len(market_weights)))
    Q = np.zeros(len(views))
    omega = np.zeros((len(views), len(views)))

    for i, (ticker, view) in enumerate(views.items()):
        if ticker not in market_weights.index:
            continue
        P[i, market_weights.index.get_loc(ticker)] = 1
        Q[i] = view
        conf = max(view_confidences.get(ticker, 0.5), 1e-6)
        omega[i, i] = (1.0 / conf) * tau

    # Robustly regularize omega before solving the linear system
    omega += np.eye(len(views)) * 1e-6

    cov_arr = cov_matrix.values
    # BL formula using scipy.linalg.solve for numerical stability
    # Posterior = pi + tau*Sigma*P' * inv(P*tau*Sigma*P' + Omega) * (Q - P*pi)
    M = P.dot(tau * cov_arr).dot(P.T) + omega  # (K x K)
    rhs = Q - P.dot(pi.values)
    try:
        adjustment = tau * cov_arr.dot(P.T).dot(linalg.solve(M, rhs))
    except linalg.LinAlgError:
        logger.warning("Black-Litterman linear solve failed; returning CAPM prior.")
        return pi

    pi_bl = pi + adjustment
    return pi_bl


def apply_fx_overlay(
    expected_returns: pd.Series,
    usd_exposure: pd.Series,
    usd_mxn_level: float,
    expected_usdmxn_return: float,
    hedge_ratio: float = 0.5,
) -> pd.Series:
    """Apply FX overlay to expected returns."""
    fx_adjustment = usd_exposure * (1 - hedge_ratio) * expected_usdmxn_return
    adjusted = expected_returns + fx_adjustment
    return adjusted
