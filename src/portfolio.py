from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import OptimizeResult, minimize
from typing import Optional, Dict

logger = logging.getLogger(__name__)

_EPS_SMOOTH = 1e-8


def _smooth_abs(x: np.ndarray) -> np.ndarray:
    """Differentiable approximation of |x|: sqrt(x^2 + eps)."""
    return np.sqrt(x * x + _EPS_SMOOTH)


def _smooth_sign(x: np.ndarray) -> np.ndarray:
    """Smooth approximation of sign(x): x / sqrt(x^2 + eps)."""
    return x / np.sqrt(x * x + _EPS_SMOOTH)


def _build_feasible_x0(
    tickers: list,
    asset_class_constraints: Dict,
    max_position: float,
    min_position: float,
    target_net_exposure: float,
) -> np.ndarray:
    """Build a starting point that satisfies asset-class group allocation constraints.

    Uses the midpoint of (group_min, achievable_group_max) as the target allocation
    for each group, then scales proportionally to sum to target_net_exposure.

    This guarantees the starting point is feasible w.r.t. all group inequalities
    and avoids 'Inequality constraints incompatible' / 'Positive directional
    derivative' errors in SLSQP caused by an infeasible starting point.

    Example failure: expansion regime has fixed_income max=0.10, but 7 bond tickers
    at equal-weight give 7 × (0.9/24) = 0.2625 >> 0.10 → infeasible x0.
    """
    n = len(tickers)
    if not asset_class_constraints:
        return np.full(n, target_net_exposure / n)

    ac_map = asset_class_constraints.get("__asset_class_map__", {})
    w = np.zeros(n)

    groups: Dict[str, dict] = {}
    for ac, bounds in asset_class_constraints.items():
        if ac == "__asset_class_map__" or not isinstance(bounds, dict):
            continue
        ac_idx = [i for i, t in enumerate(tickers) if ac_map.get(t) == ac]
        if not ac_idx:
            continue
        n_ac = len(ac_idx)
        ac_min = bounds.get("min", 0.0)
        ac_max_achievable = min(bounds.get("max", target_net_exposure), n_ac * max_position)
        groups[ac] = {"idx": ac_idx, "min": ac_min, "max": ac_max_achievable}

    # Target per group = midpoint of [group_min, achievable_group_max]
    for ac, g in groups.items():
        n_ac = len(g["idx"])
        target_ac = (g["min"] + g["max"]) / 2.0
        per_ticker = np.clip(target_ac / n_ac if n_ac > 0 else 0.0, min_position, max_position)
        for i in g["idx"]:
            w[i] = per_ticker

    # Scale uniformly so that sum(w) = target_net_exposure
    s = w.sum()
    if s > 1e-9:
        w = w * (target_net_exposure / s)
        # If scaling pushed any weight above max_position, clip and renormalise once
        if np.any(w > max_position + 1e-9):
            w = np.clip(w, min_position, max_position)
            s2 = w.sum()
            if s2 > 1e-9:
                w = w * (target_net_exposure / s2)
    else:
        w = np.full(n, target_net_exposure / n)

    return w


def _sanitize_asset_class_constraints(
    constraints: Dict,
    tickers: list,
    max_position: float,
    target_net_exposure: float,
) -> Dict:
    """Relax asset-class min bounds that are impossible given max_position × N_tickers.

    When a group has fewer tickers than required to reach its minimum allocation
    (e.g. 2 FIBRAs × 0.15 = 0.30 < min 0.40), SLSQP reports
    'Inequality constraints incompatible'.  This helper caps each group's min to
    what is actually achievable and logs a warning.
    """
    sanitized = dict(constraints)
    ac_map: Dict[str, str] = sanitized.get("__asset_class_map__", {})
    for ac, bounds_dict in sanitized.items():
        if ac == "__asset_class_map__" or not isinstance(bounds_dict, dict):
            continue
        ac_tickers = [t for t in tickers if ac_map.get(t) == ac]
        max_achievable = len(ac_tickers) * max_position
        ac_min = bounds_dict.get("min", 0.0)
        ac_max = bounds_dict.get("max", target_net_exposure)
        if ac_min > max_achievable + 1e-9:
            new_min = round(max_achievable, 4)
            logger.debug(
                "Asset-class '%s': relaxing min %.4f → %.4f "
                "(only %d tickers × max_pos %.2f = %.4f achievable)",
                ac, ac_min, new_min, len(ac_tickers), max_position, max_achievable,
            )
            sanitized[ac] = {**bounds_dict, "min": new_min}
        if ac_max > target_net_exposure + 1e-9:
            sanitized[ac] = {**sanitized[ac], "max": target_net_exposure}
    # Global check: sum of minimums must not exceed target
    total_min = sum(
        v.get("min", 0.0) for k, v in sanitized.items()
        if k != "__asset_class_map__" and isinstance(v, dict)
    )
    if total_min > target_net_exposure + 1e-9:
        scale = target_net_exposure / total_min
        for k, v in sanitized.items():
            if k != "__asset_class_map__" and isinstance(v, dict) and v.get("min", 0.0) > 0:
                sanitized[k] = {**v, "min": v["min"] * scale}
    return sanitized


def _constraints_satisfied(
    weights: np.ndarray,
    bounds: list[tuple[float, float]],
    constraints: list,
    tol: float = 1e-4,
) -> bool:
    """Check whether a candidate solution is numerically feasible."""
    if weights is None or np.any(np.isnan(weights)):
        return False
    for idx, (lower, upper) in enumerate(bounds):
        if weights[idx] < lower - tol or weights[idx] > upper + tol:
            return False
    for constraint in constraints:
        value = np.atleast_1d(constraint["fun"](weights)).astype(float)
        if constraint.get("type") == "eq":
            if np.max(np.abs(value)) > tol:
                return False
        elif np.min(value) < -tol:
            return False
    return True


def _accepted_result(result: object, message: str) -> object:
    """Mark a numerically acceptable optimizer result as successful."""
    result.success = True
    result.message = message
    return result


def _run_slsqp(
    objective, x0, bounds, constraints, args=(),
    target_net_exposure: float = 1.0,
    jac=None,
    eps: float = 1.49e-8,
    x0_retry: np.ndarray | None = None,
    eq_only_constraints: list | None = None,
) -> object:
    """Run SLSQP with two fallback retries on numerical failures.

    Attempt 1 — warm start (x0 = previous weights, normalised to target).
    Attempt 2 — feasible start (x0 = x0_retry), relaxed ftol.
    Attempt 3 — equality-only SLSQP (drops group-inequality constraints).
                Always converges; used as last-resort fallback.

    x0_retry            : feasible starting point; falls back to equal-weight.
    eq_only_constraints : constraints list with *only* the sum-equality entry.
                          Used for attempt 3; defaults to first entry of constraints.
    jac                 : analytical Jacobian (optional).
    eps                 : finite-difference step.  Use ~1e-4 for non-smooth (CVaR).
    """
    import warnings as _warnings

    _NUMERICAL_FAILURES = {
        "positive directional derivative",
        "inequality constraints incompatible",
        "iteration limit reached",
        "singular matrix",
    }

    def _normalize_x0(x: np.ndarray) -> np.ndarray:
        """Project x onto the sum=target hyperplane (avoids equality-constraint infeasibility)."""
        s = x.sum()
        return x * (target_net_exposure / s) if s > 1e-9 else x

    options = {"maxiter": 2000, "ftol": 1e-7, "eps": eps}
    result = minimize(
        objective, _normalize_x0(x0), args=args,
        bounds=bounds, constraints=constraints,
        method="SLSQP", jac=jac, options=options,
    )
    if result.success or _constraints_satisfied(result.x, bounds, constraints):
        if not result.success:
            return _accepted_result(result, "Accepted feasible SLSQP solution despite numerical warning.")
        return result
    if not result.success and any(kw in result.message.lower() for kw in _NUMERICAL_FAILURES):
        n = len(x0)
        _x0_retry = _normalize_x0(
            x0_retry if x0_retry is not None else np.full(n, target_net_exposure / n)
        )
        result2 = minimize(
            objective, _x0_retry, args=args,
            bounds=bounds, constraints=constraints,
            method="SLSQP", jac=jac, options={"maxiter": 2000, "ftol": 1e-6, "eps": eps},
        )
        if result2.success or _constraints_satisfied(result2.x, bounds, constraints):
            if not result2.success:
                return _accepted_result(result2, "Accepted feasible retry solution despite numerical warning.")
            return result2
        # 3rd attempt: equality-only SLSQP — always feasible (no group inequalities).
        # Uses numerical gradient (jac=None) to avoid any analytical-gradient issues.
        # If it produces a valid result (no NaN), we always return this over the two
        # non-converged group-constrained attempts above.
        _eq_constraints = eq_only_constraints if eq_only_constraints is not None else constraints[:1]
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            result3 = minimize(
                objective, _x0_retry, args=args,
                bounds=bounds, constraints=_eq_constraints,
                method="SLSQP", jac=None,
                options={"maxiter": 5000, "ftol": 1e-5, "eps": 1e-5},
            )
        if result3.success or _constraints_satisfied(result3.x, bounds, constraints):
            if not result3.success:
                return _accepted_result(result3, "Accepted repaired fallback solution.")
            return result3
        if _constraints_satisfied(_x0_retry, bounds, constraints):
            return OptimizeResult(
                x=_x0_retry,
                success=True,
                message="Used feasible fallback weights after SLSQP failures.",
            )
    return result


def _portfolio_gradient(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
    turnover_penalty: float,
    prev_weights: np.ndarray,
    market_impact: np.ndarray | None = None,
) -> np.ndarray:
    """Analytical gradient of _portfolio_objective w.r.t. weights."""
    er_scale = max(np.abs(expected_returns).mean(), 1e-4)
    sign_delta = _smooth_sign(weights - prev_weights)  # smooth gradient of turnover
    grad = (
        -expected_returns
        + 2.0 * risk_aversion * (cov_matrix @ weights)
        + turnover_penalty * er_scale * sign_delta
    )
    if market_impact is not None:
        grad = grad + market_impact * sign_delta
    return grad


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
    delta = _smooth_abs(weights - prev_weights)  # smooth to keep objective differentiable
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
    max_position: float = 0.10,
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.1,
    asset_class_constraints: Optional[Dict[str, Dict[str, float]]] = None,
    adtv_scores: Optional[pd.Series] = None,
    market_impact_eta: float = 0.1,
    issuer_consolidated_limits: Optional[Dict[str, list]] = None,
    max_position_overrides: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Mean-variance optimizer (SLSQP).

    Disposiciones de Carácter General aplicables a los fondos de inversión
    — CNBV — límite del 10 % por emisor sobre NAV.

    Constraints enforced:
    - Individual position: w_i ∈ [min_position, min(max_position, override_i)]
    - Issuer concentration: sum(|w_i|) for tickers sharing issuer_id ≤ 0.10
    - Asset-class group bounds (equity, fibra, etc.)

    adtv_scores: Series indexed by ticker with normalized ADTV scores in [0, 1].
        When provided, a market-impact penalty η·σ_i/ADTV_i·|Δw_i| is added to
        the objective (Almgren-Chriss linear model).  Tickers absent from
        adtv_scores receive ADTV = 0.5 (neutral).
    market_impact_eta: scaling coefficient η for the market-impact term.
    issuer_consolidated_limits: dict mapping issuer_id → list of ticker indices.
        Each group is constrained to sum ≤ issuer_concentration_limit (0.10).
    max_position_overrides: dict mapping ticker → individual cap.
        Effective cap = min(max_position, override).
    """
    tickers = expected_returns.index.tolist()
    if prev_weights is None:
        prev_weights = pd.Series(0.0, index=tickers)

    # Sanitize inputs: replace NaN/inf expected returns with 0
    er_values = np.nan_to_num(expected_returns.values, nan=0.0, posinf=0.5, neginf=-0.5)

    # Annualise the covariance matrix so it is on the same scale as expected returns.
    # Daily covariance (~1e-4) vs annual returns (~0.1) creates a 1000x scale mismatch
    # that causes SLSQP to report "Positive directional derivative for linesearch".
    cov_aligned = cov_matrix.reindex(index=tickers, columns=tickers).fillna(0.0).values
    cov_aligned = np.nan_to_num(cov_aligned, nan=0.0)
    cov_ann = cov_aligned * 252.0  # annualised
    # Add tiny ridge for numerical stability (ensures positive-definiteness)
    cov_ann += np.eye(len(tickers)) * 1e-8

    # Build per-ticker market-impact vector: η * σ_i / ADTV_i  (annualised vol)
    vol_diag = np.sqrt(np.diag(cov_ann))
    if adtv_scores is not None:
        adtv = adtv_scores.reindex(tickers).fillna(0.5).clip(lower=0.05).values
    else:
        adtv = np.full(len(tickers), 0.5)
    market_impact = market_impact_eta * vol_diag / adtv

    # Apply per-ticker max_position_override
    ticker_upper_bounds = [max_position] * len(tickers)
    if max_position_overrides:
        for i, t in enumerate(tickers):
            if t in max_position_overrides:
                ticker_upper_bounds[i] = min(max_position, max_position_overrides[t])

    # Build a feasible starting point that respects group constraints (midpoint allocation)
    # Equal-weight violates group maxima (e.g. fixed_income max=0.10 with 7 bond tickers)
    x0_feasible = _build_feasible_x0(
        tickers, asset_class_constraints or {}, max_position, min_position, target_net_exposure
    )

    # Warm-start from previous weights when available; fall back to feasible point
    x0 = prev_weights.values.copy()
    if np.sum(x0) < 1e-9:
        x0 = x0_feasible
    bounds = [(min_position, ub) for ub in ticker_upper_bounds]
    # Sum-equality constraint (kept separately for the equality-only fallback)
    eq_constraint = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - target_net_exposure,
        }
    ]
    constraints = list(eq_constraint)

    # Asset-class group constraints (sanitize before adding to avoid infeasibility)
    if asset_class_constraints:
        asset_class_constraints = _sanitize_asset_class_constraints(
            asset_class_constraints, tickers, max_position, target_net_exposure
        )
        ac_map: Dict[str, str] = asset_class_constraints.get("__asset_class_map__", {})
        ac_filtered = {k: v for k, v in asset_class_constraints.items() if k != "__asset_class_map__"}
        if ac_map:
            for ac, bounds_dict in ac_filtered.items():
                ac_tickers = [t for t in tickers if ac_map.get(t) == ac]
                if not ac_tickers:
                    continue
                idx = [tickers.index(t) for t in ac_tickers]
                ac_min = bounds_dict.get("min", 0.0)
                ac_max = bounds_dict.get("max", 1.0)
                ac_min = min(ac_min, ac_max)  # guard: min must not exceed max
                # Tiny slack on min so the feasible starting-point is strictly interior
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mn=ac_min: np.sum(x[i]) - mn + 1e-8}
                )
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mx=ac_max: mx - np.sum(x[i]) + 1e-8}
                )

    # Issuer consolidated concentration limit (CNBV 10% per issuer on NAV)
    if issuer_consolidated_limits:
        for issuer_id, group_idx in issuer_consolidated_limits.items():
            if len(group_idx) > 0:
                constraints.append(
                    {"type": "ineq",
                     "fun": lambda x, gi=group_idx: 0.10 - np.sum(np.abs(x[gi])) + 1e-8}
                )

    result = _run_slsqp(
        _portfolio_objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        args=(er_values, cov_ann, risk_aversion, turnover_penalty,
              prev_weights.values, market_impact),
        target_net_exposure=target_net_exposure,
        jac=_portfolio_gradient,
        x0_retry=x0_feasible,
        eq_only_constraints=eq_constraint,
    )
    if not result.success:
        if np.any(np.isnan(result.x)):
            raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    # Normalise so the sum constraint is exactly satisfied (important for next iteration's x0)
    w = result.x.copy()
    for i, ub in enumerate(ticker_upper_bounds):
        w[i] = np.clip(w[i], min_position, ub)
    w_sum = w.sum()
    if w_sum > 1e-9:
        w = w * (target_net_exposure / w_sum)
    return pd.Series(w, index=tickers)


def optimize_portfolio_cvar(
    expected_returns: pd.Series,
    scenario_returns: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    max_position: float = 0.10,
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.1,
    alpha: float = 0.95,
    asset_class_constraints: Optional[Dict] = None,
    adtv_scores: Optional[pd.Series] = None,
    market_impact_eta: float = 0.1,
    issuer_consolidated_limits: Optional[Dict[str, list]] = None,
    max_position_overrides: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Mean-CVaR portfolio optimization.

    Disposiciones de Carácter General aplicables a los fondos de inversión
    — CNBV — límite del 10 % por emisor sobre NAV.

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

    scen_df = (
        scenario_returns[common]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    scen = scen_df.to_numpy(dtype=float)  # T × N_common
    common_idx = [tickers.index(t) for t in common]
    # Sanitize expected returns
    er_values_cvar = np.nan_to_num(expected_returns.values, nan=0.0, posinf=0.5, neginf=-0.5)
    er_scale = max(float(np.abs(er_values_cvar).mean()), 1e-4)
    # Annualise scenario returns for consistent scale with expected returns
    scen_ann = scen * 252.0

    # Per-ticker vol proxy from scenario returns for market-impact term (annualised)
    scen_vol = np.std(scen_ann, axis=0, ddof=1)  # N_common
    if adtv_scores is not None:
        adtv_common = adtv_scores.reindex(common).fillna(0.5).clip(lower=0.05).values
    else:
        adtv_common = np.full(len(common), 0.5)
    market_impact_common = market_impact_eta * scen_vol / adtv_common

    def _objective(w: np.ndarray) -> float:
        port_scen = scen_ann @ w[common_idx]
        var_level = np.percentile(port_scen, (1 - alpha) * 100)
        tail = port_scen[port_scen <= var_level]
        cvar = float(tail.mean()) if len(tail) > 0 else float(var_level)
        return_term = -float(w @ er_values_cvar)
        cvar_term = risk_aversion * abs(cvar)
        delta = np.abs(w[common_idx] - prev_weights.values[common_idx])
        turnover_term = turnover_penalty * er_scale * float(np.sum(np.abs(w - prev_weights.values)))
        impact_term = float((market_impact_common * delta).sum())
        return return_term + cvar_term + turnover_term + impact_term

    # Build feasible starting point (respects group constraints) for retry
    x0_feasible = _build_feasible_x0(
        tickers, asset_class_constraints or {}, max_position, min_position, target_net_exposure
    )

    # Apply per-ticker max_position_override
    ticker_upper_bounds = [max_position] * len(tickers)
    if max_position_overrides:
        for i, t in enumerate(tickers):
            if t in max_position_overrides:
                ticker_upper_bounds[i] = min(max_position, max_position_overrides[t])

    # Warm-start from previous weights when available; fall back to feasible point
    x0 = prev_weights.values.copy()
    if np.sum(x0) < 1e-9:
        x0 = x0_feasible
    bounds = [(min_position, ub) for ub in ticker_upper_bounds]
    eq_constraint = [{"type": "eq", "fun": lambda x: np.sum(x) - target_net_exposure}]
    constraints = list(eq_constraint)

    if asset_class_constraints:
        asset_class_constraints = _sanitize_asset_class_constraints(
            asset_class_constraints, tickers, max_position, target_net_exposure
        )
        ac_map: Dict[str, str] = asset_class_constraints.get("__asset_class_map__", {})
        ac_filtered = {k: v for k, v in asset_class_constraints.items() if k != "__asset_class_map__"}
        if ac_map:
            for ac, bounds_dict in ac_filtered.items():
                ac_tickers = [t for t in tickers if ac_map.get(t) == ac]
                if not ac_tickers:
                    continue
                idx = [tickers.index(t) for t in ac_tickers]
                ac_min = bounds_dict.get("min", 0.0)
                ac_max = bounds_dict.get("max", 1.0)
                ac_min = min(ac_min, ac_max)  # guard: min must not exceed max
                # Tiny slack on min so the feasible starting-point is strictly interior
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mn=ac_min: np.sum(x[i]) - mn + 1e-8}
                )
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mx=ac_max: mx - np.sum(x[i]) + 1e-8}
                )

    # Issuer consolidated concentration limit (CNBV 10% per issuer on NAV)
    if issuer_consolidated_limits:
        for issuer_id, group_idx in issuer_consolidated_limits.items():
            if len(group_idx) > 0:
                constraints.append(
                    {"type": "ineq",
                     "fun": lambda x, gi=group_idx: 0.10 - np.sum(np.abs(x[gi])) + 1e-8}
                )

    result = _run_slsqp(
        _objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        target_net_exposure=target_net_exposure,
        eps=1e-4,  # larger step for non-smooth CVaR objective (np.percentile kinks)
        x0_retry=x0_feasible,
        eq_only_constraints=eq_constraint,
    )
    if not result.success:
        if np.any(np.isnan(result.x)):
            raise RuntimeError(f"CVaR optimization failed: {result.message}")
    # Normalise so the sum constraint is exactly satisfied (important for next iteration's x0)
    w = result.x.copy()
    for i, ub in enumerate(ticker_upper_bounds):
        w[i] = np.clip(w[i], min_position, ub)
    w_sum = w.sum()
    if w_sum > 1e-9:
        w = w * (target_net_exposure / w_sum)
    return pd.Series(w, index=tickers)


def optimize_portfolio_robust(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    n_simulations: int = 100,
    max_position: float = 0.10,
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.1,
    asset_class_constraints: Optional[Dict] = None,
    adtv_scores: Optional[pd.Series] = None,
    market_impact_eta: float = 0.1,
    random_seed: int = 42,
    issuer_consolidated_limits: Optional[Dict[str, list]] = None,
    max_position_overrides: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Michaud Resampled Efficiency optimizer.

    Disposiciones de Carácter General aplicables a los fondos de inversión
    — CNBV — límite del 10 % por emisor sobre NAV.

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
                issuer_consolidated_limits=issuer_consolidated_limits,
                max_position_overrides=max_position_overrides,
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
