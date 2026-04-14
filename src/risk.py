from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import genextreme, kstest
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / (excess.std(ddof=0) + 1e-9)


def compute_sortino(returns: pd.Series, required_return: float = 0.0) -> float:
    downside = returns[returns < required_return] - required_return  # excess loss below MAR
    # Lower partial moment (standard semideviation)
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 1 else 0.0
    excess_mean = returns.mean() - required_return
    return excess_mean / (downside_std + 1e-9) * np.sqrt(252)


def max_drawdown(returns: pd.Series) -> float:
    cumulative = np.exp(returns.fillna(0.0).cumsum())
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def compute_var(returns: pd.Series, alpha: float = 0.95) -> float:
    return np.percentile(returns.dropna(), 100 * (1 - alpha))


def compute_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    var_level = compute_var(returns, alpha)
    tail = returns[returns <= var_level]
    if len(tail) == 0:
        return float(var_level)
    if len(tail) == 1:
        return float(tail.iloc[0])
    return float(tail.mean())


def stress_test(
    portfolio_returns: pd.Series,
    scenario_shocks: dict[str, float],
    exposures: dict[str, float],
    risk_free_rate: float = 0.02,
    shock_days: int = 5,
    event_spacing_days: int = 126,
) -> pd.DataFrame:
    scenario_results = []
    n = max(int(shock_days), 1)
    spacing = max(int(event_spacing_days), n)
    for label, shock in scenario_shocks.items():
        exposure = exposures.get(label, 0.0)
        adjusted = portfolio_returns.copy()
        # Apply repeated finite stress windows over the sample so scenarios remain
        # comparable and visible in aggregate metrics, while avoiding permanent drift.
        shock_per_day = (shock * exposure) / n
        if len(adjusted) > 0:
            for start in range(0, len(adjusted), spacing):
                end = min(start + n, len(adjusted))
                adjusted.iloc[start:end] = adjusted.iloc[start:end] + shock_per_day
        scenario_results.append(
            {
                "scenario": label,
                "mean_return": adjusted.mean(),
                "volatility": adjusted.std(ddof=0),
                "sharpe": compute_sharpe(adjusted, risk_free_rate=risk_free_rate),
                "max_drawdown": max_drawdown(adjusted),
            }
        )
    return pd.DataFrame(scenario_results)


def fit_garch(returns: pd.Series, model: str = "GJR") -> arch_model:
    """Fit a GARCH model to returns."""
    import warnings

    if model == "GARCH":
        mod = arch_model(returns, vol="Garch", p=1, q=1, rescale=True)
    elif model == "GJR":
        mod = arch_model(returns, vol="Garch", p=1, o=1, q=1, rescale=True)
    elif model == "EGARCH":
        mod = arch_model(returns, vol="EGarch", p=1, q=1, rescale=True)
    else:
        raise ValueError("Unsupported model")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mod.fit(disp="off", show_warning=False, options={"maxiter": 500})
    if result.convergence_flag != 0 and model != "GARCH":
        # Fallback to simpler standard GARCH(1,1) which is easier to fit
        logger.debug("GARCH (%s) did not converge — retrying with GARCH(1,1).", model)
        mod2 = arch_model(returns, vol="Garch", p=1, q=1, rescale=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result2 = mod2.fit(disp="off", show_warning=False, options={"maxiter": 500})
        if result2.convergence_flag == 0:
            return result2
    return result


def garch_forecast_vol(fitted_result, horizon: int = 21) -> float:
    """Forecast annualized volatility for given horizon."""
    forecast = fitted_result.forecast(horizon=horizon)
    horizon_var = forecast.variance.iloc[-1].values
    mean_daily_var = float(np.nanmean(horizon_var))
    vol = np.sqrt(max(mean_daily_var, 0.0) * 252)
    return vol


def rolling_garch_forecast(
    returns: pd.Series,
    horizon: int = 21,
    lookback: int = 504,
    refit_every: int = 21,
) -> pd.Series:
    """Build a rolling annualized GARCH forecast series.

    The model is re-fit every ``refit_every`` observations using a trailing
    ``lookback`` window, then the forecast is forward-filled until next refit.
    """
    r = returns.dropna()
    if len(r) < max(lookback, 100):
        return pd.Series(index=returns.index, dtype=float)

    out = pd.Series(index=r.index, dtype=float)
    step = max(int(refit_every), 1)
    lb = max(int(lookback), 100)

    for i in range(lb, len(r), step):
        train = r.iloc[i - lb:i]
        if len(train) < 100:
            continue
        try:
            # Robustify fit inputs and cap extreme forecasts vs recent realized vol.
            lo, hi = train.quantile([0.01, 0.99])
            train_fit = train.clip(lower=float(lo), upper=float(hi))
            fitted = fit_garch(train_fit, "GJR")
            fcst = garch_forecast_vol(fitted, horizon=horizon)

            recent_realized = float(train.std(ddof=0) * np.sqrt(252))
            if np.isfinite(recent_realized) and recent_realized > 1e-9:
                lower = 0.60 * recent_realized
                upper = 1.60 * recent_realized
                fcst = float(np.clip(fcst, lower, upper))

            out.loc[r.index[i]] = fcst
        except Exception:
            continue

    return out.reindex(returns.index).ffill()


def dynamic_var(returns: pd.Series, alpha: float = 0.95, method: str = "garch") -> pd.Series:
    """Rolling 1-day VaR."""
    if method == "garch":
        r = returns.dropna()
        if len(r) < 60:
            return r.reindex(returns.index).rolling(252, min_periods=20).quantile(1 - alpha)

        # Fit in percent space for numerical stability, then convert sigma back to decimal.
        fitted = fit_garch(r * 100.0, "GJR")
        sigma = pd.Series(fitted.conditional_volatility, index=r.index) / 100.0

        std_resid = pd.Series(fitted.std_resid, index=r.index).replace([np.inf, -np.inf], np.nan).dropna()
        if len(std_resid) >= 30:
            z_alpha = float(np.percentile(std_resid, (1 - alpha) * 100))
        else:
            z_alpha = float(np.percentile((r / (r.std(ddof=0) + 1e-9)).dropna(), (1 - alpha) * 100))

        # VaR remains in decimal return units (typically a small negative number).
        var = sigma * z_alpha
        var = var.reindex(returns.index).ffill()
    elif method == "empirical":
        var = returns.rolling(252).quantile(1 - alpha)
    else:
        raise ValueError("Unsupported method")
    return var


def monte_carlo_var(
    returns: pd.Series,
    asset_returns: pd.DataFrame | None = None,
    weights: pd.Series | None = None,
    alpha: float = 0.95,
    n_sim: int = 10000,
    horizon: int = 1,
) -> float:
    """Monte Carlo VaR.

    If *asset_returns* (T × N) and *weights* (N,) are supplied, simulates
    correlated multi-asset portfolio returns using a Ledoit-Wolf covariance
    estimated from the individual asset return history.  This captures
    cross-ticker correlation that the univariate fallback misses.

    Falls back to fitting a 1-D normal on the portfolio return series when
    per-asset data is unavailable.
    """
    rng = np.random.default_rng(42)

    if asset_returns is not None and weights is not None and not asset_returns.empty:
        # Align weights to asset_returns columns and normalise to sum = 1
        common = [c for c in asset_returns.columns if c in weights.index]
        ar = asset_returns[common].fillna(0.0)
        w = weights.reindex(common).fillna(0.0)
        w_sum = w.sum()
        if w_sum > 1e-9:
            w = w / w_sum
        else:
            # degenerate — fall through to univariate
            asset_returns = None

    if asset_returns is not None and weights is not None and not asset_returns.empty:
        means = ar.mean().values
        cov = LedoitWolf().fit(ar.values).covariance_
        # Simulate horizon-day asset returns
        sim_asset = rng.multivariate_normal(means * horizon, cov * horizon, size=n_sim)
        sim_portfolio = sim_asset @ w.values
    else:
        # Univariate fallback
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        sim_portfolio = rng.normal(mean * horizon, std * np.sqrt(horizon), size=n_sim)

    return float(np.percentile(sim_portfolio, (1 - alpha) * 100))


def gev_var(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """VaR and CVaR using GEV distribution on left tail."""
    tail = -returns[returns < 0]
    if len(tail) < 20:
        # Not enough tail observations — fall back to empirical
        empirical_var = float(np.percentile(returns.dropna(), (1 - alpha) * 100))
        return empirical_var, float(returns[returns <= empirical_var].mean() if any(returns <= empirical_var) else empirical_var)
    params = genextreme.fit(tail)
    # Goodness-of-fit check: reject GEV if K-S p-value < 0.05
    _ks_stat, ks_pval = kstest(tail, "genextreme", args=params)
    if ks_pval < 0.05:
        logger.debug(
            "GEV fit rejected by K-S test (p=%.4f); falling back to empirical quantile.", ks_pval
        )
        empirical_var = float(np.percentile(returns.dropna(), (1 - alpha) * 100))
        return empirical_var, float(returns[returns <= empirical_var].mean() if any(returns <= empirical_var) else empirical_var)
    # ppf gives positive loss magnitude; negate to match return convention (negative = loss)
    loss_var = float(genextreme.ppf(alpha, *params))
    x_tail = np.linspace(loss_var, genextreme.ppf(0.9999, *params), 1000)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    loss_cvar = float(_trapz(x_tail * genextreme.pdf(x_tail, *params), x_tail) / (1 - alpha))
    return -loss_var, -loss_cvar


def detect_macro_regime(macro: pd.DataFrame) -> str:
    """Classify the current macro regime into one of three states.

    Uses three signals from the most recent macro snapshot:

    - Banxico rate momentum (3-month change): rising = tightening
    - Industrial production YoY: negative = contraction
    - USD/MXN momentum (3-month %change): > +5% = stress

    Returns one of:
        "expansion"  — growth, low rates or falling rates, stable FX
        "tightening" — Banxico raising, but still positive growth
        "stress"     — contraction OR high FX stress

    Falls back to "expansion" if data is insufficient.
    """
    if macro is None or len(macro) < 4:
        return "expansion"

    df = macro.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()

    # --- Banxico momentum: 3-month change in rate ---
    banxico = df["banxico_rate"].dropna()
    if len(banxico) >= 3:
        banxico_delta = float(banxico.iloc[-1] - banxico.iloc[-3])
    else:
        banxico_delta = 0.0

    # --- Industrial production YoY ---
    ip = df["industrial_production_yoy"].dropna()
    ip_latest = float(ip.iloc[-1]) if len(ip) > 0 else 0.02

    # --- USD/MXN 3-month momentum ---
    fx = df["usd_mxn"].dropna()
    if len(fx) >= 3:
        fx_mom = float((fx.iloc[-1] - fx.iloc[-3]) / (fx.iloc[-3] + 1e-9))
    else:
        fx_mom = 0.0

    # --- Regime classification ---
    if ip_latest < 0.0 or fx_mom > 0.05:
        return "stress"
    elif banxico_delta > 0.25:          # >25bps cumulative tightening in 3m
        return "tightening"
    else:
        return "expansion"


def regime_asset_class_constraints(regime: str) -> dict:
    """Return asset-class min/max constraints keyed by regime.

    expansion : full risk-on — max equity, minimal bonds
    tightening: reduce equity, add bonds as vol buffer
    stress    : defensive — cut equity, max bonds

    The __asset_class_map__ key must be added by the caller.
    """
    _regimes = {
        "expansion": {
            "equity":       {"min": 0.55, "max": 0.90},
            "fibra":        {"min": 0.05, "max": 0.25},
            "fixed_income": {"min": 0.00, "max": 0.10},
        },
        "tightening": {
            "equity":       {"min": 0.45, "max": 0.75},
            "fibra":        {"min": 0.05, "max": 0.20},
            "fixed_income": {"min": 0.05, "max": 0.20},
        },
        "stress": {
            "equity":       {"min": 0.35, "max": 0.60},
            "fibra":        {"min": 0.03, "max": 0.15},
            "fixed_income": {"min": 0.10, "max": 0.30},
        },
    }
    return dict(_regimes.get(regime, _regimes["expansion"]))
