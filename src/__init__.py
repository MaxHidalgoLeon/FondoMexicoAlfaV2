"""Fondo Mexico research package."""

from .pipeline import run_pipeline, print_summary
from .data_loader import load_mock_data, load_data
from .features import build_signal_matrix
from .signals import score_cross_section, forecast_returns
from .portfolio import optimize_portfolio, optimize_portfolio_cvar, optimize_portfolio_robust, black_litterman, apply_fx_overlay
from .risk import compute_sharpe, compute_sortino, compute_cvar, max_drawdown, gev_var, detect_macro_regime, regime_asset_class_constraints
from .backtest import run_backtest
from .hedge_overlay import run_hedge_backtest

__all__ = [
    "run_pipeline",
    "print_summary",
    "load_mock_data",
    "load_data",
    "build_signal_matrix",
    "score_cross_section",
    "forecast_returns",
    "optimize_portfolio",
    "optimize_portfolio_cvar",
    "optimize_portfolio_robust",
    "black_litterman",
    "apply_fx_overlay",
    "compute_sharpe",
    "compute_sortino",
    "compute_cvar",
    "max_drawdown",
    "gev_var",
    "detect_macro_regime",
    "regime_asset_class_constraints",
    "run_backtest",
    "run_hedge_backtest",
]
