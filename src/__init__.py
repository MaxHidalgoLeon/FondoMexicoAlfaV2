"""Fondo Mexico research package."""

from .pipeline import run_pipeline, print_summary
from .data_loader import load_mock_data, load_data
from .features import build_signal_matrix
from .signals import score_cross_section, forecast_returns
from .bootstrap import bootstrap_metric, bootstrap_paired_difference, bootstrap_block_size_selector
from .portfolio import optimize_portfolio, optimize_portfolio_cvar, optimize_portfolio_robust, black_litterman, apply_fx_overlay
from .risk import compute_sharpe, compute_sortino, compute_cvar, max_drawdown, gev_var, detect_macro_regime, regime_asset_class_constraints
from .alpha_significance import compute_benchmark_alpha_significance
from .signal_diagnostics import compute_signal_ic_diagnostics
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
    "bootstrap_metric",
    "bootstrap_paired_difference",
    "bootstrap_block_size_selector",
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
    "compute_benchmark_alpha_significance",
    "compute_signal_ic_diagnostics",
    "run_backtest",
    "run_hedge_backtest",
]
