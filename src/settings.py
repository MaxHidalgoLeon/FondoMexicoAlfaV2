from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    # ------------------------------------------------------------------
    # Covariance / EWMA
    # ------------------------------------------------------------------
    "covariance_method": "ewma_ledoit_wolf",
    "rolling_cov_window": 63,
    "ewma_lambda_cov": 0.94,
    "ewma_window_cov": 252,
    "ewma_min_periods_cov": 60,
    "realized_vol_method": "ewma",
    "realized_vol_span": 21,
    # ------------------------------------------------------------------
    # Macro regime
    # ------------------------------------------------------------------
    "regime_method": "ewma_composite",
    "regime_ewma_span": 6,
    "regime_threshold_expansion": 0.5,
    "regime_threshold_stress": -0.5,
    "regime_min_confidence_for_switch": 0.3,
    # ------------------------------------------------------------------
    # Liquidity
    # ------------------------------------------------------------------
    "adtv_method": "ewma",
    "adtv_window": 252,
    "adtv_ewma_lambda": 0.97,
    "adtv_min_periods": 60,
    # ------------------------------------------------------------------
    # Bootstrap / significance
    # ------------------------------------------------------------------
    "bootstrap_enabled": True,
    "bootstrap_n_reps": 5000,
    "bootstrap_block_size": 20,
    "bootstrap_confidence": 0.95,
    "bootstrap_seed": 42,
    # ------------------------------------------------------------------
    # Fan chart
    # ------------------------------------------------------------------
    "fan_chart_enabled": True,
    "fan_chart_n_paths": 1000,
    "fan_chart_block_size": 20,
    # ------------------------------------------------------------------
    # Signal IC
    # ------------------------------------------------------------------
    "ic_diagnostics_enabled": True,
    "ic_bootstrap_block_size": 6,
    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------
    "stress_distributional_enabled": True,
    "stress_window_days": 21,
    # ------------------------------------------------------------------
    # Reporting / comparisons
    # ------------------------------------------------------------------
    "enable_method_comparison": True,
    # Numerical tolerances
    "covariance_psd_tolerance": 1e-9,
}


def resolve_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy of the project settings merged over the defaults."""
    merged = deepcopy(DEFAULT_SETTINGS)
    if settings:
        for key, value in settings.items():
            if value is not None:
                merged[key] = value
    return merged
