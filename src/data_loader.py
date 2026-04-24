from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)


def compute_adtv_liquidity_scores(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 252,
    method: str = "uniform",
    ewma_lambda: float = 0.97,
    min_periods: int = 60,
) -> pd.Series:
    """
    Compute liquidity scores from real ADTV (Average Daily Traded Value).

    ADTV = mean(close_price * volume) over the last `window` trading days.
    Scores are normalized to [0.0, 1.0] using min-max scaling across the
    equity/FIBRA universe. Fixed-income tickers (not in prices) get 1.0.

    Returns a Series indexed by canonical ticker name.
    """
    common = prices.columns.intersection(volume.columns)
    pv = (prices[common] * volume[common]).replace([np.inf, -np.inf], np.nan)
    method_lc = str(method).lower().strip()
    if method_lc == "ewma":
        adtv = pv.tail(window).ewm(alpha=1.0 - float(ewma_lambda), min_periods=min_periods, adjust=False).mean().iloc[-1]
    else:
        adtv = pv.tail(window).mean()
    score_min, score_max = adtv.min(), adtv.max()
    if score_max - score_min < 1e-9:
        scores = pd.Series(1.0, index=common)
    else:
        scores = (adtv - score_min) / (score_max - score_min)
    return scores


def get_investable_universe() -> pd.DataFrame:
    """Create the nearshoring industrial universe for México (verified on Yahoo Finance 2026-04).

    Thematic mandate: ≥30% revenue exposure to nearshoring / industrial activity.
    Removed (consumer staples / beverages, outside thematic mandate):
        FEMSAUBD, KIMBERA, BIMBOA, GRUMAB, BECLE.
    Removed (fixed income — outside liquidity-sleeve scope or issuer double-count):
        CORP1 (CEMEX 2030, same issuer as CEMEXCPO), CORP2 (KOF consumer),
        MBONO5Y, MBONO10Y (no liquidity-sleeve role).

    Fields added for CNBV compliance:
        thematic_purity: "pure" (>70% nearshoring), "mixed" (30-70%), "proxy" (<30% — future use)
        issuer_id: consolidated issuer key for the 10% issuer concentration limit
        max_position_override: per-ticker cap (NaN = use global max_position)
    """
    tickers = [
        # Equities — industrial / logistics / infrastructure / nearshoring
        "NEMAKA", "GISSAA", "CEMEXCPO", "ICHB", "GCARSOA1",
        "ASURB", "GAPB", "OMAB", "PINFRA", "ORBIA",
        "VESTA",
        # Equities — IPC industrials expanding cross-sectional model power
        "ALPEK", "GMEXICOB", "ALFA", "SIMECB", "VITRO",
        # FIBRAs — industrial / logistics focus
        "FUNO11", "FIBRAPL14", "FIBRAMQ12", "TERRA13", "FMTY14",
        # Fixed income — liquidity sleeve only (CETES + MBONO3Y)
        "CETES28", "CETES91", "MBONO3Y",
    ]
    names = [
        "Nemak", "Grupo Industrial Saltillo", "CEMEX", "Ternium México", "Grupo Carso",
        "Aeropuertos del Sureste", "Grupo Aeroportuario del Pacífico", "Grupo Aeroportuario Centro Norte", "Pinfra", "Orbia",
        "Vesta",
        "Alpek", "Grupo México", "Alfa", "Grupo Simec", "Vitro",
        "FIBRA Uno", "FIBRA Prologis", "FIBRA Macquarie", "FIBRA Terrafina", "FIBRA Monterrey",
        "Cetes 28d", "Cetes 91d", "Mbono 3yr",
    ]
    sectors = [
        "Industrial", "Industrial", "Industrial", "Industrial", "Industrial",
        "Logistics", "Logistics", "Logistics", "Infrastructure", "Industrial",
        "Industrial",
        "Industrial", "Industrial", "Industrial", "Industrial", "Industrial",
        "FIBRA", "FIBRA", "FIBRA", "FIBRA", "FIBRA",
        "Government", "Government", "Government",
    ]
    asset_classes = [
        "equity", "equity", "equity", "equity", "equity",
        "equity", "equity", "equity", "equity", "equity",
        "equity",
        "equity", "equity", "equity", "equity", "equity",
        "fibra", "fibra", "fibra", "fibra", "fibra",
        "fixed_income", "fixed_income", "fixed_income",
    ]
    investable = [
        True, True, True, True, True,
        True, True, True, True, True,
        True,
        True, True, True, True, True,
        True, True, True, True, True,
        True, True, True,
    ]
    # USD revenue exposure estimate (0.0–1.0) — nearshoring names have high USD exposure
    usd_exposure = [
        0.865, 0.70, 0.5387, 0.0038, 0.1977,   # NEMAKA, GISSAA, CEMEX, ICHB, GCARSOA1
        0.1515, 0.0924, 0.30, 0.20, 0.6961,   # ASURB, GAPB, OMAB, PINFRA, ORBIA
        0.86,                             # VESTA
        0.4207, 0.3103, 0.2671, 0.55, 0.55,   # ALPEK, GMEXICOB, ALFA, SIMECB, VITRO
        0.4381, 0.75, 0.8365, 0.9302, 0.7196,   # FUNO11, FIBRAPL14, FIBRAMQ12, TERRA13, FMTY14
        0.0, 0.0, 0.0,                   # bonds
    ]
    # Approximate market caps in MXN millions (Q1 2026 estimates, bonds = 0)
    market_caps = [
        18_000, 12_000, 250_000, 95_000, 180_000,
        120_000, 100_000, 60_000, 85_000, 55_000,
        35_000,
        42_000, 380_000, 38_000, 22_000, 7_500,
        140_000, 50_000, 30_000, 45_000, 20_000,
        0, 0, 0,
    ]
    # Liquidity score (0.0–1.0) based on average daily traded value
    liquidity = [
        0.55, 0.45, 0.95, 0.70, 0.80,
        0.85, 0.80, 0.65, 0.75, 0.60,
        0.50,
        0.62, 0.88, 0.56, 0.53, 0.47,
        0.90, 0.65, 0.55, 0.60, 0.48,
        1.0, 1.0, 1.0,
    ]
    # Thematic purity: "pure" (>70% nearshoring), "mixed" (30-70%), "proxy" (<30%, future use)
    # Only "pure" and "mixed" are investable=True.
    thematic_purity = [
        "pure", "pure", "pure", "pure", "mixed",      # NEMAKA..GCARSOA1
        "pure", "pure", "pure", "pure", "pure",        # ASURB..ORBIA
        "pure",                                         # VESTA
        "pure", "pure", "mixed", "pure", "pure",       # ALPEK..VITRO
        "mixed", "pure", "pure", "pure", "mixed",      # FUNO11..FMTY14
        "pure", "pure", "pure",                         # bonds (always pure for FI)
    ]
    # Consolidated issuer ID for CNBV 10% issuer limit
    issuer_id = [
        "NEMAK", "GISSA", "CEMEX", "TERNIUM", "GCARSO",
        "ASUR", "GAP", "OMA", "PINFRA", "ORBIA",
        "VESTA",
        "ALPEK", "GMEXICO", "ALFA", "SIMEC", "VITRO",
        "FUNO", "FIBRAPL", "FIBRAMQ", "TERRA", "FMTY",
        "GOB_MX", "GOB_MX", "GOB_MX",
    ]
    # Per-ticker max_position_override (NaN = use global max_position)
    # FUNO11 capped at 4% due to diversified exposure (~35-40% industrial)
    max_position_override = [
        float("nan")] * 16 + [  # 16 equities: no override
        0.04,  # FUNO11 — mixed thematic purity, cap at 4%
    ] + [float("nan")] * 4 + [  # FIBRAPL14, FIBRAMQ12, TERRA13, FMTY14
        float("nan"), float("nan"), float("nan"),  # bonds
    ]
    df = pd.DataFrame({
        "ticker": tickers,
        "name": names,
        "sector": sectors,
        "asset_class": asset_classes,
        "investable": investable,
        "usd_exposure": usd_exposure,
        "market_cap_mxn": market_caps,
        "liquidity_score": liquidity,
        "thematic_purity": thematic_purity,
        "issuer_id": issuer_id,
        "max_position_override": max_position_override,
    })
    return df


def generate_mock_price_series(
    tickers: list[str],
    start_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    freq: str = "B",
) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq=freq)
    n = len(dates)
    prices = {}
    np.random.seed(42)
    for ticker in tickers:
        drift = np.random.uniform(0.02, 0.12)
        vol = np.random.uniform(0.18, 0.35)
        shocks = np.random.normal(loc=(drift / 252), scale=(vol / np.sqrt(252)), size=n)
        level = 100 * np.exp(np.cumsum(shocks))
        prices[ticker] = level
    price_df = pd.DataFrame(prices, index=dates).clip(lower=1)
    return price_df


def build_mock_fundamentals(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate per-ticker fundamentals with AR(1) persistence — no fresh random draw per month."""
    rng = np.random.default_rng(24)
    records = []
    for ticker in tickers:
        # Initial values per ticker drawn once
        pe = float(rng.uniform(8.0, 24.0))
        pb = float(rng.uniform(1.0, 3.5))
        roe = float(rng.normal(0.16, 0.05))
        margin = float(rng.normal(0.18, 0.07))
        leverage = float(max(0.0, rng.normal(2.5, 1.0)))
        ebitda_growth = float(rng.normal(0.08, 0.08))
        capex = float(max(0.01, rng.normal(0.06, 0.02)))
        for date in dates:
            # Slow AR(1) random walk — economically realistic persistence
            pe = float(np.clip(pe * (1 + rng.normal(0.0, 0.03)), 4.0, 45.0))
            pb = float(np.clip(pb * (1 + rng.normal(0.0, 0.03)), 0.4, 7.0))
            roe = float(np.clip(roe + rng.normal(0.0, 0.008), 0.01, 0.40))
            margin = float(np.clip(margin + rng.normal(0.0, 0.008), 0.01, 0.50))
            leverage = float(np.clip(leverage + rng.normal(0.0, 0.08), 0.0, 8.0))
            ebitda_growth = float(np.clip(ebitda_growth + rng.normal(0.0, 0.015), -0.25, 0.35))
            capex = float(np.clip(capex + rng.normal(0.0, 0.004), 0.01, 0.15))
            records.append({
                "date": date,
                "ticker": ticker,
                "ebitda_growth": ebitda_growth,
                "net_debt_to_ebitda": leverage,
                "roe": roe,
                "profit_margin": margin,
                "capex_to_sales": capex,
                "pe_ratio": pe,
                "pb_ratio": pb,
            })
    return pd.DataFrame.from_records(records)


def build_mock_fibra_fundamentals(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate per-ticker FIBRA fundamentals with AR(1) persistence."""
    rng = np.random.default_rng(25)
    records = []
    for ticker in tickers:
        cap_rate = float(rng.uniform(0.05, 0.12))
        ffo_yield = float(rng.uniform(0.06, 0.15))
        div_yield = float(rng.uniform(0.04, 0.10))
        ltv = float(rng.uniform(0.30, 0.65))
        vacancy = float(rng.uniform(0.02, 0.12))
        for date in dates:
            cap_rate = float(np.clip(cap_rate + rng.normal(0.0, 0.003), 0.03, 0.16))
            ffo_yield = float(np.clip(ffo_yield + rng.normal(0.0, 0.004), 0.03, 0.20))
            div_yield = float(np.clip(div_yield + rng.normal(0.0, 0.003), 0.02, 0.15))
            ltv = float(np.clip(ltv + rng.normal(0.0, 0.01), 0.10, 0.80))
            vacancy = float(np.clip(vacancy + rng.normal(0.0, 0.005), 0.01, 0.25))
            records.append({
                "date": date,
                "ticker": ticker,
                "cap_rate": cap_rate,
                "ffo_yield": ffo_yield,
                "dividend_yield": div_yield,
                "ltv": ltv,
                "vacancy_rate": vacancy,
            })
    return pd.DataFrame.from_records(records)


def _bond_price(ytm: float, coupon_rate: float, n_years: float) -> float:
    """Price a bond with annual coupons using the present value formula."""
    if ytm <= 0 or n_years <= 0:
        return 100.0
    coupon = 100.0 * coupon_rate
    discount = (1.0 + ytm) ** (-n_years)
    # Price = PV of coupons + PV of face
    return coupon * (1.0 - discount) / ytm + 100.0 * discount


def build_mock_bonds(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate mock bond data for the liquidity sleeve instruments only.

    Only CETES28, CETES91, MBONO3Y remain in the investable universe.
    MBONO5Y, MBONO10Y removed (no liquidity-sleeve role).
    CORP1, CORP2 removed (issuer double-count / consumer mandate violation).
    """
    bond_tickers = ["CETES28", "CETES91", "MBONO3Y"]
    records = []
    rng = np.random.default_rng(26)
    # Base YTM per bond type with AR persistence
    base_ytm = {
        "CETES28": 0.055, "CETES91": 0.058,
        "MBONO3Y": 0.075,
    }
    ytm_state = dict(base_ytm)
    credit_base = {"CETES28": 0.0, "CETES91": 0.0, "MBONO3Y": 0.0}
    for date in dates:
        for ticker in bond_tickers:
            # Durations (fixed contract characteristics)
            if "CETES28" in ticker:
                duration = 28 / 365
                maturity = duration
            elif "CETES91" in ticker:
                duration = 91 / 365
                maturity = duration
            else:  # MBONO3Y
                duration = 2.7
                maturity = 3.0

            # AR(1) YTM random walk
            ytm_state[ticker] = float(np.clip(
                ytm_state[ticker] + rng.normal(0.0, 0.002), 0.02, 0.18
            ))
            ytm = ytm_state[ticker]
            credit_spread = float(np.clip(
                credit_base[ticker] + rng.normal(0.0, 0.001), 0.0, 0.06
            ))

            # Coupon rate set at par (at issuance the bond was priced at par)
            coupon_rate = max(ytm - credit_spread - 0.005, 0.01)  # slight below-YTM coupon
            price = _bond_price(ytm, coupon_rate, maturity)

            records.append({
                "date": date,
                "ticker": ticker,
                "asset_class": "fixed_income",
                "price": price,
                "ytm": ytm,
                "duration": duration,
                "credit_spread": credit_spread,
            })
    return pd.DataFrame.from_records(records)


def build_mock_macro_series(start_date: str = "2017-01-01", end_date: str = "2026-03-31") -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="ME")
    np.random.seed(9)

    # Compute us_fed_rate: 5.25 through 2024-Q2, then -25bps per quarter, floor at 4.0
    us_fed_rate = []
    cutoff_date = pd.Timestamp("2024-06-30")  # End of Q2 2024
    for date in dates:
        if date <= cutoff_date:
            rate = 5.25
        else:
            # Quarters since 2024-Q3
            quarters_elapsed = (date.year - 2024) * 4 + (date.quarter - 3)
            rate = 5.25 - (quarters_elapsed * 0.0025)
            rate = max(rate, 4.0)
        us_fed_rate.append(rate)

    macro = pd.DataFrame(
        {
            "date": dates,
            "IMAI": np.clip(100 + np.cumsum(np.random.normal(0.15, 0.9, len(dates))), 80, None),
            "industrial_production_yoy": np.random.normal(0.04, 0.03, len(dates)),
            "exports_yoy": np.random.normal(0.06, 0.05, len(dates)),
            "usd_mxn": np.clip(19.5 + np.cumsum(np.random.normal(0.01, 0.1, len(dates))), 17.0, None),
            "banxico_rate": np.clip(4.0 + np.cumsum(np.random.normal(0.02, 0.1, len(dates))), 4.0, 12.0),
            "inflation_yoy": np.clip(0.03 + np.random.normal(0.0, 0.01, len(dates)), 0.02, 0.09),
            "us_ip_yoy": np.random.normal(0.03, 0.03, len(dates)),
            "us_fed_rate": us_fed_rate,
        }
    )
    return macro


def load_mock_data() -> Dict[str, pd.DataFrame]:
    universe = get_investable_universe()
    tickers = universe.loc[universe["investable"], "ticker"].tolist()
    prices = generate_mock_price_series(tickers)
    fundamentals = build_mock_fundamentals(tickers, pd.date_range(prices.index[0], prices.index[-1], freq="ME"))
    fibra_tickers = universe.loc[universe["asset_class"] == "fibra", "ticker"].tolist()
    fibra_fundamentals = build_mock_fibra_fundamentals(fibra_tickers, pd.date_range(prices.index[0], prices.index[-1], freq="ME"))
    bonds = build_mock_bonds(pd.date_range(prices.index[0], prices.index[-1], freq="ME"))
    macro = build_mock_macro_series(prices.index[0].strftime("%Y-%m-%d"), prices.index[-1].strftime("%Y-%m-%d"))
    return {
        "universe": universe,
        "prices": prices,
        "fundamentals": fundamentals,
        "fibra_fundamentals": fibra_fundamentals,
        "bonds": bonds,
        "macro": macro,
    }


def load_data(
    source: str = "mock",
    start_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    strict_data_mode: bool = False,
    fundamentals_lag_days: int = 90,
    **provider_kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Load strategy data from the specified source.

    Args:
        source: Data source — "mock", "yahoo", "bloomberg", or "refinitiv".
        start_date: Backtest start date (YYYY-MM-DD).
        end_date: Backtest end date (YYYY-MM-DD).
        strict_data_mode: If True, do NOT silently fall back to mock data
            when real data fails. Instead, log an error and return empty
            DataFrames. This ensures data integrity for production runs.
        fundamentals_lag_days: Number of calendar days to lag fundamentals
            data to prevent look-ahead bias (default: 90).
        **provider_kwargs: Passed to the provider constructor (e.g., api keys).

    Returns:
        Dict with keys: universe, prices, fundamentals, fibra_fundamentals, bonds, macro.
        Same schema as load_mock_data().
    """
    from .data_providers import get_provider

    if source == "mock":
        return load_mock_data()

    dropped_tickers: list[str] = []
    mock_fallbacks_used: list[str] = []

    universe = get_investable_universe()
    provider = get_provider(source, **provider_kwargs)

    equity_tickers = universe.loc[
        universe["investable"] & universe["asset_class"].isin(["equity", "fibra"]), "ticker"
    ].tolist()
    fibra_tickers = universe.loc[
        universe["investable"] & (universe["asset_class"] == "fibra"), "ticker"
    ].tolist()
    bond_tickers = universe.loc[
        universe["investable"] & (universe["asset_class"] == "fixed_income"), "ticker"
    ].tolist()

    prices = provider.get_prices(equity_tickers, start_date, end_date)
    if source != "mock" and not prices.empty:
        min_price_history = int(provider_kwargs.get("min_price_history", 126))
        valid_counts = prices.notna().sum()
        has_recent_price = prices.ffill(limit=5).tail(1).notna().iloc[0]
        valid_price_tickers = valid_counts.index[
            (valid_counts >= min_price_history) & has_recent_price.reindex(valid_counts.index).fillna(False)
        ].tolist()
        dropped_tickers = sorted(set(equity_tickers) - set(valid_price_tickers))
        if valid_price_tickers:
            prices = prices[valid_price_tickers]
            universe = universe[
                universe["asset_class"].eq("fixed_income") | universe["ticker"].isin(valid_price_tickers)
            ].reset_index(drop=True)
            equity_tickers = universe.loc[
                universe["investable"] & universe["asset_class"].isin(["equity", "fibra"]), "ticker"
            ].tolist()
            fibra_tickers = universe.loc[
                universe["investable"] & (universe["asset_class"] == "fibra"), "ticker"
            ].tolist()
            bond_tickers = universe.loc[
                universe["investable"] & (universe["asset_class"] == "fixed_income"), "ticker"
            ].tolist()
        if dropped_tickers:
            logger.info(
                "Dropping %d ticker(s) with insufficient %s history (<%d business days or stale last price): %s",
                len(dropped_tickers), source, min_price_history, dropped_tickers,
            )

    if source != "mock":
        try:
            mcaps = provider.get_market_caps(equity_tickers + fibra_tickers)
            # Update universe market_cap_mxn for available tickers
            universe["market_cap_mxn"] = universe["market_cap_mxn"].astype(float)
            universe.loc[universe["ticker"].isin(mcaps.index), "market_cap_mxn"] = (
                universe.loc[universe["ticker"].isin(mcaps.index), "ticker"].map(mcaps)
            )
            logger.info("Dynamic market caps updated from %s.", source)
        except Exception as e:
            logger.warning("Failed to load dynamic market caps from %s (%s). Using defaults.", source, e)

    try:
        fundamentals = provider.get_fundamentals(
            [t for t in equity_tickers if t not in fibra_tickers], start_date, end_date, allow_defaults=not strict_data_mode
        )
    except Exception as e:
        logger.error("Fundamentals load failed (%s). strict_data_mode=%s", e, strict_data_mode)
        mock_fallbacks_used.append("fundamentals")
        fundamentals = pd.DataFrame(columns=["date", "ticker", "pe_ratio", "pb_ratio", "roe",
                                              "profit_margin", "net_debt_to_ebitda", "ebitda_growth", "capex_to_sales"])

    try:
        fibra_fundamentals = provider.get_fibra_fundamentals(fibra_tickers, start_date, end_date, allow_defaults=not strict_data_mode)
    except Exception as e:
        logger.error("FIBRA fundamentals load failed (%s). strict_data_mode=%s", e, strict_data_mode)
        mock_fallbacks_used.append("fibra_fundamentals")
        fibra_fundamentals = pd.DataFrame(columns=["date", "ticker", "cap_rate", "ffo_yield",
                                                    "dividend_yield", "ltv", "vacancy_rate"])

    try:
        bonds = provider.get_bonds(bond_tickers, start_date, end_date)
    except Exception as e:
        logger.error("Bond data load failed (%s). strict_data_mode=%s", e, strict_data_mode)
        mock_fallbacks_used.append("bonds")
        bonds = pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])

    try:
        macro = provider.get_macro(start_date, end_date)
    except Exception as e:
        logger.error("Macro load failed (%s). strict_data_mode=%s", e, strict_data_mode)
        mock_fallbacks_used.append("macro")
        macro = pd.DataFrame(columns=["date", "IMAI", "industrial_production_yoy", "exports_yoy",
                                      "usd_mxn", "banxico_rate", "inflation_yoy", "us_ip_yoy", "us_fed_rate"])

    if fundamentals_lag_days > 0:
        if not fundamentals.empty and "date" in fundamentals.columns:
            fundamentals["date"] = pd.to_datetime(fundamentals["date"]) + pd.Timedelta(days=fundamentals_lag_days)
        if not fibra_fundamentals.empty and "date" in fibra_fundamentals.columns:
            fibra_fundamentals["date"] = pd.to_datetime(fibra_fundamentals["date"]) + pd.Timedelta(days=fundamentals_lag_days)

    data_integrity = {
        "source": source,
        "strict_data_mode": strict_data_mode,
        "dropped_tickers": dropped_tickers,
        "mock_fallbacks_used": mock_fallbacks_used,
        "fundamentals_lag_days": fundamentals_lag_days,
    }

    return {
        "universe": universe,
        "prices": prices,
        "fundamentals": fundamentals,
        "fibra_fundamentals": fibra_fundamentals,
        "bonds": bonds,
        "macro": macro,
        "data_integrity": data_integrity,
    }


# ---------------------------------------------------------------------------
# ETF Universe — versión por clase de activo (profe)
# ---------------------------------------------------------------------------

# Ticker → (nombre, sector, asset_class_pipeline, min_weight, max_weight)
# Fixed-income sleeve uses government bonds — no EMLC.
_ETF_SPECS: Dict[str, tuple] = {
    "INDUSTRIAL":    ("S&P/BMV IPC CompMX Industrial",    "Industrial",    "equity", 0.45, 0.65),
    "FIBRATC14":     ("FIBRA TC14",                        "FIBRA",         "fibra",  0.20, 0.35),
    "CONSUMER":      ("S&P/BMV IPC CompMX Consumer",       "Consumer",      "equity", 0.05, 0.15),
    "COMMUNICATION": ("S&P/BMV IPC CompMX Communication",  "Communication", "equity", 0.05, 0.15),
    "MATERIALS":     ("S&P/BMV IPC CompMX Materials",      "Materials",     "equity", 0.00, 0.10),
    # Government bonds — fixed income sleeve (complement, 0–30% combined)
    "CETES28": ("Cetes 28d", "Government", "fixed_income", 0.00, 0.15),
    "CETES91": ("Cetes 91d", "Government", "fixed_income", 0.00, 0.15),
    "MBONO3Y": ("Mbono 3yr", "Government", "fixed_income", 0.00, 0.15),
}

# Sector indices loaded from local Excel files (index/<Name>.xls) — no external ticker
_ETF_INDEX_TICKERS = ["INDUSTRIAL", "CONSUMER", "COMMUNICATION", "MATERIALS"]
# Tickers fetched live from external providers
_ETF_PRICE_TICKERS = ["FIBRATC14"]
# Bond tickers that go through get_bonds() / mock bonds
_ETF_BOND_TICKERS  = ["CETES28", "CETES91", "MBONO3Y"]

# Map index ticker → Excel filename (relative to project root)
_ETF_INDEX_FILES: Dict[str, str] = {
    "INDUSTRIAL":    "index/Industrial.xls",
    "CONSUMER":      "index/Consumer.xls",
    "COMMUNICATION": "index/Communication.xls",
    "MATERIALS":     "index/Materials.xls",
}


def get_etf_universe() -> pd.DataFrame:
    """Investable universe for the ETF version of the strategy.

    Equity ETFs have per-ticker min/max weights (the professor's allocation ranges).
    The fixed income sleeve (CETES28, CETES91, MBONO3Y) acts as the complement,
    replacing EMLC. Combined FI allocation is capped at 30%.
    """
    tickers   = list(_ETF_SPECS.keys())
    names     = [v[0] for v in _ETF_SPECS.values()]
    sectors   = [v[1] for v in _ETF_SPECS.values()]
    ac        = [v[2] for v in _ETF_SPECS.values()]
    min_w     = [v[3] for v in _ETF_SPECS.values()]
    max_w     = [v[4] for v in _ETF_SPECS.values()]

    n = len(tickers)
    # INDUSTRIAL, FIBRATC14, CONSUMER, COMMUNICATION, MATERIALS → MXN-denominated (0.0 USD)
    # CETES28, CETES91, MBONO3Y → fixed income (0.0 USD)
    usd_exp = [0.0] * n
    liq     = [1.0] * n

    df = pd.DataFrame({
        "ticker":              tickers,
        "name":                names,
        "sector":              sectors,
        "asset_class":         ac,
        "investable":          [True] * n,
        "usd_exposure":        usd_exp,
        "market_cap_mxn":      [0.0] * n,
        "liquidity_score":     liq,
        "thematic_purity":     ["pure"] * n,
        "issuer_id":           tickers,
        "max_position_override": max_w,
        "min_weight":          min_w,
        "max_weight":          max_w,
    })
    return df


def _load_index_prices_from_excel(
    start_date: str, end_date: str
) -> pd.DataFrame:
    """Load sector index price series from local Excel files.

    Each file has columns [date, price] where date is an Excel serial number
    and price is a rebased index (100 = first date). Returns a wide DataFrame
    indexed by business-day DatetimeIndex with one column per sector ticker.
    """
    import xlrd
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    frames = {}
    for ticker, rel_path in _ETF_INDEX_FILES.items():
        path = root / rel_path
        wb = xlrd.open_workbook(str(path))
        sh = wb.sheets()[0]
        rows = [sh.row_values(r) for r in range(1, sh.nrows)  # skip header
                if isinstance(sh.cell_value(r, 0), (int, float))]
        dates = [xlrd.xldate_as_datetime(r[0], wb.datemode).date() for r in rows]
        prices = [float(r[1]) for r in rows]
        s = pd.Series(prices, index=pd.DatetimeIndex(dates), name=ticker)
        frames[ticker] = s

    df = pd.DataFrame(frames).sort_index()
    bdays = pd.bdate_range(start_date, end_date)
    df = df.reindex(bdays).ffill(limit=5)
    return df.astype("float64")


def load_etf_data(
    source: str = "yahoo",
    start_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    **provider_kwargs,
) -> Dict[str, pd.DataFrame]:
    """Load price + bond data for the ETF universe from the given provider.

    Sector index prices (INDUSTRIAL, CONSUMER, COMMUNICATION, MATERIALS) are
    loaded from local Excel files. FIBRATC14 is fetched live from the provider.
    Government bond data reuses the existing bond loading pipeline.
    Returns a dict with the same schema as load_data() so run_etf_pipeline can
    consume it.
    """
    from .data_providers import get_provider

    universe = get_etf_universe()

    # Always load sector index prices from local Excel files
    index_prices = _load_index_prices_from_excel(start_date, end_date)

    if source == "mock":
        fibra_prices = generate_mock_price_series(
            _ETF_PRICE_TICKERS, start_date=start_date, end_date=end_date
        )
        bonds = build_mock_bonds(pd.date_range(start_date, end_date, freq="ME"))
        macro = build_mock_macro_series(start_date, end_date)
    else:
        provider = get_provider(source, **provider_kwargs)
        fibra_prices = provider.get_prices(_ETF_PRICE_TICKERS, start_date, end_date)
        try:
            bonds = provider.get_bonds(_ETF_BOND_TICKERS, start_date, end_date)
        except Exception as exc:
            logger.warning("ETF bond load failed (%s) — using mock bonds.", exc)
            bonds = build_mock_bonds(pd.date_range(start_date, end_date, freq="ME"))
        try:
            macro = provider.get_macro(start_date, end_date)
        except Exception as exc:
            logger.warning("ETF macro load failed (%s) — using mock macro.", exc)
            macro = build_mock_macro_series(start_date, end_date)

    # Cast to plain float64 — LSEG returns nullable Float64 which breaks np.log
    if not fibra_prices.empty:
        fibra_prices = fibra_prices.astype("float64")

    etf_prices = pd.concat([index_prices, fibra_prices], axis=1).sort_index()

    # Convert bond long-format to daily wide prices for the optimizer
    bond_prices = pd.DataFrame()
    if not bonds.empty and "price" in bonds.columns:
        bp = bonds.pivot_table(index="date", columns="ticker", values="price")
        bp.index = pd.DatetimeIndex(bp.index)
        bdays = pd.bdate_range(start_date, end_date)
        bond_prices = bp.reindex(bdays).ffill(limit=5).astype("float64")

    prices = pd.concat([etf_prices, bond_prices], axis=1).sort_index() if not bond_prices.empty else etf_prices

    empty_fund  = pd.DataFrame(columns=["date", "ticker", "pe_ratio", "pb_ratio", "roe",
                                         "profit_margin", "net_debt_to_ebitda", "ebitda_growth",
                                         "capex_to_sales"])
    empty_fibra = pd.DataFrame(columns=["date", "ticker", "cap_rate", "ffo_yield",
                                         "dividend_yield", "ltv", "vacancy_rate"])

    return {
        "universe":           universe,
        "prices":             prices,
        "fundamentals":       empty_fund,
        "fibra_fundamentals": empty_fibra,
        "bonds":              bonds,
        "macro":              macro,
        "data_integrity": {
            "source":               source,
            "strict_data_mode":     False,
            "dropped_tickers":      [],
            "mock_fallbacks_used":  [],
            "fundamentals_lag_days": 0,
        },
    }
