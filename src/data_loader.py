from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def compute_adtv_liquidity_scores(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 252,
) -> pd.Series:
    """
    Compute liquidity scores from real ADTV (Average Daily Traded Value).

    ADTV = mean(close_price * volume) over the last `window` trading days.
    Scores are normalized to [0.0, 1.0] using min-max scaling across the
    equity/FIBRA universe. Fixed-income tickers (not in prices) get 1.0.

    Returns a Series indexed by canonical ticker name.
    """
    common = prices.columns.intersection(volume.columns)
    adtv = (prices[common].tail(window) * volume[common].tail(window)).mean()
    score_min, score_max = adtv.min(), adtv.max()
    if score_max - score_min < 1e-9:
        scores = pd.Series(1.0, index=common)
    else:
        scores = (adtv - score_min) / (score_max - score_min)
    return scores


def get_investable_universe() -> pd.DataFrame:
    """Create the nearshoring industrial universe for México (verified on Yahoo Finance 2026-04)."""
    tickers = [
        # Equities — industrial / logistics / infrastructure / nearshoring
        "NEMAKA", "GISSAA", "CEMEXCPO", "ICHB", "GCARSOA1",
        "ASURB", "GAPB", "OMAB", "PINFRA", "ORBIA",
        "VESTA", "FEMSAUBD", "KIMBERA", "BIMBOA", "GRUMAB",
        # Equities — IPC industrials expanding cross-sectional model power
        "ALPEK", "GMEXICOB", "ALFA", "SIMECB", "RASSINIA", "BECLE",
        # FIBRAs — industrial / logistics focus
        "FUNO11", "FIBRAPL14", "FIBRAMQ12",
        # Fixed income — internal identifiers (always mock-generated)
        "CETES28", "CETES91", "MBONO3Y", "MBONO5Y", "MBONO10Y", "CORP1", "CORP2",
    ]
    names = [
        "Nemak", "Grupo Industrial Saltillo", "CEMEX", "Ternium México", "Grupo Carso",
        "Aeropuertos del Sureste", "Grupo Aeroportuario del Pacífico", "Grupo Aeroportuario Centro Norte", "Pinfra", "Orbia",
        "Vesta", "FEMSA", "Kimberly-Clark México", "Bimbo", "Gruma",
        "Alpek", "Grupo México", "Alfa", "Grupo Simec", "Rassini", "Becle (Cuervo)",
        "FIBRA Uno", "FIBRA Prologis", "FIBRA Macquarie",
        "Cetes 28d", "Cetes 91d", "Mbono 3yr", "Mbono 5yr", "Mbono 10yr", "Corporate Bond 1", "Corporate Bond 2",
    ]
    sectors = [
        "Industrial", "Industrial", "Industrial", "Industrial", "Industrial",
        "Logistics", "Logistics", "Logistics", "Infrastructure", "Industrial",
        "Industrial", "Consumer/Industrial", "Industrial", "Industrial", "Industrial",
        "Industrial", "Industrial", "Industrial", "Industrial", "Industrial", "Consumer/Industrial",
        "FIBRA", "FIBRA", "FIBRA",
        "Government", "Government", "Government", "Government", "Government", "Corporate", "Corporate",
    ]
    asset_classes = [
        "equity", "equity", "equity", "equity", "equity",
        "equity", "equity", "equity", "equity", "equity",
        "equity", "equity", "equity", "equity", "equity",
        "equity", "equity", "equity", "equity", "equity", "equity",
        "fibra", "fibra", "fibra",
        "fixed_income", "fixed_income", "fixed_income", "fixed_income", "fixed_income", "fixed_income", "fixed_income",
    ]
    investable = [
        True, True, True, True, True,
        True, True, True, True, True,
        True, True, True, True, True,
        True, True, True, True, True, True,
        True, True, True,
        True, True, True, True, True, True, True,
    ]
    # USD revenue exposure estimate (0.0–1.0) — nearshoring names have high USD exposure
    usd_exposure = [
        0.85, 0.70, 0.40, 0.75, 0.30,   # NEMAKA, GISSAA, CEMEX, ICHB, GCARSOA1
        0.30, 0.25, 0.30, 0.20, 0.50,   # ASURB, GAPB, OMAB, PINFRA, ORBIA
        0.65, 0.35, 0.20, 0.15, 0.35,   # VESTA, FEMSA, KIMBERA, BIMBOA, GRUMAB
        0.60, 0.70, 0.35, 0.55, 0.80, 0.75,  # ALPEK, GMEXICOB, ALFA, SIMECB, RASSINIA, BECLE
        0.50, 0.80, 0.55,               # FUNO11, FIBRAPL14, FIBRAMQ12
        0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1,  # bonds
    ]
    # Approximate market caps in MXN millions (Q1 2026 estimates, bonds = 0)
    market_caps = [
        18_000, 12_000, 250_000, 95_000, 180_000,
        120_000, 100_000, 60_000, 85_000, 55_000,
        35_000, 300_000, 90_000, 220_000, 75_000,
        42_000, 380_000, 38_000, 22_000, 7_500, 125_000,
        140_000, 50_000, 30_000,
        0, 0, 0, 0, 0, 0, 0,
    ]
    # Liquidity score (0.0–1.0) based on average daily traded value
    liquidity = [
        0.55, 0.45, 0.95, 0.70, 0.80,
        0.85, 0.80, 0.65, 0.75, 0.60,
        0.50, 1.00, 0.75, 0.90, 0.70,
        0.62, 0.88, 0.56, 0.53, 0.47, 0.70,
        0.90, 0.65, 0.55,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
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
    bond_tickers = ["CETES28", "CETES91", "MBONO3Y", "MBONO5Y", "MBONO10Y", "CORP1", "CORP2"]
    records = []
    rng = np.random.default_rng(26)
    # Base YTM per bond type with AR persistence
    base_ytm = {
        "CETES28": 0.055, "CETES91": 0.058,
        "MBONO3Y": 0.075, "MBONO5Y": 0.080, "MBONO10Y": 0.085,
        "CORP1": 0.095, "CORP2": 0.105,
    }
    ytm_state = dict(base_ytm)
    credit_base = {"CETES28": 0.0, "CETES91": 0.0, "MBONO3Y": 0.0, "MBONO5Y": 0.0, "MBONO10Y": 0.0, "CORP1": 0.015, "CORP2": 0.022}
    for date in dates:
        for ticker in bond_tickers:
            # Durations (fixed contract characteristics)
            if "CETES28" in ticker:
                duration = 28 / 365
                maturity = duration
            elif "CETES91" in ticker:
                duration = 91 / 365
                maturity = duration
            elif "MBONO3Y" in ticker:
                duration = 2.7
                maturity = 3.0
            elif "MBONO5Y" in ticker:
                duration = 4.3
                maturity = 5.0
            elif "MBONO10Y" in ticker:
                duration = 7.5
                maturity = 10.0
            elif ticker == "CORP1":
                duration = 3.5
                maturity = 4.0
            else:  # CORP2
                duration = 5.5
                maturity = 7.0

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
    **provider_kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Load strategy data from the specified source.

    Args:
        source: Data source — "mock", "yahoo", "bloomberg", or "refinitiv".
        start_date: Backtest start date (YYYY-MM-DD).
        end_date: Backtest end date (YYYY-MM-DD).
        **provider_kwargs: Passed to the provider constructor (e.g., api keys).

    Returns:
        Dict with keys: universe, prices, fundamentals, fibra_fundamentals, bonds, macro.
        Same schema as load_mock_data().
    """
    from .data_providers import get_provider

    if source == "mock":
        return load_mock_data()

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

    import warnings

    prices = provider.get_prices(equity_tickers, start_date, end_date)

    try:
        fundamentals = provider.get_fundamentals(
            [t for t in equity_tickers if t not in fibra_tickers], start_date, end_date
        )
    except NotImplementedError as e:
        warnings.warn(f"Fundamentals not available from {source}: {e}. Falling back to mock.")
        from .data_loader import build_mock_fundamentals
        fundamentals = build_mock_fundamentals(
            [t for t in equity_tickers if t not in fibra_tickers],
            pd.date_range(start_date, end_date, freq="ME")
        )

    try:
        fibra_fundamentals = provider.get_fibra_fundamentals(fibra_tickers, start_date, end_date)
    except NotImplementedError as e:
        warnings.warn(f"FIBRA fundamentals not available from {source}: {e}. Falling back to mock.")
        from .data_loader import build_mock_fibra_fundamentals
        fibra_fundamentals = build_mock_fibra_fundamentals(
            fibra_tickers, pd.date_range(start_date, end_date, freq="ME")
        )

    try:
        bonds = provider.get_bonds(bond_tickers, start_date, end_date)
    except NotImplementedError as e:
        warnings.warn(f"Bond data not available from {source}: {e}. Falling back to mock.")
        from .data_loader import build_mock_bonds
        bonds = build_mock_bonds(pd.date_range(start_date, end_date, freq="ME"))

    try:
        macro = provider.get_macro(start_date, end_date)
    except NotImplementedError as e:
        warnings.warn(f"Macro data not available from {source}: {e}. Falling back to mock.")
        from .data_loader import build_mock_macro_series
        macro = build_mock_macro_series(start_date=start_date, end_date=end_date)

    return {
        "universe": universe,
        "prices": prices,
        "fundamentals": fundamentals,
        "fibra_fundamentals": fibra_fundamentals,
        "bonds": bonds,
        "macro": macro,
    }
