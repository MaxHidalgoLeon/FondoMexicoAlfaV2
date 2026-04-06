from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from typing import Dict, Tuple


def get_investable_universe() -> pd.DataFrame:
    """Create the initial thematic universe for Mexico industrial research."""
    tickers = [
        "GFM0", "FIBRA1", "LOGI2", "INDU3", "UTIL4",
        "TRANS5", "ENER6", "TECH7", "MEXI8", "ENGR9",
        "AUT10", "RWL11", "IND12", "FIBR13", "LOG14",
        "IND15", "FRA16", "STOR17", "CARG18", "ECO19",
    ]
    names = [
        "Grupo Fabricación", "Fibra Industrial Uno", "Logística Norte", "Industria MX", "Utility Industrial",
        "Transporte Integral", "Energía Manufacturera", "Tecnologías de Planta", "México Industrial", "Ingeniería 4.0",
        "Automotriz Supply", "Rieles Logísticos", "Industrial Desarrollo", "Fibra Park", "Logística Global",
        "Industria Sostenible", "FIBRA Renta", "Storage Infra", "Cargo Rex", "Energía Conexión",
    ]
    sectors = [
        "Industrial", "FIBRA", "Logistics", "Industrial", "Utilities",
        "Logistics", "Energy", "Industrial", "Industrial", "Industrial",
        "Industrial", "Logistics", "Industrial", "FIBRA", "Logistics",
        "Industrial", "FIBRA", "Logistics", "Logistics", "Energy",
    ]
    flags = [
        True, True, True, True, False,
        True, False, True, True, True,
        True, True, True, True, True,
        True, True, True, True, False,
    ]
    usd_exposure = [0.2, 0.4, 0.1, 0.35, 0.15, 0.25, 0.55, 0.1, 0.3, 0.2, 0.4, 0.2, 0.3, 0.45, 0.2, 0.25, 0.4, 0.05, 0.15, 0.6]
    market_caps = np.linspace(10_000, 120_000, len(tickers)).tolist()
    liquidity = np.linspace(0.25, 1.0, len(tickers)).tolist()
    df = pd.DataFrame({
        "ticker": tickers,
        "name": names,
        "sector": sectors,
        "investable": flags,
        "usd_exposure": usd_exposure,
        "market_cap_mxn": market_caps,
        "liquidity_score": liquidity,
    })
    return df


def generate_mock_price_series(
    tickers: list[str],
    start_date: str = "2018-01-01",
    end_date: str = "2025-12-31",
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
    records = []
    np.random.seed(24)
    for date in dates:
        for ticker in tickers:
            records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "ebitda_growth": np.random.normal(0.08, 0.12),
                    "net_debt_to_ebitda": max(0.0, np.random.normal(2.5, 1.1)),
                    "roe": np.random.normal(0.16, 0.08),
                    "profit_margin": np.random.normal(0.18, 0.1),
                    "capex_to_sales": np.random.normal(0.06, 0.03),
                    "pe_ratio": np.random.uniform(8, 24),
                    "pb_ratio": np.random.uniform(1.0, 3.5),
                }
            )
    return pd.DataFrame.from_records(records)


def build_mock_macro_series(start_date: str = "2018-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="ME")
    np.random.seed(9)
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
        }
    )
    return macro


def load_mock_data() -> Dict[str, pd.DataFrame]:
    universe = get_investable_universe()
    tickers = universe.loc[universe["investable"], "ticker"].tolist()
    prices = generate_mock_price_series(tickers)
    fundamentals = build_mock_fundamentals(tickers, pd.date_range(prices.index[0], prices.index[-1], freq="ME"))
    macro = build_mock_macro_series(prices.index[0].strftime("%Y-%m-%d"), prices.index[-1].strftime("%Y-%m-%d"))
    return {
        "universe": universe,
        "prices": prices,
        "fundamentals": fundamentals,
        "macro": macro,
    }
