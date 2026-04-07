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
        "CETES28", "CETES91", "MBONO3Y", "MBONO5Y", "MBONO10Y", "CORP1", "CORP2",
    ]
    names = [
        "Grupo Fabricación", "Fibra Industrial Uno", "Logística Norte", "Industria MX", "Utility Industrial",
        "Transporte Integral", "Energía Manufacturera", "Tecnologías de Planta", "México Industrial", "Ingeniería 4.0",
        "Automotriz Supply", "Rieles Logísticos", "Industrial Desarrollo", "Fibra Park", "Logística Global",
        "Industria Sostenible", "FIBRA Renta", "Storage Infra", "Cargo Rex", "Energía Conexión",
        "Cetes 28d", "Cetes 91d", "Mbono 3yr", "Mbono 5yr", "Mbono 10yr", "Corporate Bond 1", "Corporate Bond 2",
    ]
    sectors = [
        "Industrial", "FIBRA", "Logistics", "Industrial", "Utilities",
        "Logistics", "Energy", "Industrial", "Industrial", "Industrial",
        "Industrial", "Logistics", "Industrial", "FIBRA", "Logistics",
        "Industrial", "FIBRA", "Logistics", "Logistics", "Energy",
        "Government", "Government", "Government", "Government", "Government", "Corporate", "Corporate",
    ]
    asset_classes = [
        "equity", "fibra", "equity", "equity", "equity",
        "equity", "equity", "equity", "equity", "equity",
        "equity", "equity", "equity", "fibra", "equity",
        "equity", "fibra", "equity", "equity", "equity",
        "fixed_income", "fixed_income", "fixed_income", "fixed_income", "fixed_income", "fixed_income", "fixed_income",
    ]
    flags = [
        True, True, True, True, False,
        True, False, True, True, True,
        True, True, True, True, True,
        True, True, True, True, False,
        True, True, True, True, True, True, True,
    ]
    usd_exposure = [0.2, 0.4, 0.1, 0.35, 0.15, 0.25, 0.55, 0.1, 0.3, 0.2, 0.4, 0.2, 0.3, 0.45, 0.2, 0.25, 0.4, 0.05, 0.15, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1]
    market_caps = np.linspace(10_000, 120_000, 20).tolist() + [0] * 7  # bonds have no market cap
    liquidity = np.linspace(0.25, 1.0, 20).tolist() + [1.0] * 7  # bonds are liquid
    df = pd.DataFrame({
        "ticker": tickers,
        "name": names,
        "sector": sectors,
        "asset_class": asset_classes,
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


def build_mock_fibra_fundamentals(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    records = []
    np.random.seed(25)
    for date in dates:
        for ticker in tickers:
            records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "cap_rate": np.random.uniform(0.05, 0.12),
                    "ffo_yield": np.random.uniform(0.06, 0.15),
                    "dividend_yield": np.random.uniform(0.04, 0.10),
                    "ltv": np.random.uniform(0.3, 0.7),
                    "vacancy_rate": np.random.uniform(0.02, 0.15),
                }
            )
    return pd.DataFrame.from_records(records)
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


def build_mock_bonds(dates: pd.DatetimeIndex) -> pd.DataFrame:
    bond_tickers = ["CETES28", "CETES91", "MBONO3Y", "MBONO5Y", "MBONO10Y", "CORP1", "CORP2"]
    records = []
    np.random.seed(26)
    for date in dates:
        for ticker in bond_tickers:
            if "CETES" in ticker:
                duration = 0.08 if "28" in ticker else 0.25
                ytm = np.random.uniform(0.04, 0.08)
                credit_spread = 0.0
            elif "MBONO" in ticker:
                duration = 3.0 if "3Y" in ticker else 5.0 if "5Y" in ticker else 10.0
                ytm = np.random.uniform(0.06, 0.10)
                credit_spread = 0.0
            else:  # corporate
                duration = np.random.uniform(3.0, 7.0)
                ytm = np.random.uniform(0.08, 0.12)
                credit_spread = np.random.uniform(0.01, 0.03)
            price = 100 * (1 - ytm * duration / 2)  # rough approximation
            records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "asset_class": "fixed_income",
                    "price": price,
                    "ytm": ytm,
                    "duration": duration,
                    "credit_spread": credit_spread,
                }
            )
    return pd.DataFrame.from_records(records)


def build_mock_macro_series(start_date: str = "2018-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
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
