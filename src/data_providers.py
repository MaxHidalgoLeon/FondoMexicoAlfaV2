"""Multi-source data provider abstraction for the Mexico quant strategy."""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared ticker map loader
# ---------------------------------------------------------------------------

def _load_ticker_map() -> Dict[str, Dict[str, str | None]]:
    """Load config/ticker_map.yaml and return {canonical: {provider: symbol}}."""
    map_path = Path(__file__).resolve().parent.parent / "config" / "ticker_map.yaml"
    if not map_path.exists():
        return {}
    try:
        import yaml
        with open(map_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


def _resolve_symbols(tickers: List[str], provider: str, suffix: str) -> Dict[str, str]:
    """
    Build {provider_symbol: canonical} for a list of canonical tickers.

    For each ticker:
    - If found in ticker_map.yaml and the provider key is non-null → use that value.
    - Otherwise fall back to canonical.
    - Provider suffix is appended only when needed (not for already-native symbols).
    - Tickers with a null provider entry (bonds) are skipped.
    """
    def _apply_suffix(raw_symbol: str) -> str:
        if not suffix:
            return raw_symbol
        if raw_symbol.endswith(suffix):
            return raw_symbol

        # Yahoo indices like ^MXX are already fully qualified.
        if provider == "yahoo" and raw_symbol.startswith("^"):
            return raw_symbol

        # Bloomberg symbols that already include asset class should not be modified.
        if provider == "bloomberg" and any(x in raw_symbol for x in (" Equity", " Index", " Govt")):
            return raw_symbol

        # LSEG RICs are provider-native when they already contain common RIC markers.
        if provider == "lseg" and ("=" in raw_symbol or "." in raw_symbol):
            return raw_symbol

        return f"{raw_symbol}{suffix}"

    ticker_map = _load_ticker_map()
    result: Dict[str, str] = {}
    for canonical in tickers:
        entry = ticker_map.get(canonical, {})
        symbol = entry.get(provider) if entry else None
        if symbol is None and entry and entry.get(provider) is None and provider in entry:
            # Explicitly set to null in YAML — skip (bonds, internal ids)
            continue
        raw = symbol if symbol is not None else canonical
        resolved = _apply_suffix(str(raw).strip())
        result[resolved] = canonical
    return result


def _fill_numeric_defaults(df: pd.DataFrame, defaults: Dict[str, float]) -> pd.DataFrame:
    """Replace sparse Yahoo fundamentals with stable medians/defaults."""
    filled = df.copy()
    for col, default in defaults.items():
        if col not in filled.columns:
            continue
        series = pd.to_numeric(filled[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = series.dropna().median()
        fill_value = float(median) if pd.notna(median) else float(default)
        filled[col] = series.fillna(fill_value)
    return filled


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseDataProvider(ABC):
    """Abstract interface for all data providers."""

    @abstractmethod
    def get_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return adjusted close prices as a business-day DataFrame (dates × tickers)."""

    @abstractmethod
    def get_fundamentals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return long-format fundamentals: date, ticker, pe_ratio, pb_ratio, roe,
        profit_margin, net_debt_to_ebitda, ebitda_growth, capex_to_sales."""

    @abstractmethod
    def get_macro(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return monthly macro DataFrame: date, IMAI, industrial_production_yoy,
        exports_yoy, usd_mxn, banxico_rate, inflation_yoy, us_ip_yoy, us_fed_rate."""

    @abstractmethod
    def get_fibra_fundamentals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return long-format FIBRA metrics: date, ticker, cap_rate, ffo_yield,
        dividend_yield, ltv, vacancy_rate."""

    @abstractmethod
    def get_bonds(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return long-format bond data: date, ticker, asset_class, price, ytm,
        duration, credit_spread."""


# ---------------------------------------------------------------------------
# Mock provider — delegates to existing data_loader functions
# ---------------------------------------------------------------------------

class MockDataProvider(BaseDataProvider):
    """Routes all requests to the mock generators in data_loader."""

    def __init__(self) -> None:
        from .data_loader import (  # noqa: F401 — validated at init time
            generate_mock_price_series,
            build_mock_fundamentals,
            build_mock_fibra_fundamentals,
            build_mock_bonds,
            build_mock_macro_series,
            get_investable_universe,
        )

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from .data_loader import generate_mock_price_series
        return generate_mock_price_series(tickers, start_date=start_date, end_date=end_date)

    def get_fundamentals(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from .data_loader import build_mock_fundamentals
        dates = pd.date_range(start_date, end_date, freq="ME")
        return build_mock_fundamentals(tickers, dates)

    def get_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        from .data_loader import build_mock_macro_series
        return build_mock_macro_series(start_date=start_date, end_date=end_date)

    def get_fibra_fundamentals(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        from .data_loader import build_mock_fibra_fundamentals
        dates = pd.date_range(start_date, end_date, freq="ME")
        return build_mock_fibra_fundamentals(tickers, dates)

    def get_bonds(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from .data_loader import build_mock_bonds
        dates = pd.date_range(start_date, end_date, freq="ME")
        return build_mock_bonds(dates)


# ---------------------------------------------------------------------------
# Yahoo Finance provider
# ---------------------------------------------------------------------------

class YahooFinanceProvider(BaseDataProvider):
    """Fetches data via the yfinance library (equities/FIBRAs on BMV)."""

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            raise ImportError("Install yfinance: pip install yfinance")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mx_tickers(tickers: List[str]) -> dict:
        """Map canonical tickers to Yahoo Finance symbols via ticker_map.yaml (fallback: append .MX)."""
        return _resolve_symbols(tickers, "yahoo", ".MX")

    @staticmethod
    def _forward_fill_prices(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        return df.ffill(limit=limit)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import yfinance as yf

        mapping = self._to_mx_tickers(tickers)
        mx_tickers = list(mapping.keys())

        raw = yf.download(
            mx_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False,
        )["Close"]

        # yfinance may return a Series when only one ticker is requested
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=mx_tickers[0])

        # Rename .MX tickers back to originals
        raw.columns = [mapping.get(str(c), str(c)) for c in raw.columns]
        raw.index = pd.DatetimeIndex(raw.index)

        # Reindex to business days and forward-fill illiquid sessions
        bdays = pd.bdate_range(start_date, end_date)
        raw = raw.reindex(bdays).pipe(self._forward_fill_prices)
        return raw

    def get_volume(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Return daily traded volume (shares) as a business-day DataFrame (dates × tickers)."""
        import yfinance as yf

        mapping = self._to_mx_tickers(tickers)
        mx_tickers = list(mapping.keys())

        raw = yf.download(
            mx_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False,
        )["Volume"]

        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=mx_tickers[0])

        raw.columns = [mapping.get(str(c), str(c)) for c in raw.columns]
        raw.index = pd.DatetimeIndex(raw.index)
        bdays = pd.bdate_range(start_date, end_date)
        raw = raw.reindex(bdays).fillna(0.0)
        return raw

    def get_fundamentals(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import yfinance as yf

        mapping = self._to_mx_tickers(tickers)  # {yahoo_symbol: canonical}
        canonical_to_yahoo = {v: k for k, v in mapping.items()}
        today = pd.Timestamp.today().normalize()
        records = []
        for ticker in tickers:
            yahoo_symbol = canonical_to_yahoo.get(ticker, f"{ticker}.MX")
            try:
                info = yf.Ticker(yahoo_symbol).info
            except Exception:
                info = {}

            total_debt = info.get("totalDebt") or np.nan
            ebitda = info.get("ebitda") or np.nan
            capex = info.get("capitalExpenditures") or np.nan
            revenue = info.get("totalRevenue") or np.nan

            net_debt_to_ebitda = (
                total_debt / ebitda
                if (not math.isnan(total_debt) and not math.isnan(ebitda) and ebitda != 0)
                else np.nan
            )
            capex_to_sales = (
                abs(capex) / revenue
                if (not math.isnan(capex) and not math.isnan(revenue) and revenue != 0)
                else np.nan
            )

            records.append({
                "date": today,
                "ticker": ticker,
                "pe_ratio": info.get("trailingPE", np.nan),
                "pb_ratio": info.get("priceToBook", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
                "profit_margin": info.get("profitMargins", np.nan),
                "net_debt_to_ebitda": net_debt_to_ebitda,
                "ebitda_growth": info.get("revenueGrowth", np.nan),
                "capex_to_sales": capex_to_sales,
            })
        return _fill_numeric_defaults(
            pd.DataFrame.from_records(records),
            {
                "pe_ratio": 14.0,
                "pb_ratio": 1.8,
                "roe": 0.12,
                "profit_margin": 0.08,
                "net_debt_to_ebitda": 2.5,
                "ebitda_growth": 0.05,
                "capex_to_sales": 0.05,
            },
        )

    def get_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch macro from FRED + Banxico SIE + INEGI (all free, no Bloomberg/Refinitiv needed)."""
        provider = FREDBanxicoMacroProvider()
        return provider.get_macro(start_date, end_date)

    def get_fibra_fundamentals(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        import yfinance as yf

        mapping = self._to_mx_tickers(tickers)  # {yahoo_symbol: canonical}
        canonical_to_yahoo = {v: k for k, v in mapping.items()}
        today = pd.Timestamp.today().normalize()
        records = []
        for ticker in tickers:
            yahoo_symbol = canonical_to_yahoo.get(ticker, f"{ticker}.MX")
            try:
                info = yf.Ticker(yahoo_symbol).info
            except Exception:
                info = {}

            dividend_yield = info.get("dividendYield", np.nan)
            if pd.notna(dividend_yield):
                cap_rate = dividend_yield + 0.01
                ffo_yield = dividend_yield
            else:
                cap_rate = np.nan
                ffo_yield = np.nan

            records.append({
                "date": today,
                "ticker": ticker,
                "cap_rate": cap_rate,
                "ffo_yield": ffo_yield,
                "dividend_yield": dividend_yield,
                "ltv": np.nan,
                "vacancy_rate": np.nan,
            })
        return _fill_numeric_defaults(
            pd.DataFrame.from_records(records),
            {
                "cap_rate": 0.075,
                "ffo_yield": 0.065,
                "dividend_yield": 0.055,
                "ltv": 0.35,
                "vacancy_rate": 0.08,
            },
        )

    def get_bonds(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from .data_loader import build_mock_bonds

        govt_tickers = {"CETES28", "CETES91", "MBONO3Y", "MBONO5Y", "MBONO10Y"}
        requested_govt = [t for t in tickers if t in govt_tickers]
        requested_corp = [t for t in tickers if t not in govt_tickers]

        frames = []

        # --- Government bonds: real yields from Banxico SIE ---
        if requested_govt:
            macro_provider = FREDBanxicoMacroProvider()
            govt_df = macro_provider.fetch_bond_yields(start_date, end_date)
            if not govt_df.empty:
                govt_df = govt_df[govt_df["ticker"].isin(requested_govt)]
                if not govt_df.empty:
                    frames.append(govt_df)

        # --- Corporate bonds: mock until real tickers are configured ---
        if requested_corp:
            dates = pd.date_range(start_date, end_date, freq="ME")
            mock_df = build_mock_bonds(dates)
            corp_df = mock_df[mock_df["ticker"].isin(requested_corp)]
            if not corp_df.empty:
                frames.append(corp_df)

        if not frames:
            return pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Bloomberg provider
# ---------------------------------------------------------------------------

# Bond ticker mapping: internal name → Bloomberg ticker
_BBG_BOND_TICKERS: dict = {
    "CETES28": "CETES 28D Govt",
    "CETES91": "CETES 91D Govt",
    "MBONO3Y": "MBONO 3Y Govt",
    "MBONO5Y": "MBONO 5Y Govt",
    "MBONO10Y": "MBONO 10Y Govt",
    "CORP1": "MBCORP1 Corp",
    "CORP2": "MBCORP2 Corp",
}

_BBG_MACRO_TICKERS: dict = {
    "MMACTIVI Index": "IMAI",
    "MXIPYOY Index": "industrial_production_yoy",
    "MXEXPORT Index": "exports_yoy",
    "USDMXN Curncy": "usd_mxn",
    "MXONBRAN Index": "banxico_rate",
    "MXCPYOY Index": "inflation_yoy",
    "IP YOY Index": "us_ip_yoy",
    "FDTR Index": "us_fed_rate",
}

_RATE_COLUMNS = {"banxico_rate", "inflation_yoy", "us_ip_yoy", "us_fed_rate", "industrial_production_yoy", "exports_yoy"}


class BloombergProvider(BaseDataProvider):
    """Fetches data via xbbg (blpapi wrapper)."""

    def __init__(self) -> None:
        try:
            from xbbg import blp  # noqa: F401
        except ImportError:
            raise ImportError("Install xbbg and blpapi: pip install xbbg")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _equity_bbg(tickers: List[str]) -> dict:
        """Map canonical tickers to Bloomberg equity tickers via ticker_map.yaml (fallback: append ' MM Equity')."""
        return _resolve_symbols(tickers, "bloomberg", " MM Equity")

    @staticmethod
    def _collapse_multiindex(df: pd.DataFrame, reverse_map: dict) -> pd.DataFrame:
        """Flatten a MultiIndex column DataFrame produced by blp.bdh."""
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df.columns = [reverse_map.get(str(c), str(c)) for c in df.columns]
        return df

    @staticmethod
    def _forward_fill_prices(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        return df.ffill(limit=limit)

    @staticmethod
    def _maybe_divide_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Divide a rate column by 100 when Bloomberg returns it as a percentage."""
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0 and series.abs().median() > 1.0:
                df[col] = df[col] / 100.0
        return df

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from xbbg import blp

        mapping = self._equity_bbg(tickers)
        bbg_tickers = list(mapping.keys())

        # Request split/dividend adjusted history when available.
        raw = blp.bdh(
            bbg_tickers,
            "PX_LAST",
            start_date,
            end_date,
            CshAdjNormal=True,
            CshAdjAbnormal=True,
            CapChg=True,
        )
        raw = self._collapse_multiindex(raw, {v: v for v in mapping.values()})

        bdays = pd.bdate_range(start_date, end_date)
        raw.index = pd.DatetimeIndex(raw.index)
        raw = raw.reindex(bdays).pipe(self._forward_fill_prices)
        return raw

    def get_fundamentals(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from xbbg import blp

        mapping = self._equity_bbg(tickers)
        fields = [
            "PE_RATIO", "PX_TO_BOOK_RATIO", "RETURN_ON_EQUITY",
            "PROF_MARGIN", "NET_DEBT_TO_EBITDA", "EBITDA_GROWTH", "CAPEX_TO_SALES",
        ]
        field_rename = {
            "PE_RATIO": "pe_ratio",
            "PX_TO_BOOK_RATIO": "pb_ratio",
            "RETURN_ON_EQUITY": "roe",
            "PROF_MARGIN": "profit_margin",
            "NET_DEBT_TO_EBITDA": "net_debt_to_ebitda",
            "EBITDA_GROWTH": "ebitda_growth",
            "CAPEX_TO_SALES": "capex_to_sales",
        }

        records = []
        for bbg_ticker, orig_ticker in mapping.items():
            raw = blp.bdh(bbg_ticker, fields, start_date, end_date, Per="M")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=field_rename)
            raw["ticker"] = orig_ticker
            raw.index.name = "date"
            raw = raw.reset_index()
            records.append(raw)

        if not records:
            return pd.DataFrame(columns=["date", "ticker"] + list(field_rename.values()))
        return pd.concat(records, ignore_index=True)

    def get_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        from xbbg import blp

        bbg_tickers = list(_BBG_MACRO_TICKERS.keys())
        raw = blp.bdh(bbg_tickers, "PX_LAST", start_date, end_date, Per="M")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns=_BBG_MACRO_TICKERS)
        raw.index.name = "date"
        raw = raw.reset_index()

        for col in _RATE_COLUMNS:
            raw = self._maybe_divide_rates(raw, col)

        return raw

    def get_fibra_fundamentals(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        from xbbg import blp

        mapping = self._equity_bbg(tickers)
        fields = ["CAP_RATE", "FFO_YIELD", "DVD_SH_12M", "LOAN_TO_VALUE", "VACANCY_RATE"]
        field_rename = {
            "CAP_RATE": "cap_rate",
            "FFO_YIELD": "ffo_yield",
            "DVD_SH_12M": "dividend_yield",
            "LOAN_TO_VALUE": "ltv",
            "VACANCY_RATE": "vacancy_rate",
        }

        records = []
        for bbg_ticker, orig_ticker in mapping.items():
            raw = blp.bdh(bbg_ticker, fields, start_date, end_date, Per="M")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=field_rename)
            raw["ticker"] = orig_ticker
            raw.index.name = "date"
            raw = raw.reset_index()
            records.append(raw)

        if not records:
            return pd.DataFrame(columns=["date", "ticker"] + list(field_rename.values()))
        return pd.concat(records, ignore_index=True)

    def get_bonds(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from xbbg import blp

        fields = ["PX_LAST", "YLD_YTM_MID", "DUR_MID", "Z_SPRD_MID"]
        field_rename = {
            "PX_LAST": "price",
            "YLD_YTM_MID": "ytm",
            "DUR_MID": "duration",
            "Z_SPRD_MID": "credit_spread",
        }

        records = []
        for ticker in tickers:
            bbg_ticker = _BBG_BOND_TICKERS.get(ticker)
            if bbg_ticker is None:
                continue
            raw = blp.bdh(bbg_ticker, fields, start_date, end_date, Per="M")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=field_rename)
            raw["ticker"] = ticker
            raw["asset_class"] = "fixed_income"
            raw.index.name = "date"
            raw = raw.reset_index()
            records.append(raw)

        if not records:
            return pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])
        return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Refinitiv / LSEG provider
# ---------------------------------------------------------------------------

_RD_BOND_RICS: dict = {
    "CETES28": "MX0001Y=",
    "CETES91": "MX0003M=",
    "MBONO3Y": "MX3YT=RR",
    "MBONO5Y": "MX5YT=RR",
    "MBONO10Y": "MX10YT=RR",
}

_RD_MACRO_RICS: dict = {
    "MXIMAI=ECI": "IMAI",
    "MXIP=ECI": "industrial_production_yoy",
    "MXEX=ECI": "exports_yoy",
    "MXN=": "usd_mxn",
    "MXCBRATE=ECI": "banxico_rate",
    "MXCPI=ECI": "inflation_yoy",
    "USIP=ECI": "us_ip_yoy",
    "USFEDFS=ECI": "us_fed_rate",
}


class RefinitivProvider(BaseDataProvider):
    """Fetches data via the LSEG Data Library (lseg-data)."""

    _session_opened: bool = False

    def __init__(self) -> None:
        try:
            import lseg.data as ld  # noqa: F401
        except ImportError:
            raise ImportError("Install LSEG Data Library: pip install lseg-data")

    def _ensure_session(self) -> None:
        import lseg.data as ld
        if not RefinitivProvider._session_opened:
            ld.open_session()
            RefinitivProvider._session_opened = True

    @staticmethod
    def _to_rics(tickers: List[str]) -> dict:
        """Map canonical tickers to LSEG RICs via ticker_map.yaml (fallback: append .MX)."""
        return _resolve_symbols(tickers, "lseg", ".MX")

    @staticmethod
    def _forward_fill_prices(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        return df.ffill(limit=limit)

    @staticmethod
    def _maybe_divide_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0 and series.abs().median() > 1.0:
                df[col] = df[col] / 100.0
        return df

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import lseg.data as ld

        self._ensure_session()
        mapping = self._to_rics(tickers)
        rics = list(mapping.keys())

        # Try multiple field names because support differs across LSEG environments/universes.
        # Order: adjusted close first, then common close aliases.
        price_fields = [
            "TR.CLOSEPRICE(Adjusted=1)",
            "TRDPRC_1",
            "TR.CLOSEPRICE",
            "CF_CLOSE",
            "CLOSE",
        ]
        raw = None
        value_field = None
        last_exc: Exception | None = None

        for field in price_fields:
            try:
                candidate = ld.get_history(universe=rics, fields=[field], start=start_date, end=end_date)
                if candidate is None or candidate.empty:
                    continue

                # Accept the first field that returns non-empty history.
                raw = candidate
                value_field = field
                break
            except Exception as exc:
                last_exc = exc
                continue

        if raw is None or value_field is None:
            tried = ", ".join(price_fields)
            msg = f"Refinitiv get_prices failed for all candidate fields: {tried}"
            if last_exc is not None:
                raise RuntimeError(f"{msg}. Last error: {last_exc}") from last_exc
            raise RuntimeError(msg)

        def _find_value_col(df: pd.DataFrame, requested: str) -> str | None:
            candidates = [
                requested,
                requested.upper(),
                "TRDPRC_1",
                "TR.CLOSEPRICE",
                "CF_CLOSE",
                "CLOSE",
            ]
            if isinstance(df.columns, pd.MultiIndex):
                cols = list(df.columns.get_level_values(-1))
            else:
                cols = list(df.columns)
            for c in candidates:
                if c in cols:
                    return c
            return None

        # Pivot to wide format if necessary
        if isinstance(raw.index, pd.MultiIndex):
            flat = raw.reset_index()
            value_col = _find_value_col(flat, value_field)
            if value_col is None:
                raise ValueError(f"No recognized price column found for Refinitiv prices. Columns: {list(flat.columns)}")
            raw = flat.pivot(index="Date", columns="Instrument", values=value_col)
        elif "Instrument" in raw.columns:
            flat = raw.reset_index()
            value_col = _find_value_col(flat, value_field)
            if value_col is None:
                raise ValueError(f"No recognized price column found for Refinitiv prices. Columns: {list(flat.columns)}")
            raw = flat.pivot(index="Date", columns="Instrument", values=value_col)
        else:
            raw.index.name = "Date"

        raw.index = pd.DatetimeIndex(raw.index)
        raw.columns = [mapping.get(str(c), str(c)) for c in raw.columns]
        raw = raw.apply(pd.to_numeric, errors="coerce")

        bdays = pd.bdate_range(start_date, end_date)
        raw = raw.reindex(bdays).pipe(self._forward_fill_prices)
        return raw

    def get_fundamentals(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import lseg.data as ld

        self._ensure_session()
        mapping = self._to_rics(tickers)
        rics = list(mapping.keys())
        fields = [
            "TR.PE", "TR.PriceToBVPerShare", "TR.ROE", "TR.NetProfitMarginPct",
            "TR.TotalDebtToEBITDA", "TR.RevenueGrowth", "TR.CapexToRevenue",
        ]
        field_rename = {
            "TR.PE": "pe_ratio",
            "TR.PriceToBVPerShare": "pb_ratio",
            "TR.ROE": "roe",
            "TR.NetProfitMarginPct": "profit_margin",
            "TR.TotalDebtToEBITDA": "net_debt_to_ebitda",
            "TR.RevenueGrowth": "ebitda_growth",
            "TR.CapexToRevenue": "capex_to_sales",
        }

        raw = ld.get_history(
            universe=rics,
            fields=fields,
            start=start_date,
            end=end_date,
            interval="quarterly",
        )

        records = []
        for ric, orig_ticker in mapping.items():
            if ric not in raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns:
                continue
            try:
                ticker_df = raw[ric].copy() if isinstance(raw.columns, pd.MultiIndex) else raw
            except (KeyError, TypeError):
                continue

            ticker_df = ticker_df.rename(columns=field_rename)
            # Forward-fill quarterly to monthly cadence
            monthly_idx = pd.date_range(start_date, end_date, freq="ME")
            ticker_df.index = pd.DatetimeIndex(ticker_df.index)
            ticker_df = ticker_df[~ticker_df.index.duplicated(keep="last")]
            ticker_df = ticker_df.reindex(ticker_df.index.union(monthly_idx)).ffill().reindex(monthly_idx)
            ticker_df["ticker"] = orig_ticker
            ticker_df.index.name = "date"
            ticker_df = ticker_df.reset_index()
            records.append(ticker_df)

        if not records:
            return pd.DataFrame(columns=["date", "ticker"] + list(field_rename.values()))
        return pd.concat(records, ignore_index=True)

    def get_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        import lseg.data as ld

        self._ensure_session()
        rics = list(_RD_MACRO_RICS.keys())

        raw = ld.get_history(universe=rics, fields=["CLOSE"], start=start_date, end=end_date)

        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index().pivot(index="Date", columns="Instrument", values="CLOSE")
        raw.index = pd.DatetimeIndex(raw.index)
        raw = raw.rename(columns=_RD_MACRO_RICS)

        # MXN= is MXN per 1 USD; take reciprocal to get USD/MXN
        if "usd_mxn" in raw.columns:
            raw["usd_mxn"] = 1.0 / raw["usd_mxn"].replace(0, np.nan)

        for col in _RATE_COLUMNS:
            raw = self._maybe_divide_rates(raw, col)

        raw.index.name = "date"
        raw = raw.reset_index()
        return raw

    def get_fibra_fundamentals(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        import lseg.data as ld

        self._ensure_session()
        mapping = self._to_rics(tickers)
        rics = list(mapping.keys())
        fields = [
            "TR.CapRate", "TR.FFOYield", "TR.DividendYield",
            "TR.LoanToValue", "TR.VacancyRatePercent",
        ]
        field_rename = {
            "TR.CapRate": "cap_rate",
            "TR.FFOYield": "ffo_yield",
            "TR.DividendYield": "dividend_yield",
            "TR.LoanToValue": "ltv",
            "TR.VacancyRatePercent": "vacancy_rate",
        }

        raw = ld.get_history(
            universe=rics,
            fields=fields,
            start=start_date,
            end=end_date,
            interval="quarterly",
        )

        records = []
        for ric, orig_ticker in mapping.items():
            try:
                ticker_df = raw[ric].copy() if isinstance(raw.columns, pd.MultiIndex) else raw
            except (KeyError, TypeError):
                continue

            ticker_df = ticker_df.rename(columns=field_rename)
            monthly_idx = pd.date_range(start_date, end_date, freq="ME")
            ticker_df.index = pd.DatetimeIndex(ticker_df.index)
            ticker_df = ticker_df[~ticker_df.index.duplicated(keep="last")]
            ticker_df = ticker_df.reindex(ticker_df.index.union(monthly_idx)).ffill().reindex(monthly_idx)
            ticker_df["ticker"] = orig_ticker
            ticker_df.index.name = "date"
            ticker_df = ticker_df.reset_index()
            records.append(ticker_df)

        if not records:
            return pd.DataFrame(columns=["date", "ticker"] + list(field_rename.values()))
        return pd.concat(records, ignore_index=True)

    def get_bonds(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import lseg.data as ld

        self._ensure_session()
        fields = ["YIELD", "MDURATION", "ZSPREAD", "PRICE"]
        field_rename = {
            "YIELD": "ytm",
            "MDURATION": "duration",
            "ZSPREAD": "credit_spread",
            "PRICE": "price",
        }

        records = []
        for ticker in tickers:
            ric = _RD_BOND_RICS.get(ticker)
            if ric is None:
                continue
            raw = ld.get_history(universe=[ric], fields=fields, start=start_date, end=end_date)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw[ric]
            raw = raw.rename(columns=field_rename)
            raw["ticker"] = ticker
            raw["asset_class"] = "fixed_income"
            raw.index.name = "date"
            raw = raw.reset_index()
            records.append(raw)

        if not records:
            return pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])
        return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Free macro provider: FRED + Banxico SIE + INEGI (used by Yahoo mode)
# ---------------------------------------------------------------------------

class FREDBanxicoMacroProvider:
    """
    Fetches macro data from free public APIs:
      - FRED (St. Louis Fed)   : USD/MXN, Fed Funds rate, US industrial production
      - Banxico SIE API        : Banxico rate, inflation (INPC yoy), USD/MXN official
      - INEGI BIE API          : IMAI, Mexico industrial production yoy, exports yoy

    BANXICO_TOKEN must be set in the environment (obtain free at
    https://www.banxico.org.mx/SieAPIRest/service/v1/token).
    If any source is unavailable the column is filled with NaN gracefully.
    """

    # ------------------------------------------------------------------
    # FRED
    # ------------------------------------------------------------------
    _FRED_SERIES = {
        "DEXMXUS": "usd_mxn_fred",   # MXN per USD (inverted → USD/MXN)
        "FEDFUNDS": "us_fed_rate",
        "INDPRO": "us_ip_index",      # level; we compute yoy internally
    }

    # ------------------------------------------------------------------
    # Banxico SIE series IDs
    # ------------------------------------------------------------------
    _BANXICO_SERIES = {
        "SF43783": "banxico_rate",    # Tasa de fondeo bancario objetivo
        "SP1": "inflation_index",     # INPC (nivel); yoy computed internally
        "SF60653": "usd_mxn_banxico", # Tipo de cambio FIX
    }

    # ------------------------------------------------------------------
    # INEGI BIE series IDs (indicator codes)
    # ------------------------------------------------------------------
    _INEGI_SERIES = {
        "910406": "IMAI",                       # Índice de actividad industrial
        "910405": "industrial_production_idx",  # Producción industrial (nivel)
        "229954": "exports_idx",                # Exportaciones totales (nivel)
    }

    # ------------------------------------------------------------------
    # Banxico SIE bond yield series
    # ------------------------------------------------------------------
    _BANXICO_BOND_SERIES = {
        "SF43936": "CETES28",
        "SF43939": "CETES91",
        "SF43773": "MBONO3Y",
        "SF43774": "MBONO5Y",
        "SF43775": "MBONO10Y",
    }

    # Bond characteristics: maturity (years), modified duration (approx), coupon_rate
    # Coupon rates are fixed at representative benchmark issuance rates (2017-2026 era).
    # CETES are zero-coupon discount instruments (coupon_rate = 0).
    _BOND_CHARS: dict = {
        "CETES28":  {"maturity": 28 / 365,  "duration": 28 / 365,  "coupon_rate": 0.0},
        "CETES91":  {"maturity": 91 / 365,  "duration": 91 / 365,  "coupon_rate": 0.0},
        "MBONO3Y":  {"maturity": 3.0,        "duration": 2.7,        "coupon_rate": 0.075},
        "MBONO5Y":  {"maturity": 5.0,        "duration": 4.3,        "coupon_rate": 0.080},
        "MBONO10Y": {"maturity": 10.0,       "duration": 7.5,        "coupon_rate": 0.085},
    }

    def __init__(self, banxico_token: str | None = None) -> None:
        import os
        self._banxico_token = banxico_token or os.environ.get("BANXICO_TOKEN", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _yoy(series: pd.Series) -> pd.Series:
        """Compute year-over-year % change from a monthly level series."""
        return series.pct_change(12, fill_method=None)

    @staticmethod
    def _resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.DatetimeIndex(df.index)
        return df.resample("ME").last()

    # ------------------------------------------------------------------
    # FRED
    # ------------------------------------------------------------------

    def _fetch_fred(self, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import pandas_datareader.data as web
        except ImportError:
            return pd.DataFrame()

        frames = {}
        for series_id, col in self._FRED_SERIES.items():
            try:
                raw = web.DataReader(series_id, "fred", start_date, end_date)
                raw.columns = [col]
                frames[col] = raw[col]
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = self._resample_monthly(df)

        # USD/MXN from FRED is already MXN per USD — keep as-is
        if "usd_mxn_fred" in df.columns:
            df = df.rename(columns={"usd_mxn_fred": "usd_mxn"})

        # US IP: compute yoy from index level
        if "us_ip_index" in df.columns:
            df["us_ip_yoy"] = self._yoy(df["us_ip_index"])
            df = df.drop(columns=["us_ip_index"])

        # Fed Funds rate: convert from percent to decimal
        if "us_fed_rate" in df.columns:
            if df["us_fed_rate"].dropna().median() > 1.0:
                df["us_fed_rate"] = df["us_fed_rate"] / 100.0

        return df

    # ------------------------------------------------------------------
    # Banxico SIE
    # ------------------------------------------------------------------

    def _fetch_banxico(self, start_date: str, end_date: str) -> pd.DataFrame:
        if not self._banxico_token:
            return pd.DataFrame()

        import urllib.request
        import json

        series_ids = ",".join(self._BANXICO_SERIES.keys())
        url = (
            f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
            f"{series_ids}/datos/{start_date}/{end_date}"
        )
        headers = {"Bmx-Token": self._banxico_token}

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return pd.DataFrame()

        frames = {}
        for serie in data.get("bmx", {}).get("series", []):
            sid = serie.get("idSerie", "")
            col = self._BANXICO_SERIES.get(sid)
            if col is None:
                continue
            rows = serie.get("datos", [])
            if not rows:
                continue
            idx = pd.to_datetime([r["fecha"] for r in rows], dayfirst=True)
            vals = pd.to_numeric([r["dato"] for r in rows], errors="coerce")
            frames[col] = pd.Series(vals.tolist(), index=idx)

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = self._resample_monthly(df)

        # Banxico rate: convert percent → decimal
        if "banxico_rate" in df.columns:
            if df["banxico_rate"].dropna().median() > 1.0:
                df["banxico_rate"] = df["banxico_rate"] / 100.0

        # Inflation: compute yoy from INPC level
        if "inflation_index" in df.columns:
            df["inflation_yoy"] = self._yoy(df["inflation_index"])
            df = df.drop(columns=["inflation_index"])

        # Keep banxico FIX rate as fallback for usd_mxn
        if "usd_mxn_banxico" in df.columns:
            df = df.rename(columns={"usd_mxn_banxico": "usd_mxn_banxico"})

        return df

    # ------------------------------------------------------------------
    # INEGI BIE
    # ------------------------------------------------------------------

    def _fetch_inegi(self, start_date: str, end_date: str) -> pd.DataFrame:
        import urllib.request
        import json

        frames = {}
        for indicator, col in self._INEGI_SERIES.items():
            url = (
                f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/"
                f"jsonxml/INDICATOR/{indicator}/es/0700/false/BIE/2.0/data.json"
            )
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                obs = data.get("Series", [{}])[0].get("OBSERVATIONS", [])
                idx = pd.to_datetime([o["TIME_PERIOD"] for o in obs])
                vals = pd.to_numeric([o["OBS_VALUE"] for o in obs], errors="coerce")
                s = pd.Series(vals.tolist(), index=idx)
                s = s[(s.index >= start_date) & (s.index <= end_date)]
                frames[col] = s
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = self._resample_monthly(df)

        # Compute yoy for level series
        if "industrial_production_idx" in df.columns:
            df["industrial_production_yoy"] = self._yoy(df["industrial_production_idx"])
            df = df.drop(columns=["industrial_production_idx"])

        if "exports_idx" in df.columns:
            df["exports_yoy"] = self._yoy(df["exports_idx"])
            df = df.drop(columns=["exports_idx"])

        return df

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fetch_bond_yields(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch CETES and Mbono yields from Banxico SIE and return a bond DataFrame
        compatible with get_bonds() output: (date, ticker, price, ytm, duration,
        credit_spread, asset_class).  Returns empty DataFrame if token is missing
        or API is unavailable.
        """
        from .data_loader import _bond_price

        if not self._banxico_token:
            return pd.DataFrame()

        import urllib.request
        import json

        series_ids = ",".join(self._BANXICO_BOND_SERIES.keys())
        url = (
            f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
            f"{series_ids}/datos/{start_date}/{end_date}"
        )
        headers = {"Bmx-Token": self._banxico_token}

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return pd.DataFrame()

        records = []
        for serie in data.get("bmx", {}).get("series", []):
            sid = serie.get("idSerie", "")
            ticker = self._BANXICO_BOND_SERIES.get(sid)
            if ticker is None:
                continue
            chars = self._BOND_CHARS[ticker]
            for row in serie.get("datos", []):
                try:
                    ytm = float(row["dato"]) / 100.0  # Banxico returns percent
                    date = pd.to_datetime(row["fecha"], dayfirst=True)
                    price = _bond_price(ytm, chars["coupon_rate"], chars["maturity"])
                    records.append({
                        "date": date,
                        "ticker": ticker,
                        "asset_class": "fixed_income",
                        "price": price,
                        "ytm": ytm,
                        "duration": chars["duration"],
                        "credit_spread": 0.0,
                    })
                except (ValueError, KeyError):
                    continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(records)
        df["date"] = pd.to_datetime(df["date"])
        # Resample to monthly end-of-month (last observation in each month)
        df = (
            df.sort_values("date")
            .groupby("ticker", group_keys=False)
            .apply(lambda g: g.set_index("date").resample("ME").last().reset_index())
        )
        return df.reset_index(drop=True)

    def get_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch and merge macro data from FRED, Banxico and INEGI.
        Falls back gracefully to NaN for any unavailable series.
        """
        monthly_idx = pd.date_range(start_date, end_date, freq="ME")

        fred_df    = self._fetch_fred(start_date, end_date)
        banxico_df = self._fetch_banxico(start_date, end_date)
        inegi_df   = self._fetch_inegi(start_date, end_date)

        # Merge all sources on a common monthly index
        result = pd.DataFrame(index=monthly_idx)
        for df in [fred_df, banxico_df, inegi_df]:
            if not df.empty:
                result = result.join(df, how="left")

        # Prefer Banxico FIX rate over FRED for USD/MXN when both exist
        if "usd_mxn_banxico" in result.columns:
            if "usd_mxn" not in result.columns:
                result = result.rename(columns={"usd_mxn_banxico": "usd_mxn"})
            else:
                result["usd_mxn"] = result["usd_mxn"].fillna(result["usd_mxn_banxico"])
                result = result.drop(columns=["usd_mxn_banxico"])

        # Ensure all expected columns exist (fill missing with NaN)
        expected = [
            "IMAI", "industrial_production_yoy", "exports_yoy",
            "usd_mxn", "banxico_rate", "inflation_yoy",
            "us_ip_yoy", "us_fed_rate",
        ]
        for col in expected:
            if col not in result.columns:
                result[col] = np.nan

        result = result[expected].copy()
        result.index.name = "date"
        result = result.reset_index()
        return result


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_provider(source: str, **kwargs) -> BaseDataProvider:
    """
    Factory function to get the appropriate data provider.

    Args:
        source: One of "mock", "yahoo", "bloomberg", "lseg"
        **kwargs: Provider-specific keyword arguments (e.g., bloomberg_host, lseg_app_key)

    Returns:
        An instantiated BaseDataProvider subclass.

    Raises:
        ValueError: If source is not recognized.
        ImportError: If required library for the provider is not installed.
    """
    source = source.lower().strip()
    if source == "mock":
        return MockDataProvider()
    elif source in ("yahoo", "yfinance"):
        return YahooFinanceProvider()
    elif source in ("bloomberg", "bbg"):
        return BloombergProvider()
    elif source in ("refinitiv", "lseg", "eikon"):
        return RefinitivProvider()
    else:
        raise ValueError(
            f"Unknown data source: '{source}'. "
            f"Valid options: 'mock', 'yahoo', 'bloomberg', 'lseg'"
        )
