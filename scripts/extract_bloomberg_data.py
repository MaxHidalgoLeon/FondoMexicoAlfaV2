#!/usr/bin/env python
"""
extract_bloomberg_data.py

Corre en la PC con Bloomberg Terminal (requiere xbbg + blpapi).
Lee los tickers desde config/ticker_map.yaml y guarda los datos en parquet.

Uso:
    python scripts/extract_bloomberg_data.py
    python scripts/extract_bloomberg_data.py --start 2017-01-01 --end 2026-03-31
    python scripts/extract_bloomberg_data.py --output-dir data/bloomberg

Instalar en la PC Bloomberg:
    pip install xbbg pandas pyarrow pyyaml
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Bonds no están en ticker_map.yaml — tienen su propio mapeo interno
_BOND_TICKERS = {
    "CETES28":  "GCETAA28 Index",
    "CETES91":  "GCETAA91 Index",
    "MBONO3Y":  "CTMXN3Y Govt",
    "MBONO5Y":  "CTMXN5Y Govt",
    "MBONO10Y": "CTMXN10Y Govt",
}

_MACRO_TICKERS = {
    "MMACTIVI Index": "IMAI",
    "MXIPYOY Index":  "industrial_production_yoy",
    "MXEXPORT Index": "exports_yoy",
    "USDMXN Curncy":  "usd_mxn",
    "MXONBRAN Index": "banxico_rate",
    "MXCPYOY Index":  "inflation_yoy",
    "IP YOY Index":   "us_ip_yoy",
    "FDTR Index":     "us_fed_rate",
}

_RATE_COLS = {"banxico_rate", "inflation_yoy", "us_ip_yoy", "us_fed_rate",
              "industrial_production_yoy", "exports_yoy"}

_FIBRA_PREFIXES = {"FUNO", "FIBRA", "TERRA", "FMTY"}


def _load_ticker_map(repo_root: Path, explicit_path: str | None = None) -> dict:
    if explicit_path:
        path = Path(explicit_path)
    else:
        # Buscar en: 1) junto al script, 2) repo_root/config, 3) cwd/config
        candidates = [
            Path(__file__).parent / "ticker_map.yaml",
            repo_root / "config" / "ticker_map.yaml",
            Path.cwd() / "config" / "ticker_map.yaml",
            Path.cwd() / "ticker_map.yaml",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "No se encontró ticker_map.yaml. Usa --ticker-map para especificar la ruta.\n"
                f"Rutas buscadas:\n" + "\n".join(f"  {p}" for p in candidates)
            )
    with open(path) as f:
        return yaml.safe_load(f)


def _equity_tickers(ticker_map: dict) -> dict[str, str]:
    """Devuelve {bloomberg_ticker: canonical} para tickers con bloomberg != null."""
    result = {}
    for canonical, providers in ticker_map.items():
        if not isinstance(providers, dict):
            continue
        bbg = providers.get("bloomberg")
        if not bbg:
            continue
        # Agregar sufijo " MM Equity" si no tiene asset class
        if not any(x in bbg for x in (" Equity", " Index", " Govt", " Curncy", " Corp")):
            bbg = f"{bbg} MM Equity"
        result[bbg] = canonical
    return result


def _is_fibra(canonical: str) -> bool:
    return any(canonical.startswith(p) for p in _FIBRA_PREFIXES)


def _to_pandas(raw) -> pd.DataFrame:
    """Convierte a pandas si xbbg devuelve Polars (PyEngine nuevo)."""
    if hasattr(raw, "to_pandas"):
        return raw.to_pandas()
    return raw


def _is_empty(raw) -> bool:
    if raw is None:
        return True
    if hasattr(raw, "is_empty"):   # polars
        return raw.is_empty()
    return raw.empty               # pandas


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que el DataFrame tenga DatetimeIndex, sin importar si PyEngine devuelve fecha como columna o índice."""
    if "date" in df.columns:
        df = df.set_index("date")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "date"
    return df


def _collapse_multiindex(df: pd.DataFrame, reverse_map: dict) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df.columns = [reverse_map.get(str(c), str(c)) for c in df.columns]
    return df


def _to_wide_format(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Convierte df a formato ancho (DatetimeIndex × ticker canónico).
    Maneja tanto MultiIndex de columnas (xbbg clásico) como formato largo (PyEngine).
    """
    # Caso 1: MultiIndex en columnas → xbbg clásico
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
        df.columns = [mapping.get(str(c).strip(), str(c).strip()) for c in df.columns]
        df.columns.name = None
        return df

    # Caso 2: formato largo — columna con nombre de security y fechas duplicadas en índice
    sec_col = next(
        (c for c in df.columns if c.lower() in ("security", "ticker", "name")),
        None,
    )
    if sec_col is not None:
        # Buscar columna de valor: "value" primero, luego cualquier columna que no sea security/date/field
        val_col = next(
            (c for c in df.columns if c.lower() == "value"),
            next(
                (c for c in df.columns if c.lower() not in (sec_col.lower(), "date", "field")),
                None,
            ),
        )
        if val_col:
            df = df.pivot(columns=sec_col, values=val_col)
            df.columns = [mapping.get(str(c).strip(), str(c).strip()) for c in df.columns]
            df.columns.name = None
        return df

    # Caso 3: ya ancho, sin MultiIndex — solo renombrar columnas
    df.columns = [mapping.get(str(c).strip(), str(c).strip()) for c in df.columns]
    return df


def _divide_rates_if_pct(df: pd.DataFrame) -> pd.DataFrame:
    for col in _RATE_COLS:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0 and s.abs().median() > 1.0:
                df[col] = df[col] / 100.0
    return df


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------

def extract_prices(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Precios: %d tickers...", len(mapping))
    raw = blp.bdh(list(mapping), "PX_LAST", start, end,
                  adjustmentNormal=True, adjustmentAbnormal=True, adjustmentSplit=True)
    raw = _to_pandas(raw)
    if _is_empty(raw):
        logger.warning("Precios: respuesta vacía")
        return pd.DataFrame()
    raw = _normalize_index(raw)
    raw = _to_wide_format(raw, mapping)
    bdays = pd.bdate_range(start, end)
    return raw.reindex(bdays).ffill(limit=5)


def extract_volume(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Volumen: %d tickers...", len(mapping))
    try:
        raw = blp.bdh(list(mapping), "PX_VOLUME", start, end)
        raw = _to_pandas(raw)
        if _is_empty(raw):
            return pd.DataFrame()
        raw = _normalize_index(raw)
        raw = _to_wide_format(raw, mapping)
        raw = raw.apply(pd.to_numeric, errors="coerce")
        bdays = pd.bdate_range(start, end)
        return raw.reindex(bdays).fillna(0.0)
    except Exception as e:
        logger.warning("Volumen falló: %s", e)
        return pd.DataFrame()


def _bdh_monthly(blp, ticker: str, fields: list, start: str, end: str) -> pd.DataFrame:
    """bdh con periodicidad mensual — compatible con PyEngine nuevo y viejo."""
    try:
        raw = blp.bdh(ticker, fields, start, end, periodicitySelection="MONTHLY")
    except Exception:
        raw = blp.bdh(ticker, fields, start, end, Per="M")
    return _to_pandas(raw)


def extract_fundamentals(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    fields = ["PE_RATIO", "PX_TO_BOOK_RATIO", "RETURN_ON_EQUITY",
              "PROF_MARGIN", "NET_DEBT_TO_EBITDA", "EBITDA_GROWTH", "CAPEX_TO_SALES"]
    rename = {"PE_RATIO": "pe_ratio", "PX_TO_BOOK_RATIO": "pb_ratio",
              "RETURN_ON_EQUITY": "roe", "PROF_MARGIN": "profit_margin",
              "NET_DEBT_TO_EBITDA": "net_debt_to_ebitda", "EBITDA_GROWTH": "ebitda_growth",
              "CAPEX_TO_SALES": "capex_to_sales"}

    equity_map = {bbg: can for bbg, can in mapping.items() if not _is_fibra(can)}
    logger.info("Fundamentales: %d tickers...", len(equity_map))
    records = []
    for bbg, canonical in equity_map.items():
        try:
            raw = _bdh_monthly(blp, bbg, fields, start, end)
            if _is_empty(raw):
                continue
            raw = _normalize_index(raw)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=rename)
            raw["ticker"] = canonical
            records.append(raw.reset_index())
        except Exception as e:
            logger.warning("Fundamentales %s falló: %s", bbg, e)
    if not records:
        return pd.DataFrame(columns=["date", "ticker"] + list(rename.values()))
    return pd.concat(records, ignore_index=True)


def extract_fibra_fundamentals(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    fields = ["CAP_RATE", "FFO_YIELD", "DVD_SH_12M", "LOAN_TO_VALUE", "VACANCY_RATE"]
    rename = {"CAP_RATE": "cap_rate", "FFO_YIELD": "ffo_yield", "DVD_SH_12M": "dividend_yield",
              "LOAN_TO_VALUE": "ltv", "VACANCY_RATE": "vacancy_rate"}

    fibra_map = {bbg: can for bbg, can in mapping.items() if _is_fibra(can)}
    logger.info("FIBRA fundamentales: %d tickers...", len(fibra_map))
    records = []
    for bbg, canonical in fibra_map.items():
        try:
            raw = _bdh_monthly(blp, bbg, fields, start, end)
            if _is_empty(raw):
                continue
            raw = _normalize_index(raw)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=rename)
            raw["ticker"] = canonical
            records.append(raw.reset_index())
        except Exception as e:
            logger.warning("FIBRA fundamentales %s falló: %s", bbg, e)
    if not records:
        return pd.DataFrame(columns=["date", "ticker"] + list(rename.values()))
    return pd.concat(records, ignore_index=True)


def extract_macro(start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Macro: %d indicadores...", len(_MACRO_TICKERS))
    raw = _bdh_monthly(blp, list(_MACRO_TICKERS), "PX_LAST", start, end)
    if _is_empty(raw):
        logger.warning("Macro: respuesta vacía")
        return pd.DataFrame()
    raw = _normalize_index(raw)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=_MACRO_TICKERS)
    raw = raw.reset_index()
    return _divide_rates_if_pct(raw)


def extract_bonds(start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    fields = ["PX_LAST", "YLD_YTM_MID", "DUR_MID", "Z_SPRD_MID"]
    rename = {"PX_LAST": "price", "YLD_YTM_MID": "ytm",
              "DUR_MID": "duration", "Z_SPRD_MID": "credit_spread"}
    logger.info("Bonos: %d tickers...", len(_BOND_TICKERS))
    records = []
    for canonical, bbg in _BOND_TICKERS.items():
        try:
            raw = _bdh_monthly(blp, bbg, fields, start, end)
            if _is_empty(raw):
                continue
            raw = _normalize_index(raw)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=rename)
            raw["ticker"] = canonical
            raw["asset_class"] = "fixed_income"
            records.append(raw.reset_index())
        except Exception as e:
            logger.warning("Bono %s falló: %s", bbg, e)
    if not records:
        return pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])
    return pd.concat(records, ignore_index=True)


def extract_market_caps(mapping: dict[str, str]) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Market caps: %d tickers...", len(mapping))
    try:
        raw = blp.bdp(tickers=list(mapping), flds=["CUR_MKT_CAP"])
        raw = _to_pandas(raw)
        if _is_empty(raw):
            return pd.DataFrame(columns=["ticker", "market_cap_mxn"])
        # PyEngine devuelve con columnas en minúsculas
        # PyEngine puede devolver security como columna o como índice
        for sec_col in ("security", "Security", "ticker", "Ticker"):
            if sec_col in raw.columns:
                raw = raw.set_index(sec_col)
                break
        # Tomar la primera columna numérica (solo pedimos CUR_MKT_CAP)
        num_cols = raw.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            logger.warning("Market caps: no se encontró columna numérica. Columnas: %s", list(raw.columns))
            return pd.DataFrame(columns=["ticker", "market_cap_mxn"])
        cap_col = num_cols[0]
        # Normalizar espacios en el índice
        raw.index = pd.Index([re.sub(r"\s+", " ", str(s)).strip() for s in raw.index])
        records = []
        for bbg, canonical in mapping.items():
            bbg_key = re.sub(r"\s+", " ", bbg).strip()
            if bbg_key in raw.index:
                val = raw.loc[bbg_key, cap_col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if pd.notna(val):
                    records.append({"ticker": canonical, "market_cap_mxn": float(val)})
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning("Market caps falló: %s", e)
        return pd.DataFrame(columns=["ticker", "market_cap_mxn"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae datos de Bloomberg Terminal y los guarda como parquet.")
    parser.add_argument("--output-dir", default="data/bloomberg", help="Carpeta de salida (default: data/bloomberg)")
    parser.add_argument("--start", default="2017-01-01", help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end",   default="2026-12-31", help="Fecha fin YYYY-MM-DD")
    parser.add_argument("--ticker-map", default=None, help="Ruta al ticker_map.yaml (opcional, se busca automáticamente)")
    args = parser.parse_args()

    repo_root  = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Extracción Bloomberg  %s → %s ===", args.start, args.end)
    logger.info("Salida: %s", output_dir.resolve())

    ticker_map = _load_ticker_map(repo_root, explicit_path=args.ticker_map)
    mapping    = _equity_tickers(ticker_map)
    logger.info("Tickers Bloomberg encontrados en ticker_map.yaml: %d", len(mapping))

    steps = [
        ("prices.parquet",            lambda: extract_prices(mapping, args.start, args.end)),
        ("volume.parquet",            lambda: extract_volume(mapping, args.start, args.end)),
        ("fundamentals.parquet",      lambda: extract_fundamentals(mapping, args.start, args.end)),
        ("fibra_fundamentals.parquet",lambda: extract_fibra_fundamentals(mapping, args.start, args.end)),
        ("macro.parquet",             lambda: extract_macro(args.start, args.end)),
        ("bonds.parquet",             lambda: extract_bonds(args.start, args.end)),
        ("market_caps.parquet",       lambda: extract_market_caps(mapping)),
    ]

    for filename, fn in steps:
        logger.info("--- %s ---", filename)
        try:
            df = fn()
            if df is None or df.empty:
                logger.warning("%s: sin datos, archivo no guardado", filename)
                continue
            df.to_parquet(output_dir / filename)
            logger.info("%s guardado  (%d filas)", filename, len(df))
        except Exception as e:
            logger.error("%s FALLÓ: %s", filename, e)

    logger.info("=== Extracción completada. Copia la carpeta '%s' a tu laptop. ===", output_dir)


if __name__ == "__main__":
    main()
