#!/usr/bin/env python
"""
run_etf.py — Pipeline para la versión ETF del Fondo Mexico.

Universo: EWW (45-65%) | INDS (20-35%) | IGF (5-15%) | ILF (0-10%) | EMLC (complemento)
Señales:  momentum + volatilidad (price-only, sin fundamentales)
Salidas:  reports/output/output_etf_{source}.html

Uso:
    python scripts/run_etf.py
    python scripts/run_etf.py --source yahoo
    python scripts/run_etf.py --source bloomberg
    python scripts/run_etf.py --source refinitiv
    python scripts/run_etf.py --source yahoo,bloomberg
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_mpl_cache = ROOT / ".cache" / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

SUPPORTED_SOURCES = ["mock", "yahoo", "bloomberg", "refinitiv"]
DEFAULT_REPORT_BASE = "reports/output/output_etf.html"


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _load_config() -> dict:
    config_path = ROOT / "config.yaml"
    defaults = {
        "source": "yahoo",
        "start_date": "2017-01-01",
        "end_date": "2026-03-31",
        "optimizer": "mv",
        "report_output": DEFAULT_REPORT_BASE,
    }
    if not config_path.exists():
        return defaults
    try:
        import yaml
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        defaults.update({k: v for k, v in file_cfg.items() if v is not None})
    except ImportError:
        pass
    return defaults


def _normalize_sources(raw: str | list) -> list[str]:
    if isinstance(raw, list):
        candidates = [str(s).strip().lower() for s in raw if str(s).strip()]
    else:
        raw = str(raw).strip().lower()
        candidates = ["yahoo", "bloomberg", "refinitiv"] if raw == "all" else [s.strip() for s in raw.split(",") if s.strip()]
    invalid = [s for s in candidates if s not in SUPPORTED_SOURCES]
    if invalid:
        raise ValueError(f"Fuente(s) inválida(s): {', '.join(invalid)}. Válidas: {', '.join(SUPPORTED_SOURCES)}, all")
    return list(dict.fromkeys(candidates))


def _output_for_source(base: str, source: str, multi: bool) -> str:
    if not multi:
        return base
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{source}{p.suffix}"))


DEFAULT_BENCHMARKS = ["IPC", "GBMCRE", "GBMNEAR", "GBMMOD", "GBMALFA"]


def _parse_args(config: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fondo Mexico — pipeline ETF")
    p.add_argument("--source",    default=None)
    p.add_argument("--start",     default=None)
    p.add_argument("--end",       default=None)
    p.add_argument("--optimizer", choices=["mv", "cvar", "robust", "both"], default=None)
    p.add_argument("--out",       default=None)
    p.add_argument("--benchmarks", default=None,
                   help="Benchmarks separados por coma. Default: IPC,GBMCRE,GBMNEAR,GBMMOD,GBMALFA")
    p.add_argument("--hedge", action="store_true", default=None,
                   help="Activar hedge overlay (Layer 2)")
    return p.parse_args()


def run_report(
    source: str,
    start: str,
    end: str,
    out_path: str,
    optimizer: str,
    benchmark_tickers: list[str],
    hedge: bool,
    settings: dict,
) -> None:
    print(f"\n{'='*60}")
    print(f"ETF Pipeline  [{source}]  {start} → {end}  hedge={hedge}")
    print(f"{'='*60}")

    provider_kwargs: dict = {}
    if source == "refinitiv":
        app_key = os.environ.get("REFINITIV_APP_KEY", "")
        if app_key and app_key != "pega_tu_app_key_aqui":
            provider_kwargs["app_key"] = app_key
        else:
            print("[AVISO] REFINITIV_APP_KEY no configurada en .env")
    elif source == "bloomberg":
        provider_kwargs["data_dir"] = os.environ.get("BLOOMBERG_DATA_DIR", "data/bloomberg")

    from src.pipeline import run_etf_pipeline
    results = run_etf_pipeline(
        hedge_mode=hedge,
        data_source=source,
        start_date=start,
        end_date=end,
        optimizer=optimizer,
        benchmark_tickers=benchmark_tickers if benchmark_tickers else None,
        settings=settings,
        **provider_kwargs,
    )

    from reports.charts import build_dashboard_html
    html = build_dashboard_html(results, hedge_mode=hedge, data_source=source)

    out = ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n[OK] Reporte ETF guardado en: {out}")


def main() -> None:
    _load_env()
    config = _load_config()
    args   = _parse_args(config)

    raw_source  = args.source  or config.get("source", "yahoo")
    start       = args.start   or config["start_date"]
    end         = args.end     or config["end_date"]
    optimizer   = args.optimizer or config.get("optimizer", "mv")
    hedge       = args.hedge if args.hedge else config.get("hedge", False)
    out         = args.out     or DEFAULT_REPORT_BASE
    # CLI benchmarks override config; if neither, default to GBM funds
    if args.benchmarks:
        benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    elif config.get("benchmark_tickers"):
        cfg_b = config["benchmark_tickers"]
        benchmarks = [x.strip() for x in cfg_b.split(",")] if isinstance(cfg_b, str) else list(cfg_b)
    else:
        benchmarks = []   # run_etf_pipeline defaults to GBM funds when None is passed
    sources     = _normalize_sources(raw_source)
    multi       = len(sources) > 1

    print("\nFondo Mexico — Pipeline ETF")
    print(f"  Fuente(s)  : {', '.join(sources)}")
    print(f"  Periodo    : {start} → {end}")
    print(f"  Hedge      : {hedge}")
    print(f"  Optimizador: {optimizer}")
    print(f"  Universo   : EWW | INDS | IGF | ILF | CETES28 | CETES91 | MBONO3Y")
    print(f"  Benchmarks : {', '.join(benchmarks) if benchmarks else 'GBM (por defecto)'}")

    successful, failed = [], []
    for source in sources:
        out_for = _output_for_source(out, source, multi)
        try:
            run_report(source, start, end, out_for, optimizer, benchmarks, hedge, dict(config))
            successful.append(source)
        except Exception as exc:
            failed.append((source, str(exc)))
            print(f"\n[ERROR] {source}: {exc}")

    if failed:
        print(f"\n[ERRORES] {', '.join(s for s, _ in failed)}")
    if not successful:
        sys.exit(1)
    print("\n[LISTO] Pipeline ETF completado.\n")


if __name__ == "__main__":
    main()
