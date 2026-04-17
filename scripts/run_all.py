#!/usr/bin/env python
"""
run_all.py — Script maestro del proyecto Fondo Mexico.

Ejecuta en orden:
  1. Tests (pytest)
  2. Pipeline completo
  3. Generación del reporte HTML

Configuración: edita config.yaml en la raíz del proyecto.
Los argumentos de la terminal sobreescriben config.yaml.

Uso:
    python scripts/run_all.py
    python scripts/run_all.py --source yahoo
    python scripts/run_all.py --source refinitiv --start 2020-01-01 --hedge
    python scripts/run_all.py --skip-tests
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_mpl_cache = ROOT / ".cache" / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

DEFAULT_BENCHMARKS = ["IPC", "GBMCRE", "GBMNEAR", "GBMMOD", "GBMALFA"]
SUPPORTED_SOURCES = ["mock", "yahoo", "bloomberg", "refinitiv"]
DEFAULT_MULTI_PROVIDERS = ["yahoo", "refinitiv", "bloomberg"]

# ---------------------------------------------------------------------------
# Cargar .env (credenciales) si existe
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Cargar config.yaml
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    config_path = ROOT / "config.yaml"
    defaults = {
        "source": "mock",
        "start_date": "2017-01-01",
        "end_date": "2026-03-31",
        "hedge": False,
        "benchmark_tickers": [],
        "report_output": "reports/output/strategy_report.html",
        "abort_on_test_failure": True,
        "optimizer": "mv",
    }
    if not config_path.exists():
        return defaults
    try:
        import yaml
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        defaults.update({k: v for k, v in file_cfg.items() if v is not None})
    except ImportError:
        print("[AVISO] pyyaml no instalado — usando valores por defecto. Instala con: pip install pyyaml")
    return defaults

# ---------------------------------------------------------------------------
# Argumentos de la terminal (sobreescriben config.yaml)
# ---------------------------------------------------------------------------
def _parse_args(config: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fondo Mexico — pipeline completo")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Fuente(s) de datos: mock|yahoo|bloomberg|refinitiv, "
            "lista separada por comas, o 'all' para yahoo,refinitiv,bloomberg "
            f"(config.yaml: {config['source']})"
        ),
    )
    p.add_argument("--start", default=None, help=f"Fecha inicio YYYY-MM-DD (config.yaml: {config['start_date']})")
    p.add_argument("--end",   default=None, help=f"Fecha fin   YYYY-MM-DD (config.yaml: {config['end_date']})")
    p.add_argument("--hedge", action="store_true", default=None, help="Activar hedge overlay (Layer 2)")
    p.add_argument("--out",   default=None, help="Ruta del reporte HTML de salida")
    p.add_argument(
        "--benchmarks",
        default=None,
        help="Tickers benchmark separados por coma (ej: ^MXX,GBMINTBO.MX)",
    )
    p.add_argument("--skip-tests", action="store_true", help="Omitir la etapa de tests")
    p.add_argument(
        "--optimizer",
        choices=["mv", "cvar", "robust", "both"],
        default=None,
        help=f"Optimizador de portafolio (config.yaml: {config.get('optimizer', 'mv')})",
    )
    return p.parse_args()


def _normalize_sources(raw_source: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(raw_source, (list, tuple)):
        candidates = [str(s).strip().lower() for s in raw_source if str(s).strip()]
    else:
        source_text = str(raw_source).strip().lower()
        if source_text == "all":
            candidates = DEFAULT_MULTI_PROVIDERS.copy()
        else:
            candidates = [s.strip().lower() for s in source_text.split(",") if s.strip()]

    if not candidates:
        raise ValueError("No se especificó ninguna fuente de datos.")

    invalid = [s for s in candidates if s not in SUPPORTED_SOURCES]
    if invalid:
        valid = ", ".join(SUPPORTED_SOURCES + ["all"])
        raise ValueError(f"Fuente(s) inválida(s): {', '.join(invalid)}. Válidas: {valid}")

    # Preservar orden y eliminar duplicados
    return list(dict.fromkeys(candidates))


def _output_path_for_source(base_out: str, source: str, multi_source: bool) -> str:
    if not multi_source:
        return base_out
    out_path = Path(base_out)
    return str(out_path.with_name(f"{out_path.stem}_{source}{out_path.suffix}"))

# ---------------------------------------------------------------------------
# Paso 1 — Tests
# ---------------------------------------------------------------------------
def run_tests(abort_on_failure: bool) -> bool:
    print("\n" + "=" * 60)
    print("PASO 1/3 — Ejecutando tests")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        if abort_on_failure:
            print("\n[ERROR] Tests fallaron. Abortando pipeline.")
            print("        Para ignorar los tests usa: --skip-tests")
            return False
        else:
            print("\n[AVISO] Tests fallaron pero se continúa (abort_on_test_failure=false en config.yaml).")
    return True

# ---------------------------------------------------------------------------
# Paso 2 + 3 — Pipeline y reporte
# ---------------------------------------------------------------------------
def run_report(
    source: str,
    start: str,
    end: str,
    hedge: bool,
    out_path: str,
    optimizer: str = "mv",
    benchmark_tickers: list[str] | None = None,
    settings: dict | None = None,
) -> None:
    print("\n" + "=" * 60)
    print(f"PASO 2/3 — Corriendo pipeline  [{source}]  {start} → {end}")
    print("=" * 60)

    # Inyectar App Key de Refinitiv si está en el entorno
    provider_kwargs = {}
    if source == "refinitiv":
        app_key = os.environ.get("REFINITIV_APP_KEY", "")
        if app_key and app_key != "pega_tu_app_key_aqui":
            provider_kwargs["app_key"] = app_key
        else:
            print("[AVISO] REFINITIV_APP_KEY no configurada en .env")

    from src.pipeline import run_pipeline
    results = run_pipeline(
        hedge_mode=hedge,
        data_source=source,
        start_date=start,
        end_date=end,
        optimizer=optimizer,
        benchmark_tickers=benchmark_tickers,
        settings=settings,
        **provider_kwargs,
    )

    print("\n" + "=" * 60)
    print("PASO 3/3 — Generando reporte HTML")
    print("=" * 60)

    from reports.charts import build_dashboard_html
    html = build_dashboard_html(results, hedge_mode=hedge, data_source=source)

    out = ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n[OK] Reporte guardado en: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _load_env()
    config  = _load_config()
    args    = _parse_args(config)

    # Merge: CLI > config.yaml > defaults
    source_value = args.source or config["source"]
    start   = args.start      or config["start_date"]
    end     = args.end        or config["end_date"]
    hedge   = args.hedge      if args.hedge else config["hedge"]
    out     = args.out        or config["report_output"]
    abort     = config["abort_on_test_failure"]
    optimizer = args.optimizer or config.get("optimizer", "mv")
    pipeline_settings = dict(config)
    sources = _normalize_sources(source_value)
    cli_bench = [x.strip() for x in args.benchmarks.split(",") if x.strip()] if args.benchmarks else None
    cfg_bench = config.get("benchmark_tickers")
    if isinstance(cfg_bench, str):
        cfg_bench = [x.strip() for x in cfg_bench.split(",") if x.strip()]

    def _benchmarks_for_source(current_source: str) -> list[str]:
        if cli_bench is not None:
            return cli_bench
        if cfg_bench is not None:
            return cfg_bench
        return DEFAULT_BENCHMARKS if current_source in ("yahoo", "refinitiv") else []

    print(f"\nFondo Mexico — Pipeline completo")
    print(f"  Fuente(s)  : {', '.join(sources)}")
    print(f"  Periodo    : {start} → {end}")
    print(f"  Hedge      : {hedge}")
    print(f"  Optimizador: {optimizer}")
    print(f"  Reporte    : {out}")

    if not args.skip_tests:
        ok = run_tests(abort_on_failure=abort)
        if not ok:
            sys.exit(1)
    else:
        print("\n[AVISO] Tests omitidos por --skip-tests")

    multi_source = len(sources) > 1
    successful_sources: list[str] = []
    failed_sources: list[tuple[str, str]] = []

    for source in sources:
        benchmark_tickers = _benchmarks_for_source(source)
        out_for_source = _output_path_for_source(out, source, multi_source)
        print(
            f"\n[RUN] source={source} | benchmarks={', '.join(benchmark_tickers) if benchmark_tickers else 'N/A'} "
            f"| out={out_for_source}"
        )
        try:
            run_report(
                source,
                start,
                end,
                hedge,
                out_for_source,
                optimizer,
                benchmark_tickers,
                settings=pipeline_settings,
            )
            successful_sources.append(source)
        except Exception as exc:
            failed_sources.append((source, str(exc)))
            print(f"\n[ERROR] source={source} falló: {exc}")
            print("        Se continúa con el siguiente provider.")

    if failed_sources:
        print("\n" + "=" * 60)
        print("RESUMEN DE ERRORES POR PROVIDER")
        print("=" * 60)
        for source, err in failed_sources:
            print(f"- {source}: {err}")

    if not successful_sources:
        print("\n[ERROR] Ningún provider completó exitosamente.")
        sys.exit(1)

    print("\n[LISTO] Pipeline completado exitosamente.\n")

if __name__ == "__main__":
    main()
