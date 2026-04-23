#!/usr/bin/env python
"""
run_etf_hyperopt.py — Hyperopt para la versión ETF del Fondo Mexico.

Optimiza los mismos hiperparámetros del modelo (risk_aversion, tau, EWMA, etc.)
pero usando el universo ETF (EWW, INDS, IGF, ILF, EMLC) con señales price-only.

Salidas:
  reports/output/hyperopt_results_etf_{source}.json
  reports/output/hyperopt_report_etf_{source}.html
  config_optimized_etf_{source}.yaml

Uso:
    python scripts/run_etf_hyperopt.py
    python scripts/run_etf_hyperopt.py --source yahoo --n-trials 30
    python scripts/run_etf_hyperopt.py --source bloomberg --n-trials 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_mpl_cache = ROOT / ".cache" / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

SUPPORTED_SOURCES = ["mock", "yahoo", "bloomberg", "refinitiv"]


def _load_yaml_config() -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    config_path = ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(data: dict[str, Any], path: Path) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fondo Mexico ETF — hyperopt (Optuna)")
    p.add_argument("--source",      default=None)
    p.add_argument("--start",       dest="start_date",  default=None)
    p.add_argument("--end",         dest="end_date",    default=None)
    p.add_argument("--n-trials",    type=int,           default=None)
    p.add_argument("--n-folds",     type=int,           default=None)
    p.add_argument("--purge-gap-days", type=int,        default=None)
    p.add_argument("--objective",   choices=["sharpe_adj", "sortino", "calmar"], default=None)
    p.add_argument("--optimizer",   choices=["mv", "cvar", "robust"], default=None)
    p.add_argument("--seed",        type=int,           default=None)
    return p


def _run_single_source(
    source: str,
    cfg: dict,
    start_date: str,
    end_date: str,
    n_trials: int,
    n_folds: int,
    purge_gap_days: int,
    objective_metric: str,
    turnover_penalty: float,
    optimizer: str,
    seed: int,
    logger: logging.Logger,
) -> bool:
    output_path = ROOT / f"reports/output/hyperopt_results_etf_{source}.json"
    config_out  = ROOT / f"config_optimized_etf_{source}.yaml"
    report_out  = ROOT / f"reports/output/hyperopt_report_etf_{source}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== ETF [%s] trials=%d folds=%d objective=%s optimizer=%s ===",
                source, n_trials, n_folds, objective_metric, optimizer)

    provider_kwargs: dict = {}
    if source == "refinitiv":
        app_key = os.environ.get("REFINITIV_APP_KEY", "")
        if app_key and app_key != "pega_tu_app_key_aqui":
            provider_kwargs["app_key"] = app_key
    elif source == "bloomberg":
        provider_kwargs["data_dir"] = os.environ.get("BLOOMBERG_DATA_DIR", "data/bloomberg")

    from src.data_loader import load_etf_data
    from src.features import build_etf_features
    from src.hyperopt import run_hyperopt

    try:
        data = load_etf_data(source=source, start_date=start_date, end_date=end_date, **provider_kwargs)
    except Exception as exc:
        logger.error("[ETF %s] Fallo al cargar datos: %s", source, exc)
        return False

    prices   = data["prices"]
    universe = data["universe"]
    macro    = data["macro"]
    bonds    = data["bonds"]

    feature_df = build_etf_features(prices, macro, universe, bonds=bonds)
    if feature_df.empty:
        logger.error("[ETF %s] feature_df vacío — no hay suficientes precios.", source)
        return False

    result = run_hyperopt(
        prices=prices,
        feature_df=feature_df,
        universe=universe,
        macro=macro,
        n_trials=n_trials,
        n_folds=n_folds,
        purge_gap_days=purge_gap_days,
        objective_metric=objective_metric,
        turnover_penalty=turnover_penalty,
        seed=seed,
        settings=cfg,
        optimizer=optimizer,
    )

    payload = {
        "mode":                    "etf",
        "source":                  source,
        "best_params":             result.best_params,
        "best_value":              result.best_value,
        "validation_metrics":      result.validation_metrics,
        "n_trials_completed":      result.n_trials_completed,
        "optimization_time_seconds": result.optimization_time_seconds,
        "objective_metric":        result.objective_metric,
        "turnover_penalty":        result.turnover_penalty,
        "search_space": {
            k: {"kind": v[0], "low": v[1], "high": v[2], "log": v[3]} if v[0] != "categorical"
               else {"kind": "categorical", "choices": v[1]}
            for k, v in result.search_space.items()
        },
        "trial_history": result.trial_history.to_dict(orient="records") if not result.trial_history.empty else [],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("[ETF %s] JSON guardado: %s", source, output_path)

    if result.best_params:
        try:
            _dump_yaml({**cfg, **result.best_params}, config_out)
            logger.info("[ETF %s] Config optimizado: %s", source, config_out)
        except ImportError:
            logger.warning("PyYAML no disponible — omitiendo %s.", config_out)

    try:
        from reports.charts import generate_hyperopt_report
        generate_hyperopt_report(result, output_path=report_out)
        logger.info("[ETF %s] Reporte HTML: %s", source, report_out)
    except Exception as exc:
        logger.warning("[ETF %s] Reporte HTML omitido: %s", source, exc)

    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("run_etf_hyperopt")

    args = _build_parser().parse_args()
    cfg  = _load_yaml_config()

    raw_source       = args.source        or cfg.get("source") or "yahoo"
    sources          = _normalize_sources(raw_source)
    start_date       = args.start_date    or cfg.get("start_date",               "2017-01-01")
    end_date         = args.end_date      or cfg.get("end_date",                 "2026-03-31")
    n_trials         = args.n_trials      or int(cfg.get("hyperopt_n_trials",    50))
    n_folds          = args.n_folds       or int(cfg.get("hyperopt_n_folds",     3))
    purge_gap_days   = args.purge_gap_days or int(cfg.get("hyperopt_purge_gap_days", 21))
    objective_metric = args.objective     or str(cfg.get("hyperopt_objective",   "sharpe_adj"))
    turnover_penalty = float(cfg.get("hyperopt_turnover_penalty", 0.5))
    optimizer        = args.optimizer     or str(cfg.get("hyperopt_optimizer",   "mv"))
    seed             = args.seed          or int(cfg.get("hyperopt_seed",        42))

    print("\nFondo Mexico — Hyperopt ETF")
    print(f"  Fuente(s)  : {', '.join(sources)}")
    print(f"  Periodo    : {start_date} → {end_date}")
    print(f"  Trials     : {n_trials}  |  Folds: {n_folds}  |  Purge: {purge_gap_days}d")
    print(f"  Objetivo   : {objective_metric}  |  Optimizador: {optimizer}")
    print(f"  Universo   : EWW | INDS | IGF | ILF | EMLC\n")

    successful, failed = [], []
    for source in sources:
        ok = _run_single_source(
            source=source, cfg=cfg,
            start_date=start_date, end_date=end_date,
            n_trials=n_trials, n_folds=n_folds, purge_gap_days=purge_gap_days,
            objective_metric=objective_metric, turnover_penalty=turnover_penalty,
            optimizer=optimizer, seed=seed, logger=logger,
        )
        (successful if ok else failed).append(source)

    print("\n" + "=" * 60)
    if successful:
        print(f"[OK] Completado: {', '.join(successful)}")
        for s in successful:
            print(f"     config_optimized_etf_{s}.yaml  →  listo para run_etf.py")
    if failed:
        print(f"[ERROR] Fallaron: {', '.join(failed)}")
    print("=" * 60)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
