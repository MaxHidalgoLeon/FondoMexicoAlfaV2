"""
Dashboard chart builder for Fondo Mexico strategy reports.

All charts use plotly with a dark theme. The main entry point is
build_dashboard_html(), which assembles a fully standalone HTML page.
"""
from __future__ import annotations

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("Install plotly: pip install plotly>=5.18")

import numpy as np
import pandas as pd

from src.bootstrap import bootstrap_paths
from src.settings import resolve_settings

# ---------------------------------------------------------------------------
# Global theme
# ---------------------------------------------------------------------------

LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1a1d27",
    plot_bgcolor="#1a1d27",
    font=dict(family="system-ui, -apple-system, sans-serif", color="#e8eaf0", size=12),
    margin=dict(l=60, r=30, t=102, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

C_BLUE   = "#4f8ef7"
C_GREEN  = "#2ecc71"
C_RED    = "#e74c3c"
C_AMBER  = "#f39c12"
C_PURPLE = "#9b59b6"
C_GRAY   = "#666677"
C_BG     = "#0f1117"
C_CARD   = "#1a1d27"

STYLE = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f1117; color: #e8eaf0;
       font-family: system-ui, -apple-system, sans-serif;
       padding: 32px 48px; }
h1 { font-size: 1.8rem; color: #4f8ef7; margin-bottom: 6px; }
.meta { color: #666677; font-size: 0.85rem; margin-bottom: 40px; }
h2 { font-size: 1.1rem; color: #4f8ef7; text-transform: uppercase;
     letter-spacing: 0.1em; margin: 40px 0 16px; padding-bottom: 6px;
     border-bottom: 1px solid #2d3045; }
h3 { font-size: 0.9rem; color: #8892b0; margin: 24px 0 10px; }
.card { background: #1a1d27; border: 1px solid #2d3045;
        border-radius: 8px; padding: 20px; margin-bottom: 20px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th { text-align: left; padding: 8px 12px; color: #8892b0;
     border-bottom: 1px solid #2d3045; font-weight: 500; }
td { padding: 8px 12px; border-bottom: 1px solid #1e2133; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
.good { color: #2ecc71; }
.bad  { color: #e74c3c; }
.neutral { color: #f39c12; }
.mono { font-family: 'Roboto Mono', 'Courier New', monospace; }
</style>
"""

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fig_to_div(fig, div_id: str | None = None) -> str:
    kwargs = {"full_html": False, "include_plotlyjs": False, "config": {"responsive": True}}
    if div_id:
        kwargs["div_id"] = div_id
    return fig.to_html(**kwargs)


def _add_time_controls(fig):
    """Add range selector and range slider to time-series charts."""
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            x=0.99,
            xanchor="right",
            y=1.20,
            yanchor="top",
            bgcolor="#1a1d27",
            activecolor="#2d4368",
            bordercolor="#2d3045",
            borderwidth=1,
            font=dict(color="#e8eaf0", size=11),
        ),
        rangeslider=dict(visible=True, bgcolor="#131722", bordercolor="#2d3045", borderwidth=1, thickness=0.09),
        type="date",
    )
    return fig


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v * 100:.2f}%"


def _num(v, decimals=2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def _color_class(v, good_if_positive=True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "neutral"
    return "good" if (v > 0) == good_if_positive else "bad"


def _cum_from_log(returns):
    return np.exp(returns.fillna(0.0).cumsum())


def compute_realized_volatility(returns, method: str = "ewma", span: int = 21):
    clean = pd.Series(returns, dtype=float)
    if str(method).lower() == "ewma":
        return clean.ewm(span=int(span), min_periods=max(int(span // 2), 10), adjust=False).std() * np.sqrt(252)
    return clean.rolling(int(span)).std() * np.sqrt(252)


def build_bootstrap_fan_chart_data(
    returns,
    n_paths: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> dict[str, object]:
    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if series.empty:
        return {"paths": np.empty((0, 0)), "median": pd.Series(dtype=float)}
    paths = bootstrap_paths(series, n_paths=int(n_paths), block_size=int(block_size), seed=int(seed))
    equity_paths = np.exp(np.cumsum(paths, axis=1))
    index = pd.DatetimeIndex(series.index)
    return {
        "paths": equity_paths,
        "median": pd.Series(np.median(equity_paths, axis=0), index=index),
        "p05": pd.Series(np.quantile(equity_paths, 0.05, axis=0), index=index),
        "p25": pd.Series(np.quantile(equity_paths, 0.25, axis=0), index=index),
        "p75": pd.Series(np.quantile(equity_paths, 0.75, axis=0), index=index),
        "p95": pd.Series(np.quantile(equity_paths, 0.95, axis=0), index=index),
    }


def _ci_range_text(metric_stats: dict | None, fmt_fn) -> str:
    if not metric_stats:
        return ""
    low = metric_stats.get("ci_low")
    high = metric_stats.get("ci_high")
    if low is None or high is None:
        return ""
    return f" [{fmt_fn(low)}, {fmt_fn(high)}]"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_dashboard_html(results: dict, hedge_mode, data_source: str) -> str:
    import datetime

    hedge_active = bool(hedge_mode)  # True si es True, "analytical", o "regulated"
    hedge_is_analytical = (hedge_mode == "analytical" or hedge_mode is True)
    # True = tratar como analytical (default seguro)

    settings   = resolve_settings(results.get("settings"))
    summary    = results["summary"]
    backtest   = results["backtest"]
    feature_df = results["feature_df"]
    forecast_df = results.get("forecast_df")
    universe   = results["data"]["universe"]
    returns    = backtest["returns"]
    weights    = backtest["weights"]
    turnover   = backtest["turnover"]
    metrics    = summary["metrics"]
    benchmarks = results.get("benchmarks")

    hedge_returns  = results["hedge_layer"]["returns"]      if hedge_active else None
    hedge_metrics  = results["hedge_layer"]["metrics"]      if hedge_active else None
    hedge_stress   = results["hedge_layer"].get("stress")   if hedge_active else None
    hedge_overlay  = results["hedge_layer"]["fx_overlay"]   if hedge_active else None
    hedge_leverage = results["hedge_layer"]["leverage_series"] if hedge_active else None
    tail_hedge     = results["hedge_layer"]["tail_hedge"]   if hedge_active else None
    hedge_layer    = results.get("hedge_layer") if hedge_active else None
    signal_diag    = results.get("signal_diagnostics", {})
    benchmark_alpha = (benchmarks or {}).get("alpha_significance", {}) if isinstance(benchmarks, dict) else {}

    start_date = summary["start_date"].date()
    end_date   = summary["end_date"].date()
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = []
    sections.append(_section_performance(returns, hedge_returns, metrics, hedge_metrics, hedge_active, summary, settings, hedge_is_analytical))
    sections.append(_section_benchmarks(returns, hedge_returns, benchmarks, benchmark_alpha, data_source=data_source))
    sections.append(_section_risk(returns, summary, hedge_returns=hedge_returns, hedge_metrics=hedge_metrics, settings=settings))
    sections.append(_section_statistical_significance(summary, benchmark_alpha, signal_diag))
    # Optimizer comparison — only when "both" were run
    if summary.get("optimizer") == "both" and summary.get("metrics_cvar") and backtest.get("returns_cvar") is not None:
        sections.append(_section_optimizer_comparison(backtest, summary, hedge_mode=hedge_active))
    if hedge_active and hedge_layer is not None:
        sections.append(_section_hedge_engine_breakdown(hedge_layer, metrics, hedge_metrics))
    sections.append(_section_signals(feature_df, forecast_df, signal_diag))
    sections.append(_section_universe_donuts(feature_df, universe))
    sections.append(_section_portfolio(weights, turnover, universe, hedge_layer=hedge_layer))
    sections.append(_section_stress(summary["stress"], hedge_stress_df=hedge_stress, stress_distributional=summary.get("stress_test_distributional")))
    if hedge_active:
        sections.append(_section_fx_overlay(hedge_overlay, hedge_leverage))
        sections.append(_section_layer_comparison(metrics, hedge_metrics, tail_hedge, hedge_is_analytical))

    body = "\n".join(sections)

    # Optional sections inserted before Signal Quality shift later section numbers.
    has_optimizer_section = bool(
        summary.get("optimizer") == "both" and summary.get("metrics_cvar") and backtest.get("returns_cvar") is not None
    )
    has_hedge_breakdown = bool(hedge_mode and hedge_layer is not None)
    section_shift = int(has_optimizer_section) + int(has_hedge_breakdown)
    if section_shift:
        for base_num, title in [
            (5, "Signal Quality"),
            (6, "Universe Composition"),
            (7, "Portfolio Construction"),
            (8, "Stress Testing"),
            (9, "FX Overlay &amp; Dynamic Leverage"),
            (10, "Traditional vs Hedge"),
        ]:
            body = body.replace(
                f"<h2>{base_num}. {title}</h2>",
                f"<h2>{base_num + section_shift}. {title}</h2>",
            )
    if has_hedge_breakdown and not has_optimizer_section:
        body = body.replace(
            "<h2>6. Hedge Engine Breakdown</h2>",
            "<h2>5. Hedge Engine Breakdown</h2>",
        )

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fondo Mexico Strategy Report</title>
{plotly_cdn}
{STYLE}
</head>
<body>
<h1>Fondo Mexico Strategy Report</h1>
<p class="meta">
  Data source: <strong>{data_source}</strong> &nbsp;|&nbsp;
  Period: <strong>{start_date}</strong> to <strong>{end_date}</strong> &nbsp;|&nbsp;
  Generated: <strong>{timestamp}</strong> &nbsp;|&nbsp;
  Hedge overlay: <strong>{"Yes" if hedge_active else "No"}</strong>
</p>
{body}
<script>
{_build_dynamic_metric_scripts()}
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Section 1: Performance Overview
# ---------------------------------------------------------------------------

def _section_performance(returns, hedge_returns, metrics, hedge_metrics, hedge_mode, summary, settings, hedge_is_analytical=False) -> str:
    metrics_ci = summary.get("metrics_ci", {}) or {}
    # Chart 1.1 — Cumulative returns (combined)
    cum = _cum_from_log(returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index,
        y=cum.values,
        name="Traditional",
        line=dict(color=C_BLUE, width=2),
    ))
    if hedge_mode and hedge_returns is not None:
        cum_hedge = _cum_from_log(hedge_returns)
        fig.add_trace(go.Scatter(
            x=cum_hedge.index,
            y=cum_hedge.values,
            name="Hedge",
            line=dict(color=C_GREEN, width=2),
        ))
    fig.add_hline(y=1.0, line=dict(color=C_GRAY, dash="dot", width=1))
    fig.update_layout(
        **LAYOUT,
        title="Cumulative Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
    )
    _add_time_controls(fig)
    chart_cum = _fig_to_div(fig, div_id="chart-cumulative")

    fan_chart = ""
    if settings.get("fan_chart_enabled", False):
        fan = build_bootstrap_fan_chart_data(
            returns,
            n_paths=int(settings["fan_chart_n_paths"]),
            block_size=int(settings["fan_chart_block_size"]),
            seed=int(settings["bootstrap_seed"]),
        )
        if isinstance(fan.get("median"), pd.Series) and not fan["median"].empty:
            fig_fan = go.Figure()
            fig_fan.add_trace(go.Scatter(
                x=fan["p95"].index,
                y=fan["p95"].values,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig_fan.add_trace(go.Scatter(
                x=fan["p05"].index,
                y=fan["p05"].values,
                fill="tonexty",
                fillcolor="rgba(79,142,247,0.10)",
                line=dict(width=0),
                name="Bootstrap 5-95%",
                hoverinfo="skip",
            ))
            fig_fan.add_trace(go.Scatter(
                x=fan["p75"].index,
                y=fan["p75"].values,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig_fan.add_trace(go.Scatter(
                x=fan["p25"].index,
                y=fan["p25"].values,
                fill="tonexty",
                fillcolor="rgba(79,142,247,0.20)",
                line=dict(width=0),
                name="Bootstrap 25-75%",
                hoverinfo="skip",
            ))
            fig_fan.add_trace(go.Scatter(
                x=fan["median"].index,
                y=fan["median"].values,
                name="Bootstrap median",
                line=dict(color=C_AMBER, width=2),
            ))
            fig_fan.add_trace(go.Scatter(
                x=cum.index,
                y=cum.values,
                name="Observed equity",
                line=dict(color=C_BLUE, width=2),
            ))
            fig_fan.update_layout(
                **LAYOUT,
                title="Bootstrap Equity Fan Chart",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
            )
            _add_time_controls(fig_fan)
            fan_chart = _fig_to_div(fig_fan)

    # Chart 1.2 — Rolling 63-day Sharpe (separated by sleeve)
    roll_sharpe_trad = (returns.rolling(63).mean() / (returns.rolling(63).std() + 1e-9)) * np.sqrt(252)
    fig2_trad = go.Figure()
    fig2_trad.add_trace(go.Scatter(
        x=roll_sharpe_trad.index,
        y=roll_sharpe_trad.values,
        name="Rolling Sharpe (63d)",
        line=dict(color=C_BLUE, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.1)",
    ))
    fig2_trad.add_hline(y=0, line=dict(color=C_GRAY, dash="dot", width=1))
    fig2_trad.add_hline(y=1, line=dict(color=C_GREEN, dash="dash", width=1))
    fig2_trad.update_layout(
        **LAYOUT,
        title="Rolling Sharpe Ratio (63-day) — Traditional",
        xaxis_title="Date",
        yaxis_title="Sharpe",
    )
    _add_time_controls(fig2_trad)
    chart_sharpe_trad = _fig_to_div(fig2_trad)

    chart_sharpe_hedge = ""
    if hedge_mode and hedge_returns is not None:
        roll_sharpe_hedge = (hedge_returns.rolling(63).mean() / (hedge_returns.rolling(63).std() + 1e-9)) * np.sqrt(252)
        fig2_hedge = go.Figure()
        fig2_hedge.add_trace(go.Scatter(
            x=roll_sharpe_hedge.index,
            y=roll_sharpe_hedge.values,
            name="Rolling Sharpe (63d)",
            line=dict(color=C_GREEN, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(46,204,113,0.1)",
        ))
        fig2_hedge.add_hline(y=0, line=dict(color=C_GRAY, dash="dot", width=1))
        fig2_hedge.add_hline(y=1, line=dict(color=C_BLUE, dash="dash", width=1))
        fig2_hedge.update_layout(
            **LAYOUT,
            title="Rolling Sharpe Ratio (63-day) — Hedge",
            xaxis_title="Date",
            yaxis_title="Sharpe",
        )
        _add_time_controls(fig2_hedge)
        chart_sharpe_hedge = _fig_to_div(fig2_hedge)

    # Table 1.3 — Key metrics
    def metric_row(label, v1, v2, fmt_fn, good_if_positive=True, ci_stats=None):
        c1 = _color_class(v1, good_if_positive)
        ci_suffix = _ci_range_text(ci_stats, fmt_fn)
        c1_str = f'<span class="{c1} mono">{fmt_fn(v1)}{ci_suffix}</span>'
        if hedge_mode and v2 is not None:
            c2 = _color_class(v2, good_if_positive)
            c2_str = f'<span class="{c2} mono">{fmt_fn(v2)}</span>'
            return f"<tr><td>{label}</td><td>{c1_str}</td><td>{c2_str}</td></tr>"
        return f"<tr><td>{label}</td><td>{c1_str}</td></tr>"

    hdr = "<tr><th>Metric</th><th>Traditional</th>"
    if hedge_mode:
        hdr += "<th>Hedge</th>"
    hdr += "</tr>"

    hm = hedge_metrics or {}
    rows = [
        metric_row("Annualized Return",  metrics.get("annualized_return"), hm.get("annualized_return"), _pct, True, metrics_ci.get("cagr")),
        metric_row("Annualized Vol",     metrics.get("annualized_vol"),    hm.get("annualized_vol"),    _pct, False),
        metric_row("Sharpe Ratio",       metrics.get("sharpe"),            hm.get("sharpe"),            lambda v: _num(v, 2), True, metrics_ci.get("sharpe")),
        metric_row("Sortino Ratio",      metrics.get("sortino"),           hm.get("sortino"),           lambda v: _num(v, 2), True, metrics_ci.get("sortino")),
        metric_row("Max Drawdown",       metrics.get("max_drawdown"),      hm.get("max_drawdown"),      _pct, False, metrics_ci.get("max_drawdown")),
        metric_row("Calmar Ratio",        metrics.get("calmar"),            hm.get("calmar"),            lambda v: _num(v, 2), True),
        metric_row("CVaR 95% (daily)",   metrics.get("cvar_95"),           hm.get("cvar_95"),           _pct, False, metrics_ci.get("cvar_95")),
        metric_row("Avg Turnover",       metrics.get("turnover"),          hm.get("turnover"),          _pct, False),
    ]
    table = f'<table>{hdr}{"".join(rows)}</table>'

    cum_grid = f"<div class=\"card\">{chart_cum}</div>"
    if fan_chart:
        cum_grid += f"\n<div class=\"card\">{fan_chart}<p style='color:#8892b0; font-size:0.82rem; margin-top:10px;'>El fan chart muestra un envelope de trayectorias consistentes con la dependencia temporal observada via stationary bootstrap.</p></div>"
    
    if hedge_is_analytical:
        cum_grid += """
   <p style="font-size:11px;color:#9ca3af;margin-top:4px;">
   * Hedge Layer 2 mostrado como referencia analítica \u2014 no incluido en NAV regulatorio.
   </p>"""
    sharpe_grid = (
        f"<div class=\"grid2\"><div class=\"card\">{chart_sharpe_trad}</div><div class=\"card\">{chart_sharpe_hedge}</div></div>"
        if chart_sharpe_hedge
        else f"<div class=\"card\">{chart_sharpe_trad}</div>"
    )

    return f"""
<h2>1. Performance Overview</h2>
{cum_grid}
{sharpe_grid}
<div class="card"><h3>Key Metrics</h3>{table}</div>"""


def _section_benchmarks(returns, hedge_returns, benchmarks, alpha_significance=None, data_source: str = "") -> str:
    fig = go.Figure()
    base = _cum_from_log(returns)
    fig.add_trace(go.Scatter(
        x=base.index,
        y=base.values,
        name="Traditional",
        line=dict(color=C_BLUE, width=2),
    ))
    if hedge_returns is not None:
        hedge_cum = _cum_from_log(hedge_returns)
        fig.add_trace(go.Scatter(
            x=hedge_cum.index,
            y=hedge_cum.values,
            name="Hedge",
            line=dict(color=C_GREEN, width=2),
        ))

    has_bench_data = False
    if isinstance(benchmarks, dict):
        bret = benchmarks.get("returns")
        if isinstance(bret, pd.DataFrame) and not bret.empty:
            has_bench_data = True
            for i, col in enumerate(bret.columns.tolist()):
                cc = np.exp(bret[col].dropna().cumsum())
                if cc.empty:
                    continue
                color = [C_AMBER, C_PURPLE, "#1abc9c", "#e67e22", "#95a5a6"][i % 5]
                fig.add_trace(go.Scatter(
                    x=cc.index,
                    y=cc.values,
                    name=f"Benchmark: {col}",
                    line=dict(color=color, width=1.6, dash="dot"),
                ))

    fig.add_hline(y=1.0, line=dict(color=C_GRAY, dash="dot", width=1))
    fig.update_layout(**LAYOUT, title="Strategy vs Benchmarks",
                      xaxis_title="Date", yaxis_title="Cumulative Return")
    _add_time_controls(fig)
    chart = _fig_to_div(fig, div_id="chart-benchmarks")

    note = ""
    if not has_bench_data:
        note = "<p style='color:#8892b0; margin-top:8px;'>Sin benchmarks externos cargados. Cuando me pases los tickers GBM, se agregan aquí junto con IPC.</p>"

    alpha_table = ""
    if alpha_significance:
        rows = []
        for bench, stats in alpha_significance.items():
            alpha = stats.get("alpha_annualized", {})
            ir = stats.get("information_ratio", {})
            te = stats.get("tracking_error", {})
            beta_val = stats.get("beta", float("nan"))
            alpha_sig = " *" if (alpha.get("p_value", 1.0) or 1.0) < 0.05 else ""
            ir_sig = " *" if (ir.get("p_value", 1.0) or 1.0) < 0.05 else ""
            beta_str = f"{beta_val:.2f}" if isinstance(beta_val, float) and not (beta_val != beta_val) else "—"
            rows.append(
                f"<tr><td>{bench}</td>"
                f"<td class='mono'>{beta_str}</td>"
                f"<td class='mono'>{_pct(alpha.get('point'))} [{_pct(alpha.get('ci_low'))}, {_pct(alpha.get('ci_high'))}]{alpha_sig}</td>"
                f"<td class='mono'>{_num(ir.get('point'))} [{_num(ir.get('ci_low'))}, {_num(ir.get('ci_high'))}]{ir_sig}</td>"
                f"<td class='mono'>{_pct(te.get('point'))} [{_pct(te.get('ci_low'))}, {_pct(te.get('ci_high'))}]</td></tr>"
            )
        alpha_table = (
            "<div class='card'><h3>Alpha de Jensen vs Benchmarks</h3>"
            "<table><tr><th>Benchmark</th><th>Beta (β)</th><th>Alpha Jensen (95% CI)</th><th>Information Ratio (95% CI)</th><th>Tracking Error (95% CI)</th></tr>"
            + "".join(rows)
            + "</table><p style='color:#8892b0; font-size:0.82rem; margin-top:10px;'>Alpha = R<sub>p</sub> − [R<sub>f</sub> + β·(R<sub>m</sub> − R<sub>f</sub>)]. * indica significancia al 5% bajo paired stationary bootstrap.</p></div>"
        )

    source_note = ""
    if str(data_source).lower() in ("yahoo", "mock"):
        source_note = (
            "<div class='card' style='border-color:#f39c12;'>"
            "<p style='color:#f39c12; font-size:0.88rem;'>"
            "<strong>Señales en modo Yahoo:</strong> Los datos fundamentales históricos no están disponibles vía Yahoo Finance "
            "(solo snapshot actual). El backtest utiliza únicamente señales de precio: momentum y liquidez. "
            "Para señales fundamentales PIT (pe_ratio, pb_ratio, roe, etc.) se requiere Refinitiv/LSEG.</p></div>"
        )

    return f"""
<h2>2. Benchmarks</h2>
<div class=\"card\">{chart}{note}</div>
{source_note}
{alpha_table}"""


# ---------------------------------------------------------------------------
# Section 2: Risk Analysis
# ---------------------------------------------------------------------------

def _section_risk(returns, summary, hedge_returns=None, hedge_metrics=None, settings=None) -> str:
    cfg = resolve_settings(settings)

    def _risk_charts(series, label, line_color, fill_color, dd_div_id=None, garch_series=None, garch_val=None):
        # Drawdown
        cum = _cum_from_log(series)
        dd = (cum / cum.cummax()) - 1
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(color=line_color, width=1.5),
            name=f"Drawdown ({label})",
        ))
        fig_dd.update_layout(**LAYOUT, title=f"Underwater Equity Curve — {label}",
                             xaxis_title="Date", yaxis_title="Drawdown (%)")
        _add_time_controls(fig_dd)
        dd_chart = _fig_to_div(fig_dd, div_id=dd_div_id) if dd_div_id else _fig_to_div(fig_dd)

        # Distribution
        clean = series.dropna()
        var_val = float(np.percentile(clean, 5)) if len(clean) else None
        cvar_val = float(clean[clean <= var_val].mean()) if var_val is not None and len(clean[clean <= var_val]) else None
        mean_val = float(clean.mean()) if len(clean) else None
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=series * 100,
            nbinsx=60,
            marker_color=line_color,
            opacity=0.7,
            name=f"Daily Returns ({label})",
        ))
        markers = [
            (var_val * 100 if var_val is not None else None, C_RED, "VaR 95%", "dash", "top left"),
            (cvar_val * 100 if cvar_val is not None else None, C_RED, "CVaR 95%", "solid", "top right"),
            (mean_val * 100 if mean_val is not None else None, C_GREEN, "Mean", "dash", "top"),
        ]
        for x_val, color, name, dash, ann_pos in markers:
            if x_val is not None:
                fig_dist.add_vline(
                    x=x_val,
                    line=dict(color=color, dash=dash, width=1.5),
                    annotation_text=name,
                    annotation_position=ann_pos,
                )
        fig_dist.update_layout(**LAYOUT, title=f"Daily Return Distribution — {label}",
                               xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        dist_chart = _fig_to_div(fig_dist)

        # Vol
        realized_vol = compute_realized_volatility(
            series,
            method=str(cfg["realized_vol_method"]),
            span=int(cfg["realized_vol_span"]),
        ) * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=realized_vol.index,
            y=realized_vol.values,
            name=f"Realized Vol ({cfg['realized_vol_method']}) — {label}",
            line=dict(color=line_color, width=1.5),
        ))
        if garch_series is not None:
            gv = garch_series.reindex(realized_vol.index).ffill()
            if len(gv.dropna()):
                fig_vol.add_trace(go.Scatter(
                    x=gv.index,
                    y=gv.values * 100,
                    name="GARCH Forecast (rolling)",
                    line=dict(color=C_AMBER, width=1.5, dash="dash"),
                ))
        elif garch_val is not None and np.isfinite(garch_val):
            fig_vol.add_hline(y=garch_val * 100,
                              line=dict(color=C_AMBER, dash="dash", width=1.5),
                              annotation_text=f"GARCH Forecast: {garch_val*100:.1f}%",
                              annotation_position="right")
        vol_layout = {
            **LAYOUT,
            "legend": dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
        }
        fig_vol.update_layout(
            **vol_layout,
            title=f"Volatility (21d) — {label}",
            xaxis_title="Date",
            yaxis_title="Annualized Vol (%)",
        )
        _add_time_controls(fig_vol)
        vol_note = ""
        if str(cfg["realized_vol_method"]).lower() == "ewma":
            vol_note = "<p style='color:#8892b0; font-size:0.82rem; margin-top:10px;'>Realized vol EWMA (span=21, ~lambda=0.905) per RiskMetrics methodology.</p>"
        vol_chart = _fig_to_div(fig_vol) + vol_note

        return dd_chart, dist_chart, vol_chart

    # Traditional panels
    trad_dd, trad_dist, trad_vol = _risk_charts(
        returns,
        "Traditional",
        C_BLUE,
        "rgba(79,142,247,0.20)",
        dd_div_id="chart-drawdown",
        garch_series=summary.get("garch_vol_series"),
        garch_val=summary.get("garch_vol_forecast"),
    )

    # Traditional advanced metrics table (existing model-based risk)
    def risk_row(label, val, fmt_fn, good_if_positive=True):
        c = _color_class(val, good_if_positive)
        return f'<tr><td>{label}</td><td class="{c} mono">{fmt_fn(val)}</td></tr>'

    trad_metrics = summary.get("metrics") or {}
    trad_values = {
        "garch_vol_forecast": summary.get("garch_vol_forecast"),
        "dynamic_var": summary.get("dynamic_var"),
        "monte_carlo_var": summary.get("monte_carlo_var"),
        "gev_var": summary.get("gev_var"),
        "gev_cvar": summary.get("gev_cvar"),
        "annualized_vol": trad_metrics.get("annualized_vol"),
        "cvar_95": trad_metrics.get("cvar_95"),
        "max_drawdown": trad_metrics.get("max_drawdown"),
        "sharpe": trad_metrics.get("sharpe"),
        "sortino": trad_metrics.get("sortino"),
        "calmar": trad_metrics.get("calmar"),
        "turnover": trad_metrics.get("turnover"),
    }
    risk_specs = [
        ("GARCH Vol Forecast (21d)", "garch_vol_forecast", _pct, False),
        ("Dynamic VaR 95% (GARCH)", "dynamic_var", _pct, False),
        ("Monte Carlo VaR 95%", "monte_carlo_var", _pct, False),
        ("GEV VaR 95%", "gev_var", _pct, False),
        ("GEV CVaR 95%", "gev_cvar", _pct, False),
        ("Annualized Vol", "annualized_vol", _pct, False),
        ("CVaR 95% (daily)", "cvar_95", _pct, False),
        ("Max Drawdown", "max_drawdown", _pct, False),
        ("Sharpe Ratio", "sharpe", lambda v: _num(v, 2), True),
        ("Sortino Ratio", "sortino", lambda v: _num(v, 2), True),
        ("Calmar Ratio", "calmar", lambda v: _num(v, 2), True),
        ("Avg Turnover", "turnover", _pct, False),
    ]
    trad_rows = [risk_row(label, trad_values.get(key), fmt_fn, gip) for label, key, fmt_fn, gip in risk_specs]
    trad_table = f'<table><tr><th>Risk Metric</th><th>Traditional</th></tr>{"".join(trad_rows)}</table>'

    # Hedge panels (when available)
    hedge_block = ""
    if hedge_returns is not None and len(hedge_returns):
        hedge_dd, hedge_dist, hedge_vol = _risk_charts(
            hedge_returns,
            "Hedge",
            C_GREEN,
            "rgba(46,204,113,0.20)",
            dd_div_id="chart-drawdown-hedge",
            garch_series=(hedge_metrics or {}).get("garch_vol_series"),
            garch_val=(hedge_metrics or {}).get("garch_vol_forecast"),
        )

        hm = hedge_metrics or {}
        hedge_values = {
            "garch_vol_forecast": hm.get("garch_vol_forecast"),
            "dynamic_var": hm.get("dynamic_var"),
            "monte_carlo_var": hm.get("monte_carlo_var"),
            "gev_var": hm.get("gev_var"),
            "gev_cvar": hm.get("gev_cvar"),
            "annualized_vol": hm.get("annualized_vol"),
            "cvar_95": hm.get("cvar_95"),
            "max_drawdown": hm.get("max_drawdown"),
            "sharpe": hm.get("sharpe"),
            "sortino": hm.get("sortino"),
            "calmar": hm.get("calmar"),
            "turnover": hm.get("turnover"),
        }
        hedge_rows = [risk_row(label, hedge_values.get(key), fmt_fn, gip) for label, key, fmt_fn, gip in risk_specs]
        hedge_table = f'<table><tr><th>Risk Metric</th><th>Hedge</th></tr>{"".join(hedge_rows)}</table>'

        hedge_block = f"""
<div class="grid2">
  <div class="card">{trad_dd}</div>
  <div class="card">{hedge_dd}</div>
</div>
<div class="grid2">
  <div class="card">{trad_dist}</div>
  <div class="card">{hedge_dist}</div>
</div>
<div class="grid2">
  <div class="card">{trad_vol}</div>
  <div class="card">{hedge_vol}</div>
</div>
<div class="grid2">
  <div class="card"><h3>Advanced Risk Metrics — Tradicional</h3>{trad_table}</div>
  <div class="card"><h3>Risk Metrics — Hedge</h3>{hedge_table}</div>
</div>"""
    else:
        hedge_block = f"""
<div class="card">{trad_dd}</div>
<div class="grid2">
  <div class="card">{trad_dist}</div>
  <div class="card">{trad_vol}</div>
</div>
<div class="card"><h3>Advanced Risk Metrics</h3>{trad_table}</div>"""

    covariance_block = ""
    cov_diag = summary.get("covariance_diagnostics") or {}
    if cov_diag:
        rolling_corr = cov_diag.get("rolling_corr")
        ewma_corr = cov_diag.get("ewma_corr")
        det_ratio = cov_diag.get("det_ratio")
        if isinstance(rolling_corr, pd.DataFrame) and isinstance(ewma_corr, pd.DataFrame):
            fig_cov = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Rolling Correlation", "EWMA Correlation"),
            )
            fig_cov.add_trace(
                go.Heatmap(
                    z=rolling_corr.values,
                    x=rolling_corr.columns.tolist(),
                    y=rolling_corr.index.tolist(),
                    colorscale="RdBu",
                    zmid=0.0,
                    showscale=False,
                ),
                row=1,
                col=1,
            )
            fig_cov.add_trace(
                go.Heatmap(
                    z=ewma_corr.values,
                    x=ewma_corr.columns.tolist(),
                    y=ewma_corr.index.tolist(),
                    colorscale="RdBu",
                    zmid=0.0,
                    showscale=True,
                ),
                row=1,
                col=2,
            )
            fig_cov.update_layout(**LAYOUT, title="Covariance Diagnostics — Correlation Heatmaps")
            heatmap_chart = _fig_to_div(fig_cov)
        else:
            heatmap_chart = ""

        det_chart = ""
        if isinstance(det_ratio, pd.Series) and not det_ratio.dropna().empty:
            fig_det = go.Figure()
            fig_det.add_trace(go.Scatter(
                x=det_ratio.index,
                y=det_ratio.values,
                name="det(EWMA) / det(rolling)",
                line=dict(color=C_AMBER, width=1.8),
            ))
            fig_det.add_hline(y=1.0, line=dict(color=C_GRAY, dash="dot", width=1))
            fig_det.update_layout(
                **LAYOUT,
                title="Determinant Ratio Through Time",
                xaxis_title="Date",
                yaxis_title="det ratio",
            )
            _add_time_controls(fig_det)
            det_chart = _fig_to_div(fig_det)

        vol_compare = ""
        ewma_vol = compute_realized_volatility(returns, method="ewma", span=int(cfg["realized_vol_span"])) * 100
        rolling_vol = compute_realized_volatility(returns, method="rolling", span=int(cfg["realized_vol_span"])) * 100
        garch_series = summary.get("garch_vol_series")
        if len(ewma_vol.dropna()) or len(rolling_vol.dropna()):
            fig_real = go.Figure()
            fig_real.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol.values, name="Realized EWMA", line=dict(color=C_BLUE, width=2)))
            fig_real.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name="Realized rolling", line=dict(color=C_GREEN, width=1.8, dash="dash")))
            if garch_series is not None and len(garch_series.dropna()):
                fig_real.add_trace(go.Scatter(x=garch_series.index, y=garch_series.values * 100, name="GARCH forecast", line=dict(color=C_AMBER, width=1.8, dash="dot")))
            fig_real.update_layout(
                **LAYOUT,
                title="Realized Volatility Comparison",
                xaxis_title="Date",
                yaxis_title="Annualized Vol (%)",
            )
            _add_time_controls(fig_real)
            vol_compare = _fig_to_div(fig_real)

        comparison_table = ""
        method_comparison = summary.get("method_comparison") or {}
        if method_comparison:
            current_metrics = method_comparison.get("current", {})
            baseline_metrics = method_comparison.get("baseline", {})
            rows = []
            for label, key, fmt in [
                ("Sharpe", "sharpe", lambda x: _num(x, 2)),
                ("Sortino", "sortino", lambda x: _num(x, 2)),
                ("Max Drawdown", "max_drawdown", _pct),
                ("CVaR 95%", "cvar_95", _pct),
                ("Turnover", "turnover", _pct),
            ]:
                rows.append(
                    f"<tr><td>{label}</td><td class='mono'>{fmt(baseline_metrics.get(key))}</td><td class='mono'>{fmt(current_metrics.get(key))}</td></tr>"
                )
            comparison_table = (
                "<table><tr><th>Metric</th><th>Baseline</th><th>EWMA</th></tr>"
                + "".join(rows)
                + f"<tr><td>Regime switches</td><td class='mono'>{method_comparison.get('regime_switches_before', 'N/A')}</td><td class='mono'>{method_comparison.get('regime_switches_after', 'N/A')}</td></tr>"
                + f"<tr><td>Annualized TC savings</td><td class='mono'>N/A</td><td class='mono'>{method_comparison.get('transaction_cost_saved_bps_annualized', 0.0):.2f} bps</td></tr>"
                + "</table>"
            )

        covariance_block = f"""
<div class="grid2">
  <div class="card">{heatmap_chart}</div>
  <div class="card">{det_chart}</div>
</div>
<div class="grid2">
  <div class="card">{vol_compare}</div>
  <div class="card"><h3>Method Comparison</h3>{comparison_table}</div>
</div>"""

    units_note = "<p style='color:#8892b0; font-size:0.82rem; margin-top:10px;'>Nota: VaR y CVaR se muestran como retorno diario esperado en porcentaje (no anualizado).</p>"
    return f"""
<h2>3. Risk Analysis</h2>
{hedge_block}
{covariance_block}
{units_note}"""


# ---------------------------------------------------------------------------
# Section 4: Statistical Significance
# ---------------------------------------------------------------------------

def _section_statistical_significance(summary, benchmark_alpha, signal_diagnostics) -> str:
    metrics_ci = summary.get("metrics_ci") or {}
    perf_table = ""
    if metrics_ci:
        rows = []
        for label, key, fmt in [
            ("Sharpe", "sharpe", lambda x: _num(x, 2)),
            ("Sortino", "sortino", lambda x: _num(x, 2)),
            ("CVaR 95%", "cvar_95", _pct),
            ("Max Drawdown", "max_drawdown", _pct),
            ("CAGR", "cagr", _pct),
        ]:
            stats = metrics_ci.get(key, {})
            rows.append(
                f"<tr><td>{label}</td><td class='mono'>{fmt(stats.get('point'))}</td><td class='mono'>[{fmt(stats.get('ci_low'))}, {fmt(stats.get('ci_high'))}]</td><td class='mono'>{fmt(stats.get('se'))}</td></tr>"
            )
        perf_table = (
            "<div class='card'><h3>Performance Metrics with 95% CI</h3>"
            "<table><tr><th>Metric</th><th>Point</th><th>95% CI</th><th>SE</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )

    alpha_table = ""
    if benchmark_alpha:
        rows = []
        for bench, stats in benchmark_alpha.items():
            alpha = stats.get("alpha_annualized", {})
            ir = stats.get("information_ratio", {})
            beta_val = stats.get("beta", float("nan"))
            beta_str = f"{beta_val:.2f}" if isinstance(beta_val, float) and not (beta_val != beta_val) else "—"
            rows.append(
                f"<tr><td>{bench}</td><td class='mono'>{beta_str}</td><td class='mono'>{_pct(alpha.get('point'))}</td><td class='mono'>[{_pct(alpha.get('ci_low'))}, {_pct(alpha.get('ci_high'))}]</td><td class='mono'>{alpha.get('p_value', np.nan):.3f}</td><td class='mono'>{_num(ir.get('point'))}</td><td class='mono'>{ir.get('p_value', np.nan):.3f}</td></tr>"
            )
        alpha_table = (
            "<div class='card'><h3>Alpha de Jensen vs Benchmarks</h3>"
            "<table><tr><th>Benchmark</th><th>Beta (β)</th><th>Alpha Jensen</th><th>Alpha 95% CI</th><th>Alpha p-value</th><th>IR</th><th>IR p-value</th></tr>"
            + "".join(rows)
            + "<tr><td colspan='7' style='color:#8892b0; font-size:0.82rem; padding-top:10px;'>Alpha = R<sub>p</sub> − [R<sub>f</sub> + β·(R<sub>m</sub> − R<sub>f</sub>)]</td></tr>"
            + "</table></div>"
        )

    signal_table = ""
    if signal_diagnostics:
        rows = []
        for signal, stats in signal_diagnostics.items():
            status = "significant" if stats.get("significant") else "not significant"
            rows.append(
                f"<tr><td>{signal}</td><td class='mono'>{_num(stats.get('ic_mean'), 3)}</td><td class='mono'>{_num(stats.get('ic_t_stat'), 2)}</td><td class='mono'>[{_num(stats.get('ci_low'), 3)}, {_num(stats.get('ci_high'), 3)}]</td><td class='mono'>{stats.get('p_value', np.nan):.3f}</td><td>{status}</td></tr>"
            )
        signal_table = (
            "<div class='card'><h3>Signal Quality Diagnostics</h3>"
            "<table><tr><th>Signal</th><th>IC mean</th><th>IC t-stat</th><th>IC 95% CI</th><th>p-value</th><th>Status</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )

    return f"""
<h2>4. Statistical Significance</h2>
{perf_table}
{alpha_table}
{signal_table}"""


# ---------------------------------------------------------------------------
# Section 5: Signal Quality
# ---------------------------------------------------------------------------

def _section_signals(feature_df, forecast_df=None, signal_diagnostics=None) -> str:
    scored = feature_df.copy()
    if forecast_df is not None and not forecast_df.empty:
        expected_lookup = forecast_df[["date", "ticker", "expected_return"]].drop_duplicates(["date", "ticker"])
        scored = scored.merge(expected_lookup, on=["date", "ticker"], how="left")

    feature_cols = [c for c in [
        "momentum_63", "momentum_126", "volatility_63",
        "value_score", "quality_score", "macro_exposure", "liquidity_score"
    ] if c in scored.columns]

    # Chart 3.1 — Feature-return cross-sectional correlation
    corr = {}
    if "expected_return" in scored.columns and scored["expected_return"].notna().sum() >= 20:
        for col in feature_cols:
            r = scored[col].corr(scored["expected_return"])
            if r is not None and pd.notna(r):
                corr[col] = r

    if corr:
        sorted_corr = dict(sorted(corr.items(), key=lambda x: x[1]))
        colors = [C_GREEN if v >= 0 else C_RED for v in sorted_corr.values()]
        fig = go.Figure(go.Bar(
            x=list(sorted_corr.values()),
            y=list(sorted_corr.keys()),
            orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(**LAYOUT, title="Signal-Return Cross-Sectional Correlation",
                          xaxis_title="Correlation", yaxis_title="Feature")
        chart_corr = _fig_to_div(fig)
    else:
        # Fallback: feature-feature correlation heatmap so section is always informative.
        corr_df = scored[feature_cols].dropna(how="all")
        if len(corr_df) >= 10 and len(feature_cols) >= 2:
            mat = corr_df.corr().fillna(0.0)
            fig = go.Figure(data=go.Heatmap(
                z=mat.values,
                x=mat.columns.tolist(),
                y=mat.index.tolist(),
                colorscale="RdBu",
                zmid=0.0,
                colorbar=dict(title="Corr"),
            ))
            fig.update_layout(**LAYOUT, title="Feature Correlation Heatmap",
                              xaxis_title="Feature", yaxis_title="Feature")
            chart_corr = _fig_to_div(fig)
        else:
            chart_corr = "<p style='color:#666'>Insufficient data for correlation analysis.</p>"

    # Chart 3.2 — Monthly signal dispersion (last 12 months)
    if "expected_return" in scored.columns and "date" in scored.columns:
        df = scored.copy()
        df["month"] = df["date"].dt.to_period("M").astype(str)
        last_months = sorted(df["month"].unique())[-12:]
        df = df[df["month"].isin(last_months)]
        fig3 = go.Figure()
        for month in last_months:
            vals = df[df["month"] == month]["expected_return"].dropna()
            if len(vals) > 1:
                fig3.add_trace(go.Box(y=vals.values, name=month,
                                      marker_color=C_BLUE, showlegend=False,
                                      boxpoints=False, line=dict(width=1),
                                      fillcolor="rgba(79,142,247,0.25)"))
        fig3.update_layout(**LAYOUT, title="Cross-Sectional Signal Dispersion (Last 12M)",
                           xaxis_title="Month", yaxis_title="Expected Return (z-score)")
        fig3.update_xaxes(tickangle=45)
        chart_disp = _fig_to_div(fig3)
    else:
        chart_disp = ""

    diag_table = ""
    if signal_diagnostics:
        rows = []
        for signal, stats in signal_diagnostics.items():
            rows.append(
                f"<tr><td>{signal}</td><td class='mono'>{_num(stats.get('ic_mean'), 3)}</td><td class='mono'>[{_num(stats.get('ci_low'), 3)}, {_num(stats.get('ci_high'), 3)}]</td><td class='mono'>{stats.get('p_value', np.nan):.3f}</td></tr>"
            )
        diag_table = (
            "<div class='card'><h3>IC Bootstrap Summary</h3>"
            "<table><tr><th>Signal</th><th>IC mean</th><th>95% CI</th><th>p-value</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )

    return f"""
<h2>5. Signal Quality</h2>
<div class="card">{chart_corr}</div>
<div class="card">{chart_disp}</div>
{diag_table}"""


# ---------------------------------------------------------------------------
# Section 4: Universe Composition
# ---------------------------------------------------------------------------

def _section_universe_donuts(feature_df, universe_df=None) -> str:
    if "asset_class" not in feature_df.columns or "ticker" not in feature_df.columns:
        return """
<h2>6. Universe Composition</h2>
<div class=\"card\"><p style='color:#666'>Asset-class data unavailable for donut charts.</p></div>"""

    universe = feature_df.dropna(subset=["ticker", "asset_class"]).drop_duplicates("ticker").copy()
    if universe_df is not None and hasattr(universe_df, "columns") and "ticker" in universe_df.columns:
        name_cols = [c for c in ["ticker", "name", "security_name", "company_name", "short_name"] if c in universe_df.columns]
        if len(name_cols) > 1:
            names_lookup = universe_df[name_cols].drop_duplicates("ticker")
            universe = universe.merge(names_lookup, on="ticker", how="left", suffixes=("", "_u"))
    universe["asset_class_lc"] = universe["asset_class"].astype(str).str.lower()
    hedge_eligible = universe[universe["asset_class_lc"].isin(["equity", "fibra"])]

    color_map = {
        "equity": C_BLUE,
        "fibra": C_GREEN,
        "fixed_income": C_AMBER,
        "cash": C_PURPLE,
    }

    preferred_name_cols = [
        "name", "name_u",
        "security_name", "security_name_u",
        "company_name", "company_name_u",
        "short_name", "short_name_u",
    ]
    available_name_cols = [c for c in preferred_name_cols if c in universe.columns]

    def _display_name(row) -> str:
        ticker = str(row.get("ticker", ""))
        custom_labels = {
            "CORP1": "CEMEX30 (CEMEX)",
            "CORP2": "KOF26 (COCACOLA-FEMSA)",
        }
        if ticker in custom_labels:
            return custom_labels[ticker]

        for c in available_name_cols:
            val = row.get(c)
            if val is None:
                continue
            nm = str(val).strip()
            if nm and nm.lower() != "nan":
                return f"{ticker} ({nm})"
        return ticker

    def _build_donut(df, title):
        if df.empty:
            return "", "<p style='color:#666'>No data available.</p>"

        counts = df["asset_class"].value_counts()
        total = float(counts.sum())
        labels = counts.index.tolist()
        values = counts.values.tolist()
        colors = [color_map.get(str(lbl).lower(), C_GRAY) for lbl in labels]
        pct_labels = [
            f"<b><span style='color:{colors[i]}'>{(float(values[i]) / total) * 100:.0f}%</span></b>"
            for i in range(len(values))
        ]

        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.45,
            sort=False,
            text=pct_labels,
            textinfo="text",
            textposition="outside",
            textfont=dict(size=18),
        ))
        fig.update_layout(**LAYOUT, title=title)
        donut_html = _fig_to_div(fig)

        summary_rows = []
        for asset_class, count in counts.items():
            pct_total = (float(count) / total) if total > 0 else 0.0
            row_color = color_map.get(str(asset_class).lower(), C_GRAY)
            summary_rows.append(
                f"<tr><td><span style='color:{row_color}; font-weight:700'>{asset_class}</span></td>"
                f"<td class='mono'>{int(count)}</td>"
                f"<td class='mono'><b style='color:{row_color}'>{pct_total*100:.2f}%</b></td></tr>"
            )

        summary_table = f"""<table>
<tr><th>Asset Class</th><th>Assets</th><th>Weight %</th></tr>
{''.join(summary_rows)}
<tr><td><b>Total</b></td><td class='mono'><b>{int(total)}</b></td><td class='mono'><b>100.00%</b></td></tr>
</table>"""

        details = []
        for asset_class in labels:
            class_df = df[df["asset_class"] == asset_class].copy().sort_values("ticker")
            class_n = len(class_df)
            class_color = color_map.get(str(asset_class).lower(), C_GRAY)
            class_share = (class_n / total) * 100 if total > 0 else 0.0
            ticker_rows = []
            for _, row in class_df.iterrows():
                pct_in_class = (1.0 / class_n) * 100 if class_n > 0 else 0.0
                pct_total = (1.0 / total) * 100 if total > 0 else 0.0
                ticker_rows.append(
                    f"<tr><td>{_display_name(row)}</td>"
                    f"<td class='mono'>{pct_in_class:.2f}%</td>"
                    f"<td class='mono'>{pct_total:.2f}%</td></tr>"
                )

            details.append(
                f"<h3 style='margin-top:14px; color:{class_color}'>{asset_class}"
                f" <span style='font-size:0.82rem; color:#8892b0'>({class_n} assets | {class_share:.2f}% of universe)</span></h3>"
                f"<table><tr><th>Ticker / Name</th><th>% within class</th><th>% of universe</th></tr>{''.join(ticker_rows)}</table>"
            )

        table_html = summary_table + "".join(details)
        return donut_html, table_html

    trad_donut, trad_table = _build_donut(universe, "Traditional Universe by Asset Class")
    hedge_donut, hedge_table = _build_donut(hedge_eligible, "Hedge-Eligible Universe by Asset Class")

    return f"""
<h2>6. Universe Composition</h2>
<div class=\"grid2\">
  <div class=\"card\"><h3>Traditional</h3>{trad_donut}</div>
  <div class=\"card\"><h3>Traditional Breakdown</h3>{trad_table}</div>
</div>
<div class=\"grid2\">
  <div class=\"card\"><h3>Hedge</h3>{hedge_donut}</div>
  <div class=\"card\"><h3>Hedge Breakdown</h3>{hedge_table}</div>
</div>"""


# ---------------------------------------------------------------------------
# Section 5: Portfolio Construction
# ---------------------------------------------------------------------------

def _section_portfolio(weights, turnover, universe, hedge_layer=None) -> str:
    import pandas as pd

    def _build_portfolio_block(block_weights, block_turnover, title, include_fixed_income=False):
        block_weights = block_weights.copy()

        monthly_w = block_weights.resample("ME").last()

        # For Traditional, always include fixed-income sleeves in the allocation stack.
        if include_fixed_income and "asset_class" in universe.columns:
            fi_tickers = universe.loc[universe["asset_class"] == "fixed_income", "ticker"].tolist()
            for t in fi_tickers:
                if t not in block_weights.columns:
                    block_weights[t] = 0.0
            monthly_w = block_weights.resample("ME").last()
            top_core = [t for t in monthly_w.mean().nlargest(8).index.tolist() if t not in fi_tickers]
            selected = top_core + fi_tickers
        else:
            selected = monthly_w.mean().nlargest(8).index.tolist()

        selected = [t for t in selected if t in monthly_w.columns]
        if not selected:
            selected = monthly_w.columns.tolist()[:8]

        fig = go.Figure()
        for ticker in selected:
            fig.add_trace(go.Scatter(
                x=monthly_w.index, y=monthly_w[ticker] * 100,
                name=ticker, stackgroup="one",
                line=dict(width=0.5),
                mode="lines",
            ))
        fig.update_layout(**LAYOUT, title=f"{title} Allocation Over Time",
                          xaxis_title="Date", yaxis_title="Weight (%)")
        _add_time_controls(fig)
        chart_alloc = _fig_to_div(fig)

        monthly_turn = block_turnover.resample("ME").sum() * 100
        mean_turn = float(monthly_turn.mean()) if len(monthly_turn) else 0.0
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=monthly_turn.index, y=monthly_turn.values,
                              marker_color=C_AMBER, name="Turnover"))
        fig2.add_hline(y=mean_turn, line=dict(color=C_GRAY, dash="dash", width=1),
                       annotation_text=f"Mean: {mean_turn:.1f}%")
        fig2.update_layout(**LAYOUT, title=f"{title} Monthly Turnover",
                           xaxis_title="Date", yaxis_title="Turnover (%)")
        _add_time_controls(fig2)
        chart_turn = _fig_to_div(fig2)

        if "sector" in universe.columns:
            last_weights = block_weights.iloc[-1]
            merged = universe.set_index("ticker")["sector"]
            sector_w = last_weights.groupby(merged).sum().sort_values(ascending=True) * 100
            fig3 = go.Figure(go.Bar(
                x=sector_w.values,
                y=sector_w.index.tolist(),
                orientation="h",
                marker_color=C_BLUE,
            ))
            fig3.update_layout(**LAYOUT, title=f"{title} Current Sector Allocation",
                               xaxis_title="Weight (%)", yaxis_title="Sector")
            chart_sector = _fig_to_div(fig3)
        else:
            chart_sector = ""

        return f"""
<div class=\"card\"><h3>{title}</h3>{chart_alloc}</div>
<div class=\"grid2\">
  <div class=\"card\">{chart_turn}</div>
  <div class=\"card\">{chart_sector}</div>
</div>"""

    traditional_block = _build_portfolio_block(
        weights, turnover, "Traditional", include_fixed_income=True
    )

    hedge_block = ""
    if hedge_layer is not None and hedge_layer.get("long_book") is not None and hedge_layer.get("returns") is not None:
        long_book = hedge_layer["long_book"].copy()
        hedge_index = hedge_layer["returns"].index
        hedge_cols = sorted(set(weights.columns).union(set(long_book["ticker"].unique())))
        hedge_weights = pd.DataFrame(0.0, index=hedge_index, columns=hedge_cols)

        if not long_book.empty:
            reb = long_book.pivot_table(index="date", columns="ticker", values="net_weight", aggfunc="sum").fillna(0.0)
            reb = reb.reindex(columns=hedge_cols, fill_value=0.0)
            hedge_weights.update(reb)
            hedge_weights = hedge_weights.replace(0.0, np.nan).ffill().fillna(0.0)

        hedge_turnover = hedge_weights.diff().abs().sum(axis=1).fillna(0.0)
        hedge_block = _build_portfolio_block(
            hedge_weights, hedge_turnover, "Hedge", include_fixed_income=False
        )

    return f"""
<h2>7. Portfolio Construction</h2>
{traditional_block}
{hedge_block}"""


# ---------------------------------------------------------------------------
# Section 6: Stress Testing
# ---------------------------------------------------------------------------

def _section_stress(stress_df, hedge_stress_df=None, stress_distributional=None) -> str:
    """Build stress test section with Traditional and optionally Hedge stress scenarios."""
    
    def _build_stress_table(stress_data, title):
        """Helper to build table HTML for a single stress dataframe."""
        rows = []
        for _, row in stress_data.iterrows():
            mr  = row.get("mean_return", None)
            vol = row.get("volatility", None)
            sh  = row.get("sharpe", None)
            mdd = row.get("max_drawdown", None)
            c_mr  = _color_class(mr,  good_if_positive=True)
            c_mdd = _color_class(mdd, good_if_positive=False)
            rows.append(f"""<tr>
              <td>{row["scenario"]}</td>
              <td class="{c_mr} mono">{_pct(mr)}</td>
              <td class="mono">{_pct(vol)}</td>
              <td class="mono">{_num(sh)}</td>
              <td class="{c_mdd} mono">{_pct(mdd)}</td>
            </tr>""")
        table_html = f"""<table>
          <tr><th>Scenario</th><th>Mean Return</th><th>Volatility</th>
              <th>Sharpe</th><th>Max Drawdown</th></tr>
          {"".join(rows)}
        </table>"""
        return f"<h3>{title}</h3>{table_html}"

    # Traditional block
    traditional_table = _build_stress_table(stress_df, "Traditional Strategy")
    
    # Hedge block (optional)
    hedge_table = ""
    if hedge_stress_df is not None and not hedge_stress_df.empty:
        hedge_table = _build_stress_table(hedge_stress_df, "Hedge Strategy")

    # Chart — Traditional only for now (can be extended to show side-by-side)
    scenarios = stress_df["scenario"].tolist()
    mean_rets  = (stress_df["mean_return"] * 100).tolist()
    mdds       = (stress_df["max_drawdown"] * 100).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Mean Return (%)",
                         x=scenarios, y=mean_rets,
                         marker_color=[C_GREEN if v >= 0 else C_RED for v in mean_rets]))
    fig.add_trace(go.Bar(name="Max Drawdown (%)",
                         x=scenarios, y=mdds,
                         marker_color=C_RED, opacity=0.7))
    
    # Add hedge traces if available
    if hedge_stress_df is not None and not hedge_stress_df.empty:
        hedge_scenarios = hedge_stress_df["scenario"].tolist()
        hedge_mean_rets = (hedge_stress_df["mean_return"] * 100).tolist()
        hedge_mdds = (hedge_stress_df["max_drawdown"] * 100).tolist()
        fig.add_trace(go.Bar(name="Hedge Mean Return (%)",
                             x=hedge_scenarios, y=hedge_mean_rets,
                             marker_color=[C_GREEN if v >= 0 else C_RED for v in hedge_mean_rets],
                             opacity=0.6))
        fig.add_trace(go.Bar(name="Hedge Max Drawdown (%)",
                             x=hedge_scenarios, y=hedge_mdds,
                             marker_color=C_RED, opacity=0.4))
    
    fig.update_layout(**LAYOUT, barmode="group",
                      title="Stress Scenario Impact",
                      xaxis_title="Scenario", yaxis_title="Value (%)")
    chart_stress = _fig_to_div(fig)

    distributional_table = ""
    if stress_distributional:
        rows = []
        for scenario, payload in stress_distributional.items():
            pnl = payload.get("pnl_distribution", {})
            rows.append(
                f"<tr><td>{scenario}</td><td class='mono'>{payload.get('n_historical_windows', 0)}</td>"
                f"<td class='mono'>{_pct(pnl.get('mean'))}</td><td class='mono'>{_pct(pnl.get('p5'))}</td>"
                f"<td class='mono'>{_pct(pnl.get('p25'))}</td><td class='mono'>{_pct(pnl.get('p50'))}</td>"
                f"<td class='mono'>{_pct(pnl.get('p75'))}</td><td class='mono'>{_pct(pnl.get('p95'))}</td>"
                f"<td class='mono'>{_pct(payload.get('cvar_95_pnl'))}</td></tr>"
            )
        distributional_table = (
            "<div class='card'><h3>Distributional Stress Testing</h3>"
            "<table><tr><th>Scenario</th><th>N windows</th><th>Mean</th><th>P5</th><th>P25</th><th>P50</th><th>P75</th><th>P95</th><th>CVaR 95%</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )

    return f"""
<h2>8. Stress Testing</h2>
<div class="card">{traditional_table}</div>
{f'<div class="card">{hedge_table}</div>' if hedge_table else ""}
{distributional_table}
<div class="card">{chart_stress}</div>"""


# ---------------------------------------------------------------------------
# Section 7: FX Overlay & Dynamic Leverage
# ---------------------------------------------------------------------------

def _section_fx_overlay(fx_overlay, leverage_series) -> str:
    charts = []

    # Chart 6.1 — Dynamic hedge ratio
    if fx_overlay is not None and "hedge_ratio" in fx_overlay.columns:
        df = fx_overlay.copy()
        if "date" in df.columns:
            df = df.set_index("date")
        fig = go.Figure()
        fig.add_hrect(y0=0.10, y1=0.95,
                      fillcolor="rgba(79,142,247,0.05)",
                      line_width=0)
        fig.add_trace(go.Scatter(x=df.index, y=df["hedge_ratio"],
                                 name="Hedge Ratio", line=dict(color=C_BLUE, width=2)))
        if "mxn_momentum" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["mxn_momentum"],
                                     name="MXN Momentum", line=dict(color=C_AMBER, dash="dash"),
                                     yaxis="y2", opacity=0.7))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                          showgrid=False, color=C_AMBER,
                                          title="MXN Momentum"))
        fig.update_layout(**LAYOUT, title="Dynamic FX Hedge Ratio",
                          xaxis_title="Date", yaxis_title="Hedge Ratio")
        charts.append(_fig_to_div(fig))

    # Chart 6.2 — Leverage scalar
    if leverage_series is not None:
        lev = leverage_series.resample("W").mean()
        colors_lev = [C_GREEN if v >= 1.0 else C_RED for v in lev.values]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=lev.index, y=lev.values,
                              marker_color=colors_lev, name="Leverage"))
        fig2.add_hline(y=1.0, line=dict(color=C_GRAY, dash="dot", width=1))
        fig2.add_hline(y=0.5, line=dict(color=C_RED, dash="dash", width=1),
                       annotation_text="Min 0.5x")
        fig2.add_hline(y=1.5, line=dict(color=C_GREEN, dash="dash", width=1),
                       annotation_text="Max 1.5x")
        fig2.update_layout(**LAYOUT, title="Dynamic Leverage Scalar (Weekly Avg)",
                           xaxis_title="Date", yaxis_title="Leverage")
        charts.append(_fig_to_div(fig2))

    grid = f'<div class="grid2">{"".join(f"<div class=\'card\'>{c}</div>" for c in charts)}</div>'
    return f"<h2>9. FX Overlay &amp; Dynamic Leverage</h2>{grid}"


# ---------------------------------------------------------------------------
# Section 8: Traditional vs Hedge
# ---------------------------------------------------------------------------

def _section_layer_comparison(metrics, hedge_metrics, tail_hedge, hedge_is_analytical=False) -> str:
    hm = hedge_metrics or {}

    def delta_class(v1, v2, good_if_positive=True):
        if v1 is None or v2 is None:
            return "neutral", "N/A"
        delta = v2 - v1
        c = _color_class(delta, good_if_positive)
        sign = "+" if delta >= 0 else ""
        if abs(delta) < 0.01:
            return c, f"{sign}{delta:.4f}"
        return c, f"{sign}{delta*100:.2f}%"

    rows = []
    for label, key, fmt_fn, gip in [
        ("Annualized Return", "annualized_return", _pct, True),
        ("Annualized Vol",    "annualized_vol",    _pct, False),
        ("Sharpe Ratio",      "sharpe",            lambda v: _num(v, 2), True),
        ("Sortino Ratio",     "sortino",           lambda v: _num(v, 2), True),
        ("Max Drawdown",      "max_drawdown",      _pct, False),
        ("Calmar Ratio",      "calmar",            lambda v: _num(v, 2), True),
        ("CVaR 95%",          "cvar_95",           _pct, False),
    ]:
        v1 = metrics.get(key)
        v2 = hm.get(key)
        c1 = _color_class(v1, gip)
        c2 = _color_class(v2, gip)
        dc, ds = delta_class(v1, v2, gip)
        rows.append(f"""<tr>
          <td>{label}</td>
          <td class="{c1} mono">{fmt_fn(v1)}</td>
          <td class="{c2} mono">{fmt_fn(v2)}</td>
          <td class="{dc} mono">{ds}</td>
        </tr>""")

    header_trad = "Regulated NAV" if hedge_is_analytical else "Traditional"
    header_hedge = "Analytical Overlay" if hedge_is_analytical else "Hedge"

    table = f"""<table>
      <tr><th>Metric</th><th>{header_trad}</th><th>{header_hedge}</th><th>Delta</th></tr>
      {"".join(rows)}
    </table>"""

    # Chart 7.2 — Tail hedge waterfall
    chart_tail = ""
    if tail_hedge:
        labels = ["Unhedged Loss\n@99%", "Hedge Payoff", "Daily Cost Drag", "Net Benefit"]
        values = [
            -abs(tail_hedge.get("unhedged_loss_at_99", 0)),
             abs(tail_hedge.get("hedge_payoff", 0)),
            -abs(tail_hedge.get("daily_cost_drag", 0)),
             tail_hedge.get("net_benefit", 0),
        ]
        bar_colors = [C_RED, C_GREEN, C_AMBER,
                      C_GREEN if tail_hedge.get("net_benefit", 0) >= 0 else C_RED]
        fig = go.Figure(go.Bar(
            x=labels, y=[v * 100 for v in values],
            marker_color=bar_colors,
            text=[f"{v*100:.4f}%" for v in values],
            textposition="outside",
        ))
        recommended = tail_hedge.get("recommended", False)
        fig.update_layout(**LAYOUT,
                          title=f"Tail Hedge Cost-Benefit  |  Recommended: {'YES' if recommended else 'NO'}",
                          xaxis_title="Component", yaxis_title="Value (%)")
        chart_tail = _fig_to_div(fig)

    banner = ""
    if hedge_is_analytical:
        banner = """<div class="alert-banner" style="background:#2a1a00;border-left:4px solid #f59e0b;
       padding:10px 16px;margin-bottom:12px;font-size:13px;color:#fcd34d;">
       ⚠️ <strong>Analytical Overlay — Not Part of Regulated NAV.</strong>
       Las métricas del hedge Layer 2 son un ejercicio paralelo y no forman
       parte del NAV regulatorio reportable ante la CNBV.
     </div>"""

    return f"""
<h2>10. Traditional vs Hedge</h2>
{banner}
<div class="card"><h3>Performance Comparison</h3>{table}</div>
<div class="card">{chart_tail}</div>"""


# ---------------------------------------------------------------------------
# Section: Optimizer Comparison (MV vs min-CVaR)
# ---------------------------------------------------------------------------

def _section_optimizer_comparison(backtest: dict, summary: dict, hedge_mode: bool = False) -> str:
    ret_mv   = backtest["returns"]
    ret_cvar = backtest["returns_cvar"]
    m_mv     = summary["metrics"]
    m_cvar   = summary["metrics_cvar"]

    # Chart — cumulative returns MV vs min-CVaR
    cum_mv   = _cum_from_log(ret_mv)
    cum_cvar = _cum_from_log(ret_cvar)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_mv.index, y=cum_mv.values,
                             name="MV (Media-Varianza)", line=dict(color=C_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=cum_cvar.index, y=cum_cvar.values,
                             name="min-CVaR", line=dict(color=C_PURPLE, width=2)))
    fig.add_hline(y=1.0, line=dict(color=C_GRAY, dash="dot", width=1))
    fig.update_layout(**LAYOUT, title="Cumulative Performance: MV vs min-CVaR",
                      xaxis_title="Date", yaxis_title="Cumulative Return")
    _add_time_controls(fig)
    chart_cum = _fig_to_div(fig)

    # Table — side-by-side metrics with delta column
    def _delta(v1, v2, good_if_pos=True):
        if v1 is None or v2 is None:
            return "neutral", "N/A"
        delta = v2 - v1
        c = _color_class(delta, good_if_pos)
        sign = "+" if delta >= 0 else ""
        if abs(max(abs(v1), abs(v2), 1e-9)) < 5:
            return c, f"{sign}{delta*100:.2f}%"
        return c, f"{sign}{delta:.4f}"

    rows = []
    for label, key, fmt_fn, gip in [
        ("Annualized Return", "annualized_return", _pct,                        True),
        ("Annualized Vol",    "annualized_vol",    _pct,                        False),
        ("Sharpe Ratio",      "sharpe",            lambda v: _num(v, 2),        True),
        ("Sortino Ratio",     "sortino",           lambda v: _num(v, 2),        True),
        ("Max Drawdown",      "max_drawdown",      _pct,                        False),
        ("Calmar Ratio",      "calmar",            lambda v: _num(v, 2),        True),
        ("CVaR 95% (daily)",  "cvar_95",           _pct,                        False),
        ("Avg Turnover",      "turnover",          _pct,                        False),
    ]:
        v1 = m_mv.get(key)
        v2 = m_cvar.get(key)
        c1 = _color_class(v1, gip)
        c2 = _color_class(v2, gip)
        dc, ds = _delta(v1, v2, gip)
        rows.append(f"""<tr>
          <td>{label}</td>
          <td class="{c1} mono">{fmt_fn(v1)}</td>
          <td class="{c2} mono">{fmt_fn(v2)}</td>
          <td class="{dc} mono">{ds}</td>
        </tr>""")

    table = f"""<table>
      <tr><th>Metric</th><th>MV</th><th>min-CVaR</th><th>Delta (CVaR &minus; MV)</th></tr>
      {"".join(rows)}
    </table>"""

    hedge_note = ""
    if hedge_mode:
        hedge_note = """
<div class="card">
    <h3>Hedge</h3>
    <p style=\"color:#8892b0; font-size:0.9rem;\">
        La comparacion MV vs min-CVaR no aplica al Hedge en esta version del pipeline,
        porque el overlay hedge usa un motor especifico de cobertura (sin variante MV/CVaR paralela).
    </p>
</div>"""

    return f"""
<h2>5. Optimizer Comparison</h2>
<div class="card"><h3>Traditional — MV vs min-CVaR</h3>{chart_cum}</div>
<div class="card"><h3>Traditional — Side-by-Side Metrics</h3>{table}</div>
{hedge_note}"""


def _section_hedge_engine_breakdown(hedge_layer: dict, trad_metrics: dict, hedge_metrics: dict | None) -> str:
    base_returns = hedge_layer.get("base_returns")
    leveraged_returns = hedge_layer.get("leveraged_returns")
    fx_pnl = hedge_layer.get("fx_pnl")
    costs = hedge_layer.get("transaction_costs")
    final_returns = hedge_layer.get("returns")
    lev_series = hedge_layer.get("leverage_series")
    params = hedge_layer.get("params", {})
    hm = hedge_metrics or hedge_layer.get("metrics", {})

    chart = ""
    if all(s is not None for s in [base_returns, leveraged_returns, fx_pnl, costs, final_returns]):
        after_fx = leveraged_returns + fx_pnl
        cum_base = _cum_from_log(base_returns)
        cum_lev = _cum_from_log(leveraged_returns)
        cum_fx = _cum_from_log(after_fx)
        cum_final = _cum_from_log(final_returns)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_base.index, y=cum_base.values,
                                 name="Base Book", line=dict(color=C_GRAY, width=1.8)))
        fig.add_trace(go.Scatter(x=cum_lev.index, y=cum_lev.values,
                                 name="+ Dynamic Leverage", line=dict(color=C_PURPLE, width=1.8)))
        fig.add_trace(go.Scatter(x=cum_fx.index, y=cum_fx.values,
                                 name="+ FX Overlay", line=dict(color=C_AMBER, width=1.8)))
        fig.add_trace(go.Scatter(x=cum_final.index, y=cum_final.values,
                                 name="Final Hedge", line=dict(color=C_GREEN, width=2.2)))
        fig.add_hline(y=1.0, line=dict(color=C_GRAY, dash="dot", width=1))
        fig.update_layout(
            **LAYOUT,
            title="Hedge Engine Breakdown: Stage-by-Stage Cumulative Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
        )
        _add_time_controls(fig)
        chart = _fig_to_div(fig)

    lev_mean = float(lev_series.mean()) if lev_series is not None and len(lev_series.dropna()) else None
    lev_max = float(lev_series.max()) if lev_series is not None and len(lev_series.dropna()) else None

    d_ret = (hm.get("annualized_return") or 0) - (trad_metrics.get("annualized_return") or 0)
    d_vol = (hm.get("annualized_vol") or 0) - (trad_metrics.get("annualized_vol") or 0)
    d_cvar = (hm.get("cvar_95") or 0) - (trad_metrics.get("cvar_95") or 0)
    d_mdd = (hm.get("max_drawdown") or 0) - (trad_metrics.get("max_drawdown") or 0)

    rows = [
        ("Configured max_leverage", _num(params.get("max_leverage"), 2), "neutral"),
        ("Configured cvar_limit", _pct(params.get("cvar_limit")), "neutral"),
        ("Configured transaction_cost", _pct(params.get("transaction_cost")), "neutral"),
        ("Observed average leverage", _num(lev_mean, 2), "neutral"),
        ("Observed max leverage", _num(lev_max, 2), "neutral"),
        ("Hedge annualized return", _pct(hm.get("annualized_return")), "neutral"),
        ("Hedge annualized vol", _pct(hm.get("annualized_vol")), "neutral"),
        ("Hedge Sharpe", _num(hm.get("sharpe"), 2), "neutral"),
        ("Hedge CVaR 95% (daily)", _pct(hm.get("cvar_95")), "neutral"),
        ("Hedge max drawdown", _pct(hm.get("max_drawdown")), "neutral"),
        ("Hedge avg turnover", _pct(hm.get("turnover")), "neutral"),
        ("Delta annualized return vs Traditional", _pct(d_ret), _color_class(d_ret, True)),
        ("Delta annualized vol vs Traditional", _pct(d_vol), _color_class(d_vol, False)),
        ("Delta CVaR 95% vs Traditional", _pct(d_cvar), _color_class(d_cvar, True)),
        ("Delta max drawdown vs Traditional", _pct(d_mdd), _color_class(d_mdd, True)),
    ]
    table_rows = "".join([f"<tr><td>{k}</td><td class='{cls} mono'>{v}</td></tr>" for k, v, cls in rows])
    table = f"<table><tr><th>Knob / Outcome</th><th>Value</th></tr>{table_rows}</table>"
    chart_block = f"<div class='card'>{chart}</div>" if chart else ""

    return f"""
<h2>6. Hedge Engine Breakdown</h2>
{chart_block}
<div class=\"card\"><h3>Knobs vs Outcomes</h3>{table}</div>"""


def _build_dynamic_metric_scripts() -> str:
        """Client-side dynamic annotations for selected date ranges."""
        return """
function _toDate(v) {
    if (!v) return null;
    return new Date(v);
}

function _sliceTraceByRange(trace, start, end) {
    if (!trace || !trace.x || !trace.y) return {x: [], y: []};
    const outX = [];
    const outY = [];
    for (let i = 0; i < trace.x.length; i++) {
        const d = new Date(trace.x[i]);
        if ((start === null || d >= start) && (end === null || d <= end)) {
            const y = trace.y[i];
            if (Number.isFinite(y)) {
                outX.push(d);
                outY.push(y);
            }
        }
    }
    return {x: outX, y: outY};
}

function _cagrFromCum(points) {
    if (!points || !points.x || !points.y || points.y.length < 2) return null;
    const first = points.y[0];
    const last = points.y[points.y.length - 1];
    if (!(first > 0) || !(last > 0)) return null;
    const t0 = points.x[0].getTime();
    const t1 = points.x[points.x.length - 1].getTime();
    const years = (t1 - t0) / (365.25 * 24 * 60 * 60 * 1000);
    if (!(years > 0)) return null;
    const totalReturn = (last / first) - 1.0;
    const cagr = Math.pow(last / first, 1.0 / years) - 1.0;
    return { cagr: cagr, totalReturn: totalReturn, years: years };
}

function _updateCagrBox(chartId) {
    const gd = document.getElementById(chartId);
    if (!gd || !gd.data || !gd.layout) return;
    if (gd.__annoBusy) return;
    const xr = (gd.layout.xaxis && gd.layout.xaxis.range) ? gd.layout.xaxis.range : null;
    const start = xr && xr.length ? _toDate(xr[0]) : null;
    const end = xr && xr.length ? _toDate(xr[1]) : null;

    const lines = [];
    for (let i = 0; i < gd.data.length; i++) {
        const tr = gd.data[i];
        if (!tr || tr.visible === 'legendonly') continue;
        const points = _sliceTraceByRange(tr, start, end);
        const stats = _cagrFromCum(points);
        if (stats !== null) {
            lines.push(`${tr.name}: Total ${(stats.totalReturn * 100).toFixed(2)}% | CAGR ${(stats.cagr * 100).toFixed(2)}% (${stats.years.toFixed(2)}y)`);
        }
    }
    const text = lines.length ? lines.join('<br>') : 'CAGR: N/A';
    const curr = (gd.layout.annotations && gd.layout.annotations[0] && gd.layout.annotations[0].text) || '';
    if (curr === text) return;
    gd.__annoBusy = true;
    Promise.resolve(Plotly.relayout(gd, {
        annotations: [{
            xref: 'paper', yref: 'paper', x: 0.01, y: 0.99,
            text: text, showarrow: false, align: 'left',
            bgcolor: 'rgba(15,17,23,0.82)', bordercolor: '#2d3045',
            borderwidth: 1, font: {color: '#e8eaf0', size: 11}
        }]
    })).finally(() => { gd.__annoBusy = false; });
}

function _updateDrawdownChart(chartId) {
    const gd = document.getElementById(chartId);
    if (!gd || !gd.data || !gd.data.length || !gd.layout) return;
    if (gd.__annoBusy) return;
    const xr = (gd.layout.xaxis && gd.layout.xaxis.range) ? gd.layout.xaxis.range : null;
    const start = xr && xr.length ? _toDate(xr[0]) : null;
    const end = xr && xr.length ? _toDate(xr[1]) : null;
    const vals = _sliceTraceByRange(gd.data[0], start, end).y;
    let minV = null;
    if (vals.length) minV = vals.reduce((a, b) => Math.min(a, b), vals[0]);
    const text = minV !== null ? `Max DD (rango): ${minV.toFixed(2)}%` : 'Max DD (rango): N/A';
    const curr = (gd.layout.annotations && gd.layout.annotations[0] && gd.layout.annotations[0].text) || '';
    if (curr === text) return;
    gd.__annoBusy = true;
    Promise.resolve(Plotly.relayout(gd, {
        annotations: [{
            xref: 'paper', yref: 'paper', x: 0.01, y: 0.99,
            text: text, showarrow: false, align: 'left',
            bgcolor: 'rgba(15,17,23,0.82)', bordercolor: '#2d3045',
            borderwidth: 1, font: {color: '#e8eaf0', size: 11}
        }]
    })).finally(() => { gd.__annoBusy = false; });
}

function _wireDynamicAnnotations() {
    for (const id of ['chart-cumulative']) {
        const gd = document.getElementById(id);
        if (!gd) continue;
        const run = () => _updateCagrBox(id);
        gd.on('plotly_relayout', run);
        gd.on('plotly_restyle', run);
        gd.on('plotly_afterplot', run);
        run();
    }
    for (const id of ['chart-benchmarks']) {
        const gd = document.getElementById(id);
        if (!gd) continue;
        const run = () => _updateCagrBox(id);
        gd.on('plotly_relayout', run);
        gd.on('plotly_restyle', run);
        gd.on('plotly_afterplot', run);
        run();
    }
    for (const id of ['chart-drawdown', 'chart-drawdown-hedge']) {
        const dd = document.getElementById(id);
        if (!dd) continue;
        const runDd = () => _updateDrawdownChart(id);
        dd.on('plotly_relayout', runDd);
        dd.on('plotly_afterplot', runDd);
        runDd();
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _wireDynamicAnnotations);
} else {
    _wireDynamicAnnotations();
}
"""


# ---------------------------------------------------------------------------
# Hyperparameter optimization report
# ---------------------------------------------------------------------------

def _section_hyperopt_parallel(history: pd.DataFrame, objective: str) -> str:
    """Parallel-coordinates plot — one line per trial, color = objective value."""
    if history is None or history.empty or "value" not in history.columns:
        return ""

    param_cols = [c for c in history.columns if c not in {"trial_number", "value", "state"}]
    if not param_cols:
        return ""

    df = history.dropna(subset=["value"]).copy()
    if df.empty:
        return ""

    dims = []
    for col in param_cols:
        series = df[col]
        if series.dtype == object:
            codes, uniques = pd.factorize(series)
            dims.append(dict(label=col, values=codes, tickvals=list(range(len(uniques))), ticktext=[str(u) for u in uniques]))
        else:
            values = pd.to_numeric(series, errors="coerce")
            dims.append(dict(label=col, values=values.fillna(values.median()).tolist()))

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=df["value"].values, colorscale="Viridis", showscale=True,
                  colorbar=dict(title=objective)),
        dimensions=dims,
    ))
    fig.update_layout(**LAYOUT, title=f"Parallel coordinates — trials colored by {objective}")
    return f'<div class="card">{_fig_to_div(fig, div_id="hyperopt-parcoords")}</div>'


def _section_hyperopt_importance(study_importance: dict[str, float]) -> str:
    if not study_importance:
        return ""
    items = sorted(study_importance.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [float(v) for _, v in items]
    fig = go.Figure(data=go.Bar(x=vals, y=names, orientation="h", marker_color=C_BLUE))
    fig.update_layout(**LAYOUT, title="Parameter importance", xaxis_title="Importance",
                      yaxis=dict(autorange="reversed"))
    return f'<div class="card">{_fig_to_div(fig, div_id="hyperopt-importance")}</div>'


def _section_hyperopt_convergence(history: pd.DataFrame, objective: str) -> str:
    if history is None or history.empty or "value" not in history.columns:
        return ""
    df = history.sort_values("trial_number")
    running_best = df["value"].cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["trial_number"], y=df["value"], mode="markers",
                             marker=dict(color=C_GRAY, size=6), name="Trial value"))
    fig.add_trace(go.Scatter(x=df["trial_number"], y=running_best, mode="lines",
                             line=dict(color=C_GREEN, width=2), name="Running best"))
    fig.update_layout(**LAYOUT, title=f"Convergence — best {objective} vs trials",
                      xaxis_title="Trial", yaxis_title=objective)
    return f'<div class="card">{_fig_to_div(fig, div_id="hyperopt-convergence")}</div>'


def _section_hyperopt_top_trials(history: pd.DataFrame, top_n: int = 10) -> str:
    if history is None or history.empty or "value" not in history.columns:
        return ""
    df = history.dropna(subset=["value"]).sort_values("value", ascending=False).head(top_n)
    if df.empty:
        return ""
    cols = ["trial_number", "value"] + [c for c in df.columns if c not in {"trial_number", "value", "state"}]
    rows = []
    for _, r in df.iterrows():
        tds = "".join(f"<td>{_num(r[c], 4) if isinstance(r[c], (int, float)) else r[c]}</td>" for c in cols)
        rows.append(f"<tr>{tds}</tr>")
    header = "".join(f"<th>{c}</th>" for c in cols)
    return (
        f'<div class="card"><h3>Top {min(top_n, len(df))} trials</h3>'
        f'<table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _section_hyperopt_validation(validation_metrics: dict, best_params: dict,
                                 best_value: float, objective: str) -> str:
    if not best_params and not validation_metrics:
        return ""
    param_rows = "".join(
        f"<tr><td>{k}</td><td class='mono'>{v}</td></tr>" for k, v in best_params.items()
    )
    metric_rows = "".join(
        f"<tr><td>{k}</td><td class='mono'>{_num(v, 4)}</td></tr>"
        for k, v in validation_metrics.items()
    )
    return (
        f'<div class="grid2">'
        f'<div class="card"><h3>Best params ({objective} = {_num(best_value, 4)})</h3>'
        f'<table><tbody>{param_rows}</tbody></table></div>'
        f'<div class="card"><h3>Validation metrics (walk-forward mean)</h3>'
        f'<table><tbody>{metric_rows}</tbody></table></div>'
        f'</div>'
    )


def generate_hyperopt_report(result, output_path) -> str:
    """Write a standalone HTML report for a hyperopt OptimResult.

    Builds four panels (validation, parallel coordinates, importance,
    convergence) plus the top-10 trial table and writes them to
    `output_path`.  Returns the HTML string.
    """
    import datetime
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = result.trial_history if result.trial_history is not None else pd.DataFrame()
    importance: dict[str, float] = {}
    try:
        import optuna  # noqa: F401
        from optuna.importance import get_param_importances

        if not history.empty and "trial_number" in history.columns:
            # Rebuild a minimal study from the history so we can call importance
            study = _rebuild_study_from_history(result)
            if study is not None and len(study.trials) >= 2:
                importance = dict(get_param_importances(study))
    except Exception:
        importance = {}

    sections = [
        _section_hyperopt_validation(
            result.validation_metrics, result.best_params, result.best_value, result.objective_metric
        ),
        _section_hyperopt_convergence(history, result.objective_metric),
        _section_hyperopt_parallel(history, result.objective_metric),
        _section_hyperopt_importance(importance),
        _section_hyperopt_top_trials(history, top_n=10),
    ]
    body = "\n".join(s for s in sections if s)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FMIA Hyperparameter Optimization Report</title>
{plotly_cdn}
{STYLE}
</head>
<body>
<h1>Hyperparameter Optimization Report</h1>
<p class="meta">
  Objective: <strong>{result.objective_metric}</strong> &nbsp;|&nbsp;
  Trials: <strong>{result.n_trials_completed}</strong> &nbsp;|&nbsp;
  Turnover penalty: <strong>{result.turnover_penalty}</strong> &nbsp;|&nbsp;
  Elapsed: <strong>{result.optimization_time_seconds:.1f}s</strong> &nbsp;|&nbsp;
  Generated: <strong>{timestamp}</strong>
</p>
{body}
</body>
</html>"""
    with open(output_path, "w") as f:
        f.write(html)
    return html


def _rebuild_study_from_history(result):
    """Reconstruct a minimal Optuna study from a trial_history DataFrame.

    Importance analysis needs a study object.  We replay trials into an
    in-memory study so get_param_importances can run without re-executing
    the expensive objective.
    """
    try:
        import optuna
    except ImportError:
        return None
    history = result.trial_history
    if history is None or history.empty:
        return None
    study = optuna.create_study(direction="maximize")
    param_cols = [c for c in history.columns if c not in {"trial_number", "value", "state"}]
    search_space = result.search_space or {}
    for _, row in history.iterrows():
        if not np.isfinite(row.get("value", np.nan)):
            continue
        distributions = {}
        params = {}
        for col in param_cols:
            val = row[col]
            if pd.isna(val):
                continue
            spec = search_space.get(col) or search_space.get(col.replace("__idx", ""))
            if spec is None:
                continue
            kind = spec[0]
            if kind == "float":
                distributions[col] = optuna.distributions.FloatDistribution(spec[1], spec[2], log=bool(spec[3]))
                params[col] = float(val)
            elif kind == "int":
                distributions[col] = optuna.distributions.IntDistribution(int(spec[1]), int(spec[2]), log=bool(spec[3]))
                params[col] = int(val)
            elif kind == "categorical":
                choices = list(range(len(spec[1])))
                distributions[col] = optuna.distributions.CategoricalDistribution(choices)
                params[col] = int(val)
        if not params:
            continue
        trial = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=float(row["value"]),
        )
        study.add_trial(trial)
    return study
