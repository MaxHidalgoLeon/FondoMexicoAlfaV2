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

# ---------------------------------------------------------------------------
# Global theme
# ---------------------------------------------------------------------------

LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1a1d27",
    plot_bgcolor="#1a1d27",
    font=dict(family="system-ui, -apple-system, sans-serif", color="#e8eaf0", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
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
.neutral { color: #e8eaf0; }
.mono { font-family: 'Roboto Mono', 'Courier New', monospace; }
</style>
"""

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fig_to_div(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_dashboard_html(results: dict, hedge_mode: bool, data_source: str) -> str:
    import datetime
    import pandas as pd

    summary    = results["summary"]
    backtest   = results["backtest"]
    feature_df = results["feature_df"]
    universe   = results["data"]["universe"]
    returns    = backtest["returns"]
    weights    = backtest["weights"]
    turnover   = backtest["turnover"]
    metrics    = summary["metrics"]

    hedge_returns  = results["hedge_layer"]["returns"]      if hedge_mode else None
    hedge_metrics  = results["hedge_layer"]["metrics"]      if hedge_mode else None
    hedge_overlay  = results["hedge_layer"]["fx_overlay"]   if hedge_mode else None
    hedge_leverage = results["hedge_layer"]["leverage_series"] if hedge_mode else None
    tail_hedge     = results["hedge_layer"]["tail_hedge"]   if hedge_mode else None

    start_date = summary["start_date"].date()
    end_date   = summary["end_date"].date()
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = []
    sections.append(_section_performance(returns, hedge_returns, metrics, hedge_metrics, hedge_mode))
    sections.append(_section_risk(returns, summary))
    # Optimizer comparison — only when "both" were run
    if summary.get("optimizer") == "both" and summary.get("metrics_cvar") and backtest.get("returns_cvar") is not None:
        sections.append(_section_optimizer_comparison(backtest, summary))
    sections.append(_section_signals(feature_df))
    sections.append(_section_portfolio(weights, turnover, universe))
    sections.append(_section_stress(summary["stress"]))
    if hedge_mode:
        sections.append(_section_fx_overlay(hedge_overlay, hedge_leverage))
        sections.append(_section_layer_comparison(metrics, hedge_metrics, tail_hedge))

    body = "\n".join(sections)

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
  Hedge overlay: <strong>{"Yes" if hedge_mode else "No"}</strong>
</p>
{body}
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Section 1: Performance Overview
# ---------------------------------------------------------------------------

def _section_performance(returns, hedge_returns, metrics, hedge_metrics, hedge_mode) -> str:
    import pandas as pd

    # Chart 1.1 — Cumulative returns
    cum = (1 + returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values,
                             name="Layer 1", line=dict(color=C_BLUE, width=2)))
    if hedge_mode and hedge_returns is not None:
        cum2 = (1 + hedge_returns).cumprod()
        fig.add_trace(go.Scatter(x=cum2.index, y=cum2.values,
                                 name="Layer 2 (Hedge)", line=dict(color=C_GREEN, width=2)))
    fig.add_hline(y=1.0, line=dict(color=C_GRAY, dash="dot", width=1))
    fig.update_layout(**LAYOUT, title="Cumulative Performance",
                      xaxis_title="Date", yaxis_title="Cumulative Return")
    chart_cum = _fig_to_div(fig)

    # Chart 1.2 — Rolling 63-day Sharpe
    roll_sharpe = (returns.rolling(63).mean() / (returns.rolling(63).std() + 1e-9)) * np.sqrt(252)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                              name="Rolling Sharpe (63d)", line=dict(color=C_BLUE, width=1.5),
                              fill="tozeroy",
                              fillcolor="rgba(79,142,247,0.1)"))
    fig2.add_hline(y=0, line=dict(color=C_GRAY, dash="dot", width=1))
    fig2.add_hline(y=1, line=dict(color=C_GREEN, dash="dash", width=1))
    fig2.update_layout(**LAYOUT, title="Rolling Sharpe Ratio (63-day)",
                       xaxis_title="Date", yaxis_title="Sharpe")
    chart_sharpe = _fig_to_div(fig2)

    # Table 1.3 — Key metrics
    def metric_row(label, v1, v2, fmt_fn, good_if_positive=True):
        c1 = _color_class(v1, good_if_positive)
        c1_str = f'<span class="{c1} mono">{fmt_fn(v1)}</span>'
        if hedge_mode and v2 is not None:
            c2 = _color_class(v2, good_if_positive)
            c2_str = f'<span class="{c2} mono">{fmt_fn(v2)}</span>'
            return f"<tr><td>{label}</td><td>{c1_str}</td><td>{c2_str}</td></tr>"
        return f"<tr><td>{label}</td><td>{c1_str}</td></tr>"

    hdr = "<tr><th>Metric</th><th>Layer 1</th>"
    if hedge_mode:
        hdr += "<th>Layer 2</th>"
    hdr += "</tr>"

    hm = hedge_metrics or {}
    rows = [
        metric_row("Annualized Return",  metrics.get("annualized_return"), hm.get("annualized_return"), _pct, True),
        metric_row("Annualized Vol",     metrics.get("annualized_vol"),    hm.get("annualized_vol"),    _pct, False),
        metric_row("Sharpe Ratio",       metrics.get("sharpe"),            hm.get("sharpe"),            lambda v: _num(v, 2), True),
        metric_row("Sortino Ratio",      metrics.get("sortino"),           hm.get("sortino"),           lambda v: _num(v, 2), True),
        metric_row("Max Drawdown",       metrics.get("max_drawdown"),      hm.get("max_drawdown"),      _pct, False),
        metric_row("Calmar Ratio",        metrics.get("calmar"),            hm.get("calmar"),            lambda v: _num(v, 2), True),
        metric_row("CVaR 95% (daily)",   metrics.get("cvar_95"),           hm.get("cvar_95"),           _pct, False),
        metric_row("Avg Turnover",       metrics.get("turnover"),          hm.get("turnover"),          _pct, False),
    ]
    table = f'<table>{hdr}{"".join(rows)}</table>'

    return f"""
<h2>1. Performance Overview</h2>
<div class="card">{chart_cum}</div>
<div class="grid2">
  <div class="card">{chart_sharpe}</div>
  <div class="card"><h3>Key Metrics</h3>{table}</div>
</div>"""


# ---------------------------------------------------------------------------
# Section 2: Risk Analysis
# ---------------------------------------------------------------------------

def _section_risk(returns, summary) -> str:
    # Chart 2.1 — Drawdown
    cum = (1 + returns).cumprod()
    dd = (cum / cum.cummax()) - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100,
                             fill="tozeroy",
                             fillcolor="rgba(231,76,60,0.25)",
                             line=dict(color=C_RED, width=1.5),
                             name="Drawdown"))
    fig.update_layout(**LAYOUT, title="Underwater Equity Curve",
                      xaxis_title="Date", yaxis_title="Drawdown (%)")
    chart_dd = _fig_to_div(fig)

    # Chart 2.2 — Return distribution with VaR/CVaR
    cvar_val = summary["metrics"].get("cvar_95", None)
    var_val  = float(np.percentile(returns.dropna(), 5))
    mean_val = float(returns.mean())

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=returns * 100, nbinsx=60,
                                marker_color=C_BLUE, opacity=0.7,
                                name="Daily Returns"))
    for x_val, color, label, dash in [
        (var_val * 100,                                      C_RED,   "VaR 95%",  "dash"),
        (cvar_val * 100 if cvar_val is not None else None,  C_RED,   "CVaR 95%", "solid"),
        (mean_val * 100,                                     C_GREEN, "Mean",     "dash"),
    ]:
        if x_val is not None:
            fig2.add_vline(x=x_val, line=dict(color=color, dash=dash, width=1.5),
                           annotation_text=label, annotation_position="top")
    fig2.update_layout(**LAYOUT, title="Daily Return Distribution",
                       xaxis_title="Daily Return (%)", yaxis_title="Frequency")
    chart_dist = _fig_to_div(fig2)

    # Chart 2.3 — Rolling 21-day annualized vol
    roll_vol = returns.rolling(21).std() * np.sqrt(252) * 100
    garch_vol = summary.get("garch_vol_forecast")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values,
                              name="Realized Vol (21d)", line=dict(color=C_BLUE, width=1.5)))
    if garch_vol is not None and np.isfinite(garch_vol):
        fig3.add_hline(y=garch_vol * 100,
                       line=dict(color=C_AMBER, dash="dash", width=1.5),
                       annotation_text=f"GARCH Forecast: {garch_vol*100:.1f}%",
                       annotation_position="right")
    fig3.update_layout(**LAYOUT, title="Realized vs GARCH-Forecast Volatility",
                       xaxis_title="Date", yaxis_title="Annualized Vol (%)")
    chart_vol = _fig_to_div(fig3)

    # Table 2.4 — Advanced risk metrics
    def risk_row(label, val, fmt_fn, good_if_positive=True):
        c = _color_class(val, good_if_positive)
        return f'<tr><td>{label}</td><td class="{c} mono">{fmt_fn(val)}</td></tr>'

    rows = [
        risk_row("GARCH Vol Forecast (21d)",  summary.get("garch_vol_forecast"), _pct, False),
        risk_row("Dynamic VaR 95% (GARCH)",   summary.get("dynamic_var"),        _pct, False),
        risk_row("Monte Carlo VaR 95%",        summary.get("monte_carlo_var"),    _pct, False),
        risk_row("GEV VaR 95%",               summary.get("gev_var"),            _pct, False),
        risk_row("GEV CVaR 95%",              summary.get("gev_cvar"),           _pct, False),
    ]
    table = f'<table><tr><th>Risk Metric</th><th>Value</th></tr>{"".join(rows)}</table>'

    return f"""
<h2>2. Risk Analysis</h2>
<div class="card">{chart_dd}</div>
<div class="grid2">
  <div class="card">{chart_dist}</div>
  <div class="card">{chart_vol}</div>
</div>
<div class="card"><h3>Advanced Risk Metrics</h3>{table}</div>"""


# ---------------------------------------------------------------------------
# Section 3: Signal Quality
# ---------------------------------------------------------------------------

def _section_signals(feature_df) -> str:
    import pandas as pd

    feature_cols = [c for c in [
        "momentum_63", "momentum_126", "volatility_63",
        "value_score", "quality_score", "macro_exposure", "liquidity_score"
    ] if c in feature_df.columns]

    # Chart 3.1 — Feature-return cross-sectional correlation
    corr = {}
    if "expected_return" in feature_df.columns:
        for col in feature_cols:
            r = feature_df[col].corr(feature_df["expected_return"])
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
        chart_corr = "<p style='color:#666'>Insufficient data for correlation analysis.</p>"

    # Chart 3.2 — Asset class breakdown (pie)
    if "asset_class" in feature_df.columns:
        counts = feature_df.drop_duplicates("ticker")["asset_class"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            marker=dict(colors=[C_BLUE, C_AMBER, C_GREEN, C_PURPLE]),
            hole=0.4,
        ))
        fig2.update_layout(**LAYOUT, title="Universe by Asset Class")
        chart_pie = _fig_to_div(fig2)
    else:
        chart_pie = ""

    # Chart 3.3 — Monthly signal dispersion (last 24 months)
    if "expected_return" in feature_df.columns and "date" in feature_df.columns:
        df = feature_df.copy()
        df["month"] = df["date"].dt.to_period("M").astype(str)
        last_months = sorted(df["month"].unique())[-24:]
        df = df[df["month"].isin(last_months)]
        fig3 = go.Figure()
        for month in last_months:
            vals = df[df["month"] == month]["expected_return"].dropna()
            if len(vals) > 1:
                fig3.add_trace(go.Box(y=vals.values, name=month,
                                      marker_color=C_BLUE, showlegend=False,
                                      line=dict(width=1)))
        fig3.update_layout(**LAYOUT, title="Cross-Sectional Signal Dispersion (Last 24M)",
                           xaxis_title="Month", yaxis_title="Expected Return (z-score)")
        chart_disp = _fig_to_div(fig3)
    else:
        chart_disp = ""

    return f"""
<h2>3. Signal Quality</h2>
<div class="grid2">
  <div class="card">{chart_corr}</div>
  <div class="card">{chart_pie}</div>
</div>
<div class="card">{chart_disp}</div>"""


# ---------------------------------------------------------------------------
# Section 4: Portfolio Construction
# ---------------------------------------------------------------------------

def _section_portfolio(weights, turnover, universe) -> str:
    import pandas as pd

    # Chart 4.1 — Weights over time (top 8 by average weight)
    monthly_w = weights.resample("ME").last()
    top8 = monthly_w.mean().nlargest(8).index.tolist()
    colors_palette = [C_BLUE, C_GREEN, C_AMBER, C_PURPLE, C_RED,
                      "#1abc9c", "#e67e22", "#3498db"]
    fig = go.Figure()
    for i, ticker in enumerate(top8):
        fig.add_trace(go.Scatter(
            x=monthly_w.index, y=monthly_w[ticker] * 100,
            name=ticker, stackgroup="one",
            line=dict(width=0.5),
            mode="lines",
        ))
    fig.update_layout(**LAYOUT, title="Portfolio Allocation Over Time (Top 8 Positions)",
                      xaxis_title="Date", yaxis_title="Weight (%)")
    chart_alloc = _fig_to_div(fig)

    # Chart 4.2 — Monthly turnover
    monthly_turn = turnover.resample("ME").sum() * 100
    mean_turn = float(monthly_turn.mean())
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=monthly_turn.index, y=monthly_turn.values,
                          marker_color=C_AMBER, name="Turnover"))
    fig2.add_hline(y=mean_turn, line=dict(color=C_GRAY, dash="dash", width=1),
                   annotation_text=f"Mean: {mean_turn:.1f}%")
    fig2.update_layout(**LAYOUT, title="Monthly Portfolio Turnover",
                       xaxis_title="Date", yaxis_title="Turnover (%)")
    chart_turn = _fig_to_div(fig2)

    # Chart 4.3 — Sector allocation (latest)
    if "sector" in universe.columns:
        last_weights = weights.iloc[-1]
        merged = universe.set_index("ticker")["sector"]
        sector_w = last_weights.groupby(merged).sum().sort_values(ascending=True) * 100
        fig3 = go.Figure(go.Bar(
            x=sector_w.values,
            y=sector_w.index.tolist(),
            orientation="h",
            marker_color=C_BLUE,
        ))
        fig3.update_layout(**LAYOUT, title="Current Sector Allocation",
                           xaxis_title="Weight (%)", yaxis_title="Sector")
        chart_sector = _fig_to_div(fig3)
    else:
        chart_sector = ""

    return f"""
<h2>4. Portfolio Construction</h2>
<div class="card">{chart_alloc}</div>
<div class="grid2">
  <div class="card">{chart_turn}</div>
  <div class="card">{chart_sector}</div>
</div>"""


# ---------------------------------------------------------------------------
# Section 5: Stress Testing
# ---------------------------------------------------------------------------

def _section_stress(stress_df) -> str:
    rows = []
    for _, row in stress_df.iterrows():
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
    table = f"""<table>
      <tr><th>Scenario</th><th>Mean Return</th><th>Volatility</th>
          <th>Sharpe</th><th>Max Drawdown</th></tr>
      {"".join(rows)}
    </table>"""

    # Chart 5.2 — Scenario bar chart
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
    fig.update_layout(**LAYOUT, barmode="group",
                      title="Stress Scenario Impact",
                      xaxis_title="Scenario", yaxis_title="Value (%)")
    chart_stress = _fig_to_div(fig)

    return f"""
<h2>5. Stress Testing</h2>
<div class="card"><h3>Scenario Results</h3>{table}</div>
<div class="card">{chart_stress}</div>"""


# ---------------------------------------------------------------------------
# Section 6: FX Overlay & Dynamic Leverage
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
    return f"<h2>6. FX Overlay &amp; Dynamic Leverage</h2>{grid}"


# ---------------------------------------------------------------------------
# Section 7: Layer 1 vs Layer 2 Comparison
# ---------------------------------------------------------------------------

def _section_layer_comparison(metrics, hedge_metrics, tail_hedge) -> str:
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

    table = f"""<table>
      <tr><th>Metric</th><th>Layer 1</th><th>Layer 2</th><th>Delta</th></tr>
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

    return f"""
<h2>7. Layer 1 vs Layer 2 Comparison</h2>
<div class="card"><h3>Performance Comparison</h3>{table}</div>
<div class="card">{chart_tail}</div>"""


# ---------------------------------------------------------------------------
# Section: Optimizer Comparison (MV vs min-CVaR)
# ---------------------------------------------------------------------------

def _section_optimizer_comparison(backtest: dict, summary: dict) -> str:
    ret_mv   = backtest["returns"]
    ret_cvar = backtest["returns_cvar"]
    m_mv     = summary["metrics"]
    m_cvar   = summary["metrics_cvar"]

    # Chart — cumulative returns MV vs min-CVaR
    cum_mv   = (1 + ret_mv).cumprod()
    cum_cvar = (1 + ret_cvar).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_mv.index, y=cum_mv.values,
                             name="MV (Media-Varianza)", line=dict(color=C_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=cum_cvar.index, y=cum_cvar.values,
                             name="min-CVaR", line=dict(color=C_PURPLE, width=2)))
    fig.add_hline(y=1.0, line=dict(color=C_GRAY, dash="dot", width=1))
    fig.update_layout(**LAYOUT, title="Cumulative Performance: MV vs min-CVaR",
                      xaxis_title="Date", yaxis_title="Cumulative Return")
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

    return f"""
<h2>3. Optimizer Comparison: MV vs min-CVaR</h2>
<div class="card">{chart_cum}</div>
<div class="card"><h3>Side-by-Side Metrics</h3>{table}</div>"""
