"""
InfraHeal AI — 2D Visualization Module
=========================================
Plotly-based 2D visualizations for infrastructure metrics,
anomaly detection, and incident analysis.
All plots output Plotly Figures for Gradio gr.Plot embedding.
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

SEVERITY_COLORS = {
    "P1": "#FF3B3B", "P2": "#FFB800", "P3": "#FFD700", "P4": "#00FF88",
    "CRITICAL": "#FF006E", "ERROR": "#FF3B3B", "WARNING": "#FFB800",
    "INFO": "#00D4FF", "DEBUG": "#94a3b8",
}

THEME_BG = "#0a0a1a"
THEME_TEXT = "#e2e8f0"
THEME_GRID = "rgba(255,255,255,0.06)"
ACCENT = "#00D4FF"


def _apply_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply dark theme to any figure."""
    fig.update_layout(
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        font={"color": THEME_TEXT, "family": "Inter, system-ui, sans-serif"},
        title_font={"size": 15, "color": ACCENT},
        margin=dict(l=10, r=10, t=35, b=10),
        legend={"font": {"color": THEME_TEXT}, "bgcolor": "rgba(0,0,0,0)"},
        height=height,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=THEME_GRID, zerolinecolor=THEME_GRID,
                     title_font={"color": "#94a3b8"}, tickfont={"color": "#94a3b8"})
    fig.update_yaxes(gridcolor=THEME_GRID, zerolinecolor=THEME_GRID,
                     title_font={"color": "#94a3b8"}, tickfont={"color": "#94a3b8"})
    return fig


def _empty_figure(message: str = "No data.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=THEME_BG, plot_bgcolor=THEME_BG,
        font={"color": THEME_TEXT}, height=400,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=message, xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(color="#94a3b8", size=14))],
    )
    return fig


def time_series_dashboard(
    metrics: List[Dict[str, Any]],
    anomalies: Optional[List[Dict[str, Any]]] = None,
    title: str = "Metric Time-Series Dashboard",
) -> go.Figure:
    """Multi-panel time-series: CPU, Memory, Latency with anomaly overlays."""
    if not metrics:
        return _empty_figure("No metric data available.")

    timestamps = [m.get("timestamp", "") for m in metrics]
    hosts = [m.get("host", "unknown") for m in metrics]
    cpu = [m.get("cpu_percent", 0) for m in metrics]
    mem = [m.get("memory_percent", 0) for m in metrics]
    lat = [m.get("request_latency_ms", 0) for m in metrics]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=("CPU Utilization (%)", "Memory Utilization (%)", "Latency (ms)"),
                        row_heights=[0.33, 0.33, 0.33])

    for host in sorted(set(hosts)):
        mask = [h == host for h in hosts]
        h_times = [t for t, m in zip(timestamps, mask) if m]
        h_cpu = [c for c, m in zip(cpu, mask) if m]
        h_mem = [m_ for m_, m in zip(mem, mask) if m]
        h_lat = [l for l, m in zip(lat, mask) if m]

        fig.add_trace(go.Scatter(x=h_times, y=h_cpu, mode="lines+markers",
                      name=host, legendgroup=host, line=dict(width=1.5),
                      marker=dict(size=3), hovertemplate=f"{host}<br>%{{x}}<br>CPU: %{{y:.1f}}%<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=h_times, y=h_mem, mode="lines+markers",
                      name=host, legendgroup=host, line=dict(width=1.5),
                      marker=dict(size=3), showlegend=False,
                      hovertemplate=f"{host}<br>%{{x}}<br>Mem: %{{y:.1f}}%<extra></extra>"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=h_times, y=h_lat, mode="lines+markers",
                      name=host, legendgroup=host, line=dict(width=1.5),
                      marker=dict(size=3), showlegend=False,
                      hovertemplate=f"{host}<br>%{{x}}<br>Lat: %{{y:.1f}}ms<extra></extra>"),
                      row=3, col=1)

    # Anomaly markers
    if anomalies:
        for a in anomalies:
            related = a.get("related_metrics", [])
            if not related:
                continue
            r = related[0]
            ts = r.get("timestamp", "")
            sev = a.get("severity", "P3")
            color = SEVERITY_COLORS.get(sev, "#FFD700")
            label = f"{a.get('id', '')}: {a.get('description', '')[:60]}"

            for row_num, field, val in [
                (1, "cpu_percent", r.get("cpu_percent")),
                (2, "memory_percent", r.get("memory_percent")),
                (3, "request_latency_ms", r.get("request_latency_ms")),
            ]:
                if val is not None:
                    fig.add_trace(go.Scatter(x=[ts], y=[val], mode="markers",
                                  marker=dict(size=10, color=color, symbol="diamond",
                                              line=dict(width=1, color="white")),
                                  name=label, legendgroup=label,
                                  showlegend=(row_num == 1),
                                  hovertemplate=f"{label}<br>Severity: {sev}<br>%{{x}}<br>%{{y:.1f}}<extra></extra>"),
                                  row=row_num, col=1)

    # Threshold lines
    for row_num, threshold, label_text in [
        (1, 90, "CPU Critical (90%)"),
        (2, 85, "Memory Critical (85%)"),
        (3, 5000, "Latency Critical (5s)"),
    ]:
        fig.add_hline(y=threshold, line=dict(color="#FF3B3B", width=1, dash="dash"),
                      annotation_text=label_text, annotation_font_color="#FF3B3B",
                      annotation_font_size=10, row=row_num, col=1)

    _apply_theme(fig, height=650)
    fig.update_layout(title=dict(text=title, x=0.5))
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="Timestamp", row=3, col=1)
    fig.update_xaxes(tickangle=45, row=3, col=1)
    return fig


def correlation_heatmap(
    metrics: List[Dict[str, Any]],
    title: str = "Metric Correlation Heatmap",
) -> go.Figure:
    """Correlation matrix heatmap of CPU, Memory, Latency, Disk I/O."""
    if not metrics:
        return _empty_figure("No metric data available.")

    fields = ["cpu_percent", "memory_percent", "request_latency_ms"]
    labels = ["CPU %", "Memory %", "Latency (ms)"]

    data = {f: [m.get(f, 0) for m in metrics] for f in fields}
    import numpy as np
    n = len(fields)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = np.array(data[fields[i]]), np.array(data[fields[j]])
            if a.std() < 1e-10 or b.std() < 1e-10:
                corr[i][j] = 0.0
            else:
                corr[i][j] = float(np.corrcoef(a, b)[0, 1])

    fig = go.Figure(data=go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale=[[0, "#1a0533"], [0.5, "#0a0a1a"], [1, "#00D4FF"]],
        zmin=-1, zmax=1,
        text=[[f"{corr[i][j]:.2f}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}",
        textfont=dict(size=14, color=THEME_TEXT),
        hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
    ))

    _apply_theme(fig, height=420)
    fig.update_layout(title=dict(text=title, x=0.5))
    fig.update_xaxes(side="bottom")
    return fig


def anomaly_timeline(
    incidents: List[Dict[str, Any]],
    title: str = "Anomaly Timeline — Severity vs Time",
) -> go.Figure:
    """Horizontal bar timeline showing anomalies grouped by host over time."""
    if not incidents:
        return _empty_figure("No anomaly incidents to visualize.")

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, inc in enumerate(incidents):
        anomalies = inc.get("anomalies", [])
        if not anomalies:
            continue
        host = inc.get("primary_source", anomalies[0].get("source", "unknown"))
        color = colors[idx % len(colors)]

        for a in anomalies:
            ts = a.get("timestamp", "")
            sev = a.get("severity", "P4")
            sev_rank = {"P1": 4, "P2": 3, "P3": 2, "P4": 1}.get(sev, 1)
            desc = a.get("description", "Unknown")[:50]

            fig.add_trace(go.Bar(
                x=[sev_rank],
                y=[host],
                orientation="h",
                marker=dict(color=color, opacity=0.8,
                            line=dict(color="white", width=0.5)),
                name=f"{inc.get('incident_id', '')} - {desc}",
                hovertemplate=(
                    f"<b>{inc.get('incident_id', '')}</b><br>"
                    f"Host: {host}<br>Severity: {sev}<br>"
                    f"{desc}<br>Time: {ts}<extra></extra>"
                ),
                showlegend=(idx == 0),
                width=0.6,
            ))

    _apply_theme(fig, height=350)
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Severity Rank", tickvals=[1, 2, 3, 4],
                   ticktext=["P4 (Low)", "P3 (Medium)", "P2 (High)", "P1 (Critical)"]),
        yaxis=dict(title="Host", autorange="reversed"),
        barmode="stack",
        bargap=0.3,
    )
    return fig


def log_level_distribution(
    logs: List[Dict[str, Any]],
    title: str = "Log Level Distribution by Source",
) -> go.Figure:
    """Grouped bar chart of log levels per source."""
    if not logs:
        return _empty_figure("No log data available.")

    pairs = Counter()
    for log in logs:
        src = log.get("source", "unknown")
        lvl = log.get("level", "INFO").upper()
        pairs[(src, lvl)] += 1

    sources = sorted(set(p[0] for p in pairs))
    levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
    colors_map = {"CRITICAL": "#FF006E", "ERROR": "#FF3B3B", "WARNING": "#FFB800",
                  "INFO": "#00D4FF", "DEBUG": "#94a3b8"}

    fig = go.Figure()
    for lvl in levels:
        vals = [pairs.get((src, lvl), 0) for src in sources]
        if sum(vals) == 0:
            continue
        fig.add_trace(go.Bar(
            name=lvl, x=sources, y=vals,
            marker_color=colors_map.get(lvl, "#94a3b8"),
            hovertemplate=f"Level: {lvl}<br>Source: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))

    _apply_theme(fig, height=350)
    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode="group",
        xaxis=dict(title="Source", tickangle=45),
        yaxis=dict(title="Count"),
        legend=dict(title="Log Level"),
    )
    return fig


def host_radar(
    metrics: List[Dict[str, Any]],
    title: str = "Host Comparison — Radar",
) -> go.Figure:
    """Radar chart comparing average metrics across hosts."""
    if not metrics:
        return _empty_figure("No metric data available.")

    host_data = defaultdict(list)
    for m in metrics:
        host_data[m.get("host", "unknown")].append(m)

    categories = ["CPU %", "Memory %", "Latency (ms)"]
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, (host, vals) in enumerate(sorted(host_data.items())):
        avg_cpu = sum(v.get("cpu_percent", 0) for v in vals) / len(vals)
        avg_mem = sum(v.get("memory_percent", 0) for v in vals) / len(vals)
        avg_lat = sum(v.get("request_latency_ms", 0) for v in vals) / len(vals)

        # Normalize for radar display
        max_vals = {"CPU %": 100, "Memory %": 100, "Latency (ms)": max(100, avg_lat * 1.5)}
        norm_cpu = avg_cpu / max_vals["CPU %"] * 100
        norm_mem = avg_mem / max_vals["Memory %"] * 100
        norm_lat = avg_lat / max_vals["Latency (ms)"] * 100

        fig.add_trace(go.Scatterpolar(
            r=[norm_cpu, norm_mem, norm_lat],
            theta=categories,
            fill="toself",
            name=host,
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate=(
                f"<b>{host}</b><br>"
                f"CPU: {avg_cpu:.1f}%<br>"
                f"Memory: {avg_mem:.1f}%<br>"
                f"Latency: {avg_lat:.1f}ms<br>"
                "<extra></extra>"
            ),
        ))

    _apply_theme(fig, height=350)
    fig.update_layout(
        title=dict(text=title, x=0.5),
        polar=dict(
            bgcolor=THEME_BG,
            radialaxis=dict(visible=True, range=[0, 100],
                           gridcolor=THEME_GRID, color="#94a3b8"),
            angularaxis=dict(gridcolor=THEME_GRID, color="#94a3b8"),
        ),
        legend=dict(x=0.8, y=0.9),
    )
    return fig
