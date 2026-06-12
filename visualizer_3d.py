"""
InfraHeal AI — 3D Visualization Module
=========================================
Plotly-based 3D visualizations for infrastructure metrics,
anomaly clusters, and incident correlation graphs.

All plots output HTML for embedding in Gradio or Jupyter.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Color map consistent with the dashboard theme ──────────────
SEVERITY_COLORS = {
    "P1": "#FF3B3B",
    "P2": "#FFB800",
    "P3": "#FFD700",
    "P4": "#00FF88",
    "CRITICAL": "#FF006E",
    "ERROR": "#FF3B3B",
    "WARNING": "#FFB800",
    "INFO": "#00D4FF",
    "DEBUG": "#94a3b8",
}

THEME_BG = "#0a0a1a"
THEME_TEXT = "#e2e8f0"
THEME_GRID = "rgba(255,255,255,0.06)"


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply the InfraHeal dark theme to a plotly figure."""
    fig.update_layout(
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        font={"color": THEME_TEXT, "family": "Inter, system-ui, sans-serif"},
        title_font={"size": 16, "color": "#00D4FF"},
        margin=dict(l=20, r=20, t=40, b=20),
        legend={"font": {"color": THEME_TEXT}},
    )
    fig.update_xaxes(
        gridcolor=THEME_GRID,
        zerolinecolor=THEME_GRID,
        title_font={"color": "#94a3b8"},
        tickfont={"color": "#94a3b8"},
    )
    fig.update_yaxes(
        gridcolor=THEME_GRID,
        zerolinecolor=THEME_GRID,
        title_font={"color": "#94a3b8"},
        tickfont={"color": "#94a3b8"},
    )
    return fig


def metrics_3d_scatter(
    metrics: List[Dict[str, Any]],
    anomalies: Optional[List[Dict[str, Any]]] = None,
    title: str = "3D Metric Space — CPU / Memory / Latency",
    as_html: bool = True,
) -> str:
    """3D scatter plot of metrics with CPU, Memory, Latency axes.

    Args:
        metrics: List of metric dicts with cpu_percent, memory_percent,
                 request_latency_ms, host.
        anomalies: Optional list of anomaly dicts to overlay.
        title: Plot title.
        as_html: If True return HTML string, else return Plotly Figure.

    Returns:
        HTML string (as_html=True) or Plotly Figure (as_html=False).
    """
    if not metrics:
        return _empty_plot("No metric data available for 3D visualization.") if as_html else _empty_figure("No metric data available for 3D visualization.")

    fig = go.Figure()

    # Prepare data
    cpu_vals = [m.get("cpu_percent", 0) for m in metrics]
    mem_vals = [m.get("memory_percent", 0) for m in metrics]
    lat_vals = [m.get("request_latency_ms", 0) for m in metrics]
    hosts = [m.get("host", "unknown") for m in metrics]
    timestamps = [m.get("timestamp", "") for m in metrics]

    # Normalize latency for color scaling
    max_lat = max(lat_vals) if lat_vals and max(lat_vals) > 0 else 1

    fig.add_trace(go.Scatter3d(
        x=cpu_vals,
        y=mem_vals,
        z=lat_vals,
        mode="markers",
        marker=dict(
            size=5,
            color=lat_vals,
            colorscale="Viridis",
            cmin=0,
            cmax=max_lat,
            colorbar=dict(title="Latency (ms)", x=1.02),
            line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
        ),
        text=[f"Host: {h}<br>CPU: {c:.1f}%<br>Mem: {m:.1f}%<br>Lat: {l:.1f}ms<br>{t}"
              for h, c, m, l, t in zip(hosts, cpu_vals, mem_vals, lat_vals, timestamps)],
        hoverinfo="text",
        name="Metric Points",
    ))

    # Overlay anomalies if provided
    if anomalies:
        ax, ay, az = [], [], []
        a_labels = []
        a_colors = []
        for a in anomalies:
            related = a.get("related_metrics", [])
            if related:
                r = related[0]
                ax.append(r.get("cpu_percent", 0))
                ay.append(r.get("memory_percent", 0))
                az.append(r.get("request_latency_ms", 0))
            else:
                continue
            sev = a.get("severity", "P3")
            a_labels.append(f"{a.get('id', '')}<br>{a.get('description', '')[:80]}<br>Severity: {sev}")
            a_colors.append(SEVERITY_COLORS.get(sev, "#FFD700"))

        if ax:
            fig.add_trace(go.Scatter3d(
                x=ax, y=ay, z=az,
                mode="markers",
                marker=dict(
                    size=10,
                    color=a_colors,
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                ),
                text=a_labels,
                hoverinfo="text",
                name="Anomalies",
            ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="CPU (%)",
            yaxis_title="Memory (%)",
            zaxis_title="Latency (ms)",
            xaxis=dict(gridcolor=THEME_GRID, backgroundcolor=THEME_BG),
            yaxis=dict(gridcolor=THEME_GRID, backgroundcolor=THEME_BG),
            zaxis=dict(gridcolor=THEME_GRID, backgroundcolor=THEME_BG),
            bgcolor=THEME_BG,
        ),
        height=600,
    )
    _apply_dark_theme(fig)
    if not as_html:
        return fig
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def anomaly_clusters_3d(
    incidents: List[Dict[str, Any]],
    title: str = "3D Anomaly Clusters — Incident Correlation",
    as_html: bool = True,
) -> str:
    """3D visualization of correlated incident clusters.

    Each incident is a group of anomalies displayed as a 3D scatter
    with severity on the Z-axis and time on X.

    Args:
        incidents: List of incident dicts from IncidentCorrelator.
        title: Plot title.
        as_html: If True return HTML string, else return Plotly Figure.

    Returns:
        HTML string (as_html=True) or Plotly Figure (as_html=False).
    """
    if not incidents:
        return _empty_plot("No incident clusters to visualize.") if as_html else _empty_figure("No incident clusters to visualize.")

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly
    for idx, inc in enumerate(incidents):
        anomalies = inc.get("anomalies", [])
        if not anomalies:
            continue

        xs, ys, zs, labels = [], [], [], []
        for a in anomalies:
            ts = a.get("timestamp", "")
            try:
                x_val = datetime.fromisoformat(ts).timestamp()
            except (ValueError, TypeError):
                x_val = idx
            sev_rank = {"P1": 3, "P2": 2, "P3": 1, "P4": 0}.get(a.get("severity", "P4"), 0)
            xs.append(x_val)
            ys.append(hash(a.get("source", "")) % 100)  # source as y-position
            zs.append(sev_rank)
            labels.append(
                f"Incident: {inc.get('incident_id', '')}<br>"
                f"Severity: {a.get('severity', '')}<br>"
                f"Source: {a.get('source', '')}<br>"
                f"Type: {a.get('type', '')}<br>"
                f"{a.get('description', '')[:80]}"
            )

        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+lines",
            marker=dict(size=8, color=color, symbol="circle",
                        line=dict(width=0.5, color="white")),
            line=dict(color=color, width=2, dash="dot"),
            text=labels,
            hoverinfo="text",
            name=f"{inc.get('incident_id', '')} — {inc.get('primary_severity', '')}",
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Source (hash)",
            zaxis_title="Severity Rank",
            xaxis=dict(gridcolor=THEME_GRID, backgroundcolor=THEME_BG),
            yaxis=dict(gridcolor=THEME_GRID, backgroundcolor=THEME_BG, showticklabels=False),
            zaxis=dict(
                gridcolor=THEME_GRID, backgroundcolor=THEME_BG,
                tickvals=[0, 1, 2, 3], ticktext=["P4", "P3", "P2", "P1"],
            ),
            bgcolor=THEME_BG,
        ),
        height=600,
        showlegend=True,
    )
    _apply_dark_theme(fig)
    if not as_html:
        return fig
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def log_level_distribution_3d(
    logs: List[Dict[str, Any]],
    title: str = "3D Log Level Distribution — Source vs Level vs Count",
    as_html: bool = True,
) -> str:
    """3D bar chart showing log level distribution per source.

    Args:
        logs: List of log dicts.
        title: Plot title.
        as_html: If True return HTML string, else return Plotly Figure.

    Returns:
        HTML string (as_html=True) or Plotly Figure (as_html=False).
    """
    if not logs:
        return _empty_plot("No log data available.") if as_html else _empty_figure("No log data available.")

    from collections import Counter

    # Count (source, level) pairs
    pairs = Counter()
    for log in logs:
        src = log.get("source", "unknown")
        lvl = log.get("level", "INFO").upper()
        pairs[(src, lvl)] += 1

    sources = sorted(set(p[0] for p in pairs))
    levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

    # Build 3D bar chart
    fig = go.Figure()
    colors_map = {"CRITICAL": "#FF006E", "ERROR": "#FF3B3B", "WARNING": "#FFB800",
                  "INFO": "#00D4FF", "DEBUG": "#94a3b8"}

    for lvl in levels:
        z_vals = []
        for src in sources:
            z_vals.append(pairs.get((src, lvl), 0))
        if sum(z_vals) == 0:
            continue
        fig.add_trace(go.Bar(
            name=lvl,
            x=sources,
            y=z_vals,
            marker_color=colors_map.get(lvl, "#94a3b8"),
            hovertemplate=f"Level: {lvl}<br>Source: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode="stack",
        xaxis=dict(title="Source", tickangle=45),
        yaxis=dict(title="Count"),
        height=500,
        legend=dict(title="Log Level"),
        barnorm="",
    )
    _apply_dark_theme(fig)
    if not as_html:
        return fig
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def timeline_3d_surface(
    metrics: List[Dict[str, Any]],
    metric_field: str = "cpu_percent",
    title: str = "3D Metric Surface — Time vs Host",
    as_html: bool = True,
) -> str:
    """3D surface plot of a metric over time across hosts.

    Args:
        metrics: List of metric dicts.
        metric_field: Field name to plot (cpu_percent, memory_percent, etc.).
        title: Plot title.
        as_html: If True return HTML string, else return Plotly Figure.

    Returns:
        HTML string (as_html=True) or Plotly Figure (as_html=False).
    """
    if not metrics:
        return _empty_plot("No metric data available.") if as_html else _empty_figure("No metric data available.")

    # Group by host
    from collections import defaultdict
    host_data = defaultdict(list)
    for m in metrics:
        host_data[m.get("host", "unknown")].append(m)

    hosts = sorted(host_data.keys())
    if len(hosts) < 2:
        msg = "Need at least 2 hosts for surface plot."
        return _empty_plot(msg) if as_html else _empty_figure(msg)

    # Get common time points
    all_times = sorted(set(
        m.get("timestamp", "") for m in metrics
        if m.get(metric_field) is not None
    ))
    if len(all_times) < 2:
        msg = "Need at least 2 time points for surface plot."
        return _empty_plot(msg) if as_html else _empty_figure(msg)

    # Build Z matrix: hosts x times
    time_index = {t: i for i, t in enumerate(all_times)}
    z_matrix = []
    for host in hosts:
        row = [None] * len(all_times)
        for m in host_data[host]:
            ts = m.get("timestamp", "")
            val = m.get(metric_field)
            if ts in time_index and val is not None:
                row[time_index[ts]] = val
        # Fill gaps
        last_val = None
        for i in range(len(row)):
            if row[i] is None:
                row[i] = last_val or 0
            else:
                last_val = row[i]
        z_matrix.append(row)

    fig = go.Figure(data=[go.Surface(
        z=z_matrix,
        x=list(range(len(all_times))),
        y=hosts,
        colorscale="Viridis",
        hovertemplate=(
            f"Host: %{{y}}<br>"
            f"Time index: %{{x}}<br>"
            f"{metric_field}: %{{z:.1f}}<extra></extra>"
        ),
        colorbar=dict(title=metric_field),
    )])

    # Show every Nth time label
    step = max(1, len(all_times) // 10)
    tick_vals = list(range(0, len(all_times), step))
    tick_text = [all_times[i][11:19] if len(all_times[i]) > 11 else str(i) for i in tick_vals]

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(
                title="Time", tickvals=tick_vals, ticktext=tick_text,
                gridcolor=THEME_GRID, backgroundcolor=THEME_BG,
            ),
            yaxis=dict(title="Host", gridcolor=THEME_GRID, backgroundcolor=THEME_BG),
            zaxis=dict(title=metric_field, gridcolor=THEME_GRID, backgroundcolor=THEME_BG),
            bgcolor=THEME_BG,
            aspectratio=dict(x=2, y=1, z=0.8),
        ),
        height=600,
    )
    _apply_dark_theme(fig)
    if not as_html:
        return fig
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _empty_figure(message: str = "No data.") -> go.Figure:
    """Return a dark-themed empty figure with annotation message."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        font={"color": THEME_TEXT},
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="#94a3b8", size=14),
        )],
    )
    return fig


def _empty_plot(message: str = "No data.") -> str:
    """Return an empty state HTML for when no data is available."""
    return (
        f'<div style="display:flex;align-items:center;justify-content:center;'
        f'height:400px;background:#0a0a1a;border:1px solid rgba(255,255,255,0.06);'
        f'border-radius:14px;color:#94a3b8;font-size:0.92rem;">'
        f'📊 {message}</div>'
    )
