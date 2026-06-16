"""
InfraHeal AI — Premium Gradio Dashboard
=========================================
Autonomous Incident Diagnosis & Resolution Agent
Dark glassmorphism theme with neon accents.
Built for TCS & AMD AI Hackathon 2026.
"""

import logging
import os
import time
import json
import math
import html
import warnings
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Diagnostic logger (writes structured JSON lines to infraheal/diagnostics.log) ──
_DIAG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics.log")
_diag_handler = logging.FileHandler(_DIAG_LOG, mode="a", encoding="utf-8")
_diag_handler.setFormatter(logging.Formatter('%(message)s'))
_diag_logger = logging.getLogger("infraheal.diag")
_diag_logger.setLevel(logging.DEBUG)
_diag_logger.handlers.clear()
_diag_logger.addHandler(_diag_handler)
_diag_logger.propagate = False

def _diag(event: str, **kw):
    """Write a structured JSON line to diagnostics.log."""
    record = {"t": datetime.now().strftime("%H:%M:%S.%f")[:-3], "e": event, **kw}
    _diag_logger.debug(json.dumps(record))

_diag("init", file=_DIAG_LOG)

warnings.filterwarnings("ignore", message=".*422.*")
import gradio as gr
import plotly.graph_objects as go

from visualizer_2d import (
    time_series_dashboard,
    correlation_heatmap,
    anomaly_timeline,
    log_level_distribution,
    host_radar,
    draw_topology_map,
)
from log_streamer import LogStreamer, LiveAnalyzer

# Live log stream state
_live_log_cache: List[Dict[str, Any]] = []
_live_log_lock = threading.Lock()

# Pending human approvals queue
_pending_approvals: List[Dict[str, Any]] = []
_approval_history: List[Dict[str, Any]] = []
_approval_id_counter = 0
_blocked_scenarios: Dict[str, bool] = {}
_auto_approve: bool = False
_monitoring_active = False
_stop_monitoring_requested = False
MONITOR_POLL_SECONDS = 30
MONITOR_POLL_INTERVAL = 60  # mutable; updated by dropdown in UI

# Live pipeline output: background thread writes, gr.Timer polls
# Separated per pipeline type so process and monitor don't overwrite each other.
_process_live_html: str = ""
_monitor_live_html: str = ""
_live_html_lock = threading.Lock()
_process_thread: Optional[threading.Thread] = None
_monitor_thread: Optional[threading.Thread] = None
_process_completed: bool = False
_monitoring_completed: bool = False
_scenario_results: Dict[str, Dict[str, Any]] = {}  # latest result per scenario name
_static_output_active: bool = False  # True when user clicked report/scan/optimize; suppresses iframe overwrite

# Approval audit log (persistent)
APPROVAL_AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approval_audit.json")

# Head HTML injected into Gradio page (must be passed to launch())
HEAD_HTML = """<style>.agent-modal{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;z-index:9999}.agent-modal-box{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px;max-width:460px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,0.5)}.agent-modal-title{font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:12px}.agent-modal-body{font-size:0.85rem;color:#8b949e;margin-bottom:20px;line-height:1.5}.agent-modal-input{width:100%;padding:10px 12px;background:#0d1117;border:1px solid #30363d;border-radius:8px;color:#e2e8f0;font-size:0.85rem;outline:none;box-sizing:border-box;margin-bottom:16px}.agent-modal-input:focus{border-color:#58a6ff}.agent-modal-actions{display:flex;gap:10px;justify-content:flex-end}.agent-modal-btn{padding:8px 20px;border-radius:8px;border:1px solid;font-size:0.82rem;font-weight:600;cursor:pointer;background:transparent}.agent-modal-btn-primary{background:#00FF8822;border-color:#00FF88;color:#00FF88}.agent-modal-btn-primary:hover{background:#00FF8833}.agent-modal-btn-danger{background:#FF3B3B22;border-color:#FF3B3B;color:#FF3B3B}.agent-modal-btn-danger:hover{background:#FF3B3B33}.agent-modal-btn-cancel{border-color:#30363d;color:#8b949e}.agent-modal-btn-cancel:hover{background:rgba(255,255,255,0.05)}#refresh-btn{display:none!important}</style>"""

def _append_audit_log(entry: dict):
    """Append to persistent approval audit log."""
    try:
        entries = []
        if os.path.exists(APPROVAL_AUDIT_PATH):
            with open(APPROVAL_AUDIT_PATH, "r", encoding="utf-8") as f:
                entries = json.load(f)
        entries.append(entry)
        with open(APPROVAL_AUDIT_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to write approval audit log: %s", exc)

def _load_audit_log() -> list:
    try:
        if os.path.exists(APPROVAL_AUDIT_PATH):
            with open(APPROVAL_AUDIT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read approval audit log: %s", exc)
    return []

def _render_audit_log() -> str:
    entries = _load_audit_log()
    if not entries:
        return ""
    rows = "".join(
        f'<tr>'
        f'<td style="font-size:0.76rem;white-space:nowrap;">{e.get("action","")}</td>'
        f'<td style="font-size:0.76rem;">{e.get("id","")}</td>'
        f'<td style="font-size:0.76rem;">{e.get("title","")[:24]}</td>'
        f'<td style="font-size:0.76rem;">{e.get("scenario","")[:20]}</td>'
        f'<td style="font-size:0.76rem;color:#8b949e;">{e.get("reason","")[:40]}</td>'
        f'<td style="font-size:0.76rem;color:#8b949e;white-space:nowrap;">{str(e.get("timestamp",""))[:19]}</td>'
        f'</tr>'
        for e in reversed(entries[-50:])
    )
    return (
        '<div style="margin-top:16px;">'
        '<div style="font-size:0.8rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
        f'Approval Audit Log ({len(entries)})</div>'
        '<table class="styled-table" style="font-size:0.76rem;">'
        '<thead><tr><th>Action</th><th>ID</th><th>Tool</th><th>Scenario</th><th>Reason/Outcome</th><th>Timestamp</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )

# Experience store for continuous learning
_experience_store: List[Dict[str, Any]] = []
_action_preferences: Dict[str, Dict[str, float]] = {}  # tool_name -> {approved, denied, total}
EXPERIENCE_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experience_store.json")

def _save_experience_store():
    """Persist experience store and action preferences to JSON."""
    try:
        data = {
            "experiences": _experience_store,
            "preferences": _action_preferences,
        }
        with open(EXPERIENCE_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to save experience store: %s", exc)

def _load_experience_store():
    """Load experience store and action preferences from JSON."""
    global _experience_store, _action_preferences
    try:
        if os.path.exists(EXPERIENCE_STORE_PATH):
            with open(EXPERIENCE_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            _experience_store = data.get("experiences", [])
            _action_preferences = data.get("preferences", {})
            logger.info("Loaded %d experiences, %d action types from store", len(_experience_store), len(_action_preferences))
    except Exception as exc:
        logger.warning("Failed to load experience store: %s", exc)

def _add_experience(entry: dict):
    """Add an experience entry and update action preferences."""
    global _experience_store, _action_preferences
    _experience_store.append(entry)
    # Update action preferences
    for action in entry.get("actions_attempted", []):
        tool = action.get("tool_name", "unknown")
        if tool not in _action_preferences:
            _action_preferences[tool] = {"approved": 0, "denied": 0, "total": 0}
        _action_preferences[tool]["total"] = _action_preferences[tool].get("total", 0) + 1
        if action.get("approved", False) or entry.get("verdict") == "approved":
            _action_preferences[tool]["approved"] = _action_preferences[tool].get("approved", 0) + 1
        if action.get("denied", False) or entry.get("verdict") == "denied":
            _action_preferences[tool]["denied"] = _action_preferences[tool].get("denied", 0) + 1
    _save_experience_store()

def _get_few_shot_examples(severity: str, category: str, max_examples: int = 3) -> str:
    """Retrieve top similar past successful remediations as few-shot string."""
    if not _experience_store:
        return ""
    # Score by matching severity and category, then take most recent
    scored = []
    for exp in _experience_store:
        score = 0
        if exp.get("severity") == severity:
            score += 2
        if exp.get("category") == category:
            score += 3
        if exp.get("verdict") == "approved":
            score += 1
        scored.append((score, exp))
    scored.sort(key=lambda x: (-x[0], x[1].get("timestamp", "")))
    examples = scored[:max_examples]
    parts = []
    for _, exp in examples:
        actions = exp.get("actions_attempted", [])
        approved_actions = [a for a in actions if not a.get("denied")]
        if not approved_actions:
            continue
        parts.append(
            f"[Past] sev={exp.get('severity')} cat={exp.get('category')}"
            f" rc=\"{exp.get('root_cause','')[:60]}\""
            f" actions={[a['tool_name'] for a in approved_actions]}"
        )
    return "\n".join(parts) if parts else ""

def _get_preference_ranking() -> str:
    """Generate action preference string from historical approval rates."""
    if not _action_preferences:
        return ""
    ranked = sorted(
        _action_preferences.items(),
        key=lambda x: x[1].get("approved", 0) / max(x[1].get("total", 1), 1),
        reverse=True,
    )
    parts = []
    for tool, prefs in ranked:
        rate = prefs.get("approved", 0) / max(prefs.get("total", 1), 1) * 100
        parts.append(f"{tool}: {rate:.0f}% approval ({prefs.get('approved',0)}/{prefs.get('total',0)})")
    return " | ".join(parts)

logger = logging.getLogger("infraheal.dashboard")

# Load on startup
_load_experience_store()

# Use DATA_DIR from config for reliable path resolution
try:
    from config import DATA_DIR as _CFG_DATA_DIR
except ImportError:
    _CFG_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")

def _load_metrics_for_3d() -> List[Dict[str, Any]]:
    """Load full metrics from sample_data for richer 3D plots."""
    # Try all possible paths
    candidates = [
        os.path.join(_CFG_DATA_DIR, "metrics.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "metrics.json"),
        os.path.join(os.getcwd(), "sample_data", "metrics.json"),
    ]
    tried = set()
    for path in candidates:
        if path in tried:
            continue
        tried.add(path)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if data:
                    hosts = set(x.get("host") for x in data)
                    logger.info(
                        "Loaded %d metrics (%d hosts) from %s for 3D plots",
                        len(data), len(hosts), path,
                    )
                    return data
            except Exception as exc:
                logger.warning("Failed to load metrics from %s: %s", path, exc)
    logger.warning("No metrics found in %s — 3D plots will be empty", tried)
    return []

warnings.filterwarnings("ignore", message="The parameters have been moved")

try:
    from .config import (
        SEVERITY_LEVELS, INCIDENT_CATEGORIES, AVAILABLE_TOOLS,
        DASHBOARD_HOST, DASHBOARD_PORT, MODEL_NAME, VLLM_BASE_URL,
    )
except ImportError:
    from config import (
        SEVERITY_LEVELS, INCIDENT_CATEGORIES, AVAILABLE_TOOLS,
        DASHBOARD_HOST, DASHBOARD_PORT, MODEL_NAME, VLLM_BASE_URL,
        MODEL_REGISTRY, THINKING_TAGS,
    )

# ═══════════════════════════════════════════════════════════════════
#  COLOR PALETTE  (B&W + severity-only)
# ═══════════════════════════════════════════════════════════════════
_C = {
    "bg_primary":   "#0a0a1a",
    "bg_secondary": "#111128",
    "bg_card":      "rgba(17, 17, 40, 0.65)",
    "border":       "rgba(255, 255, 255, 0.08)",
    "border_hover": "rgba(255, 255, 255, 0.25)",
    "text":         "#e2e8f0",
    "text_muted":   "#64748b",
    "red":          "#FF3B3B",
    "amber":        "#FFB800",
    "green":        "#00FF88",
}

def _risk_color(risk: str) -> str:
    return {"high": _C["red"], "medium": _C["amber"], "low": _C["green"]}.get(risk.lower(), _C["amber"])

# ═══════════════════════════════════════════════════════════════════
#  CUSTOM CSS  (~250 lines)
# ═══════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
/* ── Global ────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary:   #0a0a1a;
  --bg-secondary: #111128;
  --bg-card:      rgba(17,17,40,0.65);
  --border:       rgba(255,255,255,0.08);
  --border-hover: rgba(255,255,255,0.25);
  --text:         #e2e8f0;
  --text-muted:   #64748b;
  --accent:       #e2e8f0;
  --red:          #FF3B3B;
  --amber:        #FFB800;
  --green:        #00FF88;
}

.gradio-container {
  background: var(--bg-primary) !important;
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  color: var(--text) !important;
  max-width: 1440px !important;
}

/* ── Card ──────────────────────────────────────────────────────── */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  transition: border-color 0.25s ease;
}
.glass-card:hover {
  border-color: var(--border-hover);
}

/* ── Tab Bar ───────────────────────────────────────────────────── */
.tabs > .tab-nav {
  background: var(--bg-secondary) !important;
  border-bottom: 1px solid var(--border) !important;
  border-radius: 10px 10px 0 0 !important;
  padding: 4px 8px !important;
  gap: 2px !important;
}
.tabs > .tab-nav > button {
  background: transparent !important;
  color: var(--text-muted) !important;
  border: 1px solid transparent !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  padding: 8px 16px !important;
  transition: background 0.2s ease, color 0.2s ease !important;
}
.tabs > .tab-nav > button:hover {
  color: var(--text) !important;
  background: rgba(255,255,255,0.06) !important;
}
.tabs > .tab-nav > button.selected {
  background: rgba(255,255,255,0.1) !important;
  color: #ffffff !important;
}
.tabs > .tab-nav > button .tab-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-left: 6px;
  font-size: 0.6rem;
  font-weight: 700;
  color: #0d1117;
  transition: background 0.3s ease;
}
.tabs > .tab-nav > button .tab-badge.has-pending {
  background: #FFB800;
  box-shadow: 0 0 6px rgba(255,184,0,0.6);
}
.tabs > .tab-nav > button .tab-badge.all-clear {
  background: #00FF88;
  box-shadow: 0 0 6px rgba(0,255,136,0.4);
}

/* ── Buttons ───────────────────────────────────────────────────── */
.gr-button-primary, button.primary {
  background: #e2e8f0 !important;
  border: none !important;
  color: #0a0a1a !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  padding: 10px 24px !important;
  font-size: 0.9rem !important;
  transition: opacity 0.2s ease, transform 0.15s ease !important;
}
.gr-button-primary:hover, button.primary:hover {
  opacity: 0.8 !important;
  transform: translateY(-1px) !important;
}
.gr-button-primary:active, button.primary:active {
  transform: translateY(0px) !important;
  opacity: 0.9 !important;
}
button.secondary {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  padding: 10px 24px !important;
  font-size: 0.9rem !important;
  transition: border-color 0.2s ease, background 0.2s ease, transform 0.15s ease !important;
}
button.secondary:hover {
  border-color: var(--text-muted) !important;
  background: rgba(255,255,255,0.08) !important;
  transform: translateY(-1px) !important;
}
button.secondary:active {
  transform: translateY(0px) !important;
}

/* ── Inputs / Textboxes / Dropdowns ────────────────────────────── */
.gr-input, .gr-text-input, textarea, input[type="text"],
.gr-dropdown, select {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
  transition: border-color 0.3s ease !important;
}
.gr-input:focus, textarea:focus, select:focus {
  border-color: rgba(255,255,255,0.3) !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,0.08) !important;
}

/* ── Accordion ─────────────────────────────────────────────────── */
.gr-accordion {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
.gr-accordion > .label-wrap {
  color: var(--text) !important;
  font-weight: 600 !important;
}

/* ── Severity Badges ───────────────────────────────────────────── */
.severity-p1 { background: #FF3B3B; color: #fff; }
.severity-p2 { background: #FF8C00; color: #fff; }
.severity-p3 { background: #FFD700; color: #1a1a2e; }
.severity-p4 { background: #4CAF50; color: #fff; }

.status-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 12px; border-radius: 16px;
  font-size: 0.72rem; font-weight: 600;
}

/* ── Pulse Animation ───────────────────────────────────────────── */
@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.5; }
}
.pulse-dot {
  display: inline-block; width: 6px; height: 6px;
  background: var(--green); border-radius: 50%;
  animation: pulse-dot 2s ease-in-out infinite;
}
.pulse-dot-red {
  display: inline-block; width: 6px; height: 6px;
  background: var(--red); border-radius: 50%;
  animation: pulse-dot 1.5s ease-in-out infinite;
}

/* ── Scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.3); }

/* ── Table ─────────────────────────────────────────────────────── */
.styled-table {
  width: 100%; border-collapse: separate; border-spacing: 0;
  font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;
}
.styled-table thead th {
  background: rgba(255,255,255,0.06); color: var(--text);
  padding: 10px 12px; text-align: left; font-weight: 600;
  border-bottom: 1px solid var(--border); letter-spacing: 0.5px;
  font-size: 0.72rem; white-space: nowrap;
}
.styled-table tbody td {
  padding: 10px 12px; vertical-align: middle;
  font-size: 0.82rem;
}
.styled-table tbody tr { transition: background 0.2s ease; }
.styled-table tbody tr:nth-child(even) { background: rgba(255,255,255,0.015); }
.styled-table tbody tr:hover { background: rgba(255,255,255,0.04); }
.styled-table tbody td {
  padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.04);
  color: var(--text);
}

/* ── Metric Cards ──────────────────────────────────────────────── */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px;
  text-align: center; position: relative; overflow: hidden;
}
.metric-value {
  font-size: 2rem; font-weight: 700; line-height: 1.1;
  color: var(--accent);
}
.metric-label {
  font-size: 0.7rem; color: var(--text-muted);
  letter-spacing: 0.8px; margin-top: 4px; font-weight: 500;
}

/* ── Agent Output Panels ───────────────────────────────────────── */
.agent-panel {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 8px; padding: 0; overflow: hidden;
}
.agent-panel-header {
  padding: 8px 12px;
  font-weight: 600; font-size: 0.8rem;
  letter-spacing: 0.5px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
}
.agent-panel-body { padding: 12px; }

.evidence-item {
  background: rgba(255,255,255,0.015); border-left: 2px solid var(--text-muted);
  padding: 8px 12px; margin-bottom: 6px; border-radius: 0 6px 6px 0;
  font-size: 0.82rem;
}
.action-card {
  background: rgba(255,255,255,0.015);
  border: 1px solid rgba(255,255,255,0.04);
  border-radius: 6px; padding: 8px 12px; margin-bottom: 4px;
  display: flex; align-items: center; gap: 8px;
}
.action-card .action-icon {
  width: 20px; height: 20px; min-width: 20px; flex-shrink: 0;
  border-radius: 4px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 0.65rem; font-weight: 600; color: #fff;
  background: rgba(255,255,255,0.12);
}

/* ── Utility Classes ────────────────────────────────────────────── */
.section-title {
  font-size: 1.05rem; font-weight: 600; color: var(--text);
  margin-bottom: 4px;
}
.section-subtitle {
  font-size: 0.8rem; color: var(--text-muted); margin-bottom: 16px;
}
.section-label {
  font-size: 0.72rem; font-weight: 600; color: var(--text-muted);
  letter-spacing: 0.8px; margin-bottom: 12px;
}
.flex-row { display: flex; align-items: center; gap: 8px; }
.flex-gap { gap: 12px; }
.text-accent { color: var(--accent); }
.text-muted { color: var(--text-muted); }
.text-sm { font-size: 0.82rem; }
.mt-4 { margin-top: 4px; }
.mb-8 { margin-bottom: 8px; }
.mb-16 { margin-bottom: 16px; }

.divider {
  height: 1px; background: rgba(255,255,255,0.06); margin: 16px 0;
}

/* ── Loading / Skeleton ────────────────────────────────────────── */
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
.loading-skeleton {
  background: linear-gradient(90deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.03) 100%);
  background-size: 400px 100%;
  animation: shimmer 1.8s infinite;
  border-radius: 10px; height: 120px;
}

/* ── Agent Chat ────────────────────────────────────────────────── */
.chat-container {
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 12px;
  padding: 12px;
  height: 360px;
  overflow-y: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
}
.chat-msg {
  display: flex;
  padding: 6px 4px;
  border-bottom: 1px solid #21262d;
  position: relative;
}
.chat-msg:last-child { border-bottom: none; }
.chat-msg.user { justify-content: flex-end; }
.chat-msg.assistant { justify-content: flex-start; }
.chat-msg .chat-bubble {
  max-width: 80%;
  line-height: 1.5;
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 0.82rem;
}
.chat-msg.user .chat-bubble {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  color: #c9d1d9;
}
.chat-msg.assistant .chat-bubble {
  background: transparent;
  color: #e2e8f0;
}
.chat-msg .chat-bubble strong { color: #ffffff; font-weight: 700; }
.chat-msg .chat-bubble em { color: #8b949e; font-style: italic; }
.chat-msg .chat-bubble code {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 0.78rem;
  color: #60A5FA;
}
/* Copy button: hidden by default, show on hover */
.chat-msg .chat-copy-btn {
  display: none;
  position: absolute;
  top: 8px;
  right: 4px;
  background: #21262d;
  border: 1px solid #30363d;
  color: #8b949e;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 0.7rem;
  font-family: 'JetBrains Mono', monospace;
  cursor: pointer;
  transition: all 0.15s ease;
}
.chat-msg:hover .chat-copy-btn {
  display: inline-flex;
}
.chat-msg .chat-copy-btn:hover {
  background: #30363d;
  color: #e2e8f0;
}
.chat-status-bar {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 10px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: #8b949e;
  display: flex;
  align-items: center;
  gap: 12px;
}
.chat-status-bar .status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block;
}
.chat-status-bar .status-dot.green { background: var(--green); }
.chat-status-bar .status-dot.yellow { background: var(--amber); }
.chat-status-bar .status-dot.gray { background: #484f58; }
.chat-quick-btn {
  background: #21262d !important;
  border: 1px solid #30363d !important;
  color: #c9d1d9 !important;
  border-radius: 6px !important;
  font-size: 0.8rem !important;
  font-family: 'JetBrains Mono', monospace !important;
  padding: 6px 14px !important;
  transition: all 0.2s ease !important;
}
.chat-quick-btn:hover {
  border-color: rgba(255,255,255,0.4) !important;
  color: #ffffff !important;
  background: rgba(255,255,255,0.08) !important;
}
.chat-send-btn {
  min-width: 40px !important;
  font-size: 1.1rem !important;
  background: #1a6cff22 !important;
  border-color: #1a6cff !important;
  color: #1a6cff !important;
  padding: 4px 12px !important;
}
.chat-send-btn:hover {
  background: #1a6cff44 !important;
  color: #58a6ff !important;
}
.chat-send-btn:disabled,
.chat-send-btn button:disabled {
  opacity: 0.3 !important;
  cursor: not-allowed !important;
  background: #21262d !important;
  border-color: #30363d !important;
  color: #484f58 !important;
}
.chat-clear-btn {
  min-width: 40px !important;
  font-size: 1rem !important;
  padding: 4px 12px !important;
}

/* Chat input: inline row with textbox + buttons */
.chat-input-row {
  background: #0d1117 !important;
  border: 1px solid #30363d !important;
  border-radius: 8px !important;
  margin-top: 6px !important;
  padding: 2px 2px !important;
  gap: 2px !important;
  align-items: stretch !important;
  flex-wrap: nowrap !important;
  min-height: 38px !important;
}
.chat-input-row > * {
  align-self: center !important;
  margin: 0 !important;
}
.chat-input-row .form-control,
.chat-input-row .wrap,
.chat-input-row .wrap-inner {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
.chat-input-row textarea,
.chat-input-row input[type="text"] {
  border: none !important;
  background: transparent !important;
  padding: 8px 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82rem !important;
  color: #c9d1d9 !important;
  min-height: 40px !important;
  max-height: 100px !important;
  resize: none !important;
  outline: none !important;
  box-shadow: none !important;
  line-height: 1.4 !important;
}
.chat-send-btn, .chat-clear-btn {
  min-width: 32px !important;
  height: 34px !important;
  padding: 0 !important;
  font-size: 1.1rem !important;
  border-radius: 4px !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  color: #8b949e !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
  position: relative !important;
  overflow: visible !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}
.chat-send-btn *,
.chat-clear-btn * {
  padding: 0 !important;
  margin: 0 !important;
  line-height: 1 !important;
}
.chat-send-btn {
  color: #1a6cff !important;
}
.chat-send-btn:disabled,
.chat-send-btn button:disabled {
  opacity: 0.25 !important;
  cursor: not-allowed !important;
}
.chat-send-btn:hover:not(:disabled),
.chat-clear-btn:hover {
  background: #21262d !important;
  border-color: #30363d !important;
}
#chat-send-btn:hover:not(:disabled)::after {
  content: "Send";
  position: absolute;
  bottom: calc(100% + 8px);
  right: 0;
  background: #161b22;
  border: 1px solid #30363d;
  color: #c9d1d9;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 0.7rem;
  white-space: nowrap;
  font-family: 'JetBrains Mono', monospace;
  z-index: 9999;
  pointer-events: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
#chat-clear-btn:hover::after {
  content: "Clear";
  position: absolute;
  bottom: calc(100% + 8px);
  right: 0;
  background: #161b22;
  border: 1px solid #30363d;
  color: #c9d1d9;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 0.7rem;
  white-space: nowrap;
  font-family: 'JetBrains Mono', monospace;
  z-index: 9999;
  pointer-events: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
/* Model selector: black & white */
#agent-chat-tab select,
#agent-chat-tab .gr-dropdown,
#agent-chat-tab .gr-dropdown select,
#agent-chat-tab .gr-dropdown .gr-box,
#agent-chat-tab label,
#agent-chat-tab .gr-form,
#agent-chat-tab .gr-input-field,
#agent-chat-tab .gr-dropdown-container,
#agent-chat-tab .choices-wrapper,
#agent-chat-tab .wrap-inner,
#agent-chat-tab .dropdown-option {
  background: #0d1117 !important;
  border-color: #30363d !important;
  color: #c9d1d9 !important;
  font-family: 'JetBrains Mono', monospace !important;
  border-radius: 6px !important;
}
#agent-chat-tab select:focus,
#agent-chat-tab .gr-dropdown:focus {
  border-color: #8b949e !important;
  box-shadow: none !important;
}
/* Model selector label */
#agent-chat-tab label,
#agent-chat-tab .label-text,
#agent-chat-tab .gr-dropdown label {
  color: #8b949e !important;
  font-size: 0.75rem !important;
  text-transform: uppercase !important;
  letter-spacing: 1px !important;
  font-weight: 600 !important;
}
/* Chat input and model selector: black & white */
#agent-chat-tab .gr-box,
#agent-chat-tab .gr-form {
  background: transparent !important;
  border-color: #30363d !important;
  color: #c9d1d9 !important;
}
/* Approval command input inside accordion */
#approval-cmd-input {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
  background: #0d1117 !important;
  border: 1px solid #30363d !important;
  border-radius: 6px !important;
  color: #c9d1d9 !important;
  padding: 8px 12px !important;
  width: 100% !important;
  margin-top: 8px !important;
  box-sizing: border-box !important;
}
#approval-cmd-input:focus {
  border-color: #58a6ff !important;
}
#approval-trigger-btn {
  margin-top: 6px !important;
}
/* ── Misc ─────────────────────────────────────────────────────── */
.gr-group { border: none !important; }
.gr-padded { padding: 0 !important; }
.gr-form { background: transparent !important; border: none !important; }
.gr-box { background: transparent !important; border: none !important; }
.gr-panel { background: transparent !important; }
label { color: var(--text-muted) !important; font-weight: 500 !important; }
.gr-check-radio label { color: var(--text) !important; }
footer { display: none !important; }

/* ── Rerun Buttons (black & white) ──────────────────────────── */
.rerun-btn {
  min-width: 32px !important;
  width: 32px !important;
  height: 32px !important;
  padding: 0 !important;
  font-size: 1.1rem !important;
  border-radius: 6px !important;
  border: 1px solid #484f58 !important;
  background: #0d1117 !important;
  color: #c9d1d9 !important;
  cursor: pointer !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: all 0.15s ease !important;
  line-height: 1 !important;
  box-shadow: none !important;
}
.rerun-btn * {
  padding: 0 !important;
  margin: 0 !important;
  line-height: 1 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}
.rerun-btn:hover {
  background: #21262d !important;
  border-color: #8b949e !important;
  color: #ffffff !important;
}
.rerun-btn:active {
  transform: rotate(90deg);
}
.stop-btn {
  min-width: 32px !important;
  width: 32px !important;
  height: 32px !important;
  padding: 0 !important;
  font-size: 22px !important;
  border-radius: 6px !important;
  border: 1.5px solid #ff3b3b !important;
  background: #0d1117 !important;
  color: #ff3b3b !important;
  cursor: pointer !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: all 0.15s ease !important;
  line-height: 1 !important;
  box-shadow: none !important;
  position: relative !important;
}
.stop-btn * {
  padding: 0 !important;
  margin: 0 !important;
  line-height: 1 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: none !important;
  border: none !important;
  box-shadow: none !important;
}
.stop-btn:hover {
  background: #21262d !important;
  border-color: #ff6b6b !important;
}
.stop-btn:hover::after {
  content: "Stop" !important;
  position: absolute !important;
  top: -28px !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  font-size: 0.72rem !important;
  background: #21262d !important;
  color: #e2e8f0 !important;
  padding: 2px 8px !important;
  border-radius: 4px !important;
  white-space: nowrap !important;
  border: 1px solid #484f58 !important;
  font-weight: 500 !important;
  z-index: 100 !important;
}

/* ── Pipeline Flow ──────────────────────────────────────────────── */
.pipeline-flow { margin-bottom:20px; }
.pipeline-header { display:flex; align-items:center; gap:12px; padding:14px 16px; background:#161b22; border:1px solid #30363d; border-radius:10px; margin-bottom:12px; flex-wrap:wrap; }
.pipeline-title { font-size:0.85rem; font-weight:700; color:#e2e8f0; flex:1; }
.pipeline-status { font-size:0.75rem; font-weight:600; padding:4px 10px; border-radius:12px; white-space:nowrap; }
.pipeline-status.running { background:rgba(88,166,255,0.15); color:#58a6ff; }
.pipeline-status.completed { background:rgba(0,255,136,0.15); color:#00FF88; }
.pipeline-status.failed { background:rgba(255,59,59,0.15); color:#FF3B3B; }
.pipeline-status.warning { background:rgba(255,184,0,0.15); color:#FFB800; }
.pipeline-elapsed { font-size:0.72rem; color:#8b949e; font-family:'JetBrains Mono',monospace; white-space:nowrap; }
.pipeline-steps { display:flex; flex-direction:column; gap:2px; position:relative; padding-left:20px; }
.pipeline-steps::before { content:''; position:absolute; left:9px; top:12px; bottom:12px; width:2px; background:#21262d; }
.pipeline-step { display:flex; align-items:center; gap:12px; padding:10px 14px; background:#0d1117; border:1px solid #21262d; border-radius:6px; position:relative; transition:all 0.2s; }
.pipeline-step:hover { border-color:#30363d; background:#161b22; }
.pipeline-step-icon { width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.7rem; flex-shrink:0; position:relative; z-index:1; }
.pipeline-step-icon.pending { background:#21262d; border:2px solid #30363d; color:#484f58; }
.pipeline-step-icon.running { background:rgba(88,166,255,0.2); border:2px solid #58a6ff; animation:pulse 1.2s infinite; }
.pipeline-step-icon.completed { background:rgba(0,255,136,0.2); border:2px solid #00FF88; }
.pipeline-step-icon.completed::after { content:'✓'; color:#00FF88; font-weight:700; }
.pipeline-step-icon.failed { background:rgba(255,59,59,0.2); border:2px solid #FF3B3B; }
.pipeline-step-icon.failed::after { content:'✗'; color:#FF3B3B; font-weight:700; }
.pipeline-step-icon.warning { background:rgba(255,184,0,0.2); border:2px solid #FFB800; }
.pipeline-step-icon.warning::after { content:'⚠'; color:#FFB800; font-weight:700; }
@keyframes pulse { 0%,100% { box-shadow:0 0 0 0 rgba(88,166,255,0.4); } 50% { box-shadow:0 0 0 6px rgba(88,166,255,0); } }
.pipeline-step-info { flex:1; min-width:0; }
.pipeline-step-name { font-size:0.8rem; font-weight:600; color:#c9d1d9; }
.pipeline-step-desc { font-size:0.72rem; color:#8b949e; margin-top:1px; }
.pipeline-step-duration { font-size:0.7rem; color:#64748b; font-family:'JetBrains Mono',monospace; white-space:nowrap; margin-left:auto; padding-left:12px; }
.pipeline-step-progress { width:80px; height:4px; background:#21262d; border-radius:2px; overflow:hidden; margin-left:8px; flex-shrink:0; }
.pipeline-step-progress-bar { height:100%; border-radius:2px; transition:width 0.5s ease; }
.pipeline-step-progress-bar.completed { background:#00FF88; }
.pipeline-step-progress-bar.running { background:#58a6ff; }
.pipeline-step-progress-bar.failed { background:#FF3B3B; }
.pipeline-step-progress-bar.warning { background:#FFB800; }
#refresh-btn{display:none!important}

/* ── Poll Interval Dropdown (compact, black, no box) ─────────── */
#poll-interval { 
  min-width: 90px !important; 
  max-width: 120px !important; 
}
#poll-interval .gr-dropdown-container,
#poll-interval .gr-box, 
#poll-interval .gr-dropdown,
#poll-interval .choices-wrapper {
  background: #000 !important; 
  border: 1px solid #30363d !important;
  color: #c9d1d9 !important; 
  border-radius: 6px !important;
  height: 32px !important;
}
#poll-interval select {
  background:#000 !important; color:#c9d1d9 !important;
  border:none !important; font-size:0.75rem !important;
  padding:2px 4px !important; margin:0 !important;
  min-height:26px !important; height:26px !important;
}

/* ── Collapsible pipeline steps ──────────────────────────────── */
.pipeline-step { cursor:pointer; }
.pipeline-step.collapsed .pipeline-step-desc,
.pipeline-step.collapsed .pipeline-step-progress,
.pipeline-step.collapsed .pipeline-step-duration { display:none !important; }
.pipeline-step:hover { background:rgba(255,255,255,0.03); }
/* ── Per-cycle collapsible (monitoring) ──────────────────────── */
.pipeline-cycle-group { }
.pipeline-cycle-group > .pipeline-step.cycle-header { cursor:pointer; }
.pipeline-cycle-group > .pipeline-step.cycle-header .pipeline-step-name::before { content:'\\25bc '; font-size:0.6rem; color:#8b949e; }
.pipeline-cycle-group.collapsed > .pipeline-cycle-children { display:none !important; }
.pipeline-cycle-group.collapsed > .pipeline-step.cycle-header .pipeline-step-name::before { content:'\\25b6 '; }
.pipeline-step.cycle-header { background:rgba(255,255,255,0.04); border-left:2px solid #8b949e; }
/* ── Center table headers ────────────────────────────────────── */
.styled-table th { text-align:center !important; }
"""


# ═══════════════════════════════════════════════════════════════════
#  HTML FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════

def _branding_header() -> str:
    """Return the hero header HTML with InfraHeal AI branding."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:28px 36px;
                background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.06);border-radius:18px;
                margin-bottom:24px;">
      <div style="display:flex;align-items:center;gap:18px;">
        <div style="width:54px;height:54px;border-radius:14px;
                    background:rgba(255,255,255,0.1);
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.6rem;">
          🛡️
        </div>
        <div>
          <div style="font-size:1.6rem;font-weight:800;color:{_C["text"]};letter-spacing:-0.5px;">
            InfraHeal AI
          </div>
          <div style="font-size:0.78rem;color:#64748b;letter-spacing:0.6px;">
            Autonomous Incident Diagnosis &amp; Resolution Agent
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:20px;">
        <div style="text-align:right;">
          <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Status</div>
          <div style="display:flex;align-items:center;gap:6px;margin-top:2px;">
            <span class="pulse-dot"></span>
            <span style="font-size:0.82rem;color:#00FF88;font-weight:600;">Operational</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Last Refresh</div>
          <div style="font-size:0.82rem;color:#e2e8f0;font-weight:500;margin-top:2px;">{now}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Model</div>
          <div style="font-size:0.82rem;color:{_C["text"]};font-weight:500;margin-top:2px;">
            {MODEL_NAME.split("/")[-1]}
          </div>
        </div>
      </div>
    </div>
    """


def format_severity_badge(severity: str) -> str:
    """Return HTML for a colored severity badge.

    Args:
        severity: One of P1, P2, P3, P4.

    Returns:
        Inline-styled HTML ``<span>`` element.
    """
    info = SEVERITY_LEVELS.get(severity.upper(), SEVERITY_LEVELS["P3"])
    bg_map = {
        "P1": "linear-gradient(135deg,#FF3B3B,#FF3B3B)",
        "P2": "linear-gradient(135deg,#FF8C00,#FFB800)",
        "P3": "linear-gradient(135deg,#FFD700,#FFA500)",
        "P4": "linear-gradient(135deg,#4CAF50,#00FF88)",
    }
    txt_color = "#fff" if severity.upper() == "P1" else "#0a0a1a"
    bg = bg_map.get(severity.upper(), bg_map["P3"])
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;padding:4px 14px;'
        f'border-radius:20px;font-size:0.75rem;font-weight:700;letter-spacing:0.8px;'
        f'text-transform:uppercase;background:{bg};color:{txt_color};">'
        f'{severity.upper()} — {info["label"]}'
        f'</span>'
    )


def _hl(text: str) -> str:
    """Wrap key technical terms in light-blue spans. HTML-escapes first."""
    import html as _html
    import re as _re
    t = _html.escape(str(text))  # escape HTML first so <> in original text are safe
    # Severity/criticality labels
    t = _re.sub(r'\b(P[1-4])\b', r'<span style="color:#60A5FA;font-weight:600;">\1</span>', t)
    t = _re.sub(r'\b(CRITICAL|ERROR|FATAL|WARNING)\b', r'<span style="color:#60A5FA;font-weight:600;">\1</span>', t)
    # Service/host names (alphanumeric with hyphens/underscores, 5+ chars)
    t = _re.sub(r'\b([a-z][a-z0-9_-]{5,30})\b', lambda m: f'<span style="color:#60A5FA;">{m.group(1)}</span>' if any(c in m.group(1) for c in '-_') else m.group(1), t)
    # HTTP status codes
    t = _re.sub(r'\b([45]\d{2})\b', r'<span style="color:#60A5FA;font-weight:600;">\1</span>', t)
    # URLs
    t = _re.sub(r'(https?://[^\s<]+)', r'<span style="color:#60A5FA;">\1</span>', t)
    # IP addresses
    t = _re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', r'<span style="color:#60A5FA;">\1</span>', t)
    # Percentages and time values
    t = _re.sub(r'\b(\d+(?:\.\d+)?%)\b', r'<span style="color:#60A5FA;font-weight:600;">\1</span>', t)
    t = _re.sub(r'\b(\d+ms|\d+s|\d+ minutes?)\b', r'<span style="color:#60A5FA;font-weight:600;">\1</span>', t)
    return t


def _mhl(text: str) -> str:
    """Markdown-compatible highlighting — wraps key terms in **bold** for chat."""
    import re as _re
    t = str(text)
    t = _re.sub(r'\b(P[1-4])\b', r'**\1**', t)
    t = _re.sub(r'\b(CRITICAL|ERROR|FATAL|WARNING)\b', r'**\1**', t)
    t = _re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', r'`\1`', t)
    t = _re.sub(r'\b(\d+(?:\.\d+)?%)\b', r'**\1**', t)
    t = _re.sub(r'\b(\d+ms|\d+s)\b', r'**\1**', t)
    return t


def format_agent_output(agent_name: str, result: Dict[str, Any]) -> str:
    """Format one agent's output as a rich HTML panel.

    Args:
        agent_name: Display name of the agent (e.g. ``Triage``, ``RCA``).
        result: Dictionary returned by the agent's ``run()`` method.

    Returns:
        Full HTML string ready for ``gr.HTML``.
    """
    # Build body content based on what keys the result dict contains
    body_parts: List[str] = []

    # Severity
    if "severity" in result:
        body_parts.append(
            f'<div style="margin-bottom:8px;">'
            f'<span style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Severity</span><br>'
            f'{format_severity_badge(result["severity"])}'
            f'</div>'
        )

    # Category / Impact
    for key in ("category", "impact", "impact_summary", "summary"):
        if key in result:
            label = key.replace("_", " ").title()
            body_parts.append(
                f'<div style="margin-bottom:8px;">'
                f'<span style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">{label}</span>'
                f'<div style="margin-top:3px;font-size:0.85rem;color:#e2e8f0;">{_hl(result[key])}</div>'
                f'</div>'
            )

    # Confidence
    if "confidence" in result:
        conf = result["confidence"]
        conf_pct = conf * 100 if isinstance(conf, float) and conf <= 1 else conf
        bar_color = _C["green"] if conf_pct >= 80 else (_C["amber"] if conf_pct >= 50 else _C["red"])
        body_parts.append(
            f'<div style="margin-bottom:8px;">'
            f'<span style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Confidence</span>'
            f'<div style="margin-top:4px;display:flex;align-items:center;gap:10px;">'
            f'  <div style="flex:1;height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;">'
            f'    <div style="width:{conf_pct}%;height:100%;background:{bar_color};border-radius:3px;"></div>'
            f'  </div>'
            f'  <span style="font-size:0.8rem;font-weight:700;color:{bar_color};">{conf_pct:.0f}%</span>'
            f'</div></div>'
        )

    # Evidence
    if "evidence" in result and isinstance(result["evidence"], list):
        items = "".join(
            f'<div class="evidence-item">{_hl(e)}</div>' for e in result["evidence"]
        )
        body_parts.append(
            f'<div style="margin-bottom:8px;">'
            f'<span style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Evidence Chain</span>'
            f'<div style="margin-top:6px;">{items}</div></div>'
        )

    # Root causes
    if "root_causes" in result and isinstance(result["root_causes"], list):
        items = "".join(
            f'<div class="evidence-item">{_hl(rc)}</div>' for rc in result["root_causes"]
        )
        body_parts.append(
            f'<div style="margin-bottom:8px;">'
            f'<span style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Root Causes</span>'
            f'<div style="margin-top:6px;">{items}</div></div>'
        )

    # Actions / Steps
    for key in ("actions", "resolution_steps", "steps"):
        if key in result and isinstance(result[key], list):
            action_html = ""
            for idx, action in enumerate(result[key], 1):
                if isinstance(action, dict):
                    name = action.get("name", action.get("action", f"Step {idx}"))
                    desc = action.get("description", action.get("detail", ""))
                    action_html += (
                        f'<div class="action-card">'
                        f'<div class="action-icon">{idx}</div>'
                        f'<div><div style="font-weight:600;font-size:0.85rem;color:#e2e8f0;">{_hl(name)}</div>'
                        f'<div style="font-size:0.76rem;color:#64748b;margin-top:1px;">{_hl(desc)}</div></div>'
                        f'</div>'
                    )
                else:
                    action_html += (
                        f'<div class="action-card">'
                        f'<div class="action-icon">{idx}</div>'
                        f'<div style="font-size:0.85rem;color:#e2e8f0;">{_hl(action)}</div>'
                        f'</div>'
                    )
            body_parts.append(
                f'<div style="margin-bottom:8px;">'
                f'<span style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">'
                f'{key.replace("_", " ").title()}</span>'
                f'<div style="margin-top:6px;">{action_html}</div></div>'
            )

    # Report (markdown)
    if "report" in result:
        body_parts.append(
            f'<div style="margin-bottom:8px;">'
            f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);'
            f'border-radius:8px;padding:12px;font-size:0.82rem;color:#e2e8f0;'
            f'line-height:1.6;white-space:pre-wrap;">{_hl(result["report"])}</div></div>'
        )

    # Fallback: render remaining keys
    rendered_keys = {
        "severity", "category", "impact", "impact_summary", "summary",
        "confidence", "evidence", "root_causes", "actions",
        "resolution_steps", "steps", "report",
    }
    for k, v in result.items():
        if k not in rendered_keys and not k.startswith("_"):
            if k == "kb_consulted" and v:
                body_parts.append(
                    f'<div style="margin-bottom:8px;padding:6px 10px;background:rgba(96,165,250,0.08);'
                    f'border-left:2px solid #60A5FA;border-radius:0 6px 6px 0;font-size:0.8rem;">'
                    f'<span style="color:#60A5FA;font-weight:600;">📚 Knowledge Base consulted</span>'
                    f'<span style="color:#64748b;margin-left:6px;">— relevant context retrieved from past incidents &amp; runbooks</span></div>'
                )
                continue
            if k == "kb_findings" and v:
                body_parts.append(
                    f'<details style="margin-bottom:8px;font-size:0.78rem;">'
                    f'<summary style="color:#64748b;cursor:pointer;font-weight:500;">📖 Retrieved KB Context</summary>'
                    f'<div style="margin-top:4px;padding:8px;background:rgba(255,255,255,0.02);border-radius:6px;'
                    f'color:#64748b;line-height:1.6;white-space:pre-wrap;">{_hl(v)}</div></details>'
                )
                continue
            if k == "llm_generated":
                label = "LLM Generated" if v else "Template-based"
                body_parts.append(
                    f'<div style="margin-bottom:8px;font-size:0.75rem;">'
                    f'<span style="padding:2px 8px;border-radius:4px;font-weight:600;'
                    f'background:{_C["green"]}22;color:{_C["green"]};">{label}</span></div>'
                )
                continue
            val = v if isinstance(v, str) else json.dumps(v, indent=2, default=str) if v else "None"
            body_parts.append(
                f'<div style="margin-bottom:8px;">'
                f'<span style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">{k.replace("_"," ").title()}</span>'
                f'<div style="margin-top:3px;font-size:0.82rem;color:#e2e8f0;white-space:pre-wrap;">{_hl(val) if isinstance(v, str) else val}</div>'
                f'</div>'
            )

    body = "".join(body_parts) if body_parts else (
        f'<div style="color:#64748b;font-style:italic;">No data returned by {agent_name} agent.</div>'
    )

    return (
        f'<div class="agent-panel">'
        f'  <div class="agent-panel-header">{agent_name}</div>'
        f'  <div class="agent-panel-body">{body}</div>'
        f'</div>'
    )


def format_log_table(logs: List[Dict[str, Any]]) -> str:
    """Format a list of log dicts as a styled HTML table.

    Args:
        logs: List of log dictionaries with keys like timestamp, source,
              level, service, message.

    Returns:
        HTML string of a styled table.
    """
    if not logs:
        return _empty_state("No logs available", "Run an anomaly scan to see live logs.")

    level_colors = {
        "ERROR": _C["red"], "CRITICAL": _C["red"],
        "WARNING": _C["amber"], "INFO": _C["text"],
        "DEBUG": _C["text_muted"],
    }

    rows = ""
    for log in logs[:100]:  # Cap at 100 rows
        ts = log.get("timestamp", "—")
        src = log.get("source", "—")
        lvl = log.get("level", "INFO").upper()
        svc = log.get("service", "—")
        msg = log.get("message", "")
        color = level_colors.get(lvl, _C["text"])
        title_attr = msg.replace('"', "&quot;").replace("'", "&#39;")
        rows += (
            f'<tr>'
            f'<td style="white-space:nowrap;color:#64748b;" title="{ts}">{ts}</td>'
            f'<td><span style="color:{color};font-weight:600;" title="{lvl}">{lvl}</span></td>'
            f'<td title="{svc}">{svc}</td>'
            f'<td style="color:#64748b;" title="{src}">{src}</td>'
            f'<td style="max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" '
            f'title="{title_attr}">{msg}</td>'
            f'</tr>'
        )

    return (
        f'<div style="overflow-x:auto;border-radius:14px;border:1px solid rgba(255,255,255,0.06);">'
        f'<table class="styled-table">'
        f'<thead><tr><th>Timestamp</th><th>Level</th><th>Service</th><th>Source</th><th>Message</th></tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table></div>'
    )


def format_metrics_panel(metrics: Dict[str, Any]) -> str:
    """Format performance metrics as rich HTML cards.

    Args:
        metrics: Dict with keys like total_time, tokens_used, latency, etc.

    Returns:
        HTML string of metric cards.
    """
    cards_data = [
        ("Total Time", f'{metrics.get("total_time_seconds", 0):.1f}s', _C["text"]),
        ("Tokens Used", f'{metrics.get("total_tokens", 0):,}', _C["text"]),
        ("LLM Calls", str(metrics.get("llm_calls", 0)), _C["text"]),
        ("Avg Latency", f'{metrics.get("avg_latency_ms", 0):.0f}ms', _C["text"]),
        ("GPU Memory", f'{metrics.get("gpu_memory_mb", 0):.0f} MB', _C["text"]),
        ("Model", metrics.get("model", MODEL_NAME.split("/")[-1]), _C["text"]),
    ]

    cards_html = ""
    for label, value, color in cards_data:
        cards_html += (
            f'<div style="background:rgba(17,17,40,0.65);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:10px;padding:12px 14px;'
            f'text-align:center;flex:1;min-width:100px;position:relative;overflow:hidden;">'
            f'<div style="position:absolute;top:0;left:0;right:0;height:2px;background:{_C["text"]};border-radius:10px 10px 0 0;"></div>'
            f'<div style="font-size:1.2rem;font-weight:800;color:{_C["text"]};">{value}</div>'
            f'<div style="font-size:0.65rem;color:{_C["text_muted"]};text-transform:uppercase;letter-spacing:0.8px;margin-top:2px;">{label}</div>'
            f'</div>'
        )

    return f'<div style="display:flex;gap:14px;flex-wrap:wrap;">{cards_html}</div>'


def _metric_card_html(label: str, value: str, accent: str) -> str:
    """Build a single metric card block."""
    return (
        f'<div class="metric-card">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:2px;background:{accent};'
        f'border-radius:14px 14px 0 0;"></div>'
        f'<div class="metric-value" style="color:{accent};">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )


def _empty_state(title: str, subtitle: str = "") -> str:
    """Placeholder panel when no data is loaded."""
    return (
        f'<div style="text-align:center;padding:48px 24px;">'
        f'<div style="font-size:1rem;color:#64748b;font-weight:600;">{title}</div>'
        f'<div style="font-size:0.82rem;color:#64748b;margin-top:6px;">{subtitle}</div>'
        f'</div>'
    )


def _loading_html(message: str = "Analyzing...") -> str:
    """Return a loading skeleton HTML."""
    return (
        f'<div style="padding:24px;">'
        f'<div class="loading-skeleton" style="margin-bottom:12px;height:28px;width:60%;"></div>'
        f'<div class="loading-skeleton" style="margin-bottom:12px;height:18px;width:90%;"></div>'
        f'<div class="loading-skeleton" style="margin-bottom:12px;height:18px;width:75%;"></div>'
        f'<div style="text-align:center;margin-top:16px;color:#64748b;font-size:0.85rem;">'
        f'⏳ {message}</div></div>'
    )


def _format_anomalies_html(anomalies: List[Dict[str, Any]]) -> str:
    """Format anomaly results as rich cards."""
    if not anomalies:
        return _empty_state("No anomalies detected", "All systems are operating normally.")

    cards = ""
    for a in anomalies:
        sev = a.get("severity", "P3").upper()
        badge = format_severity_badge(sev)
        desc = a.get("description", "Unknown anomaly")
        atype = a.get("type", "unknown")
        source = a.get("source", "—")
        conf = a.get("confidence", 0)
        if isinstance(conf, (int, float)):
            conf_pct = conf * 100 if conf <= 1 else conf
            conf_str = f"{conf_pct:.0f}%"
        else:
            conf_str = str(conf)
        ts = a.get("timestamp", "")

        evidence_html = ""
        for e in a.get("evidence", []):
            evidence_html += f'<div class="evidence-item">{e}</div>'

        cards += (
            f'<div class="glass-card" style="margin-bottom:14px;">'
            f'  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            f'    <div style="display:flex;align-items:center;gap:10px;">{badge}'
            f'      <span style="color:#64748b;font-size:0.78rem;">{atype}</span></div>'
            f'    <span style="color:#64748b;font-size:0.75rem;">{ts}</span>'
            f'  </div>'
            f'  <div style="font-size:0.92rem;color:#e2e8f0;margin-bottom:8px;">{desc}</div>'
            f'  <div style="font-size:0.78rem;color:#64748b;">Source: {source} · Confidence: {conf_str}</div>'
            f'  {evidence_html}'
            f'</div>'
        )
    return cards


def _format_runbook_html(runbook: Dict[str, Any]) -> str:
    """Format a single runbook as a rich card."""
    title = runbook.get("title", "Untitled")
    cat = runbook.get("category", "—")
    symptoms = runbook.get("symptoms", [])
    root_causes = runbook.get("root_causes", [])
    steps = runbook.get("resolution_steps", [])
    prevention = runbook.get("prevention", [])
    tags = runbook.get("tags", [])

    def _list_html(items: List[str], color: str) -> str:
        return "".join(
            f'<div style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);'
            f'font-size:0.85rem;color:#e2e8f0;display:flex;gap:8px;">'
            f'<span style="color:{color};font-weight:600;">›</span> {item}</div>'
            for item in items
        )

    tags_html = "".join(
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'background:rgba(255,255,255,0.06);color:{_C["text"]};font-size:0.7rem;'
        f'font-weight:600;margin-right:6px;margin-bottom:4px;">{t}</span>'
        for t in tags
    )

    return (
        f'<div class="glass-card">'
        f'  <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">'
        f'    <span style="font-size:1.4rem;">📘</span>'
        f'    <div>'
        f'      <div style="font-size:1.05rem;font-weight:700;color:{_C["text"]};">{title}</div>'
        f'      <div style="font-size:0.75rem;color:{_C["text_muted"]};">{cat}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div style="margin-bottom:12px;">{tags_html}</div>'
        f'  <div style="margin-bottom:14px;">'
        f'    <div style="color:{_C["text"]};font-size:0.73rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Symptoms</div>'
        f'    {_list_html(symptoms, _C["text"])}'
        f'  </div>'
        f'  <div style="margin-bottom:14px;">'
        f'    <div style="color:{_C["text"]};font-size:0.73rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Root Causes</div>'
        f'    {_list_html(root_causes, _C["text"])}'
        f'  </div>'
        f'  <div style="margin-bottom:14px;">'
        f'    <div style="color:{_C["text"]};font-size:0.73rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Resolution Steps</div>'
        f'    {_list_html(steps, _C["text"])}'
        f'  </div>'
        f'  <div>'
        f'    <div style="color:{_C["text"]};font-size:0.73rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Prevention</div>'
        f'    {_list_html(prevention, _C["text"])}'
        f'  </div>'
        f'</div>'
    )


def _build_agent_latency_chart(agent_timings: Dict[str, float]) -> str:
    """Horizontal bar chart showing agent latency breakdown."""
    if not agent_timings:
        return _empty_state("No timing data", "Run an analysis to see latency breakdown.")

    max_val = max(agent_timings.values()) if agent_timings else 1
    colors = ["rgba(255,255,255,0.7)", "rgba(255,255,255,0.55)", "rgba(255,255,255,0.4)", "rgba(255,255,255,0.3)", "rgba(255,255,255,0.2)"]

    bars = ""
    for idx, (agent, ms) in enumerate(agent_timings.items()):
        color = colors[idx % len(colors)]
        width_pct = max(5, (ms / max_val) * 100)
        bars += (
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'  <div style="width:120px;font-size:0.8rem;color:#64748b;font-weight:500;text-align:right;">{agent}</div>'
            f'  <div style="flex:1;height:6px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;">'
            f'    <div style="width:{width_pct}%;height:100%;background:{color};'
            f'border-radius:3px;"></div>'
            f'  </div>'
            f'  <span style="font-size:0.72rem;font-weight:700;color:{color};min-width:40px;">{ms:.0f}ms</span>'
            f'</div>'
        )
    return (
        f'<div class="glass-card">'
        f'<div style="font-size:0.8rem;font-weight:700;color:#64748b;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:14px;">Agent Latency Breakdown</div>'
        f'{bars}</div>'
    )


def _build_token_chart(token_data: Dict[str, int]) -> str:
    """Circular ring chart showing token usage per agent."""
    if not token_data:
        return _empty_state("No token data", "Run an analysis to see token usage.")

    total = sum(token_data.values()) or 1
    colors = ["rgba(255,255,255,0.7)", "rgba(255,255,255,0.55)", "rgba(255,255,255,0.4)", "rgba(255,255,255,0.25)"]
    radius, stroke = 28, 5
    circumference = 2 * math.pi * radius

    rings_html = ""
    for idx, (agent, tokens) in enumerate(token_data.items()):
        pct = tokens / total
        offset = circumference * (1 - pct)
        color = colors[idx % len(colors)]
        rings_html += (
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<svg width="64" height="64" viewBox="0 0 64 64">'
            f'<circle cx="32" cy="32" r="{radius}" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="{stroke}"/>'
            f'<circle cx="32" cy="32" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" '
            f'transform="rotate(-90 32 32)" stroke-linecap="round"/>'
            f'<text x="32" y="30" text-anchor="middle" dominant-baseline="middle" '
            f'fill="{color}" font-size="11" font-weight="700">{tokens}</text>'
            f'<text x="32" y="42" text-anchor="middle" dominant-baseline="middle" '
            f'fill="{color}88" font-size="7">{pct*100:.0f}%</text>'
            f'</svg>'
            f'<div><div style="font-weight:600;font-size:0.82rem;color:#e2e8f0;">{agent}</div>'
            f'<div style="font-size:0.7rem;color:#64748b;">{tokens:,} tokens</div></div>'
            f'</div>'
        )

    return (
        f'<div class="glass-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
        f'  <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Token Usage per Agent</div>'
        f'  <div style="font-size:0.8rem;color:{_C["text"]};font-weight:600;">Total: {total:,}</div>'
        f'</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{rings_html}</div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════
#  DEMO DATA (works without live components)
# ═══════════════════════════════════════════════════════════════════

def _demo_scenarios() -> Dict[str, Dict[str, Any]]:
    """Built-in demo scenarios for when data_generator is not available."""
    return {
        "Database Connection Pool Exhaustion": {
            "id": "INC-001",
            "title": "Database Connection Pool Exhaustion",
            "severity": "P1",
            "description": "PostgreSQL connection pool fully exhausted on prod-db-01. "
                           "Applications experiencing timeouts and 500 errors across payment and user services.",
            "logs": [
                {"timestamp": "2026-06-11T22:30:01Z", "source": "prod-db-01", "level": "ERROR",
                 "service": "postgresql", "message": "FATAL: too many connections for role 'app_user'"},
                {"timestamp": "2026-06-11T22:30:05Z", "source": "payment-svc", "level": "ERROR",
                 "service": "payment-service", "message": "Connection pool exhausted, cannot acquire connection within 5000ms"},
                {"timestamp": "2026-06-11T22:30:08Z", "source": "api-gw-01", "level": "ERROR",
                 "service": "api-gateway", "message": "Upstream service returned 503: payment-service unavailable"},
                {"timestamp": "2026-06-11T22:30:12Z", "source": "user-svc", "level": "WARNING",
                 "service": "user-service", "message": "Database connection timeout after 3 retries"},
                {"timestamp": "2026-06-11T22:30:18Z", "source": "monitoring", "level": "CRITICAL",
                 "service": "alertmanager", "message": "P1 Alert: prod-db-01 active connections 500/500 (100%)"},
            ],
            "metrics": [
                {"timestamp": "2026-06-11T22:30:00Z", "host": "prod-db-01", "cpu_percent": 92.3,
                 "memory_percent": 87.1, "disk_percent": 45.2, "network_in_mbps": 125.8,
                 "network_out_mbps": 340.2, "request_latency_ms": 15200, "error_rate": 0.43,
                 "active_connections": 500},
            ],
        },
        "Memory Leak in Auth Service": {
            "id": "INC-002",
            "title": "Memory Leak in Authentication Service",
            "severity": "P2",
            "description": "Auth service memory usage climbing steadily. JWT token cache not being evicted properly. "
                           "OOM kills expected within 2 hours at current rate.",
            "logs": [
                {"timestamp": "2026-06-11T21:00:00Z", "source": "auth-svc-01", "level": "WARNING",
                 "service": "auth-service", "message": "Heap usage at 78% (6.2GB/8GB), GC frequency increasing"},
                {"timestamp": "2026-06-11T21:30:00Z", "source": "auth-svc-01", "level": "WARNING",
                 "service": "auth-service", "message": "JWT token cache size: 2.4M entries, eviction policy not triggering"},
                {"timestamp": "2026-06-11T22:00:00Z", "source": "auth-svc-01", "level": "ERROR",
                 "service": "auth-service", "message": "GC pause exceeded 5 seconds, request queuing detected"},
                {"timestamp": "2026-06-11T22:15:00Z", "source": "k8s-master", "level": "WARNING",
                 "service": "kubelet", "message": "Pod auth-svc-01 memory 89% of limit (7.1GB/8GB)"},
            ],
            "metrics": [
                {"timestamp": "2026-06-11T22:15:00Z", "host": "auth-svc-01", "cpu_percent": 65.4,
                 "memory_percent": 89.2, "disk_percent": 32.1, "network_in_mbps": 45.3,
                 "network_out_mbps": 22.1, "request_latency_ms": 3200, "error_rate": 0.12,
                 "active_connections": 180},
            ],
        },
        "Disk Space Critical on Log Server": {
            "id": "INC-003",
            "title": "Disk Space Critical on Centralized Log Server",
            "severity": "P3",
            "description": "Log aggregation server running out of disk space. Log rotation misconfigured after "
                           "last deployment. Ingestion rate 3x normal due to debug logging left enabled.",
            "logs": [
                {"timestamp": "2026-06-11T20:00:00Z", "source": "log-server-01", "level": "WARNING",
                 "service": "elasticsearch", "message": "Disk watermark [high] exceeded on node, shards will be relocated"},
                {"timestamp": "2026-06-11T21:00:00Z", "source": "log-server-01", "level": "ERROR",
                 "service": "elasticsearch", "message": "Disk watermark [flood] exceeded, index [logs-2026.06.11] set to read-only"},
                {"timestamp": "2026-06-11T22:00:00Z", "source": "log-server-01", "level": "CRITICAL",
                 "service": "system", "message": "Filesystem /data usage at 96%, only 42GB remaining"},
                {"timestamp": "2026-06-11T22:10:00Z", "source": "api-svc-01", "level": "WARNING",
                 "service": "fluentd", "message": "Log forwarding buffer full, dropping messages"},
            ],
            "metrics": [
                {"timestamp": "2026-06-11T22:10:00Z", "host": "log-server-01", "cpu_percent": 45.0,
                 "memory_percent": 62.3, "disk_percent": 96.1, "network_in_mbps": 280.5,
                 "network_out_mbps": 12.3, "request_latency_ms": 890, "error_rate": 0.08,
                 "active_connections": 95},
            ],
        },
        "Suspicious API Access Pattern": {
            "id": "INC-004",
            "title": "Suspicious API Access Pattern Detected",
            "severity": "P4",
            "description": "Anomalous API call patterns from multiple IPs. Possible credential stuffing attack "
                           "targeting /api/v2/auth/login endpoint. Rate limiting triggered.",
            "logs": [
                {"timestamp": "2026-06-11T22:40:00Z", "source": "waf-01", "level": "WARNING",
                 "service": "cloudflare-waf", "message": "Rate limit triggered: 450 requests/min from 198.51.100.23 to /api/v2/auth/login"},
                {"timestamp": "2026-06-11T22:40:05Z", "source": "auth-svc", "level": "WARNING",
                 "service": "auth-service", "message": "Failed login attempts spike: 340 failures in last 5 minutes (baseline: 12)"},
                {"timestamp": "2026-06-11T22:40:10Z", "source": "ids-01", "level": "ERROR",
                 "service": "suricata", "message": "ET SCAN Possible credential stuffing - distributed source IPs (23 unique)"},
                {"timestamp": "2026-06-11T22:40:20Z", "source": "auth-svc", "level": "ERROR",
                 "service": "auth-service", "message": "Account lockout threshold reached for 18 accounts in 5 minutes"},
            ],
            "metrics": [
                {"timestamp": "2026-06-11T22:40:00Z", "host": "api-gw-01", "cpu_percent": 55.2,
                 "memory_percent": 48.7, "disk_percent": 35.0, "network_in_mbps": 520.3,
                 "network_out_mbps": 180.1, "request_latency_ms": 450, "error_rate": 0.38,
                 "active_connections": 1250},
            ],
        },
        "Application-Level Cascade Failure": {
            "id": "INC-005",
            "title": "Application-Level Cascade Failure",
            "severity": "P1",
            "description": "Client SDK serialization bug causes malformed API requests. Downstream circuit-breakers open on microservice timeouts. DB schema migration adds NOT NULL constraint without updating app INSERT code. All three failure modes cascade — illustrating common application-layer issues.",
            "logs": [
                {"timestamp": "2026-06-11T21:00:00Z", "source": "api-gateway", "level": "ERROR",
                 "service": "api-gateway", "message": "POST /api/v2/orders — 400 Bad Request: JSON parse failure — double-escaped quotes"},
                {"timestamp": "2026-06-11T21:05:00Z", "source": "api-gateway", "level": "WARNING",
                 "service": "api-gateway", "message": "Content-Type mismatch: client sent 'text/html' for endpoint expecting 'application/json'"},
                {"timestamp": "2026-06-11T21:10:00Z", "source": "order-service", "level": "ERROR",
                 "service": "order-service", "message": "circuit breaker OPEN for downstream 'inventory-svc' after gRPC deadline exceeded"},
                {"timestamp": "2026-06-11T21:15:00Z", "source": "postgresql", "level": "ERROR",
                 "service": "postgresql", "message": "null value in column 'tax_region' violates NOT NULL constraint — INSERT from order-service failed"},
                {"timestamp": "2026-06-11T21:20:00Z", "source": "application", "level": "CRITICAL",
                 "service": "application", "message": "order submission error rate at 100% — malformed requests + DB constraints + timeout — P1 declared"},
            ],
            "metrics": [
                {"timestamp": "2026-06-11T21:20:00Z", "host": "api-gw-01", "cpu_percent": 70.2,
                 "memory_percent": 65.5, "disk_percent": 40.1, "network_in_mbps": 420.0,
                 "network_out_mbps": 155.3, "request_latency_ms": 2100, "error_rate": 0.58,
                 "active_connections": 720},
            ],
        },
    }


def _demo_runbooks() -> List[Dict[str, Any]]:
    """Built-in demo runbooks."""
    return [
        {
            "id": "RB-001", "title": "Database Connection Pool Exhaustion",
            "category": "database",
            "symptoms": ["Connection timeout errors", "503 responses from dependent services",
                         "High active connection count at pool limit"],
            "root_causes": ["Connection leak in application code", "Missing connection timeout settings",
                            "Sudden traffic spike exceeding pool capacity"],
            "resolution_steps": ["Identify services holding idle connections",
                                 "Restart affected application pods to release connections",
                                 "Increase connection pool size temporarily",
                                 "Deploy connection leak fix"],
            "prevention": ["Implement connection health checks", "Set max connection lifetime",
                           "Add connection pool monitoring alerts"],
            "tags": ["database", "postgresql", "connection-pool", "P1"],
        },
        {
            "id": "RB-002", "title": "Memory Leak Diagnosis and Mitigation",
            "category": "application",
            "symptoms": ["Steadily increasing memory usage", "GC pause times increasing",
                         "OOM kill events", "Performance degradation over time"],
            "root_causes": ["Cache without eviction policy", "Event listener not unsubscribed",
                            "Large object retained in closures"],
            "resolution_steps": ["Capture heap dump for analysis",
                                 "Rolling restart affected pods",
                                 "Implement LRU eviction on caches",
                                 "Deploy memory-profiled version"],
            "prevention": ["Regular memory profiling in staging", "Set memory resource limits",
                           "Automated canary deployments with memory baseline"],
            "tags": ["memory", "leak", "application", "jvm", "P2"],
        },
        {
            "id": "RB-003", "title": "Disk Space Exhaustion Recovery",
            "category": "infrastructure",
            "symptoms": ["Disk usage above 90%", "Read-only filesystem errors",
                         "Log ingestion failures", "Service write failures"],
            "root_causes": ["Log rotation misconfiguration", "Debug logging left enabled",
                            "Old data not cleaned up", "Index growth unbounded"],
            "resolution_steps": ["Identify largest directories with du -sh",
                                 "Clear old log files and temp data",
                                 "Fix log rotation configuration",
                                 "Disable debug logging in production"],
            "prevention": ["Automated disk usage alerts at 80%", "Log retention policies",
                           "Regular cleanup cron jobs"],
            "tags": ["disk", "storage", "logs", "infrastructure", "P2"],
        },
        {
            "id": "RB-004", "title": "Credential Stuffing Attack Response",
            "category": "security",
            "symptoms": ["Spike in failed login attempts", "Multiple accounts locked",
                         "Traffic from distributed IPs", "Rate limiting triggered"],
            "root_causes": ["Compromised credential database sold on dark web",
                            "Insufficient rate limiting", "No CAPTCHA on login endpoint"],
            "resolution_steps": ["Enable enhanced rate limiting on login endpoint",
                                 "Block identified attacker IP ranges",
                                 "Force password reset for affected accounts",
                                 "Enable CAPTCHA challenge"],
            "prevention": ["Implement MFA for all accounts", "Deploy bot detection",
                           "Monitor for credential dumps"],
            "tags": ["security", "attack", "credential-stuffing", "authentication", "P1"],
        },
    ]

# ── Past-incidents helpers (for enriching chat context) ──────────

_PAST_INCIDENTS_CACHE: Optional[List[Dict[str, Any]]] = None

def _load_past_incidents() -> List[Dict[str, Any]]:
    global _PAST_INCIDENTS_CACHE
    if _PAST_INCIDENTS_CACHE is not None:
        return _PAST_INCIDENTS_CACHE
    paths = [
        os.path.join(_CFG_DATA_DIR, "past_incidents.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "past_incidents.json"),
        os.path.join(os.getcwd(), "sample_data", "past_incidents.json"),
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    _PAST_INCIDENTS_CACHE = data
                    logger.info("Loaded %d past incidents from %s", len(data), path)
                    return data
            except Exception as exc:
                logger.warning("Failed to load past incidents from %s: %s", path, exc)
    _PAST_INCIDENTS_CACHE = []
    return []

def _past_incidents_summary() -> str:
    """Build a markdown summary of all past incidents for LLM context."""
    incidents = _load_past_incidents()
    if not incidents:
        return ""
    sev_counts = {"P1": 0, "P2": 0, "P3": 0, "P4": 0}
    cat_counts: Dict[str, int] = {}
    resolved = 0
    for inc in incidents:
        s = inc.get("severity", "P3")
        sev_counts[s] = sev_counts.get(s, 0) + 1
        c = inc.get("category", "unknown")
        cat_counts[c] = cat_counts.get(c, 0) + 1
        if inc.get("resolution"):
            resolved += 1
    sev_row = " | ".join(f"{s}: {sev_counts.get(s,0)}" for s in ["P1","P2","P3","P4"])
    cat_row = " | ".join(f"{k}: {v}" for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]))
    top = sorted(incidents, key=lambda x: x.get("duration_minutes", 9999), reverse=True)[:5]
    top_rows = "\n".join(
        f'| {inc.get("title","?")} | {inc.get("severity","?")} | '
        f'{inc.get("category","?")} | {inc.get("duration_minutes","?")}m | '
        f'{inc.get("resolution","?")[:80]} |'
        for inc in top
    )
    return (
        f"\n**Past Incidents Database** ({len(incidents)} total)\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Severity distribution | {sev_row} |\n"
        f"| Category distribution | {cat_row} |\n"
        f"| Resolved | {resolved}/{len(incidents)} |\n\n"
        f"**Top 5 Longest-Duration Incidents:**\n\n"
        f"| Title | Sev | Category | Duration | Resolution |\n"
        f"|-------|-----|----------|----------|------------|\n{top_rows}"
    )


# ═══════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD FACTORY
# ═══════════════════════════════════════════════════════════════════

def create_dashboard(
    orchestrator: Optional[Any] = None,
    anomaly_detector: Optional[Any] = None,
    data_gen_func: Optional[Callable] = None,
) -> gr.Blocks:
    """Create and return the Gradio Blocks app.

    All three arguments are optional so the dashboard can be previewed
    without a live vLLM backend or fully-wired agents.

    Args:
        orchestrator: An ``InfraHealOrchestrator`` instance (or None for demo).
        anomaly_detector: An anomaly detection module/object (or None).
        data_gen_func: A callable that returns incident scenarios (or None).

    Returns:
        A ``gr.Blocks`` instance ready to ``.launch()``.
    """
    # ------- data sources -------------------------------------------------
    if data_gen_func is not None:
        try:
            raw = data_gen_func()
            if isinstance(raw, list):
                scenarios = {
                    s.get("name", s.get("id", f"Scenario {i}")): s
                    for i, s in enumerate(raw)
                }
            else:
                scenarios = raw
        except Exception:
            logger.warning("data_gen_func failed, falling back to built-in demos")
            scenarios = _demo_scenarios()
    else:
        scenarios = _demo_scenarios()

    scenario_names = list(scenarios.keys())
    runbooks = _demo_runbooks()

    # state holders
    _perf_state: Dict[str, Any] = {}
    _last_pipeline_state: Dict[str, Any] = {}
    _result_cache: Dict[str, Any] = {}
    _last_pipeline: Optional[str] = None  # "process" or "monitor"; used by _poll_live_html fallback

    # ── Pipeline flow state ────────────────────────────────────────
    _pipeline_run: Dict[str, Any] = {"name": "", "status": "idle", "steps": [], "start_time": 0, "elapsed": 0}

    def _pipeline_step(name: str, desc: str = "", status: str = "pending", progress: int = 0):
        return {"name": name, "desc": desc, "status": status, "progress": progress, "duration": 0, "start": time.time()}

    def _render_pipeline_flow() -> str:
        pr = _pipeline_run
        steps = pr.get("steps", [])
        if not steps:
            return ""
        now = time.time()
        elapsed = now - pr.get("start_time", now)
        pr["elapsed"] = elapsed
        status = pr.get("status", "idle")
        status_color = {"running":"58a6ff","completed":"00FF88","failed":"FF3B3B","warning":"FFB800","idle":"8b949e"}.get(status, "8b949e")
        status_label = {"running":"Running...","completed":"Completed","failed":"Failed","warning":"Warning","idle":"Idle"}.get(status, status)
        done = sum(1 for s in steps if s["status"] in ("completed","failed","warning"))
        total = len(steps)
        pct = int(done / total * 100) if total else 0

        def _render_one(s, extra_class=""):
            cls = "pipeline-step" + (" " + extra_class if extra_class else "")
            st = s["status"]
            icon_html = f'<div class="pipeline-step-icon {st}"></div>'
            dur = s.get("duration", 0)
            dur_str = f"{dur:.1f}s" if dur > 0 else ("..." if st == "running" else "--")
            bar_class = st if st in ("completed","failed","warning") else "running"
            prog = s.get("progress", 100 if st in ("completed","failed","warning") else 0)
            if st == "running" and s.get("start", 0) > 0:
                dur_html = f'<span class="step-timer" data-start="{s["start"]}" data-status="running">00:00</span>'
            elif st == "completed" and s.get("start", 0) > 0:
                dur_html = f'{dur_str}'
            else:
                dur_html = dur_str
            return (
                f'<div class="{cls}">'
                f'{icon_html}'
                f'<div class="pipeline-step-info">'
                f'<div class="pipeline-step-name">{s["name"]}</div>'
                f'<div class="pipeline-step-desc">{s.get("desc","")}</div>'
                f'</div>'
                f'<div class="pipeline-step-progress"><div class="pipeline-step-progress-bar {bar_class}" style="width:{prog}%"></div></div>'
                f'<div class="pipeline-step-duration">{dur_html}</div>'
                f'</div>'
            )

        step_html = ""
        i = 0
        while i < len(steps):
            s = steps[i]
            name = s.get("name", "")
            is_parent = (not name.startswith("  ") and i + 1 < len(steps) and steps[i + 1].get("name", "").startswith("  "))
            if is_parent:
                children_html = ""
                i += 1
                while i < len(steps) and steps[i].get("name", "").startswith("  "):
                    children_html += _render_one(steps[i])
                    i += 1
                step_html += f'<div class="pipeline-cycle-group">{_render_one(s, "cycle-header")}<div class="pipeline-cycle-children">{children_html}</div></div>'
            else:
                step_html += _render_one(s)
                i += 1

        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        timer_color = "#58a6ff" if status == "running" else ("#00FF88" if status == "completed" else status_color)
        start_ts = pr.get("start_time", now)
        js = ""
        return (
            f'<div class="pipeline-flow">'
            f'<div class="pipeline-header">'
            f'<div class="pipeline-title">{pr.get("name","Pipeline")}</div>'
            f'<div class="pipeline-progress" style="flex:1;max-width:160px;height:6px;background:#21262d;border-radius:3px;overflow:hidden;">'
            f'<div style="width:{pct}%;height:100%;background:#{status_color};border-radius:3px;transition:width 0.5s;"></div></div>'
            f'<div class="pipeline-status {status}" style="color:#{status_color};">{status_label} — {pct}%</div>'
            f'<span class="pipeline-timer" data-start="{start_ts}" data-status="{status}" style="color:#{timer_color};font-weight:600;font-size:0.95rem;font-variant-numeric:tabular-nums;">{elapsed_str}</span>'
            f'</div>'
            f'<div class="pipeline-steps">{step_html}</div>'
            f'{js}</div>'
        )

    def _with_pipeline(name: str, steps: list) -> str:
        """Execute a list of step dicts with timing and return pipeline-flow HTML.
        Each step dict: {"name": ..., "desc": ..., "fn": callable}
        """
        nonlocal _pipeline_run
        _pipeline_run = {"name": name, "status": "running", "steps": [], "start_time": time.time(), "elapsed": 0}
        for s in steps:
            step_data = {"name": s["name"], "desc": s.get("desc", ""), "status": "running", "progress": 0, "duration": 0, "start": time.time()}
            _pipeline_run["steps"].append(step_data)
            try:
                s["fn"]()
                step_data["status"] = "completed"
            except Exception as exc:
                logger.warning("Pipeline step '%s' failed: %s", s["name"], exc)
                step_data["status"] = "failed"
                step_data["desc"] = str(exc)[:120]
            step_data["duration"] = time.time() - step_data["start"]
            step_data["progress"] = 100
        _pipeline_run["status"] = "completed" if all(st["status"] == "completed" for st in _pipeline_run["steps"]) else "warning"
        return _render_pipeline_flow()

    # ------- event handlers -----------------------------------------------

    def _on_scenario_selected(name: str) -> Tuple[str, str]:
        """When user picks a scenario, show its description and logs."""
        if not name or name not in scenarios:
            return _empty_state("Select a scenario"), _empty_state("No logs")
        sc = scenarios[name]
        desc_html = (
            f'<div class="glass-card">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
            f'<span style="font-size:1.05rem;font-weight:700;color:#e2e8f0;">{sc.get("title", name)}</span>'
            f'<span style="color:#64748b;font-size:0.78rem;">({sc.get("id", "")})</span>'
            f'</div>'
            f'<div style="font-size:0.88rem;color:#e2e8f0;line-height:1.7;">{sc.get("description", "")}</div>'
            f'</div>'
        )
        logs_html = format_log_table(sc.get("logs", []))
        return desc_html, logs_html

    def _run_analysis(scenario_name: str) -> Tuple[str, str, str, str, str]:
        """Run the full orchestrator pipeline on a scenario."""
        if not scenario_name or scenario_name not in scenarios:
            empty = _empty_state("Select a scenario first")
            return empty, empty, empty, empty, empty, \
                   _chat_update_status()[0], _chat_update_status()[1], _chat_refresh_risk()

        sc = dict(scenarios[scenario_name])  # shallow copy so we can inject anomalies

        # If orchestrator is available, use it
        if orchestrator is not None:
            try:
                start = time.time()

                # Run anomaly detection first if available
                if anomaly_detector is not None:
                    detected = anomaly_detector.detect_all(
                        logs=sc.get("logs", []),
                        metrics=sc.get("metrics", []),
                    )
                    sc["anomalies"] = detected
                    logger.info("Pre-run anomaly detection: %d anomalies found", len(detected))

                # Inject few-shot examples and preference ranking from experience store
                sev = sc.get("severity", "P3")
                cat = sc.get("title", "").split()[0].lower() if sc.get("title") else "infrastructure"
                sc["few_shot_examples"] = _get_few_shot_examples(sev, cat)
                sc["action_preferences"] = _get_preference_ranking()
                if sc.get("few_shot_examples"):
                    logger.info("Injected %d chars of few-shot examples for %s", len(sc["few_shot_examples"]), scenario_name)

                result = orchestrator.process_scenario(sc)
                elapsed = time.time() - start

                # Use aliases set by process_scenario (triage, rca, remediation) + report
                triage_out = result.get("triage", result.get("triage_result", {}))
                rca_out = result.get("rca", result.get("rca_result", {}))
                remed_out = result.get("remediation", result.get("remediation_result", {}))
                report_out = result.get("report", {})

                triage_html = format_agent_output("Triage", triage_out)
                rca_html = format_agent_output("Root Cause Analysis", rca_out)
                remed_html = format_agent_output("Remediation", remed_out)

                # Append safety audit info to remediation panel
                safety_results = result.get("safety_results", [])
                if safety_results:
                    safety_lines = []
                    for s in safety_results:
                        v = s.get("verdict", "unknown")
                        r = s.get("reason", "")
                        if v == "block":
                            safety_lines.append(f'<div style="padding:6px 10px;margin:4px 0;border-left:3px solid {_C["red"]};font-size:0.82rem;background:rgba(255,59,59,0.06);border-radius:0 6px 6px 0;"><b style="color:{_C["red"]};">🛑 Blocked</b> {r}</div>')
                        elif v == "flag":
                            safety_lines.append(f'<div style="padding:6px 10px;margin:4px 0;border-left:3px solid {_C["amber"]};font-size:0.82rem;background:rgba(255,184,0,0.06);border-radius:0 6px 6px 0;"><b style="color:{_C["amber"]};">⚠️ Flagged</b> {r}</div>')
                    if safety_lines:
                        audit_summary = result.get("safety_audit_summary", {})
                        total = audit_summary.get("total_checks", 0)
                        blocked = audit_summary.get("blocked", 0)
                        flagged = audit_summary.get("flagged", 0)
                        remed_html = remed_html.replace(
                            '</div></div>',
                            f'<div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.06);">'
                            f'<span style="color:#64748b;font-size:0.73rem;text-transform:uppercase;letter-spacing:1px;">🛡️ SafetyGuard Audit</span>'
                            f'<div style="margin-top:6px;display:flex;gap:12px;font-size:0.78rem;">'
                            f'<span>✅ {total - blocked - flagged}/{total} passed</span>'
                            f'<span style="color:{_C["red"]};">🛑 {blocked} blocked</span>'
                            f'<span style="color:{_C["amber"]};">⚠️ {flagged} flagged</span>'
                            f'</div>{"".join(safety_lines)}</div></div>', 1
                        )
                report_html = format_agent_output("Incident Report", report_out)

                # Build reasoning chain
                reasoning_parts = []
                for step in result.get("reasoning_chain", []):
                    agent_name = step.get("agent", "Unknown")
                    thought = step.get("reasoning", step.get("thought", ""))
                    reasoning_parts.append(
                        f'<div class="evidence-item" style="border-left-color:{_C["text_muted"]};">'
                        f'<span style="color:{_C["text"]};font-weight:600;">{agent_name}:</span>'
                        f' <span style="color:#e2e8f0;">{thought}</span></div>'
                    )
                reasoning_html = (
                    f'<div class="glass-card"><div style="font-size:0.8rem;font-weight:700;color:#64748b;'
                    f'text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Agent Reasoning Chain</div>'
                    f'{"".join(reasoning_parts)}</div>'
                ) if reasoning_parts else _empty_state("No reasoning chain captured")

                # Store perf data from pipeline_metrics
                perf = result.get("pipeline_metrics", {})
                agent_m = perf.get("agent_metrics", {})
                totals = agent_m.get("totals", {})
                agents_data = agent_m.get("agents", {})

                def _avg_latency(agent_key):
                    a = agents_data.get(agent_key, {})
                    return a.get("avg_latency", 0) * 1000 if a.get("successful_calls", 0) > 0 else 0

                def _agent_tokens(agent_key):
                    return agents_data.get(agent_key, {}).get("total_tokens", 0)

                _perf_state.update({
                    "total_time_seconds": perf.get("total_time_seconds", elapsed),
                    "total_tokens": totals.get("total_tokens", 0),
                    "llm_calls": totals.get("total_calls", 4),
                    "avg_latency_ms": (
                        (perf.get("triage_latency", 0)
                         + perf.get("rca_latency", 0)
                         + perf.get("remediation_latency", 0)
                         + perf.get("report_latency", 0)) * 250
                    ),
                    "gpu_memory_mb": 0,
                    "model": MODEL_NAME.split("/")[-1],
                    "agent_timings": {
                        "Triage Agent": _avg_latency("triage"),
                        "RCA Agent": _avg_latency("rca"),
                        "Remediation Agent": _avg_latency("remediation"),
                        "Reporting Agent": _avg_latency("reporting"),
                    },
                    "agent_tokens": {
                        "Triage Agent": _agent_tokens("triage"),
                        "RCA Agent": _agent_tokens("rca"),
                        "Remediation Agent": _agent_tokens("remediation"),
                        "Reporting Agent": _agent_tokens("reporting"),
                    },
                })

                _last_pipeline_state.update({
                    "scenario": scenario_name,
                    "triage": triage_out,
                    "rca": rca_out,
                    "remediation": remed_out,
                    "report": report_out,
                    "reasoning_chain": reasoning_html,
                    "critique": result.get("critique", {}),
                    "anomalies": sc.get("anomalies", []),
                    "pipeline_metrics": result.get("pipeline_metrics", {}),
                    "safety_results": result.get("safety_results", []),
                    "safety_audit_summary": result.get("safety_audit_summary", {}),
                    "execution_results": result.get("execution_results", []),
                })

                # Queue high-risk actions for human approval
                actions = remed_out.get("recommended_actions", [])
                report_text = report_out.get("summary", report_out.get("narrative", ""))
                q_count = _queue_actions_for_approval(scenario_name, actions, report_text)
                if q_count:
                    logger.info("Queued %d actions for human approval in scenario %s", q_count, scenario_name)

                # ── Build pipeline flow from agent timings ──
                agent_timings = _perf_state.get("agent_timings", {})
                _pipeline_run = {
                    "name": f"Incident Analysis: {scenario_name}",
                    "status": "completed",
                    "steps": [
                        {"name": "Anomaly Detection", "desc": f"{len(sc.get('anomalies',[]))} anomalies found", "status": "completed", "progress": 100, "duration": agent_timings.get("Triage Agent", 0)/1000, "start": 0},
                        {"name": "Triage", "desc": f"Severity: {triage_out.get('severity','?')}", "status": "completed", "progress": 100, "duration": agent_timings.get("Triage Agent", 0)/1000, "start": 0},
                        {"name": "Root Cause Analysis", "desc": f"Confidence: {rca_out.get('confidence_score',0):.0%}", "status": "completed", "progress": 100, "duration": agent_timings.get("RCA Agent", 0)/1000, "start": 0},
                        {"name": "Remediation Planning", "desc": f"{len(remed_out.get('recommended_actions',[]))} actions generated", "status": "completed", "progress": 100, "duration": agent_timings.get("Remediation Agent", 0)/1000, "start": 0},
                        {"name": "Report Generation", "desc": f"Total: {_perf_state.get('total_tokens',0)} tokens", "status": "completed", "progress": 100, "duration": agent_timings.get("Reporting Agent", 0)/1000, "start": 0},
                    ],
                    "start_time": start,
                    "elapsed": elapsed,
                }
                if q_count:
                    _pipeline_run["steps"].append({"name": "Human Approval", "desc": f"{q_count} actions queued", "status": "warning", "progress": 100, "duration": 0, "start": 0})
                pipeline_html = _render_pipeline_flow()

                return triage_html, rca_html, remed_html, report_html, reasoning_html, \
                       _chat_update_status()[0], _chat_update_status()[1], _chat_refresh_risk()

            except Exception as exc:
                logger.error("Orchestrator failed: %s", exc, exc_info=True)
                _pipeline_run = {
                    "name": f"Incident Analysis: {scenario_name}",
                    "status": "failed",
                    "steps": [
                        {"name": "Pipeline Execution", "desc": f"Failed: {exc}", "status": "failed", "progress": 100, "duration": 0, "start": 0},
                    ],
                    "start_time": time.time(),
                    "elapsed": 0,
                }
                pipeline_html = _render_pipeline_flow()
                error_html = (
                    f'<div class="glass-card" style="border-left:3px solid {_C["red"]};">'
                    f'<div style="color:{_C["red"]};font-weight:700;margin-bottom:8px;">Analysis Error</div>'
                    f'<div style="color:#e2e8f0;font-size:0.88rem;">{exc}</div>'
                    f'<div style="color:#64748b;font-size:0.78rem;margin-top:8px;">'
                    f'Ensure vLLM server is running at {VLLM_BASE_URL}</div>'
                    f'</div>'
                )
                return error_html, error_html, error_html, error_html, error_html, \
                       _chat_update_status()[0], _chat_update_status()[1], _chat_refresh_risk()

        # ---- Demo mode (no orchestrator) ----
        demo_triage = format_agent_output("Triage", {
            "severity": sc.get("severity", "P3"),
            "category": sc.get("title", "").split()[0].lower() if sc.get("title") else "infrastructure",
            "impact_summary": sc.get("description", "")[:120] + "...",
            "affected_services": ["payment-service", "user-service", "api-gateway"],
            "confidence": 0.92,
        })
        demo_rca = format_agent_output("Root Cause Analysis", {
            "root_causes": [
                "Primary: " + (sc.get("logs", [{}])[0].get("message", "Unknown") if sc.get("logs") else "Unknown root cause"),
                "Contributing factor: Insufficient monitoring thresholds",
                "Contributing factor: Missing automated scaling policies",
            ],
            "evidence": [log.get("message", "") for log in sc.get("logs", [])[:3]],
            "confidence": 0.87,
        })
        demo_remed = format_agent_output("Remediation", {
            "actions": [
                {"name": "Immediate Mitigation", "description": "Restart affected services to clear stale state"},
                {"name": "Short-term Fix", "description": "Apply configuration patch to increase resource limits"},
                {"name": "Long-term Resolution", "description": "Deploy code fix with proper resource management"},
                {"name": "Validation", "description": "Monitor for 30 minutes to confirm resolution"},
            ],
            "confidence": 0.85,
        })
        demo_report = format_agent_output("Incident Report", {
            "report": (
                f"## Incident Report: {sc.get('title', 'Unknown')}\n\n"
                f"**Incident ID:** {sc.get('id', 'INC-XXX')}\n"
                f"**Severity:** {sc.get('severity', 'P3')}\n"
                f"**Status:** Under Investigation\n\n"
                f"### Summary\n{sc.get('description', '')}\n\n"
                f"### Timeline\n"
                + "\n".join(f"- **{log.get('timestamp', '')}** [{log.get('level', '')}] {log.get('message', '')}"
                           for log in sc.get("logs", [])[:5])
                + "\n\n### Recommended Actions\n"
                  "1. Execute immediate mitigation steps\n"
                  "2. Apply configuration fixes\n"
                  "3. Monitor system health for 30 minutes\n"
                  "4. Conduct post-incident review within 24 hours"
            ),
        })
        demo_reasoning = (
            f'<div class="glass-card">'
            f'<div style="font-size:0.8rem;font-weight:700;color:#64748b;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:12px;">Agent Reasoning Chain</div>'
            f'<div class="evidence-item" style="border-left-color:{_C["text_muted"]};">'
            f'<span style="color:{_C["text"]};font-weight:600;">Triage Agent:</span> '
            f'Analyzed {len(sc.get("logs", []))} log entries and {len(sc.get("metrics", []))} metric snapshots. '
            f'Detected critical-level errors indicating service degradation.</div>'
            f'<div class="evidence-item" style="border-left-color:{_C["text_muted"]};">'
            f'<span style="color:{_C["text"]};font-weight:600;">RCA Agent:</span> '
            f'Cross-referenced error patterns with knowledge base. Identified primary root cause with 87% confidence. '
            f'Used BM25 retrieval to match 3 relevant runbooks.</div>'
            f'<div class="evidence-item" style="border-left-color:{_C["text_muted"]};">'
            f'<span style="color:{_C["text"]};font-weight:600;">Remediation Agent:</span> '
            f'Generated 4-step remediation plan based on runbook RB-001. Validated action safety and '
            f'estimated rollback risk as LOW.</div>'
            f'<div class="evidence-item" style="border-left-color:{_C["text_muted"]};">'
            f'<span style="color:{_C["text"]};font-weight:600;">Reporting Agent:</span> '
            f'Compiled incident timeline, root cause summary, and action items into structured report. '
            f'SLA compliance check: within P1 15-minute window.</div>'
            f'</div>'
        )

        _perf_state.update({
            "total_time_seconds": 4.2,
            "total_tokens": 3847,
            "llm_calls": 4,
            "avg_latency_ms": 1050,
            "gpu_memory_mb": 5832,
            "model": MODEL_NAME.split("/")[-1],
            "agent_timings": {
                "Triage Agent": 820,
                "RCA Agent": 1340,
                "Remediation Agent": 1180,
                "Reporting Agent": 860,
            },
            "agent_tokens": {
                "Triage Agent": 712,
                "RCA Agent": 1245,
                "Remediation Agent": 1038,
                "Reporting Agent": 852,
            },
        })

        # Update pipeline state for chat context
        _last_pipeline_state.update({
            "triage": {
            "severity": sc.get("severity", "P3"),
                "category": "application" if "Application" in scenario_name else "infrastructure",
                "impact_assessment": sc.get("description", "")[:120],
            },
            "rca": {"root_cause": sc.get("description", "Unknown")[:100], "confidence_score": 0.87},
            "remediation": {"recommended_actions": [{"tool_name": "restart_service"}]},
            "critique": {"confirmed": True},
            "anomalies": sc.get("logs", []),
        })

        # ── Build demo pipeline flow ──
        _pipeline_run = {
            "name": f"Incident Analysis: {scenario_name}",
            "status": "completed",
            "steps": [
                {"name": "Anomaly Detection", "desc": f"{len(sc.get('logs',[]))} log entries", "status": "completed", "progress": 100, "duration": 1.2, "start": 0},
                {"name": "Triage", "desc": f"Severity: {sc.get('severity','P3')}", "status": "completed", "progress": 100, "duration": 0.8, "start": 0},
                {"name": "Root Cause Analysis", "desc": "Confidence: 87%", "status": "completed", "progress": 100, "duration": 1.3, "start": 0},
                {"name": "Remediation Planning", "desc": "4 actions generated", "status": "completed", "progress": 100, "duration": 1.2, "start": 0},
                {"name": "Report Generation", "desc": "3,847 tokens", "status": "completed", "progress": 100, "duration": 0.9, "start": 0},
            ],
            "start_time": time.time() - 5.4,
            "elapsed": 5.4,
        }
        pipeline_html = _render_pipeline_flow()

        return demo_triage, demo_rca, demo_remed, demo_report, demo_reasoning, \
               _chat_update_status()[0], _chat_update_status()[1], _chat_refresh_risk()

    def _run_error_level_resolution(scenario_name: str, level_filter: str) -> str:
        """Run error-level specific resolution analysis with progress timing."""
        if not scenario_name or scenario_name not in scenarios:
            return _empty_state("Select a scenario first")

        sc = scenarios[scenario_name]
        scenario_logs = sc.get("logs", [])
        scenario_metrics = sc.get("metrics", [])

        if not scenario_logs:
            return _empty_state("No logs in this scenario")

        # Determine which levels to process
        if level_filter and level_filter != "ALL":
            levels_to_process = [level_filter]
        else:
            levels_to_process = ["CRITICAL", "ERROR", "WARNING"]

        # ── Progress simulation with realistic timing ────────────
        progress_parts = []
        import time as _time

        LEVEL_META = {
            "CRITICAL": {"icon": "🔴", "accent": "#FF3B3B", "label": "Critical"},
            "ERROR": {"icon": "🟠", "accent": "#FF3B3B", "label": "Error"},
            "WARNING": {"icon": "🟡", "accent": "#FFB800", "label": "Warning"},
        }

        # Run once with orchestrator (fast mode, no LLM per level) or use demo
        if orchestrator is not None:
            try:
                level_result = orchestrator.process_by_error_level(
                    logs=scenario_logs,
                    metrics=scenario_metrics,
                    use_llm=True,
                )
                per_level = level_result.get("per_level", {})
            except Exception as exc:
                logger.error("Error-level resolution failed: %s", exc)
                per_level = {}
        else:
            per_level = {}

        html_parts = []
        for level in levels_to_process:
            meta = LEVEL_META.get(level, {"icon": "⚪", "accent": "#64748b", "label": level})

            # Simulate realistic analysis delay per level
            _time.sleep(1.5)

            lr = per_level.get(level) if per_level else None
            if lr:
                an_count = lr.get("anomaly_count", 0)
                summary = lr.get("resolution_summary", "Analysis complete")
                steps = lr.get("resolution_steps", [])
                root_cause = lr.get("root_cause", "Analysis in progress")
                confidence = lr.get("confidence", 0)
                llm_gen = lr.get("llm_generated", False)
            else:
                # Fallback: count log entries for demo
                an_count = sum(1 for l in scenario_logs if l.get("level", "").upper() == level)
                summary = (
                    f"Analyzed {an_count} {level} log entries. "
                    f"Recommended actions generated based on standard runbook procedures."
                )
                steps = [
                    f"Immediate: Isolate affected {level}-level services",
                    f"Diagnose: Run health checks on related infrastructure",
                    f"Mitigate: Apply auto-remediation for {level} incidents",
                    f"Monitor: Track resolution for 15-minute window",
                ]
                root_cause = f"{level} indicators detected in system logs. Identifying root cause pattern."
                confidence = 0.75

            if an_count == 0 and lr is None:
                html_parts.append(
                    f'<div class="agent-panel" style="margin-bottom:16px;">'
                    f'<div class="agent-panel-header" style="background:linear-gradient(135deg,{meta["accent"]}15,transparent);">'
                    f'<span style="font-size:1.1rem;">{meta["icon"]}</span>'
                    f'<span style="color:{meta["accent"]};">{level} — No Data</span></div>'
                    f'<div class="agent-panel-body">'
                    f'<div style="color:#64748b;font-style:italic;">No {level}-level logs present in this scenario.</div>'
                    f'</div></div>'
                )
                continue

            conf_pct = confidence * 100 if isinstance(confidence, float) and confidence <= 1 else confidence

            def _render_step(step_text: str) -> str:
                parts = step_text.split(":", 1)
                if len(parts) == 2:
                    tag, desc = parts[0].strip(), parts[1].strip()
                    return f'<strong>{tag}</strong>: {desc}'
                return step_text

            steps_html = "".join(
                f'<div class="action-card">'
                f'<div class="action-icon" style="background:rgba(255,255,255,0.1);color:#64748b;">{i}</div>'
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-size:0.82rem;color:#e2e8f0;">{_render_step(s)}</div>'
                f'</div>'
                f'</div>'
                for i, s in enumerate(steps, 1)
            ) if steps else (
                f'<div style="color:#64748b;font-style:italic;padding:12px;">'
                f'No automated resolution steps generated.</div>'
            )

            html_parts.append(
                f'<div class="agent-panel" style="margin-bottom:16px;'
                f'border-left:3px solid {meta["accent"]};">'
                f'<div class="agent-panel-header" style="background:linear-gradient(135deg,{meta["accent"]}15,transparent);">'
                f'<span style="font-size:1.1rem;">{meta["icon"]}</span>'
                f'<span style="color:{meta["accent"]};font-weight:700;">{level}</span>'
                f'<span style="color:#64748b;font-size:0.78rem;margin-left:8px;">'
                f'{an_count} anomaly{"ies" if an_count != 1 else "y"}</span>'
                f'<span style="margin-left:auto;display:flex;align-items:center;gap:6px;">'
                f'<span class="pulse-dot" style="width:6px;height:6px;"></span>'
                f'<span style="font-size:0.72rem;color:{meta["accent"]};">Live</span>'
                f'<span style="font-size:0.72rem;color:#64748b;margin-left:4px;">'
                f'{conf_pct:.0f}% confidence</span>'
                f'</span>'
                f'</div>'
                f'<div class="agent-panel-body">'
                f'<div style="margin-bottom:14px;">'
                f'<span style="color:#64748b;font-size:0.73rem;text-transform:uppercase;letter-spacing:1px;">Resolution Summary</span>'
                f'<div style="margin-top:6px;font-size:0.88rem;color:#e2e8f0;line-height:1.6;">{summary}</div>'
                f'</div>'
                f'<div style="margin-bottom:12px;">'
                f'<span style="color:#64748b;font-size:0.73rem;text-transform:uppercase;letter-spacing:1px;">Root Cause</span>'
                f'<div style="margin-top:6px;font-size:0.85rem;color:#e2e8f0;background:rgba(255,255,255,0.02);'
                f'padding:10px 14px;border-radius:8px;border-left:3px solid {meta["accent"]};">'
                f'{root_cause}</div>'
                f'</div>'
                f'<div>'
                f'<span style="color:#64748b;font-size:0.73rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;display:block;">Resolution Steps</span>'
                f'{steps_html}'
                f'</div>'
                f'</div></div>'
            )

        result_html = "".join(html_parts) if html_parts else _empty_state(
            "No results", "No matching error levels in this scenario."
        )

        # Wrap with execution header
        now = datetime.now().strftime("%H:%M:%S")
        return (
            f'<div style="margin-bottom:12px;display:flex;align-items:center;gap:12px;'
            f'padding:10px 16px;background:rgba(0,255,136,0.04);border:1px solid rgba(0,255,136,0.12);'
            f'border-radius:10px;">'
            f'<span class="pulse-dot"></span>'
            f'<span style="font-size:0.82rem;color:#00FF88;font-weight:600;">Analysis Complete</span>'
            f'<span style="font-size:0.75rem;color:#64748b;">'
            f'Processed {len(levels_to_process)} level(s) · {now} · '
            f'{"Template-based" if (orchestrator is not None) else "Demo"} mode</span>'
            f'</div>'
            f'{result_html}'
        )

    # Pre-load metrics at dashboard creation time
    _full_3d_metrics: List[Dict[str, Any]] = []
    _metrics_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "metrics.json"),
    ]
    for _p in _metrics_paths:
        if os.path.exists(_p):
            try:
                import json as _json
                with open(_p) as _f:
                    _full_3d_metrics = _json.load(_f)
                logger.info("3D: pre-loaded %d metrics from %s", len(_full_3d_metrics), _p)
                break
            except Exception as _exc:
                logger.warning("3D: failed to load %s: %s", _p, _exc)

    def _get_visualizations(scenario_name: str) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
        """Generate all 2D plots for the selected scenario or all data."""

        if scenario_name and scenario_name in scenarios:
            sc = scenarios[scenario_name]
            logs = sc.get("logs", [])
            metrics = sc.get("metrics", [])
            _scenario_hosts = len(set(m.get("host") for m in metrics))
            if _scenario_hosts < 2 and len(_full_3d_metrics) >= 20:
                metrics = _full_3d_metrics
                logger.info("VIZ: supplemented scenario metrics (%d hosts -> %d hosts)",
                           _scenario_hosts, len(set(m.get("host") for m in metrics)))
        else:
            all_logs = []
            for sc in scenarios.values():
                all_logs.extend(sc.get("logs", []))
            logs = all_logs
            metrics = _full_3d_metrics if len(_full_3d_metrics) >= 20 else []

        # Run anomaly detection for timeline / scatter overlays
        from anomaly_detector import IncidentCorrelator
        detected = anomaly_detector.detect(metrics=metrics, logs=logs) if anomaly_detector else []
        incidents = IncidentCorrelator().correlate(detected)

        plot1 = time_series_dashboard(metrics, anomalies=detected,
                                      title="Metric Time-Series Dashboard")
        plot2 = correlation_heatmap(metrics, title="Metric Correlation Heatmap")
        plot3 = anomaly_timeline(incidents, title="Anomaly Timeline by Host") if incidents else \
                host_radar(metrics, title="Host Comparison — Radar")
        plot4 = log_level_distribution(logs, title="Log Level Distribution by Source")

        root_host = _last_pipeline_state.get("root_cause_host", "")
        aff_hosts = _last_pipeline_state.get("affected_hosts", [])
        all_hosts = sorted(set(m.get("host", "") for m in metrics)) if metrics else []
        plot5 = draw_topology_map(
            root_cause_host=root_host,
            affected_hosts=aff_hosts,
            all_hosts=all_hosts if all_hosts else None,
            title="Infrastructure Topology — Failure Origin",
        )
        return plot1, plot2, plot3, plot4, plot5

    def _run_anomaly_scan() -> str:
        """Run anomaly detection and return results."""
        if anomaly_detector is not None:
            try:
                all_metrics = []
                all_logs = []
                for sc in scenarios.values():
                    all_metrics.extend(sc.get("metrics", []))
                    all_logs.extend(sc.get("logs", []))
                anomalies = anomaly_detector.detect(metrics=all_metrics, logs=all_logs)
                return _format_anomalies_html(anomalies)
            except Exception as exc:
                logger.error("Anomaly detection failed: %s", exc)
                return (
                    f'<div class="glass-card" style="border-left:3px solid {_C["red"]};">'
                    f'<div style="color:{_C["red"]};font-weight:700;">⚠️ Scan Error</div>'
                    f'<div style="color:#e2e8f0;font-size:0.88rem;margin-top:6px;">{exc}</div></div>'
                )

        # Demo anomalies
        demo_anomalies = [
            {
                "id": "ANO-001", "timestamp": "2026-06-11T22:30:01Z", "type": "resource_exhaustion",
                "severity": "P1", "source": "prod-db-01",
                "description": "Database connection pool at 100% capacity (500/500 connections)",
                "evidence": ["Active connections: 500 (threshold: 450)", "Error rate: 43% (threshold: 15%)"],
                "confidence": 0.95,
            },
            {
                "id": "ANO-002", "timestamp": "2026-06-11T22:15:00Z", "type": "memory_anomaly",
                "severity": "P2", "source": "auth-svc-01",
                "description": "Memory usage trending upward — 89.2% of limit with no GC recovery",
                "evidence": ["Memory: 7.1GB/8GB (89.2%)", "GC pause: >5s (threshold: 2s)"],
                "confidence": 0.88,
            },
            {
                "id": "ANO-003", "timestamp": "2026-06-11T22:10:00Z", "type": "disk_critical",
                "severity": "P2", "source": "log-server-01",
                "description": "Disk usage at 96.1% — flood watermark exceeded on Elasticsearch",
                "evidence": ["Disk: 96.1% (threshold: 90%)", "Remaining: 42GB"],
                "confidence": 0.97,
            },
            {
                "id": "ANO-004", "timestamp": "2026-06-11T22:40:00Z", "type": "security_anomaly",
                "severity": "P1", "source": "api-gw-01",
                "description": "Credential stuffing attack in progress — 340 failed logins in 5 minutes",
                "evidence": ["Failed logins: 340 (baseline: 12)", "Unique source IPs: 23", "Rate limit triggered"],
                "confidence": 0.91,
            },
        ]
        return _format_anomalies_html(demo_anomalies)

    def _get_command_center_metrics() -> str:
        """Build the command center metric cards from actual data."""
        total_incidents = len(scenarios)
        past_inc = _load_past_incidents()
        if past_inc:
            avg_resolution = sum(i.get("duration_minutes", 30) for i in past_inc) / len(past_inc)
            resolution_str = f"{avg_resolution:.1f} min"
        else:
            resolution_str = "—"

        # Compute anomaly count from all scenario logs
        all_logs = []
        for sc in scenarios.values():
            all_logs.extend(sc.get("logs", []))
        error_logs = [l for l in all_logs if l.get("level") in ("CRITICAL", "ERROR")]
        anomaly_count = len(error_logs)

        # Compute system health from average error rate across all scenario metrics
        all_metrics = []
        for sc in scenarios.values():
            all_metrics.extend(sc.get("metrics", []))
        if all_metrics:
            avg_err = sum(m.get("error_rate", 0) for m in all_metrics) / len(all_metrics)
            health_pct = max(0, min(100, round((1 - avg_err) * 100, 1)))
        else:
            health_pct = 100.0

        return (
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">'
            f'{_metric_card_html("Active Incidents", str(total_incidents), _C["red"])}'
            f'{_metric_card_html("Anomalies Detected", str(anomaly_count), _C["amber"])}'
            f'{_metric_card_html("Mean Resolution", resolution_str, _C["text"])}'
            f'{_metric_card_html("System Health", f"{health_pct}%", _C["green"])}'
            f'</div>'
        )

    def _get_command_center_logs() -> str:
        """Aggregate all scenario logs for the live log stream."""
        all_logs: List[Dict[str, Any]] = []
        for sc in scenarios.values():
            all_logs.extend(sc.get("logs", []))
        all_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return format_log_table(all_logs[:50])

    def _generate_live_logs() -> List[Dict[str, Any]]:
        """Generate synthetic live logs using the LogStreamer."""
        global _live_log_cache, _live_log_lock
        try:
            if not _live_log_cache:
                for sc in scenarios.values():
                    _live_log_cache.extend(sc.get("logs", []))
            streamer = LogStreamer(logs=_live_log_cache, replay_speed=2.0)
            batch = []
            for logs_batch in streamer.stream(batch_size=3, delay_per_log=0.3):
                batch.extend(logs_batch)
                if len(batch) >= 10:
                    break
            with _live_log_lock:
                _live_log_cache[:] = _live_log_cache[len(batch):] + batch
            return batch
        except Exception as exc:
            logger.warning("Live stream error: %s", exc)
            return []

    def _render_live_logs() -> str:
        """Render the current live log stream as HTML."""
        try:
            logs = _generate_live_logs()
            if not logs:
                return _get_command_center_logs()
            return format_log_table(logs)
        except Exception as exc:
            _diag("render_live_logs_error", exc=str(exc))
            return _get_command_center_logs()

    def _get_perf_metrics_html() -> str:
        """Build the performance metrics tab."""
        if not _perf_state:
            return _empty_state(
                "No performance data yet",
                "Run an incident analysis on the Incident Analysis tab first."
            )

        overview = format_metrics_panel(_perf_state)
        latency = _build_agent_latency_chart(_perf_state.get("agent_timings", {}))
        tokens = _build_token_chart(_perf_state.get("agent_tokens", {}))

        return (
            f'<div style="display:flex;flex-direction:column;gap:20px;">'
            f'{overview}{latency}{tokens}'
            f'</div>'
        )

    def _search_runbooks(query: str) -> str:
        """Search the knowledge base runbooks."""
        if not query or not query.strip():
            return "".join(_format_runbook_html(rb) for rb in runbooks)

        q = query.lower().strip()
        matched = [
            rb for rb in runbooks
            if q in rb.get("title", "").lower()
            or q in rb.get("category", "").lower()
            or any(q in s.lower() for s in rb.get("symptoms", []))
            or any(q in t.lower() for t in rb.get("tags", []))
        ]
        if not matched:
            return _empty_state(f"No runbooks matching '{query}'", "Try a different search term.")
        return "".join(_format_runbook_html(rb) for rb in matched)

    def _generate_report() -> str:
        """Generate a summary report of all incidents."""
        _diag("generate_report_enter", scenarios=len(scenarios), scenario_results=len(_scenario_results))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        incident_rows = ""
        for name, sc in scenarios.items():
            sev = sc.get("severity", "P3")
            rc_result = _scenario_results.get(name, {})
            rca_out = rc_result.get("rca", rc_result.get("rca_result", {}))
            rc_text = "—"
            rcs = []
            if rca_out:
                rcs = rca_out.get("root_causes", []) if isinstance(rca_out, dict) else []
                if rcs:
                    rc_text = html.escape(rcs[0][:100])
            if not rcs and sc.get("description"):
                rc_text = html.escape(sc["description"][:60] + "...")
            analyzed = name in _scenario_results
            status_label = "Analyzed" if analyzed else "Detected"
            status_color = _C["green"] if analyzed else _C["text"]
            incident_rows += (
                f'<tr>'
                f'<td>{sc.get("id", "—")}</td>'
                f'<td>{format_severity_badge(sev)}</td>'
                f'<td>{sc.get("title", name)}</td>'
                f'<td>{rc_text}</td>'
                f'<td>{len(sc.get("logs", []))}</td>'
                f'<td style="color:{status_color};">{status_label}</td>'
                f'</tr>'
            )

        html_out = (
            f'<div class="glass-card">'
            f'<div style="margin-bottom:18px;">'
            f'<div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;">InfraHeal AI — Summary Report</div>'
            f'<div style="font-size:0.75rem;color:#64748b;">Generated {now}</div>'
            f'</div>'
            f'<table class="styled-table">'
            f'<thead><tr><th>ID</th><th>Severity</th><th>Title</th><th>Root Cause</th><th>Log Entries</th><th>Status</th></tr></thead>'
            f'<tbody>{incident_rows}</tbody>'
            f'</table>'
            f'<div style="margin-top:18px;padding:14px;background:rgba(0,255,136,0.04);'
            f'border:1px solid rgba(0,255,136,0.15);border-radius:10px;">'
            f'<div style="font-size:0.82rem;color:{_C["green"]};font-weight:600;">✅ Report Complete</div>'
            f'<div style="font-size:0.78rem;color:#64748b;margin-top:4px;">'
            f'{len(scenarios)} incidents catalogued · AMD GPU accelerated inference · Model: {MODEL_NAME.split("/")[-1]}</div>'
            f'</div></div>'
        )
        _diag("generate_report_exit", html_len=len(html_out), has_rows=bool(incident_rows))
        return html_out

    def _run_pipeline_thread(gen_func: Callable, pipeline_type: str = "process"):
        global _process_live_html, _monitor_live_html, _process_completed, _monitoring_completed, _static_output_active
        import sys
        fn_name = gen_func.__name__
        _diag("pipeline_thread_start", fn=fn_name, static=_static_output_active)
        print("[THREAD] Starting pipeline thread", flush=True)
        is_monitor = (pipeline_type == "monitor")
        writes = 0
        skips = 0
        try:
            for partial in gen_func():
                if not _static_output_active:
                    with _live_html_lock:
                        if is_monitor:
                            _monitor_live_html = partial
                        else:
                            _process_live_html = partial
                        writes += 1
                else:
                    skips += 1
        except Exception:
            import traceback
            exc = traceback.format_exc()
            print("[THREAD] Pipeline thread FAILED:\n" + exc, flush=True)
            _diag("pipeline_thread_error", fn=fn_name, exc=str(sys.exc_info()[1]))
        if is_monitor:
            _monitoring_completed = True
        else:
            _process_completed = True
        _diag("pipeline_thread_exit", fn=fn_name, writes=writes, skips=skips, completed=True)
        print("[THREAD] Pipeline thread exiting", flush=True)

    def _start_process():
        global _process_thread, _process_live_html, _process_completed, _scenario_results, _static_output_active
        nonlocal _last_pipeline
        p_alive = _process_thread and _process_thread.is_alive()
        _diag("start_process_click", process_alive=p_alive, completed=_process_completed, static=_static_output_active)
        print("[START_PROCESS] Button clicked!", flush=True)
        if p_alive:
            _static_output_active = False
            with _live_html_lock:
                return _process_live_html
        if _process_completed:
            _last_pipeline = "process"
            print("[START_PROCESS] Already completed, showing cached result", flush=True)
            with _live_html_lock:
                return _process_live_html
        _static_output_active = False
        _process_completed = False
        _last_pipeline = "process"
        _result_cache.pop("report", None)
        _scenario_results.clear()
        _process_live_html = '<div style="color:#8b949e;text-align:center;padding:12px;">Starting... </div>'
        print("[START_PROCESS] Starting thread...", flush=True)
        _process_thread = threading.Thread(target=_run_pipeline_thread, args=(_process_all_incidents, "process"), daemon=True)
        _process_thread.start()
        return _process_live_html

    def _start_monitor():
        global _monitor_thread, _monitor_live_html, _scenario_results
        global _stop_monitoring_requested, _monitoring_active, _monitoring_completed, _static_output_active
        nonlocal _last_pipeline
        m_alive = _monitor_thread and _monitor_thread.is_alive()
        _diag("start_monitor_click", monitor_alive=m_alive, completed=_monitoring_completed, static=_static_output_active)
        print("[START_MONITOR] Button clicked!", flush=True)
        if m_alive:
            _static_output_active = False
            with _live_html_lock:
                return _monitor_live_html
        if _monitoring_completed:
            _last_pipeline = "monitor"
            print("[START_MONITOR] Already completed, showing cached result", flush=True)
            with _live_html_lock:
                return _monitor_live_html
        _static_output_active = False
        _monitoring_completed = False
        _stop_monitoring_requested = False
        _monitoring_active = False
        _last_pipeline = "monitor"
        _result_cache.pop("report", None)
        _scenario_results.clear()
        _monitor_live_html = '<div style="color:#8b949e;text-align:center;padding:12px;">Starting continuous monitoring...</div>'
        print("[START_MONITOR] Starting thread...", flush=True)
        _monitor_thread = threading.Thread(target=_run_pipeline_thread, args=(_continuous_monitor, "monitor"), daemon=True)
        _monitor_thread.start()
        return _monitor_live_html

    def _poll_live_html():
        """Called by gr.Timer to poll pipeline thread progress."""
        try:
            global _process_thread, _monitor_thread, _process_live_html, _monitor_live_html, _static_output_active
            nonlocal _last_pipeline
            p_alive = _process_thread and _process_thread.is_alive()
            m_alive = _monitor_thread and _monitor_thread.is_alive()
            if _static_output_active:
                return None
            with _live_html_lock:
                if p_alive:
                    return _process_live_html
                if m_alive:
                    return _monitor_live_html
                if _last_pipeline == "monitor":
                    return _monitor_live_html or _process_live_html
                return _process_live_html or _monitor_live_html
        except Exception as exc:
            _diag("poll_live_html_error", exc=str(exc))
            return None

    def _process_all_incidents():
        """Run the pipeline on every scenario and produce a comprehensive report."""
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = ""
        total_anomalies = 0
        total_actions = 0
        processed = 0
        failures = 0
        agg_time = 0.0
        agg_tokens = 0
        agg_calls = 0
        agg_latency_sum = 0.0
        agg_latency_count = 0
        agg_timings: Dict[str, float] = {}
        agg_tokens_per_agent: Dict[str, int] = {}
        scenario_list = list(scenarios.items())
        total_scenarios = len(scenario_list)

        # Initialize pipeline
        _pipeline_run.clear()
        _pipeline_run.update({
            "name": "Process All Incidents",
            "status": "running",
            "steps": [],
            "start_time": time.time(),
            "elapsed": 0,
        })

        def _add_step(name, desc="", status="pending"):
            s = {"name": name, "desc": desc, "status": status, "progress": 0 if status == "running" else 0, "duration": 0, "start": time.time() if status == "running" else 0}
            _pipeline_run["steps"].append(s)
            return s

        def _complete_step(step, status="completed"):
            step["status"] = status
            step["progress"] = 100
            step["duration"] = time.time() - step["start"]

        # Step 1: Load experience store
        _add_step("Load Experience Store", "Injecting few-shot examples & preferences", "running")
        yield _render_pipeline_flow() + '<div style="color:#8b949e;font-size:0.78rem;text-align:center;padding:12px;">Processing scenarios...</div>'
        _complete_step(_pipeline_run["steps"][-1])

        # Pre-populate all scenario steps as "pending" so progress starts at 0% not 50%
        _scenario_steps = []
        for si, (name, sc) in enumerate(scenario_list):
            title = sc.get("title", name)
            _scenario_steps.append(_add_step(f"[{si+1}/{total_scenarios}] {title}", "Waiting...", "pending"))
        yield _render_pipeline_flow() + '<div style="color:#8b949e;font-size:0.78rem;text-align:center;padding:12px;">Processing scenarios...</div>'

        for i, (name, sc) in enumerate(scenario_list):
            sc_data = dict(sc)
            result = None
            title = sc_data.get("title", name)
            step = _scenario_steps[i]
            step["status"] = "running"
            step["start"] = time.time()
            step["desc"] = "Running anomaly detection & agents"
            yield _render_pipeline_flow() + '<div style="color:#8b949e;font-size:0.78rem;text-align:center;padding:12px;">Processing scenarios...</div>'

            if orchestrator is not None:
                try:
                    if anomaly_detector is not None:
                        detected = anomaly_detector.detect_all(
                            logs=sc_data.get("logs", []),
                            metrics=sc_data.get("metrics", []),
                        )
                        sc_data["anomalies"] = detected
                        total_anomalies += len(detected)
                    from concurrent.futures import ThreadPoolExecutor
                    pool = ThreadPoolExecutor(max_workers=1)
                    fut = pool.submit(orchestrator.process_scenario, sc_data)
                    try:
                        result = fut.result(timeout=120)
                    finally:
                        pool.shutdown(wait=False)
                        if not fut.done():
                            fut.cancel()
                    processed += 1
                    _complete_step(step, "completed")
                except Exception as exc:
                    logger.warning("Pipeline failed for %s: %s", name, exc)
                    failures += 1
                    _complete_step(step, "failed")
                    step["desc"] = f"Timeout or error: {str(exc)[:70]}"
            else:
                # Demo mode: simulate
                time.sleep(0.3)
                processed += 1
                _complete_step(step, "completed")

            yield _render_pipeline_flow() + '<div style="color:#8b949e;font-size:0.78rem;text-align:center;padding:12px;">Processing scenarios...</div>'

            if result:
                perf = result.get("pipeline_metrics", {})
                agg_time += perf.get("total_time_seconds", 0)
                agent_m = perf.get("agent_metrics", {})
                totals = agent_m.get("totals", {})
                agg_tokens += totals.get("total_tokens", 0)
                agg_calls += totals.get("total_calls", 0)
                agents_data = agent_m.get("agents", {})
                for akey, adata in agents_data.items():
                    lat = adata.get("avg_latency", 0)
                    tok = adata.get("total_tokens", 0)
                    alabel = akey.replace("_", " ").title()
                    agg_timings[alabel] = agg_timings.get(alabel, 0) + lat
                    agg_tokens_per_agent[alabel] = agg_tokens_per_agent.get(alabel, 0) + tok
                    if adata.get("successful_calls", 0) > 0:
                        agg_latency_sum += lat
                        agg_latency_count += 1

            sev = "P3"
            cat = "unknown"
            rc = "—"
            action_count = 0
            if result:
                tri = result.get("triage", result.get("triage_result", {}))
                rca = result.get("rca", result.get("rca_result", {}))
                remed = result.get("remediation", result.get("remediation_result", {}))
                rep = result.get("report", {})
                sev = tri.get("severity", "P3")
                cat = tri.get("category", "unknown")
                rc = rca.get("root_cause", "—")[:80]
                action_count = len(remed.get("recommended_actions", []))
                total_actions += action_count

                _scenario_results[name] = result
                report_text = rep.get("summary", rep.get("narrative", ""))
                q_count = _queue_actions_for_approval(name, remed.get("recommended_actions", []), report_text)
                if q_count:
                    logger.info("Queued %d actions for human approval in scenario %s", q_count, name)

                # Block on high-risk actions
                if _blocked_scenarios.get(name):
                    _diag("pipeline_blocked", scenario=name, blocking=_blocked_scenarios.get(name))
                    step["status"] = "blocked"
                    step["desc"] = "BLOCKED — high-risk action pending approval"
                    yield _render_pipeline_flow() + _render_blocked_banner()
                    while _blocked_scenarios.get(name):
                        yield _render_pipeline_flow() + _render_blocked_banner()
                        time.sleep(2)
                    _diag("pipeline_unblocked", scenario=name)
                    _complete_step(step, "completed")
                    yield _render_pipeline_flow()

            badge = format_severity_badge(sev)
            rows += (
                f'<tr>'
                f'<td style="white-space:nowrap;">{sc_data.get("id", "—")}</td>'
                f'<td style="white-space:nowrap;">{badge}</td>'
                f'<td>{sc_data.get("title", name)}</td>'
                f'<td>{cat}</td>'
                f'<td style="font-size:0.78rem;">{rc}</td>'
                f'<td>{action_count}</td>'
                f'<td style="color:{"#FF3B3B" if result is None else _C["green"]};">{"Failed" if result is None else "Analyzed"}</td>'
                f'</tr>'
            )

        # Update perf state with aggregated data
        _perf_state.update({
            "total_time_seconds": round(agg_time, 1),
            "total_tokens": agg_tokens,
            "llm_calls": agg_calls,
            "avg_latency_ms": round((agg_latency_sum / agg_latency_count) * 1000, 0) if agg_latency_count > 0 else 0,
            "gpu_memory_mb": 0,
            "model": MODEL_NAME.split("/")[-1],
            "agent_timings": agg_timings,
            "agent_tokens": agg_tokens_per_agent,
        })

        _pipeline_run["status"] = "completed" if failures == 0 else "warning"
        yield _render_pipeline_flow() + (
            f'<div class="glass-card">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">'
            f'<span style="font-size:1.05rem;font-weight:700;color:#e2e8f0;">InfraHeal AI — Process All Incidents</span>'
            f'<div style="font-size:0.75rem;color:#64748b;">Generated {now}</div>'
            f'</div></div>'
            f'<table class="styled-table">'
            f'<thead><tr><th style="white-space:nowrap;">ID</th><th style="white-space:nowrap;">Sev</th><th>Title</th><th>Category</th><th>Root Cause</th><th>Actions</th><th>Status</th></tr></thead>'
            f'<tbody>{rows}</tbody>'
            f'</table>'
            f'<div style="margin-top:18px;padding:14px;background:rgba(0,255,136,0.04);'
            f'border:1px solid rgba(0,255,136,0.15);border-radius:10px;">'
            f'<div style="font-size:0.82rem;color:{_C["green"]};font-weight:600;">{"✅ Pipeline Complete" if not failures else "⚠️ Pipeline Complete"}</div>'
            f'<div style="font-size:0.78rem;color:#64748b;margin-top:4px;">'
            f'{processed} scenarios processed · {total_anomalies} anomalies · {total_actions} actions generated · '
            f'{failures} failures · Model: {MODEL_NAME.split("/")[-1]}</div>'
            f'</div></div>'
        )

    # ═══════════════════════════════════════════════════════════════
    #  HUMAN APPROVAL & CONTINUOUS MONITORING
    # ═══════════════════════════════════════════════════════════════

    def _render_approval_panel() -> str:
        global _pending_approvals
        blocked_banner = ""
        if _blocked_scenarios:
            blocked_banner = (
                f'<div style="padding:10px 14px;margin:8px 0;border:2px solid #FF3B3B;border-radius:8px;'
                f'background:rgba(255,59,59,0.08);font-size:0.82rem;">'
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">'
                f'<span style="font-size:1rem;">🛑</span>'
                f'<span style="color:#FF3B3B;font-weight:700;">Pipeline BLOCKED by high-risk action(s)</span></div>'
                f'<div style="color:#e2e8f0;">Scenario(s): '
                f'<span style="color:#FFB800;font-weight:600;">{", ".join(html.escape(s) for s in _blocked_scenarios)}</span></div>'
                f'<div style="color:#8b949e;font-size:0.78rem;">The pipeline is paused until you approve or deny the high-risk action(s) below.</div>'
                f'</div>'
            )
        pending = [a for a in _pending_approvals if a.get("status") == "pending"]
        if not pending:
            return blocked_banner or (
                '<div style="font-size:0.82rem;color:#64748b;text-align:center;padding:16px;">'
                'No pending approvals.</div>'
            )
        items = "".join(
            f'<div style="padding:12px;margin:8px 0;border:1px solid rgba(255,255,255,0.08);border-radius:8px;'
            f'background:rgba(255,255,255,0.02);{"border-left:3px solid #FF3B3B;" if a.get("blocks_pipeline") else "border-left:3px solid #8b949e;"}">'
            f'<div style="display:flex;justify-content:space-between;align-items:start;">'
            f'<div style="flex:1;">'
            f'<div style="font-size:0.85rem;font-weight:600;color:#e2e8f0;">'
            f'{"🛑 " if a.get("blocks_pipeline") else ""}{a.get("title","Action")}</div>'
            f'<div style="font-size:0.75rem;color:#8b949e;margin:4px 0;">{a.get("id","")} — {a.get("scenario","?")}'
            f'{" | Cycle #"+str(a.get("cycle","")) if a.get("cycle") else ""} | '
            f'Risk: <span style="color:{_risk_color(a.get("risk","medium"))};">{a.get("risk","medium")}</span>'
            f'{" | <span style=\"color:#FF3B3B;font-weight:600;\">BLOCKING</span>" if a.get("blocks_pipeline") else ""}</div>'
            f'<div style="font-size:0.78rem;color:#c9d1d9;white-space:pre-wrap;">{a.get("summary","")[:200]}</div>'
            f'</div></div>'
            f'</div>'
            for a in pending
        )
        if not items:
            return blocked_banner
        return blocked_banner + (
            f'<div style="margin-top:12px;">'
            f'<div style="font-size:0.8rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
            f'Pending Approvals ({len(pending)})</div>{items}'
            f'<div style="font-size:0.75rem;color:#8b949e;text-align:right;margin-top:6px;">'
            f'Go to the Approvals tab to review and act on these items.</div>'
            f'</div>'
        )

    def _clear_scenario_block(scenario_name: str):
        """Remove block flag for a scenario if no more blocking actions are pending."""
        remaining = any(
            x.get("scenario") == scenario_name
            and x.get("status") == "pending"
            and x.get("blocks_pipeline")
            for x in _pending_approvals
        )
        if not remaining:
            _blocked_scenarios.pop(scenario_name, None)
            _diag("scenario_unblocked", scenario=scenario_name)

    def _approve_action(action_id: str, reason: str = "") -> str:
        """Approve a pending action and execute it."""
        global _pending_approvals, _approval_history
        for a in _pending_approvals:
            if a.get("id") == action_id and a.get("status") == "pending":
                a["status"] = "approved"
                a["reason"] = reason or "Approved"
                a["resolved_at"] = datetime.now().isoformat()
                logger.info("Action %s approved by human", action_id)

                # Execute the action if orchestrator is available
                exec_result = {"status": "approved_pending"}
                if orchestrator is not None and a.get("action_payload"):
                    try:
                        exec_result = orchestrator.remediation_agent.execute_action(a["action_payload"])
                        exec_result["step"] = a.get("step", 0)
                        logger.info("Action %s executed: %s", action_id, exec_result.get("status"))
                    except Exception as exc:
                        logger.error("Action %s execution failed: %s", action_id, exc)
                        exec_result = {"status": "failed", "error": str(exc)}

                exec_msg = exec_result.get("message", exec_result.get("error", ""))
                a["execution_status"] = exec_result.get("status", "unknown")
                a["execution_detail"] = exec_msg[:200]
                a["execution_steps"] = exec_result.get("steps", [])
                _approval_history.append({
                    "id": action_id,
                    "scenario": a.get("scenario", "?"),
                    "title": a.get("title", "?"),
                    "action": "approved",
                    "cycle": a.get("cycle", 0),
                    "timestamp": a["resolved_at"],
                    "execution": exec_result.get("status", "unknown"),
                    "execution_detail": exec_msg[:120],
                    "execution_steps": exec_result.get("steps", []),
                    "reason": a["reason"],
                })
                # Log to experience store
                _add_experience({
                    "scenario": a.get("scenario", "?"),
                    "severity": a.get("severity", "P3"),
                    "category": a.get("category", "infrastructure"),
                    "root_cause": a.get("summary", "")[:120],
                    "actions_attempted": [{
                        "tool_name": a.get("title", "?"),
                        "risk_level": a.get("risk", "medium"),
                        "approved": True,
                        "denied": False,
                    }] if a.get("action_payload") else [],
                    "verdict": "approved",
                    "execution_status": exec_result.get("status", "unknown"),
                    "timestamp": a["resolved_at"],
                })
                # Write audit log
                _append_audit_log({
                    "action": "approved",
                    "id": action_id,
                    "title": a.get("title", "?"),
                    "scenario": a.get("scenario", "?"),
                    "risk": a.get("risk", "medium"),
                    "execution": exec_result.get("status", "unknown"),
                    "reason": a["reason"],
                    "timestamp": a["resolved_at"],
                })
                logger.info("Audit: approved %s (%s)", action_id, a.get("title", "?"))
                _clear_scenario_block(a.get("scenario", ""))
                return _render_approval_panel()
        return _render_approval_panel()

    def _deny_action(action_id: str, reason: str = "") -> str:
        """Deny a pending action with optional reason."""
        global _pending_approvals, _approval_history
        for a in _pending_approvals:
            if a.get("id") == action_id and a.get("status") == "pending":
                a["status"] = "denied"
                a["reason"] = reason or "No reason provided"
                a["resolved_at"] = datetime.now().isoformat()
                logger.info("Action %s denied by human: %s", action_id, a["reason"])
                _approval_history.append({
                    "id": action_id,
                    "scenario": a.get("scenario", "?"),
                    "title": a.get("title", "?"),
                    "action": "denied",
                    "cycle": a.get("cycle", 0),
                    "reason": a["reason"],
                    "timestamp": a["resolved_at"],
                })
                # Log to experience store
                _add_experience({
                    "scenario": a.get("scenario", "?"),
                    "severity": a.get("severity", "P3"),
                    "category": a.get("category", "infrastructure"),
                    "root_cause": a.get("summary", "")[:120],
                    "actions_attempted": [{
                        "tool_name": a.get("title", "?"),
                        "risk_level": a.get("risk", "medium"),
                        "approved": False,
                        "denied": True,
                    }] if a.get("action_payload") else [],
                    "verdict": "denied",
                    "reason": a["reason"],
                    "timestamp": a["resolved_at"],
                })
                # Write audit log
                a["execution_status"] = "skipped"
                a["execution_detail"] = a["reason"]
                _append_audit_log({
                    "action": "denied",
                    "id": action_id,
                    "title": a.get("title", "?"),
                    "scenario": a.get("scenario", "?"),
                    "risk": a.get("risk", "medium"),
                    "execution": "skipped",
                    "reason": a["reason"],
                    "timestamp": a["resolved_at"],
                })
                logger.info("Audit: denied %s (%s) — %s", action_id, a.get("title", "?"), a["reason"])
                _clear_scenario_block(a.get("scenario", ""))
                return _render_approval_panel()
        return _render_approval_panel()

    def _queue_actions_for_approval(scenario_name: str, actions: list, report_summary: str, cycle: int = 0) -> int:
        """Queue remediation actions that require human approval. Returns count."""
        global _pending_approvals, _approval_id_counter, _auto_approve
        count = 0
        existing = {(a.get("scenario",""), a.get("title","")) for a in _pending_approvals if a.get("status") == "pending"}
        for step_idx, action in enumerate(actions):
            if action.get("requires_approval", False):
                risk_level = action.get("risk_level", "medium")
                key = (scenario_name, action.get("tool_name", "Unknown action"))
                if key in existing:
                    logger.info("Skipping duplicate approval: %s", key)
                    continue
                is_high_risk = risk_level == "high"
                if _auto_approve:
                    _approval_id_counter += 1
                    aid = f"AUTO-{_approval_id_counter:04d}"
                    exec_result = {"status": "skipped", "message": "auto-approved"}
                    if orchestrator is not None:
                        try:
                            exec_result = orchestrator.remediation_agent.execute_action(action)
                        except Exception as exc:
                            logger.error("Auto-approve exec failed: %s", exc)
                            exec_result = {"status": "failed", "error": str(exc)}
                    _approval_history.append({
                        "id": aid,
                        "scenario": scenario_name,
                        "title": action.get("tool_name", "Unknown action"),
                        "action": "auto-approved",
                        "risk": risk_level,
                        "reason": "Auto-approved (auto-approve mode)",
                        "timestamp": datetime.now().isoformat(),
                        "execution": exec_result.get("status", "unknown"),
                        "execution_detail": exec_result.get("message", exec_result.get("error", ""))[:200],
                        "execution_steps": exec_result.get("steps", []),
                    })
                    _diag("auto_approved", action=action.get("tool_name"), scenario=scenario_name, status=exec_result.get("status"))
                    logger.info("Auto-approved action %s in scenario %s", action.get("tool_name"), scenario_name)
                    count += 1
                    continue
                _approval_id_counter += 1
                _pending_approvals.append({
                    "id": f"APP-{_approval_id_counter:04d}",
                    "scenario": scenario_name,
                    "title": action.get("tool_name", "Unknown action"),
                    "risk": risk_level,
                    "blocks_pipeline": is_high_risk,
                    "summary": report_summary[:200],
                    "timestamp": datetime.now().isoformat(),
                    "status": "pending",
                    "reason": "",
                    "step": step_idx + 1,
                    "cycle": cycle,
                    "action_payload": action,
                })
                existing.add(key)
                count += 1
                if is_high_risk:
                    _blocked_scenarios[scenario_name] = True
                    _diag("high_risk_queued", action=action.get("tool_name"), scenario=scenario_name)
                    logger.info("Pipeline BLOCKED by high-risk action %s in scenario %s",
                                action.get("tool_name"), scenario_name)
        return count

    # ── Shared helper: renders blocked pipeline banner ──
    def _render_blocked_banner() -> str:
        blocked_scenarios = list(_blocked_scenarios.keys())
        if not blocked_scenarios:
            return ""
        return (
            f'<div style="padding:12px 16px;margin:12px 0;border:2px solid #FF3B3B;border-radius:8px;'
            f'background:rgba(255,59,59,0.1);font-size:0.85rem;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<span style="font-size:1.2rem;">🛑</span>'
            f'<span style="color:#FF3B3B;font-weight:700;">Pipeline BLOCKED</span></div>'
            f'<div style="color:#e2e8f0;">High-risk action(s) pending approval in: '
            f'<span style="color:#FFB800;font-weight:600;">{", ".join(html.escape(s) for s in blocked_scenarios)}</span></div>'
            f'<div style="color:#8b949e;font-size:0.80rem;margin-top:4px;">'
            f'Approve or deny the high-risk action(s) in the Approvals tab to resume the pipeline.</div>'
            f'</div>'
        )

    def _continuous_monitor():
        """Run continuous monitoring loop: process incidents, poll for new anomalies, report.
        Loops every MONITOR_POLL_SECONDS until _stop_monitoring_requested is True.
        """
        logger.info("=== _continuous_monitor generator started ===")
        global _monitoring_active, _stop_monitoring_requested
        if _monitoring_active:
            yield _render_pipeline_flow() + '<div style="color:orange;">Monitoring already active</div>'
            return
        _monitoring_active = True
        _stop_monitoring_requested = False
        logger.info("Continuous monitoring started")

        _pipeline_run.clear()
        _pipeline_run.update({
            "name": "Continuous Monitoring",
            "status": "running",
            "steps": [],
            "start_time": time.time(),
            "elapsed": 0,
        })

        def _add_step(name, desc="", status="pending"):
            s = {"name": name, "desc": desc, "status": status, "progress": 0 if status == "running" else 0, "duration": 0, "start": time.time() if status == "running" else 0}
            _pipeline_run["steps"].append(s)
            return s

        def _complete_step(step, status="completed"):
            step["status"] = status
            step["progress"] = 100
            step["duration"] = time.time() - step["start"]

        try:
            _add_step("Initialize Monitor", "Setting up log stream & anomaly detection", "running")
            yield _render_pipeline_flow()
            _complete_step(_pipeline_run["steps"][-1])
            yield _render_pipeline_flow()

            scenario_list = list(scenarios.items())
            total_scenarios = len(scenario_list)
            summary_rows = ""
            total_anomalies = 0
            total_actions = 0
            processed = 0
            failures = 0
            poll_cycle = 0

            # Pre-populate scenario steps so progress starts at 0% not 50% for initial scan
            _prebuilt_steps = []
            def _prepopulate_initial_scan():
                nonlocal _prebuilt_steps
                _prebuilt_steps = [_add_step(f"Initial Scan: Process {total_scenarios} Scenarios", f"0/{total_scenarios}", "pending")]
                for si, (name, sc) in enumerate(scenario_list):
                    title = sc.get("title", name)
                    _prebuilt_steps.append(_add_step(f"  [{si+1}/{total_scenarios}] {title}", "Waiting...", "pending"))
                yield _render_pipeline_flow()
            for chunk in _prepopulate_initial_scan():
                yield chunk

            def _process_scenarios_cycle(slist, cycle_label, prebuilt_steps=None):
                nonlocal summary_rows, total_anomalies, total_actions, processed, failures
                if prebuilt_steps:
                    ps = prebuilt_steps[0]
                    ps["status"] = "running"
                    ps["desc"] = f"0/{len(slist)}"
                    ps["start"] = time.time()
                else:
                    ps = _add_step(f"{cycle_label}: Process {len(slist)} Scenarios", f"0/{len(slist)}", "running")
                yield _render_pipeline_flow()
                for i, (name, sc) in enumerate(slist):
                    if _stop_monitoring_requested:
                        break

                    # If pipeline is blocked by a high-risk action, stop processing
                    if _blocked_scenarios:
                        blocked_names = ", ".join(html.escape(s) for s in _blocked_scenarios)
                        sub = _add_step(f"  🛑 Pipeline blocked by: {blocked_names}", "Awaiting approval...", "blocked") if not (prebuilt_steps and i < len(prebuilt_steps) - 1) else prebuilt_steps[i + 1]
                        if prebuilt_steps and i < len(prebuilt_steps) - 1:
                            sub = prebuilt_steps[i + 1]
                            sub["status"] = "blocked"
                            sub["desc"] = "BLOCKED — awaiting approval"
                        yield _render_pipeline_flow() + _render_blocked_banner()
                        break

                    sc_data = dict(sc)
                    result = None
                    dl = []
                    title = sc_data.get("title", name)
                    if prebuilt_steps and i < len(prebuilt_steps) - 1:
                        sub = prebuilt_steps[i + 1]
                        sub["status"] = "running"
                        sub["desc"] = "Anomaly detection & agents"
                        sub["start"] = time.time()
                    else:
                        sub = _add_step(f"  [{i+1}/{len(slist)}] {title}", "Anomaly detection & agents", "running")
                    yield _render_pipeline_flow()

                    print(f"[MONITOR] orchestrator={orchestrator}, anomaly_detector={anomaly_detector}", flush=True)
                    if orchestrator is not None:
                        try:
                            print(f"[MONITOR] Calling orchestrator for {name}", flush=True)
                            if anomaly_detector is not None:
                                dl = anomaly_detector.detect_all(
                                    logs=sc_data.get("logs", []),
                                    metrics=sc_data.get("metrics", []),
                                )
                                sc_data["anomalies"] = dl
                                total_anomalies += len(dl)
                                print(f"[MONITOR] Detected {len(dl)} anomalies", flush=True)
                            from concurrent.futures import ThreadPoolExecutor
                            pool = ThreadPoolExecutor(max_workers=1)
                            fut = pool.submit(orchestrator.process_scenario, sc_data)
                            try:
                                result = fut.result(timeout=120)
                            finally:
                                pool.shutdown(wait=False)
                                if not fut.done():
                                    fut.cancel()
                            print(f"[MONITOR] orchestrator returned result={result is not None}, actions={len(result.get('remediation',{}).get('recommended_actions',[])) if result else 0}", flush=True)
                            processed += 1
                            _complete_step(sub, "completed")
                        except Exception as exc:
                            logger.warning("Monitor scenario failed for %s: %s", name, exc)
                            print(f"[MONITOR] Exception: {exc}", flush=True)
                            import traceback; traceback.print_exc()
                            failures += 1
                            _complete_step(sub, "failed")
                            sub["desc"] = f"Timeout or error: {str(exc)[:70]}"
                    else:
                        print(f"[MONITOR] No orchestrator (demo mode), sleeping 0.3s", flush=True)
                        time.sleep(0.3)
                        processed += 1
                        _complete_step(sub, "completed")

                    ps["desc"] = f"{i+1}/{len(slist)} scenarios"
                    yield _render_pipeline_flow()

                    if result:
                        remed = result.get("remediation", {})
                        actions = remed.get("recommended_actions", [])
                        total_actions += len(actions)
                        report_out = result.get("report", {})
                        severity = sc_data.get("severity", "P3")
                        rc_text = html.escape(sc_data.get("description", "—")[:60] + "...") if sc_data.get("description") else "—"
                        rca_out = result.get("rca", result.get("rca_result", {}))
                        if rca_out:
                            rcs = rca_out.get("root_causes", []) if isinstance(rca_out, dict) else []
                            if rcs:
                                rc_text = html.escape(rcs[0][:80])
                        report_text = report_out.get("summary", report_out.get("narrative", "")) if report_out else ""
                        if result:
                            _scenario_results[name] = result
                        if actions:
                            _queue_actions_for_approval(name, actions, report_text, cycle=poll_cycle)
                        # Check if this scenario just became blocked by a high-risk action
                        if _blocked_scenarios.get(name):
                            _complete_step(sub, "blocked")
                            sub["desc"] = "BLOCKED — high-risk action pending approval"
                            yield _render_pipeline_flow() + _render_blocked_banner()
                            break
                        summary_rows += (
                            f'<tr><td style="padding:4px 8px;border-bottom:1px solid #21262d;color:#e2e8f0;">{html.escape(title)}</td>'
                            f'<td style="padding:4px 8px;border-bottom:1px solid #21262d;"><span class="severity-{severity}">{severity}</span></td>'
                            f'<td style="padding:4px 8px;border-bottom:1px solid #21262d;color:#8b949e;">{rc_text}</td>'
                            f'<td style="padding:4px 8px;border-bottom:1px solid #21262d;color:#c9d1d9;text-align:center;">{len(dl)}</td>'
                            f'<td style="padding:4px 8px;border-bottom:1px solid #21262d;color:#c9d1d9;text-align:center;">{len(actions)}</td></tr>'
                        )
                _complete_step(ps)
                yield _render_pipeline_flow()

            # First pass: process all known scenarios (use pre-populated steps)
            for chunk in _process_scenarios_cycle(scenario_list, "Initial Scan", _prebuilt_steps):
                yield chunk
            # If pipeline is blocked after initial scan, wait
            if _blocked_scenarios:
                _diag("monitor_blocked", phase="initial_scan", scenarios=list(_blocked_scenarios.keys()))
            while _blocked_scenarios and not _stop_monitoring_requested:
                yield _render_pipeline_flow() + _render_blocked_banner()
                time.sleep(2)
            if _stop_monitoring_requested:
                return

            # Polling loop: run full scenario processing each cycle
            while not _stop_monitoring_requested:
                poll_cycle += 1
                # Show a pending step that becomes "complete" after processing
                poll_step = _add_step(f"Poll Cycle #{poll_cycle}", "Running...", "running")
                yield _render_pipeline_flow()
                for chunk in _process_scenarios_cycle(scenario_list, f"Cycle #{poll_cycle}"):
                    yield chunk
                # If pipeline is blocked, wait until all blocks are cleared
                if _blocked_scenarios:
                    _diag("monitor_blocked", phase=f"cycle_{poll_cycle}", scenarios=list(_blocked_scenarios.keys()))
                while _blocked_scenarios and not _stop_monitoring_requested:
                    yield _render_pipeline_flow() + _render_blocked_banner()
                    time.sleep(2)
                if _stop_monitoring_requested:
                    break
                _complete_step(poll_step, "completed")
                yield _render_pipeline_flow()

                # Countdown until next cycle
                remaining = MONITOR_POLL_INTERVAL
                while remaining > 0 and not _stop_monitoring_requested:
                    poll_step["desc"] = f"Next scan in {remaining}s"
                    yield _render_pipeline_flow()
                    time.sleep(1)
                    remaining -= 1
                poll_step["desc"] = f"Next scan in 0s"
                yield _render_pipeline_flow()

            # Final report
            pending = len([a for a in _pending_approvals if a.get("status") == "pending"])
            _add_step("Queue Approvals", f"{pending} pending human review", "running")
            _complete_step(_pipeline_run["steps"][-1], "warning" if pending else "completed")

            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            final_output = (
                f'<div class="glass-card" style="margin-top:16px;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
                f'<span style="font-size:1.05rem;font-weight:700;color:#e2e8f0;">Continuous Monitoring Report</span>'
                f'<span style="font-size:0.75rem;color:#64748b;">({poll_cycle} poll cycles)</span>'
                f'</div>'
                f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px;">'
                f'<div class="metric-card"><div class="metric-value">{total_scenarios}</div><div class="metric-label">Scenarios</div></div>'
                f'<div class="metric-card"><div class="metric-value">{processed}</div><div class="metric-label">Processed</div></div>'
                f'<div class="metric-card"><div class="metric-value">{failures}</div><div class="metric-label">Failures</div></div>'
                f'<div class="metric-card"><div class="metric-value">{total_anomalies}</div><div class="metric-label">Anomalies</div></div>'
                f'<div class="metric-card"><div class="metric-value">{total_actions}</div><div class="metric-label">Actions</div></div>'
                f'</div>'
                f'<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">'
                f'<thead><tr style="border-bottom:2px solid #30363d;">'
                f'<th style="padding:6px 8px;text-align:left;color:#8b949e;">Incident</th>'
                f'<th style="padding:6px 8px;text-align:left;color:#8b949e;">Severity</th>'
                f'<th style="padding:6px 8px;text-align:left;color:#8b949e;">Root Cause</th>'
                f'<th style="padding:6px 8px;text-align:center;color:#8b949e;">Anomalies</th>'
                f'<th style="padding:6px 8px;text-align:center;color:#8b949e;">Actions</th>'
                f'</tr></thead><tbody>{summary_rows}</tbody></table>'
                f'<div style="color:#8b949e;font-size:0.72rem;margin-top:12px;text-align:right;">Monitoring stopped at {now_ts}</div>'
                f'</div>'
            )

            _pipeline_run["status"] = "completed"
            yield _render_pipeline_flow() + final_output
        except Exception as exc:
            logger.error("Continuous monitoring failed: %s", exc, exc_info=True)
            _pipeline_run.clear()
            _pipeline_run.update({
                "name": "Continuous Monitoring",
                "status": "failed",
                "steps": [{"name": "Monitoring Error", "desc": f"Failed: {exc}", "status": "failed", "progress": 100, "duration": 0, "start": 0}],
                "start_time": time.time(),
                "elapsed": 0,
            })
            yield _render_pipeline_flow() + f'<div style="color:red;">Monitoring failed: {exc}</div>'
        finally:
            _monitoring_active = False
            _stop_monitoring_requested = False

    def _run_optimize() -> str:
        """Run LoRA fine-tuning on approved experiences."""
        try:
            store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experience_store.json")
            if not os.path.exists(store_path):
                return f'<div style="color:{_C["amber"]};">No experience store found. Approve some actions first.</div>'
            with open(store_path) as f:
                data = json.load(f)
            experiences = data.get("experiences", [])
            approved = [e for e in experiences if e.get("verdict") == "approved"]
            if len(approved) < 3:
                return (f'<div style="color:{_C["amber"]};">Only {len(approved)} approved experiences '
                        '(need at least 3 for meaningful training). Keep approving actions.</div>')

            # Run optimize.py as a subprocess
            opt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimize.py")
            import subprocess
            result = subprocess.run(
                [sys.executable, opt_path, "--adapter-name", "remediation"],
                capture_output=True, text=True, timeout=600,
            )
            output = result.stdout + "\n" + result.stderr
            logger.info("Optimize output:\n%s", output)

            if result.returncode == 0:
                return (
                    f'<div style="padding:12px;background:rgba(0,255,136,0.04);border:1px solid rgba(0,255,136,0.15);'
                    f'border-radius:8px;">'
                    f'<div style="color:{_C["green"]};font-weight:600;">Agent optimized successfully</div>'
                    f'<div style="font-size:0.78rem;color:#8b949e;margin-top:4px;">'
                    f'Trained on {len(approved)} approved experiences. '
                    f'Adapter saved to adapters/remediation/</div>'
                    f'<pre style="font-size:0.7rem;color:#8b949e;margin-top:8px;max-height:200px;overflow:auto;">'
                    f'{output[:500]}</pre></div>'
                )
            else:
                return (f'<div style="color:{_C["red"]};">Training failed</div>'
                        f'<pre style="font-size:0.7rem;color:#8b949e;margin-top:4px;max-height:200px;overflow:auto;">'
                        f'{output[:500]}</pre>')
        except subprocess.TimeoutExpired:
            return f'<div style="color:{_C["red"]};">Training timed out after 10 minutes.</div>'
        except Exception as exc:
            logger.error("Optimize failed: %s", exc)
            return f'<div style="color:{_C["red"]};">Optimization failed: {exc}</div>'

    def _render_approval_history() -> str:
        """Render approval history as HTML."""
        global _approval_history
        if not _approval_history:
            return ""
        rows = "".join(
            f'<tr>'
            f'<td style="white-space:nowrap;font-size:0.78rem;color:#8b949e;">{h.get("id","")}</td>'
            f'<td style="white-space:nowrap;font-size:0.78rem;color:#8b949e;">C{str(h.get("cycle",0))}</td>'
            f'<td style="white-space:nowrap;font-size:0.78rem;">{h.get("scenario","")[:20]}</td>'
            f'<td style="font-size:0.78rem;">{h.get("title","")}</td>'
            f'<td style="font-size:0.78rem;"><span style="color:{"#00FF88" if h.get("action") in ("approved","auto-approved") else "#FF3B3B"};">{h.get("action","")}</span></td>'
            f'<td style="font-size:0.78rem;color:#8b949e;">{h.get("reason","—")[:40]}</td>'
            f'<td style="font-size:0.78rem;color:#8b949e;white-space:nowrap;">{h.get("timestamp","")[:19].replace("T"," ")}</td>'
            f'<td style="font-size:0.78rem;" title="{html.escape(h.get("execution_detail",""))}">'
            f'<span style="color:{"#00FF88" if h.get("execution") in ("success","skipped") else "#FF3B3B"};">{h.get("execution","—")}</span></td>'
            f'</tr>'
            for h in reversed(_approval_history)
        )
        return (
            '<div style="margin-top:16px;">'
            '<div style="font-size:0.8rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
            f'Approval History ({len(_approval_history)})</div>'
            '<table class="styled-table">'
            '<thead><tr><th>ID</th><th>Cycle</th><th>Scenario</th><th>Action</th><th>Verdict</th><th>Reason</th><th>Timestamp</th><th>Execution</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )

    # ═══════════════════════════════════════════════════════════════
    #  MODEL / THINKING HELPERS
    # ═══════════════════════════════════════════════════════════════

    def _get_model_label(model_id: str) -> str:
        info = MODEL_REGISTRY.get(model_id, {})
        return info.get("label", model_id.split("/")[-1])

    def _extract_think(raw: str) -> tuple:
        """Extract thinking trace and clean answer from model output.
        Returns (thinking_trace, clean_answer)."""
        raw = raw.strip()
        for open_tag, close_tag in THINKING_TAGS:
            if open_tag in raw and close_tag in raw:
                try:
                    start = raw.index(open_tag) + len(open_tag)
                    end = raw.index(close_tag)
                    thinking = raw[start:end].strip()
                    answer = raw[end + len(close_tag):].strip()
                    return thinking, answer
                except ValueError:
                    continue
        return "", raw

    def _run_risk_assessment():
        """Generate risk & security summary from pipeline state."""
        tri = _last_pipeline_state.get("triage", {})
        rca = _last_pipeline_state.get("rca", {})
        remed = _last_pipeline_state.get("remediation", {})
        actions = remed.get("recommended_actions", [])
        rca_cat = rca.get("root_cause_category", "")
        security_related = rca_cat in ("security", "network")

        risk_items = []
        risk_levels = {"low": 0, "medium": 0, "high": 0}
        for a in actions:
            rl = a.get("risk_level", "medium")
            risk_levels[rl] = risk_levels.get(rl, 0) + 1
            risk_items.append({
                "action": a.get("tool_name", "?"),
                "risk": rl,
                "rationale": a.get("rationale", "")[:80],
            })

        return {
            "security_incident": security_related,
            "risk_levels": risk_levels,
            "total_actions": len(actions),
            "risk_items": risk_items,
            "severity": tri.get("severity", "?"),
            "sla_minutes": SEVERITY_LEVELS.get(tri.get("severity", "P3"), {}).get("sla_minutes", 240),
            "recommendation": _generate_risk_recommendation(tri, rca, risk_levels, security_related),
        }

    def _generate_risk_recommendation(tri, rca, risk_levels, security_related) -> str:
        sev = tri.get("severity", "P3")
        if sev in ("P1", "P2"):
            base = "**Critical incident** — immediate action required."
        else:
            base = "**Non-critical** — standard response timeline applies."

        if security_related:
            base += " 🔒 **Security incident** — isolate affected systems and preserve forensic evidence."
        if risk_levels.get("high", 0) > 0:
            base += f" ⚠ **{risk_levels['high']} high-risk action(s)** require manual approval."
        if rca.get("confidence_score", 0) < 0.5:
            base += " ⚠ **Low confidence RCA** — verify root cause before executing remediation."
        return base

    # ═══════════════════════════════════════════════════════════════
    #  MULTI-TURN CHAT
    # ═══════════════════════════════════════════════════════════════

    CHAT_SYSTEM_PROMPT = """You are InfraHeal AI, an autonomous incident diagnosis agent. You just analyzed an infrastructure incident. Answer questions about your analysis, reasoning, and decisions. Be concise and technical. If asked "why", explain your reasoning chain step by step. If asked "re-analyze", acknowledge and suggest running a new analysis."""

    def _chat_respond(message: str, history: list, model_id: str = "") -> str:
        # Handle approval commands in chat
        msg_lower = message.strip().lower()
        if msg_lower.startswith("approve ") or msg_lower == "approve all":
            parts = msg_lower.split()
            if len(parts) >= 2 and parts[1] == "all":
                approved = 0
                for a in _pending_approvals:
                    if a.get("status") == "pending":
                        _approve_action(a["id"])
                        approved += 1
                return f"Approved **{approved}** pending action(s). They will be executed."
            elif len(parts) >= 2:
                aid = parts[1].upper()
                for a in _pending_approvals:
                    if a.get("id") == aid and a.get("status") == "pending":
                        _approve_action(aid)
                        return f"Action **{aid}** approved and executed."
                return f"Action **{aid}** not found or already resolved."
        if msg_lower.startswith("deny ") or msg_lower.startswith("deny all"):
            parts = msg_lower.split()
            reason_start = message.find("because ")
            reason = message[reason_start + 8:].strip() if reason_start >= 0 else "No reason provided"
            if len(parts) >= 2 and parts[1] == "all":
                denied = 0
                for a in _pending_approvals:
                    if a.get("status") == "pending":
                        _deny_action(a["id"], reason)
                        denied += 1
                return f"Denied **{denied}** pending action(s). Reason: {reason}"
            elif len(parts) >= 2:
                aid = parts[1].upper()
                for a in _pending_approvals:
                    if a.get("id") == aid and a.get("status") == "pending":
                        _deny_action(aid, reason)
                        return f"Action **{aid}** denied. Reason: {reason}"
                return f"Action **{aid}** not found or already resolved."

        # ── normal LLM chat response below ──
        if not _last_pipeline_state.get("triage"):
            return "**No analysis data yet.**\n\nRun an incident analysis first from the **Incident Analysis** tab, then I can answer questions about it."

        tri = _last_pipeline_state.get("triage", {})
        rca = _last_pipeline_state.get("rca", {})
        remed = _last_pipeline_state.get("remediation", {})
        crit = _last_pipeline_state.get("critique", {})

        if not model_id or model_id not in MODEL_REGISTRY:
            model_id = MODEL_NAME
        model_info = MODEL_REGISTRY.get(model_id, {})
        has_think = model_info.get("has_thinking", False)
        model_max = model_info.get("max_tokens", 512)

        ctx = (
            f"severity={tri.get('severity','?')} category={tri.get('category','?')} "
            f"impact={tri.get('impact_assessment','?')[:120]} "
            f"root_cause={rca.get('root_cause','?')[:200]} "
            f"confidence={rca.get('confidence_score',0):.0%} "
            f"actions={len(remed.get('recommended_actions',[]))} "
            f"critique_confirmed={crit.get('confirmed',True)}"
        )

        if orchestrator is not None:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
                history_msgs = []
                for h in history[-6:]:
                    if isinstance(h, dict):
                        history_msgs.append({"role": h.get("role","user"), "content": h.get("content","")})
                    elif isinstance(h, (list, tuple)) and len(h) == 2:
                        history_msgs.append({"role": "user", "content": str(h[0])})
                        history_msgs.append({"role": "assistant", "content": str(h[1])})
                history_msgs.append({"role": "user", "content": f"Current incident: {ctx}\n\nQuestion: {message}\n\nUse clean markdown: start with a direct answer, use ## headers for sections, **bold** key terms, tables for data, and `code` for metrics."})

                system = "You are InfraHeal AI, an autonomous incident diagnosis agent running on AMD ROCm + vLLM. Follow these formatting rules strictly:\n- Use **bold** for key terms and important phrases\n- Use ## or ### headers to separate sections\n- Use bullet points or numbered lists instead of long paragraphs\n- Use markdown tables for data, comparisons, or lists with multiple traits\n- Use `code` for commands, metrics, and technical values\n- Start with a direct 1-2 sentence answer\n- Never repeat or restate the user's question\n- Keep responses under 250 words"
                past = _past_incidents_summary()
                if past:
                    system += (
                        "\n\nYou have access to a full database of past incidents. "
                        "When asked about historical incidents, overall counts, trends across "
                        "all incidents, or comparisons — use the data below.\n" + past
                    )
                if has_think:
                    system += " Think step by step before answering. Show your reasoning clearly."

                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system},
                    ] + history_msgs,
                    max_tokens=model_max,
                    temperature=0.3,
                )
                content = resp.choices[0].message.content or "I don't have a specific answer."

                if has_think:
                    thinking, clean = _extract_think(content)
                    if thinking:
                        return (
                            f"**Thinking Trace**\n\n```\n{thinking}\n```\n\n---\n\n{clean}"
                        )
                return content
            except Exception as exc:
                logger.warning("Chat LLM failed: %s", exc)

        return (
            f"**Incident Summary**\n\n"
            f"**Severity:** {tri.get('severity','?')}  **Category:** {tri.get('category','?')}  "
            f"**Confidence:** {rca.get('confidence_score',0):.0%}  "
            f"**Actions:** {len(remed.get('recommended_actions',[]))}\n\n"
            f"**Root cause:** {rca.get('root_cause','unknown')}\n\n"
            f"{len(remed.get('recommended_actions',[]))} remediation actions. "
            f"{'Critique confirmed.' if crit.get('confirmed',True) else 'Critique found gaps.'}"
        )

    def _fm(text: str) -> str:
        """Convert _mhl-style markers to HTML (bold, italic, code, newlines, tables)."""
        import re as _re
        t = _mhl(text)
        # Strip markdown/code fences: ```markdown ... ```, ```code ... ```, triple backticks
        t = _re.sub(r'```(?:markdown|code|)\s*\n?(.*?)\n?```', r'\1', t, flags=_re.DOTALL)
        # Convert markdown tables to HTML tables
        lines = t.split('\n')
        converted = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith('|') and line.count('|') > 2:
                table_rows = []
                while i < len(lines) and lines[i].strip().startswith('|') and lines[i].count('|') > 2:
                    table_rows.append(lines[i].strip())
                    i += 1
                if len(table_rows) >= 2:
                    header_cells = [c.strip() for c in table_rows[0].split('|') if c.strip()]
                    data_rows = table_rows[2:]  # skip separator row (row 1)
                    if data_rows:
                        html = '<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:0.82rem;">'
                        html += '<thead><tr>' + ''.join(f'<th style="padding:6px 8px;border:1px solid #30363d;color:#8b949e;text-align:left;">{h}</th>' for h in header_cells) + '</tr></thead>'
                        html += '<tbody>'
                        for row in data_rows:
                            cells = [c.strip() for c in row.split('|') if c.strip()]
                            html += '<tr>' + ''.join(f'<td style="padding:6px 8px;border:1px solid #30363d;color:#c9d1d9;">{c}</td>' for c in cells) + '</tr>'
                        html += '</tbody></table>'
                        converted.append(html)
                        continue
                else:
                    converted.append(line)
            else:
                converted.append(line)
                i += 1
        t = '\n'.join(converted)
        t = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
        t = _re.sub(r'\*(.+?)\*', r'<em>\1</em>', t)
        t = _re.sub(r'`(.+?)`', r'<code>\1</code>', t)
        t = t.replace('\n', '<br>')
        return t

    def _render_chat_html(messages: list) -> str:
        if not messages:
            return '<div style="padding:20px;text-align:center;color:#8b949e;font-size:0.85rem;">No messages yet.</div>'
        items = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted = _fm(content)
            is_user = role == "user"
            items.append(
                f'<div class="chat-msg {role}" data-idx="{i}">'
                f'<div class="chat-bubble">{formatted}</div>'
                f'<button class="chat-copy-btn" onclick="copyChatMsg(this)">Copy</button>'
                f'</div>'
            )
        return f'<div id="infraheal-chat" class="chat-container">{"".join(items)}</div>'

    # ═══════════════════════════════════════════════════════════════
    #  BUILD THE UI
    # ═══════════════════════════════════════════════════════════════
    _theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e0f7ff", c100="#b3ecff", c200="#80dfff",
            c300="#4dd2ff", c400="#1ac5ff", c500="#00b8f0",
            c600="#009dd4", c700="#0082b8", c800="#00669c",
            c900="#004b80", c950="#003366",
        ),
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.Color(
            c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0",
            c300="#cbd5e1", c400="#64748b", c500="#64748b",
            c600="#475569", c700="#334155", c800="#1e293b",
            c900="#111128", c950="#0a0a1a",
        ),
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )
    with gr.Blocks(title="InfraHeal AI — Autonomous Incident Resolution", css=CUSTOM_CSS, theme=_theme, head=HEAD_HTML) as demo:
        try:
            demo.queue()
        except Exception:
            pass

        # ──────────────────────────────────────────────────────────
        #  TAB 1 — COMMAND CENTER
        # ──────────────────────────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("Command Center"):
                header = gr.HTML(value=_branding_header)
                metrics_row = gr.HTML(value=_get_command_center_metrics)

                gr.HTML(
                    '<div class="section-label" style="margin-top:16px;">Live Log Stream</div>'
                )

                # Hidden JS bridge textbox
                log_stream = gr.HTML(value=_get_command_center_logs)
                live_timer = gr.Timer(value=3.0, active=True)
                live_timer.tick(fn=_render_live_logs, inputs=[], outputs=[log_stream])

                gr.HTML('<div style="height:12px;"></div>')
                # ── Action buttons with rerun ──
                with gr.Row():
                    btn_scan = gr.Button("Run Anomaly Scan", variant="secondary", scale=1)
                    btn_scan_rerun = gr.Button("\u21bb", scale=0, elem_classes="rerun-btn", elem_id="rerun-scan")
                    btn_process = gr.Button("Process All Incidents", variant="secondary", scale=1)
                    btn_process_rerun = gr.Button("\u21bb", scale=0, elem_classes="rerun-btn", elem_id="rerun-process")
                    btn_report = gr.Button("Generate Report", variant="secondary", scale=1)
                    btn_report_rerun = gr.Button("\u21bb", scale=0, elem_classes="rerun-btn", elem_id="rerun-report")
                with gr.Row(variant="compact", equal_height=True):
                    btn_monitor = gr.Button("Start Continuous Process", variant="secondary", scale=4)
                    btn_monitor_rerun = gr.Button("\u21bb", scale=0, elem_classes="rerun-btn", elem_id="rerun-monitor")
                    btn_stop_monitor = gr.Button("\u25a0", scale=0, elem_classes="stop-btn", elem_id="stop-monitor", visible=True)
                    drp_poll_interval = gr.Dropdown(
                        choices=[("1 min", 60), ("2 min", 120), ("3 min", 180),
                                 ("5 min", 300), ("10 min", 600), ("30 min", 1800), ("1 hr", 3600)],
                        value=60, label="", scale=2, min_width=100,
                        elem_id="poll-interval", show_label=False, container=False
                    )
                    btn_optimize = gr.Button("Optimize Agent", variant="secondary", scale=4, elem_id="btn-optimize")
                    btn_optimize_rerun = gr.Button("\u21bb", scale=0, elem_classes="rerun-btn", elem_id="rerun-optimize")
                with gr.Row(variant="compact", equal_height=True):
                    cmd_auto_approve = gr.Checkbox(
                        value=False,
                        label="Auto-approve actions",
                        elem_id="auto-approve-chk",
                        scale=1,
                    )
                    def _toggle_auto_approve(v: bool):
                        global _auto_approve
                        _auto_approve = v
                        logger.info("Auto-approve %s", "enabled" if v else "disabled")
                    cmd_auto_approve.change(fn=_toggle_auto_approve, inputs=[cmd_auto_approve], outputs=[])

                scan_output = gr.HTML(
                    value=_empty_state("Anomaly scan results will appear here",
                                       "Click 'Run Anomaly Scan' to start."),
                    elem_id="scan-output"
                )

                # ── Gradio-native timer: polls pipeline progress every 1s ──
                # The timer fires a server-side event that calls _poll_live_html
                # and updates scan_output through Gradio's normal event queue.
                # Combined with a lightweight iframe JS that only updates timer
                # displays (.pipeline-timer / .step-timer) — no /live-html fetch,
                # no scan-output alteration, eliminating the race condition that
                # plagued the old full-HTML-replacement iframe.
                _live_poll_timer = gr.Timer(value=1.0, active=True)
                _live_poll_timer.tick(fn=_poll_live_html, inputs=[], outputs=[scan_output])

                # ── Minimal iframe: real-time timer display only ──
                # No fetch('/live-html'), no scan-output.innerHTML modification.
                # Safely co-exists with the Gradio-native timer above.
                _TIMER_JS = "<scri" + "pt>\n" + """
setInterval(function(){
  var d=parent.document,pt=d.querySelector('.pipeline-timer');
  if(pt&&pt.dataset.status!=='completed'&&pt.dataset.start){var n=Date.now()/1e3,e=Math.max(0,Math.floor(n-parseFloat(pt.dataset.start)));pt.textContent=String(Math.floor(e/60)).padStart(2,'0')+':'+String(e%60).padStart(2,'0');}
  d.querySelectorAll('.step-timer').forEach(function(e){if(e.dataset.status==='completed'||!e.dataset.start)return;var n=Date.now()/1e3,t=Math.max(0,Math.floor(n-parseFloat(e.dataset.start)));e.textContent=String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0');});
  d.querySelectorAll('.pipeline-step').forEach(function(s){s.onclick=function(){this.classList.toggle('collapsed');var g=this.parentElement;if(g&&g.classList.contains('pipeline-cycle-group'))g.classList.toggle('collapsed');};});
  /* ── Approvals tab notification badge ── */
  var pc=d.getElementById('pending-count');
  if(pc){var cnt=parseInt(pc.getAttribute('data-count')||'0');
  var tabs=d.querySelectorAll('.tabs > .tab-nav > button');
  for(var i=0;i<tabs.length;i++){var b=tabs[i];
  if(b.textContent.trim()==='Approvals'){var bg=b.querySelector('.tab-badge');
  if(!bg){bg=d.createElement('span');bg.className='tab-badge';b.appendChild(bg);}
  bg.textContent=cnt>0?cnt:'';bg.className='tab-badge'+(cnt>0?' has-pending':' all-clear');break;}}}
},1000);
""" + "</scri" + "pt>"
                gr.HTML(value='<iframe srcdoc="' + _TIMER_JS.replace('"', '&quot;') + '" style="width:0;height:0;border:none;display:none"></iframe>')

                # ── Rerun-aware wrappers ──
                def _cached_scan():
                    global _static_output_active
                    _diag("cached_scan", cached="report" in _result_cache, static=_static_output_active)
                    _static_output_active = True
                    if "anomaly_scan" in _result_cache:
                        return _result_cache["anomaly_scan"]
                    r = _run_anomaly_scan()
                    _result_cache["anomaly_scan"] = r
                    return r

                def _rerun_scan():
                    global _static_output_active
                    _diag("rerun_scan", static=_static_output_active)
                    _static_output_active = True
                    r = _run_anomaly_scan()
                    _result_cache["anomaly_scan"] = r
                    return r

                def _cached_report():
                    global _static_output_active
                    _diag("cached_report", cached="report" in _result_cache, static=_static_output_active)
                    _static_output_active = True
                    if "report" in _result_cache:
                        return _result_cache["report"]
                    r = _generate_report()
                    _result_cache["report"] = r
                    return r

                def _rerun_report():
                    global _static_output_active
                    _diag("rerun_report", static=_static_output_active)
                    _static_output_active = True
                    r = _generate_report()
                    _result_cache["report"] = r
                    return r

                def _cached_optimize():
                    global _static_output_active
                    _diag("cached_optimize", cached="optimize" in _result_cache, static=_static_output_active)
                    _static_output_active = True
                    if "optimize" in _result_cache:
                        return _result_cache["optimize"]
                    r = _run_optimize()
                    _result_cache["optimize"] = r
                    return r

                def _rerun_optimize():
                    global _static_output_active
                    _diag("rerun_optimize", static=_static_output_active)
                    _static_output_active = True
                    r = _run_optimize()
                    _result_cache["optimize"] = r
                    return r

                def _stop_monitoring():
                    global _stop_monitoring_requested
                    _diag("stop_monitoring")
                    _stop_monitoring_requested = True
                    return '<div style="color:orange;">Stop requested — waiting for current cycle to finish...</div>'

                # ── Wire main buttons (show cached) and rerun buttons (force fresh) ──
                for btn, fn in [(btn_scan, _cached_scan), (btn_report, _cached_report), (btn_optimize, _cached_optimize)]:
                    btn.click(fn=fn, inputs=[], outputs=[scan_output])
                for btn, fn in [(btn_scan_rerun, _rerun_scan), (btn_report_rerun, _rerun_report), (btn_optimize_rerun, _rerun_optimize)]:
                    btn.click(fn=fn, inputs=[], outputs=[scan_output])

                # Process / Monitor buttons use background thread + timer polling
                # REALTIME UPDATES: hidden button for REST API polling
                _refresh_btn = gr.Button("Refresh", elem_id="refresh-btn")
                _refresh_btn.click(fn=_poll_live_html, inputs=[], outputs=[scan_output], api_name="poll_live")

                def _rerun_process():
                    global _process_completed, _process_live_html, _scenario_results, _static_output_active
                    _diag("rerun_process", completed=_process_completed, static=_static_output_active)
                    _process_completed = False
                    _process_live_html = ""
                    _result_cache.pop("report", None)
                    _scenario_results.clear()
                    _static_output_active = False
                    return _start_process()
                def _rerun_monitor():
                    global _monitoring_completed, _stop_monitoring_requested, _monitor_live_html, _scenario_results, _static_output_active
                    _diag("rerun_monitor", completed=_monitoring_completed, static=_static_output_active)
                    _stop_monitoring_requested = True
                    _monitoring_completed = False
                    _monitor_live_html = ""
                    _result_cache.pop("report", None)
                    _scenario_results.clear()
                    _static_output_active = False
                    return _start_monitor()
                btn_process.click(fn=_start_process, inputs=[], outputs=[scan_output])
                btn_monitor.click(fn=_start_monitor, inputs=[], outputs=[scan_output])
                btn_process_rerun.click(fn=_rerun_process, inputs=[], outputs=[scan_output])
                btn_monitor_rerun.click(fn=_rerun_monitor, inputs=[], outputs=[scan_output])
                btn_stop_monitor.click(fn=_stop_monitoring, inputs=[], outputs=[scan_output])
                def _set_poll_interval(v):
                    global MONITOR_POLL_INTERVAL
                    MONITOR_POLL_INTERVAL = v
                    print(f"[POLL] Interval set to {v}s", flush=True)
                drp_poll_interval.change(fn=_set_poll_interval, inputs=[drp_poll_interval], outputs=[])

            # ──────────────────────────────────────────────────────
            #  TAB 2 — INCIDENT ANALYSIS
            # ──────────────────────────────────────────────────────
            with gr.Tab("Incident Analysis"):
                gr.HTML(
                    '<div class="section-title">Incident Analysis Pipeline</div>'
                    '<div class="section-subtitle">'
                    'Select a scenario and run the full multi-agent analysis pipeline.</div>'
                )

                with gr.Row():
                    scenario_dropdown = gr.Dropdown(
                        choices=scenario_names,
                        value=scenario_names[0] if scenario_names else None,
                        label="Select Incident Scenario",
                        scale=3,
                    )
                    analyze_btn = gr.Button("Analyze Incident", variant="primary", scale=1)

                scenario_desc = gr.HTML(
                    value=_empty_state("Select a scenario", "Choose from the dropdown above.")
                )
                scenario_logs = gr.HTML(value=_empty_state("Scenario logs"))

                gr.HTML('<div class="divider"></div>')
                gr.HTML(
                    '<div class="section-label">Agent Outputs</div>'
                )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=300):
                        triage_panel = gr.HTML(
                            value=_empty_state("Triage", "Awaiting analysis…")
                        )
                    with gr.Column(scale=1, min_width=300):
                        rca_panel = gr.HTML(
                            value=_empty_state("Root Cause Analysis", "Awaiting analysis…")
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=300):
                        remed_panel = gr.HTML(
                            value=_empty_state("Remediation Plan", "Awaiting analysis…")
                        )
                    with gr.Column(scale=1, min_width=300):
                        report_panel = gr.HTML(
                            value=_empty_state("Incident Report", "Awaiting analysis…")
                        )

                with gr.Accordion("Agent Reasoning Chain", open=False):
                    reasoning_panel = gr.HTML(
                        value=_empty_state("Reasoning chain", "Run an analysis to see step-by-step reasoning.")
                    )

                gr.HTML('<div class="divider"></div>')

                # ── Error-Level Resolution Section ──────────────
                gr.HTML(
                    '<div class="section-title">Resolution by Error Level</div>'
                    '<div class="section-subtitle">'
                    'Pipeline outputs grouped by log severity — shows resolution steps '
                    'for each error level independently.</div>'
                )
                with gr.Row():
                    level_filter = gr.Dropdown(
                        choices=["ALL", "CRITICAL", "ERROR", "WARNING"],
                        value="ALL",
                        label="Filter by Log Level",
                        scale=1,
                    )
                    level_resolve_btn = gr.Button(
                        "Run Level-Specific Resolution", variant="secondary", scale=2
                    )
                level_resolution_panel = gr.HTML(
                    value=_empty_state(
                        "Level-specific resolution",
                        "Select a scenario and click 'Run Level-Specific Resolution'."
                    )
                )

                # Wire events
                scenario_dropdown.change(
                    fn=_on_scenario_selected,
                    inputs=[scenario_dropdown],
                    outputs=[scenario_desc, scenario_logs],
                )
                level_resolve_btn.click(
                    fn=_run_error_level_resolution,
                    inputs=[scenario_dropdown, level_filter],
                    outputs=[level_resolution_panel],
                )

            # ──────────────────────────────────────────────────────────
            #  TAB 3 — PERFORMANCE METRICS
            # ──────────────────────────────────────────────────────
            with gr.Tab("Performance Metrics"):
                gr.HTML(
                    '<div class="section-title">Agent Performance Dashboard</div>'
                    '<div class="section-subtitle">'
                    'Token usage, latency breakdown, and system metrics from the latest analysis run.</div>'
                )
                refresh_perf_btn = gr.Button("Refresh Metrics", variant="secondary")
                perf_output = gr.HTML(
                    value=_empty_state(
                        "No performance data yet",
                        "Run an incident analysis on the Incident Analysis tab first."
                    )
                )
                refresh_perf_btn.click(fn=_get_perf_metrics_html, inputs=[], outputs=[perf_output])

                # ── GPU Benchmark Panel ───────────────────────────
                gr.HTML('<div style="height:16px;"></div>')
                gr.HTML(
                    '<div class="section-title">ROCm GPU Benchmarking</div>'
                    '<div class="section-subtitle">'
                    'Throughput profiling for Qwen2.5-7B-Instruct on AMD ROCm. '
                    'Measures tokens/sec across batch sizes and prompt lengths.</div>'
                )
                with gr.Row():
                    tune_btn = gr.Button("Run GPU Benchmark", variant="primary", scale=1)
                    tune_status = gr.HTML(value=_empty_state("Benchmark not run", "Click to profile GPU throughput."), scale=3)
                tune_config = gr.HTML(value=_empty_state("Optimal config", "Run benchmark to get recommendations."))
                _tune_empty_fig = go.Figure()
                _tune_empty_fig.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
                                              xaxis=dict(visible=False), yaxis=dict(visible=False), height=350)
                tune_plot = gr.Plot(value=_tune_empty_fig, show_label=False)

                def _on_tune():
                    from gpu_autotuner import GPUTuner
                    tuner = GPUTuner(client=None, model_name=MODEL_NAME.split("/")[-1])
                    summary = tuner.benchmark()
                    bc = summary.get("best_config", {})
                    rec = summary.get("recommendation", {})
                    best_tok = summary.get("best_tokens_per_sec", 0)
                    avg_tok = summary.get("avg_tokens_per_sec", 0)

                    curve_data = tuner.get_benchmark_curve()
                    fig = go.Figure()
                    colors = ["rgba(255,255,255,0.8)", "rgba(255,255,255,0.55)", "rgba(255,255,255,0.35)", "rgba(255,255,255,0.2)"]
                    for idx, curve in enumerate(curve_data.get("curves", [])):
                        plen = curve["prompt_length"]
                        fig.add_trace(go.Scatter(
                            x=curve_data["batch_sizes"],
                            y=curve["tokens_per_sec"],
                            mode="lines+markers",
                            name=f"Prompt {plen}",
                            line=dict(color=colors[idx % len(colors)], width=2),
                            marker=dict(size=8),
                        ))
                    fig.update_layout(
                        paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
                        font=dict(color="#e2e8f0"), height=350,
                        title=dict(text="Throughput: Tokens/sec vs Batch Size", x=0.5, font=dict(color="#e2e8f0", size=13)),
                        xaxis=dict(title="Batch Size", gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(title="Tokens/sec", gridcolor="rgba(255,255,255,0.06)"),
                        legend=dict(font=dict(color="#64748b")),
                        margin=dict(l=10, r=10, t=35, b=10),
                    )

                    status_html = (
                        f'<div class="glass-card">'
                        f'<div style="display:flex;gap:20px;flex-wrap:wrap;">'
                        f'<div style="flex:1;min-width:120px;text-align:center;">'
                        f'  <div style="font-size:1.8rem;font-weight:800;color:#e2e8f0;">{avg_tok}</div>'
                        f'  <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;">Avg Tokens/s</div>'
                        f'</div>'
                        f'<div style="flex:1;min-width:120px;text-align:center;">'
                        f'  <div style="font-size:1.8rem;font-weight:800;color:#e2e8f0;">{best_tok}</div>'
                        f'  <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;">Peak Tokens/s</div>'
                        f'</div>'
                        f'<div style="flex:1;min-width:120px;text-align:center;">'
                        f'  <div style="font-size:1.8rem;font-weight:800;color:#e2e8f0;">{bc.get("batch_size","?")}</div>'
                        f'  <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;">Optimal Batch</div>'
                        f'</div>'
                        f'<div style="flex:1;min-width:120px;text-align:center;">'
                        f'  <div style="font-size:1.8rem;font-weight:800;color:#e2e8f0;">{bc.get("prompt_length","?")}</div>'
                        f'  <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;">Optimal Prompt Len</div>'
                        f'</div>'
                        f'</div></div>'
                    )
                    config_html = (
                        f'<div class="glass-card" style="margin-top:12px;">'
                        f'<div style="font-size:0.85rem;font-weight:700;color:#e2e8f0;margin-bottom:8px;">⚙️ Recommended vLLM Configuration</div>'
                        f'<div style="font-size:0.88rem;color:#e2e8f0;">'
                        f'<code>--max-model-len {rec.get("max_context_length",2048)}</code><br>'
                        f'<code>--gpu-memory-utilization 0.9</code><br>'
                        f'<code>Batch concurrency: {rec.get("batch_concurrency","?")}</code>'
                        f'</div>'
                        f'<div style="font-size:0.78rem;color:#64748b;margin-top:8px;">{rec.get("note","")}</div>'
                        f'</div>'
                    )
                    return status_html, config_html, fig

                tune_btn.click(fn=_on_tune, inputs=[], outputs=[tune_status, tune_config, tune_plot])

                # Also show model / system info
                gr.HTML(
                    f'<div class="glass-card" style="margin-top:20px;">'
                    f'<div class="section-label">System Information</div>'
                    f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
                    f'<div>'
                    f'  <div class="section-label" style="margin:0 0 2px;">Model</div>'
                    f'  <div class="text-cyan" style="font-size:0.88rem;font-weight:600;">{MODEL_NAME.split("/")[-1]}</div>'
                    f'</div>'
                    f'<div>'
                    f'  <div class="section-label" style="margin:0 0 2px;">Runtime</div>'
                    f'  <div class="text-green" style="font-size:0.88rem;font-weight:600;">vLLM + ROCm (AMD GPU)</div>'
                    f'</div>'
                    f'<div>'
                    f'  <div class="section-label" style="margin:0 0 2px;">API Endpoint</div>'
                    f'  <div style="font-size:0.88rem;font-weight:600;color:var(--amber);">{VLLM_BASE_URL}</div>'
                    f'</div>'
                    f'<div>'
                    f'  <div class="section-label" style="margin:0 0 2px;">RAG Backend</div>'
                    f'  <div style="font-size:0.88rem;font-weight:600;color:var(--text);">BM25 (rank_bm25)</div>'
                    f'</div>'
                    f'<div>'
                    f'  <div class="section-label" style="margin:0 0 2px;">Agent Framework</div>'
                    f'  <div style="font-size:0.88rem;font-weight:600;color:var(--text);">4-Agent Pipeline</div>'
                    f'</div>'
                    f'<div>'
                    f'  <div class="section-label" style="margin:0 0 2px;">Hackathon</div>'
                    f'  <div style="font-size:0.88rem;color:var(--text);font-weight:600;">TCS &amp; AMD Build AI 2026</div>'
                    f'</div>'
                    f'</div></div>'
                )

            # ──────────────────────────────────────────────────────
            #  TAB 4 — APPROVALS
            # ──────────────────────────────────────────────────────
            with gr.Tab("Approvals") as appr_tab:
                gr.HTML(
                    '<div class="section-title">Human Approval Center</div>'
                    '<div class="section-subtitle">'
                    'Review, approve, or deny remediation actions queued by the agent pipeline.</div>'
                )

                appr_panel = gr.HTML(value=_render_approval_panel())

                def _safe_refresh_approvals():
                    try:
                        return _render_approval_panel(), _render_approval_history(), _render_audit_log(), _refresh_approval_selector()
                    except Exception as exc:
                        _diag("refresh_approvals_error", exc=str(exc))
                        fallback = '<div style="color:#64748b;text-align:center;padding:16px;">Error loading approvals.</div>'
                        return fallback, "", "", gr.update(choices=[], value=None, interactive=False)

                appr_refresh_btn = gr.Button("Refresh", variant="secondary", size="sm", scale=0)

                def _refresh_approval_selector():
                    pending = [a for a in _pending_approvals if a.get("status") == "pending"]
                    if not pending:
                        return gr.update(choices=[], value=None, interactive=False)
                    # Flat strings (value=label) — some Gradio versions flatten tuples
                    choices = [f'{a["id"]}|{a.get("title","?")} ({a.get("risk","?")})' for a in pending]
                    return gr.update(choices=choices, value=None, interactive=True)

                appr_approval_selector = gr.Dropdown(
                    choices=[], label="Select Action to Review",
                    interactive=False,
                    scale=3,
                )
                with gr.Row():
                    appr_reason = gr.Textbox(
                        label="Reason (required for deny, optional for approve)",
                        placeholder="e.g. Action approved, proceed | Action not needed, already resolved",
                        scale=3, container=True,
                    )
                with gr.Row():
                    appr_btn_approve = gr.Button("Approve Selected", variant="secondary", scale=1)
                    appr_btn_deny = gr.Button("Deny Selected", variant="secondary", scale=1)
                with gr.Row():
                    appr_btn_approve_all = gr.Button("Approve All Pending", variant="secondary", scale=1)
                    appr_btn_deny_all = gr.Button("Deny All Pending", variant="secondary", scale=1)

                appr_status = gr.HTML(value="")
                appr_history_panel = gr.HTML(value=_render_approval_history())
                appr_audit_panel = gr.HTML(value=_render_audit_log())

                # Auto-refresh approval components when user navigates to this tab
                appr_tab.select(
                    fn=_safe_refresh_approvals,
                    inputs=[], outputs=[appr_panel, appr_history_panel, appr_audit_panel, appr_approval_selector],
                )

                # Hidden element: drives tab badge via JS (updated by timer below)
                appr_pending_count = gr.HTML(
                    value='<span id="pending-count" data-count="0"></span>',
                    visible=False,
                )

                # Safe wrapper for auto-refresh timer
                def _safe_approval_panel():
                    try:
                        return _render_approval_panel()
                    except Exception as exc:
                        _diag("auto_refresh_panel_error", exc=str(exc))
                        return '<div style="color:#64748b;text-align:center;padding:16px;">Approval panel temporarily unavailable.</div>'

                # Auto-refresh badge count + panel every 5s so the tab dot and content stay accurate
                def _poll_pending_count():
                    try:
                        cnt = len([a for a in _pending_approvals if a.get("status") == "pending"])
                        return f'<span id="pending-count" data-count="{cnt}"></span>'
                    except Exception as exc:
                        _diag("poll_pending_count_error", exc=str(exc))
                        return '<span id="pending-count" data-count="0"></span>'

                appr_badge_timer = gr.Timer(value=5.0, active=True)
                appr_badge_timer.tick(fn=_poll_pending_count, inputs=[], outputs=[appr_pending_count])

                # Auto-refresh the approval panel itself
                appr_auto_refresh = gr.Timer(value=5.0, active=True)
                appr_auto_refresh.tick(
                    fn=lambda: _safe_approval_panel(),
                    inputs=[], outputs=[appr_panel],
                )

                def _render_action_log(a: dict, action_label: str, color: str) -> str:
                    # Build structured steps collapsible
                    steps_html = ""
                    steps = a.get("execution_steps", [])
                    if steps:
                        step_items = "".join(
                            f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:0.76rem;">'
                            f'<span style="color:{"#00FF88" if s.get("status")=="completed" or s.get("status")=="success" else "#FF3B3B" if s.get("status")=="failed" else "#FFB800"};">'
                            f'{"✓" if s.get("status")=="completed" or s.get("status")=="success" else "✗" if s.get("status")=="failed" else "⏳"}</span>'
                            f'<span style="color:#e2e8f0;">{html.escape(s.get("title","?"))}</span>'
                            f'<span style="color:#8b949e;font-size:0.70rem;">({s.get("duration",0):.1f}s)</span>'
                            f'<span style="color:#8b949e;font-size:0.72rem;margin-left:4px;">— {html.escape(s.get("description",""))}</span>'
                            f'</div>'
                            for s in steps
                        )
                        steps_html = (
                            f'<details style="margin-top:6px;">'
                            f'<summary style="cursor:pointer;font-size:0.76rem;color:#8b949e;padding:4px 0;">'
                            f'▸ Agent Execution Steps ({len(steps)})</summary>'
                            f'<div style="padding:4px 0 4px 8px;border-left:2px solid rgba(255,255,255,0.08);margin-top:2px;">'
                            f'{step_items}</div></details>'
                        )
                    log = (
                        f'<div style="padding:12px 16px;margin:8px 0;border:1px solid rgba({color},0.3);border-radius:8px;'
                        f'background:rgba({color},0.06);font-size:0.82rem;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                        f'<span style="color:#{color};font-weight:700;">{html.escape(action_label)}</span>'
                        f'<span style="color:#8b949e;font-size:0.75rem;">{a.get("id","")} at {datetime.now().strftime("%H:%M:%S")}</span>'
                        f'</div>'
                        f'<div style="color:#e2e8f0;font-weight:600;">{html.escape(a.get("title",""))}</div>'
                        f'<div style="color:#8b949e;font-size:0.76rem;">Scenario: {html.escape(a.get("scenario","?"))}'
                        f' | Cycle #{a.get("cycle",0)}'
                        f' | Risk: <span style="color:{_risk_color(a.get("risk","medium"))};">{a.get("risk","medium")}</span></div>'
                        f'<div style="margin-top:6px;padding:6px 10px;background:rgba(255,255,255,0.03);border-radius:4px;'
                        f'font-size:0.76rem;color:#c9d1d9;white-space:pre-wrap;">'
                        f'<span style="color:#8b949e;">Reason:</span> {html.escape(a.get("reason",""))}'
                        f'</div>'
                        f'<div style="margin-top:4px;padding:6px 10px;background:rgba(255,255,255,0.03);border-radius:4px;'
                        f'font-size:0.76rem;color:#c9d1d9;white-space:pre-wrap;">'
                        f'<span style="color:#8b949e;">Execution:</span> '
                        f'<span style="color:{"#00FF88" if a.get("execution_status")=="success" else "#FF3B3B" if a.get("execution_status")=="failed" else "#FFB800"};">'
                        f'{html.escape(a.get("execution_status","—"))}</span>'
                        f' — {html.escape(a.get("execution_detail","")[:200])}'
                        f'</div>'
                        f'{steps_html}'
                        f'</div>'
                    )
                    return log

                def _extract_aid(raw: str) -> str:
                    """Extract action ID from dropdown choice string (format: 'APP-0001|title (risk)')."""
                    return raw.split("|")[0] if raw and "|" in raw else raw or ""

                def _on_approve_selected(aid: str, reason: str):
                    aid = _extract_aid(aid)
                    _diag("approve_selected_called", aid=aid, reason_len=len(reason or ""))
                    if not aid:
                        _diag("approve_selected_no_aid", aid=aid)
                        return _render_approval_panel(), _render_approval_history(), _render_audit_log(), _refresh_approval_selector(), ""
                    reason = reason.strip()
                    try:
                        _approve_action(aid, reason)
                    except Exception as exc:
                        _diag("approve_action_exception", aid=aid, exc=str(exc))
                    a = next((x for x in _pending_approvals if x.get("id") == aid), None)
                    _diag("approve_selected_found", aid=aid, found=a is not None)
                    status = _render_action_log(a, "Approved", "00FF88") if a else ""
                    return (
                        _render_approval_panel(), _render_approval_history(), _render_audit_log(),
                        _refresh_approval_selector(), status,
                    )

                def _on_deny_selected(aid: str, reason: str):
                    aid = _extract_aid(aid)
                    _diag("deny_selected_called", aid=aid, reason_len=len(reason or ""))
                    if not aid:
                        _diag("deny_selected_no_aid", aid=aid)
                        return _render_approval_panel(), _render_approval_history(), _render_audit_log(), _refresh_approval_selector(), ""
                    reason = reason.strip() or "No reason provided"
                    try:
                        _deny_action(aid, reason)
                    except Exception as exc:
                        _diag("deny_action_exception", aid=aid, exc=str(exc))
                    a = next((x for x in _pending_approvals if x.get("id") == aid), None)
                    _diag("deny_selected_found", aid=aid, found=a is not None)
                    status = _render_action_log(a, "Denied", "FF3B3B") if a else ""
                    return (
                        _render_approval_panel(), _render_approval_history(), _render_audit_log(),
                        _refresh_approval_selector(), status,
                    )

                def _on_approve_all(reason: str = ""):
                    try:
                        count = 0
                        reason = reason.strip() or "Bulk approved"
                        logs_html = ""
                        for a in _pending_approvals:
                            if a.get("status") == "pending":
                                _approve_action(a["id"], reason)
                                logs_html += _render_action_log(a, "Approved", "00FF88")
                                count += 1
                        summary = (
                            f'<div style="padding:8px 12px;margin:4px 0;background:rgba(0,255,136,0.04);'
                            f'border:1px solid rgba(0,255,136,0.2);border-radius:6px;font-size:0.82rem;">'
                            f'<span style="color:#00FF88;font-weight:700;">Approved {count} action(s)</span>'
                            f'</div>'
                        )
                        return (
                            _render_approval_panel(), _render_approval_history(), _render_audit_log(),
                            _refresh_approval_selector(), summary + logs_html,
                        )
                    except Exception as exc:
                        _diag("approve_all_error", exc=str(exc))
                        fb = _safe_refresh_approvals()
                        return fb[0], fb[1], fb[2], fb[3], '<div style="color:#FF3B3B;">Bulk approve failed.</div>'

                def _on_deny_all(reason: str):
                    try:
                        reason = reason.strip() or "No reason provided"
                        count = 0
                        logs_html = ""
                        for a in _pending_approvals:
                            if a.get("status") == "pending":
                                _deny_action(a["id"], reason)
                                logs_html += _render_action_log(a, "Denied", "FF3B3B")
                                count += 1
                        summary = (
                            f'<div style="padding:8px 12px;margin:4px 0;background:rgba(255,59,59,0.04);'
                            f'border:1px solid rgba(255,59,59,0.2);border-radius:6px;font-size:0.82rem;">'
                            f'<span style="color:#FF3B3B;font-weight:700;">Denied {count} action(s)</span>'
                            f'</div>'
                        )
                        return (
                            _render_approval_panel(), _render_approval_history(), _render_audit_log(),
                            _refresh_approval_selector(), summary + logs_html,
                        )
                    except Exception as exc:
                        _diag("deny_all_error", exc=str(exc))
                        fb = _safe_refresh_approvals()
                        return fb[0], fb[1], fb[2], fb[3], '<div style="color:#FF3B3B;">Bulk deny failed.</div>'

                outputs_approval = [
                    appr_panel, appr_history_panel, appr_audit_panel,
                    appr_approval_selector, appr_status,
                ]
                appr_btn_approve.click(fn=_on_approve_selected, inputs=[appr_approval_selector, appr_reason], outputs=outputs_approval)
                appr_btn_deny.click(fn=_on_deny_selected, inputs=[appr_approval_selector, appr_reason], outputs=outputs_approval)
                appr_btn_approve_all.click(fn=_on_approve_all, inputs=[appr_reason], outputs=outputs_approval)
                appr_btn_deny_all.click(fn=_on_deny_all, inputs=[appr_reason], outputs=outputs_approval)
                appr_refresh_btn.click(fn=_safe_refresh_approvals, inputs=[], outputs=[appr_panel, appr_history_panel, appr_audit_panel, appr_approval_selector])

            # ──────────────────────────────────────────────────────
            #  TAB 5 — VISUALIZATION
            # ──────────────────────────────────────────────────────
            with gr.Tab("Visualization"):
                gr.HTML(
                    '<div class="section-title">Metric &amp; Anomaly Visualization</div>'
                    '<div class="section-subtitle">'
                    'Interactive 2D visualizations powered by Plotly — time-series analysis, '
                    'correlation heatmaps, anomaly timelines, and log distributions.</div>'
                )
                with gr.Row():
                    viz_scenario = gr.Dropdown(
                        choices=["ALL SCENARIOS"] + scenario_names,
                        value="ALL SCENARIOS",
                        label="Select Scenario",
                        scale=3,
                    )
                    viz_refresh_btn = gr.Button("Generate Plots", variant="primary", scale=1)

                _empty_fig = go.Figure()
                _empty_fig.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
                                         xaxis=dict(visible=False), yaxis=dict(visible=False),
                                         height=400,
                                         annotations=[dict(text="Select a scenario and click Generate",
                                                           xref="paper", yref="paper", x=0.5, y=0.5,
                                                           showarrow=False,
                                                           font=dict(color="#64748b", size=14))])
                _empty_fig2 = go.Figure()
                _empty_fig2.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
                                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                                          height=400)

                viz_plot1 = gr.Plot(value=_empty_fig, show_label=False)
                viz_plot2 = gr.Plot(value=_empty_fig2, show_label=False)
                with gr.Row(equal_height=True):
                    viz_plot3 = gr.Plot(value=_empty_fig2, show_label=False)
                    viz_plot4 = gr.Plot(value=_empty_fig2, show_label=False)
                gr.HTML(
                    '<div class="section-label" style="margin-top:8px;">🌐 Topology Map</div>'
                )
                viz_plot5 = gr.Plot(value=_empty_fig2, show_label=False)

                def _on_viz_generate(name: str):
                    sc_name = name if name != "ALL SCENARIOS" else None
                    return _get_visualizations(sc_name)

                viz_refresh_btn.click(
                    fn=_on_viz_generate,
                    inputs=[viz_scenario],
                    outputs=[viz_plot1, viz_plot2, viz_plot3, viz_plot4, viz_plot5],
                )

            # ──────────────────────────────────────────────────────
            #  TAB 5 — KNOWLEDGE BASE
            # ──────────────────────────────────────────────────────
            with gr.Tab("Knowledge Base"):
                gr.HTML(
                    '<div class="section-title">Runbook Knowledge Base</div>'
                    '<div class="section-subtitle">'
                    'Search and browse operational runbooks used by the RAG pipeline.</div>'
                )

                with gr.Row():
                    kb_search = gr.Textbox(
                        placeholder="Search runbooks (e.g. 'database', 'memory', 'security')…",
                        label="Search",
                        scale=3,
                    )
                    kb_btn = gr.Button("Search", variant="primary", scale=1)

                kb_results = gr.HTML(
                    value="".join(_format_runbook_html(rb) for rb in runbooks)
                )

                kb_btn.click(fn=_search_runbooks, inputs=[kb_search], outputs=[kb_results])
                kb_search.submit(fn=_search_runbooks, inputs=[kb_search], outputs=[kb_results])

            # ──────────────────────────────────────────────────────
            #  TAB 6 — HELP & FAQ
            # ──────────────────────────────────────────────────────
            with gr.Tab("Help & FAQ"):
                gr.HTML(
                    '<div class="section-title">Help & Frequently Asked Questions</div>'
                    '<div class="section-subtitle">Quick answers to common questions about InfraHeal AI.</div>'
                )

                def _render_faq() -> str:
                    faqs = [
                        ("What is InfraHeal AI?",
                         "Autonomous incident diagnosis & resolution system built for the TCS & AMD Build AI 2026 hackathon. "
                         "Uses a <b>4-agent LLM pipeline</b> (Triage &rarr; RCA &rarr; Remediation &rarr; Report) powered by "
                         "<b>Qwen/Qwen2.5-7B-Instruct</b> on <b>AMD ROCm + vLLM</b>. Features real-time log streaming, "
                         "RAG-based knowledge retrieval, human-in-the-loop approvals, and continuous learning via LoRA fine-tuning."),
                        ("How do I run an analysis?",
                         "Go to <b>Incident Analysis</b> tab &rarr; select a scenario from the dropdown &rarr; "
                         "click <b>Analyze Incident</b>. The pipeline runs sequentially: "
                         "Triage classifies severity/category, RCA identifies root cause, Remediation generates actions, "
                         "Reporting produces a summary. Results appear in the four agent output panels below. "
                         "The <b>Agent Reasoning Chain</b> accordion shows step-by-step reasoning from each agent."),
                        ("How do I process all incidents at once?",
                         "Go to <b>Command Center</b> &rarr; click <b>Process All Incidents</b>. "
                         "This iterates over every scenario, runs the full pipeline on each, and produces "
                         "a comprehensive summary table with severity, category, root cause, and action counts. "
                         "Performance metrics are aggregated across all scenarios."),
                        ("What is the approval queue for?",
                         "High-risk remediation actions (<code>requires_approval=true</code>) are held for human review. "
                         "The <b>Approvals Required</b> accordion in Command Center lists pending actions with Approve/Deny buttons. "
                         "You can also type <b>APPROVE APP-0001</b> or <b>DENY APP-0002 because already resolved</b> "
                         "in Agent Chat. Approved actions are executed; denied ones are logged with the reason. "
                         "All decisions feed into the experience store for continuous learning."),
                        ("How do I use the Agent Chat?",
                         "Go to <b>Agent Chat</b> &rarr; type any question about the current analysis. "
                         "The bot has full context from the last pipeline run (severity, category, root cause, actions, critique). "
                         "Pre-loaded quick questions are available: <b>Why P1?</b>, <b>What is the root cause?</b>, "
                         "<b>What should I do?</b>, <b>Explain evidence</b>, <b>Re-analyze</b>. "
                         "You can also switch models from the dropdown to compare responses. "
                         "Some models support <b>thinking traces</b> shown in expandable details."),
                        ("What models are available?",
                         "Default: <b>Qwen/Qwen2.5-7B-Instruct</b> (highly optimized on AMD ROCm). "
                         "Switch models from the dropdown in Agent Chat — other models registered in "
                         "<code>MODEL_REGISTRY</code> are available if loaded on your vLLM instance. "
                         "Models with <code>has_thinking: true</code> show step-by-step reasoning traces "
                         "in expandable details before their final answer."),
                        ("How does the agent learn over time?",
                         "Three continuous learning layers:"
                         "<br><b>Layer 1 &mdash; Experience Store:</b> Every approved/denied action is logged. "
                         "Before each analysis, the top-3 most similar past successful remediations are injected "
                         "as few-shot examples in the remediation prompt."
                         "<br><b>Layer 2 &mdash; Action Preference Ranking:</b> Approval rates per tool are tracked. "
                         "The remediation agent sees historical success rates (e.g. 'restart_service: 100% approval') "
                         "biasing recommendations toward trusted actions."
                         "<br><b>Layer 3 &mdash; LoRA Fine-Tuning:</b> Click <b>Optimize Agent (LoRA)</b> in Command Center "
                         "to fine-tune Qwen2.5-7B on approved actions. Requires 3+ approved experiences. "
                         "Adapter is saved to <code>adapters/remediation/</code>."),
                        ("What is the SafetyGuard?",
                         "Every remediation action passes through <b>SafetyGuard</b> before execution — "
                         "a rule-based validator that checks security policies, tool permissions, and severity overrides. "
                         "Each action receives a verdict: <span style='color:#00FF88;'><b>allow</b></span> (safe to execute), "
                         "<span style='color:#FFB800;'><b>flag</b></span> (risky but permitted), or "
                         "<span style='color:#FF3B3B;'><b>block</b></span> (dangerous, prevented). "
                         "Results are shown in the Remediation output panel with detailed reasoning."),
                        ("What does the critique agent do?",
                         "After RCA, the critique agent reviews the root cause analysis for evidence quality. "
                         "It either <b>confirms</b> the RCA or identifies <b>gaps</b> (e.g. 'Insufficient evidence for memory critical conditions'). "
                         "When gaps are found, it refines confidence scores and suggests improvements. "
                         "Gaps are <b>informational</b> — they indicate low-confidence or sparse evidence, not system errors. "
                         "In a production deployment, these would trigger additional data collection."),
                        ("How do I search the knowledge base?",
                         "Go to <b>Knowledge Base</b> tab &rarr; type a query (e.g. 'database', 'memory', 'security') "
                         "&rarr; click <b>Search</b> or press Enter. Results show relevant operational runbooks "
                         "used by the RAG pipeline during RCA. The search uses BM25 ranking "
                         "(via <code>rank_bm25</code>) to find the most relevant entries."),
                        ("How do I view performance metrics?",
                         "Go to <b>Performance Metrics</b> tab &rarr; click <b>Refresh Metrics</b>. "
                         "Shows: total time, tokens used, LLM calls, average latency, GPU memory, "
                         "and per-agent latency/token breakdowns. "
                         "The <b>GPU Benchmark</b> panel profiles throughput (tokens/sec) across batch sizes "
                         "and prompt lengths — useful for tuning vLLM parameters. "
                         "Run benchmark to get recommended <code>--max-model-len</code> and batch concurrency settings."),
                        ("Why is token usage high in Process All Incidents?",
                         "The pipeline calls 4 agents per scenario, each sending a prompt and receiving a response. "
                         "For 8 scenarios with ~500-1000 tokens per call, 160K+ total tokens is expected. "
                         "Each scenario generates: triage (~200 tokens), RCA (~400 tokens), remediation (~300 tokens), "
                         "report (~400 tokens), plus critique and safety checks. "
                         "To reduce usage, you can lower <code>max_tokens</code> per agent in <code>config.py</code> "
                         "or reduce the number of scenarios."),
                        ("What does Continuous Monitoring do?",
                         "Click <b>Start Continuous Monitoring</b> in Command Center to run the full pipeline "
                         "across all scenarios in a single batch. High-risk actions are automatically queued "
                         "for human approval in the Approvals panel. "
                         "Can be extended to a periodic loop with <code>gr.Timer</code> for autonomous operation."),
                        ("How do I fine-tune the model?",
                         "1. Approve several actions first (at least 3) so the experience store has training data."
                         "<br>2. Click <b>Optimize Agent (LoRA)</b> in Command Center."
                         "<br>3. The script runs LoRA fine-tuning via <code>optimize.py</code> using "
                         "<code>peft</code> + <code>bitsandbytes</code> 4-bit quantization."
                         "<br>4. Adapter weights are saved to <code>adapters/remediation/</code>."
                         "<br>5. Restart vLLM with the adapter: <code>vllm serve Qwen/Qwen2.5-7B-Instruct "
                         "--enable-lora --lora-modules remediation=adapters/remediation</code>. "
                         "Subsequent analyses will use the fine-tuned adapter."),
                        ("How is the dashboard deployed?",
                         "<b>On AMD ROCm cloud (JupyterLab):</b>"
                         "<br><code>cd infraheal-ai && git fetch origin && git reset --hard origin/master</code>"
                         "<br><code>vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 "
                         "--gpu-memory-utilization 0.9 --max-model-len 8192</code>"
                         "<br><code>python dashboard.py</code>"
                         "<br><b>Local (CPU demo mode):</b> python dashboard.py runs with pre-generated demo data."),
                        ("What hardware is required?",
                         "<b>Production:</b> AMD ROCm GPU (MI250/MI300 recommended) with vLLM for inference. "
                         "Requires ROCm 6.x + PyTorch with ROCm support."
                         "<br><b>CPU demo mode:</b> Works on any machine without a GPU — uses pre-generated "
                         "sample data and simulated agent responses. Run <code>python dashboard.py</code> directly."),
                        ("How are incidents scored?",
                         "Four severity levels: <b>P1 (Critical)</b> &rarr; immediate SLA (~15min), "
                         "<b>P2 (High)</b> &rarr; urgent (~60min), <b>P3 (Medium)</b> &rarr; standard (~240min), "
                         "<b>P4 (Low)</b> &rarr; best-effort. Severity determines escalation rules, "
                         "SafetyGuard strictness, and SLA targets. Color-coded badges: "
                         "<span style='color:#FF3B3B;'>red</span> (P1), "
                         "<span style='color:#FF8C00;'>orange</span> (P2), "
                         "<span style='color:#FFD700;'>yellow</span> (P3), "
                         "<span style='color:#4CAF50;'>green</span> (P4)."),
                        ("Can I customize the available tools?",
                         "Yes — tools are registered in <code>config.py</code> under <code>AVAILABLE_TOOLS</code>. "
                         "Each tool has: <code>name</code>, <code>description</code>, and <code>parameters</code> "
                         "(name, type, description, required). The remediation agent dynamically reads this registry "
                         "and includes available tools in its system prompt. "
                         "Add, modify, or remove tools — the agent adapts automatically."),
                        ("What is the difference between Analyze and Process All?",
                         "<b>Analyze Incident</b> runs the pipeline on a single selected scenario with detailed "
                         "per-agent output panels. Best for debugging and understanding a specific incident."
                         "<br><b>Process All Incidents</b> runs on every scenario and produces a summary table. "
                         "Performance metrics are aggregated across all runs. Best for batch analysis and benchmarking."),
                        ("Why is GPU KV cache 0%?",
                         "The KV cache fills as tokens are generated during inference. "
                         "Early in a session or after idle periods, the cache is naturally empty. "
                         "It populates after a few requests. 0% is normal behavior and not a concern."),
                        ("Can this integrate with real infrastructure?",
                         "Currently uses simulated execution — actions print success messages but don't "
                         "connect to actual servers. For production: replace <code>execute_action()</code> "
                         "in <code>remediation_agent.py</code> with real API calls (Kubernetes, AWS, Ansible, etc.). "
                         "The SafetyGuard, approval queue, and logging infrastructure are production-ready."),
                    ]
                    items = "".join(
                        f'<div class="faq-item">'
                        f'<div class="faq-q" onclick="this.nextElementSibling.classList.toggle(\'open\');'
                        f'this.querySelector(\'.faq-toggle\').textContent = '
                        f'this.nextElementSibling.classList.contains(\'open\') ? \'−\' : \'+\';">'
                        f'<span class="faq-toggle">+</span> {q}</div>'
                        f'<div class="faq-a">{a}</div>'
                        f'</div>'
                        for q, a in faqs
                    )
                    return f'''
                    <style>
                    .faq-item {{ margin-bottom: 2px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
                    .faq-q {{ padding: 14px 16px; cursor: pointer; font-size: 0.88rem; font-weight: 600; color: #e2e8f0; display: flex; align-items: center; gap: 10px; user-select: none; }}
                    .faq-q:hover {{ background: rgba(255,255,255,0.02); border-radius: 8px; }}
                    .faq-toggle {{ display: inline-flex; align-items: center; justify-content: center; min-width: 24px; height: 24px; border-radius: 4px; background: rgba(255,255,255,0.06); color: #64748b; font-size: 1rem; flex-shrink: 0; }}
                    .faq-a {{ padding: 0 16px 14px 50px; font-size: 0.82rem; color: #8b949e; line-height: 1.65; display: none; }}
                    .faq-a.open {{ display: block; }}
                    .faq-a code {{ background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 4px; font-size: 0.78rem; color: #c9d1d9; }}
                    .faq-a b {{ color: #c9d1d9; }}
                    </style>
                    <div class="glass-card">{items}</div>'''

                gr.HTML(value=_render_faq())

            # ──────────────────────────────────────────────────────
            #  TAB 7 — AGENT CHAT (CLI-style, multi-turn, multi-model)
            # ──────────────────────────────────────────────────────
            with gr.Tab("Agent Chat", elem_id="agent-chat-tab"):
                # ── Helper definitions (before components that use them) ──
                model_choices = {
                    info["label"]: model_id
                    for model_id, info in MODEL_REGISTRY.items()
                }
                default_model_label = MODEL_NAME
                for label, mid in model_choices.items():
                    if mid == MODEL_NAME:
                        default_model_label = label
                        break

                def _chat_update_status():
                    if _last_pipeline_state.get("triage"):
                        tri = _last_pipeline_state["triage"]
                        return (
                            '<span class="status-dot green"></span>',
                            f'<span style="color:#00FF88;font-family:JetBrains Mono,monospace;font-size:0.78rem;">'
                            f'● {tri.get("severity","?")} · {tri.get("category","?")} · '
                            f'{len(_last_pipeline_state.get("anomalies",[]))} anomalies</span>'
                        )
                    return (
                        '<span class="status-dot gray"></span>',
                        '<span style="color:#8b949e;font-family:JetBrains Mono,monospace;font-size:0.78rem;">'
                        '⏳ Awaiting incident…</span>'
                    )

                def _chat_refresh_risk():
                    if not _last_pipeline_state.get("triage"):
                        return '<div style="color:#8b949e;font-size:0.78rem;">Run an analysis to see risk assessment.</div>'
                    ra = _run_risk_assessment()
                    rl = ra["risk_levels"]
                    sev = ra["severity"]
                    sev_color = SEVERITY_LEVELS.get(sev, {}).get("color", "#64748b")
                    sev_label = SEVERITY_LEVELS.get(sev, {}).get("label", sev)
                    safety_summary = _last_pipeline_state.get("safety_audit_summary", {})
                    safety_html = ""
                    if safety_summary:
                        blocked = safety_summary.get("blocked", 0)
                        flagged = safety_summary.get("flagged", 0)
                        total = safety_summary.get("total_checks", 0)
                        if blocked or flagged:
                            safety_html = (
                                f'<div style="display:flex;gap:12px;margin-top:6px;padding-top:6px;border-top:1px solid #30363d;font-size:0.76rem;">'
                                f'<span>🛑 Blocked: <b style="color:#FF3B3B;">{blocked}</b></span>'
                                f'<span>⚠️ Flagged: <b style="color:#FFB800;">{flagged}</b></span>'
                                f'<span>✅ Passed: {total - blocked - flagged}/{total}</span>'
                                f'</div>'
                            )
                        else:
                            safety_html = (
                                f'<div style="color:#00FF88;font-size:0.76rem;padding-top:4px;">'
                                f'✅ SafetyGuard: {total}/{total} actions passed validation</div>'
                            )
                    return f'''
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;margin-top:10px;">
                      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                        <span style="font-size:0.85rem;font-weight:600;color:#e2e8f0;">🛡️ Risk & Security</span>
                        <span style="font-size:0.7rem;color:#8b949e;">· SLA: {ra["sla_minutes"]}min</span>
                      </div>
                      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:6px;font-size:0.78rem;">
                        <span>Severity: <b style="color:{sev_color}">{sev} ({sev_label})</b></span>
                        <span>Low risk: <b style="color:#00FF88;">{rl.get("low",0)}</b></span>
                        <span>Medium risk: <b style="color:#FFB800;">{rl.get("medium",0)}</b></span>
                        <span>High risk: <b style="color:#FF3B3B;">{rl.get("high",0)}</b></span>
                        {('🔒 <b style="color:#FF3B3B;">Security incident</b>' if ra["security_incident"] else '')}
                      </div>
                      <div style="font-size:0.76rem;color:#8b949e;padding:6px 0 0 0;border-top:1px solid #21262d;">
                        {ra["recommendation"]}
                      </div>
                      {safety_html}
                    </div>'''

                def _chat_update_model_info(model_label: str):
                    model_id = model_choices.get(model_label, MODEL_NAME)
                    info = MODEL_REGISTRY.get(model_id, {})
                    tags = []
                    if info.get("has_thinking"):
                        tags.append("🧠 thinking")
                    tags.append(f"max {info.get('max_tokens', 512)} tok")
                    return f'<span style="color:#8b949e;font-size:0.75rem;">{" · ".join(tags)}</span>'

                # ── Header ──
                gr.HTML(
                    '<div style="padding:8px 0 4px 0;">'
                    '<div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">'
                    '<span style="font-size:1.2rem;font-weight:700;color:#e2e8f0;font-family:Inter,sans-serif;">'
                    'InfraHeal AI Terminal</span>'
                    '</div>'
                    '<div style="font-size:0.78rem;color:#8b949e;font-family:JetBrains Mono,monospace;">'
                    'Ask questions about the analysis. Switch models to compare responses.</div>'
                    '</div>'
                )

                # ── Status Bar (initialised from state) ──
                _init_status = _chat_update_status()
                with gr.Row(elem_classes="chat-status-bar"):
                    status_dot = gr.HTML(value=_init_status[0])
                    status_text = gr.HTML(value=_init_status[1])

                # ── Model Selector ──
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=list(model_choices.keys()),
                        value=default_model_label,
                        label="Model",
                        scale=3,
                        container=True,
                        interactive=True,
                    )
                    model_info_html = gr.HTML(
                        value=_chat_update_model_info(default_model_label)
                    )

                # ── Custom Chat ──
                chat_state = gr.State([{
                    "role": "assistant",
                    "content": "**System Ready**\n\nInfraHeal AI v1.0 \u2014 Autonomous Incident Diagnosis\nAMD ROCm + vLLM\n\nRun an analysis first, then ask me anything."
                }])
                chat_display = gr.HTML(value=_render_chat_html(chat_state.value))

                # ── Input Row ──
                with gr.Row(elem_classes="chat-input-row", equal_height=True):
                    chat_msg = gr.Textbox(
                        placeholder="Ask a question about the analysis...",
                        label=False,
                        container=False,
                        scale=10,
                        min_width=100,
                    )
                    chat_send = gr.Button("\u2191", variant="primary", elem_classes="chat-send-btn", elem_id="chat-send-btn", interactive=False, scale=0, min_width=36)
                    chat_clear = gr.Button("\u2715", variant="secondary", elem_classes="chat-clear-btn", elem_id="chat-clear-btn", scale=0, min_width=36)

                # ── Quick Questions ──
                gr.HTML(
                    '<div style="font-size:0.72rem;color:#8b949e;font-family:JetBrains Mono,monospace;padding:4px 0 8px 0;">'
                    'Quick questions:</div>'
                )
                with gr.Row():
                    q1 = gr.Button("Why P1?", elem_classes="chat-quick-btn", scale=1)
                    q2 = gr.Button("What's the root cause?", elem_classes="chat-quick-btn", scale=1)
                    q3 = gr.Button("What should I do?", elem_classes="chat-quick-btn", scale=1)
                    q4 = gr.Button("Explain evidence", elem_classes="chat-quick-btn", scale=1)
                    q5 = gr.Button("Re-analyze", elem_classes="chat-quick-btn", scale=1)

                # ── Risk Panel (initialised from state) ──
                risk_panel = gr.HTML(value=_chat_refresh_risk())

                # Pending quick question state (for generator chaining)
                pending_quick_q = gr.State("")

                # ── Event Wiring ──
                model_selector.change(
                    fn=_chat_update_model_info,
                    inputs=[model_selector],
                    outputs=[model_info_html],
                )

                def _chat_handler(message: str, history: list, model_label: str):
                    if not message or not message.strip():
                        yield history, _render_chat_html(history), gr.update()
                        return
                    history.append({"role": "user", "content": message})
                    yield history, _render_chat_html(history), ""
                    history.append({"role": "assistant", "content": "*Thinking...*"})
                    yield history, _render_chat_html(history), ""
                    model_id = model_choices.get(model_label, MODEL_NAME)
                    ctx = [h for h in history if h.get("content") != "*Thinking...*"]
                    response = _chat_respond(message, ctx, model_id=model_id)
                    history[-1] = {"role": "assistant", "content": response}
                    yield history, _render_chat_html(history), ""

                chat_send.click(
                    fn=_chat_handler,
                    inputs=[chat_msg, chat_state, model_selector],
                    outputs=[chat_state, chat_display, chat_msg],
                )
                chat_msg.submit(
                    fn=_chat_handler,
                    inputs=[chat_msg, chat_state, model_selector],
                    outputs=[chat_state, chat_display, chat_msg],
                )
                chat_msg.change(
                    fn=lambda v: gr.update(interactive=bool(v.strip())),
                    inputs=[chat_msg],
                    outputs=[chat_send],
                )
                chat_clear.click(
                    fn=lambda: ([{
                        "role": "assistant",
                        "content": "**System Ready**\n\nInfraHeal AI v1.0 \u2014 Terminal cleared. Ready for new questions."
                    }], _render_chat_html([{
                        "role": "assistant",
                        "content": "**System Ready**\n\nInfraHeal AI v1.0 \u2014 Terminal cleared. Ready for new questions."
                    }]), ""),
                    inputs=[],
                    outputs=[chat_state, chat_display, chat_msg],
                )

                for btn, q_text in [(q1, "Why P1?"), (q2, "What's the root cause?"), (q3, "What should I do?"), (q4, "Explain evidence"), (q5, "Re-analyze")]:
                    btn.click(
                        fn=lambda q=q_text: q,
                        inputs=[],
                        outputs=[pending_quick_q],
                    ).then(
                        fn=_chat_handler,
                        inputs=[pending_quick_q, chat_state, model_selector],
                        outputs=[chat_state, chat_display, chat_msg],
                    )

                # Wire analysis button to also update chat status/risk components
                analyze_btn.click(
                    fn=_run_analysis,
                    inputs=[scenario_dropdown],
                    outputs=[triage_panel, rca_panel, remed_panel, report_panel, reasoning_panel,
                             status_dot, status_text, risk_panel],
                )
        # On first load, auto-select the first scenario in the Incident Analysis tab
        if scenario_names:
            demo.load(
                fn=_on_scenario_selected,
                inputs=[scenario_dropdown],
                outputs=[scenario_desc, scenario_logs],
            )

    logger.info("InfraHeal AI dashboard created successfully")

    return demo


# ═══════════════════════════════════════════════════════════════════
#  STANDALONE LAUNCH
# ═══════════════════════════════════════════════════════════════════

def launch_dashboard(
    orchestrator: Optional[Any] = None,
    anomaly_detector: Optional[Any] = None,
    data_gen_func: Optional[Callable] = None,
    share: bool = False,
) -> None:
    """Create and launch the dashboard.

    Args:
        orchestrator: InfraHealOrchestrator instance (optional).
        anomaly_detector: Anomaly detection component (optional).
        data_gen_func: Function returning incident scenarios (optional).
        share: If True, create a public Gradio share link.
    """
    demo = create_dashboard(orchestrator, anomaly_detector, data_gen_func)
    demo.launch(
        server_name=DASHBOARD_HOST,
        server_port=DASHBOARD_PORT,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    launch_dashboard()
