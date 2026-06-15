"""
InfraHeal AI — JupyterLab Helper
===================================
Provides convenience functions for running InfraHeal AI
in AMD GPU-powered JupyterLab environments.

Usage in a notebook cell::

    from jupyter_helper import launch_in_jupyter
    demo = launch_in_jupyter()
    # or for inline plots:
    from jupyter_helper import generate_3d_plots_inline
    generate_3d_plots_inline()
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("infraheal.jupyter")

# Ensure the infraheal package is importable
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def setup_environment() -> Dict[str, Any]:
    """Initialize InfraHeal AI systems for Jupyter.

    Returns dict with: client, detector, knowledge_base, orchestrator,
    scenarios, logs, metrics, data.
    """
    from openai import OpenAI
    from config import VLLM_BASE_URL, VLLM_API_KEY, detect_model, MODEL_NAME
    from data_generator import generate_all_data, create_incident_scenarios
    from anomaly_detector import AnomalyDetector
    from rag.knowledge_base import KnowledgeBase
    from agents.orchestrator import InfraHealOrchestrator

    logger.info("Initializing InfraHeal AI for JupyterLab...")

    # Connect to vLLM
    client = None
    model = None
    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        model = detect_model(client)
        logger.info("vLLM connected, using model: %s", model)
    except Exception as e:
        logger.warning("vLLM unavailable: %s", e)

    # Generate data
    data = generate_all_data(save_to_disk=True)
    logger.info("Data ready: %d logs, %d runbooks", len(data["logs"]), len(data["runbooks"]))

    # Setup components
    kb = KnowledgeBase(runbooks=data["runbooks"], past_incidents=data["past_incidents"])
    detector = AnomalyDetector()
    orchestrator = InfraHealOrchestrator(
        client=client, model_name=model, knowledge_base=kb
    ) if client else None

    scenarios = create_incident_scenarios()

    return {
        "client": client,
        "model": model or MODEL_NAME,
        "detector": detector,
        "knowledge_base": kb,
        "orchestrator": orchestrator,
        "scenarios": scenarios,
        "data": data,
        "logs": data["logs"],
        "metrics": data["metrics"],
    }


def _find_free_port(start: int, max_attempts: int = 10) -> int:
    """Find the first free port starting from *start*."""
    import socket
    for port in range(start, start + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start}-{start + max_attempts - 1}")


def launch_dashboard(
    share: bool = True,
    port: int = 7860,
    show_error: bool = True,
) -> Any:
    """Launch the InfraHeal AI Gradio dashboard from Jupyter.

    Tries *port* first; if busy, scans upward for a free port.

    Args:
        share: Create a public Gradio share link (requires internet).
        port: Local port for the dashboard server.
        show_error: Show detailed errors in the UI.

    Returns:
        The running Gradio Blocks instance.
    """
    from config import DASHBOARD_HOST
    from dashboard import create_dashboard
    from data_generator import create_incident_scenarios
    from anomaly_detector import AnomalyDetector

    free_port = _find_free_port(port)
    if free_port != port:
        logger.warning("Port %d is busy, using port %d instead", port, free_port)

    env = setup_environment()

    demo = create_dashboard(
        orchestrator=env["orchestrator"],
        anomaly_detector=env["detector"],
        data_gen_func=create_incident_scenarios,
    )

    import dashboard as _dash_mod

    # ── Add live-html endpoint for pipeline progress polling ──
    try:
        from fastapi.responses import PlainTextResponse
        import sys
        print(f"[LIVE-HTML] demo.app type: {type(demo.app)}", flush=True)
        @demo.app.get("/live-html")
        def _serve_live_html():
            alive = (_dash_mod._process_thread is not None and _dash_mod._process_thread.is_alive()) or (_dash_mod._monitor_thread is not None and _dash_mod._monitor_thread.is_alive())
            if alive or _dash_mod._live_html:
                with _dash_mod._live_html_lock:
                    html = _dash_mod._live_html or ""
                    if alive:
                        print(f"[LIVE-HTML] Returning {len(html)} bytes (alive={alive})", flush=True)
                    return PlainTextResponse(html)
            return PlainTextResponse("")
        print("[LIVE-HTML] Endpoint registered successfully", flush=True)
    except Exception as ex:
        logger.warning("live-html endpoint failed: %s", ex)

    logger.info("Launching dashboard on port %d (share=%s)...", free_port, share)
    demo.launch(
        server_name=DASHBOARD_HOST,
        server_port=free_port,
        head=_dash_mod.HEAD_HTML,
        share=share,
        show_error=show_error,
    )
    return demo


def run_pipeline_on_scenario(
    scenario_idx: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full agent pipeline on a single scenario.

    Uses the LLM model (Qwen2.5-7B-Instruct via vLLM on AMD GPU)
    for all 4 agent stages: Triage → RCA → Remediation → Report.

    Args:
        scenario_idx: Index into the demo scenarios list.
        verbose: Print progress and results.

    Returns:
        Full pipeline result dict with triage_result, rca_result,
        remediation_result, report, pipeline_metrics, reasoning_chain.
    """
    env = setup_environment()
    scenarios = env["scenarios"]
    detector = env["detector"]
    orchestrator = env["orchestrator"]

    if scenario_idx >= len(scenarios):
        raise ValueError(f"Scenario index {scenario_idx} out of range (max {len(scenarios) - 1})")

    scenario = scenarios[scenario_idx]
    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'='*60}\n")

    # Detect anomalies
    anomalies = detector.detect_all(scenario["logs"], scenario["metrics"])
    if verbose:
        print(f"Detected {len(anomalies)} anomalies\n")

    # Run pipeline
    if orchestrator:
        result = orchestrator.process_incident(
            anomalies=anomalies,
            logs=scenario["logs"],
            metrics=scenario["metrics"],
        )
        if verbose:
            triage = result.get("triage_result", {})
            rca = result.get("rca_result", {})
            remed = result.get("remediation_result", {})
            report = result.get("report", {})
            print(f"Triage: Severity={triage.get('severity')}, Category={triage.get('category')}")
            print(f"Root Cause: {rca.get('root_cause', 'N/A')}")
            print(f"Remediation: {len(remed.get('recommended_actions', []))} actions")
            print(f"Report: {report.get('title', 'N/A')}")
            perf = result.get("pipeline_metrics", {})
            print(f"\nPipeline: {perf.get('total_time_seconds', 0):.2f}s total")
        return result
    else:
        if verbose:
            print("No orchestrator available (vLLM not connected).")
        return {"error": "vLLM not available", "anomalies": anomalies}


def run_error_level_resolution(
    scenario_idx: int = 0,
    verbose: bool = True,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Run error-level specific resolution on a scenario.

    Args:
        scenario_idx: Index into demo scenarios.
        verbose: Print per-level results.
        use_llm: If True, runs the full LLM pipeline per level
                 (requires vLLM; 4 LLM calls per level with anomalies).
                 If False (default), uses fast template-based resolution.

    Returns:
        Dict with per_level resolution results.
    """
    env = setup_environment()
    scenarios = env["scenarios"]
    orchestrator = env["orchestrator"]

    if scenario_idx >= len(scenarios):
        raise ValueError(f"Scenario index {scenario_idx} out of range")

    scenario = scenarios[scenario_idx]

    if orchestrator:
        if use_llm:
            print("  [LLM mode] Running per-level pipeline with vLLM...")
            print("  This runs 4 agent calls per level with anomalies.\n")
        result = orchestrator.process_by_error_level(
            logs=scenario["logs"],
            metrics=scenario["metrics"],
            use_llm=use_llm,
        )
        if verbose:
            print(f"\n{'='*60}")
            print(f"Error-Level Resolution: {scenario['name']}")
            print(f"{'='*60}")
            for lr in result.get("level_summary", []):
                level = lr.get("level", "?")
                an_count = lr.get("anomaly_count", 0)
                llm_gen = lr.get("llm_generated", False)
                print(f"\n  [{level}] {an_count} anomalies "
                      f"{'(LLM-generated)' if llm_gen else '(template-based)'}")
                print(f"  Resolution: {lr.get('resolution_summary', 'N/A')[:120]}")
                for step in lr.get("resolution_steps", []):
                    print(f"    - {step}")
        return result
    else:
        if verbose:
            print("No orchestrator available (vLLM not connected).")
            print("Run with `use_llm=False` for template-based resolution, or")
            print("connect to vLLM and set `use_llm=True` for model-generated steps.")
        return {"error": "vLLM not available"}


def generate_3d_plots_inline(
    scenario_idx: Optional[int] = None,
    width: int = 800,
    height: int = 500,
) -> None:
    """Display 3D plots inline in a Jupyter notebook cell.

    Args:
        scenario_idx: Scenario index (None = use all data).
        width: Plot width in pixels.
        height: Plot height in pixels.
    """
    import plotly.io as pio
    pio.renderers.default = "notebook"

    from visualizer_2d import (
        time_series_dashboard,
        correlation_heatmap,
        anomaly_timeline,
        log_level_distribution,
        host_radar,
    )

    env = setup_environment()
    data = env["data"]

    if scenario_idx is not None:
        scenarios = env["scenarios"]
        if scenario_idx < len(scenarios):
            sc = scenarios[scenario_idx]
            logs = sc.get("logs", [])
            metrics = sc.get("metrics", [])
        else:
            logs, metrics = data["logs"], data["metrics"]
    else:
        logs, metrics = data["logs"], data["metrics"]

    from IPython.display import display, HTML
    display(HTML("<h3>3D Metric Space — CPU / Memory / Latency</h3>"))
    display(HTML(metrics_3d_scatter(metrics)))

    display(HTML("<h3>Log Level Distribution</h3>"))
    display(HTML(log_level_distribution_3d(logs)))

    display(HTML("<h3>CPU % Surface — Time vs Host</h3>"))
    display(HTML(timeline_3d_surface(metrics, metric_field="cpu_percent")))
