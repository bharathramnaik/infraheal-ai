#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║                    InfraHeal AI v1.0                            ║
║        Autonomous Incident Diagnosis & Resolution Agent         ║
║                                                                  ║
║  TCS & AMD AI Hackathon 2026                                    ║
║  Team: team-790 | bharathram.naiktv                             ║
║  Track: Agents (AGENTS_026)                                     ║
║  Stack: AMD ROCm + vLLM + Qwen2.5-7B-Instruct                 ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python main.py                    # Launch dashboard
    python main.py --demo             # Run demo scenario without dashboard
    python main.py --generate-data    # Generate sample data only
    python main.py --test-llm         # Test vLLM connectivity
"""

import argparse
import json
import logging
import os
import sys
import time

# ─── Setup Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("infraheal.main")

# ─── Ensure proper imports ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print the InfraHeal AI banner."""
    banner = r"""
  ___      _        __  __      _         _            ___ ___
 |_ _|_ _ | |__    |  \/  |__ _| |_____ _| |_____ _ _ |_ _|_ _|
  | || ' \| / /    | |\/| / _` | / / _ \ '_/ -_) '_|  | | | |
 |___|_||_|_\_\    |_|  |_\__,_|_\_\___/_| \___|_|   |___|___|

    Autonomous Incident Diagnosis & Resolution Agent
    TCS & AMD AI Hackathon 2026 | Team team-790
    """
    try:
        print(banner)
    except UnicodeEncodeError:
        print("[InfraHeal AI] Autonomous Incident Diagnosis & Resolution Agent")


def test_vllm_connection():
    """Test connectivity to vLLM server and detect available models."""
    from openai import OpenAI
    from config import VLLM_BASE_URL, VLLM_API_KEY, detect_model

    logger.info(f"Testing vLLM connection at {VLLM_BASE_URL}...")
    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = client.models.list()
        available = [m.id for m in models.data]
        logger.info(f"✅ Connected! Available models: {available}")

        model = detect_model(client)
        logger.info(f"Selected model: {model}")

        # Quick inference test
        logger.info("Running quick inference test...")
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'InfraHeal AI is online' in exactly 5 words."}],
            max_tokens=20,
            temperature=0.1,
        )
        elapsed = time.time() - start
        reply = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else "N/A"
        logger.info(f"✅ Inference OK in {elapsed:.2f}s | Tokens: {tokens} | Reply: {reply}")
        return True, model
    except Exception as e:
        logger.error(f"❌ vLLM connection failed: {e}")
        return False, None


def generate_data():
    """Generate all synthetic data and save to disk."""
    from data_generator import generate_all_data

    logger.info("Generating synthetic infrastructure data...")
    data = generate_all_data(save_to_disk=True)
    logger.info(f"✅ Generated: {len(data['logs'])} logs, "
                f"{len(data['metrics'])} metric points, "
                f"{len(data['runbooks'])} runbooks, "
                f"{len(data['past_incidents'])} past incidents")
    return data


def run_demo():
    """Run a single demo scenario end-to-end (no dashboard)."""
    from openai import OpenAI
    from config import VLLM_BASE_URL, VLLM_API_KEY, detect_model
    from data_generator import generate_all_data, create_incident_scenarios
    from anomaly_detector import AnomalyDetector
    from rag.knowledge_base import KnowledgeBase
    from agents.orchestrator import InfraHealOrchestrator

    print_banner()

    # Step 1: Test LLM
    logger.info("=" * 60)
    logger.info("STEP 1: Connecting to vLLM...")
    logger.info("=" * 60)
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    model = detect_model(client)
    logger.info(f"Using model: {model}")

    # Step 2: Generate data
    logger.info("=" * 60)
    logger.info("STEP 2: Generating synthetic data...")
    logger.info("=" * 60)
    data = generate_all_data(save_to_disk=False)

    # Step 3: Setup
    logger.info("=" * 60)
    logger.info("STEP 3: Initializing systems...")
    logger.info("=" * 60)
    kb = KnowledgeBase(runbooks=data["runbooks"], past_incidents=data["past_incidents"])
    orchestrator = InfraHealOrchestrator(client=client, model_name=model, knowledge_base=kb)
    detector = AnomalyDetector()

    # Step 4: Pick a scenario
    logger.info("=" * 60)
    logger.info("STEP 4: Running incident scenario...")
    logger.info("=" * 60)
    scenarios = create_incident_scenarios()
    scenario = scenarios[0]  # First scenario
    logger.info(f"Scenario: {scenario['name']}")
    logger.info(f"Description: {scenario['description']}")

    # Step 5: Detect anomalies
    anomalies = detector.detect_all(scenario["logs"], scenario["metrics"])
    logger.info(f"Detected {len(anomalies)} anomalies")

    # Step 6: Process through agent pipeline
    logger.info("=" * 60)
    logger.info("STEP 5: Running agent pipeline...")
    logger.info("=" * 60)
    pipeline_start = time.time()
    result = orchestrator.process_incident(
        anomalies=anomalies,
        logs=scenario["logs"],
        metrics=scenario["metrics"],
    )
    pipeline_elapsed = time.time() - pipeline_start

    # Step 7: Display results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    triage = result.get("triage_result", {})
    rca_res = result.get("rca_result", {})
    remediation_res = result.get("remediation_result", {})
    report = result.get("report", {})

    if triage:
        logger.info("Triage: Severity=%s, Category=%s",
                     triage.get("severity", "N/A"), triage.get("category", "N/A"))

    if rca_res:
        logger.info("Root Cause: %s", rca_res.get("root_cause", "N/A"))
        logger.info("   Confidence: %s", rca_res.get("confidence_score", "N/A"))

    if remediation_res:
        actions = remediation_res.get("recommended_actions", [])
        logger.info("Remediation: %d actions recommended", len(actions))
        for i, action in enumerate(actions, 1):
            logger.info("   %d. %s: %s", i,
                        action.get("tool_name", "N/A"), action.get("rationale", "N/A"))

    if report:
        logger.info("Report: %s", report.get("title", "N/A"))
        logger.info("   Summary: %s", report.get("executive_summary", "N/A")[:200])

    # Step 8: Performance metrics
    logger.info("=" * 60)
    logger.info("PERFORMANCE METRICS")
    logger.info("=" * 60)
    perf = result.get("pipeline_metrics", {})
    logger.info("Total pipeline time: %.2fs", pipeline_elapsed)
    logger.info("Agent metrics: %s", json.dumps(perf, indent=2))

    return result


def launch_dashboard():
    """Launch the interactive Gradio dashboard."""
    from openai import OpenAI
    from config import VLLM_BASE_URL, VLLM_API_KEY, detect_model, DASHBOARD_HOST, DASHBOARD_PORT
    from data_generator import generate_all_data, create_incident_scenarios
    from anomaly_detector import AnomalyDetector
    from rag.knowledge_base import KnowledgeBase
    from agents.orchestrator import InfraHealOrchestrator
    from dashboard import create_dashboard

    print_banner()

    # Initialize components
    logger.info("Initializing InfraHeal AI systems...")

    # Try connecting to vLLM (graceful failure for layout preview)
    client = None
    model = None
    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY,
                        timeout=5.0, max_retries=0)
        model = detect_model(client)
        logger.info(f"✅ vLLM connected, using model: {model}")
    except Exception as e:
        logger.warning(f"⚠️  vLLM not available ({e}). Dashboard will run in preview mode.")

    # Generate data
    data = generate_all_data(save_to_disk=True)
    logger.info(f"✅ Data ready: {len(data['logs'])} logs, {len(data['runbooks'])} runbooks")

    # Setup components
    kb = KnowledgeBase(runbooks=data["runbooks"], past_incidents=data["past_incidents"])
    detector = AnomalyDetector()
    orchestrator = InfraHealOrchestrator(
        client=client, model_name=model, knowledge_base=kb
    ) if client else None

    # Create and launch dashboard
    demo = create_dashboard(
        orchestrator=orchestrator,
        anomaly_detector=detector,
        data_gen_func=create_incident_scenarios,
    )

    logger.info(f"🚀 Launching dashboard on {DASHBOARD_HOST}:{DASHBOARD_PORT}")
    demo.launch(
        server_name=DASHBOARD_HOST,
        server_port=DASHBOARD_PORT,
        share=True,  # Generates public URL for demo
        show_error=True,
    )


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="InfraHeal AI — Autonomous Incident Diagnosis & Resolution Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
   python main.py                    Launch interactive dashboard
   python main.py --demo             Run demo scenario in terminal
   python main.py --generate-data    Generate synthetic data only
   python main.py --test-llm         Test vLLM server connection
   python main.py --jupyter          Print JupyterLab setup instructions
        """,
    )
    parser.add_argument("--demo", action="store_true", help="Run demo scenario without dashboard")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data only")
    parser.add_argument("--test-llm", action="store_true", help="Test vLLM server connectivity")
    parser.add_argument("--tune", action="store_true", help="Run GPU throughput profiler and exit")
    parser.add_argument("--port", type=int, default=7860, help="Dashboard port (default: 7860)")
    parser.add_argument("--jupyter", action="store_true", help="Print JupyterLab setup instructions")
    parser.add_argument("--model", type=str, default=None, help="Override model name")

    args = parser.parse_args()

    # Override config if needed
    if args.model:
        import config
        config.MODEL_NAME = args.model
    if args.port:
        import config
        config.DASHBOARD_PORT = args.port

    print_banner()

    if args.test_llm:
        success, model = test_vllm_connection()
        sys.exit(0 if success else 1)
    elif args.tune:
        logger.info("Running GPU throughput profiler...")
        from gpu_autotuner import GPUTuner
        from config import VLLM_BASE_URL, VLLM_API_KEY, MODEL_NAME
        from openai import OpenAI
        try:
            client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, timeout=5.0, max_retries=0)
            _ = client.models.list()
        except Exception:
            logger.warning("vLLM not available — running simulated benchmark")
            client = None
        tuner = GPUTuner(client=client, model_name=MODEL_NAME)
        result = tuner.benchmark()
        print(json.dumps(result, indent=2))
        sys.exit(0)
    elif args.generate_data:
        generate_data()
    elif args.demo:
        run_demo()
    elif args.jupyter:
        print()
        print("=" * 60)
        print("  InfraHeal AI — JupyterLab Quick Start")
        print("=" * 60)
        print()
        print("  In a Jupyter notebook cell, run:")
        print()
        print("    from jupyter_helper import launch_dashboard, run_pipeline_on_scenario")
        print("    from jupyter_helper import generate_3d_plots_inline")
        print()
        print("  # Launch the Gradio dashboard:")
        print("  demo = launch_dashboard(share=True, port=7860)")
        print()
        print("  # Run pipeline on a scenario:")
        print("  result = run_pipeline_on_scenario(scenario_idx=0)")
        print()
        print("  # Show 3D plots inline:")
        print("  generate_3d_plots_inline(scenario_idx=0)")
        print()
        print("  # Error-level resolution:")
        print("  from jupyter_helper import run_error_level_resolution")
        print("  result = run_error_level_resolution(scenario_idx=0)")
        print()
        print("=" * 60)
    else:
        launch_dashboard()


if __name__ == "__main__":
    main()
