"""
InfraHeal AI — Orchestrator
=============================
Chains all agents (Triage → RCA → Remediation → Reporting) into a
unified incident-processing pipeline, with RAG-augmented context
injection between stages.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .triage_agent import TriageAgent
from .rca_agent import RCAAgent
from .remediation_agent import RemediationAgent
from .reporting_agent import ReportingAgent
from .safety_guard import SafetyGuard

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VLLM_BASE_URL, VLLM_API_KEY, MODEL_NAME, detect_model
from gpu_tracker import GPUMonitor

logger = logging.getLogger(__name__)


class InfraHealOrchestrator:
    """Main orchestrator for the InfraHeal AI multi-agent pipeline.

    Coordinates four specialist agents and an optional RAG knowledge base
    through a deterministic pipeline::

        anomalies → Triage → (RAG lookup) → RCA → Remediation → Report

    Each stage feeds its structured output as context to the next.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: Optional[str] = None,
        knowledge_base: Optional[Any] = None,
    ) -> None:
        """Initialise the orchestrator and all sub-agents.

        Args:
            client: Shared OpenAI client.  Creates a default one
                pointing at the local vLLM server when *None*.
            model_name: Model identifier.  When *None*, auto-detection
                is attempted, falling back to ``MODEL_NAME``.
            knowledge_base: An instance of
                ``rag.knowledge_base.KnowledgeBase`` for RAG-augmented
                root cause analysis.  When *None*, RCA runs without
                runbook context.
        """
        self.client = client or OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

        # Auto-detect model if not specified
        if model_name:
            self.model_name = model_name
        else:
            try:
                self.model_name = detect_model(self.client)
                logger.info("Auto-detected model: %s", self.model_name)
            except Exception:
                self.model_name = MODEL_NAME
                logger.info("Using default model: %s", self.model_name)

        self.knowledge_base = knowledge_base

        # Initialise all agents with shared client & model
        self.triage_agent = TriageAgent(client=self.client, model_name=self.model_name)
        self.rca_agent = RCAAgent(client=self.client, model_name=self.model_name)
        self.remediation_agent = RemediationAgent(client=self.client, model_name=self.model_name)
        self.reporting_agent = ReportingAgent(client=self.client, model_name=self.model_name)
        self.safety_guard = SafetyGuard()

        self._gpu_monitor = GPUMonitor()
        self._pipeline_log: List[Dict[str, Any]] = []
        logger.info("InfraHealOrchestrator initialised (model=%s)", self.model_name)

    # ── Main Pipeline ────────────────────────────────────────────

    def process_incident(
        self,
        anomalies: List[dict],
        logs: Optional[List[dict]] = None,
        metrics: Optional[List[dict]] = None,
    ) -> dict:
        """Run the full incident pipeline: Triage → RCA → Remediation → Report.

        Args:
            anomalies: List of anomaly dicts (required).
            logs: Optional list of log dicts for additional context.
            metrics: Optional list of metric dicts for additional context.

        Returns:
            Comprehensive result dict containing:
            - ``triage_result``: TriageAgent output
            - ``rca_result``: RCAAgent output
            - ``remediation_result``: RemediationAgent output
            - ``report``: ReportingAgent output
            - ``execution_results``: Simulated action results
            - ``pipeline_metrics``: Timing and token usage
        """
        pipeline_start = time.time()
        pipeline_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        logger.info("═══ Pipeline %s started ═══", pipeline_id)

        self._gpu_monitor = GPUMonitor()
        self._gpu_monitor.start()

        result: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "anomaly_count": len(anomalies),
        }

        # ── Step 1: Triage ──────────────────────────────────────
        logger.info("Step 1/5: Running Triage Agent…")
        step_start = time.time()
        try:
            triage_result = self.triage_agent.run({
                "anomalies": anomalies,
                "logs": logs or [],
                "metrics": metrics or [],
            })
        except Exception as exc:
            logger.error("Triage failed: %s", exc)
            triage_result = TriageAgent._default_result()
            triage_result["error"] = str(exc)

        result["triage_result"] = triage_result
        result["triage_latency"] = round(time.time() - step_start, 2)
        logger.info(
            "Triage complete: %s (%s) in %.2fs",
            triage_result.get("severity"), triage_result.get("category"),
            result["triage_latency"],
        )

        # ── Step 2: RAG Lookup ──────────────────────────────────
        logger.info("Step 2/5: RAG knowledge retrieval…")
        runbook_context = ""
        if self.knowledge_base:
            try:
                query_parts = [
                    triage_result.get("category", ""),
                    triage_result.get("impact_assessment", ""),
                ]
                # Add anomaly descriptions for richer queries
                for anomaly in anomalies[:3]:
                    query_parts.append(anomaly.get("description", ""))

                query = " ".join(filter(None, query_parts))
                runbook_context = self.knowledge_base.get_context(
                    query=query,
                    category=triage_result.get("category"),
                )
                logger.info("RAG returned %d chars of context", len(runbook_context))
            except Exception as exc:
                logger.warning("RAG lookup failed (continuing without): %s", exc)
        else:
            logger.info("No knowledge base configured — skipping RAG")

        result["rag_context_length"] = len(runbook_context)

        # ── Step 3: Root Cause Analysis ─────────────────────────
        logger.info("Step 3/5: Running RCA Agent…")
        step_start = time.time()
        try:
            rca_result = self.rca_agent.run({
                "anomalies": anomalies,
                "triage_result": triage_result,
                "runbook_context": runbook_context,
            })
        except Exception as exc:
            logger.error("RCA failed: %s", exc)
            rca_result = RCAAgent._default_result()
            rca_result["error"] = str(exc)

        result["rca_result"] = rca_result
        result["rca_latency"] = round(time.time() - step_start, 2)
        logger.info(
            "RCA complete: confidence=%.2f in %.2fs",
            rca_result.get("confidence_score", 0), result["rca_latency"],
        )

        # ── Step 3b: Self-Critique / Reflection ──────────────────
        rc_text = rca_result.get("root_cause", "")
        is_fallback = ("Unable to determine" in rc_text or "No anomalies provided" in rc_text)
        if is_fallback:
            logger.warning("Skipping critique — RCA returned fallback (no actionable root cause)")
            result["critique"] = {"confirmed": True, "refined_reasoning": "Critique skipped — RCA had no root cause to analyze.", "skipped": True}
        else:
            logger.info("Step 3b/5: Running self-critique on RCA…")
            try:
                critique = self._critique_analysis(anomalies, triage_result, rca_result)
                if critique.get("confirmed"):
                    logger.info("Critique CONFIRMED RCA (confidence=%.2f)", critique.get("refined_confidence", rca_result.get("confidence_score", 0)))
                else:
                    logger.warning("Critique found gaps: %s", critique.get("gaps", []))
                    rca_result["refined_confidence"] = critique.get("refined_confidence", rca_result.get("confidence_score", 0))
                    rca_result["refined_reasoning"] = critique.get("refined_reasoning", rca_result.get("reasoning_summary", ""))
                    rca_result["critique_gaps"] = critique.get("gaps", [])
                    rca_result["critique_suggestions"] = critique.get("suggestions", [])
                result["critique"] = critique
            except Exception as exc:
                logger.warning("Critique failed (continuing without): %s", exc)
                result["critique"] = {"confirmed": True, "error": str(exc)}

        # ── Step 4: Remediation Plan ────────────────────────────
        logger.info("Step 4/5: Running Remediation Agent…")
        step_start = time.time()
        try:
            remediation_result = self.remediation_agent.run({
                "rca_result": rca_result,
                "triage_result": triage_result,
            })
        except Exception as exc:
            logger.error("Remediation failed: %s", exc)
            remediation_result = RemediationAgent._default_result()
            remediation_result["error"] = str(exc)

        result["remediation_result"] = remediation_result
        result["remediation_latency"] = round(time.time() - step_start, 2)
        logger.info(
            "Remediation plan: %d actions in %.2fs",
            len(remediation_result.get("recommended_actions", [])),
            result["remediation_latency"],
        )

        # ── Step 4b: Execute Remediation (Simulated, Safety-Guarded) ──
        execution_results: List[dict] = []
        actions = remediation_result.get("recommended_actions", [])
        safety_results: List[dict] = []
        if actions:
            sev = triage_result.get("severity", "P3")
            logger.info("Validating %d actions with SafetyGuard (severity=%s)…", len(actions), sev)
            validated = self.safety_guard.validate_plan(actions, severity=sev)
            safety_results = [a.pop("_safety", {}) for a in validated]
            blocked = [s for s in safety_results if s.get("verdict") == "block"]
            flagged = [s for s in safety_results if s.get("verdict") == "flag"]
            allowed = [i for i, s in enumerate(safety_results) if s.get("verdict") == "allow"]

            if blocked:
                logger.warning("SafetyGuard blocked %d action(s): %s", len(blocked),
                               [b.get("reason","") for b in blocked])

            allowed_actions = [validated[i] for i in allowed]
            logger.info("Executing %d/%d actions (safety-passed)…", len(allowed_actions), len(actions))
            try:
                execution_results = self.remediation_agent.execute_plan(
                    allowed_actions, auto_approve_low_risk=True,
                )
            except Exception as exc:
                logger.error("Execution failed: %s", exc)
                execution_results = [{"error": str(exc), "status": "failed"}]

            # Append blocked/flagged as skipped entries with safety reason
            for i, s in enumerate(safety_results):
                if s.get("verdict") in ("block", "flag"):
                    execution_results.insert(i, {
                        "step": i + 1,
                        "tool_name": actions[i].get("tool_name", "?"),
                        "status": s["verdict"] + "ed_by_safety",
                        "message": s.get("reason", ""),
                        "risk_level": s.get("risk", "unknown"),
                    })

        result["execution_results"] = execution_results
        result["safety_results"] = safety_results
        result["safety_audit_summary"] = self.safety_guard.get_audit_summary()

        # ── Step 5: Incident Report ─────────────────────────────
        logger.info("Step 5/5: Running Reporting Agent…")
        step_start = time.time()
        try:
            report = self.reporting_agent.run({
                "anomalies": anomalies,
                "triage_result": triage_result,
                "rca_result": rca_result,
                "remediation_result": remediation_result,
                "execution_results": execution_results,
            })
        except Exception as exc:
            logger.error("Reporting failed: %s", exc)
            report = {
                "incident_id": f"INC-{pipeline_id}",
                "title": "Report Generation Failed",
                "error": str(exc),
            }

        result["report"] = report
        result["report_latency"] = round(time.time() - step_start, 2)

        # ── Root Cause Host / Affected Hosts ──────────────────
        anomaly_sources = [a.get("source", "") for a in anomalies if a.get("source")]
        host_counts = {h: anomaly_sources.count(h) for h in set(anomaly_sources)}
        worst_anomaly = max(anomalies, key=lambda a: {"P1":0,"P2":1,"P3":2,"P4":3}.get(a.get("severity","P4"), 4)) if anomalies else {}
        root_cause_host = rca_result.get("root_cause", worst_anomaly.get("source", anomaly_sources[0] if anomaly_sources else ""))
        affected_hosts = list(set(anomaly_sources))
        result["root_cause_host"] = root_cause_host
        result["affected_hosts"] = affected_hosts

        # ── Reasoning Chain ────────────────────────────────────
        reasoning_chain = []
        rc_items = [
            ("Triage Agent", triage_result, "triage_latency", "urgency_reasoning"),
            ("RCA Agent", rca_result, "rca_latency", "reasoning_summary"),
            ("Critique Agent", result.get("critique", {}), 0, "refined_reasoning"),
            ("Remediation Agent", remediation_result, "remediation_latency", "rollback_plan"),
            ("Reporting Agent", report, "report_latency", "executive_summary"),
        ]
        for name, agent_out, latency_key, reason_key in rc_items:
            if isinstance(agent_out, dict):
                err = agent_out.get("error", "")
                if err:
                    reasoning_chain.append({
                        "agent": name,
                        "reasoning": f"⚠ Error: {err[:500]}",
                        "latency_ms": 0,
                    })
                    continue
                thought = agent_out.get(reason_key, "")
            else:
                thought = ""
            rc_latency = result.get(latency_key, 0)
            reasoning_chain.append({
                "agent": name,
                "reasoning": thought[:500] if thought else f"{name} completed in {rc_latency}s",
                "latency_ms": round(rc_latency * 1000),
            })
        result["reasoning_chain"] = reasoning_chain

        # ── GPU Snapshot ────────────────────────────────────────
        self._gpu_monitor.stop()
        gpu_snap = self._gpu_monitor.snapshot() if hasattr(self, "_gpu_monitor") else {}

        # ── Pipeline Metrics ────────────────────────────────────
        total_time = round(time.time() - pipeline_start, 2)
        result["pipeline_metrics"] = {
            "total_time_seconds": total_time,
            "triage_latency": result.get("triage_latency", 0),
            "rca_latency": result.get("rca_latency", 0),
            "remediation_latency": result.get("remediation_latency", 0),
            "report_latency": result.get("report_latency", 0),
            "gpu_memory_mb": gpu_snap.get("gpu_memory_mb", 0),
            "gpu_peak_memory_mb": gpu_snap.get("gpu_peak_memory_mb", 0),
            "agent_metrics": self.get_all_metrics(),
        }
        result["completed_at"] = datetime.now(timezone.utc).isoformat()

        self._pipeline_log.append({
            "pipeline_id": pipeline_id,
            "total_time": total_time,
            "severity": triage_result.get("severity"),
            "root_cause": rca_result.get("root_cause", "")[:100],
            "actions_count": len(actions),
        })

        logger.info(
            "═══ Pipeline %s complete in %.2fs ═══", pipeline_id, total_time,
        )
        return result

    # ── Error-Level Resolution ───────────────────────────────────

    LOG_LEVEL_RANK = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3, "DEBUG": 4}

    def process_by_error_level(
        self,
        logs: List[dict],
        metrics: Optional[List[dict]] = None,
        use_llm: bool = False,
    ) -> dict:
        """Process anomalies grouped by log severity level.

        For each log level (CRITICAL, ERROR, WARNING) this method:
          1. Filters logs to that level.
          2. Runs anomaly detection on the filtered set.
          3. Generates level-specific resolution (optionally via LLM).

        When ``use_llm=False`` (default), resolutions are generated from
        templates based on anomaly patterns — fast and suitable for demo.
        When ``use_llm=True``, each level runs the full LLM pipeline.

        Args:
            logs: Full list of log dicts.
            metrics: Optional list of metric dicts.
            use_llm: If True, runs the full LLM pipeline per level
                     (slow — 4 LLM calls per level).

        Returns:
            Dict with keys:
            - ``per_level``: Dict of level → pipeline result
            - ``level_summary``: Aggregated summary of all levels
            - ``overall``: Full pipeline on all logs combined (when use_llm=True)
        """
        from anomaly_detector import AnomalyDetector

        detector = AnomalyDetector()
        levels = ["CRITICAL", "ERROR", "WARNING"]
        result: Dict[str, Any] = {
            "per_level": {},
            "level_summary": [],
        }

        # ── Templates for fast (non-LLM) mode ──────────────────
        RESOLUTION_TEMPLATES = {
            "CRITICAL": {
                "root_cause": "Critical infrastructure failure detected — "
                              "resource exhaustion or service crash in progress.",
                "steps": [
                    "Immediate isolation: Quarantine affected services to prevent cascading failure",
                    "Auto-remediation: Restart critical service processes",
                    "Scale vertically: Allocate additional CPU/memory resources",
                    "Traffic reroute: Redirect traffic to healthy replicas",
                    "SLA escalation: Notify on-call engineering team (15-min SLA)",
                    "Post-mortem: Initiate root cause investigation",
                ],
                "confidence": 0.92,
            },
            "ERROR": {
                "root_cause": "Application or system errors detected — "
                              "likely configuration issue or transient failure.",
                "steps": [
                    "Diagnose: Collect error logs and stack traces from affected services",
                    "Health check: Verify dependent service availability",
                    "Config review: Check for recent configuration changes",
                    "Auto-remediation: Apply known fix from runbook match",
                    "Verify: Run integration tests to confirm resolution",
                ],
                "confidence": 0.85,
            },
            "WARNING": {
                "root_cause": "Warning-level indicators suggest approaching threshold — "
                              "proactive intervention recommended.",
                "steps": [
                    "Analyze trend: Check if warning is increasing or stable",
                    "Resource check: Verify resource utilization metrics",
                    "Proactive scaling: Pre-scale resources if trend is upward",
                    "Alert tuning: Adjust thresholds if false positive",
                    "Monitor: Watch for 15 minutes to confirm stability",
                ],
                "confidence": 0.78,
            },
        }

        # Process each level separately
        for level in levels:
            level_logs = [
                l for l in logs
                if l.get("level", "").upper() == level
            ]
            if not level_logs:
                continue

            level_anomalies = detector.detect_all(logs=level_logs, metrics=metrics)
            template = RESOLUTION_TEMPLATES.get(level, RESOLUTION_TEMPLATES["WARNING"])

            if not level_anomalies:
                level_result = {
                    "level": level,
                    "anomaly_count": 0,
                    "message": f"No anomalies detected from {level} logs.",
                    "resolution_summary": (
                        f"No {level}-level anomalies found among "
                        f"{len(level_logs)} log entries. System operating normally."
                    ),
                    "resolution_steps": [],
                }
                result["per_level"][level] = level_result
                result["level_summary"].append(level_result)
                continue

            # Fast template-based resolution (default)
            if not use_llm:
                level_result = {
                    "level": level,
                    "anomaly_count": len(level_anomalies),
                    "anomalies": level_anomalies,
                    "root_cause": template["root_cause"],
                    "confidence": template["confidence"],
                    "resolution_summary": (
                        f"Resolved {len(level_anomalies)} {level}-level anomalies across "
                        f"{len(set(a.get('source','') for a in level_anomalies))} sources. "
                        f"Root cause: {template['root_cause'][:80]}... "
                        f"{len(template['steps'])} remediation steps generated."
                    ),
                    "resolution_steps": template["steps"],
                    "llm_generated": False,
                }
            else:
                # Full LLM pipeline per level (slow — 4 calls per level)
                try:
                    pipeline_out = self.process_incident(
                        anomalies=level_anomalies,
                        logs=level_logs,
                        metrics=metrics,
                    )
                    triage = pipeline_out.get("triage_result", {})
                    remediation = pipeline_out.get("remediation_result", {})
                    rca = pipeline_out.get("rca_result", {})

                    actions = remediation.get("recommended_actions", [])
                    resolution_steps = [
                        f"{a.get('tool_name', 'Action')}: {a.get('rationale', '')}"
                        for a in actions
                    ] if actions else template["steps"]

                    level_result = {
                        "level": level,
                        "anomaly_count": len(level_anomalies),
                        "anomalies": level_anomalies,
                        "root_cause": rca.get("root_cause", template["root_cause"]),
                        "confidence": rca.get("confidence_score", template["confidence"]),
                        "resolution_summary": (
                            f"Resolved {len(level_anomalies)} {level}-level anomalies. "
                            f"Root cause: {rca.get('root_cause', 'identified')[:100]}. "
                            f"{len(resolution_steps)} remediation steps generated."
                        ),
                        "resolution_steps": resolution_steps,
                        "llm_generated": True,
                    }
                except Exception as exc:
                    logger.error("Level pipeline failed for %s: %s", level, exc)
                    level_result = {
                        "level": level,
                        "anomaly_count": len(level_anomalies),
                        "error": str(exc),
                        "root_cause": template["root_cause"],
                        "confidence": template["confidence"],
                        "resolution_summary": (
                            f"Template resolution for {len(level_anomalies)} "
                            f"{level}-level anomalies (LLM unavailable: {exc})."
                        ),
                        "resolution_steps": template["steps"],
                        "llm_generated": False,
                    }

            result["per_level"][level] = level_result
            result["level_summary"].append(level_result)

        return result

    # ── Scenario Processing ──────────────────────────────────────

    def process_scenario(self, scenario: dict) -> dict:
        """Process a pre-built demo scenario.

        Demo scenarios are dicts with keys like ``anomalies``, ``logs``,
        ``metrics``, ``name``, and ``description``.

        Args:
            scenario: Pre-built scenario dict.

        Returns:
            Full pipeline result (same as :meth:`process_incident`),
            enriched with the scenario metadata.  If the scenario already
            contains ``anomalies`` (from a pre-run detector) those are used;
            otherwise anomaly detection must be done before calling.
        """
        logger.info(
            "Processing scenario: %s", scenario.get("name", "unnamed"),
        )
        result = self.process_incident(
            anomalies=scenario.get("anomalies", []),
            logs=scenario.get("logs", []),
            metrics=scenario.get("metrics", []),
        )
        result["scenario_name"] = scenario.get("name", "unnamed")
        result["scenario_description"] = scenario.get("description", "")
        # Surface agent outputs under short aliases for downstream consumers
        result["triage"] = result.get("triage_result", {})
        result["rca"] = result.get("rca_result", {})
        result["remediation"] = result.get("remediation_result", {})
        return result

    # ── Self-Critique / Reflection ───────────────────────────────

    CRITIQUE_PROMPT = """Review this RCA analysis. Identify gaps or weak evidence. Either confirm or refine.
triage: sev={sev} cat={cat} impact={impact[:80]}
rca: cause={rc} conf={conf}
evidence: {ev[:250]}
Return JSON: {{"confirmed":true/false,"refined_confidence":0-1,"refined_reasoning":"summary","gaps":["gap"],"suggestions":["suggestion"]}}"""

    def _critique_analysis(self, anomalies, triage_result, rca_result) -> dict:
        ev = "; ".join(str(e)[:60] for e in rca_result.get("evidence_chain", [])[:3])
        prompt = self.CRITIQUE_PROMPT.format(
            sev=triage_result.get("severity","?"),
            cat=triage_result.get("category","?"),
            impact=triage_result.get("impact_assessment",""),
            rc=rca_result.get("root_cause","?"),
            conf=rca_result.get("confidence_score",0),
            ev=ev,
        )
        messages = [
            {"role": "system", "content": "You are a critique agent. Respond ONLY with the JSON object."},
            {"role": "user", "content": prompt},
        ]
        raw = self.triage_agent._call_llm(messages)
        try:
            result = json.loads(raw)
        except Exception:
            result = {"confirmed": True, "refined_confidence": rca_result.get("confidence_score", 0.5), "refined_reasoning": rca_result.get("reasoning_summary", ""), "gaps": [], "suggestions": []}
        result.setdefault("refined_confidence", rca_result.get("confidence_score", 0.5))
        result.setdefault("refined_reasoning", rca_result.get("reasoning_summary", ""))
        return result

    # ── Metrics ──────────────────────────────────────────────────

    def get_all_metrics(self) -> dict:
        """Aggregate execution metrics from all agents.

        Returns:
            Dict with per-agent metrics and pipeline-level totals.
        """
        agent_metrics = {
            "triage": self.triage_agent.get_metrics(),
            "rca": self.rca_agent.get_metrics(),
            "remediation": self.remediation_agent.get_metrics(),
            "reporting": self.reporting_agent.get_metrics(),
        }

        total_calls = sum(m["total_calls"] for m in agent_metrics.values())
        total_tokens = sum(m["total_tokens"] for m in agent_metrics.values())
        total_errors = sum(m["errors"] for m in agent_metrics.values())

        return {
            "agents": agent_metrics,
            "totals": {
                "total_calls": total_calls,
                "total_tokens": total_tokens,
                "total_errors": total_errors,
                "pipelines_executed": len(self._pipeline_log),
            },
        }

    def get_pipeline_history(self) -> List[dict]:
        """Return summary of all pipelines executed in this session.

        Returns:
            List of pipeline summary dicts.
        """
        return list(self._pipeline_log)
