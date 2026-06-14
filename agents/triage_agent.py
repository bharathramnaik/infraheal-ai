"""
InfraHeal AI — Triage Agent
============================
Classifies incident severity (P1–P4) and category based on anomaly data.
Returns structured JSON with severity, category, impact assessment,
affected services, and urgency reasoning.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_agent import BaseAgent

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEVERITY_LEVELS, INCIDENT_CATEGORIES

logger = logging.getLogger(__name__)

TRIAGE_SYSTEM_PROMPT = """You triage incidents by severity (P1-P4) and category. Read anomalies, assign severity and category, output ONLY a SINGLE JSON object (NOT an array):

{"severity":"P1-P4","severity_label":"Critical/High/Medium/Low","category":"infrastructure|application|network|security|database|storage","impact_assessment":"2-3 sentence impact","affected_services":["svc1"],"urgency_reasoning":"why this severity","confidence":0-1,"escalation_needed":bool,"sla_minutes":int}

P1=production down/data loss, P2=major degradation, P3=partial, P4=minor. Category must be one of the six listed. No prose, no markdown, no arrays. ONLY a single JSON object with the keys above."""


class TriageAgent(BaseAgent):
    """Classifies incidents by severity and category.

    Analyses anomaly data to determine the appropriate severity level
    (P1–P4), assigns a category, assesses business impact, and
    identifies affected services.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialise the Triage Agent.

        Args:
            client: Pre-configured OpenAI client (optional).
            model_name: Model identifier (optional, falls back to config).
        """
        super().__init__(
            name="triage_agent",
            role="Incident Triage & Classification",
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            client=client,
            model_name=model_name,
        )

    def run(self, context: dict) -> dict:
        """Classify the incident based on anomaly data.

        Args:
            context: Must contain key ``anomalies`` — a list of anomaly
                dicts.  Optionally includes ``logs`` and ``metrics`` for
                additional signal.

        Returns:
            Dict with keys: severity, severity_label, category,
            impact_assessment, affected_services, urgency_reasoning,
            confidence, escalation_needed, sla_minutes.
        """
        anomalies: List[Dict[str, Any]] = context.get("anomalies", [])
        if not anomalies:
            self.logger.warning("Triage called with no anomalies — returning P4/Low default")
            return self._default_result()

        # Build a focused user prompt from anomaly data
        user_content = self._format_anomalies_for_prompt(anomalies, context)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = self._call_llm(messages)
        result = self._parse_json_response(raw)

        # Validate & enrich the result
        result = self._validate_result(result)
        self.logger.info(
            "Triage result: %s (%s) — %s | confidence=%.2f",
            result["severity"], result["severity_label"],
            result["category"], result.get("confidence", 0),
        )
        return result

    # ── Internal Helpers ─────────────────────────────────────────

    def _format_anomalies_for_prompt(self, anomalies: List[dict], context: dict) -> str:
        lines = ["sev type src desc conf"]
        for a in anomalies:
            desc = a.get('description','')[:60].replace(',',' ')
            lines.append(f"{a.get('severity','?')} {a.get('type','?')} {a.get('source','?')} \"{desc}\" {a.get('confidence',0)}")
        logs = context.get("logs", [])
        if logs:
            from config import MAX_CONTEXT_LOGS, MAX_CONTEXT_LOGS_CHARS
            log_lines = []
            char_count = 0
            for lg in logs[:MAX_CONTEXT_LOGS]:
                entry = f"{lg.get('level','?')} {lg.get('message','')[:80]}"
                char_count += len(entry)
                if char_count > MAX_CONTEXT_LOGS_CHARS:
                    break
                log_lines.append(entry)
            if log_lines:
                lines.append("logs: " + " | ".join(log_lines))
        metrics = context.get("metrics", [])
        if metrics:
            m = metrics[-1] if metrics else {}
            lines.append(f"metrics cpu={m.get('cpu_percent','?')} mem={m.get('memory_percent','?')} disk={m.get('disk_percent','?')} lat={m.get('request_latency_ms','?')} err={m.get('error_rate','?')}")
        lines.append("Classify incident severity P1-P4 and category. Return ONLY the JSON.")
        return "\n".join(lines)

    def _validate_result(self, result: dict) -> dict:
        """Ensure all required fields exist with valid values."""
        if "error" in result and "_partial" not in result:
            return {
                "severity": "P4",
                "severity_label": "Low",
                "category": "infrastructure",
                "impact_assessment": "No anomalies detected. System appears healthy.",
                "affected_services": [],
                "urgency_reasoning": "No anomalies were provided for analysis.",
                "confidence": 1.0,
                "escalation_needed": False,
                "sla_minutes": 1440,
                "error": result["error"],
                "raw": result.get("raw", ""),
            }

        severity = result.get("severity", "P3")
        if severity not in SEVERITY_LEVELS:
            severity = "P3"
        severity_info = SEVERITY_LEVELS[severity]

        category = result.get("category", "infrastructure")
        if category not in INCIDENT_CATEGORIES:
            category = "infrastructure"

        return {
            "severity": severity,
            "severity_label": severity_info["label"],
            "category": category,
            "impact_assessment": result.get("impact_assessment", "Impact assessment unavailable."),
            "affected_services": result.get("affected_services", []),
            "urgency_reasoning": result.get("urgency_reasoning", "No reasoning provided."),
            "confidence": min(max(float(result.get("confidence", 0.5)), 0.0), 1.0),
            "escalation_needed": result.get("escalation_needed", severity in ("P1", "P2")),
            "sla_minutes": severity_info["sla_minutes"],
        }

    @staticmethod
    def _default_result() -> dict:
        """Return a safe default triage result when no anomalies exist."""
        return {
            "severity": "P4",
            "severity_label": "Low",
            "category": "infrastructure",
            "impact_assessment": "No anomalies detected. System appears healthy.",
            "affected_services": [],
            "urgency_reasoning": "No anomalies were provided for analysis.",
            "confidence": 1.0,
            "escalation_needed": False,
            "sla_minutes": 1440,
        }
