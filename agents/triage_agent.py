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

TRIAGE_SYSTEM_PROMPT = """You are the **Triage Agent** in the InfraHeal AI incident management system.

## Your Mission
Analyse the provided anomaly data and classify the incident accurately. You MUST return a well-structured JSON response — no prose, no markdown, just raw JSON.

## Classification Rules

### Severity Levels
- **P1 (Critical)**: Production down, data loss risk, security breach, >50% users affected, SLA ≤15 min.
- **P2 (High)**: Major degradation, single critical service down, >25% users affected, SLA ≤60 min.
- **P3 (Medium)**: Partial degradation, non-critical service issues, <25% users affected, SLA ≤4 hrs.
- **P4 (Low)**: Minor issue, cosmetic, no user impact, SLA ≤24 hrs.

### Categories
Choose exactly one: infrastructure, application, network, security, database, storage.

### Decision Factors
1. **Breadth**: How many services/hosts are affected?
2. **Depth**: How severe are the anomalies (confidence, z-scores)?
3. **Duration**: How long has the anomaly been active?
4. **Cascading risk**: Could this trigger downstream failures?
5. **Data integrity**: Is data loss possible?

## Output Schema (strict)
```json
{
  "severity": "P1|P2|P3|P4",
  "severity_label": "Critical|High|Medium|Low",
  "category": "<one of: infrastructure, application, network, security, database, storage>",
  "impact_assessment": "<2-3 sentence impact summary>",
  "affected_services": ["<service1>", "<service2>"],
  "urgency_reasoning": "<detailed paragraph explaining why you chose this severity>",
  "confidence": 0.0-1.0,
  "escalation_needed": true|false,
  "sla_minutes": <number>
}
```

Respond ONLY with the JSON object. No explanation outside the JSON."""


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
        """Build a concise prompt summarising the anomalies."""
        sections: List[str] = ["## Anomalies Detected\n"]

        for idx, anomaly in enumerate(anomalies, 1):
            sections.append(
                f"### Anomaly {idx}\n"
                f"- **ID**: {anomaly.get('id', 'N/A')}\n"
                f"- **Type**: {anomaly.get('type', 'unknown')}\n"
                f"- **Severity**: {anomaly.get('severity', 'unknown')}\n"
                f"- **Source**: {anomaly.get('source', 'unknown')}\n"
                f"- **Timestamp**: {anomaly.get('timestamp', 'N/A')}\n"
                f"- **Description**: {anomaly.get('description', 'No description')}\n"
                f"- **Confidence**: {anomaly.get('confidence', 'N/A')}\n"
            )
            evidence = anomaly.get("evidence", [])
            if evidence:
                sections.append("- **Evidence**:\n" + "\n".join(f"  - {e}" for e in evidence))
            sections.append("")

        # Append optional context summaries
        logs = context.get("logs", [])
        if logs:
            log_summary = "\n".join(
                f"  [{lg.get('level', '?')}] {lg.get('service', '?')}: {lg.get('message', '')[:120]}"
                for lg in logs[:15]
            )
            sections.append(f"## Recent Logs (sample)\n{log_summary}\n")

        metrics = context.get("metrics", [])
        if metrics:
            latest = metrics[-1] if metrics else {}
            sections.append(
                f"## Latest Metrics Snapshot\n"
                f"- CPU: {latest.get('cpu_percent', 'N/A')}%\n"
                f"- Memory: {latest.get('memory_percent', 'N/A')}%\n"
                f"- Disk: {latest.get('disk_percent', 'N/A')}%\n"
                f"- Latency: {latest.get('request_latency_ms', 'N/A')} ms\n"
                f"- Error Rate: {latest.get('error_rate', 'N/A')}\n"
            )

        sections.append(
            "Classify this incident. Return ONLY the JSON object "
            "matching the schema in your instructions."
        )
        return "\n".join(sections)

    def _validate_result(self, result: dict) -> dict:
        """Ensure all required fields exist with valid values."""
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
