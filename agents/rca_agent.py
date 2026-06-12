"""
InfraHeal AI — Root Cause Analysis Agent
==========================================
Correlates anomaly evidence with runbook knowledge to identify the
most probable root cause of an incident.  Returns a structured
analysis including evidence chain, confidence score, and timeline.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

RCA_SYSTEM_PROMPT = """You are the **Root Cause Analysis (RCA) Agent** in the InfraHeal AI incident management system.

## Your Mission
Given anomaly data, a triage classification, and relevant runbook excerpts, perform deep root-cause analysis. Correlate all evidence, build a causal chain, and identify the single most likely root cause with supporting evidence.

## Analysis Methodology
1. **Evidence Correlation**: Cross-reference anomaly timestamps, affected services, and metric patterns.
2. **Causal Chain Construction**: Build a logical sequence from initial trigger → propagation → observed symptoms.
3. **Runbook Matching**: Compare symptoms against known runbook patterns to identify matching root causes.
4. **Confidence Assessment**: Rate your confidence based on evidence strength and pattern match quality.

## Reasoning Guidelines
- Distinguish between **symptoms** (what we observe) and **causes** (what triggered them).
- A CPU spike is a symptom; a memory leak in the Java heap causing excessive GC is a cause.
- Network latency spikes can be symptoms of upstream service failures.
- Always consider cascading failure chains: A → B → C.
- If runbook context is available and matches, boost confidence; if no match, note it but still reason from first principles.

## Output Schema (strict)
```json
{
  "root_cause": "<clear, specific root cause statement>",
  "root_cause_category": "<infrastructure|application|network|security|database|storage>",
  "evidence_chain": [
    "<evidence item 1 supporting the root cause>",
    "<evidence item 2>",
    "..."
  ],
  "confidence_score": 0.0-1.0,
  "related_runbook_id": "<runbook ID if matched, else null>",
  "contributing_factors": [
    "<factor 1 that worsened or enabled the incident>",
    "..."
  ],
  "timeline_of_events": [
    {"timestamp": "<ISO timestamp or relative>", "event": "<what happened>"},
    {"timestamp": "...", "event": "..."}
  ],
  "affected_components": ["<component1>", "<component2>"],
  "blast_radius": "<description of total impact scope>",
  "reasoning_summary": "<paragraph explaining the RCA logic>"
}
```

Respond ONLY with the JSON object. No markdown, no explanation outside the JSON."""


class RCAAgent(BaseAgent):
    """Performs root cause analysis by correlating evidence and runbooks.

    Combines anomaly data, triage context, and RAG-retrieved runbook
    excerpts to identify the most probable root cause, build an evidence
    chain, and produce a chronological timeline.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialise the RCA Agent.

        Args:
            client: Pre-configured OpenAI client (optional).
            model_name: Model identifier (optional).
        """
        super().__init__(
            name="rca_agent",
            role="Root Cause Analysis",
            system_prompt=RCA_SYSTEM_PROMPT,
            client=client,
            model_name=model_name,
        )

    def run(self, context: dict) -> dict:
        """Perform root cause analysis.

        Args:
            context: Must contain:
                - ``anomalies``: list of anomaly dicts.
                - ``triage_result``: output from TriageAgent.
                Optionally:
                - ``runbook_context``: formatted string of relevant
                  runbook excerpts (from RAG).

        Returns:
            Dict with root_cause, evidence_chain, confidence_score,
            related_runbook_id, contributing_factors, timeline_of_events,
            affected_components, blast_radius, reasoning_summary.
        """
        anomalies = context.get("anomalies", [])
        triage_result = context.get("triage_result", {})
        runbook_context = context.get("runbook_context", "")

        if not anomalies:
            self.logger.warning("RCA called with no anomalies")
            return self._default_result()

        user_content = self._format_rca_prompt(anomalies, triage_result, runbook_context)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = self._call_llm(messages)
        result = self._parse_json_response(raw)
        result = self._validate_result(result)

        self.logger.info(
            "RCA complete: '%s' (confidence=%.2f, runbook=%s)",
            result["root_cause"][:80],
            result["confidence_score"],
            result.get("related_runbook_id", "none"),
        )
        return result

    # ── Internal Helpers ─────────────────────────────────────────

    def _format_rca_prompt(
        self,
        anomalies: List[dict],
        triage_result: dict,
        runbook_context: str,
    ) -> str:
        """Assemble the user prompt for root cause analysis."""
        sections: List[str] = []

        # Triage summary
        sections.append(
            "## Triage Classification\n"
            f"- **Severity**: {triage_result.get('severity', 'N/A')} ({triage_result.get('severity_label', '')})\n"
            f"- **Category**: {triage_result.get('category', 'N/A')}\n"
            f"- **Impact**: {triage_result.get('impact_assessment', 'N/A')}\n"
            f"- **Affected Services**: {', '.join(triage_result.get('affected_services', []))}\n"
            f"- **Urgency Reasoning**: {triage_result.get('urgency_reasoning', 'N/A')}\n"
        )

        # Anomaly details
        sections.append("## Anomaly Evidence\n")
        for idx, anomaly in enumerate(anomalies, 1):
            sections.append(
                f"### Anomaly {idx}\n"
                f"- **ID**: {anomaly.get('id', 'N/A')}\n"
                f"- **Type**: {anomaly.get('type', 'unknown')}\n"
                f"- **Severity**: {anomaly.get('severity', 'unknown')}\n"
                f"- **Source**: {anomaly.get('source', 'unknown')}\n"
                f"- **Timestamp**: {anomaly.get('timestamp', 'N/A')}\n"
                f"- **Description**: {anomaly.get('description', '')}\n"
                f"- **Confidence**: {anomaly.get('confidence', 'N/A')}\n"
            )
            evidence = anomaly.get("evidence", [])
            if evidence:
                sections.append("- **Evidence**:\n" + "\n".join(f"  - {e}" for e in evidence))

            related_logs = anomaly.get("related_logs", [])
            if related_logs:
                log_lines = "\n".join(f"  - {lg}" for lg in related_logs[:10])
                sections.append(f"- **Related Logs**:\n{log_lines}")

            related_metrics = anomaly.get("related_metrics", [])
            if related_metrics:
                metric_lines = "\n".join(f"  - {m}" for m in related_metrics[:10])
                sections.append(f"- **Related Metrics**:\n{metric_lines}")
            sections.append("")

        # Runbook context from RAG
        if runbook_context:
            sections.append(f"## Relevant Runbook Knowledge\n{runbook_context}\n")
        else:
            sections.append(
                "## Relevant Runbook Knowledge\n"
                "No matching runbooks found. Reason from first principles.\n"
            )

        sections.append(
            "Analyse all evidence above. Identify the root cause, build "
            "the evidence chain, and return ONLY the JSON object matching "
            "the schema in your instructions."
        )
        return "\n".join(sections)

    def _validate_result(self, result: dict) -> dict:
        """Ensure all required fields are present with sensible defaults."""
        timeline = result.get("timeline_of_events", [])
        # Normalise timeline entries
        validated_timeline: List[Dict[str, str]] = []
        for entry in timeline:
            if isinstance(entry, dict):
                validated_timeline.append({
                    "timestamp": str(entry.get("timestamp", "unknown")),
                    "event": str(entry.get("event", "unknown event")),
                })
            elif isinstance(entry, str):
                validated_timeline.append({"timestamp": "unknown", "event": entry})

        return {
            "root_cause": result.get("root_cause", "Unable to determine root cause from available evidence."),
            "root_cause_category": result.get("root_cause_category", "infrastructure"),
            "evidence_chain": result.get("evidence_chain", []),
            "confidence_score": min(max(float(result.get("confidence_score", 0.3)), 0.0), 1.0),
            "related_runbook_id": result.get("related_runbook_id"),
            "contributing_factors": result.get("contributing_factors", []),
            "timeline_of_events": validated_timeline,
            "affected_components": result.get("affected_components", []),
            "blast_radius": result.get("blast_radius", "Unknown"),
            "reasoning_summary": result.get("reasoning_summary", "No reasoning summary available."),
        }

    @staticmethod
    def _default_result() -> dict:
        """Return a safe default when no anomalies are provided."""
        return {
            "root_cause": "No anomalies provided for analysis.",
            "root_cause_category": "infrastructure",
            "evidence_chain": [],
            "confidence_score": 0.0,
            "related_runbook_id": None,
            "contributing_factors": [],
            "timeline_of_events": [],
            "affected_components": [],
            "blast_radius": "None — no anomalies detected.",
            "reasoning_summary": "RCA was invoked without anomaly data.",
        }
