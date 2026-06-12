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

RCA_SYSTEM_PROMPT = """You are an RCA agent. Analyze anomalies and triage data. BE CONCISE. Output ONLY valid JSON:
{
  "root_cause": "specific root cause statement",
  "root_cause_category": "infrastructure or application or network or security or database or storage",
  "evidence_chain": ["brief evidence 1", "brief evidence 2"],
  "confidence_score": 0.0 to 1.0,
  "related_runbook_id": null,
  "contributing_factors": ["factor"],
  "timeline_of_events": [{"timestamp": "T", "event": "E"}],
  "affected_components": ["component"],
  "blast_radius": "short impact description",
  "reasoning_summary": "1 sentence explaining logic"
}
Distinguish symptoms from causes. No prose. No markdown. ONLY valid JSON."""


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
        from config import MAX_RAG_CHARS
        tri = triage_result
        parts = [
            f"triage sev={tri.get('severity','?')} cat={tri.get('category','?')} impact={str(tri.get('impact_assessment',''))[:60]}",
            "anomalies:"
        ]
        for a in anomalies:
            desc = a.get('description','')[:40].replace(',',' ')
            parts.append(f"  {a.get('severity','?')} {a.get('type','?')} {a.get('source','?')} \"{desc}\" conf={a.get('confidence',0)}")
        if runbook_context:
            parts.append(f"runbooks: {runbook_context[:MAX_RAG_CHARS]}")
        parts.append("Analyze above. Output root cause as valid JSON per system prompt. ONLY JSON.")
        return "\n".join(parts)

    def _validate_result(self, result: dict) -> dict:
        """Ensure all required fields are present with sensible defaults."""
        # If parsing failed, preserve the error and provide defaults
        if "error" in result and "_partial" not in result:
            defaults = RCAAgent._default_result()
            defaults["error"] = result["error"]
            defaults["raw"] = result.get("raw", "")
            return defaults

        timeline = result.get("timeline_of_events", [])
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
