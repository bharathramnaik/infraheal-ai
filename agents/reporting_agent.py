"""
InfraHeal AI — Reporting Agent
================================
Generates comprehensive, professional incident reports by
synthesising outputs from all upstream agents (triage, RCA,
remediation).
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

REPORTING_SYSTEM_PROMPT = """You are the **Reporting Agent** in the InfraHeal AI incident management system.

## Your Mission
Generate a comprehensive, professional incident report suitable for engineering leadership and post-mortem reviews. Synthesise all data from triage, root cause analysis, and remediation into a single cohesive document.

## Report Quality Standards
1. **Executive Summary**: 2-3 sentences a VP can understand. Lead with business impact.
2. **Timeline**: Chronological, precise, and complete.
3. **Root Cause**: Technically accurate but readable by non-experts.
4. **Actions**: Clear, with status (completed/pending/recommended).
5. **Prevention**: Specific, actionable recommendations — not generic platitudes.
6. **Tone**: Professional, factual, blameless (SRE culture).

## Output Schema (strict)
```json
{
  "incident_id": "<INC-YYYYMMDD-XXXX format>",
  "title": "<concise, descriptive incident title>",
  "severity": "<P1|P2|P3|P4>",
  "status": "resolved|mitigated|investigating",
  "executive_summary": "<2-3 sentence summary for leadership>",
  "timeline": [
    {"timestamp": "<ISO timestamp or relative>", "event": "<what happened>"}
  ],
  "root_cause_summary": "<clear explanation of root cause>",
  "actions_taken": [
    {
      "action": "<what was done>",
      "status": "completed|pending|recommended",
      "result": "<outcome>"
    }
  ],
  "business_impact": "<description of impact on users/revenue/SLA>",
  "prevention_recommendations": [
    "<specific, actionable recommendation 1>",
    "<recommendation 2>"
  ],
  "postmortem_notes": "<additional observations and lessons learned>",
  "metrics_summary": {
    "time_to_detect_minutes": "<number or estimate>",
    "time_to_resolve_minutes": "<number or estimate>",
    "affected_users_estimate": "<percentage or count>"
  },
  "tags": ["<tag1>", "<tag2>"]
}
```

Respond ONLY with the JSON object. No markdown, no explanation outside the JSON."""


class ReportingAgent(BaseAgent):
    """Generates comprehensive incident reports.

    Combines triage classification, root cause analysis, and
    remediation results into a professional post-incident report
    suitable for leadership review and postmortem archives.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialise the Reporting Agent.

        Args:
            client: Pre-configured OpenAI client (optional).
            model_name: Model identifier (optional).
        """
        super().__init__(
            name="reporting_agent",
            role="Incident Report Generation",
            system_prompt=REPORTING_SYSTEM_PROMPT,
            client=client,
            model_name=model_name,
        )

    def run(self, context: dict) -> dict:
        """Generate a full incident report.

        Args:
            context: Must contain:
                - ``anomalies``: list of anomaly dicts.
                - ``triage_result``: output from TriageAgent.
                - ``rca_result``: output from RCAAgent.
                - ``remediation_result``: output from RemediationAgent.
                Optionally:
                - ``execution_results``: list of action execution dicts.

        Returns:
            Dict with incident_id, title, executive_summary, timeline,
            root_cause_summary, actions_taken, business_impact,
            prevention_recommendations, postmortem_notes, metrics_summary,
            tags.
        """
        anomalies = context.get("anomalies", [])
        triage_result = context.get("triage_result", {})
        rca_result = context.get("rca_result", {})
        remediation_result = context.get("remediation_result", {})
        execution_results = context.get("execution_results", [])

        user_content = self._format_report_prompt(
            anomalies, triage_result, rca_result,
            remediation_result, execution_results,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = self._call_llm(messages)
        result = self._parse_json_response(raw)
        result = self._validate_result(result, triage_result)

        self.logger.info(
            "Report generated: %s — '%s' [%s]",
            result["incident_id"], result["title"], result["status"],
        )
        return result

    # ── Internal Helpers ─────────────────────────────────────────

    def _format_report_prompt(
        self,
        anomalies: List[dict],
        triage_result: dict,
        rca_result: dict,
        remediation_result: dict,
        execution_results: List[dict],
    ) -> str:
        """Assemble all agent outputs into a single user prompt."""
        sections: List[str] = []

        # ── Triage ──
        sections.append(
            "## Triage Classification\n"
            f"- **Severity**: {triage_result.get('severity', 'N/A')} ({triage_result.get('severity_label', '')})\n"
            f"- **Category**: {triage_result.get('category', 'N/A')}\n"
            f"- **Impact**: {triage_result.get('impact_assessment', 'N/A')}\n"
            f"- **Affected Services**: {', '.join(triage_result.get('affected_services', []))}\n"
            f"- **SLA**: {triage_result.get('sla_minutes', 'N/A')} minutes\n"
            f"- **Urgency**: {triage_result.get('urgency_reasoning', 'N/A')}\n"
        )

        # ── Anomalies ──
        sections.append("## Anomalies\n")
        for idx, anomaly in enumerate(anomalies, 1):
            sections.append(
                f"### Anomaly {idx}\n"
                f"- Type: {anomaly.get('type', '?')} | Source: {anomaly.get('source', '?')}\n"
                f"- Severity: {anomaly.get('severity', '?')} | Confidence: {anomaly.get('confidence', '?')}\n"
                f"- Timestamp: {anomaly.get('timestamp', '?')}\n"
                f"- Description: {anomaly.get('description', '')}\n"
            )

        # ── RCA ──
        sections.append(
            "## Root Cause Analysis\n"
            f"- **Root Cause**: {rca_result.get('root_cause', 'Unknown')}\n"
            f"- **Confidence**: {rca_result.get('confidence_score', 0)}\n"
            f"- **Blast Radius**: {rca_result.get('blast_radius', 'Unknown')}\n"
        )
        evidence = rca_result.get("evidence_chain", [])
        if evidence:
            sections.append("**Evidence Chain**:\n" + "\n".join(f"  - {e}" for e in evidence) + "\n")

        timeline = rca_result.get("timeline_of_events", [])
        if timeline:
            sections.append("**Event Timeline**:")
            for entry in timeline:
                ts = entry.get("timestamp", "?") if isinstance(entry, dict) else "?"
                ev = entry.get("event", str(entry)) if isinstance(entry, dict) else str(entry)
                sections.append(f"  - [{ts}] {ev}")
            sections.append("")

        factors = rca_result.get("contributing_factors", [])
        if factors:
            sections.append("**Contributing Factors**:\n" + "\n".join(f"  - {f}" for f in factors) + "\n")

        # ── Remediation ──
        actions = remediation_result.get("recommended_actions", [])
        sections.append(
            "## Remediation Plan\n"
            f"- **Execution Order**: {remediation_result.get('execution_order', 'sequential')}\n"
            f"- **Rollback Plan**: {remediation_result.get('rollback_plan', 'N/A')}\n"
            f"- **Estimated Resolution**: {remediation_result.get('estimated_resolution_time', 'N/A')}\n"
        )
        if actions:
            sections.append("**Planned Actions**:")
            for act in actions:
                sections.append(
                    f"  - Step {act.get('step', '?')}: {act.get('tool_name', '?')} — "
                    f"{act.get('rationale', '')} (risk: {act.get('risk_level', '?')})"
                )
            sections.append("")

        # ── Execution Results (if available) ──
        if execution_results:
            sections.append("## Execution Results\n")
            for res in execution_results:
                sections.append(
                    f"- Step {res.get('step', '?')}: {res.get('tool_name', '?')} → "
                    f"**{res.get('status', '?')}** | {res.get('message', '')}"
                )
            sections.append("")

        sections.append(
            "Generate a comprehensive incident report from all the data above. "
            "Return ONLY the JSON object matching the schema in your instructions."
        )
        return "\n".join(sections)

    def _validate_result(self, result: dict, triage_result: dict) -> dict:
        """Ensure all report fields are present with sensible defaults."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")
        short_id = uuid.uuid4().hex[:4].upper()
        default_id = f"INC-{date_str}-{short_id}"

        # Normalise timeline
        raw_timeline = result.get("timeline", [])
        validated_timeline: List[Dict[str, str]] = []
        for entry in raw_timeline:
            if isinstance(entry, dict):
                validated_timeline.append({
                    "timestamp": str(entry.get("timestamp", "unknown")),
                    "event": str(entry.get("event", "unknown event")),
                })
            elif isinstance(entry, str):
                validated_timeline.append({"timestamp": "unknown", "event": entry})

        # Normalise actions_taken
        raw_actions = result.get("actions_taken", [])
        validated_actions: List[Dict[str, str]] = []
        for act in raw_actions:
            if isinstance(act, dict):
                validated_actions.append({
                    "action": str(act.get("action", "Unknown action")),
                    "status": str(act.get("status", "recommended")),
                    "result": str(act.get("result", "Pending")),
                })
            elif isinstance(act, str):
                validated_actions.append({"action": act, "status": "recommended", "result": "Pending"})

        severity = triage_result.get("severity", result.get("severity", "P3"))

        return {
            "incident_id": result.get("incident_id", default_id),
            "title": result.get("title", "Infrastructure Incident"),
            "severity": severity,
            "status": result.get("status", "investigating"),
            "executive_summary": result.get("executive_summary", "Incident report generated by InfraHeal AI."),
            "timeline": validated_timeline,
            "root_cause_summary": result.get("root_cause_summary", "Root cause analysis pending."),
            "actions_taken": validated_actions,
            "business_impact": result.get("business_impact", "Impact assessment pending."),
            "prevention_recommendations": result.get("prevention_recommendations", []),
            "postmortem_notes": result.get("postmortem_notes", "No additional notes."),
            "metrics_summary": result.get("metrics_summary", {
                "time_to_detect_minutes": "N/A",
                "time_to_resolve_minutes": "N/A",
                "affected_users_estimate": "N/A",
            }),
            "tags": result.get("tags", [severity, triage_result.get("category", "infrastructure")]),
            "generated_at": now.isoformat(),
        }
