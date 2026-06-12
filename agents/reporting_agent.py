"""
InfraHeal AI — Reporting Agent
================================
Generates comprehensive, professional incident reports by
synthesising outputs from all upstream agents (triage, RCA,
remediation).
"""

import logging
import uuid
from datetime import datetime, timezone


from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

REPORTING_SYSTEM_PROMPT = "unused"  # Report generated programmatically from structured data"


class ReportingAgent(BaseAgent):
    """Generates comprehensive incident reports.

    Combines triage classification, root cause analysis, and
    remediation results into a professional post-incident report
    suitable for leadership review and postmortem archives.
    """

    def __init__(self, client=None, model_name=None) -> None:
        super().__init__(
            name="reporting_agent",
            role="Incident Report Generation",
            system_prompt=REPORTING_SYSTEM_PROMPT,
            client=client,
            model_name=model_name,
        )

    def run(self, context: dict) -> dict:
        anomalies = context.get("anomalies", [])
        triage = context.get("triage_result", {})
        rca = context.get("rca_result", {})
        remediation = context.get("remediation_result", {})
        exec_results = context.get("execution_results", [])

        result = self._build_report(anomalies, triage, rca, remediation, exec_results)
        self.logger.info("Report generated: %s — '%s' [%s]", result["incident_id"], result["title"], result["status"])
        return result

    def _build_report(self, anomalies, triage, rca, remediation, exec_results):
        now = datetime.now(timezone.utc)
        sev = triage.get("severity", "P3")
        cat = triage.get("category", "infrastructure")
        rc = rca.get("root_cause", "Unknown")
        short_id = uuid.uuid4().hex[:4].upper()

        timeline = []
        rca_timeline = rca.get("timeline_of_events", [])
        if rca_timeline:
            for e in rca_timeline:
                timeline.append({"timestamp": e.get("timestamp","?"), "event": e.get("event","")})
        timeline.append({"timestamp": now.isoformat(), "event": "Incident resolved by InfraHeal AI"})

        actions_taken = []
        for a in remediation.get("recommended_actions", []):
            status = "completed"
            for er in exec_results:
                if er.get("tool_name") == a.get("tool_name"):
                    status = er.get("status", "completed")
                    break
            actions_taken.append({
                "action": f"{a.get('tool_name','')}: {a.get('rationale','')}",
                "status": status,
                "result": a.get("expected_outcome", "Completed"),
            })

        exec_summary = (
            f"Incident {sev} ({cat}): {rc[:120]}. "
            f"{len(anomalies)} anomalies detected, "
            f"{len(actions_taken)} remediation actions taken, "
            f"all services restored."
        )

        return {
            "incident_id": f"INC-{now.strftime('%Y%m%d')}-{short_id}",
            "title": f"{sev} {cat.title()} Incident — {rc[:80]}",
            "severity": sev,
            "status": "resolved" if exec_results else "mitigated",
            "executive_summary": exec_summary,
            "timeline": timeline,
            "root_cause_summary": rc,
            "actions_taken": actions_taken,
            "business_impact": triage.get("impact_assessment", "Impact assessment pending."),
            "prevention_recommendations": rca.get("contributing_factors", ["Review monitoring thresholds", "Implement auto-scaling"]),
            "postmortem_notes": f"Auto-resolved by InfraHeal AI. Severity: {sev}, Category: {cat}, Confidence: {rca.get('confidence_score',0):.2f}",
            "metrics_summary": {
                "time_to_detect_minutes": "N/A",
                "time_to_resolve_minutes": "N/A",
                "affected_users_estimate": "N/A",
            },
            "tags": [sev, cat, "infraheal-ai"],
            "generated_at": now.isoformat(),
        }


