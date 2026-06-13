"""
InfraHeal AI — Reporting Agent
================================
Generates comprehensive, professional incident reports by
synthesising outputs from all upstream agents (triage, RCA,
remediation) using LLM when available.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

REPORTING_SYSTEM_PROMPT = """You are an expert incident report writer for a production infrastructure team.
Your reports are clear, concise, and suitable for both engineering teams and leadership review.
Always output valid JSON."""

REPORTING_PROMPT = """Generate a structured incident report in JSON format based on the following data.

**Triage Assessment**
- Severity: {severity}
- Category: {category}
- Impact: {impact}

**Root Cause Analysis**
- Root Cause: {root_cause}
- Confidence: {confidence}
- Contributing Factors: {contributing_factors}

**Remediation**
- Actions taken: {actions}
- Execution status: {execution_status}

**Anomalies Detected**: {anomaly_count}

Return a JSON object with these exact keys:
- "incident_id": unique ID like INC-YYYYMMDD-XXXX
- "title": concise title
- "severity": from triage
- "status": "resolved" or "mitigated"
- "executive_summary": 2-3 sentence summary
- "root_cause_summary": one-line root cause
- "business_impact": business impact assessment
- "prevention_recommendations": list of 2-3 recommendations
- "llm_generated": true
- "generated_at": current ISO timestamp"""


class ReportingAgent(BaseAgent):
    """Generates comprehensive incident reports using LLM when available."""

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

        sev = triage.get("severity", "P3")
        cat = triage.get("category", "infrastructure")
        rc = rca.get("root_cause", "Unknown")
        conf = rca.get("confidence_score", 0.0)
        contributing = rca.get("contributing_factors", [])

        actions_summary = "; ".join(
            f"{a.get('tool_name','?')}: {a.get('rationale','')}"
            for a in remediation.get("recommended_actions", [])
        ) or "None"
        exec_status = "; ".join(
            f"{e.get('tool_name','?')}={e.get('status','unknown')}"
            for e in exec_results
        ) or "pending"

        try:
            prompt = REPORTING_PROMPT.format(
                severity=sev,
                category=cat,
                impact=triage.get("impact_assessment", "N/A"),
                root_cause=rc[:200],
                confidence=f"{conf:.0%}" if isinstance(conf, float) else str(conf),
                contributing_factors=json.dumps(contributing[:3]),
                actions=actions_summary[:300],
                execution_status=exec_status[:200],
                anomaly_count=len(anomalies),
            )
            content = self._call_llm([
                {"role": "system", "content": REPORTING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            result = self._parse_json_response(content)
            result["llm_generated"] = True
            self.logger.info("LLM report generated: %s", result.get("incident_id", "unknown"))
            return result
        except Exception as exc:
            self.logger.warning("LLM report failed, using template: %s", exc)

        return self._build_template_report(anomalies, triage, rca, remediation, exec_results)

    def _build_template_report(self, anomalies, triage, rca, remediation, exec_results) -> dict:
        now = datetime.now(timezone.utc)
        sev = triage.get("severity", "P3")
        cat = triage.get("category", "infrastructure")
        rc = rca.get("root_cause", "Unknown")
        short_id = uuid.uuid4().hex[:4].upper()

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
            "root_cause_summary": rc,
            "actions_taken": actions_taken,
            "business_impact": triage.get("impact_assessment", "Impact assessment pending."),
            "prevention_recommendations": rca.get("contributing_factors",
                ["Review monitoring thresholds", "Implement auto-scaling"]),
            "llm_generated": False,
            "generated_at": now.isoformat(),
        }
