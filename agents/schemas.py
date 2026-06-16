"""
InfraHeal AI — Schema Validation
=================================
Lightweight JSON schema validation for all agent inputs/outputs.
No external dependencies — uses pure Python type/field checks.
"""

from typing import Any, Dict, List, Optional, Tuple


def _check(d: dict, field: str, typ: type, optional: bool = False) -> Optional[str]:
    v = d.get(field)
    if v is None:
        if optional:
            return None
        return f"missing required field '{field}'"
    if typ is list and isinstance(v, list):
        return None
    if typ is dict and isinstance(v, dict):
        return None
    if not isinstance(v, typ):
        return f"field '{field}' must be {typ.__name__}, got {type(v).__name__}"
    return None


def _enum(d: dict, field: str, allowed: List[str]) -> Optional[str]:
    err = _check(d, field, str)
    if err:
        return err
    if d.get(field) not in allowed:
        return f"field '{field}' must be one of {allowed}, got '{d.get(field)}'"
    return None


def _list_of_dicts(d: dict, field: str) -> Optional[str]:
    err = _check(d, field, list)
    if err:
        return err
    for i, item in enumerate(d.get(field, [])):
        if not isinstance(item, dict):
            return f"field '{field}[{i}]' must be dict, got {type(item).__name__}"
    return None


def validate_triage_output(result: dict) -> Tuple[bool, str]:
    err = _enum(result, "severity", ["P1", "P2", "P3", "P4"])
    if err: return False, err
    err = _enum(result, "severity_label", ["Critical", "High", "Medium", "Low"])
    if err: return False, err
    err = _enum(result, "category", ["infrastructure", "application", "network", "security", "database", "storage"])
    if err: return False, err
    err = _check(result, "impact_assessment", str)
    if err: return False, err
    err = _check(result, "affected_services", list)
    if err: return False, err
    err = _check(result, "urgency_reasoning", str)
    if err: return False, err
    err = _check(result, "confidence", (int, float))
    if err: return False, err
    err = _check(result, "escalation_needed", bool)
    if err: return False, err
    err = _check(result, "sla_minutes", (int, float))
    if err: return False, err
    return True, ""


def validate_rca_output(result: dict) -> Tuple[bool, str]:
    err = _check(result, "root_cause", str)
    if err: return False, err
    err = _check(result, "root_cause_category", str)
    if err: return False, err
    err = _check(result, "evidence_chain", list)
    if err: return False, err
    err = _check(result, "confidence_score", (int, float))
    if err: return False, err
    err = _check(result, "contributing_factors", list, optional=True)
    if err: return False, err
    err = _check(result, "timeline_of_events", list, optional=True)
    if err: return False, err
    err = _check(result, "affected_components", list, optional=True)
    if err: return False, err
    err = _check(result, "blast_radius", str, optional=True)
    if err: return False, err
    err = _check(result, "reasoning_summary", str, optional=True)
    if err: return False, err
    return True, ""


def validate_remediation_output(result: dict) -> Tuple[bool, str]:
    err = _check(result, "recommended_actions", list)
    if err: return False, err
    for i, act in enumerate(result.get("recommended_actions", [])):
        if not isinstance(act, dict):
            return False, f"recommended_actions[{i}] must be dict"
        for f in ("step", "tool_name", "risk_level"):
            if f not in act:
                return False, f"recommended_actions[{i}] missing '{f}'"
        rl = act.get("risk_level", "")
        if rl not in ("low", "medium", "high"):
            return False, f"recommended_actions[{i}].risk_level must be low|medium|high"
    err = _check(result, "execution_order", str, optional=True)
    if err: return False, err
    err = _check(result, "rollback_plan", str, optional=True)
    if err: return False, err
    err = _check(result, "confidence", (int, float), optional=True)
    if err: return False, err
    return True, ""


def validate_report_output(result: dict) -> Tuple[bool, str]:
    err = _check(result, "incident_id", str)
    if err: return False, err
    err = _check(result, "title", str)
    if err: return False, err
    err = _check(result, "severity", str, optional=True)
    if err: return False, err
    err = _check(result, "executive_summary", str, optional=True)
    if err: return False, err
    err = _check(result, "root_cause_summary", str, optional=True)
    if err: return False, err
    return True, ""


def validate_critique_output(result: dict) -> Tuple[bool, str]:
    err = _check(result, "confirmed", bool)
    if err: return False, err
    err = _check(result, "refined_confidence", (int, float), optional=True)
    if err: return False, err
    err = _check(result, "refined_reasoning", str, optional=True)
    if err: return False, err
    return True, ""


SCHEMA_VALIDATORS = {
    "triage_agent": validate_triage_output,
    "rca_agent": validate_rca_output,
    "remediation_agent": validate_remediation_output,
    "reporting_agent": validate_report_output,
    "critique_agent": validate_critique_output,
}


# ── Few-Shot Examples for each agent ──────────────────────────────

TRIAGE_FEW_SHOT = """
Example 1:
Input anomalies: [{"severity":"P1","type":"service_down","source":"payment-service","description":"Payment service returning 503 for all requests"}]
Output: {"severity":"P1","severity_label":"Critical","category":"application","impact_assessment":"Payment service outage blocks all transactions, directly impacting revenue. Estimated 10K+ users affected.","affected_services":["payment-service","checkout-service"],"urgency_reasoning":"Production service down — P1 per SLA. Every minute of downtime costs approximately $5K.","confidence":0.95,"escalation_needed":true,"sla_minutes":15}

Example 2:
Input anomalies: [{"severity":"P3","type":"memory_breach","source":"user-service","description":"Memory usage at 92%, above 85% threshold"}]
Output: {"severity":"P3","severity_label":"Medium","category":"application","impact_assessment":"User service memory approaching limits. Degraded performance possible under sustained load.","affected_services":["user-service"],"urgency_reasoning":"Memory trending upward but not yet critical. Proactive intervention recommended within 4 hours.","confidence":0.85,"escalation_needed":false,"sla_minutes":240}
"""

RCA_FEW_SHOT = """
Example:
triage: sev=P1 cat=application impact=Payment service outage
anomalies:
  P1 service_down payment-service "Service returning 503" conf=0.95
  P2 latency_checkout-service "Checkout latency exceeding threshold" conf=0.80
root_cause: Database connection pool exhaustion in payment-service due to unoptimized query pattern introduced in deployment v2.3.1 caused cascading failures across checkout-service.
evidence_chain: [timeline, metrics, logs]
confidence: 0.92
"""

REMEDIATION_FEW_SHOT = """
Example:
incident: sev=P1 cat=application
rca: root=Database connection pool exhaustion conf=0.92
tools: restart_service, scale_resources, rollback_deployment, flush_cache
actions:
  [1] scale_resources: Increase database connection pool (rationale: pool at 100% capacity, expected_outcome: connections freed, requires_approval=true, risk=high)
  [2] rollback_deployment: Rollback payment-service to v2.3.0 (rationale: v2.3.1 introduced bad query pattern, expected_outcome: query load returns to normal, requires_approval=true, risk=high)
  [3] flush_cache: Clear query cache (rationale: stale cached queries slowing system, expected_outcome: cache rebuilt fresh, risk=low)
  [4] restart_service: Restart payment-service (rationale: clean slate after rollback, expected_outcome: service healthy, risk=medium)
"""

REPORTING_FEW_SHOT = """
Example:
severity=P1 category=application
root_cause=Database connection pool exhaustion in payment-service due to unoptimized query pattern introduced in deployment v2.3.1 caused cascading failures across checkout-service.
actions: [{"tool_name":"scale_resources","rationale":"Increase database connection pool"},{"tool_name":"rollback_deployment","rationale":"Rollback payment-service to v2.3.0"},{"tool_name":"flush_cache","rationale":"Clear query cache"},{"tool_name":"restart_service","rationale":"Restart payment-service"}]
execution: scale_resources=completed, rollback_deployment=completed, flush_cache=completed, restart_service=completed
"""
