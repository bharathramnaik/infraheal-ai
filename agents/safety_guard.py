"""
InfraHeal AI — Safety Guard & Action Validator
================================================
Prevents the agent from executing harmful, destructive, or
unauthorised actions.  Every remediation action is validated
against safety rules before execution.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("infraheal.safety_guard")

# ─── Safety Rules ──────────────────────────────────────────────────
# Each rule: (tool_name, param_checks, risk, reason)
# param_checks: dict of param_name -> disallowed_values (set) or callable

SAFETY_RULES: List[Dict[str, Any]] = [
    # ── Critical system paths (never clear these) ─────────────────
    {
        "tool": "clear_disk_space",
        "check": lambda p: p.get("target_path", "").rstrip("/\\") in (
            "/", "/bin", "/sbin", "/etc", "/boot", "/dev",
            "/proc", "/sys", "/lib", "/lib64",
            "C:\\", "C:\\Windows", "C:\\Windows\\System32",
            "C:\\Program Files", "C:\\Program Files (x86)",
        ),
        "risk": "blocked",
        "reason": "Target path is critical system directory — clearing would break the OS.",
    },
    {
        "tool": "clear_disk_space",
        "check": lambda p: bool(p.get("older_than_days", 0) <= 0),
        "risk": "blocked",
        "reason": "older_than_days must be >= 1 to prevent accidental mass deletion.",
    },
    # ── Force-restart critical services ────────────────────────────
    {
        "tool": "restart_service",
        "check": lambda p: p.get("service_name", "").lower() in (
            "kernel", "systemd", "init", "sshd", "vllm",
            "docker", "containerd", "kubelet",
        ) and p.get("force", False),
        "risk": "blocked",
        "reason": "Force-restarting a critical system service would cause node failure.",
    },
    {
        "tool": "restart_service",
        "check": lambda p: p.get("force", False),
        "risk": "high",
        "reason": "Force restart may cause data loss or unexpected downtime.",
    },
    # ── IP blocking ────────────────────────────────────────────────
    {
        "tool": "block_ip",
        "check": lambda p: p.get("ip_address", "") in (
            "127.0.0.1", "::1", "0.0.0.0", "255.255.255.255",
            "localhost", "localhost6",
        ),
        "risk": "blocked",
        "reason": "Blocking loopback/localhost would break local communication.",
    },
    {
        "tool": "block_ip",
        "check": lambda p: p.get("duration_minutes", 0) > 10080,
        "risk": "high",
        "reason": "Blocking an IP for >7 days (10080 min) could cause extended service disruption if incorrect.",
    },
    # ── Resource scaling ───────────────────────────────────────────
    {
        "tool": "scale_resources",
        "check": lambda p: p.get("target_value", 0) <= 0,
        "risk": "blocked",
        "reason": "Scaling resources to zero or negative would crash the service.",
    },
    {
        "tool": "scale_resources",
        "check": lambda p: str(p.get("resource_type", "")).lower() == "cpu"
                          and p.get("target_value", 0) > 64,
        "risk": "high",
        "reason": "Setting CPU >64 cores is unusually high — confirm this is intentional.",
    },
    # ── Rollback ───────────────────────────────────────────────────
    {
        "tool": "rollback_deployment",
        "check": lambda p: not p.get("target_version", "").strip(),
        "risk": "blocked",
        "reason": "Rollback requires a valid target version — empty version would be destructive.",
    },
    # ── Config updates ─────────────────────────────────────────────
    {
        "tool": "update_config",
        "check": lambda p: any(
            kw in str(p.get("config_key", "")).lower()
            for kw in ("password", "secret", "token", "key", "credential")
        ),
        "risk": "blocked",
        "reason": "Updating credential-related config keys via automation is prohibited — use secrets manager.",
    },
    {
        "tool": "update_config",
        "check": lambda p: str(p.get("config_value", "")).strip() == "",
        "risk": "blocked",
        "reason": "Setting a config value to empty could break the service.",
    },
]

# ─── Severity-based guardrails ─────────────────────────────────────
# For P1/P2 incidents, certain restrictions are relaxed to allow
# faster recovery, but blocking rules still apply.
SEVERITY_OVERRIDES = {
    "P1": {"allow_high_risk": True},
    "P2": {"allow_high_risk": True},
}


class SafetyGuard:
    """Validates remediation actions against safety rules.

    Usage:
        guard = SafetyGuard()
        result = guard.validate(action, severity="P1")
        if result["verdict"] == "block":
            logger.warning("Blocked: %s", result["reason"])
    """

    def __init__(self) -> None:
        self._audit_log: List[Dict[str, Any]] = []

    def validate(self, action: dict, severity: str = "P3") -> dict:
        """Validate a single action against all safety rules.

        Returns:
            ``{"verdict": "allow"|"block"|"flag", "reason": …, "risk": …}``
        """
        tool = action.get("tool_name", "")
        params = action.get("parameters", {})
        action_risk = action.get("risk_level", "medium")
        severity_overrides = SEVERITY_OVERRIDES.get(severity, {})

        for rule in SAFETY_RULES:
            if rule["tool"] != tool:
                continue
            try:
                if rule["check"](params):
                    verdict = rule["risk"]
                    reason = rule["reason"]
                    if verdict == "blocked":
                        verdict = "block"

                    self._audit_log.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "tool": tool,
                        "parameters": dict(params),
                        "rule_reason": reason,
                        "verdict": verdict,
                        "severity": severity,
                    })

                    return {
                        "verdict": verdict,
                        "reason": reason,
                        "risk": verdict if verdict == "block" else action_risk,
                    }
            except Exception as exc:
                logger.warning("Safety rule check failed for %s: %s", tool, exc)

        # High-risk actions require approval unless overridden by severity
        if action_risk == "high" and not severity_overrides.get("allow_high_risk", False):
            self._audit_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "parameters": dict(params),
                "rule_reason": "Default high-risk guardrail",
                "verdict": "flag",
                "severity": severity,
            })
            return {
                "verdict": "flag",
                "reason": "High-risk action requires manual approval.",
                "risk": "high",
            }

        # Passthrough for allowed actions
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "parameters": dict(params),
            "rule_reason": "All safety checks passed",
            "verdict": "allow",
            "severity": severity,
        })

        return {"verdict": "allow", "reason": "", "risk": action_risk}

    def validate_plan(self, actions: List[dict], severity: str = "P3") -> List[dict]:
        """Validate every action in a plan and return enriched results."""
        results: List[dict] = []
        for action in actions:
            safety = self.validate(action, severity)
            results.append({**action, "_safety": safety})
        return results

    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._audit_log[-limit:]

    def get_audit_summary(self) -> dict:
        blocked = sum(1 for e in self._audit_log if e["verdict"] == "block")
        flagged = sum(1 for e in self._audit_log if e["verdict"] == "flag")
        allowed = sum(1 for e in self._audit_log if e["verdict"] == "allow")
        return {
            "total_checks": len(self._audit_log),
            "blocked": blocked,
            "flagged": flagged,
            "allowed": allowed,
            "has_violations": blocked > 0 or flagged > 0,
        }
