"""
InfraHeal AI — Remediation Agent
==================================
Generates safe, actionable remediation plans using the available tool
registry.  Supports simulated execution for demo/testing and
risk-aware auto-approval for low-risk actions.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_agent import BaseAgent

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)

REMEDIATION_SYSTEM_PROMPT = """You are the **Remediation Agent** in the InfraHeal AI incident management system.

## Your Mission
Given a root cause analysis and triage classification, generate a safe, ordered remediation plan using ONLY the available tools listed below. Every action must be justified, risk-assessed, and reversible where possible.

## Available Tools
{tools_section}

## Safety Rules (CRITICAL)
1. **Least-privilege**: Always prefer the smallest, most targeted action.
2. **Risk assessment**: Tag every action as low / medium / high risk.
3. **Approval gates**: Actions with risk_level "high" MUST set requires_approval = true.
4. **Rollback plan**: Every plan MUST include a rollback strategy.
5. **Order matters**: Actions MUST be sequenced logically (e.g. scale before restart, not after).
6. **No unknown tools**: ONLY use tools from the Available Tools list above.

## Risk Classification
- **low**: Read-only, cache flush, log rotation, config reads — safe to auto-execute.
- **medium**: Service restart, scaling, config changes — may cause brief disruption.
- **high**: Rollback deployment, block IP, destructive operations — requires human approval.

## Output Schema (strict)
```json
{
  "recommended_actions": [
    {
      "step": 1,
      "tool_name": "<tool name from available tools>",
      "parameters": {"<param>": "<value>"},
      "rationale": "<why this action addresses the root cause>",
      "risk_level": "low|medium|high",
      "requires_approval": true|false,
      "expected_outcome": "<what should happen after execution>"
    }
  ],
  "execution_order": "sequential|parallel",
  "rollback_plan": "<step-by-step rollback instructions>",
  "estimated_resolution_time": "<e.g. 5-10 minutes>",
  "warnings": ["<any important caveats>"],
  "confidence": 0.0-1.0
}
```

Respond ONLY with the JSON object. No markdown, no explanation outside the JSON."""


class RemediationAgent(BaseAgent):
    """Generates and optionally executes remediation plans.

    Analyses the root cause, selects appropriate tools from the registry,
    sequences actions safely, and provides rollback plans.  Includes a
    simulated execution engine for demo environments.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the Remediation Agent.

        Args:
            client: Pre-configured OpenAI client (optional).
            model_name: Model identifier (optional).
            tools: Tool registry list.  Falls back to
                ``config.AVAILABLE_TOOLS``.
        """
        self._tool_registry = tools or AVAILABLE_TOOLS
        # Build system prompt with actual tools injected
        prompt = REMEDIATION_SYSTEM_PROMPT.replace(
            "{tools_section}", self._format_tools_section()
        )
        super().__init__(
            name="remediation_agent",
            role="Remediation Planning & Execution",
            system_prompt=prompt,
            client=client,
            model_name=model_name,
            tools=self._tool_registry,
        )

    def run(self, context: dict) -> dict:
        """Generate a remediation plan.

        Args:
            context: Must contain:
                - ``rca_result``: output from RCAAgent.
                - ``triage_result``: output from TriageAgent.
                Optionally:
                - ``available_tools``: override tool registry.

        Returns:
            Dict with recommended_actions, execution_order,
            rollback_plan, estimated_resolution_time, warnings,
            confidence.
        """
        rca_result = context.get("rca_result", {})
        triage_result = context.get("triage_result", {})

        if not rca_result.get("root_cause"):
            self.logger.warning("Remediation called without root cause")
            return self._default_result()

        user_content = self._format_remediation_prompt(rca_result, triage_result)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = self._call_llm(messages)
        result = self._parse_json_response(raw)
        result = self._validate_result(result)

        action_count = len(result.get("recommended_actions", []))
        self.logger.info(
            "Remediation plan: %d actions, order=%s, ETA=%s",
            action_count,
            result.get("execution_order", "sequential"),
            result.get("estimated_resolution_time", "unknown"),
        )
        return result

    # ── Execution Engine (Simulated) ─────────────────────────────

    def execute_action(self, action: dict) -> dict:
        """Simulate execution of a single remediation action.

        In a production environment this would dispatch to real
        infrastructure APIs.  The demo implementation simulates
        success/failure with realistic delays.

        Args:
            action: A single action dict from ``recommended_actions``.

        Returns:
            Dict with execution status, duration, and details.
        """
        action_id = str(uuid.uuid4())[:8]
        tool_name = action.get("tool_name", "unknown")
        params = action.get("parameters", {})
        risk = action.get("risk_level", "medium")

        self.logger.info(
            "Executing action [%s]: %s with params %s (risk: %s)",
            action_id, tool_name, params, risk,
        )

        # Validate tool exists in registry
        valid_tools = {t["name"] for t in self._tool_registry}
        if tool_name not in valid_tools:
            return {
                "action_id": action_id,
                "tool_name": tool_name,
                "status": "failed",
                "error": f"Unknown tool '{tool_name}'. Available: {sorted(valid_tools)}",
                "duration_seconds": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Simulate execution with realistic timing
        start = time.time()
        sim_delay = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(risk, 1.0)
        time.sleep(sim_delay)
        elapsed = round(time.time() - start, 2)

        return {
            "action_id": action_id,
            "tool_name": tool_name,
            "parameters": params,
            "status": "success",
            "message": f"[SIMULATED] {tool_name} executed successfully with params {json.dumps(params)}",
            "duration_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def execute_plan(
        self,
        actions: List[dict],
        auto_approve_low_risk: bool = True,
    ) -> List[dict]:
        """Execute all actions in the remediation plan.

        Args:
            actions: List of action dicts from ``recommended_actions``.
            auto_approve_low_risk: When *True*, actions with
                ``risk_level == "low"`` are executed without approval.
                Medium/high-risk actions that ``require_approval`` are
                marked as ``pending_approval`` instead.

        Returns:
            List of execution result dicts, one per action.
        """
        results: List[dict] = []

        for idx, action in enumerate(actions, 1):
            risk = action.get("risk_level", "medium")
            needs_approval = action.get("requires_approval", risk == "high")

            if needs_approval and not (auto_approve_low_risk and risk == "low"):
                self.logger.info(
                    "Action %d (%s) requires approval — skipping in auto mode",
                    idx, action.get("tool_name", "?"),
                )
                results.append({
                    "step": idx,
                    "tool_name": action.get("tool_name", "unknown"),
                    "status": "pending_approval",
                    "message": f"Action requires manual approval (risk: {risk})",
                    "risk_level": risk,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            exec_result = self.execute_action(action)
            exec_result["step"] = idx
            results.append(exec_result)

            # Stop execution if an action fails
            if exec_result.get("status") != "success":
                self.logger.error(
                    "Action %d failed — halting plan execution: %s",
                    idx, exec_result.get("error", "unknown error"),
                )
                break

        executed = sum(1 for r in results if r.get("status") == "success")
        pending = sum(1 for r in results if r.get("status") == "pending_approval")
        self.logger.info(
            "Plan execution complete: %d/%d executed, %d pending approval",
            executed, len(actions), pending,
        )
        return results

    # ── Internal Helpers ─────────────────────────────────────────

    def _format_tools_section(self) -> str:
        """Format the tool registry for inclusion in the system prompt."""
        lines: List[str] = []
        for tool in self._tool_registry:
            params = ", ".join(
                f"{k}: {v}" for k, v in tool.get("parameters", {}).items()
            )
            lines.append(
                f"- **{tool['name']}**: {tool['description']}\n"
                f"  Parameters: {params}"
            )
        return "\n".join(lines)

    def _format_remediation_prompt(self, rca_result: dict, triage_result: dict) -> str:
        """Build the user prompt for remediation planning."""
        sections: List[str] = [
            "## Incident Summary\n"
            f"- **Severity**: {triage_result.get('severity', 'N/A')} ({triage_result.get('severity_label', '')})\n"
            f"- **Category**: {triage_result.get('category', 'N/A')}\n"
            f"- **SLA**: {triage_result.get('sla_minutes', 'N/A')} minutes\n"
            f"- **Impact**: {triage_result.get('impact_assessment', 'N/A')}\n"
            f"- **Affected Services**: {', '.join(triage_result.get('affected_services', []))}\n",
            "## Root Cause Analysis\n"
            f"- **Root Cause**: {rca_result.get('root_cause', 'Unknown')}\n"
            f"- **Category**: {rca_result.get('root_cause_category', 'Unknown')}\n"
            f"- **Confidence**: {rca_result.get('confidence_score', 0)}\n"
            f"- **Blast Radius**: {rca_result.get('blast_radius', 'Unknown')}\n",
        ]

        evidence = rca_result.get("evidence_chain", [])
        if evidence:
            sections.append(
                "## Evidence Chain\n"
                + "\n".join(f"- {e}" for e in evidence)
                + "\n"
            )

        factors = rca_result.get("contributing_factors", [])
        if factors:
            sections.append(
                "## Contributing Factors\n"
                + "\n".join(f"- {f}" for f in factors)
                + "\n"
            )

        sections.append(
            "Generate a remediation plan using ONLY the available tools. "
            "Return ONLY the JSON object matching the schema in your instructions."
        )
        return "\n".join(sections)

    def _validate_result(self, result: dict) -> dict:
        """Ensure all fields are present and actions reference valid tools."""
        valid_tools = {t["name"] for t in self._tool_registry}
        raw_actions = result.get("recommended_actions", [])

        validated_actions: List[dict] = []
        for idx, action in enumerate(raw_actions, 1):
            tool_name = action.get("tool_name", "")
            if tool_name not in valid_tools:
                self.logger.warning(
                    "LLM suggested unknown tool '%s' — keeping but flagging", tool_name,
                )
            validated_actions.append({
                "step": action.get("step", idx),
                "tool_name": tool_name,
                "parameters": action.get("parameters", {}),
                "rationale": action.get("rationale", "No rationale provided."),
                "risk_level": action.get("risk_level", "medium"),
                "requires_approval": action.get(
                    "requires_approval",
                    action.get("risk_level", "medium") == "high",
                ),
                "expected_outcome": action.get("expected_outcome", "Expected improvement."),
            })

        return {
            "recommended_actions": validated_actions,
            "execution_order": result.get("execution_order", "sequential"),
            "rollback_plan": result.get("rollback_plan", "No rollback plan specified."),
            "estimated_resolution_time": result.get("estimated_resolution_time", "Unknown"),
            "warnings": result.get("warnings", []),
            "confidence": min(max(float(result.get("confidence", 0.5)), 0.0), 1.0),
        }

    @staticmethod
    def _default_result() -> dict:
        """Return a safe default when no root cause is available."""
        return {
            "recommended_actions": [],
            "execution_order": "sequential",
            "rollback_plan": "No actions to roll back.",
            "estimated_resolution_time": "N/A",
            "warnings": ["No root cause was identified — cannot generate remediation plan."],
            "confidence": 0.0,
        }
