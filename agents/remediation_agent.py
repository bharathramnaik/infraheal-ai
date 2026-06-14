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

REMEDIATION_SYSTEM_PROMPT = """You are a remediation planner. Given root cause + triage, generate ordered actions using ONLY tools listed below. Tools:

{tools_section}

Safety: high-risk actions must set requires_approval=true. BE CONCISE. Output ONLY this JSON:

{"recommended_actions":[{"step":1,"tool_name":"tool","parameters":{"k":"v"},"risk_level":"low|medium|high","requires_approval":bool}],"execution_order":"seq|par","rollback_plan":"brief","estimated_resolution_time":"duration","warnings":["caveat"],"confidence":0-1}

No prose, no markdown, only JSON."""


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
                - ``few_shot_examples``: past successful remediations for in-context learning.
                - ``action_preferences``: historical approval rate string.

        Returns:
            Dict with recommended_actions, execution_order,
            rollback_plan, estimated_resolution_time, warnings,
            confidence.
        """
        rca_result = context.get("rca_result", {})
        triage_result = context.get("triage_result", {})
        few_shot = context.get("few_shot_examples", "")
        preferences = context.get("action_preferences", "")

        if not rca_result.get("root_cause"):
            self.logger.warning("Remediation called without root cause")
            return self._default_result()

        user_content = self._format_remediation_prompt(rca_result, triage_result, few_shot, preferences)
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

    def _format_remediation_prompt(self, rca_result: dict, triage_result: dict,
                                    few_shot: str = "", preferences: str = "") -> str:
        tri = triage_result
        rca = rca_result
        parts = [
            f"incident sev={tri.get('severity','?')} cat={tri.get('category','?')} impact={str(tri.get('impact_assessment',''))[:60]}",
            f"rca root={rca.get('root_cause','?')} conf={rca.get('confidence_score',0)}",
        ]
        evidence = rca.get("evidence_chain", [])
        if evidence:
            parts.append("ev: " + " | ".join(e[:50] for e in evidence[:2]))
        if preferences:
            parts.append("Action success history: " + preferences)
        if few_shot:
            parts.append("Past similar incidents resolved with:\n" + few_shot)
        parts.append("Plan remediation. ONLY JSON.")
        return "\n".join(parts)

    def _validate_result(self, result: dict) -> dict:
        """Ensure all fields are present and actions reference valid tools."""
        if "error" in result and "_partial" not in result:
            return {
                "recommended_actions": [],
                "execution_order": "sequential",
                "rollback_plan": "No actions to roll back.",
                "estimated_resolution_time": "N/A",
                "warnings": ["No root cause was identified — cannot generate remediation plan."],
                "confidence": 0.0,
                "error": result["error"],
                "raw": result.get("raw", ""),
            }

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
                "rationale": action.get("rationale") or tool_name.replace("_"," ").title() + " — standard remediation step.",
                "risk_level": action.get("risk_level", "medium"),
                "requires_approval": action.get(
                    "requires_approval",
                    action.get("risk_level", "medium") == "high",
                ),
                "expected_outcome": action.get("expected_outcome") or tool_name.replace("_"," ") + " executed.",
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
