"""
InfraHeal AI — Base Agent
=========================
Abstract base class for all InfraHeal agents.
Provides LLM communication via OpenAI-compatible vLLM API,
retry logic, structured JSON parsing, and execution metrics.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    AGENT_MAX_RETRIES,
    AGENT_TIMEOUT_SECONDS,
)


class BaseAgent:
    """Base class for all InfraHeal AI agents.

    Handles LLM communication through the OpenAI-compatible vLLM API,
    automatic retries on transient failures, structured JSON response
    parsing, and per-agent execution metrics collection.

    Subclasses must override :meth:`run` with their domain-specific logic.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        client: Optional[OpenAI] = None,
        model_name: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the agent.

        Args:
            name: Human-readable agent name (used in logs & metrics).
            role: Short role description (e.g. "Triage", "RCA").
            system_prompt: Full system prompt sent as the first message.
            client: Pre-configured ``OpenAI`` client.  A default one
                pointing at the local vLLM server is created when *None*.
            model_name: Model identifier.  Falls back to ``MODEL_NAME``
                from *config.py*.
            tools: Optional list of tool definitions (for future tool-use
                extensions).
        """
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = client or OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        self.model_name = model_name or MODEL_NAME
        self.tools = tools or []
        self.logger = logging.getLogger(f"infraheal.{name}")
        self.execution_log: List[Dict[str, Any]] = []

    # ── Public API ───────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        """Execute agent logic.  **Must be overridden by subclasses.**

        Args:
            context: A dictionary whose keys vary per agent (e.g.
                ``anomalies``, ``triage_result``, etc.).

        Returns:
            Structured dictionary with agent-specific results.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError(f"{self.name}.run() must be implemented by subclass")

    # ── LLM Communication ────────────────────────────────────────

    def _call_llm(self, messages: List[Dict[str, str]], response_format: Optional[dict] = None) -> str:
        """Call vLLM with automatic retry logic.

        Args:
            messages: OpenAI-style list of message dicts.
            response_format: Optional JSON schema dict.  When provided it
                is passed as ``guided_json`` via ``extra_body`` so vLLM
                constrains the output.

        Returns:
            Raw text content from the model's first choice.

        Raises:
            Exception: Propagates the last exception after all retries are
                exhausted.
        """
        start = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(AGENT_MAX_RETRIES + 1):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
                if response_format and isinstance(response_format, dict):
                    kwargs["extra_body"] = {"guided_json": json.dumps(response_format)}

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""

                elapsed = time.time() - start
                tokens_used = response.usage.total_tokens if response.usage else 0

                self.execution_log.append(
                    {
                        "agent": self.name,
                        "attempt": attempt + 1,
                        "latency_seconds": round(elapsed, 2),
                        "tokens_used": tokens_used,
                        "success": True,
                    }
                )
                self.logger.info(
                    "%s completed in %.2fs, %d tokens (attempt %d)",
                    self.name, elapsed, tokens_used, attempt + 1,
                )
                return content

            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "%s attempt %d/%d failed: %s",
                    self.name, attempt + 1, AGENT_MAX_RETRIES + 1, exc,
                )
                if attempt < AGENT_MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 8))  # exponential back-off capped at 8s

        # All retries exhausted
        self.execution_log.append({"agent": self.name, "error": str(last_error), "success": False})
        raise last_error  # type: ignore[misc]

    def _build_messages(self, context: dict) -> List[Dict[str, str]]:
        """Build the standard system + user message list.

        Subclasses can override this to inject additional messages (e.g.
        few-shot examples) between system and user prompts.

        Args:
            context: The same dict passed to :meth:`run`.

        Returns:
            List of message dicts ready for :meth:`_call_llm`.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(context, indent=2, default=str)},
        ]

    # ── Response Parsing ─────────────────────────────────────────

    def _parse_json_response(self, text: str) -> dict:
        """Extract a JSON object from LLM output.

        Handles common LLM quirks:
        * Markdown fenced code blocks (```json … ```)
        * Leading/trailing prose around a JSON object
        * Nested braces

        Args:
            text: Raw LLM response string.

        Returns:
            Parsed ``dict``, or a fallback ``{"error": …, "raw": …}``
            when parsing fails entirely.
        """
        cleaned = text.strip()

        # Strip markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove opening fence (possibly ```json)
            lines = lines[1:]
            # Remove closing fence if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        # Attempt direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Greedy search for the outermost { … }
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        self.logger.error("Failed to parse JSON from %s: %s", self.name, cleaned[:300])
        return {"error": "Failed to parse response", "raw": text}

    # ── Metrics ──────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return execution metrics for this agent.

        Returns:
            Dictionary containing total/successful call counts, token
            usage, average latency, and error count.
        """
        successful = [e for e in self.execution_log if e.get("success")]
        failed = len(self.execution_log) - len(successful)
        total_tokens = sum(e.get("tokens_used", 0) for e in successful)
        total_latency = sum(e.get("latency_seconds", 0) for e in successful)

        return {
            "agent": self.name,
            "total_calls": len(self.execution_log),
            "successful_calls": len(successful),
            "total_tokens": total_tokens,
            "avg_latency": round(total_latency / max(len(successful), 1), 2),
            "errors": failed,
        }
