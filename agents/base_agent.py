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
    AGENT_MAX_TOKENS,
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
        max_tokens: Optional[int] = None,
        schema_validator: Optional[Any] = None,
        few_shot_examples: str = "",
    ) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = client or OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        self.model_name = model_name or MODEL_NAME
        self.tools = tools or []
        self.max_tokens = max_tokens or AGENT_MAX_TOKENS.get(name, AGENT_MAX_TOKENS.get(name.removesuffix("_agent"), MAX_TOKENS))
        self.logger = logging.getLogger(f"infraheal.{name}")
        self.execution_log: List[Dict[str, Any]] = []
        self.schema_validator = schema_validator
        self.few_shot_examples = few_shot_examples

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
                    max_tokens=self.max_tokens,
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

    def _run_with_validation(self, messages: List[Dict[str, str]], max_retries: int = 2) -> dict:
        """Call LLM, parse, and validate against schema with reject/retry.

        When the parsed output fails schema validation, the error is fed
        back to the model as a correction message and the call is retried.
        """
        if not self.schema_validator:
            raw = self._call_llm(messages)
            return self._parse_json_response(raw)

        last_result: dict = {}
        for attempt in range(max_retries + 1):
            raw = self._call_llm(messages)
            result = self._parse_json_response(raw)
            valid, err_msg = self.schema_validator(result)
            if not valid:
                self.logger.warning(
                    "%s schema validation failed (attempt %d/%d): %s",
                    self.name, attempt + 1, max_retries + 1, err_msg,
                )
                last_result = result
                if attempt < max_retries:
                    messages.append({"role": "assistant", "content": json.dumps(result, default=str)})
                    messages.append({
                        "role": "user",
                        "content": f"Output failed schema validation: {err_msg}\n"
                                   f"Fix the JSON output to match the required schema exactly. "
                                   f"Return ONLY the corrected JSON object.",
                    })
                continue
            return result

        self.logger.warning(
            "%s schema validation exhausted %d retries — returning last result",
            self.name, max_retries + 1,
        )
        return last_result

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
            result = json.loads(cleaned)
            if isinstance(result, list):
                if len(result) >= 1 and isinstance(result[0], dict):
                    self.logger.warning("LLM returned array with %d items — using first item", len(result))
                    return result[0]
                return {"error": "Expected JSON object, got array", "raw": text, "_array": result}
            return result
        except json.JSONDecodeError:
            pass

        # Greedy search for the outermost { … }
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    if len(result) >= 1 and isinstance(result[0], dict):
                        self.logger.warning("LLM returned array (greedy) — using first item")
                        return result[0]
                    return {"error": "Expected JSON object, got array", "raw": text, "_array": result}
                return result
            except json.JSONDecodeError:
                pass

        self.logger.warning("Failed to parse JSON from %s (attempting salvage): %s", self.name, cleaned[:200])

        # Detect truncation — if text is cut off mid-JSON
        truncated = False
        if "{" in cleaned and "}" not in cleaned[cleaned.rindex("{"):]:
            truncated = True
        elif cleaned.count("{") > cleaned.count("}"):
            truncated = True

        if truncated:
            # Attempt to salvage partial JSON by closing unclosed brackets/braces
            # in the correct nesting order (LIFO).
            try:
                fixed = cleaned.rstrip().rstrip(",")
                # Build a stack of opening brackets to track nesting order;
                # also track whether we're inside an unclosed string.
                stack = []
                in_string = False
                escape = False
                for ch in fixed:
                    if escape:
                        escape = False
                        continue
                    if ch == '\\' and in_string:
                        escape = True
                        continue
                    if ch == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch in ('{', '['):
                        stack.append(ch)
                    elif ch == '}' and stack and stack[-1] == '{':
                        stack.pop()
                    elif ch == ']' and stack and stack[-1] == '[':
                        stack.pop()
                # Close unclosed string, then remaining brackets in LIFO order
                if in_string:
                    fixed += '"'
                for ch in reversed(stack):
                    fixed += '}' if ch == '{' else ']'
                partial = json.loads(fixed)
                partial["_partial"] = True
                self.logger.warning("Salvaged partial JSON from %s (recovered %d fields)", self.name, len(partial))
                return partial
            except (json.JSONDecodeError, Exception) as salvage_err:
                # Fallback: strip incomplete trailing content after last `,`
                try:
                    last_comma = fixed.rfind(",")
                    if last_comma > 0:
                        stripped = fixed[:last_comma].rstrip().rstrip(",")
                        # Re-close brackets for the stripped version
                        stack = []
                        in_string = False
                        escape = False
                        for ch in stripped:
                            if escape: escape = False; continue
                            if ch == '\\' and in_string: escape = True; continue
                            if ch == '"': in_string = not in_string; continue
                            if in_string: continue
                            if ch in ('{', '['): stack.append(ch)
                            elif ch == '}' and stack and stack[-1] == '{': stack.pop()
                            elif ch == ']' and stack and stack[-1] == '[': stack.pop()
                        if in_string:
                            stripped += '"'
                        for ch in reversed(stack):
                            stripped += '}' if ch == '{' else ']'
                        fallback = json.loads(stripped)
                        fallback["_partial"] = True
                        self.logger.warning("Salvaged partial JSON from %s via fallback (recovered %d fields)", self.name, len(fallback))
                        return fallback
                except (json.JSONDecodeError, Exception):
                    pass
                self.logger.warning("Could not salvage truncated JSON from %s: %s", self.name, salvage_err)
            return {"error": "LLM response truncated (max_tokens too low). JSON incomplete.", "raw": text, "_truncated": True}
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
