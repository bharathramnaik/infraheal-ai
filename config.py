"""
InfraHeal AI — Central Configuration
=====================================
All settings, paths, and model configs in one place.
Designed for AMD GPU Cloud with vLLM (ROCm + vLLM default stack).
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "sample_data"
RUNBOOKS_DIR = PROJECT_DIR / "sample_data" / "runbooks"

# ─── vLLM / Model Configuration ─────────────────────────────────
# vLLM runs an OpenAI-compatible API server on the GPU cloud
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM default

# Model priority: try these in order, use first available
MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]
MODEL_NAME = os.getenv("MODEL_NAME", MODEL_CANDIDATES[0])

# Per-model capabilities registry
# has_thinking: model outputs chain-of-thought before answer (DeepSeek-R1, Qwen3)
# thinking_tag: how the model wraps its thinking trace
MODEL_REGISTRY = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "label": "Qwen 2.5 7B (Default)",
        "has_thinking": False,
        "max_tokens": 1024,
        "description": "Balanced instruct model — fast, reliable",
    },
    "Qwen/Qwen3-4B": {
        "label": "Qwen 3 4B (Thinking)",
        "has_thinking": True,
        "max_tokens": 2048,
        "description": "Extended thinking support",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "label": "LLaMA 3.1 8B",
        "has_thinking": False,
        "max_tokens": 1024,
        "description": "Standard instruct — good general purpose",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "label": "DeepSeek R1 7B (Thinking)",
        "has_thinking": True,
        "max_tokens": 2048,
        "description": "Reasoning model — shows internal thought chain",
    },
}

# All thinking tags to strip from output for display
THINKING_TAGS = [
    ("<think>", "</think>"),
    ("[REASONING]", "[/REASONING]"),
    ("[reasoning]", "[/reasoning]"),
    ("[THINK]", "[/THINK]"),
    ("[think]", "[/think]"),
]

# Inference settings
MAX_TOKENS = 256                # Fallback; agents use their own value below
AGENT_MAX_TOKENS = {
    "triage": 256,
    "rca": 512,
    "remediation": 512,
    "reporting": 256,
}
TEMPERATURE = 0.3         # Low for consistent agent outputs
TOP_P = 0.9
FREQUENCY_PENALTY = 0.1   # Reduce repetition

# ─── Anomaly Detection Thresholds ────────────────────────────────
ANOMALY_Z_SCORE_THRESHOLD = 2.5
ANOMALY_ERROR_RATE_THRESHOLD = 0.15   # 15% error rate triggers alert
CPU_CRITICAL_THRESHOLD = 90.0          # %
MEMORY_CRITICAL_THRESHOLD = 85.0       # %
DISK_CRITICAL_THRESHOLD = 90.0         # %
LATENCY_CRITICAL_MS = 5000             # ms

# ─── Agent Settings ──────────────────────────────────────────────
AGENT_MAX_RETRIES = 2
AGENT_TIMEOUT_SECONDS = 60
MAX_CONTEXT_LOGS = 5        # Max log lines sent to agents
MAX_CONTEXT_LOGS_CHARS = 300  # Max total chars from logs in prompt
MAX_RAG_CHARS = 200          # Max chars of runbook context in prompt

# ─── Dashboard Settings ─────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 7860
DASHBOARD_THEME = "dark"

# ─── Severity Levels ────────────────────────────────────────────
SEVERITY_LEVELS = {
    "P1": {"label": "Critical", "color": "#FF3B3B", "sla_minutes": 15},
    "P2": {"label": "High",     "color": "#FF8C00", "sla_minutes": 60},
    "P3": {"label": "Medium",   "color": "#FFD700", "sla_minutes": 240},
    "P4": {"label": "Low",      "color": "#4CAF50", "sla_minutes": 1440},
}

# ─── Incident Categories ────────────────────────────────────────
INCIDENT_CATEGORIES = [
    "infrastructure",
    "application",
    "network",
    "security",
    "database",
    "storage",
]

# ─── Remediation Actions (Tool Registry) ────────────────────────
AVAILABLE_TOOLS = [
    {
        "name": "restart_service",
        "description": "Restart a specific service/container",
        "parameters": {"service_name": "string", "force": "boolean"},
    },
    {
        "name": "scale_resources",
        "description": "Scale up/down compute resources (CPU/memory/replicas)",
        "parameters": {"resource_type": "string", "target_value": "number", "unit": "string"},
    },
    {
        "name": "rollback_deployment",
        "description": "Rollback to a previous deployment version",
        "parameters": {"service_name": "string", "target_version": "string"},
    },
    {
        "name": "clear_disk_space",
        "description": "Clear temporary files and old logs to free disk space",
        "parameters": {"target_path": "string", "older_than_days": "number"},
    },
    {
        "name": "block_ip",
        "description": "Block a suspicious IP address at firewall level",
        "parameters": {"ip_address": "string", "duration_minutes": "number"},
    },
    {
        "name": "flush_cache",
        "description": "Flush application or database cache",
        "parameters": {"cache_type": "string", "service_name": "string"},
    },
    {
        "name": "rotate_logs",
        "description": "Force log rotation to prevent disk issues",
        "parameters": {"service_name": "string", "max_size_mb": "number"},
    },
    {
        "name": "update_config",
        "description": "Update service configuration parameter",
        "parameters": {"service_name": "string", "config_key": "string", "config_value": "string"},
    },
]


def detect_model(client):
    """Auto-detect available model from vLLM server."""
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        for candidate in MODEL_CANDIDATES:
            if candidate in available:
                return candidate
        # Return first available if none match candidates
        return available[0] if available else MODEL_NAME
    except Exception:
        return MODEL_NAME
