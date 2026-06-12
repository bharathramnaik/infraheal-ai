"""
InfraHeal AI — Shared Utilities
=================================
Data validation, format helpers, and common functions used across
the InfraHeal AI platform.

Kept deliberately small — no heavy dependencies beyond stdlib.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REQUIRED_LOG_KEYS = {"timestamp", "source", "level", "service", "message"}
REQUIRED_METRIC_KEYS = {
    "timestamp", "host", "cpu_percent", "memory_percent", "disk_percent",
    "network_in_mbps", "network_out_mbps", "request_latency_ms", "error_rate",
    "active_connections",
}
REQUIRED_ANOMALY_KEYS = {"id", "timestamp", "type", "severity", "source", "description"}
REQUIRED_RUNBOOK_KEYS = {"id", "title", "category", "symptoms", "root_causes", "resolution_steps"}


def validate_log_entry(entry: dict, strict: bool = False) -> List[str]:
    """Validate a single log entry against the canonical schema.

    Args:
        entry: Log dict to validate.
        strict: If True, missing optional keys are flagged.

    Returns:
        List of validation error messages (empty = valid).
    """
    errors: List[str] = []
    missing = REQUIRED_LOG_KEYS - set(entry.keys())
    if missing:
        errors.append(f"Missing required keys: {missing}")
    if not isinstance(entry.get("timestamp"), str):
        errors.append("'timestamp' must be a string")
    if isinstance(entry.get("level"), str) and entry["level"].upper() not in {
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
    }:
        errors.append(f"Invalid level: {entry.get('level')}")
    return errors


def validate_metric_entry(entry: dict) -> List[str]:
    """Validate a single metric entry against the canonical schema."""
    errors: List[str] = []
    missing = REQUIRED_METRIC_KEYS - set(entry.keys())
    if missing:
        errors.append(f"Missing required metric keys: {missing}")
    for field in ["cpu_percent", "memory_percent", "disk_percent"]:
        val = entry.get(field)
        if isinstance(val, (int, float)) and not (0 <= val <= 100):
            errors.append(f"{field} out of range [0,100]: {val}")
    return errors


def validate_anomaly(entry: dict) -> List[str]:
    """Validate an anomaly dict produced by the anomaly detector."""
    errors: List[str] = []
    missing = REQUIRED_ANOMALY_KEYS - set(entry.keys())
    if missing:
        errors.append(f"Missing required anomaly keys: {missing}")
    sev = entry.get("severity", "")
    if sev not in {"P1", "P2", "P3", "P4"}:
        errors.append(f"Invalid severity: {sev}")
    conf = entry.get("confidence", 0)
    if isinstance(conf, (int, float)) and not (0 <= conf <= 1):
        errors.append(f"Confidence out of range [0,1]: {conf}")
    return errors


def validate_runbook(rb: dict) -> List[str]:
    """Validate a runbook dict."""
    errors: List[str] = []
    missing = REQUIRED_RUNBOOK_KEYS - set(rb.keys())
    if missing:
        errors.append(f"Missing required runbook keys: {missing}")
    return errors


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """Safely traverse nested dict keys, returning default on missing."""
    for key in keys:
        try:
            d = d[key]
        except (KeyError, TypeError, IndexError):
            return default
    return d


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"
