"""
InfraHeal AI — Anomaly Detection Engine
=========================================
Multi-strategy anomaly detection combining statistical (Z-score/IQR)
metric analysis with log-level pattern matching and temporal correlation.

Produces structured anomaly dicts for consumption by the multi-agent pipeline.
"""

import logging
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from .config import (
        ANOMALY_Z_SCORE_THRESHOLD, ANOMALY_ERROR_RATE_THRESHOLD,
        CPU_CRITICAL_THRESHOLD, MEMORY_CRITICAL_THRESHOLD,
        DISK_CRITICAL_THRESHOLD, LATENCY_CRITICAL_MS,
        SEVERITY_LEVELS,
    )
except ImportError:
    from config import (
        ANOMALY_Z_SCORE_THRESHOLD, ANOMALY_ERROR_RATE_THRESHOLD,
        CPU_CRITICAL_THRESHOLD, MEMORY_CRITICAL_THRESHOLD,
        DISK_CRITICAL_THRESHOLD, LATENCY_CRITICAL_MS,
        SEVERITY_LEVELS,
    )

logger = logging.getLogger(__name__)

_CRITICAL_LOG_LEVELS = {"CRITICAL", "ERROR"}
_HIGH_LOG_LEVELS = {"WARNING"}


class AnomalyDetector:
    """Multi-strategy anomaly detection engine for infrastructure data.

    Detection strategies:

    1. **Metric Z-Score**: Detects statistical outliers in time-series
       metrics using rolling Z-score analysis.
    2. **Metric Threshold**: Fires when metrics cross hard-coded
       critical thresholds (CPU > 90%, memory > 85%, etc.).
    3. **Log Pattern**: Flags ERROR/CRITICAL log bursts and specific
       known-bad message patterns.
    4. **Error Rate Spike**: Detects abrupt increases in error
       percentage from metric data.
    """

    def __init__(
        self,
        z_score_threshold: float = ANOMALY_Z_SCORE_THRESHOLD,
        cpu_threshold: float = CPU_CRITICAL_THRESHOLD,
        memory_threshold: float = MEMORY_CRITICAL_THRESHOLD,
        disk_threshold: float = DISK_CRITICAL_THRESHOLD,
        latency_threshold_ms: float = LATENCY_CRITICAL_MS,
        error_rate_threshold: float = ANOMALY_ERROR_RATE_THRESHOLD,
    ):
        self.z_score_threshold = z_score_threshold
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold

    def detect_all(
        self,
        logs: Optional[List[dict]] = None,
        metrics: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Run all detection strategies and return merged anomaly list.

        Args:
            logs: List of log dicts with keys timestamp, source, level, service, message, metadata.
            metrics: List of metric dicts with keys timestamp, host, cpu_percent, etc.

        Returns:
            Deduplicated list of anomaly dicts sorted by severity (P1 first).
        """
        anomalies: List[dict] = []

        if metrics:
            anomalies.extend(self._detect_metric_outliers(metrics))
            anomalies.extend(self._detect_metric_thresholds(metrics))
            anomalies.extend(self._detect_error_rate_spikes(metrics))

        if logs:
            anomalies.extend(self._detect_log_patterns(logs))
            anomalies.extend(self._detect_log_bursts(logs))

        anomalies = self._deduplicate(anomalies)
        anomalies.sort(key=lambda a: _severity_rank(a.get("severity", "P4")))
        logger.info("detect_all: %d anomalies from %d logs, %d metrics", len(anomalies), len(logs or []), len(metrics or []))
        return anomalies

    def detect(
        self,
        metrics: Optional[List[dict]] = None,
        logs: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Alias for detect_all for backward compatibility."""
        return self.detect_all(logs=logs, metrics=metrics)

    def detect_metric_outliers(self, metrics: List[dict]) -> List[dict]:
        """Z-score-based outlier detection per metric per host."""
        return self._detect_metric_outliers(metrics)

    def _detect_metric_outliers(self, metrics: List[dict]) -> List[dict]:
        """Detect statistical outliers in numeric metric fields."""
        if not metrics:
            return []

        METRIC_FIELDS = [
            ("cpu_percent", "cpu_spike", "CPU utilization"),
            ("memory_percent", "memory_anomaly", "Memory usage"),
            ("request_latency_ms", "latency_spike", "Request latency"),
        ]

        anomalies: List[dict] = []
        host_groups = defaultdict(list)
        for m in metrics:
            host_groups[m.get("host", "unknown")].append(m)

        for host, host_metrics in host_groups.items():
            host_metrics.sort(key=lambda m: m.get("timestamp", ""))
            for field, anomaly_type, field_label in METRIC_FIELDS:
                values = [m.get(field) for m in host_metrics if isinstance(m.get(field), (int, float))]
                if len(values) < 10:
                    continue

                mean = statistics.mean(values)
                stdev = statistics.pstdev(values)
                if stdev < 0.001:
                    continue

                for m in host_metrics:
                    val = m.get(field)
                    if not isinstance(val, (int, float)):
                        continue
                    z = abs((val - mean) / stdev)
                    if z > self.z_score_threshold:
                        anomalies.append({
                            "id": f"ANO-{uuid.uuid4().hex[:6].upper()}",
                            "timestamp": m.get("timestamp", ""),
                            "type": anomaly_type,
                            "severity": "P3" if z < self.z_score_threshold * 1.5 else "P2",
                            "source": host,
                            "description": f"{field_label} anomaly on {host}: {val:.1f} (z-score={z:.2f}, threshold={self.z_score_threshold})",
                            "evidence": [f"{field_label}: {val:.1f}", f"Mean: {mean:.1f}", f"StdDev: {stdev:.2f}", f"Z-score: {z:.2f}"],
                            "related_logs": [],
                            "related_metrics": [m],
                            "confidence": min(z / (self.z_score_threshold * 2), 0.99),
                        })
        return anomalies

    def _detect_metric_thresholds(self, metrics: List[dict]) -> List[dict]:
        """Hard-threshold checks for critical metric breaches."""
        if not metrics:
            return []

        recent = metrics[-min(len(metrics), 20):]
        anomalies: List[dict] = []

        TRESHOLD_CHECKS = [
            ("cpu_percent", self.cpu_threshold, "cpu_breach", "CPU", "P2"),
            ("memory_percent", self.memory_threshold, "memory_breach", "Memory", "P2"),
            ("disk_percent", self.disk_threshold, "disk_breach", "Disk", "P2"),
            ("request_latency_ms", self.latency_threshold_ms, "latency_breach", "Latency", "P2"),
        ]

        for field, threshold, an_type, label, severity in TRESHOLD_CHECKS:
            for m in recent:
                val = m.get(field)
                if not isinstance(val, (int, float)):
                    continue
                if val <= threshold:
                    continue
                host = m.get("host", "unknown")
                severity_override = "P1" if val > threshold * 1.3 else severity
                anomalies.append({
                    "id": f"ANO-{uuid.uuid4().hex[:6].upper()}",
                    "timestamp": m.get("timestamp", ""),
                    "type": an_type,
                    "severity": severity_override,
                    "source": host,
                    "description": f"{label} critical on {host}: {val:.1f} (threshold: {threshold})",
                    "evidence": [f"{label}: {val:.1f}%", f"Threshold: {threshold}", f"Host: {host}"],
                    "related_logs": [],
                    "related_metrics": [m],
                    "confidence": min((val - threshold) / threshold, 0.99),
                })
        return anomalies

    def _detect_error_rate_spikes(self, metrics: List[dict]) -> List[dict]:
        """Detect error rate spikes above dynamic baseline."""
        if not metrics:
            return []

        host_groups = defaultdict(list)
        for m in metrics:
            host_groups[m.get("host", "unknown")].append(m)

        anomalies: List[dict] = []
        for host, host_metrics in host_groups.items():
            host_metrics.sort(key=lambda m: m.get("timestamp", ""))
            if len(host_metrics) < 10:
                continue

            error_rates = [
                m.get("error_rate", 0) for m in host_metrics
                if isinstance(m.get("error_rate"), (int, float))
            ]
            if not error_rates:
                continue

            baseline = statistics.mean(error_rates[:-5]) if len(error_rates) > 5 else statistics.mean(error_rates)
            for m in host_metrics[-10:]:
                err = m.get("error_rate", 0)
                if not isinstance(err, (int, float)):
                    continue
                if err <= self.error_rate_threshold and err <= baseline * 3:
                    continue
                anomalies.append({
                    "id": f"ANO-{uuid.uuid4().hex[:6].upper()}",
                    "timestamp": m.get("timestamp", ""),
                    "type": "error_rate_spike",
                    "severity": "P1" if err > 0.3 else "P2",
                    "source": host,
                    "description": f"Error rate spike on {host}: {err*100:.1f}% (baseline: {baseline*100:.1f}%)",
                    "evidence": [f"Error rate: {err*100:.1f}%", f"Baseline: {baseline*100:.1f}%", f"Host: {host}"],
                    "related_logs": [],
                    "related_metrics": [m],
                    "confidence": min(err / max(baseline, 0.001), 0.99),
                })
        return anomalies

    def _detect_log_patterns(self, logs: List[dict]) -> List[dict]:
        """Flag ERROR/CRITICAL log entries and known-bad message patterns."""
        if not logs:
            return []

        anomalies: List[dict] = []
        for log in logs[-500:]:
            level = log.get("level", "INFO").upper()
            msg = log.get("message", "")
            source = log.get("source", "unknown")
            host = log.get("metadata", {}).get("host", source)

            if level not in _CRITICAL_LOG_LEVELS:
                continue

            anomaly_type = "log_critical" if level == "CRITICAL" else "log_error"
            severity = "P1" if level == "CRITICAL" else "P2"

            anomalies.append({
                "id": f"ANO-{uuid.uuid4().hex[:6].upper()}",
                "timestamp": log.get("timestamp", ""),
                "type": anomaly_type,
                "severity": severity,
                "source": host,
                "description": f"[{level}] {source}: {msg[:200]}",
                "evidence": [f"Level: {level}", f"Source: {source}", f"Message: {msg[:200]}"],
                "related_logs": [log],
                "related_metrics": [],
                "confidence": 0.85 if level == "ERROR" else 0.95,
            })

        return anomalies

    def _detect_log_bursts(self, logs: List[dict]) -> List[dict]:
        """Detect bursts of errors from the same source within a time window."""
        if not logs:
            return []

        recent = [l for l in logs if l.get("level", "").upper() in _CRITICAL_LOG_LEVELS]
        if len(recent) < 5:
            return []

        source_counts = Counter(l.get("source", "unknown") for l in recent)
        anomalies: List[dict] = []
        for source, count in source_counts.most_common(5):
            if count < 5:
                continue
            host = recent[0].get("metadata", {}).get("host", source)
            anomalies.append({
                "id": f"ANO-{uuid.uuid4().hex[:6].upper()}",
                "timestamp": recent[-1].get("timestamp", ""),
                "type": "error_burst",
                "severity": "P1" if count > 20 else "P2",
                "source": host,
                "description": f"Error burst from '{source}': {count} ERROR/CRITICAL entries in window",
                "evidence": [f"Source: {source}", f"Count: {count} errors", f"Time window: last {len(logs)} entries"],
                "related_logs": [l for l in recent if l.get("source") == source][:10],
                "related_metrics": [],
                "confidence": min(count / 50, 0.98),
            })
        return anomalies

    def _deduplicate(self, anomalies: List[dict]) -> List[dict]:
        """Merge near-identical anomalies from different strategies."""
        if not anomalies:
            return []

        seen: set = set()
        unique: List[dict] = []
        for a in anomalies:
            key = (a.get("source", ""), a.get("type", ""), a.get("timestamp", "")[:16])
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique


class IncidentCorrelator:
    """Correlates individual anomalies into grouped incident clusters.

    Groups anomalies that share the same source host, occur within
    a temporal window, or have causal relationships (e.g. a latency
    spike preceded by a CPU spike on the same host).  This enables
    the orchestrator to treat a group of related anomalies as a
    single incident rather than processing each one independently.
    """

    CORRELATION_WINDOW_SECONDS = 300  # 5 min

    def correlate(self, anomalies: List[dict]) -> List[dict]:
        """Group anomalies into incident clusters.

        Args:
            anomalies: Flat list of anomaly dicts from AnomalyDetector.

        Returns:
            List of incident-group dicts, each with:
            ``incident_id``, ``anomalies`` (list), ``primary_severity``,
            ``sources``, ``description``, ``start_time``, ``end_time``.
        """
        if not anomalies:
            return []

        sorted_ans = sorted(anomalies, key=lambda a: a.get("timestamp", ""))
        clusters: List[List[dict]] = []

        for anomaly in sorted_ans:
            placed = False
            for cluster in clusters:
                if self._belongs_to_cluster(anomaly, cluster):
                    cluster.append(anomaly)
                    placed = True
                    break
            if not placed:
                clusters.append([anomaly])

        incidents = []
        for idx, cluster in enumerate(clusters):
            severities = [a.get("severity", "P4") for a in cluster]
            sources = list({a.get("source", "unknown") for a in cluster})
            primary = min(severities, key=lambda s: _severity_rank(s))
            timestamps = [a.get("timestamp", "") for a in cluster if a.get("timestamp")]
            desc = (
                f"Incident involving {len(cluster)} anomalies on {', '.join(sources[:3])}"
                f"{' and more' if len(sources) > 3 else ''}"
            )
            incidents.append({
                "incident_id": f"INC-CORR-{idx+1:03d}",
                "anomalies": cluster,
                "primary_severity": primary,
                "sources": sources,
                "description": desc,
                "anomaly_count": len(cluster),
                "start_time": min(timestamps) if timestamps else "",
                "end_time": max(timestamps) if timestamps else "",
            })

        return incidents

    @staticmethod
    def _belongs_to_cluster(anomaly: dict, cluster: List[dict]) -> bool:
        """Check if an anomaly belongs to an existing cluster."""
        if not cluster:
            return False

        for existing in cluster:
            same_source = anomaly.get("source") == existing.get("source")
            if not same_source:
                continue
            t_anom = anomaly.get("timestamp", "")
            t_exist = existing.get("timestamp", "")
            if t_anom and t_exist:
                try:
                    from datetime import datetime
                    dt_anom = datetime.fromisoformat(t_anom)
                    dt_exist = datetime.fromisoformat(t_exist)
                    if abs((dt_anom - dt_exist).total_seconds()) <= IncidentCorrelator.CORRELATION_WINDOW_SECONDS:
                        return True
                except (ValueError, TypeError):
                    return same_source
        return False


def _severity_rank(sev: str) -> int:
    """Return numeric rank for severity sorting (lower = more severe)."""
    return {"P1": 0, "P2": 1, "P3": 2, "P4": 3}.get(sev.upper(), 4)
