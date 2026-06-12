"""
InfraHeal AI — Synthetic Data Generator
========================================
Generates realistic IT-infrastructure logs, metrics, runbooks,
past incidents, and pre-built demo scenarios for testing / demos.

All outputs conform to the canonical data formats defined in config.py.
"""

import json
import logging
import math
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .config import (
        CPU_CRITICAL_THRESHOLD,
        DATA_DIR,
        DISK_CRITICAL_THRESHOLD,
        INCIDENT_CATEGORIES,
        MEMORY_CRITICAL_THRESHOLD,
        SEVERITY_LEVELS,
    )
except ImportError:
    from config import (
        CPU_CRITICAL_THRESHOLD,
        DATA_DIR,
        DISK_CRITICAL_THRESHOLD,
        INCIDENT_CATEGORIES,
        MEMORY_CRITICAL_THRESHOLD,
        SEVERITY_LEVELS,
    )

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
SEED = 42

SOURCES = [
    "nginx", "postgresql", "redis", "kubernetes",
    "docker", "systemd", "auth", "application",
]

LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LEVEL_WEIGHTS = [0.10, 0.50, 0.20, 0.15, 0.05]

HOSTS = [
    "web-server-01", "web-server-02", "db-primary",
    "db-replica", "cache-server-01", "k8s-node-01",
]

# Per-source realistic message templates  (≥ 8–10 each)
_LOG_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "nginx": {
        "DEBUG": [
            'rewrite phase: regex "{pat}" matched against "{uri}"',
            "upstream response buffered to temporary file {tmpfile}",
        ],
        "INFO": [
            '{ip} - - "{method} {uri} HTTP/1.1" {status} {bytes} "{referrer}"',
            "server {host} listening on 0.0.0.0:{port}",
            "worker process {pid} started",
            "graceful shutdown initiated",
        ],
        "WARNING": [
            "upstream server temporarily disabled while connecting to upstream",
            "client body is buffered to a temporary file (size {bytes})",
            "{ip} was rate-limited on zone req_zone (excess: {excess})",
        ],
        "ERROR": [
            "connect() failed (111: Connection refused) while connecting to upstream {upstream}",
            "upstream timed out (110: Connection timed out) for client {ip}",
            "SSL_do_handshake() failed (SSL: error:1408F119) on {host}",
            "open() /var/www/{path} failed (2: No such file or directory)",
        ],
        "CRITICAL": [
            "worker process {pid} exited on signal 11 (core dumped)",
            "emergency: could not open error log file: open() /var/log/nginx/error.log failed",
        ],
    },
    "postgresql": {
        "DEBUG": [
            "duration: {ms:.1f} ms  plan: Seq Scan on {table}",
            "checkpoint starting: xlog",
        ],
        "INFO": [
            "database system is ready to accept connections",
            "autovacuum: processing table {schema}.{table}",
            "checkpoint complete: wrote {pages} buffers ({pct:.1f}%)",
            "received fast shutdown request",
        ],
        "WARNING": [
            "connection pool exhausted — {active}/{max} connections in use",
            "archive command failed with exit code 1",
            "checkpoints are occurring too frequently ({interval} seconds)",
            "could not receive data from WAL stream: SSL error",
        ],
        "ERROR": [
            "too many connections for role \"{role}\"",
            "deadlock detected: process {pid} waits for ShareLock on {table}",
            "invalid input syntax for type integer: \"{val}\"",
            "canceling statement due to statement timeout ({ms} ms)",
        ],
        "CRITICAL": [
            "data directory \"/var/lib/postgresql/data\" has wrong ownership",
            "out of shared memory for lock table; increase max_locks_per_transaction",
        ],
    },
    "redis": {
        "DEBUG": [
            "accepted connection from {ip}:{port} (fd={fd})",
            "bio thread performing lazy-free of object (mem: {mem} MB)",
        ],
        "INFO": [
            "ready to accept connections on port 6379",
            "RDB: {keys} keys saved on disk in {sec:.2f} seconds",
            "background saving started by pid {pid}",
            "slave {ip}:{port} asks for synchronization",
        ],
        "WARNING": [
            "memory usage {mem_pct:.0f}% — approaching maxmemory {max_mem}",
            "overcommit_memory is set to 0; background save may fail under low memory",
            "possible slow log detected: command={cmd} duration={us}us",
            "client {ip} scheduled to be closed for overcoming output buffer limits",
        ],
        "ERROR": [
            "MISCONF Redis is configured to save RDB snapshots but unable to persist on disk",
            "connection with replica {ip}:{port} lost",
            "OOM command not allowed when used memory > maxmemory",
        ],
        "CRITICAL": [
            "fatal error: can't open /var/lib/redis/dump.rdb for saving: Permission denied",
            "Redis crashed by signal: 11 — segfault at address {addr}",
        ],
    },
    "kubernetes": {
        "DEBUG": [
            "SyncLoop (PLEG): event for pod {pod}: ContainerStarted",
            "volume mount {vol} attached to node {node}",
        ],
        "INFO": [
            "successfully pulled image \"{image}:{tag}\"",
            "created pod {pod} in namespace {ns}",
            "started container {container} for pod {pod}",
            'node {node} condition Ready is now "True"',
        ],
        "WARNING": [
            "pod {pod} in namespace {ns} has been in Pending state for {min} minutes",
            "back-off restarting failed container {container} in pod {pod}",
            "node {node} has disk pressure: available {avail}Mi < threshold {thresh}Mi",
            "evicting pod {pod} due to node memory pressure",
        ],
        "ERROR": [
            "failed to create pod sandbox for {pod}: rpc error: code = Unknown",
            "container {container} in pod {pod} CrashLoopBackOff (exit code 137)",
            "failed to pull image \"{image}:{tag}\": ErrImagePull",
            "liveness probe failed for {pod}: HTTP probe failed with status 503",
        ],
        "CRITICAL": [
            "node {node} is NotReady — kubelet stopped posting status",
            "ETCD quorum lost: cluster cannot commit new entries",
        ],
    },
    "docker": {
        "DEBUG": [
            "layer {layer_hash} already exists, skipping push",
            "health check for container {cid}: status=healthy",
        ],
        "INFO": [
            "container {cid} started (image={image})",
            "network bridge created: {net_id}",
            "volume {vol} mounted at {mount}",
            "container {cid} stopped gracefully (exit 0)",
        ],
        "WARNING": [
            "container {cid} using 95% of memory limit ({mem_limit})",
            "no space left on device for layer storage",
            "container {cid} health check: unhealthy (3 consecutive failures)",
        ],
        "ERROR": [
            "OOM killed container {cid} (used {mem_used} > limit {mem_limit})",
            "failed to start container {cid}: bind mount {mount} not found",
            "cannot stop container {cid}: permission denied",
            "error pulling image {image}: net/http: TLS handshake timeout",
        ],
        "CRITICAL": [
            "dockerd panic: runtime error: invalid memory address or nil pointer dereference",
            "daemon storage driver overlay2 failed: no space left on device",
        ],
    },
    "systemd": {
        "DEBUG": [
            "unit {unit}.service: state changed running → running",
            "cgroup /system.slice/{unit}.service: memory usage {mem} MB",
        ],
        "INFO": [
            "started {unit}.service — {desc}",
            "stopping {unit}.service...",
            "unit {unit}.service entered 'active' state",
            "system boot completed in {sec:.1f}s",
        ],
        "WARNING": [
            "unit {unit}.service entered 'failed' state — will retry in 30s",
            "service {unit}.service watchdog timeout ({timeout}s), scheduling restart",
            "resource limit reached for {unit}.service: TasksMax={tasks}",
        ],
        "ERROR": [
            "unit {unit}.service: main process exited, code=exited, status=1/FAILURE",
            "failed to start {unit}.service: unit is masked",
            "service {unit}.service: start request repeated too quickly, refusing to start",
            "failed to write PID file /run/{unit}.pid: read-only file system",
        ],
        "CRITICAL": [
            "kernel: Out of memory: Killed process {pid} ({unit}), UID {uid}",
            "systemd[1]: crash detected — freezing execution",
        ],
    },
    "auth": {
        "DEBUG": [
            "token validation: user={user} scope={scope} exp={exp}",
            "LDAP bind successful for dn=uid={user},ou=people,dc=infraheal",
        ],
        "INFO": [
            "user '{user}' logged in from {ip} (method={method})",
            "session created for user '{user}' (session_id={sid})",
            "password changed for user '{user}'",
            "MFA challenge sent to user '{user}' device {device}",
        ],
        "WARNING": [
            "failed login attempt for user '{user}' from {ip} ({attempts} consecutive failures)",
            "session {sid} for user '{user}' expired after {dur} minutes of inactivity",
            "rate limit exceeded for IP {ip}: {rate} requests/min",
            "API key {key_prefix}*** nearing expiry ({days} days remaining)",
        ],
        "ERROR": [
            "authentication failed for user '{user}': invalid credentials (IP: {ip})",
            "LDAP connection refused: ldap://{ldap_host}:389",
            "OAuth2 token exchange failed: invalid_grant for client {client}",
        ],
        "CRITICAL": [
            "brute force attack detected: {attempts} failed logins from {ip} in {window}s",
            "auth service unresponsive — all login requests failing (circuit breaker OPEN)",
        ],
    },
    "application": {
        "DEBUG": [
            "request {req_id}: route={route} handler resolved in {us}µs",
            "cache MISS for key={key} — querying database",
        ],
        "INFO": [
            "request {req_id}: {method} {route} completed in {ms}ms (status={status})",
            "background job {job_id} completed: processed {items} items in {sec:.1f}s",
            "health check OK — uptime {uptime}h, heap {heap}MB",
            "deployment v{version} rolled out successfully",
        ],
        "WARNING": [
            "request {req_id}: response time {ms}ms exceeds SLA threshold of 2000ms",
            "thread pool utilization at {pct}% — consider scaling",
            "dependency service '{dep}' latency elevated ({lat_ms}ms avg)",
            "memory usage growing: {mem}MB (was {prev_mem}MB 10 min ago)",
        ],
        "ERROR": [
            "request {req_id}: unhandled exception in {handler}: {exc}",
            "circuit breaker OPEN for downstream service '{dep}'",
            "database query timeout after {ms}ms on table '{table}'",
            "failed to enqueue job {job_id}: message broker unreachable",
        ],
        "CRITICAL": [
            "application heap exhausted — OutOfMemoryError after {mem}MB allocation",
            "unrecoverable state: data corruption detected in {table}, shutting down",
        ],
    },
}

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _uid() -> str:
    """Short unique id fragment."""
    return uuid.uuid4().hex[:8]


def _rand_ip() -> str:
    return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def _fill_template(template: str) -> str:
    """Replace placeholders with plausible random values."""
    replacements: Dict[str, Any] = {
        "ip": _rand_ip(),
        "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
        "uri": random.choice(["/api/v1/health", "/api/v1/users", "/api/v1/orders", "/dashboard", "/metrics"]),
        "status": random.choice([200, 201, 204, 301, 400, 404, 500, 502, 503]),
        "bytes": random.randint(128, 65536),
        "referrer": random.choice(["https://app.infraheal.io", "-"]),
        "host": random.choice(HOSTS),
        "port": random.choice([80, 443, 8080, 8443]),
        "pid": random.randint(1000, 65535),
        "upstream": f"10.0.{random.randint(1,10)}.{random.randint(1,254)}:8080",
        "path": random.choice(["static/img/logo.png", "api/docs/index.html"]),
        "ms": round(random.uniform(0.5, 15000), 1),
        "table": random.choice(["users", "orders", "sessions", "audit_log", "metrics"]),
        "schema": "public",
        "pages": random.randint(50, 5000),
        "pct": round(random.uniform(1, 100), 1),
        "interval": random.randint(10, 120),
        "role": random.choice(["app_user", "readonly", "admin"]),
        "val": random.choice(["abc", "null", "NaN"]),
        "active": random.randint(80, 200),
        "max": 100,
        "excess": random.randint(1, 50),
        "keys": random.randint(1000, 500000),
        "sec": round(random.uniform(0.1, 30), 2),
        "mem": random.randint(64, 8192),
        "mem_pct": round(random.uniform(50, 99), 0),
        "max_mem": "4GB",
        "cmd": random.choice(["KEYS *", "LRANGE biglist 0 -1", "SMEMBERS largeset"]),
        "us": random.randint(100, 500000),
        "fd": random.randint(3, 1024),
        "addr": f"0x{random.randint(0, 2**32):08x}",
        "pod": f"{random.choice(['api','worker','frontend','celery'])}-{_uid()[:6]}",
        "ns": random.choice(["production", "staging", "monitoring"]),
        "container": random.choice(["api", "sidecar", "init-db", "worker"]),
        "image": random.choice(["infraheal/api", "infraheal/worker", "nginx", "redis"]),
        "tag": random.choice(["latest", "v1.2.3", "v2.0.0-rc1", "sha-abc1234"]),
        "node": random.choice(HOSTS),
        "avail": random.randint(50, 500),
        "thresh": 1024,
        "min": random.randint(2, 30),
        "vol": f"pvc-{_uid()}",
        "mount": f"/data/vol-{_uid()[:4]}",
        "cid": _uid()[:12],
        "net_id": _uid()[:12],
        "mem_limit": random.choice(["512MB", "1GB", "2GB", "4GB"]),
        "mem_used": random.choice(["520MB", "1.1GB", "2.3GB", "4.2GB"]),
        "layer_hash": f"sha256:{_uid()}{_uid()}",
        "unit": random.choice(["nginx", "postgresql", "redis-server", "docker", "kubelet", "app-api"]),
        "desc": random.choice(["The NGINX HTTP Server", "PostgreSQL RDBMS", "Application API"]),
        "timeout": random.choice([30, 60, 90]),
        "tasks": random.choice([512, 1024, 4096]),
        "uid": random.randint(1000, 65534),
        "user": random.choice(["admin", "deploy-bot", "jdoe", "svc-account", "root"]),
        "scope": random.choice(["read", "write", "admin"]),
        "exp": (datetime.now(timezone.utc) + timedelta(hours=random.randint(1, 72))).isoformat(),
        "sid": _uid(),
        "device": random.choice(["iPhone-14", "Pixel-7", "YubiKey-5"]),
        "attempts": random.randint(3, 100),
        "dur": random.randint(15, 480),
        "rate": random.randint(100, 2000),
        "key_prefix": _uid()[:6],
        "days": random.randint(1, 30),
        "ldap_host": "ldap.infraheal.internal",
        "client": f"client-{_uid()[:6]}",
        "window": random.choice([60, 120, 300]),
        "req_id": _uid(),
        "route": random.choice(["/api/v1/users", "/api/v1/orders", "/api/v1/auth/token"]),
        "handler": random.choice(["UserController.get", "OrderService.create", "AuthHandler.verify"]),
        "exc": random.choice(["NullPointerException", "ConnectionResetError", "TimeoutError", "ValueError"]),
        "job_id": f"job-{_uid()[:6]}",
        "items": random.randint(10, 10000),
        "uptime": round(random.uniform(0.1, 720), 1),
        "heap": random.randint(128, 4096),
        "version": f"{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,20)}",
        "dep": random.choice(["payment-service", "inventory-api", "notification-svc", "cache-layer"]),
        "lat_ms": random.randint(500, 8000),
        "prev_mem": random.randint(256, 2048),
        "pat": random.choice(["^/api/v2/(.*)", "^/old-path(.*)"]),
        "tmpfile": f"/tmp/nginx-body-{_uid()[:6]}",
    }
    result = template
    for key, val in replacements.items():
        result = result.replace("{" + key + "}", str(val))
    return result


# ────────────────────────────────────────────────────────────────────
# 1. Log Generation
# ────────────────────────────────────────────────────────────────────

# Pre-built incident storylines injected into the log stream.
_INCIDENT_STORYLINES: List[List[Dict[str, Any]]] = [
    # --- Memory Leak Escalation (application + systemd) ---
    [
        {"offset_min": -90, "source": "application", "level": "INFO",
         "message": "health check OK — uptime 48.3h, heap 512MB",
         "metadata": {"host": "web-server-01", "request_id": None}},
        {"offset_min": -70, "source": "application", "level": "WARNING",
         "message": "memory usage growing: 1024MB (was 512MB 10 min ago)",
         "metadata": {"host": "web-server-01"}},
        {"offset_min": -50, "source": "application", "level": "WARNING",
         "message": "memory usage growing: 2048MB (was 1024MB 10 min ago)",
         "metadata": {"host": "web-server-01"}},
        {"offset_min": -35, "source": "application", "level": "ERROR",
         "message": "thread pool utilization at 98% — consider scaling",
         "metadata": {"host": "web-server-01"}},
        {"offset_min": -20, "source": "application", "level": "CRITICAL",
         "message": "application heap exhausted — OutOfMemoryError after 3584MB allocation",
         "metadata": {"host": "web-server-01"}},
        {"offset_min": -19, "source": "systemd", "level": "CRITICAL",
         "message": "kernel: Out of memory: Killed process 4821 (app-api), UID 1001",
         "metadata": {"host": "web-server-01", "unit": "app-api"}},
    ],
    # --- Database Connection Pool Exhaustion ---
    [
        {"offset_min": -80, "source": "postgresql", "level": "INFO",
         "message": "autovacuum: processing table public.sessions",
         "metadata": {"host": "db-primary"}},
        {"offset_min": -60, "source": "postgresql", "level": "WARNING",
         "message": "connection pool exhausted — 95/100 connections in use",
         "metadata": {"host": "db-primary"}},
        {"offset_min": -45, "source": "postgresql", "level": "WARNING",
         "message": "connection pool exhausted — 100/100 connections in use",
         "metadata": {"host": "db-primary"}},
        {"offset_min": -40, "source": "application", "level": "ERROR",
         "message": "database query timeout after 15000ms on table 'orders'",
         "metadata": {"host": "web-server-01"}},
        {"offset_min": -38, "source": "postgresql", "level": "ERROR",
         "message": 'too many connections for role "app_user"',
         "metadata": {"host": "db-primary"}},
        {"offset_min": -37, "source": "nginx", "level": "ERROR",
         "message": "upstream timed out (110: Connection timed out) for client 10.0.4.55",
         "metadata": {"host": "web-server-01", "upstream": "10.0.2.10:5432"}},
    ],
    # --- Kubernetes Pod Crash Loop ---
    [
        {"offset_min": -100, "source": "kubernetes", "level": "INFO",
         "message": 'successfully pulled image "infraheal/api:v2.0.0-rc1"',
         "metadata": {"host": "k8s-node-01", "pod": "api-7f8b9c", "namespace": "production"}},
        {"offset_min": -98, "source": "kubernetes", "level": "INFO",
         "message": "started container api for pod api-7f8b9c",
         "metadata": {"host": "k8s-node-01", "pod": "api-7f8b9c", "namespace": "production"}},
        {"offset_min": -95, "source": "kubernetes", "level": "ERROR",
         "message": "liveness probe failed for api-7f8b9c: HTTP probe failed with status 503",
         "metadata": {"host": "k8s-node-01", "pod": "api-7f8b9c", "namespace": "production"}},
        {"offset_min": -92, "source": "kubernetes", "level": "ERROR",
         "message": "container api in pod api-7f8b9c CrashLoopBackOff (exit code 137)",
         "metadata": {"host": "k8s-node-01", "pod": "api-7f8b9c", "namespace": "production"}},
        {"offset_min": -88, "source": "kubernetes", "level": "WARNING",
         "message": "back-off restarting failed container api in pod api-7f8b9c",
         "metadata": {"host": "k8s-node-01", "pod": "api-7f8b9c", "namespace": "production"}},
        {"offset_min": -85, "source": "docker", "level": "ERROR",
         "message": "OOM killed container a3f8c9e12b01 (used 2.3GB > limit 2GB)",
         "metadata": {"host": "k8s-node-01", "container_id": "a3f8c9e12b01"}},
    ],
    # --- Disk Filling on k8s-node-01 ---
    [
        {"offset_min": -75, "source": "systemd", "level": "INFO",
         "message": "started logrotate.service — Rotate log files",
         "metadata": {"host": "k8s-node-01", "unit": "logrotate"}},
        {"offset_min": -55, "source": "docker", "level": "WARNING",
         "message": "no space left on device for layer storage",
         "metadata": {"host": "k8s-node-01"}},
        {"offset_min": -40, "source": "kubernetes", "level": "WARNING",
         "message": "node k8s-node-01 has disk pressure: available 200Mi < threshold 1024Mi",
         "metadata": {"host": "k8s-node-01", "node": "k8s-node-01"}},
        {"offset_min": -30, "source": "docker", "level": "CRITICAL",
         "message": "daemon storage driver overlay2 failed: no space left on device",
         "metadata": {"host": "k8s-node-01"}},
    ],
]

_METADATA_FIELDS_BY_SOURCE: Dict[str, List[str]] = {
    "nginx": ["host", "upstream", "request_id", "status_code"],
    "postgresql": ["host", "database", "query_id", "duration_ms"],
    "redis": ["host", "db_index", "client_ip"],
    "kubernetes": ["host", "pod", "namespace", "container", "node"],
    "docker": ["host", "container_id", "image"],
    "systemd": ["host", "unit", "pid"],
    "auth": ["host", "user", "ip", "session_id"],
    "application": ["host", "request_id", "endpoint", "trace_id"],
}


def _build_metadata(source: str) -> Dict[str, Any]:
    """Build a plausible metadata dict for a given log source."""
    fields = _METADATA_FIELDS_BY_SOURCE.get(source, ["host"])
    meta: Dict[str, Any] = {}
    for f in fields:
        if f == "host":
            meta[f] = random.choice(HOSTS)
        elif f in ("upstream", "client_ip", "ip"):
            meta[f] = _rand_ip()
        elif f in ("request_id", "query_id", "trace_id", "session_id"):
            meta[f] = _uid()
        elif f == "status_code":
            meta[f] = random.choice([200, 201, 301, 400, 404, 500, 502])
        elif f == "database":
            meta[f] = random.choice(["infraheal_prod", "infraheal_analytics"])
        elif f == "duration_ms":
            meta[f] = round(random.uniform(0.5, 5000), 1)
        elif f == "db_index":
            meta[f] = random.randint(0, 15)
        elif f == "pod":
            meta[f] = f"api-{_uid()[:6]}"
        elif f == "namespace":
            meta[f] = random.choice(["production", "staging"])
        elif f == "container":
            meta[f] = random.choice(["api", "worker", "sidecar"])
        elif f == "node":
            meta[f] = random.choice(HOSTS)
        elif f == "container_id":
            meta[f] = _uid()[:12]
        elif f == "image":
            meta[f] = random.choice(["infraheal/api:latest", "redis:7", "nginx:1.25"])
        elif f == "unit":
            meta[f] = random.choice(["nginx", "postgresql", "app-api"])
        elif f == "pid":
            meta[f] = random.randint(1000, 65535)
        elif f == "user":
            meta[f] = random.choice(["admin", "deploy-bot", "jdoe"])
        elif f == "endpoint":
            meta[f] = random.choice(["/api/v1/users", "/api/v1/orders"])
        else:
            meta[f] = _uid()
    return meta


def generate_system_logs(count: int = 5000, seed: int = SEED) -> List[Dict[str, Any]]:
    """Generate *count* realistic syslog-style log entries spanning the last 2 hours.

    Includes injected incident storylines that form coherent escalation patterns
    amid a background of normal log traffic.

    Returns:
        List of log dicts conforming to the canonical format:
        ``{timestamp, source, level, service, message, metadata}``.
    """
    rng = random.Random(seed)
    random.seed(seed)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=2)

    logs: List[Dict[str, Any]] = []

    # --- Inject storyline logs first ---
    for storyline in _INCIDENT_STORYLINES:
        for entry in storyline:
            ts = now + timedelta(minutes=entry["offset_min"])
            meta = entry.get("metadata", {})
            if "host" not in meta:
                meta["host"] = rng.choice(HOSTS)
            logs.append({
                "timestamp": ts.isoformat(),
                "source": entry["source"],
                "level": entry["level"],
                "service": entry["source"],
                "message": entry["message"],
                "metadata": meta,
            })

    storyline_count = len(logs)
    remaining = count - storyline_count

    # --- Generate background noise logs ---
    for _ in range(max(0, remaining)):
        source = rng.choice(SOURCES)
        level = rng.choices(LEVELS, weights=LEVEL_WEIGHTS, k=1)[0]
        templates = _LOG_TEMPLATES.get(source, {}).get(level, [])
        if not templates:
            # fall back to INFO
            templates = _LOG_TEMPLATES[source].get("INFO", ["operational"])
        msg = _fill_template(rng.choice(templates))
        ts = start + timedelta(seconds=rng.uniform(0, 7200))
        meta = _build_metadata(source)
        logs.append({
            "timestamp": ts.isoformat(),
            "source": source,
            "level": level,
            "service": source,
            "message": msg,
            "metadata": meta,
        })

    # Sort chronologically
    logs.sort(key=lambda l: l["timestamp"])
    logger.info("Generated %d logs (%d from storylines)", len(logs), storyline_count)
    return logs


# ────────────────────────────────────────────────────────────────────
# 2. Metric Generation
# ────────────────────────────────────────────────────────────────────

# Baseline profiles per host  (mean, std)
_HOST_BASELINES: Dict[str, Dict[str, tuple]] = {
    "web-server-01": {"cpu": (35, 6), "mem": (55, 4), "disk": (45, 1),
                      "net_in": (120, 20), "net_out": (80, 15),
                      "latency": (150, 40), "error_rate": (0.02, 0.01),
                      "connections": (250, 50)},
    "web-server-02": {"cpu": (30, 5), "mem": (50, 4), "disk": (42, 1),
                      "net_in": (100, 18), "net_out": (70, 12),
                      "latency": (140, 35), "error_rate": (0.01, 0.005),
                      "connections": (200, 40)},
    "db-primary":    {"cpu": (45, 8), "mem": (65, 5), "disk": (60, 2),
                      "net_in": (80, 15), "net_out": (60, 10),
                      "latency": (50, 15), "error_rate": (0.005, 0.003),
                      "connections": (80, 20)},
    "db-replica":    {"cpu": (25, 5), "mem": (55, 5), "disk": (58, 2),
                      "net_in": (60, 10), "net_out": (40, 8),
                      "latency": (45, 12), "error_rate": (0.003, 0.002),
                      "connections": (40, 10)},
    "cache-server-01": {"cpu": (20, 4), "mem": (70, 3), "disk": (30, 1),
                        "net_in": (150, 25), "net_out": (150, 25),
                        "latency": (5, 2), "error_rate": (0.001, 0.001),
                        "connections": (500, 80)},
    "k8s-node-01":   {"cpu": (40, 7), "mem": (60, 5), "disk": (50, 2),
                      "net_in": (90, 15), "net_out": (70, 12),
                      "latency": (120, 30), "error_rate": (0.015, 0.008),
                      "connections": (300, 60)},
}

# Anomaly injection windows: (host, metric, start_offset_min, duration_min, behaviour)
_ANOMALY_WINDOWS = [
    # CPU spike on web-server-01
    {
        "host": "web-server-01", "metric": "cpu",
        "start_min": 50, "dur_min": 20,
        "transform": lambda base, t, dur: min(99, base + 55 * math.sin(math.pi * t / dur)),
        "label": "CPU spike",
    },
    # Memory leak on db-primary (gradual ramp)
    {
        "host": "db-primary", "metric": "mem",
        "start_min": 30, "dur_min": 60,
        "transform": lambda base, t, dur: min(98, base + 30 * (t / dur)),
        "label": "Memory leak",
    },
    # Disk filling on k8s-node-01 (linear climb)
    {
        "host": "k8s-node-01", "metric": "disk",
        "start_min": 20, "dur_min": 80,
        "transform": lambda base, t, dur: min(97, base + 45 * (t / dur)),
        "label": "Disk filling",
    },
    # Latency spike on web-server-01 (correlated with CPU spike)
    {
        "host": "web-server-01", "metric": "latency",
        "start_min": 52, "dur_min": 18,
        "transform": lambda base, t, dur: base + 4500 * math.sin(math.pi * t / dur),
        "label": "Latency spike",
    },
]


def generate_metrics(
    duration_minutes: int = 120,
    interval_seconds: int = 30,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Generate time-series infrastructure metrics for all hosts.

    Normal baselines are perturbed with Gaussian noise.  Several anomaly
    windows are injected to produce detectable deviations.

    Returns:
        List of metric dicts conforming to the canonical format.
    """
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=duration_minutes)
    total_points = (duration_minutes * 60) // interval_seconds

    metrics: List[Dict[str, Any]] = []

    for host, baselines in _HOST_BASELINES.items():
        for i in range(total_points):
            elapsed_min = (i * interval_seconds) / 60.0
            ts = start + timedelta(seconds=i * interval_seconds)

            cpu_mean, cpu_std = baselines["cpu"]
            mem_mean, mem_std = baselines["mem"]
            disk_mean, disk_std = baselines["disk"]
            net_in_mean, net_in_std = baselines["net_in"]
            net_out_mean, net_out_std = baselines["net_out"]
            lat_mean, lat_std = baselines["latency"]
            err_mean, err_std = baselines["error_rate"]
            conn_mean, conn_std = baselines["connections"]

            cpu = max(0, rng.gauss(cpu_mean, cpu_std))
            mem = max(0, rng.gauss(mem_mean, mem_std))
            disk = max(0, rng.gauss(disk_mean, disk_std))
            net_in = max(0, rng.gauss(net_in_mean, net_in_std))
            net_out = max(0, rng.gauss(net_out_mean, net_out_std))
            latency = max(0, rng.gauss(lat_mean, lat_std))
            error_rate = max(0, rng.gauss(err_mean, err_std))
            connections = max(0, rng.gauss(conn_mean, conn_std))

            # Apply anomaly windows
            for aw in _ANOMALY_WINDOWS:
                if aw["host"] != host:
                    continue
                aw_start = aw["start_min"]
                aw_end = aw_start + aw["dur_min"]
                if aw_start <= elapsed_min <= aw_end:
                    t = elapsed_min - aw_start
                    dur = aw["dur_min"]
                    m = aw["metric"]
                    if m == "cpu":
                        cpu = aw["transform"](cpu_mean, t, dur)
                    elif m == "mem":
                        mem = aw["transform"](mem_mean, t, dur)
                    elif m == "disk":
                        disk = aw["transform"](disk_mean, t, dur)
                    elif m == "latency":
                        latency = aw["transform"](lat_mean, t, dur)
                    # Correlated side-effects
                    if m == "cpu" and cpu > 80:
                        error_rate = max(error_rate, 0.10 + rng.uniform(0, 0.15))
                        connections = connections * 1.5
                    if m == "mem" and mem > 85:
                        latency = latency * 2
                        error_rate = max(error_rate, 0.08 + rng.uniform(0, 0.10))

            metrics.append({
                "timestamp": ts.isoformat(),
                "host": host,
                "cpu_percent": round(min(cpu, 100), 2),
                "memory_percent": round(min(mem, 100), 2),
                "disk_percent": round(min(disk, 100), 2),
                "network_in_mbps": round(max(net_in, 0), 2),
                "network_out_mbps": round(max(net_out, 0), 2),
                "request_latency_ms": round(max(latency, 0), 2),
                "error_rate": round(min(max(error_rate, 0), 1.0), 4),
                "active_connections": int(max(connections, 0)),
            })

    metrics.sort(key=lambda m: (m["timestamp"], m["host"]))
    logger.info("Generated %d metric data-points across %d hosts", len(metrics), len(HOSTS))
    return metrics


# ────────────────────────────────────────────────────────────────────
# 3. Runbook Generation
# ────────────────────────────────────────────────────────────────────

_RUNBOOK_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "title": "High CPU Utilization",
        "category": "infrastructure",
        "symptoms": [
            "CPU usage consistently above 90%",
            "Increased request latency",
            "Application timeouts and 504 errors",
            "System load average > number of cores",
        ],
        "root_causes": [
            "Runaway process or infinite loop",
            "Insufficient compute resources for workload",
            "Missing rate-limiting on API endpoints",
            "Crypto-mining malware",
        ],
        "resolution_steps": [
            "1. Run `top -c` or `htop` to identify the offending process",
            "2. Check if the process is legitimate: `ps aux | grep <PID>`",
            "3. If runaway: `kill -15 <PID>`, then `kill -9` if unresponsive",
            "4. Scale horizontally: add more replicas via orchestrator",
            "5. Apply CPU resource limits in container/pod spec",
            "6. Review recent deployments for regressions",
        ],
        "prevention": [
            "Set CPU resource limits and requests in Kubernetes manifests",
            "Implement auto-scaling policies",
            "Add rate-limiting and request throttling",
            "Run periodic load tests before deployments",
        ],
        "tags": ["cpu", "performance", "infrastructure", "scaling"],
    },
    {
        "title": "Memory Leak",
        "category": "application",
        "symptoms": [
            "Memory usage steadily increasing over time",
            "OOM Killer events in system logs",
            "Application restarts due to memory limits",
            "Increasing GC pause times",
        ],
        "root_causes": [
            "Object references not released (retained caches, listeners)",
            "Connection pool or thread pool leak",
            "Growing in-memory data structures without bounds",
            "Native memory leak in JNI / C extensions",
        ],
        "resolution_steps": [
            "1. Capture heap dump: `jmap -dump:live,format=b,file=heap.hprof <PID>`",
            "2. Analyze with Eclipse MAT or VisualVM",
            "3. Identify dominator tree and retained objects",
            "4. Restart affected service as immediate mitigation",
            "5. Deploy fix for the leaking code path",
            "6. Set memory limits and configure OOM behaviour",
        ],
        "prevention": [
            "Implement bounded caches with TTL eviction",
            "Monitor memory trends in dashboards",
            "Run memory profiling in CI pipelines",
            "Set container memory limits with appropriate OOM behaviour",
        ],
        "tags": ["memory", "leak", "oom", "application", "performance"],
    },
    {
        "title": "Disk Full",
        "category": "storage",
        "symptoms": [
            "Disk usage above 90%",
            "'No space left on device' errors",
            "Failed log writes and database operations",
            "Container image pulls failing",
        ],
        "root_causes": [
            "Unbounded log growth without rotation",
            "Large temporary files or core dumps",
            "Database WAL or binlog not being cleaned",
            "Docker images / layers accumulating",
        ],
        "resolution_steps": [
            "1. Identify large files: `du -sh /* | sort -rh | head -20`",
            "2. Clear old logs: `find /var/log -name '*.gz' -mtime +7 -delete`",
            "3. Clean Docker: `docker system prune -a --volumes`",
            "4. Rotate and compress active logs",
            "5. Remove old database backups and WAL segments",
            "6. Expand volume if persistent",
        ],
        "prevention": [
            "Configure logrotate for all services",
            "Set up disk-usage alerts at 80% threshold",
            "Implement automated cleanup cron jobs",
            "Use separate volumes for data and logs",
        ],
        "tags": ["disk", "storage", "cleanup", "logs"],
    },
    {
        "title": "Service Crash / Unexpected Restart",
        "category": "application",
        "symptoms": [
            "Service process exits unexpectedly",
            "Systemd shows 'failed' state for unit",
            "Repeated restart attempts in logs",
            "Downstream services report connection errors",
        ],
        "root_causes": [
            "Unhandled exception or segmentation fault",
            "Out-of-memory kill by the kernel",
            "Configuration error after deployment",
            "Dependency service unavailable at startup",
        ],
        "resolution_steps": [
            "1. Check service status: `systemctl status <service>`",
            "2. Review logs: `journalctl -u <service> --since '1 hour ago'`",
            "3. Check for core dumps in /var/lib/systemd/coredump/",
            "4. Review recent config or deployment changes",
            "5. Restart with increased logging: set DEBUG level",
            "6. Rollback deployment if regression identified",
        ],
        "prevention": [
            "Implement health checks and readiness probes",
            "Add graceful shutdown handlers",
            "Test configuration changes in staging first",
            "Set resource limits to prevent OOM kills",
        ],
        "tags": ["crash", "service", "restart", "systemd"],
    },
    {
        "title": "Database Connection Pool Exhaustion",
        "category": "database",
        "symptoms": [
            "Application errors: 'too many connections'",
            "Connection timeouts to database",
            "Slow queries and increased latency",
            "Thread pool starvation in application",
        ],
        "root_causes": [
            "Connection leak — connections not returned to pool",
            "Slow queries holding connections too long",
            "Too many application replicas for pool size",
            "Connection idle timeout misconfiguration",
        ],
        "resolution_steps": [
            "1. Check active connections: `SELECT count(*) FROM pg_stat_activity;`",
            "2. Identify idle connections: `SELECT * FROM pg_stat_activity WHERE state='idle';`",
            "3. Kill idle connections: `SELECT pg_terminate_backend(pid);`",
            "4. Increase `max_connections` in postgresql.conf if needed",
            "5. Deploy connection pooler (PgBouncer) in front of database",
            "6. Fix application connection leak in code",
        ],
        "prevention": [
            "Use a connection pooler (PgBouncer, ProxySQL)",
            "Set connection pool max-lifetime and idle-timeout",
            "Monitor connection count with alerts",
            "Implement connection leak detection in tests",
        ],
        "tags": ["database", "postgresql", "connections", "pool"],
    },
    {
        "title": "SSL Certificate Expiry",
        "category": "security",
        "symptoms": [
            "Browser shows 'certificate expired' warning",
            "TLS handshake failures in logs",
            "API clients rejecting HTTPS connections",
            "Monitoring alerts for certificate age",
        ],
        "root_causes": [
            "Certificate renewal automation failure",
            "ACME / Let's Encrypt renewal not configured",
            "DNS validation records missing for renewal",
            "Certificate stored in wrong path after renewal",
        ],
        "resolution_steps": [
            "1. Check certificate expiry: `openssl x509 -enddate -noout -in cert.pem`",
            "2. Renew certificate: `certbot renew` or request from CA",
            "3. Deploy new certificate to all endpoints",
            "4. Reload web server: `nginx -s reload`",
            "5. Verify with `curl -vI https://your-domain.com`",
            "6. Test from external monitoring service",
        ],
        "prevention": [
            "Automate certificate renewal with cert-manager or certbot",
            "Set up alerts for certificates expiring within 30 days",
            "Use short-lived certificates with automated rotation",
            "Maintain certificate inventory",
        ],
        "tags": ["ssl", "tls", "certificate", "security", "https"],
    },
    {
        "title": "DDoS Attack Mitigation",
        "category": "security",
        "symptoms": [
            "Massive spike in inbound traffic",
            "All backends returning 503/504",
            "Connection table exhaustion",
            "Legitimate users unable to access service",
        ],
        "root_causes": [
            "Volumetric DDoS (UDP/TCP flood)",
            "Application-layer attack (HTTP flood, slowloris)",
            "DNS amplification attack",
            "Botnet targeting public endpoints",
        ],
        "resolution_steps": [
            "1. Enable DDoS protection at CDN/edge (Cloudflare, AWS Shield)",
            "2. Identify attack source IPs from access logs",
            "3. Block offending IPs/CIDRs at firewall: `iptables -A INPUT -s <IP> -j DROP`",
            "4. Enable rate-limiting on nginx/load balancer",
            "5. Scale up to absorb if traffic is semi-legitimate",
            "6. Engage ISP for upstream filtering if volumetric",
        ],
        "prevention": [
            "Use CDN with built-in DDoS protection",
            "Implement rate-limiting and CAPTCHA for public endpoints",
            "Configure SYN cookies and connection limits",
            "Have a DDoS incident response runbook drilled regularly",
        ],
        "tags": ["ddos", "security", "network", "firewall", "traffic"],
    },
    {
        "title": "DNS Resolution Failure",
        "category": "network",
        "symptoms": [
            "Services unable to resolve internal hostnames",
            "'Name or service not known' errors",
            "Intermittent connectivity to external APIs",
            "CoreDNS / kube-dns pods unhealthy",
        ],
        "root_causes": [
            "DNS server overloaded or crashed",
            "Incorrect /etc/resolv.conf configuration",
            "Network policy blocking DNS (port 53)",
            "Upstream DNS provider outage",
        ],
        "resolution_steps": [
            "1. Test resolution: `dig @<dns-server> <hostname>`",
            "2. Check DNS pod status: `kubectl get pods -n kube-system -l k8s-app=kube-dns`",
            "3. Review DNS pod logs: `kubectl logs -n kube-system <coredns-pod>`",
            "4. Verify resolv.conf: `cat /etc/resolv.conf`",
            "5. Restart DNS pods: `kubectl rollout restart deploy coredns -n kube-system`",
            "6. Switch to backup DNS resolver if primary is down",
        ],
        "prevention": [
            "Deploy DNS with multiple replicas and anti-affinity",
            "Monitor DNS query latency and failure rate",
            "Configure node-level DNS caching (NodeLocal DNSCache)",
            "Maintain fallback DNS resolvers",
        ],
        "tags": ["dns", "network", "resolution", "coredns", "kubernetes"],
    },
    {
        "title": "Deployment Rollback",
        "category": "application",
        "symptoms": [
            "Error rate spike after deployment",
            "New version returning 500 errors",
            "Health checks failing for new pods",
            "Increased latency after release",
        ],
        "root_causes": [
            "Bug introduced in new code",
            "Database schema incompatibility",
            "Missing environment variable or secret",
            "Dependency version conflict",
        ],
        "resolution_steps": [
            "1. Confirm regression: compare error rates before/after deploy",
            "2. Rollback deployment: `kubectl rollout undo deployment/<name>`",
            "3. Verify rollback: `kubectl rollout status deployment/<name>`",
            "4. Check previous version health checks pass",
            "5. Investigate root cause in staging with new version",
            "6. Hotfix and re-deploy when ready",
        ],
        "prevention": [
            "Implement canary/blue-green deployments",
            "Automate rollback on error-rate threshold breach",
            "Run comprehensive integration tests before release",
            "Keep deployment artifacts versioned and immutable",
        ],
        "tags": ["deployment", "rollback", "release", "kubernetes"],
    },
    {
        "title": "Container OOM Kill",
        "category": "infrastructure",
        "symptoms": [
            "Container exits with code 137",
            "OOMKilled reason in pod status",
            "dmesg shows 'Out of memory: Killed process'",
            "Application becomes unresponsive before crash",
        ],
        "root_causes": [
            "Container memory limit set too low",
            "Memory leak in application",
            "JVM heap sized larger than container limit",
            "In-memory caching without bounds",
        ],
        "resolution_steps": [
            "1. Check OOM status: `kubectl describe pod <pod> | grep -A5 'Last State'`",
            "2. Review memory limit vs actual usage",
            "3. Increase memory limit if under-provisioned",
            "4. If leak: capture heap dump and profile",
            "5. Tune JVM: set `-Xmx` to 75% of container limit",
            "6. Restart pod with corrected limits",
        ],
        "prevention": [
            "Set memory requests = limits (QoS Guaranteed)",
            "Profile memory usage before setting limits",
            "Implement bounded caches and connection pools",
            "Monitor container memory with Prometheus",
        ],
        "tags": ["oom", "container", "kubernetes", "memory", "137"],
    },
    {
        "title": "Network Partition",
        "category": "network",
        "symptoms": [
            "Split-brain in clustered services",
            "Intermittent connectivity between nodes",
            "Raft/consensus timeouts in distributed databases",
            "Some nodes see each other, others do not",
        ],
        "root_causes": [
            "Switch/router failure in one rack",
            "Misconfigured firewall rules or security groups",
            "MTU mismatch causing jumbo-frame drops",
            "Cloud provider AZ network issue",
        ],
        "resolution_steps": [
            "1. Verify connectivity: `ping`, `traceroute`, `mtr` between affected nodes",
            "2. Check switch/router status and interface errors",
            "3. Review firewall rules: `iptables -L -n` / security groups",
            "4. Test MTU: `ping -M do -s 1472 <target>`",
            "5. Failover affected services to healthy partition",
            "6. Engage network team / cloud provider support",
        ],
        "prevention": [
            "Deploy across multiple availability zones",
            "Use redundant network paths (bonding/LACP)",
            "Monitor inter-node latency and packet loss",
            "Test network partition scenarios in chaos engineering",
        ],
        "tags": ["network", "partition", "split-brain", "connectivity"],
    },
    {
        "title": "Slow Database Queries",
        "category": "database",
        "symptoms": [
            "Query response time above SLA thresholds",
            "Growing number of active queries in pg_stat_activity",
            "Application timeouts on database calls",
            "Lock wait timeouts increasing",
        ],
        "root_causes": [
            "Missing or stale indexes",
            "Table bloat requiring VACUUM",
            "Query plan regression after statistics update",
            "Lock contention from long transactions",
        ],
        "resolution_steps": [
            "1. Enable slow query log: `log_min_duration_statement = 1000`",
            "2. Identify slow queries: `SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;`",
            "3. Run EXPLAIN ANALYZE on problematic queries",
            "4. Create missing indexes based on query patterns",
            "5. Run VACUUM ANALYZE on bloated tables",
            "6. Optimize or rewrite inefficient queries",
        ],
        "prevention": [
            "Schedule regular VACUUM and ANALYZE jobs",
            "Monitor query performance with pg_stat_statements",
            "Set up slow query alerts",
            "Review query plans in CI for critical queries",
        ],
        "tags": ["database", "slow-query", "postgresql", "performance", "index"],
    },
    {
        "title": "Cache Miss Storm (Thundering Herd)",
        "category": "application",
        "symptoms": [
            "Sudden spike in database load",
            "Cache hit ratio drops dramatically",
            "All requests hitting origin simultaneously",
            "Service latency spike after cache flush or restart",
        ],
        "root_causes": [
            "Cache TTL expiry for hot keys at the same time",
            "Cache server restart or failover",
            "Cache invalidation of frequently accessed data",
            "Missing cache-aside pattern for cold starts",
        ],
        "resolution_steps": [
            "1. Verify cache status: `redis-cli INFO stats | grep hit`",
            "2. Implement request coalescing / singleflight for hot keys",
            "3. Pre-warm cache with critical data",
            "4. Apply jittered TTL to prevent synchronized expiry",
            "5. Scale database temporarily to handle surge",
            "6. Implement stale-while-revalidate cache strategy",
        ],
        "prevention": [
            "Use jittered TTL: base_ttl + random(0, ttl * 0.1)",
            "Implement cache warming on deployment",
            "Use multi-tier caching (local + distributed)",
            "Deploy request coalescing middleware",
        ],
        "tags": ["cache", "redis", "thundering-herd", "performance"],
    },
    {
        "title": "Log Rotation Failure",
        "category": "storage",
        "symptoms": [
            "Log files growing unbounded",
            "Disk filling up in /var/log",
            "logrotate errors in cron output",
            "Application performance degrading from large log writes",
        ],
        "root_causes": [
            "logrotate configuration syntax error",
            "Application holding file descriptor of rotated log",
            "Insufficient disk space for compressed archive",
            "Permission issues on log directory",
        ],
        "resolution_steps": [
            "1. Test logrotate config: `logrotate -d /etc/logrotate.d/<service>`",
            "2. Force rotation: `logrotate -f /etc/logrotate.d/<service>`",
            "3. Signal application to reopen log files: `kill -USR1 <PID>`",
            "4. Fix permissions: `chown syslog:adm /var/log/<service>/`",
            "5. Manually compress and truncate oversized logs",
            "6. Verify cron job for logrotate is active",
        ],
        "prevention": [
            "Use stdout/stderr logging with container log drivers",
            "Test logrotate config in deployment pipeline",
            "Monitor log file sizes with alerts",
            "Implement application-level log rotation (e.g., RotatingFileHandler)",
        ],
        "tags": ["logs", "rotation", "disk", "logrotate", "storage"],
    },
    {
        "title": "Authentication Service Failure",
        "category": "security",
        "symptoms": [
            "All login attempts failing",
            "401/403 errors across services",
            "SSO/LDAP connection errors in logs",
            "JWT validation failures",
        ],
        "root_causes": [
            "Auth service crashed or out of memory",
            "LDAP / Active Directory server unreachable",
            "JWT signing key rotation failure",
            "Database backend for auth service is down",
        ],
        "resolution_steps": [
            "1. Check auth service health: `curl http://auth-svc:8080/health`",
            "2. Verify LDAP connectivity: `ldapsearch -H ldap://<host> -x`",
            "3. Restart auth service: `systemctl restart auth-service`",
            "4. Verify JWT signing keys are valid and accessible",
            "5. Check auth database connectivity",
            "6. Temporarily enable bypass for critical services if needed",
        ],
        "prevention": [
            "Deploy auth service with high availability (3+ replicas)",
            "Implement token caching to survive short outages",
            "Monitor auth service health and latency",
            "Maintain emergency access procedures",
        ],
        "tags": ["auth", "security", "login", "ldap", "jwt"],
    },
    {
        "title": "Load Balancer 502 Bad Gateway",
        "category": "network",
        "symptoms": [
            "Users seeing 502 errors",
            "Intermittent connectivity to backend services",
            "Load balancer health checks failing",
            "No healthy upstream servers",
        ],
        "root_causes": [
            "All backend servers down or unhealthy",
            "Backend servers overloaded and not responding in time",
            "Health check misconfiguration",
            "Network connectivity issue between LB and backends",
        ],
        "resolution_steps": [
            "1. Check upstream health: `curl -I http://<backend>:<port>/health`",
            "2. Review LB config: verify upstream server list and ports",
            "3. Check backend service logs for errors",
            "4. Increase upstream timeout if backends are slow",
            "5. Restart unhealthy backend instances",
            "6. Scale up backend pool if under capacity",
        ],
        "prevention": [
            "Configure proper health check endpoints",
            "Set appropriate timeout and retry values",
            "Implement circuit breakers between LB and backends",
            "Use auto-scaling for backend services",
        ],
        "tags": ["load-balancer", "502", "nginx", "network", "upstream"],
    },
    {
        "title": "Disk I/O Bottleneck",
        "category": "infrastructure",
        "symptoms": [
            "High I/O wait (%iowait) in CPU stats",
            "Slow disk reads and writes",
            "Database queries slow despite low CPU",
            "Application throughput degradation",
        ],
        "root_causes": [
            "Too many concurrent disk operations",
            "HDD instead of SSD for high-IOPS workloads",
            "Noisy neighbour on shared storage",
            "Filesystem fragmentation",
        ],
        "resolution_steps": [
            "1. Check I/O stats: `iostat -xz 1 5`",
            "2. Identify I/O-heavy processes: `iotop -o`",
            "3. Move hot data to faster storage tier (SSD/NVMe)",
            "4. Tune I/O scheduler: `echo deadline > /sys/block/sda/queue/scheduler`",
            "5. Offload logging to separate volume",
            "6. Implement write batching and async I/O in application",
        ],
        "prevention": [
            "Use SSD/NVMe for database and high-IOPS workloads",
            "Monitor disk I/O metrics and set alerts",
            "Separate data and log volumes",
            "Provision IOPS according to workload requirements",
        ],
        "tags": ["disk", "io", "performance", "storage", "iops"],
    },
    {
        "title": "NFS Mount Stale",
        "category": "storage",
        "symptoms": [
            "'Stale file handle' errors",
            "Applications hanging on file operations",
            "df command hangs for NFS mounts",
            "Timeouts accessing shared storage",
        ],
        "root_causes": [
            "NFS server rebooted without client remount",
            "Network interruption between client and NFS server",
            "NFS export configuration changed",
            "Server-side filesystem error",
        ],
        "resolution_steps": [
            "1. Check NFS server status: `showmount -e <nfs-server>`",
            "2. Try lazy unmount: `umount -l /mnt/nfs`",
            "3. Remount: `mount -a` or `mount <nfs-server>:/export /mnt/nfs`",
            "4. If hung: force unmount `umount -f /mnt/nfs`",
            "5. Restart NFS client services: `systemctl restart nfs-client.target`",
            "6. Verify NFS exports on server: `exportfs -v`",
        ],
        "prevention": [
            "Use `soft,timeo=30,retrans=3` mount options",
            "Monitor NFS mount health with synthetic checks",
            "Implement autofs for automatic remount on failure",
            "Consider object storage for large-scale shared data",
        ],
        "tags": ["nfs", "storage", "mount", "stale", "filesystem"],
    },
    {
        "title": "Kubernetes Pod CrashLoopBackOff",
        "category": "infrastructure",
        "symptoms": [
            "Pod status: CrashLoopBackOff",
            "Container repeatedly exiting with non-zero code",
            "Exponential back-off delays between restarts",
            "Pod restart count rapidly increasing",
        ],
        "root_causes": [
            "Application crash on startup (misconfiguration)",
            "Missing required environment variables or secrets",
            "Liveness probe misconfigured (too aggressive)",
            "OOM kill due to insufficient memory limits",
        ],
        "resolution_steps": [
            "1. Check pod events: `kubectl describe pod <pod>`",
            "2. View container logs: `kubectl logs <pod> --previous`",
            "3. Check exit code: 137 = OOM, 1 = app error, 127 = binary not found",
            "4. Verify env vars and secrets: `kubectl get secret <name> -o yaml`",
            "5. Adjust liveness probe: increase initialDelaySeconds and timeoutSeconds",
            "6. Fix root cause and redeploy",
        ],
        "prevention": [
            "Set appropriate resource requests and limits",
            "Configure startup probes for slow-starting containers",
            "Validate manifests with kube-linter before deployment",
            "Test container startup locally with same env vars",
        ],
        "tags": ["kubernetes", "crashloop", "pod", "container", "restart"],
    },
    {
        "title": "Redis Connection Refused",
        "category": "database",
        "symptoms": [
            "'Connection refused' errors to Redis port 6379",
            "Cache operations failing across services",
            "Session store unavailable",
            "Application fallback to database causing overload",
        ],
        "root_causes": [
            "Redis process crashed or was killed",
            "Max connections limit reached",
            "Network policy blocking port 6379",
            "Redis running out of memory (OOM protection enabled)",
        ],
        "resolution_steps": [
            "1. Check Redis status: `redis-cli ping` or `systemctl status redis`",
            "2. Review Redis logs: `tail -100 /var/log/redis/redis-server.log`",
            "3. Check connections: `redis-cli INFO clients | grep connected_clients`",
            "4. Restart Redis: `systemctl restart redis`",
            "5. Verify memory: `redis-cli INFO memory`",
            "6. Check network policies / firewall: `ss -tlnp | grep 6379`",
        ],
        "prevention": [
            "Deploy Redis Sentinel or Cluster for HA",
            "Set `maxmemory-policy allkeys-lru`",
            "Monitor Redis memory and connection count",
            "Implement connection pooling in application clients",
        ],
        "tags": ["redis", "cache", "connection", "database"],
    },
]


def generate_runbooks(count: int = 20, seed: int = SEED) -> List[Dict[str, Any]]:
    """Generate *count* operational runbooks.

    Returns:
        List of runbook dicts conforming to the canonical format.
    """
    random.seed(seed)
    runbooks: List[Dict[str, Any]] = []
    definitions = _RUNBOOK_DEFINITIONS[:count]
    for idx, defn in enumerate(definitions, start=1):
        runbooks.append({
            "id": f"RB-{idx:03d}",
            "title": defn["title"],
            "category": defn["category"],
            "symptoms": defn["symptoms"],
            "root_causes": defn["root_causes"],
            "resolution_steps": defn["resolution_steps"],
            "prevention": defn["prevention"],
            "tags": defn["tags"],
        })
    logger.info("Generated %d runbooks", len(runbooks))
    return runbooks


# ────────────────────────────────────────────────────────────────────
# 4. Past Incidents
# ────────────────────────────────────────────────────────────────────

_PAST_INCIDENT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "title": "Web Server CPU Spike Causing 504 Timeouts",
        "severity": "P1", "category": "infrastructure",
        "root_cause": "Runaway regex evaluation in request validation middleware",
        "resolution": "Deployed hotfix to replace catastrophic backtracking regex; restarted nginx workers",
        "duration_minutes": 45, "affected_services": ["nginx", "application"],
    },
    {
        "title": "Database Primary OOM Kill",
        "severity": "P1", "category": "database",
        "root_cause": "Unbounded query result set from missing LIMIT clause in analytics job",
        "resolution": "Killed offending query, added LIMIT clause, increased shared_buffers",
        "duration_minutes": 30, "affected_services": ["postgresql", "application"],
    },
    {
        "title": "Redis Cache Failure — Session Loss",
        "severity": "P2", "category": "database",
        "root_cause": "Redis exceeded maxmemory with noeviction policy; all SET commands rejected",
        "resolution": "Changed eviction policy to allkeys-lru; flushed stale session keys",
        "duration_minutes": 20, "affected_services": ["redis", "auth", "application"],
    },
    {
        "title": "Certificate Expiry on API Gateway",
        "severity": "P2", "category": "security",
        "root_cause": "Certbot renewal cron disabled during maintenance and never re-enabled",
        "resolution": "Manually renewed certificate and restored cron job",
        "duration_minutes": 60, "affected_services": ["nginx", "application"],
    },
    {
        "title": "Kubernetes Node NotReady",
        "severity": "P1", "category": "infrastructure",
        "root_cause": "Kubelet crashed due to disk pressure; docker overlay2 consumed all inode space",
        "resolution": "Cleaned dangling images and containers; increased inode monitoring threshold",
        "duration_minutes": 55, "affected_services": ["kubernetes", "docker", "application"],
    },
    {
        "title": "Slow Query Cascade in Order Service",
        "severity": "P3", "category": "database",
        "root_cause": "Missing index on orders.customer_id after schema migration",
        "resolution": "Created btree index; ran ANALYZE on orders table",
        "duration_minutes": 90, "affected_services": ["postgresql", "application"],
    },
    {
        "title": "Auth Service Unreachable",
        "severity": "P1", "category": "security",
        "root_cause": "LDAP server ran out of file descriptors; all bind requests rejected",
        "resolution": "Increased ulimit on LDAP server; restarted slapd daemon",
        "duration_minutes": 25, "affected_services": ["auth", "application"],
    },
    {
        "title": "Docker Storage Driver Failure",
        "severity": "P2", "category": "storage",
        "root_cause": "Overlay2 filesystem corruption after unclean node shutdown during kernel update",
        "resolution": "Rebuilt overlay2 storage; re-pulled affected container images",
        "duration_minutes": 120, "affected_services": ["docker", "kubernetes"],
    },
    {
        "title": "Network Partition Between AZs",
        "severity": "P1", "category": "network",
        "root_cause": "Misconfigured security group rule blocked inter-AZ traffic on port range 30000-32767",
        "resolution": "Reverted security group change; verified all NodePort services accessible",
        "duration_minutes": 40, "affected_services": ["kubernetes", "application", "nginx"],
    },
    {
        "title": "Log Volume Disk Full",
        "severity": "P3", "category": "storage",
        "root_cause": "Debug logging accidentally left enabled in production after troubleshooting session",
        "resolution": "Set log level back to INFO; rotated and compressed old logs; freed 45GB",
        "duration_minutes": 15, "affected_services": ["application", "systemd"],
    },
    {
        "title": "DDoS on Public API Endpoint",
        "severity": "P1", "category": "security",
        "root_cause": "HTTP flood from botnet targeting /api/v1/search endpoint (200k req/s)",
        "resolution": "Enabled Cloudflare Under Attack mode; blocked offending ASN; added rate-limit rule",
        "duration_minutes": 35, "affected_services": ["nginx", "application"],
    },
    {
        "title": "NFS Stale Mount on Worker Nodes",
        "severity": "P3", "category": "storage",
        "root_cause": "NFS server kernel panic during scheduled patching; clients got stale handles",
        "resolution": "Lazy-unmounted stale mounts; remounted after NFS server recovery",
        "duration_minutes": 50, "affected_services": ["kubernetes", "application"],
    },
    {
        "title": "Deployment Rollback After Error Spike",
        "severity": "P2", "category": "application",
        "root_cause": "Breaking database migration in v2.1.0 dropped a NOT NULL constraint needed by API",
        "resolution": "Rolled back to v2.0.9; applied corrective migration in v2.1.1",
        "duration_minutes": 25, "affected_services": ["application", "postgresql"],
    },
    {
        "title": "Cache Thundering Herd After Redis Restart",
        "severity": "P2", "category": "application",
        "root_cause": "All cache keys expired simultaneously after Redis restart; database overwhelmed",
        "resolution": "Pre-warmed cache with critical keys; implemented jittered TTL",
        "duration_minutes": 30, "affected_services": ["redis", "postgresql", "application"],
    },
    {
        "title": "Container CrashLoopBackOff on New Deployment",
        "severity": "P2", "category": "infrastructure",
        "root_cause": "New image referenced non-existent secret for DB_PASSWORD env var",
        "resolution": "Created missing Kubernetes secret; pod started successfully on next retry",
        "duration_minutes": 15, "affected_services": ["kubernetes", "application"],
    },
    # ── Application-level incidents ───────────────────────────────
    {
        "title": "Malformed JSON Payload Causing 400 Errors Across API",
        "severity": "P2", "category": "application",
        "root_cause": "Client SDK shipped with broken serializer sending double-escaped JSON",
        "resolution": "Pinned SDK version to previous stable; deployed API validation middleware",
        "duration_minutes": 90, "affected_services": ["api-gateway", "application"],
    },
    {
        "title": "Order Service CRUD INSERT Failure After Schema Migration",
        "severity": "P1", "category": "database",
        "root_cause": "Migration added NOT NULL column without default; existing INSERT paths missing new field",
        "resolution": "Rolled back migration; deployed app update with new field; re-ran migration with default",
        "duration_minutes": 55, "affected_services": ["postgresql", "order-service", "application"],
    },
    {
        "title": "Payment Microservice Unreachable — Consul Service Mesh Split Brain",
        "severity": "P1", "category": "application",
        "root_cause": "Consul datacenter partition left payment-svc registered only in minority DC",
        "resolution": "Restored WAN gossip between Consul datacenters; re-registered service endpoints",
        "duration_minutes": 35, "affected_services": ["consul", "payment-svc", "api-gateway"],
    },
    {
        "title": "User Validation Errors Surge After Business Logic Update",
        "severity": "P3", "category": "application",
        "root_cause": "New input validation rule for tax ID format too strict; rejecting valid customer data",
        "resolution": "Relaxed regex to match broader tax ID formats; added unit tests for edge cases",
        "duration_minutes": 120, "affected_services": ["user-service", "application"],
    },
    {
        "title": "Data Corruption in Product Catalog — Duplicate Primary Key Violations",
        "severity": "P2", "category": "database",
        "root_cause": "Idempotency key missing on product upsert; concurrent requests created duplicate rows",
        "resolution": "Removed duplicates with CTE; added UNIQUE constraint and upsert logic",
        "duration_minutes": 40, "affected_services": ["postgresql", "catalog-service"],
    },
    {
        "title": "Content-Type Mismatch on File Upload Service",
        "severity": "P3", "category": "application",
        "root_cause": "Frontend sent multipart/form-data but API expected application/octet-stream",
        "resolution": "Updated API contract to accept both Content-Types; added content negotiation",
        "duration_minutes": 30, "affected_services": ["api-gateway", "upload-service"],
    },
    {
        "title": "Inventory Service gRPC Deadline Exceeded Under Load",
        "severity": "P2", "category": "application",
        "root_cause": "gRPC client timeout (500ms) too aggressive for inventory lookup that takes 800ms at p99",
        "resolution": "Increased deadline to 2s with jittered retry; added server-side caching for hot SKUs",
        "duration_minutes": 60, "affected_services": ["inventory-svc", "order-service"],
    },
    {
        "title": "Deadlock Chain on Order-Item Table During Flash Sale",
        "severity": "P1", "category": "database",
        "root_cause": "Two-phase DELETE + INSERT in transaction acquiring locks in conflicting orders",
        "resolution": "Applied consistent lock ordering; reduced transaction isolation to READ COMMITTED",
        "duration_minutes": 25, "affected_services": ["postgresql", "order-service", "payment-svc"],
    },
]


def generate_past_incidents(count: int = 15, seed: int = SEED) -> List[Dict[str, Any]]:
    """Generate historical incident records.

    Returns:
        List of incident dicts with id, timestamp, title, severity, category,
        root_cause, resolution, duration_minutes, affected_services.
    """
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    incidents: List[Dict[str, Any]] = []
    templates = _PAST_INCIDENT_TEMPLATES[:count]

    for idx, tpl in enumerate(templates, start=1):
        # Spread incidents over last 90 days
        ts = now - timedelta(days=rng.randint(1, 90), hours=rng.randint(0, 23),
                             minutes=rng.randint(0, 59))
        incidents.append({
            "id": f"INC-{idx:03d}",
            "timestamp": ts.isoformat(),
            "title": tpl["title"],
            "severity": tpl["severity"],
            "category": tpl["category"],
            "root_cause": tpl["root_cause"],
            "resolution": tpl["resolution"],
            "duration_minutes": tpl["duration_minutes"],
            "affected_services": tpl["affected_services"],
        })

    incidents.sort(key=lambda i: i["timestamp"])
    logger.info("Generated %d past incidents", len(incidents))
    return incidents


# ────────────────────────────────────────────────────────────────────
# 5. Full Data Bundle
# ────────────────────────────────────────────────────────────────────

def generate_all_data(
    save_to_disk: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate all synthetic data and optionally persist to JSON files.

    Args:
        save_to_disk: Whether to write JSON files to *output_dir*.
        output_dir: Destination directory (defaults to ``config.DATA_DIR``).

    Returns:
        Dict with keys ``logs``, ``metrics``, ``runbooks``, ``past_incidents``.
    """
    data: Dict[str, Any] = {
        "logs": generate_system_logs(),
        "metrics": generate_metrics(),
        "runbooks": generate_runbooks(),
        "past_incidents": generate_past_incidents(),
    }

    if save_to_disk:
        dest = Path(output_dir) if output_dir else DATA_DIR
        dest.mkdir(parents=True, exist_ok=True)
        for key, payload in data.items():
            filepath = dest / f"{key}.json"
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            logger.info("Saved %s → %s (%d records)", key, filepath, len(payload))

    return data


# ────────────────────────────────────────────────────────────────────
# 6. Pre-built Demo Scenarios
# ────────────────────────────────────────────────────────────────────

def create_incident_scenarios(seed: int = SEED) -> List[Dict[str, Any]]:
    """Return five curated incident scenarios for live demos.

    Each scenario provides a coherent slice of anomalous logs and metrics
    that together tell a realistic story an SRE would investigate.

    Returns:
        List of scenario dicts, each with:
        ``id, name, description, logs, metrics, expected_severity,
        expected_category, expected_root_cause``.
    """
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)

    scenarios: List[Dict[str, Any]] = []

    # ── Scenario 1: Memory Leak Causing OOM ──────────────────────
    s1_logs = [
        {"timestamp": (now - timedelta(minutes=90)).isoformat(), "source": "application",
         "level": "INFO", "service": "application",
         "message": "health check OK — uptime 72.4h, heap 480MB",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=60)).isoformat(), "source": "application",
         "level": "WARNING", "service": "application",
         "message": "memory usage growing: 1200MB (was 480MB 10 min ago)",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=40)).isoformat(), "source": "application",
         "level": "WARNING", "service": "application",
         "message": "memory usage growing: 2400MB (was 1200MB 10 min ago)",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=25)).isoformat(), "source": "application",
         "level": "ERROR", "service": "application",
         "message": "thread pool utilization at 99% — consider scaling",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=20)).isoformat(), "source": "application",
         "level": "CRITICAL", "service": "application",
         "message": "application heap exhausted — OutOfMemoryError after 3800MB allocation",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=19)).isoformat(), "source": "systemd",
         "level": "CRITICAL", "service": "systemd",
         "message": "kernel: Out of memory: Killed process 4821 (app-api), UID 1001",
         "metadata": {"host": "web-server-01", "unit": "app-api"}},
        {"timestamp": (now - timedelta(minutes=18)).isoformat(), "source": "nginx",
         "level": "ERROR", "service": "nginx",
         "message": "connect() failed (111: Connection refused) while connecting to upstream 127.0.0.1:8080",
         "metadata": {"host": "web-server-01"}},
    ]
    s1_metrics = []
    for i in range(180):  # 90 min, 30s intervals
        t = now - timedelta(minutes=90) + timedelta(seconds=i * 30)
        elapsed = i * 30 / 60  # minutes
        mem = 55 + 40 * (elapsed / 90)
        cpu = 35 + (20 * (elapsed / 90) if elapsed > 30 else 0)
        s1_metrics.append({
            "timestamp": t.isoformat(), "host": "web-server-01",
            "cpu_percent": round(min(cpu + rng.gauss(0, 3), 100), 2),
            "memory_percent": round(min(mem + rng.gauss(0, 2), 99.5), 2),
            "disk_percent": round(45 + rng.gauss(0, 1), 2),
            "network_in_mbps": round(120 + rng.gauss(0, 15), 2),
            "network_out_mbps": round(80 + rng.gauss(0, 10), 2),
            "request_latency_ms": round(max(150 + (elapsed / 90) * 3000 + rng.gauss(0, 50), 10), 2),
            "error_rate": round(min(max(0.02 + (elapsed / 90) * 0.25, 0), 1), 4),
            "active_connections": int(250 + elapsed * 5 + rng.gauss(0, 20)),
        })
    scenarios.append({
        "id": "SCEN-001",
        "name": "Memory Leak → OOM Kill",
        "description": ("Gradual memory leak in the application on web-server-01 leads to heap "
                         "exhaustion, OOM kill by the kernel, and cascading upstream errors in nginx."),
        "logs": s1_logs,
        "metrics": s1_metrics,
        "expected_severity": "P1",
        "expected_category": "application",
        "expected_root_cause": "Memory leak in application causing OOM kill",
    })

    # ── Scenario 2: Database Connection Pool Exhaustion ──────────
    s2_logs = [
        {"timestamp": (now - timedelta(minutes=70)).isoformat(), "source": "postgresql",
         "level": "WARNING", "service": "postgresql",
         "message": "connection pool exhausted — 90/100 connections in use",
         "metadata": {"host": "db-primary"}},
        {"timestamp": (now - timedelta(minutes=55)).isoformat(), "source": "postgresql",
         "level": "WARNING", "service": "postgresql",
         "message": "connection pool exhausted — 100/100 connections in use",
         "metadata": {"host": "db-primary"}},
        {"timestamp": (now - timedelta(minutes=50)).isoformat(), "source": "postgresql",
         "level": "ERROR", "service": "postgresql",
         "message": 'too many connections for role "app_user"',
         "metadata": {"host": "db-primary"}},
        {"timestamp": (now - timedelta(minutes=48)).isoformat(), "source": "application",
         "level": "ERROR", "service": "application",
         "message": "database query timeout after 15000ms on table 'orders'",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=45)).isoformat(), "source": "application",
         "level": "ERROR", "service": "application",
         "message": "circuit breaker OPEN for downstream service 'db-primary'",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=44)).isoformat(), "source": "nginx",
         "level": "ERROR", "service": "nginx",
         "message": "upstream timed out (110: Connection timed out) for client 10.5.3.22",
         "metadata": {"host": "web-server-01"}},
    ]
    s2_metrics = []
    for i in range(140):
        t = now - timedelta(minutes=70) + timedelta(seconds=i * 30)
        elapsed = i * 30 / 60
        conn_factor = min(elapsed / 70, 1.0)
        s2_metrics.append({
            "timestamp": t.isoformat(), "host": "db-primary",
            "cpu_percent": round(45 + 35 * conn_factor + rng.gauss(0, 4), 2),
            "memory_percent": round(65 + 15 * conn_factor + rng.gauss(0, 3), 2),
            "disk_percent": round(60 + rng.gauss(0, 1), 2),
            "network_in_mbps": round(80 + 60 * conn_factor + rng.gauss(0, 10), 2),
            "network_out_mbps": round(60 + 40 * conn_factor + rng.gauss(0, 8), 2),
            "request_latency_ms": round(max(50 + 8000 * conn_factor + rng.gauss(0, 30), 5), 2),
            "error_rate": round(min(0.005 + 0.20 * conn_factor, 1), 4),
            "active_connections": int(80 + 120 * conn_factor + rng.gauss(0, 5)),
        })
    scenarios.append({
        "id": "SCEN-002",
        "name": "Database Connection Pool Exhaustion",
        "description": ("Connections to db-primary are gradually consumed until the pool is "
                         "fully exhausted, causing query timeouts and circuit breaker activation."),
        "logs": s2_logs,
        "metrics": s2_metrics,
        "expected_severity": "P1",
        "expected_category": "database",
        "expected_root_cause": "Database connection pool exhaustion due to connection leak",
    })

    # ── Scenario 3: Disk Full on K8s Node ────────────────────────
    s3_logs = [
        {"timestamp": (now - timedelta(minutes=80)).isoformat(), "source": "systemd",
         "level": "WARNING", "service": "systemd",
         "message": "unit logrotate.service entered 'failed' state — will retry in 30s",
         "metadata": {"host": "k8s-node-01", "unit": "logrotate"}},
        {"timestamp": (now - timedelta(minutes=60)).isoformat(), "source": "docker",
         "level": "WARNING", "service": "docker",
         "message": "no space left on device for layer storage",
         "metadata": {"host": "k8s-node-01"}},
        {"timestamp": (now - timedelta(minutes=45)).isoformat(), "source": "kubernetes",
         "level": "WARNING", "service": "kubernetes",
         "message": "node k8s-node-01 has disk pressure: available 150Mi < threshold 1024Mi",
         "metadata": {"host": "k8s-node-01", "node": "k8s-node-01"}},
        {"timestamp": (now - timedelta(minutes=35)).isoformat(), "source": "kubernetes",
         "level": "WARNING", "service": "kubernetes",
         "message": "evicting pod worker-a8b2c1 due to node memory pressure",
         "metadata": {"host": "k8s-node-01", "pod": "worker-a8b2c1"}},
        {"timestamp": (now - timedelta(minutes=25)).isoformat(), "source": "docker",
         "level": "CRITICAL", "service": "docker",
         "message": "daemon storage driver overlay2 failed: no space left on device",
         "metadata": {"host": "k8s-node-01"}},
    ]
    s3_metrics = []
    for i in range(160):
        t = now - timedelta(minutes=80) + timedelta(seconds=i * 30)
        elapsed = i * 30 / 60
        disk = 50 + 47 * (elapsed / 80)
        s3_metrics.append({
            "timestamp": t.isoformat(), "host": "k8s-node-01",
            "cpu_percent": round(40 + rng.gauss(0, 5), 2),
            "memory_percent": round(60 + rng.gauss(0, 4), 2),
            "disk_percent": round(min(disk + rng.gauss(0, 1), 99), 2),
            "network_in_mbps": round(90 + rng.gauss(0, 12), 2),
            "network_out_mbps": round(70 + rng.gauss(0, 10), 2),
            "request_latency_ms": round(max(120 + rng.gauss(0, 25), 5), 2),
            "error_rate": round(min(max(0.015 + (disk / 100) * 0.05, 0), 1), 4),
            "active_connections": int(300 + rng.gauss(0, 40)),
        })
    scenarios.append({
        "id": "SCEN-003",
        "name": "Disk Full on Kubernetes Node",
        "description": ("Log rotation failure causes unbounded log growth on k8s-node-01, "
                         "filling the disk and triggering pod evictions and Docker storage failure."),
        "logs": s3_logs,
        "metrics": s3_metrics,
        "expected_severity": "P1",
        "expected_category": "storage",
        "expected_root_cause": "Log rotation failure causing disk space exhaustion",
    })

    # ── Scenario 4: Kubernetes CrashLoopBackOff ──────────────────
    s4_logs = [
        {"timestamp": (now - timedelta(minutes=50)).isoformat(), "source": "kubernetes",
         "level": "INFO", "service": "kubernetes",
         "message": 'successfully pulled image "infraheal/api:v2.1.0"',
         "metadata": {"host": "k8s-node-01", "pod": "api-d4e5f6", "namespace": "production"}},
        {"timestamp": (now - timedelta(minutes=49)).isoformat(), "source": "kubernetes",
         "level": "INFO", "service": "kubernetes",
         "message": "started container api for pod api-d4e5f6",
         "metadata": {"host": "k8s-node-01", "pod": "api-d4e5f6", "namespace": "production"}},
        {"timestamp": (now - timedelta(minutes=48)).isoformat(), "source": "application",
         "level": "ERROR", "service": "application",
         "message": "request abc123: unhandled exception in AuthHandler.verify: KeyError: 'DB_PASSWORD'",
         "metadata": {"host": "k8s-node-01"}},
        {"timestamp": (now - timedelta(minutes=47)).isoformat(), "source": "kubernetes",
         "level": "ERROR", "service": "kubernetes",
         "message": "liveness probe failed for api-d4e5f6: HTTP probe failed with status 503",
         "metadata": {"host": "k8s-node-01", "pod": "api-d4e5f6", "namespace": "production"}},
        {"timestamp": (now - timedelta(minutes=45)).isoformat(), "source": "kubernetes",
         "level": "ERROR", "service": "kubernetes",
         "message": "container api in pod api-d4e5f6 CrashLoopBackOff (exit code 137)",
         "metadata": {"host": "k8s-node-01", "pod": "api-d4e5f6", "namespace": "production"}},
        {"timestamp": (now - timedelta(minutes=43)).isoformat(), "source": "kubernetes",
         "level": "WARNING", "service": "kubernetes",
         "message": "back-off restarting failed container api in pod api-d4e5f6",
         "metadata": {"host": "k8s-node-01", "pod": "api-d4e5f6", "namespace": "production"}},
        {"timestamp": (now - timedelta(minutes=40)).isoformat(), "source": "docker",
         "level": "ERROR", "service": "docker",
         "message": "OOM killed container d4e5f67890ab (used 2.3GB > limit 2GB)",
         "metadata": {"host": "k8s-node-01", "container_id": "d4e5f67890ab"}},
    ]
    s4_metrics = []
    for i in range(100):
        t = now - timedelta(minutes=50) + timedelta(seconds=i * 30)
        spike = 1 if (i % 20) < 5 else 0  # periodic CPU spikes from restarts
        s4_metrics.append({
            "timestamp": t.isoformat(), "host": "k8s-node-01",
            "cpu_percent": round(40 + spike * 40 + rng.gauss(0, 5), 2),
            "memory_percent": round(60 + spike * 25 + rng.gauss(0, 4), 2),
            "disk_percent": round(50 + rng.gauss(0, 1), 2),
            "network_in_mbps": round(90 + rng.gauss(0, 12), 2),
            "network_out_mbps": round(70 + rng.gauss(0, 10), 2),
            "request_latency_ms": round(max(120 + spike * 2000 + rng.gauss(0, 30), 5), 2),
            "error_rate": round(min(max(0.015 + spike * 0.15, 0), 1), 4),
            "active_connections": int(300 + rng.gauss(0, 40)),
        })
    scenarios.append({
        "id": "SCEN-004",
        "name": "Kubernetes Pod CrashLoopBackOff",
        "description": ("New deployment v2.1.0 crashes on startup due to missing DB_PASSWORD secret. "
                         "The container enters CrashLoopBackOff with periodic OOM kills."),
        "logs": s4_logs,
        "metrics": s4_metrics,
        "expected_severity": "P2",
        "expected_category": "infrastructure",
        "expected_root_cause": "Missing Kubernetes secret for DB_PASSWORD environment variable",
    })

    # ── Scenario 5: Authentication Service Cascade Failure ───────
    s5_logs = [
        {"timestamp": (now - timedelta(minutes=35)).isoformat(), "source": "auth",
         "level": "ERROR", "service": "auth",
         "message": "LDAP connection refused: ldap://ldap.infraheal.internal:389",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=33)).isoformat(), "source": "auth",
         "level": "ERROR", "service": "auth",
         "message": "authentication failed for user 'jdoe': invalid credentials (IP: 10.4.5.12)",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=30)).isoformat(), "source": "auth",
         "level": "CRITICAL", "service": "auth",
         "message": "auth service unresponsive — all login requests failing (circuit breaker OPEN)",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=28)).isoformat(), "source": "application",
         "level": "ERROR", "service": "application",
         "message": "circuit breaker OPEN for downstream service 'auth-service'",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=25)).isoformat(), "source": "nginx",
         "level": "ERROR", "service": "nginx",
         "message": "connect() failed (111: Connection refused) while connecting to upstream 10.0.1.50:8080",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=22)).isoformat(), "source": "auth",
         "level": "CRITICAL", "service": "auth",
         "message": "brute force attack detected: 500 failed logins from 10.99.1.1 in 120s",
         "metadata": {"host": "web-server-01"}},
    ]
    s5_metrics = []
    for i in range(70):
        t = now - timedelta(minutes=35) + timedelta(seconds=i * 30)
        elapsed = i * 30 / 60
        err_factor = min(elapsed / 35, 1.0)
        s5_metrics.append({
            "timestamp": t.isoformat(), "host": "web-server-01",
            "cpu_percent": round(35 + 20 * err_factor + rng.gauss(0, 4), 2),
            "memory_percent": round(55 + rng.gauss(0, 3), 2),
            "disk_percent": round(45 + rng.gauss(0, 1), 2),
            "network_in_mbps": round(120 + 200 * err_factor + rng.gauss(0, 20), 2),
            "network_out_mbps": round(80 + rng.gauss(0, 12), 2),
            "request_latency_ms": round(max(150 + 5000 * err_factor + rng.gauss(0, 50), 10), 2),
            "error_rate": round(min(0.02 + 0.40 * err_factor, 1), 4),
            "active_connections": int(250 + 500 * err_factor + rng.gauss(0, 30)),
        })
    scenarios.append({
        "id": "SCEN-005",
        "name": "Authentication Service Cascade Failure",
        "description": ("LDAP server becomes unreachable, causing the auth service to fail. "
                         "All login requests are rejected, circuit breakers open, and a concurrent "
                         "brute-force attack is detected."),
        "logs": s5_logs,
        "metrics": s5_metrics,
        "expected_severity": "P1",
        "expected_category": "security",
        "expected_root_cause": "LDAP server unreachable causing authentication cascade failure",
    })

    # ── Scenario 6: Malformed API Requests Causing 400 Errors ────
    s6_logs = [
        {"timestamp": (now - timedelta(minutes=60)).isoformat(), "source": "api-gateway",
         "level": "ERROR", "service": "api-gateway",
         "message": "POST /api/v2/orders — 400 Bad Request: Unrecognized token 'npe': was expecting (JSON valid, 'value' 'string' 'number' 'object' 'array')",
         "metadata": {"host": "api-gw-01", "endpoint": "/api/v2/orders", "status_code": 400}},
        {"timestamp": (now - timedelta(minutes=55)).isoformat(), "source": "api-gateway",
         "level": "ERROR", "service": "api-gateway",
         "message": "POST /api/v2/orders — 400 Bad Request (x42 in 60s): JSON parse failure — string value contains double-escaped quotes",
         "metadata": {"host": "api-gw-01", "endpoint": "/api/v2/orders"}},
        {"timestamp": (now - timedelta(minutes=50)).isoformat(), "source": "api-gateway",
         "level": "WARNING", "service": "api-gateway",
         "message": "Content-Type mismatch: client sent 'text/html' for endpoint expecting 'application/json' (Client: mobile-app/3.2.1)",
         "metadata": {"host": "api-gw-01"}},
        {"timestamp": (now - timedelta(minutes=45)).isoformat(), "source": "application",
         "level": "WARNING", "service": "application",
         "message": "order-submission error rate at 38% — threshold 5% — likely client-side serializer bug",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=40)).isoformat(), "source": "api-gateway",
         "level": "CRITICAL", "service": "api-gateway",
         "message": "P1 Alert: 400 error rate on /api/v2/orders at 62% — upstream clients receiving failures",
         "metadata": {"host": "api-gw-01"}},
    ]
    s6_metrics = []
    for i in range(120):
        t = now - timedelta(minutes=60) + timedelta(seconds=i * 30)
        spike = 1 if i > 30 else 0
        s6_metrics.append({
            "timestamp": t.isoformat(), "host": "api-gw-01",
            "cpu_percent": round(35 + spike * 25 + rng.gauss(0, 4), 2),
            "memory_percent": round(50 + spike * 10 + rng.gauss(0, 3), 2),
            "disk_percent": round(40 + rng.gauss(0, 1), 2),
            "network_in_mbps": round(200 + spike * 300 + rng.gauss(0, 25), 2),
            "network_out_mbps": round(150 + spike * 50 + rng.gauss(0, 15), 2),
            "request_latency_ms": round(max(80 + spike * 200 + rng.gauss(0, 20), 5), 2),
            "error_rate": round(min(0.01 + spike * 0.35, 1), 4),
            "active_connections": int(400 + spike * 600 + rng.gauss(0, 50)),
        })
    scenarios.append({
        "id": "SCEN-006",
        "name": "Malformed API Requests Flood",
        "description": ("A client SDK update introduces a JSON serialization bug, causing all "
                         "order submission requests to carry malformed payloads. API gateway returns "
                         "400 errors, error rate spikes to 62%, and upstream clients experience failures."),
        "logs": s6_logs,
        "metrics": s6_metrics,
        "expected_severity": "P2",
        "expected_category": "application",
        "expected_root_cause": "Client SDK JSON serializer bug causing malformed API requests",
    })

    # ── Scenario 7: Payment Microservice Unreachable ─────────────
    s7_logs = [
        {"timestamp": (now - timedelta(minutes=45)).isoformat(), "source": "consul",
         "level": "WARNING", "service": "consul",
         "message": "wan federation link to dc-dr lost — gossip timeout after 30s",
         "metadata": {"host": "consul-server-01"}},
        {"timestamp": (now - timedelta(minutes=40)).isoformat(), "source": "consul",
         "level": "ERROR", "service": "consul",
         "message": "service 'payment-svc' deregistered in dc-main — no healthy instances in local datacenter",
         "metadata": {"host": "consul-server-01", "service": "payment-svc"}},
        {"timestamp": (now - timedelta(minutes=38)).isoformat(), "source": "order-service",
         "level": "ERROR", "service": "order-service",
         "message": "circuit breaker OPEN for downstream 'payment-svc' after 15 consecutive failures",
         "metadata": {"host": "order-svc-01"}},
        {"timestamp": (now - timedelta(minutes=35)).isoformat(), "source": "order-service",
         "level": "ERROR", "service": "order-service",
         "message": "POST /api/v1/checkout — 503 Service Unavailable: payment-svc unreachable",
         "metadata": {"host": "order-svc-01"}},
        {"timestamp": (now - timedelta(minutes=30)).isoformat(), "source": "api-gateway",
         "level": "CRITICAL", "service": "api-gateway",
         "message": "checkout failure rate at 100% — all payment processing requests failing",
         "metadata": {"host": "api-gw-01"}},
        {"timestamp": (now - timedelta(minutes=25)).isoformat(), "source": "application",
         "level": "CRITICAL", "service": "application",
         "message": "revenue impact: 0 successful transactions in last 15 min — escalation triggered",
         "metadata": {"host": "web-server-01"}},
    ]
    s7_metrics = []
    for i in range(90):
        t = now - timedelta(minutes=45) + timedelta(seconds=i * 30)
        fail = 1 if i > 10 else 0
        s7_metrics.append({
            "timestamp": t.isoformat(), "host": "order-svc-01",
            "cpu_percent": round(30 + fail * 35 + rng.gauss(0, 4), 2),
            "memory_percent": round(45 + fail * 15 + rng.gauss(0, 3), 2),
            "disk_percent": round(35 + rng.gauss(0, 1), 2),
            "network_in_mbps": round(50 + fail * 10 + rng.gauss(0, 8), 2),
            "network_out_mbps": round(30 + fail * 5 + rng.gauss(0, 5), 2),
            "request_latency_ms": round(max(100 + fail * 3000 + rng.gauss(0, 50), 5), 2),
            "error_rate": round(min(0.005 + fail * 0.75, 1), 4),
            "active_connections": int(100 + fail * 50 + rng.gauss(0, 15)),
        })
    scenarios.append({
        "id": "SCEN-007",
        "name": "Payment Microservice Unreachable",
        "description": ("Consul WAN federation link between datacenters fails, causing payment-svc to be "
                         "deregistered in dc-main. Order service circuit breakers open, all checkouts fail, "
                         "and revenue impact escalates as no transactions complete."),
        "logs": s7_logs,
        "metrics": s7_metrics,
        "expected_severity": "P1",
        "expected_category": "application",
        "expected_root_cause": "Consul WAN gossip failure causing payment service deregistration",
    })

    # ── Scenario 8: Database CRUD INSERT Failure After Migration ─
    s8_logs = [
        {"timestamp": (now - timedelta(minutes=90)).isoformat(), "source": "application",
         "level": "INFO", "service": "application",
         "message": "schema migration v0042 applied: added 'tax_region' column (NOT NULL) to 'orders'",
         "metadata": {"host": "web-server-01"}},
        {"timestamp": (now - timedelta(minutes=85)).isoformat(), "source": "application",
         "level": "ERROR", "service": "application",
         "message": "INSERT into orders failed: null value in column 'tax_region' of relation 'orders' violates NOT NULL constraint",
         "metadata": {"host": "order-svc-01"}},
        {"timestamp": (now - timedelta(minutes=80)).isoformat(), "source": "application",
         "level": "ERROR", "service": "application",
         "message": "order submission failed for 12 consecutive requests — DB constraint violation",
         "metadata": {"host": "order-svc-01"}},
        {"timestamp": (now - timedelta(minutes=75)).isoformat(), "source": "postgresql",
         "level": "ERROR", "service": "postgresql",
         "message": "ERROR: null value in column 'tax_region' violates NOT NULL constraint (x47 in 5 min)",
         "metadata": {"host": "db-primary"}},
        {"timestamp": (now - timedelta(minutes=70)).isoformat(), "source": "api-gateway",
         "level": "WARNING", "service": "api-gateway",
         "message": "POST /api/v2/orders — 500 Internal Server Error: order creation failed",
         "metadata": {"host": "api-gw-01", "endpoint": "/api/v2/orders"}},
        {"timestamp": (now - timedelta(minutes=65)).isoformat(), "source": "application",
         "level": "CRITICAL", "service": "application",
         "message": "order-placement error rate at 100% — P1 incident declared",
         "metadata": {"host": "web-server-01"}},
    ]
    s8_metrics = []
    for i in range(180):
        t = now - timedelta(minutes=90) + timedelta(seconds=i * 30)
        fail = 1 if i > 10 else 0
        s8_metrics.append({
            "timestamp": t.isoformat(), "host": "db-primary",
            "cpu_percent": round(40 + fail * 30 + rng.gauss(0, 5), 2),
            "memory_percent": round(55 + fail * 10 + rng.gauss(0, 3), 2),
            "disk_percent": round(50 + rng.gauss(0, 1), 2),
            "network_in_mbps": round(60 + fail * 40 + rng.gauss(0, 10), 2),
            "network_out_mbps": round(40 + fail * 20 + rng.gauss(0, 8), 2),
            "request_latency_ms": round(max(50 + fail * 500 + rng.gauss(0, 15), 5), 2),
            "error_rate": round(min(0.002 + fail * 0.60, 1), 4),
            "active_connections": int(120 + fail * 80 + rng.gauss(0, 10)),
        })
    scenarios.append({
        "id": "SCEN-008",
        "name": "DB CRUD INSERT Failure After Migration",
        "description": ("A schema migration adds a NOT NULL column without a default value. The application "
                         "code has not been updated to include the new field in INSERT statements, causing "
                         "all order creation requests to fail with constraint violations."),
        "logs": s8_logs,
        "metrics": s8_metrics,
        "expected_severity": "P1",
        "expected_category": "database",
        "expected_root_cause": "Schema migration added NOT NULL column without default; app code not updated",
    })

    logger.info("Created %d demo scenarios", len(scenarios))
    return scenarios


# ────────────────────────────────────────────────────────────────────
# Standalone Testing
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("InfraHeal AI — Synthetic Data Generator")
    print("=" * 60)

    data = generate_all_data(save_to_disk=True)
    print(f"\n📋 Logs generated       : {len(data['logs']):,}")
    print(f"📊 Metric points        : {len(data['metrics']):,}")
    print(f"📖 Runbooks             : {len(data['runbooks'])}")
    print(f"🗂  Past incidents       : {len(data['past_incidents'])}")

    # Show level distribution
    from collections import Counter
    level_counts = Counter(log["level"] for log in data["logs"])
    print("\n  Log level distribution:")
    for lvl in LEVELS:
        print(f"    {lvl:10s}: {level_counts.get(lvl, 0):5d}")

    # Show host metric counts
    host_counts = Counter(m["host"] for m in data["metrics"])
    print("\n  Metrics per host:")
    for host, cnt in sorted(host_counts.items()):
        print(f"    {host:20s}: {cnt:5d}")

    # Scenarios
    scenarios = create_incident_scenarios()
    print(f"\n🎬 Demo scenarios       : {len(scenarios)}")
    for sc in scenarios:
        print(f"    [{sc['id']}] {sc['name']}  "
              f"(severity={sc['expected_severity']}, logs={len(sc['logs'])}, "
              f"metrics={len(sc['metrics'])})")

    print("\n✅ All data generated successfully.")
    print(f"   Files saved to: {DATA_DIR}")
