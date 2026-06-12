"""
InfraHeal AI — GPU Memory Tracker
===================================
Monitors GPU memory usage via AMD ROCm `rocm-smi` or fallback psutil.
Captures peak memory, current usage, and temperature for dashboard metrics.

The `GPUMonitor` context manager wraps agent inference calls so the
dashboard can display live GPU metrics (Slide 4 of the presentation).
"""

import logging
import subprocess
import threading
import time
from typing import Any, Dict, Optional

try:
    import pynvml
    HAS_NVIDIA = True
except ImportError:
    HAS_NVIDIA = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


class GPUMonitor:
    """GPU memory and utilisation monitor.

    Works with both NVIDIA (via pynvml) and AMD (via rocm-smi) GPUs.
    Falls back to ``psutil`` virtual memory when no GPU is detected so
    the dashboard always shows *some* metric.

    Usage as context manager::

        with GPUMonitor() as mon:
            # … agent inference …
        print(mon.get_peak_memory_mb())
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._peak_mib: float = 0.0
        self._start_mib: float = 0.0
        self._current_mib: float = 0.0
        self._temperature: float = 0.0
        self._util_pct: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        """Begin periodic polling in a background thread."""
        self._sample()  # baseline
        self._start_mib = self._current_mib
        self._peak_mib = self._current_mib
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background poller."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()  # final reading

    # ── Public API ───────────────────────────────────────────────

    def get_current_memory_mb(self) -> float:
        return self._current_mib

    def get_peak_memory_mb(self) -> float:
        return self._peak_mib

    def get_used_memory_mb(self) -> float:
        return max(0.0, self._current_mib - self._start_mib)

    def get_temperature(self) -> float:
        return self._temperature

    def get_utilization(self) -> float:
        return self._util_pct

    def snapshot(self) -> Dict[str, Any]:
        """Return a dict suitable for dashboard metrics cards."""
        self._sample()
        return {
            "gpu_memory_mb": round(self._current_mib, 1),
            "gpu_peak_memory_mb": round(self._peak_mib, 1),
            "gpu_used_mb": round(self.get_used_memory_mb(), 1),
            "gpu_temperature_c": round(self._temperature, 1),
            "gpu_util_pct": round(self._util_pct, 1),
        }

    # ── Internal ─────────────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            self._sample()
            time.sleep(2)

    def _sample(self):
        mem, temp, util = self._read_gpu()
        self._current_mib = mem
        self._temperature = temp
        self._util_pct = util
        if mem > self._peak_mib:
            self._peak_mib = mem

    @staticmethod
    def _read_gpu() -> tuple:
        """Read GPU memory (MiB), temperature (C), util (%)."""
        # 1. Try AMD rocm-smi
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                stderr=subprocess.DEVNULL, timeout=5, text=True,
            )
            import json
            data = json.loads(out)
            for key, info in data.items():
                if key.startswith("card"):
                    total = float(info.get("VRAM Total", "0").replace(" MB", ""))
                    used = float(info.get("VRAM Used", "0").replace(" MB", ""))
                    return used, 0.0, 0.0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        # 2. Try NVIDIA pynvml
        if HAS_NVIDIA:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU,
                )
                used_mib = mem_info.used / (1024 * 1024)
                return used_mib, float(temp), float(util.gpu)
            except Exception:
                pass

        # 3. Fallback: system memory via psutil
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.used / (1024 * 1024), 0.0, mem.percent
        except ImportError:
            return 0.0, 0.0, 0.0
