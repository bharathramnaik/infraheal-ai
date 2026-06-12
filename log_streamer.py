"""
InfraHeal AI — Log Streamer
=============================
Real-time log streaming module that replays pre-generated logs
through the anomaly detector and pipeline, simulating live ingestion.
"""

import json
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"


class LogStreamer:
    """Replays pre-generated logs in real-time with configurable speed.

    Can operate in two modes:
      1.  Stream from saved JSON files on disk.
      2.  Stream from an in-memory log list (e.g. scenario logs).
    """

    def __init__(
        self,
        logs: Optional[List[dict]] = None,
        replay_speed: float = 1.0,
        shuffle: bool = False,
    ):
        self._logs: List[dict] = logs or self._load_sample_logs()
        self._speed = replay_speed
        self._shuffle = shuffle
        self._index = 0
        self._start_time: Optional[datetime] = None
        self._paused = False
        self._processed_count = 0
        self._batch: List[dict] = []

    @staticmethod
    def _load_sample_logs() -> List[dict]:
        """Load logs from sample_data/logs.json."""
        path = SAMPLE_DATA_DIR / "logs.json"
        if not path.exists():
            logger.warning("sample_data/logs.json not found, using empty log list")
            return []
        with open(path) as f:
            data = json.load(f)
        data.sort(key=lambda x: x.get("timestamp", ""))
        logger.info("Loaded %d logs from %s", len(data), path)
        return data

    def reset(self) -> None:
        """Reset the stream to the beginning."""
        self._index = 0
        self._start_time = None
        self._paused = False
        self._processed_count = 0
        self._batch = []

    def set_speed(self, speed: float) -> None:
        """Set replay speed multiplier (1.0 = real-time)."""
        self._speed = max(0.1, speed)

    def pause(self) -> None:
        """Pause the stream."""
        self._paused = True

    def resume(self) -> None:
        """Resume the stream."""
        self._paused = False

    @property
    def is_running(self) -> bool:
        return not self._paused and self._index < len(self._logs)

    @property
    def progress(self) -> float:
        return self._index / max(len(self._logs), 1) * 100

    def stream(
        self, batch_size: int = 1, delay_per_log: float = 0.5
    ) -> Generator[List[dict], None, None]:
        """Yield batches of logs with realistic timing.

        Args:
            batch_size: Number of logs per yield.
            delay_per_log: Base seconds between logs (divided by speed).

        Yields:
            Lists of log dicts representing a time window.
        """
        self.reset()
        if not self._logs:
            return

        self._start_time = datetime.now(timezone.utc)

        # Determine the overall time span of the log data
        try:
            t0 = datetime.fromisoformat(self._logs[0]["timestamp"])
            t1 = datetime.fromisoformat(self._logs[-1]["timestamp"])
            total_span = (t1 - t0).total_seconds()
        except (ValueError, KeyError):
            total_span = max(len(self._logs) * delay_per_log, 1)

        if self._shuffle:
            indices = list(range(len(self._logs)))
            random.shuffle(indices)
            self._logs = [self._logs[i] for i in indices]

        while self._index < len(self._logs):
            if self._paused:
                time.sleep(0.1)
                continue

            batch = []
            batch_end = min(self._index + batch_size, len(self._logs))

            for i in range(self._index, batch_end):
                log = self._logs[i]
                # Calculate realistic delay based on actual log timestamps
                if total_span > 0 and i > 0:
                    try:
                        prev_ts = datetime.fromisoformat(self._logs[i - 1]["timestamp"])
                        curr_ts = datetime.fromisoformat(log["timestamp"])
                        real_gap = (curr_ts - prev_ts).total_seconds()
                    except (ValueError, KeyError):
                        real_gap = delay_per_log

                    adjusted_delay = max(0.01, real_gap / self._speed)
                    if adjusted_delay > 0.01 and len(batch) > 0:
                        time.sleep(adjusted_delay)
                else:
                    pass

                batch.append(log)
                self._processed_count += 1

            self._index = batch_end
            self._batch = batch
            yield batch

    def get_stats(self) -> dict:
        """Return streaming statistics."""
        elapsed = 0
        if self._start_time:
            elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        return {
            "total_logs": len(self._logs),
            "processed": self._processed_count,
            "remaining": max(0, len(self._logs) - self._index),
            "progress_pct": self.progress,
            "elapsed_seconds": round(elapsed, 1),
            "speed_multiplier": self._speed,
            "paused": self._paused,
        }


class LiveAnalyzer:
    """Processes live-streamed logs through anomaly detector + orchestrator.

    Accumulates logs over time windows and triggers pipeline runs
    when configurable thresholds are met.
    """

    def __init__(
        self,
        detector: Any,
        orchestrator: Optional[Any] = None,
        window_size: int = 100,
        error_threshold: int = 10,
    ):
        self._detector = detector
        self._orchestrator = orchestrator
        self._window_size = window_size
        self._error_threshold = error_threshold
        self._accumulated_logs: List[dict] = []
        self._accumulated_metrics: List[dict] = []
        self._anomaly_history: List[dict] = []
        self._pipeline_results: List[dict] = []

    def ingest(self, logs: List[dict]) -> dict:
        """Ingest a batch of logs and optionally trigger analysis.

        Returns:
            Status dict with keys: logs_ingested, anomalies_found,
            pipeline_triggered, current_window.
        """
        self._accumulated_logs.extend(logs)

        # Count error/critical logs in current window
        error_count = sum(
            1 for l in self._accumulated_logs[-self._window_size:]
            if l.get("level", "").upper() in ("ERROR", "CRITICAL")
        )

        result = {
            "logs_ingested": len(logs),
            "total_accumulated": len(self._accumulated_logs),
            "error_count": error_count,
            "pipeline_triggered": False,
            "window_full": len(self._accumulated_logs) >= self._window_size,
        }

        # Run anomaly detection periodically
        if len(self._accumulated_logs) >= self._window_size or error_count >= self._error_threshold:
            anomalies = self._detector.detect_all(
                logs=self._accumulated_logs,
                metrics=self._accumulated_metrics or None,
            )
            if anomalies:
                self._anomaly_history.extend(anomalies)
                result["anomalies_found"] = len(anomalies)
                result["anomalies"] = anomalies

                # Trigger orchestrator pipeline
                if self._orchestrator is not None:
                    try:
                        pipeline_result = self._orchestrator.process_incident(
                            anomalies=anomalies,
                            logs=self._accumulated_logs[-self._window_size * 2:],
                        )
                        self._pipeline_results.append(pipeline_result)
                        result["pipeline_triggered"] = True
                        result["pipeline_id"] = pipeline_result.get("pipeline_id", "")
                    except Exception as exc:
                        logger.error("Live pipeline failed: %s", exc)
                        result["pipeline_error"] = str(exc)
            else:
                result["anomalies_found"] = 0

            # Slide window
            if len(self._accumulated_logs) >= self._window_size * 2:
                self._accumulated_logs = self._accumulated_logs[-self._window_size:]

        return result

    def get_status(self) -> dict:
        """Return live analyzer status."""
        error_count = sum(
            1 for l in self._accumulated_logs
            if l.get("level", "").upper() in ("ERROR", "CRITICAL")
        )
        return {
            "accumulated_logs": len(self._accumulated_logs),
            "error_count": error_count,
            "anomalies_detected": len(self._anomaly_history),
            "pipelines_run": len(self._pipeline_results),
            "window_size": self._window_size,
        }

    def reset(self) -> None:
        """Clear accumulated state."""
        self._accumulated_logs = []
        self._accumulated_metrics = []
        self._anomaly_history = []
        self._pipeline_results = []
