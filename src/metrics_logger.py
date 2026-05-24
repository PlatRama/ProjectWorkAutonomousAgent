from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


_COLUMNS = [
    "episode",
    "phase",
    "total_reward",
    "steps",
    "success",
    "collision",
    "timeout",
    "epsilon",
    "avg_loss",
    "avg_q",
    "global_train_steps",
]


class MetricsLogger:
    """Accumulates per-episode metrics and writes them to CSV + JSON summary."""

    def __init__(self, results_dir: str | Path, exp_name: str = "run"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name    = exp_name
        self.csv_path    = self.results_dir / "metrics.csv"
        self.json_path   = self.results_dir / "summary.json"
        self._rows: List[Dict[str, Any]] = []
        self._file_handle = None
        self._csv_writer  = None
        self._open_csv()

    # ── Writing ───────────────────────────────────────────────────────────────

    def _open_csv(self) -> None:
        """Open CSV file and write header if the file is new."""
        write_header = not self.csv_path.exists()
        self._file_handle = open(self.csv_path, "a", newline="")
        self._csv_writer  = csv.DictWriter(
            self._file_handle, fieldnames=_COLUMNS
        )
        if write_header:
            self._csv_writer.writeheader()

    def log(
        self,
        episode: int,
        phase: int,
        reward: float,
        steps: int,
        success: bool,
        collision: bool,
        timeout: bool,
        epsilon: float,
        loss: float,
        q_mean: float,
        global_train_steps: int,
    ) -> None:
        row = {
            "episode":            episode,
            "phase":              phase,
            "total_reward":       round(reward, 5),
            "steps":              steps,
            "success":            int(success),
            "collision":          int(collision),
            "timeout":            int(timeout),
            "epsilon":            round(epsilon, 6),
            "avg_loss":           round(loss, 6),
            "avg_q":              round(q_mean, 5),
            "global_train_steps": global_train_steps,
        }
        self._rows.append(row)
        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
            self._file_handle.flush()

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    # ── Reading ───────────────────────────────────────────────────────────────

    def to_dataframe(self):
        """Load the CSV into a pandas DataFrame (pandas must be installed)."""
        import pandas as pd
        return pd.read_csv(self.csv_path)

    def load_csv(self) -> List[Dict[str, Any]]:
        """Load the CSV without pandas (returns list of dicts)."""
        rows = []
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    k: (int(v) if k in ("episode", "phase", "steps", "success",
                                        "collision", "timeout", "global_train_steps")
                        else float(v))
                    for k, v in row.items()
                })
        return rows

    # ── Summaries ─────────────────────────────────────────────────────────────

    def compute_summary(self, last_n: int = 200) -> Dict[str, Any]:
        """Compute summary statistics over the last *last_n* episodes."""
        rows = self._rows if self._rows else self.load_csv()
        if not rows:
            return {}
        tail = rows[-last_n:]
        success_rate  = np.mean([r["success"]   for r in tail])
        collision_rate = np.mean([r["collision"] for r in tail])
        avg_reward    = np.mean([r["total_reward"] for r in tail])
        avg_steps     = np.mean([r["steps"]     for r in tail])
        return {
            "exp_name":       self.exp_name,
            "total_episodes": len(rows),
            "last_n":         last_n,
            "success_rate":   round(float(success_rate),  4),
            "collision_rate": round(float(collision_rate), 4),
            "avg_reward":     round(float(avg_reward),    4),
            "avg_steps":      round(float(avg_steps),     2),
        }

    def save_summary(self, last_n: int = 200, extra: Optional[Dict] = None) -> None:
        summary = self.compute_summary(last_n)
        if extra:
            summary.update(extra)
        with open(self.json_path, "w") as f:
            json.dump(summary, f, indent=2)

    # ── Smoothing helper ──────────────────────────────────────────────────────

    @staticmethod
    def rolling_mean(values: List[float], window: int) -> np.ndarray:
        """Compute rolling mean with valid convolution (output shorter than input)."""
        if len(values) < window:
            return np.array(values, dtype=float)
        arr    = np.array(values, dtype=float)
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="valid")

    @staticmethod
    def rolling_mean_padded(values: List[float], window: int) -> np.ndarray:
        """Rolling mean same length as input (uses cumulative mean for prefix)."""
        arr = np.array(values, dtype=float)
        out = np.empty_like(arr)
        for i in range(len(arr)):
            start   = max(0, i - window + 1)
            out[i]  = arr[start: i + 1].mean()
        return out

    def __del__(self):
        self.close()
