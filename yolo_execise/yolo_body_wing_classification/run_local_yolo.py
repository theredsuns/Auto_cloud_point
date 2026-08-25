#!/usr/bin/env python3
"""Run the project's bundled CPU Ultralytics runtime without pip installation."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = str(ROOT / "libraries" / "yolo_runtime")
GPU_RUNTIME = str(ROOT / "libraries" / "yolo_gpu_runtime")
# PyTorch 2.13 needs SymPy >= 1.13.3, whereas Ubuntu supplies SymPy 1.9.
# Import the bundled compatible SymPy first, then append the runtime.  Appending
# keeps OpenCV on the system's NumPy 1.x instead of loading runtime NumPy 2.x.
sys.path.insert(0, RUNTIME)
import sympy  # noqa: E402,F401
sys.path.pop(0)
# When installed, the CUDA-enabled Torch runtime comes first; the remaining
# bundled Ultralytics dependencies stay in the CPU runtime directory.
if Path(GPU_RUNTIME).is_dir():
    sys.path.append(GPU_RUNTIME)
sys.path.append(RUNTIME)

from ultralytics.cfg import entrypoint  # noqa: E402
from ultralytics.engine.trainer import BaseTrainer  # noqa: E402


def read_results_csv_without_polars(self):
    """Use the standard library for checkpoint metadata when polars is absent."""
    import csv

    try:
        with open(self.csv, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        return {key.strip(): [row[key] for row in rows] for key in (rows[0] if rows else {})}
    except (OSError, csv.Error, KeyError):
        return {}


# Ultralytics only uses this data as checkpoint metadata.  The bundled runtime
# deliberately omits the optional polars dependency, so do not let that block
# saving best.pt after a successful training epoch.
BaseTrainer.read_results_csv = read_results_csv_without_polars


if __name__ == "__main__":
    entrypoint()
