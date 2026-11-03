"""
Timing harness: measure the real execution time of a Python code snippet.
Produces the ground-truth labels for training/evaluating duration predictors.

Each snippet is run in a fresh subprocess (realistic: a coding agent spawns a
process per tool call). We take the median of N runs to reduce noise.
"""
from __future__ import annotations
import subprocess
import sys
import time
import statistics
import tempfile
import os

PYTHON = sys.executable


def measure_snippet(code: str, runs: int = 5, timeout: float = 30.0) -> dict:
    """Run `code` in a subprocess `runs` times, return timing stats (seconds).
    Measures wall-clock of the subprocess including interpreter startup, which
    is what an agent's tool call actually pays."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        samples = []
        ok = True
        err = ""
        for _ in range(runs):
            t0 = time.perf_counter()
            try:
                p = subprocess.run([PYTHON, path], capture_output=True,
                                   timeout=timeout)
                dt = time.perf_counter() - t0
                if p.returncode != 0:
                    ok = False
                    err = p.stderr.decode()[-200:]
                    break
                samples.append(dt)
            except subprocess.TimeoutExpired:
                ok = False
                err = f"timeout>{timeout}s"
                samples.append(timeout)
                break
        return {
            "ok": ok,
            "error": err,
            "median_s": statistics.median(samples) if samples else None,
            "min_s": min(samples) if samples else None,
            "mean_s": statistics.mean(samples) if samples else None,
            "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "n": len(samples),
        }
    finally:
        os.unlink(path)


if __name__ == "__main__":
    code = sys.stdin.read()
    import json
    print(json.dumps(measure_snippet(code), indent=2))
