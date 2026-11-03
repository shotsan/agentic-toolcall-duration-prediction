"""
Generate REAL mini-SWE-agent trajectories on SWE-Bench, recording per-command
wall-clock duration measured INSIDE the task's Docker environment.

This is the faithful data source for the prediction task (PREDICTION_TASK.md):
- mini-SWE-agent's only tool is bash; each action is one command via docker exec.
- We time each command exactly where it runs (authentic label).
- We record the Continuum tool key (first word) + the full command.

Output: swebench_calls.jsonl, one row per executed bash command:
  {instance_id, step, command, first_word, duration_s, returncode, out_bytes}

Model backend: local vLLM (OpenAI-compatible) at $MSWEA_MODEL_API_BASE.
We use a local model because the MODEL only determines WHICH commands are
generated; the DURATION of a given command is set by the command + environment.
"""
from __future__ import annotations
import json
import os
import time
import sys

# isolate: run under mini-env which has minisweagent installed
from minisweagent.environments.docker import DockerEnvironment

TRACE_OUT = os.environ.get("TRACE_OUT", "swebench_calls.jsonl")
_fh = open(TRACE_OUT, "a")
_INSTANCE = {"id": "unknown", "step": 0}


class TimedDockerEnvironment(DockerEnvironment):
    """DockerEnvironment that times every command and logs it."""

    def execute(self, action: dict, cwd: str = "", *, timeout=None):
        command = action.get("command", "") if isinstance(action, dict) else str(action)
        t0 = time.perf_counter()
        result = super().execute(action, cwd, timeout=timeout)
        dt = time.perf_counter() - t0
        first_word = ""
        stripped = command.strip()
        if stripped:
            # mirror Continuum: first token of the command
            first_word = stripped.split()[0]
        rec = {
            "instance_id": _INSTANCE["id"],
            "step": _INSTANCE["step"],
            "command": command,
            "first_word": first_word,
            "duration_s": dt,
            "returncode": result.get("returncode"),
            "out_bytes": len(result.get("output", "") or ""),
        }
        _fh.write(json.dumps(rec) + "\n")
        _fh.flush()
        _INSTANCE["step"] += 1
        return result


if __name__ == "__main__":
    # Smoke test: start the astropy image, run a few representative commands,
    # confirm timing + logging work before wiring the full agent.
    _INSTANCE["id"] = "astropy__astropy-12907"
    img = "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    env = TimedDockerEnvironment(image=img, cwd="/testbed", timeout=120)
    try:
        for cmd in [
            "pwd",
            "ls -la",
            "find . -name '*.py' | wc -l",
            "cat setup.py | head -50",
            "python -c 'import astropy; print(astropy.__version__)'",
            "grep -rn 'def separability_matrix' astropy/modeling | head",
            "python -m pytest astropy/modeling/tests/test_separable.py -q 2>&1 | tail -5",
        ]:
            r = env.execute({"command": cmd})
            print(f"[{r['returncode']}] {cmd[:60]}")
    finally:
        env.cleanup()
    print(f"\nwrote timed commands to {TRACE_OUT}")
