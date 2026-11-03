"""
A-priori SIMULATION predictors (predictors 4,5 from APRIORI_PLAN).
Run the code under valgrind's SIMULATED CPU (cachegrind) with branch simulation.
These are deterministic instruction/cache/branch counts -> a-priori cost. They
do NOT read the real-hardware wall-clock (that's ground truth, separate).

4. cachegrind_cost : total EXECUTED instructions (Ir) + cache-miss penalty
5. branch_penalty  : branch mispredictions (the TAGE/perceptron role)

A combined estimated-cycles number is also returned:
    cycles ~= Ir + 10*L1_misses + 100*LL_misses + 15*mispredicts
(standard first-order penalties; constants fixed, not learned).

NOTE: valgrind runs the code, but under a simulator counting operations — this
is cost ANALYSIS, not wall-clock measurement. For Python snippets we run the
CPython interpreter under cachegrind executing the snippet; the simulated counts
reflect the interpreter work the snippet causes (bytecode dispatch etc.).
"""
from __future__ import annotations
import re
import subprocess
import tempfile
import os

PYTHON = "/usr/bin/python3"

_SUMMARY = {
    "Ir": re.compile(r"I\s+refs:\s+([\d,]+)"),
    "I1_miss": re.compile(r"I1\s+misses:\s+([\d,]+)"),
    "D_refs": re.compile(r"D\s+refs:\s+([\d,]+)"),
    "D1_miss": re.compile(r"D1\s+misses:\s+([\d,]+)"),
    "LL_miss": re.compile(r"LL\s+misses:\s+([\d,]+)"),
    "Branches": re.compile(r"Branches:\s+([\d,]+)"),
    "Mispredicts": re.compile(r"Mispredicts:\s+([\d,]+)"),
}


def _num(m):
    return float(m.group(1).replace(",", "")) if m else 0.0


def run_cachegrind(src: str, timeout=300) -> dict:
    """Run python snippet under cachegrind --branch-sim=yes; parse the summary.
    Returns dict of counts + an estimated-cycles cost. nan on failure."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        p = subprocess.run(
            ["valgrind", "--tool=cachegrind", "--branch-sim=yes",
             "--cachegrind-out-file=/dev/null", PYTHON, path],
            capture_output=True, text=True, timeout=timeout)
        err = p.stderr  # cachegrind summary goes to stderr
        out = {k: _num(rx.search(err)) for k, rx in _SUMMARY.items()}
        if out["Ir"] == 0:
            return {"ok": False}
        cycles = (out["Ir"] + 10 * out["I1_miss"] + 100 * out["LL_miss"]
                  + 15 * out["Mispredicts"])
        out["est_cycles"] = cycles
        out["ok"] = True
        return out
    except subprocess.TimeoutExpired:
        return {"ok": False, "timeout": True}
    except Exception:
        return {"ok": False}
    finally:
        os.unlink(path)


def cachegrind_cost(src: str) -> float:
    r = run_cachegrind(src)
    return r["est_cycles"] if r.get("ok") else float("nan")


def branch_penalty(src: str) -> float:
    r = run_cachegrind(src)
    return r["Mispredicts"] if r.get("ok") else float("nan")


if __name__ == "__main__":
    tests = [
        'print("hi")',
        's=0\nfor i in range(1000000):\n    s+=i\nprint(s)',
    ]
    for t in tests:
        r = run_cachegrind(t)
        print(repr(t[:40]))
        if r.get("ok"):
            print(f"   Ir={r['Ir']:.0f}  LLmiss={r['LL_miss']:.0f}  "
                  f"mispred={r['Mispredicts']:.0f}  est_cycles={r['est_cycles']:.0f}")
        else:
            print("   FAILED", r)
