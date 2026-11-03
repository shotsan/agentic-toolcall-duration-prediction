"""
Track T2b (Heavy Shell-Commands): test the boundary claim from T2.

Hypothesis: code/arg conditioning beats the per-first-word CDF when duration is
COMPUTE-dominated (not spawn-overhead-dominated). Here commands do real work
(sorting large generated data, grepping large concatenations, python compute
over large N) so duration spans ms -> seconds and depends on the argument.

Same baseline keying as Continuum (first word).
"""
from __future__ import annotations
import json
import random
import subprocess
import statistics
import time

random.seed(23)
FIRST_WORDS = ["sort", "grep", "wc", "python3", "awk"]
FW_ID = {w: i for i, w in enumerate(FIRST_WORDS)}


def make_command():
    kind = random.choice(["sort", "grep", "wc", "python_sum", "python_sort", "awk"])
    # N controls work; it's an ARGUMENT, not fixed in the verb.
    N = random.choice([100_000, 500_000, 1_000_000, 3_000_000, 8_000_000])
    if kind == "sort":
        cmd = f"seq {N} | shuf | sort -n > /dev/null"
        return cmd, "sort", N
    if kind == "grep":
        cmd = f"seq {N} | grep -c 7 > /dev/null"
        return cmd, "grep", N
    if kind == "wc":
        cmd = f"seq {N} | wc -l > /dev/null"
        return cmd, "wc", N
    if kind == "python_sum":
        cmd = f'python3 -c "print(sum(range({N})))"'
        return cmd, "python3", N
    if kind == "python_sort":
        cmd = (f'python3 -c "import random;a=[random.random() for _ in '
               f'range({N})];a.sort();print(a[0])"')
        return cmd, "python3", N
    if kind == "awk":
        cmd = f"seq {N} | awk '{{s+=$1}} END{{print s}}' > /dev/null"
        return cmd, "awk", N


def time_cmd(cmd, runs=3, timeout=120.0):
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return timeout * 1000.0
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000.0


def build(n=250):
    rows = []
    for _ in range(n):
        cmd, fw, N = make_command()
        dur = time_cmd(cmd)
        rows.append({"first_word": fw, "cmd": cmd, "dur_ms": dur,
                     "feat": [float(FW_ID[fw]), float(N)]})
    return rows


if __name__ == "__main__":
    rows = build()
    with open("shell_heavy_labeled.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    d = [r["dur_ms"] for r in rows]
    print(f"wrote {len(rows)} heavy shell calls; {min(d):.1f}-{max(d):.1f} ms, "
          f"median {statistics.median(d):.1f}, mean {statistics.mean(d):.1f}")
