"""
Track T2 (Shell-Commands): benchmark-faithful to Continuum's SWE-agent setup.

Continuum keys tool name = FIRST WORD of the bash command. We emit real bash
commands an SWE-agent would run against a real repo tree, measure wall-clock,
and compare:
  BASELINE (Continuum): per-first-word CDF median -> one prediction per verb
                        (cd, cat, grep, ...), blind to the argument.
  OURS: features of the FULL command (first word one-hot + target file/dir size
        + recursive flag + output size proxy).

Repo under test: a real large codebase (vLLM source tree) so file sizes and
directory depths are realistic and varied -- the source of duration variance.
"""
from __future__ import annotations
import json
import os
import random
import subprocess
import statistics
import time

random.seed(11)

REPO = "/home/santosh/cpu-gpu-sync/vllm"

# Collect a realistic pool of files (varied sizes) and directories.
def collect_targets():
    files, dirs = [], []
    for root, dirnames, filenames in os.walk(REPO):
        if "/.git" in root:
            continue
        dirs.append(root)
        for fn in filenames:
            p = os.path.join(root, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if sz > 0:
                files.append((p, sz))
    return files, dirs


FILES, DIRS = collect_targets()
# Spread file choices across the size distribution.
FILES.sort(key=lambda x: x[1])


def rand_file():
    return random.choice(FILES)


def rand_dir():
    return random.choice(DIRS)


# Command templates: (first_word, build) -> (argv list, feature dict)
def make_command():
    kind = random.choice([
        "cat", "wc", "grep", "find", "ls", "sort", "head", "python_c", "wc_l",
    ])
    if kind == "cat":
        p, sz = rand_file()
        return ["cat", p], "cat", {"target_bytes": sz, "recursive": 0, "n_files": 1}
    if kind == "wc":
        p, sz = rand_file()
        return ["wc", p], "wc", {"target_bytes": sz, "recursive": 0, "n_files": 1}
    if kind == "wc_l":
        p, sz = rand_file()
        return ["wc", "-l", p], "wc", {"target_bytes": sz, "recursive": 0, "n_files": 1}
    if kind == "grep":
        p, sz = rand_file()
        return ["grep", "-c", "def", p], "grep", {"target_bytes": sz, "recursive": 0, "n_files": 1}
    if kind == "find":
        d = rand_dir()
        # count files under a dir -> cost scales with subtree size
        n_sub = sum(len(fs) for _, _, fs in os.walk(d))
        return ["find", d, "-name", "*.py"], "find", {"target_bytes": 0, "recursive": 1, "n_files": n_sub}
    if kind == "ls":
        d = rand_dir()
        try:
            n = len(os.listdir(d))
        except OSError:
            n = 0
        return ["ls", "-la", d], "ls", {"target_bytes": 0, "recursive": 0, "n_files": n}
    if kind == "sort":
        p, sz = rand_file()
        return ["sort", p], "sort", {"target_bytes": sz, "recursive": 0, "n_files": 1}
    if kind == "head":
        p, sz = rand_file()
        return ["head", "-n", "100", p], "head", {"target_bytes": sz, "recursive": 0, "n_files": 1}
    if kind == "python_c":
        n = random.choice([1000, 100000, 1000000, 5000000])
        return (["python3", "-c", f"print(sum(range({n})))"], "python3",
                {"target_bytes": n, "recursive": 0, "n_files": 0})


FIRST_WORDS = ["cat", "wc", "grep", "find", "ls", "sort", "head", "python3"]
FW_ID = {w: i for i, w in enumerate(FIRST_WORDS)}


def time_cmd(argv, runs=3, timeout=60.0):
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            subprocess.run(argv, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return timeout * 1000.0
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000.0


def build(n=400):
    rows = []
    for _ in range(n):
        argv, fw, feat = make_command()
        dur = time_cmd(argv)
        rows.append({
            "first_word": fw,            # Continuum's tool key
            "argv": argv,
            "dur_ms": dur,
            # OURS: full-command features
            "feat": [float(FW_ID[fw]),
                     float(feat["target_bytes"]),
                     float(feat["recursive"]),
                     float(feat["n_files"])],
        })
    return rows


if __name__ == "__main__":
    rows = build()
    with open("shell_labeled.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    d = [r["dur_ms"] for r in rows]
    print(f"wrote {len(rows)} shell tool calls; "
          f"{min(d):.1f}-{max(d):.1f} ms, median {statistics.median(d):.1f}, "
          f"mean {statistics.mean(d):.1f}")
    print(f"pool: {len(FILES)} files, {len(DIRS)} dirs")
