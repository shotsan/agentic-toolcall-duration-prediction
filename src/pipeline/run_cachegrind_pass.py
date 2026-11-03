"""
Predictors 4 & 5 pass: run cachegrind (+branch-sim) on the Python-code commands.
Expensive (valgrind per command) -> only for embedded/runs_file kinds, resumable.
Appends to cachegrind_scores.jsonl: {instance_id, step, est_cycles, mispredicts}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from run_apriori import classify, get_code
from apriori_sim import run_cachegrind

SAVE = Path("saved_code")


def done_keys(path="cachegrind_scores.jsonl"):
    s = set()
    try:
        for l in open(path):
            r = json.loads(l); s.add((r["instance_id"], r["step"]))
    except FileNotFoundError:
        pass
    return s


def main():
    rows = [json.loads(l) for l in open("swe_verified_durations.jsonl")]
    done = done_keys()
    out = open("cachegrind_scores.jsonl", "a")
    n = 0
    for r in rows:
        if (r["instance_id"], r["step"]) in done:
            continue
        kind = classify(r["command"])
        if kind not in ("embedded", "runs_file"):
            continue
        fdir = SAVE / r["instance_id"] / "files.json"
        files = json.load(open(fdir)) if fdir.exists() else {}
        code = get_code(r["instance_id"], r["command"], kind, files)
        if not code:
            continue
        res = run_cachegrind(code, timeout=120)
        out.write(json.dumps({
            "instance_id": r["instance_id"], "step": r["step"],
            "kind": kind, "duration_s": r["duration_s"],
            "est_cycles": res.get("est_cycles") if res.get("ok") else None,
            "mispredicts": res.get("Mispredicts") if res.get("ok") else None,
        }) + "\n")
        out.flush()
        n += 1
        if n % 20 == 0:
            print(f"  {n} done", file=sys.stderr)
    out.close()
    print(f"cachegrind pass: {n} commands", file=sys.stderr)


if __name__ == "__main__":
    main()
