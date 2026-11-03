"""
Show, per command: ACTUAL duration, CONTINUUM's prediction, OURS prediction.
No error metric. Just the three numbers side by side.

- ACTUAL    = measured wall-clock (swe_verified_durations.jsonl)
- CONTINUUM = per-first-word mean of actual durations (leave-one-out), i.e. their
              shipped predictor (running average per first word)
- OURS      = cachegrind cycles / real P-core frequency (5.1 GHz)
"""
import json, numpy as np
from collections import defaultdict

FREQ = 5.1e9  # measured P-core frequency

gt = {}
for l in open("swe_verified_durations.jsonl"):
    r = json.loads(l)
    if r["duration_s"] < 119.0:
        gt[(r["instance_id"], r["step"])] = r

cg = {}
for l in open("cachegrind_docker.jsonl"):
    r = json.loads(l)
    if r.get("ok") and r.get("est_cycles"):
        cg[(r["instance_id"], r["step"])] = r

# Continuum: leave-one-out per-first-word mean
fs = defaultdict(float); fc = defaultdict(int)
for k in gt:
    w = gt[k]["command"].split()[0] if gt[k]["command"].split() else ""
    fs[w] += gt[k]["duration_s"]; fc[w] += 1
gm = np.mean([gt[k]["duration_s"] for k in gt])


def continuum_pred(k):
    w = gt[k]["command"].split()[0] if gt[k]["command"].split() else ""
    return (fs[w] - gt[k]["duration_s"]) / (fc[w] - 1) if fc[w] > 1 else gm


def ours_pred(k):
    return cg[k]["est_cycles"] / FREQ if k in cg else None


keys = [k for k in gt if k in cg]
print(f"n = {len(keys)} commands (have both actual + our prediction)\n")
print(f"{'ACTUAL(s)':>10} {'CONTINUUM(s)':>13} {'OURS(s)':>10}   command")
print("-" * 80)
# sort by actual duration, show a spread: fastest, middle, slowest
rows = sorted(keys, key=lambda k: gt[k]["duration_s"])
idx = list(range(0, 5)) + list(range(len(rows)//2 - 2, len(rows)//2 + 3)) + \
      list(range(len(rows) - 8, len(rows)))
for i in idx:
    k = rows[i]
    a = gt[k]["duration_s"]; c = continuum_pred(k); o = ours_pred(k)
    cmd = gt[k]["command"].replace("\n", " ")[:42]
    print(f"{a:10.3f} {c:13.3f} {o:10.3f}   {cmd!r}")


# also dump the full table (all commands) to a CSV for inspection
import csv
with open("predictions_table.csv", "w", newline="") as f:
    wri = csv.writer(f)
    wri.writerow(["instance_id", "step", "actual_s", "continuum_s", "ours_s",
                  "first_word", "command"])
    for k in sorted(keys, key=lambda k: gt[k]["duration_s"]):
        wri.writerow([k[0], k[1], f"{gt[k]['duration_s']:.4f}",
                      f"{continuum_pred(k):.4f}", f"{ours_pred(k):.4f}",
                      gt[k]["command"].split()[0] if gt[k]["command"].split() else "",
                      gt[k]["command"].replace("\n", " ")[:120]])
print(f"\nfull table ({len(keys)} rows) -> predictions_table.csv")
