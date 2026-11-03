"""
Add a WARM-START view, communicating what happens when interpreter/process
startup is already paid (persistent/warm workers), vs the COLD-start we measured.

Measured fixed startup costs (a-priori, from cachegrind):
  bash process startup   ~  1.24e6 instructions
  CPython cold-start     ~ 80.7e6 instructions  (python `-c pass`)
Measured wall-clock floor ~ 0.09 s (p10 of durations).

COLD (what we measured): actual_s, M4_cold = cycles / FREQ
WARM (startup removed):
  M4_warm = max(cycles - startup_instr, 0) / FREQ
  actual_warm = max(actual_s - WALL_FLOOR, 0)
So warm shows the predicted/real time of just the COMMAND'S WORK, excluding the
fixed interpreter/process boot that a warm worker would not repay.
"""
import json, numpy as np, csv
from collections import defaultdict

FREQ = 5.1e9
PY_START = 80.7e6 + 1.24e6     # python launched via bash
SH_START = 1.24e6              # shell command
WALL_FLOOR = 0.09              # measured wall-clock startup floor (s)

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

fs = defaultdict(float); fc = defaultdict(int)
for k in gt:
    w = gt[k]["command"].split()[0] if gt[k]["command"].split() else ""
    fs[w] += gt[k]["duration_s"]; fc[w] += 1
gm = np.mean([gt[k]["duration_s"] for k in gt])


def cont(k):
    w = gt[k]["command"].split()[0] if gt[k]["command"].split() else ""
    return (fs[w] - gt[k]["duration_s"]) / (fc[w] - 1) if fc[w] > 1 else gm


def is_py(cmd):
    w = cmd.split()[0] if cmd.split() else ""
    return (w in ("python", "python3") or "python" in cmd[:40]
            or w.startswith("PYTHON"))


keys = [k for k in gt if k in cg]
rows = sorted(keys, key=lambda k: gt[k]["duration_s"])
idx = list(range(4)) + list(range(len(rows)//2, len(rows)//2+3)) + list(range(len(rows)-6, len(rows)))

print("COLD = as measured.  WARM = startup (interpreter/process boot) removed.\n")
hdr = (f"{'ACT_cold':>9} {'ACT_warm':>9} | {'CONTINU':>8} | "
       f"{'M4_cold':>9} {'M4_warm':>9}  cmd")
print(hdr); print("-"*len(hdr))
for i in idx:
    k = rows[i]
    a = gt[k]["duration_s"]
    aw = max(a - WALL_FLOOR, 0.0)
    cyc = cg[k]["est_cycles"]
    start = PY_START if is_py(gt[k]["command"]) else SH_START
    m4c = cyc / FREQ
    m4w = max(cyc - start, 0.0) / FREQ
    print(f"{a:9.3f} {aw:9.3f} | {cont(k):8.3f} | {m4c:9.3f} {m4w:9.3f}  "
          f"{gt[k]['command'][:28]!r}")

with open("warm_cold_table.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["instance_id","step","actual_cold_s","actual_warm_s",
                "continuum_s","m4_cold_s","m4_warm_s","command"])
    for k in rows:
        a = gt[k]["duration_s"]; cyc = cg[k]["est_cycles"]
        start = PY_START if is_py(gt[k]["command"]) else SH_START
        w.writerow([k[0],k[1],f"{a:.4f}",f"{max(a-WALL_FLOOR,0):.4f}",
                    f"{cont(k):.4f}",f"{cyc/FREQ:.4f}",
                    f"{max(cyc-start,0)/FREQ:.4f}",
                    gt[k]["command"].replace(chr(10)," ")[:120]])
print(f"\nfull warm/cold table -> warm_cold_table.csv ({len(rows)} rows)")
