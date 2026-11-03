"""
Show per command, no invented metrics, no fabricated floor:
  ACTUAL(s)              : measured wall-clock
  CONTINUUM(s)           : first-word hardcode (leave-one-out mean) -- their method
  M4_cachegrind(s)       : cachegrind cycles / real frequency (5.1 GHz) -- the ONLY
                           method whose output is physically time. No floor added.
  M1 bytecode_count      : raw count (not seconds)
  M2 weighted_bytecode   : raw score
  M3 complexity_bigO     : raw score
  M5 branch_mispredict   : raw count
  M6 wcet_bound          : raw score
Methods 1,2,3,5,6 are COST COUNTS, not seconds. Shown in native units.
"""
import json, numpy as np, csv
from collections import defaultdict

FREQ = 5.1e9

gt = {}
for l in open("swe_verified_durations.jsonl"):
    r = json.loads(l)
    if r["duration_s"] < 119.0:
        gt[(r["instance_id"], r["step"])] = r
st = {}
for l in open("apriori_scores.jsonl"):
    r = json.loads(l); st[(r["instance_id"], r["step"])] = r["scores"]
cg = {}
for l in open("cachegrind_docker.jsonl"):
    r = json.loads(l)
    if r.get("ok"):
        cg[(r["instance_id"], r["step"])] = r

fs = defaultdict(float); fc = defaultdict(int)
for k in gt:
    w = gt[k]["command"].split()[0] if gt[k]["command"].split() else ""
    fs[w] += gt[k]["duration_s"]; fc[w] += 1
gm = np.mean([gt[k]["duration_s"] for k in gt])


def cont(k):
    w = gt[k]["command"].split()[0] if gt[k]["command"].split() else ""
    return (fs[w] - gt[k]["duration_s"]) / (fc[w] - 1) if fc[w] > 1 else gm


keys = [k for k in gt if k in cg]
rows = sorted(keys, key=lambda k: gt[k]["duration_s"])
idx = list(range(4)) + list(range(len(rows)//2, len(rows)//2 + 3)) + list(range(len(rows)-6, len(rows)))

print("M4 = cachegrind cycles / 5.1GHz (SECONDS). M1,2,3,5,6 = raw COUNTS (not seconds).\n")
hdr = f"{'ACTUAL':>8} {'CONTINU':>8} {'M4_sec':>9} | {'M1_bc':>8} {'M2_wbc':>8} {'M3_cplx':>9} {'M5_bmis':>10} {'M6_wcet':>9}  cmd"
print(hdr); print("-"*len(hdr))
for i in idx:
    k = rows[i]; s = st.get(k, {})
    a = gt[k]["duration_s"]; c = cont(k)
    m4 = cg[k]["est_cycles"]/FREQ if cg[k].get("est_cycles") else float('nan')
    def f(x): return "-" if x is None else f"{x:.0f}"
    print(f"{a:8.3f} {c:8.3f} {m4:9.3f} | "
          f"{f(s.get('bytecode_count')):>8} {f(s.get('weighted_bytecode')):>8} "
          f"{f(s.get('complexity_bigO')):>9} {cg[k].get('mispredicts',0):>10.0f} "
          f"{f(s.get('wcet_bound')):>9}  {gt[k]['command'][:30]!r}")

# full CSV
with open("all_methods_table.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["instance_id","step","actual_s","continuum_s","m4_cachegrind_s",
                "m1_bytecode","m2_weighted","m3_complexity","m5_branch_mispred",
                "m6_wcet","command"])
    for k in rows:
        s = st.get(k, {})
        m4 = cg[k]["est_cycles"]/FREQ if cg[k].get("est_cycles") else ""
        w.writerow([k[0],k[1],f"{gt[k]['duration_s']:.4f}",f"{cont(k):.4f}",
                    f"{m4:.4f}" if m4!="" else "",
                    s.get("bytecode_count"),s.get("weighted_bytecode"),
                    s.get("complexity_bigO"),cg[k].get("mispredicts"),
                    s.get("wcet_bound"),gt[k]["command"].replace(chr(10)," ")[:120]])
print(f"\nfull table -> all_methods_table.csv ({len(rows)} rows)")
