"""
FINAL clean comparison: 6 a-priori predictors + Continuum baseline vs measured
wall-clock, on the FULL fixed dataset (499 instances).
- cachegrind (predictors 4,5) = environment-matched, full-command, children-traced.
- static (predictors 1,2,3,6) = from apriori_scores.jsonl.
- Continuum baseline = per-first-word mean.
Drops timeout-clamped (>=119s) rows. One global scale per predictor (no ML).
Metrics: Spearman rho (ranking), median APE, tail MAPE (slowest 20%).
"""
import json, numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

# ground truth (drop timeouts)
gt = {}
for l in open("swe_verified_durations.jsonl"):
    r = json.loads(l)
    if r["duration_s"] < 119.0:
        gt[(r["instance_id"], r["step"])] = r

# cachegrind (env-matched)
cg = {}
for l in open("cachegrind_docker.jsonl"):
    r = json.loads(l)
    if r.get("ok") and r.get("est_cycles"):
        cg[(r["instance_id"], r["step"])] = r

# static scores
st = {}
for l in open("apriori_scores.jsonl"):
    r = json.loads(l)
    st[(r["instance_id"], r["step"])] = r["scores"]

keys = [k for k in gt if k in cg or k in st]
y = np.array([gt[k]["duration_s"] for k in keys])

# Continuum first-word mean (over all gt)
fw = defaultdict(list)
for k in gt:
    w = gt[k]["command"].split()[0] if gt[k]["command"].split() else ""
    fw[w].append(gt[k]["duration_s"])
fwmean = {w: np.mean(v) for w, v in fw.items()}


def ape(p, t):
    return np.abs(p - t) / np.maximum(t, 1e-9) * 100


def rep(name, scores):
    s = np.array(scores, float)
    m = ~np.isnan(s)
    ss, yy = s[m], y[m]
    if len(ss) < 10:
        print(f"{name:24s} n={len(ss)}"); return
    k = np.median(yy) / max(np.median(ss), 1e-9)
    p = ss * k
    tail = yy >= np.quantile(yy, 0.80)
    print(f"{name:24s} n={len(ss):5d}  rho={spearmanr(ss,yy).correlation:+.3f}  "
          f"medAPE={np.median(ape(p,yy)):6.1f}%  tailMAPE={ape(p,yy)[tail].mean():7.1f}%")


print(f"=== FINAL comparison, {len(keys)} commands (499 instances) ===\n")
print("BASELINE:")
rep("Continuum(first-word)",
    [fwmean[gt[k]["command"].split()[0] if gt[k]["command"].split() else ""] for k in keys])
print("\nSTATIC a-priori (1,2,3,6):")
for nm in ["bytecode_count", "weighted_bytecode", "complexity_bigO", "wcet_bound"]:
    rep(nm, [st.get(k, {}).get(nm) if st.get(k, {}).get(nm) is not None else np.nan
             for k in keys])
print("\nSIMULATION a-priori (4,5) — env-matched cachegrind:")
rep("cachegrind_cost(4)", [cg[k]["est_cycles"] if k in cg else np.nan for k in keys])
rep("branch_mispredict(5)", [cg[k]["mispredicts"] if k in cg and cg[k].get("mispredicts") is not None else np.nan for k in keys])
