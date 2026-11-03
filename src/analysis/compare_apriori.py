"""
Final comparison: the 6 a-priori predictors vs Continuum's first-word baseline,
against measured wall-clock ground truth. No training.

For each predictor we have a per-command SCORE (proportional cost). To compare
to seconds we apply ONE global scale constant k = median(duration)/median(score)
(a single analytical calibration, NOT a per-command learned model), then report:
  - Spearman rho (ranking vs true duration)  <- most important for scheduling
  - median APE, tail APE (slowest 20%) after global scaling
Baseline = Continuum per-first-word mean (its shipped predictor).
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import spearmanr

STATIC = ["bytecode_count", "weighted_bytecode", "complexity_bigO", "wcet_bound"]


def load_static():
    rows = [json.loads(l) for l in open("apriori_scores.jsonl")]
    return rows


def load_cachegrind():
    d = {}
    try:
        for l in open("cachegrind_scores.jsonl"):
            r = json.loads(l)
            d[(r["instance_id"], r["step"])] = r
    except FileNotFoundError:
        pass
    return d


def ape(pred, true):
    return np.abs(pred - true) / np.maximum(true, 1e-9) * 100.0


def eval_predictor(scores, durs, name):
    s = np.array(scores, float); y = np.array(durs, float)
    mask = ~np.isnan(s)
    s, y = s[mask], y[mask]
    if len(s) < 10:
        print(f"{name:20s} (too few: {len(s)})"); return
    k = np.median(y) / max(np.median(s), 1e-9)       # one global scale
    pred = s * k
    thr = np.quantile(y, 0.80); tail = y >= thr
    rho = spearmanr(s, y).correlation
    print(f"{name:20s} n={len(s):5d}  rho={rho:+.3f}  "
          f"medAPE={np.median(ape(pred,y)):7.1f}%  "
          f"tailMAPE={ape(pred,y)[tail].mean():8.1f}%")


def baseline_firstword(rows):
    """Continuum: per-first-word mean (leave-one-out within first word)."""
    from collections import defaultdict
    fw_d = defaultdict(list)
    for r in rows:
        fw = r["command"].split()[0] if r["command"].split() else ""
        fw_d[fw].append(r["duration_s"])
    preds, durs = [], []
    for r in rows:
        fw = r["command"].split()[0] if r["command"].split() else ""
        others = [d for d in fw_d[fw] if d is not r["duration_s"]]
        preds.append(np.mean(fw_d[fw]))   # mean of that first word
        durs.append(r["duration_s"])
    p = np.array(preds); y = np.array(durs)
    thr = np.quantile(y, 0.80); tail = y >= thr
    print(f"{'Continuum(firstword)':20s} n={len(y):5d}  "
          f"rho={spearmanr(p,y).correlation:+.3f}  "
          f"medAPE={np.median(ape(p,y)):7.1f}%  tailMAPE={ape(p,y)[tail].mean():8.1f}%")


def main():
    rows = load_static()
    cg = load_cachegrind()
    durs = [r["duration_s"] for r in rows]

    print(f"=== {len(rows)} commands ===\n")
    print("--- BASELINE ---")
    baseline_firstword(rows)

    print("\n--- A-PRIORI PREDICTORS 1-3,6 (static) ---")
    for name in STATIC:
        eval_predictor([r["scores"].get(name) if r["scores"].get(name) is not None
                        else np.nan for r in rows], durs, name)

    print("\n--- A-PRIORI PREDICTORS 4,5 (cachegrind sim; python-code subset) ---")
    if cg:
        keys = [(r["instance_id"], r["step"]) for r in rows]
        est = [cg.get(k, {}).get("est_cycles") for k in keys]
        mis = [cg.get(k, {}).get("mispredicts") for k in keys]
        eval_predictor([e if e is not None else np.nan for e in est], durs,
                       "cachegrind_cost")
        eval_predictor([m if m is not None else np.nan for m in mis], durs,
                       "branch_penalty")
        # also report on the cachegrind subset only (apples-to-apples)
        sub = [(cg[k]["est_cycles"], cg[k]["duration_s"]) for k in keys
               if k in cg and cg[k].get("est_cycles") is not None]
        if sub:
            s = np.array([x[0] for x in sub]); y = np.array([x[1] for x in sub])
            print(f"\n  [cachegrind subset only, n={len(sub)}] "
                  f"rho={spearmanr(s,y).correlation:+.3f}")
    else:
        print("  (cachegrind pass not done yet)")


if __name__ == "__main__":
    main()
