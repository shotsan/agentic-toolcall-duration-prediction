"""
DELAY PREDICTOR (no fitting). Convert cachegrind cycles -> seconds with FIXED
physical constants:
    predicted_seconds = cycles / CLOCK_HZ + STARTUP_FLOOR
  CLOCK_HZ      = 5.0e9   (i7-13700F ~5 GHz)
  STARTUP_FLOOR = 0.10 s  (process/interpreter startup floor)
Compare predicted seconds to measured wall-clock. Also Continuum's predictor
(per-first-word mean seconds). Metric: median/mean APE on the predicted TIME.
"""
import json, numpy as np
from collections import defaultdict

CLOCK_HZ = 5.0e9
STARTUP_FLOOR = 0.10

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

keys = [k for k in gt if k in cg]
y = np.array([gt[k]["duration_s"] for k in keys])
cyc = np.array([cg[k]["est_cycles"] for k in keys])

# OUR delay predictor: cycles -> seconds, fixed constants
pred = cyc / CLOCK_HZ + STARTUP_FLOOR

# Continuum: per-first-word mean seconds, LEAVE-ONE-OUT (exclude the command
# being predicted, so Continuum does not peek at its own answer). Matches their
# online running-average behavior more faithfully.
fw_sum = defaultdict(float)
fw_cnt = defaultdict(int)
for k in gt:
    w = gt[k]["command"].split()[0] if gt[k]["command"].split() else ""
    fw_sum[w] += gt[k]["duration_s"]
    fw_cnt[w] += 1
global_mean = np.mean([gt[k]["duration_s"] for k in gt])
cont = []
for k in keys:
    w = gt[k]["command"].split()[0] if gt[k]["command"].split() else ""
    # leave-one-out mean for this first word
    if fw_cnt[w] > 1:
        loo = (fw_sum[w] - gt[k]["duration_s"]) / (fw_cnt[w] - 1)
    else:
        loo = global_mean  # unseen first word -> global average (cold start)
    cont.append(loo)
cont = np.array(cont)


def ape(p, t):
    return np.abs(p - t) / np.maximum(t, 1e-9) * 100


print(f"DELAY PREDICTION (predicted seconds vs measured), n={len(keys)}")
print(f"  constants: clock={CLOCK_HZ:.0e}Hz  floor={STARTUP_FLOOR}s  (NO fitting)\n")
for name, p in [("Ours (cycles->sec)", pred), ("Continuum (first-word)", cont)]:
    a = ape(p, y)
    print(f"  {name:24s} medAPE={np.median(a):6.1f}%  meanAPE={a.mean():7.1f}%  "
          f"within2x={np.mean((p/y<2)&(p/y>0.5))*100:4.1f}%")

# show some concrete predictions
print("\n  examples (real -> predicted seconds):")
order = np.argsort(y)
for i in list(order[:3]) + list(order[-3:]):
    print(f"    real={y[i]:7.3f}s  ours={pred[i]:7.3f}s  continuum={cont[i]:7.3f}s  "
          f"{gt[keys[i]]['command'][:40]!r}")
