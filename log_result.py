"""
Compute the prediction-study metrics on the CURRENT swe_verified_durations.jsonl
and APPEND a timestamped checkpoint row to results_log.csv + results_log.md.
Run this after each measurement batch to keep a documented history.
"""
from __future__ import annotations
import json, time, datetime, os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from predict_swe import load, ape, baseline_firstword_mean, ours

CSV = "results_log.csv"
MD = "results_log.md"


def main():
    fw, X, y, groups = load()
    n_inst = len(set(groups))
    thr = np.quantile(y, 0.80); tail = y >= thr
    repos = sorted(set(g.split("__")[0] for g in groups))

    base = baseline_firstword_mean(fw, y, groups)
    our = ours(X, y, groups)

    def m(p):
        a = ape(p, y)
        return (a.mean(), float(np.median(a)), a[tail].mean(),
                spearmanr(p, y).correlation)

    b_mape, b_med, b_tail, b_rho = m(base)
    o_mape, o_med, o_tail, o_rho = m(our)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    header = ("timestamp,n_instances,n_commands,n_repos,"
              "base_mape,base_medape,base_tailmape,base_rho,"
              "ours_mape,ours_medape,ours_tailmape,ours_rho\n")
    row = (f"{ts},{n_inst},{len(y)},{len(repos)},"
           f"{b_mape:.1f},{b_med:.1f},{b_tail:.1f},{b_rho:.3f},"
           f"{o_mape:.1f},{o_med:.1f},{o_tail:.1f},{o_rho:.3f}\n")
    if not os.path.exists(CSV):
        open(CSV, "w").write(header)
    open(CSV, "a").write(row)

    # human-readable md checkpoint
    with open(MD, "a") as f:
        f.write(f"\n### checkpoint {ts} — {n_inst} instances, {len(y)} commands, "
                f"{len(repos)} repos ({', '.join(repos)})\n")
        f.write(f"- duration: {y.min():.0f}-{y.max():.0f} ms, median {np.median(y):.0f}; "
                f"tail(>=p80)={thr:.0f}ms ({int(tail.sum())} cmds)\n")
        f.write(f"- Continuum (first-word mean): MAPE {b_mape:.1f}%  medAPE {b_med:.1f}%  "
                f"tailMAPE {b_tail:.1f}%  rho {b_rho:+.3f}\n")
        f.write(f"- Ours (full command):        MAPE {o_mape:.1f}%  medAPE {o_med:.1f}%  "
                f"tailMAPE {o_tail:.1f}%  rho {o_rho:+.3f}\n")

    print(f"logged checkpoint: {n_inst} instances, {len(y)} commands")
    print(f"  Continuum MAPE {b_mape:.1f}%  rho {b_rho:+.3f}")
    print(f"  Ours      MAPE {o_mape:.1f}%  rho {o_rho:+.3f}")


if __name__ == "__main__":
    main()
