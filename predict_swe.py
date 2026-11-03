"""
THE PREDICTION STUDY on real GPT-5 / mini-SWE-agent / SWE-bench Verified data.

Compares, at predicting per-command duration:
  BASELINE (Continuum, their actual code logic): per-FIRST-WORD mean.
           (matches ToolCallEstimator.update_func_call_exec_time: running avg
            per first word.)
  OURS:    condition on the FULL command via lightweight command features.

Features for OURS (command-string only, no execution):
  - first-word id (so we never do worse than the baseline's signal)
  - command length (chars), token/arg count, number of pipes, redirects
  - flags: has_pytest, has_python_heredoc, has_sed_range, has_grep_recursive,
    has_find, writes_file, has_subshell
  - target: longest numeric literal (e.g. sed line ranges), path depth

Protocol: GroupKFold by instance_id (train on some tasks, test on HELD-OUT
tasks -> generalization to unseen instances, not interpolation).
Metrics: MAPE, median APE, tail MAPE (slow 20%), Spearman rho.
"""
from __future__ import annotations
import json
import re
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr


def feats(cmd: str) -> list[float]:
    s = cmd.strip()
    nums = [int(x) for x in re.findall(r"\d+", s)]
    return [
        float(len(s)),                              # char length
        float(len(s.split())),                      # token count
        float(s.count("|")),                        # pipes
        float(s.count(">") + s.count(">>")),        # redirects
        float("pytest" in s),
        float("<<" in s or "<<'" in s),             # heredoc (python - <<PY)
        float(bool(re.search(r"sed -n '\d+,\d+p'", s))),  # sed line range
        float("-r" in s or "-R" in s or "grep -rn" in s or "grep -RIn" in s),
        float("find " in s),
        float(">" in s or "tee " in s or "mkdir" in s),   # writes
        float("&&" in s or ";" in s or "<<" in s),        # multi/subshell
        float(max(nums) if nums else 0),            # largest numeric literal
        float(s.count("/")),                         # path depth proxy
    ]


def load():
    cmds, fw, X, y, groups = [], [], [], [], []
    fw_ids = {}
    for line in open("swe_verified_durations.jsonl"):
        r = json.loads(line)
        if r.get("duration_s") is None:
            continue
        w = r["first_word"]
        fw_ids.setdefault(w, len(fw_ids))
        cmds.append(r["command"]); fw.append(w)
        X.append([float(fw_ids[w])] + feats(r["command"]))
        y.append(r["duration_s"] * 1000.0)
        groups.append(r["instance_id"])
    return np.array(fw), np.array(X, float), np.array(y, float), np.array(groups)


def ape(p, t):
    return np.abs(p - t) / np.maximum(t, 1e-6) * 100.0


def baseline_firstword_mean(fw, y, groups):
    """Continuum: predict per-first-word MEAN, trained on OTHER instances
    (grouped LOO) so it's a fair held-out comparison."""
    preds = np.empty_like(y)
    gkf = GroupKFold(n_splits=min(6, len(set(groups))))
    for tr, te in gkf.split(y, y, groups):
        means = {}
        for w in set(fw[tr]):
            means[w] = y[tr][fw[tr] == w].mean()
        glob = y[tr].mean()
        for i in te:
            preds[i] = means.get(fw[i], glob)
    return preds


def ours(X, y, groups):
    preds = np.empty_like(y); ylog = np.log1p(y)
    gkf = GroupKFold(n_splits=min(6, len(set(groups))))
    for tr, te in gkf.split(X, y, groups):
        m = RandomForestRegressor(n_estimators=400, random_state=0)
        m.fit(X[tr], ylog[tr])
        preds[te] = np.expm1(m.predict(X[te]))
    return np.clip(preds, 0, None)


def report(name, p, y, tail):
    a = ape(p, y)
    print(f"{name:30s} MAPE={a.mean():7.1f}%  medAPE={np.median(a):6.1f}%  "
          f"tailMAPE={a[tail].mean():7.1f}%  rho={spearmanr(p,y).correlation:+.3f}")


def main():
    fw, X, y, groups = load()
    thr = np.quantile(y, 0.80); tail = y >= thr
    print(f"REAL SWE-bench Verified / GPT-5 / mini-SWE-agent")
    print(f"n={len(y)} commands, {len(set(groups))} instances; "
          f"{y.min():.0f}-{y.max():.0f} ms, median {np.median(y):.0f}; "
          f"tail(>=p80)={thr:.0f}ms ({tail.sum()})")
    print(f"distinct first-words: {len(set(fw))}\n")
    print("=== BASELINE (Continuum: per-first-word mean, held-out instances) ===")
    report("first-word mean", baseline_firstword_mean(fw, y, groups), y, tail)
    print("\n=== OURS (full-command features, held-out instances) ===")
    report("RandomForest(full cmd)", ours(X, y, groups), y, tail)


if __name__ == "__main__":
    main()
