"""
The experiment: does conditioning on code beat Continuum's flat per-tool CDF?

Baselines (code-blind, what Continuum/vanilla engines do):
  - CDF-median : predict the median of all observed durations for the tool.
  - CDF-mean   : predict the mean.

Code-conditioned (our thesis):
  - RandomForest / Huber / MLP on AST features (PrETi-style feature model).

Eval: leave-one-out CV (dataset is small). Metric: MAPE (mean abs % error) and
median APE -- the same APE metric PrETi reports. Lower is better.

Note: all snippets share tool="execute_code", so the CDF baseline is forced to
give ONE number for everything -- exactly Continuum's limitation for coding
agents.
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut


def load():
    X, y, names = [], [], None
    with open("labeled.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if not r["ok"] or r["median_s"] is None:
                continue
            X.append(r["features"])
            y.append(r["median_s"] * 1000.0)  # ms
            names = r["feature_names"]
    return np.array(X, float), np.array(y, float), names


def ape(pred, true):
    return np.abs(pred - true) / np.maximum(true, 1e-9) * 100.0


def eval_cdf(y, stat):
    """Leave-one-out: predict the stat (median/mean) of the OTHER points.
    This is the code-blind flat-CDF baseline."""
    preds = np.empty_like(y)
    for i in range(len(y)):
        rest = np.delete(y, i)
        preds[i] = np.median(rest) if stat == "median" else np.mean(rest)
    return preds


def eval_model(make_model, X, y):
    """Leave-one-out CV for a code-conditioned regressor.
    We predict log(duration) because durations span orders of magnitude."""
    loo = LeaveOneOut()
    preds = np.empty_like(y)
    ylog = np.log1p(y)
    for tr, te in loo.split(X):
        m = make_model()
        m.fit(X[tr], ylog[tr])
        preds[te] = np.expm1(m.predict(X[te]))
    return np.clip(preds, 0, None)


def report(name, preds, y, tail_mask=None):
    a = ape(preds, y)
    # Tail-weighted: APE on the slowest-20% calls (where scheduling value is).
    tail = a[tail_mask].mean() if tail_mask is not None else float("nan")
    # Also: how well do we RANK (Spearman) -- useful for SJF-style scheduling.
    print(f"{name:26s}  MAPE={a.mean():7.1f}%  medAPE={np.median(a):6.1f}%  "
          f"tailMAPE={tail:8.1f}%")
    return a.mean()


def spearman(pred, y):
    from scipy.stats import spearmanr
    return spearmanr(pred, y).correlation


def main():
    X, y, names = load()
    # tail = slowest 20% of calls (the ones that dominate delay, per Continuum Fig5)
    thresh = np.quantile(y, 0.80)
    tail_mask = y >= thresh
    print(f"loaded {len(y)} samples; duration range "
          f"{y.min():.0f}-{y.max():.0f} ms; mean={y.mean():.0f} "
          f"std={y.std():.0f}; tail(>=p80)={thresh:.0f}ms ({tail_mask.sum()} calls)\n")

    preds_store = {}

    print("=== CODE-BLIND BASELINES (Continuum-style flat per-tool CDF) ===")
    p = eval_cdf(y, "median"); preds_store["CDF-median"] = p
    report("CDF-median (Continuum)", p, y, tail_mask)
    p = eval_cdf(y, "mean"); preds_store["CDF-mean"] = p
    report("CDF-mean", p, y, tail_mask)

    print("\n=== CODE-CONDITIONED (our thesis: AST + effective-op-count) ===")
    p = eval_model(lambda: RandomForestRegressor(
        n_estimators=400, random_state=0), X, y)
    preds_store["RF"] = p
    report("RandomForest(AST)", p, y, tail_mask)
    p = eval_model(lambda: make_pipeline(
        StandardScaler(), HuberRegressor(max_iter=3000)), X, y)
    report("Huber(AST,scaled)", p, y, tail_mask)
    p = eval_model(lambda: make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=8000,
                     random_state=0)), X, y)
    report("MLP(AST,scaled)", p, y, tail_mask)

    print("\n=== RANKING QUALITY (Spearman corr with true duration) ===")
    print(f"  CDF-median : {spearman(preds_store['CDF-median'], y):.3f} "
          f"(flat -> ~0, can't rank)")
    print(f"  RandomForest: {spearman(preds_store['RF'], y):.3f} "
          f"(higher = better SJF ordering)")

    # Feature importance from a full-data RF (which signals matter)
    rf = RandomForestRegressor(n_estimators=300, random_state=0)
    rf.fit(X, np.log1p(y))
    order = np.argsort(rf.feature_importances_)[::-1]
    print("\n=== top AST features by importance ===")
    for i in order[:5]:
        print(f"  {names[i]:22s} {rf.feature_importances_[i]:.3f}")


if __name__ == "__main__":
    main()
