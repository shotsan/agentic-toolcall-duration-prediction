"""
Real-data comparison on BFCL tool calls with varying runtime arguments.

Baselines (Continuum's information level):
  - per-tool CDF median: predict median of OTHER calls to the SAME tool f.
    (This is exactly Continuum's S[f] flat CDF.)
Our thesis (call-site conditioning):
  - RF / Huber / MLP on [tool_id, arg_len] features.

5-fold CV (random). Metrics: MAPE, median APE, tail MAPE (slowest 20%),
Spearman ranking.
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr


def load():
    tools, Xc, y = [], [], []
    with open("real_labeled.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            tools.append(r["tool"])
            Xc.append(r["feat_callsite"])
            y.append(r["dur_ms"])
    return np.array(tools), np.array(Xc, float), np.array(y, float)


def ape(p, t):
    return np.abs(p - t) / np.maximum(t, 1e-6) * 100.0


def per_tool_cdf(tools, y):
    """Continuum's flat per-tool CDF: predict median of OTHER same-tool calls.
    Leave-one-out within each tool."""
    preds = np.empty_like(y)
    for i in range(len(y)):
        same = (tools == tools[i])
        same[i] = False
        preds[i] = np.median(y[same]) if same.any() else np.median(y)
    return preds


def cv_model(make_model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    preds = np.empty_like(y)
    ylog = np.log1p(y)
    for tr, te in kf.split(X):
        m = make_model()
        m.fit(X[tr], ylog[tr])
        preds[te] = np.expm1(m.predict(X[te]))
    return np.clip(preds, 0, None)


def report(name, p, y, tail_mask):
    a = ape(p, y)
    print(f"{name:30s} MAPE={a.mean():7.1f}%  medAPE={np.median(a):6.1f}%  "
          f"tailMAPE={a[tail_mask].mean():7.1f}%  "
          f"rho={spearmanr(p, y).correlation:+.3f}")


def main():
    tools, Xc, y = load()
    thr = np.quantile(y, 0.80)
    tail = y >= thr
    print(f"{len(y)} real BFCL calls; {y.min():.2f}-{y.max():.1f} ms; "
          f"mean {y.mean():.1f}; tail(>=p80)={thr:.1f}ms ({tail.sum()} calls)\n")

    print("=== CONTINUUM INFO LEVEL (per-tool flat CDF) ===")
    report("per-tool CDF median", per_tool_cdf(tools, y), y, tail)

    print("\n=== CALL-SITE CONDITIONED (tool id + argument size) ===")
    report("RandomForest(callsite)",
           cv_model(lambda: RandomForestRegressor(n_estimators=400,
                                                  random_state=0), Xc, y), y, tail)
    report("Huber(callsite,scaled)",
           cv_model(lambda: make_pipeline(StandardScaler(),
                                          HuberRegressor(max_iter=3000)), Xc, y),
           y, tail)
    report("MLP(callsite,scaled)",
           cv_model(lambda: make_pipeline(StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=8000,
                                 random_state=0)), Xc, y), y, tail)


if __name__ == "__main__":
    main()
