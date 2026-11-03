"""
Track T2 comparison + distributional prediction (next-step #3).

Baseline: Continuum per-first-word CDF (median of other calls with same verb).
Ours: RF / GBM-quantile on full-command features.

Adds distributional eval: Quantile Gradient Boosting predicts P50/P90; we report
point error (P50) AND interval coverage (does the true value fall under the P90
upper bound at the target 90% rate? -> calibration, à la MAPS arXiv 2609.15359).
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from scipy.stats import spearmanr


def load():
    fw, X, y = [], [], []
    with open("shell_heavy_labeled.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            fw.append(r["first_word"])
            X.append(r["feat"])
            y.append(r["dur_ms"])
    return np.array(fw), np.array(X, float), np.array(y, float)


def ape(p, t):
    return np.abs(p - t) / np.maximum(t, 1e-6) * 100.0


def per_firstword_cdf(fw, y):
    preds = np.empty_like(y)
    for i in range(len(y)):
        same = (fw == fw[i]); same[i] = False
        preds[i] = np.median(y[same]) if same.any() else np.median(y)
    return preds


def cv_point(make, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    preds = np.empty_like(y); ylog = np.log1p(y)
    for tr, te in kf.split(X):
        m = make(); m.fit(X[tr], ylog[tr]); preds[te] = np.expm1(m.predict(X[te]))
    return np.clip(preds, 0, None)


def cv_quantile(X, y, q, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    preds = np.empty_like(y); ylog = np.log1p(y)
    for tr, te in kf.split(X):
        m = GradientBoostingRegressor(loss="quantile", alpha=q,
                                      n_estimators=300, max_depth=3,
                                      random_state=0)
        m.fit(X[tr], ylog[tr]); preds[te] = np.expm1(m.predict(X[te]))
    return np.clip(preds, 0, None)


def report(name, p, y, tail):
    a = ape(p, y)
    print(f"{name:28s} MAPE={a.mean():7.1f}%  medAPE={np.median(a):6.1f}%  "
          f"tailMAPE={a[tail].mean():7.1f}%  rho={spearmanr(p,y).correlation:+.3f}")


def main():
    fw, X, y = load()
    thr = np.quantile(y, 0.80); tail = y >= thr
    print(f"{len(y)} shell calls; {y.min():.1f}-{y.max():.1f} ms; "
          f"mean {y.mean():.1f}; tail(>=p80)={thr:.1f}ms ({tail.sum()})\n")

    print("=== CONTINUUM INFO (per-first-word CDF) ===")
    report("per-first-word CDF", per_firstword_cdf(fw, y), y, tail)

    print("\n=== OURS: full-command features (point) ===")
    report("RandomForest(full cmd)",
           cv_point(lambda: RandomForestRegressor(n_estimators=400,
                    random_state=0), X, y), y, tail)

    print("\n=== OURS: distributional (Quantile GBM) ===")
    p50 = cv_quantile(X, y, 0.50)
    p90 = cv_quantile(X, y, 0.90)
    report("QuantileGBM P50 (point)", p50, y, tail)
    cov90 = float(np.mean(y <= p90)) * 100.0
    print(f"{'QuantileGBM P90 coverage':28s} target=90.0%  actual={cov90:5.1f}%  "
          f"(fraction of true durations <= predicted P90 upper bound)")
    # A well-calibrated P90 upper bound => TTL that rarely under-estimates.


if __name__ == "__main__":
    main()
