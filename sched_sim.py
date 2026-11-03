"""
#4 End-to-end: faithful Continuum TTL scheduler simulation, comparing the flat
per-tool CDF predictor (Continuum) vs our call-conditioned predictor, holding
EVERYTHING ELSE identical.

Continuum's TTL objective (paper Eq. 2):
    tau* = argmax_tau  P(tau,f) * (T*eta + PrefillReload(r)) - tau
Only P(tau,f) differs between the two systems:
  - BASELINE: empirical CDF of past durations for tool f (code/arg-blind).
  - OURS:     per-call predicted duration distribution (conditioned on the call),
              turned into P(tau|call) via a lognormal around the point estimate.

Mechanism per turn (matches paper intent):
  - Request finishes decoding, emits a tool call; scheduler pins KV for tau*.
  - If the tool returns within tau*  -> HIT: instant resume (no prefill, no
    queueing delay).
  - If the tool returns after tau*   -> MISS: KV was evicted; returning request
    pays PrefillReload + queueing delay T.
  - Holding KV for tau also costs memory-time (the -tau term / opportunity cost),
    charged as delay to the system proportional to memory pressure.

Metric: total added delay across all turns (lower is better) = sum of
prefill+queueing paid on misses + memory-holding opportunity cost. This is the
quantity Continuum's TTL minimizes (job completion time).
"""
from __future__ import annotations
import json
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

RNG = np.random.default_rng(0)

# System constants (seconds), order-of-magnitude realistic for LLM serving.
PREFILL_RELOAD = 0.30   # cost to rebuild/reload a request's KV cache on a miss
T_WAIT = 0.40           # average queueing delay a returning evicted req faces
ETA = 1.0               # memoryfulness (fully memoryful => SJF benefit). paper eta
BENEFIT = T_WAIT * ETA + PREFILL_RELOAD  # coefficient on P(tau,f) in Eq.2
# Opportunity cost per second of holding one request's KV pinned. This is an
# OPPORTUNITY cost (fraction of a serving slot), not a wall-clock delay, so it
# is small relative to the miss penalty. Swept in __main__ for robustness.
MEM_COST_PER_S = 0.05


def load_calls(path, dur_key="dur_ms", feat_key=None, tool_key="tool"):
    durs, feats, tools = [], [], []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            durs.append(r[dur_key] / 1000.0)  # -> seconds
            tools.append(r.get(tool_key, r.get("first_word", "f")))
            if feat_key:
                feats.append(r[feat_key])
    return (np.array(durs, float), tools,
            np.array(feats, float) if feat_key else None)


# ---- Predictor P(tau, f) implementations -------------------------------------

def cdf_prob(sample_durs):
    """Return P(tau,f) as empirical CDF function over a pooled set of durations
    (Continuum's flat predictor)."""
    s = np.sort(np.asarray(sample_durs))
    def P(tau):
        return float(np.searchsorted(s, tau, side="right") / len(s))
    return P


def lognormal_prob(mu_point, sigma):
    """Our predictor: given a predicted point duration, model P(tau) as the CDF
    of a lognormal centered at the prediction (captures per-call uncertainty)."""
    # lognormal with median = mu_point
    logmed = math.log(max(mu_point, 1e-6))
    def P(tau):
        if tau <= 0:
            return 0.0
        z = (math.log(tau) - logmed) / sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    return P


def best_tau(P, candidates, mem_cost):
    """Solve Eq.2: argmax_tau P(tau)*BENEFIT - mem_cost*tau."""
    best, best_r = 0.0, -1e18
    for tau in candidates:
        r = P(tau) * BENEFIT - mem_cost * tau
        if r > best_r:
            best_r, best = r, tau
    return best


def turn_delay(tau, actual_dur, mem_cost):
    """Added delay for one turn given chosen TTL and the realized duration.

    Model (matches Continuum Eq.2 semantics, in delay units):
      - Memory-holding opportunity cost is charged for how long we actually PIN
        the KV: min(tau, actual_dur). (If the tool returns first we unpin then;
        if TTL expires first we unpin at tau.)
      - On a MISS (actual_dur > tau) the returning request additionally pays
        PREFILL_RELOAD + T_WAIT.
    The MEM weight is scaled so holding cost is an opportunity cost per second,
    comparable to -tau in Eq.2.
    """
    hold = mem_cost * min(tau, actual_dur)
    if actual_dur <= tau:
        return hold                 # HIT
    return hold + PREFILL_RELOAD + T_WAIT  # MISS


def oracle_delay_for(actual_dur, mem_cost):
    """True optimal: the per-call BEST achievable delay with perfect knowledge.
    Either pin until return (hold=mem*actual, hit) OR evict now (miss). Min."""
    pin = mem_cost * actual_dur
    nopin = PREFILL_RELOAD + T_WAIT
    return min(pin, nopin)


def simulate(durs, preds_point, tools, mem_cost, sigma=0.8):
    """Compare baseline (per-tool CDF) vs ours (per-call lognormal).
    preds_point: per-call predicted duration (ours). durs: realized durations."""
    hi = float(max(durs.max(), 0.01))
    cands = np.concatenate([[0.0], np.geomspace(1e-3, hi * 1.5, 40)])

    tool_to_idx = {}
    for i, t in enumerate(tools):
        tool_to_idx.setdefault(t, []).append(i)

    base_delay = ours_delay = oracle_delay = 0.0
    for i in range(len(durs)):
        others = [j for j in tool_to_idx[tools[i]] if j != i]
        pool = durs[others] if others else durs
        Pb = cdf_prob(pool)
        base_delay += turn_delay(best_tau(Pb, cands, mem_cost), durs[i], mem_cost)
        Po = lognormal_prob(preds_point[i], sigma)
        ours_delay += turn_delay(best_tau(Po, cands, mem_cost), durs[i], mem_cost)
        oracle_delay += oracle_delay_for(durs[i], mem_cost)
    n = len(durs)
    return base_delay / n, ours_delay / n, oracle_delay / n


def predict_point(X, y):
    """Leave-one-out-ish per-call point prediction via 5-fold RF (log space)."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds = np.empty_like(y)
    ylog = np.log1p(y)
    for tr, te in kf.split(X):
        m = RandomForestRegressor(n_estimators=400, random_state=0)
        m.fit(X[tr], ylog[tr])
        preds[te] = np.expm1(m.predict(X[te]))
    return np.clip(preds, 1e-4, None)


def run_track(name, path, feat_key, tool_key, mem_costs):
    durs, tools, X = load_calls(path, feat_key=feat_key, tool_key=tool_key)
    preds = predict_point(X, durs)
    print(f"\n### {name}  (n={len(durs)}, dur {durs.min()*1000:.1f}-"
          f"{durs.max()*1000:.0f} ms)")
    print(f"  {'mem_cost/s':>10} | {'Continuum':>10} {'Ours':>10} "
          f"{'Oracle':>10} | {'Ours vs Cont':>14}")
    for mc in mem_costs:
        base, ours, oracle = simulate(durs, preds, tools, mc)
        gap = (base - ours) / base * 100.0 if base else 0.0
        print(f"  {mc:10.3f} | {base:10.4f} {ours:10.4f} {oracle:10.4f} | "
              f"{gap:12.1f}%")


if __name__ == "__main__":
    print("END-TO-END TTL SCHEDULER SIM (Continuum Eq.2; only P(tau,f) differs)")
    print(f"PREFILL_RELOAD={PREFILL_RELOAD}s T_WAIT={T_WAIT}s eta={ETA} "
          f"BENEFIT={BENEFIT}  (delay units = seconds; lower is better)")
    # Sweep the memory opportunity-cost rate to show the result is not an
    # artifact of one cost setting.
    MC = [0.02, 0.05, 0.1, 0.25]
    run_track("T1 BFCL-Compute", "real_labeled.jsonl", "feat_callsite", "tool", MC)
    run_track("T2c Heavy-Shell", "shell_heavy_labeled.jsonl", "feat", "first_word", MC)
    run_track("T2b Light-Shell", "shell_labeled.jsonl", "feat", "first_word", MC)
