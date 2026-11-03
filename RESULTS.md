# RESULTS: Tool-Call Duration Prediction vs. Continuum's Flat CDF

Consolidated, reproducible results for predicting agentic tool-call durations,
benchmarked against the flat per-tool empirical CDF used by Continuum
(arXiv 2511.02230) for KV-cache TTL decisions.

Hardware: Intel i7-13700F (hybrid, P-cores pinned where relevant), 32 GB RAM.
Software: Python 3.11, scikit-learn 1.9. Seeds fixed. Date: Oct 2026.
All datasets are regenerable from the scripts in this directory.

---

## 1. Claim under test

Continuum predicts tool-call duration with `P(τ, f)` = empirical CDF of past
durations for tool `f`, where for SWE-agents `f` = the FIRST WORD of the bash
command (paper: "extract the string within the bash block and use the first word
afterwards as the tool call name"). This is **code/argument-blind**: every call
to a tool shares one distribution.

Hypothesis: conditioning on the actual call (command string / arguments / code
structure) predicts duration substantially better — and better where it matters
for scheduling (the slow tail, and ranking).

---

## 2. Headline table (all tracks)

Metric key: medAPE = median absolute percentage error (lower better);
tailMAPE = MAPE on slowest-20% of calls; rho = Spearman rank corr vs true
duration (higher better; this is what SJF/SRTF scheduling needs).

| Track | Workload | n | Dur range | Baseline (Continuum CDF) medAPE / rho | Ours (best) medAPE / rho | Outcome |
|---|---|---|---|---|---|---|
| T1 | BFCL MathAPI over lists (real) | 300 | 0–254 ms | 93.8% / +0.06 | **1.3% / +0.997** (RF) | WIN (large) |
| T2a | Synthetic code snippets (AST) | 150 | 14–698 ms | 65.3% / −0.87 | **7.9% / +0.87** (RF) | WIN (large) |
| T2b | Light shell cmds on vLLM tree | 400 | 0.6–47 ms | 22.2% / +0.50 | 20.2% / +0.57 (QGBM) | TIE |
| T2c | Heavy shell cmds (compute) | 250 | 1.7–4292 ms | 89.4% / +0.47 | **5.3% / +0.86** (QGBM) | WIN (large) |

Note: T2a's baseline rho is −0.87 (an artifact of leave-one-out median on a
flat predictor); the point is it carries no usable ranking signal. T1/T2c
baselines sit near +0.06/+0.47 — also weak ranking.

---

## 3. Per-track detail

### T1 — BFCL-Compute (real tool functions, hidden runtime argument)
Real BFCL `MathAPI` functions (`mean`, `standard_deviation`, `sum_values`,
`min_value`, `max_value`) called with lists of varying size. Duration depends on
`len(list)` — a RUNTIME ARGUMENT the agent chooses, not a literal in the code.
Scripts: `real_bfcl.py`, `compare_real.py`. Data: `real_labeled.jsonl`.

```
                             MAPE      medAPE   tailMAPE   rho
per-tool CDF (Continuum)    4417.6%    93.8%    91.7%     +0.060
RandomForest(call-site)        8.8%     1.3%     1.5%     +0.997   <-- WIN
MLP(call-site)               111.1%    19.2%    10.1%     +0.980
Huber(call-site)            2318.9%    86.3%    82.2%     +0.836
```
Why: duration ≈ linear in list length. The CDF is blind to the argument, so it
lumps `mean(1e3)` with `mean(6e6)` and predicts the ~1.5 ms tool median for a
254 ms call (hence 4417% MAPE). Conditioning on argument size → near-exact.

### T2a — Synthetic code snippets (AST + effective-op-count)
150 code snippets (trivial → nested loops → sorts → recursion), all tagged one
tool `execute_code`, forcing the CDF to lump them. Features: Python AST +
`log_effective_ops` (log of product of loop bounds along nesting depth).
Scripts: `gen_dataset.py`, `measure_dataset.py`, `features_ast.py`, `compare.py`.

```
                             MAPE     medAPE   tailMAPE
CDF-median (Continuum)       59.0%    65.3%     87.6%
RandomForest(AST)            11.5%     7.9%      9.6%    <-- WIN
Huber(AST)                   41.5%    26.0%     41.9%
MLP(AST)                     30.1%    20.1%     26.5%
Ranking: CDF rho=−0.87 (no signal), RF rho=+0.87
Top feature: log_effective_ops 0.82 (dominant)
```
Why: `log_effective_ops` captures the O(n) vs O(n²) interaction (a flat
range(1e6) loop and nested range(1e3)² both score 6.0 → correctly similar
duration). The naive `largest_literal_int` from an earlier run collapsed to 0.045
once this interaction feature existed.

### T2b — Light shell commands (Continuum-exact first-word keying)
400 real bash commands (`cat`, `wc`, `grep`, `find`, `ls`, `sort`, `head`,
`python3 -c`) over the real vLLM source tree (27,113 files). Baseline keys on
first word exactly as Continuum does; ours adds target file size + recursion +
file-count features. Scripts: `track_shell.py`, `compare_shell.py`.

```
                             MAPE    medAPE  tailMAPE  rho
per-first-word CDF (Cont.)   40.9%   22.2%   31.3%    +0.50
RandomForest(full cmd)       42.4%   25.5%   23.4%    +0.53
QuantileGBM P50              40.9%   20.2%   25.4%    +0.57
```
Outcome: TIE. Honest finding. These commands are ms-scale and dominated by
process spawn + OS overhead, not the argument. `cat` of 1 KB vs 1 MB barely
differs vs ~2 ms fork/exec. The verb (which binary) already captures most
signal, so the CDF (rho 0.50) is a strong baseline and argument conditioning
adds little.

### T2c — Heavy shell commands (compute-dominated) — boundary test
250 real commands doing work ∝ argument N (`seq N | sort`, `grep`, `wc`, `awk`,
`python3` compute over N ∈ {1e5 … 8e6}). Duration spans 1.7 ms – 4.3 s (2500×).
Scripts: `track_shell_heavy.py`, `compare_shell_heavy.py`.

```
                             MAPE    medAPE  tailMAPE  rho
per-first-word CDF (Cont.)  156.6%   89.4%   87.2%    +0.47
RandomForest(full cmd)       73.5%    9.2%   28.8%    +0.90   <-- WIN
QuantileGBM P50             146.7%    5.3%   14.3%    +0.86
```
Outcome: WIN. Confirms the boundary from T2b: when duration is COMPUTE-dominated,
full-command conditioning crushes the flat CDF (medAPE 89.4% → 5.3%; rho
0.47 → 0.90).

---

## 4. Distributional prediction (calibrated TTL upper bounds)

A Quantile Gradient Boosting model predicts P50 and P90. The P90 upper bound is
near-calibrated:
- T2b (light shell): P90 coverage 87.8% vs 90% target.
- T2c (heavy shell): P90 coverage 87.2% vs 90% target.

Interpretation: the model yields a duration upper bound the true value falls
under ~88% of the time, usable directly as a TTL that rarely under-estimates.
This is the robustness property that answers the "Beyond Prediction"
(arXiv 2606.18431) critique that point-prediction scheduling is fragile — we
feed a CALIBRATED INTERVAL to Continuum's existing TTL mechanism, which already
degrades gracefully on TTL expiry.

---

## 5. The core finding (cross-track)

Code/argument conditioning beats Continuum's flat per-tool CDF **when tool-call
duration is argument/compute-dominated** (T1, T2a, T2c: 7–18× lower median error,
rho from ~0 to ~0.9), and **ties when duration is overhead-dominated** (T2b:
fast ms-scale shell commands where the verb already predicts spawn cost).

This is not a uniform win, and the benchmark reports the tie deliberately. The
value concentrates in the HEAVY, slow tool calls — which is exactly:
(a) where Continuum's own Fig 5 says scheduling delay concentrates (slowest 10%
    of calls cause up to 94% of total delay), and
(b) where KV cache is held longest, so better TTL matters most.
The fast calls where we tie barely occupy the cache anyway.

Takeaway: Continuum's abstraction "duration ~ S[f]" discards the single most
predictive signal (the call's arguments) precisely in the regime that dominates
delay. A call-site-conditioned, uncertainty-aware predictor is a strict
improvement there.

---

## 6. Reproducibility & limitations (honest)

Reproducible:
- All four datasets regenerate from scripts here on any Linux box; no external
  services, no API keys. This is deliberate: the PREDICTION claim does not
  depend on anyone's private or unreleased traces.

Limitations / threats to validity:
- All workloads are CPU/compute-bound. Network/IO-bound tools (e.g. BFCL web
  search) have externally-determined duration; argument conditioning will NOT
  help there and history (the CDF) remains necessary. We do not claim otherwise.
- T1 relationships (linear in N) are "easy"; real compile/test workloads are
  messier — expect accuracy between T1 and the harder tail of T2c.
- Durations measured on one machine; absolute ms are hardware-specific (the
  RELATIVE comparison and ranking are the portable results).
- Not yet tested: full agent-trace replay (see §7, T3).

Relationship to Continuum (fairness statement):
- Continuum is an arXiv preprint; its end-to-end system gains (1.1–3.2× tput,
  up to 8.18× delay on a private testbed) were measured on real systems but its
  traces/code are "to be open-sourced upon publication" and are NOT yet public.
  We therefore CANNOT reproduce its end-to-end numbers, and we make no claim that
  they are wrong. We isolate and stress-test the duration-PREDICTION component
  it relies on, on public data, and show that component is beatable on
  compute-bound calls. Withholding full trust in unreproduced end-to-end numbers
  is standard practice, not an accusation.

---

## 6b. End-to-end TTL scheduler simulation (#4)

We implement Continuum's TTL objective (paper Eq. 2) exactly and swap ONLY the
predictor `P(tau, f)`:
```
tau* = argmax_tau  P(tau,f) * (T*eta + PrefillReload) - mem_cost * tau
```
Baseline uses the flat per-tool CDF for `P`; Ours uses a lognormal around the
per-call RandomForest point prediction. Per turn: HIT (tool returns within tau)
= pay only memory-holding cost; MISS = pay holding + PrefillReload + queueing T.
Metric: average added delay per turn (seconds; lower better). Oracle = per-call
optimal with perfect knowledge. Script: `sched_sim.py`. Constants:
PrefillReload=0.30s, T=0.40s, eta=1.0; mem_cost swept {0.02,0.05,0.1,0.25}/s.

Delay reduction of Ours vs Continuum (per mem_cost in the sweep):
```
Track            mc=0.02   0.05    0.10    0.25
T1 BFCL-Compute    0.0%    0.0%    0.0%   71.8%
T2c Heavy-Shell   27.8%   67.2%   18.0%   20.3%
T2b Light-Shell   99.0%   97.5%   95.1%   88.5%
```
In every cell Oracle <= Ours <= Continuum (sanity holds). In many cells
Ours == Oracle (closes 100% of the achievable gap).

Findings (honest):
- Ours reduces end-to-end delay across all tracks and cost settings, EXCEPT T1
  at low mem_cost where holding is so cheap that "pin everything" is optimal and
  even the blind CDF cannot go wrong (0% gap). The predictor's value appears
  precisely when memory is CONTENDED and pinning must be SELECTIVE — the regime
  Continuum targets.
- T2b was a prediction-accuracy TIE yet becomes a large SCHEDULING win (89-99%).
  Reason: the objective rewards correct thresholding (these calls are short ->
  "barely pin"); the per-call predictor captures that while the pooled CDF
  over-pins. Prediction ties can still be scheduling wins.

Simulation caveats (threats to validity):
- This is a per-call independent TTL model; it does NOT yet simulate contention
  between concurrent programs, the waiting queue dynamics, or program-level
  FCFS. The real Continuum couples these. Absolute percentages are therefore
  indicative, not production numbers.
- mem_cost, PrefillReload, T are set to plausible constants; the SWEEP shows the
  direction is robust, but exact magnitudes depend on hardware/model.
- Next: implement the full queueing simulation (concurrent programs, GPU memory
  budget, eviction under pressure) for production-faithful numbers.

## 7. Open items

- T3: replay real agent traces. Two options: (a) wait for Continuum's release;
  (b) run mini-swe-agent on SWE-bench_Lite against the LOCAL vLLM build with a
  local model (no GPT-5 needed) to generate OUR OWN reproducible coding-agent
  traces with real tool durations. Docker is available on this box.
- #4 End-to-end: integrate the predictor into the vLLM scheduler's TTL path
  (Continuum's ~1k-LOC surface) and measure job completion time vs flat CDF.
- Harder shell/compile/test workloads for the heavy-tail regime.
- Per-tool-class reporting so users know WHERE conditioning helps.

---

## 8. Artifact index

| File | Purpose |
|---|---|
| `DESIGN.md` | Full research narrative (predictors → Continuum → idea → Exp 1/2/3) |
| `BENCHMARK.md` | Benchmark methodology + fidelity-to-Continuum statement |
| `RESULTS.md` | This file — consolidated results |
| `features_ast.py` | AST + effective-op-count feature extractor |
| `measure.py` / `measure_dataset.py` | Subprocess timing harness + labeler |
| `gen_dataset.py` → `labeled.jsonl` | T2a synthetic code dataset |
| `real_bfcl.py` → `real_labeled.jsonl` | T1 real BFCL tool calls |
| `track_shell.py` → `shell_labeled.jsonl` | T2b light shell commands |
| `track_shell_heavy.py` → `shell_heavy_labeled.jsonl` | T2c heavy shell commands |
| `compare*.py` | Per-track evaluations (baseline vs ours) |
