# Predicting Tool-Call Durations for Agentic LLM Serving

A research/design document capturing the full investigation: from CPU
execution-time prediction tools, through branch predictors and code analysis,
to a concrete proposal for improving KV-cache management in vLLM for agentic
(especially coding-agent) workloads.

Status: active research. Contains confirmed experimental results (Oct 2026)
plus open directions.

---

## 0. One-paragraph thesis

Agentic LLM serving interleaves model inference with tool calls. During a tool
call, a request's KV cache occupies scarce GPU memory while producing nothing.
Continuum (arXiv 2511.02230) manages this with a per-tool **time-to-live (TTL)**
driven by a flat per-tool **empirical CDF** of past durations. That predictor is
**code-blind**: it gives every call to a tool the same duration distribution.
For **coding agents**, whose dominant tool is *local code execution*, the
executed code itself is strong signal the CDF throws away. Our idea: predict
tool-call duration by **conditioning on the code/arguments** (AST + LLVM IR +
microarchitecture features), output a **calibrated distribution** (not a point),
and feed it into Continuum's TTL cost model — improving KV-cache retention
decisions, especially on the long tail that dominates delay.

---

## 1. The prediction problem, layered

"How long will this take on a CPU?" is answered by four distinct layers, each
at a different granularity. Confusing them is the main source of wrong tool
choices.

| Layer | Granularity | Tools | Predict vs measure |
|---|---|---|---|
| L1 Static basic-block | instructions/cycles | llvm-mca, uiCA, OSACA, IACA | PREDICT (no run) |
| L2 ML cost models | basic block | Ithemal, DiffTune, COMET | PREDICT (learned) |
| L3 Microarch predictors | single branch/load | TAGE, perceptron (hardware) | neither — they *cause* speed |
| L4 WCET | whole task upper bound | aiT, Bound-T, pWCET | PREDICT (safe bound) |
| (systems) Request latency | request/tool call | OSDI tail-latency, ML predictors | PREDICT (from features) |

Key distinction the whole project hinges on:
- **llvm-mca / perceptron / TAGE** predict *on-core instruction timing*
  (nanoseconds, assume cache-resident data + perfect prediction).
- **Tool-call duration** is a *request-level* quantity (ms–s). For I/O-bound
  tools it's dominated by network/external services (microarch tools useless).
  For **CPU-bound code-execution tools** it's dominated by computation — and
  THEN L1/L2/L4 features become relevant, but only as *inputs* to a
  higher-level model, never as the end-to-end predictor (they can't see
  data-dependent loop trip counts).

### 1.1 Fundamental limit (recurring theme)
A program's runtime is **not fully defined by its code** — it depends on inputs,
files, network (confirmed by the runtime-error-prediction literature,
arXiv 2203.03771). `for i in range(n)` costs O(n) but `n` may not be in the
code. Therefore:
- No static tool predicts real wall-clock perfectly. Everything is
  probabilistic (so is an LLM — nobody claims 100%).
- The right bar is **"better than the code-blind baseline (flat CDF)"**, not
  "perfect".
- Coding agents are a *favorable* case: agents often emit **self-contained
  scripts where inputs are literals in the code**, so code-conditioning has
  more to work with than the general case.

---

## 2. Branch predictors (L3) — what they are and why they don't predict duration

Covered because the user asked, and because they clarify the limit above.

### 2.1 Perceptron predictor (Jiménez & Lin, HPCA 2001)
- Replaces 2-bit saturating counters with a **perceptron** (single-layer neural
  net). Prediction = sign of dot(weights, recent-history-bits); weights trained
  online.
- Breakthrough property: hardware cost scales **linearly** with history length
  (vs exponential for table schemes like gshare) → can exploit *long* histories.
- Reported: ~1.94% misprediction on SPEC2000 int at 4KB budget (53% better than
  gshare); ~15.8% IPC gain over gshare.
- Limitation: linear classifier → only linearly-separable patterns. Follow-up
  **path-based neural predictor** (TOCS 2005) +16% IPC. AMD Zen ships
  neural/perceptron-style predictors.

### 2.2 TAGE (Seznec & Michaud)
- **TAG**ged **GE**ometric history length. Multiple tagged tables indexed by
  **geometrically increasing** history lengths (4,8,16,...,128+ branches). On
  lookup, use the match from the *longest* history that hits.
- Captures correlation on both recent and very old branches; storage-efficient.
- TAGE-SC-L (TAGE + Statistical Corrector + Loop predictor): ~3.99 MPKI on
  CBP-5 — state of the art. **Intel cores use TAGE-derived predictors** (so this
  is literally inside the i7-13700F here).

### 2.3 Why they matter to us
They explain WHY static prediction is capped: branch outcome depends on the
**runtime global history** (last 100–200 branches) — data-dependent, exists only
at runtime. A static tool has no history to feed TAGE/perceptron, so it cannot
know what the hardware will predict. The thing that makes real code fast
(data-driven history correlation) is invisible to code-only analysis.

### 2.4 Installed simulators (for reference / experimentation)
- **ChampSim** (`../ChampSim`): full OoO core sim; built 4 predictor binaries —
  `champsim_{bimodal,gshare,perceptron,hashed_perceptron}`. perceptron &
  hashed_perceptron are the Jiménez&Lin implementations. Needs branch traces
  (DPC-3 `.champsimtrace.xz`; original TAMU links dead, use Google Drive mirror
  or ChampSim `tracer/`).
- **CBP-16** (`../CBP-16-Simulation`): bundles Seznec TAGE-SC-L in
  `cbp16sim/src/simnlog/predictor.h`. predictor.o builds; `simnlog` needs
  `libboost-iostreams-dev` (apt) + `.bt9.trace.gz` traces.
- Standalone reference source: `../PerceptronBranchPredictor` (64KB Jiménez,
  59-bit GHR, 1K perceptrons × 60 weights), `../TAGEBranchPredictor`.

VERDICT for our project: these are **not the predictor** for tool duration.
They are at most a *feature source* (via llvm-mca-style cycle estimates) for the
CPU-bound-code case.

---

## 3. Static / code-level execution-time prediction (the tools that DO apply)

### 3.1 Microarchitectural code analyzers (L1) — installed
- **llvm-mca** (`/usr/bin/llvm-mca`, LLVM 14): reads assembly, predicts
  cycles/basic-block from LLVM scheduling models. Assumes perfect prediction +
  cache-resident data → gives *best-case* cycles. Use `# LLVM-MCA-BEGIN/END`
  asm markers to analyze just a hot loop. CANNOT model calls (assumes 100cy) or
  data-dependent trip counts.
- **uiCA / OSACA / IACA**: same class; uiCA most accurate on recent Intel.
- Empirically (our demo): llvm-mca said ~5 cyc/iter for a loop that really ran
  3.3× slower due to branch mispredicts it can't see. Good for *relative* loop
  tuning; poor for wall-clock.

### 3.2 Dynamic / post-analysis tools — installed
- **perf stat** (unlocked via `kernel.perf_event_paranoid<=0`): real HW counters
  — cycles, IPC, branch-misses, cache-misses. Hybrid CPU reports `cpu_core`
  (P-core) & `cpu_atom` (E-core) separately; pin with `taskset -c 0`.
- **toplev** (`../pmu-tools/toplev.py`): Intel Top-Down; names the bottleneck
  (Frontend / Backend / **Bad-Speculation** / Retiring). "Triage in seconds".
- **cachegrind** (`valgrind --tool=cachegrind --branch-sim=yes`): exact
  simulated cache + branch-mispredict counts; slow but deterministic.

### 3.3 PrETi (arXiv 2503.13679) — the blueprint we follow
"Predicting Execution Time in Early Stage with LLVM and Machine Learning."
- Rejects pure source features (compiler transforms distort) and pure binary
  features (can't link blocks to input-dependent paths).
- Uses **LLVM IR instrumentation** as the sweet spot. Features:
  1. basic-block execution counts (captures loop trip counts),
  2. simulated data-cache hits/misses,
  3. branch decisions,
  combined with historical execution-time labels.
- Models: Linear Regression, Huber Regression, Random Forest, MLP (chosen as
  robust on small data). Metric: Absolute Percentage Error (APE).
- Trained on 1348 samples at -O2.
- THIS is the validated feature set + model family we adopt.

### 3.4 Other code→runtime literature
- Code runtime **complexity** prediction via code embeddings (arXiv 1911.01155)
  — predicts Big-O class; coarse, input-free.
- **TEP-GNN** (arXiv 2208.11947): GNN on flow-augmented ASTs → predicts **test
  execution times**. Directly on-point for coding agents (running tests).
- **Mantis** (arXiv 1010.0019): instrument + feature-select → program runtime.
- **Regression Language Models for Code** (arXiv 2509.26476): LLM regresses
  numeric props (memory, latency) directly from code text, multi-language.
- Comparative AST/LSTM perf prediction (arXiv 2102.07660): ~73–84% accuracy on
  the easier *relative* (faster/slower) question.

### 3.5 AST vs MLIR/LLVM-IR representations (feature engineering choice)
- **AST** (Python `ast` module): cheap, language-native, captures loop nesting,
  node counts, literals. Good first signal; misses the *interaction* (depth ×
  bound = effective op count) and runtime path. Implemented in `features_ast.py`.
- **LLVM IR / MLIR**: richer, captures basic-block structure & instruction mix;
  PrETi's choice. For Python tools this needs a lowering path (e.g., compile a
  C/Numba equivalent, or instrument CPython bytecode as an IR proxy). Heavier
  but captures what AST can't. MLIR relevant if targeting compiled/kernel tools.
- Practical plan: start AST (done), add an **effective-operation-count** feature
  (product of loop bounds along nesting), then add IR/bytecode instrumentation
  à la PrETi if AST+op-count is insufficient on the tail.

---

## 4. Continuum (arXiv 2511.02230) — the system we extend

"Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache
Time-to-Live." Berkeley (EECS tech report EECS-2026-234). PDF saved at
`../continuum/continuum_2511.02230.pdf` (+ extracted `continuum.txt`).
**No public code repo** at time of writing (authors say they will open-source
traces + code + testbed *upon publication*).

### 4.1 Problem it solves
Agentic workloads interleave LLM calls with tool calls. Standard engines (vLLM,
SGLang) evict a request's KV cache when decoding finishes (assuming it's done).
But the agent will return after the tool call — so the cache must be
recomputed (prefill) or reloaded from CPU (if offloading), adding delay EVERY
turn. Over 10+ turns this accumulates badly.

### 4.2 Mechanism
- For a request that emits a tool call, **pin** its KV cache in GPU with a
  **TTL**. If the tool returns within TTL → instant resume (saves prefill +
  queueing). If TTL expires → evict (safety net, prevents memory pressure /
  deadlock).
- TTL chosen to maximize expected reward balancing:
  - **Benefit**: avoided prefill + queueing delay if cache hit.
  - **Cost**: GPU memory occupied during TTL (opportunity cost / queueing
    induced on others).
- Combined with **program-level FCFS** scheduling (approximates SRTF since later
  turns have shorter expected remaining time).

### 4.3 How it predicts duration — THE WEAKNESS WE TARGET
- Keeps historical per-tool records `S[f]` (durations seen for tool `f`).
- Estimates `P(τ, f)` = P(tool `f` finishes within τ) as the **empirical CDF**:
  `P(τ,f) = (1/|S[f]|) · Σ_{t∈S[f]} 1[t ≤ τ]`.
- Picks TTL `τ*` maximizing expected reward over candidate τ's.
- Cold start: fall back to global CDF `P(τ, f_any)`, then to a fixed default
  assuming `ToolCallDuration ~ Exp(1)`.
- **This is code-blind / argument-blind.** Every `execute_code` call gets the
  same distribution regardless of whether it runs `print("hi")` or a training
  loop.

### 4.4 Evidence that supports OUR idea (from their own paper)
- Table 2 (collected datasets):
  - SWE-Bench: 10.9 turns/prog; tool time **(mean 925 ms, std 3550 ms)**;
    70,126 tokens/prog.
  - BFCL v4: 6.3 turns; tool time **(mean 1923 ms, std 2133 ms)**;
    93,256 tokens/prog.
  - → variance ≈ 4× the mean: a single CDF is a very wide, weak point predictor.
- Figure 5: execution time is **extremely long-tailed WITHIN a tool**:
  slowest 10% of `fetch_url` = 52.5% of total delay; slowest 10% of `cd` =
  **94.1%** of total delay. The tail is where scheduling value lives, and it is
  exactly what per-tool averaging misses.
- Implementation: ~1k lines of Python on top of vLLM, via tool-call handler
  hooks in the scheduler. → small, well-defined integration surface (same vLLM
  we built from source here).

### 4.5 Baselines Continuum compares against (our related work)
- **InferCept**: "preserve" op pins KV only when CPU-offload cost exceeds reload
  cost of the immediate next turn; ignores queueing delay → bubbles persist.
- **MORI** (arXiv 2606.00866): exploits tool-call idle windows for offloading.
- **LMCache**: fast async CPU offload (reduces reload cost).
- **AsymCache** (arXiv 2606.02964): integrates into Continuum, −18.1% job
  latency.
- **Adaptive KV Retention at human-approval timescales** (arXiv 2608.30830):
  targets minute/hour pauses (human approval), beats Continuum 41–52% there.
- Counterpoint: **Beyond Prediction** (arXiv 2606.18431): prediction-driven
  scheduling is FRAGILE under distribution shift/bursts/memory pressure; a
  prediction-FREE scheduler beats SRPT-with-perfect-prediction by 35–50% on P99.
  → our predictor must feed a mechanism (TTL) that degrades gracefully; we do
  not bet the system on a point prediction.

---

## 5. Our proposal

### 5.1 Statement
Replace Continuum's flat per-tool CDF `P(τ,f)` with a **code/argument-conditioned
predictive distribution** `P(τ | f, features(call))`, for the CPU-bound
code-execution tool class, while keeping the TTL-expiry safety net for
robustness.

### 5.2 Feature sources (nothing from the investigation wasted)
1. **Code structure (AST)** — loop nesting depth, node counts, literals,
   comprehensions, recursion hints. `features_ast.py` (done).
2. **Effective operation count** — product of loop bounds along nesting depth
   (an O() estimate). THE missing feature (see §6 results). TODO.
3. **Static microarch (llvm-mca / IR instrumentation)** — PrETi-style basic-block
   exec counts, cache-sim, branch decisions. For the compute-bound tail. TODO.
4. **History (Continuum CDF)** — fallback + calibration, esp. for I/O tail.

### 5.3 Output = distribution, not point
Predict a calibrated conditional distribution (e.g., quantile regression or a
mean+variance with coverage calibration, à la MAPS arXiv 2609.15359 "uncertainty
-aware upper bounds with target coverage"). TTL uses the distribution directly.

### 5.4 Integration point in vLLM/Continuum
Continuum's `CalcTTL(r, S[f])` → `CalcTTL(r, predictor(call))`. The tool-call
handler already parses tool calls; we attach a feature extractor + predictor and
return `P(τ|·)` to the TTL optimizer. ~Same 1k-LOC surface.

### 5.5 Why this answers the fragility critique
We do not replace the mechanism with a prediction; we give the *existing robust
mechanism* a better-informed, uncertainty-aware input. Mispredictions are caught
by TTL expiry (eviction) exactly as in vanilla Continuum.

---

## 6. Experiment #1 (DONE) — AST features vs flat CDF

Harness (all in this dir):
- `gen_dataset.py` → 29 code snippets tagged tool=`execute_code`, spanning
  trivial → nested loops → sorts → recursion. All under ONE tool name (forces
  the CDF baseline to lump them, = Continuum's coding-agent limitation).
- `measure.py` / `measure_dataset.py` → subprocess wall-clock, median of 5 runs
  → `labeled.jsonl`. Range 14–676 ms, mean 74, std 128 (same skew shape as
  Continuum Table 2).
- `compare.py` → leave-one-out CV; baselines = flat-CDF median/mean; models =
  RF/Huber/MLP on AST features. Metric: MAPE + median APE.

### 6.1 Results (i7-13700F, Oct 2026)
```
CODE-BLIND BASELINES (Continuum-style flat per-tool CDF)
  CDF-median                 MAPE= 41.3%   medAPE= 34.5%
  CDF-mean                   MAPE=233.2%   medAPE=296.5%
CODE-CONDITIONED (AST features)
  RandomForest(AST)          MAPE= 47.8%   medAPE= 34.8%
  Huber(AST,scaled)          MAPE=115.3%   medAPE= 39.6%
  MLP(AST,scaled)            MAPE= 79.6%   medAPE= 54.3%
Top AST features: largest_literal_int 0.68, n_nodes 0.13, n_calls 0.07
```

### 6.2 Honest interpretation
- AST features **did NOT beat** CDF-median on average error (tied/worse).
- Reason: (a) most snippets ~14–18 ms dominated by interpreter startup → runtime
  flat → "predict the median" already near-optimal; (b) model leans on
  `largest_literal_int` (0.68), a brittle proxy that can't capture
  depth×bound interaction (O(n) vs O(n²)); (c) 29 samples too few for RF/MLP.
- This is real evidence FOR the "Beyond Prediction" camp: crude baselines are
  robust when overhead dominates.
- BUT average MAPE is the WRONG metric for the goal: Continuum Fig 5 says the
  slow 10% cause ~94% of delay. Scheduling value is in the tail; averaging
  hides it. CDF-median predicting 18 ms for a 676 ms call is catastrophic for
  that call yet barely moves the average.

### 6.3 Conclusion
Thesis not refuted — tested with wrong features + wrong metric. Needs the
decisive version (Experiment #2).

---

## 6b. Experiment #2 (DONE) — AST + effective-op-count, tail metric, 150 samples

Changes from #1: (a) added `log_effective_ops` feature = log10(1 + Σ over loops
of product-of-enclosing-loop-bounds), capturing O(n) vs O(n^2) interaction;
(b) grew dataset to 150 randomized snippets; (c) added tail-weighted metric
(APE on slowest-20%) and Spearman ranking.

### 6b.1 Results (i7-13700F, Oct 2026)
150 samples, 14–698 ms, mean 94, std 138, tail(>=p80)=118 ms (30 calls).
```
                             MAPE     medAPE   tailMAPE
CODE-BLIND (Continuum CDF)
  CDF-median                 59.0%    65.3%     87.6%
  CDF-mean                  237.8%   231.2%     59.5%
CODE-CONDITIONED (AST + op-count)
  RandomForest               11.5%     7.9%      9.6%   <-- WIN
  Huber                      41.5%    26.0%     41.9%
  MLP                        30.1%    20.1%     26.5%
Ranking (Spearman vs true duration)
  CDF-median   -0.866  (flat -> cannot rank)
  RandomForest  0.873  (strong -> good SJF ordering)
Top features: log_effective_ops 0.82, n_nodes 0.07, largest_literal_int 0.045
```

### 6b.2 Interpretation — THESIS CONFIRMED
- RandomForest beats the Continuum flat CDF ~5x on overall MAPE (11.5 vs 59) and
  ~9x on the TAIL (9.6 vs 87.6). The tail is where Continuum Fig 5 says 94% of
  delay lives — so the win is exactly where it matters for scheduling.
- `log_effective_ops` is the dominant feature (0.82); the brittle
  `largest_literal_int` from Exp #1 collapsed to 0.045 once the interaction
  feature existed. This was the precise fix predicted in 6.2.
- Spearman: RF +0.87 vs CDF −0.87. The CDF literally cannot rank calls (one
  number for all), so it cannot support SJF/SRTF-style ordering; the
  code-conditioned predictor can.
- Caveat (honest): synthetic dataset with literal loop bounds — the favorable
  "inputs are in the code" case for coding agents. Real BFCL/SWE-Bench calls
  will have hidden inputs; expect accuracy to drop. That is the next test.

---

## 7. Next steps (prioritized)

1. **Experiment #2 (decisive)**: add effective-operation-count feature; evaluate
   with a **tail-weighted metric** (APE on top-20% slowest calls); grow dataset
   to a few hundred snippets with randomized bounds/depths. Decides if
   code-conditioning wins where it matters (the tail).
2. Add PrETi-style **IR/bytecode instrumentation** features if AST+op-count
   insufficient (basic-block exec counts via CPython bytecode tracing, or
   compile numeric kernels and use llvm-mca).
3. **Distribution output** (quantile regression) + coverage calibration (MAPS).
4. **Real data**: extract executable snippets from BFCL (`../gorilla`,
   `berkeley-function-call-leaderboard/.../func_source_code/*.py`) and measure
   their durations; later SWE-Bench; adopt Continuum traces when open-sourced.
5. **Integrate** into vLLM scheduler as a `P(τ|·)` provider; A/B vs flat CDF on
   job completion time under an agentic trace.

---

## 8. Datasets to use

| Dataset | Where | Use |
|---|---|---|
| BFCL v4 | `../gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/*.json` + `.../func_source_code/*.py` | Executable multi-turn tool calls; measure real durations |
| SWE-Bench | public (HF) | Coding-agent tool calls (cd, run tests); matches Continuum Table 2 |
| Continuum traces | not yet released (authors will open-source) | Exact replication when available |
| Synthetic (ours) | `gen_dataset.py` | Controlled code→duration sweep for predictor dev |
| CBP branch traces | Google Drive mirror / ChampSim tracer | ONLY for branch-predictor sims, NOT tool duration |

---

## 9. Environment / reproducibility notes

- vLLM built from source (v0.11.0, `vllm 0.11.1.dev0+gb8b302cde`) in
  `../vllm-env`; PyTorch 2.8.0+cu128; GPU RTX 4060 Ti (16GB, sm_89).
  `transformers` pinned to 4.57.6 (5.x breaks tokenizer).
- perf unlocked: `kernel.perf_event_paranoid=-1` (resets on reboot).
- Swap: /swapfile2 + /swapfile3 (32G) added, persisted in /etc/fstab.
- Tools: llvm-mca (LLVM14), toplev (`../pmu-tools`), valgrind 3.18.1.
- Predictor code: this directory. Python via system `python3` (sklearn 1.9).

---

## 6c. Experiment #3 (DONE) — REAL BFCL tool calls, hidden runtime inputs

The make-or-break test (next-step #1). Uses ACTUAL BFCL `MathAPI` functions
(`mean`, `standard_deviation`, `sum_values`, `min_value`, `max_value`) called
with varying runtime list arguments. Duration depends on `len(list)` — a
RUNTIME ARGUMENT the agent chooses, NOT a literal in the function body. This is
the realistic "inputs not in code" case Exp #2 did not cover.
Code: `real_bfcl.py` (dataset) + `compare_real.py`. Data: `real_labeled.jsonl`.

Two information levels:
- FUNCTION-ONLY = per-tool flat CDF (exactly Continuum's `S[f]`).
- CALL-SITE = tool id + argument-size features (from the agent's emitted call).

### 6c.1 Results (i7-13700F, Oct 2026)
300 real calls, 0.00–254.0 ms, mean 10.5, tail(>=p80)=12.2 ms (60 calls).
```
                             MAPE      medAPE   tailMAPE   Spearman
per-tool CDF (Continuum)    4417.6%    93.8%    91.7%      +0.060
RandomForest(callsite)         8.8%     1.3%     1.5%      +0.997  <-- WIN
MLP(callsite)                111.1%    19.2%    10.1%      +0.980
Huber(callsite)             2318.9%    86.3%    82.2%      +0.836
```

### 6c.2 Interpretation — THESIS CONFIRMED ON REAL DATA
- RF on call-site features: 1.3% median error, rho=0.997 (near-perfect ranking).
- Continuum per-tool CDF: 93.8% median error, rho=0.060 (essentially guessing).
- Root cause: duration is ~linear in len(list); the CDF is blind to the
  argument, so it lumps mean(1e3) and mean(6e6) into one distribution.
  Predicting the ~1.5ms tool median for a 254ms call yields the 4417% MAPE.
- CRUCIAL: this is the hidden-input case. The arg size is NOT in the function
  BODY, but it IS in the agent's emitted CALL. Conditioning on the call site
  (not just the tool name) is what wins — and that information is exactly what a
  scheduler has when the tool call is issued.
- Honest caveats: (a) pure-compute functions (no I/O); network-bound tools would
  not be this predictable from args alone — history still needed there.
  (b) Linear-in-N is an easy relationship; real compile/test tasks are messier.
  But the DIRECTIONAL result is unambiguous: argument/code conditioning
  dominates the flat CDF whenever duration depends on the call, which is the
  common case for coding/data agents.

### 6c.3 Takeaway for the proposal
Continuum's abstraction — "duration ~ S[f], a per-tool distribution" — discards
the single most predictive signal (the call's arguments). A predictor
conditioned on the call site is a strict, large improvement on accuracy AND
rankability; the rankability (rho 0.997 vs 0.06) is what SJF/SRTF-style
scheduling needs. Both synthetic (Exp #2) and real (Exp #3) confirm it.
