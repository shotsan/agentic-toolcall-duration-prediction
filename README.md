# A-Priori Tool-Call Duration Prediction for Agentic LLM Serving

Predicting how long a coding agent's tool call (bash command) will take **before
running it**, by analyzing the command's execution cost — and comparing against
the predictor used by Continuum (arXiv 2511.02230), a recent KV-cache scheduling
system for multi-turn LLM agents.

This repository contains the full pipeline: real agent trajectories, real
per-command durations measured inside the task environments, and six a-priori
cost-analysis predictors evaluated head-to-head with Continuum's method.

---

## 1. The problem

Modern LLM serving hosts **agentic** workloads: an agent interleaves LLM
inference steps with **tool calls** (bash commands, code execution, searches).
While a tool call runs, that request's **KV cache** sits in scarce GPU memory
doing nothing. The scheduler must decide: keep the KV cache pinned (betting the
agent returns soon) or evict it (freeing memory but paying to rebuild it later)?

That decision needs an estimate of **how long the tool call will take**. A good
duration estimate lets the scheduler pin short calls and evict during long ones.

So the core sub-problem is: **given a tool call, predict its duration.**

---

## 2. What Continuum does

Continuum ("Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache
Time-to-Live", arXiv 2511.02230; code: `github.com/Hanchenli/vllm-continuum`)
introduces a time-to-live (TTL) KV-cache retention mechanism. Its duration
predictor, read directly from the paper and shipped code
(`vllm/v1/core/estimate_with_func.py`):

- **Tool key = the FIRST WORD of the bash command.** From the paper: *"We
  extract the string within the bash block and use the first word afterwards as
  the tool call name."* In code: `ToolCallParser` regex-extracts the bash block
  and returns `words[0]` (e.g. `python`, `ls`, `grep`, `pytest`).
- **Prediction = per-first-word average duration.** Shipped code keeps a running
  mean per first word (literally commented `#TODO Hanchen This is currently just
  an average`); the paper describes an empirical CDF of past durations per first
  word. Either way it is keyed **only on the first word**.
- **Pin rule:** `FIXED_THRESHOLD_CONTINUUM = 2.0s` — if the first-word average is
  under 2s, pin the KV cache for 2s, else don't pin.

**Consequence:** every command sharing a first word gets the **same** predicted
duration. `python -c "print(1)"` (milliseconds) and `python -m pytest suite/`
(minutes) both key to `python` and receive the identical estimate. The predictor
is blind to the command's arguments and to the code it runs.

Continuum's paper evaluates the **system** (job completion time, throughput,
P95/P99 delay — 1.12–3.66x delay reduction) but never measures the predictor's
own accuracy. This repo isolates and studies that predictor sub-problem.

---

## 3. What we did

### 3.1 Built a real benchmark (the exact Continuum setup)
- **Agent:** mini-SWE-agent (bash-only tools) — the exact agent Continuum used.
- **Workload:** SWE-bench **Verified** (500 tasks) — the exact benchmark.
- **Trajectories:** the official GPT-5 + mini-SWE-agent runs from
  `s3://swe-bench-submissions/bash-only/20250807_mini-v1.7.0_gpt-5/` (same
  model Continuum used; no GPT-5 access needed — the trajectories are public).
- **Commands:** 6,425 real bash tool calls across 499 instances.

### 3.2 Measured ground-truth durations (not predicted — measured)
For each command we replayed it **inside that task's real SWE-bench Docker
image** (the authentic environment with the task's repo + installed packages)
and recorded wall-clock duration. Pull → measure → delete, to bound disk.

### 3.3 Extracted the real code
For every file each command references, we saved the **full file content** from
the Docker image (so the a-priori analyzers see the actual code, not a stub).

### 3.4 Implemented six a-priori predictors (NO machine learning, NO training)
Each analyzes the command's cost *before/without* trusting the wall-clock:

| # | Method | Native output | Notes |
|---|--------|---------------|-------|
| 1 | Bytecode instruction count | count | CPython `compile()`+`dis` op count |
| 2 | Weighted bytecode cost | score | per-opcode cost weights |
| 3 | Complexity / Big-O | score | AST loop-nesting × literal trip counts |
| 4 | **cachegrind cycles** | **cycles → seconds** | run under valgrind cachegrind **inside the task's Docker**, `--trace-children`, full command; output is in cycles, so dividing by CPU frequency gives seconds |
| 5 | Branch mispredictions (TAGE/perceptron role) | count | from cachegrind `--branch-sim` |
| 6 | WCET-style worst-case bound | score | worst-case path over AST |

Key point: Python is interpreted — source → **CPython bytecode** → the
interpreter's machine code. There is no per-script x86, so llvm-mca/uiCA (x86
basic-block analyzers) do not apply to Python snippets; the bytecode is the
right "compiled form". Only **method 4** produces a time estimate
(`seconds = cycles / frequency`); methods 1,2,3,5,6 are cost **counts**.

### 3.5 Converted cycles → time with a real physical constant
`predicted_seconds = cachegrind_cycles / frequency`, frequency = **5.1 GHz**
(measured P-core clock of the i7-13700F). No fitting, no fabricated constant.

### 3.6 Added a WARM-START view
The measured durations are **cold-start** (every command pays CPython/process
boot). We measured the fixed startup cost a-priori (CPython `-c pass` ≈ 80.7M
instructions; `bash -c true` ≈ 1.24M) and subtract it to show the **warm-worker**
case (persistent interpreters), where only the command's work remains.

---

## 4. Results

### 4.1 The data is overwhelmingly short commands (cold start)
- 499 instances, 6,425 commands.
- Duration: p50 = **0.103 s**, p90 = 0.324 s, p99 = 1.59 s, max = 86 s.
- **81.7% of commands are < 0.2 s** — dominated by fixed interpreter/process
  startup, not by the command's work.

This matters: any metric averaged over all commands is dominated by the trivial
~0.1s majority, where every predictor trivially outputs ~0.1s. **We therefore
report the raw predictions, not an aggregate "accuracy", and look specifically
at the commands that do more work.**

### 4.2 The six methods' native outputs
Methods 1,2,3,6 are **identical for shell commands** (they reduce to file-size)
and are `N/A` where there is no Python code. Method 4 is expressed in cycles and
converts to seconds via the CPU frequency; methods 1,2,3,5,6 are cost scores
(counts), which indicate work but are not in time units. Full per-command
values: `all_methods_table.csv`.

### 4.3 Head-to-head predictions (seconds), illustrative rows
`ACT` = actual, `CONT` = Continuum (first-word avg), `M4` = cachegrind/5.1GHz.

```
 ACT(s)  CONT(s)   M4(s)   command
 0.103    0.136    0.000   ls -la                      (trivial: all ~startup)
 2.672    0.413   11.547   python -m pip install -U pip (install-bound)
 4.922    9.168  135.446   pytest -q -x                 (test-runner)
 6.334    0.126    6.028   ls -la && printf ... REPO TREE WALK
```

Observations:
- **Continuum keys on the first word.** For `ls -la && <6.3s tree walk>` it
  predicts **0.126 s** (first word `ls`), while the command takes 6.3 s. It
  aligns well with the trivial ~0.1s majority, where the first word is a good
  guide.
- **M4 tracks command work.** It predicts **6.03 s** for that 6.3 s command —
  the case where Continuum diverges most.
- **M4 diverges on `pytest -x` / `pip install`**: cachegrind's instruction count
  diverges from native wall-clock (valgrind runs a fuller/slower path, early
  exits differ), so it over-predicts (e.g. 135 s for a 5 s run). These are a
  minority but are a clear limitation of the cycle-count approach.

### 4.4 Warm-start view (`warm_cold_table.csv`)
Removing the fixed startup cost:
```
 ACT_cold  ACT_warm  CONT     M4_cold  M4_warm  command
 0.103     0.013     0.136    0.000    0.000    ls -la
 6.334     6.244     0.126    6.028    6.028    ls && tree walk
 2.672     2.582     0.413   11.547   11.531    pip install
```
- Trivial commands collapse to **~0 s of work** when warm — i.e. Continuum's
  ~0.1 s prediction for them is predicting **boot overhead a warm worker would
  not pay**, not the command's work.
- Work-dominated commands are unchanged (startup is negligible there), and that
  is exactly where first-word prediction fails and work-measurement is needed.

### 4.5 Summary
- Continuum's first-word average matches the trivial startup-dominated commands
  and diverges whenever a command does non-trivial work (compute, large
  traversal, install, tests), because it keys only on the first word.
- All six methods are predictors. They differ in unit: method 4 (cachegrind
  cycles) is expressed directly in cycles, so dividing by the CPU frequency
  gives a time estimate in seconds; methods 1, 2, 3, 5, 6 produce cost scores
  (bytecode-op counts, complexity, branch mispredictions, worst-case bounds)
  that indicate the amount of work but are not in time units without a
  per-method scaling.
- Method 4 tracks command work (e.g. the 6.3 s tree-walk Continuum predicts as
  0.13 s) and over-predicts test-runner/install commands, where the instruction
  count measured under valgrind diverges from native wall-clock.
- No single a-priori signal predicts duration accurately across all commands on
  this workload, because duration is governed by heterogeneous causes (cold
  start, compute, I/O, package installs).

---

## 5. Related work

A fast-growing body of recent work studies KV-cache management for agentic LLM
serving. We summarize the most relevant papers, their key insight, a note on the
aspect our work complements, and how this study relates. All of these are strong
contributions; we position our work as a reproducible measurement study of the
*pre-call duration-prediction* sub-problem they each touch.

**Continuum — Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache
Time-to-Live** (arXiv 2511.02230).
- Summary: pins a request's KV cache with a time-to-live during tool calls,
  evicting on expiry.
- Key insight: a modest TTL plus program-level FCFS captures most of the reuse
  benefit while staying robust to imperfect duration estimates.
- Aspect we complement: its duration estimate keys on the command's first word,
  so calls sharing a verb receive the same estimate regardless of arguments.
- Our relation: we benchmark that estimate against six code-aware a-priori
  methods on the same SWE-bench Verified / mini-SWE-agent setup, and quantify
  where first-word estimation matches and diverges from measured durations.

**Ask the Tool, Don't Guess: Agent Tool Calls Hold Their Progress** (arXiv
2609.18849).
- Summary: proposes reading a tool's live progress while it runs, instead of
  estimating its duration beforehand, to guide KV-cache decisions.
- Key insight: a few long calls hold most tool time, and a running tool often
  contains a recoverable progress signal that is very accurate near the end.
- Aspect we complement: the comparison contrasts a signal observed partway
  through a call (50-90% elapsed) with estimates fixed before the call, which
  are different information regimes; and the remaining-time conversion still
  depends on the environment's execution rate, which no tool observes directly.
- Our relation: our work targets the *pre-call* decision (pin or not, how long),
  where no progress signal yet exists; we also decompose duration as
  work-amount x environment-rate and note the environment term is outside the
  reach of both pre-call estimators and in-flight progress readers.

**MORI — Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading**
(arXiv 2606.00866).
- Summary: ranks programs by idleness and tiers their KV cache between GPU HBM
  and CPU DRAM accordingly.
- Key insight: relative idleness across concurrent programs is a strong signal
  for where to place each cache.
- Aspect we complement: placement is driven by observed engine-side idleness
  rather than an estimate of how long the current call will take.
- Our relation: a per-call duration estimate (our focus) is complementary input
  that could inform such tiering decisions.

**TokenCake — A KV-Cache-centric Serving Framework for Multi-Agent Applications**
(arXiv 2510.18586).
- Summary: co-optimizes temporal (offload during calls) and spatial (memory
  partitioning) scheduling with an agent-aware design.
- Key insight: proactive offload during function calls plus predictive upload
  hides transfer latency on the critical path.
- Aspect we complement: it uses a declared/fixed duration estimate per call.
- Our relation: we study how accurately such a per-call estimate can be formed
  a-priori from the command itself.

**Adaptive KV Retention for LLM Agents at Human-Approval Timescales** (arXiv
2608.30830).
- Summary: a retention controller for pauses that last minutes to hours (human
  approval), balancing residency cost against recomputation.
- Key insight: at human timescales the tradeoff differs sharply from
  seconds-scale tool pauses, and a tiered controller handles it well.
- Aspect we complement: it targets a different (much longer) pause regime than
  the sub-second-to-seconds tool calls we measure.
- Our relation: our duration study is about the short/compute tool-call regime;
  the two address different points on the pause-duration spectrum.

**CacheWise — Understanding Workloads and Optimizing KVCache Management for LLM
Coding Agents** (arXiv 2606.16824).
- Summary: characterizes coding-agent workloads and optimizes eviction
  accordingly.
- Key insight: coding-agent KV reuse has structure (within-turn vs across-turn)
  that a workload-aware policy can exploit.
- Aspect we complement: its learned estimate clusters tool names and arguments.
- Our relation: we evaluate code-level (bytecode / cost-analysis) signals as an
  additional a-priori input beyond name/argument clustering.

**PBKV — Efficient Serving for Dynamic Agent Workflows with Prediction-based
KV-Cache Management** (arXiv 2605.06472).
- Summary: predicts future agent invocations by fusing historical workflows with
  the current workflow's context.
- Key insight: workflow-level prediction of *which* steps come next helps
  prefetch/retain the right caches.
- Aspect we complement: it predicts the sequence of invocations rather than the
  duration of an individual tool call.
- Our relation: our focus is orthogonal (per-call duration), and could sit
  beside workflow-level prediction.

**Characterizing Coding Agent Workloads for LLM Serving** (arXiv 2606.30560).
- Summary: a measurement study of coding-agent serving workloads.
- Key insight: explicitly names "semantic-aware tool-latency prediction" as an
  open opportunity.
- Aspect we complement: it characterizes the opportunity rather than building a
  duration predictor.
- Our relation: our study is a direct, reproducible attempt at exactly that
  tool-latency-prediction opportunity, with its successes and limits reported.

### Where this work sits
We agree with the shared observation across these works that a tool call's
absolute wall-clock time depends on the runtime environment (machine load, I/O,
remote-service rates) and is therefore hard to pin exactly before the call. Our
contribution is a careful, reproducible measurement — on the full SWE-bench
Verified / mini-SWE-agent benchmark — of how far code-aware a-priori methods get
on the *pre-call* estimation problem, separating the predictable *work-amount*
component from the environment-dependent *rate* component, and comparing six
such methods head-to-head with Continuum's first-word estimate.

## 6. Repository layout

```
.
├── README.md
├── docs/            design + result documents
│   ├── APRIORI_PLAN.md          authoritative design (hard rules, the six methods)
│   ├── CONTINUUM_EXPLAINED.md   Continuum's method, quoted from paper + code
│   ├── CONTINUUM_REPO.md        notes on the official vllm-continuum repo
│   ├── APRIORI_RESULTS.md       detailed a-priori result log
│   ├── DELAY_RESULT.md          delay-prediction result log
│   ├── PREDICTION_TASK.md       task definition
│   └── DESIGN.md, RESULTS.md, BENCHMARK.md, SWEBENCH_PIPELINE.md, ...
├── src/
│   ├── pipeline/    data collection
│   │   ├── fetch_trajs.py / extract_all_cmds.py   download + parse GPT-5 trajectories
│   │   ├── measure_trajs.py / measure.py          measure wall-clock in Docker
│   │   ├── extract_code.py                        save referenced-file content from Docker
│   │   └── run_cachegrind_docker.py               cachegrind inside task Docker (method 4,5)
│   ├── predictors/  the a-priori methods
│   │   ├── apriori_static.py    methods 1,2,3,6 (static, no run)
│   │   ├── apriori_sim.py       methods 4,5 (cachegrind / branch-sim)
│   │   └── features_ast.py      AST feature extraction
│   └── analysis/    comparison + reporting
│       ├── show_all_methods.py  per-command table of all methods
│       ├── show_warm.py         warm-start vs cold-start view
│       ├── delay_predict.py     cycles -> seconds delay predictor
│       └── final_compare.py, predict_swe.py, ...
├── data/            ground truth + measured outputs (jsonl)
│   ├── swe_verified_durations.jsonl   measured durations (ground truth)
│   ├── cachegrind_docker.jsonl        method 4/5 cycle & mispredict counts
│   ├── apriori_scores.jsonl           method 1,2,3,6 scores
│   └── trajs_commands.jsonl           parsed agent commands
├── results/         final tables + logs
│   ├── all_methods_table.csv          every command, all methods
│   ├── warm_cold_table.csv            warm vs cold per command
│   └── predictions_table.csv, results_log.{csv,md}
└── deprecated/      early synthetic / BFCL / shell / scheduler-sim experiments
                     (kept for transparency; NOT part of the final study)
```

---

## 7. Reproducibility notes
- Hardware: Intel i7-13700F (P-core 5.1 GHz), 32 GB RAM, RTX 4060 Ti 16 GB.
- Durations are machine-specific; the comparisons are what matter.
- Docker images: `swebench/sweb.eval.x86_64.*` (authenticated pulls).
- No ML / no training anywhere — all predictors are static/analytical cost
  estimators; the cycles→seconds step uses the measured CPU frequency only.
