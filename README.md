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
| 4 | **cachegrind cycles** | **cycles → seconds** | run under valgrind cachegrind **inside the task's Docker**, `--trace-children`, full command; the only method whose output is physically time |
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
case (persistent interpreters), where only the command's real work remains.

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
at commands that do real work.**

### 4.2 The six methods' native outputs
Methods 1,2,3,6 are **identical for shell commands** (they reduce to file-size)
and are `N/A` where there is no Python code. Method 4 is the only time estimate.
Method 5 (branch mispredicts) scales with work but is a count, not seconds.
Full per-command values: `all_methods_table.csv`.

### 4.3 Head-to-head predictions (seconds), illustrative rows
`ACT` = actual, `CONT` = Continuum (first-word avg), `M4` = cachegrind/5.1GHz.

```
 ACT(s)  CONT(s)   M4(s)   command
 0.103    0.136    0.000   ls -la                      (trivial: all ~startup)
 2.672    0.413   11.547   python -m pip install -U pip (install-bound)
 4.922    9.168  135.446   pytest -q -x                 (test-runner)
 6.334    0.126    6.028   ls -la && printf ... REPO TREE WALK
```

Observations (numbers, not spin):
- **Continuum is a first-word hardcode.** For `ls -la && <6.3s tree walk>` it
  predicts **0.126 s** (first word `ls` → "fast"), off by ~50×. It only matches
  the trivial ~0.1s majority by coincidence.
- **M4 tracks real work.** It predicts **6.03 s** for that 6.3 s command — the
  case where Continuum fails hardest.
- **M4 breaks on `pytest -x` / `pip install`**: cachegrind's instruction count
  diverges from native wall-clock (valgrind runs a fuller/slower path, early
  exits differ), so it over-predicts (e.g. 135 s for a 5 s run). These are a
  minority but are real failures of the cycle-count approach.

### 4.4 Warm-start view (`warm_cold_table.csv`)
Removing the fixed startup cost:
```
 ACT_cold  ACT_warm  CONT     M4_cold  M4_warm  command
 0.103     0.013     0.136    0.000    0.000    ls -la
 6.334     6.244     0.126    6.028    6.028    ls && tree walk
 2.672     2.582     0.413   11.547   11.531    pip install
```
- Trivial commands collapse to **~0 s of real work** when warm — i.e. Continuum's
  ~0.1 s prediction for them is predicting **boot overhead a warm worker would
  not pay**, not actual work.
- Work-dominated commands are unchanged (startup is negligible there), and that
  is exactly where first-word prediction fails and work-measurement is needed.

### 4.5 Honest summary
- Continuum's first-word average is **correct only for trivial startup-dominated
  commands** and **wrong whenever a command does non-trivial work** (compute, big
  traversal, install, tests) — because it ignores everything but the first word.
- Of the six a-priori methods, **only method 4 (cachegrind cycles / frequency)
  is a physical time predictor.** It captures real work (where Continuum fails)
  but over-predicts test-runner/install commands where instruction count ≠ native
  time. Methods 1,2,3,5,6 are cost indicators, not times.
- **No method is a universally accurate seconds-predictor** on this workload;
  duration is governed by heterogeneous causes (cold start, compute, I/O,
  package installs) that no single a-priori signal fully captures.

---

## 5. Repository layout

Pipeline:
- `fetch_trajs.py`, `extract_all_cmds.py` — download + parse GPT-5 trajectories
- `gen_swebench_traces.py`, `measure_trajs.py` — measure wall-clock in Docker
- `extract_code.py` — save full referenced-file content from Docker
- `apriori_static.py` — predictors 1,2,3,6 (static, no run)
- `apriori_sim.py`, `run_cachegrind_docker.py` — predictors 4,5 (cachegrind in
  the task Docker, `--trace-children`, full command)
- `show_all_methods.py`, `show_warm.py` — produce the result tables

Data / results:
- `swe_verified_durations.jsonl` — measured durations (ground truth)
- `cachegrind_docker.jsonl` — method 4/5 cycle & mispredict counts
- `apriori_scores.jsonl` — method 1,2,3,6 scores
- `all_methods_table.csv`, `warm_cold_table.csv` — final per-command tables

Docs:
- `APRIORI_PLAN.md` — the authoritative design (hard rules, the six methods)
- `CONTINUUM_EXPLAINED.md` — Continuum's method, quoted from paper + code
- `APRIORI_RESULTS.md`, `DELAY_RESULT.md` — detailed result logs

---

## 6. Reproducibility notes
- Hardware: Intel i7-13700F (P-core 5.1 GHz), 32 GB RAM, RTX 4060 Ti 16 GB.
- Durations are machine-specific; the comparisons are what matter.
- Docker images: `swebench/sweb.eval.x86_64.*` (authenticated pulls).
- No ML / no training anywhere — all predictors are static/analytical cost
  estimators; the cycles→seconds step uses the measured CPU frequency only.
