# A-Priori Tool-Call Duration Prediction — Plan (authoritative reference)

This document defines EXACTLY what we are building. It is the reference to
prevent drift. Read this before touching code.

================================================================================
## 0. HARD RULES (do not violate)
================================================================================
- A-PRIORI ONLY: every predictor estimates duration by ANALYZING the code
  BEFORE/ WITHOUT depending on the real wall-clock. (Simulators like
  cachegrind/callgrind count instructions deterministically — that is analysis,
  allowed. Running the code on real hardware and reading the stopwatch is NOT a
  predictor; it is ground truth.)
- NO MACHINE LEARNING / NO TRAINING. No RandomForest, no MLP, no fitting a model
  per command. These are static/analytical cost estimators only.
- Ground truth already exists: swe_verified_durations.jsonl
  (6435 commands, 499 SWE-bench Verified instances, measured wall-clock
  durations). We DO NOT re-measure durations.
- Python is interpreted by CPython: source -> CPython BYTECODE -> the
  interpreter's (already-compiled) machine code dispatches each opcode. There is
  NO per-script x86 machine code. Therefore for Python the correct "compiled
  form" to analyze is the BYTECODE, not x86 assembly. llvm-mca (x86) only
  applies where a real compiled form exists; for Python it is N/A and we say so.

================================================================================
## 1. GOAL
================================================================================
For each tool call (a bash command a coding agent issued), produce an A-PRIORI
estimate of how long it will take, by analyzing the code it runs. Compare each
estimate against the measured wall-clock ground truth, and against Continuum's
baseline (per-first-word average). This augments Continuum by replacing its
argument-blind first-word predictor with code-aware a-priori estimators.

================================================================================
## 2. THE SIX A-PRIORI PREDICTORS
================================================================================
Each emits a per-command duration estimate (or a proportional cost score).
For Python commands we use the Python-relevant form (bytecode/AST/interpreter
branch stream). For shell commands we use the file-size/line cost form.

| # | Predictor | What it computes (a-priori) | Python form | Shell form |
|---|-----------|------------------------------|-------------|------------|
| 1 | Bytecode instruction count        | number of CPython bytecode ops the code compiles to | compile()+dis: count ops; multiply by est loop trip counts | n/a (bytes-based, see shell note) |
| 2 | Weighted bytecode cost            | sum of per-opcode cost weights (not all ops equal) | weight each opcode by relative cost table | n/a |
| 3 | Complexity / loop-nesting (Big-O) | effective op count = product of loop bounds along nesting depth | AST: loop nesting x literal/range trip counts | n/a |
| 4 | cachegrind instruction+cache count| total EXECUTED instructions + cache misses via valgrind simulated CPU | run snippet under `valgrind --tool=cachegrind` | run shell cmd under cachegrind |
| 5 | Branch-misprediction (TAGE/perceptron role) | mispredicts the snippet's branch stream causes (pipeline-stall penalty) | `valgrind --tool=cachegrind/callgrind --branch-sim=yes` on the CPython run; (optionally feed the branch trace to our ChampSim/CBP-16 TAGE sim) | same via branch-sim |
| 6 | WCET-style worst-case bound       | static worst-case path cost (upper bound) | worst-case path over AST/bytecode (max loop bounds) | n/a |

NOTES on specific tools we already have:
- llvm-mca (/usr/bin/llvm-mca) & uiCA/OSACA: x86 basic-block cycle estimators.
  Apply ONLY where a real compiled artifact exists (rare in our Python/shell
  workload). For Python: N/A (interpreter dispatch, not per-script asm). We
  record N/A honestly rather than feeding them meaningless input.
- TAGE (CBP-16-Simulation TAGE-SC-L) & perceptron (ChampSim binaries): branch
  predictors. They predict MISPREDICT RATE on a branch trace, NOT time. Role =
  predictor #5's branch-penalty signal. Practical path = valgrind branch-sim
  (works directly on the running CPython); raw-trace -> TAGE sim is optional.
- cachegrind/callgrind: deterministic simulated instruction & cache & branch
  counts -> a-priori cost. Allowed (no real-HW timing).

SHELL COMMANDS (cat/grep/sed/wc/nl/find/ls): there is no "code snippet" to
compile. Their a-priori cost is ~linear in bytes/lines of the target file(s).
So for shell commands, predictors 1-3,6 reduce to a bytes/lines cost model over
the referenced files; 4-5 (cachegrind/branch-sim) can still run the actual
utility under valgrind for instruction/branch counts.

================================================================================
## 3. DATA WE NEED (and the Docker re-pull, done RIGHT this time)
================================================================================
Command categories (counts from swe_verified_durations.jsonl):
- 1037 python_embedded_code  : code is INSIDE the command (heredoc/`-c`). HAVE IT.
- 4103 shell_reads_repo_files: cat/grep/sed/wc/nl/find/ls over repo files.
-  294 python_runs_repo_file : pytest X / python file.py -> repo files.
- 1001 other.

=> ~5400 commands reference REPO FILES we deleted with the images. We must:
  FOR EACH instance:
    1. docker pull <image>            (authenticated: shotsan2, ~200/6h)
    2. for every file PATH referenced by that instance's commands:
         save FULL CONTENT locally -> saved_code/<instance_id>/<path>
    3. docker rmi <image>             (bound disk; prune each loop)
  Resumable (skip instances already extracted). Durations are NOT re-measured.
  Embedded code (heredoc/-c) is parsed straight from the command string.

Storage: full file contents, but only the files actually referenced (not whole
repos) -> small. Saved under saved_code/.

================================================================================
## 4. EXPERIMENT / COMPARISON
================================================================================
- For each command: run all applicable predictors -> 6 estimates (+ N/A where a
  predictor doesn't apply, recorded explicitly).
- Baseline: Continuum per-first-word average (and the paper's per-first-word CDF).
- Ground truth: measured wall-clock (swe_verified_durations.jsonl).
- Each a-priori predictor yields a RELATIVE cost score; to compare to seconds we
  calibrate a single global scale factor per predictor (one constant, NOT a
  trained model) — e.g. median(ground_truth)/median(score). This keeps it
  analytical, no per-command learning.
- Metrics (per predictor, held-out by instance where any calibration is used):
  Spearman rho (ranking — most important for scheduling), median APE, tail APE
  on slowest 20%. Report all six + baseline in one table.

================================================================================
## 5. WHAT THIS IS NOT
================================================================================
- Not training a model per command (the earlier RandomForest approach is
  DEPRECATED and must not be used for any claim).
- Not re-measuring wall-clock (already have it).
- Not pretending llvm-mca works on Python (recorded N/A where it doesn't apply).

================================================================================
## 6. STATUS / NEXT STEPS
================================================================================
[ ] Docker extraction: pull -> save full file contents -> delete (resumable).
[ ] Implement predictors 1-3,6 (pure static: bytecode/AST, no run).
[ ] Implement predictors 4-5 (valgrind cachegrind/callgrind + branch-sim).
[ ] Calibrate one global scale per predictor; compute metrics vs ground truth
    and vs Continuum baseline.
[ ] Document results in results_log (append-only, like before).
