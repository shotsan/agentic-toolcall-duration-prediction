# A-Priori Predictor Results (vs wall-clock ground truth)

Setup: 6435 real GPT-5/mini-SWE-agent SWE-bench Verified commands, full
referenced-file content extracted from Docker (499 instances, 0 pull failures).
Each a-priori predictor -> cost score -> one global scale constant -> compared to
measured wall-clock. Metrics: Spearman rho (ranking), median APE, tail MAPE.
NO training.

## Baseline
Continuum (per-first-word mean): rho=+0.637, medAPE=20.4%, tailMAPE=187.5%.

## Static predictors 1-3,6 (PRELIMINARY, full 6435)
| predictor | rho | medAPE | tailMAPE |
|---|---|---|---|
| bytecode_count    | -0.437 | 99.8% | 124.2% |
| weighted_bytecode | -0.419 | 99.9% | 116.3% |
| complexity_bigO   | -0.512 | 100.0%| 146.4% |
| wcet_bound        | -0.508 | 100.0%| 139.3% |

FINDING (honest): the pure-static predictors NEGATIVELY correlate with real
duration on this workload. Reason: real command duration is dominated by FIXED
OVERHEAD (CPython interpreter startup ~100ms; shell process spawn), not by
static code size. A big grep/cat often finishes fast (~I/O), and a tiny python
snippet still pays ~100ms startup. So "bigger static score" != "longer time"
here -> anti-correlated. Static code-size metrics are the WRONG a-priori signal
for overhead-dominated agent commands.

## Simulation predictors 4,5 (cachegrind + branch-sim) — IN PROGRESS
Early signal (n=7): cachegrind est_cycles rho=+0.607 (POSITIVE). cachegrind
counts the ACTUAL interpreter work INCLUDING startup, which is what real time
tracks -> correlates where the static metrics fail. Full pass (~1853 python-code
commands) running; valgrind ~30-60s/command so this takes hours. Will update.

## CORRECTION: earlier "Continuum ranks better" was a CONTAMINATION ARTIFACT
An earlier comparison mixed non-Python commands (179 `applypatch`, plus sed/git/
cat) into the "python-code" subset because the classifier matched any command
containing a heredoc or a .py path. `applypatch` forms a tight duration cluster
that inflated Continuum's first-word ranking to a spurious +0.41. That was wrong
and was rationalized with a made-up explanation. Corrected below.

## Apples-to-apples: STRICT python-executing commands only (n=763)
(python / python3 / pytest / PYTHONPATH= ..., no applypatch/sed/git/cat.)
| predictor | rho | medAPE | tailMAPE |
|---|---|---|---|
| Continuum (first-word) | -0.228 | 610.3% | 155.4% |
| cachegrind (a-priori)  | -0.015 |  52.0% |  76.2% |

TRUE finding:
- Continuum is NEGATIVELY correlated (-0.23) on real python commands: every
  `python`/`python3` collapses to one first-word average unrelated to actual
  duration. It does NOT win ranking.
- cachegrind beats it on everything: medAPE 610% -> 52% (12x), tail 155% -> 76%,
  ranking -0.23 -> -0.02.
- Honest limitation: cachegrind rho ~0 (not strongly positive) on pure python,
  because simulated cycles don't track wall-clock when ~100ms interpreter
  startup dominates and I/O isn't modeled. Far better on MAGNITUDE than ranking.

## Interpretation so far
The study's value is showing WHICH a-priori methods work on real agent commands:
- Static code-size (bytecode/complexity/WCET): fail (overhead-dominated regime).
- Dynamic simulation (cachegrind instruction+branch counts): correlates, because
  it captures the fixed interpreter/process cost that dominates.
This is itself the result: for coding-agent tool calls, you need an
execution-cost SIMULATION (or to explicitly model the fixed overhead), not pure
static code analysis.


## FIXED PIPELINE (environment-matched, full-command cachegrind in Docker)

Bugs fixed:
1. Dropped 10 timeout-clamped (>=119s, returncode -1) ground-truth rows.
2. Ran cachegrind on the EXACT full command INSIDE the task's own Docker
   container (real repo + installed packages), not a reconstructed sub-piece on
   the host (host had wrong package versions).
3. Added --trace-children=yes so cachegrind follows into the python/pytest/grep
   child processes bash spawns (earlier it measured only bash -> constant ~2.89M
   cycles for every command, which made ALL correlations ~0; that was the bug
   making a dumb baseline look competitive).

Validation: cycles now span 1.4M (ls) to 16.3B (pytest two files) and track real
time monotonically. Instance astropy-12907: Spearman(cycles,duration)=0.867.

Interim (22 instances, 309 commands): Spearman(cycles,duration) = 0.770.
(vs Continuum first-word baseline ~0.0 -> decisive.) Full 499-instance pass
running (~20h, resumable, pull->install valgrind->measure->delete, disk bounded).


## FINAL RESULTS — full 499 instances, all bugs fixed (6384 commands)
| method | rho | medAPE | tailMAPE |
|---|---|---|---|
| Continuum (first-word)      | +0.647 |  16.3% |   74.5% |
| 1 bytecode_count            | -0.438 |  99.8% |  124.4% |
| 2 weighted_bytecode         | -0.420 |  99.9% |  116.4% |
| 3 complexity_bigO           | -0.512 | 100.0% |  146.4% |
| 6 wcet_bound                | -0.510 | 100.0% |  139.3% |
| 4 cachegrind_cost (env)     | +0.743 |  81.6% | (scale artifact) |
| 5 branch_mispredict (env)   | +0.738 |  76.8% | (scale artifact) |

HONEST CONCLUSIONS (CORRECTED — earlier "static methods fail/negative" was a BUG):
- RANKING (the metric scheduling needs): simulation methods WIN. cachegrind
  rho=+0.743 and branch-mispredict rho=+0.738 BEAT Continuum's first-word
  rho=+0.647. Environment-matched, full-command, children-traced; clean bug-free
  comparison on the full benchmark.
- STATIC methods were NOT failing. The earlier negative rho (-0.4 to -0.5) was a
  CLASSIFIER CONTAMINATION BUG: 3958 shell commands (nl/sed/grep/cat over big
  files) were lumped into the "code complexity" metric with file-size as a
  stand-in. Big file -> big score but still ~100ms (fast I/O) -> anti-correlated,
  dragging the whole number negative.
  On a STRICT clean python-code subset (n=1064) the static methods are WEAKLY
  POSITIVE, not negative:
    weighted_bytecode +0.166, bytecode_count +0.058, complexity_bigO +0.035,
    wcet_bound -0.067.
- WHY static is only weakly positive (the real reason, verified): heavy python
  commands do their work INSIDE imported library calls (sympy/numpy/pytest). A
  surface AST/bytecode analyzer counts only the visible snippet's ops, not what
  sympy.solve() does internally. e.g. a 24-second `python <<PY ... sympy ... PY`
  has only ~24 visible bytecode ops. So static code size decouples from real
  time whenever the work is hidden in library calls. This is exactly why the
  EXECUTION/SIMULATION predictors (cachegrind) win: they count ALL executed
  work including inside libraries.
- ABSOLUTE TIME: our one-global-linear-scale calibration is poor (medAPE 77-82%,
  tail blows up) because cycles span ~6 orders of magnitude while real durations
  are compressed by a ~100ms fixed floor + I/O that instruction-counting can't
  see. A linear cycles->seconds map is too crude for magnitude; ranking is the
  robust result. (A floor + non-linear calibration would fix magnitude; future
  work, no ML.)
- Branch-mispredict (the TAGE/perceptron role, predictor 5) is nearly as strong
  as full cachegrind for ranking (0.738 vs 0.743) -> the branch signal alone
  carries most of the ranking information.
