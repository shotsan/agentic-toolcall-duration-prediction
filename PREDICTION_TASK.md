# Tool-Call Duration Prediction on mini-SWE-agent / SWE-Bench

Clean, self-contained definition of the PREDICTION task. This document is scoped
to prediction ONLY. LLM serving and scheduler integration are explicitly out of
scope here (later work).

---

## 1. The task (one sentence)

Given a tool call that a coding agent (mini-SWE-agent) emits while solving a
SWE-Bench task, predict how long that tool call will take to execute (wall-clock
duration), BEFORE running it.

## 2. Why this task, and why THIS data

- Continuum (arXiv 2511.02230) decides KV-cache TTL from a predicted tool-call
  duration. It predicts duration with a per-tool empirical CDF where, for
  SWE-agents, the "tool" = the FIRST WORD of the bash command (paper: "extract
  the string within the bash block and use the first word afterwards as the tool
  call name").
- mini-SWE-agent is the exact agent Continuum used for its SWE-Bench workload.
  Its ONLY tool is `bash`; every action runs via `subprocess.run`. So its tool
  calls ARE bash commands. This is the correct, faithful data source.
- We therefore evaluate duration prediction on REAL mini-SWE-agent bash commands
  from SWE-Bench trajectories — NOT on hand-picked functions. (A previous
  5-function BFCL experiment is deprecated and must not be used for any claim.)

## 3. Inputs and outputs (precise)

For each tool call (one bash command executed in the task's repo environment):

INPUT (features available to a scheduler at dispatch time):
- the full bash command string (e.g. `python -m pytest tests/test_x.py -q`)
- its first word / verb (e.g. `python`) — this is all Continuum uses
- derived command features (to be defined in FEATURES below)
- optionally: repo/environment context (repo size, file targeted)

OUTPUT (label):
- measured wall-clock execution time of that command in its environment (seconds)

## 4. Baseline vs our method (the comparison under test)

- BASELINE (Continuum): predict duration = median (or empirical-CDF statistic)
  of PAST durations for the same FIRST WORD. Argument-blind.
- OURS: predict duration conditioned on the FULL command (verb + arguments +
  command structure). Hypothesis: the command string carries signal the
  first-word bucket discards (e.g. `pytest wholesuite` vs `pytest -q one_test`).

## 5. Features (our method) — to be finalized against real data

Candidate, command-string-derived (no execution needed):
- verb / first word (categorical)
- subcommand / flags (e.g. pytest vs pytest -q; grep -r vs grep)
- target path(s) and their size/characteristics when resolvable
- command length / token count / number of args / pipes / redirects
- for `python ...`: whether it runs a file vs -c; target file size / LoC
NOTE on "compiled form": Python is compiled to BYTECODE at run; where a command
invokes Python code, bytecode-level features (instruction counts) are a valid
analysis target (analogous to llvm-mca on machine code). To be integrated where
applicable; NOT hand-waved.

## 6. Metrics (report all; same family as our prior benchmark)

- Median APE and MAPE (central-tendency error).
- Tail MAPE on slowest-20% (Continuum Fig 5: slow 10% of calls cause up to 94%
  of total delay — tail is what matters).
- Spearman rank correlation (scheduling needs correct ORDERING).
- Evaluation protocol: train on a subset of tasks, test on HELD-OUT tasks
  (so we measure generalization to unseen SWE-Bench instances, not interpolation
  within one task).

## 7. Honest scope and the GPT-5 deviation

- Continuum generated trajectories with GPT-5. We do NOT need GPT-5 to study
  PREDICTION: we need real bash-command trajectories + their durations. Options:
  (a) use PUBLIC mini-SWE-agent / SWE-agent trajectory datasets (bash commands);
  (b) run mini-SWE-agent ourselves on SWE-Bench Docker images with a local model
      (vLLM) to generate our own trajectories.
- The LLM/model choice affects WHICH commands get generated, not the duration of
  a given command. For the prediction task, the command + its environment
  determine the label. We will document whichever trajectory source we use.

## 8. What we do NOT claim (guardrails)

- We do NOT claim to "beat Continuum" end-to-end from a prediction result alone.
- We do NOT use the deprecated 5-BFCL experiment for any claim.
- Prediction accuracy is reported on held-out SWE-Bench tasks; absolute ms are
  hardware-specific (ranking and relative error are the portable results).

## 9. Status / next concrete step

- Obtain real mini-SWE-agent SWE-Bench trajectories with per-command durations
  (public dataset or self-generated via Docker + local model).
- Build the command-feature extractor.
- Run baseline (first-word CDF) vs ours (full-command) on held-out tasks.
- Report Sec 6 metrics.
