# Continuum's Official Repo — What's There and What We Reuse

Repo: https://github.com/Hanchenli/vllm-continuum  (cited in the paper as
"Preview code"). Cloned to ../vllm-continuum. It is a FULL vLLM fork + their
experiment harness. This document records exactly what exists and how our
augmentation plugs in, so we build on their code, not a reimplementation.

## 1. The predictor we augment (the key file)
`vllm/v1/core/estimate_with_func.py`:
- `ToolCallParser.parse(text)` — regex-extracts the bash block
  `r"```bash\s*\n(.*?)\n```"` and returns `words[0]` (the FIRST WORD). This is
  Continuum's tool key `f`. Confirmed: argument-blind.
- `ToolCallEstimator`:
  - `update_func_call_exec_time()` — stores per-first-word durations and returns
    their running AVERAGE. Carries the comment:
    `#TODO Hanchen This is currently just an average`.
  - `set_up_pin(request)` — if predicted time for the first-word <
    `FIXED_THRESHOLD_CONTINUUM = 2.0`s, pin for 2.0s; else 0 (don't pin).
- So the SHIPPED predictor = per-first-word average + fixed 2s threshold. The
  README states the fork is "without the estimation in the paper" — i.e. the
  public baseline is even cruder than the paper's CDF.

## 2. The EXACT integration point (one line)
`vllm/v1/core/sched/scheduler.py`, in `_free_blocks()`:
```python
if self.policy == SchedulingPolicy.CONTINUUM and not request.is_last_step:
    length_of_pin = self.tool_call_estimator.set_up_pin(request)   # <-- predictor
    if length_of_pin > 0.01:
        self.pin_request(request, length_of_pin)
```
=> Our augmentation = a drop-in `set_up_pin(request)` that conditions on the
FULL command (available via the tokenizer-decoded output / request.this_func_call
chain) instead of just the first word. Same signature, same call site.
Estimator is constructed at scheduler.py:131 (`ToolCallEstimator(...)`).

## 3. Their measurement harness (reuse verbatim)
- `Continuum_Recorder` (in estimate_with_func.py) logs per-request timestamps:
  arrival, departure, pinned, unpinned, evicted, waiting_to_running
  (+ prompt_length, hit_length). Dumps `scheduler_timestamps`.
- `continuum_exp/analyze.py` computes job duration = last departure − first
  arrival, plus avg/median/P95/P99. Use this so our metrics == theirs.

## 4. Their REAL results (reproduced from continuum_exp/*/results.json)
Llama-3.1-70B, SWE-bench, 500 jobs:
| Metric | vanilla vLLM | Continuum | speedup |
|---|---|---|---|
| average | 465.2 s | 215.5 s | 2.16x |
| median  | 420.4 s | 180.7 s | 2.33x |
| P95     | 928.5 s | 492.3 s | 1.89x |
| P99     | 952.9 s | 926.5 s | 1.03x |
These match the paper's "up to ~2x" claim.

## 5. How to run their system (from their README)
- Install fork editable; `vllm serve <model> --scheduling-policy continuum`.
- Generate workload with their bundled `mini-swe-agent/`:
  `mini-extra swebench --model-class vllm --model <m> --port <p>
   --subset verified --split test --use-jps --jps 1.0 --output ./swebench_output`
- `python continuum_exp/analyze.py --output-dir <dir>`.
- Default eval: Llama-70B on 4×H100. README warns smaller models may fail the
  agent task (same issue we hit with Qwen-3B).

## 6. What we reuse vs build
REUSE (their code):
- `ToolCallEstimator` as the BASELINE predictor (don't reimplement).
- `ToolCallParser` (first-word) as the baseline tool key.
- `Continuum_Recorder` + `analyze.py` for identical metrics.
- `--scheduling-policy continuum` scheduler + the `set_up_pin` hook.
- their bundled mini-swe-agent for trajectory generation.
BUILD (our augmentation):
- `CallConditionedEstimator.set_up_pin(request)` — same interface, predicts from
  the FULL command (verb + args + structure), not just first word.
- Comparison harness: baseline `set_up_pin` vs ours, on the SAME trajectories.

## 7. Hardware reality (honest)
- Their eval: Llama-70B / 4×H100. Ours: 1× RTX 4060 Ti (16 GB).
- We cannot serve 70B. For the PREDICTION study we do not need to: we need
  trajectories (commands) + per-command durations. Commands can come from a
  capable agent (their mini-swe-agent with a stronger-but-fitting model, a cheap
  hosted model, or public strong-agent traces); durations are measured locally
  in the task Docker image via our TimedDockerEnvironment.
- End-to-end serving numbers at their scale need their hardware; that is the
  optional GPU "nice-to-have", not required for the prediction contribution.
