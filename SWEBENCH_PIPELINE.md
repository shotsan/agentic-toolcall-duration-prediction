# SWE-Bench Trajectory Generation Pipeline (for the prediction task)

Status log for Option B: self-generate real mini-SWE-agent trajectories on
SWE-Bench, measuring per-command duration inside the task's Docker image.

## What works (infrastructure — DONE)
- SWE-Bench Docker images pull & run (tested: astropy__astropy-12907).
- `TimedDockerEnvironment` (gen_swebench_traces.py) subclasses mini-SWE-agent's
  DockerEnvironment and records per-command wall-clock + first-word + returncode
  to swebench_calls.jsonl. Verified: smoke test showed real duration variance,
  e.g. `python -c import` 211ms vs `python -m pytest` 1166ms (same first word!).
- Full agent driver (run_agent_swebench.py): mini-SWE-agent (isolated mini-env)
  driving the real Docker task, model served by LOCAL vLLM (Qwen2.5-3B) via
  litellm hosted_vllm. End-to-end run completed, 13 commands captured.

## Current blocker (data QUALITY, not infra)
- Qwen2.5-3B is too weak: on astropy it looped issuing
  `cd /path/to/testbed && ...` (hallucinated path) -> all 13 commands failed
  (rc=1), ~100ms each. No useful duration variance.
- Root cause: 3B model chosen only because Qwen-7B bf16 (~15GB) + KV cache does
  not fit the 16GB RTX 4060 Ti (hit CUDA OOM at sampler warmup, then "no KV
  cache memory"). 3B fits but produces degenerate trajectories.

## Options to get good trajectories (pick)
1. Quantized 7B/14B (AWQ/GPToQ ~5-8GB) on the local GPU -> stronger agent, fits
   16GB, keeps everything local & reproducible. PREFERRED.
2. Use a hosted API model (e.g. GPT-4o-mini) just to GENERATE trajectories.
   Durations are still measured locally in Docker, so labels remain valid.
   Deviates from "fully local" but is cheap and gives realistic commands.
3. Use PUBLIC SWE-agent trajectory datasets for the COMMANDS, then RE-EXECUTE
   each command in the task's Docker image to measure duration locally. Gets
   realistic commands without needing a strong local model at all.

## Key reminder
For the PREDICTION task, the model only determines WHICH commands are generated.
Durations are measured in the real Docker environment regardless. So option 3
(replay public commands, measure locally) is fully valid and sidesteps model
quality entirely.

## Artifacts
- gen_swebench_traces.py : TimedDockerEnvironment + smoke test
- run_agent_swebench.py  : agent driver (local vLLM model)
- swebench_calls.jsonl   : captured commands (currently low-quality, to redo)
- smoke_calls.jsonl      : smoke-test commands (good variance, hand-picked)
