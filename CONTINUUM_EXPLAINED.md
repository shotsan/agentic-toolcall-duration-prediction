# Continuum, Explained from Paper + Code (ground-truth reference)

Source-grounded notes on what Continuum (arXiv 2511.02230) actually is, how its
predictor works, what GPT-5 and trajectories were used for, and what results it
showed. Everything here is quoted/traced from the paper text (continuum.txt) and
the official repo (../vllm-continuum), NOT from memory.

---

## 1. What a "trajectory" is

A trajectory = the full conversation log of ONE agent solving ONE task. It is an
ordered list of messages (verified from a real file,
gpt5_trajs_all/astropy__astropy-12907/...traj.json, 26 messages):

- role=system   : agent instructions ("use bash to fix the bug").
- role=user     : the TASK (the GitHub issue to fix).
- role=assistant: the model's reply = a THOUGHT + ONE bash command in a
                  ```bash ... ``` block  (this is the TOOL CALL).
- role=user     : the OUTPUT of running that command (<returncode>, <output>).
- ... assistant/user alternate until the agent submits a fix.

So: assistant messages carry the bash commands (tool calls); user messages carry
their outputs. A trajectory is the play-by-play of one bug fix.

## 2. What GPT-5 was used for (3 mentions in the paper)

GPT-5 was the AGENT'S BRAIN that GENERATED the trajectories — i.e. it decided
which bash commands the agent issues. It is NOT the model their vLLM serves.

- (line 411) "We collect and analyze 100 mini-swe-agent traces running
  SWE-Bench. We also collect 100 traces from BFCL V4 Web [Search]."
- (line 616) "both workloads use GPT-5 as the base model."
- (line 1415) "we evaluate on two collected workloads running GPT5 ... poisson
  distribution for the arrival pattern."
- (footnote 2, line 1445) "We use GPT-5 for the better model capabilities to
  ensure that the workflow generated are mostly correct. Base small models often
  fail to accomplish the task."  <-- exactly the failure we saw with Qwen-3B.

Two distinct model roles:
- Trajectory/workload generation -> GPT-5 (defines WHICH commands arrive).
- Served model under test        -> Llama-3.1-8B/70B, Gemma-3-12B, GLM-4.5
  (what the scheduler manages; what delay/throughput numbers measure).

## 3. How the released experiment runs (code-verified, NOT a replay)

In ../vllm-continuum/mini-swe-agent/src/minisweagent/run/extra/swebench.py:
- process_instance() calls agent.run(task) -> a LIVE agent loop: start the real
  SWE-bench Docker container, query the served (vLLM) model for the next bash
  command, exec it in Docker, feed output back, repeat.
- Load generation (JPS mode, --use-jps --jps X):
    inter_arrival_time = random.expovariate(jps)   # Poisson arrivals
    each instance launched in its own thread at its scheduled arrival time.
  => Many agent jobs run concurrently against one vLLM server; the KV-cache
     contention is what Continuum's scheduler optimizes.
(Correction: it is a live run with Poisson-arrival concurrency, not a replay of
pre-recorded commands.)

## 4. How Continuum "predicts" tool-call duration (paper + code)

Tool KEY = first word of the bash command.
- Paper (line 1111): "We extract the string within the bash block and use the
  first word afterwards as the tool call name."
- Code (estimate_with_func.py, ToolCallParser.parse):
    actions = re.findall(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)
    words = actions[0].strip().split(); return words[0]   # first word = key `f`

Prediction P(tau,f):
- Paper (line 1012): "estimate P(tau,f) using the empirical CDF derived from
  historical tool-call records S[f]."  (CDF of past durations for first-word f.)
- Code (shipped, simpler): per-first-word running AVERAGE
    #TODO Hanchen This is currently just an average
    func_call_to_exec_time[func] = sum(records[func]) / len(records[func])

Pin decision (set_up_pin + scheduler, code-traced):
    FIXED_THRESHOLD_CONTINUUM = 2.0  # seconds
    t = avg_duration_for_first_word(cmd) or 0.0
    if t > 2.0: return 0            # predicted slow -> DON'T pin
    else:       return 2.0          # -> PIN the KV cache for 2 seconds
  scheduler._free_blocks: if length_of_pin>0.01: pin_request(req, 2.0) else free.
  unpin_requests_regular: once now >= pin_end_time and agent hasn't returned ->
  free the KV cache.

CONSEQUENCE (the crudeness): every command sharing a first word gets the SAME
prediction. `python -c "print(1)"` (ms) and `python -m pytest suite/` (minutes)
both key to "python" and get the same number; pin length is a hardcoded 2s, not
derived from the command. This argument-blindness is what our code-analysis
predictor targets.

## 5. What results Continuum SHOWED (with the trajectories)

(A) MOTIVATION / characterization (from the 100+100 GPT-5 traces):
- Many turns/program (SWE-bench ~10.9, BFCL ~6.3; Table 2).
- Tool time (mean, std): SWE-bench (925 ms, 3550 ms); BFCL (1923 ms, 2133 ms)
  -> variance >> mean.
- Figure 5 (key): tool execution time extremely LONG-TAILED within a tool:
  slowest 10% of fetch_url = 52.5% of total delay; slowest 10% of cd = 94.1%.
- Figure 3: later turns have shorter remaining time -> justifies program-level
  FCFS as an SRTF approximation.

(B) END-TO-END scheduling (traces as workload, Poisson arrivals):
- Metrics: Average Job Delay and P95 Job Delay vs Jobs-Per-Second throughput.
- Baselines: vanilla vLLM, CPU-offload (vLLM+LMCache), Autellix(PLAS),
  InferCept, SGLang, Nvidia Dynamo.
- Headline: delay reduction 1.12x-3.66x, throughput 1.10x-3.22x; up to 8.18x on
  a private "Company A" testbed.
- From their repo results.json (Llama-70B, SWE-bench, 500 jobs):
  avg 465.2s -> 215.5s (2.16x), median 420.4 -> 180.7 (2.33x),
  P95 928.5 -> 492.3, P99 952.9 -> 926.5.
- Table 5 (OpenHands, steps/min): vLLM 93.4, ThunderAgent 114.8, Continuum 144.9.

## 6. Why this motivates OUR work

Figure 5 proves a few slow tool calls dominate delay. Continuum only predicts by
first-word average/CDF -> blind to the arguments/code that determine whether a
call is in that slow tail. A predictor that reads the actual code (bytecode / IR
/ static cost analysis) targets exactly the tail Continuum shows matters, and
could set a tailored pin length instead of a flat 2s. Their own motivation
figure is the argument for a code-aware duration predictor.
