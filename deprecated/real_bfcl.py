"""
REAL-DATA test (DESIGN.md next-step #1).

Uses actual BFCL executable tool functions (MathAPI) called with VARYING runtime
arguments -- the realistic "inputs are NOT literals in the code" case that the
synthetic Experiment #2 did not test.

Two prediction regimes, mirroring information an online scheduler would have:
  (A) FUNCTION-ONLY features (what Continuum sees): identity of the tool `f`
      only -> degenerates to the flat per-tool CDF.
  (B) CALL-SITE features (our thesis): tool identity + ARGUMENT features
      (e.g. len(list), magnitude) extracted from the actual call the agent
      emitted. This is still "code/args conditioning" -- the arg size is in the
      agent's generated tool call, even if not in the function body.

This asks the sharp question: when duration depends on a runtime argument the
agent chooses, can conditioning on that argument beat the flat CDF?
"""
from __future__ import annotations
import sys
import json
import time
import random
import statistics

sys.path.insert(0, "/home/santosh/cpu-gpu-sync/gorilla/"
                   "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/"
                   "multi_turn_eval/func_source_code")

from math_api import MathAPI  # real BFCL tool implementation

random.seed(7)
api = MathAPI()

# Tool calls a coding/data agent might emit: (tool_name, build_args) where
# build_args() returns a concrete argument instance with a chosen size.
# Duration depends on the argument SIZE, which the model must infer from the
# call, not the function body.

TOOLS = {
    "mean":               lambda n: ([random.random() for _ in range(n)],),
    "standard_deviation": lambda n: ([random.random() for _ in range(n)],),
    "sum_values":         lambda n: ([random.random() for _ in range(n)],),
    "min_value":          lambda n: ([random.random() for _ in range(n)],),
    "max_value":          lambda n: ([random.random() for _ in range(n)],),
}
FUNCS = {
    "mean": api.mean,
    "standard_deviation": api.standard_deviation,
    "sum_values": api.sum_values,
    "min_value": api.min_value,
    "max_value": api.max_value,
}
TOOL_IDS = {name: i for i, name in enumerate(TOOLS)}

SIZES = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 3_000_000,
         6_000_000]


def time_call(fn, args, runs=5):
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def build_dataset():
    rows = []
    for _ in range(300):
        name = random.choice(list(TOOLS))
        n = random.choice(SIZES)
        args = TOOLS[name](n)         # the concrete list the agent would pass
        dur_ms = time_call(FUNCS[name], args) * 1000.0
        # CALL-SITE features: tool id + argument-size features (what the agent's
        # emitted call reveals). NOT the function body literals.
        arg = args[0]
        feats_callsite = [
            float(TOOL_IDS[name]),      # which tool
            float(len(arg)),            # argument length (the hidden input!)
            float(len(arg)),            # (duplicate kept for parity; harmless)
        ]
        # FUNCTION-ONLY features: tool id alone (Continuum's info).
        feats_funconly = [float(TOOL_IDS[name])]
        rows.append({
            "tool": name,
            "n": n,
            "dur_ms": dur_ms,
            "feat_callsite": feats_callsite,
            "feat_funconly": feats_funconly,
        })
    return rows


if __name__ == "__main__":
    rows = build_dataset()
    with open("real_labeled.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ds = [r["dur_ms"] for r in rows]
    print(f"wrote {len(rows)} real BFCL tool calls to real_labeled.jsonl")
    print(f"duration range {min(ds):.2f}-{max(ds):.1f} ms, "
          f"mean {statistics.mean(ds):.1f}, median {statistics.median(ds):.1f}")
