"""
Drive mini-SWE-agent on SWE-Bench tasks against the LOCAL vLLM server, using a
TimedDockerEnvironment that records per-command wall-clock duration inside the
task's real Docker image.

Produces swebench_calls.jsonl: one row per executed bash command with
{instance_id, step, command, first_word, duration_s, returncode, out_bytes}.

Faithful to Continuum's SWE-Bench workload: mini-SWE-agent, bash-only tools,
first-word tool keying. Model = local Qwen (documented deviation from GPT-5;
affects WHICH commands are generated, not the duration of a given command).
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("MSWEA_CONFIGURED", "true")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")  # local model, no prices
# Point litellm at the local vLLM OpenAI-compatible server.
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")
os.environ.setdefault("HOSTED_VLLM_API_BASE", "http://localhost:8000/v1")
os.environ.setdefault("HOSTED_VLLM_API_KEY", "EMPTY")

import yaml
from datasets import load_dataset
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel

# our timed env
sys.path.insert(0, os.path.dirname(__file__))
from gen_swebench_traces import TimedDockerEnvironment, _INSTANCE

MINI = Path("/home/santosh/cpu-gpu-sync/mini-swe-agent")
CFG = MINI / "src/minisweagent/config/benchmarks/swebench.yaml"

MODEL = "hosted_vllm/Qwen/Qwen2.5-3B-Instruct"  # litellm routes to OPENAI_API_BASE


def docker_image(instance_id: str) -> str:
    iid = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{iid}:latest".lower()


def run_instance(instance: dict, cfg: dict, step_limit: int = 15):
    iid = instance["instance_id"]
    _INSTANCE["id"] = iid
    _INSTANCE["step"] = 0
    env = TimedDockerEnvironment(image=docker_image(iid), cwd="/testbed", timeout=120)
    model = LitellmModel(model_name=MODEL, model_kwargs={"temperature": 0.0,
                                                         "api_base": os.environ["OPENAI_API_BASE"],
                                                         "api_key": "EMPTY"})
    agent_cfg = {**cfg.get("agent", {}), "step_limit": step_limit, "cost_limit": 0.0}
    agent = DefaultAgent(model, env, **agent_cfg)
    try:
        agent.run(task=instance["problem_statement"])
    except Exception as e:
        print(f"  [{iid}] agent ended: {type(e).__name__}: {str(e)[:100]}")
    finally:
        env.cleanup()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = yaml.safe_load(CFG.read_text())
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    for i in range(min(n, len(ds))):
        inst = ds[i]
        print(f"=== [{i+1}/{n}] {inst['instance_id']} ({inst['repo']}) ===")
        run_instance(inst, cfg)


if __name__ == "__main__":
    main()
