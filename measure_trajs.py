"""
Measure real durations for the GPT-5/mini-SWE-agent commands by replaying each
instance's commands inside that instance's SWE-bench Docker image.

Per instance: start container once, run its commands IN ORDER (agents depend on
prior state), record wall-clock per command. This yields authentic per-command
durations on the exact benchmark Continuum used.

Output: swe_verified_durations.jsonl
  {instance_id, step, command, first_word, duration_s, returncode}
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from gen_swebench_traces import TimedDockerEnvironment  # timed docker exec


def docker_image(iid: str) -> str:
    return f"swebench/sweb.eval.x86_64.{iid.replace('__','_1776_')}:latest".lower()


def load_commands(path="trajs_commands.jsonl"):
    by_inst = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        by_inst[r["instance_id"]].append(r)
    return by_inst


def pull_image(img: str, timeout=1200) -> bool:
    import subprocess
    r = subprocess.run(["docker", "pull", img], capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode == 0


def measure_instance(iid: str, cmds: list[dict], out_fh):
    img = docker_image(iid)
    import subprocess
    # ensure image present
    exists = subprocess.run(["docker", "image", "inspect", img],
                            capture_output=True).returncode == 0
    if not exists:
        print(f"  pulling {img} ...", file=sys.stderr)
        if not pull_image(img):
            print(f"  SKIP {iid}: image pull failed", file=sys.stderr)
            return 0
    try:
        env = TimedDockerEnvironment(image=img, cwd="/testbed", timeout=120)
    except Exception as e:
        print(f"  SKIP {iid}: container start failed: {e}", file=sys.stderr)
        return 0
    n = 0
    try:
        for r in cmds:
            t0 = time.perf_counter()
            try:
                res = env.execute({"command": r["command"]})
                rc = res.get("returncode")
            except Exception:
                rc = -1
            dt = time.perf_counter() - t0
            out_fh.write(json.dumps({
                "instance_id": iid, "step": r["step"],
                "command": r["command"], "first_word": r["first_word"],
                "duration_s": dt, "returncode": rc,
            }) + "\n")
            out_fh.flush()
            n += 1
    finally:
        env.cleanup()
        time.sleep(2)  # let --rm container detach before image rmi
        # DELETE the image to keep disk bounded (pull -> measure -> delete).
        subprocess.run(["docker", "rmi", "-f", img], capture_output=True)
        # also prune the trajectory json we no longer need
    return n


def already_done(path="swe_verified_durations.jsonl"):
    done = set()
    try:
        for line in open(path):
            done.add(json.loads(line)["instance_id"])
    except FileNotFoundError:
        pass
    return done


def main():
    # args: [start_index] [count]  -> process by_inst[start:start+count]
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    by_inst = load_commands()
    insts = list(by_inst.keys())[start:start + count]
    done = already_done()
    out = open("swe_verified_durations.jsonl", "a")  # APPEND, resumable
    total = 0
    for i, iid in enumerate(insts):
        if iid in done:
            print(f"=== skip {iid} (done) ===", file=sys.stderr)
            continue
        print(f"=== [{i+1}/{len(insts)}] {iid} ({len(by_inst[iid])} cmds) ===",
              file=sys.stderr)
        total += measure_instance(iid, by_inst[iid], out)
    out.close()
    print(f"\nmeasured {total} commands across {len(insts)} instances", file=sys.stderr)


if __name__ == "__main__":
    main()
