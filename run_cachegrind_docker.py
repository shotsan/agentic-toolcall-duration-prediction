"""
FULL FIX: run cachegrind on the EXACT full command INSIDE the task's own Docker
container (same environment where the wall-clock ground truth was measured).

Per instance:
  pull image -> start container -> apt-get install valgrind (once per container)
  -> for each of that instance's commands: run
       valgrind --tool=cachegrind --branch-sim=yes bash -lc "<command>"
     inside the container at /testbed, parse Ir/misses/mispredicts
  -> kill container, rmi image.

Environment-matched: the command sees the task's real repo + installed packages.
Drops timeout-clamped (>=119s) ground-truth rows. Resumable per instance.

Output: cachegrind_docker.jsonl
  {instance_id, step, command, duration_s, est_cycles, mispredicts, ok}
"""
from __future__ import annotations
import json, re, subprocess, sys
from collections import defaultdict

SUMMARY = {
    "Ir": re.compile(r"I\s+refs:\s+([\d,]+)"),
    "I1_miss": re.compile(r"I1\s+misses:\s+([\d,]+)"),
    "LL_miss": re.compile(r"LL\s+misses:\s+([\d,]+)"),
    "Mispredicts": re.compile(r"Mispredicts:\s+([\d,]+)"),
}
CMD_TIMEOUT = 180  # seconds per command under valgrind (slower than native)


def num(m):
    return float(m.group(1).replace(",", "")) if m else 0.0


def image(iid):
    return f"swebench/sweb.eval.x86_64.{iid.replace('__','_1776_')}:latest".lower()


def load_cmds():
    by = defaultdict(list)
    for l in open("swe_verified_durations.jsonl"):
        r = json.loads(l)
        if r["duration_s"] < 119.0:
            by[r["instance_id"]].append(r)
    return by


def done_instances(path="cachegrind_docker.jsonl"):
    s = set()
    try:
        for l in open(path):
            s.add(json.loads(l)["instance_id"])
    except FileNotFoundError:
        pass
    return s


def dexec(cid, bash_cmd, timeout):
    return subprocess.run(
        ["docker", "exec", "-w", "/testbed", cid, "bash", "-lc", bash_cmd],
        capture_output=True, text=True, errors="replace", timeout=timeout)


def process_instance(iid, cmds, out):
    img = image(iid)
    if subprocess.run(["docker", "image", "inspect", img],
                      capture_output=True).returncode != 0:
        if subprocess.run(["docker", "pull", img], capture_output=True,
                          text=True, timeout=1800).returncode != 0:
            print(f"  {iid}: pull failed", file=sys.stderr)
            return 0
    cid = subprocess.run(["docker", "run", "-d", "-w", "/testbed", "--rm", img,
                          "sleep", "7200"], capture_output=True,
                         text=True).stdout.strip()
    n = 0
    try:
        # install valgrind once (try apt; fall back silently if offline)
        subprocess.run(["docker", "exec", cid, "bash", "-lc",
                        "command -v valgrind || (apt-get update -y && "
                        "apt-get install -y valgrind)"],
                       capture_output=True, text=True, timeout=600)
        has_vg = subprocess.run(["docker", "exec", cid, "bash", "-lc",
                                 "command -v valgrind"],
                                capture_output=True).returncode == 0
        for r in cmds:
            rec = {"instance_id": iid, "step": r["step"],
                   "command": r["command"], "duration_s": r["duration_s"],
                   "est_cycles": None, "mispredicts": None, "ok": False}
            if has_vg:
                try:
                    # --trace-children follows into python/pytest/grep that bash
                    # spawns. Write per-PID cg.out.<pid> files, then SUM them.
                    outdir = "/tmp/cg"
                    inner = (
                        "rm -rf %s && mkdir -p %s && "
                        "valgrind --tool=cachegrind --branch-sim=yes "
                        "--trace-children=yes "
                        "--cachegrind-out-file=%s/cg.out.%%p bash -c %s ; "
                        "python3 - <<'AGG'\n"
                        "import glob,re\n"
                        "tot={'Ir':0,'I1mr':0,'DLmr':0,'ILmr':0,'Bcm':0,'Bim':0}\n"
                        "events=[]\n"
                        "for f in glob.glob('%s/cg.out.*'):\n"
                        "    ev=None\n"
                        "    for line in open(f,errors='ignore'):\n"
                        "        if line.startswith('events:'): ev=line.split()[1:]\n"
                        "        if line.startswith('summary:') and ev:\n"
                        "            vals=list(map(int,line.split()[1:]))\n"
                        "            d=dict(zip(ev,vals))\n"
                        "            for k in tot: tot[k]+=d.get(k,0)\n"
                        "print('AGG_Ir',tot['Ir'])\n"
                        "print('AGG_ILmr',tot.get('ILmr',0))\n"
                        "print('AGG_DLmr',tot.get('DLmr',0))\n"
                        "print('AGG_Bcm',tot.get('Bcm',0)+tot.get('Bim',0))\n"
                        "AGG\n"
                    ) % (outdir, outdir, outdir, json_quote(r["command"]), outdir)
                    p = dexec(cid, inner, CMD_TIMEOUT)
                    def g(tag):
                        m = re.search(tag + r"\s+(\d+)", p.stdout)
                        return float(m.group(1)) if m else 0.0
                    Ir = g("AGG_Ir")
                    if Ir > 0:
                        LLmiss = g("AGG_ILmr") + g("AGG_DLmr")
                        mis = g("AGG_Bcm")
                        rec["est_cycles"] = Ir + 100 * LLmiss + 15 * mis
                        rec["mispredicts"] = mis
                        rec["ok"] = True
                except subprocess.TimeoutExpired:
                    pass
            out.write(json.dumps(rec) + "\n"); out.flush()
            n += 1
    finally:
        subprocess.run(["docker", "kill", cid], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", img], capture_output=True)
    return n


def json_quote(s: str) -> str:
    # safe single-arg quoting for bash -lc "<...>"
    return "'" + s.replace("'", "'\\''") + "'"


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    by = load_cmds()
    insts = list(by.keys())[start:start + count]
    done = done_instances()
    out = open("cachegrind_docker.jsonl", "a")
    for i, iid in enumerate(insts):
        if iid in done:
            print(f"[{i+1}/{len(insts)}] {iid}: skip", file=sys.stderr)
            continue
        n = process_instance(iid, by[iid], out)
        print(f"[{i+1}/{len(insts)}] {iid}: {n} cmds", file=sys.stderr)
    out.close()


if __name__ == "__main__":
    main()
