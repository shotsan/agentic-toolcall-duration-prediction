"""
Step 1 of APRIORI_PLAN.md: pull each SWE-bench image, SAVE FULL CONTENT of every
file each instance's commands reference, then delete the image. Resumable,
disk-bounded. Durations are NOT re-measured (already have them).

Saves:
  saved_code/<instance_id>/files.json   -> {path: full_text_content}
  saved_code/<instance_id>/embedded.json-> [{step, command, embedded_code}]
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

SAVE = Path("saved_code")
SAVE.mkdir(exist_ok=True)

# ---- parse file paths and embedded code from a bash command -----------------
PATH_RE = re.compile(r"[\w./\-]+\.(?:py|pyx|txt|cfg|toml|ini|rst|md|c|h|cpp|pyi)")
HEREDOC_RE = re.compile(r"<<\s*'?(\w+)'?\n(.*?)\n\1", re.DOTALL)
DASHC_RE = re.compile(r"""python3?\s+-c\s+(['"])(.*?)\1""", re.DOTALL)


def embedded_code(cmd: str):
    codes = []
    for m in HEREDOC_RE.finditer(cmd):
        codes.append(m.group(2))
    for m in DASHC_RE.finditer(cmd):
        codes.append(m.group(2))
    return codes


def referenced_paths(cmd: str):
    # candidate file paths that look like repo files
    paths = set()
    for m in PATH_RE.finditer(cmd):
        p = m.group(0)
        if not p.startswith("-") and "://" not in p:
            paths.add(p)
    return paths


def docker_image(iid):
    return f"swebench/sweb.eval.x86_64.{iid.replace('__','_1776_')}:latest".lower()


def load_cmds():
    by = defaultdict(list)
    for line in open("trajs_commands.jsonl"):
        r = json.loads(line)
        by[r["instance_id"]].append(r)
    return by


def extract_instance(iid, cmds):
    out_dir = SAVE / iid
    if (out_dir / "files.json").exists():
        return "skip"
    # embedded code needs no docker
    embedded = []
    all_paths = set()
    for r in cmds:
        ec = embedded_code(r["command"])
        if ec:
            embedded.append({"step": r["step"], "command": r["command"],
                             "embedded_code": ec})
        all_paths |= referenced_paths(r["command"])

    files = {}
    if all_paths:
        img = docker_image(iid)
        exists = subprocess.run(["docker", "image", "inspect", img],
                                capture_output=True).returncode == 0
        if not exists:
            pr = subprocess.run(["docker", "pull", img], capture_output=True,
                                text=True, timeout=1800)
            if pr.returncode != 0:
                # save embedded even if files unavailable
                out_dir.mkdir(parents=True, exist_ok=True)
                json.dump(files, open(out_dir / "files.json", "w"))
                json.dump(embedded, open(out_dir / "embedded.json", "w"))
                return "pull_failed"
        # start a container, cat each path
        cid = subprocess.run(["docker", "run", "-d", "-w", "/testbed", "--rm",
                              img, "sleep", "600"], capture_output=True,
                             text=True).stdout.strip()
        try:
            for p in all_paths:
                # try relative to /testbed then absolute
                for target in (p, "/testbed/" + p.lstrip("./")):
                    r = subprocess.run(["docker", "exec", cid, "cat", target],
                                       capture_output=True, text=True)
                    if r.returncode == 0 and r.stdout:
                        files[p] = r.stdout
                        break
        finally:
            subprocess.run(["docker", "kill", cid], capture_output=True)
            subprocess.run(["docker", "rmi", "-f", img], capture_output=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(files, open(out_dir / "files.json", "w"))
    json.dump(embedded, open(out_dir / "embedded.json", "w"))
    return f"ok files={len(files)} embedded={len(embedded)}"


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    by = load_cmds()
    insts = list(by.keys())[start:start + count]
    for i, iid in enumerate(insts):
        status = extract_instance(iid, by[iid])
        print(f"[{i+1}/{len(insts)}] {iid}: {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
