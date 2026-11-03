"""
Download real GPT-5 + mini-SWE-agent trajectories (SWE-bench Verified) from the
official swe-bench submissions S3 bucket (anonymous, no creds), and extract the
bash commands per instance.

This is the EXACT setup Continuum used: mini-SWE-agent on SWE-bench, GPT-5.
Output: trajs_commands.jsonl  {instance_id, step, command, first_word}
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

S3 = "s3://swe-bench-submissions/bash-only/20250807_mini-v1.7.0_gpt-5/trajs"
OUT_DIR = Path("gpt5_trajs")
BASH_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)


def list_instances(limit=None):
    r = subprocess.run(["aws", "s3", "ls", f"{S3}/", "--no-sign-request"],
                       capture_output=True, text=True, check=True)
    ids = [ln.split("PRE ")[1].strip().rstrip("/")
           for ln in r.stdout.splitlines() if "PRE " in ln]
    return ids[:limit] if limit else ids


def fetch_traj(iid: str) -> Path | None:
    dst = OUT_DIR / iid
    f = dst / f"{iid}.traj.json"
    if f.exists():
        return f
    subprocess.run(["aws", "s3", "cp", f"{S3}/{iid}/", str(dst) + "/",
                    "--recursive", "--no-sign-request"],
                   capture_output=True, text=True)
    return f if f.exists() else None


def extract_commands(traj_path: Path) -> list[str]:
    d = json.loads(traj_path.read_text())
    msgs = d.get("messages", d)
    cmds = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for a in BASH_RE.findall(str(m.get("content", ""))):
                cmds.append(a.strip())
    return cmds


def first_word(cmd: str) -> str:
    s = cmd.strip()
    return s.split()[0] if s else ""


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ids = list_instances(limit=n)
    print(f"Found {len(ids)} instances (showing first {n})", file=sys.stderr)
    out = open("trajs_commands.jsonl", "w")
    total = 0
    for iid in ids:
        f = fetch_traj(iid)
        if not f:
            continue
        cmds = extract_commands(f)
        for step, c in enumerate(cmds):
            out.write(json.dumps({"instance_id": iid, "step": step,
                                  "command": c, "first_word": first_word(c)}) + "\n")
            total += 1
        print(f"  {iid}: {len(cmds)} cmds", file=sys.stderr)
    out.close()
    print(f"wrote {total} commands from {len(ids)} instances to trajs_commands.jsonl",
          file=sys.stderr)


if __name__ == "__main__":
    main()
