"""
Run the 6 a-priori predictors on each SWE command and compare to the measured
wall-clock ground truth (swe_verified_durations.jsonl). No ML, no training.

Code source per command:
- embedded (python -c / heredoc): analyze the embedded code directly.
- runs a repo file (python X.py / pytest X): analyze the saved file content.
- shell (grep/cat/sed/wc...): shell cost model over referenced file sizes.

Output: apriori_scores.jsonl {instance_id, step, command, duration_s, kind,
         scores:{predictor:score or null}}
Then compute, per predictor, Spearman rho vs duration (ranking), + calibrated
median/tail APE with a single global scale constant (not learned per-command).
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import apriori_static as S

SAVE = Path("saved_code")
HEREDOC_RE = re.compile(r"<<\s*'?(\w+)'?\n(.*?)\n\1", re.DOTALL)
DASHC_RE = re.compile(r"""python3?\s+-c\s+(['"])(.*?)\1""", re.DOTALL)
PATH_RE = re.compile(r"[\w./\-]+\.(?:py|pyx)")


def embedded_code(cmd):
    out = []
    for m in HEREDOC_RE.finditer(cmd): out.append(m.group(2))
    for m in DASHC_RE.finditer(cmd): out.append(m.group(2))
    return out


def classify(cmd):
    c = cmd.strip()
    fw = c.split()[0] if c.split() else ""
    if embedded_code(cmd):
        return "embedded"
    if fw in ("python", "python3") or "pytest" in c:
        return "runs_file"
    if fw in ("cat", "grep", "sed", "wc", "nl", "head", "tail", "find", "ls", "diff"):
        return "shell"
    return "other"


def get_code(iid, cmd, kind, files):
    if kind == "embedded":
        return "\n".join(embedded_code(cmd))
    if kind == "runs_file":
        paths = PATH_RE.findall(cmd)
        texts = [files[p] for p in paths if p in files]
        return "\n".join(texts) if texts else None
    return None


def shell_cost(cmd, files):
    """Shell cost model: ~ bytes + lines of referenced files (predictors 1-3,6
    all reduce to this for shell; returned as one score)."""
    paths = re.findall(r"[\w./\-]+\.\w+", cmd)
    total_bytes = sum(len(files.get(p, "")) for p in paths)
    total_lines = sum(files.get(p, "").count("\n") for p in paths)
    # if no file content available, fall back to command length
    if total_bytes == 0:
        return float(len(cmd))
    return float(total_bytes + total_lines)


def main():
    rows = [json.loads(l) for l in open("swe_verified_durations.jsonl")]
    out = open("apriori_scores.jsonl", "w")
    n_done = 0
    for r in rows:
        iid = r["instance_id"]
        fdir = SAVE / iid / "files.json"
        files = json.load(open(fdir)) if fdir.exists() else {}
        kind = classify(r["command"])
        code = get_code(iid, r["command"], kind, files)
        scores = {}
        if code:  # python code: run static predictors (1,2,3,6)
            for name, fn in S.PREDICTORS.items():
                v = fn(code)
                scores[name] = None if (v != v) else v  # nan->None
        elif kind == "shell":
            sc = shell_cost(r["command"], files)
            for name in S.PREDICTORS:
                scores[name] = sc
        # predictors 4,5 (cachegrind/branch) are expensive -> done separately
        out.write(json.dumps({
            "instance_id": iid, "step": r["step"], "command": r["command"],
            "duration_s": r["duration_s"], "kind": kind, "scores": scores,
        }) + "\n")
        n_done += 1
    out.close()
    print(f"scored {n_done} commands (static predictors) -> apriori_scores.jsonl",
          file=sys.stderr)


if __name__ == "__main__":
    main()
