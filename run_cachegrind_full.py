"""
FIX: estimate the SAME full command that was timed (not a sub-piece).

For each command we run the ENTIRE bash command under cachegrind, in a temp dir
staged with the instance's saved repo files, so the predictor input == the
ground-truth input. Drops timeout-clamped (>=119s) ground-truth rows.

Output: cachegrind_full.jsonl {instance_id, step, command, duration_s,
         est_cycles, mispredicts, ok}
Resumable. Only commands whose referenced files we actually saved are runnable
faithfully; others are marked ok=False (skipped from the clean comparison).
"""
from __future__ import annotations
import json, os, re, subprocess, sys, tempfile, shutil
from pathlib import Path

SAVE = Path("saved_code")
SUMMARY = {
    "Ir": re.compile(r"I\s+refs:\s+([\d,]+)"),
    "I1_miss": re.compile(r"I1\s+misses:\s+([\d,]+)"),
    "LL_miss": re.compile(r"LL\s+misses:\s+([\d,]+)"),
    "Mispredicts": re.compile(r"Mispredicts:\s+([\d,]+)"),
}


def num(m):
    return float(m.group(1).replace(",", "")) if m else 0.0


def done_keys(path="cachegrind_full.jsonl"):
    s = set()
    try:
        for l in open(path):
            r = json.loads(l); s.add((r["instance_id"], r["step"]))
    except FileNotFoundError:
        pass
    return s


def stage_files(workdir, files: dict):
    """Write saved repo files into workdir preserving their relative paths."""
    for path, content in files.items():
        p = path.lstrip("./")
        full = Path(workdir) / p
        full.parent.mkdir(parents=True, exist_ok=True)
        try:
            full.write_text(content)
        except Exception:
            pass


def run_full(command: str, files: dict, timeout=150):
    """Run the FULL bash command under cachegrind in a staged temp dir."""
    workdir = tempfile.mkdtemp(prefix="cg_")
    try:
        stage_files(workdir, files)
        # valgrind wraps bash -c "<command>"
        cmd = ["valgrind", "--tool=cachegrind", "--branch-sim=yes",
               "--cachegrind-out-file=/dev/null", "bash", "-c", command]
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=workdir)
        err = p.stderr
        vals = {k: num(rx.search(err)) for k, rx in SUMMARY.items()}
        if vals["Ir"] == 0:
            return {"ok": False}
        cycles = (vals["Ir"] + 10 * vals["I1_miss"] + 100 * vals["LL_miss"]
                  + 15 * vals["Mispredicts"])
        return {"ok": True, "est_cycles": cycles,
                "mispredicts": vals["Mispredicts"], "Ir": vals["Ir"]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "timeout": True}
    except Exception as e:
        return {"ok": False, "err": str(e)[:80]}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    rows = [json.loads(l) for l in open("swe_verified_durations.jsonl")]
    # drop timeout-clamped ground truth
    rows = [r for r in rows if r["duration_s"] < 119.0]
    done = done_keys()
    out = open("cachegrind_full.jsonl", "a")
    n = ok = 0
    for r in rows:
        key = (r["instance_id"], r["step"])
        if key in done:
            continue
        fdir = SAVE / r["instance_id"] / "files.json"
        files = json.load(open(fdir)) if fdir.exists() else {}
        res = run_full(r["command"], files, timeout=150)
        rec = {"instance_id": r["instance_id"], "step": r["step"],
               "command": r["command"], "duration_s": r["duration_s"],
               "est_cycles": res.get("est_cycles"),
               "mispredicts": res.get("mispredicts"), "ok": res.get("ok", False)}
        out.write(json.dumps(rec) + "\n"); out.flush()
        n += 1; ok += 1 if res.get("ok") else 0
        if n % 25 == 0:
            print(f"  {n} done ({ok} ok)", file=sys.stderr)
    out.close()
    print(f"full-command cachegrind: {n} commands, {ok} ok", file=sys.stderr)


if __name__ == "__main__":
    main()
