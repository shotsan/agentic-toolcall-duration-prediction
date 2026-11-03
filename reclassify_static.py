"""
Recompute static predictors (1,2,3,6) on a STRICT, correctly-classified subset
and report rho vs duration. Fixes the classifier so shell commands
(nl/sed/grep/cat even when they mention a .py file) are NOT treated as
python-code, and compound commands (A && B, A | B) are excluded (ambiguous).

Strict python-code = the command is ONLY a python/pytest invocation (optionally
with a heredoc), with NO leading shell utility and NO pipe/&&/; chaining to
non-python work.
"""
import json, re, numpy as np
from scipy.stats import spearmanr
import apriori_static as S

gt = {}
for l in open("swe_verified_durations.jsonl"):
    r = json.loads(l)
    if r["duration_s"] < 119.0:
        gt[(r["instance_id"], r["step"])] = r


def embedded_code(cmd):
    out = []
    for m in re.finditer(r"<<\s*'?(\w+)'?\n(.*?)\n\1", cmd, re.DOTALL):
        out.append(m.group(2))
    for m in re.finditer(r"""python3?\s+-c\s+(['"])(.*?)\1""", cmd, re.DOTALL):
        out.append(m.group(2))
    return out


def is_strict_python(cmd):
    c = cmd.strip()
    # must START with python/python3 (allow env-var prefix) and have embedded code
    first = c.split()[0] if c.split() else ""
    starts_py = first in ("python", "python3") or first.startswith("PYTHONPATH") \
        or first.startswith("PYTHONWARNINGS")
    if not starts_py:
        return False
    # exclude chaining to other shell tools AFTER the python part
    # (heredoc content may contain anything; we only forbid shell-chaining the
    #  python invocation to nl/sed/grep/cat pipelines)
    if re.search(r"\|\s*(nl|sed|grep|cat|head|tail|wc|awk)\b", c):
        return False
    return True


def analyzable_code(cmd):
    ec = embedded_code(cmd)
    return "\n".join(ec) if ec else None


# strict python subset with embedded code we can statically analyze
rows = []
for k, r in gt.items():
    if is_strict_python(r["command"]):
        code = analyzable_code(r["command"])
        if code:
            rows.append((r["duration_s"], code))

print(f"STRICT python-code-with-embedded commands: n={len(rows)}")
y = np.array([d for d, _ in rows])
for nm, fn in S.PREDICTORS.items():
    s = np.array([fn(code) for _, code in rows], float)
    m = ~np.isnan(s)
    if m.sum() < 10:
        print(f"  {nm}: too few"); continue
    print(f"  {nm:20s} rho={spearmanr(s[m], y[m]).correlation:+.3f} (n={m.sum()})")
