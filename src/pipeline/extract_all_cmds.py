import json, re
from pathlib import Path
BASH_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
out = open("trajs_commands.jsonl", "w")
n_inst = n_cmd = 0
for f in sorted(Path("gpt5_trajs_all").rglob("*.traj.json")):
    iid = f.stem.replace(".traj", "")
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    msgs = d.get("messages", d)
    step = 0
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for a in BASH_RE.findall(str(m.get("content", ""))):
                c = a.strip()
                fw = c.split()[0] if c.split() else ""
                out.write(json.dumps({"instance_id": iid, "step": step,
                                      "command": c, "first_word": fw}) + "\n")
                step += 1; n_cmd += 1
    n_inst += 1
out.close()
print(f"instances={n_inst} commands={n_cmd}")
