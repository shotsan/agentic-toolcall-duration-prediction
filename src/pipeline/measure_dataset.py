"""
Read dataset.jsonl, measure each snippet's real duration, extract AST features,
write labeled.jsonl with {tool, code, features, names, median_s, ...}.
"""
from __future__ import annotations
import json
import sys
from measure import measure_snippet
from features_ast import extract, ASTFeatures


def main() -> None:
    rows = []
    with open("dataset.jsonl") as fh:
        for line in fh:
            rows.append(json.loads(line))

    names = ASTFeatures.names()
    out = []
    for i, r in enumerate(rows):
        feats = extract(r["code"])
        timing = measure_snippet(r["code"], runs=3, timeout=90.0)
        rec = {
            "tool": r["tool"],
            "code": r["code"],
            "feature_names": names,
            "features": feats.vector(),
            **timing,
        }
        out.append(rec)
        ms = (timing["median_s"] or 0) * 1000
        print(f"[{i+1}/{len(rows)}] ok={timing['ok']} "
              f"median={ms:8.1f}ms  depth={feats.max_loop_depth} "
              f"bound={feats.largest_literal_int}", file=sys.stderr)

    with open("labeled.jsonl", "w") as fh:
        for rec in out:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(out)} labeled rows to labeled.jsonl", file=sys.stderr)


if __name__ == "__main__":
    main()
