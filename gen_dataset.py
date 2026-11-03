"""
Generate a LARGE, randomized dataset of code-execution "tool calls" spanning a
wide, long-tailed duration range -- mimicking what a coding agent runs.

All snippets tagged tool="execute_code" so the flat per-tool CDF baseline must
lump them together (Continuum's coding-agent limitation).

Randomized bounds/depths across templates let a model learn the
depth x bound -> duration interaction instead of memorizing literals.
"""
from __future__ import annotations
import json
import random

random.seed(42)
SNIPPETS: list[str] = []


def add(code: str) -> None:
    SNIPPETS.append(code)


# --- trivial / constant (fast majority) ---
TRIVIAL = [
    'print("hello")',
    'x=1+1\nprint(x)',
    'print(sum([1,2,3,4,5]))',
    'import os\nprint(os.getpid())',
    'd={"a":1}\nprint(d.get("a"))',
    's="abc".upper()\nprint(s)',
    'print(len("hello world"))',
    'print(max([3,1,2]))',
]
for s in TRIVIAL:
    add(s)
    add(s)  # duplicate the common case (realistic: most calls are trivial)

# --- flat O(n) loops, random bound ---
for _ in range(25):
    n = random.choice([1_000, 10_000, 100_000, 500_000, 1_000_000,
                       3_000_000, 8_000_000])
    add(f's=0\nfor i in range({n}):\n    s+=i\nprint(s)')

# --- nested O(n^2) loops, random bound ---
for _ in range(25):
    n = random.choice([50, 100, 200, 400, 800, 1500, 2500])
    add(f's=0\nfor i in range({n}):\n    for j in range({n}):\n        s+=i*j\nprint(s)')

# --- triple nested O(n^3) ---
for _ in range(10):
    n = random.choice([20, 40, 80, 150])
    add(f's=0\nfor i in range({n}):\n    for j in range({n}):\n        for k in range({n}):\n            s+=1\nprint(s)')

# --- sorting O(n log n) ---
for _ in range(15):
    n = random.choice([10_000, 100_000, 500_000, 1_000_000, 3_000_000])
    add(f'import random\nrandom.seed(0)\na=[random.random() for _ in range({n})]\na.sort()\nprint(a[0])')

# --- string building ---
for _ in range(12):
    n = random.choice([10_000, 100_000, 1_000_000, 5_000_000])
    add(f'x="a"*{n}\ny=x[::-1]\nprint(len(y))')

# --- comprehensions ---
for _ in range(15):
    n = random.choice([100_000, 1_000_000, 4_000_000, 8_000_000])
    add(f'v=[i*i for i in range({n})]\nprint(v[-1])')

# --- recursion (fib) exponential tail ---
for _ in range(10):
    n = random.choice([18, 22, 26, 30, 32])
    add(f'def fib(k):\n    return k if k<2 else fib(k-1)+fib(k-2)\nprint(fib({n}))')

# --- mixed math-in-loop ---
for _ in range(12):
    n = random.choice([200_000, 1_000_000, 3_000_000, 6_000_000])
    add(f'import math\ndef f(x): return math.sqrt(x)+x*x\n'
        f't=0\nfor i in range({n}):\n    t+=f(i)\nprint(t)')

# --- dict/set building ---
for _ in range(10):
    n = random.choice([100_000, 1_000_000, 4_000_000])
    add(f'd={{}}\nfor i in range({n}):\n    d[i]=i*i\nprint(len(d))')


def main() -> None:
    random.shuffle(SNIPPETS)
    with open("dataset.jsonl", "w") as fh:
        for code in SNIPPETS:
            fh.write(json.dumps({"tool": "execute_code", "code": code}) + "\n")
    print(f"wrote {len(SNIPPETS)} snippets to dataset.jsonl")


if __name__ == "__main__":
    main()
