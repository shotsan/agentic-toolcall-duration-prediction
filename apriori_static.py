"""
A-priori STATIC predictors (no execution): predictors 1,2,3,6 from APRIORI_PLAN.
Each takes Python source and returns a cost SCORE (proportional to predicted
time). No ML, no training. Pure analysis of the code / its bytecode / AST.

1. bytecode_count      : number of CPython bytecode instructions (static)
2. weighted_bytecode   : sum of per-opcode cost weights
3. complexity_bigO     : effective op count = product of loop bounds x nesting
6. wcet_bound          : worst-case path cost (upper bound over branches/loops)

For loops, static bytecode count alone misses trip counts; predictors 3 and 6
incorporate loop-bound estimates from AST literals/range().
"""
from __future__ import annotations
import ast
import dis
import math

# Rough relative cost weights for bytecode op families (cycles-ish, a-priori).
# Not learned — assigned from known interpreter costs.
_OP_WEIGHT = {
    "CALL": 5.0, "CALL_FUNCTION": 5.0, "CALL_METHOD": 5.0,
    "BINARY": 2.0, "INPLACE": 2.0, "COMPARE": 2.0,
    "LOAD": 1.0, "STORE": 1.0, "RETURN": 1.0, "POP": 0.5,
    "JUMP": 1.0, "FOR_ITER": 2.0, "GET_ITER": 2.0,
    "BUILD": 3.0, "IMPORT": 20.0, "MAKE_FUNCTION": 3.0,
}


def _op_weight(opname: str) -> float:
    for prefix, w in _OP_WEIGHT.items():
        if opname.startswith(prefix):
            return w
    return 1.0


def _safe_compile(src: str):
    try:
        return compile(src, "<snippet>", "exec")
    except Exception:
        return None


def bytecode_count(src: str) -> float:
    """Predictor 1: total static bytecode instruction count (incl. nested code
    objects like functions/comprehensions)."""
    code = _safe_compile(src)
    if code is None:
        return float("nan")
    total = 0
    stack = [code]
    while stack:
        c = stack.pop()
        for ins in dis.get_instructions(c):
            total += 1
        for const in c.co_consts:
            if hasattr(const, "co_code"):
                stack.append(const)
    return float(total)


def weighted_bytecode(src: str) -> float:
    """Predictor 2: sum of per-opcode cost weights over all code objects."""
    code = _safe_compile(src)
    if code is None:
        return float("nan")
    total = 0.0
    stack = [code]
    while stack:
        c = stack.pop()
        for ins in dis.get_instructions(c):
            total += _op_weight(ins.opname)
        for const in c.co_consts:
            if hasattr(const, "co_code"):
                stack.append(const)
    return total


# ---- AST loop-bound analysis for complexity / WCET -------------------------
def _trip_count(node: ast.AST) -> float:
    UNKNOWN = 10.0
    if isinstance(node, ast.For):
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) \
                and it.func.id == "range":
            vals = [a.value for a in it.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, int)]
            if len(it.args) == 1 and vals:
                return float(max(vals[0], 0))
            if len(it.args) >= 2 and len(vals) >= 2:
                return float(max(vals[1] - vals[0], 0))
            return UNKNOWN
        if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
            return float(len(it.elts))
        if isinstance(it, ast.Constant) and isinstance(it.value, str):
            return float(len(it.value))
        return UNKNOWN
    return UNKNOWN  # while: unknown statically


class _LoopCost(ast.NodeVisitor):
    def __init__(self, worst: bool):
        self.worst = worst
        self.stack = []        # enclosing trip counts
        self.total = 0.0

    def _body_base(self, body):
        # count statements as unit work at this level
        return float(max(len(body), 1))

    def visit_For(self, node):
        trip = _trip_count(node)
        self.stack.append(trip)
        prod = 1.0
        for b in self.stack:
            prod *= max(b, 1.0)
        self.total += prod * self._body_base(node.body)
        self.generic_visit(node)
        self.stack.pop()

    def visit_While(self, node):
        self.stack.append(100.0 if self.worst else 10.0)
        prod = 1.0
        for b in self.stack:
            prod *= max(b, 1.0)
        self.total += prod * self._body_base(node.body)
        self.generic_visit(node)
        self.stack.pop()


def complexity_bigO(src: str) -> float:
    """Predictor 3: effective operation count from loop nesting x trip counts."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return float("nan")
    v = _LoopCost(worst=False)
    v.visit(tree)
    # base cost = number of top-level statements, plus loop contributions
    base = float(len(tree.body))
    return base + v.total


def wcet_bound(src: str) -> float:
    """Predictor 6: worst-case path upper bound (use max loop bounds, both
    branches of conditionals assumed taken)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return float("nan")
    v = _LoopCost(worst=True)
    v.visit(tree)
    # add a branch-penalty: count If nodes (both arms considered)
    n_if = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
    return float(len(tree.body)) + v.total + n_if * 2.0


PREDICTORS = {
    "bytecode_count": bytecode_count,
    "weighted_bytecode": weighted_bytecode,
    "complexity_bigO": complexity_bigO,
    "wcet_bound": wcet_bound,
}


if __name__ == "__main__":
    tests = [
        'print("hi")',
        's=0\nfor i in range(1000000):\n    s+=i\nprint(s)',
        's=0\nfor i in range(1000):\n    for j in range(1000):\n        s+=i*j\nprint(s)',
    ]
    for t in tests:
        print(repr(t[:40]))
        for name, fn in PREDICTORS.items():
            print(f"   {name:18s} {fn(t):.1f}")
