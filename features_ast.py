"""
Python AST-based static feature extractor for code-execution tool calls.

Extracts structural features that correlate with runtime WITHOUT running the
code. These are the "code structure" feature source (source #2 in the README).
Complements llvm-mca (microarch) and historical CDF.

Rationale (PrETi, arXiv 2503.13679): pure source features are imperfect because
compilers transform code, but for interpreted Python (no -O2 transforms at the
AST level) structural features like loop nesting and node counts are a strong,
cheap signal — and crucially they distinguish `print("hi")` from a nested loop,
which a flat per-tool CDF cannot.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, asdict


@dataclass
class ASTFeatures:
    n_nodes: int = 0            # total AST nodes (overall size)
    max_loop_depth: int = 0     # deepest nested loop (drives polynomial cost)
    n_loops: int = 0            # for + while count
    n_calls: int = 0            # function calls (I/O / heavy-op hints)
    n_comprehensions: int = 0   # list/set/dict/generator comps (implicit loops)
    n_arith_ops: int = 0        # BinOp count (compute intensity)
    has_recursion_hint: int = 0 # a function that calls itself by name
    n_assigns: int = 0
    n_ranges: int = 0           # range(...) calls (loop-bound hints)
    largest_literal_int: int = 0  # biggest int constant (often a loop bound)
    # Effective operation count: sum over loops of (product of enclosing loop
    # bounds). Captures the O(n) vs O(n^2) interaction the raw literal misses.
    # Stored as log10(1+ops) because it spans many orders of magnitude.
    log_effective_ops: float = 0.0

    def vector(self) -> list[float]:
        return [float(v) for v in asdict(self).values()]

    @staticmethod
    def names() -> list[str]:
        return list(ASTFeatures().__dataclass_fields__.keys())


def _estimate_trip_count(node: ast.AST) -> float:
    """Best-effort static trip-count estimate for a loop header.
    - `for x in range(N)` / `range(a, b)` / `range(a, b, s)` with int literals.
    - `for x in <literal list/tuple/str>`: its length.
    - comprehension targets handled by caller.
    Fallback: a generic constant (unknown bound) so the loop still contributes.
    """
    UNKNOWN = 32.0  # neutral guess for loops whose bound isn't a literal
    if isinstance(node, (ast.For,)):
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) \
                and it.func.id == "range":
            args = it.args
            def val(a):
                return a.value if isinstance(a, ast.Constant) and \
                    isinstance(a.value, int) else None
            nums = [val(a) for a in args]
            if len(nums) == 1 and nums[0] is not None:
                return max(0.0, float(nums[0]))
            if len(nums) >= 2 and None not in nums[:2]:
                step = nums[2] if len(nums) == 3 and nums[2] else 1
                return max(0.0, float((nums[1] - nums[0]) / (step or 1)))
            return UNKNOWN
        if isinstance(it, (ast.List, ast.Tuple)):
            return float(len(it.elts))
        if isinstance(it, ast.Constant) and isinstance(it.value, str):
            return float(len(it.value))
        return UNKNOWN
    return UNKNOWN  # while loops: bound unknown statically


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.f = ASTFeatures()
        self._loop_depth = 0
        self._bound_stack: list[float] = []  # enclosing loop trip counts
        self._effective_ops = 0.0
        self._func_names: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.f.n_nodes += 1
        super().generic_visit(node)

    def _visit_loop(self, node: ast.AST) -> None:
        self.f.n_loops += 1
        self._loop_depth += 1
        self.f.max_loop_depth = max(self.f.max_loop_depth, self._loop_depth)
        trip = _estimate_trip_count(node)
        self._bound_stack.append(trip)
        # effective ops contributed at this nesting = product of all enclosing
        # bounds including this one.
        prod = 1.0
        for b in self._bound_stack:
            prod *= max(b, 1.0)
        self._effective_ops += prod
        self.generic_visit(node)
        self._bound_stack.pop()
        self._loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.f.n_calls += 1
        fn = node.func
        if isinstance(fn, ast.Name):
            if fn.id == "range":
                self.f.n_ranges += 1
            if fn.id in self._func_names:
                self.f.has_recursion_hint = 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_names.append(node.name)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.f.n_arith_ops += 1
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.f.n_assigns += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.f.n_comprehensions += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.f.n_comprehensions += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.f.n_comprehensions += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.f.n_comprehensions += 1
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            self.f.largest_literal_int = max(self.f.largest_literal_int,
                                             abs(node.value))
        self.generic_visit(node)


def extract(code: str) -> ASTFeatures:
    """Parse Python source and return structural features. Returns zeros on
    syntax error (caller can treat unparseable code as a feature too)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ASTFeatures()
    v = _Visitor()
    v.visit(tree)
    # Fold comprehension iteration into effective ops: each comprehension's
    # generator contributes its (estimated) length as a loop.
    import math
    for comp in ast.walk(tree):
        if isinstance(comp, (ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp)):
            prod = 1.0
            for gen in comp.generators:
                it = gen.iter
                cnt = 32.0
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) \
                        and it.func.id == "range":
                    a = it.args
                    if len(a) == 1 and isinstance(a[0], ast.Constant) \
                            and isinstance(a[0].value, int):
                        cnt = float(a[0].value)
                prod *= max(cnt, 1.0)
            v._effective_ops += prod
    v.f.log_effective_ops = math.log10(1.0 + v._effective_ops)
    return v.f


if __name__ == "__main__":
    import sys, json
    src = sys.stdin.read()
    feats = extract(src)
    print(json.dumps({"features": feats.vector(), "names": ASTFeatures.names(),
                      "detail": asdict(feats)}, indent=2))
