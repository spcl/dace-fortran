# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Retype integer-valued float ``**`` exponents to ``int`` so codegen emits
``ipow`` (repeated multiply) instead of libm ``pow``, which isn't bit-identical
to Fortran/C's integer-power codegen and drifts across real(8) reduction chains.
Only the exponent literal is rewritten; base and fractional exponents are untouched.
"""

import ast
from typing import Optional

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible


class _ExponentIntegerizer(ast.NodeTransformer):
    """Rewrite integer-valued float ``**`` exponents to ``int`` literals."""

    def __init__(self):
        self.rewrites = 0

    @staticmethod
    def _as_int_constant(node: ast.AST) -> Optional[ast.AST]:
        """Int-valued replacement for a float ``**`` exponent (``2.0`` or
        ``-2.0``), or ``None`` if not integer-valued. Sign folds into the int."""
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, float) and node.operand.value.is_integer()):
            return ast.copy_location(ast.Constant(value=-int(node.operand.value)), node)
        if (isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value.is_integer()):
            return ast.copy_location(ast.Constant(value=int(node.value)), node)
        return None

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Retype an integer-valued float ``**`` exponent to ``int``."""
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            repl = self._as_int_constant(node.right)
            if repl is not None:
                node.right = repl
                self.rewrites += 1
        return node


@explicit_cf_compatible
class IntegerizePowerExponents(ppl.Pass):
    """Retype integer-valued float ``**`` exponents in tasklets to ``int``.

    Must run after tasklet splitting -- flips codegen from libm ``pow`` to
    ``ipow`` without touching the base expression or any connector.
    """

    def modifies(self) -> ppl.Modifies:
        """This pass only mutates tasklet bodies."""
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """One-shot: the retype doesn't re-trigger the pass."""
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Rewrite every Python tasklet's integer-valued float ``**`` exponents
        to ``int``, recursively including nested SDFGs. Returns rewrite count, or ``None`` if unchanged."""
        total = 0
        for node, _parent in sdfg.all_nodes_recursive():
            if not isinstance(node, dace.nodes.Tasklet):
                continue
            if node.code.language != dace.dtypes.Language.Python:
                continue
            body = node.code.code
            if not isinstance(body, list):
                continue
            tr = _ExponentIntegerizer()
            for stmt in body:
                tr.visit(stmt)
            if tr.rewrites:
                for stmt in body:
                    ast.fix_missing_locations(stmt)
                total += tr.rewrites
        return total or None
