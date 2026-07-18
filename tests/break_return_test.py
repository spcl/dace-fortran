"""Builder-side emit handlers for BreakBlock/ReturnBlock: given an ``SDFGBuilder``
seeded with a stub AST (``kind="break"``/``"return"``), the emitted SDFG must
validate, compile, and run correctly. Bridge-side EXIT detection (real Fortran
source via lift-cf-to-scf) is covered separately by do_loop_exit_test.py; this
is the focused unit test for the emit handlers themselves.
"""

from dataclasses import dataclass, field

import numpy as np
import pytest


@dataclass
class _Node:
    """Minimal stand-in for the nanobind ASTNode."""
    kind: str
    target: str = ""
    expr: str = ""
    target_is_array: bool = False
    loop_iter: str = ""
    loop_lower: int = 1
    loop_bound: str = ""
    condition: str = ""
    callee: str = ""
    call_args: list = field(default_factory=list)
    reduce_src: str = ""
    reduce_wcr: str = ""
    reduce_identity: str = ""
    reduce_axes: list = field(default_factory=list)
    children: list = field(default_factory=list)
    else_children: list = field(default_factory=list)
    accesses: list = field(default_factory=list)


def test_return_block_wired_at_top_level(tmp_path):
    """Top-level RETURN emits a ReturnBlock -> early C++ ``return``; calling the
    SDFG must leave inputs untouched (no compute before the return)."""
    import dace
    from dace import SDFG
    from dace_fortran.hlfir_to_sdfg import SDFGBuilder

    builder = SDFGBuilder.__new__(SDFGBuilder)
    builder.variables = []
    builder.arrays = {}
    builder.symbols = {}
    builder.scalars = {}
    builder._id_counter = 0
    sdfg = SDFG("early_ret")
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)

    from dace_fortran.hlfir_to_sdfg import _Ctx
    ctx = _Ctx(sdfg, builder)

    ast = [_Node(kind="return")]
    builder._emit(ctx, ast, sdfg)
    ctx.flush(builder, sdfg)
    sdfg.validate()

    # Bare top-level RETURN: must compile, run, and leave a element-wise unchanged.
    a_init = np.array([1.5, -2.5, 3.5, 4.5], dtype=np.float64)
    a = a_init.copy()
    sdfg(a=a, n=a.size)
    np.testing.assert_array_equal(a, a_init)


def test_break_block_inside_loop_region(tmp_path):
    """LoopRegion with a ConditionalBlock whose true-arm is a BreakBlock behaves like
    an early-exit while; the empty body never writes ``a``, so it comes back unchanged either way."""
    import dace
    from dace import SDFG
    from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
    from dace_fortran.hlfir_to_sdfg import SDFGBuilder, _Ctx

    builder = SDFGBuilder.__new__(SDFGBuilder)
    builder.variables = []
    builder.arrays = {}
    builder.symbols = {}
    builder.scalars = {}
    builder._id_counter = 0

    sdfg = SDFG("early_break")
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("n", dace.int64)
    sdfg.add_array("a", shape=(dace.symbol("n"), ), dtype=dace.float64, transient=False)

    # Manually wires: LoopRegion(i=1..n) { ConditionalBlock: break-arm if a[i-1]>100, else no-op body }
    loop = LoopRegion(label="loop_0",
                      condition_expr="i < n + 1",
                      loop_var="i",
                      initialize_expr="i = 1",
                      update_expr="i = i + 1")
    sdfg.add_node(loop)

    cond_block = ConditionalBlock("if_exit")
    loop.add_node(cond_block, ensure_unique_name=True)

    break_region = ControlFlowRegion("exit_branch", sdfg=sdfg)
    cond_block.add_branch("(a[i - 1] > 100)", break_region)
    builder._emit(_Ctx(sdfg, builder), [_Node(kind="break")], break_region)

    else_region = ControlFlowRegion("body_branch", sdfg=sdfg)
    else_region.add_state("body_noop", is_start_block=True)
    cond_block.add_branch(None, else_region)

    sdfg.validate()

    # Loop body has no writes; whether the break fires (a[3]>100 below) or not, a is unchanged.
    a_init = np.array([1.0, 2.0, 3.0, 200.0, 5.0], dtype=np.float64)
    a = a_init.copy()
    sdfg(a=a, n=a.size, i=0)
    np.testing.assert_array_equal(a, a_init)

    # No-break path: every element below threshold, loop exits naturally on the counter.
    a_no_break = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    a_nb = a_no_break.copy()
    sdfg(a=a_nb, n=a_nb.size, i=0)
    np.testing.assert_array_equal(a_nb, a_no_break)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
