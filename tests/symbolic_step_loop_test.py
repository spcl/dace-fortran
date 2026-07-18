"""Coverage for ``DO i = lo, hi, step`` loops whose step is a RUNTIME value (scalar arg or
array element), not a compile-time literal. The bridge previously refused all non-constant
steps with a defensive throw; encountered upstream in QE's ``vexx_bp_k_gpu``
(``DO jbnd = jstart, jend, many_fft``). Fix: capture the symbolic step on
``ASTNode.loop_step_expr``, threaded through by emit_cfg as the loop-region update
expression. Forward iteration is assumed; runtime-negative step symbols yield zero-or-one
iterations under ``uid <= bound``, matching Fortran's trip-count formula.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_run(src: str, tmp: Path, *, ref_kwargs: dict, sdfg_kwargs: dict, mod_name: str = "kern"):
    """Build f2py reference + SDFG, run each with the right kwarg shape. f2py returns
    INTENT(OUT) arrays as the return value; the SDFG takes every dummy including ``out``.
    Returns ``(ref_out_array, sdfg_kwargs_after_call)``.
    """
    mod = f2py_compile(src, tmp / "ref", mod_name)
    sdfg = build_sdfg(src, tmp / "sdfg", name=mod_name, entry="kernel_mod::kernel").build()
    sdfg_copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sdfg_kwargs.items()}
    # kernel lives in kernel_mod, so f2py exposes it under the module's submodule namespace
    ref_out = mod.kernel_mod.kernel(**ref_kwargs)
    sdfg(**sdfg_copy)
    return ref_out, sdfg_copy


def test_step_from_scalar_argument(tmp_path: Path):
    """``DO jbnd = jstart, jend, batch`` where ``batch`` is a runtime scalar (QE ``many_fft``
    shape); ``out`` accumulates one entry per stride step, checked bit-exact against f2py."""
    src = """
module kernel_mod
  implicit none
contains
subroutine kernel(out, jstart, jend, batch, n)
  implicit none
  integer, intent(in) :: jstart, jend, batch, n
  integer, intent(out) :: out(n)
  integer :: j, k
  k = 0
  out(:) = -1
  do j = jstart, jend, batch
    k = k + 1
    if (k <= n) out(k) = j
  end do
end subroutine kernel
end module kernel_mod
"""
    out = np.zeros(8, dtype=np.int32, order="F")
    ref_out, sdfg = _build_and_run(src,
                                   tmp_path,
                                   ref_kwargs=dict(jstart=2, jend=20, batch=3, n=8),
                                   mod_name="kern_scalar_batch3",
                                   sdfg_kwargs=dict(out=out,
                                                    jstart=np.int32(2),
                                                    jend=np.int32(20),
                                                    batch=np.int32(3),
                                                    n=np.int32(8)))
    np.testing.assert_array_equal(sdfg["out"], ref_out)
    # Sanity: the batch=3 stride captured ``2, 5, 8, 11, 14, 17, 20``.
    expected = np.array([2, 5, 8, 11, 14, 17, 20, -1], dtype=np.int32)
    np.testing.assert_array_equal(sdfg["out"], expected)


def test_step_from_scalar_argument_with_batch_one(tmp_path: Path):
    """``batch = 1``: step evaluates to 1 at runtime, loop runs every iteration in
    ``[jstart, jend]`` -- symbolic-step path must match the constant-step path."""
    src = """
module kernel_mod
  implicit none
contains
subroutine kernel(out, jstart, jend, batch, n)
  implicit none
  integer, intent(in) :: jstart, jend, batch, n
  integer, intent(out) :: out(n)
  integer :: j, k
  k = 0
  out(:) = -1
  do j = jstart, jend, batch
    k = k + 1
    if (k <= n) out(k) = j
  end do
end subroutine kernel
end module kernel_mod
"""
    out = np.zeros(8, dtype=np.int32, order="F")
    ref_out, sdfg = _build_and_run(src,
                                   tmp_path,
                                   ref_kwargs=dict(jstart=1, jend=5, batch=1, n=8),
                                   mod_name="kern_scalar_batch1",
                                   sdfg_kwargs=dict(out=out,
                                                    jstart=np.int32(1),
                                                    jend=np.int32(5),
                                                    batch=np.int32(1),
                                                    n=np.int32(8)))
    np.testing.assert_array_equal(sdfg["out"], ref_out)
    expected = np.array([1, 2, 3, 4, 5, -1, -1, -1], dtype=np.int32)
    np.testing.assert_array_equal(sdfg["out"], expected)


def test_step_from_array_element(tmp_path: Path):
    """``DO j = 1, n, stride_arr(idx)``: step reads from an array. The bridge hoists the
    non-trivial step to a fresh ``loopstep_<nid>`` symbol via a pre-LoopRegion interstate
    edge (mirrors bound-hoist); ``arr(idx)``->``arr[idx-1]`` conversion happens once there via ``_fortran_subs_to_dace``.
    """
    src = """
module kernel_mod
  implicit none
contains
subroutine kernel(out, stride_arr, idx, n, m)
  implicit none
  integer, intent(in) :: idx, n, m
  integer, intent(in) :: stride_arr(m)
  integer, intent(out) :: out(n)
  integer :: j, k
  k = 0
  out(:) = -1
  do j = 1, n, stride_arr(idx)
    k = k + 1
    out(k) = j
  end do
end subroutine kernel
end module kernel_mod
"""
    stride_arr = np.array([2, 3, 5], dtype=np.int32, order="F")
    out = np.zeros(8, dtype=np.int32, order="F")
    ref_out, sdfg = _build_and_run(src,
                                   tmp_path,
                                   ref_kwargs=dict(stride_arr=stride_arr, idx=2, n=8, m=3),
                                   mod_name="kern_array_stride",
                                   sdfg_kwargs=dict(out=out,
                                                    stride_arr=stride_arr,
                                                    idx=np.int32(2),
                                                    n=np.int32(8),
                                                    m=np.int32(3)))
    np.testing.assert_array_equal(sdfg["out"], ref_out)
    # stride=3: iterations 1, 4, 7.
    expected = np.array([1, 4, 7, -1, -1, -1, -1, -1], dtype=np.int32)
    np.testing.assert_array_equal(sdfg["out"], expected)


def test_step_expr_field_is_populated_on_symbolic_step(tmp_path: Path):
    """``ASTNode.loop_step_expr`` carries the symbolic-step string; drives AST extraction
    directly so the contract is pinned independent of any downstream emit path."""
    from dace_fortran.build_bridge import hb
    from dace_fortran import DEFAULT_PIPELINE
    src = """
subroutine kernel(jstart, jend, batch)
  implicit none
  integer, intent(in) :: jstart, jend, batch
  integer :: j, n_iters
  n_iters = 0
  do j = jstart, jend, batch
    n_iters = n_iters + 1
  end do
end subroutine kernel
"""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path as _P
        f = _P(td) / "k.f90"
        f.write_text(src)
        h = _P(td) / "k.hlfir"
        subprocess.check_call([
            "flang-new-21", "-fc1", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang", "-emit-hlfir",
            str(f), "-o",
            str(h)
        ],
                              cwd=td)
        mod = hb.HLFIRModule()
        mod.parse_file(str(h))
        mod.set_entry_symbol("kernel")
        mod.run_passes(DEFAULT_PIPELINE)
        ast = mod.get_ast()

    # get_ast() returns a list of top-level nodes; flatten and walk every subtree
    def walk(node):
        yield node
        for c in node.children:
            yield from walk(c)

    roots = list(ast) if isinstance(ast, list) else [ast]
    loop_nodes = [n for r in roots for n in walk(r) if n.kind == "loop"]
    assert loop_nodes, "no loop node found in AST"
    # The kernel's only loop has the symbolic step.
    step_exprs = [n.loop_step_expr for n in loop_nodes if n.loop_step_expr]
    assert step_exprs, \
        f"no loop carries loop_step_expr; loops: {[n.loop_step for n in loop_nodes]}"
    assert any("batch" in s for s in step_exprs), \
        f"step expression should mention 'batch', got {step_exprs}"
