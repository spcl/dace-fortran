"""Reduction intrinsics as inline expression operands.

A reduction as the immediate RHS (``out = MAXVAL(arr)``) is handled by ``buildReduceNode`` /
``buildSectionReduceAssign``. As an OPERAND of a larger expression (``out = max(scalar,
MAXVAL(arr(s:e)))``), ``buildExpr`` used to return ``"?"`` and the tasklet failed to parse.

``hlfir-lift-reduction-operands`` now rewrites every nested reduction into a preceding
scalar-temp assign:

    %tmp = fir.alloca f64
    %tmp_decl = hlfir.declare %tmp ...
    hlfir.assign %maxval_result to %tmp_decl#0
    %loaded = fir.load %tmp_decl#0
    ...uses of %maxval_result rewritten to %loaded...

so the lifted ``temp = MAXVAL(...)`` becomes a top-level reduction the dispatcher already
handles.

Each test pairs an SDFG run against an f2py/numpy reference on identical random inputs.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_run(src: str, tmp_path: Path, **kwargs) -> dict:
    """Build an SDFG via the bridge, call it with kwargs, return the final buffer state."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    sdfg(**kwargs)
    return kwargs


def test_inline_maxval_in_max_expression(tmp_path: Path):
    """``out = max(scalar, MAXVAL(arr(1:n)))`` -- the failure repro from the velocity_tendencies probe."""
    src = """
subroutine kernel(arr, scalar, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: arr(n)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = max(scalar, maxval(arr(1:n)))
end subroutine kernel
"""
    rng = np.random.default_rng(0)
    n = 16
    arr = np.asfortranarray(rng.standard_normal(n))
    scalar = np.float64(0.5)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out, n=n)
    assert out[0] == max(scalar, arr.max())


def test_inline_minval_in_min_expression(tmp_path: Path):
    """``out = min(scalar, MINVAL(arr))`` -- symmetric of the maxval case."""
    src = """
subroutine kernel(arr, scalar, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: arr(n)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = min(scalar, minval(arr(1:n)))
end subroutine kernel
"""
    rng = np.random.default_rng(1)
    n = 16
    arr = np.asfortranarray(rng.standard_normal(n))
    scalar = np.float64(-0.5)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out, n=n)
    assert out[0] == min(scalar, arr.min())


def test_inline_sum_in_arithmetic(tmp_path: Path):
    """``out = scalar + SUM(arr(1:n))`` -- sum used additively in a larger expression."""
    src = """
subroutine kernel(arr, scalar, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: arr(n)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = scalar + sum(arr(1:n))
end subroutine kernel
"""
    rng = np.random.default_rng(2)
    n = 16
    arr = np.asfortranarray(rng.standard_normal(n))
    scalar = np.float64(1.25)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out, n=n)
    np.testing.assert_allclose(out[0], scalar + arr.sum(), rtol=1e-12)


def test_inline_product_in_arithmetic(tmp_path: Path):
    """``out = scalar * PRODUCT(arr(1:n))`` -- product used multiplicatively in a larger expression."""
    src = """
subroutine kernel(arr, scalar, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: arr(n)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = scalar * product(arr(1:n))
end subroutine kernel
"""
    rng = np.random.default_rng(3)
    n = 4  # small n to keep product manageable
    arr = np.asfortranarray(rng.uniform(0.5, 1.5, n))
    scalar = np.float64(2.0)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out, n=n)
    np.testing.assert_allclose(out[0], scalar * arr.prod(), rtol=1e-12)


def test_two_inline_reductions_in_same_expression(tmp_path: Path):
    """``out = MAXVAL(a(1:n)) + MINVAL(b(1:n))`` -- two distinct reductions in one expression, each must lift independently."""
    src = """
subroutine kernel(a, b, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a(n), b(n)
  real(8), intent(out) :: out
  out = maxval(a(1:n)) + minval(b(1:n))
end subroutine kernel
"""
    rng = np.random.default_rng(4)
    n = 16
    a = np.asfortranarray(rng.standard_normal(n))
    b = np.asfortranarray(rng.standard_normal(n))
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, a=a, b=b, out=out, n=n)
    np.testing.assert_allclose(out[0], a.max() + b.min(), rtol=1e-12)


def test_inline_maxval_no_section(tmp_path: Path):
    """``out = max(scalar, MAXVAL(arr))`` -- whole-array reduction (no slice) used inline;
    routes through the whole-array Reduce path instead of the section-reduce loop."""
    src = """
subroutine kernel(arr, scalar, out)
  implicit none
  real(8), intent(in) :: arr(8)
  real(8), intent(in) :: scalar
  real(8), intent(out) :: out
  out = max(scalar, maxval(arr))
end subroutine kernel
"""
    rng = np.random.default_rng(5)
    arr = np.asfortranarray(rng.standard_normal(8))
    scalar = np.float64(0.3)
    out = np.zeros((1, ), dtype=np.float64)
    _build_and_run(src, tmp_path, arr=arr, scalar=scalar, out=out)
    assert out[0] == max(scalar, arr.max())


def test_dimensional_sum_does_not_corrupt_verifier(tmp_path: Path):
    """``SUM(arr, DIM=1)`` produces an ARRAY (``!hlfir.expr<NxT>``), not a scalar -- the lift
    pass must skip these (a ``fir.alloca``+``fir.load`` round-trip on an ``!hlfir.expr`` is
    invalid IR; ``fir.load`` only returns FIR-dialect scalar types). Encountered upstream in
    QE's ``vexx_bp_k_gpu``: ``SQRT(SUM(matrix**2, DIM=1))``.

    Before the ``isScalar`` guard in ``LiftReductionOperands.cpp``, the pipeline died at
    ``hlfir-lift-reduction-operands`` with a verifier complaint. This test pins the verifier
    contract only; SDFG construction may still hit a separate downstream gap for dimensional
    reductions (``emit_reduce`` doesn't yet handle non-named-array sources).
    """
    import tempfile
    src = """
subroutine kernel(matrix, out)
  implicit none
  real(8), intent(in) :: matrix(3, 3)
  real(8), intent(out) :: out
  out = minval(sqrt(sum(matrix**2, dim=1)))
end subroutine kernel
"""
    # drive the bridge pipeline directly: parse HLFIR, run DEFAULT_PIPELINE; a resurfaced
    # verifier complaint makes run_passes raise
    import subprocess
    from dace_fortran import DEFAULT_PIPELINE
    from dace_fortran.build_bridge import hb
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
        # before the fix this raised at hlfir-lift-reduction-operands
        mod.run_passes(DEFAULT_PIPELINE)
