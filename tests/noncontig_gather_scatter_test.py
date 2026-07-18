"""Numerical correctness for Fortran array-based gather (``rhs = a(idx)``) and scatter (``a(idx) = rhs``).

Both lower through a per-iteration loop reassigning a single indirection symbol (``<arr>_at<gid>``) -- the symbol SLOT is reused, respecting DaCe's "no array of symbols" constraint.  Pipeline: ``hlfir-expand-vector-subscript-gather``/``-scatter`` replace flang's hlfir.associate/region_assign with explicit DO loops.  SDFG-vs-numpy at rtol=1e-12; ``ported/noncontig_pardecls_test.py`` covers the cross-subroutine-call variant via hlfir-inline-all.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------------------
# GATHER  --  ``out = d(cols)`` shape patterns
# ---------------------------------------------------------------------------


def test_gather_into_local_then_use(tmp_path: Path):
    """d(cols) materialised as a local array, then doubled elementwise -- no subroutine call, exercises the gather expression directly."""
    src = """
subroutine main(d, cols, out)
  double precision, intent(in)  :: d(8)
  integer,          intent(in)  :: cols(4)
  double precision, intent(out) :: out(4)
  double precision              :: tmp(4)
  integer :: i
  tmp = d(cols)
  do i = 1, 4
    out(i) = tmp(i) * 2.0d0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(0)
    d = rng.random(8).astype(np.float64)
    cols = np.array([2, 5, 1, 7], dtype=np.int32)  # 1-based Fortran indices
    out = np.zeros(4, dtype=np.float64)
    sdfg(d=d, cols=cols, out=out)
    expected = d[cols - 1] * 2.0  # numpy uses 0-based indexing
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_gather_inline_in_expression(tmp_path: Path):
    """out(i) = d(cols(i)) + d(cols(i))**2 -- gather inside a bigger arithmetic tree; each indirect read gets its own indirection-symbol slot."""
    src = """
subroutine main(d, cols, out)
  double precision, intent(in)  :: d(10)
  integer,          intent(in)  :: cols(5)
  double precision, intent(out) :: out(5)
  integer :: i
  do i = 1, 5
    out(i) = d(cols(i)) + d(cols(i)) ** 2
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(1)
    d = rng.random(10).astype(np.float64)
    cols = np.array([1, 4, 7, 9, 2], dtype=np.int32)
    out = np.zeros(5, dtype=np.float64)
    sdfg(d=d, cols=cols, out=out)
    g = d[cols - 1]
    np.testing.assert_allclose(out, g + g**2, rtol=1e-12)


def test_inline_gather_with_symbolic_extent(tmp_path: Path):
    """tmp = d(cols) where cols has runtime length n.  Positive counterpart of the symbolic-extent bail-out in noncontig_unsupported_test.py: inline gathers skip hlfir.associate entirely, so the static-extent guard never fires -- the regular elementwise-assign path handles symbolic n directly via indirection memlets."""
    src = """
subroutine main(n, d, cols, out)
  integer, intent(in) :: n
  double precision, intent(in)  :: d(2 * n)
  integer,          intent(in)  :: cols(n)
  double precision, intent(out) :: out(n)
  double precision              :: tmp(n)
  tmp = d(cols)
  out = tmp
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 4
    d = np.arange(8, dtype=np.float64)
    cols = np.array([1, 3, 5, 7], dtype=np.int32)
    out = np.zeros(4, dtype=np.float64)
    sdfg(n=n, d=d, cols=cols, out=out)
    np.testing.assert_allclose(out, d[cols - 1], rtol=1e-12)


def test_gather_via_call(tmp_path: Path):
    """call fun(d(cols), out) -- noncontig slice passed to a callee taking a contiguous array; flang produces hlfir.associate, hlfir-expand-vector-subscript-gather lowers it.  Constant-shape dummy avoids a separate n symbol."""
    src = """
subroutine fun(d, out)
  double precision, intent(in)  :: d(4)
  double precision, intent(out) :: out(4)
  integer :: i
  do i = 1, 4
    out(i) = d(i) * 3.0d0
  end do
end subroutine fun

subroutine main(d, cols, out)
  double precision, intent(in)  :: d(10)
  integer,          intent(in)  :: cols(4)
  double precision, intent(out) :: out(4)
  call fun(d(cols), out)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    rng = np.random.default_rng(2)
    d = rng.random(10).astype(np.float64)
    cols = np.array([3, 6, 1, 9], dtype=np.int32)
    out = np.zeros(4, dtype=np.float64)
    sdfg(d=d, cols=cols, out=out)
    np.testing.assert_allclose(out, d[cols - 1] * 3.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# SCATTER  --  ``d(cols) = source`` shape patterns
# ---------------------------------------------------------------------------


def test_scatter_into_local_array(tmp_path: Path):
    """d(cols) = source -- writes 4 elements at cols, leaves the rest untouched; exercises hlfir-expand-vector-subscript-scatter."""
    src = """
subroutine main(d, cols, source)
  double precision, intent(inout) :: d(8)
  integer,          intent(in)    :: cols(4)
  double precision, intent(in)    :: source(4)
  d(cols) = source
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(3)
    d = rng.random(8).astype(np.float64)
    d_orig = d.copy()
    cols = np.array([2, 4, 6, 8], dtype=np.int32)  # 1-based
    source = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
    sdfg(d=d, cols=cols, source=source)
    expected = d_orig.copy()
    expected[cols - 1] = source
    np.testing.assert_allclose(d, expected, rtol=1e-12)


def test_scatter_overwrites_specific_indices(tmp_path: Path):
    """Sanity: d(cols) only touches the listed positions; everything else is preserved."""
    src = """
subroutine main(d, cols, source)
  double precision, intent(inout) :: d(6)
  integer,          intent(in)    :: cols(2)
  double precision, intent(in)    :: source(2)
  d(cols) = source
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    d = np.array([10., 20., 30., 40., 50., 60.], dtype=np.float64)
    cols = np.array([2, 5], dtype=np.int32)
    source = np.array([-1.0, -2.0], dtype=np.float64)
    sdfg(d=d, cols=cols, source=source)
    expected = np.array([10., -1., 30., 40., -2., 60.], dtype=np.float64)
    np.testing.assert_allclose(d, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# GATHER + SCATTER combined  --  ``a(b) = c(d)`` round-trip
# ---------------------------------------------------------------------------


def test_scatter_with_symbolic_extent(tmp_path: Path):
    """d(cols) = source with runtime-symbolic length n.  Scatter pass uses cols/source directly (no temp needed for a contiguous source); n controls the loop trip count, not the symbol count, so DaCe's "no array of symbols" invariant holds."""
    src = """
subroutine main(n, d, cols, source)
  integer, intent(in)             :: n
  double precision, intent(inout) :: d(2 * n)
  integer,          intent(in)    :: cols(n)
  double precision, intent(in)    :: source(n)
  d(cols) = source
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 4
    d = np.array([10., 20., 30., 40., 50., 60., 70., 80.], dtype=np.float64)
    d_orig = d.copy()
    cols = np.array([1, 3, 5, 7], dtype=np.int32)
    source = np.array([100., 200., 300., 400.], dtype=np.float64)
    sdfg(n=n, d=d, cols=cols, source=source)
    expected = d_orig.copy()
    expected[cols - 1] = source
    np.testing.assert_allclose(d, expected, rtol=1e-12)


def test_scatter_with_symbolic_extent_two_index_arrays(tmp_path: Path):
    """a(cols2) = c(cols1) with symbolic n and two distinct index arrays.  Source is an hlfir.expr (gather of c via cols1) so the fused single-loop path fires; disjoint arrays make that correct."""
    src = """
subroutine main(n, a, c, cols1, cols2)
  integer, intent(in)             :: n
  double precision, intent(inout) :: a(2 * n)
  double precision, intent(in)    :: c(2 * n)
  integer,          intent(in)    :: cols1(n)
  integer,          intent(in)    :: cols2(n)
  a(cols2) = c(cols1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 4
    rng = np.random.default_rng(7)
    a = rng.random(8).astype(np.float64)
    a_orig = a.copy()
    c = rng.random(8).astype(np.float64)
    cols1 = np.array([1, 3, 5, 7], dtype=np.int32)
    cols2 = np.array([2, 4, 6, 8], dtype=np.int32)
    sdfg(n=n, a=a, c=c, cols1=cols1, cols2=cols2)
    expected = a_orig.copy()
    expected[cols2 - 1] = c[cols1 - 1]
    np.testing.assert_allclose(a, expected, rtol=1e-12)


def test_gather_then_scatter_roundtrip(tmp_path: Path):
    """a(write_idx) = c(read_idx) -- gather from c via read_idx, scatter into a via write_idx; disjoint arrays, no aliasing."""
    src = """
subroutine main(a, c, read_idx, write_idx)
  double precision, intent(inout) :: a(8)
  double precision, intent(in)    :: c(8)
  integer,          intent(in)    :: read_idx(4)
  integer,          intent(in)    :: write_idx(4)
  a(write_idx) = c(read_idx)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    rng = np.random.default_rng(4)
    a = rng.random(8).astype(np.float64)
    a_orig = a.copy()
    c = rng.random(8).astype(np.float64)
    read_idx = np.array([1, 3, 5, 7], dtype=np.int32)
    write_idx = np.array([2, 4, 6, 8], dtype=np.int32)
    sdfg(a=a, c=c, read_idx=read_idx, write_idx=write_idx)
    expected = a_orig.copy()
    expected[write_idx - 1] = c[read_idx - 1]
    np.testing.assert_allclose(a, expected, rtol=1e-12)


def test_gather_scatter_aliasing_same_array(tmp_path: Path):
    """a(write_idx) = a(read_idx), same array both sides.  Fortran 2003 requires the RHS fully evaluated before any LHS write; a naive fused read-write loop breaks when read_idx/write_idx overlap (write at k clobbers a value a later iteration reads).  Pins the contract: bridge must materialise the gather to a temp first, then scatter -- same as overlapping section assignments like a(:) = a(2:)."""
    src = """
subroutine main(a, read_idx, write_idx)
  double precision, intent(inout) :: a(8)
  integer,          intent(in)    :: read_idx(4)
  integer,          intent(in)    :: write_idx(4)
  a(write_idx) = a(read_idx)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.array([10., 20., 30., 40., 50., 60., 70., 80.], dtype=np.float64)
    a_orig = a.copy()
    # Overlapping: positions 3,4 are BOTH read and written.
    read_idx = np.array([1, 2, 3, 4], dtype=np.int32)
    write_idx = np.array([3, 4, 5, 6], dtype=np.int32)
    sdfg(a=a, read_idx=read_idx, write_idx=write_idx)
    # Correct (Fortran-semantics) result: a_orig[0..3] copied into a[2..5].
    expected = a_orig.copy()
    expected[write_idx - 1] = a_orig[read_idx - 1]
    np.testing.assert_allclose(a,
                               expected,
                               rtol=1e-12,
                               err_msg="gather/scatter on the same array must materialise the "
                               "RHS to a temp first; check that hlfir-expand-vector-subscript-scatter "
                               "emits a separate gather loop into a transient before the "
                               "scatter loop.")


def test_gather_scatter_aliasing_same_array_symbolic(tmp_path: Path):
    """Same aliased pattern as test_gather_scatter_aliasing_same_array but with a symbolic extent -- drives the dynamic-extent scatter-source temp path (fir.alloca array<?xT> shaped by the runtime gather extent)."""
    src = """
subroutine main(n, a, read_idx, write_idx)
  integer,          intent(in)    :: n
  double precision, intent(inout) :: a(2 * n)
  integer,          intent(in)    :: read_idx(n)
  integer,          intent(in)    :: write_idx(n)
  a(write_idx) = a(read_idx)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 4
    a = np.array([10., 20., 30., 40., 50., 60., 70., 80.], dtype=np.float64)
    a_orig = a.copy()
    read_idx = np.array([1, 2, 3, 4], dtype=np.int32)
    write_idx = np.array([3, 4, 5, 6], dtype=np.int32)
    sdfg(n=n, a=a, read_idx=read_idx, write_idx=write_idx)
    expected = a_orig.copy()
    expected[write_idx - 1] = a_orig[read_idx - 1]
    np.testing.assert_allclose(a, expected, rtol=1e-12)
