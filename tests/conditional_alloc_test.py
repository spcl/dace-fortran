"""Conditional and sequential ALLOCATE of the same allocatable.

Two distinct patterns the bridge must tell apart:

* **Conditional ALLOCATE** -- ``IF (c) ALLOCATE(a(n)) ELSE ALLOCATE(a(m))``
  (and the nested ``IF/ELSEIF/.../ELSE`` form).  The allocate sites are in
  mutually-exclusive branches and store to the *same* descriptor, so ``a``
  must stay ONE transient whose extent is a branch-dependent symbol
  (``a_d0`` is assigned the branch's extent; the assignments merge at the
  IF join).  Versioning it into ``a_alloc1`` would split it into two
  buffers and bind post-IF reads statically to one -- wrong.

* **Sequential re-allocation** -- ``ALLOCATE(a(n)); ...; DEALLOCATE(a);
  ALLOCATE(a(m))``.  Here the bridge DOES want a fresh buffer with a fresh
  name (``a_alloc1``) per ALLOCATE site; reads after the K-th site route to
  the K-th buffer.

Each kernel is checked against an f2py reference on several inputs.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _run(tmp_path, src, cases, argnames):
    """Build ``src`` through the bridge and f2py; run both on each
    ``cases`` tuple (mapped to ``argnames``); assert the ``out(10)``
    arrays match."""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="probe", entry="probe_mod::probe").build()
    mod = f2py_compile(src, tmp_path / "ref", f"ca_ref_{tmp_path.name}")
    for args in cases:
        s = np.zeros(10, dtype=np.float64)
        r = np.zeros(10, dtype=np.float64)
        kw = {k: np.int32(v) for k, v in zip(argnames, args)}
        sdfg(out=s, **kw)
        # ``probe`` now lives in ``probe_mod`` so f2py exposes it under
        # the module's submodule namespace.
        mod.probe_mod.probe(*args, r)
        np.testing.assert_array_equal(s, r)
    return sdfg


def _size_loop_parts(hop):
    """Fortran fragments for a ``DO i = 1, SIZE(a)`` loop bound in the two
    equivalent spellings the bridge must both handle: the direct form, and
    the ``sz = SIZE(a)`` scalar hop.  Returns ``(declaration, loop_header)``;
    the caller appends the loop body and ``end do``.  Both are exercised
    (``@pytest.mark.parametrize``) so a regression in either is caught."""
    if hop:
        return "integer :: i, sz", "  sz = size(a)\n  do i = 1, sz"
    return "integer :: i", "  do i = 1, size(a)"


_SIZE_LOOP_FORMS = pytest.mark.parametrize("hop", [False, True], ids=["direct", "hop"])


@_SIZE_LOOP_FORMS
def test_cond_alloc_if_else(tmp_path, hop):
    """``IF (c) ALLOCATE(a(n)) ELSE ALLOCATE(a(m))`` -- one transient with a
    branch-dependent extent symbol; ``size(a)`` is ``n`` or ``m`` per branch."""
    decl, loop = _size_loop_parts(hop)
    src = f"""
module probe_mod
  implicit none
contains
subroutine probe(cond, n, m, out)
  implicit none
  integer, intent(in) :: cond, n, m
  real(8), intent(inout) :: out(10)
  real(8), allocatable :: a(:)
  {decl}
  if (cond > 0) then
    allocate(a(n))
  else
    allocate(a(m))
  end if
{loop}
    a(i) = real(i, 8)
  end do
  out(1) = a(1)
  out(2) = real(size(a), 8)
  deallocate(a)
end subroutine probe
end module probe_mod
"""
    sdfg = _run(tmp_path, src, [(1, 5, 3), (0, 5, 3), (1, 2, 7), (0, 2, 7)],
                ["cond", "n", "m"])
    # Single array, not versioned, with the branch-extent symbol shape.
    assert "a" in sdfg.arrays and "a_alloc1" not in sdfg.arrays
    assert str(sdfg.arrays["a"].shape) == "(a_d0,)"


@_SIZE_LOOP_FORMS
def test_cond_alloc_if_elif_else(tmp_path, hop):
    """Nested ``IF/ELSEIF/ELSEIF/ELSE`` (four mutually-exclusive branches)."""
    decl, loop = _size_loop_parts(hop)
    src = f"""
module probe_mod
  implicit none
contains
subroutine probe(sel, n1, n2, n3, n4, out)
  implicit none
  integer, intent(in) :: sel, n1, n2, n3, n4
  real(8), intent(inout) :: out(10)
  real(8), allocatable :: a(:)
  {decl}
  if (sel == 1) then
    allocate(a(n1))
  else if (sel == 2) then
    allocate(a(n2))
  else if (sel == 3) then
    allocate(a(n3))
  else
    allocate(a(n4))
  end if
{loop}
    a(i) = real(i, 8)
  end do
  out(1) = real(size(a), 8)
  deallocate(a)
end subroutine probe
end module probe_mod
"""
    sdfg = _run(tmp_path, src,
                [(1, 5, 3, 7, 2), (2, 5, 3, 7, 2), (3, 5, 3, 7, 2), (4, 5, 3, 7, 2)],
                ["sel", "n1", "n2", "n3", "n4"])
    assert "a_alloc1" not in sdfg.arrays
    assert str(sdfg.arrays["a"].shape) == "(a_d0,)"


def test_cond_alloc_single_branch(tmp_path):
    """``IF (c) THEN; ALLOCATE(a(n)); ...; ENDIF`` -- a single alloc site;
    ``a`` is used only on the allocated path."""
    src = """
module probe_mod
  implicit none
contains
subroutine probe(sel, n, out)
  implicit none
  integer, intent(in) :: sel, n
  real(8), intent(inout) :: out(10)
  real(8), allocatable :: a(:)
  integer :: i
  if (sel > 0) then
    allocate(a(n))
    do i = 1, n
      a(i) = real(i, 8)
    end do
    out(1) = a(n)
    deallocate(a)
  end if
end subroutine probe
end module probe_mod
"""
    _run(tmp_path, src, [(1, 6), (0, 6)], ["sel", "n"])


def test_realloc_sequential_new_buffer(tmp_path):
    """``ALLOCATE(a(n)); ...; DEALLOCATE(a); ALLOCATE(a(m)); ...`` -- the
    sequential re-allocation gets a fresh buffer (``a_alloc1``); this is NOT
    the conditional case and must stay versioned."""
    src = """
module probe_mod
  implicit none
contains
subroutine probe(n, m, out)
  implicit none
  integer, intent(in) :: n, m
  real(8), intent(inout) :: out(10)
  real(8), allocatable :: a(:)
  integer :: i
  allocate(a(n))
  do i = 1, n
    a(i) = real(i, 8)
  end do
  out(1) = a(n)
  deallocate(a)
  allocate(a(m))
  do i = 1, m
    a(i) = real(2*i, 8)
  end do
  out(2) = a(m)
  deallocate(a)
end subroutine probe
end module probe_mod
"""
    sdfg = _run(tmp_path, src, [(5, 3), (4, 8)], ["n", "m"])
    # Sequential realloc -> a fresh versioned buffer (new name).
    assert "a_alloc1" in sdfg.arrays


def test_realloc_chain_four_buffers(tmp_path):
    """``ALLOC; DEALLOC`` repeated four times -> four versioned buffers
    (``a``, ``a_alloc1``, ``a_alloc2``, ``a_alloc3``), one per epoch."""
    src = """
module probe_mod
  implicit none
contains
subroutine probe(n1, n2, n3, n4, out)
  implicit none
  integer, intent(in) :: n1, n2, n3, n4
  real(8), intent(inout) :: out(10)
  real(8), allocatable :: a(:)
  integer :: i
  allocate(a(n1)); do i=1,n1; a(i)=real(i,8);   end do; out(1)=a(n1); deallocate(a)
  allocate(a(n2)); do i=1,n2; a(i)=real(2*i,8); end do; out(2)=a(n2); deallocate(a)
  allocate(a(n3)); do i=1,n3; a(i)=real(3*i,8); end do; out(3)=a(n3); deallocate(a)
  allocate(a(n4)); do i=1,n4; a(i)=real(4*i,8); end do; out(4)=a(n4); deallocate(a)
end subroutine probe
end module probe_mod
"""
    sdfg = _run(tmp_path, src, [(5, 3, 7, 2), (1, 9, 4, 6)],
                ["n1", "n2", "n3", "n4"])
    assert {"a", "a_alloc1", "a_alloc2", "a_alloc3"} <= set(sdfg.arrays)


@_SIZE_LOOP_FORMS
def test_cond_alloc_then_realloc(tmp_path, hop):
    """Conditional ALLOCATE, used (via ``size``), deallocated, then
    re-ALLOCATEd to a new size -- conditional + sequential realloc on one
    array.  Buffer-class grouping: then/else -> the conditional buffer
    ``a`` (branch extent ``a_d0``); the realloc -> ``a_alloc1`` (extent
    ``a_alloc1_d0`` so ``size`` resolves on the versioned buffer too)."""
    decl, loop = _size_loop_parts(hop)
    src = f"""
module probe_mod
  implicit none
contains
subroutine probe(cond, n, m, k, out)
  implicit none
  integer, intent(in) :: cond, n, m, k
  real(8), intent(inout) :: out(10)
  real(8), allocatable :: a(:)
  {decl}
  if (cond > 0) then
    allocate(a(n))
  else
    allocate(a(m))
  end if
{loop}
    a(i) = real(i, 8)
  end do
  out(1) = real(size(a), 8)
  out(2) = a(1)
  deallocate(a)
  allocate(a(k))
  do i = 1, k
    a(i) = real(2*i, 8)
  end do
  out(3) = real(size(a), 8)
  out(4) = a(k)
  deallocate(a)
end subroutine probe
end module probe_mod
"""
    sdfg = _run(tmp_path, src, [(1, 5, 3, 4), (0, 5, 3, 4)],
                ["cond", "n", "m", "k"])
    assert str(sdfg.arrays["a"].shape) == "(a_d0,)"   # conditional buffer
    assert "a_alloc1" in sdfg.arrays                  # realloc buffer


@_SIZE_LOOP_FORMS
def test_realloc_chain_inside_if(tmp_path, hop):
    """A realloc chain inside one ``IF`` branch (``alloc; use; dealloc;
    alloc``) with a single alloc in the other branch.  The post-``IF`` use
    is reached by the then-branch's LAST buffer and the else buffer -- they
    merge into one conditional buffer; the then-branch's first (freed)
    buffer is a separate transient."""
    decl, loop = _size_loop_parts(hop)
    src = f"""
module probe_mod
  implicit none
contains
subroutine probe(cond, n, n2, m, out)
  implicit none
  integer, intent(in) :: cond, n, n2, m
  real(8), intent(inout) :: out(10)
  real(8), allocatable :: a(:)
  {decl}
  if (cond > 0) then
    allocate(a(n))
    a(1) = 99.0d0
    deallocate(a)
    allocate(a(n2))
  else
    allocate(a(m))
  end if
{loop}
    a(i) = real(i, 8)
  end do
  out(1) = real(size(a), 8)
  out(2) = a(1)
  deallocate(a)
end subroutine probe
end module probe_mod
"""
    sdfg = _run(tmp_path, src, [(1, 5, 8, 3), (0, 5, 8, 3), (1, 2, 6, 9)],
                ["cond", "n", "n2", "m"])
    # post-IF buffer is the merged then-last/else class (conditional);
    # the then's first (freed) buffer is a separate transient.
    assert "a_alloc1" in sdfg.arrays


@_SIZE_LOOP_FORMS
def test_size_of_concrete_base_buffer(tmp_path, hop):
    """``SIZE(a)`` on a plain ``allocate(a(n))`` base buffer.  ``SIZE``
    lowers to ``fir.box_dims`` which the bridge renders as the extent symbol
    ``a_d0``; binding ``a_d0 = n`` at the ALLOCATE site keeps it from leaking
    onto the program signature as a free symbol (it was an unbound ``a_d0``
    -> ``KeyError`` before)."""
    decl, loop = _size_loop_parts(hop)
    src = f"""
module probe_mod
  implicit none
contains
subroutine probe(n, out)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: out(10)
  real(8), allocatable :: a(:)
  {decl}
  allocate(a(n))
{loop}
    a(i) = real(i, 8)
  end do
  out(1) = a(1)
  out(2) = a(n)
  out(3) = real(size(a), 8)
  deallocate(a)
end subroutine probe
end module probe_mod
"""
    _run(tmp_path, src, [(5,), (3,), (8,)], ["n"])
