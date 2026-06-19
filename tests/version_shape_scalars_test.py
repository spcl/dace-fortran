"""Shape-variable SSA-versioning  --  ``hlfir-version-shape-scalars``.

A local integer scalar reassigned between two ALLOCATEs would otherwise be a
MUTABLE shape symbol shared by both arrays:

    m = base * 2
    ALLOCATE(x(m))     ! x's extent is m's FIRST value
    m = m + 3
    ALLOCATE(y(m))     ! y's extent is m's SECOND value
    x = x * 2.0        ! whole-array op maps over x's shape symbol

With a single mutable ``m`` symbol the whole-array op runs ``m = base*2+3``
iterations over an array allocated to ``base*2`` -> heap corruption.  The pass
splits ``m`` into one immutable version per straight-line store (``m``,
``m_2``), so ``x`` binds to ``m`` and ``y`` to ``m_2``.

THE PRECISE HAZARD is a reassignment ORDERED AFTER an array was allocated from
the scalar (the array's live extent symbol then mutates).  Two common shapes
are therefore NOT hazards and must be LEFT UNTOUCHED:

  * accumulate-then-allocate-once -- ``nij = 0; do .. nij = nij + ..;
    ALLOCATE(qgm(.., nij))`` -- every reassignment precedes the single
    allocation, so the extent is frozen for the array's lifetime; and
  * data-access scalars -- a loop bound (``do jb = .., i_endblk``) or a
    subscript (``z(1:jb)``) mints a trip-count / section ``fir.shape`` but never
    an ``fir.allocmem`` extent, so it is not an array shape at all.  (Loop
    iterators in particular are never versioned.)

ONLY a post-allocation reassignment is acted on: straight-line -> versioned;
inside a loop / conditional branch -> REFUSED with a clear message.
"""
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


# ---------------------------------------------------------------------------
# Straight-line versioning  --  the patterns we SUPPORT
# ---------------------------------------------------------------------------
def test_two_versions_whole_array_op_no_corruption(tmp_path):
    """``alloc x(m); m=m+3; alloc y(m); x = x*2`` -- the whole-array op over
    x's shape must map over x's OWN extent (m's first value), not the
    reassigned one.  Pre-fix this corrupted the heap; here it is exact and
    ``x``/``y`` carry DISTINCT immutable shape symbols."""
    import numpy as np

    src = """
subroutine vss_two(base, xsum, xlast)
  implicit none
  integer, intent(in) :: base
  real(8), intent(out) :: xsum, xlast
  integer :: m, i
  real(8), allocatable :: x(:), y(:)
  m = base * 2
  allocate(x(m))
  do i = 1, m
    x(i) = real(i, 8)
  end do
  m = m + 3
  allocate(y(m))
  y = 9.0d0
  x = x * 2.0d0
  xsum = sum(x)
  xlast = x(base * 2)
end subroutine
"""
    builder = build_sdfg(src, tmp_path, name="vss_two", entry="vss_two")
    sdfg = builder.build()
    # x sizes from m (version 1), y from m_2 (version 2) -- both immutable.
    assert str(sdfg.arrays["x"].shape[0]) == "m"
    assert str(sdfg.arrays["y"].shape[0]) == "m_2"
    assert "m_2" in sdfg.symbols

    sdfg.name = "vss_two"
    compiled = sdfg.compile()
    base = 4
    xsum = np.array([0.0])
    xlast = np.array([0.0])
    compiled(base=np.int32(base), xsum=xsum, xlast=xlast)
    # x(i) = i for i in 1..8, then x = x*2 -> sum = 2*(1+..+8) = 72, x(8) = 16.
    np.testing.assert_allclose(xsum[0], 2.0 * sum(range(1, base * 2 + 1)))
    np.testing.assert_allclose(xlast[0], 2.0 * base * 2)


def test_three_versions(tmp_path):
    """Three straight-line reassignments -> ``m``, ``m_2``, ``m_3``; each array
    binds to the version live at its own ALLOCATE."""
    src = """
subroutine vss_three(n, sa, sb, sc)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: sa, sb, sc
  integer :: m
  real(8), allocatable :: a(:), b(:), c(:)
  m = n
  allocate(a(m))
  m = m + 1
  allocate(b(m))
  m = m + 1
  allocate(c(m))
  a = 0.0d0
  b = 0.0d0
  c = 0.0d0
  sa = size(a)
  sb = size(b)
  sc = size(c)
end subroutine
"""
    import numpy as np

    builder = build_sdfg(src, tmp_path, name="vss_three", entry="vss_three")
    sdfg = builder.build()
    assert str(sdfg.arrays["a"].shape[0]) == "m"
    assert str(sdfg.arrays["b"].shape[0]) == "m_2"
    assert str(sdfg.arrays["c"].shape[0]) == "m_3"

    sdfg.name = "vss_three"
    compiled = sdfg.compile()
    n = 5
    sa, sb, sc = (np.array([0], dtype=np.int32) for _ in range(3))
    compiled(n=np.int32(n), sa=sa, sb=sb, sc=sc)
    assert (int(sa[0]), int(sb[0]), int(sc[0])) == (n, n + 1, n + 2)


def test_single_assignment_not_versioned(tmp_path):
    """A shape scalar assigned exactly once is left untouched -- no ``m_2``."""
    src = """
subroutine vss_one(n, s)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: s
  integer :: m
  real(8), allocatable :: x(:)
  m = n * 2
  allocate(x(m))
  x = 1.0d0
  s = size(x)
end subroutine
"""
    builder = build_sdfg(src, tmp_path, name="vss_one", entry="vss_one")
    sdfg = builder.build()
    assert "m_2" not in sdfg.symbols
    assert str(sdfg.arrays["x"].shape[0]) == "m"


# ---------------------------------------------------------------------------
# Control: a reassigned NON-shape scalar is NOT touched / NOT refused
# ---------------------------------------------------------------------------
def test_loop_accumulator_not_refused(tmp_path):
    """A scalar reassigned inside a loop but NEVER used as an array shape (a
    plain accumulator) must build cleanly -- the pass only acts on shape
    scalars."""
    import numpy as np

    src = """
subroutine vss_acc(n, r)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: r
  integer :: acc, i
  acc = 0
  do i = 1, n
    acc = acc + i
  end do
  r = acc
end subroutine
"""
    builder = build_sdfg(src, tmp_path, name="vss_acc", entry="vss_acc")
    sdfg = builder.build()
    sdfg.name = "vss_acc"
    compiled = sdfg.compile()
    n = 6
    r = np.array([0], dtype=np.int32)
    compiled(n=np.int32(n), r=r)
    assert int(r[0]) == sum(range(1, n + 1))


# ---------------------------------------------------------------------------
# Refusal  --  the GENUINE hazard: a reassignment AFTER an allocation, in a
# loop / branch, so the live array's extent symbol mutates and cannot be named
# ---------------------------------------------------------------------------
def test_refuse_post_alloc_reassign_in_loop(tmp_path, capfd):
    """``x`` is allocated from ``m``; then INSIDE a loop ``m`` is reassigned and
    another array is allocated from the new value.  ``x``'s extent symbol ``m``
    mutates while ``x`` is live -- the live value is ambiguous (which
    iteration?), so the pass refuses rather than emit a corrupting shape."""
    src = """
subroutine vss_loop(n, k, s)
  implicit none
  integer, intent(in) :: n, k
  integer, intent(out) :: s
  integer :: m, i
  real(8), allocatable :: x(:), y(:)
  m = n
  allocate(x(m))
  do i = 1, k
    m = m + 1
    allocate(y(m))
    y = 0.0d0
    deallocate(y)
  end do
  x = x * 2.0d0
  s = size(x)
end subroutine
"""
    with pytest.raises(RuntimeError) as exc:
        build_sdfg(src, tmp_path, name="vss_loop", entry="vss_loop").build()
    assert "pipeline failed" in str(exc.value)
    err = capfd.readouterr().err
    assert "shape variable 'm'" in err
    assert "AFTER an array was allocated from it" in err
    assert "Hoist the size to a single assignment" in err


def test_refuse_post_alloc_reassign_in_branch(tmp_path, capfd):
    """``x`` is allocated from ``m``; then inside an ``if`` branch ``m`` is
    reassigned and another array is allocated.  The value of ``x``'s extent
    symbol depends on the branch taken -- refused."""
    src = """
subroutine vss_branch(n, c, s)
  implicit none
  integer, intent(in) :: n
  logical, intent(in) :: c
  integer, intent(out) :: s
  integer :: m
  real(8), allocatable :: x(:), y(:)
  m = n
  allocate(x(m))
  if (c) then
    m = m + 1
    allocate(y(m))
    y = 0.0d0
  end if
  x = x * 2.0d0
  s = size(x)
end subroutine
"""
    with pytest.raises(RuntimeError) as exc:
        build_sdfg(src, tmp_path, name="vss_branch", entry="vss_branch").build()
    assert "pipeline failed" in str(exc.value)
    err = capfd.readouterr().err
    assert "shape variable 'm'" in err
    assert "AFTER an array was allocated from it" in err


# ---------------------------------------------------------------------------
# False positives  --  patterns that LOOK like a reassigned shape scalar but
# carry no cross-version hazard, so the pass must leave them ALONE (build OK,
# no ``m_2`` minted, no refusal).  These regression-guard the over-refusal that
# rejected ICON velocity (loop bounds) and QE vexx (the ``nij`` accumulate).
# ---------------------------------------------------------------------------
def test_accumulate_then_allocate_once_not_refused(tmp_path):
    """QE's ``nij`` pattern: a scalar accumulated INSIDE a loop (and a branch),
    then used ONCE to size an array AFTER the loop.  Every reassignment precedes
    the single allocation, so the extent is frozen for the array's lifetime --
    this is the dominant HPC sizing idiom and must build untouched."""
    import numpy as np

    src = """
subroutine vss_accum(ntyp, nh, tot)
  implicit none
  integer, intent(in) :: ntyp
  integer, intent(in) :: nh(ntyp)
  integer, intent(out) :: tot
  integer :: nij, nt
  real(8), allocatable :: qgm(:)
  nij = 0
  do nt = 1, ntyp
    if (nh(nt) > 0) nij = nij + (nh(nt) * (nh(nt) + 1)) / 2
  end do
  allocate(qgm(nij))
  qgm = 1.0d0
  tot = int(sum(qgm))
end subroutine
"""
    builder = build_sdfg(src, tmp_path, name="vss_accum", entry="vss_accum")
    sdfg = builder.build()
    assert "nij_2" not in sdfg.symbols  # not versioned -- no hazard

    sdfg.name = "vss_accum"
    compiled = sdfg.compile()
    ntyp = 3
    nh = np.array([2, 3, 4], dtype=np.int32)
    tot = np.array([0], dtype=np.int32)
    compiled(ntyp=np.int32(ntyp), nh=nh, tot=tot)
    expected = sum(h * (h + 1) // 2 for h in nh)  # qgm has that many 1.0s
    assert int(tot[0]) == expected


def test_loop_iterator_subscript_not_refused(tmp_path):
    """A loop iterator (``jb``) used as a section subscript ``z(1:jb)`` mints a
    section-length ``fir.shape`` but never an ALLOCATE extent.  ``z`` is sized
    by a separate parameter ``n``.  The iterator must NOT be read as a shape
    (loop iterators are never versioned) and the kernel must build."""
    import numpy as np

    src = """
subroutine vss_iter(n, r)
  implicit none
  integer, intent(in) :: n
  real(8), intent(out) :: r
  integer :: jb
  real(8), allocatable :: z(:)
  allocate(z(n))
  z = 0.0d0
  do jb = 1, n
    z(1:jb) = z(1:jb) + 1.0d0
  end do
  r = sum(z)
end subroutine
"""
    builder = build_sdfg(src, tmp_path, name="vss_iter", entry="vss_iter")
    sdfg = builder.build()
    assert "jb_2" not in sdfg.symbols  # iterator not versioned
    assert str(sdfg.arrays["z"].shape[0]) == "n"  # z sized by n, not jb

    sdfg.name = "vss_iter"
    compiled = sdfg.compile()
    n = 5
    r = np.array([0.0])
    compiled(n=np.int32(n), r=r)
    # z(i) gets +1 once for every jb >= i, i.e. z(i) = n - i + 1 -> sum = n(n+1)/2.
    np.testing.assert_allclose(r[0], n * (n + 1) / 2)


def test_loop_bound_only_reassigned_not_refused(tmp_path):
    """An ICON-style block bound (``i_endblk``) reassigned in a STRAIGHT LINE but
    used ONLY as a loop range -- the arrays are sized by a separate ``nproma``.
    A data-access (range) use feeds no ALLOCATE extent, so the reassigned bound
    must NOT be versioned or refused."""
    import numpy as np

    src = """
subroutine vss_bound(nproma, nblk, r)
  implicit none
  integer, intent(in) :: nproma, nblk
  real(8), intent(out) :: r
  integer :: i_startblk, i_endblk, jb, jc
  real(8), allocatable :: z(:,:)
  allocate(z(nproma, nblk))
  z = 0.0d0
  i_startblk = 1
  i_endblk = nblk
  i_endblk = min(i_endblk, nblk)
  do jb = i_startblk, i_endblk
    do jc = 1, nproma
      z(jc, jb) = real(jc + jb, 8)
    end do
  end do
  r = sum(z)
end subroutine
"""
    builder = build_sdfg(src, tmp_path, name="vss_bound", entry="vss_bound")
    sdfg = builder.build()
    assert "i_endblk_2" not in sdfg.symbols  # range bound not versioned
    assert str(sdfg.arrays["z"].shape[0]) == "nproma"

    sdfg.name = "vss_bound"
    compiled = sdfg.compile()
    nproma, nblk = 4, 3
    r = np.array([0.0])
    compiled(nproma=np.int32(nproma), nblk=np.int32(nblk), r=r)
    expected = sum(jc + jb for jb in range(1, nblk + 1) for jc in range(1, nproma + 1))
    np.testing.assert_allclose(r[0], expected)
