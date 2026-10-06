"""Whole-array copy -> ``CopyLibraryNode`` and zero-fill -> ``FillLibraryNode``.

Exercises the two ``hlfir.assign`` shapes that skip the tasklet/loop path
and go straight to library nodes on FaCe.  Compared numerically against
the gfortran/f2py-compiled reference on seeded random input.
"""

from pathlib import Path

import numpy as np

from tests._util import build_sdfg, f2py_compile

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "copy_memset.f90"


def test_copy_and_memset_numerical(tmp_path):
    mod = f2py_compile(_SRC_PATH, tmp_path / "ref", "copy_and_memset_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(
        _SRC_PATH.read_text(), sdfg_dir, name="copy_and_memset", pipeline="hlfir-propagate-shapes"
    ).build()
    sdfg.validate()

    rng = np.random.default_rng(9)
    n = 17
    a = rng.standard_normal(n)

    # Start b and c with nonzero values to catch cases where the copy or
    # memset didn't run.
    b_sdfg = np.full(n, 7.25, dtype=np.float64)
    c_sdfg = np.full(n, 7.25, dtype=np.float64)

    sdfg(a=np.ascontiguousarray(a), b=b_sdfg, c=c_sdfg, n=n)

    # Reference  --  gfortran-compiled whole-array assign / zero-fill.
    b_ref = np.full(n, 7.25, order="F", dtype=np.float64)
    c_ref = np.full(n, 7.25, order="F", dtype=np.float64)
    mod.copy_and_memset(np.asfortranarray(a), b_ref, c_ref)

    np.testing.assert_allclose(b_sdfg, b_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_sdfg, c_ref, rtol=1e-12, atol=1e-12)


def test_copy_and_memset_structure(tmp_path):
    """The SDFG should carry exactly one CopyLibraryNode (for ``b = a``)
    and one FillLibraryNode (for ``c = 0.0``), rather than tasklet-and-
    loop decompositions."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(
        _SRC_PATH.read_text(), sdfg_dir, name="copy_and_memset", pipeline="hlfir-propagate-shapes"
    ).build()

    from dace.libraries.standard.nodes import CopyLibraryNode, FillLibraryNode
    from dace.sdfg.state import LoopRegion, SDFGState

    def iter_states(region):
        for n in region.nodes():
            if isinstance(n, LoopRegion):
                yield from iter_states(n)
            elif isinstance(n, SDFGState):
                yield n

    nodes = [n for s in iter_states(sdfg) for n in s.nodes()]
    copies = [n for n in nodes if isinstance(n, CopyLibraryNode)]
    memsets = [n for n in nodes if isinstance(n, FillLibraryNode)]
    assert len(copies) == 1, f"expected 1 CopyLibraryNode, got {len(copies)}"
    assert len(memsets) == 1, f"expected 1 FillLibraryNode, got {len(memsets)}"


_NESTED_ALLOCATABLE_SRC = """
module nested_fill
  implicit none
contains
  subroutine fill_diag(n, mat)
    integer, intent(in) :: n
    real(kind=8), intent(out) :: mat(n, n)
    integer :: i
    mat = 0.0d0
    do i = 1, n
      mat(i, i) = i
    end do
  end subroutine
  subroutine owner(n, total)
    integer, intent(in) :: n
    real(kind=8), intent(out) :: total
    real(kind=8), allocatable :: buf(:, :)
    allocate(buf(n, n))
    call fill_diag(n, buf)
    total = sum(buf)
    deallocate(buf)
  end subroutine
  subroutine top(n, total)
    integer, intent(in) :: n
    real(kind=8), intent(out) :: total
    call owner(n, total)
  end subroutine
end module
"""


def test_memset_of_a_dummy_bound_to_an_inlined_callees_allocatable(tmp_path):
    """``mat = 0`` on an explicit-shape dummy whose actual is a local allocatable of a routine that is itself inlined
    (not the entry) fills that allocatable: the dummy is named after its root declare (QE matcalc_gpu's ``mat`` over
    vexxace_gamma_gpu's ``rmexx_d`` raised ``KeyError: 'mat'``)."""
    sdfg = build_sdfg(_NESTED_ALLOCATABLE_SRC, tmp_path, name="nested_fill", entry="nested_fill::top").build()
    sdfg.validate()
    n = 6
    total = np.full(1, -1.0)
    sdfg(n=n, total=total)
    assert total[0] == n * (n + 1) / 2


_GLOBAL_STRUCT_MEMBER_SRC = """
module desc_types
  implicit none
  type desc_t
    integer :: nproc = 1
    integer, allocatable :: counts(:)
  end type
contains
  subroutine desc_allocate(desc, n, total)
    type(desc_t), intent(inout) :: desc
    integer, intent(in) :: n
    integer, intent(out) :: total
    desc%nproc = n
    allocate(desc%counts(desc%nproc))
    desc%counts = 0
    desc%counts(1) = n
    total = sum(desc%counts)
  end subroutine
  subroutine desc_init(dfft, n, total)
    type(desc_t), intent(inout) :: dfft
    integer, intent(in) :: n
    integer, intent(out) :: total
    call desc_allocate(dfft, n, total)
  end subroutine
end module
module owner_mod
  use desc_types
  implicit none
  type(desc_t) :: global_desc
contains
  subroutine setup(n, total)
    integer, intent(in) :: n
    integer, intent(out) :: total
    call desc_init(global_desc, n, total)
  end subroutine
end module
"""


def test_memset_of_a_module_struct_member_through_two_inlined_dummies(tmp_path):
    """``desc % counts = 0`` where ``desc`` aliases ``dfft`` aliases the module global ``global_desc`` (QE's
    ``fft_type_init(dffts_exx)`` -> ``fft_type_allocate(desc)``): the member reached only through the two-dummy chain
    still gets its descriptor (``KeyError: 'dffts_exx_nr2p'`` before)."""
    sdfg = build_sdfg(_GLOBAL_STRUCT_MEMBER_SRC, tmp_path, name="global_member", entry="owner_mod::setup").build()
    sdfg.validate()
    n = 5
    # The module global's member is host storage the SDFG writes through (the bindings marshal it).
    counts = np.full(n, -1, dtype=np.int32)
    total = np.full(1, -1, dtype=np.int32)
    sdfg(n=n, total=total, global_desc_counts=counts, global_desc_counts_d0=n, offset_global_desc_counts_d0=1)
    assert total[0] == n
    np.testing.assert_array_equal(counts, [n, 0, 0, 0, 0])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
