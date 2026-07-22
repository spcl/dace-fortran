"""The C++ DaCe generates must survive static analysis.

Motivated by a real miscompile: an inlined ALLOCATE whose extent symbol was never bound produced
``new double[<uninitialised local>]``, and the only symptom was a glibc abort inside an unrelated ``free()`` two
kernels later.  ``-Wmaybe-uninitialized`` names that at compile time.

The gcc pass runs in the normal sweep; ``-fanalyzer``/clang-tidy/cppcheck are slower and marked ``long`` so CI
runs them and routine local sweeps skip the cost with ``-m "not long"``.
"""

import pytest

from _util import build_sdfg, have_flang
from dace_fortran.codegen_check import CRITICAL_WARNINGS, analyze

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# An ALLOCATE whose extent comes from a runtime scalar, plus a loop nest over it -- the shape that produced the
# uninitialised-extent miscompile.  Kept tiny so the analysis, not the build, dominates the runtime.
ALLOCATE_SRC = """
subroutine main(n, out)
  integer, intent(in) :: n
  double precision, intent(out) :: out(n)
  double precision, allocatable :: buf(:)
  integer :: i
  allocate (buf(n))
  do i = 1, n
    buf(i) = real(i, 8) * 2.0d0
  end do
  do i = 1, n
    out(i) = buf(i)
  end do
  deallocate (buf)
end subroutine main"""

# Two-dimensional ALLOCATE with a derived extent: exercises the extent-expression path (clamp over a product)
# rather than a bare symbol copy.
DERIVED_EXTENT_SRC = """
subroutine main(n, m, out)
  integer, intent(in) :: n, m
  double precision, intent(out) :: out(n, m)
  double precision, allocatable :: buf(:, :)
  integer :: i, j
  allocate (buf(n, m))
  do j = 1, m
    do i = 1, n
      buf(i, j) = real(i + j, 8)
    end do
  end do
  out(:, :) = buf(:, :)
  deallocate (buf)
end subroutine main"""

SOURCES = {"allocate": ALLOCATE_SRC, "derived_extent": DERIVED_EXTENT_SRC}


def built_sdfg(source, tmp_path, name):
    sdfg = build_sdfg(source, tmp_path, name=name).build()
    sdfg.compile()
    return sdfg


@pytest.mark.parametrize("kernel", sorted(SOURCES))
def test_generated_cpp_has_no_critical_warnings(kernel, tmp_path):
    """No generated TU may emit a UB-class warning; see ``CRITICAL_WARNINGS`` for the list and why."""
    sdfg = built_sdfg(SOURCES[kernel], tmp_path, name=kernel)
    found = analyze(sdfg, "warnings")
    assert not found, ("generated C++ emits critical warnings (" + ", ".join(CRITICAL_WARNINGS) + "):\n" +
                       "\n".join(found))


@pytest.mark.long
@pytest.mark.parametrize("tool", ["analyzer", "clang-tidy", "cppcheck"])
@pytest.mark.parametrize("kernel", sorted(SOURCES))
def test_generated_cpp_passes_deep_analysis(tool, kernel, tmp_path):
    """Interprocedural analysis catches the leaks/overruns a single-TU warning pass cannot see."""
    sdfg = built_sdfg(SOURCES[kernel], tmp_path, name=kernel)
    found = analyze(sdfg, tool)
    assert not found, f"{tool} reported on generated C++:\n" + "\n".join(found)
