"""Regression: two hlfir-inlined copies of a POINTER-rebind local must keep
their own rebind targets.

``exchange_data_r3d``'s ``send_ptr => recv`` inlined once per call site binds a
DIFFERENT ``recv`` per copy, but downstream keying is name-based end to end
(VarInfos land in a ``{fortran_name: v}`` dict; memlets carry the extracted
short name), so same-named copies collapsed last-wins.  Worse, the pointer-view
source trace stopped at the inlined DUMMY's declare, recording a
``view_source`` (``recv``) that has no SDFG descriptor -- the reads then
surfaced as bare View AccessNodes and the whole solve_nh SDFG failed
validation with "Ambiguous or invalid edge to/from a View access node".

Two-part fix under test:

* ``prepareExtractionState`` suffixes the 2nd+ ``pointer_view``-tagged declare
  of each uniq_name with ``_pv<N>`` ON THE OP, so every consumer that resolves
  an access chain to its declare sees a distinct per-copy name, and
* the pointer-view source walk peels through inlined-dummy declares (a
  declare whose memref chains onward) to ROOT storage, so ``view_source``
  lands on a registered descriptor.

The kernel below inlines ``bump`` twice; each copy rebinds ``p`` to a
different actual.  Correct lowering increments BOTH arrays; the pre-fix
collapse either fails validation or links both copies' reads to one source.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_pv
  implicit none
contains
  subroutine bump(dst, n)
    integer, intent(in) :: n
    real(8), intent(inout), target :: dst(:)
    real(8), pointer :: p(:)
    integer :: i
    p => dst
    do i = 1, n
      p(i) = p(i) + 1.0d0
    end do
  end subroutine bump

  subroutine run(a, b, n)
    integer, intent(in) :: n
    real(8), intent(inout), target :: a(:), b(:)
    call bump(a, n)
    call bump(b, n)
  end subroutine run
end module mo_pv
"""


def test_inlined_pointer_rebind_copies_keep_their_targets(tmp_path: Path):
    """Both inlined copies' rebinds land on their own actual: a AND b bump."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="pvcopies", entry="mo_pv::run").build()
    n = 6
    a = np.zeros(n, dtype=np.float64, order="F")
    b = np.full(n, 10.0, dtype=np.float64, order="F")
    sdfg(a=a, b=b, n=np.int32(n), a_d0=n, b_d0=n)
    np.testing.assert_allclose(a, np.ones(n), rtol=0, atol=0)
    np.testing.assert_allclose(b, np.full(n, 11.0), rtol=0, atol=0)
