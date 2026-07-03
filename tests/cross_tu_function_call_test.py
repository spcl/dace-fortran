"""A scalar/expression assigned from a function defined in another module
(translation unit) must lower correctly -- the function has to be inlined so
its body (here ``min``) is visible to the expression builder.

This is the pattern ICON's dynamical core hits via
``nproma_gradp = cpu_min_nproma(nproma, 256)`` (``mo_solve_nonhydro``), where
``cpu_min_nproma`` is defined in ``mo_parallel_config`` and on CPU is just
``MIN(nproma, min_nproma)``.  When the callee's TU is not merged in, the bridge
sees an opaque ``fir.call`` and the expression builder emits ``?`` (an invalid
tasklet ``_out = ?``).  Merging the modules (``build_sdfg_from_files`` /
``merge_used_modules`` + ``hlfir-inline-all``) splices the body in, so the call
becomes ``min`` and lowers.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import have_flang
from dace_fortran import build_sdfg_from_files

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HELPER = """
module mo_clamp
  implicit none
contains
  pure integer function clamp_to(n, m) result(r)
    integer, intent(in) :: n, m
    r = min(n, m)
  end function clamp_to
end module mo_clamp
"""

_CALLER = """
module mo_apply_clamp
  use mo_clamp, only: clamp_to
  implicit none
contains
subroutine apply_clamp(nproma, x, out)
  implicit none
  integer, intent(in) :: nproma
  real(8), intent(in) :: x(8)
  real(8), intent(out) :: out(8)
  integer :: nb, i
  nb = clamp_to(nproma, 4)          ! cross-TU function result in a scalar assign
  do i = 1, 8
    if (i <= nb) then
      out(i) = x(i) * 2.0d0
    else
      out(i) = 0.0d0
    end if
  end do
end subroutine apply_clamp
end module mo_apply_clamp
"""


@pytest.mark.parametrize("merge_engine", ["fparser", "regex"])
def test_cross_tu_function_result_inlines(tmp_path: Path, merge_engine):
    """``nb = clamp_to(nproma, 4)`` with ``clamp_to`` in another module: the
    callee inlines (to ``min``) when both TUs are merged, the SDFG builds, and
    the result matches the reference for nproma above and below the clamp.

    Run with both ``USE``-merge engines (fparser default + legacy regex)."""
    caller = tmp_path / "apply_clamp.f90"
    caller.write_text(_CALLER)
    helper = tmp_path / "mo_clamp.f90"
    helper.write_text(_HELPER)

    sdfg = build_sdfg_from_files([caller, helper],
                                 entry="mo_apply_clamp::apply_clamp",
                                 name="apply_clamp",
                                 out_dir=tmp_path / "build",
                                 merge_engine=merge_engine)

    for nproma in (2, 6):  # below and above the clamp of 4
        x = np.arange(1, 9, dtype=np.float64)
        out = np.zeros(8, dtype=np.float64)
        sdfg(nproma=np.int32(nproma), x=x, out=out)
        nb = min(nproma, 4)
        ref = np.where(np.arange(1, 9) <= nb, x * 2.0, 0.0)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
