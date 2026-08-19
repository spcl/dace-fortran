"""Reproducer for bug 4 (dace-fortran-fixes-needed-6c99810.md):
non-unit / negative lower bounds of module arrays lost in lowering.

QE declares ``eigts1/2/3(-nr:nr, nat)`` (module allocatables, allocated by
the caller with a NEGATIVE lower bound) and indexes them with the
possibly-negative Miller indices ``mill``.  The bridge marshals only the
extent symbol (``eig_d0``) and normalizes the index as if lbound were 1
(``eig[idx - 1]``), so every read lands ``lbound-1`` slots too low: wrong
structure-factor phases everywhere, out-of-bounds for indices <= 0.

``test_module_allocatable_runtime_negative_lbound`` encodes the CORRECT
numeric result and FAILS at 6c99810.  A fix may either honor the declared /
runtime lower bound in index normalization, or marshal a per-dimension
``offset_eig_d0`` symbol (the mechanism already used for ``dfftt%nl``) --
the test binds that symbol if the fixed build exposes it.

``test_static_negative_lbound_module_array`` is a CONTROL: STATIC-shape
module arrays already get an ``offset_<name>_d0`` and index correctly (see
tests/negative_lower_bound_test.py), which localizes the defect to
DEFERRED-shape (allocatable/pointer) module arrays whose bounds only exist
in the runtime descriptor.

Companion doc: bug4_module_array_lbound.md
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

# Minimal QE eigts shape: deferred-shape module allocatable, caller allocates
# with lbound -nmax, kernel reads through an indirect (Miller-style) index.
_KERNEL = """
module gdat_mod
  implicit none
  complex(8), allocatable :: eig(:)
end module gdat_mod

module phase_mod
  implicit none
contains
subroutine phase(m, mill, y)
  use gdat_mod, only: eig
  implicit none
  integer, intent(in) :: m
  integer, intent(in) :: mill(m)
  complex(8), intent(inout) :: y(m)
  integer :: ig
  do ig = 1, m
    y(ig) = y(ig) * eig(mill(ig))
  end do
end subroutine phase
end module phase_mod
"""

_KERNEL_STATIC = """
module sdat_mod
  implicit none
  integer :: arr(-3:3) = 0
end module sdat_mod

module pick_mod
  implicit none
contains
subroutine pick(idx, out)
  use sdat_mod, only: arr
  implicit none
  integer, intent(in) :: idx
  integer, intent(out) :: out
  out = arr(idx)
end subroutine pick
end module pick_mod
"""


def _bind_optional_syms(sdfg, kw, nmax):
    """Bind extent/offset symbols for ``eig`` against whatever surface this
    build exposes (a fixed build may add ``offset_eig_d0``)."""
    free = {str(s) for s in sdfg.free_symbols}
    if "eig_d0" in free:
        kw["eig_d0"] = np.int64(2 * nmax + 1)
    if "offset_eig_d0" in free:
        kw["offset_eig_d0"] = np.int64(-nmax)
    return kw


def test_module_allocatable_runtime_negative_lbound(tmp_path: Path):
    """FAILS at 6c99810: the caller allocated ``eig(-3:3)``; reading
    ``eig(mill(ig))`` must honor the runtime lower bound.  The buggy build
    indexes 1-based (``eig[mill-1]``) and returns [-3,-2,-1,-3] here."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_KERNEL, sdfg_dir, name="phase", entry="phase_mod::phase").build()

    nmax, m = 3, 4
    # Fortran view: eig(i) == complex(i) for i in -3..3; storage slot k holds
    # Fortran element (k - nmax).
    eig = np.array([complex(v, 0) for v in range(-nmax, nmax + 1)], dtype=np.complex128)
    # All indices >= 1 so the BUGGY read stays in bounds and the mismatch is
    # deterministic values, not a crash.  (For mill <= 0 the buggy index is
    # negative: genuine out-of-bounds, the QE crash/garbage mode.)
    mill = np.array([1, 2, 3, 1], dtype=np.int32)
    y = np.ones(m, dtype=np.complex128)

    kw = _bind_optional_syms(sdfg, {"m": np.int32(m), "mill": mill, "y": y, "eig": eig}, nmax)
    sdfg(**kw)

    expected = np.array([complex(v, 0) for v in mill], dtype=np.complex128)
    np.testing.assert_allclose(y,
                               expected,
                               err_msg="module allocatable's runtime lower bound ignored -- the "
                               "kernel indexed eig[] 1-based (QE eigts structure-factor bug)")


def test_static_negative_lbound_module_array(tmp_path: Path):
    """CONTROL (passes at 6c99810): a STATIC ``arr(-3:3)`` module array gets
    a marshalled offset and indexes correctly -- the bug is specific to
    deferred-shape module arrays."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_KERNEL_STATIC, sdfg_dir, name="pick", entry="pick_mod::pick").build()

    arr = np.array(range(-3, 4), dtype=np.int32) * 10  # arr(i) = 10*i
    out = np.zeros(1, dtype=np.int32)
    kw = {"idx": np.int32(-2), "out": out, "arr": arr}
    free = {str(s) for s in sdfg.free_symbols}
    if "arr_d0" in free:
        kw["arr_d0"] = np.int64(7)
    if "offset_arr_d0" in free:
        kw["offset_arr_d0"] = np.int64(-3)
    sdfg(**kw)
    assert out[0] == -20, f"static module-array lbound broken too: {out[0]}"
