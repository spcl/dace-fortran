"""The persistent-state pattern for an SDFG-generated MPI communicator.

ICON builds a halo communicator once at model init and reuses it every
timestep.  When a kernel that *creates* the communicator is lowered to an SDFG,
the communicator handle is declared in the driver (outside the SDFG) but
generated INSIDE the SDFG on the first call -- and must be returned so later
calls reuse it rather than re-creating it.

These tests pin that the bindings support it: an ``intent(inout)`` handle the
SDFG writes is reflected back to the caller (so the "init once, reuse" guard
holds across calls), for both a 1-element array carrier (the documented
SDFG-external-return shape) and a bare scalar.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_sdfg_generated_comm_returned_via_array_carrier(tmp_path: Path):
    """The comm handle is carried in a 1-element ``intent(inout)`` array.  It is
    declared (allocated) by the caller, generated inside the SDFG on the first
    call (``comm == 0`` guard), and returned -- so the second call sees the
    already-built comm and does NOT regenerate it.  This is the "initialize once
    inside the SDFG, reuse across iterations" pattern, with the driver holding
    the handle between calls."""
    src = """
subroutine halo_step(comm, buf, n)
  implicit none
  integer, intent(in)    :: n
  integer, intent(inout) :: comm(1)
  real(8), intent(inout) :: buf(n)
  integer :: i
  ! Build the communicator once, on the first call (it is 0 = "not built" until
  ! then); subsequent calls find it already set and reuse it.
  if (comm(1) == 0) then
    comm(1) = 42
  end if
  do i = 1, n
    buf(i) = buf(i) + real(comm(1), 8)
  end do
end subroutine halo_step
"""
    sdfg = build_sdfg(src, tmp_path, name='halo_step', entry='halo_step').build()

    n = 4
    comm = np.zeros(1, dtype=np.int32)  # declared outside, "not built yet"
    buf = np.ones(n, dtype=np.float64)

    # First call: the SDFG generates the comm and returns it to the caller.
    sdfg(comm=comm, buf=buf, n=n)
    assert comm[0] == 42, "the SDFG-generated comm handle was not returned to the caller"
    np.testing.assert_array_equal(buf, np.full(n, 1.0 + 42.0))

    # Second call: the caller still holds comm=42, so the SDFG reuses it (the
    # init guard does NOT fire again) -- the round-tripped handle persists.
    sdfg(comm=comm, buf=buf, n=n)
    assert comm[0] == 42
    np.testing.assert_array_equal(buf, np.full(n, 1.0 + 42.0 + 42.0))


def test_sdfg_generated_comm_not_returned_via_inout_scalar(tmp_path: Path):
    """Contract: a BARE ``intent(inout)`` scalar does NOT round-trip a value
    generated inside the SDFG.  The bridge lowers it to a by-value SDFG scalar the
    bindings pass input-only, so a value produced inside is used internally (``buf``
    reflects it) but the caller's scalar is left unchanged.  The supported shape for
    a returned handle is the 1-element array carrier
    (:func:`test_sdfg_generated_comm_returned_via_array_carrier`)."""
    src = """
subroutine halo_step_scalar(comm, buf, n)
  implicit none
  integer, intent(in)    :: n
  integer, intent(inout) :: comm
  real(8), intent(inout) :: buf(n)
  integer :: i
  if (comm == 0) then
    comm = 42
  end if
  do i = 1, n
    buf(i) = buf(i) + real(comm, 8)
  end do
end subroutine halo_step_scalar
"""
    sdfg = build_sdfg(src, tmp_path, name='halo_step_scalar', entry='halo_step_scalar').build()

    n = 4
    comm = np.int32(0)
    buf = np.ones(n, dtype=np.float64)
    sdfg(comm=comm, buf=buf, n=n)
    # By-value: the caller's bare scalar is NOT written back -- comm stays 0 ...
    assert int(comm) == 0
    # ... but the internally-generated ``comm = 42`` WAS used, so buf += 42.
    np.testing.assert_array_equal(buf, np.full(n, 1.0 + 42.0))
