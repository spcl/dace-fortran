"""SUM over an elemental product with a DATA-DEPENDENT extent (QE h_psi pattern).

``w1(ih) = fac * SUM(deeq(ih, 1:nh(nt), ia) * v(1:nh(nt)))`` -- the elemental's
extent is the array load ``nh(nt)``.  ``resolveExtent`` used to return the bare
base-array name (``nh``), which ``declare_synth_array`` then registered as a
symbol, colliding with the existing ``nh`` data descriptor
(``FileExistsError: Cannot create symbol "nh"`` -- the QE h_psi build's second
fatal error).  Even with the subscript kept, a mask transient cannot be shaped
by an array element read, so the bridge lowers these reductions as an explicit
scalar-accumulate loop (``buildElementalAccumulateReduce``) whose ``nh[nt]``
bound rides emit_loop's bound-hoist machinery.

Pinned: (1) build + validate, (2) numerics vs a numpy reference.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
subroutine probe_sum(deeq, nh, becpr, nhm, nt, ia, m, fac, w1)
  implicit none
  integer, intent(in) :: nhm, nt, ia, m
  integer, intent(in) :: nh(2)
  real(8), intent(in) :: deeq(nhm, nhm, 2), becpr(m)
  real(8), intent(in) :: fac
  real(8), intent(out) :: w1(nhm)
  integer :: ih
  do ih = 1, nh(nt)
    w1(ih) = fac * sum(deeq(ih, 1:nh(nt), ia) * becpr(1:nh(nt)))
  end do
end subroutine probe_sum
"""


def _inputs():
    rng = np.random.default_rng(11)
    nhm, m = 6, 10
    deeq = np.asfortranarray(rng.standard_normal((nhm, nhm, 2)))
    becpr = np.asfortranarray(rng.standard_normal(m))
    nh = np.array([3, 5], dtype=np.int32)
    nt, ia, fac = 2, 1, 1.5
    k = nh[nt - 1]
    w1_ref = np.zeros(nhm)
    for ih in range(1, k + 1):
        w1_ref[ih - 1] = fac * float(np.dot(deeq[ih - 1, :k, ia - 1], becpr[:k]))
    return deeq, nh, becpr, nhm, nt, ia, m, fac, w1_ref


def test_sum_dynamic_extent_builds(tmp_path):
    """The SDFG builds and validates (was: FileExistsError on symbol 'nh')."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="probe_sum").build()
    sdfg.validate()


def test_sum_dynamic_extent_numerical(tmp_path):
    """Accumulate-loop lowering matches the numpy reference."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="probe_sum").build()
    deeq, nh, becpr, nhm, nt, ia, m, fac, w1_ref = _inputs()
    w1 = np.zeros(nhm, dtype=np.float64)
    sdfg(deeq=deeq, nh=nh, becpr=becpr, nhm=nhm, nt=nt, ia=ia, m=m, fac=fac, w1=w1)
    np.testing.assert_allclose(w1, w1_ref, rtol=1e-12, atol=1e-12)
