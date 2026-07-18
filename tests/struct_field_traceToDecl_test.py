"""Verify ``traceToDecl`` returns the FLATTENED struct-field name
(``<parent>_<member>``) for ``hlfir.designate`` ops with a component attribute.

Before this fix, QE's ``vcut_get`` calling a libcall over ``vcut % a`` made
``traceToDecl`` walk through the designate to the parent (``vcut``), so the
libcall dispatcher raised ``KeyError: 'vcut'`` instead of finding the flattened
``vcut_a`` the ``hlfir-flatten-structs`` pass actually produces.  ``traceToDecl``
now builds the flattened name from the component attribute + recursively-traced
parent.

These probes pin the contract; MODULE-LEVEL struct fields newly progress past
resolution but hit a separate, clean "not registered as SDFG data" gap downstream.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_dummy_arg_struct_field_used_as_matmul_input(tmp_path):
    """``MATMUL(TRANSPOSE(vcut % a), q)`` with vcut a DUMMY arg -- builds + numerical
    correctness via the flatten pass producing ``vcut_a`` as a flat arg."""
    src = """
module m
  type :: vcut_type
    real(kind=8) :: a(3, 3)
    real(kind=8) :: cutoff
  end type
contains
  subroutine vcut_get(vcut, q, res)
    type(vcut_type), intent(in) :: vcut
    real(kind=8), intent(in) :: q(3)
    real(kind=8), intent(out) :: res(3)
    res = MATMUL(TRANSPOSE(vcut % a), q)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="vcut_get", entry="m::vcut_get").build()
    # Flatten pass produces vcut_a/vcut_cutoff as top-level SDFG arrays/scalars; the struct base is gone.
    assert "vcut_a" in sdfg.arrays, f"expected vcut_a in arrays: {sorted(sdfg.arrays.keys())}"
    assert "vcut" not in sdfg.arrays, f"struct base should be flattened away"
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    res = np.zeros(3, dtype=np.float64, order='F')
    sdfg(vcut_a=A, vcut_cutoff=np.zeros((1, ), dtype=np.float64, order='F'), q=q, res=res)
    np.testing.assert_allclose(res, A.T @ q)
