"""Verify ``traceToDecl`` returns the FLATTENED struct-field name
(``<parent>_<member>``) for ``hlfir.designate`` ops that carry a
component attribute.

QE's ``vcut_get`` (in ``vexx_bp_k_gpu``) calls a libcall over
``vcut % a`` -- the matmul's first operand is the struct-field
designate.  Before this fix, ``traceToDecl`` walked THROUGH the
designate to the parent (``vcut``) and returned that name; the
libcall dispatcher then tried to look up ``vcut`` as an SDFG
array and raised ``KeyError: 'vcut'``.

The bridge's ``hlfir-flatten-structs`` pass produces flat declares
named ``vcut_a`` etc. for DUMMY-arg struct fields, and the
bindings layer maps a MODULE-LEVEL struct global by the same
convention.  ``traceToDecl`` now builds that flattened name
directly from the designate's component attribute + the
recursively-traced parent name -- so it returns ``vcut_a`` for
``vcut % a``, ``vcut_corrected`` for ``vcut % corrected``, etc.

These probes pin the contract.  The DUMMY-arg variants were
working before this fix via the flatten pass; the MODULE-LEVEL
variants newly progress past the ``vcut`` -> ``vcut_a`` resolution
(downstream module-level field SDFG-array registration is still a
separate gap surfaced as a clean ``not registered as SDFG data``
error rather than ``KeyError``).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_dummy_arg_struct_field_used_as_matmul_input(tmp_path):
    """``MATMUL(TRANSPOSE(vcut % a), q)`` with vcut a DUMMY arg.
    Builds + numerical correctness.  After my trace fix, this still
    works via the flatten pass producing ``vcut_a`` as a flat
    arg."""
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
    # The flatten pass produces ``vcut_a`` (and ``vcut_cutoff``) as
    # top-level SDFG arrays/scalars; the struct base is GONE.
    assert "vcut_a" in sdfg.arrays, f"expected vcut_a in arrays: {sorted(sdfg.arrays.keys())}"
    assert "vcut" not in sdfg.arrays, f"struct base should be flattened away"
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    res = np.zeros(3, dtype=np.float64, order='F')
    sdfg(vcut_a=A, vcut_cutoff=np.zeros((1, ), dtype=np.float64, order='F'), q=q, res=res)
    np.testing.assert_allclose(res, A.T @ q)
