# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``ddot`` over a slice with data-dependent bounds (QE ``calbec_rs_gamma`` pattern).

``r = ddot(m, a(s(ia):e(ia), 1), 1, b, 1)`` slices ``a`` with bounds read from
arrays (``s(ia)`` / ``e(ia)``).  A ``blas.Dot`` libcall needs static or symbolic
memlet subsets; a bound that reads an array element renders with a subscript
(``s[ia]``), which DaCe's memlet parser rejects
(``ValueError: too many values to unpack`` in ``emit_library`` -- the QE
``h_psi`` build's fatal error).  The bridge lowers such dots as the equivalent
explicit reduction loop instead (``buildDdotDynamicSliceNodes``).

These tests pin (1) the SDFG builds + validates and (2) the lowered loop is
numerically exact against the gfortran/f2py reference.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

_SRC = """
subroutine probe_ddot(a, s, e, b, n, r)
  implicit none
  integer, intent(in) :: n, s(3), e(3)
  real(8), intent(in) :: a(n,3), b(n)
  real(8), intent(out) :: r
  integer :: ia, m
  real(8), external :: ddot
  ia = 2
  m = e(ia) - s(ia) + 1
  r = ddot(m, a(s(ia):e(ia), 1), 1, b, 1)
end subroutine probe_ddot
"""


def _f2py_ref(tmp_path: Path):
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir = tmp_path / "ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "probe_ddot.f90"
    src.write_text(_SRC)
    subprocess.check_call(
        [sys.executable, "-m", "numpy.f2py", "-c",
         str(src), "-m", "probe_ddot_ref", "-lblas", "--quiet"], cwd=out_dir)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__("probe_ddot_ref")
    return sys.modules["probe_ddot_ref"]


def _inputs():
    rng = np.random.default_rng(7)
    n = 12
    a = np.asfortranarray(rng.standard_normal((n, 3)))
    b = np.asfortranarray(rng.standard_normal(n))
    s = np.array([2, 4, 1], dtype=np.int32)
    e = np.array([5, 9, 3], dtype=np.int32)
    return a, s, e, b, n


def test_ddot_dynamic_slice_builds(tmp_path):
    """The SDFG builds and validates (was: Memlet parse ValueError at emit)."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="probe_ddot").build()
    sdfg.validate()


def test_ddot_dynamic_slice_numerical(tmp_path):
    """Explicit-loop lowering == gfortran reference, bit-exact."""
    mod = _f2py_ref(tmp_path)
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="probe_ddot").build()

    a, s, e, b, n = _inputs()
    r_ref = mod.probe_ddot(a, s, e, b, n)

    r_sdfg = np.zeros(1, dtype=np.float64)
    sdfg(a=a, s=s, e=e, b=b, n=n, r=r_sdfg)

    np.testing.assert_allclose(r_sdfg[0], r_ref, rtol=1e-12, atol=1e-12)
