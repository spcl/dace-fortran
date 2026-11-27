# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``.and.`` / ``.or.`` / ``.eqv.`` / ``.neqv.`` on LOGICAL values, as a stored value and as a branch condition.

LLVM 23 lowers these to ``fir.logical_and`` / ``fir.logical_or`` / ``fir.eqv`` / ``fir.neqv``, which the bridge
rejected as unlowered ``?`` placeholders until the ``lower-fir-logical-ops`` pass rewrote them to the LLVM 21/22 shape.
"""

import numpy as np

SRC = """
subroutine lg(a, b, out, n)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: a(n), b(n)
  integer, intent(inout) :: out(n)
  logical :: p, q, r
  integer :: i
  do i = 1, n
    p = a(i) > 0
    q = b(i) > 0
    r = (p .and. q) .or. (p .eqv. q)
    out(i) = 0
    if (r) out(i) = 1
    if (p .neqv. q) out(i) = out(i) + 2
    if (p .or. q) out(i) = out(i) + 4
    if (.not. (p .and. q)) out(i) = out(i) + 8
  end do
end subroutine lg
"""


def test_logical_binary_ops_match_gfortran(tmp_path):
    from tests._util import build_sdfg, f2py_compile

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir()
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    sdfg = build_sdfg(SRC, sdfg_dir, name="lg", entry="lg").build()
    ref = f2py_compile(SRC, ref_dir, "lg_ref")

    a = np.asfortranarray(np.array([0, 0, 1, 1], dtype=np.int32))
    b = np.asfortranarray(np.array([0, 1, 0, 1], dtype=np.int32))
    o_ref = np.zeros(4, dtype=np.int32, order="F")
    o_sdfg = np.zeros(4, dtype=np.int32, order="F")
    ref.lg(a, b, o_ref)
    sdfg(n=np.int32(4), a=a, b=b, out=o_sdfg)
    np.testing.assert_array_equal(o_sdfg, o_ref)
